"""
Drug Response Prediction (GDSC) — end‑to‑end baseline

What this script does in ONE go:
1) Loads your uploaded GDSC files (GDSC1/2 IC50s, screened_compounds, Cell_Lines_Details, RNA expression zip)
2) Harmonises into a single tidy table: (cell_id, drug_id, drug_name, tissue, smiles, ln_ic50)
3) Builds features:
   • Expression → variance filter (top 5000) → StandardScaler → PCA(512)
   • Drug SMILES → Morgan FP(1024, r=2) + 6 simple RDKit descriptors
   • (Optional) one‑hot Tissue
4) Trains a LightGBM regressor with GroupKFold (leave‑drug‑out) and Optuna HPO
5) Prints CV RMSE/MAE/Spearman, saves artifacts under ./artifacts/

Usage (from project root):
    python drp_baseline.py \
      --gdsc1 /mnt/data/GDSC1_fitted_dose_response_27Oct23.xlsx \
      --gdsc2 /mnt/data/GDSC2_fitted_dose_response_27Oct23.xlsx \
      --compounds /mnt/data/screened_compounds_rel_8.5.csv \
      --cells /mnt/data/Cell_Lines_Details.xlsx \
      --expr_zip /mnt/data/Cell_line_RMA_proc_basalExp.txt.zip

Requires: pandas, numpy, scikit-learn, rdkit-pypi, lightgbm, optuna, scipy
"""

import argparse
import os
import math
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import optuna

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

warnings.filterwarnings("ignore", category=UserWarning)


# ------------------------
# Utility: robust column access
# ------------------------

def _ci(df: pd.DataFrame):
    return {c.lower(): c for c in df.columns}


# ------------------------
# Loaders
# ------------------------

def load_gdsc_ic50(gdsc_path: str, tag: str) -> pd.DataFrame:
    df = pd.read_excel(gdsc_path)
    cols = _ci(df)
    keep = [
        cols.get("cell_line_name", "CELL_LINE_NAME"),
        cols.get("cosmic_id", "COSMIC_ID"),
        cols.get("drug_id", "DRUG_ID"),
        cols.get("drug_name", "DRUG_NAME"),
        cols.get("ln_ic50", "LN_IC50"),
    ]
    df = df[keep].copy()
    df.columns = ["CELL_LINE_NAME", "COSMIC_ID", "DRUG_ID", "DRUG_NAME", "LN_IC50"]
    df["dataset"] = tag
    return df


def load_compounds(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = _ci(df)
    # Expect columns like "Drug id", "SMILES"
    drug_col = cols.get("drug id", None) or cols.get("drug_id", None) or "Drug id"
    smiles_col = cols.get("smiles", "SMILES")
    out = df[[drug_col, smiles_col]].drop_duplicates()
    out.columns = ["DRUG_ID", "SMILES"]
    return out


def load_cells(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    cols = _ci(df)
    cosmic = cols.get("cosmic identifier", None) or cols.get("cosmic_id", None) or "COSMIC identifier"
    tissue = cols.get("tissue descriptor", None) or cols.get("primary site", None) or "Tissue descriptor"
    out = df[[cosmic, tissue]].drop_duplicates()
    out.columns = ["COSMIC_ID", "tissue"]
    return out


def load_expression_zip(expr_zip: str, topk: int = 5000, n_pcs: int = 512) -> Tuple[pd.DataFrame, PCA, StandardScaler]:
    """Reads zipped tab‑separated expression matrix (genes x samples or vice‑versa) and returns PCA PCs per COSMIC_ID."""
    expr = pd.read_csv(expr_zip, sep="\t", compression="zip")
    # Heuristics: first column is gene/probe id; remaining columns are COSMIC IDs
    gene_col = expr.columns[0]
    expr = expr.set_index(gene_col).T
    expr.index.name = "COSMIC_ID"
    # Keep numeric cols only
    expr = expr.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="any")
    # Variance filter
    variances = expr.var(axis=0)
    top_genes = variances.sort_values(ascending=False).head(topk).index
    expr = expr[top_genes]
    # Scale + PCA
    scaler = StandardScaler()
    Xs = scaler.fit_transform(expr.values)
    pca = PCA(n_components=n_pcs, random_state=42)
    PCs = pca.fit_transform(Xs)
    expr_pca = pd.DataFrame(PCs, index=expr.index, columns=[f"PC{i+1}" for i in range(n_pcs)])
    return expr_pca, pca, scaler


# ------------------------
# Drug featurisation
# ------------------------

def morgan_fp(smiles: str, n_bits: int = 1024, radius: int = 2) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.uint8)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def rdkit_desc(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(6, dtype=np.float32)
    return np.array([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
    ], dtype=np.float32)


def build_drug_matrix(smiles_series: pd.Series, n_bits: int = 1024) -> np.ndarray:
    fps = []
    descs = []
    for s in smiles_series.fillna(""):
        fps.append(morgan_fp(s, n_bits=n_bits, radius=2))
        descs.append(rdkit_desc(s))
    return np.hstack([np.array(fps), np.array(descs)])


# ------------------------
# Metrics
# ------------------------

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def spearman(y_true, y_pred):
    try:
        return float(spearmanr(y_true, y_pred).correlation)
    except Exception:
        return np.nan


# ------------------------
# Training loop (LightGBM + Optuna) with leave‑drug‑out
# ------------------------

def train_lgbm_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_trials: int = 30, seed: int = 42):
    gkf = GroupKFold(n_splits=5)

    def objective(trial):
        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": trial.suggest_float("lr", 1e-3, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 300),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 10.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 10.0),
            "verbose": -1,
            "seed": seed,
            "num_threads": 0,
        }
        rmses = []
        for tr, va in gkf.split(X, y, groups):
            dtrain = lgb.Dataset(X[tr], label=y[tr])
            dvalid = lgb.Dataset(X[va], label=y[va])
            model = lgb.train(
                params,
                dtrain,
                valid_sets=[dvalid],
                num_boost_round=5000,
                early_stopping_rounds=200,
                verbose_eval=False,
            )
            pred = model.predict(X[va], num_iteration=model.best_iteration)
            rmses.append(rmse(y[va], pred))
        return float(np.mean(rmses))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params

    # Final CV with metrics
    fold_metrics = []
    models = []
    for tr, va in gkf.split(X, y, groups):
        dtrain = lgb.Dataset(X[tr], label=y[tr])
        dvalid = lgb.Dataset(X[va], label=y[va])
        model = lgb.train(
            {**best, "objective": "regression", "metric": "rmse", "verbose": -1, "seed": seed},
            dtrain,
            valid_sets=[dvalid],
            num_boost_round=5000,
            early_stopping_rounds=200,
            verbose_eval=False,
        )
        pred = model.predict(X[va], num_iteration=model.best_iteration)
        fold_metrics.append({
            "rmse": rmse(y[va], pred),
            "mae": mae(y[va], pred),
            "spearman": spearman(y[va], pred),
        })
        models.append(model)

    return best, fold_metrics, models


# ------------------------
# Main
# ------------------------

def main(args):
    os.makedirs("artifacts", exist_ok=True)

    # 1) Responses
    gdsc1 = load_gdsc_ic50(args.gdsc1, "GDSC1")
    gdsc2 = load_gdsc_ic50(args.gdsc2, "GDSC2")
    resp = pd.concat([gdsc1, gdsc2], ignore_index=True)

    # 2) Compounds (SMILES)
    cmpd = load_compounds(args.compounds)
    resp = resp.merge(cmpd, on="DRUG_ID", how="left")

    # 3) Cell metadata (tissue)
    cells = load_cells(args.cells)
    resp = resp.merge(cells, left_on="COSMIC_ID", right_on="COSMIC_ID", how="left")

    # Cleanup + rename
    resp = resp.rename(columns={
        "COSMIC_ID": "cell_id",
        "DRUG_ID": "drug_id",
        "DRUG_NAME": "drug_name",
        "LN_IC50": "ln_ic50",
        "SMILES": "smiles",
    })

    # Drop missing essentials
    resp = resp.dropna(subset=["smiles", "ln_ic50"]).reset_index(drop=True)

    # Save tidy pairs
    resp.to_parquet("artifacts/gdsc_pairs.parquet", index=False)

    # 4) Expression PCs
    expr_pca, pca, scaler = load_expression_zip(args.expr_zip, topk=args.top_genes, n_pcs=args.pcs)
    expr_pca.to_parquet("artifacts/gdsc_expr_pca.parquet")

    # 5) Merge expression into responses
    merged = resp.merge(expr_pca, left_on="cell_id", right_index=True, how="inner")

    # 6) Features
    # Drug features
    X_drug = build_drug_matrix(merged["smiles"], n_bits=args.fp_bits).astype(np.float32)
    # Expression PCs already numeric
    X_expr = merged[[f"PC{i+1}" for i in range(args.pcs)]].values.astype(np.float32)

    # Optional tissue one‑hot
    if args.use_tissue:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        Xt = ohe.fit_transform(merged[["tissue"].fillna("unknown")])
        X = np.hstack([X_expr, X_drug, Xt]).astype(np.float32)
    else:
        X = np.hstack([X_expr, X_drug]).astype(np.float32)

    y = merged["ln_ic50"].values.astype(np.float32)
    groups = merged["drug_id"].astype(str).values  # leave‑drug‑out

    # 7) Train CV
    best, fold_metrics, models = train_lgbm_cv(X, y, groups, n_trials=args.trials, seed=42)

    # 8) Report
    rmse_vals = [m["rmse"] for m in fold_metrics]
    mae_vals = [m["mae"] for m in fold_metrics]
    sp_vals = [m["spearman"] for m in fold_metrics]

    print("Leave‑DRUG‑out CV:")
    print(f"  RMSE  mean±sd: {np.mean(rmse_vals):.4f} ± {np.std(rmse_vals):.4f}")
    print(f"  MAE   mean±sd: {np.mean(mae_vals):.4f} ± {np.std(mae_vals):.4f}")
    print(f"  Spearman mean±sd: {np.mean(sp_vals):.4f} ± {np.std(sp_vals):.4f}")

    # 9) Save artifacts
    pd.DataFrame(fold_metrics).to_csv("artifacts/cv_metrics.csv", index=False)
    pd.Series(best).to_json("artifacts/best_params.json")
    # Save merged index mapping for reproducibility
    merged[["cell_id", "drug_id", "drug_name", "tissue", "smiles", "ln_ic50"]].to_csv(
        "artifacts/merged_pairs_used.csv", index=False
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gdsc1", required=True)
    ap.add_argument("--gdsc2", required=True)
    ap.add_argument("--compounds", required=True)
    ap.add_argument("--cells", required=True)
    ap.add_argument("--expr_zip", required=True, help="Path to Cell_line_RMA_proc_basalExp.txt.zip")
    ap.add_argument("--top_genes", type=int, default=5000)
    ap.add_argument("--pcs", type=int, default=512)
    ap.add_argument("--fp_bits", type=int, default=1024)
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--use_tissue", action="store_true")
    args = ap.parse_args()
    main(args)
