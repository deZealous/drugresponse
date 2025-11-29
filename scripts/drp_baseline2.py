"""
Drug Response Prediction (GDSC) — baseline on processed data

Pipeline:
1) Assumes you already ran scripts/make_dataset.py, which produces:
       data/processed/merged.parquet
   with columns such as:
       cell_id, drug_id, drug_name, tissue, smiles, ln_ic50, PC1..PCk

2) Builds features:
   - Expression PCs: all columns starting with "PC"
   - Drug: Morgan fingerprint (fp_bits, radius=2) + 6 RDKit descriptors

3) Trains LightGBM with GroupKFold CV and Optuna hyperparameter search.

Run from project root:
    python scripts/drp_baseline.py \
        --data data/processed/merged.parquet \
        --out results/baseline_lgbm \
        --group_by drug \
        --use_tissue
"""

import argparse
import os
import warnings
from typing import Tuple, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb
import optuna

# RDKit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------
# Drug featurisation
# ------------------------

def morgan_fp(smiles: str, n_bits: int = 1024, radius: int = 2) -> np.ndarray:
    """Return Morgan fingerprint as numpy array of bits for a SMILES string."""
    if not isinstance(smiles, str) or smiles == "":
        return np.zeros(n_bits, dtype=np.uint8)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.uint8)
    # Use the modern MorganGenerator API
    gen = GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = gen.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def rdkit_desc(smiles: str) -> np.ndarray:
    """Simple RDKit descriptors: MW, LogP, TPSA, HBD, HBA, RotBonds."""
    if not isinstance(smiles, str) or smiles == "":
        return np.zeros(6, dtype=np.float32)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(6, dtype=np.float32)
    return np.array(
        [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
        ],
        dtype=np.float32,
    )


def build_drug_matrix(smiles_series: pd.Series, n_bits: int = 1024) -> np.ndarray:
    fps = []
    descs = []
    for s in smiles_series.fillna(""):
        fps.append(morgan_fp(s, n_bits=n_bits))
        descs.append(rdkit_desc(s))
    return np.hstack([np.array(fps, dtype=np.float32),
                      np.array(descs, dtype=np.float32)])


# ------------------------
# Metrics
# ------------------------

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def spearman(y_true, y_pred) -> float:
    try:
        return float(spearmanr(y_true, y_pred).correlation)
    except Exception:
        return float("nan")


# ------------------------
# Training loop (LightGBM + Optuna)
# ------------------------

def train_lgbm_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_trials: int = 30,
    seed: int = 42,
    n_splits: int = 5,
):
    gkf = GroupKFold(n_splits=n_splits)

    def objective(trial: optuna.Trial) -> float:
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
        rmses: List[float] = []
        for tr, va in gkf.split(X, y, groups):
            dtrain = lgb.Dataset(X[tr], label=y[tr])
            dvalid = lgb.Dataset(X[va], label=y[va])
            model = lgb.train(
                params,
                dtrain,
                valid_sets=[dvalid],
                num_boost_round=5000,
                callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)],
            )
            pred = model.predict(X[va], num_iteration=model.best_iteration)
            rmses.append(rmse(y[va], pred))
        return float(np.mean(rmses))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params

    # Final CV with metrics and saved models
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
            callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)],
        )
        pred = model.predict(X[va], num_iteration=model.best_iteration)
        fold_metrics.append(
            {
                "rmse": rmse(y[va], pred),
                "mae": mae(y[va], pred),
                "spearman": spearman(y[va], pred),
            }
        )
        models.append(model)

    return best, fold_metrics, models


# ------------------------
# Main
# ------------------------

def main(args):
    os.makedirs(args.out, exist_ok=True)

    # 1) Load processed data
    df = pd.read_parquet(args.data)
    print("Loaded data:", df.shape)
    # expect columns: cell_id, drug_id, drug_name, tissue, smiles, ln_ic50, PC1..PCk

    # 2) Build feature matrices
    # Expression PCs
    pc_cols = [c for c in df.columns if c.startswith("PC")]
    if not pc_cols:
        raise RuntimeError("No PC* columns found in the data. Did you run make_dataset.py?")
    X_expr = df[pc_cols].to_numpy(dtype=np.float32)

    # Drug features
    X_drug = build_drug_matrix(df["smiles"], n_bits=args.fp_bits)
    print("Expr shape:", X_expr.shape, "Drug featurization shape:", X_drug.shape)

    # Optional tissue one-hot
    if args.use_tissue:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        Xt = ohe.fit_transform(df[["tissue"]].fillna("unknown"))
        X = np.hstack([X_expr, X_drug, Xt]).astype(np.float32)
        print("Using tissue; Xt shape:", Xt.shape)
    else:
        X = np.hstack([X_expr, X_drug]).astype(np.float32)

    y = df["ln_ic50"].to_numpy(dtype=np.float32)

    # 3) Groups for CV
    if args.group_by == "drug":
        groups = df["drug_id"].astype(str).values
    elif args.group_by == "cell":
        groups = df["cell_id"].astype(str).values
    else:
        # no grouping: use indices
        groups = np.arange(len(df))

    # 4) Train CV
    best, fold_metrics, models = train_lgbm_cv(
        X, y, groups, n_trials=args.trials, seed=args.seed, n_splits=args.cv
    )

    # 5) Report metrics
    rmse_vals = [m["rmse"] for m in fold_metrics]
    mae_vals = [m["mae"] for m in fold_metrics]
    sp_vals = [m["spearman"] for m in fold_metrics]

    print(f"GroupKFold ({args.group_by}) CV:")
    print(f"  RMSE    mean±sd: {np.mean(rmse_vals):.4f} ± {np.std(rmse_vals):.4f}")
    print(f"  MAE     mean±sd: {np.mean(mae_vals):.4f} ± {np.std(mae_vals):.4f}")
    print(f"  Spearman mean±sd: {np.mean(sp_vals):.4f} ± {np.std(sp_vals):.4f}")

    # 6) Save artifacts
    pd.DataFrame(fold_metrics).to_csv(os.path.join(args.out, "cv_metrics.csv"), index=False)
    pd.Series(best).to_json(os.path.join(args.out, "best_params.json"), indent=2)
    # We don’t save all models to keep things light; you can add joblib if you want.


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/merged.parquet",
                    help="Path to processed merged parquet from make_dataset.py")
    ap.add_argument("--out", default="results/baseline_lgbm",
                    help="Output directory for metrics/params")
    ap.add_argument("--fp_bits", type=int, default=1024,
                    help="Number of bits in Morgan fingerprint")
    ap.add_argument("--trials", type=int, default=25,
                    help="Number of Optuna trials")
    ap.add_argument("--group_by", choices=["drug", "cell", "none"], default="drug",
                    help="Grouping for CV (leave-drug-out / leave-cell-out / no grouping)")
    ap.add_argument("--use_tissue", action="store_true",
                    help="Include tissue one-hot features")
    ap.add_argument("--cv", type=int, default=5, help="Number of CV folds")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args)
