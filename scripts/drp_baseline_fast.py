import argparse
import os
import warnings
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb
import optuna

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------
# Drug featurisation
# ------------------------

def morgan_fp(smiles: str, n_bits: int = 1024, radius: int = 2) -> np.ndarray:
    if not isinstance(smiles, str) or smiles == "":
        return np.zeros(n_bits, dtype=np.uint8)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.uint8)
    gen = GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = gen.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def rdkit_desc(smiles: str) -> np.ndarray:
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


def build_drug_matrix_cached(smiles_series: pd.Series, n_bits: int = 1024) -> np.ndarray:
    """Compute RDKit features once per unique SMILES, reuse for all rows."""
    smiles_series = smiles_series.fillna("")
    unique_smiles = smiles_series.unique()

    fp_cache = {}
    desc_cache = {}

    for s in unique_smiles:
        fp_cache[s] = morgan_fp(s, n_bits=n_bits)
        desc_cache[s] = rdkit_desc(s)

    fps = []
    descs = []
    for s in smiles_series:
        fps.append(fp_cache[s])
        descs.append(desc_cache[s])

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
    n_trials: int = 10,
    seed: int = 42,
    n_splits: int = 3,
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

    df = pd.read_parquet(args.data)
    print("Loaded data:", df.shape)

    if args.max_rows > 0 and len(df) > args.max_rows:
        df = df.sample(n=args.max_rows, random_state=args.seed).reset_index(drop=True)
        print(f"Subsampled to {len(df)} rows")

    pc_cols = [c for c in df.columns if c.startswith("PC")]
    if not pc_cols:
        raise RuntimeError("No PC* columns found in the data.")
    X_expr = df[pc_cols].to_numpy(dtype=np.float32)

    X_drug = build_drug_matrix_cached(df["smiles"], n_bits=args.fp_bits)
    print("Expr shape:", X_expr.shape, "Drug featurization shape:", X_drug.shape)

    if args.use_tissue:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        Xt = ohe.fit_transform(df[["tissue"]].fillna("unknown"))
        X = np.hstack([X_expr, X_drug, Xt]).astype(np.float32)
        print("Using tissue; Xt shape:", Xt.shape)
    else:
        X = np.hstack([X_expr, X_drug]).astype(np.float32)

    y = df["ln_ic50"].to_numpy(dtype=np.float32)

    if args.group_by == "drug":
        groups = df["drug_id"].astype(str).values
    elif args.group_by == "cell":
        groups = df["cell_id"].astype(str).values
    else:
        groups = np.arange(len(df))

    best, fold_metrics, models = train_lgbm_cv(
        X, y, groups, n_trials=args.trials, seed=args.seed, n_splits=args.cv
    )

    rmse_vals = [m["rmse"] for m in fold_metrics]
    mae_vals = [m["mae"] for m in fold_metrics]
    sp_vals = [m["spearman"] for m in fold_metrics]

    print(f"GroupKFold ({args.group_by}) CV:")
    print(f"  RMSE    mean±sd: {np.mean(rmse_vals):.4f} ± {np.std(rmse_vals):.4f}")
    print(f"  MAE     mean±sd: {np.mean(mae_vals):.4f} ± {np.std(mae_vals):.4f}")
    print(f"  Spearman mean±sd: {np.mean(sp_vals):.4f} ± {np.std(sp_vals):.4f}")

    pd.DataFrame(fold_metrics).to_csv(os.path.join(args.out, "cv_metrics.csv"), index=False)
    pd.Series(best).to_json(os.path.join(args.out, "best_params.json"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/merged.parquet")
    ap.add_argument("--out", default="results/baseline_lgbm")
    ap.add_argument("--fp_bits", type=int, default=256)   # smaller than 1024
    ap.add_argument("--trials", type=int, default=10)     # fewer trials than 25
    ap.add_argument("--group_by", choices=["drug", "cell", "none"], default="drug")
    ap.add_argument("--use_tissue", action="store_true")
    ap.add_argument("--cv", type=int, default=3)          # fewer folds than 5
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_rows", type=int, default=0,    # 0 = use all rows
                    help="If >0, randomly subsample this many rows for a lighter run.")
    args = ap.parse_args()
    main(args)
