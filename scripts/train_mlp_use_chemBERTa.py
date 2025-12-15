import argparse
import os
import warnings
from typing import List, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GroupKFold

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

warnings.filterwarnings("ignore", category=UserWarning)


# -----------------------
# Drug featurisation (Morgan + RDKit)
# -----------------------

def morgan_fp(smiles: str, n_bits: int = 1024, radius: int = 2) -> np.ndarray:
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


def build_drug_matrix_morgan(smiles_series: pd.Series, n_bits: int = 1024) -> np.ndarray:
    fps = []
    descs = []
    for s in smiles_series.fillna(""):
        fps.append(morgan_fp(s, n_bits=n_bits))
        descs.append(rdkit_desc(s))
    fps = np.asarray(fps, dtype=np.float32)
    descs = np.asarray(descs, dtype=np.float32)
    return np.hstack([fps, descs])


def build_drug_matrix_chemberta(df: pd.DataFrame, chemberta_npz: str) -> np.ndarray:
    """Map each row's drug_id to ChemBERTa embedding."""
    data = np.load(chemberta_npz)
    feats = data["feats"]          # (n_drugs, d)
    drug_ids = data["drug_id"]     # same length
    # build mapping drug_id -> row index
    id_to_idx = {int(d): i for i, d in enumerate(drug_ids)}
    idx = df["drug_id"].astype(int).map(id_to_idx).to_numpy()
    if np.any(pd.isna(idx)):
        raise ValueError("Some drug_id not found in ChemBERTa features.")
    idx = idx.astype(int)
    return feats[idx].astype(np.float32)


# -----------------------
# MLP model
# -----------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: Sequence[int], dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = []
        last = in_dim
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# -----------------------
# Metrics
# -----------------------

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def spearman(y_true, y_pred):
    try:
        return float(spearmanr(y_true, y_pred).correlation)
    except Exception:
        return float("nan")


# -----------------------
# Training on one train/val split
# -----------------------

def train_one_split(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    args,
    device: torch.device,
):
    train_ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    model = MLP(
        in_dim=X_tr.shape[1],
        hidden=[int(h) for h in args.hidden.split(",")],
        dropout=args.dropout,
    ).to(device)

    opt = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loss_fn = nn.MSELoss()

    best_val_rmse = float("inf")
    best_state = None
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        # validation
        model.eval()
        val_preds = []
        val_true = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                val_preds.append(preds.cpu().numpy())
                val_true.append(yb.cpu().numpy())
        val_preds = np.concatenate(val_preds)
        val_true = np.concatenate(val_true)
        val_rmse = rmse(val_true, val_preds)

        if val_rmse < best_val_rmse - 1e-4:
            best_val_rmse = val_rmse
            best_state = model.state_dict()
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_preds = model(torch.tensor(X_val, dtype=torch.float32, device=device))
        val_preds = val_preds.cpu().numpy()
    metrics = {
        "rmse": rmse(y_val, val_preds),
        "mae": mae(y_val, val_preds),
        "spearman": spearman(y_val, val_preds),
    }
    return model, metrics


# -----------------------
# Main
# -----------------------

def main(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print("Using device:", device)

    df = pd.read_parquet(args.data)
    print("Loaded data:", df.shape)

    # Expression PCs
    pc_cols = [c for c in df.columns if c.startswith("PC")]
    if not pc_cols:
        raise RuntimeError("No PC* columns found in data.")
    X_expr = df[pc_cols].to_numpy(dtype=np.float32)

    # Drug features: choose Morgan or ChemBERTa
    if args.chemberta_npz:
        print("Using ChemBERTa drug features from", args.chemberta_npz)
        X_drug = build_drug_matrix_chemberta(df, args.chemberta_npz)
    else:
        print("Using Morgan+RDKit drug features.")
        X_drug = build_drug_matrix_morgan(df["smiles"], n_bits=args.fp_bits)

    print("Expr shape:", X_expr.shape, "Drug features shape:", X_drug.shape)

    # Optional tissue one-hot
    if args.use_tissue:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        Xt = ohe.fit_transform(df[["tissue"]].fillna("unknown"))
        Xt = Xt.astype(np.float32)
        print("Using tissue; Xt shape:", Xt.shape)
        X = np.hstack([X_expr, X_drug, Xt]).astype(np.float32)
    else:
        X = np.hstack([X_expr, X_drug]).astype(np.float32)

    y = df["ln_ic50"].to_numpy(dtype=np.float32)

    # ----------------- CASE 1: fixed splits_dir -----------------
    if args.splits_dir:
        print("Using fixed splits from:", args.splits_dir)
        train_idx = np.load(os.path.join(args.splits_dir, "train_idx.npy"))
        val_idx = np.load(os.path.join(args.splits_dir, "val_idx.npy"))
        test_idx = np.load(os.path.join(args.splits_dir, "test_idx.npy"))

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        model, val_metrics = train_one_split(X_tr, y_tr, X_val, y_val, args, device)

        model.eval()
        with torch.no_grad():
            preds_test = (
                model(torch.tensor(X_test, dtype=torch.float32, device=device))
                .cpu()
                .numpy()
            )
        test_metrics = {
            "rmse": rmse(y_test, preds_test),
            "mae": mae(y_test, preds_test),
            "spearman": spearman(y_test, preds_test),
        }

        print("\nValidation metrics:", val_metrics)
        print("Test metrics      :", test_metrics)

        os.makedirs(args.out, exist_ok=True)
        pd.DataFrame(
            [
                {"split": "val", **val_metrics},
                {"split": "test", **test_metrics},
            ]
        ).to_csv(os.path.join(args.out, "metrics_val_test.csv"), index=False)

        np.save(os.path.join(args.out, "test_preds.npy"), preds_test)
        torch.save(model.state_dict(), os.path.join(args.out, "mlp_model.pt"))

    # ----------------- CASE 2: GroupKFold CV -----------------
    else:
        print("No splits_dir given → using GroupKFold CV")

        if args.group_by == "drug":
            groups = df["drug_id"].astype(str).values
        elif args.group_by == "cell":
            groups = df["cell_id"].astype(str).values
        else:
            groups = np.arange(len(df))

        gkf = GroupKFold(n_splits=args.cv)
        fold_metrics: List[dict] = []

        for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            print(f"\nFold {fold + 1}/{args.cv} (n_train={len(tr_idx)}, n_val={len(val_idx)})")
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            _, metrics = train_one_split(X_tr, y_tr, X_val, y_val, args, device)
            fold_metrics.append(metrics)
            print("  Fold metrics:", metrics)

        rmse_vals = [m["rmse"] for m in fold_metrics]
        mae_vals = [m["mae"] for m in fold_metrics]
        sp_vals = [m["spearman"] for m in fold_metrics]

        print("\nGroupKFold CV results:")
        print(f"  RMSE    mean±sd: {np.mean(rmse_vals):.4f} ± {np.std(rmse_vals):.4f}")
        print(f"  MAE     mean±sd: {np.mean(mae_vals):.4f} ± {np.std(mae_vals):.4f}")
        print(f"  Spearman mean±sd: {np.mean(sp_vals):.4f} ± {np.std(sp_vals):.4f}")

        os.makedirs(args.out, exist_ok=True)
        pd.DataFrame(fold_metrics).to_csv(os.path.join(args.out, "cv_metrics.csv"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/merged.parquet")
    ap.add_argument("--out", default="results/mlp_baseline")
    ap.add_argument("--fp_bits", type=int, default=1024)
    ap.add_argument("--group_by", choices=["drug", "cell", "none"], default="drug")
    ap.add_argument("--use_tissue", action="store_true")
    ap.add_argument("--hidden", default="1024,512")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument(
        "--splits_dir",
        default="",
        help="Directory with train_idx.npy / val_idx.npy / test_idx.npy. "
             "If empty, use GroupKFold CV.",
    )
    ap.add_argument(
        "--chemberta_npz",
        default="",
        help="If set, use ChemBERTa embeddings from this .npz instead of Morgan.",
    )
    args = ap.parse_args()
    main(args)
