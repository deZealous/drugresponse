# scripts/train_mlp.py

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# RDKit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------
# Drug featurisation
# ------------------------

def morgan_fp(smiles: str, n_bits: int = 1024, radius: int = 2) -> np.ndarray:
    """Return Morgan fingerprint bit vector for a SMILES string."""
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
    return np.hstack(
        [np.array(fps, dtype=np.float32), np.array(descs, dtype=np.float32)]
    )


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
# MLP model
# ------------------------

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(1024, 512), dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ------------------------
# Training utilities
# ------------------------

def train_one_fold(
    X_tr, y_tr, X_va, y_va, input_dim, args, device
):
    model = MLP(
        input_dim=input_dim,
        hidden_dims=tuple(args.hidden),
        dropout=args.dropout,
    ).to(device)

    train_ds = TensorDataset(
        torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).float()
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_va).float(), torch.from_numpy(y_va).float()
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_rmse = float("inf")
    best_state = None
    patience = args.patience
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        train_loss = running_loss / len(train_ds)

        # validation
        model.eval()
        preds_all = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                preds = model(xb)
                preds_all.append(preds.cpu().numpy())
        preds_all = np.concatenate(preds_all)
        val_rmse = rmse(y_va, preds_all)

        if val_rmse < best_val_rmse - 1e-4:
            best_val_rmse = val_rmse
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 10 == 0 or epochs_no_improve == patience:
            print(f"  epoch {epoch+1:03d} | train_loss={train_loss:.4f} | val_rmse={val_rmse:.4f}")
        if epochs_no_improve >= patience:
            break

    # load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # final val predictions and metrics
    model.eval()
    preds_all = []
    with torch.no_grad():
        for xb, _ in val_loader:
            xb = xb.to(device)
            preds = model(xb)
            preds_all.append(preds.cpu().numpy())
    preds_all = np.concatenate(preds_all)

    metrics = {
        "rmse": rmse(y_va, preds_all),
        "mae": mae(y_va, preds_all),
        "spearman": spearman(y_va, preds_all),
    }

    return model, metrics, preds_all


# ------------------------
# Main
# ------------------------

def main(args):
    os.makedirs(args.out, exist_ok=True)

    # 1) Load processed data
    df = pd.read_parquet(args.data)
    print("Loaded data:", df.shape)

    # 2) Build base features
    pc_cols = [c for c in df.columns if c.startswith("PC")]
    if not pc_cols:
        raise RuntimeError("No PC* columns found in dataset.")

    X_expr = df[pc_cols].to_numpy(dtype=np.float32)
    X_drug = build_drug_matrix(df["smiles"], n_bits=args.fp_bits)
    print("Expr shape:", X_expr.shape, "Drug shape:", X_drug.shape)

    # Optional tissue one-hot
    if args.use_tissue:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        Xt = ohe.fit_transform(df[["tissue"]].fillna("unknown"))
        Xt = Xt.astype(np.float32)
        print("Tissue one-hot shape:", Xt.shape)
        X = np.hstack([X_expr, X_drug, Xt]).astype(np.float32)
    else:
        X = np.hstack([X_expr, X_drug]).astype(np.float32)

    y = df["ln_ic50"].to_numpy(dtype=np.float32)

    # 3) Groups for CV
    if args.group_by == "drug":
        groups = df["drug_id"].astype(str).values
    elif args.group_by == "cell":
        groups = df["cell_id"].astype(str).values
    else:
        groups = np.arange(len(df))

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Using device:", device)

    gkf = GroupKFold(n_splits=args.cv)
    fold_metrics = []
    all_preds = np.zeros_like(y, dtype=np.float32)

    input_dim = X.shape[1]

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"\n=== Fold {fold} ===")
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]

        model, metrics, preds_va = train_one_fold(
            X_tr, y_tr, X_va, y_va, input_dim, args, device
        )
        fold_metrics.append(metrics)
        all_preds[va_idx] = preds_va

        # save model per fold (optional)
        torch.save(
            model.state_dict(),
            os.path.join(args.out, f"mlp_fold{fold}.pt"),
        )
        print(f"Fold {fold} metrics:", metrics)

    # aggregate metrics
    rmse_vals = [m["rmse"] for m in fold_metrics]
    mae_vals = [m["mae"] for m in fold_metrics]
    sp_vals = [m["spearman"] for m in fold_metrics]

    print("\nCV summary:")
    print(f"  RMSE    mean±sd: {np.mean(rmse_vals):.4f} ± {np.std(rmse_vals):.4f}")
    print(f"  MAE     mean±sd: {np.mean(mae_vals):.4f} ± {np.std(mae_vals):.4f}")
    print(f"  Spearman mean±sd: {np.mean(sp_vals):.4f} ± {np.std(sp_vals):.4f}")

    # save metrics and OOF predictions
    pd.DataFrame(fold_metrics).to_csv(
        os.path.join(args.out, "cv_metrics.csv"), index=False
    )
    pd.DataFrame(
        {
            "ln_ic50": y,
            "pred": all_preds,
            "cell_id": df["cell_id"],
            "drug_id": df["drug_id"],
        }
    ).to_csv(os.path.join(args.out, "oof_predictions.csv"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/merged.parquet")
    ap.add_argument("--out", default="results/mlp_baseline")
    ap.add_argument("--fp_bits", type=int, default=1024)
    ap.add_argument("--group_by", choices=["drug", "cell", "none"], default="drug")
    ap.add_argument("--use_tissue", action="store_true")

    # MLP hyperparams
    ap.add_argument("--hidden", type=lambda s: [int(x) for x in s.split(",")],
                    default="1024,512",
                    help="Comma-separated hidden layer sizes, e.g. '1024,512'")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    args = ap.parse_args()

    # parse hidden if string default
    if isinstance(args.hidden, str):
        args.hidden = [int(x) for x in args.hidden.split(",") if x]

    main(args)
