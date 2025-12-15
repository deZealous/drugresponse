import os
import argparse
import warnings
from typing import List, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import optuna

from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", category=UserWarning)


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
# Training for one config
# -----------------------

def train_one(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden: Sequence[int],
    dropout: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    patience: int,
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
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )

    model = MLP(in_dim=X_tr.shape[1], hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val_rmse = float("inf")
    best_state = None
    patience_left = patience

    for epoch in range(1, epochs + 1):
        # train
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        # validate
        model.eval()
        val_preds = []
        val_true = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                val_preds.append(pred.cpu().numpy())
                val_true.append(yb.cpu().numpy())
        val_preds = np.concatenate(val_preds)
        val_true = np.concatenate(val_true)
        val_rmse = rmse(val_true, val_preds)

        if val_rmse < best_val_rmse - 1e-4:
            best_val_rmse = val_rmse
            best_state = model.state_dict()
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_rmse


# -----------------------
# Main
# -----------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Using device:", device)

    # 1) Load data
    df = pd.read_parquet(args.data)
    print("Loaded:", df.shape)

    # Expression PCs
    pc_cols = [c for c in df.columns if c.startswith("PC")]
    if not pc_cols:
        raise RuntimeError("No PC* columns in data.")
    X_expr = df[pc_cols].to_numpy(dtype=np.float32)
    expr_dim = X_expr.shape[1]
    print("Expr dim:", expr_dim)

    # 2) Load ChemBERTa embeddings
    chem = np.load(args.chemberta_npz)
    feats = chem["feats"]          # (n_drugs, d)
    drug_ids = chem["drug_id"]

    id_to_idx = {int(d): i for i, d in enumerate(drug_ids)}
    idx = df["drug_id"].astype(int).map(id_to_idx).to_numpy()
    if np.any(pd.isna(idx)):
        raise ValueError("Some drug_id not found in ChemBERTa embeddings.")
    idx = idx.astype(int)
    X_drug = feats[idx].astype(np.float32)
    drug_dim = X_drug.shape[1]
    print("Drug dim:", drug_dim)

    # 3) Tissue one-hot
    Xt = None
    if args.use_tissue:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        Xt = ohe.fit_transform(df[["tissue"]].fillna("unknown")).astype(np.float32)
        print("Tissue dim:", Xt.shape[1])

    if Xt is not None:
        X = np.hstack([X_expr, X_drug, Xt]).astype(np.float32)
    else:
        X = np.hstack([X_expr, X_drug]).astype(np.float32)

    y = df["ln_ic50"].to_numpy(dtype=np.float32)

    # 4) Splits
    print("Using splits:", args.splits_dir)
    train_idx = np.load(os.path.join(args.splits_dir, "train_idx.npy"))
    val_idx = np.load(os.path.join(args.splits_dir, "val_idx.npy"))
    test_idx = np.load(os.path.join(args.splits_dir, "test_idx.npy"))

    X_tr, y_tr = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print("Train/Val/Test sizes:", len(X_tr), len(X_val), len(X_test))

    # 5) Optuna objective
    def objective(trial: optuna.Trial):
        # hidden dims: e.g. 1–3 layers
        n_layers = trial.suggest_int("n_layers", 1, 3)
        hidden = []
        for i in range(n_layers):
            h = trial.suggest_int(f"hidden_{i}", 128, 1024, log=True)
            hidden.append(h)

        dropout = trial.suggest_float("dropout", 0.0, 0.4)
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
        epochs = args.epochs
        patience = args.patience

        model, best_val_rmse = train_one(
            X_tr,
            y_tr,
            X_val,
            y_val,
            hidden=hidden,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            device=device,
        )

        # you can add pruning if you like
        trial.report(best_val_rmse, step=0)
        return best_val_rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    print("Best trial:", study.best_trial.number)
    print("Best value (val RMSE):", study.best_value)
    print("Best params:", study.best_params)

    os.makedirs(args.out, exist_ok=True)
    study.trials_dataframe().to_csv(os.path.join(args.out, "optuna_trials.csv"), index=False)

    # 6) Retrain with best params on train+val, evaluate on test
    bp = study.best_params
    n_layers = bp["n_layers"]
    best_hidden = [bp[f"hidden_{i}"] for i in range(n_layers)]
    dropout = bp["dropout"]
    lr = bp["lr"]
    weight_decay = bp["weight_decay"]
    batch_size = bp["batch_size"]

    X_train_full = np.vstack([X_tr, X_val])
    y_train_full = np.concatenate([y_tr, y_val])

    model, _ = train_one(
        X_train_full,
        y_train_full,
        X_val=X_val,   # val not used for early stop now, but okay
        y_val=y_val,
        hidden=best_hidden,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        epochs=args.final_epochs,
        patience=args.patience,
        device=device,
    )

    # evaluate on test
    model.eval()
    with torch.no_grad():
        preds_test = model(torch.tensor(X_test, dtype=torch.float32, device=device)).cpu().numpy()

    test_metrics = {
        "rmse": rmse(y_test, preds_test),
        "mae": mae(y_test, preds_test),
        "spearman": spearman(y_test, preds_test),
    }
    print("Final test metrics:", test_metrics)

    pd.DataFrame(
        [
            {"split": "test", **test_metrics},
        ]
    ).to_csv(os.path.join(args.out, "final_test_metrics.csv"), index=False)

    # save model + best params
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": X.shape[1],
            "best_params": bp,
        },
        os.path.join(args.out, "mlp_chemberta_best.pt"),
    )
    print("Saved artifacts to:", args.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/merged.parquet")
    ap.add_argument("--chemberta_npz", default="data/processed/chemberta_drug_feats.npz")
    ap.add_argument("--splits_dir", required=True)
    ap.add_argument("--out", default="results/mlp_chemberta_tuned")

    ap.add_argument("--use_tissue", action="store_true")

    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--epochs", type=int, default=20, help="epochs per trial")
    ap.add_argument("--final_epochs", type=int, default=40, help="epochs for final retrain")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    main(args)
