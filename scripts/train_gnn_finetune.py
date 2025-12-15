import os
import argparse
import warnings
from typing import List, Sequence, Tuple, Dict, Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import OneHotEncoder

from torch_geometric.data import Batch
from torch_geometric.nn import GINConv, global_mean_pool

import optuna

warnings.filterwarnings("ignore", category=UserWarning)


# -----------------------
# GNN encoder
# -----------------------

class GINEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        layers = []
        last = in_dim
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(last, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            layers.append(GINConv(mlp))
            last = hidden_dim
        self.convs = nn.ModuleList(layers)
        self.act = nn.ReLU()
        self.out_dim = hidden_dim

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.act(x)
        # global mean pool over nodes -> graph embedding
        return global_mean_pool(x, batch)


# -----------------------
# MLP regressor on top of (expr + gnn + optional tissue)
# -----------------------

class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden: Sequence[int], dropout: float = 0.1):
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

    def forward(self, z):
        return self.net(z).squeeze(-1)


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
# Dataset of (drug_idx, expr, tissue_ohe, y)
# -----------------------

class PairDataset(Dataset):
    def __init__(self, drug_idx, expr, y, tissue=None):
        self.drug_idx = drug_idx.astype(np.int64)
        self.expr = expr.astype(np.float32)
        self.y = y.astype(np.float32)
        self.tissue = tissue.astype(np.float32) if tissue is not None else None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        if self.tissue is None:
            return self.drug_idx[i], self.expr[i], self.y[i]
        else:
            tissue_i = self.tissue[i]
            if tissue_i.ndim == 0:
                tissue_i = tissue_i.reshape(1)
            return self.drug_idx[i], self.expr[i], tissue_i, self.y[i]


# -----------------------
# Train / eval loop
# -----------------------

def run_epoch(loader, graphs, gnn, head, device, optim=None):
    train_mode = optim is not None
    if train_mode:
        gnn.train()
        head.train()
    else:
        gnn.eval()
        head.eval()

    losses = []
    all_pred = []
    all_true = []

    loss_fn = nn.MSELoss()

    for batch in loader:
        if len(batch) == 3:
            drug_idx, expr, y = batch
            tissue = None
        else:
            drug_idx, expr, tissue, y = batch
            tissue = tissue.to(device)

        drug_idx = drug_idx.to(device)
        expr = expr.to(device)
        y = y.to(device)

        # build batched graph from indices
        graphs_batch = [graphs[i] for i in drug_idx.cpu().numpy().tolist()]
        batch_graph = Batch.from_data_list(graphs_batch).to(device)

        mol_emb = gnn(batch_graph.x, batch_graph.edge_index, batch_graph.batch)

        # Concatenate features
        if tissue is not None:
            z = torch.cat([expr, mol_emb, tissue], dim=1)
        else:
            z = torch.cat([expr, mol_emb], dim=1)

        pred = head(z)

        loss = loss_fn(pred, y)
        if train_mode:
            optim.zero_grad()
            loss.backward()
            optim.step()

        losses.append(loss.item())
        all_pred.append(pred.detach().cpu().numpy())
        all_true.append(y.detach().cpu().numpy())

    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)

    metrics = {
        "rmse": rmse(all_true, all_pred),
        "mae": mae(all_true, all_pred),
        "spearman": spearman(all_true, all_pred),
        "loss": float(np.mean(losses)),
    }
    return metrics


# -----------------------
# Data prep (load once, reuse for all trials)
# -----------------------

def prepare_data(args) -> Tuple:
    # device is not needed here, everything is numpy / cpu
    df = pd.read_parquet(args.data)
    print("Loaded:", df.shape)

    # Expression PCs
    pc_cols = [c for c in df.columns if c.startswith("PC")]
    if not pc_cols:
        raise RuntimeError("No PC* columns in data.")
    X_expr = df[pc_cols].to_numpy(dtype=np.float32)
    expr_dim = X_expr.shape[1]

    # Load graphs
    graphs_obj = torch.load(args.graphs_pt, weights_only=False)
    graphs = graphs_obj["graphs"]
    g_drug_ids = graphs_obj["drug_id"]
    id_to_gidx = {int(d): i for i, d in enumerate(g_drug_ids)}
    drug_idx = df["drug_id"].astype(int).map(id_to_gidx).to_numpy()
    if np.any(pd.isna(drug_idx)):
        raise ValueError("Some drug_id not found in graphs.")
    drug_idx = drug_idx.astype(np.int64)
    in_dim_atom = graphs[0].x.shape[1]

    # Tissue one-hot (optional)
    tissue_ohe = None
    if args.use_tissue:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        tissue_ohe = ohe.fit_transform(df[["tissue"]].fillna("unknown")).astype(np.float32)
        print("Tissue one-hot shape:", tissue_ohe.shape)
        tissue_dim = tissue_ohe.shape[1]
    else:
        tissue_dim = 0

    y = df["ln_ic50"].to_numpy(dtype=np.float32)

    # Splits
    print("Using splits from:", args.splits_dir)
    train_idx = np.load(os.path.join(args.splits_dir, "train_idx.npy"))
    val_idx = np.load(os.path.join(args.splits_dir, "val_idx.npy"))
    test_idx = np.load(os.path.join(args.splits_dir, "test_idx.npy"))

    def subset(idx):
        if tissue_ohe is None:
            return (
                drug_idx[idx],
                X_expr[idx],
                None,
                y[idx],
            )
        else:
            return (
                drug_idx[idx],
                X_expr[idx],
                tissue_ohe[idx],
                y[idx],
            )

    tr_parts = subset(train_idx)
    val_parts = subset(val_idx)
    te_parts = subset(test_idx)

    if tissue_ohe is None:
        tr_ds = PairDataset(*tr_parts[:3])
        val_ds = PairDataset(*val_parts[:3])
        te_ds = PairDataset(*te_parts[:3])
    else:
        tr_ds = PairDataset(tr_parts[0], tr_parts[1], tr_parts[3], tissue=tr_parts[2])
        val_ds = PairDataset(val_parts[0], val_parts[1], val_parts[3], tissue=val_parts[2])
        te_ds = PairDataset(te_parts[0], te_parts[1], te_parts[3], tissue=te_parts[2])

    # Collate fn (works with or without tissue)
    def collate_fn(batch):
        if len(batch[0]) == 3:
            drug_idx_b, expr_b, y_b = zip(*batch)
            return (
                torch.from_numpy(np.stack(drug_idx_b)),
                torch.from_numpy(np.stack(expr_b)),
                torch.from_numpy(np.stack(y_b)),
            )
        else:
            drug_idx_b, expr_b, tissue_b, y_b = zip(*batch)
            tissue_stacked = np.stack(tissue_b)
            return (
                torch.from_numpy(np.stack(drug_idx_b)),
                torch.from_numpy(np.stack(expr_b)),
                torch.from_numpy(tissue_stacked),
                torch.from_numpy(np.stack(y_b)),
            )

    return (
        graphs,
        tr_ds,
        val_ds,
        te_ds,
        collate_fn,
        in_dim_atom,
        expr_dim,
        tissue_dim,
    )


# -----------------------
# Train + evaluate once (for fixed hyperparams)
# -----------------------

def train_and_eval(
    params: Dict[str, Any],
    graphs,
    tr_ds,
    val_ds,
    te_ds,
    collate_fn,
    in_dim_atom: int,
    expr_dim: int,
    tissue_dim: int,
    device,
    save_dir: str = None,
    save_model: bool = False,
    verbose: bool = True,
) -> Tuple[Dict[str, float], Dict[str, float]]:

    batch_size = params["batch_size"]
    epochs = params["epochs"]
    patience = params["patience"]

    train_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    gnn = GINEncoder(
        in_dim=in_dim_atom,
        hidden_dim=params["gnn_hidden"],
        num_layers=params["num_layers"],
    ).to(device)

    if tissue_dim == 0:
        head_in_dim = expr_dim + gnn.out_dim
    else:
        head_in_dim = expr_dim + gnn.out_dim + tissue_dim

    head = MLPHead(
        head_in_dim,
        hidden=params["mlp_hidden"],
        dropout=params["dropout"],
    ).to(device)

    optim = torch.optim.Adam(
        list(gnn.parameters()) + list(head.parameters()),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )

    best_val_rmse = float("inf")
    best_state = None
    patience_left = patience

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(train_loader, graphs, gnn, head, device, optim)
        val_metrics = run_epoch(val_loader, graphs, gnn, head, device, optim=None)

        if verbose:
            print(
                f"Epoch {epoch:03d} | "
                f"train_rmse={train_metrics['rmse']:.4f} "
                f"val_rmse={val_metrics['rmse']:.4f} "
                f"val_spearman={val_metrics['spearman']:.4f}"
            )

        if val_metrics["rmse"] < best_val_rmse - 1e-4:
            best_val_rmse = val_metrics["rmse"]
            best_state = {
                "gnn": gnn.state_dict(),
                "head": head.state_dict(),
            }
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                if verbose:
                    print("Early stopping.")
                break

    if best_state is not None:
        gnn.load_state_dict(best_state["gnn"])
        head.load_state_dict(best_state["head"])

    val_metrics = run_epoch(val_loader, graphs, gnn, head, device, optim=None)
    test_metrics = run_epoch(test_loader, graphs, gnn, head, device, optim=None)

    if verbose:
        print("\nValidation metrics:", val_metrics)
        print("Test metrics      :", test_metrics)

    if save_dir is not None and save_model:
        os.makedirs(save_dir, exist_ok=True)
        pd.DataFrame(
            [
                {"split": "val", **val_metrics},
                {"split": "test", **test_metrics},
            ]
        ).to_csv(os.path.join(save_dir, "metrics_val_test.csv"), index=False)

        torch.save(
            {
                "gnn": gnn.state_dict(),
                "head": head.state_dict(),
                "in_dim_atom": in_dim_atom,
                "expr_dim": expr_dim,
                "params": params,
            },
            os.path.join(save_dir, "gnn_model.pt"),
        )

    return val_metrics, test_metrics


# -----------------------
# Main
# -----------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Using device:", device)

    (
        graphs,
        tr_ds,
        val_ds,
        te_ds,
        collate_fn,
        in_dim_atom,
        expr_dim,
        tissue_dim,
    ) = prepare_data(args)

    if not args.tune:
        # Just run once with CLI hyperparams (baseline behaviour)
        hidden = [int(h) for h in args.mlp_hidden.split(",")]
        params = {
            "gnn_hidden": args.gnn_hidden,
            "num_layers": 3,
            "mlp_hidden": hidden,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
        }
        train_and_eval(
            params,
            graphs,
            tr_ds,
            val_ds,
            te_ds,
            collate_fn,
            in_dim_atom,
            expr_dim,
            tissue_dim,
            device,
            save_dir=args.out,
            save_model=True,
            verbose=True,
        )
        return

    # -----------------------
    # Optuna tuning branch
    # -----------------------
    print(f"Running Optuna tuning for {args.n_trials} trials...")

    def objective(trial: optuna.Trial):
        gnn_hidden = trial.suggest_categorical("gnn_hidden", [64, 128, 256, 512])
        num_layers = trial.suggest_int("num_layers", 2, 4)
        mlp_width = trial.suggest_categorical("mlp_width", [128, 256, 512, 1024])
        mlp_layers = trial.suggest_int("mlp_layers", 1, 3)
        mlp_hidden = [mlp_width] * mlp_layers
        dropout = trial.suggest_float("dropout", 0.0, 0.4)
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])

        params = {
            "gnn_hidden": gnn_hidden,
            "num_layers": num_layers,
            "mlp_hidden": mlp_hidden,
            "dropout": dropout,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "epochs": args.tune_epochs,
            "patience": args.patience,
        }

        val_metrics, _ = train_and_eval(
            params,
            graphs,
            tr_ds,
            val_ds,
            te_ds,
            collate_fn,
            in_dim_atom,
            expr_dim,
            tissue_dim,
            device,
            save_dir=None,
            save_model=False,
            verbose=False,
        )
        return val_metrics["rmse"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.n_trials)

    print("Best trial:", study.best_trial.number)
    print("Best RMSE :", study.best_value)
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Retrain with best params using full epochs and save
    best = study.best_params
    best_params = {
        "gnn_hidden": best["gnn_hidden"],
        "num_layers": best["num_layers"],
        "mlp_hidden": [best["mlp_width"]] * best["mlp_layers"],
        "dropout": best["dropout"],
        "lr": best["lr"],
        "weight_decay": best["weight_decay"],
        "batch_size": best["batch_size"],
        "epochs": args.epochs,
        "patience": args.patience,
    }

    print("\nRetraining final GNN with best hyperparameters...")
    train_and_eval(
        best_params,
        graphs,
        tr_ds,
        val_ds,
        te_ds,
        collate_fn,
        in_dim_atom,
        expr_dim,
        tissue_dim,
        device,
        save_dir=args.out,
        save_model=True,
        verbose=True,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/merged.parquet")
    ap.add_argument("--graphs_pt", default="data/processed/mol_graphs.pt")
    ap.add_argument("--splits_dir", required=True)
    ap.add_argument("--out", default="results/gnn_baseline")

    ap.add_argument("--use_tissue", action="store_true")

    # Baseline hyperparams (also used as defaults for final retrain)
    ap.add_argument("--gnn_hidden", type=int, default=128)
    ap.add_argument("--mlp_hidden", default="256,128")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=8)

    ap.add_argument("--cpu", action="store_true")

    # Tuning options
    ap.add_argument("--tune", action="store_true",
                    help="If set, run Optuna hyperparameter search instead of a single run.")
    ap.add_argument("--n_trials", type=int, default=30,
                    help="Number of Optuna trials.")
    ap.add_argument("--tune_epochs", type=int, default=30,
                    help="Epochs per trial during tuning (smaller than full epochs).")

    args = ap.parse_args()
    main(args)
