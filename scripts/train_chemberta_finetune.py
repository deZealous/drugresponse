import os
import argparse
import warnings
from typing import List, Sequence, Dict, Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import OneHotEncoder
from transformers import AutoTokenizer, AutoModel, AutoConfig

warnings.filterwarnings("ignore", category=UserWarning)


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
# Dataset / collate
# -----------------------

class ChemDataset(Dataset):
    def __init__(self, smiles, expr, y, tissue=None):
        self.smiles = smiles
        self.expr = expr.astype(np.float32)
        self.y = y.astype(np.float32)
        self.tissue = tissue.astype(np.float32) if tissue is not None else None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        if self.tissue is None:
            return {
                "smiles": self.smiles[i],
                "expr": self.expr[i],
                "y": self.y[i],
            }
        else:
            return {
                "smiles": self.smiles[i],
                "expr": self.expr[i],
                "tissue": self.tissue[i],
                "y": self.y[i],
            }


def make_collate(tokenizer, max_len: int):
    def collate_fn(batch: List[Dict[str, Any]]):
        smiles_list = [b["smiles"] for b in batch]
        enc = tokenizer(
            smiles_list,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        expr = torch.tensor(np.stack([b["expr"] for b in batch]), dtype=torch.float32)
        y = torch.tensor([b["y"] for b in batch], dtype=torch.float32)

        out = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "expr": expr,
            "y": y,
        }
        if "tissue" in batch[0]:
            tissue = torch.tensor(np.stack([b["tissue"] for b in batch]), dtype=torch.float32)
            out["tissue"] = tissue
        return out

    return collate_fn


# -----------------------
# Model
# -----------------------

class ChemBERTaRegressor(nn.Module):
    def __init__(self, model_name: str, expr_dim: int, tissue_dim: int = 0,
                 hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)

        chem_dim = self.config.hidden_size
        in_dim = chem_dim + expr_dim + tissue_dim

        layers: List[nn.Module] = [
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        ]
        self.head = nn.Sequential(*layers)

    def forward(self, input_ids, attention_mask, expr, tissue=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # CLS token embedding (index 0)
        pooled = out.last_hidden_state[:, 0, :]
        if tissue is not None:
            z = torch.cat([pooled, expr, tissue], dim=-1)
        else:
            z = torch.cat([pooled, expr], dim=-1)
        pred = self.head(z)
        return pred.squeeze(-1)


def freeze_all_bert(model: ChemBERTaRegressor):
    for p in model.encoder.parameters():
        p.requires_grad = False


def unfreeze_last_n_layers(model: ChemBERTaRegressor, n_layers: int):
    # keep embeddings frozen, unfreeze last n encoder layers
    if not hasattr(model.encoder, "encoder") or not hasattr(model.encoder.encoder, "layer"):
        # Some models may have different structure, but ChemBERTa is RoBERTa-like
        return
    encoder_layers = model.encoder.encoder.layer
    n_total = len(encoder_layers)
    n_layers = min(n_layers, n_total)
    for layer in encoder_layers[-n_layers:]:
        for p in layer.parameters():
            p.requires_grad = True


# -----------------------
# Train / eval
# -----------------------

def run_epoch(loader, model, device, optimizer=None, scaler=None):
    train_mode = optimizer is not None
    if train_mode:
        model.train()
    else:
        model.eval()

    all_pred = []
    all_true = []
    losses = []
    loss_fn = nn.MSELoss()

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        expr = batch["expr"].to(device)
        y = batch["y"].to(device)
        tissue = batch.get("tissue", None)
        if tissue is not None:
            tissue = tissue.to(device)

        with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
            pred = model(input_ids=input_ids,
                         attention_mask=attention_mask,
                         expr=expr,
                         tissue=tissue)
            loss = loss_fn(pred, y)

        if train_mode:
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

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
# Main
# -----------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Using device:", device)

    # 1) Load data
    df = pd.read_parquet(args.data)
    print("Loaded data:", df.shape)

    # Expression PCs
    pc_cols = [c for c in df.columns if c.startswith("PC")]
    if not pc_cols:
        raise RuntimeError("No PC* columns in data.")
    X_expr = df[pc_cols].to_numpy(dtype=np.float32)
    expr_dim = X_expr.shape[1]
    print("Expr dim:", expr_dim)

    # Tissue one-hot
    Xt = None
    tissue_dim = 0
    if args.use_tissue:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        Xt = ohe.fit_transform(df[["tissue"]].fillna("unknown")).astype(np.float32)
        tissue_dim = Xt.shape[1]
        print("Tissue dim:", tissue_dim)

    smiles = df["smiles"].astype(str).to_numpy()
    y = df["ln_ic50"].to_numpy(dtype=np.float32)

    if Xt is not None:
        X_full = (smiles, X_expr, Xt, y)
    else:
        X_full = (smiles, X_expr, None, y)

    # 2) Splits
    print("Using splits from:", args.splits_dir)
    train_idx = np.load(os.path.join(args.splits_dir, "train_idx.npy"))
    val_idx = np.load(os.path.join(args.splits_dir, "val_idx.npy"))
    test_idx = np.load(os.path.join(args.splits_dir, "test_idx.npy"))

    def make_subset(idx):
        if Xt is not None:
            return ChemDataset(
                smiles[idx],
                X_expr[idx],
                y[idx],
                Xt[idx],
            )
        else:
            return ChemDataset(
                smiles[idx],
                X_expr[idx],
                y[idx],
                None,
            )

    train_ds = make_subset(train_idx)
    val_ds = make_subset(val_idx)
    test_ds = make_subset(test_idx)

    # 3) Tokenizer, collate, dataloaders
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    collate_fn = make_collate(tokenizer, max_len=args.max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # 4) Model
    model = ChemBERTaRegressor(
        model_name=args.model_name,
        expr_dim=expr_dim,
        tissue_dim=tissue_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    # Freeze everything at start
    freeze_all_bert(model)

    # Two param groups: head & (later) unfrozen encoder layers
    # Initially only head is trainable
    def get_param_groups():
        head_params = [p for p in model.head.parameters() if p.requires_grad]
        encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
        param_groups = []
        if head_params:
            param_groups.append({"params": head_params, "lr": args.lr_head})
        if encoder_params:
            param_groups.append({"params": encoder_params, "lr": args.lr_bert})
        return param_groups

    optimizer = torch.optim.Adam(get_param_groups(), weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    best_val_rmse = float("inf")
    best_state = None
    patience_left = args.patience

    # 5) Training loop with gradual unfreezing
    for epoch in range(1, args.epochs + 1):
        # Unfreeze last n layers after freeze_epochs
        if epoch == args.freeze_epochs + 1 and args.unfreeze_layers > 0:
            print(f"Unfreezing last {args.unfreeze_layers} encoder layers at epoch {epoch}")
            unfreeze_last_n_layers(model, args.unfreeze_layers)
            # Rebuild optimizer with encoder params now trainable
            optimizer = torch.optim.Adam(get_param_groups(), weight_decay=args.weight_decay)

        train_metrics = run_epoch(train_loader, model, device, optimizer, scaler)
        val_metrics = run_epoch(val_loader, model, device, optimizer=None, scaler=None)

        print(
            f"Epoch {epoch:03d} | "
            f"train_rmse={train_metrics['rmse']:.4f} "
            f"val_rmse={val_metrics['rmse']:.4f} "
            f"val_spearman={val_metrics['spearman']:.4f}"
        )

        if val_metrics["rmse"] < best_val_rmse - 1e-4:
            best_val_rmse = val_metrics["rmse"]
            best_state = model.state_dict()
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # 6) Final eval on val + test
    val_metrics = run_epoch(val_loader, model, device, optimizer=None, scaler=None)
    test_metrics = run_epoch(test_loader, model, device, optimizer=None, scaler=None)

    print("\nValidation metrics:", val_metrics)
    print("Test metrics      :", test_metrics)

    os.makedirs(args.out, exist_ok=True)
    pd.DataFrame(
        [
            {"split": "val", **val_metrics},
            {"split": "test", **test_metrics},
        ]
    ).to_csv(os.path.join(args.out, "metrics_val_test.csv"), index=False)

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "model_name": args.model_name,
                "expr_dim": expr_dim,
                "tissue_dim": tissue_dim,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
            },
            "train_args": vars(args),
        },
        os.path.join(args.out, "chemberta_finetuned.pt"),
    )
    print("Saved model + metrics to:", args.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/merged.parquet")
    ap.add_argument("--splits_dir", required=True)
    ap.add_argument("--out", default="results/chemberta_finetune")

    ap.add_argument("--model_name", default="seyonec/ChemBERTa-zinc-base-v1")
    ap.add_argument("--use_tissue", action="store_true")

    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--freeze_epochs", type=int, default=2,
                    help="Epochs to train only the head before unfreezing encoder layers")
    ap.add_argument("--unfreeze_layers", type=int, default=2,
                    help="How many last encoder layers to unfreeze after freeze_epochs")

    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_bert", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--patience", type=int, default=3)

    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    main(args)
