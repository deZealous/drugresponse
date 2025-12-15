# scripts/make_splits.py

import argparse, os
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def make_group_split(df, group_col, test_size=0.2, val_size=0.1, seed=42):
    rng = np.random.RandomState(seed)

    groups = df[group_col].astype(str).values

    # 1) train+val vs test (no group leakage)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss1.split(df, groups=groups))

    # 2) within train+val, split train vs val
    df_trainval = df.iloc[trainval_idx].copy()
    groups_tv = df_trainval[group_col].astype(str).values
    # proportion of val relative to trainval
    val_rel = val_size / (1.0 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_rel, random_state=seed+1)
    train_idx_tv, val_idx_tv = next(gss2.split(df_trainval, groups=groups_tv))

    train_idx = trainval_idx[train_idx_tv]
    val_idx   = trainval_idx[val_idx_tv]

    return train_idx, val_idx, test_idx

def main(args):
    os.makedirs(args.out, exist_ok=True)
    df = pd.read_parquet(args.data)
    print("Loaded:", df.shape)

    if args.group_by not in ["drug_id", "cell_id"]:
        raise ValueError("--group_by must be 'drug_id' or 'cell_id'")

    train_idx, val_idx, test_idx = make_group_split(
        df,
        group_col=args.group_by,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
    )

    splits = {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }

    # Save as numpy + human-readable CSV with IDs
    np.save(os.path.join(args.out, "train_idx.npy"), train_idx)
    np.save(os.path.join(args.out, "val_idx.npy"),   val_idx)
    np.save(os.path.join(args.out, "test_idx.npy"),  test_idx)

    df_assign = pd.DataFrame({
        "idx": np.arange(len(df)),
        "cell_id": df["cell_id"],
        "drug_id": df["drug_id"],
        "split": "train",
    })
    df_assign.loc[val_idx, "split"] = "val"
    df_assign.loc[test_idx, "split"] = "test"
    df_assign.to_csv(os.path.join(args.out, "splits_assignment.csv"), index=False)

    print("Saved splits to", args.out)
    print("Counts:", df_assign["split"].value_counts())

if __name__ == "__main__":
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/merged.parquet")
    ap.add_argument("--out", default="data/processed/splits_drug")
    ap.add_argument("--group_by", default="drug_id", choices=["drug_id", "cell_id"])
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--val_size", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args)
