import os
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Using device:", device)

    df = pd.read_parquet(args.data)
    print("Loaded merged data:", df.shape)

    # unique (drug_id, smiles)
    drugs = df[["drug_id", "smiles"]].drop_duplicates().reset_index(drop=True)
    print("Unique drugs:", len(drugs))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    model.to(device)
    model.eval()

    all_embs = []
    batch_smiles = []

    def encode_batch(smiles_list):
        with torch.no_grad():
            enc = tokenizer(
                smiles_list,
                padding=True,
                truncation=True,
                max_length=args.max_len,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            # use CLS token (index 0) representation
            cls = out.last_hidden_state[:, 0, :]
            return cls.cpu().numpy()

    for i in tqdm(range(0, len(drugs), args.batch_size), desc="Encoding drugs"):
        batch = drugs["smiles"].iloc[i : i + args.batch_size].tolist()
        embs = encode_batch(batch)
        all_embs.append(embs)

    feats = np.vstack(all_embs).astype(np.float32)
    print("ChemBERTa feature matrix:", feats.shape)  # (n_drugs, d)

    os.makedirs(os.path.dirname(args.out_npz), exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        feats=feats,
        drug_id=drugs["drug_id"].to_numpy(),
        smiles=drugs["smiles"].to_numpy(),
        model_name=args.model_name,
    )
    print("Saved:", args.out_npz)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/merged.parquet")
    ap.add_argument("--out_npz", default="data/processed/chemberta_drug_feats.npz")
    ap.add_argument("--model_name", default="seyonec/ChemBERTa-zinc-base-v1")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    main(args)
