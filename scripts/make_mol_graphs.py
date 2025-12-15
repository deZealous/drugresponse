import os
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from rdkit import Chem

import torch
from torch_geometric.data import Data


def atom_features(atom: Chem.rdchem.Atom) -> np.ndarray:
    """Very simple numeric atom features."""
    return np.array(
        [
            atom.GetAtomicNum(),                # Z
            atom.GetTotalDegree(),              # degree
            atom.GetFormalCharge(),             # formal charge
            atom.GetTotalNumHs(),               # hydrogens
            int(atom.GetIsAromatic()),          # aromatic flag
        ],
        dtype=np.float32,
    )


def mol_to_graph(smiles: str) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        # empty graph placeholder
        x = torch.zeros((1, 5), dtype=torch.float32)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        return Data(x=x, edge_index=edge_index)

    # nodes
    atom_list = [atom_features(a) for a in mol.GetAtoms()]
    if len(atom_list) == 0:
        # fallback for edge case
        x = torch.zeros((1, 5), dtype=torch.float32)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        return Data(x=x, edge_index=edge_index)
    
    x = np.stack(atom_list, axis=0)
    x = torch.tensor(x, dtype=torch.float32)

    # edges (undirected)
    rows = []
    cols = []
    for b in mol.GetBonds():
        u = b.GetBeginAtomIdx()
        v = b.GetEndAtomIdx()
        rows += [u, v]
        cols += [v, u]
    if len(rows) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor([rows, cols], dtype=torch.long)

    return Data(x=x, edge_index=edge_index)


def main(args):
    df = pd.read_parquet(args.data)
    print("Loaded merged data:", df.shape)

    drugs = df[["drug_id", "smiles"]].drop_duplicates().reset_index(drop=True)
    print("Unique drugs:", len(drugs))

    graphs = []
    for smi in tqdm(drugs["smiles"], desc="Building graphs"):
        g = mol_to_graph(str(smi))
        graphs.append(g)

    os.makedirs(os.path.dirname(args.out_pt), exist_ok=True)
    obj = {
        "graphs": graphs,
        "drug_id": drugs["drug_id"].to_numpy(),
        "smiles": drugs["smiles"].to_numpy(),
    }
    torch.save(obj, args.out_pt)
    print("Saved graphs to:", args.out_pt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/merged.parquet")
    ap.add_argument("--out_pt", default="data/processed/mol_graphs.pt")
    args = ap.parse_args()
    main(args)
