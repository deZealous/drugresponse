import os, argparse, warnings
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from rdkit import Chem

warnings.filterwarnings("ignore", category=UserWarning)

def ci_map(df):  # case-insensitive column map
    return {c.lower(): c for c in df.columns}

def load_ic50(xlsx_path, tag):
    df = pd.read_excel(xlsx_path)
    c = ci_map(df)
    cols = [
        c.get("cell_line_name","CELL_LINE_NAME"),
        c.get("cosmic_id","COSMIC_ID"),
        c.get("drug_id","DRUG_ID"),
        c.get("drug_name","DRUG_NAME"),
        c.get("ln_ic50","LN_IC50"),
    ]
    df = df[cols].copy()
    df.columns = ["CELL_LINE_NAME","COSMIC_ID","DRUG_ID","DRUG_NAME","LN_IC50"]
    df["dataset"] = tag
    return df

def load_cells(path):
    df = pd.read_excel(path)
    print(f"Reading cell metadata from: {path}")
    print("Available columns:", list(df.columns))

    # Case-insensitive header map
    cols = {c.lower().strip(): c for c in df.columns}

    # --- COSMIC column ---
    cosmic_keys = [
        "cosmic identifier", "cosmic id", "cosmic_id", "cosmicid",
        "cosmic cell line id", "cosmic cell-line id"
    ]
    cosmic_col = next((cols[k] for k in cosmic_keys if k in cols), None)
    if cosmic_col is None:
        raise KeyError(f"No COSMIC column found. Got: {list(df.columns)}")

    # --- Tissue columns (GDSC Tissue descriptor 1/2, or fallbacks) ---
    # Check for keys with potential newlines or spaces
    gdsc_t1 = next((cols[k] for k in cols if "gdsc" in k and "tissue" in k and "descriptor" in k and ("1" in k or k.endswith("descriptor"))), None)
    gdsc_t2 = next((cols[k] for k in cols if "gdsc" in k and "tissue" in k and "descriptor" in k and "2" in k), None)
    other_tissue_keys = ["tissue descriptor","tissue","primary tissue","primary site","site","cancer type"]
    fallback_tissue = next((cols[k] for k in other_tissue_keys if k in cols), None)

    out = pd.DataFrame()
    out["COSMIC_ID"] = df[cosmic_col]

    if gdsc_t1 or gdsc_t2:
        t1 = df[gdsc_t1].astype(str) if gdsc_t1 else ""
        t2 = df[gdsc_t2].astype(str) if gdsc_t2 else ""
        t1s = t1.fillna("").str.strip()
        t2s = t2.fillna("").str.strip()
        use_both = (t2s != "") & (t2s.str.lower() != t1s.str.lower())
        tissue = np.where(use_both, t1s + " / " + t2s, t1s)
        out["tissue"] = pd.Series(tissue).replace({"": np.nan})
    elif fallback_tissue:
        out["tissue"] = df[fallback_tissue].astype(str)
    else:
        print("  load_cells: no tissue columns found; setting tissue='unknown'.")
        out["tissue"] = "unknown"

    out["tissue"] = out["tissue"].fillna("unknown").str.strip()
    out = out.drop_duplicates()
    print("Parsed columns -> COSMIC_ID + tissue (example):")
    print(out.head())
    return out

def canonical_smiles(s):
    try:
        mol = Chem.MolFromSmiles(s)
        if mol is None: return np.nan
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return np.nan

def winsorize(series, lo=0.005, hi=0.995):
    ql, qh = series.quantile([lo, hi])
    return series.clip(lower=ql, upper=qh)

def process_expression_zip(expr_zip, topk=5000, pcs=512):
    # GDSC1000: genes x cell_lines (or the reverse) tsv (can be plain text or zipped)
    # Try to read as plain text first, then try zip
    if not os.path.exists(expr_zip):
        zip_path = expr_zip + ".zip"
        if os.path.exists(zip_path):
            expr = pd.read_csv(zip_path, sep="\t", compression="zip")
        else:
            raise FileNotFoundError(f"Expression file not found: {expr_zip}")
    else:
        # File exists, try reading it
        print(f"Reading expression file: {expr_zip}")
        try:
            expr = pd.read_csv(expr_zip, sep="\t")
            print(f"Successfully read as plain text. Shape: {expr.shape}")
        except Exception as e:
            print(f"Plain text read failed: {e}")
            # If plain text fails, try as zip
            try:
                expr = pd.read_csv(expr_zip, sep="\t", compression="zip")
                print(f"Successfully read as zip. Shape: {expr.shape}")
            except Exception as e2:
                raise Exception(f"Failed to read {expr_zip} as both plain text and zip. Plain text error: {e}, Zip error: {e2}")
    
    gene_col = expr.columns[0]
    expr = expr.set_index(gene_col).T  # rows = cell lines (COSMIC), cols = genes
    expr.index.name = "cell_id"

    # numeric only
    expr = expr.apply(pd.to_numeric, errors="coerce")
    expr = expr.dropna(axis=1, how="any")

    # variance filter
    var = expr.var(axis=0)
    keep = var.sort_values(ascending=False).head(topk).index
    expr = expr[keep]

    # z-score then PCA
    scaler = StandardScaler()
    X = scaler.fit_transform(expr.values)
    pca = PCA(n_components=min(pcs, X.shape[1]), random_state=42)
    PCs = pca.fit_transform(X)

    expr_pca = pd.DataFrame(PCs, index=expr.index,
                            columns=[f"PC{i+1}" for i in range(pca.n_components_)])
    return expr_pca

def main(args):
    print(f"Using cell file: {os.path.join(args.raw_dir, args.cell_file)}")
    raw = args.raw_dir
    out = args.proc_dir
    os.makedirs(out, exist_ok=True)

    # 1) Responses
    gdsc1 = load_ic50(os.path.join(raw, "GDSC1_fitted_dose_response_27Oct23.xlsx"), "GDSC1")
    gdsc2 = load_ic50(os.path.join(raw, "GDSC2_fitted_dose_response_27Oct23.xlsx"), "GDSC2")
    resp = pd.concat([gdsc1, gdsc2], ignore_index=True)

    # 2) Compounds (SMILES) — use enriched mapping
    cmpd = pd.read_csv("data/processed/compounds_with_smiles.csv")[["DRUG_ID", "SMILES"]]
    resp = resp.merge(cmpd, on="DRUG_ID", how="left")

    # 3) Cell metadata (tissue)
    cells = load_cells(os.path.join(raw, args.cell_file))
    resp = resp.merge(cells, left_on="COSMIC_ID", right_on="COSMIC_ID", how="left")

    # 4) Basic cleaning — drop missing IC50; canonicalize SMILES; keep unmatched drugs
    resp = resp.dropna(subset=["LN_IC50"])
    resp["SMILES"] = resp["SMILES"].astype(str).replace({"nan": np.nan})
    resp["SMILES"] = resp["SMILES"].apply(canonical_smiles).fillna("")

    # (cell,drug) replicates  median ln_ic50
    resp = (resp
            .groupby(["COSMIC_ID","DRUG_ID","DRUG_NAME","SMILES","tissue","dataset"], as_index=False)["LN_IC50"]
            .median())

    # winsorize ln_ic50 to tame extreme tails
    resp["LN_IC50"] = winsorize(resp["LN_IC50"], 0.005, 0.995)

    # rename to final schema
    resp = resp.rename(columns={
        "COSMIC_ID":"cell_id","DRUG_ID":"drug_id","DRUG_NAME":"drug_name",
        "SMILES":"smiles","LN_IC50":"ln_ic50"
    })

    # save tidy pairs
    pairs_path = os.path.join(out, "gdsc_pairs.parquet")
    resp.to_parquet(pairs_path, index=False)

    # 5) Expression PCs (note: .zip!)
    expr_zip = os.path.join(raw, "Cell_line_RMA_proc_basalExp.txt")
    expr_pca = process_expression_zip(expr_zip, topk=args.top_genes, pcs=args.pcs)
    expr_path = os.path.join(out, "gdsc_expr_pca.parquet")
    expr_pca.to_parquet(expr_path)

    # 6) Merge
    merged = resp.merge(expr_pca, left_on="cell_id", right_index=True, how="inner")
    merged_path = os.path.join(out, "merged.parquet")
    merged.to_parquet(merged_path, index=False)

    # sanity prints
    print("pairs:", resp.shape, " ->", pairs_path)
    print("expr_pca:", expr_pca.shape, " ->", expr_path)
    print("merged:", merged.shape, " ->", merged_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--proc_dir", default="data/processed")
    ap.add_argument("--cell_file", default="Cell_Lines_Details.xlsx",
                    help="Name of the cell metadata Excel file in raw_dir")
    ap.add_argument("--top_genes", type=int, default=5000)
    ap.add_argument("--pcs", type=int, default=512)
    args = ap.parse_args()
    main(args)
