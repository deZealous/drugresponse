import os, re, time, json
import pandas as pd
import numpy as np
from urllib.parse import quote_plus
import requests
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# --- Config ---
RAW_PATH = "data/raw/screened_compounds_rel_8.5.csv"
OUT_PATH = "data/processed/compounds_with_smiles.csv"
CACHE_PATH = "data/processed/chem_name_to_smiles_cache.json"
TIMEOUT = 20

# --- Helpers ---
def norm(s):
    if pd.isna(s): return ""
    return re.sub(r"\s+", " ", str(s)).strip()

def parse_synonyms(s):
    if pd.isna(s) or not str(s).strip():
        return []
    # Split on comma/semicolon; strip whitespace
    parts = re.split(r"[;,]", str(s))
    return [norm(p) for p in parts if norm(p)]

def load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

class NotFound(Exception): pass

# ---------- ChEMBL (REST) ----------
@retry(reraise=True,
       stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=1, min=1, max=8),
       retry=retry_if_exception_type((requests.RequestException,)))
def chembl_search_name(name):
    """Return list of candidate molecules from ChEMBL name/synonym search."""
    url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/search.json?q={quote_plus(name)}&limit=10"
    r = requests.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return data.get("molecules", [])

@retry(reraise=True,
       stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=1, min=1, max=8),
       retry=retry_if_exception_type((requests.RequestException,)))
def chembl_get_molecule(chembl_id):
    url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json"
    r = requests.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def pick_best_chembl_match(name, molecules):
    """Pick the best match using exact-name preference and ChEMBL preferred name."""
    name_l = name.lower()
    # 1) exact match on pref_name
    exact = [m for m in molecules if str(m.get("pref_name","")).lower() == name_l]
    if exact: return exact[0]
    # 2) exact match on molecule synonyms (molecule_synonyms not returned in search; rely on search score order)
    # ChEMBL search already sorts reasonably, so pick first
    if molecules: return molecules[0]
    return None

def chembl_name_to_smiles(name):
    mols = chembl_search_name(name)
    best = pick_best_chembl_match(name, mols)
    if not best:
        raise NotFound()
    chembl_id = best["molecule_chembl_id"]
    mol = chembl_get_molecule(chembl_id)
    # structure fields may be under "molecule_structures"
    ms = mol.get("molecule_structures") or {}
    smiles = ms.get("canonical_smiles") or ms.get("standard_inchi_key")
    if not smiles:
        raise NotFound()
    return chembl_id, smiles

# ---------- PubChem fallback ----------
@retry(reraise=True,
       stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=1, min=1, max=8),
       retry=retry_if_exception_type((requests.RequestException,)))
def pubchem_name_to_smiles(name):
    # PUG REST: name to CID
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote_plus(name)}/property/CanonicalSMILES/JSON"
    r = requests.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    js = r.json()
    props = js.get("PropertyTable", {}).get("Properties", [])
    if not props:
        raise NotFound()
    return props[0]["CanonicalSMILES"]

def resolve_smiles_for_names(names, cache):
    """Try a list of names/synonyms; return (source, chembl_id_or_none, smiles) or (None,None,None)."""
    for nm in names:
        key = f"name::{nm.lower()}"
        if key in cache:
            v = cache[key]
            if v is None:  # negative cache
                continue
            return v["source"], v.get("chembl_id"), v["smiles"]

        # Try ChEMBL
        try:
            chembl_id, smiles = chembl_name_to_smiles(nm)
            cache[key] = {"source": "ChEMBL", "chembl_id": chembl_id, "smiles": smiles}
            return "ChEMBL", chembl_id, smiles
        except Exception:
            pass
        # Try PubChem
        try:
            smiles = pubchem_name_to_smiles(nm)
            cache[key] = {"source": "PubChem", "chembl_id": None, "smiles": smiles}
            return "PubChem", None, smiles
        except Exception:
            cache[key] = None  # negative cache and continue

    return None, None, None

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    cache = load_cache()

    df = pd.read_csv(RAW_PATH)
    # Normalize columns present in your file: DRUG_ID, DRUG_NAME, SYNONYMS, TARGET, TARGET_PATHWAY
    req = ["DRUG_ID", "DRUG_NAME"]
    for r in req:
        if r not in df.columns:
            raise SystemExit(f"Column {r} not found in {RAW_PATH}. Found: {df.columns.tolist()}")

    rows = []
    unmatched = []

    print(f"Resolving {len(df)} compounds to SMILES (with caching & retries)…")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        drug_id = int(row["DRUG_ID"])
        dn = norm(row.get("DRUG_NAME", ""))
        syns = parse_synonyms(row.get("SYNONYMS", ""))

        # Try order: DRUG_NAME, then synonyms
        cand_names = [n for n in [dn] + syns if n]
        source, chembl_id, smiles = resolve_smiles_for_names(cand_names, cache)

        if smiles:
            rows.append({
                "DRUG_ID": drug_id,
                "DRUG_NAME": dn,
                "ChEMBL_ID": chembl_id,
                "SMILES": smiles,
                "SOURCE": source
            })
        else:
            unmatched.append({"DRUG_ID": drug_id, "DRUG_NAME": dn})

        # be polite to APIs
        time.sleep(0.1)

    save_cache(cache)

    out = pd.DataFrame(rows).drop_duplicates(subset=["DRUG_ID"])
    out.to_csv(OUT_PATH, index=False)
    print(f"\n✅ Saved {len(out)} mappings to: {OUT_PATH}")

    if unmatched:
        um = pd.DataFrame(unmatched).drop_duplicates("DRUG_ID")
        miss_path = OUT_PATH.replace(".csv", "_UNMATCHED.csv")
        um.to_csv(miss_path, index=False)
        print(f"⚠️ {len(um)} drugs unmatched. See: {miss_path}")
        print("Tip: create a small overrides CSV with columns [DRUG_ID, SMILES] and we’ll merge it later.")

if __name__ == "__main__":
    main()
