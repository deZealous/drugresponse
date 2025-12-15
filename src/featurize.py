import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

def morgan_fp(smiles: str, nbits=2048, radius=2):
    if not smiles:
        return np.zeros(nbits, dtype=np.float32)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nbits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros((nbits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr
