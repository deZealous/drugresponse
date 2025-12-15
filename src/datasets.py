"""
PyTorch dataset classes for drug response prediction.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, Union


class PairDataset(Dataset):
    """
    Dataset for drug-cell line pairs with expression and response data.
    
    Supports optional tissue features for multi-modal learning.
    
    Args:
        drug_idx: Drug indices for lookup in graph/embedding collections
        expr: Gene expression features [n_samples, n_features]
        y: Drug response values (e.g., ln_IC50)
        tissue: Optional tissue one-hot encodings [n_samples, n_tissues]
    
    Returns:
        If tissue is None: (drug_idx, expr, y)
        If tissue provided: (drug_idx, expr, tissue, y)
    
    Example:
        >>> dataset = PairDataset(
        ...     drug_idx=np.array([0, 1, 2]),
        ...     expr=np.random.randn(3, 50),
        ...     y=np.array([1.2, 3.4, 2.1])
        ... )
        >>> drug_idx, expr, y = dataset[0]
    """
    def __init__(
        self,
        drug_idx: np.ndarray,
        expr: np.ndarray,
        y: np.ndarray,
        tissue: Optional[np.ndarray] = None
    ):
        self.drug_idx = drug_idx.astype(np.int64)
        self.expr = expr.astype(np.float32)
        self.y = y.astype(np.float32)
        self.tissue = tissue.astype(np.float32) if tissue is not None else None

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> Union[Tuple[int, np.ndarray, float], 
                                             Tuple[int, np.ndarray, np.ndarray, float]]:
        if self.tissue is None:
            return self.drug_idx[i], self.expr[i], self.y[i]
        else:
            tissue_i = self.tissue[i]
            if tissue_i.ndim == 0:
                tissue_i = tissue_i.reshape(1)
            return self.drug_idx[i], self.expr[i], tissue_i, self.y[i]


class SimpleDataset(Dataset):
    """
    Simple dataset for feature matrix and target values.
    
    Used for MLP models with pre-computed features.
    
    Args:
        X: Feature matrix [n_samples, n_features]
        y: Target values [n_samples]
    
    Returns:
        (features, target) tuple
    
    Example:
        >>> dataset = SimpleDataset(
        ...     X=np.random.randn(100, 50),
        ...     y=np.random.randn(100)
        ... )
        >>> features, target = dataset[0]
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[i], self.y[i]


class SMILESDataset(Dataset):
    """
    Dataset for SMILES strings with expression and response data.
    
    Used with ChemBERTa models that require tokenized SMILES inputs.
    
    Args:
        smiles: SMILES strings
        expr: Gene expression features [n_samples, n_features]
        y: Drug response values
        tissue: Optional tissue one-hot encodings
    
    Note: This returns raw data; tokenization should be done in collate_fn
    
    Example:
        >>> dataset = SMILESDataset(
        ...     smiles=["CCO", "CC(=O)O", "c1ccccc1"],
        ...     expr=np.random.randn(3, 50),
        ...     y=np.array([1.2, 3.4, 2.1])
        ... )
    """
    def __init__(
        self,
        smiles: list,
        expr: np.ndarray,
        y: np.ndarray,
        tissue: Optional[np.ndarray] = None
    ):
        self.smiles = smiles
        self.expr = torch.from_numpy(expr.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.tissue = torch.from_numpy(tissue.astype(np.float32)) if tissue is not None else None

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        if self.tissue is None:
            return self.smiles[i], self.expr[i], self.y[i]
        else:
            return self.smiles[i], self.expr[i], self.tissue[i], self.y[i]
