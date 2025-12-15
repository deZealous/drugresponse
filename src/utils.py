"""
Utility functions for training and model management.
"""
import os
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, Any, Optional


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    
    Args:
        patience: Number of epochs to wait before stopping (default: 10)
        min_delta: Minimum change to qualify as improvement (default: 0.0)
        mode: 'min' for loss, 'max' for accuracy/correlation (default: 'min')
    
    Example:
        >>> early_stop = EarlyStopping(patience=5, mode='min')
        >>> for epoch in range(100):
        ...     val_loss = train_epoch()
        ...     if early_stop(val_loss):
        ...         print(f"Early stopping at epoch {epoch}")
        ...         break
    """
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'min':
            self.is_better = lambda new, best: new < best - min_delta
            self.best_score = float('inf')
        else:  # mode == 'max'
            self.is_better = lambda new, best: new > best + min_delta
            self.best_score = float('-inf')
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.early_stop = False
        if self.mode == 'min':
            self.best_score = float('inf')
        else:
            self.best_score = float('-inf')


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    metrics: Dict[str, float],
    filepath: str
):
    """
    Save model checkpoint with training state.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer (optional)
        epoch: Current epoch number
        metrics: Dictionary of metric values
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
    }
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: PyTorch model to load weights into
        optimizer: Optional optimizer to load state into
        device: Device to load tensors to
        
    Returns:
        Dictionary with checkpoint information
    """
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def save_metrics(metrics: Dict[str, Any], filepath: str, format: str = 'csv'):
    """
    Save metrics to file.
    
    Args:
        metrics: Dictionary of metrics
        filepath: Output file path
        format: 'csv' or 'json' (default: 'csv')
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if format == 'csv':
        df = pd.DataFrame([metrics])
        df.to_csv(filepath, index=False)
    elif format == 'json':
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_splits(splits_dir: str) -> tuple:
    """
    Load train/val/test split indices.
    
    Args:
        splits_dir: Directory containing split files
        
    Returns:
        Tuple of (train_idx, val_idx, test_idx) arrays
    """
    train_idx = np.load(os.path.join(splits_dir, "train_idx.npy"))
    val_idx = np.load(os.path.join(splits_dir, "val_idx.npy"))
    test_idx = np.load(os.path.join(splits_dir, "test_idx.npy"))
    return train_idx, val_idx, test_idx


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(cpu: bool = False) -> torch.device:
    """
    Get appropriate torch device.
    
    Args:
        cpu: Force CPU usage
        
    Returns:
        torch.device
    """
    if cpu or not torch.cuda.is_available():
        return torch.device('cpu')
    return torch.device('cuda')


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable
    }
