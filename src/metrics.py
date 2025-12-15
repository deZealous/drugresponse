"""
Evaluation metrics for drug response prediction models.
"""
import numpy as np
from scipy.stats import spearmanr, pearsonr
from typing import Union


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        RMSE score
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MAE score
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Spearman correlation coefficient.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Spearman correlation (returns NaN if calculation fails)
    """
    try:
        return float(spearmanr(y_true, y_pred).correlation)
    except Exception:
        return float("nan")


def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Pearson correlation coefficient.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Pearson correlation (returns NaN if calculation fails)
    """
    try:
        return float(pearsonr(y_true, y_pred)[0])
    except Exception:
        return float("nan")


def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate all metrics at once.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary with all metrics
    """
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "spearman": spearman(y_true, y_pred),
        "pearson": pearson(y_true, y_pred),
    }
