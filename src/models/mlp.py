"""
Multi-layer perceptron models for drug response prediction.
"""
import torch
import torch.nn as nn
from typing import List, Sequence


class MLP(nn.Module):
    """
    Standard MLP for regression tasks.
    
    Args:
        in_dim: Input feature dimension
        hidden: List of hidden layer sizes
        dropout: Dropout probability (default: 0.0)
    """
    def __init__(self, in_dim: int, hidden: Sequence[int], dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = []
        last = in_dim
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPHead(nn.Module):
    """
    MLP head for regression on concatenated features.
    Typically used after combining expression + drug embeddings.
    
    Args:
        in_dim: Input feature dimension
        hidden: List of hidden layer sizes
        dropout: Dropout probability (default: 0.1)
    """
    def __init__(self, in_dim: int, hidden: Sequence[int], dropout: float = 0.1):
        super().__init__()
        layers: List[nn.Module] = []
        last = in_dim
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z).squeeze(-1)
