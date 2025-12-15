"""
Graph Neural Network models for molecular representation learning.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, global_mean_pool


class GINEncoder(nn.Module):
    """
    Graph Isomorphism Network (GIN) encoder for molecular graphs.
    
    Encodes molecular graph structures into fixed-size embeddings using
    graph convolutional layers followed by global mean pooling.
    
    Args:
        in_dim: Input node feature dimension
        hidden_dim: Hidden dimension for GIN layers (default: 128)
        num_layers: Number of GIN convolutional layers (default: 3)
    
    Example:
        >>> encoder = GINEncoder(in_dim=9, hidden_dim=128, num_layers=3)
        >>> mol_emb = encoder(node_features, edge_index, batch)
    """
    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        layers = []
        last = in_dim
        for _ in range(num_layers):
            # Each GIN layer has a 2-layer MLP
            mlp = nn.Sequential(
                nn.Linear(last, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            layers.append(GINConv(mlp))
            last = hidden_dim
        self.convs = nn.ModuleList(layers)
        self.act = nn.ReLU()
        self.out_dim = hidden_dim

    def forward(self, x, edge_index, batch):
        """
        Forward pass through GIN layers.
        
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment vector [num_nodes]
            
        Returns:
            Graph-level embeddings [batch_size, hidden_dim]
        """
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.act(x)
        # Global mean pool over nodes -> graph embedding
        return global_mean_pool(x, batch)
