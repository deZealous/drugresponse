"""
ChemBERTa-based models for chemical representation learning.
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import List, Optional


class ChemBERTaRegressor(nn.Module):
    """
    ChemBERTa transformer model for drug response prediction.
    
    Combines ChemBERTa molecular embeddings with gene expression and
    optional tissue features for regression.
    
    Args:
        model_name: HuggingFace model name (e.g., "seyonec/ChemBERTa-zinc-base-v1")
        expr_dim: Gene expression feature dimension
        tissue_dim: Tissue one-hot encoding dimension (default: 0)
        hidden_dim: Hidden layer size for regression head (default: 256)
        dropout: Dropout probability (default: 0.1)
    
    Example:
        >>> model = ChemBERTaRegressor(
        ...     model_name="seyonec/ChemBERTa-zinc-base-v1",
        ...     expr_dim=50,
        ...     tissue_dim=20,
        ...     hidden_dim=256
        ... )
    """
    def __init__(
        self, 
        model_name: str, 
        expr_dim: int, 
        tissue_dim: int = 0,
        hidden_dim: int = 256, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)

        chem_dim = self.config.hidden_size
        in_dim = chem_dim + expr_dim + tissue_dim

        layers: List[nn.Module] = [
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        ]
        self.head = nn.Sequential(*layers)

    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        expr: torch.Tensor, 
        tissue: Optional[torch.Tensor] = None
    ):
        """
        Forward pass through ChemBERTa encoder and regression head.
        
        Args:
            input_ids: Tokenized SMILES [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            expr: Gene expression features [batch_size, expr_dim]
            tissue: Optional tissue features [batch_size, tissue_dim]
            
        Returns:
            Predicted drug response [batch_size]
        """
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # CLS token embedding (index 0)
        pooled = out.last_hidden_state[:, 0, :]
        if tissue is not None:
            z = torch.cat([pooled, expr, tissue], dim=-1)
        else:
            z = torch.cat([pooled, expr], dim=-1)
        pred = self.head(z)
        return pred.squeeze(-1)


def freeze_all_bert(model: ChemBERTaRegressor):
    """
    Freeze all ChemBERTa encoder parameters.
    
    Args:
        model: ChemBERTaRegressor instance
    """
    for param in model.encoder.parameters():
        param.requires_grad = False


def unfreeze_all_bert(model: ChemBERTaRegressor):
    """
    Unfreeze all ChemBERTa encoder parameters.
    
    Args:
        model: ChemBERTaRegressor instance
    """
    for param in model.encoder.parameters():
        param.requires_grad = True


def freeze_bert_layers(model: ChemBERTaRegressor, num_layers_to_freeze: int):
    """
    Freeze bottom N layers of ChemBERTa encoder.
    
    Args:
        model: ChemBERTaRegressor instance
        num_layers_to_freeze: Number of bottom layers to freeze
    """
    # Freeze embeddings
    for param in model.encoder.embeddings.parameters():
        param.requires_grad = False
    
    # Freeze first N layers
    for i in range(num_layers_to_freeze):
        if i < len(model.encoder.encoder.layer):
            for param in model.encoder.encoder.layer[i].parameters():
                param.requires_grad = False
