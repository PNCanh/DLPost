import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualMLP(nn.Module):
    """
    Residual MLP Head: feature -> linear -> GELU -> drop out -> Linear -> Residual -> Classifier
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout_prob: float = 0.1):
        super(ResidualMLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_prob)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.classifier = nn.Linear(input_dim, output_dim)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # feature -> linear -> GELU -> drop out
        residual = x
        h = self.linear1(x)
        h = self.act(h)
        h = self.dropout(h)
        # Linear -> Residual
        h = self.linear2(h)
        h = self.layer_norm(h + residual)
        # Classifier
        out = self.classifier(h)
        return out
