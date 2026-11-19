"""Architektury sieci neuronowych do klasyfikacji sentymentu."""
import torch
from torch import nn

from .preprocessing import PAD_IDX


class MeanEmbeddingClassifier(nn.Module):
    """Model bazowy (etap 1).

    Embedding -> średnia z wektorów słów (bez <PAD>) -> Linear + ReLU + Dropout
    -> Linear -> 1 logit. Prawdopodobieństwo klasy pozytywnej = sigmoid(logit).

    Model nie zna kolejności słów ("bag of words"), więc słabo radzi sobie
    z negacjami typu "not good" — to motywacja dla LSTM w etapie 2.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 32, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = (x != PAD_IDX).unsqueeze(-1).float()          # (batch, len, 1)
        emb = self.embedding(x) * mask                         # (batch, len, embed)
        lengths = mask.sum(dim=1).clamp(min=1.0)               # (batch, 1)
        pooled = emb.sum(dim=1) / lengths                      # (batch, embed)
        hidden = self.dropout(torch.relu(self.fc1(pooled)))
        return self.fc2(hidden).squeeze(-1)                    # (batch,) logity


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
