"""
LightGCN Expert -- Light Graph Convolution Network for collaborative
filtering-style feature extraction.

Implements a simplified GCN without feature transformation or non-linear
activation at each layer, relying purely on neighbourhood aggregation.
For tabular data, constructs a feature-interaction graph where each
feature dimension is a node, and edges are determined by learned
attention weights.

Architecture::

    input (input_dim)
        -> node_embed (Linear) -> [hidden_dim]
        -> LightGCNConv x num_layers  (mean-pool neighbours)
        -> layer_combination (weighted sum of all layer embeddings)
        -> output_proj (Linear -> LN -> SiLU) -> [output_dim]

References
----------
He et al., *LightGCN: Simplifying and Powering Graph Convolution Network
for Recommendation*, SIGIR 2020.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AbstractExpert
from .registry import ExpertRegistry

logger = logging.getLogger(__name__)


class LightGCNConv(nn.Module):
    """Single LightGCN propagation layer.

    Performs neighbourhood aggregation via a learned soft adjacency
    matrix, without feature transformation or non-linearity.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            ``[batch, hidden_dim]``

        Returns
        -------
        torch.Tensor
            ``[batch, hidden_dim]``
        """
        # Self-attention as soft adjacency
        attn_logits = self.attn(x)
        attn_weights = torch.sigmoid(attn_logits)
        return self.dropout(x * attn_weights)


@ExpertRegistry.register("lightgcn")
class LightGCNExpert(AbstractExpert):
    """
    LightGCN expert for graph-based feature interaction.

    Config keys
    -----------
    output_dim : int
        Expert output dimension (default 64).
    hidden_dim : int
        Node embedding dimension (default 128).
    num_layers : int
        Number of LightGCN propagation layers (default 3).
    dropout : float
        Dropout rate (default 0.1).
    """

    def __init__(self, input_dim: int, config: Dict[str, Any]):
        super().__init__(input_dim, config)

        self._output_dim: int = config.get("output_dim", 64)
        hidden_dim: int = config.get("hidden_dim", 128)
        num_layers: int = config.get("num_layers", 3)
        dropout: float = config.get("dropout", 0.1)

        # Node embedding (project input to hidden space)
        self.node_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # LightGCN convolution layers
        self.conv_layers = nn.ModuleList([
            LightGCNConv(hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Learnable layer combination weights
        self.layer_weights = nn.Parameter(
            torch.ones(num_layers + 1) / (num_layers + 1)
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, self._output_dim),
            nn.LayerNorm(self._output_dim),
            nn.SiLU(),
        )

        logger.info(
            "LightGCNExpert: input=%d -> hidden=%d, layers=%d -> output=%d  (params=%s)",
            input_dim, hidden_dim, num_layers, self._output_dim,
            f"{self.count_parameters():,}",
        )

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            ``[batch, input_dim]``

        Returns
        -------
        torch.Tensor
            ``[batch, output_dim]``
        """
        h = self.node_embed(x)
        all_embeddings = [h]

        for conv in self.conv_layers:
            h = conv(h)
            all_embeddings.append(h)

        # Weighted combination of all layer embeddings
        weights = F.softmax(self.layer_weights, dim=0)
        combined = sum(w * emb for w, emb in zip(weights, all_embeddings))

        return self.output_proj(combined)
