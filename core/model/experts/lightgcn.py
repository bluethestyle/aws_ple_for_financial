"""
LightGCN Expert -- Refine MLP for pre-computed graph embeddings.

The actual graph convolution (LightGCN propagation) is performed offline
in Phase 0 by the GraphEmbeddingGenerator.  At training time this expert
simply refines the pre-computed collaborative filtering embeddings with
a lightweight MLP + residual connection.

This matches the on-prem design where:
  Phase 0: GraphEmbeddingGenerator computes LightGCN embeddings offline
  Phase 1: LightGCNExpert refines them with MLP + residual

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

from .base import AbstractExpert
from .registry import ExpertRegistry

logger = logging.getLogger(__name__)


@ExpertRegistry.register("lightgcn")
class LightGCNExpert(AbstractExpert):
    """
    LightGCN expert that refines pre-computed graph embeddings.

    Graph convolution is done offline in Phase 0.  This expert applies
    a simple refine MLP with residual connection to adapt the embeddings
    for downstream multi-task prediction.

    Config keys
    -----------
    output_dim : int
        Expert output dimension (default 64).
    hidden_dim : int
        MLP hidden dimension (default 128).
    dropout : float
        Dropout rate (default 0.1).
    """

    INTERPRET_DIM = 4

    def __init__(self, input_dim: int, config: Dict[str, Any]):
        super().__init__(input_dim, config)

        self._output_dim: int = config.get("output_dim", 64)
        hidden_dim: int = config.get("hidden_dim", 128)
        dropout: float = config.get("dropout", 0.1)

        # Simple refine MLP on pre-computed graph embeddings
        self.refine_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self._output_dim),
            nn.LayerNorm(self._output_dim),
        )

        # Residual projection if dims differ
        if input_dim != self._output_dim:
            self.skip = nn.Linear(input_dim, self._output_dim)
        else:
            self.skip = nn.Identity()

        # Interpretable projection (4D)
        self.interpret_proj = nn.Linear(self._output_dim, self.INTERPRET_DIM)
        self._interpretable_scores = None

        # Pre-computed embeddings are flat, not sequential
        self.expects_sequence = False

        logger.info(
            "LightGCNExpert: input=%d -> hidden=%d -> output=%d, dropout=%.2f  "
            "(refine MLP on pre-computed graph embeddings, params=%s)",
            input_dim, hidden_dim, self._output_dim, dropout,
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
            ``[batch, input_dim]`` -- pre-computed graph embeddings
            from Phase 0 GraphEmbeddingGenerator.

        Returns
        -------
        torch.Tensor
            ``[batch, output_dim]``
        """
        residual = self.skip(x)
        out = self.refine_mlp(x) + residual
        self._interpretable_scores = self.interpret_proj(out).detach()
        return out
