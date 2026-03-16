"""
Hyperbolic Graph Convolutional Network Expert -- HGCN feature refinement.

Refines pre-computed hyperbolic and merchant embeddings through a gated
MLP with residual connections.  Optionally produces auxiliary brand/MCC
prediction heads for multi-task supervision.

Architecture::

    input (hyperbolic_dim + merchant_dim = 47D default)
        -> refine_mlp (Linear -> GELU -> Linear + residual)
        -> output_proj (Linear -> LayerNorm -> SiLU)  ->  output (output_dim)

    Optional heads:
        refined -> mcc_level1_head  ->  logits
        refined -> mcc_level2_head  ->  logits
        refined -> brand_head       ->  logits

The hyperbolic embeddings are expected to be pre-computed (e.g. via
Poincare-ball GCN) and concatenated with merchant-level features before
being fed to this expert.

References
----------
Chami et al., *Hyperbolic Graph Convolutional Neural Networks*, NeurIPS 2019.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .base import AbstractExpert
from .registry import ExpertRegistry

logger = logging.getLogger(__name__)


# =============================================================================
# Config
# =============================================================================

@dataclass
class HGCNConfig:
    """Configuration for :class:`UnifiedHGCNExpert`."""

    output_dim: int = 128
    hyperbolic_dim: int = 20
    merchant_dim: int = 27
    hidden_dim: int = 128
    dropout: float = 0.2
    use_brand_heads: bool = False
    mcc_level1_classes: int = 20
    mcc_level2_classes: int = 100
    brand_classes: int = 500

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "HGCNConfig":
        """Build from a plain dict, ignoring unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in cfg.items() if k in valid_keys})


# =============================================================================
# Sub-modules
# =============================================================================

class RefineMLP(nn.Module):
    """
    Two-layer MLP with GELU activation and a residual connection.

    If ``input_dim != hidden_dim`` a skip projection is added so that
    the residual shapes match.
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

        # Skip projection when dimensions differ
        self.skip = (
            nn.Linear(input_dim, hidden_dim, bias=False)
            if input_dim != hidden_dim
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h = self.fc1(x)
        h = self.act(h)
        h = self.dropout(h)
        h = self.fc2(h)
        return self.norm(h + residual)


# =============================================================================
# HGCN Expert
# =============================================================================

@ExpertRegistry.register("hgcn")
class UnifiedHGCNExpert(AbstractExpert):
    """
    Hyperbolic GCN expert for refining pre-computed hyperbolic embeddings.

    Config keys
    -----------
    output_dim : int
        Expert output dimension (default 128).
    hyperbolic_dim : int
        Dimension of hyperbolic (Poincare) embeddings (default 20).
    merchant_dim : int
        Dimension of merchant-level features (default 27).
    hidden_dim : int
        Hidden layer width in the refinement MLP (default 128).
    dropout : float
        Dropout rate (default 0.2).
    use_brand_heads : bool
        Enable auxiliary brand / MCC prediction heads (default ``False``).
    mcc_level1_classes : int
        Number of MCC level-1 categories (default 20).
    mcc_level2_classes : int
        Number of MCC level-2 categories (default 100).
    brand_classes : int
        Number of brand classes (default 500).
    """

    def __init__(self, input_dim: int, config: Dict[str, Any]):
        super().__init__(input_dim, config)
        cfg = HGCNConfig.from_dict(config)

        self._output_dim: int = cfg.output_dim
        self.hyperbolic_dim = cfg.hyperbolic_dim
        self.merchant_dim = cfg.merchant_dim
        self.use_brand_heads = cfg.use_brand_heads

        combined_dim = cfg.hyperbolic_dim + cfg.merchant_dim

        # Validate input_dim
        if input_dim != combined_dim:
            logger.warning(
                "HGCNExpert: input_dim=%d != hyperbolic_dim(%d) + merchant_dim(%d) = %d. "
                "Using input_dim as-is.",
                input_dim, cfg.hyperbolic_dim, cfg.merchant_dim, combined_dim,
            )
            combined_dim = input_dim

        # -- Refinement MLP (Linear -> GELU -> Linear + residual) ---------------
        self.refine_mlp = RefineMLP(
            input_dim=combined_dim,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
        )

        # -- Output projection ---------------------------------------------------
        self.output_proj = nn.Sequential(
            nn.Linear(cfg.hidden_dim, self._output_dim),
            nn.LayerNorm(self._output_dim),
            nn.SiLU(),
        )

        # -- Optional brand prediction heads ------------------------------------
        if self.use_brand_heads:
            self.mcc_level1_head = nn.Linear(cfg.hidden_dim, cfg.mcc_level1_classes)
            self.mcc_level2_head = nn.Linear(cfg.hidden_dim, cfg.mcc_level2_classes)
            self.brand_head = nn.Linear(cfg.hidden_dim, cfg.brand_classes)

        logger.info(
            "UnifiedHGCNExpert: input=%d (hyper=%d + merch=%d), "
            "hidden=%d, brand_heads=%s -> output=%d  (params=%s)",
            input_dim, cfg.hyperbolic_dim, cfg.merchant_dim,
            cfg.hidden_dim, self.use_brand_heads,
            self._output_dim, f"{self.count_parameters():,}",
        )

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            ``[batch, input_dim]`` concatenation of hyperbolic and merchant features.

        Returns
        -------
        torch.Tensor
            ``[batch, output_dim]``
        """
        refined = self.refine_mlp(x)
        return self.output_proj(refined)

    def predict_brand_heads(
        self, x: torch.Tensor
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Compute auxiliary brand / MCC logits for multi-task training.

        Returns ``None`` if brand heads are disabled.

        Returns
        -------
        dict or None
            ``{"mcc_level1": (B, C1), "mcc_level2": (B, C2), "brand": (B, C3)}``
        """
        if not self.use_brand_heads:
            return None

        refined = self.refine_mlp(x)
        return {
            "mcc_level1": self.mcc_level1_head(refined),
            "mcc_level2": self.mcc_level2_head(refined),
            "brand": self.brand_head(refined),
        }
