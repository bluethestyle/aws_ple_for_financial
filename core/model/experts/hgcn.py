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
from typing import Any, Dict, List, Optional

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

    Matches the on-prem refine MLP structure:
    ``Linear(input, hidden) -> GELU -> Linear(hidden, input) + residual``

    No dropout or LayerNorm inside the refine block -- regularisation is
    handled at the expert level (output_proj has LayerNorm).

    If ``input_dim != hidden_dim`` the second linear projects back to
    ``input_dim`` so the residual addition works without a skip projection.
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.act(h)
        h = self.fc2(h)
        return h + x


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
        # On-prem structure: no dropout in refine MLP, output dim == input dim
        self.refine_mlp = RefineMLP(
            input_dim=combined_dim,
            hidden_dim=cfg.hidden_dim,
        )
        # refine_mlp output dimension is combined_dim (residual preserves dim)
        refine_out_dim = combined_dim

        # -- Output projection ---------------------------------------------------
        self.output_proj = nn.Sequential(
            nn.Linear(refine_out_dim, self._output_dim),
            nn.LayerNorm(self._output_dim),
            nn.SiLU(),
        )

        # -- Interpretable 6D projection (detached for audit) -------------------
        # Produces scores along 3 interpretive axes (2D each):
        #   - hierarchy_activation_intensity (2D)
        #   - depth_importance (2D)
        #   - cross_level_interaction (2D)
        self.interpretable_proj = nn.Linear(refine_out_dim, 6)
        self._interpretable_scores: Optional[torch.Tensor] = None

        # Interpretable dimension labels for downstream consumers
        self.interpretable_labels: List[str] = [
            "hierarchy_activation_intensity_0",
            "hierarchy_activation_intensity_1",
            "depth_importance_0",
            "depth_importance_1",
            "cross_level_interaction_0",
            "cross_level_interaction_1",
        ]

        # -- Optional brand prediction heads ------------------------------------
        if self.use_brand_heads:
            self.mcc_level1_head = nn.Linear(refine_out_dim, cfg.mcc_level1_classes)
            self.mcc_level2_head = nn.Linear(refine_out_dim, cfg.mcc_level2_classes)
            self.brand_head = nn.Linear(refine_out_dim, cfg.brand_classes)

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

        # Compute 6D interpretable scores (detached for audit -- no grad)
        self._interpretable_scores = self.interpretable_proj(refined).detach()

        return self.output_proj(refined)

    @property
    def interpretable_scores(self) -> Optional[torch.Tensor]:
        """Return the most recent 6D interpretable projection.

        Shape: ``[batch, 6]`` -- 3 axes x 2 dimensions each:
          - ``[0:2]`` hierarchy_activation_intensity
          - ``[2:4]`` depth_importance
          - ``[4:6]`` cross_level_interaction

        Returns ``None`` if :meth:`forward` has not been called yet.
        """
        return getattr(self, "_interpretable_scores", None)

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
