"""
PersLay Expert -- Topological feature learning from persistence diagrams.

Implements a differentiable PersLay pipeline that maps persistence diagrams
(or pre-computed TDA feature vectors) to fixed-size representations.

Two operating modes:

* **Legacy** (``use_raw_diagram=False``) -- the input is a pre-computed TDA
  feature vector; a simple MLP maps it to the output.
* **Full PersLay** (``use_raw_diagram=True``) -- the input is a raw
  persistence diagram (set of birth/death pairs); the full PersLay pipeline
  (phi -> weight -> rho) is applied.

Architecture (full PersLay mode)::

    diagram [batch, max_points, 2]
        -> RationalHatPhi  (pointwise feature map)  [batch, max_points, phi_dim]
        -> WeightFunction   (per-point importance)   [batch, max_points, 1]
        -> weighted sum / max (PermutationInvariantRho)
        -> output_proj  ->  output (output_dim)

References
----------
Carriere et al., *PersLay: A Neural Network Layer for Persistence Diagrams
and New Graph Topological Signatures*, AISTATS 2020.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AbstractExpert
from .registry import ExpertRegistry

logger = logging.getLogger(__name__)


# =============================================================================
# Config
# =============================================================================

@dataclass
class PersLayConfig:
    """Configuration for :class:`PersLayExpert`."""

    output_dim: int = 64
    tda_dim: int = 70
    hidden_dims: List[int] = None  # type: ignore[assignment]
    dropout: float = 0.1
    use_raw_diagram: bool = False
    # Full PersLay parameters (only used when use_raw_diagram=True)
    phi_dim: int = 32
    num_hat_centers: int = 20
    rho_type: str = "sum"  # "sum" or "max"
    max_diagram_points: int = 100

    def __post_init__(self):
        if self.hidden_dims is None:
            # 3-layer MLP matching on-prem PersLay hidden structure
            self.hidden_dims = [128, 96, 64]

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "PersLayConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in cfg.items() if k in valid_keys})


# =============================================================================
# PersLay sub-modules
# =============================================================================

class RationalHatPhi(nn.Module):
    """
    Rational hat function feature map for persistence diagram points.

    Maps each (birth, death) pair through a set of learnable hat-shaped
    basis functions centred at ``centers`` with learnable widths.

    Output for each center::

        phi_j(b, d) = max(0, 1 - |[b, d] - c_j| / sigma_j)

    Parameters
    ----------
    num_centers : int
        Number of hat function centres.
    input_features : int
        Dimensionality of each point (default 2 for birth/death).
    """

    def __init__(self, num_centers: int = 20, input_features: int = 2):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_centers, input_features) * 0.5)
        self.log_sigmas = nn.Parameter(torch.zeros(num_centers))

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        points : torch.Tensor
            ``[batch, max_points, 2]``

        Returns
        -------
        torch.Tensor
            ``[batch, max_points, num_centers]``
        """
        # points: [B, P, 2], centers: [C, 2]
        sigmas = torch.exp(self.log_sigmas).clamp(min=1e-6)  # [C]
        diff = points.unsqueeze(2) - self.centers.unsqueeze(0).unsqueeze(0)  # [B,P,C,2]
        dist = diff.norm(dim=-1)  # [B, P, C]
        return F.relu(1.0 - dist / sigmas.unsqueeze(0).unsqueeze(0))


class WeightFunction(nn.Module):
    """
    Per-point importance weighting for persistence diagram points.

    A small MLP that maps each point to a scalar weight, normalised
    via softmax across all points in the diagram.

    Parameters
    ----------
    input_features : int
        Dimensionality of each point (default 2).
    hidden_dim : int
        Hidden layer width (default 32).
    """

    def __init__(self, input_features: int = 2, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, points: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        points : torch.Tensor
            ``[batch, max_points, 2]``
        mask : torch.Tensor or None
            ``[batch, max_points]`` boolean; ``True`` for valid points.

        Returns
        -------
        torch.Tensor
            ``[batch, max_points, 1]`` softmax-normalised weights.
        """
        logits = self.net(points)  # [B, P, 1]
        if mask is not None:
            logits = logits.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        return F.softmax(logits, dim=1)


class PermutationInvariantRho(nn.Module):
    """
    Permutation-invariant aggregation: weighted sum or max pooling.

    Parameters
    ----------
    rho_type : str
        ``"sum"`` for weighted sum, ``"max"`` for max pooling.
    """

    def __init__(self, rho_type: str = "sum"):
        super().__init__()
        if rho_type not in ("sum", "max"):
            raise ValueError(f"rho_type must be 'sum' or 'max', got {rho_type!r}")
        self.rho_type = rho_type

    def forward(
        self,
        features: torch.Tensor,
        weights: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        features : torch.Tensor
            ``[batch, max_points, phi_dim]``
        weights : torch.Tensor
            ``[batch, max_points, 1]``
        mask : torch.Tensor or None
            ``[batch, max_points]`` boolean.

        Returns
        -------
        torch.Tensor
            ``[batch, phi_dim]``
        """
        if self.rho_type == "sum":
            return (features * weights).sum(dim=1)
        else:
            # Max pooling (mask invalid points with large negative)
            if mask is not None:
                features = features.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            return features.max(dim=1).values


class PersLayBlock(nn.Module):
    """
    Full PersLay block: phi -> weight -> rho.

    Combines :class:`RationalHatPhi`, :class:`WeightFunction`, and
    :class:`PermutationInvariantRho` into a single module.

    Parameters
    ----------
    num_centers : int
        Number of hat centres in phi.
    rho_type : str
        Aggregation type (``"sum"`` or ``"max"``).
    """

    def __init__(self, num_centers: int = 20, rho_type: str = "sum"):
        super().__init__()
        self.phi = RationalHatPhi(num_centers=num_centers, input_features=2)
        self.weight_fn = WeightFunction(input_features=2)
        self.rho = PermutationInvariantRho(rho_type=rho_type)

    def forward(
        self, diagram: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        diagram : torch.Tensor
            ``[batch, max_points, 2]`` persistence diagram (birth, death).
        mask : torch.Tensor or None
            ``[batch, max_points]`` boolean mask for valid points.

        Returns
        -------
        torch.Tensor
            ``[batch, num_centers]``
        """
        features = self.phi(diagram)            # [B, P, C]
        weights = self.weight_fn(diagram, mask)  # [B, P, 1]
        return self.rho(features, weights, mask)  # [B, C]


# =============================================================================
# PersLay Expert
# =============================================================================

@ExpertRegistry.register("perslay")
class PersLayExpert(AbstractExpert):
    """
    PersLay expert for topological feature extraction.

    Config keys
    -----------
    output_dim : int
        Expert output dimension (default 64).
    tda_dim : int
        Dimension of pre-computed TDA feature vector (default 70).
        Only used in legacy mode (``use_raw_diagram=False``).
    hidden_dims : list[int]
        Hidden layer sizes for the legacy MLP (default ``[128, 96, 64]``).
    dropout : float
        Dropout rate (default 0.1).
    use_raw_diagram : bool
        If ``True``, expect raw persistence diagrams and use the full
        PersLay pipeline.  If ``False`` (default), expect a pre-computed
        TDA feature vector and use a simple MLP.
    phi_dim : int
        Number of hat centres in the phi feature map (default 32).
    num_hat_centers : int
        Alias for phi_dim (default 20).
    rho_type : str
        Aggregation: ``"sum"`` or ``"max"`` (default ``"sum"``).
    max_diagram_points : int
        Maximum number of points per persistence diagram (default 100).
    """

    def __init__(self, input_dim: int, config: Dict[str, Any]):
        super().__init__(input_dim, config)
        cfg = PersLayConfig.from_dict(config)

        self._output_dim: int = cfg.output_dim
        self.use_raw_diagram = cfg.use_raw_diagram

        if self.use_raw_diagram:
            # -- Full PersLay pipeline ------------------------------------------
            self.perslay_block = PersLayBlock(
                num_centers=cfg.num_hat_centers,
                rho_type=cfg.rho_type,
            )
            perslay_out_dim = cfg.num_hat_centers

            self.output_proj = nn.Sequential(
                nn.Linear(perslay_out_dim, cfg.output_dim),
                nn.LayerNorm(cfg.output_dim),
                nn.SiLU(),
            )

            logger.info(
                "PersLayExpert (raw diagram): centers=%d, rho=%s -> output=%d  (params=%s)",
                cfg.num_hat_centers, cfg.rho_type,
                self._output_dim, f"{self.count_parameters():,}",
            )
        else:
            # -- Legacy MLP mode ------------------------------------------------
            # Pre-computed TDA features -> 3-layer refine MLP -> output
            # Matches on-prem hidden structure: SiLU activation, LayerNorm,
            # dropout after each hidden layer, final LayerNorm on output.
            layers: List[nn.Module] = []
            prev_dim = input_dim
            for hdim in cfg.hidden_dims:
                layers.append(nn.Linear(prev_dim, hdim))
                layers.append(nn.SiLU())
                layers.append(nn.Dropout(cfg.dropout))
                prev_dim = hdim
            layers.append(nn.Linear(prev_dim, cfg.output_dim))
            layers.append(nn.LayerNorm(cfg.output_dim))

            self.network = nn.Sequential(*layers)

        # -- Interpretable 4D projection (detached for audit) -----------------
        # Produces scores along 2 interpretive axes (2D each):
        #   - topological_complexity (2D): H0/H1 feature importance
        #   - persistence_scale (2D): local vs global structure
        self.interpretable_proj = nn.Linear(cfg.output_dim, 4)
        self._interpretable_scores: Optional[torch.Tensor] = None

        # Interpretable dimension labels for downstream consumers
        self.interpretable_labels: List[str] = [
            "topological_complexity_0",
            "topological_complexity_1",
            "persistence_scale_0",
            "persistence_scale_1",
        ]

        # -- Output LayerNorm (on-prem normalization) -------------------------
        self.output_norm = nn.LayerNorm(cfg.output_dim)

        if not self.use_raw_diagram:
            logger.info(
                "PersLayExpert (legacy MLP): input_dim=%d (tda_dim=%d) -> hidden=%s -> output=%d  (params=%s)",
                input_dim, cfg.tda_dim, cfg.hidden_dims,
                self._output_dim, f"{self.count_parameters():,}",
            )

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def interpretable_scores(self) -> Optional[torch.Tensor]:
        """Return the most recent 4D interpretable projection.

        Shape: ``[batch, 4]`` -- 2 axes x 2 dimensions each:
          - ``[0:2]`` topological_complexity
          - ``[2:4]`` persistence_scale

        Returns ``None`` if :meth:`forward` has not been called yet.
        """
        return getattr(self, "_interpretable_scores", None)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Legacy mode: ``[batch, tda_dim]`` pre-computed TDA features.
            Raw mode: ``[batch, max_points, 2]`` persistence diagram.
        mask : torch.Tensor, optional
            ``[batch, max_points]`` boolean mask (raw mode only).

        Returns
        -------
        torch.Tensor
            ``[batch, output_dim]``
        """
        if self.use_raw_diagram:
            mask = kwargs.get("mask")
            pooled = self.perslay_block(x, mask=mask)
            out = self.output_proj(pooled)
        else:
            out = self.network(x)

        # Compute 4D interpretable scores (detached -- no grad backprop)
        self._interpretable_scores = self.interpretable_proj(out).detach()

        # On-prem output normalization
        return self.output_norm(out)
