"""
Optimal Transport Expert -- Sinkhorn-based Wasserstein feature extraction.

Projects each sample's features onto a probability simplex and computes
approximate Wasserstein distances to a set of learnable reference
distributions.  These OT distances form a rich, geometry-aware
representation.

Architecture::

    input (input_dim)
        -> dist_projector (MLP + softmax)  ->  sample distribution  [D]
        -> Sinkhorn distance to each of n_ref learnable references  [n_ref]
        -> wasserstein_encoder (MLP)       ->  output (output_dim)

The ground cost matrix ``C`` is learnable and kept positive semi-definite
via the factorisation ``C = M^T M``.

References
----------
Cuturi, *Sinkhorn Distances: Lightspeed Computation of Optimal Transport
Distances*, NeurIPS 2013.
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


@ExpertRegistry.register("optimal_transport")
class OptimalTransportExpert(AbstractExpert):
    """
    Optimal Transport expert via log-domain Sinkhorn algorithm.

    Config keys
    -----------
    output_dim : int
        Expert output dimension (default 64).
    hidden_dim : int
        Hidden layer size in projector / encoder (default 128).
    n_reference_distributions : int
        Number of learnable reference distributions (default 16).
    sinkhorn_iterations : int
        Sinkhorn algorithm iterations (default 10).
    sinkhorn_epsilon : float
        Entropic regularisation coefficient (default 0.1).
    distribution_dim : int
        Probability simplex dimension (default 32).
    dropout : float
        Dropout rate (default 0.2).
    """

    def __init__(self, input_dim: int, config: Dict[str, Any]):
        super().__init__(input_dim, config)

        self._output_dim: int = config.get("output_dim", 64)
        hidden_dim: int = config.get("hidden_dim", 128)
        self.n_ref: int = config.get("n_reference_distributions", 16)
        self.sinkhorn_iters: int = config.get("sinkhorn_iterations", 10)
        self.sinkhorn_eps: float = config.get("sinkhorn_epsilon", 0.1)
        dist_dim: int = config.get("distribution_dim", 32)
        dropout: float = config.get("dropout", 0.2)

        # Feature -> probability simplex (logits; softmax in forward)
        self.dist_projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dist_dim),
        )

        # Learnable reference distribution logits [n_ref, D]
        self.reference_logits = nn.Parameter(
            torch.randn(self.n_ref, dist_dim) * 0.1
        )

        # Learnable ground cost matrix (PSD via M^T M in forward)
        self.cost_matrix = nn.Parameter(
            torch.eye(dist_dim) + torch.randn(dist_dim, dist_dim) * 0.01
        )

        # Wasserstein distance vector -> output representation
        self.wasserstein_encoder = nn.Sequential(
            nn.Linear(self.n_ref, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self._output_dim),
            nn.LayerNorm(self._output_dim),
            nn.SiLU(),
        )

        logger.info(
            "OptimalTransportExpert: input=%d, n_ref=%d, "
            "sinkhorn_iters=%d -> output=%d  (params=%s)",
            input_dim, self.n_ref, self.sinkhorn_iters,
            self._output_dim, f"{self.count_parameters():,}",
        )

    @property
    def output_dim(self) -> int:
        return self._output_dim

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

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
        # Project to probability simplex
        dist_logits = self.dist_projector(x)
        sample_dist = F.softmax(dist_logits, dim=-1)  # [B, D]

        # Reference distributions
        ref_dists = F.softmax(self.reference_logits, dim=-1)  # [n_ref, D]

        # PSD cost matrix: C = M^T M
        cost = self.cost_matrix.T @ self.cost_matrix  # [D, D]

        # Wasserstein distances to each reference
        ot_distances = []
        for i in range(self.n_ref):
            d = self._sinkhorn_distance(
                sample_dist, ref_dists[i], cost,
                self.sinkhorn_iters, self.sinkhorn_eps,
            )
            ot_distances.append(d)

        ot_features = torch.stack(ot_distances, dim=-1)  # [B, n_ref]
        return self.wasserstein_encoder(ot_features)

    # ------------------------------------------------------------------
    # Sinkhorn
    # ------------------------------------------------------------------

    @staticmethod
    def _sinkhorn_distance(
        a: torch.Tensor,
        b: torch.Tensor,
        cost: torch.Tensor,
        num_iters: int,
        epsilon: float,
    ) -> torch.Tensor:
        """
        Log-domain Sinkhorn algorithm for approximate Wasserstein distance.

        Parameters
        ----------
        a : torch.Tensor
            ``[B, D]`` source distributions (probability simplex).
        b : torch.Tensor
            ``[D]`` single reference distribution.
        cost : torch.Tensor
            ``[D, D]`` ground cost matrix (PSD).
        num_iters : int
            Number of Sinkhorn iterations.
        epsilon : float
            Entropic regularisation.

        Returns
        -------
        torch.Tensor
            ``[B]`` approximate Wasserstein distance per sample.
        """
        # Cast to FP32 before log-domain Sinkhorn to avoid FP16 underflow
        a = a.float()
        b = b.float()
        cost = cost.float()
        log_a = torch.log(a.clamp(min=1e-6))
        log_b = torch.log(b.clamp(min=1e-6))
        log_K = -cost / epsilon  # [D, D]

        u = torch.zeros_like(log_a)  # [B, D]

        for _ in range(num_iters):
            # v update
            v = log_b.unsqueeze(0) - torch.logsumexp(
                u.unsqueeze(2) + log_K.unsqueeze(0), dim=1
            )
            # u update
            u = log_a - torch.logsumexp(
                v.unsqueeze(1) + log_K.unsqueeze(0), dim=2
            )

        # Transport plan and Wasserstein distance
        log_pi = u.unsqueeze(2) + log_K.unsqueeze(0) + v.unsqueeze(1)
        pi = log_pi.exp()
        distances = (pi * cost.unsqueeze(0)).sum(dim=(1, 2))

        return distances
