"""
Causal Expert -- Structural Causal Model with NOTEARS DAG constraint.

Learns a directed acyclic graph (DAG) over a compressed variable space
and uses it to produce causally-informed representations.

Architecture::

    input (input_dim)
        -> feature_compressor (MLP)  ->  z  (n_causal_vars)
        -> SCM mechanism:  z_hat = z + z @ (W * W)
        -> causal_encoder  (MLP)     ->  output (output_dim)

The adjacency matrix ``W`` is continuously optimised subject to the
NOTEARS acyclicity constraint:

    h(W) = tr(exp(W . W)) - d = 0

where the matrix exponential is approximated with a 10-term Taylor expansion.

References
----------
Zheng et al., *DAGs with NO TEARS: Continuous Optimization for Structure
Learning*, NeurIPS 2018.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import torch
import torch.nn as nn

from .base import AbstractExpert
from .registry import ExpertRegistry

logger = logging.getLogger(__name__)


@ExpertRegistry.register("causal")
class CausalExpert(AbstractExpert):
    """
    Causal expert using a differentiable DAG (NOTEARS).

    Config keys
    -----------
    output_dim : int
        Expert output dimension (default 64).
    hidden_dim : int
        Hidden layer size in compressor/encoder (default 128).
    n_causal_vars : int
        DAG variable space dimension (default 32).
    dag_lambda : float
        Weight of the acyclicity loss term (default 0.01).
    sparsity_lambda : float
        Weight of the L1 sparsity regulariser on the adjacency
        matrix (default 0.001).
    dropout : float
        Dropout rate (default 0.2).
    """

    def __init__(self, input_dim: int, config: Dict[str, Any]):
        super().__init__(input_dim, config)

        self._output_dim: int = config.get("output_dim", 64)
        hidden_dim: int = config.get("hidden_dim", 128)
        self.n_causal_vars: int = config.get("n_causal_vars", 32)
        self.dag_lambda: float = config.get("dag_lambda", 0.01)
        self.sparsity_lambda: float = config.get("sparsity_lambda", 0.001)
        # NOTEARS-style reconstruction penalty: enforces that W actually
        # describes how each latent variable is a linear function of the
        # others. Without this, W has no gradient incentive beyond the
        # sparsity penalty and collapses to zero.
        # Reference: Zheng et al. (NeurIPS 2018) min ||X - X W||_F^2.
        self.recon_lambda: float = config.get("recon_lambda", 0.5)
        dropout: float = config.get("dropout", 0.2)

        # -- Feature compressor: input_dim -> n_causal_vars --------------------
        self.feature_compressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.n_causal_vars),
        )

        # -- Learnable weighted adjacency matrix [d, d] -----------------------
        # W[i, j] encodes the causal influence j -> i.
        # Init scale 0.1 (was 0.01) so gradient on W escapes the W=0 saddle
        # point: both task loss and the reconstruction loss have a W factor
        # in their gradient (d(z @ W^2)/dW ∝ 2zW), so a near-zero init
        # gives near-zero gradient, and sparsity then pulls W all the way
        # to exactly 0. A 10× larger init keeps W² on the O(0.01) scale
        # initially, which is small enough not to disrupt early forward
        # passes but large enough to carry meaningful gradient.
        self.W = nn.Parameter(
            torch.randn(self.n_causal_vars, self.n_causal_vars) * 0.1
        )

        # Cache of the last feature_compressor output (z), used by the
        # reconstruction loss in ``get_dag_regularization``. Populated on
        # every forward call so the most recent batch's z is available.
        self._last_z: torch.Tensor | None = None

        # -- Causal encoder: n_causal_vars -> output_dim -----------------------
        self.causal_encoder = nn.Sequential(
            nn.Linear(self.n_causal_vars, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self._output_dim),
            nn.LayerNorm(self._output_dim),
            nn.SiLU(),
        )

        logger.info(
            "CausalExpert: input=%d -> vars=%d -> output=%d  (params=%s)",
            input_dim, self.n_causal_vars, self._output_dim,
            f"{self.count_parameters():,}",
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
            ``[batch, output_dim]`` causally-informed representation.
        """
        z = self.feature_compressor(x)
        # Keep z accessible to get_dag_regularization so the NOTEARS-
        # style reconstruction loss ||z - z W^2||^2 can be computed.
        # Stored as a live (with-grad) tensor; gets overwritten every
        # forward, so only the most recent batch contributes.
        self._last_z = z
        z_hat = self._apply_causal_mechanism(z)
        return self.causal_encoder(z_hat)

    def _apply_causal_mechanism(self, z: torch.Tensor) -> torch.Tensor:
        """
        Linear SCM intervention::

            z_hat = z + z @ (W * W)

        Element-wise squaring ensures non-negative causal strengths.
        The residual connection preserves the original representation.
        """
        W_sq = self.W * self.W  # [d, d] non-negative
        return z + z @ W_sq

    # ------------------------------------------------------------------
    # Regularisation
    # ------------------------------------------------------------------

    def get_dag_regularization(self) -> torch.Tensor:
        """
        Compute the NOTEARS acyclicity + sparsity + reconstruction
        regularisation loss.

        Three terms:

          1. Acyclicity (10-term Taylor approximation of
             ``h(W) = tr(exp(W . W)) - d``)
          2. Sparsity (L1-like, ``||W . W||_1``)
          3. *Reconstruction* (NOTEARS-style ``||z - z W^2||_F^2``):
             ensures ``W`` has a load-bearing gradient path. Without
             this, the residual connection
             ``z_hat = z + z W^2`` lets task loss flow entirely through
             ``z``, W has no learning incentive beyond the sparsity
             penalty, and collapses to zero (observed empirically on
             all checkpoints before this patch).

        Total loss::

            dag_loss = dag_lambda    * h(W)
                     + sparsity_lambda * ||W.W||_1
                     + recon_lambda   * ||z - z W^2||_F^2

        The reconstruction term is a safeguard against degenerate
        ``W = 0`` and restores the standard NOTEARS gradient signal that
        the original paper relies on.

        Returns
        -------
        torch.Tensor
            Scalar loss (differentiable).
        """
        with torch.amp.autocast('cuda', enabled=False):
            W_f32 = self.W.float()
            W_sq = W_f32 * W_f32
            d = self.n_causal_vars

            # 1. Taylor expansion for tr(exp(W_sq)) in FP32 to avoid overflow
            h = torch.tensor(0.0, device=self.W.device)
            M_power = torch.eye(d, device=self.W.device)
            for i in range(1, 11):
                M_power = M_power @ W_sq / i
                h = h + torch.trace(M_power)

            # 2. Sparsity (existing)
            sparsity = W_sq.sum()

            # 3. NOTEARS-style reconstruction on the most recent z.
            # Guards against missing cache (e.g. called before forward)
            # and against recon_lambda == 0 (skip MSE compute entirely).
            recon = torch.tensor(0.0, device=self.W.device)
            if self.recon_lambda > 0.0 and self._last_z is not None:
                z = self._last_z.float()
                recon = torch.mean((z - z @ W_sq) ** 2)

            return (self.dag_lambda * h
                    + self.sparsity_lambda * sparsity
                    + self.recon_lambda * recon)

    def get_causal_graph(self) -> torch.Tensor:
        """
        Return the learned causal adjacency matrix (detached).

        Returns
        -------
        torch.Tensor
            ``[n_causal_vars, n_causal_vars]`` non-negative matrix where
            ``graph[i, j] = W[i, j]^2`` represents causal influence j -> i.
        """
        return (self.W * self.W).detach()
