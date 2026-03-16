"""
Sparse Autoencoder (SAE) layer for mechanistic interpretability.

This module implements an Anthropic-style Sparse Autoencoder that extracts
interpretable, sparse features from dense neural-network representations.
It is designed to be attached to any intermediate representation (e.g. the
gated expert output of a PLE model) as an *analysis sidecar* -- gradients
do **not** flow back into the main network unless explicitly desired.

Architecture overview::

    x  ─── (- pre_bias) ──> Encoder ──> ReLU ──> latent (sparse)
                                                      │
                                               Decoder (tied W^T)
                                                      │
                                              (+ post_bias) ──> x_hat

Loss = MSE(x_hat, x) + l1_lambda * |latent|_1

Key design choices:

* **Tied weights** (default): The decoder weight matrix is the transpose of
  the encoder weight, reducing parameter count and improving training
  stability.
* **Pre-bias centering**: A learnable bias is subtracted before encoding,
  effectively learning the data mean.
* **ReLU activation**: Induces non-negative sparse codes without
  additional penalties.
* **Column-normalised decoder**: Prevents individual latent features from
  growing unboundedly.

Reference:
    Bricken et al., "Towards Monosemanticity: Decomposing Language Models
    With Dictionary Learning", Anthropic 2023.

Example::

    sae = SparseAutoencoder(input_dim=128, expansion_factor=8)

    # Analysis mode (no gradient to backbone):
    with torch.no_grad():
        recon, latent, sae_loss = sae(backbone_output.detach())

    # Joint training mode (gradient flows to backbone):
    recon, latent, sae_loss = sae(backbone_output)
    total_loss = task_loss + 0.01 * sae_loss
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SparseAutoencoder"]


class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder for interpretable feature extraction.

    Args:
        input_dim: Dimensionality of the input representation.
        expansion_factor: The latent dimension is ``input_dim * expansion_factor``.
            Larger factors allow finer-grained feature decomposition at the
            cost of memory and compute.
        l1_lambda: Weight of the L1 sparsity penalty on latent activations.
        tied_weights: If ``True``, the decoder reuses the encoder weight
            matrix (transposed).
        normalize_decoder: If ``True``, decoder columns are L2-normalised
            before each forward pass.
    """

    def __init__(
        self,
        input_dim: int,
        expansion_factor: int = 4,
        l1_lambda: float = 0.001,
        tied_weights: bool = True,
        normalize_decoder: bool = True,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = input_dim * expansion_factor
        self.l1_lambda = l1_lambda
        self.tied_weights = tied_weights
        self.normalize_decoder = normalize_decoder

        # Pre-encoding centering bias
        self.pre_bias = nn.Parameter(torch.zeros(input_dim))

        # Encoder: input_dim -> latent_dim
        self.encoder = nn.Linear(input_dim, self.latent_dim, bias=True)

        # Decoder (only when weights are NOT tied)
        if not self.tied_weights:
            self.decoder = nn.Linear(self.latent_dim, input_dim, bias=False)

        # Post-decoding bias
        self.post_bias = nn.Parameter(torch.zeros(input_dim))

    # ── Core operations ─────────────────────────────────────────

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input into a sparse latent representation.

        Args:
            x: Input tensor of shape ``(B, input_dim)``.

        Returns:
            Non-negative latent activations of shape ``(B, latent_dim)``.
        """
        x_centered = x - self.pre_bias
        return F.relu(self.encoder(x_centered))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode sparse latent codes back to input space.

        Args:
            latent: Latent activations of shape ``(B, latent_dim)``.

        Returns:
            Reconstruction of shape ``(B, input_dim)``.
        """
        W = self._decoder_weight()  # (latent_dim, input_dim)
        return latent @ W + self.post_bias

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: encode, decode, and compute the SAE loss.

        Callers that want to prevent gradient flow into the upstream network
        should pass ``x.detach()`` explicitly.

        Args:
            x: Input tensor of shape ``(B, input_dim)``.

        Returns:
            A 3-tuple ``(reconstruction, latent, sae_loss)`` where:

            * ``reconstruction`` -- ``(B, input_dim)`` reconstructed input.
            * ``latent`` -- ``(B, latent_dim)`` sparse feature activations.
            * ``sae_loss`` -- scalar ``MSE + l1_lambda * mean(|latent|)``.
        """
        latent = self.encode(x)
        reconstruction = self.decode(latent)

        mse_loss = F.mse_loss(reconstruction, x)
        l1_loss = latent.abs().mean()
        sae_loss = mse_loss + self.l1_lambda * l1_loss

        return reconstruction, latent, sae_loss

    # ── Utilities ───────────────────────────────────────────────

    def get_top_features(
        self, x: torch.Tensor, k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the top-*k* activated latent features per sample.

        Args:
            x: Input tensor of shape ``(B, input_dim)``.
            k: Number of top features to return.

        Returns:
            ``(values, indices)`` -- both of shape ``(B, k)``.  *values*
            are the activation magnitudes; *indices* are positions in the
            latent space.
        """
        latent = self.encode(x)
        return torch.topk(latent, k=k, dim=-1)

    def get_feature_attributions(self, feature_idx: int) -> torch.Tensor:
        """Return the decoder column for a given latent feature.

        This vector shows how a single latent feature maps back to the
        original input dimensions, providing a simple attribution.

        Args:
            feature_idx: Index into the latent dimension.

        Returns:
            Attribution vector of shape ``(input_dim,)``.
        """
        W = self._decoder_weight()  # (latent_dim, input_dim)
        return W[feature_idx].detach()

    @property
    def sparsity(self) -> float:
        """Fraction of dead (always-zero) latent neurons.

        This is computed from the encoder bias; neurons with very negative
        bias are unlikely to ever fire after ReLU.  A high sparsity (> 0.9)
        is typical and desirable.

        Note:
            This is a rough heuristic.  For a precise measurement, run a
            forward pass on representative data and count zero activations.
        """
        with torch.no_grad():
            return (self.encoder.bias < -5.0).float().mean().item()

    # ── Private helpers ─────────────────────────────────────────

    def _decoder_weight(self) -> torch.Tensor:
        """Return the effective decoder weight matrix ``(latent_dim, input_dim)``.

        When tied, this is simply the encoder weight matrix (which PyTorch
        stores as ``(out_features, in_features) = (latent_dim, input_dim)``).
        When untied, the decoder stores ``(input_dim, latent_dim)`` so we
        transpose it.
        """
        if self.tied_weights:
            W = self.encoder.weight  # (latent_dim, input_dim)
        else:
            W = self.decoder.weight.T  # (input_dim, latent_dim) -> (latent_dim, input_dim)

        if self.normalize_decoder:
            W = F.normalize(W, dim=1)  # normalise along input_dim axis

        return W

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, latent_dim={self.latent_dim}, "
            f"l1_lambda={self.l1_lambda}, tied={self.tied_weights}, "
            f"norm_decoder={self.normalize_decoder}"
        )
