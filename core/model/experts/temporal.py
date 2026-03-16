"""
Temporal Ensemble Expert -- Mamba + PatchTST with learned gating.

Ensembles multiple temporal modelling paradigms to capture different aspects
of sequential data:

* **Mamba (SSM)** -- linear-complexity long-range dependency modelling.
* **PatchTST (Transformer)** -- patch-based attention for periodic patterns.

A learned gating network decides per-sample how much to trust each sub-model,
providing both capacity and interpretability (gate entropy monitoring).

The original implementation also included Liquid Neural Networks (LNN).
LNN support can be re-added as a plugin; this version keeps Mamba + Transformer
as the two production-ready branches.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AbstractExpert
from .mamba import MambaBlock
from .registry import ExpertRegistry

logger = logging.getLogger(__name__)


# =============================================================================
# PatchTST components
# =============================================================================

class PatchEmbedding(nn.Module):
    """
    Patch Embedding for time series.

    Splits the sequence into non-overlapping (or strided) patches,
    projects each patch into ``d_model``, and adds sinusoidal positional
    encoding.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        patch_size: int = 16,
        stride: Optional[int] = None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride or patch_size
        self.projection = nn.Linear(input_dim * patch_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Parameters
        ----------
        x : torch.Tensor
            ``[batch, seq_len, input_dim]``

        Returns
        -------
        patches : torch.Tensor
            ``[batch, num_patches, d_model]``
        num_patches : int
        """
        batch_size, seq_len, input_dim = x.shape

        # Pad if needed
        if seq_len % self.patch_size != 0:
            pad_len = self.patch_size - (seq_len % self.patch_size)
            x = F.pad(x, (0, 0, 0, pad_len))
            seq_len = x.size(1)

        num_patches = seq_len // self.patch_size
        x = x.view(batch_size, num_patches, self.patch_size * input_dim)
        patches = self.projection(x)

        # Sinusoidal positional encoding
        pos = torch.arange(num_patches, device=x.device).unsqueeze(0)
        patches = patches + self._positional_encoding(pos, self.d_model)

        return patches, num_patches

    @staticmethod
    def _positional_encoding(positions: torch.Tensor, d_model: int) -> torch.Tensor:
        pe = torch.zeros(
            positions.size(0), positions.size(1), d_model,
            device=positions.device,
        )
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=positions.device).float()
            * (-torch.log(torch.tensor(10_000.0)) / d_model)
        )
        pe[:, :, 0::2] = torch.sin(positions.unsqueeze(-1) * div_term)
        pe[:, :, 1::2] = torch.cos(
            positions.unsqueeze(-1) * div_term[: pe[:, :, 1::2].size(-1)]
        )
        return pe


class PatchTST(nn.Module):
    """
    Patch Time Series Transformer (PatchTST-style).

    Splits the input sequence into patches, encodes with a Transformer,
    and pools to a fixed-size vector.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        patch_size: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.patch_embed = PatchEmbedding(
            input_dim=input_dim,
            d_model=d_model,
            patch_size=patch_size,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            ``[batch, seq_len, input_dim]``

        Returns
        -------
        torch.Tensor
            ``[batch, d_model]``
        """
        patches, _ = self.patch_embed(x)
        encoded = self.encoder(patches)
        return self.pool(encoded.transpose(1, 2)).squeeze(-1)


# =============================================================================
# Temporal Ensemble Expert
# =============================================================================

@ExpertRegistry.register("temporal_ensemble")
class TemporalEnsembleExpert(AbstractExpert):
    """
    Temporal ensemble expert: Mamba + Transformer with learned gating.

    Processes one or more sequence streams (e.g. primary + auxiliary).
    Each enabled sub-model produces a representation that is projected to
    ``output_dim``, then combined via a learned soft gate.

    Config keys
    -----------
    output_dim : int
        Expert output dimension (default 64).
    aux_input_dim : int or None
        Auxiliary sequence feature dimension.  If ``None`` (default),
        only the primary stream is used.
    mamba.enabled : bool
        Enable Mamba branch (default ``True``).
    mamba.d_model : int
        Mamba hidden dimension (default 128).
    mamba.d_state : int
        SSM state dimension (default 16).
    mamba.expand : int
        Expansion factor (default 2).
    transformer.enabled : bool
        Enable Transformer branch (default ``True``).
    transformer.d_model : int
        Transformer model dimension (default 64).
    transformer.nhead : int
        Number of attention heads (default 4).
    transformer.num_layers : int
        Encoder layers (default 2).
    transformer.patch_size : int
        Patch size (default 16).
    ensemble_gating : bool
        Use learned gating (default ``True``).  If ``False``,
        concatenate + MLP fusion.
    dropout : float
        Dropout rate (default 0.2).
    """

    def __init__(self, input_dim: int, config: Dict[str, Any]):
        super().__init__(input_dim, config)

        self._output_dim: int = config.get("output_dim", 64)
        aux_input_dim: Optional[int] = config.get("aux_input_dim")
        dropout: float = config.get("dropout", 0.2)
        self.ensemble_gating: bool = config.get("ensemble_gating", True)

        # Sub-model configs (nested dicts)
        m_cfg = config.get("mamba", {})
        t_cfg = config.get("transformer", {})

        self.mamba_enabled: bool = m_cfg.get("enabled", True)
        self.transformer_enabled: bool = t_cfg.get("enabled", True)

        num_models = self.mamba_enabled + self.transformer_enabled
        if num_models == 0:
            raise ValueError("At least one sub-model (mamba or transformer) must be enabled.")

        # -- Mamba branch ------------------------------------------------------
        mamba_out_dim = 0
        if self.mamba_enabled:
            m_d_model = m_cfg.get("d_model", 128)
            m_expand = m_cfg.get("expand", 2)
            m_d_state = m_cfg.get("d_state", 16)

            self.mamba_primary = MambaBlock(
                d_model=m_d_model,
                d_input=input_dim,
                expand=m_expand,
                d_state=m_d_state,
            )
            mamba_out_dim += m_d_model

            if aux_input_dim is not None:
                m_aux_d = m_d_model // 2
                self.mamba_aux = MambaBlock(
                    d_model=m_aux_d,
                    d_input=aux_input_dim,
                    expand=m_expand,
                    d_state=m_d_state,
                )
                mamba_out_dim += m_aux_d

        # -- Transformer branch ------------------------------------------------
        transformer_out_dim = 0
        if self.transformer_enabled:
            t_d_model = t_cfg.get("d_model", 64)
            t_nhead = t_cfg.get("nhead", 4)
            t_num_layers = t_cfg.get("num_layers", 2)
            t_patch_size = t_cfg.get("patch_size", 16)

            self.transformer_primary = PatchTST(
                input_dim=input_dim,
                d_model=t_d_model,
                nhead=t_nhead,
                num_layers=t_num_layers,
                patch_size=t_patch_size,
                dropout=dropout,
            )
            transformer_out_dim += t_d_model

            if aux_input_dim is not None:
                t_aux_d = t_d_model // 2
                self.transformer_aux = PatchTST(
                    input_dim=aux_input_dim,
                    d_model=t_aux_d,
                    nhead=max(1, t_nhead // 2),
                    num_layers=t_num_layers,
                    patch_size=t_patch_size,
                    dropout=dropout,
                )
                transformer_out_dim += t_aux_d

        self._has_aux = aux_input_dim is not None
        total_dim = mamba_out_dim + transformer_out_dim

        # -- Gating / Fusion ---------------------------------------------------
        if self.ensemble_gating and num_models > 1:
            self.gate = nn.Sequential(
                nn.Linear(total_dim, num_models * 2),
                nn.ReLU(),
                nn.Linear(num_models * 2, num_models),
                nn.Softmax(dim=-1),
            )
            model_dims: list[int] = []
            if self.mamba_enabled:
                model_dims.append(mamba_out_dim)
            if self.transformer_enabled:
                model_dims.append(transformer_out_dim)
            self.model_projs = nn.ModuleList([
                nn.Linear(d, self._output_dim) for d in model_dims
            ])
        else:
            self.gate = None
            self.fusion = nn.Sequential(
                nn.Linear(total_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, self._output_dim),
            )

        logger.info(
            "TemporalEnsembleExpert: input=%d, aux=%s, "
            "Mamba=%s, Transformer=%s -> output=%d",
            input_dim, aux_input_dim,
            self.mamba_enabled, self.transformer_enabled,
            self._output_dim,
        )

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            ``[batch, seq_len, input_dim]`` primary sequence.
        aux_seq : torch.Tensor, optional
            ``[batch, seq_len, aux_input_dim]`` auxiliary sequence.
            Required if ``aux_input_dim`` was set in config.

        Returns
        -------
        torch.Tensor
            ``[batch, output_dim]``
        """
        aux_seq: Optional[torch.Tensor] = kwargs.get("aux_seq")
        outputs: list[torch.Tensor] = []

        # -- Mamba branch ------------------------------------------------------
        if self.mamba_enabled:
            h_mamba = self.mamba_primary(x)[:, -1, :]
            parts = [h_mamba]
            if self._has_aux and aux_seq is not None:
                parts.append(self.mamba_aux(aux_seq)[:, -1, :])
            outputs.append(torch.cat(parts, dim=-1))

        # -- Transformer branch ------------------------------------------------
        if self.transformer_enabled:
            h_tf = self.transformer_primary(x)
            parts = [h_tf]
            if self._has_aux and aux_seq is not None:
                parts.append(self.transformer_aux(aux_seq))
            outputs.append(torch.cat(parts, dim=-1))

        # -- Ensemble ----------------------------------------------------------
        if self.gate is not None and len(outputs) > 1:
            concat = torch.cat(outputs, dim=-1)
            gate_weights = self.gate(concat)  # [B, num_models]
            projected = [proj(out) for proj, out in zip(self.model_projs, outputs)]
            expert_output = sum(
                w.unsqueeze(-1) * p
                for w, p in zip(gate_weights.unbind(dim=-1), projected)
            )
        else:
            concat = torch.cat(outputs, dim=-1)
            expert_output = self.fusion(concat)

        return expert_output

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def compute_gate_entropy(self, x: torch.Tensor, **kwargs) -> Optional[torch.Tensor]:
        """
        Compute the Shannon entropy of the gating weights.

        Useful for monitoring gate collapse during training: low entropy
        means the ensemble degenerates to a single sub-model.

        Returns ``None`` if gating is disabled or only one sub-model is active.
        """
        if self.gate is None:
            return None

        with torch.no_grad():
            # Forward just enough to get gate weights
            out = self.forward(x, **kwargs)  # noqa: F841 -- side-effect free
            # Re-compute gate weights (lightweight)
            parts: list[torch.Tensor] = []
            aux_seq = kwargs.get("aux_seq")
            if self.mamba_enabled:
                h = self.mamba_primary(x)[:, -1, :]
                ps = [h]
                if self._has_aux and aux_seq is not None:
                    ps.append(self.mamba_aux(aux_seq)[:, -1, :])
                parts.append(torch.cat(ps, dim=-1))
            if self.transformer_enabled:
                h = self.transformer_primary(x)
                ps = [h]
                if self._has_aux and aux_seq is not None:
                    ps.append(self.transformer_aux(aux_seq))
                parts.append(torch.cat(ps, dim=-1))

            concat = torch.cat(parts, dim=-1)
            gate_weights = self.gate(concat)
            entropy = -(gate_weights * torch.log2(gate_weights + 1e-8)).sum(dim=-1)
            return entropy.mean()
