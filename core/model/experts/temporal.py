"""
Temporal Ensemble Expert -- Mamba + PatchTST + LNN with learned gating.

Ensembles multiple temporal modelling paradigms to capture different aspects
of sequential data:

* **Mamba (SSM)** -- linear-complexity long-range dependency modelling.
* **LNN (LNNSingleStep)** -- processes Mamba's last hidden state with
  input-dependent time constants.  This is a **serial** pipeline
  (Mamba -> LNN), matching the on-prem design.
* **PatchTST (Transformer)** -- patch-based attention for periodic patterns.
  Runs independently on the raw sequence.

Architecture::

    raw_seq ──> Mamba ──> last_hidden ──> LNNSingleStep ──> lnn_out
    raw_seq ──> PatchTST ──────────────────────────────────> tf_out
                                                              |
    gate([mamba_out, lnn_out, tf_out]) ──> ensemble_output

A learned gating network decides per-sample how much to trust each sub-model,
providing both capacity and interpretability (gate entropy monitoring).
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
# Liquid Neural Network components
# =============================================================================

class LiquidTimeConstantCell(nn.Module):
    """
    Single Liquid Time-Constant (LTC) cell — ODE-based dynamics.

    Ported from on-prem LiquidCell. Pure ODE formulation:
        dh/dt = (-h + f(x, h)) / tau(x, h)
    Discretised via Euler method:
        h_new = h + dt * (-h + f(x, h)) / tau

    tau: input-dependent time constant (ReLU + Softplus → always positive).
    f:   state update network (Tanh bounded).
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1,
                 num_units: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Adaptive time constant: tau = f(x, h), always > 0
        self.tau_net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, num_units),
            nn.ReLU(),
            nn.Linear(num_units, hidden_dim),
            nn.Softplus(),  # tau > 0
        )

        # State update network: f(x, h)
        self.state_net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Time scaling (learnable)
        self.time_scale = nn.Parameter(torch.tensor(1.0))

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        dt: float = 1.0,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x_t : torch.Tensor
            ``[batch, input_dim]`` -- input at time *t*.
        h_prev : torch.Tensor
            ``[batch, hidden_dim]`` -- previous hidden state.
        dt : float or torch.Tensor
            Scalar or ``[batch]`` time delta since last event.

        Returns
        -------
        torch.Tensor
            ``[batch, hidden_dim]`` -- updated hidden state.
        """
        combined = torch.cat([x_t, h_prev], dim=-1)

        # Adaptive time constant (always positive via Softplus + floor)
        tau = self.tau_net(combined) + 0.1  # min tau = 0.1 for stability

        # State update function
        f_xh = self.state_net(combined)

        # Discretised ODE: h_new = h + dt * (-h + f(x,h)) / tau
        if isinstance(dt, torch.Tensor):
            dt_scaled = (dt * self.time_scale).unsqueeze(-1).clamp(min=0.001, max=30.0)
        else:
            dt_scaled = (dt * self.time_scale).clamp(min=0.001, max=30.0)
        h_new = h_prev + dt_scaled * (-h_prev + f_xh) / tau

        h_new = self.layer_norm(h_new)
        h_new = self.dropout(h_new)
        return h_new


class LNNSingleStep(nn.Module):
    """
    Single-step Liquid Neural Network that processes Mamba's last hidden state.

    On-prem design: Mamba processes the full sequence and produces a hidden
    state at the last timestep.  LNN then takes this single hidden vector
    and applies a single LTC update step, enabling adaptive temporal dynamics
    on top of Mamba's SSM representation.

    This is a **serial** pipeline: Mamba -> LNN (not parallel).

    Parameters
    ----------
    input_dim : int
        Dimension of Mamba's last hidden state (= Mamba d_model).
    hidden_dim : int
        LNN hidden dimension (output dimension).
    n_layers : int
        Number of stacked LTC cells applied sequentially in one step.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Project Mamba's output dim to LNN hidden dim if they differ
        if input_dim != hidden_dim:
            self.input_proj: Optional[nn.Linear] = nn.Linear(input_dim, hidden_dim)
        else:
            self.input_proj = None

        # Stacked LTC cells for the single step
        self.cells = nn.ModuleList([
            LiquidTimeConstantCell(hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, mamba_last_hidden: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        mamba_last_hidden : torch.Tensor
            ``[batch, input_dim]`` -- Mamba's last timestep hidden state.

        Returns
        -------
        torch.Tensor
            ``[batch, hidden_dim]`` -- LNN-refined representation.
        """
        x = mamba_last_hidden
        if self.input_proj is not None:
            x = self.input_proj(x)

        # Initialise hidden states as zeros (single-step, no recurrence)
        batch = x.size(0)
        h_states = [
            torch.zeros(batch, self.hidden_dim, device=x.device, dtype=x.dtype)
            for _ in self.cells
        ]

        # Single-step: pass through stacked LTC cells
        x_t = x
        for layer_idx, cell in enumerate(self.cells):
            x_t = cell(x_t, h_states[layer_idx], dt=1.0)

        return self.norm(x_t)


class LiquidNeuralNetwork(nn.Module):
    """
    Stacked Liquid Time-Constant cells processing a full sequence.

    .. deprecated::
        This class is retained for backward compatibility. New code should
        use :class:`LNNSingleStep` in the serial Mamba -> LNN pipeline.

    Each layer consists of an :class:`LiquidTimeConstantCell`.  The network
    processes the sequence step-by-step and returns the mean-pooled hidden
    state across time.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Optional input projection
        if input_dim != hidden_dim:
            self.input_proj: Optional[nn.Linear] = nn.Linear(input_dim, hidden_dim)
        else:
            self.input_proj = None

        # Stacked LTC cells -- first cell receives hidden_dim input
        self.cells = nn.ModuleList([
            LiquidTimeConstantCell(hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        time_delta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            ``[batch, seq_len, input_dim]``
        time_delta : torch.Tensor or None
            ``[batch, seq_len]`` inter-event time deltas.  If ``None``,
            uniform spacing ``dt=1.0`` is assumed.

        Returns
        -------
        torch.Tensor
            ``[batch, hidden_dim]`` -- mean-pooled hidden states.
        """
        batch, seq_len, _ = x.shape

        # Project input if dimensions differ
        if self.input_proj is not None:
            x = self.input_proj(x)

        # Initialise hidden states for each layer
        h_states = [
            torch.zeros(batch, self.hidden_dim, device=x.device, dtype=x.dtype)
            for _ in self.cells
        ]

        # Collect final-layer hidden states for pooling
        collector: list[torch.Tensor] = []

        for t in range(seq_len):
            x_t = x[:, t, :]
            dt = time_delta[:, t] if time_delta is not None else 1.0
            for layer_idx, cell in enumerate(self.cells):
                x_t = cell(x_t, h_states[layer_idx], dt=dt)
                h_states[layer_idx] = x_t
            collector.append(x_t)

        # Mean pool across time
        stacked = torch.stack(collector, dim=1)  # [batch, seq_len, hidden_dim]
        pooled = stacked.mean(dim=1)  # [batch, hidden_dim]
        return self.norm(pooled)


# =============================================================================
# Temporal Ensemble Expert
# =============================================================================

@ExpertRegistry.register("temporal_ensemble")
class TemporalEnsembleExpert(AbstractExpert):
    """
    Temporal ensemble expert: Mamba + Transformer + LNN with learned gating.

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
    lnn.enabled : bool
        Enable Liquid Neural Network branch (default ``True``).
    lnn.hidden_dim : int
        LNN hidden dimension (default 64).
    lnn.n_layers : int
        Number of stacked LTC cells (default 2).
    ensemble_gating : bool
        Use learned gating (default ``True``).  If ``False``,
        concatenate + MLP fusion.
    dropout : float
        Dropout rate (default 0.2).
    """

    def __init__(self, input_dim: int, config: Dict[str, Any]):
        super().__init__(input_dim, config)
        self.expects_sequence = True

        self._output_dim: int = config.get("output_dim", 64)
        aux_input_dim: Optional[int] = config.get("aux_input_dim")
        dropout: float = config.get("dropout", 0.2)
        self.ensemble_gating: bool = config.get("ensemble_gating", True)

        # Sub-model configs (nested dicts)
        m_cfg = config.get("mamba", {})
        t_cfg = config.get("transformer", {})
        l_cfg = config.get("lnn", {})

        self.mamba_enabled: bool = m_cfg.get("enabled", True)
        self.transformer_enabled: bool = t_cfg.get("enabled", True)
        self.lnn_enabled: bool = l_cfg.get("enabled", True)

        num_models = self.mamba_enabled + self.transformer_enabled + self.lnn_enabled
        if num_models == 0:
            raise ValueError(
                "At least one sub-model (mamba, transformer, or lnn) must be enabled."
            )

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

        # -- LNN branch (serial: takes Mamba's last hidden state) -------------
        # On-prem design: Mamba -> LNN single-step (serial pipeline).
        # LNN input_dim = Mamba d_model (not raw input_dim).
        lnn_out_dim = 0
        if self.lnn_enabled:
            if not self.mamba_enabled:
                raise ValueError(
                    "LNN requires Mamba to be enabled (serial Mamba -> LNN pipeline). "
                    "Either enable Mamba or disable LNN."
                )
            l_hidden_dim = l_cfg.get("hidden_dim", 64)
            l_n_layers = l_cfg.get("n_layers", 2)

            # LNN receives Mamba's d_model as input (serial pipeline)
            self.lnn_single_step = LNNSingleStep(
                input_dim=m_d_model,
                hidden_dim=l_hidden_dim,
                n_layers=l_n_layers,
                dropout=dropout,
            )
            lnn_out_dim += l_hidden_dim

            if aux_input_dim is not None:
                m_aux_d = m_cfg.get("d_model", 128) // 2
                l_aux_d = l_hidden_dim // 2
                self.lnn_aux_single_step = LNNSingleStep(
                    input_dim=m_aux_d,
                    hidden_dim=l_aux_d,
                    n_layers=l_n_layers,
                    dropout=dropout,
                )
                lnn_out_dim += l_aux_d

        self._has_aux = aux_input_dim is not None
        total_dim = mamba_out_dim + transformer_out_dim + lnn_out_dim

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
            if self.lnn_enabled:
                model_dims.append(lnn_out_dim)
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
            "Mamba=%s, Transformer=%s, LNN=%s -> output=%d",
            input_dim, aux_input_dim,
            self.mamba_enabled, self.transformer_enabled, self.lnn_enabled,
            self._output_dim,
        )

        # Sprint 2 S4: HMM-smoothed ensemble gating (config single-source).
        # Disabled by default; activated via `set_hmm_routing(True, ...)` or
        # via `config["hmm_routing"]` block.
        self._hmm_routing_enabled: bool = False
        self._hmm_smoothing: float = 0.0
        self._hmm_transition: Optional[torch.Tensor] = None
        hmm_cfg = config.get("hmm_routing") or {}
        if hmm_cfg.get("enabled", False):
            self.set_hmm_routing(
                enabled=True,
                smoothing=float(hmm_cfg.get("smoothing", 0.1)),
                transition_prior=hmm_cfg.get("transition_prior"),
            )

    # ------------------------------------------------------------------
    # Sprint 2 S4: HMM routing
    # ------------------------------------------------------------------

    def set_hmm_routing(
        self,
        enabled: bool,
        smoothing: float = 0.1,
        transition_prior: Optional[Any] = None,
    ) -> None:
        """Toggle HMM-style smoothing on the ensemble gating weights.

        Args:
            enabled: If True, gating weights produced by ``self.gate`` are
                post-multiplied by a row-stochastic transition matrix before
                being applied to the sub-model outputs. This reduces
                per-sample variance of the gating decision (analogous to
                an HMM forward pass over a 1-step sequence).
            smoothing: Off-diagonal weight in the transition matrix
                (``0.0`` = identity = no smoothing; ``1 / n`` = uniform).
                Must be in ``[0, 1 / (n_models - 1)]``; clipped otherwise.
            transition_prior: Optional ``(n_models, n_models)`` tensor /
                list overriding the smoothing-based default. Rows are
                re-normalised to sum to 1.
        """
        if self.gate is None:
            # Either gating is off, or only one sub-model → nothing to smooth.
            self._hmm_routing_enabled = False
            return

        num_models = sum([
            self.mamba_enabled, self.transformer_enabled, self.lnn_enabled,
        ])
        if num_models <= 1:
            self._hmm_routing_enabled = False
            return

        self._hmm_routing_enabled = bool(enabled)
        if not enabled:
            return

        if transition_prior is not None:
            mat = torch.as_tensor(transition_prior, dtype=torch.float32)
            if mat.shape != (num_models, num_models):
                raise ValueError(
                    f"transition_prior shape {tuple(mat.shape)} "
                    f"must be ({num_models}, {num_models})"
                )
            row_sum = mat.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            mat = mat / row_sum
        else:
            s = max(0.0, min(smoothing, 1.0 / max(1, num_models - 1)))
            off = s
            diag = 1.0 - s * (num_models - 1)
            mat = torch.full(
                (num_models, num_models), off, dtype=torch.float32,
            )
            mat.fill_diagonal_(diag)

        self._hmm_smoothing = float(smoothing)
        # Store as buffer so it moves with .to(device) but doesn't train.
        self.register_buffer("_hmm_transition_buf", mat, persistent=False)
        self._hmm_transition = self._hmm_transition_buf  # alias
        logger.info(
            "HMM routing enabled: n_models=%d smoothing=%.3f",
            num_models, self._hmm_smoothing,
        )

    def _apply_hmm_smoothing(
        self, gate_weights: torch.Tensor
    ) -> torch.Tensor:
        """Multiply gate weights by the learned / configured transition
        matrix and re-normalise. Left alone when HMM routing is disabled
        or the transition buffer is missing."""
        if (not self._hmm_routing_enabled) or self._hmm_transition is None:
            return gate_weights
        smoothed = gate_weights @ self._hmm_transition.to(gate_weights.device)
        smoothed = smoothed.clamp(min=1e-12)
        return smoothed / smoothed.sum(dim=-1, keepdim=True)

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
        time_delta : torch.Tensor, optional
            ``[batch, seq_len]`` inter-event time deltas for the LNN branch.
            If ``None``, uniform spacing is assumed.

        Returns
        -------
        torch.Tensor
            ``[batch, output_dim]``
        """
        # Guard: if routing collapsed the input to 2D, add a length-1
        # sequence dimension so sub-models (Mamba, PatchTST, LNN) work.
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, features)

        aux_seq: Optional[torch.Tensor] = kwargs.get("aux_seq")
        time_delta: Optional[torch.Tensor] = kwargs.get("time_delta")
        outputs: list[torch.Tensor] = []

        # -- Mamba branch ------------------------------------------------------
        # Mamba processes the full sequence; we keep full output for LNN
        mamba_last_primary: Optional[torch.Tensor] = None
        mamba_last_aux: Optional[torch.Tensor] = None
        if self.mamba_enabled:
            mamba_full = self.mamba_primary(x)  # (batch, seq_len, d_model)
            mamba_last_primary = mamba_full[:, -1, :]  # (batch, d_model)
            parts = [mamba_last_primary]
            if self._has_aux and aux_seq is not None:
                mamba_aux_full = self.mamba_aux(aux_seq)
                mamba_last_aux = mamba_aux_full[:, -1, :]
                parts.append(mamba_last_aux)
            outputs.append(torch.cat(parts, dim=-1))

        # -- Transformer branch ------------------------------------------------
        if self.transformer_enabled:
            h_tf = self.transformer_primary(x)
            parts = [h_tf]
            if self._has_aux and aux_seq is not None:
                parts.append(self.transformer_aux(aux_seq))
            outputs.append(torch.cat(parts, dim=-1))

        # -- LNN branch (serial: Mamba last hidden -> LNN single step) --------
        if self.lnn_enabled:
            # mamba_last_primary is guaranteed non-None because __init__
            # enforces mamba_enabled when lnn_enabled
            h_lnn = self.lnn_single_step(mamba_last_primary)
            parts = [h_lnn]
            if self._has_aux and aux_seq is not None and mamba_last_aux is not None:
                parts.append(self.lnn_aux_single_step(mamba_last_aux))
            outputs.append(torch.cat(parts, dim=-1))

        # -- Ensemble ----------------------------------------------------------
        if self.gate is not None and len(outputs) > 1:
            concat = torch.cat(outputs, dim=-1)
            gate_weights = self.gate(concat)  # [B, num_models]
            # Sprint 2 S4: HMM-smoothed gating (no-op when disabled)
            gate_weights = self._apply_hmm_smoothing(gate_weights)
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
            # Compute gate weights directly without a full forward pass
            parts: list[torch.Tensor] = []
            aux_seq = kwargs.get("aux_seq")
            mamba_last_primary: Optional[torch.Tensor] = None
            mamba_last_aux: Optional[torch.Tensor] = None
            if self.mamba_enabled:
                mamba_full = self.mamba_primary(x)
                mamba_last_primary = mamba_full[:, -1, :]
                ps = [mamba_last_primary]
                if self._has_aux and aux_seq is not None:
                    mamba_last_aux = self.mamba_aux(aux_seq)[:, -1, :]
                    ps.append(mamba_last_aux)
                parts.append(torch.cat(ps, dim=-1))
            if self.transformer_enabled:
                h = self.transformer_primary(x)
                ps = [h]
                if self._has_aux and aux_seq is not None:
                    ps.append(self.transformer_aux(aux_seq))
                parts.append(torch.cat(ps, dim=-1))
            if self.lnn_enabled:
                h = self.lnn_single_step(mamba_last_primary)
                ps = [h]
                if self._has_aux and aux_seq is not None and mamba_last_aux is not None:
                    ps.append(self.lnn_aux_single_step(mamba_last_aux))
                parts.append(torch.cat(ps, dim=-1))

            concat = torch.cat(parts, dim=-1)
            gate_weights = self.gate(concat)
            entropy = -(gate_weights * torch.log2(gate_weights + 1e-8)).sum(dim=-1)
            return entropy.mean()
