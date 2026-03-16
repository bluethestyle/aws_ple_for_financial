"""
Mamba Block -- Selective State Space Model for efficient sequence modelling.

Implements the S6 (Selective Structured State Space) mechanism from the
Mamba paper.  Key properties:

* **Linear complexity** O(L) in sequence length (vs O(L^2) for attention).
* **Input-dependent** discretisation -- the SSM parameters (delta, B, C) are
  dynamically generated from each input token, enabling selective
  memorisation / forgetting.
* **Gated architecture** -- Conv1d for local patterns, SSM for global
  dependencies, element-wise gating for output.

This is a pure-PyTorch reference implementation using a sequential scan.
For production workloads with long sequences, consider replacing the scan
with the ``mamba-ssm`` package's fused CUDA kernel (associative-scan).

References
----------
Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*,
arXiv 2312.00752, 2023.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AbstractExpert
from .registry import ExpertRegistry

logger = logging.getLogger(__name__)


# =============================================================================
# Selective SSM core
# =============================================================================

class SelectiveSSM(nn.Module):
    """
    Selective State Space Model (S6 mechanism).

    Generates SSM parameters (delta, B, C) from the input itself so that the
    model can selectively remember or forget information at each time step.

    Parameters
    ----------
    d_model : int
        Model / channel dimension.
    d_state : int
        SSM latent state dimension *N*.
    dt_rank : int or None
        Rank for the delta projection (default ``ceil(d_model / 16)``).
    dt_min, dt_max : float
        Range for delta initialisation.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        dt_rank: Optional[int] = None,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank or math.ceil(d_model / 16)

        # Input-dependent parameter projections
        self.x_proj = nn.Linear(d_model, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)

        # State transition matrix A -- diagonal, initialised negative for stability
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_model, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # Skip connection parameter
        self.D = nn.Parameter(torch.ones(d_model))

        # Delta initialisation
        dt = torch.exp(
            torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            ``[batch, seq_len, d_model]``

        Returns
        -------
        torch.Tensor
            ``[batch, seq_len, d_model]``
        """
        batch, seq_len, _ = x.shape

        # Input-dependent parameters (S6)
        x_proj = self.x_proj(x)  # [B, L, dt_rank + 2*N]
        dt, B, C = torch.split(
            x_proj, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        dt = F.softplus(self.dt_proj(dt))  # [B, L, D]
        A = -torch.exp(self.A_log.float())  # [D, N]

        # Zero-Order Hold discretisation
        dA = torch.exp(torch.einsum("bld,dn->bldn", dt, A))  # [B,L,D,N]
        dB = torch.einsum("bld,bln->bldn", dt, B)            # [B,L,D,N]

        y = self._sequential_scan(x, dA, dB, C)
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        return y

    def _sequential_scan(
        self,
        x: torch.Tensor,
        dA: torch.Tensor,
        dB: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """
        Recurrent-mode sequential scan.

        For production, replace with a fused CUDA associative-scan kernel
        (e.g. ``mamba_ssm.ops.selective_scan``).
        """
        batch, seq_len, d_model = x.shape
        d_state = dA.shape[-1]

        dBx = dB * x.unsqueeze(-1)  # [B, L, D, N]
        h = torch.zeros(batch, d_model, d_state, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(seq_len):
            h = dA[:, t] * h + dBx[:, t]
            y_t = torch.einsum("bdn,bn->bd", h, C[:, t])
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)


# =============================================================================
# Mamba Block
# =============================================================================

class MambaBlock(nn.Module):
    """
    Single Mamba block: input projection -> LayerNorm -> gated (Conv1d + SSM).

    Parameters
    ----------
    d_model : int
        Hidden dimension.
    d_input : int
        Input feature dimension (projected to ``d_model`` internally).
    expand : int
        Expansion factor for the inner dimension (default 2).
    d_state : int
        SSM state dimension (default 16).
    d_conv : int
        1-D convolution kernel size (default 4).
    dropout : float
        Dropout rate (default 0.1).
    """

    def __init__(
        self,
        d_model: int,
        d_input: int,
        expand: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        dt_rank: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand

        self.input_proj = nn.Linear(d_input, d_model)
        self.norm = nn.LayerNorm(d_model)

        # Two parallel paths: SSM path + gate path
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Local pattern capture via depthwise conv
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        self.ssm = SelectiveSSM(
            d_model=self.d_inner,
            d_state=d_state,
            dt_rank=dt_rank,
        )

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            ``[batch, seq_len, d_input]``

        Returns
        -------
        torch.Tensor
            ``[batch, seq_len, d_model]``
        """
        x = self.input_proj(x)
        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)

        # Conv1d path
        x_conv = x_ssm.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, : x_ssm.shape[1]]  # causal trim
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)

        # SSM path
        x_ssm_out = self.ssm(x_conv)

        # Gated output
        z = F.silu(z)
        output = x_ssm_out * z

        output = self.out_proj(output)
        output = self.dropout(output) + residual
        return output


class StackedMambaBlocks(nn.Module):
    """
    Stack of ``n_layers`` :class:`MambaBlock` modules with a final LayerNorm.
    """

    def __init__(
        self,
        d_model: int,
        d_input: int,
        n_layers: int = 1,
        expand: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(
                MambaBlock(
                    d_model=d_model,
                    d_input=d_input if i == 0 else d_model,
                    expand=expand,
                    d_state=d_state,
                    d_conv=d_conv,
                    dropout=dropout,
                )
            )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            ``[batch, seq_len, d_input]``

        Returns
        -------
        torch.Tensor
            ``[batch, seq_len, d_model]``
        """
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


# =============================================================================
# Mamba as a registered AbstractExpert
# =============================================================================

@ExpertRegistry.register("mamba")
class MambaExpert(AbstractExpert):
    """
    Mamba expert that processes a sequence and returns the last hidden state.

    Config keys
    -----------
    output_dim : int
        Expert output dimension (default 64). A linear projection is applied
        from ``d_model`` to ``output_dim`` if they differ.
    d_model : int
        Hidden dimension of the Mamba block (default 128).
    n_layers : int
        Number of stacked Mamba blocks (default 1).
    expand : int
        Expansion factor (default 2).
    d_state : int
        SSM state dimension (default 16).
    d_conv : int
        Conv1d kernel size (default 4).
    dropout : float
        Dropout rate (default 0.1).
    pool : str
        Pooling strategy: ``"last"`` (default) or ``"mean"``.
    """

    def __init__(self, input_dim: int, config: Dict[str, Any]):
        super().__init__(input_dim, config)

        self._output_dim: int = config.get("output_dim", 64)
        d_model: int = config.get("d_model", 128)
        n_layers: int = config.get("n_layers", 1)
        expand: int = config.get("expand", 2)
        d_state: int = config.get("d_state", 16)
        d_conv: int = config.get("d_conv", 4)
        dropout: float = config.get("dropout", 0.1)
        self._pool: str = config.get("pool", "last")

        self.blocks = StackedMambaBlocks(
            d_model=d_model,
            d_input=input_dim,
            n_layers=n_layers,
            expand=expand,
            d_state=d_state,
            d_conv=d_conv,
            dropout=dropout,
        )

        self.proj = (
            nn.Linear(d_model, self._output_dim)
            if d_model != self._output_dim
            else nn.Identity()
        )

        logger.info(
            "MambaExpert: input=%d, d_model=%d, layers=%d, pool=%s -> output=%d",
            input_dim, d_model, n_layers, self._pool, self._output_dim,
        )

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            ``[batch, seq_len, input_dim]``

        Returns
        -------
        torch.Tensor
            ``[batch, output_dim]``
        """
        h = self.blocks(x)  # [B, L, d_model]
        if self._pool == "mean":
            h = h.mean(dim=1)
        else:
            h = h[:, -1, :]
        return self.proj(h)
