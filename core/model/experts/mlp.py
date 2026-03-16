"""
MLP Expert -- simple multi-layer perceptron baseline.

This is the default expert type: a straightforward feedforward network
with configurable hidden layers, normalisation, and dropout.
It serves as the baseline against which specialised experts
(DeepFM, Causal, OT, ...) are compared.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .base import AbstractExpert
from .registry import ExpertRegistry

logger = logging.getLogger(__name__)


@ExpertRegistry.register("mlp")
class MLPExpert(AbstractExpert):
    """
    Multi-Layer Perceptron expert.

    Config keys
    -----------
    output_dim : int
        Output dimensionality (default 64).
    hidden_dims : list[int]
        Hidden layer sizes (default ``[128, 64]``).
    dropout : float
        Dropout rate (default 0.2).
    use_layer_norm : bool
        Use LayerNorm instead of BatchNorm (default ``True``).
    use_batch_norm : bool
        Use BatchNorm (default ``False``). Ignored if ``use_layer_norm``
        is ``True``.
    activation : str
        ``"relu"`` or ``"silu"`` (default ``"relu"``).
    """

    def __init__(self, input_dim: int, config: Dict[str, Any]):
        super().__init__(input_dim, config)

        self._output_dim: int = config.get("output_dim", 64)
        hidden_dims: List[int] = config.get("hidden_dims", [128, 64])
        dropout: float = config.get("dropout", 0.2)
        use_layer_norm: bool = config.get("use_layer_norm", True)
        use_batch_norm: bool = config.get("use_batch_norm", False)
        activation: str = config.get("activation", "relu")

        act_cls = nn.SiLU if activation == "silu" else nn.ReLU

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hdim))
            elif use_batch_norm:
                layers.append(nn.BatchNorm1d(hdim))
            layers.append(act_cls())
            layers.append(nn.Dropout(dropout))
            prev_dim = hdim

        layers.append(nn.Linear(prev_dim, self._output_dim))

        self.network = nn.Sequential(*layers)

        logger.info(
            "MLPExpert: input=%d -> hidden=%s -> output=%d  (params=%s)",
            input_dim, hidden_dims, self._output_dim,
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
            ``[batch, input_dim]``

        Returns
        -------
        torch.Tensor
            ``[batch, output_dim]``
        """
        return self.network(x)
