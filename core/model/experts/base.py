"""
Abstract base class for all Expert modules.

Every expert in the PLE architecture inherits from ``AbstractExpert``
and must implement:

* ``__init__(self, input_dim, config)`` -- build sub-modules
* ``forward(self, x, **kwargs) -> torch.Tensor`` -- expert representation
* ``output_dim`` property -- dimensionality of the expert's output

The base class provides common utilities (weight init, param counting)
and enforces a uniform interface so that the registry / PLE gating layer
can treat experts interchangeably.
"""

from __future__ import annotations

import abc
import logging
from typing import Any, Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class AbstractExpert(nn.Module, abc.ABC):
    """
    Base class for all expert networks.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input tensor.
    config : dict
        Expert-specific configuration. Sub-classes pull whatever keys
        they need; unknown keys are silently ignored.
    """

    def __init__(self, input_dim: int, config: Dict[str, Any]):
        super().__init__()
        self._input_dim = input_dim
        self._config = config

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abc.abstractmethod
    def output_dim(self) -> int:
        """Return the dimensionality of this expert's output."""
        ...

    @abc.abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute the expert representation.

        Parameters
        ----------
        x : torch.Tensor
            ``[batch, input_dim]`` (tabular) or
            ``[batch, seq_len, input_dim]`` (sequential).
        **kwargs
            Expert-specific extra inputs (e.g. ``time_deltas``).

        Returns
        -------
        torch.Tensor
            ``[batch, output_dim]`` expert output.
        """
        ...

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        return f"input_dim={self._input_dim}, output_dim={self.output_dim}"


# ---------------------------------------------------------------------------
# Weight initialisation utility
# ---------------------------------------------------------------------------

_INIT_FN_MAP = {
    "xavier_uniform": nn.init.xavier_uniform_,
    "xavier_normal": nn.init.xavier_normal_,
    "kaiming_normal": lambda w: nn.init.kaiming_normal_(w, nonlinearity="relu"),
    "kaiming_uniform": lambda w: nn.init.kaiming_uniform_(w, nonlinearity="relu"),
}


def init_expert_weights(expert: nn.Module, method: str = "xavier_uniform") -> int:
    """
    Apply a consistent weight initialisation strategy to all
    ``Linear`` / ``Conv1d`` / ``Conv2d`` layers in *expert*.

    Parameters
    ----------
    expert : nn.Module
        The expert (or any module) to initialise.
    method : str
        One of ``xavier_uniform``, ``xavier_normal``,
        ``kaiming_normal``, ``kaiming_uniform``.

    Returns
    -------
    int
        Number of layers initialised.

    Raises
    ------
    ValueError
        If *method* is not recognised.
    """
    init_fn = _INIT_FN_MAP.get(method)
    if init_fn is None:
        raise ValueError(
            f"Unknown init method: {method!r}. "
            f"Available: {list(_INIT_FN_MAP.keys())}"
        )

    count = 0
    for module in expert.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            init_fn(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            count += 1

    logger.debug("Expert weight init: method=%s, layers=%d", method, count)
    return count
