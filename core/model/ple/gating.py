"""
Gating Networks for PLE.

Provides three gating strategies for combining expert outputs:
  - **SoftmaxGate**: Static learnable weights with softmax normalization.
  - **AttentionGate**: Query-key attention between a learned task query
    and expert output projections.
  - **MLPGate**: Input-dependent gating via a small MLP.

All gates produce ``(batch, num_experts)`` weight vectors that are used
to combine expert outputs in the CGC layer.

These are *standalone* modules -- the CGC layer in ``experts.py`` uses
a simple linear gate by default, but these can be swapped in for
more expressive gating.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SoftmaxGate(nn.Module):
    """Static learnable softmax gate.

    Maintains a single weight vector that is softmax-normalized.
    The gate is input-independent (same weights for all samples),
    but the weights are learnable parameters.

    Args:
        num_experts: Number of experts to gate over.
        temperature: Softmax temperature (higher = more uniform).
        learnable: If ``False``, weights are frozen.
    """

    def __init__(
        self,
        num_experts: int,
        temperature: float = 1.0,
        learnable: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.gate_weights = nn.Parameter(
            torch.ones(num_experts) / num_experts,
            requires_grad=learnable,
        )

    def forward(
        self,
        x: torch.Tensor,
        expert_outputs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute gated output.

        Args:
            x: ``(batch, input_dim)`` -- input features (unused, kept for
               interface compatibility).
            expert_outputs: ``(batch, num_experts, hidden_dim)``.

        Returns:
            ``(batch, hidden_dim)`` -- weighted sum of expert outputs.
        """
        weights = F.softmax(
            self.gate_weights / self.temperature, dim=0,
        )  # (num_experts,)
        weights = weights.unsqueeze(0).expand(x.size(0), -1)  # (batch, num_experts)
        return (weights.unsqueeze(-1) * expert_outputs).sum(dim=1)

    def get_weights(self) -> torch.Tensor:
        """Return current softmax-normalized weights."""
        with torch.no_grad():
            return F.softmax(self.gate_weights / self.temperature, dim=0)


class AttentionGate(nn.Module):
    """Query-key attention gate.

    Each expert output is projected to a key space, and a learnable
    task-specific query computes scaled dot-product attention weights.

    Args:
        expert_dim: Dimension of each expert output.
        hidden_dim: Attention key/query dimension.
        num_experts: Number of experts.
        temperature: Softmax temperature.
    """

    def __init__(
        self,
        expert_dim: int,
        hidden_dim: int = 64,
        num_experts: int = 4,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.scale = hidden_dim ** 0.5

        # Learnable task query
        self.task_query = nn.Parameter(torch.randn(1, hidden_dim))

        # Key projection for expert outputs
        self.key_proj = nn.Linear(expert_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        expert_outputs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attention-gated output.

        Args:
            x: ``(batch, input_dim)`` -- input features (unused).
            expert_outputs: ``(batch, num_experts, expert_dim)``.

        Returns:
            ``(batch, expert_dim)`` -- attention-weighted sum.
        """
        batch_size = x.size(0)

        # Project expert outputs to key space: (batch, num_experts, hidden_dim)
        keys = self.key_proj(expert_outputs)

        # Query: (batch, 1, hidden_dim)
        query = self.task_query.expand(batch_size, -1).unsqueeze(1)

        # Attention scores: (batch, 1, num_experts)
        scores = torch.bmm(query, keys.transpose(1, 2)) / self.scale

        # Weights: (batch, num_experts)
        weights = F.softmax(scores.squeeze(1) / self.temperature, dim=-1)

        return (weights.unsqueeze(-1) * expert_outputs).sum(dim=1)


class MLPGate(nn.Module):
    """Input-dependent MLP gate.

    A small MLP takes the concatenated expert outputs and produces
    per-expert weights.  This is the most expressive gate type.

    Args:
        total_expert_dim: Total concatenated dimension of all expert outputs.
        num_experts: Number of experts.
        hidden_dim: MLP hidden layer width.
        temperature: Softmax temperature.
    """

    def __init__(
        self,
        total_expert_dim: int,
        num_experts: int,
        hidden_dim: int = 64,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.mlp = nn.Sequential(
            nn.Linear(total_expert_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(
        self,
        x: torch.Tensor,
        expert_outputs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MLP-gated output.

        Args:
            x: ``(batch, input_dim)`` -- input features (unused).
            expert_outputs: ``(batch, num_experts, expert_dim)``.

        Returns:
            ``(batch, expert_dim)`` -- gated sum.
        """
        # Concatenate all expert outputs for the MLP input
        concat = expert_outputs.flatten(start_dim=1)  # (batch, total_expert_dim)
        logits = self.mlp(concat)
        weights = F.softmax(logits / self.temperature, dim=-1)
        return (weights.unsqueeze(-1) * expert_outputs).sum(dim=1)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_GATE_REGISTRY = {
    "softmax": SoftmaxGate,
    "attention": AttentionGate,
    "mlp": MLPGate,
}


def build_gate(gate_type: str, **kwargs) -> nn.Module:
    """Factory function to create a gate by type string.

    Args:
        gate_type: One of ``"softmax"``, ``"attention"``, ``"mlp"``.
        **kwargs: Passed to the gate constructor.

    Returns:
        Instantiated gate module.
    """
    if gate_type not in _GATE_REGISTRY:
        raise ValueError(
            f"Unknown gate type '{gate_type}'. Options: {list(_GATE_REGISTRY)}"
        )
    return _GATE_REGISTRY[gate_type](**kwargs)
