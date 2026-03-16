"""
Expert Networks for PLE.

Provides the base expert abstraction, concrete MLP expert, and the
Customized Gate Control (CGC) layer that combines shared + task-specific
expert outputs per task.

Architecture (per CGC layer, per task *k*):
    shared_experts  -> [E_s1, E_s2, ...]   -> stack -> \\
    task_experts[k] -> [E_tk1, E_tk2, ...] -> stack ->  > gate_k -> task_output_k
    input_x  ------------------------------------------/

Design decisions:
  - Expert types are loaded from ``ExpertRegistry`` (not hardcoded).
  - Input/output dimensions come from ``PLEConfig``.
  - All ``nn.Module`` classes accept and return ``torch.Tensor``.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from .feature_router import FeatureRouter

logger = logging.getLogger(__name__)


# ============================================================================
# Expert Registry
# ============================================================================

class ExpertRegistry:
    """Plugin registry for expert network types.

    Experts are registered by string key and instantiated via
    ``ExpertRegistry.build(key, **kwargs)``.  The default ``"mlp"`` expert
    is auto-registered at module load time.

    Usage::

        @ExpertRegistry.register("my_expert")
        class MyExpert(BaseExpert):
            ...

        expert = ExpertRegistry.build("my_expert", input_dim=128, output_dim=64)
    """

    _registry: Dict[str, Type[BaseExpert]] = {}

    @classmethod
    def register(cls, key: str):
        """Decorator to register an expert class."""
        def decorator(expert_cls: Type[BaseExpert]):
            cls._registry[key] = expert_cls
            return expert_cls
        return decorator

    @classmethod
    def build(cls, key: str, **kwargs) -> "BaseExpert":
        """Instantiate a registered expert by key."""
        if key not in cls._registry:
            raise KeyError(
                f"Unknown expert type '{key}'. "
                f"Registered: {list(cls._registry.keys())}"
            )
        return cls._registry[key](**kwargs)

    @classmethod
    def list_registered(cls) -> List[str]:
        return list(cls._registry.keys())


# ============================================================================
# Base Expert
# ============================================================================

class BaseExpert(ABC, nn.Module):
    """Abstract base class for all expert networks.

    Every expert maps ``(batch, input_dim) -> (batch, output_dim)``
    through a learnable transformation.

    Args:
        input_dim: Width of the input feature vector.
        output_dim: Width of the output representation.
        dropout: Dropout probability applied inside the expert.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform input features.

        Args:
            x: ``(batch, input_dim)`` tensor.

        Returns:
            ``(batch, output_dim)`` tensor.
        """


# ============================================================================
# MLP Expert
# ============================================================================

@ExpertRegistry.register("mlp")
class MLPExpert(BaseExpert):
    """Multi-layer perceptron expert.

    Supports configurable depth, normalization, and activation.

    Args:
        input_dim: Input feature width.
        output_dim: Output representation width.
        hidden_dims: List of hidden layer widths (e.g. ``[256, 256]``).
        dropout: Dropout probability.
        activation: One of ``"relu"``, ``"silu"``, ``"gelu"``, ``"leaky_relu"``.
        use_layer_norm: Apply LayerNorm after each hidden layer.
        use_batch_norm: Apply BatchNorm1d (ignored if ``use_layer_norm=True``).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        use_layer_norm: bool = True,
        use_batch_norm: bool = False,
    ):
        super().__init__(input_dim, output_dim, dropout)
        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.network = self._build(
            input_dim, output_dim, hidden_dims,
            dropout, activation, use_layer_norm, use_batch_norm,
        )

    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
            "gelu": nn.GELU(),
            "leaky_relu": nn.LeakyReLU(0.2),
        }
        if name not in activations:
            raise ValueError(f"Unknown activation '{name}'. Options: {list(activations)}")
        return activations[name]

    @staticmethod
    def _build(
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        dropout: float,
        activation: str,
        use_layer_norm: bool,
        use_batch_norm: bool,
    ) -> nn.Sequential:
        layers: List[nn.Module] = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            elif use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(MLPExpert._get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ============================================================================
# CGC Layer (Customized Gate Control)
# ============================================================================

class CGCLayer(nn.Module):
    """Customized Gate Control layer -- the core building block of PLE.

    For each task, a gating network learns to combine:
      - ``num_shared_experts`` shared experts (seen by all tasks)
      - ``num_task_experts`` task-specific experts (private to each task)

    The output for each task is a weighted sum of all expert outputs,
    preserving the expert hidden dimension.

    Optionally accepts a ``FeatureRouter`` to route different feature
    subsets to different experts. When no router is provided, all experts
    receive the full input tensor (backward compatible).

    Args:
        input_dim: Input feature width for this layer (used for gating
            and as the default expert input width when no router is used).
        num_tasks: Number of tasks.
        num_shared_experts: Number of shared experts.
        num_task_experts: Number of per-task experts.
        expert_hidden_dim: Internal width of each expert.
        dropout: Dropout probability.
        expert_type: Key into ``ExpertRegistry`` (default ``"mlp"``).
        expert_hidden_dims: Hidden layer widths within each expert.
        feature_router: Optional ``FeatureRouter`` for expert-specific
            feature routing. When ``None``, all experts receive the full
            input tensor.
        shared_expert_names: Names for shared experts (required when
            ``feature_router`` is provided). Defaults to
            ``["shared_0", "shared_1", ...]``.
    """

    def __init__(
        self,
        input_dim: int,
        num_tasks: int,
        num_shared_experts: int = 4,
        num_task_experts: int = 1,
        expert_hidden_dim: int = 64,
        dropout: float = 0.1,
        expert_type: str = "mlp",
        expert_hidden_dims: Optional[List[int]] = None,
        feature_router: Optional["FeatureRouter"] = None,
        shared_expert_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_shared_experts = num_shared_experts
        self.num_task_experts = num_task_experts
        self.expert_hidden_dim = expert_hidden_dim
        self.feature_router = feature_router

        if expert_hidden_dims is None:
            expert_hidden_dims = [expert_hidden_dim]

        # Generate expert names
        if shared_expert_names is None:
            shared_expert_names = [f"shared_{i}" for i in range(num_shared_experts)]
        self.shared_expert_names = shared_expert_names

        # Build shared experts -- each may have a different input_dim if routed
        self.shared_experts = nn.ModuleList()
        for i in range(num_shared_experts):
            expert_name = self.shared_expert_names[i]
            if feature_router is not None and expert_name in feature_router.expert_names:
                expert_in_dim = feature_router.get_expert_input_dim(expert_name)
            else:
                expert_in_dim = input_dim

            self.shared_experts.append(
                ExpertRegistry.build(
                    expert_type,
                    input_dim=expert_in_dim,
                    output_dim=expert_hidden_dim,
                    hidden_dims=expert_hidden_dims,
                    dropout=dropout,
                )
            )

        # Per-task experts (these always receive the full input -- routing
        # is only for shared experts, which represent domain-specific
        # feature extractors like PersLay, HGCN, Mamba, etc.)
        task_expert_kwargs = dict(
            input_dim=input_dim,
            output_dim=expert_hidden_dim,
            hidden_dims=expert_hidden_dims,
            dropout=dropout,
        )
        self.task_experts = nn.ModuleList([
            nn.ModuleList([
                ExpertRegistry.build(expert_type, **task_expert_kwargs)
                for _ in range(num_task_experts)
            ])
            for _ in range(num_tasks)
        ])

        # Per-task gating networks (gate always receives full input_dim)
        num_total_experts = num_shared_experts + num_task_experts
        self.gating = nn.ModuleList([
            nn.Linear(input_dim, num_total_experts)
            for _ in range(num_tasks)
        ])

    def forward(
        self,
        shared_input: torch.Tensor,
        task_inputs: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Forward pass through one CGC layer.

        Args:
            shared_input: ``(batch, input_dim)`` -- input to shared experts.
                When a ``FeatureRouter`` is configured, this is the full
                concatenated feature tensor; the router handles slicing.
            task_inputs: List of ``(batch, input_dim)`` per task.

        Returns:
            Tuple of:
              - List of ``(batch, expert_hidden_dim)`` per task (gated outputs).
              - ``(batch, num_shared * expert_hidden_dim)`` concatenated shared
                expert outputs (for downstream CGC attention / monitoring).
        """
        # Shared expert outputs: route features if router is available
        shared_expert_outputs: List[torch.Tensor] = []
        for i, expert in enumerate(self.shared_experts):
            expert_name = self.shared_expert_names[i]
            if (self.feature_router is not None
                    and expert_name in self.feature_router.expert_names):
                expert_input = self.feature_router.route(shared_input, expert_name)
            else:
                expert_input = shared_input
            shared_expert_outputs.append(expert(expert_input))

        shared_outs = torch.stack(shared_expert_outputs, dim=1)

        # Concatenated shared outputs for downstream use
        shared_concat = torch.cat(shared_expert_outputs, dim=-1)

        outputs: List[torch.Tensor] = []
        for task_idx in range(self.num_tasks):
            # Task-specific expert outputs (always full input)
            task_outs = torch.stack(
                [expert(task_inputs[task_idx]) for expert in self.task_experts[task_idx]],
                dim=1,
            )
            # Combine: (batch, num_total, hidden)
            all_outs = torch.cat([task_outs, shared_outs], dim=1)

            # Gate: (batch, num_total)
            gate_logits = self.gating[task_idx](task_inputs[task_idx])
            gate_weights = F.softmax(gate_logits, dim=-1)

            # Weighted sum: (batch, hidden)
            gated = (gate_weights.unsqueeze(-1) * all_outs).sum(dim=1)
            outputs.append(gated)

        return outputs, shared_concat


# ============================================================================
# CGC Attention (post-hoc expert weighting per task)
# ============================================================================

class CGCAttention(nn.Module):
    """Per-task attention over concatenated shared expert outputs.

    When experts have heterogeneous output dimensions, this module can
    optionally normalize contributions via ``dim_normalize=True``.

    This is the "second-stage" CGC used in PLE-Cluster-adaTT: after the
    shared experts produce a concatenated vector, each task applies a
    learned softmax attention to scale each expert's block.

    Args:
        task_names: List of task name strings.
        expert_dims: List of output dimensions per shared expert.
        expert_names: List of shared expert name strings (for logging).
        bias_high: Initial bias for domain-relevant experts.
        bias_low: Initial bias for other experts.
        dim_normalize: Scale expert blocks by ``sqrt(mean_dim / dim)``
            to equalize contribution when expert dims differ.
        domain_experts_map: ``{task_name: [expert_name, ...]}`` for
            initializing attention bias.
    """

    def __init__(
        self,
        task_names: List[str],
        expert_dims: List[int],
        expert_names: Optional[List[str]] = None,
        bias_high: float = 1.0,
        bias_low: float = -1.0,
        dim_normalize: bool = False,
        domain_experts_map: Optional[Dict[str, List[str]]] = None,
    ):
        super().__init__()
        self.task_names = task_names
        self.expert_dims = expert_dims
        self.n_experts = len(expert_dims)
        self.dim_normalize = dim_normalize
        self._expert_names = expert_names or [f"expert_{i}" for i in range(self.n_experts)]
        self._mean_dim = sum(expert_dims) / max(len(expert_dims), 1)

        concat_dim = sum(expert_dims)
        domain_map = domain_experts_map or {}

        self.attention_modules = nn.ModuleDict()
        for task_name in task_names:
            attn = nn.Sequential(
                nn.Linear(concat_dim, self.n_experts),
                nn.Softmax(dim=-1),
            )
            # Apply bias initialization
            domain_experts = domain_map.get(task_name, [])
            if domain_experts:
                linear = attn[0]
                with torch.no_grad():
                    linear.weight.zero_()
                    for i, ename in enumerate(self._expert_names):
                        linear.bias[i] = bias_high if ename in domain_experts else bias_low

            self.attention_modules[task_name] = attn

    def forward(
        self,
        shared_concat: torch.Tensor,
        task_name: str,
    ) -> torch.Tensor:
        """Apply per-task attention scaling to the shared expert concat.

        Args:
            shared_concat: ``(batch, sum(expert_dims))``.
            task_name: Which task's attention to use.

        Returns:
            ``(batch, sum(expert_dims))`` -- scaled expert concat.
        """
        if task_name not in self.attention_modules:
            return shared_concat

        weights = self.attention_modules[task_name](shared_concat)  # (batch, n_experts)

        parts: List[torch.Tensor] = []
        offset = 0
        for i, dim in enumerate(self.expert_dims):
            block = shared_concat[:, offset:offset + dim]
            if self.dim_normalize and dim != self._mean_dim:
                block = block * math.sqrt(self._mean_dim / dim)
            parts.append(block * weights[:, i:i + 1])
            offset += dim

        return torch.cat(parts, dim=-1)

    def get_attention_weights(
        self,
        shared_concat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Return attention weights for all tasks (for monitoring).

        Returns:
            ``{task_name: (batch, n_experts)}`` dict.
        """
        result: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for task_name, attn in self.attention_modules.items():
                result[task_name] = attn(shared_concat)
        return result

    def entropy_regularization(
        self,
        shared_concat: torch.Tensor,
    ) -> torch.Tensor:
        """Negative entropy regularizer to prevent expert collapse.

        Minimizing this value maximizes entropy (uniform attention), which
        prevents CGC from collapsing onto a single expert.

        Returns:
            Scalar negative-entropy loss (lower = more uniform).
        """
        total = torch.tensor(0.0, device=shared_concat.device)
        for attn in self.attention_modules.values():
            w = attn(shared_concat)  # (batch, n_experts)
            log_w = torch.log(w.clamp(min=1e-8))
            entropy = -(w * log_w).sum(dim=-1).mean()
            total = total - entropy
        return total / max(len(self.attention_modules), 1)
