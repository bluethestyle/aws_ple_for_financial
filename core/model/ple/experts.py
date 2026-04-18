"""
Expert Networks for PLE.

Provides the base expert abstraction, concrete MLP expert, the
Customized Gate Control (CGC) layer that combines shared + task-specific
expert outputs per task, and the Expert Basket for config-driven expert
subset selection.

Architecture (per CGC layer, per task *k*):
    shared_experts  -> [E_s1, E_s2, ...]   -> stack -> \\
    task_experts[k] -> [E_tk1, E_tk2, ...] -> stack ->  > gate_k -> task_output_k
    input_x  ------------------------------------------/

3-tier expert selection:
    Expert Pool   -- all experts registered via @ExpertRegistry.register
    Expert Basket -- config-defined subset selected from the pool
    CGC Gating    -- runtime weighted selection from the basket

Design decisions:
  - Expert types are loaded from ``ExpertRegistry`` (not hardcoded).
  - Input/output dimensions come from ``PLEConfig``.
  - All ``nn.Module`` classes accept and return ``torch.Tensor``.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from .config import ExpertBasketConfig
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
        gate_type: str = "softmax",
        fusion_type: str = "cgc",
        native_residual_weight_init: float = 1.0,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.gate_type = gate_type
        self.num_shared_experts = num_shared_experts
        self.num_task_experts = num_task_experts
        self.expert_hidden_dim = expert_hidden_dim
        self.feature_router = feature_router
        self._last_gate_weights: Dict[int, torch.Tensor] = {}

        _valid_fusion = ("cgc", "adatt_sp", "residual_complement", "eceb")
        if fusion_type not in _valid_fusion:
            raise ValueError(
                f"fusion_type must be one of {_valid_fusion}, got {fusion_type!r}"
            )
        self.fusion_type = fusion_type
        # Native expert residual (AdaTT-sp, Li et al. KDD 2023).
        # Learnable scalar so the network can weight or suppress the residual.
        if fusion_type == "adatt_sp":
            self.native_residual_weight = nn.Parameter(
                torch.tensor(float(native_residual_weight_init))
            )
        else:
            self.native_residual_weight = None

        # Residual recovery (Paper 3): recover signal from experts *not*
        # selected by the gate, within the same task. Mutually exclusive
        # with adatt_sp at this layer.
        if fusion_type == "residual_complement":
            self.residual_recovery_weight = nn.Parameter(
                torch.tensor(float(native_residual_weight_init))
            )
        else:
            self.residual_recovery_weight = None

        # ECEB (uncertainty-conditioned recovery, Paper 3 MV).
        # Per-task learnable scalar modulated at forward time by the gate's
        # normalised entropy — high entropy (gate confused) → recovery on;
        # low entropy (gate confident) → recovery suppressed.
        if fusion_type == "eceb":
            self.eceb_gate = nn.Parameter(
                torch.full((num_tasks,), float(native_residual_weight_init))
            )
        else:
            self.eceb_gate = None

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

        # Per-task gating networks
        # On-prem design: gate on concatenated expert outputs (post-hoc),
        # not on raw input.  This lets the gate see the actual expert
        # representations before deciding how to combine them.
        num_total_experts = num_shared_experts + num_task_experts
        gate_input_dim = num_total_experts * expert_hidden_dim
        self.gating = nn.ModuleList([
            nn.Linear(gate_input_dim, num_total_experts)
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

            # Handle 3D sequence tensors: flatten to 2D for non-sequence
            # experts; keep 3D for sequence-aware experts (Mamba, etc.).
            if expert_input.dim() == 3 and not getattr(expert, "expects_sequence", False):
                batch_size = expert_input.size(0)
                expert_input = expert_input.reshape(batch_size, -1)

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

            # Gate on concatenated expert outputs (post-hoc, on-prem design)
            # all_outs: (batch, num_total, hidden) -> flatten to (batch, num_total * hidden)
            gate_input = all_outs.reshape(all_outs.size(0), -1)
            gate_logits = self.gating[task_idx](gate_input)
            if self.gate_type == "sigmoid":
                gate_weights = torch.sigmoid(gate_logits)
                gate_weights = gate_weights / (gate_weights.sum(dim=-1, keepdim=True) + 1e-6)
            else:
                gate_weights = F.softmax(gate_logits, dim=-1)

            # Weighted sum: (batch, hidden)
            gated = (gate_weights.unsqueeze(-1) * all_outs).sum(dim=1)

            # AdaTT-sp native expert residual (Li et al., KDD 2023).
            # After gated weighted sum, add the task's own task-specific
            # experts' mean output back as a residual so the native expert
            # always contributes regardless of what the gate learns.
            if self.fusion_type == "adatt_sp":
                native = task_outs.mean(dim=1)  # (batch, hidden)
                gated = gated + self.native_residual_weight * native

            # Residual recovery --- complementary gate (Paper 3, M1).
            # Primary = gated weighted sum (experts the gate "selected").
            # Residual = weighted sum with (1 - gate) weights renormalised
            # over the expert axis — this recovers signal from experts the
            # gate down-weighted, *within the same task*, without any
            # cross-task mixing.
            elif self.fusion_type == "residual_complement":
                complement = (1.0 - gate_weights).clamp(min=0.0)  # (batch, num_total)
                complement_sum = complement.sum(dim=-1, keepdim=True)
                # Avoid division by zero when the gate saturates.
                complement = complement / (complement_sum + 1e-6)
                residual = (complement.unsqueeze(-1) * all_outs).sum(dim=1)
                gated = gated + self.residual_recovery_weight * residual

            # ECEB (Paper 3 MV): uncertainty-conditioned task-agnostic
            # recovery. Recovery path = mean over all experts (gate-
            # independent "consensus"). Recovery weight is scaled by the
            # *normalised* entropy of the gate's own distribution per sample
            # — high entropy (gate confused) → recovery on; low entropy
            # (gate confident) → recovery near zero.
            elif self.fusion_type == "eceb":
                num_total = gate_weights.size(-1)
                # Per-sample entropy of the task's gate distribution.
                gate_entropy = -(
                    gate_weights * (gate_weights + 1e-8).log()
                ).sum(dim=-1)  # (batch,)
                max_entropy = math.log(float(num_total))
                entropy_ratio = (gate_entropy / max_entropy).clamp(0.0, 1.0)  # (batch,)
                # Task-agnostic consensus = mean over all experts.
                consensus = all_outs.mean(dim=1)  # (batch, hidden)
                # Per-task scalar gate (sigmoid) modulated by per-sample entropy.
                task_gate = torch.sigmoid(self.eceb_gate[task_idx])  # scalar
                recovery_weight = (task_gate * entropy_ratio).unsqueeze(-1)  # (batch, 1)
                gated = gated + recovery_weight * consensus

            outputs.append(gated)
            if not self.training:
                self._last_gate_weights[task_idx] = gate_weights.detach()

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
            with torch.amp.autocast('cuda', enabled=False):
                w_f32 = w.float()
                log_w = torch.log(w_f32.clamp(min=1e-6))
                entropy = -(w_f32 * log_w).sum(dim=-1).mean()
            total = total - entropy
        return total / max(len(self.attention_modules), 1)


# ============================================================================
# Expert Basket
# ============================================================================

class ExpertBasket:
    """Config-driven expert subset selection from the Expert Pool.

    The Expert Basket implements the middle tier of the 3-tier expert
    selection architecture:

    1. **Expert Pool** -- all experts registered via
       ``@ExpertRegistry.register`` in ``core.model.experts.registry``
       (e.g. 11 total).  Uses ``AbstractExpert(input_dim, config)``.
    2. **Expert Basket** -- a config-defined subset selected from the
       pool for a specific pipeline (this class).
    3. **CGC Gating** -- runtime weighted selection from the basket
       (handled by :class:`CGCLayer`).

    .. note:: Dual-registry architecture

       There are two ``ExpertRegistry`` classes in the codebase:

       - ``core.model.experts.registry.ExpertRegistry`` (the **Pool
         Registry**) -- uses ``AbstractExpert(input_dim, config)`` and
         is where new experts (HGCN, PersLay, etc.) are registered.
       - ``core.model.ple.experts.ExpertRegistry`` (the **PLE
         Registry**) -- uses ``BaseExpert(input_dim, output_dim,
         dropout)`` and is used by :class:`CGCLayer` for default expert
         construction.

       The Expert Basket bridges these two registries: it builds
       heterogeneous experts from the Pool Registry and assigns them to
       ``CGCLayer.shared_experts``.  This works because
       ``CGCLayer.forward()`` only calls ``expert(input)`` on each
       shared expert, which is satisfied by both ``AbstractExpert``
       and ``BaseExpert`` subclasses (both are ``nn.Module`` with a
       ``forward(x) -> Tensor`` method).

    The basket validates that all requested expert names exist in the
    pool, builds only the selected experts with their config overrides,
    and provides introspection methods for monitoring.

    Args:
        basket_config: :class:`ExpertBasketConfig` defining which experts
            to select and how to configure them.
        input_dim: Default input dimension for expert construction.
        default_output_dim: Default output dimension when not overridden.
        default_hidden_dims: Default hidden layer sizes for experts that
            accept ``hidden_dims``.
        dropout: Default dropout rate.

    Usage::

        from core.model.ple.config import ExpertBasketConfig
        from core.model.ple.experts import ExpertBasket

        basket_cfg = ExpertBasketConfig(
            shared_experts=["deepfm", "hgcn", "perslay"],
            task_experts=["mlp", "deepfm"],
            expert_configs={"hgcn": {"output_dim": 128}},
        )
        basket = ExpertBasket(basket_cfg, input_dim=644, default_output_dim=64)
        shared = basket.build_shared_experts()  # nn.ModuleList of 3 experts
        task = basket.build_task_expert("mlp")   # single expert instance
    """

    def __init__(
        self,
        basket_config: "ExpertBasketConfig",
        input_dim: int,
        default_output_dim: int = 64,
        default_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
    ):
        from core.model.experts import ExpertRegistry as PoolRegistry

        self._config = basket_config
        self._input_dim = input_dim
        self._default_output_dim = default_output_dim
        self._default_hidden_dims = default_hidden_dims or [256, 256]
        self._dropout = dropout
        self._pool_registry = PoolRegistry

        # Validate all requested experts exist in the pool
        pool_names = set(PoolRegistry.list_available())
        all_requested = set(basket_config.shared_experts) | set(basket_config.task_experts)
        # Include group-specific task experts in validation
        for group_experts in basket_config.group_task_experts.values():
            all_requested |= set(group_experts)
        missing = all_requested - pool_names
        if missing:
            raise ValueError(
                f"Expert Basket references unregistered experts: {sorted(missing)}. "
                f"Expert Pool contains: {sorted(pool_names)}"
            )

        # Log the basket selection summary
        logger.info(
            "Expert Pool: %d registered %s -> Basket: %d shared + %d task selected",
            len(pool_names),
            sorted(pool_names),
            len(basket_config.shared_experts),
            len(basket_config.task_experts),
        )
        if basket_config.group_task_experts:
            logger.info(
                "  Group task experts: %s",
                basket_config.group_task_experts,
            )
        logger.info(
            "  Shared basket: %s",
            basket_config.shared_experts,
        )
        logger.info(
            "  Task basket: %s",
            basket_config.task_experts,
        )
        if basket_config.expert_configs:
            logger.info(
                "  Config overrides for: %s",
                list(basket_config.expert_configs.keys()),
            )

    def _get_expert_config(self, expert_name: str) -> Dict[str, Any]:
        """Build the config dict for a specific expert.

        Merges default values with any per-expert overrides from
        ``ExpertBasketConfig.expert_configs``.
        """
        base_cfg: Dict[str, Any] = {
            "output_dim": self._default_output_dim,
            "hidden_dims": list(self._default_hidden_dims),
            "dropout": self._dropout,
        }
        overrides = self._config.expert_configs.get(expert_name, {})
        base_cfg.update(overrides)
        return base_cfg

    def build_shared_experts(
        self,
        input_dim_overrides: Optional[Dict[str, int]] = None,
    ) -> nn.ModuleList:
        """Build all shared experts defined in the basket.

        All shared experts are forced to use ``self._default_output_dim``
        regardless of any per-expert ``output_dim`` override in
        ``expert_configs``.  This is required because
        :meth:`CGCLayer.forward` stacks all shared expert outputs with
        ``torch.stack``, which requires identical tensor shapes.

        Args:
            input_dim_overrides: Optional dict mapping expert name to its
                actual input dimension.  When provided (e.g. from
                ``FeatureRouter._expert_input_dims`` or
                ``PLEConfig.expert_input_dims``), each expert is built
                with its routed input size instead of the full feature
                tensor width.  Experts not in the dict fall back to
                ``self._input_dim``.

        Returns:
            ``nn.ModuleList`` of expert modules, one per entry in
            ``ExpertBasketConfig.shared_experts``.
        """
        overrides = input_dim_overrides or {}
        experts = nn.ModuleList()
        for name in self._config.shared_experts:
            cfg = self._get_expert_config(name)
            # Force uniform output_dim so torch.stack in CGCLayer.forward()
            # does not fail with a shape mismatch (CRITICAL: all shared
            # experts must produce (batch, default_output_dim) tensors).
            cfg["output_dim"] = self._default_output_dim
            expert_input = overrides.get(name, self._input_dim)
            expert = self._pool_registry.create(
                name,
                input_dim=expert_input,
                config=cfg,
            )
            if expert_input != self._input_dim:
                logger.info(
                    "ExpertBasket: '%s' input_dim=%d (override from %d)",
                    name, expert_input, self._input_dim,
                )
            experts.append(expert)
        return experts

    def build_task_expert(self, expert_name: str) -> nn.Module:
        """Build a single task expert by name.

        Args:
            expert_name: Registered expert name from the basket's
                ``task_experts`` list.

        Returns:
            An instantiated expert module.
        """
        cfg = self._get_expert_config(expert_name)
        return self._pool_registry.create(
            expert_name,
            input_dim=self._input_dim,
            config=cfg,
        )

    def build_task_experts_for_task(self) -> nn.ModuleList:
        """Build one expert for each entry in the task expert list.

        Returns:
            ``nn.ModuleList`` of task expert modules.
        """
        experts = nn.ModuleList()
        for name in self._config.task_experts:
            experts.append(self.build_task_expert(name))
        return experts

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_pool(self) -> List[str]:
        """Return all expert names registered in the Expert Pool."""
        return self._pool_registry.list_available()

    def list_basket(self) -> Dict[str, List[str]]:
        """Return the basket selection as ``{shared: [...], task: [...]}``.

        Returns:
            Dict with ``"shared"`` and ``"task"`` keys mapping to lists
            of expert names selected for this pipeline.
        """
        return {
            "shared": list(self._config.shared_experts),
            "task": list(self._config.task_experts),
        }

    @property
    def num_shared_experts(self) -> int:
        """Number of shared experts in the basket."""
        return len(self._config.shared_experts)

    @property
    def num_task_experts(self) -> int:
        """Number of task expert types in the basket."""
        return len(self._config.task_experts)

    @property
    def shared_expert_names(self) -> List[str]:
        """Names of shared experts in the basket."""
        return list(self._config.shared_experts)

    @property
    def task_expert_names(self) -> List[str]:
        """Names of task experts in the basket."""
        return list(self._config.task_experts)


# ============================================================================
# BRP — Residual Expert Bank (Paper 3 MV)
# ============================================================================

class ResidualExpertBank(nn.Module):
    """Per-task residual expert producing logit residual at output scale.

    Operates outside the CGC gate. Takes the last extraction layer's
    ``shared_concat`` (concatenated shared-expert outputs) as a
    gate-bypass feature view and produces per-task logit residuals with
    shapes matching each task's primary tower output.

    Training discipline (see ``BRPConfig``): the residual is supervised
    only against ``target - activation(primary.detach())``; this keeps
    the primary's gradient isolated from the residual expert's
    learning signal (single-stage boosting).

    The per-task combining scalar ``sigmoid(λ_t)`` lives on ``PLEModel``
    as a single ``nn.Parameter`` — not here — so that one residual bank
    serves all tasks with a consistent interface.
    """

    def __init__(
        self,
        task_names: List[str],
        input_dim: int,
        task_output_dims: Dict[str, int],
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]
        self.task_names = list(task_names)
        self.input_dim = input_dim
        self.task_output_dims = dict(task_output_dims)

        self.heads = nn.ModuleDict()
        for tn in self.task_names:
            out_dim = int(task_output_dims[tn])
            layers: List[nn.Module] = []
            prev = input_dim
            for h in hidden_dims:
                layers.append(nn.Linear(prev, h))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                prev = h
            layers.append(nn.Linear(prev, out_dim))
            self.heads[tn] = nn.Sequential(*layers)

    def forward(self, task_name: str, x: torch.Tensor) -> torch.Tensor:
        """Return per-task residual logit of shape ``(batch, task_output_dim)``."""
        return self.heads[task_name](x)
