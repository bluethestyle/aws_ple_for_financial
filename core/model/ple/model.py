"""
Progressive Layered Extraction (PLE) Model.

Full PLE model with:
  - Stacked CGC layers (shared + task-specific experts with gating)
  - CGC Attention (per-task expert weighting over shared outputs)
  - Adaptive Task Transfer (adaTT) for inter-task knowledge transfer
  - Configurable loss weighting (GradNorm / DWA / Uncertainty)
  - Logit transfer (source task output feeds into target task tower)
  - Task towers with configurable activation and output dimensions

This implementation is **domain-agnostic**: all task names, dimensions,
expert types, and transfer relationships come from ``PLEConfig``.

Architecture overview::

    features ─┬─ SharedExperts ─── CGCAttention(task_k) ──┐
              │                                            ├─ TaskExpert_k ─ Tower_k ─ prediction_k
              └─ TaskExpert_k(per task) ──────────────────┘
                                                    ↑
                                              logit_transfer (source_task → target_task)

References:
  - Tang et al., "Progressive Layered Extraction (PLE): A Novel Multi-Task
    Learning (MTL) Model for Personalized Recommendations" (RecSys 2020)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import PLEConfig, ExpertInputConfig, ExpertBasketConfig, GroupTaskExpertConfig
from .experts import CGCLayer, CGCAttention, MLPExpert, ExpertRegistry, ExpertBasket
from .feature_router import FeatureRouter
from .adatt import AdaptiveTaskTransfer
from .loss_weighting import (
    BaseLossWeighting,
    UncertaintyWeighting,
    create_loss_weighting,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class PLEInput:
    """Model input container.

    Args:
        features: ``(batch, input_dim)`` static feature tensor (e.g. 644D).
        feature_group_ranges: Optional mapping from feature group name to
            ``(start_col, end_col)`` index range within ``features``.
            When provided together with a configured ``FeatureRouter``,
            enables expert-specific feature routing.  When ``None``, all
            experts receive the full feature tensor (backward compatible).
        expert_routing: Optional runtime override for expert-to-group
            mapping.  ``{expert_name: [group_names]}``.  Typically set by
            the ``FeatureGroupPipeline`` when dynamic routing is needed.
            When ``None``, the static config from ``PLEConfig`` is used.
        cluster_ids: ``(batch,)`` cluster assignment (or zeros if no clustering).
        cluster_probs: ``(batch, n_clusters)`` soft cluster probabilities.
        targets: ``{task_name: label_tensor}`` for training.
        hyperbolic_features: ``(batch, 20)`` embeddings for HGCN expert.
        tda_features: ``(batch, 70)`` precomputed TDA features for PersLay.
        tda_diagrams: ``(batch, max_pairs, 3)`` raw persistence diagrams.
        tda_diagram_mask: ``(batch, max_pairs)`` validity mask for diagrams.
        collaborative_features: ``(batch, 64)`` LightGCN embeddings.
        hmm_journey: ``(batch, 16)`` HMM journey-mode state posteriors.
        hmm_lifecycle: ``(batch, 16)`` HMM lifecycle-mode state posteriors.
        hmm_behavior: ``(batch, 16)`` HMM behavior-mode state posteriors.
        event_sequences: ``(batch, seq_len, feat_dim)`` event-level sequences.
        session_sequences: ``(batch, seq_len, feat_dim)`` session-level sequences.
        event_time_delta: ``(batch, seq_len)`` inter-event time deltas.
        session_time_delta: ``(batch, seq_len)`` inter-session time deltas.
        sequence_lengths: Actual pre-padding lengths per sequence type.
        multidisciplinary_features: ``(batch, 24)`` cross-domain features.
        coldstart_features: ``(batch, 40)`` cold-start indicator features.
        anonymous_features: ``(batch, 15)`` anonymous-user features.
        edge_index: ``(2, num_edges)`` graph edge indices for runtime GNN.
        edge_weight: ``(num_edges,)`` edge weights for runtime GNN.
        sample_weights: ``(batch,)`` per-sample importance weights.
    """
    features: torch.Tensor

    # Existing fields
    feature_group_ranges: Optional[Dict[str, Tuple[int, int]]] = None
    expert_routing: Optional[Dict[str, List[str]]] = None
    cluster_ids: Optional[torch.Tensor] = None
    cluster_probs: Optional[torch.Tensor] = None
    targets: Optional[Dict[str, torch.Tensor]] = None

    # Specialized expert inputs
    hyperbolic_features: Optional[torch.Tensor] = None
    tda_features: Optional[torch.Tensor] = None
    tda_diagrams: Optional[torch.Tensor] = None
    tda_diagram_mask: Optional[torch.Tensor] = None
    collaborative_features: Optional[torch.Tensor] = None

    # HMM triple-mode states
    hmm_journey: Optional[torch.Tensor] = None
    hmm_lifecycle: Optional[torch.Tensor] = None
    hmm_behavior: Optional[torch.Tensor] = None

    # Sequence tensors (for Temporal / Mamba experts)
    event_sequences: Optional[torch.Tensor] = None
    session_sequences: Optional[torch.Tensor] = None
    event_time_delta: Optional[torch.Tensor] = None
    session_time_delta: Optional[torch.Tensor] = None
    sequence_lengths: Optional[Dict[str, torch.Tensor]] = None

    # Auxiliary features
    multidisciplinary_features: Optional[torch.Tensor] = None
    coldstart_features: Optional[torch.Tensor] = None
    anonymous_features: Optional[torch.Tensor] = None

    # Graph structure (for runtime GNN)
    edge_index: Optional[torch.Tensor] = None
    edge_weight: Optional[torch.Tensor] = None

    # Sample metadata
    sample_weights: Optional[torch.Tensor] = None

    # Names of all tensor fields for generic iteration in .to()
    _TENSOR_FIELDS: List[str] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # Pre-compute the list once so .to() doesn't introspect every call
        object.__setattr__(self, "_TENSOR_FIELDS", [
            "features", "cluster_ids", "cluster_probs",
            "hyperbolic_features", "tda_features", "tda_diagrams",
            "tda_diagram_mask", "collaborative_features",
            "hmm_journey", "hmm_lifecycle", "hmm_behavior",
            "event_sequences", "session_sequences",
            "event_time_delta", "session_time_delta",
            "multidisciplinary_features", "coldstart_features",
            "anonymous_features", "edge_index", "edge_weight",
            "sample_weights",
        ])

    @property
    def batch_size(self) -> int:
        return self.features.size(0)

    @property
    def device(self) -> torch.device:
        return self.features.device

    def to(self, device: torch.device) -> "PLEInput":
        """Move all tensors to *device*, leaving non-tensor fields unchanged."""
        kwargs: Dict[str, object] = {}
        for name in self._TENSOR_FIELDS:
            val = getattr(self, name)
            kwargs[name] = val.to(device) if val is not None else None

        # Dict[str, Tensor] fields
        kwargs["targets"] = (
            {k: v.to(device) for k, v in self.targets.items()}
            if self.targets else None
        )
        kwargs["sequence_lengths"] = (
            {k: v.to(device) for k, v in self.sequence_lengths.items()}
            if self.sequence_lengths else None
        )

        # Non-tensor fields pass through
        kwargs["feature_group_ranges"] = self.feature_group_ranges
        kwargs["expert_routing"] = self.expert_routing

        return PLEInput(**kwargs)


@dataclass
class PLEOutput:
    """Model output container.

    Args:
        predictions: ``{task_name: prediction_tensor}``.
        total_loss: Scalar total loss (training only).
        task_losses: ``{task_name: scalar_loss}`` before transfer enhancement.
        transfer_weights: ``(n_tasks, n_tasks)`` adaTT transfer matrix.
        cgc_attention_weights: ``{task_name: (batch, n_experts)}``.
        aux_losses: Additional regularization losses (e.g. CGC entropy).
    """
    predictions: Dict[str, torch.Tensor]
    total_loss: Optional[torch.Tensor] = None
    task_losses: Optional[Dict[str, torch.Tensor]] = None
    transfer_weights: Optional[torch.Tensor] = None
    cgc_attention_weights: Optional[Dict[str, torch.Tensor]] = None
    aux_losses: Optional[Dict[str, float]] = None


# ============================================================================
# Task Tower
# ============================================================================

# ============================================================================
# Tower Registry
# ============================================================================

class TowerRegistry:
    """Plugin registry for task tower types.

    Towers are registered by string key and instantiated via
    ``TowerRegistry.build(key, **kwargs)``.  The default ``"standard"``
    tower is auto-registered at module load time.

    Usage::

        @TowerRegistry.register("contrastive")
        class ContrastiveTower(nn.Module):
            ...

        tower = TowerRegistry.build("contrastive", input_dim=96, output_dim=128)
    """

    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, key: str):
        def decorator(tower_cls):
            cls._registry[key] = tower_cls
            return tower_cls
        return decorator

    @classmethod
    def build(cls, key: str, **kwargs) -> nn.Module:
        if key not in cls._registry:
            raise KeyError(
                f"Unknown tower type '{key}'. "
                f"Registered: {list(cls._registry.keys())}"
            )
        return cls._registry[key](**kwargs)

    @classmethod
    def list_registered(cls) -> List[str]:
        return list(cls._registry.keys())


# ============================================================================
# Standard Task Tower
# ============================================================================

@TowerRegistry.register("standard")
class TaskTower(nn.Module):
    """Per-task output tower (standard MLP).

    Maps the task expert output to the final prediction.

    Args:
        input_dim: Width of the task expert output.
        output_dim: Number of output units (1 for binary/regression,
            N for multiclass).
        hidden_dims: Hidden layer widths.
        activation: Output activation (``"sigmoid"``, ``"softmax"``, or ``None``).
        dropout: Dropout probability.
        task_type: ``"binary"``, ``"multiclass"``, or ``"regression"``.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dims: Optional[List[int]] = None,
        activation: Optional[str] = "sigmoid",
        dropout: float = 0.2,
        task_type: str = "binary",
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        if task_type == "regression" and activation not in (None, "none"):
            raise ValueError(
                f"Regression tasks require activation=None, got '{activation}'"
            )

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hd),
                nn.LayerNorm(hd),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hd

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.network(x)
        if self.activation == "sigmoid":
            return torch.sigmoid(out)
        elif self.activation == "softmax":
            return F.softmax(out, dim=-1)
        return out


# ============================================================================
# Contrastive Tower (for brand_prediction etc.)
# ============================================================================

@TowerRegistry.register("contrastive")
class ContrastiveTower(nn.Module):
    """Contrastive embedding tower for tasks with many classes.

    Instead of softmax over N classes, learns an embedding space where
    similar items are close.  Output is an L2-normalized embedding vector
    used with contrastive loss (e.g. InfoNCE, triplet).

    Args:
        input_dim: Width of the task expert output.
        output_dim: Embedding dimension (e.g. 128 for brand_prediction).
        hidden_dims: Hidden layer widths.
        dropout: Dropout probability.
        task_type: Ignored (always contrastive).
        activation: Ignored (always L2 normalize).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 128,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        task_type: str = "contrastive",
        activation: Optional[str] = None,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 64]

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hd),
                nn.LayerNorm(hd),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hd

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.network(x)
        return F.normalize(emb, p=2, dim=-1)


# ============================================================================
# Main PLE Model
# ============================================================================

class PLEModel(nn.Module):
    """Progressive Layered Extraction model.

    This is the top-level model class that assembles all PLE components.
    Task definitions are fully config-driven -- no hardcoded task names,
    dimensions, or domain knowledge.

    Usage::

        config = PLEConfig(
            input_dim=128,
            task_names=["task_a", "task_b", "task_c"],
            ...
        )
        model = PLEModel(config)

        inputs = PLEInput(features=x, targets={"task_a": y_a, ...})
        output = model(inputs)
        loss = output.total_loss

    Args:
        config: ``PLEConfig`` instance with all hyperparameters.
    """

    def __init__(self, config: PLEConfig):
        super().__init__()
        self.config = config
        self.task_names = list(config.task_names)

        if not self.task_names:
            raise ValueError("PLEConfig.task_names must not be empty")

        # -- Expert Basket (optional) ----------------------------------------
        # When expert_basket is configured, build an ExpertBasket that
        # selects a subset of experts from the pool.  Otherwise, fall back
        # to legacy behaviour where all shared experts use the same type.
        self.expert_basket: Optional[ExpertBasket] = None
        if config.expert_basket is not None:
            # Ensure expert modules are imported (triggers registration)
            import core.model.experts  # noqa: F401

            self.expert_basket = ExpertBasket(
                basket_config=config.expert_basket,
                input_dim=config.input_dim,
                default_output_dim=config.shared_expert.output_dim,
                default_hidden_dims=config.shared_expert.hidden_dims,
                dropout=config.dropout,
            )

        # -- Build components ------------------------------------------------
        self._build_extraction_layers()
        self._build_cgc_attention()
        self._build_task_experts()
        self._build_adatt()
        self._build_logit_transfer()
        self._build_task_towers()
        self._build_loss_weighting()

        # -- Training state --------------------------------------------------
        self.current_epoch = 0
        self.global_step = 0
        self._adatt_grad_interval = config.adatt.grad_interval

        logger.info(f"PLEModel init: {len(self.task_names)} tasks, "
                     f"input_dim={config.input_dim}")
        logger.info(self.summary())

    # ------------------------------------------------------------------
    # Build methods
    # ------------------------------------------------------------------

    def _build_extraction_layers(self) -> None:
        """Build stacked CGC extraction layers.

        When an Expert Basket is configured, the first CGC layer's shared
        experts are replaced with the basket-built experts (heterogeneous
        types from the Expert Pool).  The CGC gating logic is unchanged --
        it simply receives whatever experts the basket provides.

        If ``config.expert_input_routing`` is configured, creates a
        ``FeatureRouter`` and passes it to the first CGC layer so that
        each shared expert receives only its designated feature groups.

        Subsequent stacked layers always receive the full expert output
        dimension (routing only applies to the raw feature input in layer 0).
        """
        cfg = self.config
        self.extraction_layers = nn.ModuleList()

        # Determine number of shared experts and their names
        if self.expert_basket is not None:
            num_shared = self.expert_basket.num_shared_experts
            shared_expert_names = [
                f"shared_{i}" for i in range(num_shared)
            ]
        else:
            num_shared = cfg.num_shared_experts
            shared_expert_names = [
                f"shared_{i}" for i in range(num_shared)
            ]

        # Build FeatureRouter for the first layer if routing is configured.
        self.feature_router: Optional[FeatureRouter] = None

        if cfg.expert_input_routing:
            group_ranges = getattr(cfg, "feature_group_ranges", None)
            if group_ranges:
                self.feature_router = FeatureRouter(
                    expert_names=shared_expert_names,
                    routing_config=cfg.expert_input_routing,
                    group_ranges=group_ranges,
                )
                logger.info(
                    f"FeatureRouter created for {len(cfg.expert_input_routing)} "
                    f"routing rules"
                )

        in_dim = cfg.input_dim
        for layer_idx in range(cfg.num_extraction_layers):
            # Only the first layer gets the feature router
            router = self.feature_router if layer_idx == 0 else None
            expert_names = shared_expert_names if layer_idx == 0 else None

            layer = CGCLayer(
                input_dim=in_dim,
                num_tasks=len(self.task_names),
                num_shared_experts=num_shared,
                num_task_experts=cfg.num_task_experts_per_task,
                expert_hidden_dim=cfg.shared_expert.output_dim,
                dropout=cfg.dropout,
                expert_hidden_dims=cfg.shared_expert.hidden_dims,
                feature_router=router,
                shared_expert_names=expert_names,
            )

            # When Expert Basket is configured, replace the first layer's
            # shared experts with the basket-built heterogeneous experts.
            # Subsequent stacked layers keep the default homogeneous MLP
            # experts (they operate on expert output vectors, not raw features).
            if self.expert_basket is not None and layer_idx == 0:
                basket_experts = self.expert_basket.build_shared_experts()
                layer.shared_experts = basket_experts
                layer.num_shared_experts = len(basket_experts)
                logger.info(
                    "CGC layer 0: replaced shared experts with Expert Basket "
                    "(%d experts: %s)",
                    len(basket_experts),
                    self.expert_basket.shared_expert_names,
                )

            self.extraction_layers.append(layer)
            in_dim = cfg.shared_expert.output_dim  # next layer input = previous output

        self._extraction_output_dim = in_dim

    def _build_cgc_attention(self) -> None:
        """Build per-task CGC attention over shared expert outputs."""
        cfg = self.config
        if not cfg.cgc.enabled:
            self.cgc_attention = None
            return

        # Determine number of shared experts (basket or legacy)
        if self.expert_basket is not None:
            num_shared = self.expert_basket.num_shared_experts
            expert_names = [
                f"shared_{i}" for i in range(num_shared)
            ]
        else:
            num_shared = cfg.num_shared_experts
            expert_names = [f"shared_{i}" for i in range(num_shared)]

        # In stacked PLE, attention operates on the final extraction output
        # Each expert output has the same dimension
        expert_dims = [cfg.shared_expert.output_dim] * num_shared

        domain_map = {
            task_name: cfg.get_domain_experts(task_name)
            for task_name in self.task_names
        }

        self.cgc_attention = CGCAttention(
            task_names=self.task_names,
            expert_dims=expert_dims,
            expert_names=expert_names,
            bias_high=cfg.cgc.bias_high,
            bias_low=cfg.cgc.bias_low,
            dim_normalize=cfg.cgc.dim_normalize,
            domain_experts_map=domain_map,
        )

    def _build_task_experts(self) -> None:
        """Build per-task expert networks.

        When ``GroupTaskExpertConfig.enabled`` is ``True`` and a
        ``task_group_map`` is available, uses the efficient GroupEncoder +
        ClusterEmbedding + TaskHead architecture (original v3.2 design).

        Falls back to legacy per-task MLP experts otherwise.
        """
        cfg = self.config

        if cfg.group_task_expert.enabled and cfg.task_group_map:
            # Use GroupEncoder + ClusterEmbedding + TaskHead architecture
            from .task_experts import GroupTaskExpertBasket

            gte = cfg.group_task_expert
            n_clusters = cfg.cluster.n_clusters

            self.group_task_expert_basket = GroupTaskExpertBasket(
                input_dim=self._extraction_output_dim,
                task_names=self.task_names,
                task_group_map=cfg.task_group_map,
                group_hidden_dim=gte.group_hidden_dim,
                group_output_dim=gte.group_output_dim,
                cluster_embed_dim=gte.cluster_embed_dim if n_clusters > 0 else 0,
                n_clusters=n_clusters,
                dropout=gte.dropout,
            )
            # output_dim = group_output_dim + cluster_embed_dim (e.g. 64+32=96)
            # This goes directly to TaskTower (no TaskHead bottleneck)
            self._task_expert_output_dim = self.group_task_expert_basket.output_dim
        else:
            # Fallback: plain MLP per task (legacy)
            self.group_task_expert_basket = None
            self.task_expert_networks = nn.ModuleDict()
            for task_name in self.task_names:
                self.task_expert_networks[task_name] = MLPExpert(
                    input_dim=self._extraction_output_dim,
                    output_dim=cfg.task_expert_output_dim,
                    hidden_dims=cfg.task_expert.hidden_dims,
                    dropout=cfg.task_expert.dropout,
                    activation=cfg.task_expert.activation,
                    use_layer_norm=cfg.task_expert.use_layer_norm,
                )
            self._task_expert_output_dim = cfg.task_expert_output_dim

    def _build_adatt(self) -> None:
        """Build the Adaptive Task Transfer module."""
        cfg = self.config.adatt
        if not cfg.enabled:
            self.adatt = None
            return

        # Convert TaskGroupDef objects to plain dicts for AdaTT
        task_groups = None
        if cfg.task_groups:
            task_groups = {
                gname: {
                    "members": gdef.members if hasattr(gdef, "members") else gdef.get("members", []),
                    "intra_strength": gdef.intra_strength if hasattr(gdef, "intra_strength") else gdef.get("intra_strength", 0.7),
                }
                for gname, gdef in cfg.task_groups.items()
            }

        self.adatt = AdaptiveTaskTransfer(
            task_names=self.task_names,
            transfer_lambda=cfg.transfer_lambda,
            temperature=cfg.temperature,
            use_group_prior=bool(task_groups),
            warmup_epochs=cfg.warmup_epochs,
            freeze_epoch=cfg.freeze_epoch,
            negative_transfer_threshold=cfg.negative_transfer_threshold,
            task_groups=task_groups,
            inter_group_strength=cfg.inter_group_strength,
            ema_decay=cfg.ema_decay,
            prior_blend_start=cfg.prior_blend_start,
            prior_blend_end=cfg.prior_blend_end,
            max_transfer_ratio=cfg.max_transfer_ratio,
        )

    def _build_logit_transfer(self) -> None:
        """Build logit transfer projections (source task -> target task)."""
        cfg = self.config
        self.logit_transfer_sources: Dict[str, str] = {}
        self.logit_transfer_proj = nn.ModuleDict()
        self.logit_transfer_strength = cfg.logit_transfer_strength

        for lt in cfg.logit_transfers:
            if not lt.enabled:
                continue
            src, tgt = lt.source, lt.target
            if src in self.task_names and tgt in self.task_names:
                self.logit_transfer_sources[tgt] = src
                src_output_dim = cfg.get_task_output_dim(src)
                self.logit_transfer_proj[tgt] = nn.Sequential(
                    nn.Linear(src_output_dim, self._task_expert_output_dim),
                    nn.LayerNorm(self._task_expert_output_dim),
                    nn.SiLU(),
                )

        if self.logit_transfer_sources:
            logger.info(f"Logit transfer: {self.logit_transfer_sources} "
                        f"(strength={self.logit_transfer_strength})")

    def _build_task_towers(self) -> None:
        """Build per-task output towers.

        Each task can override tower_type and tower_dims via
        ``task_overrides``.  When no override is set, falls back to the
        global ``TaskTowerConfig`` defaults.
        """
        cfg = self.config
        self.task_towers = nn.ModuleDict()

        # Use _task_expert_output_dim set by _build_task_experts()
        tower_input_dim = self._task_expert_output_dim

        for task_name in self.task_names:
            tower_type = cfg.get_tower_type(task_name)
            tower_dims = cfg.get_tower_dims(task_name) or cfg.task_tower.hidden_dims

            self.task_towers[task_name] = TowerRegistry.build(
                tower_type,
                input_dim=tower_input_dim,
                output_dim=cfg.get_task_output_dim(task_name),
                hidden_dims=tower_dims,
                activation=cfg.get_task_activation(task_name),
                dropout=cfg.task_tower.dropout,
                task_type=cfg.get_task_type(task_name),
            )

    def _build_loss_weighting(self) -> None:
        """Build loss weighting strategy."""
        cfg = self.config.loss_weighting
        self.loss_weighting = create_loss_weighting(
            strategy=cfg.strategy,
            num_tasks=len(self.task_names),
            task_names=self.task_names,
            config={
                "alpha": cfg.gradnorm_alpha,
                "grad_interval": cfg.gradnorm_interval,
                "temperature": cfg.dwa_temperature,
                "window_size": cfg.dwa_window_size,
            },
        )

    # ------------------------------------------------------------------
    # Task execution order (topological sort)
    # ------------------------------------------------------------------

    def _get_task_execution_order(self) -> List[str]:
        """Derive task execution order from logit transfer dependencies.

        Uses Kahn's algorithm for topological sort.  Tasks without
        dependencies are sorted alphabetically.
        """
        if not self.logit_transfer_sources:
            return sorted(self.task_names)

        # Build dependency graph: target -> {sources}
        deps: Dict[str, set] = defaultdict(set)
        for target, source in self.logit_transfer_sources.items():
            deps[target].add(source)

        # In-degree
        in_degree = {t: 0 for t in self.task_names}
        for target, sources in deps.items():
            in_degree[target] = len([s for s in sources if s in self.task_names])

        # Kahn's algorithm
        queue = sorted([t for t in self.task_names if in_degree[t] == 0])
        result: List[str] = []

        while queue:
            queue.sort()
            task = queue.pop(0)
            result.append(task)
            for target, sources in deps.items():
                if task in sources and target in in_degree:
                    in_degree[target] -= 1
                    if in_degree[target] == 0 and target not in result:
                        queue.append(target)

        # Add remaining tasks (handles cycles gracefully)
        remaining = sorted(t for t in self.task_names if t not in result)
        result.extend(remaining)
        return result

    # ------------------------------------------------------------------
    # Expert input preparation
    # ------------------------------------------------------------------

    # Mapping from PLEInput field names to expert type identifiers.
    # When a PLEInput has a dedicated tensor field, it is routed to the
    # matching expert instead of slicing from the main features tensor.
    _EXPERT_FIELD_MAP: Dict[str, str] = {
        "hgcn": "hyperbolic_features",
        "perslay": "tda_features",
        "lightgcn": "collaborative_features",
        "temporal": "event_sequences",
        "mamba": "event_sequences",
    }

    def _prepare_expert_inputs(
        self,
        ple_input: PLEInput,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Map PLEInput fields to expert-specific inputs.

        Priority:
            1. Dedicated tensor (e.g., PLEInput.hyperbolic_features for hgcn)
            2. Feature router slice (when feature_group_ranges provided)
            3. Full features tensor (fallback -- all experts get everything)

        When no specialized inputs are available (no dedicated tensors and
        no feature_group_ranges), returns ``None`` to signal that the
        legacy code path (all experts receive full features) should be used.

        Args:
            ple_input: Model input container.

        Returns:
            Dict mapping expert name to its input tensor, or ``None`` if
            no routing is needed (backward compatible fallback).
        """
        features = ple_input.features
        has_dedicated = False
        has_routing = False
        expert_inputs: Dict[str, torch.Tensor] = {}

        # Determine which expert names we need to prepare inputs for.
        # These come from the first CGC layer's shared expert names.
        if self.expert_basket is not None:
            expert_names = self.expert_basket.shared_expert_names
        else:
            expert_names = [
                f"shared_{i}" for i in range(self.config.num_shared_experts)
            ]

        # Step 1: Check for dedicated tensor fields
        for expert_key, field_name in self._EXPERT_FIELD_MAP.items():
            tensor = getattr(ple_input, field_name, None)
            if tensor is not None:
                # Find which shared expert(s) match this key.
                # Expert basket names use the expert type directly (e.g. "hgcn").
                # Legacy shared expert names are "shared_0", "shared_1", etc.
                for ename in expert_names:
                    if expert_key in ename.lower() or ename.lower() == expert_key:
                        expert_inputs[ename] = tensor
                        has_dedicated = True
                        logger.debug(
                            "Expert routing: %s receiving %s (%dD)",
                            ename, field_name,
                            tensor.shape[-1] if tensor.dim() >= 2 else tensor.numel(),
                        )

        # Step 2: Use FeatureRouter for remaining experts
        if self.feature_router is not None:
            # Use runtime group_ranges if provided, else fall back to config
            runtime_ranges = ple_input.feature_group_ranges
            if runtime_ranges is not None:
                # Create a temporary router with runtime ranges if they differ
                # from the config-time ranges.  In practice, they usually match.
                pass  # The existing self.feature_router handles this

            for ename in expert_names:
                if ename not in expert_inputs:
                    if self.feature_router.has_routing(ename):
                        expert_inputs[ename] = self.feature_router.route(
                            features, ename,
                        )
                        has_routing = True
                    else:
                        expert_inputs[ename] = features

        if not has_dedicated and not has_routing:
            return None

        # Step 3: Fill in any remaining experts with full features
        for ename in expert_names:
            if ename not in expert_inputs:
                expert_inputs[ename] = features

        # Log routing summary (once, at debug level)
        if logger.isEnabledFor(logging.DEBUG):
            for ename, tensor in expert_inputs.items():
                dim_str = "x".join(str(s) for s in tensor.shape[1:])
                logger.debug("Expert input: %s -> %sD", ename, dim_str)

        return expert_inputs

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        inputs: PLEInput,
        compute_loss: bool = True,
    ) -> PLEOutput:
        """Full forward pass.

        Args:
            inputs: ``PLEInput`` container.
            compute_loss: Whether to compute and return losses.

        Returns:
            ``PLEOutput`` with predictions and optional losses.
        """
        features = inputs.features

        # Prepare expert-specific inputs (returns None when no routing needed)
        expert_inputs = self._prepare_expert_inputs(inputs)

        # Log routing decisions at INFO level on first forward pass
        if not hasattr(self, "_routing_logged") and expert_inputs is not None:
            self._routing_logged = True
            for ename, tensor in expert_inputs.items():
                dim_str = "x".join(str(s) for s in tensor.shape[1:])
                logger.info(
                    "Expert routing: %s receiving %sD input", ename, dim_str,
                )

        # 1. Stacked CGC extraction layers
        task_representations = [features] * len(self.task_names)
        shared_input = features
        shared_concat = None

        for layer_idx, layer in enumerate(self.extraction_layers):
            if layer_idx == 0 and expert_inputs is not None:
                # For the first layer, pass expert-specific inputs via
                # the feature_router mechanism.  The CGCLayer already
                # supports FeatureRouter -- we just need to ensure the
                # shared_input contains the full features tensor so the
                # router can slice from it.  For dedicated tensors that
                # bypass the router, we inject them by temporarily
                # replacing the shared experts' forward methods.
                #
                # However, the cleaner approach is: when expert_inputs
                # are available and a FeatureRouter is configured, the
                # CGCLayer already handles routing.  When dedicated
                # tensors are provided (not from the main features tensor),
                # we need to handle them specially.
                #
                # Strategy: For dedicated tensors (from PLEInput fields),
                # we override the expert's input by wrapping the layer call.
                task_representations, shared_concat = self._forward_cgc_with_routing(
                    layer, shared_input, task_representations, expert_inputs,
                )
            else:
                task_representations, shared_concat = layer(
                    shared_input, task_representations,
                )
            shared_input = shared_concat

        # 2. Per-task expert processing (GroupEncoder + ClusterEmbedding + TaskHead)
        task_expert_outputs: Dict[str, torch.Tensor] = {}
        if self.group_task_expert_basket is not None:
            cluster_ids = inputs.cluster_ids
            cluster_probs = inputs.cluster_probs
            for i, task_name in enumerate(self.task_names):
                task_repr = task_representations[i]
                task_expert_outputs[task_name] = self.group_task_expert_basket(
                    task_repr, task_name,
                    cluster_ids=cluster_ids,
                    cluster_probs=cluster_probs,
                )
        else:
            for i, task_name in enumerate(self.task_names):
                task_repr = task_representations[i]
                task_expert_outputs[task_name] = self.task_expert_networks[task_name](task_repr)

        # 3. Task towers with logit transfer (in dependency order)
        execution_order = self._get_task_execution_order()
        predictions: Dict[str, torch.Tensor] = {}

        for task_name in execution_order:
            tower_input = task_expert_outputs[task_name]

            # Logit transfer from source task
            if task_name in self.logit_transfer_sources:
                source = self.logit_transfer_sources[task_name]
                if source in predictions:
                    src_out = predictions[source]
                    if src_out.dim() == 1:
                        src_out = src_out.unsqueeze(-1)
                    proj = self.logit_transfer_proj[task_name](src_out)
                    tower_input = tower_input + self.logit_transfer_strength * proj

            predictions[task_name] = self.task_towers[task_name](tower_input)

        # 4. Loss computation
        total_loss = None
        task_losses = None
        transfer_weights = None
        aux_losses: Dict[str, float] = {}

        if compute_loss and inputs.targets is not None:
            task_losses = self._compute_task_losses(predictions, inputs.targets)

            # Gradient extraction for adaTT (at configured interval)
            task_gradients = None
            if self.training and self.adatt is not None:
                if self.global_step % self._adatt_grad_interval == 0:
                    task_gradients = self._extract_task_gradients(task_losses)

            # Apply adaTT transfer enhancement
            if self.adatt is not None:
                enhanced_losses = self.adatt.compute_transfer_loss(
                    task_losses,
                    task_gradients=task_gradients,
                )
                total_loss = sum(enhanced_losses.values())
                transfer_weights = self.adatt.get_transfer_matrix()
            elif self.loss_weighting is not None:
                # Apply loss weighting strategy
                if isinstance(self.loss_weighting, UncertaintyWeighting):
                    total_loss = self.loss_weighting.weighted_loss(task_losses)
                else:
                    weights = self.loss_weighting.compute_weights()
                    total_loss = sum(
                        weights.get(name, 1.0) * loss
                        for name, loss in task_losses.items()
                    )
            else:
                total_loss = sum(task_losses.values())

            # CGC entropy regularization
            if (self.training
                    and self.cgc_attention is not None
                    and shared_concat is not None
                    and self.config.cgc.entropy_lambda > 0):
                ent_loss = self.cgc_attention.entropy_regularization(shared_concat)
                ent_lambda = self.config.cgc.entropy_lambda
                total_loss = total_loss + ent_lambda * ent_loss
                aux_losses["cgc_entropy"] = (ent_lambda * ent_loss).item()

            # CausalExpert DAG regularization (NOTEARS acyclicity + sparsity)
            if self.training:
                for expert in self._iter_shared_experts():
                    if hasattr(expert, "get_dag_regularization"):
                        dag_reg = expert.get_dag_regularization()
                        total_loss = total_loss + dag_reg
                        aux_losses["dag_regularization"] = dag_reg.item()

            # TemporalEnsemble gate entropy monitoring (logging only, no reg)
            if self.training:
                for expert in self._iter_shared_experts():
                    if hasattr(expert, "compute_gate_entropy"):
                        ent = expert.compute_gate_entropy(
                            task_representations[0].unsqueeze(1),
                        )
                        if ent is not None:
                            aux_losses["temporal_gate_entropy"] = ent.item()

        # CGC attention weights for monitoring (inference only)
        cgc_weights = None
        if not self.training and self.cgc_attention is not None and shared_concat is not None:
            cgc_weights = self.cgc_attention.get_attention_weights(shared_concat)

        return PLEOutput(
            predictions=predictions,
            total_loss=total_loss,
            task_losses=task_losses,
            transfer_weights=transfer_weights,
            cgc_attention_weights=cgc_weights,
            aux_losses=aux_losses if aux_losses else None,
        )

    # ------------------------------------------------------------------
    # CGC routing helper
    # ------------------------------------------------------------------

    def _forward_cgc_with_routing(
        self,
        layer: CGCLayer,
        shared_input: torch.Tensor,
        task_inputs: List[torch.Tensor],
        expert_inputs: Dict[str, torch.Tensor],
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Forward through a CGC layer with expert-specific input routing.

        This method replicates ``CGCLayer.forward()`` logic but substitutes
        expert-specific inputs from ``expert_inputs`` when available.  This
        handles the case where dedicated PLEInput tensors (e.g.,
        ``hyperbolic_features``) need to bypass the FeatureRouter and be
        fed directly to their matching expert.

        When a FeatureRouter is configured on the layer, it handles
        standard group-based slicing.  This method adds support for
        dedicated tensor override on top of that.

        Expert forward() signatures are NOT modified -- routing happens
        BEFORE passing to experts.

        Args:
            layer: The CGC layer to forward through.
            shared_input: Full features tensor ``(batch, input_dim)``.
            task_inputs: Per-task input tensors.
            expert_inputs: ``{expert_name: tensor}`` from
                ``_prepare_expert_inputs()``.

        Returns:
            Same as ``CGCLayer.forward()``: list of per-task gated outputs
            and concatenated shared expert outputs.
        """
        # Shared expert outputs: use expert_inputs dict for overrides
        shared_expert_outputs: List[torch.Tensor] = []
        for i, expert in enumerate(layer.shared_experts):
            expert_name = layer.shared_expert_names[i]
            if expert_name in expert_inputs:
                expert_input = expert_inputs[expert_name]
            elif (layer.feature_router is not None
                    and expert_name in layer.feature_router.expert_names):
                expert_input = layer.feature_router.route(
                    shared_input, expert_name,
                )
            else:
                expert_input = shared_input

            # Handle 3D sequence tensors: if the expert input is 3D
            # (batch, seq_len, feat_dim) but the expert expects 2D,
            # reshape to (batch, seq_len * feat_dim) for compatibility.
            # The expert's forward() expects (batch, input_dim).
            if expert_input.dim() == 3 and expert.input_dim != expert_input.shape[1]:
                batch_size = expert_input.size(0)
                expert_input = expert_input.reshape(batch_size, -1)

            shared_expert_outputs.append(expert(expert_input))

        shared_outs = torch.stack(shared_expert_outputs, dim=1)
        shared_concat = torch.cat(shared_expert_outputs, dim=-1)

        # Per-task expert outputs (always receive full input -- same as
        # CGCLayer.forward())
        outputs: List[torch.Tensor] = []
        for task_idx in range(layer.num_tasks):
            task_outs = torch.stack(
                [expert(task_inputs[task_idx])
                 for expert in layer.task_experts[task_idx]],
                dim=1,
            )
            all_outs = torch.cat([task_outs, shared_outs], dim=1)

            gate_logits = layer.gating[task_idx](task_inputs[task_idx])
            gate_weights = F.softmax(gate_logits, dim=-1)

            gated = (gate_weights.unsqueeze(-1) * all_outs).sum(dim=1)
            outputs.append(gated)

        return outputs, shared_concat

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------

    def _compute_task_losses(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute per-task losses using task-type-appropriate loss functions.

        Override this method to plug in custom loss functions.
        """
        losses: Dict[str, torch.Tensor] = {}

        for task_name in self.task_names:
            if task_name not in targets:
                continue

            pred = predictions[task_name]
            target = targets[task_name]
            task_type = self.config.get_task_type(task_name)

            if task_type == "binary":
                loss = F.binary_cross_entropy(
                    pred.squeeze(-1), target.float(),
                )
            elif task_type == "multiclass":
                loss = F.cross_entropy(pred, target.long())
            elif task_type == "regression":
                loss = F.huber_loss(pred.squeeze(-1), target.float())
            else:
                # Fallback: MSE
                loss = F.mse_loss(pred.squeeze(-1), target.float())

            losses[task_name] = loss

        return losses

    def _iter_shared_experts(self):
        """Iterate over shared expert modules from extraction layers.

        Yields individual expert nn.Module instances from CGC shared_experts
        or from the ExpertBasket's built shared experts.
        """
        for layer in self.extraction_layers:
            if hasattr(layer, "shared_experts"):
                yield from layer.shared_experts

    def _extract_task_gradients(
        self,
        task_losses: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Extract flattened gradients for adaTT affinity computation.

        Uses the shared extraction layer parameters as the reference.
        Gradients are zero-padded for unused parameters so that all
        tasks produce the same-length gradient vector.
        """
        shared_params = list(self.extraction_layers.parameters())
        if not shared_params:
            return {}

        gradients: Dict[str, torch.Tensor] = {}
        for task_name, loss in task_losses.items():
            grads = torch.autograd.grad(
                loss,
                shared_params,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
            # Zero-pad for parameters that received no gradient
            flat_parts = []
            for param, g in zip(shared_params, grads):
                if g is not None:
                    flat_parts.append(g.flatten())
                else:
                    flat_parts.append(torch.zeros(param.numel(), device=param.device))
            gradients[task_name] = torch.cat(flat_parts)

        return gradients

    # ------------------------------------------------------------------
    # Epoch / step lifecycle
    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        """Update epoch state for adaTT and loss weighting."""
        self.current_epoch = epoch
        if self.adatt is not None:
            self.adatt.on_epoch_end(epoch)

    def set_global_step(self, step: int) -> None:
        """Update global step counter (called by Trainer)."""
        self.global_step = step

    def update_loss_weights(
        self,
        losses: Dict[str, torch.Tensor],
        epoch: int = 0,
    ) -> None:
        """Update loss weighting strategy (call after each batch/epoch)."""
        if self.loss_weighting is not None:
            shared_params = list(self.extraction_layers.parameters())
            self.loss_weighting.update(
                losses,
                shared_params=shared_params,
                epoch=epoch,
            )

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Human-readable model summary."""
        lines = [
            "PLEModel:",
            f"  Tasks: {self.task_names}",
            f"  Input dim: {self.config.input_dim}",
            f"  Extraction layers: {self.config.num_extraction_layers}",
        ]

        # Expert basket summary
        if self.expert_basket is not None:
            basket = self.expert_basket.list_basket()
            pool = self.expert_basket.list_pool()
            lines.append(
                f"  Expert Pool: {len(pool)} registered {pool}"
            )
            lines.append(
                f"  Expert Basket: {len(basket['shared'])} shared {basket['shared']} "
                f"+ {len(basket['task'])} task {basket['task']}"
            )
        else:
            lines.append(f"  Shared experts: {self.config.num_shared_experts} (legacy mode)")

        # Task expert architecture summary
        if self.group_task_expert_basket is not None:
            n_groups = len(self.group_task_expert_basket.group_encoders)
            out_dim = self.group_task_expert_basket.output_dim
            lines.append(
                f"  Task experts: GroupTaskExpertBasket "
                f"({n_groups} groups, output_dim={out_dim} → TaskTower directly)"
            )
        else:
            lines.append(f"  Task experts: legacy MLP per task")

        lines.extend([
            f"  Task experts per task: {self.config.num_task_experts_per_task}",
            f"  Expert hidden dim: {self.config.shared_expert.output_dim}",
            f"  Task expert output dim: {self._task_expert_output_dim}",
            f"  adaTT: {'enabled' if self.adatt is not None else 'disabled'}",
            f"  CGC attention: {'enabled' if self.cgc_attention is not None else 'disabled'}",
            f"  Logit transfers: {self.logit_transfer_sources or 'none'}",
            f"  Loss weighting: {self.config.loss_weighting.strategy}",
        ])

        # Feature routing summary
        if self.feature_router is not None:
            lines.append(f"  Feature routing: enabled ({len(self.config.expert_input_routing)} rules)")
            for name in self.feature_router.expert_names:
                dim = self.feature_router.get_expert_input_dim(name)
                routed = "routed" if self.feature_router.has_routing(name) else "full"
                lines.append(f"    {name}: dim={dim} ({routed})")
        else:
            lines.append("  Feature routing: disabled (all experts receive full input)")

        n_params = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lines.append(f"  Parameters: {n_params:,} total, {n_trainable:,} trainable")
        return "\n".join(lines)

    def get_loss_weights(self) -> Dict[str, float]:
        """Return current loss weights (for logging)."""
        if self.loss_weighting is not None:
            return self.loss_weighting.compute_weights()
        return {name: 1.0 for name in self.task_names}
