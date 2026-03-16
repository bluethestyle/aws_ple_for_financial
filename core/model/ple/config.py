"""
PLE Model Configuration.

All hyperparameters for the Progressive Layered Extraction model are defined
here as a single dataclass.  YAML config maps directly to these fields via
``PLEConfig(**yaml_block)``.

Design principles:
  - No hardcoded task names or domain references.
  - Dimensions are explicitly passed, never derived from magic numbers.
  - Optional sub-configs (adaTT, CGC, loss weighting) are nested dataclasses
    so that ``None`` cleanly disables the feature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class ExpertConfig:
    """Configuration for a single expert type."""
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    output_dim: int = 64
    dropout: float = 0.1
    activation: str = "relu"
    use_layer_norm: bool = True
    use_batch_norm: bool = False


@dataclass
class CGCConfig:
    """Customized Gate Control configuration.

    CGC applies per-task attention weights over the shared expert outputs so
    that each task can learn which experts are most relevant.
    """
    enabled: bool = True
    bias_high: float = 1.0
    bias_low: float = -1.0
    dim_normalize: bool = False
    entropy_lambda: float = 0.01


@dataclass
class TaskGroupDef:
    """A single task group for adaTT."""
    members: List[str] = field(default_factory=list)
    intra_strength: float = 0.7


@dataclass
class AdaTTConfig:
    """Adaptive Task Transfer configuration.

    Controls the gradient-based affinity measurement and inter-task knowledge
    transfer mechanism.
    """
    enabled: bool = True
    transfer_lambda: float = 0.1
    temperature: float = 1.0
    warmup_epochs: int = 10
    freeze_epoch: Optional[int] = None
    negative_transfer_threshold: float = -0.1
    ema_decay: float = 0.9
    prior_blend_start: float = 0.5
    prior_blend_end: float = 0.1
    inter_group_strength: float = 0.3
    task_groups: Dict[str, TaskGroupDef] = field(default_factory=dict)
    grad_interval: int = 10
    max_transfer_ratio: float = 0.5


@dataclass
class LossWeightingConfig:
    """Loss weighting strategy configuration."""
    strategy: str = "fixed"               # "fixed" | "uncertainty" | "gradnorm" | "dwa"
    # GradNorm
    gradnorm_alpha: float = 1.5
    gradnorm_interval: int = 1
    # DWA
    dwa_temperature: float = 2.0
    dwa_window_size: int = 5


@dataclass
class TaskTowerConfig:
    """Per-task output tower configuration."""
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    dropout: float = 0.2


@dataclass
class ClusterConfig:
    """Cluster-specific sub-head configuration.

    Set ``n_clusters=0`` to disable clustering entirely.
    """
    n_clusters: int = 0
    cluster_embed_dim: int = 32
    subhead_hidden_dim: int = 64
    subhead_output_dim: int = 32


@dataclass
class ExpertInputConfig:
    """Defines which feature groups a specific expert receives.

    When used in ``PLEConfig.expert_input_routing``, this controls which
    subset of the concatenated feature tensor is routed to each expert.
    Experts not listed in the routing config (or with an empty
    ``input_groups`` list) receive **all** features -- preserving backward
    compatibility.

    Args:
        expert_name: Identifier matching the expert name in the CGC layer
            (e.g. ``"shared_0"``, ``"shared_1"``).
        input_groups: List of feature group names that this expert should
            receive (e.g. ``["base_profile", "tda_topology"]``).
            If empty, the expert receives ALL features.
    """
    expert_name: str = ""
    input_groups: List[str] = field(default_factory=list)


@dataclass
class LogitTransferDef:
    """A single source->target logit transfer relationship."""
    source: str = ""
    target: str = ""
    enabled: bool = True


# ---------------------------------------------------------------------------
# Main config
# ---------------------------------------------------------------------------

@dataclass
class PLEConfig:
    """
    Progressive Layered Extraction model configuration.

    Covers the full PLE + CGC + adaTT + cluster sub-heads architecture.
    All dimensions and task definitions come from external config -- nothing
    is hardcoded to a specific domain.

    Usage::

        import yaml
        with open("model_config.yaml") as f:
            raw = yaml.safe_load(f)
        config = PLEConfig(**raw["model"])
    """

    # -- Global dimensions ---------------------------------------------------
    input_dim: int = 128                          # feature vector width
    task_expert_output_dim: int = 32              # output dim of each task expert

    # -- Task definitions (list of task name strings) ------------------------
    task_names: List[str] = field(default_factory=list)

    # -- Shared experts ------------------------------------------------------
    num_shared_experts: int = 4
    shared_expert: ExpertConfig = field(default_factory=ExpertConfig)

    # -- Task experts --------------------------------------------------------
    num_task_experts_per_task: int = 1
    task_expert: ExpertConfig = field(default_factory=lambda: ExpertConfig(
        hidden_dims=[128, 64], output_dim=32,
    ))

    # -- PLE stacking --------------------------------------------------------
    num_extraction_layers: int = 1                # number of stacked CGC layers

    # -- CGC (Customized Gate Control) --------------------------------------
    cgc: CGCConfig = field(default_factory=CGCConfig)

    # -- Cluster sub-heads ---------------------------------------------------
    cluster: ClusterConfig = field(default_factory=ClusterConfig)

    # -- adaTT (Adaptive Task Transfer) ------------------------------------
    adatt: AdaTTConfig = field(default_factory=AdaTTConfig)

    # -- Logit transfer relationships ----------------------------------------
    logit_transfers: List[LogitTransferDef] = field(default_factory=list)
    logit_transfer_strength: float = 0.5

    # -- Task tower ----------------------------------------------------------
    task_tower: TaskTowerConfig = field(default_factory=TaskTowerConfig)

    # -- Loss weighting ------------------------------------------------------
    loss_weighting: LossWeightingConfig = field(default_factory=LossWeightingConfig)

    # -- Training knobs ------------------------------------------------------
    dropout: float = 0.1

    # -- Expert input routing ------------------------------------------------
    # Maps each expert to the feature groups it receives.
    # If empty, all experts receive all features (backward compatible).
    expert_input_routing: List[ExpertInputConfig] = field(default_factory=list)

    # Feature group ranges: {group_name: (start_col, end_col)} in the
    # concatenated feature tensor.  Set by the FeatureGroupPipeline before
    # model construction.  Required if expert_input_routing is non-empty.
    feature_group_ranges: Optional[Dict[str, tuple]] = None

    # -- Per-task overrides --------------------------------------------------
    # Maps task_name -> {output_dim, activation, task_type, domain_experts, ...}
    task_overrides: Dict[str, dict] = field(default_factory=dict)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def get_task_output_dim(self, task_name: str) -> int:
        """Return the output dimension for a specific task tower."""
        override = self.task_overrides.get(task_name, {})
        return override.get("output_dim", 1)

    def get_task_activation(self, task_name: str) -> Optional[str]:
        """Return the activation function name for a task tower (or None)."""
        override = self.task_overrides.get(task_name, {})
        act = override.get("activation", "sigmoid")
        return None if act in (None, "none", "null") else act

    def get_task_type(self, task_name: str) -> str:
        """Return the task type string for a specific task."""
        override = self.task_overrides.get(task_name, {})
        return override.get("task_type", "binary")

    def get_domain_experts(self, task_name: str) -> List[str]:
        """Return the list of domain-relevant expert names for a task."""
        override = self.task_overrides.get(task_name, {})
        return override.get("domain_experts", [])
