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
    dim_normalize: bool = True
    entropy_lambda: float = 0.01


@dataclass
class AdaTTSPConfig:
    """AdaTT-sp fusion configuration (Li et al., KDD 2023).

    When enabled, the per-task fusion in CGCLayer adds the task's own
    task-specific experts' mean output as a learnable-scalar residual on
    top of the standard gated weighted sum. This implements the core
    mechanism of AdaTT-sp (the ``sp`` / specific-fusion variant of the
    AdaTT paper) without the optional AllExpertGF second-stage fusion.

    Unlike the prior ``AdaTTConfig`` (which controls the TAG + GradNorm
    inspired *loss-level* transfer retained under the ``adaTT`` label for
    historical continuity), AdaTTSPConfig controls *representation-level*
    fusion at the CGC gate layer.
    """
    enabled: bool = False
    native_residual_weight_init: float = 1.0


@dataclass
class BRPConfig:
    """Boosting-Residual Path (Paper 3, MV).

    Motivation: the four preceding recovery mechanisms (loss-level adaTT,
    AdaTT-sp, residual_complement, ECEB MV) all inject a residual
    *additively into the primary representation* and empirically degrade
    AUC in monotone proportion to the invasiveness of the intervention.
    BRP is the one remaining design direction in Paper 3's scoping that
    does not share this structure: it places a separate residual expert
    bank that predicts *in output space* (per-task logit residual) and
    combines only at the final prediction step, leaving the primary
    representation untouched.

    Training (MV): primary task tower is trained against ground-truth
    targets as in baseline PLE. The residual expert for task *t* is
    trained on ``e_t = y_t - activation(primary_t.detach())``, i.e. it
    fits the primary's prediction error with the primary's gradient cut
    off — the single-stage boosting discipline. Inference and evaluation
    use the combined prediction ``primary + sigmoid(λ_t) * residual`` so
    eval metrics reflect the ensemble.

    Mutually exclusive with adatt_sp, residual_recovery, and eceb at the
    PLE model level (a single fusion-augmentation family is active at
    once). Off by default.
    """
    enabled: bool = False
    residual_hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    residual_weight_init: float = -2.0   # sigmoid(-2) ≈ 0.12: residual starts suppressed
    residual_loss_weight: float = 0.1    # weight of residual-error loss in total
    dropout: float = 0.1


@dataclass
class ECEBConfig:
    """Error-Conditioned Expert Bank — uncertainty-gated recovery (Paper 3, MV).

    Rationale: the three preceding recovery mechanisms (loss-level adaTT,
    AdaTT-sp, residual_complement) all inject a *gate-derived* residual at
    the primary's fusion point and were empirically null-to-negative. ECEB
    replaces the gate-inverse signal with an *uncertainty-conditioned*
    residual: when the CGC gate is confused (high entropy), recovery is
    activated; when the gate is confident (low entropy), recovery is
    suppressed. The uncertainty signal is the gate's own entropy, measured
    per-sample per-task — a direct structural property, not a post-hoc
    estimate.

    MV-ECEB (minimum-viable) implementation:
      - Recovery path = task-agnostic consensus (mean over all expert
        outputs), placed in parallel with the gated primary.
      - Recovery weight = sigmoid(learnable_gate) * normalised_entropy,
        where normalised_entropy = H(gate) / log(num_total_experts).
      - Output = gated + recovery_weight * consensus.

    Mutually exclusive with adatt_sp and residual_recovery at the CGC gate
    layer. Off by default.
    """
    enabled: bool = False
    uncertainty_source: str = "gate_entropy"  # gate_entropy (only option in MV)
    recovery_source: str = "uniform"          # uniform (mean over experts)
    weight_init: float = 0.0                  # sigmoid(0)=0.5 → balanced start


@dataclass
class ResidualRecoveryConfig:
    """Intra-task residual expert recovery (Paper 3).

    Motivation: PLE's per-task CGC gate selects primary experts and
    down-weights others. Cross-task fusion mechanisms (loss-level adaTT,
    Li 2023 AdaTT-sp) do not recover the signal that the gate attenuated
    — they mix losses / other tasks' experts instead. This config controls
    *intra-task* residual recovery mechanisms that reclaim signal from the
    unselected experts *within the same task's forward pass*, without any
    cross-task mixing.

    Methods:
      - ``complement``: primary = gated sum; residual = weighted sum with
        (1 - gate) weights renormalised over the expert axis; output =
        primary + α * residual (α learnable scalar).
      - ``orthogonal``: project each expert's output onto the complement
        of the primary direction before aggregating as residual (Paper 3
        future extension).
      - ``dualgate``: learn a second per-task gate explicitly for residual
        weighting (Paper 3 future extension).

    Mutually exclusive with AdaTT-sp at the CGC gate layer.
    """
    enabled: bool = False
    method: str = "complement"  # complement | orthogonal | dualgate
    weight_init: float = 0.5


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

    @classmethod
    def from_pipeline_groups(
        cls,
        pipeline_groups: list,
        **kwargs,
    ) -> "AdaTTConfig":
        """Create an AdaTTConfig from pipeline-level TaskGroupConfig list.

        Args:
            pipeline_groups: List of
                :class:`~core.pipeline.config.TaskGroupConfig` instances.
            **kwargs: Additional AdaTTConfig fields to override.

        Returns:
            A new ``AdaTTConfig`` with ``task_groups`` and
            ``inter_group_strength`` derived from *pipeline_groups*.
        """
        if not pipeline_groups:
            return cls(**kwargs)

        task_groups: Dict[str, TaskGroupDef] = {}
        # Use the first group's inter strength as the default; all groups
        # share the same inter_group_strength at the AdaTT level.
        inter_strength = pipeline_groups[0].adatt_inter_strength

        for pg in pipeline_groups:
            task_groups[pg.name] = TaskGroupDef(
                members=list(pg.tasks),
                intra_strength=pg.adatt_intra_strength,
            )

        return cls(
            task_groups=task_groups,
            inter_group_strength=inter_strength,
            **kwargs,
        )


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
    """A single source->target logit transfer relationship.

    Args:
        source: Source task name.
        target: Target task name.
        enabled: Whether this transfer is active.
        transfer_method: One of ``"residual"`` (default -- add projected
            source output as residual), ``"output_concat"`` (concatenate
            source task output to target tower input and re-project), or
            ``"hidden_concat"`` (concatenate source task's pre-tower hidden
            representation to target tower input and re-project).
    """
    source: str = ""
    target: str = ""
    enabled: bool = True
    transfer_method: str = "residual"


@dataclass
class ExpertBasketConfig:
    """Expert Basket configuration -- selects a subset from the Expert Pool.

    The Expert Basket sits between the full Expert Pool (all experts
    registered via ``@ExpertRegistry.register``) and the CGC runtime
    gating.  It defines *which* experts are included in a specific
    pipeline and with what configuration overrides.

    .. note:: Per-task expert routing

       For per-task expert routing, prefer :class:`GroupTaskExpertConfig`
       which implements the efficient GroupEncoder + ClusterEmbedding +
       TaskHead architecture.  The ``group_task_experts`` field here is
       retained for potential future use but is superseded by
       ``GroupTaskExpertConfig`` for the standard pipeline.

    Args:
        shared_experts: List of registered expert names to use as shared
            experts in the CGC layer (e.g. ``["deepfm", "hgcn", "perslay"]``).
        task_experts: Default per-task expert names used when a task's
            group does not specify its own experts
            (e.g. ``["mlp", "deepfm"]``).
        group_task_experts: Per-group task expert overrides.  Keys are
            task group names; values are lists of expert names.  Tasks
            belonging to a group use these experts instead of the global
            ``task_experts``.  Superseded by ``GroupTaskExpertConfig``
            for production use.
        expert_configs: Per-expert configuration overrides.  Keys are
            expert names from ``shared_experts`` or ``task_experts``;
            values are dicts forwarded to the expert constructor's
            ``config`` parameter (e.g. ``{"hgcn": {"output_dim": 128}}``).
    """
    shared_experts: List[str] = field(default_factory=list)
    task_experts: List[str] = field(default_factory=list)
    group_task_experts: Dict[str, List[str]] = field(default_factory=dict)
    expert_configs: Dict[str, dict] = field(default_factory=dict)

    def get_task_experts_for(self, task_name: str, task_group: Optional[str] = None) -> List[str]:
        """Return the task expert names for a specific task.

        Looks up the task's group in ``group_task_experts`` first;
        falls back to the global ``task_experts`` if the group has no
        override or if *task_group* is ``None``.
        """
        if task_group and task_group in self.group_task_experts:
            return self.group_task_experts[task_group]
        return self.task_experts


@dataclass
class GradSurgeryConfig:
    """Gradient Surgery configuration.

    Gradient Surgery (PCGrad) projects conflicting task gradients onto each
    other's normal planes to reduce destructive interference during
    multi-task learning.  Only tasks belonging to different
    ``task_type_groups`` entries are compared; within-group pairs are always
    co-operative.

    Args:
        enabled: Whether gradient surgery is active during training.
        task_type_groups: Maps a group label (e.g. ``"binary"``,
            ``"multiclass"``, ``"regression"``) to the list of task names
            in that group.  Conflict detection is performed across groups.
            When empty, all tasks are treated as a single group (no
            cross-group conflicts resolved).
        conflict_threshold: Cosine-similarity threshold below which two
            task gradients are considered conflicting.  ``0.0`` (default)
            means any negative cosine similarity triggers projection.
        warmup_epochs: Number of epochs to skip gradient surgery at the
            start of training.  Allows the shared trunk to learn a common
            representation before conflict resolution is applied.
        ema_decay: Exponential moving average decay for the per-task-pair
            conflict-frequency tracker.  Used for logging and optional
            adaptive thresholding.
        log_interval: Log gradient conflict statistics every N steps.
    """
    enabled: bool = False
    task_type_groups: Dict[str, List[str]] = field(default_factory=dict)
    conflict_threshold: float = 0.0
    warmup_epochs: int = 2
    ema_decay: float = 0.9
    log_interval: int = 1


@dataclass
class GroupTaskExpertConfig:
    """Configuration for the GroupEncoder + ClusterEmbedding architecture.

    This is the preferred per-task expert approach (original v3.2 design):
    - GroupEncoder: shared MLP per task group (4 groups, not 16 individual)
    - ClusterEmbedding: GMM cluster -> learned embedding (user segment conditioning)
    - Output goes directly to TaskTower (no intermediate TaskHead bottleneck)

    Output dim = group_output_dim + cluster_embed_dim (e.g. 64 + 32 = 96D).
    Parameter efficiency: ~88% reduction vs independent experts per task.

    When ``enabled=False``, falls back to legacy per-task MLP experts.
    """
    enabled: bool = True
    group_hidden_dim: int = 128
    group_output_dim: int = 64
    cluster_embed_dim: int = 32
    dropout: float = 0.2


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

    # -- adaTT (Adaptive Task Transfer — loss-level TAG + GradNorm variant)
    adatt: AdaTTConfig = field(default_factory=AdaTTConfig)

    # -- AdaTT-sp (representation-level fusion, Li et al. KDD 2023) ---------
    adatt_sp: AdaTTSPConfig = field(default_factory=AdaTTSPConfig)

    # -- Residual recovery (intra-task, Paper 3) ----------------------------
    residual_recovery: ResidualRecoveryConfig = field(default_factory=ResidualRecoveryConfig)

    # -- ECEB (uncertainty-conditioned recovery, Paper 3 MV) ----------------
    eceb: ECEBConfig = field(default_factory=ECEBConfig)

    # -- BRP (boosting-residual path, Paper 3 MV) ---------------------------
    brp: BRPConfig = field(default_factory=BRPConfig)

    # -- Gradient Surgery (PCGrad) ------------------------------------------
    grad_surgery: GradSurgeryConfig = field(default_factory=GradSurgeryConfig)

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

    # -- Expert basket -------------------------------------------------------
    # When set, selects a subset of experts from the Expert Pool for this
    # pipeline.  When None, falls back to the legacy behaviour (all shared
    # experts are the same type defined by ``shared_expert``).
    expert_basket: Optional[ExpertBasketConfig] = None

    # -- Per-expert input dimensions -----------------------------------------
    # Maps expert name -> actual input dimension.  Used by ExpertBasket to
    # build each expert with its correct input size (e.g. LightGCN: 64,
    # HGCN: 47, PersLay: 70) instead of the full concatenated feature dim.
    # When empty, all experts receive input_dim.  Set from pipeline.yaml
    # ``model.expert_input_dims`` or computed from FeatureRouter at runtime.
    expert_input_dims: Dict[str, int] = field(default_factory=dict)

    # -- GroupEncoder + ClusterEmbedding + TaskHead (v3.2 architecture) ------
    group_task_expert: GroupTaskExpertConfig = field(
        default_factory=GroupTaskExpertConfig,
    )

    # -- Multidisciplinary feature routing ------------------------------------
    # Maps task_group_name -> list of feature indices within the 24D
    # multidisciplinary feature vector.  When non-empty, each task group
    # receives only its designated subgroup of multidisciplinary features
    # (projected and concatenated to tower input) instead of the full 24D.
    multidisciplinary_routing: Dict[str, List[int]] = field(default_factory=dict)

    # -- Task group mapping --------------------------------------------------
    # Maps task_name -> group_name.  Populated from pipeline-level
    # TaskGroupConfig list before model construction.  Used by
    # _build_task_experts() for group-specific expert selection.
    task_group_map: Dict[str, str] = field(default_factory=dict)

    # -- HMM group-to-mode mapping ------------------------------------------
    # Maps task_group_name -> hmm_mode (journey / lifecycle / behavior).
    # When empty, falls back to PLEModel._DEFAULT_HMM_GROUP_MODE_MAP.
    # Set from pipeline.yaml ``model.hmm_group_mode_map``.
    hmm_group_mode_map: Dict[str, str] = field(default_factory=dict)

    # -- HMM projectors toggle (ablation) ------------------------------------
    # When False, HMM triple-mode projectors are not built even if
    # task_group_map is populated.  Set via HP ``use_hmm_projectors=false``.
    hmm_projectors_enabled: bool = True

    # -- Per-task loss configuration -----------------------------------------
    # Maps task_name -> loss type string (e.g. "focal", "ce", "huber", "mse")
    task_loss_types: Dict[str, str] = field(default_factory=dict)
    # Maps task_name -> loss params dict (e.g. {"alpha": 0.25, "gamma": 2.0})
    task_loss_params: Dict[str, Dict] = field(default_factory=dict)
    # Maps task_name -> scalar loss weight (e.g. {"has_nba": 2.5, "churn": 2.0})
    # Applied as a multiplier to each task's loss before aggregation.
    task_loss_weights: Dict[str, float] = field(default_factory=dict)

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
        task_type = self.get_task_type(task_name)
        default_act = None if task_type == "regression" else "sigmoid"
        act = override.get("activation", default_act)
        return None if act in (None, "none", "null") else act

    def get_task_type(self, task_name: str) -> str:
        """Return the task type string for a specific task."""
        override = self.task_overrides.get(task_name, {})
        return override.get("task_type", "binary")

    def get_task_topk_k(self, task_name: str) -> Optional[List[int]]:
        """Return the list of K values for top-K metrics, or None if not configured.

        Only multiclass tasks that explicitly set ``topk_k`` in their task
        definition will have top-K accuracy and NDCG@K computed.
        Returns ``None`` for tasks that have not configured this field.
        """
        override = self.task_overrides.get(task_name, {})
        k_values = override.get("topk_k")
        if k_values is None:
            return None
        return [int(k) for k in k_values]

    def get_domain_experts(self, task_name: str) -> List[str]:
        """Return the list of domain-relevant expert names for a task."""
        override = self.task_overrides.get(task_name, {})
        return override.get("domain_experts", [])

    def get_tower_type(self, task_name: str) -> str:
        """Return the tower type for a specific task.

        Looks up ``tower_type`` in task_overrides; defaults to
        ``"standard"``.  Use ``"contrastive"`` for tasks with many
        classes (e.g. brand_prediction).
        """
        override = self.task_overrides.get(task_name, {})
        return override.get("tower_type", "standard")

    def get_task_loss_type(self, task_name: str) -> str:
        """Return the loss type string for a specific task.

        Priority:
          1. Explicit entry in ``task_loss_types``.
          2. ``loss`` key in ``task_overrides``.
          3. Auto-infer from task_type: binary->bce, multiclass->ce,
             regression->huber.
        """
        # 1. Explicit task_loss_types mapping
        if task_name in self.task_loss_types:
            return self.task_loss_types[task_name]

        # 2. task_overrides "loss" key
        override = self.task_overrides.get(task_name, {})
        if "loss" in override:
            return override["loss"]

        # 3. Auto-infer from task_type
        task_type = self.get_task_type(task_name)
        _default_loss = {
            "binary": "bce",
            "multiclass": "ce",
            "regression": "huber",
        }
        return _default_loss.get(task_type, "mse")

    def get_task_loss_params(self, task_name: str) -> Dict:
        """Return loss hyperparameters for a specific task.

        Merges ``task_loss_params[task_name]`` with any ``loss_params``
        key in ``task_overrides[task_name]``.
        """
        params: Dict = {}
        override = self.task_overrides.get(task_name, {})
        if "loss_params" in override:
            params.update(override["loss_params"])
        if task_name in self.task_loss_params:
            params.update(self.task_loss_params[task_name])
        return params

    # -- Evidential Deep Learning ---------------------------------------------
    evidential_enabled: bool = False
    evidential_kl_lambda: float = 0.01
    evidential_annealing_epochs: int = 10

    # -- Sparse Autoencoder (SAE) --------------------------------------------
    sae_enabled: bool = False
    sae_weight: float = 0.01
    sae_expansion_factor: int = 4
    sae_l1_lambda: float = 0.001

    # -----------------------------------------------------------------------
    # Helpers (continued)
    # -----------------------------------------------------------------------

    def get_tower_dims(self, task_name: str) -> Optional[List[int]]:
        """Return per-task tower hidden_dims override, or ``None`` for default.

        When ``None``, ``_build_task_towers`` uses
        ``TaskTowerConfig.hidden_dims`` as the global default.
        """
        override = self.task_overrides.get(task_name, {})
        return override.get("tower_dims")

    def build_task_group_map_from_groups(self) -> None:
        """Populate ``task_group_map`` from ``adatt.task_groups``.

        Iterates over the task groups defined in the AdaTTConfig and
        constructs a ``{task_name: group_name}`` mapping.  This enables
        GroupEncoder, HMM triple-mode routing, and multidisciplinary
        feature routing -- all of which require a non-empty
        ``task_group_map``.

        Safe to call multiple times; only overwrites if task_group_map
        is currently empty.
        """
        if self.task_group_map:
            return  # already populated, don't overwrite

        if not self.adatt.task_groups:
            return  # no groups defined

        new_map: Dict[str, str] = {}
        for group_name, group_def in self.adatt.task_groups.items():
            members = (
                group_def.members
                if hasattr(group_def, "members")
                else group_def.get("members", [])
            )
            for task_name in members:
                new_map[task_name] = group_name

        self.task_group_map = new_map
