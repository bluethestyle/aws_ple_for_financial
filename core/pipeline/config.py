"""
Pipeline configuration -- YAML to PipelineConfig parsing.

A single YAML file defines data sources, feature schema, model architecture,
task definitions, training parameters, and AWS deployment settings.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class TaskGroupConfig:
    """Definition of a task group for adaTT transfer and monitoring.

    Task groups are defined once in pipeline.yaml and propagated to adaTT
    (intra/inter transfer strengths), loss weighting, monitoring, and
    per-group task expert selection.

    When ``task_experts`` is non-empty, tasks in this group use the
    specified experts instead of the global ``expert_basket.task_experts``.
    This enables domain-specific expert assignment per task group
    (e.g. lifecycle tasks use causal expert, consumption tasks use deepfm).
    """

    name: str = ""
    tasks: List[str] = field(default_factory=list)
    task_experts: List[str] = field(default_factory=list)
    adatt_intra_strength: float = 0.8   # intra-group transfer strength
    adatt_inter_strength: float = 0.3   # inter-group transfer strength
    description: str = ""


@dataclass
class TaskSpec:
    """Specification for a single learning task."""

    name: str
    type: str           # binary | multiclass | regression | ranking | contrastive
    loss: str
    loss_weight: float = 1.0
    label_col: str = "label"
    num_classes: int = 1
    tower_type: str = ""        # "" = use default ("standard")
    tower_dims: List[int] = field(default_factory=list)  # [] = use global default
    description: str = ""
    # Top-K accuracy / NDCG@K gate (multiclass only). When the YAML task
    # block declares e.g. ``topk_k: [1, 3]``, the trainer's
    # _compute_topk_metrics path activates and ``avg_ndcg@3`` flows
    # through to the saved checkpoint metrics + SageMaker FinalMetrics.
    # Previously absent from the dataclass: yaml ``topk_k`` was silently
    # dropped at load time, so PLEConfig.get_task_topk_k returned None
    # for nba_primary / next_mcc and NDCG never appeared in any v14 run.
    topk_k: List[int] = field(default_factory=list)
    # Per-task domain expert preferences (optional task-level routing
    # hint).  Read by PLEConfig.get_domain_experts.
    domain_experts: List[str] = field(default_factory=list)
    # Label-derivation rule when the YAML uses
    # ``derive: {method, source_col, filter_col, ...}``. The runner reads
    # this dict to (a) build labels in Stage 4 and (b) exclude the
    # referenced helper columns (e.g. ``has_nba`` for the
    # ``nba_primary.derive.filter_col`` link) from the post-Stage-6
    # feature matrix. Previously absent from the dataclass, so the YAML
    # block was silently dropped at load time.
    derive: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataSpec:
    """Data source specification."""

    source: str = ""    # s3://bucket/path or local file path
    format: str = "parquet"
    train_split: float = 0.8
    val_split: float = 0.1
    backend: Any = "pandas"  # str or List[str] (e.g. ["cudf", "duckdb", "pandas"])
    train_path: str = ""
    s3_path: str = ""
    parquet_file: str = ""
    temporal_split: Optional[Dict[str, Any]] = None  # temporal split config
    preprocessing: Optional[Dict[str, Any]] = None    # preprocessing config
    # Adapter-required identifiers. SantanderAdapter (and any adapter that
    # builds row-level joins) reads ``id_col`` from this section; without
    # it the adapter raises ``data.id_col must be specified``.  Yaml has
    # always carried these keys -- DataSpec just wasn't surfacing them.
    id_col: Optional[str] = None
    total_rows: Optional[int] = None


@dataclass
class FeatureSpec:
    """Feature column layout and transformer definitions.

    The ``transformers`` list maps directly to FeaturePipelineBuilder steps.
    Each entry must have a ``transformer`` (or ``name``) key and optional
    ``params``.
    """

    numeric: List[str] = field(default_factory=list)
    categorical: List[Any] = field(default_factory=list)
    sequence: List[str] = field(default_factory=list)
    embedding_dim: int = 16
    id_cols: List[str] = field(default_factory=list)
    # Pure-metadata columns (snapshot_date, partitioning keys, etc.).
    # The runner's Stage 6 feature filter excludes these from the feature
    # matrix; without this field, ``meta_cols: [snapshot_date]`` declared
    # in the dataset YAML was silently dropped by the dataclass loader,
    # leaving snapshot_date in features.parquet at column 0.
    meta_cols: List[str] = field(default_factory=list)
    transformers: List[dict] = field(default_factory=list)
    input_dim: int = 0

    @property
    def categorical_names(self) -> List[str]:
        """Extract plain column names from categorical, which may contain
        dicts with a ``name`` key or plain strings."""
        names: List[str] = []
        for entry in self.categorical:
            if isinstance(entry, dict):
                names.append(entry.get("name", ""))
            else:
                names.append(str(entry))
        return names


@dataclass
class ModelSpec:
    """Model architecture specification.

    When ``architecture`` is ``"ple"``, the PLE-specific fields
    (num_shared_experts, tower_dims, etc.) are forwarded to PLEConfig.
    When ``architecture`` is ``"lgbm"``, they are ignored.

    The optional ``expert_basket`` dict, when present, is forwarded to
    :class:`ExpertBasketConfig` during PLE model construction.
    """

    architecture: str = "ple"   # ple | lgbm
    num_shared_experts: int = 2
    num_task_experts: int = 2
    expert_hidden_dim: int = 256
    num_layers: int = 2
    tower_dims: List[int] = field(default_factory=lambda: [128, 64])
    dropout: float = 0.1
    expert_basket: Optional[dict] = None
    group_task_expert: Optional[dict] = None
    # Per-expert MLP / DeepFM / Causal / OT hidden dim configuration. The
    # config_builder reads ``expert_config.mlp.hidden_dims`` to size the
    # shared/task expert internal MLPs. Without this field on the
    # dataclass, ``_config_to_dict`` silently dropped the entire pipeline
    # ``model.expert_config`` block from label_schema.json, and
    # config_builder fell through to its dynamic fallback
    # ``[input_dim*2, input_dim]`` — which on a 1355-D feature matrix
    # ballooned the model from ~2.6 M params (v12, [128, 64]) to ~171 M
    # params (v13, [2710, 1355]).
    expert_config: dict = field(default_factory=dict)
    ple: dict = field(default_factory=dict)
    task_tower: dict = field(default_factory=dict)
    cgc: dict = field(default_factory=dict)
    adatt: dict = field(default_factory=dict)


@dataclass
class SecuritySpec:
    """Security / encryption specification."""

    enabled: bool = False
    salt_source: str = "env"  # "env" | "ssm" | "file"
    policies: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LabelSpec:
    """Specification for a single label (target column or derived label)."""

    name: str = ""
    source: str = "column"  # "column" | "derive"
    method: str = ""        # derivation method when source="derive"
    input_col: str = ""
    type: str = "binary"    # "binary" | "multiclass" | "regression"
    num_classes: int = 2
    # Type-specific config (indices for list_intersect, mapping for
    # string_map, mode/top_k for sequence_last, weights for weighted_sum,
    # threshold/operator for threshold, etc). Stored verbatim so the
    # downstream :class:`LabelDeriver` receives whatever the YAML author
    # supplied. Without this field every non-canonical key gets dropped
    # at dataclass-parse time (see ``load_config``) and label derivation
    # blows up with KeyError mid-pipeline.
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SequenceSpec:
    """Specification for a sequential / temporal input."""

    source: str = "parquet_list"  # "parquet_list" | "npy"
    columns: List[str] = field(default_factory=list)
    seq_len: int = 50
    pad_value: float = 0.0
    mode: str = "count_based"  # "count_based" | "time_based"
    window_days: int = 90
    stride_days: int = 0
    timestamp_col: str = ""
    truncate_last: int = 0


@dataclass
class ItemUniverseSpec:
    """Item universe definition for recommendation tasks."""

    name: str = ""
    items: List[str] = field(default_factory=list)
    hierarchy: List[str] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class FeatureGroupSpec:
    """Thin wrapper bridging pipeline config to FeatureGroupConfig.

    Raw dicts from the YAML ``feature_groups`` section are stored here and
    later forwarded to :class:`core.feature.group.FeatureGroupConfig` by
    :class:`FeatureGroupPipeline`.
    """

    name: str = ""
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AWSSpec:
    """AWS / SageMaker deployment configuration.

    ``region`` has no hardcoded default per CLAUDE.md §1.1. Callers must
    supply it via ``pipeline.yaml::aws.region``; leaving it unset lets
    boto3 resolve from env / credentials.
    """

    region: Optional[str] = None
    s3_bucket: str = ""
    instance_type: str = "ml.g4dn.xlarge"
    cpu_instance_type: str = "ml.m5.2xlarge"
    gpu_instance_type: Optional[str] = None
    use_spot: bool = True
    max_run_seconds: int = 7200
    role_arn: str = ""
    # Pre-baked ECR image for the Mamba GPU pre-compute job
    # (mamba_ssm + causal-conv1d + ninja). Built once via
    # scripts/build_mamba_image.sh; absent value falls back to
    # the stock PyTorch GPU DLC and runtime wheel build.
    mamba_image_uri: Optional[str] = None


@dataclass
class TrainingSpec:
    """Training hyperparameters.

    For PLE models, finer-grained control (phase1/phase2 epochs, optimizer,
    scheduler, AMP) is available via a separate TrainingConfig loaded from
    the ``training`` block of the YAML.
    """

    batch_size: int = 2048
    epochs: int = 20
    learning_rate: float = 1e-3
    early_stopping_patience: int = 5
    seed: int = 42


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration.

    Aggregates all sub-configs required by :class:`PipelineRunner` to
    execute the full training pipeline (data loading, feature engineering,
    model training, saving).
    """

    task_name: str
    tasks: List[TaskSpec]
    data: DataSpec
    features: FeatureSpec
    model: ModelSpec
    training: TrainingSpec
    aws: AWSSpec
    task_groups: List[TaskGroupConfig] = field(default_factory=list)
    security: SecuritySpec = field(default_factory=SecuritySpec)
    labels: List[LabelSpec] = field(default_factory=list)
    sequences: Dict[str, SequenceSpec] = field(default_factory=dict)
    item_universe: Optional[ItemUniverseSpec] = None
    feature_groups: List[dict] = field(default_factory=list)
    adapter: str = ""
    task_relationships: List[dict] = field(default_factory=list)
    logit_transfer_strength: float = 0.5
    adatt: dict = field(default_factory=dict)

    def get_task_group(self, task_name: str) -> Optional[str]:
        """Return the group name that *task_name* belongs to, or ``None``."""
        for group in self.task_groups:
            if task_name in group.tasks:
                return group.name
        return None


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *override* into *base*. Override wins on conflicts.

    Lists and scalar values are replaced (not appended).  Only dict-valued
    keys are merged recursively.

    Args:
        base: Base configuration dict (e.g. from common ``pipeline.yaml``).
        override: Dataset-specific overrides that take precedence.

    Returns:
        A new dict with overrides applied.
    """
    result: Dict[str, Any] = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_merged_config(
    pipeline_path: Union[str, Path],
    dataset_path: Union[str, Path],
) -> Dict[str, Any]:
    """Load and deep-merge a common pipeline config with a dataset-specific config.

    Dataset config values override pipeline config where they overlap.

    Args:
        pipeline_path: Path to the common ``configs/pipeline.yaml``.
        dataset_path:  Path to the dataset-specific ``configs/datasets/{name}.yaml``.

    Returns:
        Merged raw config dict (not yet parsed into :class:`PipelineConfig`).
    """
    with open(pipeline_path, encoding="utf-8") as f:
        base: Dict[str, Any] = yaml.safe_load(f) or {}
    with open(dataset_path, encoding="utf-8") as f:
        dataset: Dict[str, Any] = yaml.safe_load(f) or {}
    return deep_merge(base, dataset)


def load_config(path: Union[str, Path], dataset_path: Optional[Union[str, Path]] = None) -> PipelineConfig:
    """Parse a YAML file into a :class:`PipelineConfig` instance.

    When *dataset_path* is provided the two files are deep-merged before
    parsing (dataset values override common pipeline values).

    Args:
        path: Path to the pipeline config YAML (common or legacy single-file).
        dataset_path: Optional dataset-specific YAML to merge on top.
    """
    if dataset_path is not None:
        raw: Dict[str, Any] = load_merged_config(path, dataset_path)
    else:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    tasks = [TaskSpec(**{k: v for k, v in t.items() if k in TaskSpec.__dataclass_fields__}) for t in raw.get("tasks", [])]
    data = DataSpec(**{
        k: v for k, v in raw.get("data", {}).items()
        if k in DataSpec.__dataclass_fields__
    })
    features = FeatureSpec(**{
        k: v for k, v in raw.get("features", {}).items()
        if k in FeatureSpec.__dataclass_fields__
    })
    model = ModelSpec(**{
        k: v for k, v in raw.get("model", {}).items()
        if k in ModelSpec.__dataclass_fields__
    })
    training = TrainingSpec(**{
        k: v for k, v in raw.get("training", {}).items()
        if k in TrainingSpec.__dataclass_fields__
    })
    aws = AWSSpec(**{
        k: v for k, v in raw.get("aws", {}).items()
        if k in AWSSpec.__dataclass_fields__
    })
    task_groups = [TaskGroupConfig(**tg) for tg in raw.get("task_groups", [])]

    # --- New pipeline sections (optional, backward-compatible) ---
    security = SecuritySpec(**{
        k: v for k, v in raw.get("security", {}).items()
        if k in SecuritySpec.__dataclass_fields__
    })
    # labels: accept both list-of-dicts and dict-of-dicts (name→spec) formats
    _labels_raw = raw.get("labels", [])
    _label_fields = LabelSpec.__dataclass_fields__.keys()

    def _build_label_spec(spec: Dict[str, Any], name: str = "") -> LabelSpec:
        """Split spec dict into canonical LabelSpec fields + extras."""
        canonical = {k: v for k, v in spec.items()
                     if k in _label_fields and k != "extra"}
        extras = {k: v for k, v in spec.items()
                  if k not in _label_fields}
        if name and "name" not in canonical:
            canonical["name"] = name
        if "extra" in spec and isinstance(spec["extra"], dict):
            extras = {**spec["extra"], **extras}
        return LabelSpec(extra=extras, **canonical)

    if isinstance(_labels_raw, dict):
        labels = [
            _build_label_spec(spec, name=name)
            for name, spec in _labels_raw.items()
            if isinstance(spec, dict)
        ]
    elif isinstance(_labels_raw, list):
        labels = [
            _build_label_spec(lb)
            for lb in _labels_raw
            if isinstance(lb, dict)
        ]
    else:
        labels = []

    # Fallback: extract labels from tasks[].derive blocks if no top-level labels
    if not labels:
        for t in raw.get("tasks", []):
            derive_block = t.get("derive")
            if derive_block:
                labels.append(LabelSpec(
                    name=t.get("label_col", t.get("name", "")),
                    source="derive",
                    method=derive_block.get("method", ""),
                    input_col=derive_block.get("input_col", derive_block.get("source_col", "")),
                    type=t.get("type", "binary"),
                    num_classes=t.get("num_classes", 2),
                ))

    # Top-level sequences section
    sequences_raw = raw.get("sequences", {})

    # Fallback: extract sequences from features.sequences if no top-level sequences
    if not sequences_raw:
        feat_raw = raw.get("features", {})
        sequences_raw = feat_raw.get("sequences", {})

    sequences = {
        name: SequenceSpec(**{
            k: v for k, v in spec.items()
            if k in SequenceSpec.__dataclass_fields__
        })
        for name, spec in sequences_raw.items()
    }
    raw_item_universe = raw.get("item_universe")
    item_universe = ItemUniverseSpec(**{
        k: v for k, v in raw_item_universe.items()
        if k in ItemUniverseSpec.__dataclass_fields__
    }) if raw_item_universe else None
    feature_groups_raw = raw.get("feature_groups", [])

    # If no inline feature_groups, look for feature_groups_path or
    # auto-discover configs/<adapter>/feature_groups.yaml next to the
    # pipeline YAML (or next to the dataset YAML when using split configs).
    if not feature_groups_raw:
        # Accept both ``feature_groups_path`` (canonical) and
        # ``feature_groups_file`` (the key the santander dataset YAML
        # has been using). Without this alias the dataset's explicit
        # pointer at ``configs/santander/feature_groups.yaml`` was
        # silently ignored and auto-discovery served the legacy
        # top-level ``configs/feature_groups.yaml`` instead.
        fg_path_str = (
            raw.get("feature_groups_path", "")
            or raw.get("feature_groups_file", "")
        )
        if fg_path_str:
            fg_path = Path(fg_path_str)
            if not fg_path.is_absolute():
                # Resolve against the pipeline YAML's directory; if that
                # resolution misses, fall back to the project root so
                # values like "configs/santander/feature_groups.yaml"
                # work regardless of which dir the pipeline YAML lives in.
                resolved = Path(path).parent / fg_path
                if not resolved.exists() and not str(fg_path).startswith("configs"):
                    resolved = fg_path
                elif not resolved.exists():
                    project_root = Path(path).resolve().parents[1]
                    candidate = project_root / fg_path
                    if candidate.exists():
                        resolved = candidate
                fg_path = resolved
        else:
            # Auto-discover sibling feature_groups.yaml.
            # When using split configs, prefer the dataset YAML's directory
            # (configs/datasets/) before falling back to the pipeline YAML's dir.
            fg_path = Path(path).parent / "feature_groups.yaml"
            if dataset_path is not None:
                dataset_sibling = Path(dataset_path).parent / "feature_groups.yaml"
                if dataset_sibling.exists():
                    fg_path = dataset_sibling

        if fg_path.exists():
            import logging as _logging
            _lg = _logging.getLogger(__name__)
            _lg.info("Loading feature groups from %s", fg_path)
            with open(fg_path, "r", encoding="utf-8") as _fh:
                _fg_raw = yaml.safe_load(_fh)
            feature_groups_raw = _fg_raw.get("feature_groups", [])

    adapter_name = raw.get("adapter", "")

    # Top-level task relationship / transfer config (optional)
    task_relationships = raw.get("task_relationships", [])
    logit_transfer_strength = float(raw.get("logit_transfer_strength", 0.5))
    adatt = raw.get("adatt", {})

    return PipelineConfig(
        task_name=raw["task_name"],
        tasks=tasks,
        data=data,
        features=features,
        model=model,
        training=training,
        aws=aws,
        task_groups=task_groups,
        security=security,
        labels=labels,
        sequences=sequences,
        item_universe=item_universe,
        feature_groups=feature_groups_raw,
        adapter=adapter_name,
        task_relationships=task_relationships,
        logit_transfer_strength=logit_transfer_strength,
        adatt=adatt,
    )
