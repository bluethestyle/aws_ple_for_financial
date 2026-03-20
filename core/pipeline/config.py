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


@dataclass
class DataSpec:
    """Data source specification."""

    source: str = ""    # s3://bucket/path or local file path
    format: str = "parquet"
    train_split: float = 0.8
    val_split: float = 0.1
    backend: str = "pandas"
    train_path: str = ""
    s3_path: str = ""


@dataclass
class FeatureSpec:
    """Feature column layout and transformer definitions.

    The ``transformers`` list maps directly to FeaturePipelineBuilder steps.
    Each entry must have a ``transformer`` (or ``name``) key and optional
    ``params``.
    """

    numeric: List[str] = field(default_factory=list)
    categorical: List[str] = field(default_factory=list)
    sequence: List[str] = field(default_factory=list)
    embedding_dim: int = 16
    id_cols: List[str] = field(default_factory=list)
    transformers: List[dict] = field(default_factory=list)
    input_dim: int = 0


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


@dataclass
class AWSSpec:
    """AWS / SageMaker deployment configuration."""

    region: str = "ap-northeast-2"
    s3_bucket: str = ""
    instance_type: str = "ml.g4dn.xlarge"
    use_spot: bool = True
    max_run_seconds: int = 7200
    role_arn: str = ""


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

    def get_task_group(self, task_name: str) -> Optional[str]:
        """Return the group name that *task_name* belongs to, or ``None``."""
        for group in self.task_groups:
            if task_name in group.tasks:
                return group.name
        return None


def load_config(path: Union[str, Path]) -> PipelineConfig:
    """Parse a YAML file into a :class:`PipelineConfig` instance."""
    with open(path) as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    tasks = [TaskSpec(**t) for t in raw.get("tasks", [])]
    data = DataSpec(**raw.get("data", {}))
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
    aws = AWSSpec(**raw.get("aws", {}))
    task_groups = [TaskGroupConfig(**tg) for tg in raw.get("task_groups", [])]

    return PipelineConfig(
        task_name=raw["task_name"],
        tasks=tasks,
        data=data,
        features=features,
        model=model,
        training=training,
        aws=aws,
        task_groups=task_groups,
    )
