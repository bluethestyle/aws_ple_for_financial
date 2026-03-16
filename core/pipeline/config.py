"""
Pipeline configuration -- YAML to PipelineConfig parsing.

A single YAML file defines data sources, feature schema, model architecture,
task definitions, training parameters, and AWS deployment settings.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TaskSpec:
    """Specification for a single learning task."""

    name: str
    type: str           # binary | multiclass | regression | ranking
    loss: str
    loss_weight: float = 1.0
    label_col: str = "label"
    num_classes: int = 1


@dataclass
class DataSpec:
    """Data source specification."""

    source: str         # s3://bucket/path or local file path
    format: str = "parquet"
    train_split: float = 0.8
    val_split: float = 0.1


@dataclass
class FeatureSpec:
    """Feature column layout and transformer definitions.

    The ``transformers`` list maps directly to FeaturePipelineBuilder steps.
    Each entry must have a ``transformer`` (or ``name``) key and optional
    ``params``.
    """

    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    sequence: list[str] = field(default_factory=list)
    embedding_dim: int = 16
    id_cols: list[str] = field(default_factory=list)
    transformers: list[dict] = field(default_factory=list)


@dataclass
class ModelSpec:
    """Model architecture specification.

    When ``architecture`` is ``"ple"``, the PLE-specific fields
    (num_shared_experts, tower_dims, etc.) are forwarded to PLEConfig.
    When ``architecture`` is ``"lgbm"``, they are ignored.
    """

    architecture: str = "ple"   # ple | lgbm
    num_shared_experts: int = 2
    num_task_experts: int = 2
    expert_hidden_dim: int = 256
    num_layers: int = 2
    tower_dims: list[int] = field(default_factory=lambda: [128, 64])
    dropout: float = 0.1


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
    tasks: list[TaskSpec]
    data: DataSpec
    features: FeatureSpec
    model: ModelSpec
    training: TrainingSpec
    aws: AWSSpec


def load_config(path: str | Path) -> PipelineConfig:
    """Parse a YAML file into a :class:`PipelineConfig` instance."""
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)

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

    return PipelineConfig(
        task_name=raw["task_name"],
        tasks=tasks,
        data=data,
        features=features,
        model=model,
        training=training,
        aws=aws,
    )
