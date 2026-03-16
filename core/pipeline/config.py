"""
YAML 설정 파일 → PipelineConfig 파싱.

하나의 YAML 파일로 데이터, 피처, 모델, 태스크, AWS 환경을 모두 정의합니다.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TaskSpec:
    name: str
    type: str           # binary | multiclass | regression | ranking
    loss: str
    loss_weight: float = 1.0
    label_col: str = "label"
    num_classes: int = 1


@dataclass
class DataSpec:
    source: str         # s3://bucket/path 또는 로컬 경로
    format: str = "parquet"
    train_split: float = 0.8
    val_split: float = 0.1


@dataclass
class FeatureSpec:
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    sequence: list[str] = field(default_factory=list)
    embedding_dim: int = 16
    transformers: list[dict] = field(default_factory=list)  # [{name: "standard_scaler", cols: [...]}]


@dataclass
class ModelSpec:
    architecture: str = "ple"   # ple | lgbm
    num_shared_experts: int = 2
    num_task_experts: int = 2
    expert_hidden_dim: int = 256
    num_layers: int = 2
    tower_dims: list[int] = field(default_factory=lambda: [128, 64])
    dropout: float = 0.1


@dataclass
class AWSSpec:
    region: str = "ap-northeast-2"
    s3_bucket: str = ""
    instance_type: str = "ml.g4dn.xlarge"
    use_spot: bool = True
    max_run_seconds: int = 7200
    role_arn: str = ""


@dataclass
class TrainingSpec:
    batch_size: int = 2048
    epochs: int = 20
    learning_rate: float = 1e-3
    early_stopping_patience: int = 5
    seed: int = 42


@dataclass
class PipelineConfig:
    task_name: str
    tasks: list[TaskSpec]
    data: DataSpec
    features: FeatureSpec
    model: ModelSpec
    training: TrainingSpec
    aws: AWSSpec


def load_config(path: str | Path) -> PipelineConfig:
    """YAML 파일을 읽어 PipelineConfig로 변환합니다."""
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    tasks = [TaskSpec(**t) for t in raw.get("tasks", [])]
    data = DataSpec(**raw.get("data", {}))
    features = FeatureSpec(**raw.get("features", {}))
    model = ModelSpec(**raw.get("model", {}))
    training = TrainingSpec(**raw.get("training", {}))
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
