from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from .types import LossType, TaskType


@dataclass
class TaskConfig:
    name: str
    task_type: TaskType
    loss_type: LossType
    loss_weight: float = 1.0
    num_classes: int = 1          # multiclass일 때 클래스 수
    output_dim: int = 1
    label_col: str = "label"
    extra: dict = field(default_factory=dict)


@dataclass
class TaskOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    predictions: torch.Tensor | None = None


class AbstractTask(ABC, nn.Module):
    """
    모든 태스크 헤드의 기반 클래스.

    새로운 태스크를 추가하려면 이 클래스를 상속하고
    `compute_loss`와 `predict`를 구현하면 됩니다.

    Example:
        class MyBinaryTask(AbstractTask):
            def compute_loss(self, logits, labels):
                return F.binary_cross_entropy_with_logits(logits, labels)

            def predict(self, logits):
                return torch.sigmoid(logits)
    """

    def __init__(self, config: TaskConfig, tower_input_dim: int):
        super().__init__()
        self.config = config
        self.tower = self._build_tower(tower_input_dim)

    def _build_tower(self, input_dim: int) -> nn.Sequential:
        """태스크별 출력 레이어. 필요시 오버라이드."""
        return nn.Linear(input_dim, self.config.output_dim)

    def forward(self, expert_output: torch.Tensor, labels: torch.Tensor | None = None) -> TaskOutput:
        logits = self.tower(expert_output)
        loss = self.compute_loss(logits, labels) if labels is not None else None
        predictions = self.predict(logits)
        return TaskOutput(logits=logits, loss=loss, predictions=predictions)

    @abstractmethod
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        """logits → 사람이 읽을 수 있는 예측값 (확률, 클래스, 점수 등)"""
        ...

    @property
    def name(self) -> str:
        return self.config.name
