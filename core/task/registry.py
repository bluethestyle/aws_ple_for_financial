from typing import Type

from .base import AbstractTask, TaskConfig
from .types import LossType, TaskType


class TaskRegistry:
    """
    태스크 헤드 플러그인 레지스트리.

    기본 구현체(binary, multiclass, regression, ranking)가 자동 등록됩니다.
    커스텀 태스크는 @TaskRegistry.register("my_task")로 추가합니다.

    Example:
        @TaskRegistry.register("my_custom_task")
        class MyTask(AbstractTask):
            ...

        task = TaskRegistry.build(TaskConfig(name="t1", task_type="my_custom_task", ...))
    """

    _registry: dict[str, Type[AbstractTask]] = {}

    @classmethod
    def register(cls, task_type: str):
        def decorator(task_cls: Type[AbstractTask]):
            cls._registry[task_type] = task_cls
            return task_cls
        return decorator

    @classmethod
    def build(cls, config: TaskConfig, tower_input_dim: int) -> AbstractTask:
        key = config.task_type.value if isinstance(config.task_type, TaskType) else config.task_type
        if key not in cls._registry:
            raise KeyError(
                f"Unknown task type '{key}'. "
                f"Registered: {list(cls._registry.keys())}"
            )
        return cls._registry[key](config, tower_input_dim)

    @classmethod
    def list_registered(cls) -> list[str]:
        return list(cls._registry.keys())


# ── 기본 구현체 등록 ────────────────────────────────────────────────

import torch
import torch.nn.functional as F


@TaskRegistry.register(TaskType.BINARY.value)
class BinaryTask(AbstractTask):
    def compute_loss(self, logits, labels):
        return F.binary_cross_entropy_with_logits(logits.squeeze(-1), labels.float())

    def predict(self, logits):
        return torch.sigmoid(logits)


@TaskRegistry.register(TaskType.MULTICLASS.value)
class MulticlassTask(AbstractTask):
    def _build_tower(self, input_dim):
        import torch.nn as nn
        return nn.Linear(input_dim, self.config.num_classes)

    def compute_loss(self, logits, labels):
        return F.cross_entropy(logits, labels.long())

    def predict(self, logits):
        return torch.softmax(logits, dim=-1)


@TaskRegistry.register(TaskType.REGRESSION.value)
class RegressionTask(AbstractTask):
    def compute_loss(self, logits, labels):
        return F.huber_loss(logits.squeeze(-1), labels.float())

    def predict(self, logits):
        return logits


@TaskRegistry.register(TaskType.RANKING.value)
class RankingTask(AbstractTask):
    def compute_loss(self, logits, labels):
        # ListNet loss (softmax over scores vs. softmax over labels)
        pred_probs = torch.softmax(logits.squeeze(-1), dim=-1)
        true_probs = torch.softmax(labels.float(), dim=-1)
        return -(true_probs * torch.log(pred_probs + 1e-10)).sum(dim=-1).mean()

    def predict(self, logits):
        return logits
