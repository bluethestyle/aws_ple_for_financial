"""
Task head subsystem for the PLE multi-task learning platform.

Public API::

    from core.task import (
        # Enums
        TaskType, LossType, ActivationType,
        # Config & output
        TaskConfig, TaskOutput,
        # Base class (for custom tasks)
        AbstractTask,
        # Registry (build tasks from config)
        TaskRegistry,
        # Built-in task classes
        BinaryTask, MulticlassTask, RegressionTask, RankingTask, ContrastiveTask,
        # Loss functions
        FocalLoss, QuantileLoss, InfoNCELoss, ListNetLoss, build_loss,
    )

Typical workflow::

    cfg = TaskConfig(name="my_task", task_type=TaskType.BINARY, output_dim=1)
    task = TaskRegistry.build(cfg, tower_input_dim=128)
    output = task(expert_output, labels=labels)
"""

from .types import ActivationType, LossType, TaskType
from .base import AbstractTask, TaskConfig, TaskOutput
from .losses import FocalLoss, InfoNCELoss, ListNetLoss, QuantileLoss, build_loss
from .registry import (
    BinaryTask,
    ContrastiveTask,
    MulticlassTask,
    RankingTask,
    RegressionTask,
    TaskRegistry,
)

__all__ = [
    # Enums
    "TaskType",
    "LossType",
    "ActivationType",
    # Data classes
    "TaskConfig",
    "TaskOutput",
    # Base
    "AbstractTask",
    # Registry
    "TaskRegistry",
    # Built-in tasks
    "BinaryTask",
    "MulticlassTask",
    "RegressionTask",
    "RankingTask",
    "ContrastiveTask",
    # Losses
    "FocalLoss",
    "QuantileLoss",
    "InfoNCELoss",
    "ListNetLoss",
    "build_loss",
]
