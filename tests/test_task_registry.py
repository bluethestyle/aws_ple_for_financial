"""Tests for the TaskRegistry and built-in task implementations."""

import torch
import pytest

from core.task.base import TaskConfig, AbstractTask, TaskOutput
from core.task.registry import TaskRegistry
from core.task.types import TaskType, LossType


def make_config(task_type: TaskType, **kwargs) -> TaskConfig:
    return TaskConfig(
        name="test_task",
        task_type=task_type,
        loss_type=LossType.BCE,
        **kwargs,
    )


@pytest.mark.parametrize("task_type", [
    TaskType.BINARY,
    TaskType.REGRESSION,
    TaskType.RANKING,
])
def test_registry_builds_default_tasks(task_type):
    config = make_config(task_type)
    task = TaskRegistry.build(config, tower_input_dim=64)
    assert task is not None
    assert task.name == "test_task"


def test_binary_task_forward():
    config = make_config(TaskType.BINARY)
    task = TaskRegistry.build(config, tower_input_dim=64)

    x = torch.randn(8, 64)
    labels = torch.randint(0, 2, (8,)).float()
    output = task(x, labels)

    assert output.logits.shape == (8, 1)
    assert output.loss is not None
    assert output.predictions.shape == (8, 1)


def test_multiclass_task_forward():
    config = make_config(TaskType.MULTICLASS, num_classes=5, output_dim=5)
    task = TaskRegistry.build(config, tower_input_dim=64)

    x = torch.randn(8, 64)
    labels = torch.randint(0, 5, (8,))
    output = task(x, labels)

    assert output.predictions.shape == (8, 5)


def test_unknown_task_type_raises():
    config = make_config(TaskType.BINARY)
    config.task_type = "nonexistent_type"
    with pytest.raises(KeyError):
        TaskRegistry.build(config, tower_input_dim=64)


def test_custom_task_registration():
    import torch.nn.functional as F

    @TaskRegistry.register("test_custom")
    class CustomTask(AbstractTask):
        def compute_loss(self, logits, labels, sample_weights=None, **kwargs):
            return F.mse_loss(logits.squeeze(-1), labels.float())

        def predict(self, logits):
            return logits

    assert "test_custom" in TaskRegistry.list_registered()
