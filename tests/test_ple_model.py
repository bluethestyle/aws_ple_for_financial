import torch
import pytest

from core.model.ple import PLEModel, PLEConfig
from core.task.base import TaskConfig
from core.task.registry import TaskRegistry
from core.task.types import TaskType, LossType


def build_tasks(tower_input_dim: int):
    return [
        TaskRegistry.build(
            TaskConfig(name="click",   task_type=TaskType.BINARY,     loss_type=LossType.BCE),
            tower_input_dim=tower_input_dim,
        ),
        TaskRegistry.build(
            TaskConfig(name="convert", task_type=TaskType.REGRESSION,  loss_type=LossType.MSE),
            tower_input_dim=tower_input_dim,
        ),
    ]


def test_ple_forward_no_labels():
    config = PLEConfig(input_dim=32, num_tasks=2, expert_hidden_dim=64, num_layers=2)
    tower_dim = config.expert_hidden_dim
    tasks = build_tasks(tower_dim)
    model = PLEModel(config, tasks)

    x = torch.randn(16, 32)
    outputs = model(x)

    assert set(outputs.keys()) == {"click", "convert"}
    assert outputs["click"].logits.shape == (16, 1)
    assert outputs["click"].loss is None


def test_ple_forward_with_labels():
    config = PLEConfig(input_dim=32, num_tasks=2, expert_hidden_dim=64, num_layers=2)
    tasks = build_tasks(config.expert_hidden_dim)
    model = PLEModel(config, tasks)

    x = torch.randn(16, 32)
    labels = {
        "click":   torch.randint(0, 2, (16,)).float(),
        "convert": torch.randn(16),
    }
    outputs = model(x, labels)

    assert outputs["click"].loss is not None
    assert outputs["convert"].loss is not None


def test_ple_total_loss():
    config = PLEConfig(input_dim=32, num_tasks=2, expert_hidden_dim=64, num_layers=2)
    tasks = build_tasks(config.expert_hidden_dim)
    model = PLEModel(config, tasks)

    x = torch.randn(16, 32)
    labels = {
        "click":   torch.randint(0, 2, (16,)).float(),
        "convert": torch.randn(16),
    }
    outputs = model(x, labels)
    total_loss = model.compute_total_loss(outputs)

    assert total_loss.item() > 0
    assert total_loss.requires_grad


def test_ple_single_task():
    config = PLEConfig(input_dim=16, num_tasks=1, expert_hidden_dim=32, num_layers=1)
    tasks = [TaskRegistry.build(
        TaskConfig(name="rating", task_type=TaskType.REGRESSION, loss_type=LossType.MSE),
        tower_input_dim=config.expert_hidden_dim,
    )]
    model = PLEModel(config, tasks)
    x = torch.randn(8, 16)
    outputs = model(x)
    assert "rating" in outputs
