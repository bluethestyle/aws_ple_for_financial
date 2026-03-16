"""Tests for the PLEModel using the current API (PLEConfig, PLEInput, PLEOutput)."""

import torch
import pytest

from core.model.ple import PLEModel, PLEConfig, PLEInput, PLEOutput
from core.model.ple.config import (
    ExpertConfig,
    AdaTTConfig,
    CGCConfig,
    LossWeightingConfig,
    TaskTowerConfig,
)


def _make_config(**overrides) -> PLEConfig:
    """Build a minimal PLEConfig suitable for unit tests."""
    defaults = dict(
        input_dim=32,
        task_names=["click", "convert"],
        num_shared_experts=2,
        num_task_experts_per_task=1,
        num_extraction_layers=1,
        task_expert_output_dim=16,
        shared_expert=ExpertConfig(hidden_dims=[32], output_dim=32, dropout=0.0),
        task_expert=ExpertConfig(hidden_dims=[32], output_dim=16, dropout=0.0),
        task_tower=TaskTowerConfig(hidden_dims=[16], dropout=0.0),
        cgc=CGCConfig(enabled=False),
        adatt=AdaTTConfig(enabled=False),
        loss_weighting=LossWeightingConfig(strategy="fixed"),
        dropout=0.0,
        task_overrides={
            "click":   {"task_type": "binary",     "output_dim": 1, "activation": "sigmoid"},
            "convert": {"task_type": "regression",  "output_dim": 1, "activation": None},
        },
    )
    defaults.update(overrides)
    return PLEConfig(**defaults)


def test_ple_forward_no_labels():
    """Forward pass without labels should produce predictions but no loss."""
    config = _make_config()
    model = PLEModel(config)

    x = torch.randn(16, config.input_dim)
    inputs = PLEInput(features=x)
    output = model(inputs, compute_loss=False)

    assert isinstance(output, PLEOutput)
    assert set(output.predictions.keys()) == {"click", "convert"}
    assert output.predictions["click"].shape[0] == 16
    assert output.predictions["convert"].shape[0] == 16
    assert output.total_loss is None


def test_ple_forward_with_labels():
    """Forward pass with labels should produce per-task losses."""
    config = _make_config()
    model = PLEModel(config)

    x = torch.randn(16, config.input_dim)
    targets = {
        "click":   torch.randint(0, 2, (16,)).float(),
        "convert": torch.randn(16),
    }
    inputs = PLEInput(features=x, targets=targets)
    output = model(inputs)

    assert output.total_loss is not None
    assert output.task_losses is not None
    assert "click" in output.task_losses
    assert "convert" in output.task_losses


def test_ple_total_loss():
    """Total loss should be a positive scalar that supports backpropagation."""
    config = _make_config()
    model = PLEModel(config)

    x = torch.randn(16, config.input_dim)
    targets = {
        "click":   torch.randint(0, 2, (16,)).float(),
        "convert": torch.randn(16),
    }
    inputs = PLEInput(features=x, targets=targets)
    output = model(inputs)

    total_loss = output.total_loss
    assert total_loss.item() > 0
    assert total_loss.requires_grad


def test_ple_single_task():
    """Model should work with a single task."""
    config = _make_config(
        input_dim=16,
        task_names=["rating"],
        task_overrides={
            "rating": {"task_type": "regression", "output_dim": 1, "activation": None},
        },
    )
    model = PLEModel(config)

    x = torch.randn(8, 16)
    inputs = PLEInput(features=x)
    output = model(inputs, compute_loss=False)

    assert "rating" in output.predictions
    assert output.predictions["rating"].shape[0] == 8
    assert output.total_loss is None
