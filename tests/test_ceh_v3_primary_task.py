"""Tests for CEH v3 primary-task-gradient target mode (Paper 3 Finding 13).

The task-logit gradient target is computed externally by
``PLEModel._inject_ceh_v3_target`` (because it needs access to the
task towers); the causal expert exposes a setter and a new target
mode but does not run the task path itself. These tests isolate the
causal-expert side of the contract.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from core.model.experts.causal import CausalExpert


def _v3_expert(input_dim: int = 32) -> CausalExpert:
    return CausalExpert(
        input_dim=input_dim,
        config={
            "output_dim": 16,
            "hidden_dim": 32,
            "n_causal_vars": 8,
            "ceh": {
                "enabled": True,
                "hidden_dim": 16,
                "target_mode": "primary_task",
                "primary_task_name": "churn_signal",
            },
        },
    )


def test_primary_task_mode_skips_internal_target_computation():
    """In primary_task mode, forward MUST NOT populate attr_target."""
    torch.manual_seed(0)
    expert = _v3_expert()
    expert.train()
    x = torch.randn(6, 32)
    _ = expert(x)
    assert expert._last_attribution is not None
    assert expert._last_attr_target is None, \
        "primary_task mode must not compute target internally"


def test_set_attr_target_external_populates_cache():
    torch.manual_seed(1)
    expert = _v3_expert()
    expert.train()
    x = torch.randn(4, 32)
    _ = expert(x)
    injected = torch.randn(4, 32)
    expert.set_attr_target_external(injected)
    assert expert._last_attr_target is not None
    assert torch.allclose(expert._last_attr_target, injected)
    # Detached
    assert expert._last_attr_target.grad_fn is None


def test_get_attribution_loss_uses_injected_target():
    torch.manual_seed(2)
    expert = _v3_expert()
    expert.train()
    x = torch.randn(5, 32)
    _ = expert(x)
    target = torch.randn(5, 32)
    expert.set_attr_target_external(target)
    loss = expert.get_attribution_loss()
    # Should match the expert's internal MSE computation
    expected = F.mse_loss(expert._last_attribution.float(), target.float())
    assert torch.allclose(loss, expected, atol=1e-6)


def test_set_attr_target_external_rejects_wrong_shape():
    expert = _v3_expert()
    expert.train()
    _ = expert(torch.randn(3, 32))
    with pytest.raises(ValueError):
        expert.set_attr_target_external(torch.randn(3, 99))  # wrong last dim


def test_unknown_target_mode_raises():
    with pytest.raises(ValueError):
        CausalExpert(
            input_dim=16,
            config={
                "output_dim": 8, "hidden_dim": 16, "n_causal_vars": 4,
                "ceh": {"enabled": True, "target_mode": "not_a_mode"},
            },
        )


def test_zero_loss_when_target_not_injected():
    """If PLE loop fails to inject the target, loss must be zero not NaN."""
    expert = _v3_expert()
    expert.train()
    _ = expert(torch.randn(3, 32))
    # External injection NOT called: target stays None
    loss = expert.get_attribution_loss()
    assert torch.isfinite(loss)
    assert float(loss) == 0.0
