"""Tests for the Causal Counterfactual Probe (Paper 3 Axis-3 E).

Covers the ``CausalExpert.get_counterfactual`` method:
  - output keys + shapes + detachedness
  - factual reproduces the vanilla forward pass
  - direct_only == full_cf when W == 0 (decorative-DAG limit)
  - direct_only != full_cf when W carries non-trivial structure
  - out-of-range feature_idx raises
"""
from __future__ import annotations

import pytest
import torch

from core.model.experts.causal import CausalExpert


def _expert(w_init_scale: float = 0.1) -> CausalExpert:
    return CausalExpert(
        input_dim=32,
        config={
            "output_dim": 16,
            "hidden_dim": 32,
            "n_causal_vars": 8,
            "w_init_scale": w_init_scale,
            "ceh": {"enabled": False},
        },
    )


def test_counterfactual_shapes_and_detached():
    torch.manual_seed(0)
    expert = _expert()
    expert.eval()
    x = torch.randn(5, 32)
    out = expert.get_counterfactual(x, feature_idx=3, intervention_value=1.5)
    assert set(out.keys()) == {"factual", "direct_only", "full_cf"}
    for key in out:
        assert out[key].shape == (5, 16)
        assert out[key].grad_fn is None


def test_factual_matches_vanilla_forward():
    torch.manual_seed(1)
    expert = _expert()
    expert.eval()
    x = torch.randn(7, 32)
    factual = expert.get_counterfactual(
        x, feature_idx=0, intervention_value=0.0
    )["factual"]
    with torch.no_grad():
        vanilla = expert(x)
    assert torch.allclose(factual, vanilla, atol=1e-6)


def test_zero_W_collapses_direct_and_full_cf():
    """When W = 0, the SCM residual z @ W^2 is zero and the full CF is
    identical to the direct-only branch."""
    torch.manual_seed(2)
    expert = _expert()
    expert.eval()
    with torch.no_grad():
        expert.W.zero_()
    x = torch.randn(4, 32)
    out = expert.get_counterfactual(x, feature_idx=2, intervention_value=3.0)
    assert torch.allclose(out["direct_only"], out["full_cf"], atol=1e-6)


def test_large_W_separates_direct_and_full_cf():
    """With a non-trivial W, mediation makes full_cf differ from
    direct_only. We use a deliberately oversized W to make the gap
    numerically detectable under untrained weights."""
    torch.manual_seed(3)
    expert = _expert(w_init_scale=1.0)
    expert.eval()
    x = torch.randn(4, 32)
    out = expert.get_counterfactual(x, feature_idx=5, intervention_value=2.0)
    diff = (out["full_cf"] - out["direct_only"]).norm().item()
    assert diff > 1e-3


def test_feature_idx_out_of_range_raises():
    expert = _expert()
    with pytest.raises(ValueError):
        expert.get_counterfactual(
            torch.randn(1, 32), feature_idx=99, intervention_value=0.0
        )
