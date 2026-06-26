"""
Tests for OCP (orthogonal-complement residual recovery) fusion and the
Expert Redundancy (CCA) analyzer core.

OCP (``fusion_type="residual_orthogonal"``, selected via
``residual_recovery.method="orthogonal"``) recovers signal from gate-down-
weighted experts but, unlike ``residual_complement`` (M1), projects out the
gated primary direction first so only the orthogonal (genuinely new)
component is added back.
"""
import numpy as np
import torch

from core.model.ple.experts import CGCLayer
from core.evaluation.expert_redundancy import ExpertRedundancyAnalyzer


def _make_layer(fusion_type: str, seed: int = 0) -> CGCLayer:
    torch.manual_seed(seed)
    return CGCLayer(
        input_dim=16,
        num_tasks=2,
        num_shared_experts=3,
        num_task_experts=1,
        expert_hidden_dim=8,
        fusion_type=fusion_type,
        native_residual_weight_init=0.5,
    )


def test_residual_orthogonal_forward_shapes_and_finite():
    """OCP fusion is a valid fusion type and produces correct, finite output."""
    layer = _make_layer("residual_orthogonal")
    layer.eval()
    x = torch.randn(32, 16)
    outs, shared_concat = layer(x, [x, x])

    assert len(outs) == 2  # one output per task
    for o in outs:
        assert o.shape == (32, 8)  # (batch, expert_hidden_dim)
        assert torch.isfinite(o).all()
    # learnable mixing scalar is created for the orthogonal method
    assert layer.residual_recovery_weight is not None


def test_residual_orthogonal_differs_from_complement():
    """OCP and M1 share the (1-gate) weighting but differ by the projection,
    so with identical init/weights they must produce different outputs."""
    x = torch.randn(32, 16)

    layer_ocp = _make_layer("residual_orthogonal", seed=7)
    layer_m1 = _make_layer("residual_complement", seed=7)
    layer_ocp.eval()
    layer_m1.eval()

    out_ocp = layer_ocp(x, [x, x])[0][0]
    out_m1 = layer_m1(x, [x, x])[0][0]

    # Same seed => identical expert/gate params; only the fusion math differs.
    assert not torch.allclose(out_ocp, out_m1, atol=1e-5)


def test_ocp_projection_is_orthogonal_to_primary():
    """The residual OCP adds is orthogonal to the gated primary direction.

    Replicates the projection formula used in CGCLayer.forward and checks
    that a (1-gate)-weighted sum of primary-projected expert outputs is
    orthogonal to the primary, per sample.
    """
    torch.manual_seed(1)
    batch, num_total, hidden = 16, 4, 8
    all_outs = torch.randn(batch, num_total, hidden)
    gate_weights = torch.softmax(torch.randn(batch, num_total), dim=-1)
    primary = (gate_weights.unsqueeze(-1) * all_outs).sum(dim=1)  # (batch, hidden)

    pp = (primary * primary).sum(dim=-1, keepdim=True)
    dot = (all_outs * primary.unsqueeze(1)).sum(dim=-1)
    coef = dot / (pp + 1e-6)
    all_perp = all_outs - coef.unsqueeze(-1) * primary.unsqueeze(1)

    complement = (1.0 - gate_weights).clamp(min=0.0)
    complement = complement / (complement.sum(dim=-1, keepdim=True) + 1e-6)
    residual = (complement.unsqueeze(-1) * all_perp).sum(dim=1)  # (batch, hidden)

    # <residual, primary> ~ 0 per sample (orthogonality of the recovered signal)
    inner = (residual * primary).sum(dim=-1)  # (batch,)
    primary_norm = primary.norm(dim=-1)
    residual_norm = residual.norm(dim=-1)
    cos = inner / (primary_norm * residual_norm + 1e-8)
    assert torch.abs(cos).max().item() < 1e-4


def test_cca_high_for_correlated_low_for_orthogonal():
    """ExpertRedundancyAnalyzer._compute_cca: ~1 for shared subspace, ~0 for
    independent representations — the property the analysis relies on."""
    rng = np.random.default_rng(0)
    n = 512
    base = rng.standard_normal((n, 6))

    # Y_corr shares the latent subspace with X (high canonical correlation).
    x = base @ rng.standard_normal((6, 8))
    y_corr = base @ rng.standard_normal((6, 8)) + 0.01 * rng.standard_normal((n, 8))
    # Y_indep is drawn independently (low canonical correlation).
    y_indep = rng.standard_normal((n, 8))

    corr_high = ExpertRedundancyAnalyzer._compute_cca(x, y_corr, n_components=5)
    corr_low = ExpertRedundancyAnalyzer._compute_cca(x, y_indep, n_components=5)

    assert corr_high is not None and corr_low is not None
    assert float(np.mean(corr_high)) > 0.9
    assert float(np.mean(corr_low)) < 0.5
