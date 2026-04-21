"""
Phase 2 Should remaining tests (S2, S3, S4, S12).

Run: pytest tests/test_phase2_remaining.py -v
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest


# ---------------------------------------------------------------------------
# S2 - Feature selector Stage 3 (mandatory feature guarantee)
# ---------------------------------------------------------------------------

class TestS2FeatureSelectorStage3:
    """Exercise the Stage 3 mandatory-feature logic via select()."""

    def _make_selector(self, mandatory: List[str]):
        from core.training.feature_selector import (
            FeatureSelectionConfig, FeatureSelector,
        )
        cfg = FeatureSelectionConfig(
            cumulative_threshold=0.95,
            min_features=1,
            max_features=100,
            mandatory_features=mandatory,
        )
        return FeatureSelector(config=cfg)

    def test_stage3_restores_dropped_mandatory(self, monkeypatch):
        """When IG drops a mandatory feature, Stage 3 must add it back."""
        from core.training.feature_selector import (
            FeatureSelectionResult,
        )
        selector = self._make_selector(mandatory=["amt_regulatory_flag"])
        feat_names = [
            "spend_dining", "txn_count_monthly", "amt_regulatory_flag",
            "life_stage_retirement",
        ]

        # Fake select_by_ig -> drops the regulatory flag
        def fake_select_by_ig(model, features, task_name, feature_names=None):
            return FeatureSelectionResult(
                task_name=task_name,
                original_count=4,
                selected_count=2,
                reduction_pct=50.0,
                cumulative_threshold_used=0.95,
                selection_method="ig",
                selected_indices=[0, 1],  # mandatory (idx 2) dropped
                selected_names=["spend_dining", "txn_count_monthly"],
                feature_importances={
                    "spend_dining": 0.6, "txn_count_monthly": 0.4,
                },
                mandatory_included=[],
            )

        monkeypatch.setattr(selector, "select_by_ig", fake_select_by_ig)
        result = selector.select(
            model=None, features=None, task_name="churn",
            lgbm_model=None, feature_names=feat_names,
        )
        assert "amt_regulatory_flag" in result.selected_names
        assert "amt_regulatory_flag" in result.mandatory_included
        assert "mandatory" in result.selection_method

    def test_stage3_passthrough_when_mandatory_already_selected(
        self, monkeypatch,
    ):
        from core.training.feature_selector import FeatureSelectionResult
        selector = self._make_selector(mandatory=["spend_dining"])

        def fake_select_by_ig(model, features, task_name, feature_names=None):
            return FeatureSelectionResult(
                task_name=task_name, original_count=3,
                selected_count=1, reduction_pct=66.0,
                cumulative_threshold_used=0.95, selection_method="ig",
                selected_indices=[0],
                selected_names=["spend_dining"],
                feature_importances={"spend_dining": 1.0},
                mandatory_included=[],
            )

        monkeypatch.setattr(selector, "select_by_ig", fake_select_by_ig)
        result = selector.select(
            model=None, features=None, task_name="churn",
            lgbm_model=None,
            feature_names=["spend_dining", "other_a", "other_b"],
        )
        # Already selected — Stage 3 is a no-op
        assert result.selected_names == ["spend_dining"]
        assert result.selection_method == "ig"

    def test_stage3_warns_on_unknown_mandatory(self, monkeypatch, caplog):
        from core.training.feature_selector import FeatureSelectionResult
        selector = self._make_selector(mandatory=["missing_feature"])

        def fake_select_by_ig(model, features, task_name, feature_names=None):
            return FeatureSelectionResult(
                task_name=task_name, original_count=2, selected_count=1,
                reduction_pct=50.0, cumulative_threshold_used=0.95,
                selection_method="ig", selected_indices=[0],
                selected_names=["feat_a"],
                feature_importances={"feat_a": 1.0},
                mandatory_included=[],
            )

        monkeypatch.setattr(selector, "select_by_ig", fake_select_by_ig)
        import logging
        with caplog.at_level(logging.WARNING):
            selector.select(
                model=None, features=None, task_name="churn",
                lgbm_model=None, feature_names=["feat_a", "feat_b"],
            )
        assert any(
            "Mandatory features not in feature_names" in r.message
            for r in caplog.records
        )


# ---------------------------------------------------------------------------
# S3 - Evidential valid_mask
# ---------------------------------------------------------------------------

class TestS3EvidentialValidMask:
    def _layer(self, task_type="binary"):
        torch = pytest.importorskip("torch")
        from core.model.layers.evidential import EvidentialLayer
        return EvidentialLayer(
            input_dim=4, task_type=task_type, output_dim=3,
        )

    def test_no_mask_finite_input_produces_prediction(self):
        torch = pytest.importorskip("torch")
        layer = self._layer("binary")
        x = torch.randn(2, 4)
        pred, info = layer(x)
        assert pred.shape == (2,)
        assert torch.isfinite(pred).all()
        assert "valid_mask" in info
        assert info["valid_mask"].shape == (2,)
        assert (info["valid_mask"] == 1).all()

    def test_nan_input_is_auto_masked(self):
        torch = pytest.importorskip("torch")
        layer = self._layer("binary")
        x = torch.randn(3, 4)
        x[1, 2] = float("nan")  # row 1 becomes invalid
        pred, info = layer(x)
        assert torch.isfinite(pred).all()
        assert info["valid_mask"][0] == 1
        assert info["valid_mask"][1] == 0
        assert info["valid_mask"][2] == 1
        # Invalid row predicts neutral 0.5
        assert pred[1].item() == pytest.approx(0.5)

    def test_explicit_mask_overrides_input(self):
        torch = pytest.importorskip("torch")
        layer = self._layer("binary")
        x = torch.randn(3, 4)
        mask = torch.tensor([1.0, 0.0, 1.0])
        pred, info = layer(x, valid_mask=mask)
        assert pred[1].item() == pytest.approx(0.5)
        # Uncertainty for invalid row must be max (1.0 canonical)
        assert info["uncertainty"][1].item() == pytest.approx(1.0)

    def test_multiclass_invalid_row_neutral(self):
        torch = pytest.importorskip("torch")
        layer = self._layer("multiclass")
        x = torch.randn(2, 4)
        mask = torch.tensor([0.0, 1.0])
        pred, info = layer(x, valid_mask=mask)
        # Row 0 invalid → uniform 1/K
        assert pred[0].tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=1e-6)

    def test_regression_invalid_row_zero(self):
        torch = pytest.importorskip("torch")
        layer = self._layer("regression")
        x = torch.randn(2, 4)
        mask = torch.tensor([1.0, 0.0])
        pred, info = layer(x, valid_mask=mask)
        assert pred[1].item() == pytest.approx(0.0)

    def test_bad_mask_shape_raises(self):
        torch = pytest.importorskip("torch")
        layer = self._layer("binary")
        x = torch.randn(3, 4)
        with pytest.raises(ValueError):
            layer(x, valid_mask=torch.tensor([1.0, 0.0]))


# ---------------------------------------------------------------------------
# S4 - TemporalEnsemble HMM routing
# ---------------------------------------------------------------------------

class TestS4HMMRouting:
    def _expert(self, enabled_via_config=False):
        torch = pytest.importorskip("torch")
        from core.model.experts.temporal import TemporalEnsembleExpert
        cfg: Dict[str, Any] = {
            "output_dim": 8,
            "ensemble_gating": True,
            "mamba": {"enabled": True, "d_model": 16},
            "transformer": {"enabled": True, "d_model": 8, "patch_size": 4},
            "lnn": {"enabled": False},
        }
        if enabled_via_config:
            cfg["hmm_routing"] = {"enabled": True, "smoothing": 0.1}
        return TemporalEnsembleExpert(input_dim=4, config=cfg)

    def test_default_disabled(self):
        torch = pytest.importorskip("torch")
        expert = self._expert()
        assert expert._hmm_routing_enabled is False

    def test_set_hmm_routing_enables(self):
        torch = pytest.importorskip("torch")
        expert = self._expert()
        expert.set_hmm_routing(enabled=True, smoothing=0.1)
        assert expert._hmm_routing_enabled is True
        assert expert._hmm_transition is not None
        assert expert._hmm_transition.shape == (2, 2)
        # Rows sum to 1
        row_sums = expert._hmm_transition.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(2), atol=1e-5)

    def test_config_block_auto_enables(self):
        torch = pytest.importorskip("torch")
        expert = self._expert(enabled_via_config=True)
        assert expert._hmm_routing_enabled is True
        assert expert._hmm_transition is not None

    def test_set_hmm_routing_accepts_explicit_prior(self):
        torch = pytest.importorskip("torch")
        expert = self._expert()
        prior = [[0.8, 0.2], [0.3, 0.7]]
        expert.set_hmm_routing(enabled=True, transition_prior=prior)
        rows = expert._hmm_transition.sum(dim=-1)
        assert torch.allclose(rows, torch.ones(2), atol=1e-5)

    def test_rejects_wrong_prior_shape(self):
        torch = pytest.importorskip("torch")
        expert = self._expert()
        with pytest.raises(ValueError, match="transition_prior shape"):
            expert.set_hmm_routing(
                enabled=True, transition_prior=[[1.0, 0.0, 0.0]],
            )

    def test_disable_is_idempotent(self):
        torch = pytest.importorskip("torch")
        expert = self._expert()
        expert.set_hmm_routing(enabled=True, smoothing=0.1)
        expert.set_hmm_routing(enabled=False)
        assert expert._hmm_routing_enabled is False

    def test_apply_smoothing_preserves_shape(self):
        torch = pytest.importorskip("torch")
        expert = self._expert()
        expert.set_hmm_routing(enabled=True, smoothing=0.2)
        gw = torch.tensor([[0.7, 0.3], [0.5, 0.5]])
        out = expert._apply_hmm_smoothing(gw)
        assert out.shape == gw.shape
        # Still row-stochastic
        assert torch.allclose(out.sum(dim=-1), torch.ones(2), atol=1e-5)


# ---------------------------------------------------------------------------
# S12 - Multidisciplinary interpreter hook
# ---------------------------------------------------------------------------

class TestS12MultidisciplinaryInterpreter:
    def _assembler(self, interpreter=None):
        from core.recommendation.reason.context_assembler import (
            ContextAssembler,
        )
        return ContextAssembler(
            reverse_mapper=None,
            product_catalog={"P001": {"name": "Test Card",
                                       "category": "deposit"}},
            consultation_store={"C001": "No recent consultation."},
            segment_store={"C001": {"segment": "stable"}},
            multidisciplinary_interpreter=interpreter,
        )

    def test_no_interpreter_no_insights(self):
        from core.recommendation.reason.context_assembler import AssembledContext
        asm = self._assembler()
        ctx = asm.assemble(
            customer_id="C001", task_name="churn",
            product_id="P001",
        )
        assert isinstance(ctx, AssembledContext)
        assert ctx.multidisciplinary_insights == {}

    def test_callable_interpreter_populates_insights(self):
        def interp(ctx):
            return {
                "behavioral_economics": "Loss aversion indicated",
                "risk_mgmt": "Low risk customer",
            }
        asm = self._assembler(interpreter=interp)
        ctx = asm.assemble(
            customer_id="C001", task_name="churn",
            product_id="P001",
        )
        assert ctx.multidisciplinary_insights == {
            "behavioral_economics": "Loss aversion indicated",
            "risk_mgmt": "Low risk customer",
        }

    def test_object_interpreter_with_interpret_method(self):
        class _Interp:
            def interpret(self, ctx):
                return {"domain": "value"}
        asm = self._assembler(interpreter=_Interp())
        ctx = asm.assemble(
            customer_id="C001", task_name="churn",
            product_id="P001",
        )
        assert ctx.multidisciplinary_insights == {"domain": "value"}

    def test_interpreter_failure_is_swallowed(self):
        def boom(ctx):
            raise RuntimeError("interpreter down")
        asm = self._assembler(interpreter=boom)
        ctx = asm.assemble(
            customer_id="C001", task_name="churn",
            product_id="P001",
        )
        assert ctx.multidisciplinary_insights == {}

    def test_attach_interpreter_runtime(self):
        asm = self._assembler()
        asm.attach_interpreter(lambda c: {"d1": "v1"})
        ctx = asm.assemble(
            customer_id="C001", task_name="churn",
            product_id="P001",
        )
        assert ctx.multidisciplinary_insights["d1"] == "v1"

    def test_detach_interpreter_with_none(self):
        asm = self._assembler(interpreter=lambda c: {"d": "v"})
        asm.attach_interpreter(None)
        ctx = asm.assemble(
            customer_id="C001", task_name="churn",
            product_id="P001",
        )
        assert ctx.multidisciplinary_insights == {}

    def test_to_dict_includes_insights(self):
        def interp(ctx):
            return {"x": "y"}
        asm = self._assembler(interpreter=interp)
        ctx = asm.assemble(
            customer_id="C001", task_name="churn",
            product_id="P001",
        )
        d = ctx.to_dict()
        assert d["multidisciplinary_insights"] == {"x": "y"}
