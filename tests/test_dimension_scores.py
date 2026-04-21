"""
Tests for dimension-score providers (plug into PromotionGate).

Run: pytest tests/test_dimension_scores.py -v
"""

from __future__ import annotations

import pytest

from core.compliance.ai_risk_classifier import AIRiskClassifier, AIRiskConfig
from core.compliance.dimension_scores import (
    CompositeProvider,
    DefaultScoreProvider,
    HEURISTIC_RULES,
    HeuristicRule,
    ManualScoreProvider,
    MetricsDerivedScoreProvider,
    _walk,
)
from core.compliance.fria_assessment import (
    FRIA_DIMENSIONS,
    FRIAConfig,
    KoreanFRIAAssessor,
)
from core.compliance.store import InMemoryComplianceStore
from core.evaluation.promotion_gate import PromotionGate


# ---------------------------------------------------------------------------
# _walk helper
# ---------------------------------------------------------------------------

class TestWalk:
    def test_missing_key_returns_none(self):
        assert _walk({}, "a") is None

    def test_nested_path(self):
        assert _walk({"a": {"b": {"c": 42}}}, "a.b.c") == 42

    def test_path_leads_to_non_mapping(self):
        assert _walk({"a": 5}, "a.b") is None


# ---------------------------------------------------------------------------
# ManualScoreProvider
# ---------------------------------------------------------------------------

class TestManualScoreProvider:
    def test_returns_operator_scores(self):
        p = ManualScoreProvider({
            "v1": {"data_sensitivity": 0.7, "fairness_risk": 0.3},
        })
        scores = p("v1")
        assert scores["data_sensitivity"] == 0.7
        assert scores["fairness_risk"] == 0.3

    def test_unknown_model_returns_empty_when_no_dims(self):
        p = ManualScoreProvider({"v1": {"a": 0.1}})
        assert p("v2") == {}

    def test_unknown_model_returns_empty_even_with_dims(self):
        # Unknown models must fall through to the next layer in a
        # CompositeProvider; per dimension defaults only apply when the
        # operator actually registered scores for this model version.
        p = ManualScoreProvider(
            {"v1": {"a": 0.1}},
            default_score=0.5,
            dimensions=("a", "b"),
        )
        assert p("unknown") == {}

    def test_missing_dim_uses_default(self):
        p = ManualScoreProvider(
            {"v1": {"a": 0.9}},
            default_score=0.2,
            dimensions=("a", "b"),
        )
        scores = p("v1")
        assert scores == {"a": 0.9, "b": 0.2}


# ---------------------------------------------------------------------------
# DefaultScoreProvider
# ---------------------------------------------------------------------------

class TestDefaultScoreProvider:
    def test_emits_scalar_for_all_dims(self):
        p = DefaultScoreProvider(value=0.4, dimensions=("a", "b", "c"))
        assert p("v1") == {"a": 0.4, "b": 0.4, "c": 0.4}

    def test_no_dimensions_returns_empty(self):
        p = DefaultScoreProvider(value=0.5)
        assert p("v1") == {}

    def test_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            DefaultScoreProvider(value=1.5)


# ---------------------------------------------------------------------------
# HeuristicRule
# ---------------------------------------------------------------------------

class TestHeuristicRule:
    def test_identity(self):
        rule = HeuristicRule(path="x", transform="identity", fallback=0.5)
        assert rule.apply({"x": 0.8}) == pytest.approx(0.8)

    def test_one_minus(self):
        rule = HeuristicRule(path="x", transform="one_minus", fallback=0.5)
        assert rule.apply({"x": 0.2}) == pytest.approx(0.8)

    def test_log10_ratio_scales(self):
        rule = HeuristicRule(
            path="x", transform="log10_ratio",
            fallback=0.5, ratio_denominator=1_000_000,
        )
        # 1M → ratio = 1.0
        assert rule.apply({"x": 1_000_000}) == pytest.approx(1.0, rel=0.05)
        # 1 → near 0
        assert rule.apply({"x": 1}) < 0.1

    def test_fallback_on_missing(self):
        rule = HeuristicRule(path="y", transform="identity", fallback=0.3)
        assert rule.apply({"x": 0.8}) == 0.3

    def test_unknown_transform_falls_back(self):
        rule = HeuristicRule(
            path="x", transform="bogus", fallback=0.42,
        )
        assert rule.apply({"x": 0.8}) == 0.42

    def test_clips_to_unit_interval(self):
        rule = HeuristicRule(path="x", transform="identity", fallback=0.5)
        assert rule.apply({"x": 1.5}) == 1.0
        assert rule.apply({"x": -0.2}) == 0.0


# ---------------------------------------------------------------------------
# MetricsDerivedScoreProvider
# ---------------------------------------------------------------------------

class TestMetricsDerivedScoreProvider:
    def test_heuristic_defaults_produce_all_fria_dims(self):
        # Empty metadata → every dim falls back to 0.5
        p = MetricsDerivedScoreProvider(lambda _v: {})
        scores = p("v1")
        for dim in HEURISTIC_RULES:
            assert scores[dim] == pytest.approx(0.5)

    def test_metadata_drives_scores(self):
        meta = {
            "pii_ratio": 0.8,
            "human_review_fraction": 0.2,
            "customer_count": 1_000_000,
            "param_count": 10_000_000,
            "llm_provider_ratio": 0.6,
            "disparate_impact_min": 0.4,
            "reason_coverage": 0.7,
        }
        p = MetricsDerivedScoreProvider(lambda _v: meta)
        s = p("v1")
        assert s["data_sensitivity"] == pytest.approx(0.8)
        assert s["automation_level"] == pytest.approx(0.8)   # 1 - 0.2
        assert s["scope_of_impact"] == pytest.approx(1.0, rel=0.05)
        assert s["fairness_risk"] == pytest.approx(0.6)      # 1 - 0.4
        assert s["explainability_gap"] == pytest.approx(0.3) # 1 - 0.7

    def test_lookup_failure_is_swallowed(self):
        def boom(_v):
            raise RuntimeError("metadata DB down")
        p = MetricsDerivedScoreProvider(boom)
        scores = p("v1")
        # Every dim falls back to per-rule default (0.5)
        assert all(v == 0.5 for v in scores.values())

    def test_restricted_dimensions_only(self):
        p = MetricsDerivedScoreProvider(
            lambda _v: {},
            dimensions=("data_sensitivity", "fairness_risk"),
        )
        s = p("v1")
        assert set(s.keys()) == {"data_sensitivity", "fairness_risk"}


# ---------------------------------------------------------------------------
# CompositeProvider
# ---------------------------------------------------------------------------

class TestCompositeProvider:
    def test_first_provider_wins(self):
        manual = ManualScoreProvider({
            "v1": {"data_sensitivity": 0.9},
        })
        default = DefaultScoreProvider(
            value=0.1, dimensions=("data_sensitivity", "automation_level"),
        )
        comp = CompositeProvider([manual, default])
        scores = comp("v1")
        # manual wins on data_sensitivity; default covers automation_level
        assert scores["data_sensitivity"] == 0.9
        assert scores["automation_level"] == 0.1

    def test_failing_provider_is_skipped(self):
        class _Bad:
            def __call__(self, _v):
                raise RuntimeError("down")
        good = DefaultScoreProvider(
            value=0.5, dimensions=("data_sensitivity",),
        )
        comp = CompositeProvider([_Bad(), good])
        s = comp("v1")
        assert s == {"data_sensitivity": 0.5}

    def test_rejects_empty_list(self):
        with pytest.raises(ValueError):
            CompositeProvider([])


# ---------------------------------------------------------------------------
# Integration with PromotionGate
# ---------------------------------------------------------------------------

class TestPromotionGateIntegration:
    def _gate(self, fria_provider=None, ai_provider=None):
        store = InMemoryComplianceStore()
        return PromotionGate(
            fria_assessor=KoreanFRIAAssessor(store=store),
            ai_risk_classifier=AIRiskClassifier(store=store),
            enabled=True,
            fria_scores_provider=fria_provider,
            ai_risk_scores_provider=ai_provider,
        )

    def test_manual_provider_drives_unacceptable(self):
        provider = ManualScoreProvider(
            {"v_bad": {d: 0.95 for d in FRIA_DIMENSIONS}},
            dimensions=FRIA_DIMENSIONS,
        )
        gate = self._gate(fria_provider=provider)
        verdict = gate.evaluate(model_version="v_bad")
        assert verdict.decision == "reject"
        assert verdict.fria.risk_category == "UNACCEPTABLE"

    def test_metadata_driven_pass(self):
        def lookup(_v):
            # Low-risk metadata → low scores for every dim
            return {
                "pii_ratio": 0.1,
                "human_review_fraction": 0.9,  # automation_level low
                "customer_count": 10,
                "param_count": 1000,
                "llm_provider_ratio": 0.0,
                "disparate_impact_min": 0.95,
                "reason_coverage": 0.95,
            }
        provider = MetricsDerivedScoreProvider(
            lookup, dimensions=FRIA_DIMENSIONS,
        )
        gate = self._gate(fria_provider=provider)
        verdict = gate.evaluate(model_version="v_safe")
        assert verdict.decision == "pass"
        assert verdict.fria.risk_category == "MINIMAL"

    def test_composite_layer_manual_over_derived(self):
        # metadata says LIMITED, manual says UNACCEPTABLE → manual wins
        def lookup(_v):
            return {
                "pii_ratio": 0.5, "human_review_fraction": 0.5,
                "customer_count": 10, "param_count": 10_000,
                "llm_provider_ratio": 0.3, "disparate_impact_min": 0.8,
                "reason_coverage": 0.8,
            }
        manual = ManualScoreProvider(
            {"v_override": {d: 0.95 for d in FRIA_DIMENSIONS}},
            dimensions=FRIA_DIMENSIONS,
        )
        derived = MetricsDerivedScoreProvider(
            lookup, dimensions=FRIA_DIMENSIONS,
        )
        comp = CompositeProvider([manual, derived])
        gate = self._gate(fria_provider=comp)
        verdict = gate.evaluate(model_version="v_override")
        assert verdict.decision == "reject"
        assert verdict.fria.risk_category == "UNACCEPTABLE"

    def test_composite_falls_through_to_derived(self):
        def lookup(_v):
            # Metadata that resolves to ~0.3 for every heuristic ->
            # with equal weights total ~0.3 → MINIMAL (< 0.40 threshold)
            return {
                "pii_ratio": 0.3,
                "human_review_fraction": 0.7,     # 1 - 0.7 = 0.3
                "customer_count": 10,             # log10_ratio ~ 0
                "param_count": 100,               # log10_ratio ~ 0
                "llm_provider_ratio": 0.3,
                "disparate_impact_min": 0.7,      # 1 - 0.7 = 0.3
                "reason_coverage": 0.7,           # 1 - 0.7 = 0.3
            }
        manual = ManualScoreProvider(
            {"v_other": {"data_sensitivity": 0.9}},
            dimensions=FRIA_DIMENSIONS,
        )
        derived = MetricsDerivedScoreProvider(
            lookup, dimensions=FRIA_DIMENSIONS,
        )
        comp = CompositeProvider([manual, derived])
        gate = self._gate(fria_provider=comp)
        # v_fall is unknown to manual → derived fills all dims → MINIMAL
        verdict = gate.evaluate(model_version="v_fall")
        assert verdict.decision == "pass"
        assert verdict.fria.risk_category == "MINIMAL"
