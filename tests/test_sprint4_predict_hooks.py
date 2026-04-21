"""
Sprint 4 integration tests: verify that predict.py invokes the Sprint 1~3
hooks (opt-out / profiling / explanation SLA / AI risk / review queue /
marker applier / item universe) with the documented behaviour.

The tests exercise each helper method directly rather than running a full
predict() end-to-end, since the existing hot path already has its own
coverage. This keeps the surface under test tight and fast.

Run: pytest tests/test_sprint4_predict_hooks.py -v
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, List

import pytest

from core.compliance.ai_risk_classifier import AIRiskClassifier
from core.compliance.fria_assessment import FRIA_DIMENSIONS
from core.compliance.rights.explanation_sla import ExplanationSLATracker
from core.compliance.rights.opt_out import OptOutManager
from core.compliance.rights.profiling import ProfilingWorkflow
from core.compliance.store import InMemoryComplianceStore
from core.compliance.types import RequestType, utcnow
from core.recommendation.reason.marker_applier import (
    DEFAULT_MARKER_TEXT,
    MarkerApplier,
)
from core.recommendation.universe.dynamic_loader import (
    DynamicItemUniverseLoader,
    Item,
    ItemUniverseConfig,
)
from core.serving.feature_store import AbstractFeatureStore
from core.serving.predict import RecommendationService
from core.serving.review.human_review_queue import HumanReviewQueue


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

class _FakeModel:
    """Minimal model stand-in; predict() not called in these tests."""
    def predict(self, feature_df):
        return {"task_a": [0.5]}


class _InMemoryFeatureStore(AbstractFeatureStore):
    """Testing-only feature store with no S3 dependency."""

    def __init__(self, data=None):
        self._data = dict(data or {})

    def get(self, user_id):
        return self._data.get(user_id)

    def get_batch(self, user_ids):
        return {u: self._data[u] for u in user_ids if u in self._data}

    def health_check(self):
        return {"healthy": True, "users": len(self._data)}


def _make_service(**hooks) -> RecommendationService:
    return RecommendationService(
        model=_FakeModel(),
        feature_store=_InMemoryFeatureStore(),
        tasks_meta=[{"name": "task_a", "type": "binary"}],
        **hooks,
    )


# ---------------------------------------------------------------------------
# Explanation SLA annotation (M6)
# ---------------------------------------------------------------------------

class TestExplanationSLAAnnotation:
    def test_no_tracker_no_annotation(self):
        service = _make_service()
        meta: Dict[str, Any] = {}
        service._annotate_explanation_sla("u1", meta)
        assert meta == {}

    def test_pending_surfaced(self):
        store = InMemoryComplianceStore()
        tracker = ExplanationSLATracker(store=store)
        tracker.open_request(
            user_id="u1", request_type=RequestType.EXPLANATION,
        )
        service = _make_service(explanation_sla_tracker=tracker)
        meta: Dict[str, Any] = {}
        service._annotate_explanation_sla("u1", meta)
        assert meta["pending_explanations"] == 1
        assert "explanation_sla_breaches" not in meta

    def test_breach_surfaced(self):
        store = InMemoryComplianceStore()
        tracker = ExplanationSLATracker(store=store)
        submitted = utcnow() - timedelta(days=11)
        tracker.open_request(
            user_id="u1", request_type=RequestType.EXPLANATION,
            submitted_at=submitted,
        )
        service = _make_service(explanation_sla_tracker=tracker)
        meta: Dict[str, Any] = {}
        service._annotate_explanation_sla("u1", meta)
        assert meta["pending_explanations"] == 1
        assert meta["explanation_sla_breaches"] == 1

    def test_failure_is_swallowed(self):
        class _Bad:
            def list_pending(self, **_):
                raise RuntimeError("DB down")
        service = _make_service(explanation_sla_tracker=_Bad())
        meta: Dict[str, Any] = {}
        service._annotate_explanation_sla("u1", meta)  # must not raise
        assert meta == {}


# ---------------------------------------------------------------------------
# Profiling rights annotation (M5)
# ---------------------------------------------------------------------------

class TestProfilingRightsAnnotation:
    def test_pending_surfaced(self):
        store = InMemoryComplianceStore()
        wf = ProfilingWorkflow(store=store)
        wf.request_access(user_id="u1")
        wf.request_correction(user_id="u1", field_name="age", new_value=30)
        service = _make_service(profiling_workflow=wf)
        meta: Dict[str, Any] = {}
        service._annotate_profiling_rights("u1", meta)
        assert meta["pending_profiling_requests"] == 2
        assert set(meta["pending_profiling_types"]) == {
            RequestType.PROFILING_ACCESS,
            RequestType.PROFILING_CORRECTION,
        }

    def test_no_pending_no_annotation(self):
        store = InMemoryComplianceStore()
        wf = ProfilingWorkflow(store=store)
        service = _make_service(profiling_workflow=wf)
        meta: Dict[str, Any] = {}
        service._annotate_profiling_rights("u1", meta)
        assert meta == {}


# ---------------------------------------------------------------------------
# AI Risk grade annotation (M9)
# ---------------------------------------------------------------------------

class TestAIRiskAnnotation:
    def test_requires_model_version_in_metadata(self):
        store = InMemoryComplianceStore()
        clf = AIRiskClassifier(store=store)
        clf.classify(
            model_version="v1",
            dimension_scores={
                "data_sensitivity": 0.8, "automation_level": 0.8,
                "scope_of_impact": 0.8, "model_complexity": 0.8,
                "external_dependency": 0.8, "fairness_risk": 0.8,
            },
        )
        service = _make_service(ai_risk_classifier=clf)
        meta: Dict[str, Any] = {}
        service._annotate_ai_risk_grade(meta)
        # no model_version → no annotation
        assert "ai_risk_grade" not in meta

    def test_annotates_when_model_version_present(self):
        store = InMemoryComplianceStore()
        clf = AIRiskClassifier(store=store)
        clf.classify(
            model_version="v1",
            dimension_scores={
                "data_sensitivity": 0.8, "automation_level": 0.8,
                "scope_of_impact": 0.8, "model_complexity": 0.8,
                "external_dependency": 0.8, "fairness_risk": 0.8,
            },
        )
        service = _make_service(ai_risk_classifier=clf)
        meta = {"model_version": "v1"}
        service._annotate_ai_risk_grade(meta)
        assert meta["ai_risk_grade"] == "high"
        assert meta["ai_risk_score"] > 0.7

    def test_grade_change_flag(self):
        store = InMemoryComplianceStore()
        clf = AIRiskClassifier(store=store)
        clf.classify(
            model_version="v1",
            dimension_scores={
                "data_sensitivity": 0.2, "automation_level": 0.2,
                "scope_of_impact": 0.2, "model_complexity": 0.2,
                "external_dependency": 0.2, "fairness_risk": 0.2,
            },
        )
        clf.classify(
            model_version="v1",
            dimension_scores={
                "data_sensitivity": 0.8, "automation_level": 0.8,
                "scope_of_impact": 0.8, "model_complexity": 0.8,
                "external_dependency": 0.8, "fairness_risk": 0.8,
            },
        )
        service = _make_service(ai_risk_classifier=clf)
        meta = {"model_version": "v1"}
        service._annotate_ai_risk_grade(meta)
        assert meta["ai_risk_grade_change"] is True


# ---------------------------------------------------------------------------
# Human review triage (M1)
# ---------------------------------------------------------------------------

class TestReviewTriage:
    def test_tier_2_enqueues(self):
        q = HumanReviewQueue(seed=0)
        service = _make_service(review_queue=q)
        recs = [{"recommendation_id": "rec_1", "product_id": "p1"}]
        meta: Dict[str, Any] = {}
        service._triage_for_review(
            "u1", recs, {"agent_tier": 2}, meta,
        )
        assert "review_id" in meta
        assert meta["review_tier"] == 2

    def test_no_recs_no_enqueue(self):
        q = HumanReviewQueue(seed=0)
        service = _make_service(review_queue=q)
        meta: Dict[str, Any] = {}
        service._triage_for_review("u1", [], {"agent_tier": 2}, meta)
        assert "review_id" not in meta

    def test_invalid_tier_silently_skipped(self):
        q = HumanReviewQueue(seed=0)
        service = _make_service(review_queue=q)
        recs = [{"recommendation_id": "rec_1"}]
        meta: Dict[str, Any] = {}
        service._triage_for_review(
            "u1", recs, {"agent_tier": "not_a_tier"}, meta,
        )
        assert meta == {}

    def test_no_queue_is_noop(self):
        service = _make_service()
        recs = [{"recommendation_id": "rec_1"}]
        meta: Dict[str, Any] = {}
        service._triage_for_review(
            "u1", recs, {"agent_tier": 2}, meta,
        )
        assert meta == {}


# ---------------------------------------------------------------------------
# LLM marker application (M12)
# ---------------------------------------------------------------------------

class TestMarkerApplication:
    def test_marker_applied_to_reason_text(self):
        applier = MarkerApplier()
        service = _make_service(marker_applier=applier)
        recs: List[Dict[str, Any]] = [
            {"reason_text": "이 상품을 추천합니다."},
            {"reason_text": "또 다른 이유입니다."},
        ]
        service._apply_llm_marker_to_recommendations(recs)
        for r in recs:
            assert DEFAULT_MARKER_TEXT in r["reason_text"]

    def test_skips_entries_without_reason_text(self):
        applier = MarkerApplier()
        service = _make_service(marker_applier=applier)
        recs = [{"product_id": "p1"}]  # no reason_text
        service._apply_llm_marker_to_recommendations(recs)
        assert "reason_text" not in recs[0]

    def test_idempotent_on_already_marked(self):
        applier = MarkerApplier()
        service = _make_service(marker_applier=applier)
        once_applied = applier.apply("hello")
        recs = [{"reason_text": once_applied}]
        service._apply_llm_marker_to_recommendations(recs)
        # Count of marker substring should stay at 1
        assert recs[0]["reason_text"].count("AI기본법") == 1

    def test_no_applier_is_noop(self):
        service = _make_service()
        recs = [{"reason_text": "unchanged"}]
        service._apply_llm_marker_to_recommendations(recs)
        assert recs[0]["reason_text"] == "unchanged"


# ---------------------------------------------------------------------------
# Hook wiring smoke: __init__ does not explode
# ---------------------------------------------------------------------------

class TestServiceConstruction:
    def test_all_sprint_hooks_accepted(self):
        store = InMemoryComplianceStore()
        service = _make_service(
            opt_out_manager=OptOutManager(store=store),
            profiling_workflow=ProfilingWorkflow(store=store),
            explanation_sla_tracker=ExplanationSLATracker(store=store),
            ai_risk_classifier=AIRiskClassifier(store=store),
            review_queue=HumanReviewQueue(seed=0),
            item_universe_loader=DynamicItemUniverseLoader(
                config=ItemUniverseConfig(enabled=False),
            ),
            marker_applier=MarkerApplier(),
        )
        # All hooks stored
        assert service._opt_out_manager is not None
        assert service._profiling_workflow is not None
        assert service._explanation_sla_tracker is not None
        assert service._ai_risk_classifier is not None
        assert service._review_queue is not None
        assert service._item_universe_loader is not None
        assert service._marker_applier is not None

    def test_legacy_construction_still_works(self):
        # Zero compliance hooks = strictly the pre-Sprint-4 behavior.
        service = _make_service()
        assert service._opt_out_manager is None
        assert service._marker_applier is None
