"""Tests for the regulatory serving-hook builder (core/serving/hook_builder.py).

Verifies that the single hook builder constructs every Sprint 1~4 hook
(M1/M4/M5/M6/M9/M10/M12) from a pipeline.yaml-shaped config, honours the
enable gate and the fail-closed (strict) posture, and that the produced dict
splats cleanly into RecommendationService so the hooks actually reach the
serving object (closing the "hooks only ran in tests" wiring gap).

Run: pytest tests/test_serving_hook_builder.py -v
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from core.serving.hook_builder import (
    HOOK_KEYS,
    build_compliance_hooks,
    hooks_enabled,
)
from core.serving.feature_store import AbstractFeatureStore
from core.serving.predict import RecommendationService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _FakeModel:
    def predict(self, feature_df):
        return {"task_a": [0.5]}


class _InMemoryFeatureStore(AbstractFeatureStore):
    def __init__(self):
        self._data: Dict[str, Any] = {}

    def get(self, user_id):
        return self._data.get(user_id)

    def get_batch(self, user_ids):
        return {u: self._data[u] for u in user_ids if u in self._data}

    def health_check(self):
        return {"healthy": True, "users": 0}


def _full_config() -> Dict[str, Any]:
    """A pipeline.yaml-shaped config carrying every consumed block."""
    return {
        "aws": {"region": "ap-northeast-2"},
        "compliance": {
            "store": {"backend": "in_memory"},
            "sla": {"explanation_response_days": 10},
            "opt_out": {"default_fallback": "rule_based"},
            "profiling": {},
            "ai_risk": {
                "dimensions": {"data_sensitivity": 1.0},
                "grade_thresholds": {"high": 0.7, "medium": 0.4},
            },
            "llm_marker": {
                "enabled": True,
                "marker_text": "본 사유는 AI가 생성하였습니다. (AI기본법 §31)",
            },
        },
        "serving": {
            "review": {"queue_backend": "in_memory"},
            "item_universe": {},
        },
    }


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestBuildsAllHooks:
    def test_all_seven_hooks_built(self):
        hooks = build_compliance_hooks(_full_config())
        assert set(hooks.keys()) == set(HOOK_KEYS)

    def test_hook_types(self):
        from core.compliance.ai_risk_classifier import AIRiskClassifier
        from core.compliance.rights.explanation_sla import ExplanationSLATracker
        from core.compliance.rights.opt_out import OptOutManager
        from core.compliance.rights.profiling import ProfilingWorkflow
        from core.recommendation.reason.marker_applier import MarkerApplier
        from core.recommendation.universe.dynamic_loader import (
            DynamicItemUniverseLoader,
        )
        from core.serving.review.human_review_queue import HumanReviewQueue

        hooks = build_compliance_hooks(_full_config())
        assert isinstance(hooks["review_queue"], HumanReviewQueue)
        assert isinstance(hooks["opt_out_manager"], OptOutManager)
        assert isinstance(hooks["profiling_workflow"], ProfilingWorkflow)
        assert isinstance(hooks["explanation_sla_tracker"], ExplanationSLATracker)
        assert isinstance(hooks["ai_risk_classifier"], AIRiskClassifier)
        assert isinstance(hooks["item_universe_loader"], DynamicItemUniverseLoader)
        assert isinstance(hooks["marker_applier"], MarkerApplier)

    def test_store_shared_across_rights_modules(self):
        """M4/M5/M6/M9 must share one ComplianceStore instance."""
        hooks = build_compliance_hooks(_full_config())
        stores = {
            id(hooks["opt_out_manager"]._store),
            id(hooks["profiling_workflow"]._store),
            id(hooks["explanation_sla_tracker"]._store),
            id(hooks["ai_risk_classifier"]._store),
        }
        assert len(stores) == 1

    def test_splats_into_recommendation_service(self):
        """The builder output must reach the serving object unchanged."""
        hooks = build_compliance_hooks(_full_config())
        svc = RecommendationService(
            model=_FakeModel(),
            feature_store=_InMemoryFeatureStore(),
            tasks_meta=[{"name": "task_a", "type": "binary"}],
            **hooks,
        )
        assert svc._review_queue is hooks["review_queue"]
        assert svc._opt_out_manager is hooks["opt_out_manager"]
        assert svc._profiling_workflow is hooks["profiling_workflow"]
        assert svc._explanation_sla_tracker is hooks["explanation_sla_tracker"]
        assert svc._ai_risk_classifier is hooks["ai_risk_classifier"]
        assert svc._item_universe_loader is hooks["item_universe_loader"]
        assert svc._marker_applier is hooks["marker_applier"]


# ---------------------------------------------------------------------------
# Enable gate
# ---------------------------------------------------------------------------

class TestEnableGate:
    def test_enabled_by_default(self):
        assert hooks_enabled({"compliance": {}}) is True
        assert hooks_enabled(None) is True

    def test_disabled_returns_empty(self):
        cfg = _full_config()
        cfg["compliance"]["hooks"] = {"enabled": False}
        assert build_compliance_hooks(cfg) == {}

    def test_none_config_returns_empty(self):
        assert build_compliance_hooks(None) == {}


# ---------------------------------------------------------------------------
# Failure posture (fail-closed vs degraded)
# ---------------------------------------------------------------------------

class TestStrictPosture:
    def test_degraded_omits_failing_hook(self):
        """Non-strict: a bad sub-block is logged & omitted, others survive."""
        cfg = _full_config()
        cfg["compliance"]["ai_risk"] = "NOT_A_DICT"  # forces build failure
        hooks = build_compliance_hooks(cfg, strict=False)
        assert "ai_risk_classifier" not in hooks
        # the rest are still wired
        assert "review_queue" in hooks
        assert "opt_out_manager" in hooks

    def test_strict_raises_on_failure(self):
        cfg = _full_config()
        cfg["compliance"]["ai_risk"] = "NOT_A_DICT"
        with pytest.raises(Exception):
            build_compliance_hooks(cfg, strict=True)

    def test_strict_read_from_config(self):
        cfg = _full_config()
        cfg["compliance"]["hooks"] = {"strict": True}
        cfg["compliance"]["ai_risk"] = "NOT_A_DICT"
        with pytest.raises(Exception):
            build_compliance_hooks(cfg)  # strict resolved from config
