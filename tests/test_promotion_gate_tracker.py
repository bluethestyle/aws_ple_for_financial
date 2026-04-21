"""
Tracker integration tests for PromotionGate (CLAUDE.md §1.14).

Validates that ``_run_promotion_gate()`` in submit_pipeline records a
``promotion_gate_verdict`` artifact via ``SageMakerComplianceTracker``
on every non-skip verdict. The in-memory backend is used to assert
the artifact shape without requiring boto3 / AWS credentials.

Run: ``pytest tests/test_promotion_gate_tracker.py -v``
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from core.compliance.fria_assessment import FRIA_DIMENSIONS
from core.compliance.sagemaker_compliance_tracker import (
    InMemoryTrackerBackend,
    SageMakerComplianceTracker,
    TrackingConfig,
)


def _base_compliance_config(enabled: bool = True,
                             manual_overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "compliance": {
            "store": {"backend": "in_memory"},
            "fria": {},
            "ai_risk": {},
            "promotion_gate": {
                "enabled": enabled,
                "require_approval_on_escalation": True,
                "default_score": 0.5,
                "providers": {
                    "manual_overrides": manual_overrides or {},
                },
            },
            "tracking": {
                "backend": "in_memory",
                "experiment_name": "test-compliance",
            },
        },
        "llm_provider": {"backend": "dummy"},
    }


class TestTrackerWiring:
    def test_pass_verdict_logged_as_artifact(self, monkeypatch):
        """A PASS verdict produces a promotion_gate_verdict artifact."""
        from scripts import submit_pipeline

        backend = InMemoryTrackerBackend()

        def _fake_tracker(pipeline_config):
            return SageMakerComplianceTracker(
                config=TrackingConfig(
                    backend="in_memory",
                    experiment_name="test-compliance",
                ),
                backend=backend,
            )

        monkeypatch.setattr(
            "core.compliance.sagemaker_compliance_tracker"
            ".build_sagemaker_compliance_tracker",
            _fake_tracker,
        )

        cfg = _base_compliance_config(enabled=True)
        verdict = submit_pipeline._run_promotion_gate(
            new_version="v_pass", pipeline_config=cfg,
        )
        assert verdict is not None
        assert verdict.decision == "pass"

        artifacts = backend.list_artifacts(
            artifact_type="promotion_gate_verdict",
        )
        assert len(artifacts) == 1
        a = artifacts[0]
        assert a.model_version == "v_pass"
        assert a.parameters.get("decision") == "pass"
        assert a.tags.get("decision") == "pass"
        assert "fria_total_score" in a.metrics
        assert "ai_risk_total_score" in a.metrics

    def test_reject_verdict_logged_with_decision_tag(self, monkeypatch):
        from scripts import submit_pipeline

        backend = InMemoryTrackerBackend()
        monkeypatch.setattr(
            "core.compliance.sagemaker_compliance_tracker"
            ".build_sagemaker_compliance_tracker",
            lambda cfg: SageMakerComplianceTracker(
                config=TrackingConfig(
                    backend="in_memory",
                    experiment_name="test-compliance",
                ),
                backend=backend,
            ),
        )

        cfg = _base_compliance_config(
            enabled=True,
            manual_overrides={
                "v_bad": {d: 0.99 for d in FRIA_DIMENSIONS},
            },
        )
        verdict = submit_pipeline._run_promotion_gate(
            new_version="v_bad", pipeline_config=cfg,
        )
        assert verdict.decision == "reject"

        artifacts = backend.list_artifacts(
            artifact_type="promotion_gate_verdict",
        )
        assert len(artifacts) == 1
        assert artifacts[0].tags.get("decision") == "reject"
        # fria_category is present on reject (FRIA result attached)
        assert artifacts[0].parameters.get("fria_category") == "UNACCEPTABLE"

    def test_disabled_gate_logs_nothing(self, monkeypatch):
        from scripts import submit_pipeline

        backend = InMemoryTrackerBackend()
        monkeypatch.setattr(
            "core.compliance.sagemaker_compliance_tracker"
            ".build_sagemaker_compliance_tracker",
            lambda cfg: SageMakerComplianceTracker(
                config=TrackingConfig(
                    backend="in_memory",
                    experiment_name="test-compliance",
                ),
                backend=backend,
            ),
        )

        cfg = _base_compliance_config(enabled=False)
        verdict = submit_pipeline._run_promotion_gate(
            new_version="v_x", pipeline_config=cfg,
        )
        assert verdict is None
        assert backend.list_artifacts() == []

    def test_tracker_failure_does_not_block_verdict(self, monkeypatch):
        """Tracker exception must be swallowed; verdict still returned."""
        from scripts import submit_pipeline

        def _raising_tracker(cfg):
            raise RuntimeError("tracker down")

        monkeypatch.setattr(
            "core.compliance.sagemaker_compliance_tracker"
            ".build_sagemaker_compliance_tracker",
            _raising_tracker,
        )

        cfg = _base_compliance_config(enabled=True)
        verdict = submit_pipeline._run_promotion_gate(
            new_version="v_pass", pipeline_config=cfg,
        )
        # Verdict preserved even though tracker failed
        assert verdict is not None
        assert verdict.decision == "pass"

    def test_reason_truncated_to_400_chars(self, monkeypatch):
        """Tracker clips reason to 400 chars — verify via artifact shape."""
        from scripts import submit_pipeline

        backend = InMemoryTrackerBackend()
        monkeypatch.setattr(
            "core.compliance.sagemaker_compliance_tracker"
            ".build_sagemaker_compliance_tracker",
            lambda cfg: SageMakerComplianceTracker(
                config=TrackingConfig(
                    backend="in_memory",
                    experiment_name="test-compliance",
                ),
                backend=backend,
            ),
        )

        cfg = _base_compliance_config(enabled=True)
        submit_pipeline._run_promotion_gate(
            new_version="v_any", pipeline_config=cfg,
        )
        artifacts = backend.list_artifacts(
            artifact_type="promotion_gate_verdict",
        )
        assert len(artifacts) == 1
        assert len(artifacts[0].parameters.get("reason", "")) <= 400
