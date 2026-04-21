"""
Sprint 2 S5 tests: SageMakerComplianceTracker in both in-memory and
fake-SageMaker modes.

Run: pytest tests/test_sagemaker_compliance_tracker.py -v
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

from core.compliance.ai_risk_classifier import AIRiskAssessment
from core.compliance.fria_assessment import FRIA_DIMENSIONS, FRIAResult
from core.compliance.sagemaker_compliance_tracker import (
    ARTIFACT_TYPES,
    InMemoryTrackerBackend,
    SageMakerComplianceTracker,
    SageMakerTrackerBackend,
    TrackedArtifact,
    TrackingConfig,
    build_sagemaker_compliance_tracker,
)


# ---------------------------------------------------------------------------
# TrackingConfig
# ---------------------------------------------------------------------------

class TestTrackingConfig:
    def test_defaults(self):
        cfg = TrackingConfig()
        assert cfg.backend == "in_memory"
        assert "compliance" in cfg.experiment_name

    def test_rejects_unknown_backend(self):
        with pytest.raises(ValueError):
            TrackingConfig(backend="mlflow")

    def test_rejects_blank_experiment_name(self):
        with pytest.raises(ValueError):
            TrackingConfig(experiment_name="   ")

    def test_from_dict(self):
        cfg = TrackingConfig.from_dict({
            "backend": "sagemaker",
            "experiment_name": "abc",
            "region": "us-east-1",
        })
        assert cfg.backend == "sagemaker"
        assert cfg.experiment_name == "abc"
        assert cfg.region == "us-east-1"

    def test_from_empty_dict_uses_defaults(self):
        cfg = TrackingConfig.from_dict(None)
        assert cfg.backend == "in_memory"


# ---------------------------------------------------------------------------
# In-memory backend
# ---------------------------------------------------------------------------

class TestInMemoryBackend:
    def test_put_and_list(self):
        backend = InMemoryTrackerBackend()
        backend.ensure_experiment("exp1")
        art = TrackedArtifact(
            artifact_id="a1",
            artifact_type="fria_assessment",
            model_version="v1",
            recorded_at=datetime.now(timezone.utc),
            parameters={"risk_category": "LIMITED"},
            metrics={"total_score": 0.45},
            tags={},
        )
        backend.put_artifact(art)
        lst = backend.list_artifacts()
        assert len(lst) == 1
        assert lst[0].artifact_id == "a1"

    def test_filter_by_type(self):
        backend = InMemoryTrackerBackend()
        backend.put_artifact(TrackedArtifact(
            artifact_id="a1", artifact_type="fria_assessment",
            model_version="v1", recorded_at=datetime.now(timezone.utc),
        ))
        backend.put_artifact(TrackedArtifact(
            artifact_id="a2", artifact_type="ai_risk_assessment",
            model_version="v1", recorded_at=datetime.now(timezone.utc),
        ))
        fria = backend.list_artifacts(artifact_type="fria_assessment")
        assert len(fria) == 1
        ai = backend.list_artifacts(artifact_type="ai_risk_assessment")
        assert len(ai) == 1

    def test_filter_by_model_version(self):
        backend = InMemoryTrackerBackend()
        backend.put_artifact(TrackedArtifact(
            artifact_id="a1", artifact_type="custom",
            model_version="v1", recorded_at=datetime.now(timezone.utc),
        ))
        backend.put_artifact(TrackedArtifact(
            artifact_id="a2", artifact_type="custom",
            model_version="v2", recorded_at=datetime.now(timezone.utc),
        ))
        v1 = backend.list_artifacts(model_version="v1")
        assert len(v1) == 1
        assert v1[0].model_version == "v1"


# ---------------------------------------------------------------------------
# Fake SageMaker client backend
# ---------------------------------------------------------------------------

class _FakeSageMakerClient:
    """Captures calls so we can assert on the shape of the boto3 requests."""

    def __init__(self):
        self.experiments: List[Dict[str, Any]] = []
        self.trial_components: List[Dict[str, Any]] = []
        self.describe_calls: List[str] = []

    def describe_experiment(self, ExperimentName):
        self.describe_calls.append(ExperimentName)
        if ExperimentName not in [e["ExperimentName"] for e in self.experiments]:
            raise RuntimeError("not found")
        return {"ExperimentName": ExperimentName}

    def create_experiment(self, ExperimentName, Description=None):
        self.experiments.append({
            "ExperimentName": ExperimentName, "Description": Description,
        })
        return {}

    def create_trial_component(self, **kwargs):
        self.trial_components.append(kwargs)
        return {
            "TrialComponentArn": (
                f"arn:aws:sagemaker:ap-northeast-2:000:trial-component/"
                f"{kwargs['TrialComponentName']}"
            ),
        }

    def list_trial_components(self, **kwargs):
        summaries = []
        for tc in self.trial_components:
            summaries.append({
                "TrialComponentName": tc["TrialComponentName"],
                "TrialComponentArn": (
                    f"arn:.../{tc['TrialComponentName']}"
                ),
                "CreationTime": datetime.now(timezone.utc).isoformat(),
                "Tags": tc.get("Tags", []),
                "Parameters": tc.get("Parameters", {}),
            })
        return {"TrialComponentSummaries": summaries}


class TestSageMakerTrackerBackend:
    def test_create_experiment_on_first_put(self):
        fake = _FakeSageMakerClient()
        backend = SageMakerTrackerBackend(sagemaker_client=fake)
        backend.ensure_experiment("exp1")
        assert any(e["ExperimentName"] == "exp1" for e in fake.experiments)

    def test_put_artifact_creates_trial_component(self):
        fake = _FakeSageMakerClient()
        backend = SageMakerTrackerBackend(sagemaker_client=fake)
        backend.ensure_experiment("exp1")
        art = TrackedArtifact(
            artifact_id="a1",
            artifact_type="fria_assessment",
            model_version="v1",
            recorded_at=datetime.now(timezone.utc),
            parameters={"risk_category": "HIGH"},
            metrics={"total_score": 0.75},
            tags={"risk_category": "HIGH"},
        )
        stored = backend.put_artifact(art)
        assert stored.trial_component_arn is not None
        assert len(fake.trial_components) == 1
        tc = fake.trial_components[0]
        # Tags include our artifact_type label
        assert any(
            t.get("Key") == "artifact_type" and t.get("Value") == "fria_assessment"
            for t in tc["Tags"]
        )
        # model_version and artifact_id must be present as parameters
        assert tc["Parameters"]["model_version"]["StringValue"] == "v1"

    def test_list_round_trip(self):
        fake = _FakeSageMakerClient()
        backend = SageMakerTrackerBackend(sagemaker_client=fake)
        backend.ensure_experiment("exp1")
        backend.put_artifact(TrackedArtifact(
            artifact_id="a1", artifact_type="custom", model_version="v1",
            recorded_at=datetime.now(timezone.utc),
            parameters={"k": "v"}, tags={"custom": "yes"},
        ))
        backend.put_artifact(TrackedArtifact(
            artifact_id="a2", artifact_type="fria_assessment",
            model_version="v1",
            recorded_at=datetime.now(timezone.utc),
            parameters={}, tags={},
        ))
        fria = backend.list_artifacts(artifact_type="fria_assessment")
        assert len(fria) == 1

    def test_failed_create_trial_component_still_returns_artifact(self):
        class _Bad(_FakeSageMakerClient):
            def create_trial_component(self, **kwargs):
                raise RuntimeError("API down")

        fake = _Bad()
        backend = SageMakerTrackerBackend(sagemaker_client=fake)
        backend.ensure_experiment("exp1")
        art = TrackedArtifact(
            artifact_id="a_err", artifact_type="custom",
            model_version="v1",
            recorded_at=datetime.now(timezone.utc),
        )
        # Must not raise — audit is best-effort
        stored = backend.put_artifact(art)
        assert stored.trial_component_arn is None


# ---------------------------------------------------------------------------
# Public API: SageMakerComplianceTracker
# ---------------------------------------------------------------------------

class TestComplianceTrackerPublicAPI:
    def _tracker(self, fake_client=None) -> SageMakerComplianceTracker:
        backend = (
            SageMakerTrackerBackend(sagemaker_client=fake_client)
            if fake_client else InMemoryTrackerBackend()
        )
        return SageMakerComplianceTracker(
            config=TrackingConfig(experiment_name="exp1"),
            backend=backend,
        )

    def _fria(self, model_version="v1", category="LIMITED", total=0.5):
        return FRIAResult(
            assessment_id="fr-1",
            model_version=model_version,
            operator_type="국가기관등",
            assessed_at=datetime.now(timezone.utc),
            retention_expiry=datetime.now(timezone.utc) + timedelta(days=1825),
            dimensions={d: 0.5 for d in FRIA_DIMENSIONS},
            total_score=total,
            risk_category=category,
        )

    def _ai_risk(self, grade="medium", total=0.5, prev_grade=None):
        return AIRiskAssessment(
            assessment_id="ar-1",
            model_version="v1",
            assessed_at=datetime.now(timezone.utc),
            dimensions={
                "data_sensitivity": 0.5, "automation_level": 0.5,
                "scope_of_impact": 0.5, "model_complexity": 0.5,
                "external_dependency": 0.5, "fairness_risk": 0.5,
            },
            weights={
                "data_sensitivity": 0.25, "automation_level": 0.20,
                "scope_of_impact": 0.20, "model_complexity": 0.15,
                "external_dependency": 0.10, "fairness_risk": 0.10,
            },
            total_score=total,
            grade=grade,
            prev_grade=prev_grade,
            grade_change=prev_grade is not None and prev_grade != grade,
        )

    def test_log_fria_assessment(self):
        tracker = self._tracker()
        art = tracker.log_fria_assessment(self._fria(category="HIGH"))
        assert art.artifact_type == "fria_assessment"
        assert art.parameters["risk_category"] == "HIGH"
        assert 0.0 <= art.metrics["total_score"] <= 1.0
        # Dimension metrics were forwarded
        assert any(k.startswith("dim.") for k in art.metrics)

    def test_log_ai_risk_assessment(self):
        tracker = self._tracker()
        art = tracker.log_ai_risk_assessment(
            self._ai_risk(grade="high", total=0.8, prev_grade="medium"),
        )
        assert art.artifact_type == "ai_risk_assessment"
        assert art.parameters["grade"] == "high"
        assert art.parameters["grade_change"] == "True"

    def test_log_compliance_sweep(self):
        tracker = self._tracker()
        art = tracker.log_compliance_sweep({
            "total": 36,
            "by_status": {"compliant": 30, "non_compliant": 5,
                          "not_applicable": 1},
            "critical_failures": ["A-06", "GAP-03"],
        })
        assert art.artifact_type == "compliance_registry_sweep"
        assert art.parameters["total"] == 36
        assert art.parameters["critical_failure_count"] == 2
        assert art.metrics["status.compliant"] == 30.0

    def test_log_promotion_decision(self):
        tracker = self._tracker()
        fria = self._fria(category="HIGH", total=0.72)
        ai_risk = self._ai_risk(grade="high", total=0.75, prev_grade="medium")
        art = tracker.log_promotion_decision(
            model_version="v2",
            decision="require_approval",
            reason="FRIA HIGH + AI risk escalation",
            fria_result=fria,
            ai_risk_assessment=ai_risk,
        )
        assert art.artifact_type == "promotion_gate_verdict"
        assert art.parameters["decision"] == "require_approval"
        assert art.metrics["fria_total_score"] == pytest.approx(0.72)
        assert art.metrics["ai_risk_total_score"] == pytest.approx(0.75)

    def test_log_custom_artifact(self):
        tracker = self._tracker()
        art = tracker.log_custom_artifact(
            name="monthly_sla_report",
            model_version="operations",
            parameters={"month": "2026-04"},
            metrics={"breaches": 2.0},
            tags={"severity": "medium"},
        )
        assert art.artifact_type == "custom"
        assert art.parameters["custom_name"] == "monthly_sla_report"
        assert art.tags["severity"] == "medium"

    def test_list_artifacts_by_type(self):
        tracker = self._tracker()
        tracker.log_fria_assessment(self._fria())
        tracker.log_ai_risk_assessment(self._ai_risk())
        fria_list = tracker.list_artifacts(artifact_type="fria_assessment")
        assert len(fria_list) == 1
        ai_list = tracker.list_artifacts(artifact_type="ai_risk_assessment")
        assert len(ai_list) == 1

    def test_summary(self):
        tracker = self._tracker()
        tracker.log_fria_assessment(self._fria())
        tracker.log_fria_assessment(self._fria(model_version="v2"))
        tracker.log_ai_risk_assessment(self._ai_risk())
        s = tracker.summary()
        assert s["total_artifacts"] == 3
        assert s["by_type"]["fria_assessment"] == 2
        assert s["by_type"]["ai_risk_assessment"] == 1


# ---------------------------------------------------------------------------
# Fake SageMaker end-to-end
# ---------------------------------------------------------------------------

class TestEndToEndWithFakeSageMaker:
    def test_full_flow(self):
        fake = _FakeSageMakerClient()
        tracker = SageMakerComplianceTracker(
            config=TrackingConfig(
                backend="sagemaker", experiment_name="e1",
            ),
            backend=SageMakerTrackerBackend(sagemaker_client=fake),
        )
        tracker.log_custom_artifact(
            name="test", model_version="v1",
            parameters={"k": "v"}, metrics={"m": 1.0},
        )
        tracker.log_compliance_sweep({
            "total": 36, "by_status": {"compliant": 36},
            "critical_failures": [],
        }, model_version="registry")

        # One experiment should have been created, two trial components
        # should have been put.
        assert any(e["ExperimentName"] == "e1" for e in fake.experiments)
        assert len(fake.trial_components) == 2

    def test_boto3_payload_shape(self):
        fake = _FakeSageMakerClient()
        backend = SageMakerTrackerBackend(sagemaker_client=fake)
        tracker = SageMakerComplianceTracker(
            config=TrackingConfig(
                backend="sagemaker", experiment_name="e1",
            ),
            backend=backend,
        )
        tracker.log_fria_assessment(FRIAResult(
            assessment_id="fr-2",
            model_version="v2",
            operator_type="국가기관등",
            assessed_at=datetime.now(timezone.utc),
            retention_expiry=datetime.now(timezone.utc) + timedelta(days=1825),
            dimensions={d: 0.7 for d in FRIA_DIMENSIONS},
            total_score=0.7,
            risk_category="HIGH",
        ))
        tc = fake.trial_components[0]
        # The TrialComponentName must be non-empty and <= 120 chars (SageMaker cap)
        assert 1 <= len(tc["TrialComponentName"]) <= 120
        assert tc["Status"]["PrimaryStatus"] == "Completed"
        # risk_category tag must be present
        risk_tags = [
            t for t in tc["Tags"]
            if t.get("Key") == "risk_category"
        ]
        assert len(risk_tags) == 1


# ---------------------------------------------------------------------------
# Factory + pipeline.yaml integration
# ---------------------------------------------------------------------------

class TestFactoryFromPipelineYAML:
    def test_build_default_is_in_memory(self):
        cfg = yaml.safe_load(
            Path("configs/pipeline.yaml").read_text(encoding="utf-8")
        )
        tracker = build_sagemaker_compliance_tracker(cfg)
        # Default is in_memory per pipeline.yaml
        assert tracker.summary()["backend"] == "in_memory"

    def test_build_sagemaker_backend_with_fake(self):
        cfg = {
            "compliance": {
                "tracking": {
                    "backend": "sagemaker",
                    "experiment_name": "from_factory",
                }
            }
        }
        fake = _FakeSageMakerClient()
        tracker = build_sagemaker_compliance_tracker(
            cfg, backend=SageMakerTrackerBackend(sagemaker_client=fake),
        )
        assert tracker.summary()["experiment_name"] == "from_factory"
        assert tracker.summary()["backend"] == "sagemaker"


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------

class TestConstants:
    def test_artifact_types_stable(self):
        assert set(ARTIFACT_TYPES) == {
            "fria_assessment",
            "ai_risk_assessment",
            "compliance_registry_sweep",
            "promotion_gate_verdict",
            "custom",
        }
