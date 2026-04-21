"""
SageMakerComplianceTracker (Sprint 2 S5 — SageMaker-native variant).

AWS SageMaker ships several services that collectively replace the
open-source MLflow + DVC stack used on-premises:

- **SageMaker Experiments** (Trials / TrialComponents) → MLflow tracking
- **SageMaker Model Registry** (handled elsewhere; see
  ``core/serving/model_registry.py``) → MLflow model registry
- **SageMaker Lineage** → MLflow lineage + DVC dataset provenance
- **S3 versioning** → DVC artifact storage
- **SageMaker Managed MLflow** (2024) → MLflow as-a-service

This module wraps the Experiments API (+ Lineage) with a regulatory
focus: every Sprint 0~4 compliance artifact (FRIA report, AI-risk
assessment, compliance registry sweep, promotion-gate verdict) is
logged as a TrialComponent under a dedicated experiment so that an
auditor can reconstruct the chain of regulatory checkpoints from
the SageMaker console or via Athena over the experiments export.

Backends:
- `sagemaker`    - boto3 `sagemaker` client (production).
- `in_memory`    - append-only list (tests, local dev).

Both expose the same :class:`SageMakerComplianceTracker` API so callers
remain backend-agnostic.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "TrackingConfig",
    "TrackedArtifact",
    "ComplianceTrackerBackend",
    "InMemoryTrackerBackend",
    "SageMakerTrackerBackend",
    "SageMakerComplianceTracker",
    "build_sagemaker_compliance_tracker",
]


# ---------------------------------------------------------------------------
# Config + records
# ---------------------------------------------------------------------------

ARTIFACT_TYPES = (
    "fria_assessment",
    "ai_risk_assessment",
    "compliance_registry_sweep",
    "promotion_gate_verdict",
    "custom",
)


@dataclass
class TrackingConfig:
    """Driven by the ``compliance.tracking`` block of pipeline.yaml.

    AWS-specific identifiers (region, experiment_name tied to a bucket) are
    not hardcoded here per CLAUDE.md §1.1. The factory
    :func:`build_sagemaker_compliance_tracker` derives them from
    ``pipeline.yaml::aws.region`` and ``aws.s3_bucket`` when the
    ``compliance.tracking`` block does not set them explicitly.
    """

    backend: str = "in_memory"          # "sagemaker" | "in_memory"
    # Generic placeholder — carries no AWS-specific identifier. Factory
    # overrides with ``{aws.s3_bucket}-compliance`` when available, or the
    # caller supplies an explicit value via pipeline.yaml.
    experiment_name: str = "compliance"
    trial_prefix: str = "compliance-trial"
    # None → boto3 resolves from env / shared credentials; factory injects
    # pipeline.yaml::aws.region when the tracking block omits it.
    region: Optional[str] = None
    artifact_s3_prefix: str = ""        # optional: s3://bucket/prefix
    display_name_tag: str = "compliance_type"

    def __post_init__(self) -> None:
        if self.backend not in ("sagemaker", "in_memory"):
            raise ValueError(
                f"backend={self.backend!r} must be 'sagemaker' or 'in_memory'"
            )
        if self.experiment_name is None or not self.experiment_name.strip():
            raise ValueError("experiment_name must be non-empty")

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "TrackingConfig":
        if not data:
            return cls()
        kwargs: Dict[str, Any] = {}
        if "backend" in data and data["backend"] is not None:
            kwargs["backend"] = str(data["backend"])
        if "experiment_name" in data and data["experiment_name"] is not None:
            kwargs["experiment_name"] = str(data["experiment_name"])
        if "trial_prefix" in data and data["trial_prefix"] is not None:
            kwargs["trial_prefix"] = str(data["trial_prefix"])
        if "region" in data:
            kwargs["region"] = (
                str(data["region"]) if data["region"] is not None else None
            )
        if "artifact_s3_prefix" in data and data["artifact_s3_prefix"] is not None:
            kwargs["artifact_s3_prefix"] = str(data["artifact_s3_prefix"])
        if "display_name_tag" in data and data["display_name_tag"] is not None:
            kwargs["display_name_tag"] = str(data["display_name_tag"])
        return cls(**kwargs)


@dataclass
class TrackedArtifact:
    """A single regulatory artifact recorded in the tracker."""

    artifact_id: str
    artifact_type: str                  # one of ARTIFACT_TYPES
    model_version: str
    recorded_at: datetime
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    s3_uri: Optional[str] = None
    trial_component_arn: Optional[str] = None   # populated by SageMaker backend

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["recorded_at"] = self.recorded_at.isoformat()
        return d


# ---------------------------------------------------------------------------
# Backend ABC
# ---------------------------------------------------------------------------

class ComplianceTrackerBackend:
    """Storage abstraction for compliance tracking records."""

    def ensure_experiment(self, experiment_name: str) -> None:
        raise NotImplementedError

    def put_artifact(self, artifact: TrackedArtifact) -> TrackedArtifact:
        raise NotImplementedError

    def list_artifacts(
        self,
        artifact_type: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> List[TrackedArtifact]:
        raise NotImplementedError


class InMemoryTrackerBackend(ComplianceTrackerBackend):
    """Append-only in-memory backend for tests / local dev."""

    def __init__(self) -> None:
        self._experiments: List[str] = []
        self._artifacts: List[TrackedArtifact] = []

    def ensure_experiment(self, experiment_name: str) -> None:
        if experiment_name not in self._experiments:
            self._experiments.append(experiment_name)

    def put_artifact(self, artifact: TrackedArtifact) -> TrackedArtifact:
        self._artifacts.append(artifact)
        return artifact

    def list_artifacts(
        self,
        artifact_type: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> List[TrackedArtifact]:
        out = list(self._artifacts)
        if artifact_type is not None:
            out = [a for a in out if a.artifact_type == artifact_type]
        if model_version is not None:
            out = [a for a in out if a.model_version == model_version]
        out.sort(key=lambda a: a.recorded_at)
        return out


class SageMakerTrackerBackend(ComplianceTrackerBackend):
    """boto3 SageMaker Experiments backend (production)."""

    def __init__(
        self,
        sagemaker_client: Any = None,
        region: Optional[str] = None,
    ) -> None:
        if sagemaker_client is None:
            try:
                import boto3  # type: ignore
                sagemaker_client = boto3.client(
                    "sagemaker", region_name=region,
                )
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "boto3 not installed; install or inject a sagemaker "
                    "client"
                ) from exc
        self._client = sagemaker_client
        self._ensured: set = set()

    def ensure_experiment(self, experiment_name: str) -> None:
        if experiment_name in self._ensured:
            return
        try:
            self._client.describe_experiment(ExperimentName=experiment_name)
        except Exception:
            # Create if missing. A conflicting-name error is treated as
            # "already exists" (e.g. eventual-consistency).
            try:
                self._client.create_experiment(
                    ExperimentName=experiment_name,
                    Description=(
                        "Regulatory compliance tracking for AIOps PLE. "
                        "Auto-created by SageMakerComplianceTracker."
                    ),
                )
            except Exception:
                logger.debug(
                    "SageMaker create_experiment failed (assuming exists): %s",
                    experiment_name,
                    exc_info=True,
                )
        self._ensured.add(experiment_name)

    def put_artifact(self, artifact: TrackedArtifact) -> TrackedArtifact:
        component_name = (
            f"{artifact.artifact_type}-{artifact.artifact_id}"
        )[:120]  # SageMaker name cap

        parameters = {
            k: {"StringValue": str(v)}
            for k, v in artifact.parameters.items()
        }
        parameters["model_version"] = {
            "StringValue": str(artifact.model_version),
        }
        parameters["artifact_id"] = {"StringValue": artifact.artifact_id}

        metrics_payload: Dict[str, Dict[str, Any]] = {}
        for k, v in artifact.metrics.items():
            try:
                metrics_payload[k] = {"NumberValue": float(v)}
            except (TypeError, ValueError):
                logger.debug(
                    "Dropping non-numeric metric %r=%r on artifact %s",
                    k, v, artifact.artifact_id,
                )

        tags_payload = [
            {"Key": k, "Value": str(v)}
            for k, v in artifact.tags.items()
        ]
        tags_payload.append({"Key": "artifact_type", "Value": artifact.artifact_type})

        request: Dict[str, Any] = {
            "TrialComponentName": component_name,
            "Status": {"PrimaryStatus": "Completed", "Message": "OK"},
            "Parameters": parameters,
            "Tags": tags_payload,
        }
        if metrics_payload:
            request["Parameters"].update(
                {k: {"StringValue": str(v["NumberValue"])}
                 for k, v in metrics_payload.items()}
            )
        if artifact.s3_uri:
            request["OutputArtifacts"] = {
                "compliance_artifact": {
                    "Value": artifact.s3_uri,
                    "MediaType": "application/json",
                },
            }

        try:
            resp = self._client.create_trial_component(**request)
            artifact.trial_component_arn = resp.get("TrialComponentArn")
        except Exception:
            logger.exception(
                "SageMaker create_trial_component failed for %s",
                component_name,
            )
        return artifact

    def list_artifacts(
        self,
        artifact_type: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> List[TrackedArtifact]:
        try:
            kwargs: Dict[str, Any] = {"MaxResults": 100}
            resp = self._client.list_trial_components(**kwargs)
        except Exception:
            logger.exception("SageMaker list_trial_components failed")
            return []

        out: List[TrackedArtifact] = []
        for summary in resp.get("TrialComponentSummaries", []):
            tc_name = summary.get("TrialComponentName", "")
            tags = {
                t.get("Key"): t.get("Value")
                for t in summary.get("Tags", []) or []
            }
            this_type = tags.get("artifact_type", "custom")
            if artifact_type is not None and this_type != artifact_type:
                continue
            model_v = "unknown"
            params = summary.get("Parameters", {}) or {}
            if "model_version" in params:
                model_v = params["model_version"].get("StringValue", "unknown")
            if model_version is not None and model_v != model_version:
                continue
            try:
                recorded = summary.get("CreationTime")
                if isinstance(recorded, str):
                    recorded_dt = datetime.fromisoformat(recorded)
                else:
                    recorded_dt = recorded or datetime.now(timezone.utc)
            except Exception:
                recorded_dt = datetime.now(timezone.utc)
            out.append(TrackedArtifact(
                artifact_id=params.get("artifact_id", {}).get(
                    "StringValue", tc_name,
                ),
                artifact_type=this_type,
                model_version=model_v,
                recorded_at=recorded_dt,
                parameters={
                    k: v.get("StringValue")
                    for k, v in params.items()
                },
                tags=tags,
                trial_component_arn=summary.get("TrialComponentArn"),
            ))
        out.sort(key=lambda a: a.recorded_at)
        return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class SageMakerComplianceTracker:
    """Regulatory-artifact tracker layered on SageMaker Experiments.

    Each :meth:`log_*` call records one TrialComponent whose tags mark
    the artifact type (``fria_assessment`` / ``ai_risk_assessment`` /
    ``compliance_registry_sweep`` / ``promotion_gate_verdict`` / ``custom``).
    """

    def __init__(
        self,
        config: Optional[TrackingConfig] = None,
        backend: Optional[ComplianceTrackerBackend] = None,
    ) -> None:
        self._cfg = config or TrackingConfig()
        if backend is not None:
            self._backend = backend
        elif self._cfg.backend == "sagemaker":
            self._backend = SageMakerTrackerBackend(region=self._cfg.region)
        else:
            self._backend = InMemoryTrackerBackend()
        self._backend.ensure_experiment(self._cfg.experiment_name)

    # ------------------------------------------------------------------
    # Regulatory-artifact-specific helpers
    # ------------------------------------------------------------------

    def log_fria_assessment(
        self,
        fria_result: Any,
        s3_uri: Optional[str] = None,
    ) -> TrackedArtifact:
        """Record a :class:`FRIAResult` as a TrialComponent."""
        params = {
            "operator_type": getattr(fria_result, "operator_type", ""),
            "risk_category": getattr(fria_result, "risk_category", ""),
            "assessment_id": getattr(fria_result, "assessment_id", ""),
            "retention_expiry": _iso(
                getattr(fria_result, "retention_expiry", None),
            ),
        }
        metrics = {}
        total = getattr(fria_result, "total_score", None)
        if total is not None:
            metrics["total_score"] = float(total)
        dims = getattr(fria_result, "dimensions", {}) or {}
        for d, v in dims.items():
            try:
                metrics[f"dim.{d}"] = float(v)
            except (TypeError, ValueError):
                pass
        return self._log(
            artifact_type="fria_assessment",
            model_version=getattr(fria_result, "model_version", "unknown"),
            parameters=params,
            metrics=metrics,
            tags={"risk_category": str(params.get("risk_category", ""))},
            s3_uri=s3_uri,
        )

    def log_ai_risk_assessment(
        self,
        assessment: Any,
        s3_uri: Optional[str] = None,
    ) -> TrackedArtifact:
        params = {
            "grade": getattr(assessment, "grade", ""),
            "prev_grade": str(getattr(assessment, "prev_grade", "") or ""),
            "grade_change": str(getattr(assessment, "grade_change", False)),
            "assessment_id": getattr(assessment, "assessment_id", ""),
        }
        metrics = {}
        total = getattr(assessment, "total_score", None)
        if total is not None:
            metrics["total_score"] = float(total)
        dims = getattr(assessment, "dimensions", {}) or {}
        for d, v in dims.items():
            try:
                metrics[f"dim.{d}"] = float(v)
            except (TypeError, ValueError):
                pass
        return self._log(
            artifact_type="ai_risk_assessment",
            model_version=getattr(assessment, "model_version", "unknown"),
            parameters=params,
            metrics=metrics,
            tags={"grade": str(params.get("grade", ""))},
            s3_uri=s3_uri,
        )

    def log_compliance_sweep(
        self,
        summary: Dict[str, Any],
        model_version: str = "registry",
        s3_uri: Optional[str] = None,
    ) -> TrackedArtifact:
        """Record the output of :meth:`ComplianceRegistry.summary`."""
        critical = summary.get("critical_failures", []) or []
        params = {
            "total": summary.get("total", 0),
            "critical_failure_count": len(critical),
            "critical_failures": ",".join(critical)[:400],
        }
        metrics = {}
        by_status = summary.get("by_status", {}) or {}
        for status, count in by_status.items():
            try:
                metrics[f"status.{status}"] = float(count)
            except (TypeError, ValueError):
                pass
        return self._log(
            artifact_type="compliance_registry_sweep",
            model_version=model_version,
            parameters=params,
            metrics=metrics,
            s3_uri=s3_uri,
        )

    def log_promotion_decision(
        self,
        model_version: str,
        decision: str,
        reason: str,
        fria_result: Any = None,
        ai_risk_assessment: Any = None,
        s3_uri: Optional[str] = None,
    ) -> TrackedArtifact:
        params = {
            "decision": decision,
            "reason": reason[:400],
            "fria_category": (
                getattr(fria_result, "risk_category", "")
                if fria_result is not None else ""
            ),
            "ai_risk_grade": (
                getattr(ai_risk_assessment, "grade", "")
                if ai_risk_assessment is not None else ""
            ),
        }
        metrics = {}
        if fria_result is not None:
            total = getattr(fria_result, "total_score", None)
            if total is not None:
                metrics["fria_total_score"] = float(total)
        if ai_risk_assessment is not None:
            total = getattr(ai_risk_assessment, "total_score", None)
            if total is not None:
                metrics["ai_risk_total_score"] = float(total)
        return self._log(
            artifact_type="promotion_gate_verdict",
            model_version=model_version,
            parameters=params,
            metrics=metrics,
            tags={"decision": decision},
            s3_uri=s3_uri,
        )

    def log_custom_artifact(
        self,
        name: str,
        model_version: str,
        parameters: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[Dict[str, str]] = None,
        s3_uri: Optional[str] = None,
    ) -> TrackedArtifact:
        params = dict(parameters or {})
        params["custom_name"] = name
        return self._log(
            artifact_type="custom",
            model_version=model_version,
            parameters=params,
            metrics=dict(metrics or {}),
            tags=dict(tags or {}),
            s3_uri=s3_uri,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_artifacts(
        self,
        artifact_type: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> List[TrackedArtifact]:
        return self._backend.list_artifacts(
            artifact_type=artifact_type, model_version=model_version,
        )

    def summary(self) -> Dict[str, Any]:
        """Return a small dict suitable for dashboards or governance reports."""
        all_artifacts = self._backend.list_artifacts()
        counts: Dict[str, int] = {}
        for a in all_artifacts:
            counts[a.artifact_type] = counts.get(a.artifact_type, 0) + 1
        return {
            "experiment_name": self._cfg.experiment_name,
            "backend": self._cfg.backend,
            "total_artifacts": len(all_artifacts),
            "by_type": counts,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _log(
        self,
        artifact_type: str,
        model_version: str,
        parameters: Dict[str, Any],
        metrics: Dict[str, float],
        tags: Optional[Dict[str, str]] = None,
        s3_uri: Optional[str] = None,
    ) -> TrackedArtifact:
        if artifact_type not in ARTIFACT_TYPES:
            raise ValueError(
                f"artifact_type={artifact_type!r} must be in {ARTIFACT_TYPES}"
            )
        artifact = TrackedArtifact(
            artifact_id=f"{artifact_type}-{uuid.uuid4().hex[:12]}",
            artifact_type=artifact_type,
            model_version=str(model_version),
            recorded_at=datetime.now(timezone.utc),
            parameters=dict(parameters),
            metrics=dict(metrics),
            tags=dict(tags or {}),
            s3_uri=s3_uri,
        )
        return self._backend.put_artifact(artifact)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_sagemaker_compliance_tracker(
    pipeline_config: Optional[Dict[str, Any]] = None,
    backend: Optional[ComplianceTrackerBackend] = None,
) -> SageMakerComplianceTracker:
    """Instantiate from the ``compliance.tracking`` block of pipeline.yaml.

    AWS constants are derived from the top-level ``aws`` block when the
    tracking block omits them:

    - ``region``          ← ``aws.region``
    - ``experiment_name`` ← ``{aws.s3_bucket}-compliance`` (derived identifier)
    """
    pc = pipeline_config or {}
    compliance_cfg = pc.get("compliance") or {}
    aws_cfg = pc.get("aws") or {}
    tracking_data: Dict[str, Any] = dict(compliance_cfg.get("tracking") or {})

    # region ← aws.region (when not explicitly overridden in tracking block)
    if "region" not in tracking_data and aws_cfg.get("region"):
        tracking_data["region"] = aws_cfg["region"]

    # experiment_name ← derived from aws.s3_bucket so the tracking stream is
    # tagged with the deployment's bucket identifier without repeating it.
    if "experiment_name" not in tracking_data:
        s3_bucket = aws_cfg.get("s3_bucket")
        if s3_bucket:
            tracking_data["experiment_name"] = f"{s3_bucket}-compliance"

    cfg = TrackingConfig.from_dict(tracking_data)
    return SageMakerComplianceTracker(config=cfg, backend=backend)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iso(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)
