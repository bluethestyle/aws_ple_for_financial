"""
Ops Collector — 7 Checkpoint Measurement Collection
=====================================================

Collects measurements from 7 pipeline checkpoints (CP1-CP7):
    CP1: Ingestion complete
    CP2: Phase 0 complete (feature engineering)
    CP3: Training complete
    CP4: Distillation complete
    CP5: Serving health
    CP6: Recommendation response
    CP7: A/B test

Each checkpoint returns a standardized CheckpointResult.
Uses ToolRegistry for all data access.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.agent.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)

__all__ = ["OpsCollector", "CheckpointResult"]


@dataclass
class CheckpointResult:
    """Standardized result from a single checkpoint collection."""
    checkpoint: str       # CP1-CP7
    name: str            # human-readable name
    status: str          # GREEN / YELLOW / RED
    measurements: Dict[str, Any] = field(default_factory=dict)
    anomalies: List[Dict[str, Any]] = field(default_factory=list)
    collected_at: str = ""
    error: Optional[str] = None

    def __post_init__(self):
        if not self.collected_at:
            self.collected_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "name": self.name,
            "status": self.status,
            "measurements": self.measurements,
            "anomalies": self.anomalies,
            "collected_at": self.collected_at,
            "error": self.error,
        }


class OpsCollector:
    """Collects measurements from 7 pipeline checkpoints.

    Args:
        registry: ToolRegistry for tool invocation.
        config: Checkpoint-specific config (thresholds, etc.)
        history: Previous checkpoint results for delta comparison.
    """

    def __init__(
        self,
        registry: "ToolRegistry",
        config: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self._registry = registry
        self._config = config or {}
        self._history = history or []

    def collect_all(self) -> List[CheckpointResult]:
        """Collect from all 7 checkpoints."""
        collectors = [
            self.collect_cp1_ingestion,
            self.collect_cp2_phase0,
            self.collect_cp3_training,
            self.collect_cp4_distillation,
            self.collect_cp5_serving_health,
            self.collect_cp6_recommendation,
            self.collect_cp7_ab_test,
        ]
        results = []
        for collector in collectors:
            try:
                result = collector()
                results.append(result)
            except Exception as e:
                cp_name = collector.__name__.replace("collect_", "").upper()
                logger.error("Checkpoint %s collection failed: %s", cp_name, e)
                results.append(CheckpointResult(
                    checkpoint=cp_name,
                    name=collector.__doc__ or cp_name,
                    status="RED",
                    error=str(e),
                ))
        return results

    def collect_cp1_ingestion(self) -> CheckpointResult:
        """CP1: Ingestion complete"""
        data = self._safe_call("read_ingestion_manifest")
        if data is None:
            return self._error_result("CP1", "인제스천", "Manifest not available")

        measurements = {
            "total_domains": data.get("total_domains", 0),
            "domains_passed": data.get("domains_passed", 0),
            "domains_failed": data.get("domains_failed", 0),
            "total_rows": data.get("total_rows", 0),
            "total_pii_encrypted": data.get("total_pii_encrypted", 0),
            "duration_seconds": data.get("total_duration_seconds", 0),
        }

        anomalies = []
        if data.get("domains_failed", 0) > 0:
            anomalies.append({"type": "domain_failure", "count": data["domains_failed"]})

        status = "RED" if anomalies else "GREEN"
        return CheckpointResult("CP1", "인제스천", status, measurements, anomalies)

    def collect_cp2_phase0(self) -> CheckpointResult:
        """CP2: Phase 0 complete"""
        state = self._safe_call("read_pipeline_state") or {}
        stats = self._safe_call("read_feature_stats") or {}

        measurements = {
            "completed_stages": state.get("completed_stages", []),
            "zero_variance_cols": stats.get("zero_variance_count", 0),
            "nan_ratio_max": stats.get("nan_ratio_max", 0.0),
            "total_features": stats.get("total_features", 0),
        }

        anomalies = []
        if measurements["zero_variance_cols"] > 0:
            anomalies.append({
                "type": "zero_variance",
                "count": measurements["zero_variance_cols"],
            })
        if measurements["nan_ratio_max"] > 0.3:
            anomalies.append({
                "type": "high_nan_ratio",
                "max_ratio": measurements["nan_ratio_max"],
            })

        status = "RED" if any(a["type"] == "high_nan_ratio" for a in anomalies) else (
            "YELLOW" if anomalies else "GREEN"
        )
        return CheckpointResult("CP2", "Phase 0", status, measurements, anomalies)

    def collect_cp3_training(self) -> CheckpointResult:
        """CP3: Training complete"""
        metrics = self._safe_call("read_experiment_metrics") or {}

        measurements = {
            "final_loss": metrics.get("final_loss"),
            "best_val_auc": metrics.get("best_val_auc"),
            "epochs_completed": metrics.get("epochs_completed", 0),
            "grad_norm_max": metrics.get("grad_norm_max"),
            "nan_loss_count": metrics.get("nan_loss_count", 0),
        }

        anomalies = []
        if measurements.get("nan_loss_count", 0) > 0:
            anomalies.append({"type": "nan_loss", "count": measurements["nan_loss_count"]})
        if measurements.get("grad_norm_max") and measurements["grad_norm_max"] > 100:
            anomalies.append({"type": "grad_explosion", "max_norm": measurements["grad_norm_max"]})

        status = "RED" if measurements.get("nan_loss_count", 0) > 0 else (
            "YELLOW" if anomalies else "GREEN"
        )
        return CheckpointResult("CP3", "학습", status, measurements, anomalies)

    def collect_cp4_distillation(self) -> CheckpointResult:
        """CP4: Distillation complete"""
        data = self._safe_call("read_distillation_fidelity") or {}

        measurements = {
            "task_fidelity": data.get("task_fidelity", {}),
            "max_gap": data.get("max_fidelity_gap", 0.0),
            "tasks_above_threshold": data.get("tasks_above_threshold", []),
        }

        anomalies = []
        if measurements["max_gap"] > 0.05:
            anomalies.append({
                "type": "fidelity_gap",
                "max_gap": measurements["max_gap"],
                "tasks": measurements["tasks_above_threshold"],
            })

        status = "YELLOW" if anomalies else "GREEN"
        return CheckpointResult("CP4", "증류", status, measurements, anomalies)

    def collect_cp5_serving_health(self) -> CheckpointResult:
        """CP5: Serving health"""
        health = self._safe_call("check_feature_store_health") or {}

        measurements = {
            "healthy": health.get("healthy", False),
            "backend": health.get("backend", "unknown"),
            "record_count": health.get("record_count", 0),
        }

        anomalies = []
        if not measurements["healthy"]:
            anomalies.append({"type": "unhealthy", "detail": health.get("error", "")})

        status = "RED" if not measurements["healthy"] else "GREEN"
        return CheckpointResult("CP5", "서빙 헬스", status, measurements, anomalies)

    def collect_cp6_recommendation(self) -> CheckpointResult:
        """CP6: Recommendation response.

        The ``p95_latency_ms`` field represents the warm-only predict
        Lambda p95 when the upstream tool delivers it that way (the
        AWS-backed pipeline_reports registry does). ``latency_sla_ms``
        accordingly refers to the warm-SLA; Bedrock-bound rewrite
        latencies belong in ``per_lambda[<l2a>]`` with its own SLA and
        are surfaced via ``extra_anomalies`` from the upstream tool.
        """
        data = self._safe_call("read_audit_archive") or {}

        measurements = {
            "p50_latency_ms": data.get("p50_latency_ms"),
            "p95_latency_ms": data.get("p95_latency_ms"),
            "filter_pass_rate": data.get("filter_pass_rate"),
            "total_requests": data.get("total_requests", 0),
            # Per-Lambda breakdown — opt-in payload from the AWS registry.
            # Stays empty when the legacy / local-file registry is used
            # so existing callers see unchanged behaviour.
            "per_lambda": data.get("per_lambda", {}),
        }

        anomalies: List[Dict[str, Any]] = list(
            data.get("extra_anomalies", []) or [],
        )
        sla = self._config.get("latency_sla_ms", 300)
        p95 = measurements.get("p95_latency_ms")
        if p95 and p95 > sla:
            anomalies.append({
                "type": "latency_sla_breach",
                "p95": p95,
                "sla": sla,
            })

        status = "RED" if anomalies else "GREEN"
        return CheckpointResult("CP6", "추천 응답", status, measurements, anomalies)

    def collect_cp7_ab_test(self) -> CheckpointResult:
        """CP7: A/B test"""
        data = self._safe_call("query_cloudwatch_metrics") or {}

        measurements = {
            "active_experiment": data.get("active_experiment"),
            "variant_metrics": data.get("variant_metrics", {}),
            "significance": data.get("significance_test"),
        }

        status = "GREEN"  # A/B tests are informational unless significance found
        anomalies = []
        if data.get("significance_test", {}).get("significant"):
            anomalies.append({
                "type": "significant_result",
                "winner": data["significance_test"].get("winner"),
                "p_value": data["significance_test"].get("p_value"),
            })
            status = "YELLOW"

        return CheckpointResult("CP7", "A/B 테스트", status, measurements, anomalies)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _safe_call(self, tool_name: str, params: Optional[Dict] = None) -> Optional[Any]:
        """Call a tool via registry, returning None on failure."""
        try:
            return self._registry.call(tool_name, params)
        except Exception as e:
            logger.warning("Tool '%s' call failed: %s", tool_name, e)
            return None

    def _error_result(self, cp: str, name: str, error: str) -> CheckpointResult:
        return CheckpointResult(checkpoint=cp, name=name, status="RED", error=error)
