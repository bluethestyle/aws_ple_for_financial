"""
Ops Reporter — Template-Based Report Generation
==================================================

Generates structured ops reports in the finding + likely_cause + suggested_action format.
Supports YAML/JSON output for downstream consumption.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.agent.ops.collector import CheckpointResult
    from core.agent.ops.diagnoser import Diagnosis

logger = logging.getLogger(__name__)

__all__ = ["OpsReporter", "OpsReport"]


@dataclass
class OpsReport:
    """Structured ops report."""
    generated_at: str = ""
    period: str = "daily"
    status: str = "GREEN"
    attention_required: List[Dict[str, Any]] = field(default_factory=list)
    all_checkpoints: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    diagnoses: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ops_report": {
                "generated_at": self.generated_at,
                "period": self.period,
                "status": self.status,
                "attention_required": self.attention_required,
                "all_checkpoints": self.all_checkpoints,
                "diagnoses": self.diagnoses,
                "metadata": self.metadata,
            }
        }

    def save(self, path: str) -> None:
        """Save report as JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info("Ops report saved to %s", path)


class OpsReporter:
    """Generates structured ops reports.

    Args:
        config: Reporter configuration.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._config = config or {}

    def generate(
        self,
        checkpoints: List["CheckpointResult"],
        diagnoses: Optional[List["Diagnosis"]] = None,
        period: str = "daily",
    ) -> OpsReport:
        """Generate an ops report from checkpoint results and diagnoses.

        Args:
            checkpoints: Results from OpsCollector.
            diagnoses: Cross-checkpoint diagnoses from OpsDiagnoser.
            period: Report period label.
        """
        diagnoses = diagnoses or []

        # Overall status: worst of all checkpoints
        statuses = [cp.status for cp in checkpoints]
        if "RED" in statuses:
            overall = "RED"
        elif "YELLOW" in statuses:
            overall = "YELLOW"
        else:
            overall = "GREEN"

        # Escalate if CRITICAL diagnosis exists
        if any(d.severity == "CRITICAL" for d in diagnoses):
            overall = "RED"

        # Build attention_required from non-GREEN checkpoints + diagnoses
        attention = []
        for cp in checkpoints:
            if cp.status != "GREEN":
                attention.append({
                    "checkpoint": cp.checkpoint,
                    "severity": "FAIL" if cp.status == "RED" else "WARNING",
                    "finding": self._summarize_anomalies(cp),
                })

        for diag in diagnoses:
            attention.append({
                "checkpoint": ", ".join(diag.checkpoints),
                "severity": diag.severity,
                "finding": diag.finding,
                "likely_cause": diag.likely_cause,
                "suggested_action": diag.suggested_action,
            })

        # Sort: CRITICAL/FAIL first
        severity_order = {"CRITICAL": 0, "FAIL": 1, "WARNING": 2, "INFO": 3}
        attention.sort(key=lambda a: severity_order.get(a.get("severity", "INFO"), 99))

        # Build all_checkpoints summary
        all_cps = {}
        for cp in checkpoints:
            summary = {"status": cp.status}
            # Add key measurement as detail
            if cp.measurements:
                key_metrics = list(cp.measurements.items())[:3]
                for k, v in key_metrics:
                    summary[k] = v
            if cp.error:
                summary["error"] = cp.error
            all_cps[cp.checkpoint] = summary

        return OpsReport(
            period=period,
            status=overall,
            attention_required=attention,
            all_checkpoints=all_cps,
            diagnoses=[d.to_dict() for d in diagnoses],
        )

    def _summarize_anomalies(self, cp: "CheckpointResult") -> str:
        """Generate a finding summary from checkpoint anomalies."""
        if cp.error:
            return f"{cp.name}: {cp.error}"
        if cp.anomalies:
            types = [a.get("type", "unknown") for a in cp.anomalies]
            return f"{cp.name}: {', '.join(types)}"
        return f"{cp.name}: status {cp.status}"
