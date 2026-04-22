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
    """Generates structured ops reports with optional 3-agent consensus.

    Args:
        config: Reporter configuration.
        consensus_arbiter: Optional ConsensusArbiter for multi-agent verdict.
            When provided, non-GREEN checkpoints and CRITICAL/WARNING diagnoses
            are evaluated via 3-agent consensus (unanimous PASS required).
        case_store: Optional DiagnosticCaseStore instance. When provided,
            attention items are saved as diagnostic cases after report generation.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        consensus_arbiter: Optional[Any] = None,
        case_store: Optional[Any] = None,
    ) -> None:
        self._config = config or {}
        self._consensus = consensus_arbiter
        self._case_store = case_store  # DiagnosticCaseStore | None

    def _get_case_store(self) -> Optional[Any]:
        """Lazy-init DiagnosticCaseStore if not provided at construction time."""
        if self._case_store is None:
            store_path = self._config.get("case_store_path", "")
            if store_path:
                try:
                    from core.agent.case_store import DiagnosticCaseStore
                    self._case_store = DiagnosticCaseStore(store_path=store_path)
                except Exception as e:
                    logger.warning("DiagnosticCaseStore init failed: %s", e)
                    self._case_store = False  # sentinel: skip future attempts
        obj = self._case_store
        return obj if obj is not False else None

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

        # 3-agent consensus on attention items (if arbiter available)
        if self._consensus and attention:
            for item in attention:
                try:
                    consensus_result = self._consensus.evaluate(
                        item_description=item.get("finding", ""),
                        measurements=item,
                        rule_engine_verdict=item.get("severity", "WARNING"),
                    )
                    item["consensus_verdict"] = consensus_result.final_verdict
                    item["consensus_type"] = consensus_result.consensus_type
                    if consensus_result.minority_report:
                        item["minority_report"] = consensus_result.minority_report
                except Exception as _ce:
                    logger.warning("Consensus failed for %s: %s", item.get("checkpoint"), _ce)

        # Sort: CRITICAL/FAIL first
        severity_order = {"CRITICAL": 0, "FAIL": 1, "WARNING": 2, "INFO": 3}
        attention.sort(key=lambda a: severity_order.get(a.get("severity", "INFO"), 99))

        # Build all_checkpoints summary.
        # Keep every scalar measurement plus any structured detail
        # explicitly attached (e.g. per-Lambda latency breakdowns on
        # CP6). The earlier ``[:3]`` truncation silently dropped these
        # richer fields — observed 2026-04-22 when the AWS-backed CP6
        # per_lambda payload disappeared from the report output.
        all_cps = {}
        for cp in checkpoints:
            summary = {"status": cp.status}
            if cp.measurements:
                for k, v in cp.measurements.items():
                    summary[k] = v
            if cp.error:
                summary["error"] = cp.error
            all_cps[cp.checkpoint] = summary

        report = OpsReport(
            period=period,
            status=overall,
            attention_required=attention,
            all_checkpoints=all_cps,
            diagnoses=[d.to_dict() for d in diagnoses],
        )

        # Save attention items as diagnostic cases (non-fatal)
        try:
            cs = self._get_case_store()
            if cs is not None and attention:
                for item in attention:
                    case = {
                        "agent": "OpsAgent",
                        "pipeline_part": item.get("checkpoint", ""),
                        "check_item": item.get("checkpoint", ""),
                        "verdict": item.get("severity", "WARNING"),
                        "severity": item.get("severity", "WARNING"),
                        "finding": item.get("finding", ""),
                        "likely_cause": item.get("likely_cause", ""),
                        "suggested_action": item.get("suggested_action", ""),
                        "metrics": {},
                        "consensus_type": item.get("consensus_type", "none"),
                    }
                    cs.save_case(case)
                logger.debug("DiagnosticCaseStore: saved %d ops attention items", len(attention))
        except Exception:
            logger.debug("DiagnosticCaseStore save failed (non-fatal)", exc_info=True)

        return report

    def _summarize_anomalies(self, cp: "CheckpointResult") -> str:
        """Generate a finding summary from checkpoint anomalies."""
        if cp.error:
            return f"{cp.name}: {cp.error}"
        if cp.anomalies:
            types = [a.get("type", "unknown") for a in cp.anomalies]
            return f"{cp.name}: {', '.join(types)}"
        return f"{cp.name}: status {cp.status}"
