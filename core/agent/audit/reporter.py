"""
Audit Reporter — Audit Report Generation
============================================

Generates structured audit reports in the focus_areas + regulatory_summary
+ reason_quality_dashboard format defined in the design doc.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.agent.audit.diagnoser import FocusArea

logger = logging.getLogger(__name__)

__all__ = ["AuditReporter", "AuditReport"]


@dataclass
class AuditReport:
    """Structured audit report."""
    generated_at: str = ""
    period: str = "weekly"
    audit_type: str = "automated"
    risk_level: str = "LOW"
    focus_areas: List[Dict[str, Any]] = field(default_factory=list)
    regulatory_summary: Dict[str, Any] = field(default_factory=dict)
    reason_quality_dashboard: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "audit_report": {
                "generated_at": self.generated_at,
                "period": self.period,
                "audit_type": self.audit_type,
                "risk_level": self.risk_level,
                "focus_areas": self.focus_areas,
                "regulatory_summary": self.regulatory_summary,
                "reason_quality_dashboard": self.reason_quality_dashboard,
                "metadata": self.metadata,
            }
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info("Audit report saved to %s", path)


class AuditReporter:
    """Generates structured audit reports with optional 3-agent consensus.

    Args:
        config: Reporter configuration.
        consensus_arbiter: Optional ConsensusArbiter for multi-agent verdict.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        consensus_arbiter: Optional[Any] = None,
    ) -> None:
        self._config = config or {}
        self._consensus = consensus_arbiter

    def generate(
        self,
        focus_areas: Optional[List["FocusArea"]] = None,
        regulatory_results: Optional[Dict[str, Any]] = None,
        reason_quality: Optional[Dict[str, Any]] = None,
        period: str = "weekly",
    ) -> AuditReport:
        """Generate an audit report.

        Args:
            focus_areas: From AuditDiagnoser.diagnose().
            regulatory_results: From regulatory compliance checks.
            reason_quality: From Tier1Aggregator + GroundingValidator.
            period: Report period label.
        """
        focus_areas = focus_areas or []

        # Determine risk level
        risk_level = self._determine_risk_level(focus_areas)

        # Build regulatory summary
        reg_summary = self._build_regulatory_summary(regulatory_results or {})

        # Build reason quality dashboard
        rq_dashboard = self._build_reason_dashboard(reason_quality or {})

        # 3-agent consensus on focus areas (if arbiter available)
        consensus_results = {}
        if self._consensus and focus_areas:
            for fa in focus_areas:
                try:
                    cr = self._consensus.evaluate(
                        item_description=fa.description,
                        measurements=fa.to_dict(),
                        rule_engine_verdict=fa.priority,
                    )
                    consensus_results[fa.area_id] = {
                        "verdict": cr.final_verdict,
                        "type": cr.consensus_type,
                        "minority_report": cr.minority_report,
                    }
                except Exception as _ce:
                    logger.warning("Consensus failed for %s: %s", fa.area_id, _ce)

        fa_dicts = []
        for fa in focus_areas:
            d = fa.to_dict()
            if fa.area_id in consensus_results:
                d["consensus"] = consensus_results[fa.area_id]
            fa_dicts.append(d)

        return AuditReport(
            period=period,
            risk_level=risk_level,
            focus_areas=fa_dicts,
            regulatory_summary=reg_summary,
            reason_quality_dashboard=rq_dashboard,
        )

    def _determine_risk_level(self, focus_areas: List["FocusArea"]) -> str:
        if any(fa.priority == "HIGH" for fa in focus_areas):
            return "HIGH"
        if any(fa.priority == "MEDIUM" for fa in focus_areas):
            return "MEDIUM"
        return "LOW"

    def _build_regulatory_summary(self, results: Dict) -> Dict[str, Any]:
        return {
            "domestic": results.get("domestic", {}),
            "eu_ai_act": results.get("eu_ai_act", {}),
            "fria": results.get("fria", {}),
        }

    def _build_reason_dashboard(self, quality: Dict) -> Dict[str, Any]:
        return {
            "tier1_auto": quality.get("tier1", {}),
            "tier2_sample": quality.get("tier2", {}),
            "tier3_expert": quality.get("tier3", {}),
        }
