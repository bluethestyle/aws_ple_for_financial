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
        case_store: Optional DiagnosticCaseStore for saving/searching audit cases.
        temporal_fact_store: Optional TemporalFactStore for recording audit events
            with valid_from timestamps and querying temporal trends.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        consensus_arbiter: Optional[Any] = None,
        case_store: Optional[Any] = None,
        temporal_fact_store: Optional[Any] = None,
    ) -> None:
        self._config = config or {}
        self._consensus = consensus_arbiter
        self._case_store = case_store              # DiagnosticCaseStore | None
        self._temporal_fact_store = temporal_fact_store  # TemporalFactStore | None

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
                    self._case_store = False
        obj = self._case_store
        return obj if obj is not False else None

    def _get_temporal_fact_store(self) -> Optional[Any]:
        """Lazy-init TemporalFactStore if not provided at construction time."""
        if self._temporal_fact_store is None:
            store_path = self._config.get("temporal_fact_store_path", "")
            if store_path:
                try:
                    from core.agent.temporal_fact_store import TemporalFactStore
                    self._temporal_fact_store = TemporalFactStore(store_path=store_path)
                except Exception as e:
                    logger.warning("TemporalFactStore init failed: %s", e)
                    self._temporal_fact_store = False
        obj = self._temporal_fact_store
        return obj if obj is not False else None

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

        # 3-agent consensus on focus areas (if arbiter available). As
        # with the Ops reporter, we keep every per-vote reasoning so
        # the audit trail shows *why* the Regulator / Risk / AuditTrail
        # personas reached their verdicts, including dissenting
        # positions that would otherwise vanish behind the majority
        # label.
        consensus_results: Dict[str, Dict[str, Any]] = {}
        if self._consensus and focus_areas:
            for fa in focus_areas:
                try:
                    cr = self._consensus.evaluate(
                        item_description=fa.finding,
                        measurements=fa.to_dict(),
                        rule_engine_verdict=fa.priority,
                    )
                    consensus_results[fa.area] = {
                        "verdict": cr.final_verdict,
                        "type": cr.consensus_type,
                        "minority_report": cr.minority_report,
                        "votes": [
                            {
                                "agent_id": v.agent_id,
                                "perspective": v.perspective,
                                "verdict": v.verdict,
                                "confidence": v.confidence,
                                "reasoning": v.reasoning,
                            }
                            for v in cr.votes
                        ],
                    }
                except Exception as _ce:
                    logger.warning("Consensus failed for %s: %s", fa.area, _ce)

        # Search similar past audit cases for context (non-fatal)
        try:
            cs = self._get_case_store()
            if cs is not None and cs.case_count > 0 and focus_areas:
                import numpy as _np
                # Query vector: encode priority as numeric (HIGH=1, MEDIUM=0.5, LOW=0)
                priority_map = {"HIGH": 1.0, "MEDIUM": 0.5, "LOW": 0.0}
                q_vals = [priority_map.get(fa.priority, 0.0) for fa in focus_areas]
                dim = cs._embedding_dim
                q_arr = _np.array(q_vals, dtype=_np.float32)
                if len(q_arr) < dim:
                    q_arr = _np.pad(q_arr, (0, dim - len(q_arr)))
                else:
                    q_arr = q_arr[:dim]
                similar = cs.search_similar(q_arr, k=3, pipeline_part=None)
                if similar:
                    logger.info(
                        "AuditReporter: found %d similar past audit cases (top score=%.3f)",
                        len(similar), similar[0][1],
                    )
        except Exception:
            logger.debug("DiagnosticCaseStore audit search failed (non-fatal)", exc_info=True)

        fa_dicts = []
        for fa in focus_areas:
            d = fa.to_dict()
            if fa.area in consensus_results:
                d["consensus"] = consensus_results[fa.area]
            fa_dicts.append(d)

        report = AuditReport(
            period=period,
            risk_level=risk_level,
            focus_areas=fa_dicts,
            regulatory_summary=reg_summary,
            reason_quality_dashboard=rq_dashboard,
        )

        # Save audit findings as diagnostic cases (non-fatal)
        try:
            cs = self._get_case_store()
            if cs is not None and fa_dicts:
                for fa_d in fa_dicts:
                    case = {
                        "agent": "AuditAgent",
                        "pipeline_part": fa_d.get("area", ""),
                        "check_item": fa_d.get("area", ""),
                        "verdict": fa_d.get("priority", "LOW"),
                        "severity": fa_d.get("priority", "LOW"),
                        "finding": fa_d.get("finding", ""),
                        "likely_cause": "",
                        "suggested_action": "",
                        "metrics": {},
                        "consensus_type": fa_d.get("consensus", {}).get("type", "none"),
                    }
                    cs.save_case(case)
                # Flush to disk — save_case only appends in-memory.
                try:
                    cs.save()
                except Exception:
                    logger.debug(
                        "DiagnosticCaseStore.save() flush failed (non-fatal)",
                        exc_info=True,
                    )
                logger.info(
                    "DiagnosticCaseStore: saved %d audit focus areas",
                    len(fa_dicts),
                )
        except Exception:
            logger.debug("DiagnosticCaseStore audit save failed (non-fatal)", exc_info=True)

        # Record audit event in TemporalFactStore for trend analysis (non-fatal)
        try:
            tfs = self._get_temporal_fact_store()
            if tfs is not None:
                tfs.save_fact({
                    "entity_type": "audit_report",
                    "entity_id": f"audit_{period}",
                    "attribute": "risk_level",
                    "value": risk_level,
                    "source": "AuditAgent",
                })
                # Record each focus area as a temporal fact
                for fa_d in fa_dicts:
                    tfs.save_fact({
                        "entity_type": "audit_focus_area",
                        "entity_id": fa_d.get("area", "unknown"),
                        "attribute": "priority",
                        "value": fa_d.get("priority", "LOW"),
                        "source": "AuditAgent",
                    })
                try:
                    tfs.save()
                except Exception:
                    logger.debug(
                        "TemporalFactStore.save() flush failed (non-fatal)",
                        exc_info=True,
                    )
                logger.info(
                    "TemporalFactStore: recorded audit event + %d focus areas",
                    len(fa_dicts),
                )
        except Exception:
            logger.debug("TemporalFactStore audit record failed (non-fatal)", exc_info=True)

        return report

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
