"""
EU AI Act Annex IV technical documentation mapper (Sprint 2 S10).

EU AI Act Article 11 + Annex IV requires providers of high-risk AI systems
to compile a technical documentation package before putting the system on
the market. Annex IV enumerates 12 information items (Sections 1-12).

This module maps each of the 12 sections to a concrete **evidence source**
in the repository (code path, config key, or document path) so that a
conformity-assessment auditor can follow the chain from the regulatory
item to a verifiable artefact.

The existing `core.monitoring.eu_ai_act_mapper.EUAIActMapper` covers
Articles 9-17 compliance; Annex IV is a separate documentation obligation
(Art. 11), so this module is kept distinct.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "AnnexIVSection",
    "AnnexIVEvidenceCheck",
    "AnnexIVReport",
    "AnnexIVMapper",
    "ANNEX_IV_SECTIONS",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnnexIVSection:
    section_id: str        # "1", "2", ..., "12"
    title: str
    description: str
    evidence_sources: List[str] = field(default_factory=list)


@dataclass
class AnnexIVEvidenceCheck:
    section_id: str
    title: str
    evidence_sources: List[str]
    resolved_evidence: List[Dict[str, Any]]
    is_documented: bool
    gap: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnnexIVReport:
    assessed_at: str
    system_name: str
    total_sections: int
    documented_count: int
    gaps: List[str]
    checks: List[AnnexIVEvidenceCheck]

    @property
    def coverage_rate(self) -> float:
        return self.documented_count / self.total_sections if self.total_sections else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assessed_at": self.assessed_at,
            "system_name": self.system_name,
            "total_sections": self.total_sections,
            "documented_count": self.documented_count,
            "coverage_rate": round(self.coverage_rate, 4),
            "gaps": list(self.gaps),
            "checks": [c.to_dict() for c in self.checks],
        }


# ---------------------------------------------------------------------------
# Default 12-section catalog
# ---------------------------------------------------------------------------

ANNEX_IV_SECTIONS: List[AnnexIVSection] = [
    AnnexIVSection(
        section_id="1",
        title="General description of the AI system",
        description="Intended purpose, provider identity, system version, "
                    "interaction with hardware/software, form in which the "
                    "system is placed on the market.",
        evidence_sources=[
            "doc:paper/typst/paper2.typ",
            "doc:README.md",
        ],
    ),
    AnnexIVSection(
        section_id="2",
        title="Detailed description of elements and development process",
        description="System architecture, training methodology, design "
                    "choices, logical diagrams.",
        evidence_sources=[
            "doc:paper/typst/paper1.typ",
            "module:core.model.ple.model",
            "doc:docs/aws_build_plan.md",
        ],
    ),
    AnnexIVSection(
        section_id="3",
        title="Monitoring, functioning, and control of the AI system",
        description="Capabilities and limitations in performance, degree of "
                    "accuracy, foreseeable unintended outcomes.",
        evidence_sources=[
            "module:core.monitoring.drift_detector",
            "module:core.monitoring.fairness_monitor",
            "module:core.monitoring.incident_reporter",
            "config:monitoring",
        ],
    ),
    AnnexIVSection(
        section_id="4",
        title="Risk management system",
        description="Reference to Art. 9 risk management system description.",
        evidence_sources=[
            "module:core.monitoring.fria_evaluator",
            "module:core.compliance.fria_assessment",
            "module:core.compliance.ai_risk_classifier",
            "config:compliance.fria",
        ],
    ),
    AnnexIVSection(
        section_id="5",
        title="Description of changes made to the system through its lifecycle",
        description="Version history, retraining events, rollback records.",
        evidence_sources=[
            "module:core.serving.model_registry",
            "module:core.monitoring.lineage_tracker",
            "module:core.monitoring.audit_logger",
        ],
    ),
    AnnexIVSection(
        section_id="6",
        title="List of harmonised standards applied",
        description="Full or partial application; evidence of conformity.",
        evidence_sources=[
            "doc:docs/pipeline_comparison_matrix.md",
            "doc:paper/typst/paper2.typ",
        ],
    ),
    AnnexIVSection(
        section_id="7",
        title="EU declaration of conformity",
        description="Copy of the Art. 47 declaration (template provided by "
                    "the provider).",
        evidence_sources=[
            "doc:docs/eu_declaration_of_conformity.md",  # not yet present
        ],
    ),
    AnnexIVSection(
        section_id="8",
        title="Post-market monitoring system",
        description="Monitoring plan, data collection, analysis methods, "
                    "corrective measures.",
        evidence_sources=[
            "module:core.monitoring.model_monitor",  # legacy placeholder
            "module:core.monitoring.drift_detector",
            "module:core.monitoring.audit_logger",
            "config:monitoring.governance_report",
        ],
    ),
    AnnexIVSection(
        section_id="9",
        title="Training, validation, and testing datasets description",
        description="Provenance, scope, main characteristics, data "
                    "preparation, labelling procedures, cleaning.",
        evidence_sources=[
            "module:core.monitoring.lineage_tracker",
            "config:data",
            "doc:docs/aws_build_plan.md",
        ],
    ),
    AnnexIVSection(
        section_id="10",
        title="Cybersecurity measures",
        description="Security features to protect the system against attacks "
                    "(Art. 15).",
        evidence_sources=[
            "module:core.security",
            "module:core.monitoring.audit_logger",  # HMAC + hash chain
        ],
    ),
    AnnexIVSection(
        section_id="11",
        title="Human oversight measures",
        description="Organizational and technical measures enabling human "
                    "oversight (Art. 14).",
        evidence_sources=[
            "module:core.serving.review.human_review_queue",
            "module:core.recommendation.fallback_router",   # Layer 4
            "module:core.serving.kill_switch",
            "config:serving.review",
            "config:serving.competition",                    # auto_promote=false
        ],
    ),
    AnnexIVSection(
        section_id="12",
        title="Accuracy, robustness, and cybersecurity specification",
        description="Expected metrics and thresholds; Art. 15 requirements.",
        evidence_sources=[
            "module:core.evaluation.model_competition",
            "module:core.evaluation.evaluator",
            "config:evaluation",
            "config:distillation.teacher_threshold",
            "config:distillation.fidelity",
        ],
    ),
]

assert len(ANNEX_IV_SECTIONS) == 12, (
    f"ANNEX_IV_SECTIONS must hold exactly 12 sections, got {len(ANNEX_IV_SECTIONS)}"
)


# ---------------------------------------------------------------------------
# Mapper
# ---------------------------------------------------------------------------

class AnnexIVMapper:
    """Resolves Annex IV evidence references and produces a coverage report."""

    def __init__(
        self,
        sections: Optional[List[AnnexIVSection]] = None,
        project_root: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._sections = list(sections or ANNEX_IV_SECTIONS)
        self._root = project_root or Path(".").resolve()
        self._config = config or {}

    def check_section(self, section: AnnexIVSection) -> AnnexIVEvidenceCheck:
        resolved: List[Dict[str, Any]] = []
        missing: List[str] = []
        for source in section.evidence_sources:
            status = self._resolve(source)
            resolved.append(status)
            if not status["ok"]:
                missing.append(source)
        is_documented = bool(resolved) and not missing
        gap = ""
        if missing:
            gap = f"Missing evidence: {', '.join(missing)}"
        return AnnexIVEvidenceCheck(
            section_id=section.section_id,
            title=section.title,
            evidence_sources=list(section.evidence_sources),
            resolved_evidence=resolved,
            is_documented=is_documented,
            gap=gap,
        )

    def generate_report(
        self, system_name: str = "AIOps PLE Platform",
    ) -> AnnexIVReport:
        checks = [self.check_section(s) for s in self._sections]
        documented = sum(1 for c in checks if c.is_documented)
        gaps = [f"Section {c.section_id} ({c.title}): {c.gap}"
                for c in checks if not c.is_documented]
        now = datetime.now(timezone.utc).isoformat()
        return AnnexIVReport(
            assessed_at=now,
            system_name=system_name,
            total_sections=len(checks),
            documented_count=documented,
            gaps=gaps,
            checks=checks,
        )

    # ------------------------------------------------------------------
    # Resolver
    # ------------------------------------------------------------------

    def _resolve(self, source: str) -> Dict[str, Any]:
        try:
            if source.startswith("module:"):
                mod_name = source[len("module:"):]
                importlib.import_module(mod_name)
                return {
                    "source": source, "type": "module",
                    "ok": True, "target": mod_name,
                }
            if source.startswith("doc:"):
                doc_path = source[len("doc:"):]
                p = self._root / doc_path
                ok = p.exists()
                return {
                    "source": source, "type": "doc",
                    "ok": ok, "target": str(p),
                }
            if source.startswith("config:"):
                key_path = source[len("config:"):]
                value, ok = self._get_config_key(key_path)
                return {
                    "source": source, "type": "config",
                    "ok": ok, "target": key_path,
                    "value_summary": _summarise(value) if ok else None,
                }
            return {
                "source": source, "type": "unknown",
                "ok": False, "target": source,
            }
        except Exception as exc:  # pragma: no cover (defensive)
            logger.debug("Annex IV resolve failed for %s: %s", source, exc)
            return {
                "source": source, "type": "error", "ok": False,
                "target": source, "error": str(exc),
            }

    def _get_config_key(self, key_path: str) -> tuple:
        cursor: Any = self._config
        for part in key_path.split("."):
            if isinstance(cursor, dict) and part in cursor:
                cursor = cursor[part]
            else:
                return None, False
        return cursor, True


def _summarise(value: Any) -> str:
    s = repr(value)
    return s[:120] + "..." if len(s) > 120 else s
