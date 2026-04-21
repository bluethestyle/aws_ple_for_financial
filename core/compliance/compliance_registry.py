"""
ComplianceRegistry - 36-item regulatory compliance catalog (M8).

A registered catalog of 36 compliance items spanning the project:
- **A-group (18 items)**: items with existing implementation in the AWS
  codebase. The registry points to concrete file paths / config keys
  / module imports so that an auditor can follow the chain.
- **GAP-group (18 items)**: items that are KNOWN gaps -- either being
  implemented (Sprint 0~3 of this plan) or accepted as "won't" in the
  AWS context (typically infra / org-specific).

Each :class:`ComplianceItem` declares a ``check_type`` that the registry
can execute:
- ``file_exists``   : the configured ``path`` must exist on disk.
- ``config_key``    : ``path`` (dot-delimited) must resolve to a truthy
  value in pipeline.yaml.
- ``module_exists`` : ``path`` must be an importable Python module.
- ``custom_check``  : caller supplies a callable via ``register_check``.

Unknown / not-applicable check types are recorded as ``not_applicable``.

Results are persisted as ComplianceEvent entries so a quarterly / annual
inspection can reconstruct the compliance state at any point in time.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import yaml

from core.compliance.store import ComplianceStore
from core.compliance.types import (
    ComplianceEvent,
    EventType,
    new_event_id,
    utcnow,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ComplianceItem",
    "ComplianceItemResult",
    "ComplianceRegistry",
    "DEFAULT_CATALOG",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ComplianceItem:
    item_id: str                # "A-01" / "GAP-01"
    group: str                  # "A" | "GAP"
    description: str
    legal_basis: List[str]
    check_type: str             # file_exists | config_key | module_exists | custom_check
    check_params: Dict[str, Any] = field(default_factory=dict)
    severity: str = "medium"    # critical | high | medium | low
    owner: str = ""             # who is responsible


@dataclass
class ComplianceItemResult:
    item_id: str
    status: str                 # compliant | non_compliant | not_applicable | error
    checked_at: datetime
    message: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["checked_at"] = self.checked_at.isoformat()
        return d


# ---------------------------------------------------------------------------
# Default catalog (36 items)
# ---------------------------------------------------------------------------

# A-group: already implemented. The registry points to actual code
# evidence so an auditor can open the file and verify.
A_GROUP: List[ComplianceItem] = [
    ComplianceItem(
        item_id="A-01", group="A",
        description="Compliance audit store (7 tables) implemented",
        legal_basis=["AI기본법 §34", "금감원 원칙 5"],
        check_type="module_exists",
        check_params={"module": "core.compliance.audit_store"},
        severity="high", owner="compliance-eng",
    ),
    ComplianceItem(
        item_id="A-02", group="A",
        description="Marketing consent manager (channel-level)",
        legal_basis=["개보법 §22", "금소법 §17"],
        check_type="module_exists",
        check_params={"module": "core.compliance.consent_manager"},
        severity="critical", owner="compliance-eng",
    ),
    ComplianceItem(
        item_id="A-03", group="A",
        description="AI decision opt-out registered",
        legal_basis=["AI기본법 §31", "개보법 §37의2"],
        check_type="module_exists",
        check_params={"module": "core.compliance.ai_opt_out"},
        severity="critical", owner="compliance-eng",
    ),
    ComplianceItem(
        item_id="A-04", group="A",
        description="Profiling rights manager (access/correct/delete)",
        legal_basis=["신정법 §36의2", "개보법 §35-37"],
        check_type="module_exists",
        check_params={"module": "core.compliance.profiling_rights"},
        severity="critical", owner="compliance-eng",
    ),
    ComplianceItem(
        item_id="A-05", group="A",
        description="Regulatory compliance checker (20 baseline items)",
        legal_basis=["금소법", "개보법", "AI기본법"],
        check_type="module_exists",
        check_params={"module": "core.compliance.regulatory_checker"},
        severity="high", owner="compliance-eng",
    ),
    ComplianceItem(
        item_id="A-06", group="A",
        description="Kill switch (runtime disable)",
        legal_basis=["AI기본법 §34 ③ 안전성"],
        check_type="module_exists",
        check_params={"module": "core.serving.kill_switch"},
        severity="critical", owner="serving-eng",
    ),
    ComplianceItem(
        item_id="A-07", group="A",
        description="Audit logger with HMAC + hash chain",
        legal_basis=["AI기본법 §34", "SR 11-7 (MRM)"],
        check_type="module_exists",
        check_params={"module": "core.monitoring.audit_logger"},
        severity="critical", owner="monitoring-eng",
    ),
    ComplianceItem(
        item_id="A-08", group="A",
        description="Fairness monitor (DI / SPD / EOD thresholds)",
        legal_basis=["금소법 §15 차별금지", "EU AI Act Art. 10"],
        check_type="module_exists",
        check_params={"module": "core.monitoring.fairness_monitor"},
        severity="high", owner="monitoring-eng",
    ),
    ComplianceItem(
        item_id="A-09", group="A",
        description="Drift detector (PSI-based)",
        legal_basis=["SR 11-7 ongoing monitoring"],
        check_type="module_exists",
        check_params={"module": "core.monitoring.drift_detector"},
        severity="high", owner="monitoring-eng",
    ),
    ComplianceItem(
        item_id="A-10", group="A",
        description="EU AI Act classification + Annex IV mapper",
        legal_basis=["EU AI Act Art. 6 / Annex IV"],
        check_type="module_exists",
        check_params={"module": "core.monitoring.eu_ai_act_mapper"},
        severity="high", owner="compliance-eng",
    ),
    ComplianceItem(
        item_id="A-11", group="A",
        description="EU-FRIA evaluator (Art. 9 risk mgmt)",
        legal_basis=["EU AI Act Art. 9"],
        check_type="module_exists",
        check_params={"module": "core.monitoring.fria_evaluator"},
        severity="high", owner="compliance-eng",
    ),
    ComplianceItem(
        item_id="A-12", group="A",
        description="Model card generator",
        legal_basis=["EU AI Act Annex IV §2(d)", "SR 11-7"],
        check_type="module_exists",
        check_params={"module": "core.evaluation.model_card"},
        severity="medium", owner="evaluation-eng",
    ),
    ComplianceItem(
        item_id="A-13", group="A",
        description="Champion-Challenger ModelCompetition gate",
        legal_basis=["SR 11-7 effective challenge"],
        check_type="module_exists",
        check_params={"module": "core.evaluation.model_competition"},
        severity="critical", owner="evaluation-eng",
    ),
    ComplianceItem(
        item_id="A-14", group="A",
        description="Lineage tracker (data + model provenance)",
        legal_basis=["EU AI Act Annex IV §2(e)"],
        check_type="module_exists",
        check_params={"module": "core.monitoring.lineage_tracker"},
        severity="high", owner="monitoring-eng",
    ),
    ComplianceItem(
        item_id="A-15", group="A",
        description="PIA evaluator (개보법 영향평가)",
        legal_basis=["개보법 §33", "개보법 시행령 §35"],
        check_type="module_exists",
        check_params={"module": "core.monitoring.pia_evaluator"},
        severity="high", owner="compliance-eng",
    ),
    ComplianceItem(
        item_id="A-16", group="A",
        description="Governance report weekly artifact",
        legal_basis=["AI기본법 §34", "SR 11-7"],
        check_type="config_key",
        check_params={"path": "monitoring.governance_report.enabled"},
        severity="medium", owner="monitoring-eng",
    ),
    ComplianceItem(
        item_id="A-17", group="A",
        description="3-layer rule engine fallback (serving)",
        legal_basis=["AI기본법 §34 안전성", "금소법 §17"],
        check_type="config_key",
        check_params={"path": "rule_engine.enabled"},
        severity="critical", owner="serving-eng",
    ),
    ComplianceItem(
        item_id="A-18", group="A",
        description="LLM provider safe-mode defaults",
        legal_basis=["AI기본법 §31 생성 표시 의무"],
        check_type="config_key",
        check_params={"path": "llm_provider"},
        severity="medium", owner="serving-eng",
    ),
]

# GAP-group: the sync gaps identified in docs/aws_work_plan.md. Each item
# maps to a Sprint deliverable (M1~M12, S*) or is explicitly "won't".
GAP_GROUP: List[ComplianceItem] = [
    ComplianceItem(
        item_id="GAP-01", group="GAP",
        description="Human Review Queue (M1)",
        legal_basis=["AI기본법 §34 인간 감독"],
        check_type="module_exists",
        check_params={"module": "core.serving.review.human_review_queue"},
        severity="critical", owner="serving-eng",
    ),
    ComplianceItem(
        item_id="GAP-02", group="GAP",
        description="Kill switch expansion (global/task/cluster) (M2)",
        legal_basis=["AI기본법 §34 안전성"],
        check_type="custom_check",
        check_params={"check_name": "kill_switch_tiers"},
        severity="high", owner="serving-eng",
    ),
    ComplianceItem(
        item_id="GAP-03", group="GAP",
        description="Consent channel config-driven 5-channel (M3)",
        legal_basis=["개보법 §22", "금소법 §17"],
        check_type="config_key",
        check_params={"path": "compliance.consent.channels"},
        severity="critical", owner="compliance-eng",
    ),
    ComplianceItem(
        item_id="GAP-04", group="GAP",
        description="Opt-out Manager with SLA (M4)",
        legal_basis=["개보법 §37의2", "AI기본법 §31"],
        check_type="module_exists",
        check_params={"module": "core.compliance.rights.opt_out"},
        severity="critical", owner="compliance-eng",
    ),
    ComplianceItem(
        item_id="GAP-05", group="GAP",
        description="Profiling rights workflow + SLA (M5)",
        legal_basis=["신정법 §36의2"],
        check_type="module_exists",
        check_params={"module": "core.compliance.rights.profiling"},
        severity="critical", owner="compliance-eng",
    ),
    ComplianceItem(
        item_id="GAP-06", group="GAP",
        description="Explanation SLA Tracker (M6)",
        legal_basis=["개보법 시행령 §44의2~4"],
        check_type="module_exists",
        check_params={"module": "core.compliance.rights.explanation_sla"},
        severity="high", owner="compliance-eng",
    ),
    ComplianceItem(
        item_id="GAP-07", group="GAP",
        description="Korean AI기본법 FRIA (M7)",
        legal_basis=["AI기본법 §35 ②③"],
        check_type="module_exists",
        check_params={"module": "core.compliance.fria_assessment"},
        severity="critical", owner="compliance-eng",
    ),
    ComplianceItem(
        item_id="GAP-08", group="GAP",
        description="36-item Compliance Registry (M8)",
        legal_basis=["AI기본법 §34", "금감원 RMF"],
        check_type="module_exists",
        check_params={"module": "core.compliance.compliance_registry"},
        severity="high", owner="compliance-eng",
    ),
    ComplianceItem(
        item_id="GAP-09", group="GAP",
        description="금감원 RMF 6-dim AI Risk Classifier (M9)",
        legal_basis=["금감원 AI RMF 2024"],
        check_type="module_exists",
        check_params={"module": "core.compliance.ai_risk_classifier"},
        severity="high", owner="compliance-eng",
    ),
    ComplianceItem(
        item_id="GAP-10", group="GAP",
        description="Dynamic item universe loader (M10)",
        legal_basis=["금소법 §17 적합성"],
        check_type="module_exists",
        check_params={"module": "core.recommendation.universe.dynamic_loader"},
        severity="medium", owner="serving-eng",
    ),
    ComplianceItem(
        item_id="GAP-11", group="GAP",
        description="Audit archive extended columns (M11)",
        legal_basis=["AI기본법 §34 ⑤ 감사"],
        check_type="custom_check",
        check_params={"check_name": "audit_archive_schema"},
        severity="medium", owner="monitoring-eng",
    ),
    ComplianceItem(
        item_id="GAP-12", group="GAP",
        description="LLM generation marker auto-insertion (M12)",
        legal_basis=["AI기본법 §31 생성 표시 의무"],
        check_type="config_key",
        check_params={"path": "compliance.llm_marker.enabled"},
        severity="high", owner="serving-eng",
    ),
    ComplianceItem(
        item_id="GAP-13", group="GAP",
        description="Human fallback router (Should S1)",
        legal_basis=["AI기본법 §34 인간 감독"],
        check_type="module_exists",
        check_params={"module": "core.recommendation.human_fallback_router"},
        severity="medium", owner="serving-eng",
    ),
    ComplianceItem(
        item_id="GAP-14", group="GAP",
        description="Fairness persistence (Should S7)",
        legal_basis=["금소법 §15 차별금지"],
        check_type="custom_check",
        check_params={"check_name": "fairness_persistence"},
        severity="medium", owner="monitoring-eng",
    ),
    ComplianceItem(
        item_id="GAP-15", group="GAP",
        description="L2a safety gate 3-layer (Should S11)",
        legal_basis=["AI기본법 §31 설명 품질"],
        check_type="custom_check",
        check_params={"check_name": "l2a_safety_gate"},
        severity="medium", owner="serving-eng",
    ),
    ComplianceItem(
        item_id="GAP-16", group="GAP",
        description="Counterfactual C-C gate (Should S14)",
        legal_basis=["Pearl Rung 3 + SR 11-7"],
        check_type="custom_check",
        check_params={"check_name": "counterfactual_gate"},
        severity="medium", owner="evaluation-eng",
    ),
    ComplianceItem(
        item_id="GAP-17", group="GAP",
        description="Auto-promote=False enforcement (Should S15)",
        legal_basis=["SR 11-7 effective challenge"],
        check_type="config_key",
        check_params={"path": "serving.competition.auto_promote"},
        severity="high", owner="evaluation-eng",
    ),
    ComplianceItem(
        item_id="GAP-18", group="GAP",
        description="Uplift T-Learner (Could C1)",
        legal_basis=["Pearl Rung 2"],
        check_type="custom_check",
        check_params={"check_name": "uplift_t_learner"},
        severity="low", owner="research-eng",
    ),
]

DEFAULT_CATALOG: List[ComplianceItem] = list(A_GROUP) + list(GAP_GROUP)
assert len(DEFAULT_CATALOG) == 36, (
    f"DEFAULT_CATALOG expected 36 items, got {len(DEFAULT_CATALOG)}"
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CustomCheckFn = Callable[[ComplianceItem], ComplianceItemResult]


class ComplianceRegistry:
    """36-item compliance registry with periodic sweep support."""

    def __init__(
        self,
        store: Optional[ComplianceStore] = None,
        items: Optional[Sequence[ComplianceItem]] = None,
        config: Optional[Dict[str, Any]] = None,
        project_root: Optional[Path] = None,
    ) -> None:
        self._store = store
        self._items: List[ComplianceItem] = list(items or DEFAULT_CATALOG)
        self._config = config or {}
        self._project_root = project_root or Path(".").resolve()
        self._custom_checks: Dict[str, CustomCheckFn] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_check(self, name: str, fn: CustomCheckFn) -> None:
        self._custom_checks[name] = fn

    def items(self) -> List[ComplianceItem]:
        return list(self._items)

    def item(self, item_id: str) -> ComplianceItem:
        for it in self._items:
            if it.item_id == item_id:
                return it
        raise KeyError(f"Unknown compliance item_id={item_id!r}")

    # ------------------------------------------------------------------
    # Checking
    # ------------------------------------------------------------------

    def check_single(self, item_id: str) -> ComplianceItemResult:
        return self._check(self.item(item_id))

    def check_all(self) -> List[ComplianceItemResult]:
        results = [self._check(it) for it in self._items]
        if self._store is not None:
            self._persist_summary(results)
        return results

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(
        self, results: Optional[List[ComplianceItemResult]] = None,
    ) -> Dict[str, Any]:
        results = results if results is not None else self.check_all()
        by_status: Dict[str, int] = {}
        by_group: Dict[str, Dict[str, int]] = {}
        critical_failures: List[str] = []
        for r in results:
            by_status[r.status] = by_status.get(r.status, 0) + 1
            item = self.item(r.item_id)
            g = item.group
            by_group.setdefault(g, {}).setdefault(r.status, 0)
            by_group[g][r.status] = by_group[g].get(r.status, 0) + 1
            if r.status == "non_compliant" and item.severity == "critical":
                critical_failures.append(r.item_id)
        return {
            "total": len(results),
            "by_status": by_status,
            "by_group": by_group,
            "critical_failures": critical_failures,
        }

    def generate_quarterly_report(
        self, quarter: str,
        results: Optional[List[ComplianceItemResult]] = None,
    ) -> str:
        """Generate a markdown report for the given quarter (e.g. '2026-Q2')."""
        results = results if results is not None else self.check_all()
        s = self.summary(results)
        lines: List[str] = []
        lines.append(f"# Compliance Registry Report {quarter}")
        lines.append("")
        lines.append(f"- Total items: {s['total']}")
        lines.append(f"- Compliant: {s['by_status'].get('compliant', 0)}")
        lines.append(
            f"- Non-compliant: {s['by_status'].get('non_compliant', 0)}"
        )
        lines.append(
            f"- Not applicable: {s['by_status'].get('not_applicable', 0)}"
        )
        lines.append(f"- Errors: {s['by_status'].get('error', 0)}")
        lines.append("")
        lines.append("## Critical failures")
        if s["critical_failures"]:
            for fid in s["critical_failures"]:
                it = self.item(fid)
                lines.append(f"- **{fid}** {it.description}")
        else:
            lines.append("None.")
        lines.append("")
        lines.append("## By group")
        for g, counts in s["by_group"].items():
            lines.append(f"### {g}")
            for status, count in counts.items():
                lines.append(f"- {status}: {count}")
            lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check(self, item: ComplianceItem) -> ComplianceItemResult:
        now = utcnow()
        try:
            if item.check_type == "module_exists":
                return self._check_module(item, now)
            if item.check_type == "file_exists":
                return self._check_file(item, now)
            if item.check_type == "config_key":
                return self._check_config(item, now)
            if item.check_type == "custom_check":
                return self._check_custom(item, now)
            return ComplianceItemResult(
                item_id=item.item_id,
                status="not_applicable",
                checked_at=now,
                message=f"unknown check_type={item.check_type!r}",
            )
        except Exception as exc:
            logger.exception(
                "Compliance check error for %s", item.item_id,
            )
            return ComplianceItemResult(
                item_id=item.item_id,
                status="error",
                checked_at=now,
                message=str(exc),
            )

    def _check_module(
        self, item: ComplianceItem, now: datetime,
    ) -> ComplianceItemResult:
        mod_name = item.check_params["module"]
        try:
            importlib.import_module(mod_name)
            return ComplianceItemResult(
                item_id=item.item_id, status="compliant",
                checked_at=now, message=f"module importable: {mod_name}",
                evidence={"module": mod_name},
            )
        except Exception as exc:
            return ComplianceItemResult(
                item_id=item.item_id, status="non_compliant",
                checked_at=now,
                message=f"import failed: {mod_name} ({exc})",
                evidence={"module": mod_name, "error": str(exc)},
            )

    def _check_file(
        self, item: ComplianceItem, now: datetime,
    ) -> ComplianceItemResult:
        path = Path(item.check_params["path"])
        if not path.is_absolute():
            path = self._project_root / path
        if path.exists():
            return ComplianceItemResult(
                item_id=item.item_id, status="compliant",
                checked_at=now, message=f"file exists: {path}",
                evidence={"path": str(path)},
            )
        return ComplianceItemResult(
            item_id=item.item_id, status="non_compliant",
            checked_at=now, message=f"file missing: {path}",
            evidence={"path": str(path)},
        )

    def _check_config(
        self, item: ComplianceItem, now: datetime,
    ) -> ComplianceItemResult:
        key_path: str = item.check_params["path"]
        cfg = self._config
        cursor: Any = cfg
        for part in key_path.split("."):
            if isinstance(cursor, dict) and part in cursor:
                cursor = cursor[part]
            else:
                return ComplianceItemResult(
                    item_id=item.item_id, status="non_compliant",
                    checked_at=now,
                    message=f"config key missing: {key_path}",
                    evidence={"key": key_path},
                )
        is_set = bool(cursor) if not isinstance(cursor, (list, dict)) \
            else bool(cursor)
        return ComplianceItemResult(
            item_id=item.item_id,
            status="compliant" if is_set else "non_compliant",
            checked_at=now,
            message=f"{key_path}={cursor!r}",
            evidence={"key": key_path, "value": cursor},
        )

    def _check_custom(
        self, item: ComplianceItem, now: datetime,
    ) -> ComplianceItemResult:
        name = item.check_params.get("check_name")
        fn = self._custom_checks.get(name)
        if fn is None:
            return ComplianceItemResult(
                item_id=item.item_id, status="not_applicable",
                checked_at=now,
                message=f"no custom check registered for {name!r}",
            )
        return fn(item)

    def _persist_summary(
        self, results: List[ComplianceItemResult],
    ) -> None:
        if self._store is None:
            return
        # Write one event per result plus a rolled-up summary.
        summary = self.summary(results)
        evt = ComplianceEvent(
            event_id=new_event_id(),
            user_id="system:compliance_registry",
            event_type=EventType.REQUEST_PROCESSED,  # reuse generic type
            timestamp=utcnow(),
            payload={
                "kind": "compliance_registry_sweep",
                "summary": summary,
                "results": [r.to_dict() for r in results],
            },
        )
        try:
            self._store.put_event(evt)
        except Exception:
            logger.exception(
                "ComplianceRegistry: failed to persist sweep event"
            )


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

def load_registry_from_pipeline_yaml(
    yaml_path: str = "configs/pipeline.yaml",
    store: Optional[ComplianceStore] = None,
) -> ComplianceRegistry:
    p = Path(yaml_path)
    cfg: Dict[str, Any] = {}
    if p.exists():
        cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return ComplianceRegistry(store=store, config=cfg)
