"""
Public Disclosure Report Generator (금융위 공시용).

Generates structured quarterly reports for regulatory filing with the
Financial Services Commission (금융위원회).

Covers AI system transparency, model performance, fairness metrics,
incident summary, and customer impact assessment.

Output: JSON (machine-readable) + optional Markdown (human-readable).
Storage: S3 with versioning.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["PublicDisclosureGenerator", "DisclosureReport"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DisclosureReport:
    """Container for a quarterly regulatory disclosure report."""

    report_id: str
    report_period: str              # "2026-Q1"
    generated_at: str               # ISO 8601
    sections: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# PublicDisclosureGenerator
# ---------------------------------------------------------------------------

class PublicDisclosureGenerator:
    """Quarterly regulatory disclosure report generator.

    Sections:
      1. AI System Overview (model type, task count, serving mode)
      2. Model Performance Summary (per-task AUC, F1, MAE)
      3. Fairness Assessment (DI/SPD/EOD per protected attribute)
      4. Incident Summary (count by severity, resolution rate)
      5. Customer Impact (recommendations served, opt-out rate)
      6. Data Governance (PII handling, encryption status, retention)
      7. Compliance Status (regulatory check pass rate)
      8. AI Disclosure Statement (금소법/AI기본법 compliance text)

    Parameters
    ----------
    audit_store : ComplianceAuditStore or InMemoryAuditStore, optional
        Audit store for querying incident and opt-out data.
    config : dict, optional
        System configuration overrides.
    s3_bucket : str, optional
        S3 bucket for report storage.
    s3_prefix : str
        S3 key prefix for archived reports.
    system_name : str
        Name of the AI system (appears in reports).
    """

    # Quarter boundaries: month ranges for each quarter
    _QUARTER_MONTHS = {
        "Q1": (1, 3),
        "Q2": (4, 6),
        "Q3": (7, 9),
        "Q4": (10, 12),
    }

    def __init__(
        self,
        audit_store: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        s3_bucket: Optional[str] = None,
        s3_prefix: str = "disclosure_reports",
        system_name: str = "PLE-Cluster-adaTT",
    ) -> None:
        self._store = audit_store
        self._config = config or {}
        self.s3_bucket = s3_bucket or os.environ.get("DISCLOSURE_S3_BUCKET", "")
        self.s3_prefix = s3_prefix.strip("/")
        self.system_name = system_name

        self._s3_client = None
        if self.s3_bucket:
            try:
                import boto3

                region = os.environ.get("AWS_DEFAULT_REGION")
                self._s3_client = boto3.client("s3", region_name=region)
            except Exception as exc:
                logger.warning("S3 client init failed (disclosure): %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        period: str = "",
        metrics: Optional[Dict[str, Any]] = None,
        incidents: Optional[List[Dict[str, Any]]] = None,
        fairness: Optional[Dict[str, Any]] = None,
    ) -> DisclosureReport:
        """Generate a quarterly disclosure report.

        Parameters
        ----------
        period : str
            Quarter string in ``"YYYY-QN"`` format (e.g. ``"2026-Q1"``).
            Auto-detected from the current date if empty.
        metrics : dict, optional
            Model performance metrics keyed by task name.
            Each value should contain ``auc``, ``f1``, ``mae``, etc.
        incidents : list of dict, optional
            Incident records for the period.
        fairness : dict, optional
            Fairness evaluation results keyed by protected attribute.

        Returns
        -------
        DisclosureReport
        """
        now = datetime.now(timezone.utc)
        if not period:
            quarter = (now.month - 1) // 3 + 1
            period = f"{now.year}-Q{quarter}"

        report_id = f"DISC-{period}-{uuid.uuid4().hex[:8].upper()}"

        sections = {
            "ai_overview": self._section_ai_overview(),
            "performance": self._section_performance(metrics or {}),
            "fairness": self._section_fairness(fairness or {}),
            "incidents": self._section_incidents(incidents or []),
            "customer_impact": self._section_customer_impact(),
            "data_governance": self._section_data_governance(),
            "compliance_status": self._section_compliance_status(),
            "ai_disclosure": self._section_ai_disclosure(),
        }

        report = DisclosureReport(
            report_id=report_id,
            report_period=period,
            generated_at=now.isoformat(),
            sections=sections,
            metadata={
                "system_name": self.system_name,
                "generator_version": "1.0.0",
                "regulatory_framework": ["금소법", "AI기본법", "신용정보법"],
            },
        )

        logger.info("Disclosure report generated: %s (period=%s)", report_id, period)
        return report

    def to_markdown(self, report: DisclosureReport) -> str:
        """Convert a disclosure report to human-readable Markdown.

        Parameters
        ----------
        report : DisclosureReport
            The report to convert.

        Returns
        -------
        str
            Markdown text.
        """
        lines: List[str] = []
        lines.append(f"# 금융위 공시보고서 ({report.report_period})")
        lines.append("")
        lines.append(f"**보고서 ID**: {report.report_id}")
        lines.append(f"**생성일시**: {report.generated_at}")
        lines.append(f"**시스템**: {report.metadata.get('system_name', self.system_name)}")
        lines.append("")

        sections = report.sections

        # 1. AI System Overview
        lines.append("## 1. AI 시스템 개요")
        overview = sections.get("ai_overview", {})
        lines.append(f"- **시스템 유형**: {overview.get('system_type', 'N/A')}")
        lines.append(f"- **모델 아키텍처**: {overview.get('architecture', 'N/A')}")
        lines.append(f"- **태스크 수**: {overview.get('task_count', 'N/A')}")
        lines.append(f"- **서빙 모드**: {overview.get('serving_mode', 'N/A')}")
        lines.append("")

        # 2. Model Performance
        lines.append("## 2. 모델 성능 요약")
        perf = sections.get("performance", {})
        task_metrics = perf.get("task_metrics", {})
        if task_metrics:
            lines.append("| 태스크 | AUC | F1 | MAE |")
            lines.append("|--------|-----|-----|-----|")
            for task, m in task_metrics.items():
                auc = m.get("auc", "N/A")
                f1 = m.get("f1", "N/A")
                mae = m.get("mae", "N/A")
                auc_str = f"{auc:.4f}" if isinstance(auc, (int, float)) else str(auc)
                f1_str = f"{f1:.4f}" if isinstance(f1, (int, float)) else str(f1)
                mae_str = f"{mae:.4f}" if isinstance(mae, (int, float)) else str(mae)
                lines.append(f"| {task} | {auc_str} | {f1_str} | {mae_str} |")
        else:
            lines.append("_성능 데이터 없음._")
        lines.append("")

        # 3. Fairness Assessment
        lines.append("## 3. 공정성 평가")
        fair = sections.get("fairness", {})
        lines.append(f"- **전체 상태**: {fair.get('overall_status', 'N/A')}")
        lines.append(f"- **검사 속성 수**: {fair.get('attributes_checked', 0)}")
        lines.append(f"- **위반 건수**: {fair.get('total_violations', 0)}")
        attr_details = fair.get("attribute_details", {})
        if attr_details:
            lines.append("")
            lines.append("| 보호속성 | DI | SPD | EOD | 판정 |")
            lines.append("|----------|-----|------|------|------|")
            for attr, detail in attr_details.items():
                di = detail.get("disparate_impact", "N/A")
                spd = detail.get("statistical_parity_diff", "N/A")
                eod = detail.get("equal_opportunity_diff", "N/A")
                status = "Pass" if detail.get("is_fair", True) else "**FAIL**"
                di_str = f"{di:.4f}" if isinstance(di, (int, float)) else str(di)
                spd_str = f"{spd:.4f}" if isinstance(spd, (int, float)) else str(spd)
                eod_str = f"{eod:.4f}" if isinstance(eod, (int, float)) else str(eod)
                lines.append(f"| {attr} | {di_str} | {spd_str} | {eod_str} | {status} |")
        lines.append("")

        # 4. Incident Summary
        lines.append("## 4. 사고 요약")
        inc = sections.get("incidents", {})
        lines.append(f"- **총 건수**: {inc.get('total', 0)}")
        by_sev = inc.get("by_severity", {})
        for sev in ("critical", "high", "medium", "low"):
            cnt = by_sev.get(sev, 0)
            if cnt > 0:
                lines.append(f"- **{sev}**: {cnt}건")
        lines.append(f"- **해결율**: {inc.get('resolution_rate', 'N/A')}")
        lines.append("")

        # 5. Customer Impact
        lines.append("## 5. 고객 영향")
        ci = sections.get("customer_impact", {})
        lines.append(f"- **추천 제공 건수**: {ci.get('recommendations_served', 'N/A')}")
        lines.append(f"- **AI 결정 거부(opt-out) 건수**: {ci.get('opt_out_count', 'N/A')}")
        lines.append(f"- **민원 건수**: {ci.get('complaint_count', 'N/A')}")
        lines.append("")

        # 6. Data Governance
        lines.append("## 6. 데이터 거버넌스")
        dg = sections.get("data_governance", {})
        lines.append(f"- **PII 암호화 상태**: {dg.get('pii_encryption_status', 'N/A')}")
        lines.append(f"- **데이터 보존 정책**: {dg.get('retention_policy', 'N/A')}")
        lines.append(f"- **동의 관리**: {dg.get('consent_management', 'N/A')}")
        lines.append("")

        # 7. Compliance Status
        lines.append("## 7. 규제 준수 현황")
        cs = sections.get("compliance_status", {})
        lines.append(f"- **총 검사 항목**: {cs.get('total_checks', 0)}")
        lines.append(f"- **통과**: {cs.get('passed', 0)}")
        lines.append(f"- **실패**: {cs.get('failed', 0)}")
        lines.append(f"- **통과율**: {cs.get('pass_rate', 'N/A')}")
        lines.append("")

        # 8. AI Disclosure Statement
        lines.append("## 8. AI 공시 의무사항")
        disc = sections.get("ai_disclosure", {})
        lines.append(f"- **금소법 준수**: {disc.get('financial_consumer_protection', 'N/A')}")
        lines.append(f"- **AI기본법 준수**: {disc.get('ai_basic_act', 'N/A')}")
        lines.append(f"- **공시 문구**:")
        lines.append(f"  > {disc.get('disclosure_text', 'N/A')}")
        lines.append("")

        lines.append("---")
        lines.append(f"_본 보고서는 자동 생성되었습니다. (generator v{report.metadata.get('generator_version', '1.0.0')})_")
        return "\n".join(lines)

    def save_to_s3(self, report: DisclosureReport, s3_path: str = "") -> Optional[str]:
        """Save report JSON to S3 with versioning.

        Parameters
        ----------
        report : DisclosureReport
            The report to save.
        s3_path : str, optional
            Custom S3 key. Auto-generated if empty.

        Returns
        -------
        str or None
            S3 URI on success, ``None`` on failure.
        """
        if not self._s3_client or not self.s3_bucket:
            logger.warning("S3 not configured; disclosure report not saved.")
            return None

        if not s3_path:
            s3_path = (
                f"{self.s3_prefix}/{report.report_period}/{report.report_id}.json"
            )

        try:
            report_dict = {
                "report_id": report.report_id,
                "report_period": report.report_period,
                "generated_at": report.generated_at,
                "system_name": self.system_name,
                "sections": report.sections,
                "metadata": report.metadata,
            }
            body = json.dumps(report_dict, ensure_ascii=False, indent=2).encode("utf-8")
            self._s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=s3_path,
                Body=body,
                ContentType="application/json",
            )
            uri = f"s3://{self.s3_bucket}/{s3_path}"
            logger.info("Disclosure report saved: %s", uri)
            return uri
        except Exception as exc:
            logger.warning("Failed to save disclosure report to S3: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _section_ai_overview(self) -> Dict[str, Any]:
        """AI system type, architecture, task count, serving infrastructure."""
        cfg = self._config
        return {
            "system_type": cfg.get("system_type", "AI-based product recommendation"),
            "architecture": cfg.get("architecture", "Multi-task expert ensemble (adaTT)"),
            "task_count": cfg.get("task_count", 0),
            "tasks": cfg.get("tasks", []),
            "serving_mode": cfg.get("serving_mode", "batch (SageMaker Processing)"),
            "infrastructure": cfg.get("infrastructure", "AWS SageMaker + S3 + DynamoDB"),
            "model_version": cfg.get("model_version", "N/A"),
        }

    @staticmethod
    def _section_performance(metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Per-task metrics table."""
        if not metrics:
            return {"status": "no_data", "task_count": 0, "task_metrics": {}}

        task_metrics: Dict[str, Any] = {}
        for task_name, task_data in metrics.items():
            if isinstance(task_data, dict):
                task_metrics[task_name] = {
                    "auc": task_data.get("auc"),
                    "f1": task_data.get("f1"),
                    "mae": task_data.get("mae"),
                    "accuracy": task_data.get("accuracy"),
                    "sample_count": task_data.get("sample_count"),
                }
            else:
                task_metrics[task_name] = {"raw": task_data}

        return {
            "status": "available",
            "task_count": len(task_metrics),
            "task_metrics": task_metrics,
        }

    @staticmethod
    def _section_fairness(fairness: Dict[str, Any]) -> Dict[str, Any]:
        """Protected attributes, DI/SPD/EOD values."""
        if not fairness:
            return {
                "overall_status": "no_data",
                "attributes_checked": 0,
                "total_violations": 0,
                "attribute_details": {},
            }

        total_violations = 0
        attribute_details: Dict[str, Any] = {}

        for attr, data in fairness.items():
            if isinstance(data, dict):
                violations = data.get("violations", [])
                total_violations += len(violations)
                attribute_details[attr] = {
                    "disparate_impact": data.get("disparate_impact"),
                    "statistical_parity_diff": data.get("statistical_parity_diff"),
                    "equal_opportunity_diff": data.get("equal_opportunity_diff"),
                    "is_fair": len(violations) == 0,
                    "violations": violations,
                }
            elif hasattr(data, "violations"):
                viol = data.violations if hasattr(data, "violations") else []
                total_violations += len(viol)
                attribute_details[attr] = {
                    "disparate_impact": getattr(data, "disparate_impact", None),
                    "statistical_parity_diff": getattr(data, "statistical_parity_diff", None),
                    "equal_opportunity_diff": getattr(data, "equal_opportunity_diff", None),
                    "is_fair": len(viol) == 0,
                    "violations": viol,
                }

        overall = "pass" if total_violations == 0 else "fail"
        return {
            "overall_status": overall,
            "attributes_checked": len(fairness),
            "total_violations": total_violations,
            "attribute_details": attribute_details,
        }

    @staticmethod
    def _section_incidents(incidents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Count by severity, resolution times."""
        if not incidents:
            return {
                "total": 0,
                "by_severity": {},
                "resolution_rate": "N/A",
                "avg_resolution_hours": None,
            }

        by_severity: Dict[str, int] = {}
        resolved_count = 0
        resolution_hours: List[float] = []

        for inc in incidents:
            sev = inc.get("severity", "low")
            by_severity[sev] = by_severity.get(sev, 0) + 1
            status = inc.get("status", "open")
            if status in ("resolved", "closed"):
                resolved_count += 1
                hours = inc.get("resolution_hours")
                if hours is not None:
                    resolution_hours.append(float(hours))

        total = len(incidents)
        rate = f"{resolved_count / total * 100:.1f}%" if total > 0 else "N/A"
        avg_hours = (
            sum(resolution_hours) / len(resolution_hours)
            if resolution_hours
            else None
        )

        return {
            "total": total,
            "by_severity": by_severity,
            "resolution_rate": rate,
            "resolved_count": resolved_count,
            "avg_resolution_hours": avg_hours,
        }

    def _section_customer_impact(self) -> Dict[str, Any]:
        """Recommendations served, opt-out count, complaint count."""
        cfg = self._config
        result: Dict[str, Any] = {
            "recommendations_served": cfg.get("recommendations_served", 0),
            "opt_out_count": cfg.get("opt_out_count", 0),
            "opt_out_rate": cfg.get("opt_out_rate", "N/A"),
            "complaint_count": cfg.get("complaint_count", 0),
            "customer_satisfaction": cfg.get("customer_satisfaction", "N/A"),
        }

        # Try to pull opt-out counts from audit store
        if self._store is not None:
            try:
                events = self._store.get_all_events("optout") if hasattr(
                    self._store, "get_all_events"
                ) else []
                if events:
                    opt_outs = [e for e in events if e.get("action") == "opt_out"]
                    result["opt_out_count"] = len(opt_outs)
            except Exception:
                pass

        return result

    def _section_data_governance(self) -> Dict[str, Any]:
        """PII encryption status, data retention, consent management."""
        cfg = self._config
        return {
            "pii_encryption_status": cfg.get(
                "pii_encryption_status", "AES-256 at rest, TLS 1.2+ in transit"
            ),
            "pii_fields_count": cfg.get("pii_fields_count", 0),
            "anonymization_method": cfg.get("anonymization_method", "integer index encryption"),
            "retention_policy": cfg.get(
                "retention_policy", "Audit logs: 10 years; model artifacts: 5 years"
            ),
            "consent_management": cfg.get(
                "consent_management", "DynamoDB-backed consent store with full audit trail"
            ),
            "data_location": cfg.get("data_location", "unspecified"),
            "cross_border_transfer": cfg.get("cross_border_transfer", False),
        }

    def _section_compliance_status(self) -> Dict[str, Any]:
        """Regulatory check pass/fail summary."""
        cfg = self._config
        total = cfg.get("compliance_total_checks", 0)
        passed = cfg.get("compliance_passed", 0)
        failed = total - passed

        return {
            "total_checks": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": f"{passed / total * 100:.1f}%" if total > 0 else "N/A",
            "frameworks": cfg.get(
                "compliance_frameworks",
                ["금소법", "AI기본법", "신용정보법", "개인정보보호법"],
            ),
            "last_audit_date": cfg.get("last_audit_date", "N/A"),
        }

    def _section_ai_disclosure(self) -> Dict[str, Any]:
        """금소법/AI기본법 mandatory disclosure text."""
        cfg = self._config
        return {
            "financial_consumer_protection": cfg.get(
                "financial_consumer_protection_status",
                "compliant",
            ),
            "ai_basic_act": cfg.get("ai_basic_act_status", "compliant"),
            "disclosure_text": cfg.get(
                "disclosure_text",
                (
                    "본 금융상품 추천은 AI 알고리즘에 의해 자동 생성되었습니다. "
                    "AI 추천 결과는 참고용이며 최종 투자 판단은 고객님께서 직접 하셔야 합니다. "
                    "AI 기반 의사결정에 동의하지 않으시는 경우 언제든지 거부(opt-out)하실 수 있으며, "
                    "이 경우 규칙 기반 추천 또는 인적 상담이 제공됩니다."
                ),
            ),
            "opt_out_mechanism": cfg.get(
                "opt_out_mechanism",
                "DynamoDB opt-out store with immediate effect",
            ),
            "human_oversight": cfg.get(
                "human_oversight",
                "Governance committee quarterly review; real-time kill switch available",
            ),
        }
