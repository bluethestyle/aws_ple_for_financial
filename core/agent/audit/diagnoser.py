"""
Audit Diagnoser — Focus Area Synthesis
=========================================

Synthesizes outputs from AV1-AV5 audit viewpoints into prioritized
focus_areas with recommended review actions.

AV1: Fairness (IntersectionalFairnessAnalyzer)
AV2: Concentration (HerdingDetector)
AV3: Reason Quality (GroundingValidator + Tier1Aggregator)
AV4: Regulatory Compliance (RegulatoryComplianceChecker + EUAIActMapper)
AV5: Data Lineage (DataLineageTracker)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["AuditDiagnoser", "FocusArea"]


@dataclass
class FocusArea:
    """A prioritized audit focus area for human review."""
    area: str               # e.g., "추천사유 품질", "교차속성 공정성"
    priority: str           # HIGH / MEDIUM / LOW
    finding: str            # what was observed
    evidence: Dict[str, Any] = field(default_factory=dict)
    recommended_review: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "area": self.area,
            "priority": self.priority,
            "finding": self.finding,
            "evidence": self.evidence,
            "recommended_review": self.recommended_review,
        }


class AuditDiagnoser:
    """Synthesizes audit viewpoint results into focus areas.

    Args:
        config: Audit-specific thresholds and config.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._config = config or {}

    def diagnose(
        self,
        fairness_results: Optional[Dict[str, Any]] = None,
        herding_results: Optional[Dict[str, Any]] = None,
        reason_quality: Optional[Dict[str, Any]] = None,
        regulatory_results: Optional[Dict[str, Any]] = None,
        lineage_results: Optional[Dict[str, Any]] = None,
    ) -> List[FocusArea]:
        """Analyze all viewpoint results and generate focus areas.

        Each parameter is the output from the corresponding audit component.
        """
        focus_areas = []

        # AV1: Fairness
        if fairness_results:
            areas = self._analyze_fairness(fairness_results)
            focus_areas.extend(areas)

        # AV2: Concentration
        if herding_results:
            areas = self._analyze_herding(herding_results)
            focus_areas.extend(areas)

        # AV3: Reason Quality
        if reason_quality:
            areas = self._analyze_reason_quality(reason_quality)
            focus_areas.extend(areas)

        # AV4: Regulatory
        if regulatory_results:
            areas = self._analyze_regulatory(regulatory_results)
            focus_areas.extend(areas)

        # AV5: Lineage
        if lineage_results:
            areas = self._analyze_lineage(lineage_results)
            focus_areas.extend(areas)

        # Sort by priority
        priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        focus_areas.sort(key=lambda f: priority_order.get(f.priority, 99))

        return focus_areas

    def _analyze_fairness(self, results: Dict) -> List[FocusArea]:
        areas = []
        # Check for intersectional violations
        violations = results.get("violations", [])
        hidden = results.get("hidden_violations", 0)

        if hidden > 0:
            areas.append(FocusArea(
                area="교차속성 공정성",
                priority="HIGH",
                finding=f"단일 속성은 통과하지만 교차에서 {hidden}건 위반 발견",
                evidence={"hidden_violations": hidden, "details": results.get("details", [])},
                recommended_review="교차 위반 그룹의 constraint_engine 필터 비례성 점검",
            ))
        elif violations:
            areas.append(FocusArea(
                area="공정성",
                priority="MEDIUM",
                finding=f"공정성 위반 {len(violations)}건",
                evidence={"violation_count": len(violations)},
                recommended_review="위반 보호속성 그룹별 추천 비율 상세 분석",
            ))
        return areas

    def _analyze_herding(self, results: Dict) -> List[FocusArea]:
        areas = []
        if results.get("is_herding"):
            severity = results.get("severity", "unknown")
            priority = "HIGH" if severity in ("critical", "high") else "MEDIUM"
            areas.append(FocusArea(
                area="추천 집중도",
                priority=priority,
                finding=f"쏠림 감지 (severity: {severity})",
                evidence={
                    "hhi": results.get("metrics", {}).get("hhi_index"),
                    "gini": results.get("metrics", {}).get("gini_coefficient"),
                    "top3_concentration": results.get("metrics", {}).get("top3_concentration"),
                },
                recommended_review="상품별 추천 분포 확인, diversity method 파라미터 조정 검토",
            ))
        return areas

    def _analyze_reason_quality(self, results: Dict) -> List[FocusArea]:
        areas = []
        # Tier 1 trend
        tier1 = results.get("tier1", {})
        if tier1.get("trend_alert"):
            areas.append(FocusArea(
                area="추천사유 품질 (Tier 1)",
                priority="HIGH",
                finding=tier1.get("trend_detail", "reject/revise 비율 추이 악화"),
                evidence={
                    "pass_rate": tier1.get("pass_rate"),
                    "reject_rate": tier1.get("reject_rate"),
                },
                recommended_review="compliance_rules 업데이트 또는 L2a 프롬프트 검토",
            ))

        # Tier 2 grounding
        tier2 = results.get("tier2", {})
        grounding_avg = tier2.get("avg_grounding_score", 1.0)
        if grounding_avg < self._config.get("min_grounding_score", 0.7):
            areas.append(FocusArea(
                area="추천사유 품질 (Tier 2 Grounding)",
                priority="HIGH",
                finding=f"평균 grounding score {grounding_avg:.2f} < 임계값",
                evidence={
                    "avg_grounding": grounding_avg,
                    "sample_size": tier2.get("sample_size", 0),
                    "worst_task": tier2.get("worst_task"),
                },
                recommended_review="IG top-K 피처와 사유 텍스트 불일치 원인 분석, InterpretationRegistry 커버리지 확인",
            ))
        return areas

    def _analyze_regulatory(self, results: Dict) -> List[FocusArea]:
        areas = []
        critical_failures = results.get("critical_failures", 0)
        if critical_failures > 0:
            areas.append(FocusArea(
                area="규제 적합성",
                priority="HIGH",
                finding=f"규제 critical failure {critical_failures}건",
                evidence={
                    "pass_rate": results.get("pass_rate"),
                    "by_regulation": results.get("by_regulation", {}),
                },
                recommended_review="critical failure 항목 즉시 조치 필요",
            ))
        elif results.get("pass_rate", 1.0) < 1.0:
            areas.append(FocusArea(
                area="규제 적합성",
                priority="MEDIUM",
                finding=f"규제 준수율 {results.get('pass_rate', 0):.0%}",
                evidence=results,
                recommended_review="미충족 항목 개선 계획 수립",
            ))
        return areas

    def _analyze_lineage(self, results: Dict) -> List[FocusArea]:
        areas = []
        unmapped = results.get("unmapped_ratio", 0)
        if unmapped > 0.05:
            areas.append(FocusArea(
                area="데이터 계보",
                priority="MEDIUM",
                finding=f"미매핑 피처 비율 {unmapped:.1%}",
                evidence={"unmapped_ratio": unmapped},
                recommended_review="미매핑 피처의 원천 데이터 소스 확인 및 lineage 등록",
            ))
        return areas
