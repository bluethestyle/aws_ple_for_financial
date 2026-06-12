"""Regression tests for the 3 pure bugfixes back-ported from on-prem
(on-prem commit 8f006186, 2026-06-05 — "AWS 역이식 권장" 항목).

1. AuditDiagnoser._analyze_fairness — IntersectionalFairnessAnalyzer.summary()
   returns violation *counts* (int); len() on int raised TypeError.
2. IntersectionalFairnessAnalyzer.summary — worst_intersection was a raw
   dataclass, breaking json.dumps of the audit report.
3. GroundingValidator.compute_readability — jargon counting used search()
   (max 1 per pattern), under-counting repeated jargon occurrences.

Run: pytest tests/test_agent_audit_backports.py -v
"""

from __future__ import annotations

import json

from core.agent.audit.diagnoser import AuditDiagnoser
from core.agent.audit.grounding_validator import GroundingValidator
from core.agent.audit.intersectional_fairness import (
    IntersectionalFairnessAnalyzer,
    IntersectionalResult,
)


def _violation_result(di: float = 0.6, single_di: float = 0.7) -> IntersectionalResult:
    return IntersectionalResult(
        attribute_pair=("age_group", "income_tier"),
        subgroup="elderly ∩ low_income",
        subgroup_size=50,
        total_size=500,
        di_value=di,
        threshold=0.8,
        is_violation=True,
        single_attr_di={"age_group": single_di},
        detail="DI below threshold",
    )


class TestSummaryDiagnoseChain:
    """summary() (int counts) → diagnose() must not raise TypeError."""

    def test_diagnose_accepts_summary_int_counts(self):
        analyzer = IntersectionalFairnessAnalyzer()
        # single_attr_di 0.7 < threshold 0.8 → violation but NOT hidden,
        # so diagnose hits the `elif violations:` branch (the len(int) crash site).
        summary = analyzer.summary([_violation_result()])
        assert summary["violations"] == 1
        assert summary["hidden_violations"] == 0

        areas = AuditDiagnoser().diagnose(fairness_results=summary)
        assert len(areas) == 1
        assert areas[0].area == "공정성"
        assert areas[0].priority == "MEDIUM"
        assert "1건" in areas[0].finding
        assert areas[0].evidence["violation_count"] == 1

    def test_diagnose_still_accepts_list_violations(self):
        # Defensive path: other sources may pass violations as a list.
        areas = AuditDiagnoser().diagnose(
            fairness_results={"violations": ["v1", "v2"], "hidden_violations": 0}
        )
        assert len(areas) == 1
        assert areas[0].evidence["violation_count"] == 2


class TestSummaryJsonSerializable:
    """summary() output must survive json.dumps (audit report persistence)."""

    def test_worst_intersection_is_dict(self):
        analyzer = IntersectionalFairnessAnalyzer()
        summary = analyzer.summary([_violation_result(di=0.6)])
        assert summary["worst_intersection"] == {
            "subgroup": "elderly ∩ low_income",
            "di": 0.6,
        }
        json.dumps(summary)  # must not raise

    def test_empty_results_serializable(self):
        analyzer = IntersectionalFairnessAnalyzer()
        summary = analyzer.summary([])
        assert summary["worst_intersection"] is None
        json.dumps(summary)  # must not raise


class TestJargonOccurrenceCount:
    """Jargon must be counted by occurrence (findall), not pattern kind (search)."""

    def test_repeated_jargon_lowers_readability(self):
        validator = GroundingValidator()
        # Same word count (8), same single short sentence (length_score == 1.0
        # for both), no vague expressions — only jargon occurrences differ.
        text_once = "PSI 안정 추세 확인 결과 이상 없음 유지"
        text_twice = "PSI PSI 안정 추세 확인 결과 이상 없음"
        score_once = validator.compute_readability(text_once)
        score_twice = validator.compute_readability(text_twice)
        # With the search() bug both texts count jargon=1 and the scores tie.
        assert score_twice < score_once

    def test_occurrence_count_arithmetic(self):
        validator = GroundingValidator(
            config={"jargon_patterns": [r"PSI"], "max_jargon_ratio": 0.5}
        )
        # 8 words, "PSI" x2 → ratio 0.25 → jargon_score 0.5;
        # sentence ≤ 30 chars → length 1.0; no vague terms → vague 1.0.
        # 0.4*1.0 + 0.35*0.5 + 0.25*1.0 = 0.825
        score = validator.compute_readability("PSI PSI 안정 추세 확인 결과 이상 없음")
        assert score == 0.825
