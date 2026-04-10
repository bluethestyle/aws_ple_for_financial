"""
Intersectional Fairness Analyzer for Audit Agent AV1
======================================================

Detects fairness violations at the intersection of protected attributes
that are invisible to single-attribute analysis.

Example: age_group DI=0.85 (pass), income_tier DI=0.88 (pass),
but elderly ∩ low_income DI=0.62 (violation).

Config-driven attribute combinations and thresholds.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = ["IntersectionalFairnessAnalyzer", "IntersectionalResult"]


@dataclass
class IntersectionalResult:
    """Result of intersectional fairness analysis for one attribute pair."""
    attribute_pair: Tuple[str, str]
    subgroup: str  # e.g., "elderly ∩ low_income"
    subgroup_size: int
    total_size: int
    di_value: float  # Disparate Impact
    threshold: float
    is_violation: bool
    single_attr_di: Dict[str, float] = field(default_factory=dict)  # per single attribute
    detail: str = ""


class IntersectionalFairnessAnalyzer:
    """Analyzes fairness across intersections of protected attributes.

    Args:
        config: Config dict with keys:
            - attribute_pairs: list of [attr1, attr2] pairs to check
            - di_threshold: float (default 0.80)
            - min_sample_size: int (default 30)
            - protected_values: dict of {attribute: [values_to_check]}
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self.attribute_pairs = [
            tuple(p) for p in cfg.get("attribute_pairs", [
                ["age_group", "income_tier"],
                ["gender", "region_type"],
                ["life_stage", "income_tier"],
            ])
        ]
        self.di_threshold = cfg.get("di_threshold", 0.80)
        self.min_sample_size = cfg.get("min_sample_size", 30)
        self.protected_values = cfg.get("protected_values", {
            "age_group": ["elderly"],
            "income_tier": ["low_income"],
            "gender": ["female"],
            "region_type": ["rural"],
            "life_stage": ["retired"],
        })

    def analyze(
        self,
        recommendations: List[Dict[str, Any]],
    ) -> List[IntersectionalResult]:
        """Analyze intersectional fairness across all configured attribute pairs.

        Args:
            recommendations: List of recommendation dicts, each containing:
                - "recommended": bool (was the item recommended)
                - Protected attribute values as keys

        Returns:
            List of IntersectionalResult for all checked intersections.
        """
        results = []

        for attr1, attr2 in self.attribute_pairs:
            pair_results = self._analyze_pair(recommendations, attr1, attr2)
            results.extend(pair_results)

        violations = [r for r in results if r.is_violation]
        if violations:
            logger.warning(
                "Intersectional fairness: %d violations found across %d pairs",
                len(violations), len(self.attribute_pairs),
            )

        return results

    def _analyze_pair(
        self,
        recommendations: List[Dict[str, Any]],
        attr1: str,
        attr2: str,
    ) -> List[IntersectionalResult]:
        """Analyze intersections for a single attribute pair."""
        results = []

        # Get protected values for each attribute
        vals1 = self.protected_values.get(attr1, [])
        vals2 = self.protected_values.get(attr2, [])

        if not vals1 or not vals2:
            return results

        # Compute overall positive rate
        total_recs = [r for r in recommendations if attr1 in r and attr2 in r]
        if not total_recs:
            return results

        overall_positive = sum(1 for r in total_recs if r.get("recommended", False))
        overall_rate = overall_positive / len(total_recs) if total_recs else 0

        if overall_rate == 0:
            return results

        # Check each intersection of protected values
        for v1 in vals1:
            for v2 in vals2:
                subgroup = [
                    r for r in total_recs
                    if r.get(attr1) == v1 and r.get(attr2) == v2
                ]

                if len(subgroup) < self.min_sample_size:
                    logger.debug(
                        "Skipping %s=%s ∩ %s=%s: sample size %d < %d",
                        attr1, v1, attr2, v2, len(subgroup), self.min_sample_size,
                    )
                    continue

                subgroup_positive = sum(
                    1 for r in subgroup if r.get("recommended", False)
                )
                subgroup_rate = subgroup_positive / len(subgroup) if subgroup else 0

                di = subgroup_rate / overall_rate if overall_rate > 0 else 0.0

                # Compute single-attribute DIs for comparison
                single_di = {}
                for attr, val in [(attr1, v1), (attr2, v2)]:
                    single_group = [r for r in total_recs if r.get(attr) == val]
                    if single_group:
                        single_rate = sum(
                            1 for r in single_group if r.get("recommended", False)
                        ) / len(single_group)
                        single_di[attr] = round(single_rate / overall_rate, 4) if overall_rate > 0 else 0.0

                is_violation = di < self.di_threshold
                subgroup_label = f"{v1} ∩ {v2}"

                detail = ""
                if is_violation:
                    # Check if single attributes pass
                    singles_pass = all(
                        d >= self.di_threshold for d in single_di.values()
                    )
                    if singles_pass:
                        detail = (
                            f"단일 속성은 모두 통과하지만 교차에서 위반. "
                            f"단일: {single_di}. 교차 DI={di:.4f}"
                        )
                    else:
                        detail = f"교차 DI={di:.4f}, 단일: {single_di}"

                results.append(IntersectionalResult(
                    attribute_pair=(attr1, attr2),
                    subgroup=subgroup_label,
                    subgroup_size=len(subgroup),
                    total_size=len(total_recs),
                    di_value=round(di, 4),
                    threshold=self.di_threshold,
                    is_violation=is_violation,
                    single_attr_di=single_di,
                    detail=detail,
                ))

        return results

    def summary(self, results: List[IntersectionalResult]) -> Dict[str, Any]:
        """Generate summary of intersectional analysis."""
        violations = [r for r in results if r.is_violation]
        hidden = [
            r for r in violations
            if all(d >= self.di_threshold for d in r.single_attr_di.values())
        ]

        return {
            "total_intersections_checked": len(results),
            "violations": len(violations),
            "hidden_violations": len(hidden),  # only visible at intersection level
            "worst_intersection": min(
                (r for r in results),
                key=lambda r: r.di_value,
                default=None,
            ),
            "details": [
                {
                    "subgroup": r.subgroup,
                    "di": r.di_value,
                    "threshold": r.threshold,
                    "single_attr_di": r.single_attr_di,
                    "detail": r.detail,
                }
                for r in violations
            ],
        }
