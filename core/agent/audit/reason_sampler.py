"""
Stratified Reason Sampler for Audit Agent AV3
===============================================

Samples recommendation reasons for Tier 2 quality evaluation using
stratified sampling across three axes:
    - task_type: binary / multiclass / regression
    - customer_segment: mass / affluent / vip
    - reason_layer: L1 / L2a / L2b

Oversampling for:
    - Borderline cases (selfcheck confidence 0.6-0.8): 3x
    - Protected groups (elderly, low_income): 2x
    - Human review flagged L2b: all
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["StratifiedReasonSampler", "SampledCase"]


@dataclass
class SampledCase:
    """A single sampled reason case for Tier 2 evaluation."""
    customer_id: str
    item_id: str
    task_type: str
    customer_segment: str
    reason_layer: str
    reason_text: str
    ig_top_features: List[Dict[str, Any]] = field(default_factory=list)
    selfcheck_confidence: float = 1.0
    selfcheck_verdict: str = "pass"
    human_review_flagged: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class StratifiedReasonSampler:
    """Stratified sampler for recommendation reason quality evaluation.

    Args:
        config: Sampling configuration dict with keys:
            - samples_per_stratum: int (default 15)
            - oversampling_weights: dict of condition -> multiplier
            - task_types: list of task types
            - customer_segments: list of segments
            - reason_layers: list of layers
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self.samples_per_stratum = cfg.get("samples_per_stratum", 15)
        self.task_types = cfg.get("task_types", ["binary", "multiclass", "regression"])
        self.customer_segments = cfg.get("customer_segments", ["mass", "affluent", "vip"])
        self.reason_layers = cfg.get("reason_layers", ["L1", "L2a", "L2b"])

        self.oversampling = cfg.get("oversampling_weights", {
            "borderline_confidence": 3.0,  # selfcheck confidence 0.6-0.8
            "protected_group": 2.0,         # elderly, low_income
            "human_review_flagged": -1,     # -1 = include all
        })

        self._protected_segments = cfg.get("protected_segments", ["elderly", "low_income"])

    @property
    def strata_count(self) -> int:
        return len(self.task_types) * len(self.customer_segments) * len(self.reason_layers)

    def sample(self, reason_records: List[Dict[str, Any]]) -> List[SampledCase]:
        """Sample from a pool of reason records using stratified sampling.

        Args:
            reason_records: List of dicts, each containing:
                customer_id, item_id, task_type, customer_segment,
                reason_layer, reason_text, ig_top_features,
                selfcheck_confidence, selfcheck_verdict,
                human_review_flagged, metadata (all optional except customer_id)

        Returns:
            List of SampledCase instances for Tier 2 evaluation.
        """
        # Step 1: Stratify records
        strata = self._stratify(reason_records)

        # Step 2: Force-include all human_review_flagged L2b
        forced = self._force_include(reason_records)
        forced_ids = {(r["customer_id"], r.get("item_id", "")) for r in forced}

        # Step 3: Sample from each stratum with oversampling
        sampled_records = list(forced)

        for key, records in strata.items():
            # Remove already-forced records
            available = [
                r for r in records
                if (r["customer_id"], r.get("item_id", "")) not in forced_ids
            ]

            n = self._compute_stratum_size(available)

            # Priority sort: borderline first, then protected, then random
            prioritized = self._prioritize(available)
            selected = prioritized[:n]
            sampled_records.extend(selected)

        # Convert to SampledCase
        cases = [self._to_case(r) for r in sampled_records]

        logger.info(
            "Sampled %d cases from %d records (%d strata, %d forced)",
            len(cases), len(reason_records), len(strata), len(forced),
        )
        return cases

    def _stratify(self, records: List[Dict]) -> Dict[str, List[Dict]]:
        """Group records by (task_type, customer_segment, reason_layer)."""
        strata: Dict[str, List[Dict]] = defaultdict(list)
        for r in records:
            key = (
                r.get("task_type", "unknown"),
                r.get("customer_segment", "unknown"),
                r.get("reason_layer", "L1"),
            )
            strata[str(key)].append(r)
        return strata

    def _force_include(self, records: List[Dict]) -> List[Dict]:
        """Force-include all human_review_flagged L2b records."""
        return [
            r for r in records
            if r.get("human_review_flagged") and r.get("reason_layer") == "L2b"
        ]

    def _compute_stratum_size(self, available: List[Dict]) -> int:
        """Compute sample size for a stratum (may exceed base due to oversampling)."""
        n = min(self.samples_per_stratum, len(available))
        return n

    def _prioritize(self, records: List[Dict]) -> List[Dict]:
        """Sort records by priority: borderline > protected > random."""
        def priority_score(r: Dict) -> float:
            score = 0.0
            # Borderline confidence (0.6-0.8) gets highest priority
            conf = r.get("selfcheck_confidence", 1.0)
            if 0.6 <= conf < 0.8:
                score += self.oversampling.get("borderline_confidence", 3.0)
            # Protected group
            seg = r.get("customer_segment", "")
            if seg in self._protected_segments:
                score += self.oversampling.get("protected_group", 2.0)
            # Add small random tiebreaker
            score += random.random() * 0.1
            return -score  # negative for descending sort

        return sorted(records, key=priority_score)

    def _to_case(self, r: Dict) -> SampledCase:
        return SampledCase(
            customer_id=r.get("customer_id", ""),
            item_id=r.get("item_id", ""),
            task_type=r.get("task_type", "unknown"),
            customer_segment=r.get("customer_segment", "unknown"),
            reason_layer=r.get("reason_layer", "L1"),
            reason_text=r.get("reason_text", ""),
            ig_top_features=r.get("ig_top_features", []),
            selfcheck_confidence=r.get("selfcheck_confidence", 1.0),
            selfcheck_verdict=r.get("selfcheck_verdict", "pass"),
            human_review_flagged=r.get("human_review_flagged", False),
            metadata=r.get("metadata", {}),
        )

    def get_sampling_stats(self, cases: List[SampledCase]) -> Dict[str, Any]:
        """Get statistics about a sampling result."""
        by_task = defaultdict(int)
        by_segment = defaultdict(int)
        by_layer = defaultdict(int)
        borderline = 0
        protected = 0

        for c in cases:
            by_task[c.task_type] += 1
            by_segment[c.customer_segment] += 1
            by_layer[c.reason_layer] += 1
            if 0.6 <= c.selfcheck_confidence < 0.8:
                borderline += 1
            if c.customer_segment in self._protected_segments:
                protected += 1

        return {
            "total_sampled": len(cases),
            "by_task_type": dict(by_task),
            "by_customer_segment": dict(by_segment),
            "by_reason_layer": dict(by_layer),
            "borderline_cases": borderline,
            "protected_group_cases": protected,
            "human_review_flagged": sum(1 for c in cases if c.human_review_flagged),
        }
