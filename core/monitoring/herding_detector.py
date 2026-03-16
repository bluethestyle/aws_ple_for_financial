"""
Herding (concentration) detection for recommendation outputs.

Measures recommendation diversity using four complementary metrics:

- **HHI** (Herfindahl-Hirschman Index): Market concentration (0--1)
- **Gini Coefficient**: Distribution inequality (0--1)
- **Shannon Entropy**: Information-theoretic diversity
- **Herding Rate**: Composite risk score

Severity is classified as ``none`` / ``low`` / ``medium`` / ``high`` / ``critical``
based on the number of threshold violations.
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HerdingMetrics:
    """Container for herding / concentration metrics."""

    product_entropy: float
    normalized_entropy: float
    hhi_index: float
    top3_concentration: float
    gini_coefficient: float
    herding_risk_score: float
    total_recommendations: int
    unique_products: int


# ---------------------------------------------------------------------------
# HerdingDetector
# ---------------------------------------------------------------------------

class HerdingDetector:
    """Detect recommendation herding (concentration bias).

    Parameters
    ----------
    thresholds : dict, optional
        Override default concentration thresholds.
    auto_incident : bool
        When ``True``, auto-escalate critical herding to the incident
        management system.
    """

    DEFAULT_THRESHOLDS: Dict[str, float] = {
        "max_hhi": 0.25,
        "min_entropy_ratio": 0.5,
        "max_top3_concentration": 0.6,
        "max_gini": 0.7,
    }

    SEVERITY_MAP = {
        0: "none",
        1: "low",
        2: "medium",
        3: "high",
        4: "critical",
    }

    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
        auto_incident: bool = True,
    ) -> None:
        self.thresholds: Dict[str, float] = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.auto_incident = auto_incident

    # ------------------------------------------------------------------
    # Core metrics computation
    # ------------------------------------------------------------------

    def compute_metrics(self, recommendations: List[Dict[str, Any]]) -> HerdingMetrics:
        """Compute all herding metrics for a set of recommendations.

        Parameters
        ----------
        recommendations : list of dict
            Each dict should have a ``"product_id"`` (or ``"item_code"``) key.

        Returns
        -------
        HerdingMetrics
        """
        product_ids = [
            r.get("product_id", r.get("item_code", "unknown"))
            for r in recommendations
        ]
        total = len(product_ids)
        if total == 0:
            return HerdingMetrics(0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0, 0)

        counter = Counter(product_ids)
        unique = len(counter)
        probs = [count / total for count in counter.values()]

        # Shannon entropy
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        max_entropy = math.log2(unique) if unique > 1 else 1.0
        normalized = entropy / max_entropy if max_entropy > 0 else 0.0

        # HHI
        hhi = sum(p ** 2 for p in probs)

        # Top-3 concentration
        sorted_counts = sorted(counter.values(), reverse=True)
        top3 = sum(sorted_counts[:3]) / total

        # Gini coefficient
        gini = self._compute_gini(list(counter.values()))

        # Composite herding risk score
        risk = 0.3 * hhi + 0.3 * (1 - normalized) + 0.2 * top3 + 0.2 * gini

        return HerdingMetrics(
            product_entropy=round(entropy, 4),
            normalized_entropy=round(normalized, 4),
            hhi_index=round(hhi, 4),
            top3_concentration=round(top3, 4),
            gini_coefficient=round(gini, 4),
            herding_risk_score=round(min(risk, 1.0), 4),
            total_recommendations=total,
            unique_products=unique,
        )

    # ------------------------------------------------------------------
    # Detection and severity classification
    # ------------------------------------------------------------------

    def detect_herding(
        self,
        recommendations: List[Dict[str, Any]],
        thresholds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Detect herding and classify severity.

        Parameters
        ----------
        recommendations : list of dict
            Recommendation results.
        thresholds : dict, optional
            Per-call threshold overrides.

        Returns
        -------
        dict
            Detection result including ``is_herding``, ``severity``,
            ``violations``, ``metrics``, and optionally ``alert_message``.
        """
        t = {**self.thresholds, **(thresholds or {})}
        metrics = self.compute_metrics(recommendations)

        violations: List[str] = []
        if metrics.hhi_index > t["max_hhi"]:
            violations.append(f"HHI exceeded: {metrics.hhi_index:.4f} > {t['max_hhi']}")
        if metrics.normalized_entropy < t["min_entropy_ratio"]:
            violations.append(
                f"Entropy too low: {metrics.normalized_entropy:.4f} < {t['min_entropy_ratio']}"
            )
        if metrics.top3_concentration > t["max_top3_concentration"]:
            violations.append(
                f"Top-3 concentration: {metrics.top3_concentration:.4f} > {t['max_top3_concentration']}"
            )
        if metrics.gini_coefficient > t["max_gini"]:
            violations.append(
                f"Gini exceeded: {metrics.gini_coefficient:.4f} > {t['max_gini']}"
            )

        num_violations = len(violations)
        severity = self.SEVERITY_MAP.get(min(num_violations, 4), "critical")
        is_herding = num_violations >= 2

        result: Dict[str, Any] = {
            "is_herding": is_herding,
            "severity": severity,
            "violations": violations,
            "metrics": {
                "hhi_index": metrics.hhi_index,
                "normalized_entropy": metrics.normalized_entropy,
                "top3_concentration": metrics.top3_concentration,
                "gini_coefficient": metrics.gini_coefficient,
                "herding_risk_score": metrics.herding_risk_score,
            },
            "total_recommendations": metrics.total_recommendations,
            "unique_products": metrics.unique_products,
            "alert_message": (
                f"Herding detected ({severity}): {'; '.join(violations)}"
                if is_herding
                else None
            ),
        }

        # Auto-escalation for critical severity
        if severity == "critical" and self.auto_incident:
            self._escalate_to_incident(result)

        return result

    def detect_task_herding(
        self,
        task_contribution_map: Dict[str, Dict[str, float]],
        dominance_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Detect per-task contribution herding.

        Parameters
        ----------
        task_contribution_map : dict
            ``{customer_id: {task_name: contribution_ratio}}``.
        dominance_threshold : float
            A task is flagged as dominant if its average contribution
            exceeds this threshold.

        Returns
        -------
        dict
            ``{"is_herding", "dominant_tasks", "task_averages", "alert_message"}``.
        """
        if not task_contribution_map:
            return {
                "is_herding": False,
                "dominant_tasks": [],
                "task_averages": {},
                "alert_message": None,
            }

        task_totals: Dict[str, List[float]] = defaultdict(list)
        for contributions in task_contribution_map.values():
            for task, ratio in contributions.items():
                task_totals[task].append(ratio)

        task_averages = {
            task: sum(ratios) / len(ratios) for task, ratios in task_totals.items()
        }

        dominant: List[Tuple[str, float]] = [
            (task, avg) for task, avg in task_averages.items() if avg > dominance_threshold
        ]

        return {
            "is_herding": len(dominant) > 0,
            "dominant_tasks": dominant,
            "task_averages": task_averages,
            "alert_message": (
                f"Task herding warning: {', '.join(f'{t}={a:.1%}' for t, a in dominant)}"
                if dominant
                else None
            ),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_gini(values: List[int]) -> float:
        """Compute the Gini coefficient for a list of counts."""
        if not values or sum(values) == 0:
            return 0.0
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        total = sum(sorted_vals)
        gini_sum = sum((2 * (i + 1) - n - 1) * val for i, val in enumerate(sorted_vals))
        return gini_sum / (n * total) if total > 0 else 0.0

    def _escalate_to_incident(self, herding_result: Dict[str, Any]) -> None:
        """Escalate a critical herding detection to the incident system."""
        try:
            from core.monitoring.incident_reporter import IncidentReporter

            reporter = IncidentReporter()
            reporter.create_incident(
                event_type="herding_critical",
                metrics=herding_result.get("metrics", {}),
                source_module="herding_detector",
                description=herding_result.get("alert_message", "Critical herding detected"),
            )
            logger.warning("Herding incident escalated (critical severity).")
        except Exception as exc:
            logger.warning("Herding incident escalation failed: %s", exc)


__all__ = ["HerdingDetector", "HerdingMetrics"]
