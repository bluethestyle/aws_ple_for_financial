"""
Fairness monitoring for recommendation systems.

Evaluates bias across **config-driven** protected attributes using three
standard metrics:

- **DI**  (Disparate Impact):             0.8 <= DI <= 1.25
- **SPD** (Statistical Parity Difference): |SPD| <= 0.1
- **EOD** (Equal Opportunity Difference):  |EOD| <= 0.1

When thresholds are violated an incident is auto-generated via
:class:`~core.monitoring.incident_reporter.IncidentReporter`.

All thresholds and protected attributes are configurable at instantiation
time -- nothing is hardcoded to a specific domain.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default configuration (overridable via constructor)
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS: Dict[str, float] = {
    "di_lower": 0.8,
    "di_upper": 1.25,
    "spd_max": 0.1,
    "eod_max": 0.1,
}

DEFAULT_PROTECTED_ATTRIBUTES: List[str] = [
    "age_group",
    "gender",
    "region_type",
    "income_tier",
    "life_stage",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FairnessMetrics:
    """Result of a single-attribute fairness evaluation."""

    attribute: str
    disparate_impact: float
    statistical_parity_diff: float
    equal_opportunity_diff: float
    group_counts: Dict[str, int]
    group_positive_rates: Dict[str, float]
    is_fair: bool
    violations: List[str]


# ---------------------------------------------------------------------------
# FairnessMonitor
# ---------------------------------------------------------------------------

class FairnessMonitor:
    """Evaluate recommendation fairness across protected attributes.

    Parameters
    ----------
    thresholds : dict, optional
        Override default thresholds.  Keys: ``di_lower``, ``di_upper``,
        ``spd_max``, ``eod_max``.
    protected_attributes : list of str, optional
        Override the default list of protected attribute names.
    auto_incident : bool
        When ``True`` (default), automatically create an incident report
        when a fairness violation is detected.
    """

    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
        protected_attributes: Optional[List[str]] = None,
        auto_incident: bool = True,
    ) -> None:
        self.thresholds: Dict[str, float] = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.protected_attributes: List[str] = (
            list(protected_attributes) if protected_attributes else list(DEFAULT_PROTECTED_ATTRIBUTES)
        )
        self.auto_incident = auto_incident

    # ------------------------------------------------------------------
    # Individual metric computation
    # ------------------------------------------------------------------

    def compute_disparate_impact(
        self,
        recommendations: List[Dict[str, Any]],
        attribute: str,
        privileged_group: str,
        unprivileged_group: str,
    ) -> float:
        """Compute Disparate Impact.

        DI = P(recommended | unprivileged) / P(recommended | privileged)

        Returns 0.0 when the privileged group has zero positive rate.
        """
        priv_rate = self._positive_rate(recommendations, attribute, privileged_group)
        unpriv_rate = self._positive_rate(recommendations, attribute, unprivileged_group)

        if priv_rate == 0.0:
            logger.warning(
                "[DI] Privileged group '%s' has 0%% positive rate; returning DI=0.0",
                privileged_group,
            )
            return 0.0
        return round(unpriv_rate / priv_rate, 6)

    def compute_statistical_parity_difference(
        self,
        recommendations: List[Dict[str, Any]],
        attribute: str,
        privileged: str,
        unprivileged: str,
    ) -> float:
        """Compute Statistical Parity Difference.

        SPD = P(recommended | unprivileged) - P(recommended | privileged)
        """
        priv_rate = self._positive_rate(recommendations, attribute, privileged)
        unpriv_rate = self._positive_rate(recommendations, attribute, unprivileged)
        return round(unpriv_rate - priv_rate, 6)

    def compute_equal_opportunity_difference(
        self,
        recommendations: List[Dict[str, Any]],
        attribute: str,
        privileged: str,
        unprivileged: str,
    ) -> float:
        """Compute Equal Opportunity Difference.

        EOD = TPR(unprivileged) - TPR(privileged)

        Records without ``actual_positive`` are excluded.
        """
        priv_tpr = self._true_positive_rate(recommendations, attribute, privileged)
        unpriv_tpr = self._true_positive_rate(recommendations, attribute, unprivileged)
        return round(unpriv_tpr - priv_tpr, 6)

    # ------------------------------------------------------------------
    # Comprehensive evaluation
    # ------------------------------------------------------------------

    def evaluate_fairness(
        self,
        recommendations: List[Dict[str, Any]],
        attribute: str,
        group_pairs: List[Tuple[str, str]],
    ) -> FairnessMetrics:
        """Evaluate fairness for a single protected attribute.

        Parameters
        ----------
        recommendations : list of dict
            Each dict must contain a ``"recommended"`` boolean key and the
            protected attribute key.
        attribute : str
            Name of the protected attribute.
        group_pairs : list of (privileged, unprivileged) tuples
            All pairwise comparisons to evaluate.

        Returns
        -------
        FairnessMetrics
            Worst-case metrics across all group pairs.
        """
        if not recommendations:
            logger.warning("Empty recommendations for attribute=%s", attribute)
            return FairnessMetrics(
                attribute=attribute,
                disparate_impact=1.0,
                statistical_parity_diff=0.0,
                equal_opportunity_diff=0.0,
                group_counts={},
                group_positive_rates={},
                is_fair=True,
                violations=[],
            )

        # Compute per-group statistics
        all_groups = {g for pair in group_pairs for g in pair}
        group_counts: Dict[str, int] = {}
        group_positive_rates: Dict[str, float] = {}
        for group in all_groups:
            members = [r for r in recommendations if r.get(attribute) == group]
            group_counts[group] = len(members)
            group_positive_rates[group] = self._positive_rate(recommendations, attribute, group)

        # Worst-case across pairs
        worst_di = 1.0
        worst_spd = 0.0
        worst_eod = 0.0

        for privileged, unprivileged in group_pairs:
            di = self.compute_disparate_impact(recommendations, attribute, privileged, unprivileged)
            spd = self.compute_statistical_parity_difference(
                recommendations, attribute, privileged, unprivileged,
            )
            eod = self.compute_equal_opportunity_difference(
                recommendations, attribute, privileged, unprivileged,
            )
            if di < worst_di:
                worst_di = di
            if abs(spd) > abs(worst_spd):
                worst_spd = spd
            if abs(eod) > abs(worst_eod):
                worst_eod = eod

        # Violation checks
        t = self.thresholds
        violations: List[str] = []

        if worst_di < t["di_lower"]:
            violations.append(
                f"DI below lower bound: {worst_di:.4f} < {t['di_lower']} (unprivileged disadvantaged)"
            )
        if worst_di > t["di_upper"]:
            violations.append(
                f"DI above upper bound: {worst_di:.4f} > {t['di_upper']} (privileged disadvantaged)"
            )
        if abs(worst_spd) > t["spd_max"]:
            violations.append(f"SPD exceeded: |{worst_spd:.4f}| > {t['spd_max']}")
        if abs(worst_eod) > t["eod_max"]:
            violations.append(f"EOD exceeded: |{worst_eod:.4f}| > {t['eod_max']}")

        is_fair = len(violations) == 0

        metrics = FairnessMetrics(
            attribute=attribute,
            disparate_impact=round(worst_di, 6),
            statistical_parity_diff=round(worst_spd, 6),
            equal_opportunity_diff=round(worst_eod, 6),
            group_counts=group_counts,
            group_positive_rates={k: round(v, 6) for k, v in group_positive_rates.items()},
            is_fair=is_fair,
            violations=violations,
        )

        # Auto-incident when violations detected
        if not is_fair and self.auto_incident:
            self._raise_incident(metrics)

        return metrics

    def evaluate_all_attributes(
        self,
        recommendations: List[Dict[str, Any]],
        group_pairs_by_attribute: Dict[str, List[Tuple[str, str]]],
    ) -> Dict[str, FairnessMetrics]:
        """Evaluate fairness across all configured protected attributes.

        Parameters
        ----------
        recommendations : list of dict
            Recommendation results.
        group_pairs_by_attribute : dict
            ``{attribute: [(privileged, unprivileged), ...]}``.

        Returns
        -------
        dict
            ``{attribute: FairnessMetrics}``.
        """
        results: Dict[str, FairnessMetrics] = {}
        for attribute, group_pairs in group_pairs_by_attribute.items():
            logger.info("Evaluating fairness: attribute=%s, pairs=%s", attribute, group_pairs)
            metrics = self.evaluate_fairness(recommendations, attribute, group_pairs)
            results[attribute] = metrics

            if metrics.is_fair:
                logger.info("[%s] Fairness PASSED (DI=%.4f)", attribute, metrics.disparate_impact)
            else:
                logger.warning(
                    "[%s] Fairness VIOLATED (%d issues): %s",
                    attribute,
                    len(metrics.violations),
                    "; ".join(metrics.violations),
                )
        return results

    def to_dict(self, metrics: FairnessMetrics) -> Dict[str, Any]:
        """Serialize a ``FairnessMetrics`` instance to a plain dict."""
        return {
            "attribute": metrics.attribute,
            "disparate_impact": metrics.disparate_impact,
            "statistical_parity_diff": metrics.statistical_parity_diff,
            "equal_opportunity_diff": metrics.equal_opportunity_diff,
            "group_counts": metrics.group_counts,
            "group_positive_rates": metrics.group_positive_rates,
            "is_fair": metrics.is_fair,
            "violations": metrics.violations,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _positive_rate(
        recommendations: List[Dict[str, Any]],
        attribute: str,
        group: str,
    ) -> float:
        """P(recommended=True | attribute == group)."""
        members = [r for r in recommendations if r.get(attribute) == group]
        if not members:
            return 0.0
        return sum(1 for r in members if r.get("recommended", False)) / len(members)

    @staticmethod
    def _true_positive_rate(
        recommendations: List[Dict[str, Any]],
        attribute: str,
        group: str,
    ) -> float:
        """P(recommended=True | actual_positive=True, attribute == group)."""
        actual_positives = [
            r for r in recommendations
            if r.get(attribute) == group and r.get("actual_positive", False)
        ]
        if not actual_positives:
            return 0.0
        return sum(1 for r in actual_positives if r.get("recommended", False)) / len(actual_positives)

    def _raise_incident(self, metrics: FairnessMetrics) -> None:
        """Create an incident report for a fairness violation."""
        try:
            from core.monitoring.incident_reporter import IncidentReporter

            reporter = IncidentReporter()
            reporter.create_incident(
                event_type="fairness_violation",
                metrics={
                    "attribute": metrics.attribute,
                    "di_value": metrics.disparate_impact,
                    "spd_value": metrics.statistical_parity_diff,
                    "eod_value": metrics.equal_opportunity_diff,
                    "violations": metrics.violations,
                },
                source_module="fairness_monitor",
                description=(
                    f"Fairness violation on '{metrics.attribute}': "
                    + "; ".join(metrics.violations)
                ),
            )
        except Exception as exc:
            logger.warning("Auto-incident creation failed (fairness): %s", exc)


__all__ = ["FairnessMonitor", "FairnessMetrics"]
