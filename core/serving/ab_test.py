"""
A/B Test Manager
================

Deterministic traffic splitting for model A/B tests.

Key design decisions:

* **Deterministic assignment** -- the same ``user_id`` always maps to the
  same variant (via a stable hash).  This prevents experience
  inconsistency across repeat visits.
* **Weighted variants** -- each :class:`~core.serving.config.ABVariant`
  carries a ``weight`` that determines its traffic share.
* **Metric recording** -- per-variant latency and prediction metrics are
  emitted to CloudWatch.
* **Auto-promote** -- optional statistical significance testing can
  automatically promote the winning variant and retire losers.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["ABTestManager", "VariantAssignment"]


@dataclass
class VariantAssignment:
    """Result of assigning a user to an A/B variant.

    Attributes:
        variant_name: The selected variant (e.g. ``"control"``).
        model_path: S3 URI of the model for this variant.
        hash_value: The normalised hash used for bucketing (in ``[0, 1)``).
    """

    variant_name: str
    model_path: str
    hash_value: float = 0.0


class ABTestManager:
    """Manages deterministic A/B variant assignment and metric emission.

    Args:
        variants: List of :class:`~core.serving.config.ABVariant`.
        experiment_name: Logical experiment name (used in CloudWatch
            namespace and DynamoDB metrics table).
        salt: Extra salt mixed into the hash to allow re-randomising
            users across experiments.
        cloudwatch_namespace: CloudWatch custom namespace for metrics.
    """

    def __init__(
        self,
        variants: List["ABVariant"],
        experiment_name: str = "ple_ab_test",
        salt: str = "ple_default_salt",
        cloudwatch_namespace: str = "PLE/ABTest",
    ) -> None:
        from .config import ABVariant  # noqa: F811 -- deferred import

        if not variants:
            raise ValueError("ABTestManager requires at least one variant")

        self._variants = sorted(variants, key=lambda v: v.name)
        self._experiment = experiment_name
        self._salt = salt
        self._cw_namespace = cloudwatch_namespace

        # Pre-compute cumulative weight boundaries for O(1) lookup
        total_weight = sum(v.weight for v in self._variants)
        if abs(total_weight - 1.0) > 1e-4:
            logger.warning(
                "ABTestManager: variant weights sum to %.4f, not 1.0.  "
                "Normalising.",
                total_weight,
            )
            for v in self._variants:
                v.weight /= total_weight

        self._boundaries: List[float] = []
        cumulative = 0.0
        for v in self._variants:
            cumulative += v.weight
            self._boundaries.append(cumulative)

        logger.info(
            "ABTestManager: experiment=%s, variants=%s",
            experiment_name,
            [(v.name, round(v.weight, 4)) for v in self._variants],
        )

    # ------------------------------------------------------------------
    # Variant selection
    # ------------------------------------------------------------------

    def assign(self, user_id: str) -> VariantAssignment:
        """Deterministically assign a user to a variant.

        The assignment is stable: the same ``user_id`` always returns the
        same variant (as long as variant definitions and salt are unchanged).

        Args:
            user_id: Unique user identifier.

        Returns:
            :class:`VariantAssignment` with the selected variant info.
        """
        h = self._hash_user(user_id)

        for boundary, variant in zip(self._boundaries, self._variants):
            if h < boundary:
                return VariantAssignment(
                    variant_name=variant.name,
                    model_path=variant.model_path,
                    hash_value=h,
                )

        # Fallback (rounding edge case)
        last = self._variants[-1]
        return VariantAssignment(
            variant_name=last.name,
            model_path=last.model_path,
            hash_value=h,
        )

    def _hash_user(self, user_id: str) -> float:
        """Produce a stable float in ``[0, 1)`` from a user_id.

        Uses SHA-256 with the experiment salt for uniform distribution
        and cross-experiment independence.
        """
        raw = f"{self._salt}:{self._experiment}:{user_id}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        # Take the first 8 hex chars (32 bits) for sufficient precision
        return int(digest[:8], 16) / 0x1_0000_0000

    # ------------------------------------------------------------------
    # Metric recording
    # ------------------------------------------------------------------

    def record_metric(
        self,
        variant_name: str,
        metric_name: str,
        value: float,
        unit: str = "None",
    ) -> None:
        """Emit a per-variant metric to CloudWatch.

        Args:
            variant_name: The variant this metric belongs to.
            metric_name: Metric name (e.g. ``"Latency"``, ``"Score"``).
            value: Metric value.
            unit: CloudWatch unit (``"Milliseconds"``, ``"None"``, etc.).
        """
        try:
            import boto3

            cw = boto3.client("cloudwatch")
            cw.put_metric_data(
                Namespace=self._cw_namespace,
                MetricData=[
                    {
                        "MetricName": metric_name,
                        "Dimensions": [
                            {"Name": "Experiment", "Value": self._experiment},
                            {"Name": "Variant", "Value": variant_name},
                        ],
                        "Value": value,
                        "Unit": unit,
                    }
                ],
            )
        except Exception:
            # Metric emission must never fail the request
            logger.debug(
                "ABTestManager: failed to emit CloudWatch metric %s=%s",
                metric_name, value,
                exc_info=True,
            )

    def record_latency(
        self, variant_name: str, elapsed_ms: float,
    ) -> None:
        """Convenience: emit a latency metric in milliseconds."""
        self.record_metric(variant_name, "Latency", elapsed_ms, "Milliseconds")

    # ------------------------------------------------------------------
    # Auto-promote (significance test)
    # ------------------------------------------------------------------

    def evaluate_significance(
        self,
        control_conversions: int,
        control_total: int,
        challenger_conversions: int,
        challenger_total: int,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """Run a two-proportion z-test to decide if the challenger wins.

        This is a simplified frequentist test suitable for binary outcomes
        (e.g. click / no-click).

        Args:
            control_conversions: Number of conversions in the control group.
            control_total: Total observations in the control group.
            challenger_conversions: Number of conversions in the challenger.
            challenger_total: Total observations in the challenger.
            alpha: Significance level (default 0.05).

        Returns:
            Dict with keys: ``significant`` (bool), ``p_value`` (float),
            ``winner`` (str), ``lift`` (float).
        """
        import math

        if control_total == 0 or challenger_total == 0:
            return {
                "significant": False,
                "p_value": 1.0,
                "winner": "none",
                "lift": 0.0,
                "reason": "insufficient_data",
            }

        p_c = control_conversions / control_total
        p_t = challenger_conversions / challenger_total

        # Pooled proportion
        p_pool = (
            (control_conversions + challenger_conversions)
            / (control_total + challenger_total)
        )

        # Standard error
        se = math.sqrt(
            p_pool * (1 - p_pool) * (1 / control_total + 1 / challenger_total)
        ) if p_pool > 0 and p_pool < 1 else 0.0

        if se == 0:
            return {
                "significant": False,
                "p_value": 1.0,
                "winner": "none",
                "lift": 0.0,
                "reason": "zero_variance",
            }

        z = (p_t - p_c) / se

        # Two-tailed p-value via normal CDF approximation
        p_value = 2.0 * (1.0 - self._normal_cdf(abs(z)))

        lift = (p_t - p_c) / p_c if p_c > 0 else 0.0
        significant = p_value < alpha
        winner = "challenger" if significant and p_t > p_c else (
            "control" if significant else "none"
        )

        result = {
            "significant": significant,
            "p_value": round(p_value, 6),
            "winner": winner,
            "lift": round(lift, 4),
            "control_rate": round(p_c, 4),
            "challenger_rate": round(p_t, 4),
            "z_score": round(z, 4),
        }

        if significant:
            logger.info(
                "ABTestManager: significance reached -- winner=%s, "
                "lift=%.2f%%, p=%.4f",
                winner, lift * 100, p_value,
            )

        return result

    def auto_promote(
        self,
        control_conversions: int,
        control_total: int,
        challenger_conversions: int,
        challenger_total: int,
        min_observations: int = 10_000,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """Evaluate significance and return a promotion recommendation.

        Args:
            control_conversions: Conversions in the control arm.
            control_total: Total samples in the control arm.
            challenger_conversions: Conversions in the challenger arm.
            challenger_total: Total samples in the challenger arm.
            min_observations: Minimum per-arm observations before testing.
            alpha: Significance level.

        Returns:
            Dict with ``action`` (``"promote"``, ``"keep"``, ``"wait"``),
            plus the significance test result.
        """
        if control_total < min_observations or challenger_total < min_observations:
            return {
                "action": "wait",
                "reason": (
                    f"insufficient observations: control={control_total}, "
                    f"challenger={challenger_total}, min={min_observations}"
                ),
            }

        sig = self.evaluate_significance(
            control_conversions, control_total,
            challenger_conversions, challenger_total,
            alpha=alpha,
        )

        if sig["significant"] and sig["winner"] == "challenger":
            action = "promote"
        elif sig["significant"] and sig["winner"] == "control":
            action = "keep"
        else:
            action = "wait"

        return {"action": action, **sig}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normal_cdf(x: float) -> float:
        """Approximate the standard normal CDF using the error function.

        Good to ~7 decimal places for |x| < 6.
        """
        import math
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
