"""
Tier 1 SelfChecker Aggregator for Audit Agent AV3
====================================================

Aggregates SelfChecker pass/revise/reject results over time periods
for trend analysis and dashboard reporting.

Monitors:
    - Overall pass/revise/reject rates
    - Per-task breakdown
    - Per-layer (L1/L2a/L2b) breakdown
    - Trend detection (reject rate increasing over N-day window)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["Tier1Aggregator", "Tier1Summary"]


@dataclass
class Tier1Summary:
    """Aggregated Tier 1 SelfChecker summary."""
    period: str  # e.g., "2026-04-10" or "2026-W15"
    total_checked: int = 0
    pass_count: int = 0
    revise_count: int = 0
    reject_count: int = 0

    by_task: Dict[str, Dict[str, int]] = field(default_factory=dict)
    by_layer: Dict[str, Dict[str, int]] = field(default_factory=dict)

    trend_alert: bool = False
    trend_detail: str = ""

    @property
    def pass_rate(self) -> float:
        return self.pass_count / self.total_checked if self.total_checked else 0.0

    @property
    def reject_rate(self) -> float:
        return self.reject_count / self.total_checked if self.total_checked else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "period": self.period,
            "total_checked": self.total_checked,
            "pass_rate": round(self.pass_rate, 4),
            "reject_rate": round(self.reject_rate, 4),
            "revise_rate": round(
                self.revise_count / self.total_checked if self.total_checked else 0.0, 4
            ),
            "by_task": self.by_task,
            "by_layer": self.by_layer,
            "trend_alert": self.trend_alert,
            "trend_detail": self.trend_detail,
        }


class Tier1Aggregator:
    """Aggregates SelfChecker results for trend analysis.

    Args:
        config: Config dict with keys:
            - pass_rate_warn_threshold: float (default 0.95)
            - trend_window_days: int (default 7)
            - trend_reject_increase_threshold: float (default 0.02)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self._pass_rate_warn = cfg.get("pass_rate_warn_threshold", 0.95)
        self._trend_window = cfg.get("trend_window_days", 7)
        self._trend_threshold = cfg.get("trend_reject_increase_threshold", 0.02)
        self._history: List[Tier1Summary] = []

    def aggregate(
        self,
        selfcheck_results: List[Dict[str, Any]],
        period: str = "",
    ) -> Tier1Summary:
        """Aggregate a batch of SelfChecker results.

        Args:
            selfcheck_results: List of dicts with keys:
                verdict ("pass"/"revise"/"reject"), task_type, reason_layer
            period: Label for this aggregation period.
        """
        if not period:
            period = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        summary = Tier1Summary(period=period)
        summary.total_checked = len(selfcheck_results)

        by_task: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        by_layer: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for r in selfcheck_results:
            verdict = r.get("verdict", "pass")
            task = r.get("task_type", "unknown")
            layer = r.get("reason_layer", "L1")

            if verdict == "pass":
                summary.pass_count += 1
            elif verdict == "revise":
                summary.revise_count += 1
            elif verdict == "reject":
                summary.reject_count += 1

            by_task[task][verdict] = by_task[task].get(verdict, 0) + 1
            by_layer[layer][verdict] = by_layer[layer].get(verdict, 0) + 1

        summary.by_task = {k: dict(v) for k, v in by_task.items()}
        summary.by_layer = {k: dict(v) for k, v in by_layer.items()}

        # Trend detection
        self._history.append(summary)
        summary.trend_alert, summary.trend_detail = self._detect_trend()

        return summary

    def _detect_trend(self) -> tuple:
        """Detect increasing reject rate trend over window."""
        if len(self._history) < 2:
            return False, ""

        window = self._history[-self._trend_window:]
        if len(window) < 2:
            return False, ""

        rates = [s.reject_rate for s in window]

        # Check if reject rate is monotonically increasing over window
        increasing_days = sum(
            1 for i in range(1, len(rates)) if rates[i] > rates[i - 1]
        )

        # Alert if reject rate increased in majority of window AND
        # overall increase exceeds threshold
        if len(rates) >= 3 and increasing_days >= len(rates) - 1:
            increase = rates[-1] - rates[0]
            if increase >= self._trend_threshold:
                return True, (
                    f"reject_rate가 {len(window)}일간 지속 상승 "
                    f"({rates[0]:.4f} → {rates[-1]:.4f}, +{increase:.4f})"
                )

        # Alert if current pass rate below warning threshold
        latest = self._history[-1]
        if latest.pass_rate < self._pass_rate_warn:
            return True, (
                f"pass_rate {latest.pass_rate:.4f} < "
                f"임계값 {self._pass_rate_warn}"
            )

        return False, ""

    def get_dashboard(self) -> Dict[str, Any]:
        """Get Tier 1 dashboard data for the audit report."""
        if not self._history:
            return {"status": "no_data"}

        latest = self._history[-1]
        return {
            "latest": latest.to_dict(),
            "history_length": len(self._history),
            "trend": "declining" if latest.trend_alert else "stable",
        }
