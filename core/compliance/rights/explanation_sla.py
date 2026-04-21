"""
ExplanationSLATracker - SLATracker specialization for 10-day explanation SLA.

Legal basis: 개보법 시행령 §44의2~4 (설명요구권 답변 10일 이내).

This is a thin specialization that wires the generic SLATracker to the
explanation request type and supplies a monthly compliance report helper.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from core.compliance.sla_tracker import (
    SLADefinition,
    SLAReport,
    SLATracker,
)
from core.compliance.store import ComplianceStore
from core.compliance.types import RequestType, utcnow

logger = logging.getLogger(__name__)


class ExplanationSLATracker(SLATracker):
    """Pre-wired SLATracker for explanation requests (10-day SLA)."""

    DEFAULT_SLA_DAYS = 10
    DEFAULT_WARNING_DAYS = 2

    def __init__(
        self,
        store: ComplianceStore,
        sla_days: int = DEFAULT_SLA_DAYS,
        warning_days_before: int = DEFAULT_WARNING_DAYS,
    ) -> None:
        definition = SLADefinition(
            name="explanation",
            request_types=(RequestType.EXPLANATION,),
            sla_days=sla_days,
            warning_days_before=warning_days_before,
        )
        super().__init__(store=store, definition=definition)

    def generate_monthly_report(
        self,
        year: int,
        month: int,
    ) -> SLAReport:
        """Generate a UTC-monthly SLA compliance report."""
        from datetime import timezone

        start = datetime(year, month, 1, tzinfo=timezone.utc)
        if month == 12:
            end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            end = datetime(year, month + 1, 1, tzinfo=timezone.utc)
        end = end - timedelta(microseconds=1)
        return self.generate_report(window_start=start, window_end=end)

    def summarize_current_state(
        self, now: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Return a small dict suitable for dashboards / logs."""
        now = now or utcnow()
        pending = self.list_pending()
        approaching = self.list_approaching(now=now)
        breached = self.list_breached(now=now)
        return {
            "pending": len(pending),
            "approaching_deadline": len(approaching),
            "breached": len(breached),
            "as_of": now.isoformat(),
        }


def build_explanation_sla_tracker(
    store: ComplianceStore, config: Optional[Dict[str, Any]] = None,
) -> ExplanationSLATracker:
    """Factory consuming the `compliance.sla` block from pipeline.yaml."""
    if not config:
        return ExplanationSLATracker(store=store)
    return ExplanationSLATracker(
        store=store,
        sla_days=int(config.get("explanation_response_days", 10)),
        warning_days_before=int(
            config.get("explanation_warning_days_before", 2)
        ),
    )
