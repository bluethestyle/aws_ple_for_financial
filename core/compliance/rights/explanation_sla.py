"""
ExplanationSLATracker - SLATracker specialization for explanation requests.

Legal basis
-----------
- 개인정보 보호법 §37의2 + 시행령 §44의2 (자동화된 결정에 대한 거부·설명
  요구의 방법 및 절차).
- 응답(처리) 법정 기한: 시행령 §44의3⑤ — 요구를 받은 날부터 30일 이내.
  본 트래커는 이보다 엄격한 내부 SLA(기본 10일)를 적용한다(과준수, config로
  조정 가능). 10일은 법정 기한이 아니라 내부 목표치이며, 법정 한도는 30일이다.

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
    """Pre-wired SLATracker for explanation requests.

    Internal SLA defaults to 10 days — stricter than the 30-day legal maximum
    (개인정보 보호법 시행령 §44의3⑤). The 10-day value is an internal target,
    not the statutory deadline.
    """

    # 내부 SLA 기본 10일 = 법정 30일(시행령 §44의3⑤)보다 엄격한 과준수 목표치.
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
