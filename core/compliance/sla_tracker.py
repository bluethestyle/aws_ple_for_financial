"""
SLATracker - deadline enforcement for compliance requests.

Rather than per-right-type trackers, a single base class reads an SLA
definition (from config) and queries a ComplianceStore to surface pending
requests + breaches.

Consumers (M6 ExplanationSLATracker, M4 OptOutManager, M5 ProfilingRights)
subclass only to declare which request_type(s) they own.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Optional, Sequence

from core.compliance.store import ComplianceStore
from core.compliance.types import (
    ComplianceEvent,
    ComplianceRequest,
    EventType,
    RequestStatus,
    new_event_id,
    new_request_id,
    utcnow,
)

logger = logging.getLogger(__name__)

__all__ = [
    "SLADefinition",
    "SLATracker",
    "SLAReport",
]


@dataclass
class SLADefinition:
    """Config-driven SLA for a family of request types."""

    name: str
    request_types: Sequence[str]
    sla_days: int
    warning_days_before: int = 2   # amber window

    def deadline_from(self, submitted_at: datetime) -> datetime:
        return submitted_at + timedelta(days=self.sla_days)

    def warning_threshold(self, submitted_at: datetime) -> datetime:
        return self.deadline_from(submitted_at) - timedelta(
            days=self.warning_days_before
        )


@dataclass
class SLAReport:
    """Snapshot of SLA state for a given window."""

    window_start: datetime
    window_end: datetime
    total_requests: int
    processed_on_time: int
    processed_late: int
    breached_pending: int
    approaching: int
    by_type: dict = field(default_factory=dict)

    @property
    def compliance_rate(self) -> float:
        completed = self.processed_on_time + self.processed_late
        if completed == 0:
            return 1.0
        return self.processed_on_time / completed


class SLATracker:
    """Generic SLA tracker. Subclass by supplying an SLADefinition."""

    def __init__(
        self,
        store: ComplianceStore,
        definition: SLADefinition,
    ) -> None:
        self._store = store
        self._def = definition

    # -- Creation ---------------------------------------------------------

    def open_request(
        self,
        user_id: str,
        request_type: str,
        submitted_at: Optional[datetime] = None,
        metadata: Optional[dict] = None,
    ) -> ComplianceRequest:
        if request_type not in self._def.request_types:
            raise ValueError(
                f"request_type={request_type!r} not owned by "
                f"SLA={self._def.name!r} "
                f"(owns {tuple(self._def.request_types)})"
            )
        submitted_at = submitted_at or utcnow()
        req = ComplianceRequest(
            request_id=new_request_id(),
            user_id=user_id,
            request_type=request_type,
            submitted_at=submitted_at,
            sla_deadline=self._def.deadline_from(submitted_at),
            status=RequestStatus.PENDING,
            metadata=dict(metadata or {}),
        )
        self._store.put_request(req)
        self._store.put_event(
            ComplianceEvent(
                event_id=new_event_id(),
                user_id=user_id,
                event_type=EventType.REQUEST_CREATED,
                timestamp=submitted_at,
                payload={
                    "request_type": request_type,
                    "sla_deadline": req.sla_deadline.isoformat(),
                    "sla_name": self._def.name,
                },
                request_id=req.request_id,
            )
        )
        return req

    # -- Completion -------------------------------------------------------

    def mark_processed(
        self,
        request_id: str,
        processed_at: Optional[datetime] = None,
        payload: Optional[dict] = None,
    ) -> None:
        processed_at = processed_at or utcnow()
        req = self._store.get_request(request_id)
        if req is None:
            raise KeyError(f"Unknown request_id={request_id!r}")
        was_breached = processed_at > req.sla_deadline
        self._store.update_request_status(
            request_id, RequestStatus.PROCESSED, processed_at=processed_at
        )
        self._store.put_event(
            ComplianceEvent(
                event_id=new_event_id(),
                user_id=req.user_id,
                event_type=EventType.REQUEST_PROCESSED,
                timestamp=processed_at,
                payload={
                    "sla_name": self._def.name,
                    "on_time": not was_breached,
                    **(payload or {}),
                },
                request_id=request_id,
            )
        )
        if was_breached:
            self._emit_breach(req, processed_at, breach_type="late_processing")

    # -- Sweep queries ----------------------------------------------------

    def list_pending(self, user_id: Optional[str] = None) -> List[ComplianceRequest]:
        pending = self._store.list_requests(
            user_id=user_id, status=RequestStatus.PENDING
        )
        owned = set(self._def.request_types)
        return [r for r in pending if r.request_type in owned]

    def list_approaching(
        self, now: Optional[datetime] = None
    ) -> List[ComplianceRequest]:
        now = now or utcnow()
        out = []
        for req in self.list_pending():
            threshold = self._def.warning_threshold(req.submitted_at)
            if threshold <= now <= req.sla_deadline:
                out.append(req)
        return out

    def list_breached(
        self, now: Optional[datetime] = None
    ) -> List[ComplianceRequest]:
        now = now or utcnow()
        return [
            r for r in self.list_pending()
            if now > r.sla_deadline
        ]

    def sweep_breaches(self, now: Optional[datetime] = None) -> List[ComplianceRequest]:
        """Emit SLA_BREACH event for each newly-breached pending request."""
        now = now or utcnow()
        breached = self.list_breached(now=now)
        for req in breached:
            self._emit_breach(req, now, breach_type="pending_overdue")
        return breached

    # -- Reporting --------------------------------------------------------

    def generate_report(
        self,
        window_start: datetime,
        window_end: datetime,
    ) -> SLAReport:
        owned = set(self._def.request_types)
        all_in_window: List[ComplianceRequest] = []
        for rtype in owned:
            all_in_window.extend(
                self._store.list_requests(request_type=rtype)
            )
        all_in_window = [
            r for r in all_in_window
            if window_start <= r.submitted_at <= window_end
        ]

        on_time = late = breached = approaching = 0
        by_type: dict = {}
        now = utcnow()
        for r in all_in_window:
            bucket = by_type.setdefault(r.request_type, {
                "total": 0, "on_time": 0, "late": 0,
                "breached_pending": 0, "approaching": 0,
            })
            bucket["total"] += 1
            if r.status == RequestStatus.PROCESSED and r.processed_at:
                if r.processed_at <= r.sla_deadline:
                    on_time += 1
                    bucket["on_time"] += 1
                else:
                    late += 1
                    bucket["late"] += 1
            elif r.status == RequestStatus.PENDING:
                if now > r.sla_deadline:
                    breached += 1
                    bucket["breached_pending"] += 1
                elif self._def.warning_threshold(r.submitted_at) <= now:
                    approaching += 1
                    bucket["approaching"] += 1

        return SLAReport(
            window_start=window_start,
            window_end=window_end,
            total_requests=len(all_in_window),
            processed_on_time=on_time,
            processed_late=late,
            breached_pending=breached,
            approaching=approaching,
            by_type=by_type,
        )

    # -- Internals --------------------------------------------------------

    def _emit_breach(
        self,
        req: ComplianceRequest,
        at: datetime,
        breach_type: str,
    ) -> None:
        self._store.put_event(
            ComplianceEvent(
                event_id=new_event_id(),
                user_id=req.user_id,
                event_type=EventType.SLA_BREACH,
                timestamp=at,
                payload={
                    "sla_name": self._def.name,
                    "sla_days": self._def.sla_days,
                    "request_type": req.request_type,
                    "breach_type": breach_type,
                    "submitted_at": req.submitted_at.isoformat(),
                    "sla_deadline": req.sla_deadline.isoformat(),
                },
                request_id=req.request_id,
            )
        )
        logger.warning(
            "SLA breach: name=%s request_id=%s user_id=%s type=%s deadline=%s",
            self._def.name,
            req.request_id,
            req.user_id,
            breach_type,
            req.sla_deadline.isoformat(),
        )
