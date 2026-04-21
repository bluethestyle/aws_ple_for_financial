"""
ProfilingWorkflow - profiling rights (access / correction / deletion).

Legal basis: 신정법 §36의2, 개보법 §35~37.

Three rights with a shared lifecycle:
- access     : disclose what profiling data / features were used
- correction : update profiling data (field + new value)
- deletion   : erase profiling data (scope = full | profiling_only)

Each request has a 30-day SLA by default (신정법 §36의2 시행령).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

from core.compliance.store import ComplianceStore
from core.compliance.types import (
    ComplianceEvent,
    ComplianceRequest,
    EventType,
    RequestStatus,
    RequestType,
    new_event_id,
    new_request_id,
    utcnow,
)

logger = logging.getLogger(__name__)


VALID_DELETION_SCOPES = ("full", "profiling_only")


@dataclass
class ProfilingConfig:
    """Config for ProfilingWorkflow. Consumes compliance.profiling block."""

    sla_days: int = 30
    warning_days_before: int = 5

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ProfilingConfig":
        if not data:
            return cls()
        return cls(
            sla_days=int(data.get("sla_days", 30)),
            warning_days_before=int(data.get("warning_days_before", 5)),
        )


@dataclass
class ProfilingAccessResult:
    """Payload returned when fulfilling an access request."""

    request_id: str
    user_id: str
    snapshot: Dict[str, Any] = field(default_factory=dict)
    disclosed_features: List[str] = field(default_factory=list)
    legal_basis: str = "신정법 §36의2 / 개보법 §35"
    fulfilled_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "snapshot": self.snapshot,
            "disclosed_features": list(self.disclosed_features),
            "legal_basis": self.legal_basis,
            "fulfilled_at": (
                self.fulfilled_at.isoformat() if self.fulfilled_at else None
            ),
        }


ProfileProviderFn = Callable[[str], Dict[str, Any]]


class ProfilingWorkflow:
    """Profiling access / correction / deletion workflow (M5)."""

    def __init__(
        self,
        store: ComplianceStore,
        config: Optional[ProfilingConfig] = None,
        profile_provider: Optional[ProfileProviderFn] = None,
    ) -> None:
        self._store = store
        self._cfg = config or ProfilingConfig()
        self._provider = profile_provider

    # ------------------------------------------------------------------
    # Request submission
    # ------------------------------------------------------------------

    def request_access(
        self,
        user_id: str,
        reason: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ComplianceRequest:
        return self._open_request(
            user_id=user_id,
            request_type=RequestType.PROFILING_ACCESS,
            metadata={
                "reason": reason,
                "legal_basis": "신정법 §36의2 / 개보법 §35",
                **(metadata or {}),
            },
        )

    def request_correction(
        self,
        user_id: str,
        field_name: str,
        new_value: Any,
        reason: str = "",
    ) -> ComplianceRequest:
        return self._open_request(
            user_id=user_id,
            request_type=RequestType.PROFILING_CORRECTION,
            metadata={
                "field": field_name,
                "new_value": new_value,
                "reason": reason,
                "legal_basis": "개보법 §36",
            },
        )

    def request_deletion(
        self,
        user_id: str,
        scope: str = "profiling_only",
        reason: str = "",
    ) -> ComplianceRequest:
        if scope not in VALID_DELETION_SCOPES:
            raise ValueError(
                f"scope={scope!r} must be in {VALID_DELETION_SCOPES}"
            )
        return self._open_request(
            user_id=user_id,
            request_type=RequestType.PROFILING_DELETION,
            metadata={
                "scope": scope,
                "reason": reason,
                "legal_basis": "개보법 §37",
            },
        )

    # ------------------------------------------------------------------
    # Fulfillment
    # ------------------------------------------------------------------

    def fulfill_access(
        self,
        request_id: str,
        disclosed_features: Optional[List[str]] = None,
    ) -> ProfilingAccessResult:
        """Resolve an access request and return a ProfilingAccessResult."""
        req = self._require_request(
            request_id, RequestType.PROFILING_ACCESS
        )

        if self._provider is None:
            snapshot: Dict[str, Any] = {}
            logger.warning(
                "ProfilingWorkflow has no profile_provider; returning "
                "empty snapshot for request_id=%s",
                request_id,
            )
        else:
            try:
                snapshot = self._provider(req.user_id) or {}
            except Exception:
                logger.exception(
                    "profile_provider failed for user_id=%s request_id=%s",
                    req.user_id, request_id,
                )
                snapshot = {}

        now = utcnow()
        self._close_request(req, now, extra_payload={
            "disclosed_feature_count": len(disclosed_features or []),
        })
        return ProfilingAccessResult(
            request_id=request_id,
            user_id=req.user_id,
            snapshot=snapshot,
            disclosed_features=list(disclosed_features or []),
            fulfilled_at=now,
        )

    def fulfill_correction(
        self,
        request_id: str,
        applied: bool,
        notes: str = "",
    ) -> None:
        req = self._require_request(
            request_id, RequestType.PROFILING_CORRECTION
        )
        now = utcnow()
        self._close_request(req, now, extra_payload={
            "applied": applied,
            "notes": notes,
        })

    def fulfill_deletion(
        self,
        request_id: str,
        deleted: bool,
        notes: str = "",
    ) -> None:
        req = self._require_request(
            request_id, RequestType.PROFILING_DELETION
        )
        now = utcnow()
        self._close_request(req, now, extra_payload={
            "deleted": deleted,
            "notes": notes,
        })

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_pending_for_user(
        self, user_id: str
    ) -> List[ComplianceRequest]:
        owned = (
            RequestType.PROFILING_ACCESS,
            RequestType.PROFILING_CORRECTION,
            RequestType.PROFILING_DELETION,
        )
        out: List[ComplianceRequest] = []
        for rtype in owned:
            out.extend(
                self._store.list_requests(
                    user_id=user_id,
                    status=RequestStatus.PENDING,
                    request_type=rtype,
                )
            )
        return sorted(out, key=lambda r: r.submitted_at)

    def list_all_pending(self) -> List[ComplianceRequest]:
        owned = (
            RequestType.PROFILING_ACCESS,
            RequestType.PROFILING_CORRECTION,
            RequestType.PROFILING_DELETION,
        )
        out: List[ComplianceRequest] = []
        for rtype in owned:
            out.extend(
                self._store.list_requests(
                    status=RequestStatus.PENDING,
                    request_type=rtype,
                )
            )
        return sorted(out, key=lambda r: r.submitted_at)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _open_request(
        self,
        user_id: str,
        request_type: str,
        metadata: Dict[str, Any],
    ) -> ComplianceRequest:
        now = utcnow()
        deadline = now + timedelta(days=self._cfg.sla_days)
        req = ComplianceRequest(
            request_id=new_request_id(),
            user_id=user_id,
            request_type=request_type,
            submitted_at=now,
            sla_deadline=deadline,
            metadata=metadata,
        )
        self._store.put_request(req)
        self._store.put_event(
            ComplianceEvent(
                event_id=new_event_id(),
                user_id=user_id,
                event_type=EventType.REQUEST_CREATED,
                timestamp=now,
                payload={
                    "sla_name": "profiling",
                    "request_type": request_type,
                    "sla_deadline": deadline.isoformat(),
                },
                request_id=req.request_id,
            )
        )
        logger.info(
            "Profiling request opened: type=%s user_id=%s request_id=%s",
            request_type, user_id, req.request_id,
        )
        return req

    def _require_request(
        self, request_id: str, expected_type: str
    ) -> ComplianceRequest:
        req = self._store.get_request(request_id)
        if req is None:
            raise KeyError(f"Unknown request_id={request_id!r}")
        if req.request_type != expected_type:
            raise ValueError(
                f"request_id={request_id!r} is type={req.request_type!r}, "
                f"expected {expected_type}"
            )
        return req

    def _close_request(
        self,
        req: ComplianceRequest,
        processed_at: datetime,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        on_time = processed_at <= req.sla_deadline
        self._store.update_request_status(
            req.request_id, RequestStatus.PROCESSED, processed_at=processed_at
        )
        self._store.put_event(
            ComplianceEvent(
                event_id=new_event_id(),
                user_id=req.user_id,
                event_type=EventType.REQUEST_PROCESSED,
                timestamp=processed_at,
                payload={
                    "sla_name": "profiling",
                    "on_time": on_time,
                    "request_type": req.request_type,
                    **(extra_payload or {}),
                },
                request_id=req.request_id,
            )
        )
        if not on_time:
            self._store.put_event(
                ComplianceEvent(
                    event_id=new_event_id(),
                    user_id=req.user_id,
                    event_type=EventType.SLA_BREACH,
                    timestamp=processed_at,
                    payload={
                        "sla_name": "profiling",
                        "breach_type": "late_processing",
                        "request_type": req.request_type,
                        "sla_deadline": req.sla_deadline.isoformat(),
                    },
                    request_id=req.request_id,
                )
            )
            logger.warning(
                "Profiling SLA breach: request_id=%s user_id=%s type=%s",
                req.request_id, req.user_id, req.request_type,
            )
