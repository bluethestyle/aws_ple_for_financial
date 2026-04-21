"""
Compliance foundation types.

ComplianceRequest = a user-facing regulatory request (consent / opt-out /
profiling access / explanation) that has an SLA and a lifecycle.

ComplianceEvent = an immutable audit record of a regulatory-relevant action.

Both are deliberately minimal dataclasses. Persistence is delegated to the
ComplianceStore backends defined in core/compliance/store.py.

Legal context
-------------
- 개보법 시행령 §44의2~4 (설명요구권 10일 SLA)
- AI기본법 §31 (자동화 결정 거부권), §35 (FRIA 5년 보존)
- 신정법 §36의2 (프로파일링 권리)
- 금소법 §17 (적합성 원칙) - suitability requests
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional


__all__ = [
    "ComplianceRequest",
    "ComplianceEvent",
    "RequestStatus",
    "RequestType",
    "EventType",
    "new_request_id",
    "new_event_id",
    "utcnow",
]


# ---------------------------------------------------------------------------
# Enumerations kept as string constants (stable over wire / across processes)
# ---------------------------------------------------------------------------

class RequestStatus:
    PENDING = "pending"
    PROCESSED = "processed"
    EXPIRED = "expired"
    REJECTED = "rejected"

    VALID = frozenset({PENDING, PROCESSED, EXPIRED, REJECTED})


class RequestType:
    CONSENT_GRANT = "consent_grant"
    CONSENT_REVOKE = "consent_revoke"
    OPT_OUT = "opt_out"
    OPT_OUT_REVOKE = "opt_out_revoke"
    PROFILING_ACCESS = "profiling_access"
    PROFILING_CORRECTION = "profiling_correction"
    PROFILING_DELETION = "profiling_deletion"
    EXPLANATION = "explanation"

    VALID = frozenset({
        CONSENT_GRANT, CONSENT_REVOKE,
        OPT_OUT, OPT_OUT_REVOKE,
        PROFILING_ACCESS, PROFILING_CORRECTION, PROFILING_DELETION,
        EXPLANATION,
    })


class EventType:
    REQUEST_CREATED = "request_created"
    REQUEST_PROCESSED = "request_processed"
    REQUEST_EXPIRED = "request_expired"
    SLA_BREACH = "sla_breach"
    FRIA_ASSESSMENT = "fria_assessment"
    AI_RISK_ASSESSMENT = "ai_risk_assessment"
    HUMAN_REVIEW_DISPOSITION = "human_review_disposition"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_request_id() -> str:
    return f"req_{uuid.uuid4().hex}"


def new_event_id() -> str:
    return f"evt_{uuid.uuid4().hex}"


def _iso(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    return datetime.fromisoformat(value)


# ---------------------------------------------------------------------------
# ComplianceRequest
# ---------------------------------------------------------------------------

@dataclass
class ComplianceRequest:
    """A user-filed regulatory request with an SLA."""

    request_id: str
    user_id: str
    request_type: str
    submitted_at: datetime
    sla_deadline: datetime
    status: str = RequestStatus.PENDING
    processed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.request_type not in RequestType.VALID:
            raise ValueError(
                f"Unknown request_type={self.request_type!r}; "
                f"must be one of {sorted(RequestType.VALID)}"
            )
        if self.status not in RequestStatus.VALID:
            raise ValueError(
                f"Unknown status={self.status!r}; "
                f"must be one of {sorted(RequestStatus.VALID)}"
            )
        if self.sla_deadline < self.submitted_at:
            raise ValueError("sla_deadline must be >= submitted_at")

    def is_overdue(self, now: Optional[datetime] = None) -> bool:
        if self.status != RequestStatus.PENDING:
            return False
        current = now or utcnow()
        return current > self.sla_deadline

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["submitted_at"] = _iso(self.submitted_at)
        d["sla_deadline"] = _iso(self.sla_deadline)
        d["processed_at"] = _iso(self.processed_at)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComplianceRequest":
        return cls(
            request_id=data["request_id"],
            user_id=data["user_id"],
            request_type=data["request_type"],
            submitted_at=_parse_iso(data["submitted_at"]),  # type: ignore[arg-type]
            sla_deadline=_parse_iso(data["sla_deadline"]),  # type: ignore[arg-type]
            status=data.get("status", RequestStatus.PENDING),
            processed_at=_parse_iso(data.get("processed_at")),
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def from_json(cls, raw: str) -> "ComplianceRequest":
        return cls.from_dict(json.loads(raw))


# ---------------------------------------------------------------------------
# ComplianceEvent
# ---------------------------------------------------------------------------

@dataclass
class ComplianceEvent:
    """An immutable audit record of a regulatory action."""

    event_id: str
    user_id: str
    event_type: str
    timestamp: datetime
    payload: Dict[str, Any] = field(default_factory=dict)
    request_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = _iso(self.timestamp)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComplianceEvent":
        return cls(
            event_id=data["event_id"],
            user_id=data["user_id"],
            event_type=data["event_type"],
            timestamp=_parse_iso(data["timestamp"]),  # type: ignore[arg-type]
            payload=dict(data.get("payload", {})),
            request_id=data.get("request_id"),
        )

    @classmethod
    def from_json(cls, raw: str) -> "ComplianceEvent":
        return cls.from_dict(json.loads(raw))
