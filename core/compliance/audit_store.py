"""
Compliance Audit Store
======================

DynamoDB-backed unified compliance audit store -- the cloud equivalent of
the original project's DuckDB audit database.

Seven audit tables (DynamoDB tables with partition key ``pk`` + sort key ``sk``):

- ``ple-audit-killswitch``   Kill switch activation / deactivation events
- ``ple-audit-consent``      Marketing consent changes
- ``ple-audit-profiling``    GDPR profiling rights exercises
- ``ple-audit-optout``       AI decision opt-out records
- ``ple-audit-incident``     Regulatory violation incidents
- ``ple-audit-distillation`` Model distillation validation records
- ``ple-audit-embedding``    Embedding quality audit records

All writes are append-only (no update/delete by design).
Timestamps are ISO 8601 UTC strings; event IDs are UUID4.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "ComplianceAuditStore",
    "InMemoryAuditStore",
    "get_audit_store",
]

# ---------------------------------------------------------------------------
# Table suffixes
# ---------------------------------------------------------------------------

TABLE_SUFFIXES = (
    "killswitch",
    "consent",
    "profiling",
    "optout",
    "incident",
    "distillation",
    "embedding",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decimal_safe(value: Any) -> Any:
    """Convert float values to Decimal for DynamoDB compatibility."""
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, dict):
        return {k: _decimal_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decimal_safe(v) for v in value]
    return value


def _from_decimal(value: Any) -> Any:
    """Convert Decimal values back to float for JSON serialisation."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {k: _from_decimal(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_from_decimal(v) for v in value]
    return value


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# DynamoDB-backed store
# ---------------------------------------------------------------------------

class ComplianceAuditStore:
    """DynamoDB-backed unified compliance audit store.

    Tables:
      - ple-audit-killswitch:   Kill switch activation/deactivation events
      - ple-audit-consent:      Marketing consent changes
      - ple-audit-profiling:    GDPR profiling rights exercises
      - ple-audit-optout:       AI decision opt-out records
      - ple-audit-incident:     Regulatory violation incidents
      - ple-audit-distillation: Model distillation validation records
      - ple-audit-embedding:    Embedding quality audit records

    Parameters
    ----------
    table_prefix : str
        Prefix for DynamoDB table names (e.g. ``"ple-audit"``).
    region : str
        AWS region for DynamoDB.
    """

    def __init__(
        self,
        table_prefix: str = "ple-audit",
        region: str = "ap-northeast-2",
    ) -> None:
        self._prefix = table_prefix
        self._region = region
        self._dynamo = None
        self._tables: Dict[str, Any] = {}

        try:
            import boto3
            self._dynamo = boto3.resource("dynamodb", region_name=region)
        except Exception as exc:
            logger.warning("boto3 DynamoDB init failed (%s); audit writes will be no-ops.", exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _table_name(self, suffix: str) -> str:
        return f"{self._prefix}-{suffix}"

    def _get_table(self, suffix: str) -> Any:
        """Return a cached DynamoDB Table resource."""
        if self._dynamo is None:
            return None
        if suffix not in self._tables:
            self._tables[suffix] = self._dynamo.Table(self._table_name(suffix))
        return self._tables[suffix]

    # ------------------------------------------------------------------
    # Generic log / query
    # ------------------------------------------------------------------

    def log_event(self, table_suffix: str, event: Dict[str, Any]) -> None:
        """Write an audit event.  Auto-adds ``event_id`` and ``sk`` (timestamp).

        Parameters
        ----------
        table_suffix : str
            One of the seven table suffixes (e.g. ``"killswitch"``).
        event : dict
            Must contain a ``"pk"`` field.  All other fields are stored as-is.
        """
        table = self._get_table(table_suffix)
        if table is None:
            logger.error(
                "DynamoDB not available; audit event dropped (table=%s).",
                table_suffix,
            )
            return

        ts = _now_iso()
        record = {
            "sk": ts,
            "event_id": str(uuid.uuid4()),
            **_decimal_safe(event),
        }
        # Remove None values (DynamoDB rejects None)
        record = {k: v for k, v in record.items() if v is not None}

        try:
            table.put_item(Item=record)
        except Exception:
            logger.exception("Failed to write audit event to %s", table_suffix)

    def query_events(
        self,
        table_suffix: str,
        partition_key: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query events by partition key and optional time range.

        Parameters
        ----------
        table_suffix : str
            Table suffix to query.
        partition_key : str
            Partition key value.
        start_time, end_time : str, optional
            ISO 8601 bounds for the sort key.
        limit : int
            Maximum records to return.

        Returns
        -------
        list of dict
            Matching records, newest first.
        """
        table = self._get_table(table_suffix)
        if table is None:
            logger.warning("DynamoDB not available; returning empty result.")
            return []

        try:
            from boto3.dynamodb.conditions import Key

            key_cond = Key("pk").eq(partition_key)
            if start_time and end_time:
                key_cond = key_cond & Key("sk").between(start_time, end_time)
            elif start_time:
                key_cond = key_cond & Key("sk").gte(start_time)
            elif end_time:
                key_cond = key_cond & Key("sk").lte(end_time)

            response = table.query(
                KeyConditionExpression=key_cond,
                ScanIndexForward=False,
                Limit=limit,
            )
            return [_from_decimal(item) for item in response.get("Items", [])]
        except Exception:
            logger.exception("Query failed on %s", table_suffix)
            return []

    # ------------------------------------------------------------------
    # Domain-specific convenience methods
    # ------------------------------------------------------------------

    def log_killswitch(
        self,
        action: str,
        level: str,
        target: str,
        reason: str,
        actor: str,
    ) -> None:
        """Log a kill switch activation or deactivation event.

        Parameters
        ----------
        action : str
            ``"ACTIVATE"`` or ``"DEACTIVATE"``.
        level : str
            Scope level (``"global"`` / ``"per_task"`` / ``"per_cluster"``).
        target : str
            Scope key (e.g. ``"global"``, ``"task:cvr"``).
        reason : str
            Free-text reason.
        actor : str
            Identity of the operator.
        """
        self.log_event("killswitch", {
            "pk": f"{action}#{level}",
            "action": action,
            "level": level,
            "target": target,
            "reason": reason,
            "actor": actor,
        })

    def log_consent(
        self,
        customer_id: str,
        channel: str,
        action: str,
        source: str,
    ) -> None:
        """Log a marketing consent change event.

        Parameters
        ----------
        customer_id : str
            Customer identifier.
        channel : str
            Marketing channel (e.g. ``"email"``, ``"sms"``, ``"push"``).
        action : str
            ``"grant"`` or ``"revoke"``.
        source : str
            Where the consent change originated.
        """
        self.log_event("consent", {
            "pk": customer_id,
            "channel": channel,
            "action": action,
            "source": source,
        })

    def log_profiling(
        self,
        customer_id: str,
        right_type: str,
        status: str,
    ) -> None:
        """Log a GDPR profiling rights exercise.

        Parameters
        ----------
        customer_id : str
            Customer identifier.
        right_type : str
            Type of right exercised (e.g. ``"access"``, ``"rectification"``,
            ``"erasure"``, ``"restriction"``, ``"objection"``).
        status : str
            Processing status (``"requested"`` / ``"processing"`` /
            ``"completed"`` / ``"denied"``).
        """
        self.log_event("profiling", {
            "pk": customer_id,
            "right_type": right_type,
            "status": status,
        })

    def log_optout(
        self,
        customer_id: str,
        action: str,
        fallback_type: str,
    ) -> None:
        """Log an AI decision opt-out event.

        Parameters
        ----------
        customer_id : str
            Customer identifier.
        action : str
            ``"opt_out"`` or ``"opt_in"``.
        fallback_type : str
            Fallback strategy applied (e.g. ``"rule_based"``, ``"human"``).
        """
        self.log_event("optout", {
            "pk": customer_id,
            "action": action,
            "fallback_type": fallback_type,
        })

    def log_incident(
        self,
        severity: str,
        category: str,
        description: str,
        affected_count: int,
    ) -> None:
        """Log a regulatory violation incident.

        Parameters
        ----------
        severity : str
            ``"critical"`` / ``"high"`` / ``"medium"`` / ``"low"``.
        category : str
            Incident category (e.g. ``"data_breach"``, ``"model_failure"``).
        description : str
            Human-readable description.
        affected_count : int
            Number of affected customers / records.
        """
        incident_id = str(uuid.uuid4())
        self.log_event("incident", {
            "pk": f"{severity}#{category}",
            "incident_id": incident_id,
            "severity": severity,
            "category": category,
            "description": description,
            "affected_count": affected_count,
        })

    def log_distillation(
        self,
        task_name: str,
        teacher_version: str,
        student_version: str,
        metrics: Dict[str, Any],
    ) -> None:
        """Log a model distillation validation record.

        Parameters
        ----------
        task_name : str
            Task being distilled (e.g. ``"ctr"``, ``"cvr"``).
        teacher_version : str
            Teacher model version identifier.
        student_version : str
            Student model version identifier.
        metrics : dict
            Validation metrics (e.g. ``{"auc_gap": 0.02, "ranking_corr": 0.95}``).
        """
        self.log_event("distillation", {
            "pk": task_name,
            "teacher_version": teacher_version,
            "student_version": student_version,
            **metrics,
        })

    def log_embedding(
        self,
        expert_name: str,
        quality_metrics: Dict[str, Any],
    ) -> None:
        """Log an embedding quality audit record.

        Parameters
        ----------
        expert_name : str
            Name of the expert / embedding type.
        quality_metrics : dict
            Quality metrics (e.g. ``{"final_loss": 0.12, "nan_count": 0}``).
        """
        self.log_event("embedding", {
            "pk": expert_name,
            **quality_metrics,
        })


# ---------------------------------------------------------------------------
# In-memory fallback for local testing
# ---------------------------------------------------------------------------

class InMemoryAuditStore(ComplianceAuditStore):
    """In-memory audit store for local development and testing.

    API-compatible with :class:`ComplianceAuditStore` but stores all events
    in a plain Python dict instead of DynamoDB.

    Usage::

        store = InMemoryAuditStore()
        store.log_killswitch("ACTIVATE", "global", "global", "test", "admin")
        events = store.query_events("killswitch", "ACTIVATE#global")
    """

    def __init__(self, **kwargs: Any) -> None:  # noqa: ARG002
        # Intentionally skip parent __init__ (no boto3 needed)
        self._prefix = kwargs.get("table_prefix", "ple-audit")
        self._region = kwargs.get("region", "ap-northeast-2")
        self._dynamo = None
        self._tables: Dict[str, Any] = {}
        self._store: Dict[str, List[Dict[str, Any]]] = {
            suffix: [] for suffix in TABLE_SUFFIXES
        }

    def log_event(self, table_suffix: str, event: Dict[str, Any]) -> None:
        """Append an event to the in-memory store."""
        ts = _now_iso()
        record = {
            "sk": ts,
            "event_id": str(uuid.uuid4()),
            **event,
        }
        record = {k: v for k, v in record.items() if v is not None}
        self._store.setdefault(table_suffix, []).append(record)

    def query_events(
        self,
        table_suffix: str,
        partition_key: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query the in-memory store."""
        events = self._store.get(table_suffix, [])
        results = [e for e in events if e.get("pk") == partition_key]

        if start_time:
            results = [e for e in results if e.get("sk", "") >= start_time]
        if end_time:
            results = [e for e in results if e.get("sk", "") <= end_time]

        # Newest first
        results.sort(key=lambda e: e.get("sk", ""), reverse=True)
        return results[:limit]

    def get_all_events(self, table_suffix: str) -> List[Dict[str, Any]]:
        """Return all events for a given table (test helper)."""
        return list(self._store.get(table_suffix, []))

    def clear(self) -> None:
        """Clear all stored events (test helper)."""
        for suffix in self._store:
            self._store[suffix] = []


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_store: Optional[ComplianceAuditStore] = None


def get_audit_store(
    use_memory: bool = False,
    **kwargs: Any,
) -> ComplianceAuditStore:
    """Return a module-level singleton audit store.

    Parameters
    ----------
    use_memory : bool
        If ``True``, return an :class:`InMemoryAuditStore` instead.
    **kwargs
        Forwarded to the store constructor.
    """
    global _default_store
    if _default_store is None:
        if use_memory:
            _default_store = InMemoryAuditStore(**kwargs)
        else:
            _default_store = ComplianceAuditStore(**kwargs)
    return _default_store
