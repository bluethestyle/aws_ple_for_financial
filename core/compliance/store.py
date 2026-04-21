"""
ComplianceStore - persistence backends for ComplianceRequest / ComplianceEvent.

Three backends:
- InMemoryComplianceStore: unit tests, local dev.
- DynamoDBComplianceStore: production online path (low latency, per-user query).
- S3ParquetComplianceStore: batch / archive (cheap long-term retention).

All are swappable behind the ComplianceStore ABC. The caller chooses via
`pipeline.yaml::compliance.store.backend`.
"""

from __future__ import annotations

import json
import logging
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.compliance.types import (
    ComplianceEvent,
    ComplianceRequest,
    RequestStatus,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ComplianceStore",
    "InMemoryComplianceStore",
    "DynamoDBComplianceStore",
    "S3ParquetComplianceStore",
    "build_compliance_store",
]


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class ComplianceStore(ABC):
    """Persistence abstraction for compliance requests and events."""

    @abstractmethod
    def put_request(self, req: ComplianceRequest) -> None: ...

    @abstractmethod
    def get_request(self, request_id: str) -> Optional[ComplianceRequest]: ...

    @abstractmethod
    def update_request_status(
        self,
        request_id: str,
        status: str,
        processed_at: Optional[datetime] = None,
    ) -> None: ...

    @abstractmethod
    def list_requests(
        self,
        user_id: Optional[str] = None,
        status: Optional[str] = None,
        request_type: Optional[str] = None,
    ) -> List[ComplianceRequest]: ...

    @abstractmethod
    def put_event(self, evt: ComplianceEvent) -> None: ...

    @abstractmethod
    def query_events(
        self,
        user_id: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        event_type: Optional[str] = None,
    ) -> List[ComplianceEvent]: ...

    # Convenience wrapper (pending requests only, optional per-user filter).
    def list_pending(
        self, user_id: Optional[str] = None
    ) -> List[ComplianceRequest]:
        return self.list_requests(user_id=user_id, status=RequestStatus.PENDING)


# ---------------------------------------------------------------------------
# InMemory backend (tests)
# ---------------------------------------------------------------------------

class InMemoryComplianceStore(ComplianceStore):
    """Thread-safe in-memory store for unit tests."""

    def __init__(self) -> None:
        self._requests: Dict[str, ComplianceRequest] = {}
        self._events_by_user: Dict[str, List[ComplianceEvent]] = defaultdict(list)
        self._all_events: List[ComplianceEvent] = []
        self._lock = threading.RLock()

    # -- Requests ---------------------------------------------------------

    def put_request(self, req: ComplianceRequest) -> None:
        with self._lock:
            self._requests[req.request_id] = req

    def get_request(self, request_id: str) -> Optional[ComplianceRequest]:
        with self._lock:
            return self._requests.get(request_id)

    def update_request_status(
        self,
        request_id: str,
        status: str,
        processed_at: Optional[datetime] = None,
    ) -> None:
        if status not in RequestStatus.VALID:
            raise ValueError(f"Invalid status={status!r}")
        with self._lock:
            req = self._requests.get(request_id)
            if req is None:
                raise KeyError(f"Unknown request_id={request_id!r}")
            req.status = status
            if processed_at is not None:
                req.processed_at = processed_at
            elif status == RequestStatus.PROCESSED:
                req.processed_at = datetime.now(timezone.utc)

    def list_requests(
        self,
        user_id: Optional[str] = None,
        status: Optional[str] = None,
        request_type: Optional[str] = None,
    ) -> List[ComplianceRequest]:
        with self._lock:
            result = list(self._requests.values())
        if user_id is not None:
            result = [r for r in result if r.user_id == user_id]
        if status is not None:
            result = [r for r in result if r.status == status]
        if request_type is not None:
            result = [r for r in result if r.request_type == request_type]
        result.sort(key=lambda r: r.submitted_at)
        return result

    # -- Events -----------------------------------------------------------

    def put_event(self, evt: ComplianceEvent) -> None:
        with self._lock:
            self._events_by_user[evt.user_id].append(evt)
            self._all_events.append(evt)

    def query_events(
        self,
        user_id: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        event_type: Optional[str] = None,
    ) -> List[ComplianceEvent]:
        with self._lock:
            if user_id is not None:
                pool = list(self._events_by_user.get(user_id, []))
            else:
                pool = list(self._all_events)
        if since is not None:
            pool = [e for e in pool if e.timestamp >= since]
        if until is not None:
            pool = [e for e in pool if e.timestamp <= until]
        if event_type is not None:
            pool = [e for e in pool if e.event_type == event_type]
        pool.sort(key=lambda e: e.timestamp)
        return pool


# ---------------------------------------------------------------------------
# DynamoDB backend (production)
# ---------------------------------------------------------------------------

class DynamoDBComplianceStore(ComplianceStore):
    """
    DynamoDB-backed store.

    Schema (two tables; both on-demand billing):

    Table: ${requests_table}
      PK: request_id (S)
      Attributes: user_id (S), request_type (S), status (S), submitted_at (S),
                  sla_deadline (S), processed_at (S?), metadata (M)
      GSI: user_id-submitted_at-index  (per-user history)
      GSI: status-sla_deadline-index   (pending / SLA sweeps)

    Table: ${events_table}
      PK: user_id (S)     SK: timestamp#event_id (S)
      Attributes: event_type (S), request_id (S?), payload (M)
    """

    def __init__(
        self,
        requests_table: str,
        events_table: str,
        dynamodb_resource: Any = None,
        region: Optional[str] = None,
    ) -> None:
        self.requests_table_name = requests_table
        self.events_table_name = events_table
        self._region = region

        if dynamodb_resource is None:
            try:
                import boto3  # type: ignore
                dynamodb_resource = boto3.resource(
                    "dynamodb", region_name=region
                )
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "boto3 not installed; install or inject a resource"
                ) from exc

        self._requests = dynamodb_resource.Table(requests_table)
        self._events = dynamodb_resource.Table(events_table)

    # -- Requests ---------------------------------------------------------

    def put_request(self, req: ComplianceRequest) -> None:
        self._requests.put_item(Item=_serialize_for_ddb(req.to_dict()))

    def get_request(self, request_id: str) -> Optional[ComplianceRequest]:
        resp = self._requests.get_item(Key={"request_id": request_id})
        item = resp.get("Item")
        if item is None:
            return None
        return ComplianceRequest.from_dict(_deserialize_from_ddb(item))

    def update_request_status(
        self,
        request_id: str,
        status: str,
        processed_at: Optional[datetime] = None,
    ) -> None:
        if status not in RequestStatus.VALID:
            raise ValueError(f"Invalid status={status!r}")
        update = "SET #s = :s"
        values: Dict[str, Any] = {":s": status}
        names = {"#s": "status"}
        if processed_at is not None:
            update += ", processed_at = :p"
            values[":p"] = processed_at.astimezone(timezone.utc).isoformat()
        elif status == RequestStatus.PROCESSED:
            update += ", processed_at = :p"
            values[":p"] = datetime.now(timezone.utc).isoformat()
        self._requests.update_item(
            Key={"request_id": request_id},
            UpdateExpression=update,
            ExpressionAttributeValues=values,
            ExpressionAttributeNames=names,
        )

    def list_requests(
        self,
        user_id: Optional[str] = None,
        status: Optional[str] = None,
        request_type: Optional[str] = None,
    ) -> List[ComplianceRequest]:
        # Minimal implementation - production path should use GSIs.
        # We prefer a Scan with FilterExpression since the requests table is
        # expected to be small (audit + per-user history).
        from boto3.dynamodb.conditions import Attr  # type: ignore

        filters = None
        if user_id is not None:
            filters = Attr("user_id").eq(user_id)
        if status is not None:
            f = Attr("status").eq(status)
            filters = f if filters is None else filters & f
        if request_type is not None:
            f = Attr("request_type").eq(request_type)
            filters = f if filters is None else filters & f

        kwargs: Dict[str, Any] = {}
        if filters is not None:
            kwargs["FilterExpression"] = filters

        items: List[Dict[str, Any]] = []
        resp = self._requests.scan(**kwargs)
        items.extend(resp.get("Items", []))
        while "LastEvaluatedKey" in resp:
            kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
            resp = self._requests.scan(**kwargs)
            items.extend(resp.get("Items", []))

        reqs = [
            ComplianceRequest.from_dict(_deserialize_from_ddb(it))
            for it in items
        ]
        reqs.sort(key=lambda r: r.submitted_at)
        return reqs

    # -- Events -----------------------------------------------------------

    def put_event(self, evt: ComplianceEvent) -> None:
        ts_iso = evt.timestamp.astimezone(timezone.utc).isoformat()
        item = {
            "user_id": evt.user_id,
            "sk": f"{ts_iso}#{evt.event_id}",
            "event_id": evt.event_id,
            "event_type": evt.event_type,
            "timestamp": ts_iso,
            "payload": _serialize_for_ddb(dict(evt.payload)),
        }
        if evt.request_id is not None:
            item["request_id"] = evt.request_id
        self._events.put_item(Item=item)

    def query_events(
        self,
        user_id: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        event_type: Optional[str] = None,
    ) -> List[ComplianceEvent]:
        from boto3.dynamodb.conditions import Attr, Key  # type: ignore

        items: List[Dict[str, Any]] = []

        if user_id is not None:
            key = Key("user_id").eq(user_id)
            if since is not None:
                key = key & Key("sk").gte(
                    since.astimezone(timezone.utc).isoformat()
                )
            kwargs: Dict[str, Any] = {"KeyConditionExpression": key}
            if event_type is not None:
                kwargs["FilterExpression"] = Attr("event_type").eq(event_type)
            resp = self._events.query(**kwargs)
            items.extend(resp.get("Items", []))
            while "LastEvaluatedKey" in resp:
                kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
                resp = self._events.query(**kwargs)
                items.extend(resp.get("Items", []))
        else:
            kwargs = {}
            if event_type is not None:
                kwargs["FilterExpression"] = Attr("event_type").eq(event_type)
            resp = self._events.scan(**kwargs)
            items.extend(resp.get("Items", []))
            while "LastEvaluatedKey" in resp:
                kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
                resp = self._events.scan(**kwargs)
                items.extend(resp.get("Items", []))

        events = [
            ComplianceEvent.from_dict(_deserialize_from_ddb(it))
            for it in items
        ]
        if until is not None:
            events = [e for e in events if e.timestamp <= until]
        if since is not None:
            events = [e for e in events if e.timestamp >= since]
        events.sort(key=lambda e: e.timestamp)
        return events


# ---------------------------------------------------------------------------
# S3 Parquet backend (archive / batch)
# ---------------------------------------------------------------------------

class S3ParquetComplianceStore(ComplianceStore):
    """
    Archive-oriented backend. Writes are batched into daily Parquet partitions.

    Intentionally minimal: reads require pyarrow; writes require pyarrow or
    pandas. Used for long-term retention (5-year / 7-year) and for offline
    reconstruction / reporting, not for online request lookup.
    """

    REQUESTS_PREFIX = "requests"
    EVENTS_PREFIX = "events"

    def __init__(
        self,
        bucket: str,
        key_prefix: str = "compliance",
        s3_client: Any = None,
    ) -> None:
        self.bucket = bucket
        self.key_prefix = key_prefix.rstrip("/")
        self._buffer_requests: List[Dict[str, Any]] = []
        self._buffer_events: List[Dict[str, Any]] = []

        if s3_client is None:
            try:
                import boto3  # type: ignore
                s3_client = boto3.client("s3")
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "boto3 not installed; install or inject an s3_client"
                ) from exc
        self._s3 = s3_client

    def put_request(self, req: ComplianceRequest) -> None:
        self._buffer_requests.append(req.to_dict())

    def get_request(self, request_id: str) -> Optional[ComplianceRequest]:
        raise NotImplementedError(
            "S3ParquetComplianceStore is archive-only; "
            "use DynamoDBComplianceStore for online lookup"
        )

    def update_request_status(
        self, request_id: str, status: str,
        processed_at: Optional[datetime] = None,
    ) -> None:
        raise NotImplementedError(
            "S3ParquetComplianceStore is append-only; "
            "cannot mutate archived requests"
        )

    def list_requests(
        self,
        user_id: Optional[str] = None,
        status: Optional[str] = None,
        request_type: Optional[str] = None,
    ) -> List[ComplianceRequest]:
        raise NotImplementedError(
            "S3ParquetComplianceStore requires batch scan; "
            "use a Glue/Athena query instead"
        )

    def put_event(self, evt: ComplianceEvent) -> None:
        self._buffer_events.append(evt.to_dict())

    def query_events(
        self,
        user_id: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        event_type: Optional[str] = None,
    ) -> List[ComplianceEvent]:
        raise NotImplementedError(
            "S3ParquetComplianceStore requires batch scan; "
            "use a Glue/Athena query instead"
        )

    def flush(self, partition_date: Optional[str] = None) -> None:
        """Write buffered requests + events as Parquet to S3."""
        try:
            import pyarrow as pa  # type: ignore
            import pyarrow.parquet as pq  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "pyarrow required for S3ParquetComplianceStore.flush()"
            ) from exc

        if partition_date is None:
            partition_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        import io

        if self._buffer_requests:
            table = pa.Table.from_pylist(self._buffer_requests)
            buf = io.BytesIO()
            pq.write_table(table, buf)
            key = (
                f"{self.key_prefix}/{self.REQUESTS_PREFIX}/"
                f"dt={partition_date}/requests_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.parquet"
            )
            self._s3.put_object(
                Bucket=self.bucket, Key=key, Body=buf.getvalue()
            )
            self._buffer_requests.clear()
            logger.info(
                "Flushed %d requests to s3://%s/%s",
                len(self._buffer_requests),
                self.bucket,
                key,
            )

        if self._buffer_events:
            # pyarrow needs a schema hint when `payload` is a mixed dict
            payloads_as_json = [
                {**e, "payload": json.dumps(e.get("payload", {}),
                                            ensure_ascii=False)}
                for e in self._buffer_events
            ]
            table = pa.Table.from_pylist(payloads_as_json)
            buf = io.BytesIO()
            pq.write_table(table, buf)
            key = (
                f"{self.key_prefix}/{self.EVENTS_PREFIX}/"
                f"dt={partition_date}/events_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.parquet"
            )
            self._s3.put_object(
                Bucket=self.bucket, Key=key, Body=buf.getvalue()
            )
            self._buffer_events.clear()
            logger.info(
                "Flushed %d events to s3://%s/%s",
                len(self._buffer_events),
                self.bucket,
                key,
            )


# ---------------------------------------------------------------------------
# DynamoDB (de)serialization helpers
# ---------------------------------------------------------------------------

def _serialize_for_ddb(value: Any) -> Any:
    """Convert floats → Decimal so boto3 can store the value."""
    from decimal import Decimal

    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, dict):
        return {k: _serialize_for_ddb(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize_for_ddb(v) for v in value]
    return value


def _deserialize_from_ddb(item: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Decimal → float for JSON-friendly reconstruction."""
    from decimal import Decimal

    def _walk(v: Any) -> Any:
        if isinstance(v, Decimal):
            return float(v)
        if isinstance(v, dict):
            return {k: _walk(x) for k, x in v.items()}
        if isinstance(v, list):
            return [_walk(x) for x in v]
        return v

    return _walk(item)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_compliance_store(config: Dict[str, Any]) -> ComplianceStore:
    """
    Instantiate a ComplianceStore from the `compliance.store` config block.

    Expected config shape (pipeline.yaml):

        compliance:
          store:
            backend: "dynamodb" | "s3_parquet" | "in_memory"
            requests_table: "ple-compliance-requests"
            events_table: "ple-compliance-events"
            s3_bucket: "aiops-ple-financial"
            s3_prefix: "compliance"
            region: "ap-northeast-2"
    """
    store_cfg = config.get("store", config)  # accept nested or flat
    backend = store_cfg.get("backend", "in_memory")

    if backend == "in_memory":
        return InMemoryComplianceStore()

    if backend == "dynamodb":
        requests_table = store_cfg.get("requests_table")
        events_table = store_cfg.get("events_table")
        if not requests_table or not events_table:
            raise ValueError(
                "dynamodb backend requires requests_table and events_table"
            )
        return DynamoDBComplianceStore(
            requests_table=requests_table,
            events_table=events_table,
            region=store_cfg.get("region"),
        )

    if backend == "s3_parquet":
        bucket = store_cfg.get("s3_bucket")
        if not bucket:
            raise ValueError("s3_parquet backend requires s3_bucket")
        return S3ParquetComplianceStore(
            bucket=bucket,
            key_prefix=store_cfg.get("s3_prefix", "compliance"),
        )

    raise ValueError(f"Unknown compliance store backend: {backend!r}")
