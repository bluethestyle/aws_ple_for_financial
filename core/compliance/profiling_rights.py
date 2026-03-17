"""
Profiling Rights Manager
========================

Data subject profiling rights management in compliance with 개보법
(Personal Information Protection Act) and GDPR.

Supported rights requests:

* **access** -- view what data/profiling is held.
* **rectify** -- correct inaccurate profiling data.
* **delete** -- erase profiling data (right to be forgotten).
* **restrict** -- limit processing of profiling data.
* **port** -- export profiling data in machine-readable format.

Storage: DynamoDB (production) or in-memory dict (testing/local).

DynamoDB Table Schema
---------------------

Partition key: ``customer_id`` (String)
Sort key:      ``request_id`` (String)

Example items::

    {"customer_id": "C001", "request_id": "REQ-abc123",
     "right_type": "access", "status": "pending", ...}
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["RightsRequest", "ProfilingRightsManager"]

_VALID_RIGHT_TYPES = {"access", "rectify", "delete", "restrict", "port"}
_VALID_STATUSES = {"pending", "processing", "completed", "denied"}


@dataclass
class RightsRequest:
    """A data subject rights request."""

    request_id: str
    customer_id: str
    right_type: str      # "access" | "rectify" | "delete" | "restrict" | "port"
    status: str          # "pending" | "processing" | "completed" | "denied"
    requested_at: str    # ISO 8601
    completed_at: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class ProfilingRightsManager:
    """Data subject profiling rights management (개보법 + GDPR).

    Manages the lifecycle of customer rights requests from submission
    through processing to completion or denial.

    Args:
        table_name: DynamoDB table name.
        region: AWS region.
        audit_store: Optional callable ``(event_dict) -> None`` for audit
            logging.
        use_dynamo: If ``False``, use in-memory dict.  Defaults to ``True``;
            falls back to in-memory if boto3 is unavailable.
    """

    def __init__(
        self,
        table_name: str = "ple-profiling-rights",
        region: str = "ap-northeast-2",
        audit_store: Any = None,
        use_dynamo: bool = True,
    ) -> None:
        self._table_name = table_name
        self._region = region
        self._audit_store = audit_store
        self._table = None
        # In-memory: keyed by request_id
        self._memory: Dict[str, Dict[str, Any]] = {}

        if use_dynamo:
            try:
                import boto3

                dynamodb = boto3.resource("dynamodb", region_name=region)
                self._table = dynamodb.Table(table_name)
                logger.info(
                    "ProfilingRightsManager: DynamoDB table=%s, region=%s",
                    table_name, region,
                )
            except Exception:
                logger.warning(
                    "ProfilingRightsManager: boto3 unavailable, "
                    "using in-memory store",
                )
        else:
            logger.info("ProfilingRightsManager: using in-memory store")

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------

    def submit_request(
        self,
        customer_id: str,
        right_type: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Submit a rights request.

        Args:
            customer_id: Customer identifier.
            right_type: One of ``access``, ``rectify``, ``delete``,
                ``restrict``, ``port``.
            details: Optional dict with request-specific details
                (e.g. fields to rectify, export format preference).

        Returns:
            The generated ``request_id``.
        """
        if right_type not in _VALID_RIGHT_TYPES:
            raise ValueError(
                f"Invalid right_type '{right_type}'. "
                f"Must be one of {_VALID_RIGHT_TYPES}"
            )

        request_id = f"REQ-{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()

        request = RightsRequest(
            request_id=request_id,
            customer_id=customer_id,
            right_type=right_type,
            status="pending",
            requested_at=now,
            details=details or {},
        )

        self._put_request(request)
        self._audit("RIGHTS_REQUEST_SUBMITTED", customer_id,
                     request_id=request_id, right_type=right_type)
        logger.info(
            "Rights request submitted: request_id=%s, customer=%s, type=%s",
            request_id, customer_id, right_type,
        )
        return request_id

    # ------------------------------------------------------------------
    # Process
    # ------------------------------------------------------------------

    def process_request(self, request_id: str) -> RightsRequest:
        """Process a pending rights request.

        Transitions the request from ``pending`` to ``processing``, then
        to ``completed``.  In a real system, each right type would trigger
        different downstream actions (data export, deletion, etc.).

        Args:
            request_id: The request identifier.

        Returns:
            The updated :class:`RightsRequest`.

        Raises:
            KeyError: If the request is not found.
            ValueError: If the request is not in a processable state.
        """
        request = self._get_request(request_id)
        if request is None:
            raise KeyError(f"Rights request not found: {request_id}")

        if request.status not in ("pending", "processing"):
            raise ValueError(
                f"Request {request_id} is in status '{request.status}' "
                f"and cannot be processed"
            )

        now = datetime.now(timezone.utc).isoformat()

        # Transition to completed
        request.status = "completed"
        request.completed_at = now

        self._put_request(request)
        self._audit(
            "RIGHTS_REQUEST_COMPLETED",
            request.customer_id,
            request_id=request_id,
            right_type=request.right_type,
        )
        logger.info(
            "Rights request completed: request_id=%s, type=%s",
            request_id, request.right_type,
        )
        return request

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_customer_requests(
        self, customer_id: str,
    ) -> List[RightsRequest]:
        """Get all rights requests for a customer."""
        if self._table is not None:
            return self._dynamo_query_customer(customer_id)

        results = []
        for item in self._memory.values():
            if item.get("customer_id") == customer_id:
                results.append(self._item_to_request(item))
        return sorted(results, key=lambda r: r.requested_at, reverse=True)

    def get_pending_requests(self) -> List[RightsRequest]:
        """Get all pending requests (for admin dashboard).

        Note: in production with DynamoDB, this uses a scan with filter.
        For high-volume systems, consider a GSI on ``status``.
        """
        if self._table is not None:
            return self._dynamo_scan_pending()

        results = []
        for item in self._memory.values():
            if item.get("status") == "pending":
                results.append(self._item_to_request(item))
        return sorted(results, key=lambda r: r.requested_at)

    # ------------------------------------------------------------------
    # Storage helpers
    # ------------------------------------------------------------------

    def _get_request(self, request_id: str) -> Optional[RightsRequest]:
        if self._table is not None:
            return self._dynamo_get(request_id)

        item = self._memory.get(request_id)
        if item is None:
            return None
        return self._item_to_request(item)

    def _put_request(self, request: RightsRequest) -> None:
        if self._table is not None:
            self._dynamo_put(request)
            return

        self._memory[request.request_id] = asdict(request)

    def _dynamo_get(self, request_id: str) -> Optional[RightsRequest]:
        """Get a request by scanning for request_id.

        Since the table uses customer_id as PK and request_id as SK,
        we need the customer_id for a direct get.  For lookup by
        request_id alone, we scan (or use a GSI in production).
        For simplicity, we scan the in-memory index first.
        """
        # Optimisation: if we cached customer_id->request_id mapping
        # we could do a direct get.  For now, scan.
        try:
            response = self._table.scan(
                FilterExpression="request_id = :rid",
                ExpressionAttributeValues={":rid": request_id},
                Limit=1,
            )
            items = response.get("Items", [])
            if not items:
                return None
            return self._item_to_request(items[0])
        except Exception:
            logger.exception(
                "ProfilingRightsManager: DynamoDB get failed request_id=%s",
                request_id,
            )
            return None

    def _dynamo_put(self, request: RightsRequest) -> None:
        item: Dict[str, Any] = {
            "customer_id": request.customer_id,
            "request_id": request.request_id,
            "right_type": request.right_type,
            "status": request.status,
            "requested_at": request.requested_at,
        }
        if request.completed_at:
            item["completed_at"] = request.completed_at
        if request.details:
            item["details"] = request.details

        try:
            self._table.put_item(Item=item)
        except Exception:
            logger.exception("ProfilingRightsManager: DynamoDB put failed")
            raise

    def _dynamo_query_customer(
        self, customer_id: str,
    ) -> List[RightsRequest]:
        results: List[RightsRequest] = []
        try:
            response = self._table.query(
                KeyConditionExpression="customer_id = :cid",
                ExpressionAttributeValues={":cid": customer_id},
                ScanIndexForward=False,
            )
            for item in response.get("Items", []):
                results.append(self._item_to_request(item))
        except Exception:
            logger.exception(
                "ProfilingRightsManager: DynamoDB query failed customer=%s",
                customer_id,
            )
        return results

    def _dynamo_scan_pending(self) -> List[RightsRequest]:
        results: List[RightsRequest] = []
        try:
            response = self._table.scan(
                FilterExpression="#s = :pending",
                ExpressionAttributeNames={"#s": "status"},
                ExpressionAttributeValues={":pending": "pending"},
            )
            for item in response.get("Items", []):
                results.append(self._item_to_request(item))

            while "LastEvaluatedKey" in response:
                response = self._table.scan(
                    FilterExpression="#s = :pending",
                    ExpressionAttributeNames={"#s": "status"},
                    ExpressionAttributeValues={":pending": "pending"},
                    ExclusiveStartKey=response["LastEvaluatedKey"],
                )
                for item in response.get("Items", []):
                    results.append(self._item_to_request(item))

        except Exception:
            logger.exception(
                "ProfilingRightsManager: DynamoDB scan pending failed",
            )
        return sorted(results, key=lambda r: r.requested_at)

    @staticmethod
    def _item_to_request(item: Dict[str, Any]) -> RightsRequest:
        return RightsRequest(
            request_id=str(item["request_id"]),
            customer_id=str(item["customer_id"]),
            right_type=str(item.get("right_type", "")),
            status=str(item.get("status", "pending")),
            requested_at=str(item.get("requested_at", "")),
            completed_at=item.get("completed_at"),
            details=dict(item.get("details", {})),
        )

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    def _audit(self, action: str, customer_id: str, **kwargs: Any) -> None:
        event = {
            "action": action,
            "customer_id": customer_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **kwargs,
        }
        logger.info("PROFILING_RIGHTS_AUDIT | %s", event)
        if self._audit_store is not None:
            try:
                self._audit_store(event)
            except Exception:
                logger.exception(
                    "ProfilingRightsManager: audit_store callback failed",
                )
