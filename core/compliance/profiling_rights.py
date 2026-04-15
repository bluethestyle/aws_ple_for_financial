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

__all__ = [
    "RightsRequest",
    "ProfilingRightsManager",
    "RecommendationReviewHandler",
]

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


# ======================================================================
# Recommendation Review Handler -- 이의제기 → 재심사
# ======================================================================
# AI기본법 '이의제기 절차 + 설명요구권' 충족을 위한 구현.
# 고객이 AI 추천 결과에 이의를 제기하면, 현재 feature 기반으로
# 재추천을 수행하고 원본 대비 변경사항을 투명하게 제공한다.
# ======================================================================


class RecommendationReviewHandler:
    """AI 추천 결과 이의제기 및 재심사 처리기.

    AI기본법이 요구하는 '이의제기 절차'와 '설명요구권'을 충족하기 위해,
    고객이 추천 결과에 대해 이의를 제기하면 현재 데이터를 기반으로
    재추천을 수행하고 원본과 비교 결과를 투명하게 반환한다.

    Args:
        pipeline: 추천 파이프라인 (``recommend(customer_id, ...)`` 메서드 필요).
        feature_store: 고객 feature 조회용 스토어
            (``get_features(customer_id)`` 메서드 필요).
        cold_start_handler: 콜드스타트 핸들러 (feature 부재 시 대체 로직).
        audit_store: 감사 로그 저장 콜러블 ``(event_dict) -> None``.
    """

    def __init__(
        self,
        pipeline: Any,
        feature_store: Any,
        cold_start_handler: Any = None,
        audit_store: Any = None,
    ) -> None:
        self._pipeline = pipeline
        self._feature_store = feature_store
        self._cold_start_handler = cold_start_handler
        self._audit_store = audit_store
        # ProfilingRightsManager for rights request lifecycle
        self._rights_manager = ProfilingRightsManager(
            audit_store=audit_store,
            use_dynamo=False,
        )
        # In-memory review registry: review_id -> review record
        self._reviews: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------

    def submit_review(
        self,
        customer_id: str,
        original_recommendation_id: str,
        reason: str,
    ) -> str:
        """이의제기 접수 — 고객의 추천 결과 재심사 요청을 등록한다.

        내부적으로 :class:`ProfilingRightsManager` 의 ``submit_request``
        를 활용하여 "access" 타입 권리 요청을 생성한다.

        Args:
            customer_id: 고객 식별자.
            original_recommendation_id: 이의 대상인 원본 추천 ID.
            reason: 이의제기 사유.

        Returns:
            생성된 ``review_id``.
        """
        # 1) ProfilingRightsManager 를 통해 "access" 권리 요청 등록
        rights_request_id = self._rights_manager.submit_request(
            customer_id=customer_id,
            right_type="access",
            details={
                "purpose": "recommendation_review",
                "original_recommendation_id": original_recommendation_id,
                "reason": reason,
            },
        )

        # 2) review 레코드 생성
        review_id = f"RVW-{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()

        review_record: Dict[str, Any] = {
            "review_id": review_id,
            "customer_id": customer_id,
            "original_recommendation_id": original_recommendation_id,
            "reason": reason,
            "rights_request_id": rights_request_id,
            "status": "pending",
            "submitted_at": now,
            "completed_at": None,
            "result": None,
        }
        self._reviews[review_id] = review_record

        # 3) 감사 로그
        self._audit(
            "REVIEW_SUBMITTED",
            customer_id,
            review_id=review_id,
            original_recommendation_id=original_recommendation_id,
            reason=reason,
        )
        logger.info(
            "Review submitted: review_id=%s, customer=%s, "
            "original_rec=%s",
            review_id, customer_id, original_recommendation_id,
        )
        return review_id

    # ------------------------------------------------------------------
    # Process
    # ------------------------------------------------------------------

    def process_review(self, review_id: str) -> Dict[str, Any]:
        """재심사 수행 — 현재 feature 기반으로 재추천하고 원본과 비교한다.

        1. ``feature_store`` 에서 고객의 현재 features 조회.
        2. ``pipeline.recommend()`` 로 재추천 생성.
        3. 원본 추천과 재추천 비교 (score 차이, 순위 변경).
        4. 결과를 ``audit_store`` 에 기록.

        Args:
            review_id: :meth:`submit_review` 가 반환한 리뷰 ID.

        Returns:
            ``{"review_id", "status", "original", "revised", "changes"}``
            형태의 딕셔너리.

        Raises:
            KeyError: 리뷰를 찾을 수 없을 때.
            ValueError: 이미 처리된 리뷰일 때.
        """
        record = self._reviews.get(review_id)
        if record is None:
            raise KeyError(f"Review not found: {review_id}")

        if record["status"] == "completed":
            raise ValueError(
                f"Review {review_id} is already completed"
            )

        customer_id: str = record["customer_id"]
        now = datetime.now(timezone.utc).isoformat()

        # Mark as processing
        record["status"] = "processing"

        # 1) 현재 features 조회
        features = self._get_customer_features(customer_id)

        # 2) 재추천 생성
        revised_recommendation = self._pipeline.recommend(
            customer_id=customer_id,
            features=features,
        )

        # 3) 원본 추천 정보 수집
        original_recommendation = self._get_original_recommendation(
            record["original_recommendation_id"],
        )

        # 4) 비교
        changes = self._compare_recommendations(
            original_recommendation,
            revised_recommendation,
        )

        # 5) 결과 조립
        result: Dict[str, Any] = {
            "review_id": review_id,
            "status": "completed",
            "original": original_recommendation,
            "revised": revised_recommendation,
            "changes": changes,
        }

        # 6) 레코드 업데이트
        record["status"] = "completed"
        record["completed_at"] = now
        record["result"] = result

        # 7) 연결된 rights request 도 완료 처리
        try:
            self._rights_manager.process_request(
                record["rights_request_id"],
            )
        except Exception:
            logger.warning(
                "Could not complete rights request %s for review %s",
                record["rights_request_id"], review_id,
            )

        # 8) 감사 로그
        self._audit(
            "REVIEW_COMPLETED",
            customer_id,
            review_id=review_id,
            changes_summary={
                "score_delta": changes.get("score_delta"),
                "rank_changes_count": len(changes.get("rank_changes", [])),
            },
        )
        logger.info(
            "Review completed: review_id=%s, customer=%s, "
            "score_delta=%.4f",
            review_id, customer_id,
            changes.get("score_delta", 0.0),
        )
        return result

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_review_status(self, review_id: str) -> Dict[str, Any]:
        """리뷰 상태 조회.

        Args:
            review_id: 조회할 리뷰 ID.

        Returns:
            리뷰 레코드 딕셔너리 (결과 포함).

        Raises:
            KeyError: 리뷰를 찾을 수 없을 때.
        """
        record = self._reviews.get(review_id)
        if record is None:
            raise KeyError(f"Review not found: {review_id}")
        return dict(record)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_customer_features(
        self, customer_id: str,
    ) -> Dict[str, Any]:
        """feature_store 에서 고객 features 조회, 실패 시 cold_start."""
        try:
            features = self._feature_store.get_features(customer_id)
            if features:
                return features
        except Exception:
            logger.warning(
                "Feature lookup failed for customer=%s, "
                "falling back to cold_start",
                customer_id,
            )

        # cold start fallback
        if self._cold_start_handler is not None:
            try:
                return self._cold_start_handler.get_default_features(
                    customer_id,
                )
            except Exception:
                logger.warning(
                    "Cold start handler failed for customer=%s",
                    customer_id,
                )
        return {}

    @staticmethod
    def _get_original_recommendation(
        recommendation_id: str,
    ) -> Dict[str, Any]:
        """원본 추천 정보를 조회한다.

        실제 운영 환경에서는 추천 결과 저장소(DynamoDB/S3)에서
        조회한다.  현재는 ID 참조를 포함한 placeholder 를 반환.

        TODO: Implement full retrieval from DynamoDB/S3.
              Expected table: ple-recommendation-log (same table used by
              predict.py _log_prediction).  Query by recommendation_id
              (partition key) and return items + scores from the stored
              prediction payload.  Until this is implemented, reconsideration
              comparisons will show empty original items/scores.
        """
        logger.warning(
            "_get_original_recommendation called for id=%s but DynamoDB/S3 "
            "retrieval is not yet implemented — returning empty placeholder. "
            "Reconsideration comparison will be incomplete.",
            recommendation_id,
        )
        return {
            "recommendation_id": recommendation_id,
            "items": [],
            "scores": [],
        }

    @staticmethod
    def _compare_recommendations(
        original: Dict[str, Any],
        revised: Dict[str, Any],
    ) -> Dict[str, Any]:
        """원본과 재추천 결과를 비교하여 변경사항을 산출한다.

        Returns:
            ``{"score_delta", "rank_changes", "added_items",
            "removed_items"}`` 딕셔너리.
        """
        orig_scores = original.get("scores", [])
        rev_scores = revised.get("scores", [])

        # 평균 score 차이
        orig_mean = (
            sum(orig_scores) / len(orig_scores) if orig_scores else 0.0
        )
        rev_mean = (
            sum(rev_scores) / len(rev_scores) if rev_scores else 0.0
        )
        score_delta = round(rev_mean - orig_mean, 6)

        # 순위 변경 추적
        orig_items = original.get("items", [])
        rev_items = revised.get("items", [])

        orig_rank = {item: idx for idx, item in enumerate(orig_items)}
        rev_rank = {item: idx for idx, item in enumerate(rev_items)}

        rank_changes: List[Dict[str, Any]] = []
        all_items = set(orig_items) | set(rev_items)
        for item in all_items:
            o_rank = orig_rank.get(item)
            r_rank = rev_rank.get(item)
            if o_rank != r_rank:
                rank_changes.append({
                    "item": item,
                    "original_rank": o_rank,
                    "revised_rank": r_rank,
                })

        added_items = [i for i in rev_items if i not in orig_rank]
        removed_items = [i for i in orig_items if i not in rev_rank]

        return {
            "score_delta": score_delta,
            "rank_changes": rank_changes,
            "added_items": added_items,
            "removed_items": removed_items,
        }

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
        logger.info("RECOMMENDATION_REVIEW_AUDIT | %s", event)
        if self._audit_store is not None:
            try:
                self._audit_store(event)
            except Exception:
                logger.exception(
                    "RecommendationReviewHandler: "
                    "audit_store callback failed",
                )
