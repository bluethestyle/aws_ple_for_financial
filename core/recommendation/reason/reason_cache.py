"""
Reason Cache -- DynamoDB-backed cache for generated recommendation reasons.

Stores generated reasons keyed by (customer_id, product_id, task_name).
Supports both in-memory (testing) and DynamoDB (production) backends.

When a reason is generated (L1/L2a/L2b), it's cached here.
Next request for the same customer+product+task returns the cached reason
instead of regenerating.

Cache entries have TTL (default 24h) for freshness.

Usage::

    cache = ReasonCache(backend="dynamodb", table_name="ple-reason-cache")

    # Store
    cache.put(customer_id="C001", product_id="P100", task_name="churn",
              reason_text="...", layer="L2a", metadata={...})

    # Retrieve
    entry = cache.get(customer_id="C001", product_id="P100", task_name="churn")
    if entry:
        print(entry.reason_text, entry.layer)

    # Batch retrieve (for recommendation list)
    entries = cache.get_batch(customer_id="C001", product_ids=["P100", "P200", "P300"])
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["ReasonCache", "CacheEntry"]


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    """A single cached reason entry.

    Attributes:
        customer_id: Customer identifier.
        product_id: Product identifier.
        task_name: Task / model name (e.g. ``"churn"``, ``"cross_sell"``).
        reason_text: The generated reason text.
        layer: Which layer produced this (``"L1"`` / ``"L2a"`` / ``"L2b"``).
        confidence: Confidence score (1.0 for L1, LLM-derived for L2).
        quality_passed: Whether L2b quality validation passed.
        created_at: ISO-8601 creation timestamp.
        ttl: Unix timestamp for DynamoDB TTL expiry.
        metadata: Arbitrary key-value metadata (template_id, job_id, etc.).
    """

    customer_id: str
    product_id: str
    task_name: str
    reason_text: str
    layer: str              # "L1" | "L2a" | "L2b"
    confidence: float = 1.0
    quality_passed: bool = True
    created_at: str = ""    # ISO 8601
    ttl: int = 0            # Unix timestamp for DynamoDB TTL
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Layer ranking for get_best()
# ---------------------------------------------------------------------------

_LAYER_RANK = {"L2b": 3, "L2a": 2, "L1": 1}


# ---------------------------------------------------------------------------
# ReasonCache
# ---------------------------------------------------------------------------

class ReasonCache:
    """Dual-backend reason cache (DynamoDB + in-memory fallback).

    Follows the same Memory + DynamoDB pattern used by
    :class:`~core.serving.feature_store.DynamoDBFeatureStore`.

    DynamoDB table schema:
        PK: ``customer_id`` (String)
        SK: ``product_id#task_name`` (String)
        reason_text: str
        layer: str
        confidence: Decimal
        quality_passed: bool
        created_at: str
        ttl: int  (DynamoDB TTL attribute -- auto-deletes expired items)
        metadata: str  (JSON-encoded)

    Args:
        backend: ``"dynamodb"`` or ``"memory"``.
        table_name: DynamoDB table name.
        region: AWS region.
        ttl_hours: Cache entry TTL in hours (default 24).
    """

    # DynamoDB BatchGetItem limit
    _BATCH_SIZE = 100

    def __init__(
        self,
        backend: str = "memory",
        table_name: str = "ple-reason-cache",
        region: Optional[str] = None,
        ttl_hours: int = 24,
    ) -> None:
        self._backend = backend
        self._table_name = table_name
        self._region = region
        self._ttl_hours = ttl_hours

        # In-memory store: {composite_key: CacheEntry}
        self._memory_store: Dict[str, CacheEntry] = {}

        # DynamoDB table resource (lazy for memory backend)
        self._table = None

        if backend == "dynamodb":
            try:
                import boto3
                dynamo = boto3.resource("dynamodb", region_name=region)
                self._table = dynamo.Table(table_name)
                logger.info(
                    "ReasonCache: DynamoDB backend, table=%s, region=%s, ttl=%dh",
                    table_name, region, ttl_hours,
                )
            except Exception as e:
                logger.warning(
                    "ReasonCache: DynamoDB unavailable (%s), falling back to memory", e,
                )
                self._backend = "memory"

        if self._backend == "memory":
            logger.info("ReasonCache: in-memory backend, ttl=%dh", ttl_hours)

    # ------------------------------------------------------------------
    # Public API: put
    # ------------------------------------------------------------------

    def put(
        self,
        customer_id: str,
        product_id: str,
        task_name: str,
        reason_text: str,
        layer: str = "L1",
        confidence: float = 1.0,
        quality_passed: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store a generated reason.

        Args:
            customer_id: Customer identifier.
            product_id: Product identifier.
            task_name: Task / model name.
            reason_text: Generated reason text.
            layer: Generation layer (``"L1"`` / ``"L2a"`` / ``"L2b"``).
            confidence: Confidence score.
            quality_passed: Whether quality validation passed.
            metadata: Optional extra metadata dict.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        ttl_ts = self._compute_ttl()
        meta = metadata or {}

        entry = CacheEntry(
            customer_id=customer_id,
            product_id=product_id,
            task_name=task_name,
            reason_text=reason_text,
            layer=layer,
            confidence=confidence,
            quality_passed=quality_passed,
            created_at=now_iso,
            ttl=ttl_ts,
            metadata=meta,
        )

        # Always write to in-memory store (serves as L1 cache)
        key = self._make_key(customer_id, product_id, task_name)
        self._memory_store[key] = entry

        # Persist to DynamoDB if available
        if self._backend == "dynamodb" and self._table is not None:
            self._dynamo_put(entry)

    # ------------------------------------------------------------------
    # Public API: get
    # ------------------------------------------------------------------

    def get(
        self,
        customer_id: str,
        product_id: str,
        task_name: str,
    ) -> Optional[CacheEntry]:
        """Retrieve a cached reason.  Returns ``None`` if not found or expired.

        Checks in-memory store first, then DynamoDB.

        Args:
            customer_id: Customer identifier.
            product_id: Product identifier.
            task_name: Task / model name.

        Returns:
            :class:`CacheEntry` or ``None``.
        """
        key = self._make_key(customer_id, product_id, task_name)

        # In-memory lookup
        entry = self._memory_store.get(key)
        if entry is not None:
            if self._is_expired(entry):
                del self._memory_store[key]
            else:
                return entry

        # DynamoDB lookup
        if self._backend == "dynamodb" and self._table is not None:
            entry = self._dynamo_get(customer_id, product_id, task_name)
            if entry is not None:
                # Populate in-memory for subsequent requests
                self._memory_store[key] = entry
                return entry

        return None

    # ------------------------------------------------------------------
    # Public API: get_batch
    # ------------------------------------------------------------------

    def get_batch(
        self,
        customer_id: str,
        product_ids: List[str],
        task_name: Optional[str] = None,
    ) -> Dict[str, CacheEntry]:
        """Batch retrieve reasons for multiple products.

        Args:
            customer_id: Customer identifier.
            product_ids: List of product identifiers.
            task_name: If provided, filter by this task.  If ``None``,
                       returns any task for each product.

        Returns:
            Dict mapping ``product_id`` to :class:`CacheEntry`.
        """
        result: Dict[str, CacheEntry] = {}
        missing_ids: List[str] = []

        # Phase 1: in-memory scan
        for pid in product_ids:
            if task_name:
                key = self._make_key(customer_id, pid, task_name)
                entry = self._memory_store.get(key)
                if entry is not None and not self._is_expired(entry):
                    result[pid] = entry
                else:
                    missing_ids.append(pid)
            else:
                # Find any task for this customer+product
                found = self._memory_scan_product(customer_id, pid)
                if found:
                    result[pid] = found
                else:
                    missing_ids.append(pid)

        # Phase 2: DynamoDB batch for remaining
        if missing_ids and self._backend == "dynamodb" and self._table is not None:
            dynamo_results = self._dynamo_batch_get(
                customer_id, missing_ids, task_name,
            )
            for pid, entry in dynamo_results.items():
                result[pid] = entry
                key = self._make_key(
                    customer_id, pid, entry.task_name,
                )
                self._memory_store[key] = entry

        return result

    # ------------------------------------------------------------------
    # Public API: get_best
    # ------------------------------------------------------------------

    def get_best(
        self,
        customer_id: str,
        product_id: str,
        task_name: str,
    ) -> Optional[CacheEntry]:
        """Get the best available reason (L2b > L2a > L1).

        Searches across all layers and returns the highest-ranked entry
        that has passed quality validation.

        Args:
            customer_id: Customer identifier.
            product_id: Product identifier.
            task_name: Task / model name.

        Returns:
            Best :class:`CacheEntry` or ``None``.
        """
        # In the standard flow a single key stores the latest/best result,
        # so a simple get() usually suffices.  This method adds an explicit
        # layer preference when multiple entries coexist (e.g. in-memory
        # might hold L1 while DynamoDB holds L2b).

        candidates: List[CacheEntry] = []

        key = self._make_key(customer_id, product_id, task_name)
        mem_entry = self._memory_store.get(key)
        if mem_entry is not None and not self._is_expired(mem_entry):
            candidates.append(mem_entry)

        if self._backend == "dynamodb" and self._table is not None:
            ddb_entry = self._dynamo_get(customer_id, product_id, task_name)
            if ddb_entry is not None:
                candidates.append(ddb_entry)

        if not candidates:
            return None

        # Filter to quality-passed entries only
        passed = [c for c in candidates if c.quality_passed]
        if not passed:
            return None

        # Return highest layer rank
        passed.sort(key=lambda c: _LAYER_RANK.get(c.layer, 0), reverse=True)
        best = passed[0]

        # Ensure in-memory is up-to-date with best
        self._memory_store[key] = best
        return best

    # ------------------------------------------------------------------
    # Public API: invalidate
    # ------------------------------------------------------------------

    def invalidate(
        self,
        customer_id: str,
        product_id: Optional[str] = None,
        task_name: Optional[str] = None,
    ) -> int:
        """Invalidate cached reasons.  Returns count of entries removed.

        Granularity:
            - ``customer_id`` only: remove all entries for that customer.
            - ``customer_id`` + ``product_id``: remove all tasks for that pair.
            - ``customer_id`` + ``product_id`` + ``task_name``: exact entry.

        Args:
            customer_id: Customer identifier.
            product_id: Optional product identifier.
            task_name: Optional task name.

        Returns:
            Number of entries removed from in-memory store.  DynamoDB
            entries are deleted asynchronously (TTL handles the rest).
        """
        removed = 0

        if product_id and task_name:
            # Exact key removal
            key = self._make_key(customer_id, product_id, task_name)
            if key in self._memory_store:
                del self._memory_store[key]
                removed = 1
            if self._backend == "dynamodb" and self._table is not None:
                self._dynamo_delete(customer_id, product_id, task_name)
        else:
            # Prefix-based removal from in-memory store
            prefix = f"{customer_id}#"
            if product_id:
                prefix = f"{customer_id}#{product_id}#"

            keys_to_remove = [
                k for k in self._memory_store if k.startswith(prefix)
            ]
            for k in keys_to_remove:
                del self._memory_store[k]
            removed = len(keys_to_remove)

            # For DynamoDB, query + batch delete
            if self._backend == "dynamodb" and self._table is not None:
                self._dynamo_invalidate_prefix(customer_id, product_id)

        logger.info(
            "ReasonCache.invalidate: customer=%s, product=%s, task=%s -> %d removed",
            customer_id, product_id, task_name, removed,
        )
        return removed

    # ------------------------------------------------------------------
    # Public API: stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, int]:
        """Return cache statistics (in-memory store only).

        Returns:
            Dict with keys ``total``, ``by_layer`` (dict), ``by_task`` (dict),
            ``expired`` count.
        """
        now = int(time.time())
        by_layer: Dict[str, int] = {}
        by_task: Dict[str, int] = {}
        expired = 0

        for entry in self._memory_store.values():
            if entry.ttl > 0 and entry.ttl < now:
                expired += 1
                continue
            by_layer[entry.layer] = by_layer.get(entry.layer, 0) + 1
            by_task[entry.task_name] = by_task.get(entry.task_name, 0) + 1

        return {
            "total": len(self._memory_store) - expired,
            "expired": expired,
            "by_layer": by_layer,
            "by_task": by_task,
            "backend": self._backend,
        }

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_key(customer_id: str, product_id: str, task_name: str) -> str:
        """Build composite in-memory cache key."""
        return f"{customer_id}#{product_id}#{task_name}"

    @staticmethod
    def _make_sk(product_id: str, task_name: str) -> str:
        """Build DynamoDB sort key."""
        return f"{product_id}#{task_name}"

    def _compute_ttl(self) -> int:
        """Compute Unix timestamp for TTL expiry."""
        return int(time.time()) + self._ttl_hours * 3600

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if a cache entry has expired."""
        if entry.ttl <= 0:
            return False
        return int(time.time()) > entry.ttl

    # ------------------------------------------------------------------
    # In-memory helpers
    # ------------------------------------------------------------------

    def _memory_scan_product(
        self, customer_id: str, product_id: str,
    ) -> Optional[CacheEntry]:
        """Scan in-memory store for any task matching customer+product."""
        prefix = f"{customer_id}#{product_id}#"
        best: Optional[CacheEntry] = None
        for key, entry in self._memory_store.items():
            if key.startswith(prefix) and not self._is_expired(entry):
                if best is None or _LAYER_RANK.get(entry.layer, 0) > _LAYER_RANK.get(best.layer, 0):
                    best = entry
        return best

    # ------------------------------------------------------------------
    # DynamoDB operations
    # ------------------------------------------------------------------

    def _dynamo_put(self, entry: CacheEntry) -> None:
        """Write a cache entry to DynamoDB."""
        from decimal import Decimal

        sk = self._make_sk(entry.product_id, entry.task_name)

        item = {
            "customer_id": entry.customer_id,
            "sk": sk,
            "product_id": entry.product_id,
            "task_name": entry.task_name,
            "reason_text": entry.reason_text,
            "layer": entry.layer,
            "confidence": Decimal(str(entry.confidence)),
            "quality_passed": entry.quality_passed,
            "created_at": entry.created_at,
            "ttl": entry.ttl,
            "metadata": json.dumps(entry.metadata, default=str),
        }

        try:
            self._table.put_item(Item=item)
        except Exception as exc:
            logger.warning(
                "ReasonCache: DynamoDB put_item failed for %s/%s: %s",
                entry.customer_id, sk, exc,
            )

    def _dynamo_get(
        self,
        customer_id: str,
        product_id: str,
        task_name: str,
    ) -> Optional[CacheEntry]:
        """Read a single cache entry from DynamoDB."""
        sk = self._make_sk(product_id, task_name)

        try:
            resp = self._table.get_item(
                Key={"customer_id": customer_id, "sk": sk},
                ConsistentRead=False,
            )
        except Exception as exc:
            logger.warning(
                "ReasonCache: DynamoDB get_item failed for %s/%s: %s",
                customer_id, sk, exc,
            )
            return None

        item = resp.get("Item")
        if item is None:
            return None

        return self._item_to_entry(item)

    def _dynamo_batch_get(
        self,
        customer_id: str,
        product_ids: List[str],
        task_name: Optional[str],
    ) -> Dict[str, CacheEntry]:
        """Batch-fetch entries from DynamoDB.

        When ``task_name`` is ``None``, falls back to query per product
        (BatchGetItem requires exact keys).
        """
        result: Dict[str, CacheEntry] = {}

        if task_name:
            # Exact keys -- use BatchGetItem
            try:
                import boto3
                dynamo = boto3.resource("dynamodb", region_name=self._region)
            except Exception:
                return result

            for i in range(0, len(product_ids), self._BATCH_SIZE):
                chunk = product_ids[i: i + self._BATCH_SIZE]
                keys = [
                    {
                        "customer_id": customer_id,
                        "sk": self._make_sk(pid, task_name),
                    }
                    for pid in chunk
                ]

                try:
                    resp = dynamo.batch_get_item(
                        RequestItems={
                            self._table_name: {
                                "Keys": keys,
                                "ConsistentRead": False,
                            }
                        }
                    )
                except Exception as exc:
                    logger.warning(
                        "ReasonCache: DynamoDB batch_get_item failed: %s", exc,
                    )
                    continue

                for item in resp.get("Responses", {}).get(self._table_name, []):
                    entry = self._item_to_entry(item)
                    if entry is not None:
                        result[entry.product_id] = entry

                unprocessed = resp.get("UnprocessedKeys", {}).get(
                    self._table_name, {},
                )
                if unprocessed:
                    logger.warning(
                        "ReasonCache: %d unprocessed keys in batch",
                        len(unprocessed.get("Keys", [])),
                    )
        else:
            # No task_name -- query by customer_id + product_id prefix
            for pid in product_ids:
                entry = self._dynamo_query_best(customer_id, pid)
                if entry is not None:
                    result[pid] = entry

        return result

    def _dynamo_query_best(
        self, customer_id: str, product_id: str,
    ) -> Optional[CacheEntry]:
        """Query DynamoDB for all tasks of a customer+product, return best."""
        from boto3.dynamodb.conditions import Key

        try:
            resp = self._table.query(
                KeyConditionExpression=(
                    Key("customer_id").eq(customer_id)
                    & Key("sk").begins_with(f"{product_id}#")
                ),
                ConsistentRead=False,
            )
        except Exception as exc:
            logger.warning(
                "ReasonCache: DynamoDB query failed for %s/%s: %s",
                customer_id, product_id, exc,
            )
            return None

        items = resp.get("Items", [])
        if not items:
            return None

        entries = [self._item_to_entry(it) for it in items]
        entries = [e for e in entries if e is not None and e.quality_passed]
        if not entries:
            return None

        entries.sort(key=lambda e: _LAYER_RANK.get(e.layer, 0), reverse=True)
        return entries[0]

    def _dynamo_delete(
        self,
        customer_id: str,
        product_id: str,
        task_name: str,
    ) -> None:
        """Delete a single entry from DynamoDB."""
        sk = self._make_sk(product_id, task_name)
        try:
            self._table.delete_item(
                Key={"customer_id": customer_id, "sk": sk},
            )
        except Exception as exc:
            logger.warning(
                "ReasonCache: DynamoDB delete_item failed for %s/%s: %s",
                customer_id, sk, exc,
            )

    def _dynamo_invalidate_prefix(
        self,
        customer_id: str,
        product_id: Optional[str],
    ) -> None:
        """Delete all DynamoDB entries matching a customer (+ optional product)."""
        from boto3.dynamodb.conditions import Key

        try:
            kce = Key("customer_id").eq(customer_id)
            if product_id:
                kce = kce & Key("sk").begins_with(f"{product_id}#")

            resp = self._table.query(
                KeyConditionExpression=kce,
                ProjectionExpression="customer_id, sk",
                ConsistentRead=False,
            )
            items = resp.get("Items", [])

            with self._table.batch_writer() as batch:
                for item in items:
                    batch.delete_item(
                        Key={
                            "customer_id": item["customer_id"],
                            "sk": item["sk"],
                        }
                    )

            logger.debug(
                "ReasonCache: DynamoDB invalidated %d items for customer=%s",
                len(items), customer_id,
            )
        except Exception as exc:
            logger.warning(
                "ReasonCache: DynamoDB prefix invalidation failed: %s", exc,
            )

    # ------------------------------------------------------------------
    # DynamoDB item deserialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _item_to_entry(item: Dict[str, Any]) -> Optional[CacheEntry]:
        """Convert a DynamoDB item dict to a :class:`CacheEntry`.

        Handles both raw-attribute format (boto3 Table resource) and
        Decimal conversion.
        """
        from decimal import Decimal

        try:
            confidence = item.get("confidence", 1.0)
            if isinstance(confidence, Decimal):
                confidence = float(confidence)

            meta_raw = item.get("metadata", "{}")
            if isinstance(meta_raw, str):
                meta = json.loads(meta_raw)
            else:
                meta = meta_raw if isinstance(meta_raw, dict) else {}

            ttl_val = item.get("ttl", 0)
            if isinstance(ttl_val, Decimal):
                ttl_val = int(ttl_val)

            return CacheEntry(
                customer_id=str(item["customer_id"]),
                product_id=str(item.get("product_id", "")),
                task_name=str(item.get("task_name", "")),
                reason_text=str(item.get("reason_text", "")),
                layer=str(item.get("layer", "L1")),
                confidence=confidence,
                quality_passed=bool(item.get("quality_passed", True)),
                created_at=str(item.get("created_at", "")),
                ttl=ttl_val,
                metadata=meta,
            )
        except Exception as exc:
            logger.warning("ReasonCache: failed to deserialise item: %s", exc)
            return None
