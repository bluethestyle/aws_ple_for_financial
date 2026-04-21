"""
Feature Store Abstraction
=========================

Two concrete implementations behind a common abstract interface:

* **MemoryFeatureStore** -- loads a Parquet file from S3 into an in-memory
  dict keyed by ``user_id``.  Fast lookup, but requires the full dataset
  to fit in Lambda / ECS memory.  Best for user populations under ~5 M.

* **DynamoDBFeatureStore** -- key-value lookup from a DynamoDB table
  partitioned by ``user_id``.  Scales horizontally and is suitable for
  arbitrarily large populations.

The factory :class:`FeatureStoreFactory` selects the right backend
automatically based on :class:`~core.serving.config.ServingConfig`.
"""

from __future__ import annotations

import io
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "AbstractFeatureStore",
    "MemoryFeatureStore",
    "DynamoDBFeatureStore",
    "FeatureStoreFactory",
]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class AbstractFeatureStore(ABC):
    """Base class for all feature store backends.

    Subclasses must implement :meth:`get`, :meth:`get_batch`, and
    :meth:`health_check`.
    """

    @abstractmethod
    def get(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Return the feature vector for a single user.

        Args:
            user_id: Unique user identifier.

        Returns:
            A dict of feature-name to feature-value, or ``None`` if the
            user is not found.
        """

    @abstractmethod
    def get_batch(self, user_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Return feature vectors for multiple users.

        Args:
            user_ids: List of user identifiers.

        Returns:
            A dict mapping each found ``user_id`` to its feature dict.
            Missing users are silently omitted.
        """

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Return health / readiness information.

        Returns:
            A dict with at least ``{"healthy": bool}``.
        """


# ---------------------------------------------------------------------------
# In-memory implementation (S3 Parquet)
# ---------------------------------------------------------------------------

class MemoryFeatureStore(AbstractFeatureStore):
    """Load Parquet features from S3 into an in-memory dictionary.

    At construction time the store downloads a Parquet file, converts it
    to a dict keyed by ``user_id_column``, and serves all reads from
    memory.

    Args:
        s3_uri: Full S3 URI (``s3://bucket/path/features.parquet``).
        user_id_column: Column name to use as the lookup key.
        region: AWS region for the S3 client. ``None`` lets boto3 resolve
            from env / credentials; callers should pass
            ``pipeline.yaml::aws.region`` via feature_store_config.
    """

    def __init__(
        self,
        s3_uri: str,
        user_id_column: str = "user_id",
        region: Optional[str] = None,
    ) -> None:
        self._s3_uri = s3_uri
        self._user_id_column = user_id_column
        self._region = region
        self._store: Dict[str, Dict[str, Any]] = {}
        self._load_time_ms: float = 0.0
        self._record_count: int = 0

        self._load()

    # -- internal ----------------------------------------------------------

    def _load(self) -> None:
        """Download Parquet from S3 and build the lookup dict."""
        import boto3
        import pyarrow.parquet as pq

        t0 = time.perf_counter()
        logger.info("MemoryFeatureStore: loading %s", self._s3_uri)

        # Parse s3://bucket/key
        parts = self._s3_uri.replace("s3://", "").split("/", 1)
        bucket, key = parts[0], parts[1]

        s3 = boto3.client("s3", region_name=self._region)
        response = s3.get_object(Bucket=bucket, Key=key)
        body = response["Body"].read()

        table = pq.read_table(io.BytesIO(body))
        df = table.to_pandas()

        if self._user_id_column not in df.columns:
            raise KeyError(
                f"MemoryFeatureStore: column '{self._user_id_column}' "
                f"not found in Parquet.  Available: {list(df.columns)}"
            )

        # Build lookup dict -- drop the key column from feature values
        df = df.set_index(self._user_id_column)
        self._store = df.to_dict(orient="index")
        self._record_count = len(self._store)

        elapsed = (time.perf_counter() - t0) * 1000.0
        self._load_time_ms = round(elapsed, 1)
        logger.info(
            "MemoryFeatureStore: loaded %d users in %.1f ms",
            self._record_count, self._load_time_ms,
        )

    # -- public API --------------------------------------------------------

    def get(self, user_id: str) -> Optional[Dict[str, Any]]:
        return self._store.get(user_id)

    def get_batch(self, user_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        for uid in user_ids:
            features = self._store.get(uid)
            if features is not None:
                result[uid] = features
        return result

    def health_check(self) -> Dict[str, Any]:
        return {
            "healthy": self._record_count > 0,
            "backend": "memory",
            "record_count": self._record_count,
            "load_time_ms": self._load_time_ms,
            "s3_uri": self._s3_uri,
        }


# ---------------------------------------------------------------------------
# DynamoDB implementation
# ---------------------------------------------------------------------------

class DynamoDBFeatureStore(AbstractFeatureStore):
    """Key-value feature lookup backed by DynamoDB.

    The table must have a partition key named ``user_id`` (string).
    Each item stores feature values as top-level DynamoDB attributes.

    Args:
        table_name: DynamoDB table name.
        region: AWS region. ``None`` lets boto3 resolve from env /
            credentials; callers should pass ``pipeline.yaml::aws.region``
            via feature_store_config.
        user_id_key: Name of the partition key attribute.
    """

    # DynamoDB BatchGetItem limit
    _BATCH_SIZE = 100

    def __init__(
        self,
        table_name: str,
        region: Optional[str] = None,
        user_id_key: str = "user_id",
    ) -> None:
        import boto3

        self._table_name = table_name
        self._user_id_key = user_id_key
        self._region = region

        dynamodb = boto3.resource("dynamodb", region_name=region)
        self._table = dynamodb.Table(table_name)

        logger.info(
            "DynamoDBFeatureStore: table=%s, region=%s",
            table_name, region,
        )

    # -- public API --------------------------------------------------------

    def get(self, user_id: str) -> Optional[Dict[str, Any]]:
        try:
            response = self._table.get_item(
                Key={self._user_id_key: user_id},
                ConsistentRead=False,
            )
        except Exception:
            logger.exception(
                "DynamoDBFeatureStore.get failed for user_id=%s", user_id,
            )
            return None

        item = response.get("Item")
        if item is None:
            return None

        # Strip the key column before returning
        features = {k: v for k, v in item.items() if k != self._user_id_key}
        return self._deserialise_numerics(features)

    def get_batch(self, user_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Batch-fetch features using DynamoDB BatchGetItem.

        Automatically pages in chunks of 100 (the DynamoDB limit).
        """
        import boto3

        result: Dict[str, Dict[str, Any]] = {}
        dynamodb = boto3.resource("dynamodb", region_name=self._region)

        for i in range(0, len(user_ids), self._BATCH_SIZE):
            chunk = user_ids[i : i + self._BATCH_SIZE]
            keys = [{self._user_id_key: uid} for uid in chunk]

            try:
                response = dynamodb.batch_get_item(
                    RequestItems={
                        self._table_name: {"Keys": keys, "ConsistentRead": False}
                    }
                )
            except Exception:
                logger.exception(
                    "DynamoDBFeatureStore.get_batch failed for chunk %d-%d",
                    i, i + len(chunk),
                )
                continue

            for item in response.get("Responses", {}).get(self._table_name, []):
                uid = str(item[self._user_id_key])
                features = {
                    k: v for k, v in item.items() if k != self._user_id_key
                }
                result[uid] = self._deserialise_numerics(features)

            # Handle unprocessed keys (throttled)
            unprocessed = response.get("UnprocessedKeys", {}).get(
                self._table_name, {},
            )
            if unprocessed:
                logger.warning(
                    "DynamoDBFeatureStore: %d unprocessed keys in batch",
                    len(unprocessed.get("Keys", [])),
                )

        return result

    def health_check(self) -> Dict[str, Any]:
        try:
            self._table.table_status  # noqa: B018 -- side-effect read
            return {
                "healthy": True,
                "backend": "dynamodb",
                "table_name": self._table_name,
            }
        except Exception as exc:
            return {
                "healthy": False,
                "backend": "dynamodb",
                "table_name": self._table_name,
                "error": str(exc),
            }

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _deserialise_numerics(features: Dict[str, Any]) -> Dict[str, Any]:
        """Convert DynamoDB Decimal values to native Python floats."""
        from decimal import Decimal

        return {
            k: float(v) if isinstance(v, Decimal) else v
            for k, v in features.items()
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class FeatureStoreFactory:
    """Create the appropriate feature store from a :class:`ServingConfig`.

    When the config specifies ``feature_store: auto``, the factory counts
    users (either from a provided count or from the data source) and picks
    the backend accordingly.

    Example::

        from core.serving.config import ServingConfig
        config = ServingConfig.from_dict(yaml_dict)
        store = FeatureStoreFactory.create(config, user_count=3_000_000)
    """

    @staticmethod
    def create(
        config: "ServingConfig",
        user_count: Optional[int] = None,
    ) -> AbstractFeatureStore:
        """Instantiate the correct feature store backend.

        Args:
            config: A fully hydrated :class:`ServingConfig`.
            user_count: Optional known user population size.  Used when
                ``feature_store`` mode is ``auto``.

        Returns:
            An instance of :class:`AbstractFeatureStore`.

        Raises:
            ValueError: If the resolved mode is unknown or required config
                fields are missing.
        """
        from .config import FeatureStoreMode

        resolved = config.resolve_feature_store_mode(user_count)
        fs_cfg = config.feature_store_config

        if resolved == FeatureStoreMode.MEMORY:
            s3_uri = fs_cfg.get("s3_uri", "")
            if not s3_uri:
                raise ValueError(
                    "MemoryFeatureStore requires 'feature_store_config.s3_uri'"
                )
            return MemoryFeatureStore(
                s3_uri=s3_uri,
                user_id_column=fs_cfg.get("user_id_column", "user_id"),
                region=fs_cfg.get("region"),
            )

        if resolved == FeatureStoreMode.DYNAMODB:
            table_name = fs_cfg.get("dynamodb_table", "")
            if not table_name:
                raise ValueError(
                    "DynamoDBFeatureStore requires "
                    "'feature_store_config.dynamodb_table'"
                )
            return DynamoDBFeatureStore(
                table_name=table_name,
                region=fs_cfg.get("region"),
                user_id_key=fs_cfg.get("user_id_key", "user_id"),
            )

        raise ValueError(f"Unknown feature store mode: {resolved}")
