"""
Dataset Registry -- S3-based dataset versioning to replace DVC.

Stores versioned datasets as Parquet with full traceability:
- Schema hash (SHA-256) for change detection
- Per-column feature statistics (mean, std, null_pct)
- Row/column counts for reproducibility verification
- Quality gate validation status
- Parent version lineage tracking

Storage: S3 with versioned directory structure.

Usage::

    registry = DatasetRegistry(
        s3_base="s3://bucket/datasets/",
        region="ap-northeast-2",
    )

    # Register a new dataset version
    version = registry.register(
        source_name="user_events",
        df=df,
        version="v1.0.0",
        metadata={"git_sha": "abc123", "pipeline_run_id": "run-42"},
    )

    # Load for training
    manifest = registry.load_manifest("user_events", "v1.0.0")
    df = registry.load_data("user_events", "v1.0.0")

    # List versions
    versions = registry.list_versions("user_events")

    # Compare two versions
    diff = registry.diff("user_events", "v1.0.0", "v1.1.0")
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "DatasetVersion",
    "DatasetRegistry",
]


# ============================================================================
# Version manifest dataclass
# ============================================================================


@dataclass
class DatasetVersion:
    """Metadata manifest for a single versioned dataset snapshot.

    Serialized as ``manifest.json`` inside each version directory.
    This is the single source of truth for everything in a version
    directory.

    Attributes:
        version: Semantic version string (e.g. ``"v1.0.0"``).
        created_at: ISO 8601 timestamp when the version was registered.
        source_name: Logical dataset name (e.g. ``"user_events"``).
        s3_uri: Full S3 URI to the ``data.parquet`` file.
        row_count: Number of rows in the dataset.
        column_count: Number of columns in the dataset.
        schema_hash: SHA-256 hash of the sorted column-name + dtype list.
            Used for schema change detection across versions.
        validation_passed: Whether the quality gate check passed for
            this snapshot.
        feature_stats: Per-column summary statistics.  Each key is a
            column name mapping to ``{"mean": ..., "std": ...,
            "null_pct": ...}``.
        parent_version: Previous version string for lineage tracking.
            ``None`` if this is the first version.
        metadata: Arbitrary extra metadata (git SHA, pipeline run ID,
            etc.).
    """

    version: str
    created_at: str
    source_name: str
    s3_uri: str
    row_count: int
    column_count: int
    schema_hash: str
    validation_passed: bool = False
    feature_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    parent_version: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Dataset Registry
# ============================================================================


class DatasetRegistry:
    """S3-based dataset registry with versioning and manifest tracking.

    Operates local-first: artifacts are written to ``local_base`` and
    then optionally uploaded to S3 when ``s3_base`` is configured.
    For loading, artifacts are downloaded from S3 to a local cache
    directory first.

    Directory structure per version::

        s3://bucket/datasets/{source_name}/{version}/
            data.parquet
            manifest.json
            stats.json

    Parameters:
        s3_base: S3 URI prefix (e.g. ``"s3://bucket/datasets/"``).
            Leave empty for local-only operation.
        local_base: Local directory for storing version artifacts.
        region: AWS region for boto3 S3 client.
    """

    def __init__(
        self,
        s3_base: str = "",
        local_base: str = "datasets/",
        region: str = "ap-northeast-2",
    ) -> None:
        self._s3_base = s3_base.rstrip("/")
        self._local_base = local_base
        self._region = region

    # ------------------------------------------------------------------
    # Register
    # ------------------------------------------------------------------

    def register(
        self,
        source_name: str,
        df: Any,
        version: str,
        metadata: Optional[Dict[str, Any]] = None,
        parent_version: Optional[str] = None,
        validation_passed: bool = True,
    ) -> DatasetVersion:
        """Register a new dataset version from a pandas DataFrame.

        Computes schema hash and feature statistics automatically,
        writes Parquet to local storage, and uploads to S3 when
        configured.

        Parameters:
            source_name: Logical dataset name (e.g. ``"user_events"``).
            df: A ``pandas.DataFrame`` to register.
            version: Semantic version string (e.g. ``"v1.0.0"``).
            metadata: Arbitrary metadata to attach to the manifest.
            parent_version: Previous version string for lineage.
                If ``None``, attempts to auto-detect from
                :meth:`get_latest`.
            validation_passed: Whether the quality gate passed for this
                snapshot.

        Returns:
            A :class:`DatasetVersion` manifest describing the registered
            snapshot.
        """
        import pandas as pd  # noqa: F811 — lazy import

        metadata = metadata or {}

        # Auto-detect parent version if not provided
        if parent_version is None:
            parent_version = self.get_latest(source_name)

        version_dir = self._get_version_dir(source_name, version)

        # -- Compute statistics -----------------------------------------------
        row_count = len(df)
        column_count = len(df.columns)
        schema_hash = self._compute_schema_hash(df)
        feature_stats = self._compute_feature_stats(df)

        # -- Write Parquet ----------------------------------------------------
        parquet_path = os.path.join(version_dir, "data.parquet")
        df.to_parquet(parquet_path, index=False)
        logger.info(
            "Dataset '%s/%s' saved to %s (%d rows, %d cols)",
            source_name, version, parquet_path, row_count, column_count,
        )

        # -- Write stats.json -------------------------------------------------
        self._write_json(feature_stats, os.path.join(version_dir, "stats.json"))

        # -- Build S3 URI -----------------------------------------------------
        if self._s3_base:
            s3_uri = (
                f"{self._s3_base}/{source_name}/{version}/data.parquet"
            )
        else:
            s3_uri = ""

        # -- Build manifest ---------------------------------------------------
        now = datetime.now(timezone.utc).isoformat()
        manifest = DatasetVersion(
            version=version,
            created_at=now,
            source_name=source_name,
            s3_uri=s3_uri,
            row_count=row_count,
            column_count=column_count,
            schema_hash=schema_hash,
            validation_passed=validation_passed,
            feature_stats=feature_stats,
            parent_version=parent_version,
            metadata=metadata,
        )

        self._write_json(
            asdict(manifest), os.path.join(version_dir, "manifest.json"),
        )
        logger.info(
            "Dataset version '%s/%s' registered: %d rows, %d cols, "
            "schema_hash=%s, validation=%s",
            source_name, version, row_count, column_count,
            schema_hash[:12], validation_passed,
        )

        # -- Upload to S3 if configured ---------------------------------------
        if self._s3_base:
            s3_prefix = f"{self._s3_base}/{source_name}/{version}"
            self._upload_to_s3(version_dir, s3_prefix)

        return manifest

    # ------------------------------------------------------------------
    # Load manifest
    # ------------------------------------------------------------------

    def load_manifest(
        self, source_name: str, version: str,
    ) -> DatasetVersion:
        """Load the manifest for a specific dataset version.

        Downloads from S3 if the local copy is not available.

        Parameters:
            source_name: Logical dataset name.
            version: Version string (e.g. ``"v1.0.0"``).

        Returns:
            A :class:`DatasetVersion` with all manifest fields.

        Raises:
            FileNotFoundError: If the manifest cannot be found locally
                or on S3.
        """
        version_dir = self._ensure_local(source_name, version)
        manifest_path = os.path.join(version_dir, "manifest.json")

        data = self._read_json(manifest_path)
        return DatasetVersion(
            version=data["version"],
            created_at=data["created_at"],
            source_name=data.get("source_name", source_name),
            s3_uri=data.get("s3_uri", ""),
            row_count=data.get("row_count", 0),
            column_count=data.get("column_count", 0),
            schema_hash=data.get("schema_hash", ""),
            validation_passed=data.get("validation_passed", False),
            feature_stats=data.get("feature_stats", {}),
            parent_version=data.get("parent_version"),
            metadata=data.get("metadata", {}),
        )

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------

    def load_data(self, source_name: str, version: str) -> Any:
        """Load the Parquet data for a specific dataset version.

        Downloads from S3 if the local copy is not available.

        Parameters:
            source_name: Logical dataset name.
            version: Version string.

        Returns:
            A ``pandas.DataFrame`` loaded from the stored Parquet file.

        Raises:
            FileNotFoundError: If the Parquet file cannot be found.
        """
        import pandas as pd  # noqa: F811 — lazy import

        version_dir = self._ensure_local(source_name, version)
        parquet_path = os.path.join(version_dir, "data.parquet")

        if not os.path.exists(parquet_path):
            raise FileNotFoundError(
                f"Dataset parquet not found: {parquet_path}"
            )

        df = pd.read_parquet(parquet_path)
        logger.info(
            "Dataset '%s/%s' loaded: %d rows, %d cols",
            source_name, version, len(df), len(df.columns),
        )
        return df

    # ------------------------------------------------------------------
    # List versions
    # ------------------------------------------------------------------

    def list_versions(self, source_name: str) -> List[DatasetVersion]:
        """List all versions for a source, sorted by ``created_at`` descending.

        Scans the local base directory for version directories
        containing ``manifest.json``.

        Parameters:
            source_name: Logical dataset name.

        Returns:
            List of :class:`DatasetVersion` manifests, most recent first.
        """
        source_dir = Path(self._local_base) / source_name
        if not source_dir.exists():
            return []

        manifests: List[DatasetVersion] = []
        for child in sorted(source_dir.iterdir()):
            manifest_path = child / "manifest.json"
            if child.is_dir() and manifest_path.exists():
                try:
                    data = self._read_json(str(manifest_path))
                    manifests.append(
                        DatasetVersion(
                            version=data["version"],
                            created_at=data["created_at"],
                            source_name=data.get("source_name", source_name),
                            s3_uri=data.get("s3_uri", ""),
                            row_count=data.get("row_count", 0),
                            column_count=data.get("column_count", 0),
                            schema_hash=data.get("schema_hash", ""),
                            validation_passed=data.get(
                                "validation_passed", False,
                            ),
                            feature_stats=data.get("feature_stats", {}),
                            parent_version=data.get("parent_version"),
                            metadata=data.get("metadata", {}),
                        )
                    )
                except Exception:
                    logger.warning(
                        "Failed to load manifest from %s", manifest_path,
                        exc_info=True,
                    )

        manifests.sort(key=lambda m: m.created_at, reverse=True)
        return manifests

    # ------------------------------------------------------------------
    # Get latest
    # ------------------------------------------------------------------

    def get_latest(self, source_name: str) -> Optional[str]:
        """Get the latest version string for a source.

        Parameters:
            source_name: Logical dataset name.

        Returns:
            Version string of the most recently created version, or
            ``None`` if no versions exist.
        """
        versions = self.list_versions(source_name)
        return versions[0].version if versions else None

    # ------------------------------------------------------------------
    # Diff
    # ------------------------------------------------------------------

    def diff(
        self,
        source_name: str,
        version_a: str,
        version_b: str,
    ) -> Dict[str, Any]:
        """Compare two dataset versions and return the differences.

        Reports schema changes, row/column count deltas, and per-column
        statistical drift between the two versions.

        Parameters:
            source_name: Logical dataset name.
            version_a: First version string (typically the older one).
            version_b: Second version string (typically the newer one).

        Returns:
            Dict with keys:

            - ``schema_changed`` (bool): Whether the schema hash differs.
            - ``row_count_delta`` (int): ``version_b.row_count -
              version_a.row_count``.
            - ``column_count_delta`` (int): ``version_b.column_count -
              version_a.column_count``.
            - ``columns_added`` (List[str]): Columns present in B but
              not A.
            - ``columns_removed`` (List[str]): Columns present in A but
              not B.
            - ``stat_diffs`` (Dict): Per-column drift for columns
              present in both versions.  Each entry has
              ``{"mean_delta", "std_delta", "null_pct_delta"}``.
        """
        manifest_a = self.load_manifest(source_name, version_a)
        manifest_b = self.load_manifest(source_name, version_b)

        cols_a = set(manifest_a.feature_stats.keys())
        cols_b = set(manifest_b.feature_stats.keys())

        # Per-column stat diffs for shared columns
        stat_diffs: Dict[str, Dict[str, float]] = {}
        for col in sorted(cols_a & cols_b):
            stats_a = manifest_a.feature_stats[col]
            stats_b = manifest_b.feature_stats[col]
            stat_diffs[col] = {
                "mean_delta": stats_b.get("mean", 0.0) - stats_a.get("mean", 0.0),
                "std_delta": stats_b.get("std", 0.0) - stats_a.get("std", 0.0),
                "null_pct_delta": (
                    stats_b.get("null_pct", 0.0) - stats_a.get("null_pct", 0.0)
                ),
            }

        return {
            "version_a": version_a,
            "version_b": version_b,
            "schema_changed": manifest_a.schema_hash != manifest_b.schema_hash,
            "row_count_delta": manifest_b.row_count - manifest_a.row_count,
            "column_count_delta": manifest_b.column_count - manifest_a.column_count,
            "columns_added": sorted(cols_b - cols_a),
            "columns_removed": sorted(cols_a - cols_b),
            "stat_diffs": stat_diffs,
        }

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DatasetRegistry":
        """Create a DatasetRegistry from a configuration dict.

        Expected keys::

            {
                "s3_base": "s3://bucket/datasets/",
                "local_base": "datasets/",
                "region": "ap-northeast-2"
            }

        Parameters:
            config: Configuration dictionary.

        Returns:
            A configured :class:`DatasetRegistry` instance.
        """
        return cls(
            s3_base=config.get("s3_base", ""),
            local_base=config.get("local_base", "datasets/"),
            region=config.get("region", "ap-northeast-2"),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_version_dir(self, source_name: str, version: str) -> str:
        """Get local directory path for a source/version pair."""
        version_dir = os.path.join(self._local_base, source_name, version)
        os.makedirs(version_dir, exist_ok=True)
        return version_dir

    def _ensure_local(self, source_name: str, version: str) -> str:
        """Ensure version artifacts exist locally, downloading if needed.

        Returns the local version directory path.
        """
        version_dir = self._get_version_dir(source_name, version)
        manifest_path = os.path.join(version_dir, "manifest.json")

        if not os.path.exists(manifest_path) and self._s3_base:
            s3_prefix = f"{self._s3_base}/{source_name}/{version}"
            self._download_from_s3(s3_prefix, version_dir)

        return version_dir

    @staticmethod
    def _compute_schema_hash(df: Any) -> str:
        """Compute a SHA-256 hash of the DataFrame schema.

        The schema is defined as the sorted list of ``(column_name,
        dtype_str)`` pairs.  This means reordering columns does NOT
        change the hash, but renaming or re-typing a column does.

        Parameters:
            df: A ``pandas.DataFrame``.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        schema_items = sorted(
            (str(col), str(dtype)) for col, dtype in zip(df.columns, df.dtypes)
        )
        schema_str = json.dumps(schema_items, sort_keys=True)
        return hashlib.sha256(schema_str.encode("utf-8")).hexdigest()

    @staticmethod
    def _compute_feature_stats(df: Any) -> Dict[str, Dict[str, float]]:
        """Compute per-column summary statistics.

        For each column, computes:

        - ``mean``: Arithmetic mean (numeric columns only; ``0.0`` for
          non-numeric).
        - ``std``: Standard deviation (numeric columns only; ``0.0`` for
          non-numeric).
        - ``null_pct``: Percentage of null values (0.0 -- 100.0).

        Parameters:
            df: A ``pandas.DataFrame``.

        Returns:
            Dict mapping column name to ``{"mean", "std", "null_pct"}``.
        """
        import numpy as np  # noqa: F811 — lazy import

        stats: Dict[str, Dict[str, float]] = {}
        total_rows = len(df)

        for col in df.columns:
            col_str = str(col)
            null_count = int(df[col].isna().sum())
            null_pct = (null_count / total_rows * 100.0) if total_rows > 0 else 0.0

            if np.issubdtype(df[col].dtype, np.number):
                col_mean = float(df[col].mean()) if total_rows > 0 else 0.0
                col_std = float(df[col].std()) if total_rows > 0 else 0.0
                # Handle NaN from all-null columns
                if np.isnan(col_mean):
                    col_mean = 0.0
                if np.isnan(col_std):
                    col_std = 0.0
            else:
                col_mean = 0.0
                col_std = 0.0

            stats[col_str] = {
                "mean": round(col_mean, 6),
                "std": round(col_std, 6),
                "null_pct": round(null_pct, 4),
            }

        return stats

    def _read_json(self, path: str) -> Dict:
        """Read JSON from a local path.

        Parameters:
            path: Absolute or relative path to a JSON file.

        Returns:
            Parsed dict.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_json(self, data: Any, path: str) -> None:
        """Write JSON to a local path.

        Creates parent directories as needed.

        Parameters:
            data: Object to serialize.
            path: Target file path.
        """
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def _upload_to_s3(self, local_dir: str, s3_prefix: str) -> None:
        """Upload an entire local directory to S3.

        Walks the directory tree and uploads each file, preserving the
        relative path structure under ``s3_prefix``.

        Parameters:
            local_dir: Local directory to upload.
            s3_prefix: S3 URI prefix (e.g.
                ``"s3://bucket/datasets/user_events/v1.0.0"``).
        """
        try:
            import boto3
        except ImportError:
            logger.warning("boto3 not available; skipping S3 upload")
            return

        s3 = boto3.client("s3", region_name=self._region)
        bucket, prefix = self._parse_s3_uri(s3_prefix)

        for root, _dirs, files in os.walk(local_dir):
            for fname in files:
                local_path = os.path.join(root, fname)
                rel_path = os.path.relpath(local_path, local_dir).replace(
                    "\\", "/",
                )
                s3_key = f"{prefix}/{rel_path}" if prefix else rel_path
                s3.upload_file(local_path, bucket, s3_key)

        logger.info(
            "Uploaded %s to s3://%s/%s", local_dir, bucket, prefix,
        )

    def _download_from_s3(self, s3_prefix: str, local_dir: str) -> None:
        """Download all objects under an S3 prefix to a local directory.

        Parameters:
            s3_prefix: S3 URI prefix (e.g.
                ``"s3://bucket/datasets/user_events/v1.0.0"``).
            local_dir: Local directory to download into.
        """
        try:
            import boto3
        except ImportError:
            logger.warning("boto3 not available; skipping S3 download")
            return

        s3 = boto3.client("s3", region_name=self._region)
        bucket, prefix = self._parse_s3_uri(s3_prefix)

        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                rel_path = key[len(prefix):].lstrip("/")
                if not rel_path:
                    continue

                local_path = os.path.join(local_dir, rel_path)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                s3.download_file(bucket, key, local_path)

        logger.info(
            "Downloaded s3://%s/%s to %s", bucket, prefix, local_dir,
        )

    @staticmethod
    def _parse_s3_uri(uri: str) -> Tuple[str, str]:
        """Parse an S3 URI into (bucket, key/prefix).

        Parameters:
            uri: S3 URI (e.g. ``"s3://bucket/path/to/object"``).

        Returns:
            ``(bucket, key)`` tuple.
        """
        path = uri.replace("s3://", "")
        parts = path.split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        return bucket, key
