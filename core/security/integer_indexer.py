"""
PII Integer Indexer -- Hash BLOB -> INT32 global index per domain.

Maintains a persistent, append-only mapping table per domain.
New hashes get the next available integer. Existing hashes return
their previously assigned integer.

Storage: One Parquet file per domain on S3 or local filesystem.
  {index_store_path}/{domain_name}_index.parquet
  Columns: hash_hex (string, hex-encoded SHA256), integer_id (int32)

Usage:
    indexer = PIIIntegerIndexer("s3://bucket/pii-indices/")
    int_series = indexer.index_column(hashed_series, PIIDomain.CUSTOMER)
"""
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .domains import PIIDomain

logger = logging.getLogger(__name__)


class PIIIntegerIndexer:
    """Maps hash BLOBs to INT32 global indices per domain.

    Persistent mapping stored as Parquet on S3 (or local).
    """

    def __init__(self, index_store_path: str = "pii_indices/"):
        """
        Args:
            index_store_path: S3 URI or local directory for index Parquet files.
        """
        self._store_path = index_store_path
        # In-memory indices: {domain: {hash_hex: int_id}}
        self._indices: Dict[str, Dict[str, int]] = {}
        self._next_id: Dict[str, int] = {}
        self._dirty: Dict[str, bool] = {}  # track which domains need saving

    # ── Single-value API ──────────────────────────────────────────────

    def get_or_create_index(self, hash_value: bytes, domain: PIIDomain) -> int:
        """Return existing INT32 index or assign next available."""
        self._ensure_loaded(domain)
        hex_key = hash_value.hex()
        domain_key = domain.value

        if hex_key in self._indices[domain_key]:
            return self._indices[domain_key][hex_key]

        # Assign next ID
        new_id = self._next_id[domain_key]
        self._indices[domain_key][hex_key] = new_id
        self._next_id[domain_key] = new_id + 1
        self._dirty[domain_key] = True
        return new_id

    # ── Column-level API ──────────────────────────────────────────────

    def index_column(
        self, hashed_series: pd.Series, domain: PIIDomain
    ) -> pd.Series:
        """Map entire column of hash bytes to INT32 indices."""
        self._ensure_loaded(domain)
        domain_key = domain.value
        idx_map = self._indices[domain_key]

        results = np.zeros(len(hashed_series), dtype=np.int32)
        for i, hash_val in enumerate(hashed_series):
            if hash_val is None or (
                isinstance(hash_val, bytes) and hash_val == b"\x00" * 32
            ):
                results[i] = -1  # null sentinel
                continue
            hex_key = (
                hash_val.hex() if isinstance(hash_val, bytes) else str(hash_val)
            )
            if hex_key in idx_map:
                results[i] = idx_map[hex_key]
            else:
                new_id = self._next_id[domain_key]
                idx_map[hex_key] = new_id
                self._next_id[domain_key] = new_id + 1
                self._dirty[domain_key] = True
                results[i] = new_id

        return pd.Series(results, index=hashed_series.index, dtype=np.int32)

    # ── DataFrame-level API ───────────────────────────────────────────

    def index_dataframe(
        self,
        df: pd.DataFrame,
        column_domain_map: Dict[str, PIIDomain],
    ) -> pd.DataFrame:
        """Process all hashed columns: replace hash bytes with INT32 indices.

        Expects columns named ``'{original}_hashed'``.  Renames output to
        ``'{original}_idx'``.
        """
        result = df.copy()
        for col, domain in column_domain_map.items():
            hashed_col = f"{col}_hashed"
            if hashed_col not in result.columns:
                continue
            idx_series = self.index_column(result[hashed_col], domain)
            result[f"{col}_idx"] = idx_series
            result = result.drop(columns=[hashed_col])
            logger.debug(
                "Indexed column '%s' -> '%s_idx' (domain=%s, unique=%d)",
                hashed_col,
                col,
                domain.value,
                len(set(idx_series.values)),
            )
        return result

    # ── Persistence ───────────────────────────────────────────────────

    def save_indices(self) -> None:
        """Persist all dirty domain indices to storage."""
        for domain_key, is_dirty in self._dirty.items():
            if not is_dirty:
                continue
            self._save_domain(domain_key)
            self._dirty[domain_key] = False

    def load_indices(self, domains: Optional[list] = None) -> None:
        """Pre-load index tables for specified domains."""
        targets = domains or [d.value for d in PIIDomain]
        for domain_key in targets:
            self._load_domain(domain_key)

    # ── Internal helpers ──────────────────────────────────────────────

    def _ensure_loaded(self, domain: PIIDomain) -> None:
        domain_key = domain.value
        if domain_key not in self._indices:
            self._load_domain(domain_key)

    def _load_domain(self, domain_key: str) -> None:
        """Load a domain's index from storage."""
        path = self._get_domain_path(domain_key)

        if self._is_s3_path(path):
            df = self._read_s3_parquet(path)
        elif os.path.exists(path):
            df = pd.read_parquet(path)
        else:
            df = None

        if df is not None and len(df) > 0:
            idx_map = dict(zip(df["hash_hex"].values, df["integer_id"].values))
            next_id = int(df["integer_id"].max()) + 1
            logger.info(
                "Loaded %d indices for domain '%s' (next_id=%d)",
                len(idx_map),
                domain_key,
                next_id,
            )
        else:
            idx_map = {}
            next_id = 1  # start from 1 (0 reserved, -1 for null)

        self._indices[domain_key] = idx_map
        self._next_id[domain_key] = next_id
        self._dirty[domain_key] = False

    def _save_domain(self, domain_key: str) -> None:
        """Save a domain's index to storage."""
        idx_map = self._indices.get(domain_key, {})
        if not idx_map:
            return

        df = pd.DataFrame(
            {
                "hash_hex": list(idx_map.keys()),
                "integer_id": list(idx_map.values()),
            }
        )
        df["integer_id"] = df["integer_id"].astype(np.int32)

        path = self._get_domain_path(domain_key)
        if self._is_s3_path(path):
            self._write_s3_parquet(df, path)
        else:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            df.to_parquet(path, index=False)

        logger.info(
            "Saved %d indices for domain '%s' to %s",
            len(idx_map),
            domain_key,
            path,
        )

    def _get_domain_path(self, domain_key: str) -> str:
        base = self._store_path.rstrip("/")
        return f"{base}/{domain_key}_index.parquet"

    @staticmethod
    def _is_s3_path(path: str) -> bool:
        return path.startswith("s3://")

    def _read_s3_parquet(self, s3_uri: str):
        """Read Parquet from S3."""
        try:
            import boto3
            import io

            parts = s3_uri.replace("s3://", "").split("/", 1)
            bucket, key = parts[0], parts[1]
            s3 = boto3.client("s3")
            obj = s3.get_object(Bucket=bucket, Key=key)
            return pd.read_parquet(io.BytesIO(obj["Body"].read()))
        except Exception:
            return None

    def _write_s3_parquet(self, df: pd.DataFrame, s3_uri: str) -> None:
        """Write Parquet to S3."""
        import boto3
        import io

        parts = s3_uri.replace("s3://", "").split("/", 1)
        bucket, key = parts[0], parts[1]
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        buf.seek(0)
        s3 = boto3.client("s3")
        s3.upload_fileobj(buf, bucket, key)

    # ── Stats ─────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, int]:
        """Return index statistics per domain."""
        return {k: len(v) for k, v in self._indices.items() if v}
