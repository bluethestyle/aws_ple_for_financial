"""
PII Integer Indexer -- Hash BLOB -> INT32 global index per domain.

Maintains a persistent, append-only mapping table per domain. New
hashes get the next available integer; existing hashes return their
previously assigned integer.

Storage: one Parquet file per domain on S3 or the local filesystem.

    {index_store_path}/{domain_name}_index.parquet
    Columns: hash_hex (string, hex-encoded SHA256), integer_id (int32)

Pandas-free implementation (CLAUDE.md §3.3 pandas 금지). Uses DuckDB
for Parquet reads (and httpfs for ``s3://`` URIs) and pyarrow for
Parquet writes. Returns INT32 numpy arrays so serving callers can
tensor-convert without round-tripping through pandas.

Usage::

    indexer = PIIIntegerIndexer("s3://bucket/pii-indices/")
    int_array = indexer.index_column(hashed_bytes_list, PIIDomain.CUSTOMER)
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .domains import PIIDomain

logger = logging.getLogger(__name__)

_NULL_DIGEST = b"\x00" * 32


class PIIIntegerIndexer:
    """Maps hash BLOBs to INT32 global indices per domain."""

    def __init__(self, index_store_path: str = "pii_indices/"):
        self._store_path = index_store_path
        # In-memory indices: {domain_key: {hash_hex: int_id}}
        self._indices: Dict[str, Dict[str, int]] = {}
        self._next_id: Dict[str, int] = {}
        self._dirty: Dict[str, bool] = {}

    # ------------------------------------------------------------------
    # Single value
    # ------------------------------------------------------------------

    def get_or_create_index(
        self, hash_value: bytes, domain: PIIDomain,
    ) -> int:
        """Return the existing INT32 index or assign the next available."""
        self._ensure_loaded(domain)
        hex_key = hash_value.hex()
        domain_key = domain.value

        idx_map = self._indices[domain_key]
        if hex_key in idx_map:
            return idx_map[hex_key]

        new_id = self._next_id[domain_key]
        idx_map[hex_key] = new_id
        self._next_id[domain_key] = new_id + 1
        self._dirty[domain_key] = True
        return new_id

    # ------------------------------------------------------------------
    # Column
    # ------------------------------------------------------------------

    def index_column(
        self,
        hashed_values: Iterable[Any],
        domain: PIIDomain,
    ) -> np.ndarray:
        """Map an iterable of hash bytes to an INT32 numpy array.

        ``None`` and the all-zero sentinel digest map to ``-1``.
        """
        self._ensure_loaded(domain)
        domain_key = domain.value
        idx_map = self._indices[domain_key]

        materialised = list(hashed_values)
        results = np.full(len(materialised), -1, dtype=np.int32)
        for i, hash_val in enumerate(materialised):
            if hash_val is None:
                continue
            if isinstance(hash_val, bytes):
                if hash_val == _NULL_DIGEST:
                    continue
                hex_key = hash_val.hex()
            else:
                hex_key = str(hash_val)
            if hex_key in idx_map:
                results[i] = idx_map[hex_key]
            else:
                new_id = self._next_id[domain_key]
                idx_map[hex_key] = new_id
                self._next_id[domain_key] = new_id + 1
                self._dirty[domain_key] = True
                results[i] = new_id
        return results

    # ------------------------------------------------------------------
    # Columnar dict
    # ------------------------------------------------------------------

    def index_columns(
        self,
        columns: Dict[str, Sequence[Any]],
        column_domain_map: Dict[str, PIIDomain],
    ) -> Dict[str, Sequence[Any]]:
        """Replace ``{col}_hashed`` entries with INT32 ``{col}_idx``.

        Columns that do not carry a hashed variant are passed through
        unchanged, so callers can hand over partial payloads safely.
        """
        result: Dict[str, Sequence[Any]] = dict(columns)
        for col, domain in column_domain_map.items():
            hashed_col = f"{col}_hashed"
            if hashed_col not in result:
                continue
            idx_array = self.index_column(result[hashed_col], domain)
            result[f"{col}_idx"] = idx_array
            del result[hashed_col]
            unique = int(len(set(idx_array.tolist())))
            logger.debug(
                "Indexed '%s' -> '%s_idx' (domain=%s, unique=%d)",
                hashed_col, col, domain.value, unique,
            )
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_indices(self) -> None:
        """Persist every domain whose in-memory map has been mutated."""
        for domain_key, is_dirty in list(self._dirty.items()):
            if not is_dirty:
                continue
            self._save_domain(domain_key)
            self._dirty[domain_key] = False

    def load_indices(self, domains: Optional[List[str]] = None) -> None:
        """Pre-load a set of domain indices into memory."""
        targets = domains or [d.value for d in PIIDomain]
        for domain_key in targets:
            self._load_domain(domain_key)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_loaded(self, domain: PIIDomain) -> None:
        if domain.value not in self._indices:
            self._load_domain(domain.value)

    def _load_domain(self, domain_key: str) -> None:
        """Load the per-domain Parquet index from storage."""
        path = self._get_domain_path(domain_key)
        rows = self._read_parquet_as_rows(path)

        if rows:
            idx_map = {h: int(i) for h, i in rows}
            next_id = max(idx_map.values()) + 1
            logger.info(
                "Loaded %d indices for domain '%s' (next_id=%d)",
                len(idx_map), domain_key, next_id,
            )
        else:
            idx_map = {}
            # start from 1 (0 reserved, -1 is the null sentinel)
            next_id = 1

        self._indices[domain_key] = idx_map
        self._next_id[domain_key] = next_id
        self._dirty[domain_key] = False

    def _save_domain(self, domain_key: str) -> None:
        """Persist the in-memory index as a Parquet file."""
        idx_map = self._indices.get(domain_key, {})
        if not idx_map:
            return
        path = self._get_domain_path(domain_key)
        self._write_parquet_rows(path, idx_map)
        logger.info(
            "Saved %d indices for domain '%s' to %s",
            len(idx_map), domain_key, path,
        )

    def _get_domain_path(self, domain_key: str) -> str:
        return f"{self._store_path.rstrip('/')}/{domain_key}_index.parquet"

    # ------------------------------------------------------------------
    # Parquet I/O (DuckDB for read, pyarrow for write)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_s3(path: str) -> bool:
        return path.startswith("s3://")

    def _read_parquet_as_rows(self, path: str) -> List[tuple]:
        """Return the index Parquet as a list of ``(hash_hex, integer_id)``.

        Non-existent / unreadable files return an empty list so first-
        time domains bootstrap cleanly.
        """
        try:
            import duckdb
        except ImportError:  # pragma: no cover — DuckDB is a core dep
            duckdb = None

        if duckdb is not None:
            try:
                con = duckdb.connect(database=":memory:")
                if self._is_s3(path):
                    con.execute("INSTALL httpfs; LOAD httpfs;")
                rows = con.execute(
                    f"SELECT hash_hex, integer_id FROM '{path}'",
                ).fetchall()
                con.close()
                return rows
            except Exception:
                # Fall through to pyarrow/S3 fallback below
                pass

        return self._read_parquet_pyarrow(path)

    def _read_parquet_pyarrow(self, path: str) -> List[tuple]:
        try:
            if self._is_s3(path):
                import io
                import boto3
                s3 = boto3.client("s3")
                bucket, key = path.replace("s3://", "").split("/", 1)
                try:
                    obj = s3.get_object(Bucket=bucket, Key=key)
                except s3.exceptions.ClientError:
                    return []
                data = io.BytesIO(obj["Body"].read())
            else:
                if not os.path.exists(path):
                    return []
                data = path  # type: ignore[assignment]

            import pyarrow.parquet as pq
            table = pq.read_table(data)
            cols = table.column_names
            if "hash_hex" not in cols or "integer_id" not in cols:
                return []
            hashes = table["hash_hex"].to_pylist()
            ids = table["integer_id"].to_pylist()
            return list(zip(hashes, ids))
        except Exception:
            logger.exception("Failed to read index parquet at %s", path)
            return []

    def _write_parquet_rows(
        self, path: str, idx_map: Dict[str, int],
    ) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pydict({
            "hash_hex": pa.array(list(idx_map.keys()), type=pa.string()),
            "integer_id": pa.array(
                list(idx_map.values()), type=pa.int32(),
            ),
        })

        if self._is_s3(path):
            import io
            import boto3
            buf = io.BytesIO()
            pq.write_table(table, buf)
            buf.seek(0)
            s3 = boto3.client("s3")
            bucket, key = path.replace("s3://", "").split("/", 1)
            s3.upload_fileobj(buf, bucket, key)
        else:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            pq.write_table(table, path)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, int]:
        """Return per-domain unique-index counts."""
        return {k: len(v) for k, v in self._indices.items() if v}
