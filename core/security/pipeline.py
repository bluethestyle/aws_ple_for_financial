"""
Encryption Pipeline -- orchestrates PII detection -> hashing -> integer indexing.

Full flow for a data source:
  1. Load encryption policies (from schema or explicit config)
  2. Validate PII columns exist in the input payload
  3. Hash PII columns with domain-specific salts (SHA256)
  4. Convert hashes to INT32 global indices
  5. Drop raw PII columns (phone, email, SSN)
  6. Save updated index tables
  7. Return a clean columnar payload + audit report

Pandas-free (CLAUDE.md §3.3). Inputs and outputs are plain Python
``dict[column, list|np.ndarray]`` so this module works inside Lambda
(no pandas in the managed layer) and under the DuckDB-first processing
policy.

Usage::

    salt_mgr = LocalSaltManager()
    indexer = PIIIntegerIndexer("s3://bucket/pii-indices/")
    policies = derive_from_schema(schema_registry)

    pipeline = EncryptionPipeline(salt_mgr, indexer, policies)
    clean_cols = pipeline.process_source("customer_master", raw_cols)
    pipeline.save_indices()
    report = pipeline.get_audit_report()
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Sequence

from .domains import PIIDomain
from .encryption_policy import SourceEncryptionPolicy
from .encryptor import PIIEncryptor
from .integer_indexer import PIIIntegerIndexer
from .salt_manager import SaltManager

logger = logging.getLogger(__name__)


class EncryptionPipeline:
    """Top-level orchestrator that ties PII hashing and integer indexing together."""

    def __init__(
        self,
        salt_manager: SaltManager,
        indexer: PIIIntegerIndexer,
        policies: Dict[str, SourceEncryptionPolicy],
        validator=None,
        audit_store=None,
    ):
        self._encryptor = PIIEncryptor(salt_manager)
        self._indexer = indexer
        self._policies = policies
        self._validator = validator
        self._audit: List[Dict[str, Any]] = []
        self._audit_store = audit_store

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def process_source(
        self,
        source_name: str,
        columns: Dict[str, Sequence[Any]],
    ) -> Dict[str, Sequence[Any]]:
        """Full encryption pipeline for one data source.

        ``columns`` is a ``{column_name: list-or-array}`` payload. The
        return value has the same shape with:

        * Columns marked for deletion dropped outright.
        * PII columns replaced by ``{col}_idx`` INT32 arrays (hashed
          then indexed).

        Row-count is preserved; callers that need a DataFrame should
        rebuild it themselves (``{col: values}`` → their library of
        choice).
        """
        start = time.time()
        policy = self._policies.get(source_name)
        if policy is None:
            logger.info(
                "No encryption policy for source '%s', passing through",
                source_name,
            )
            return columns

        # Determine row count from the first non-empty column.
        row_count = 0
        for v in columns.values():
            try:
                row_count = len(v)  # type: ignore[arg-type]
                break
            except Exception:
                continue

        result: Dict[str, Sequence[Any]] = dict(columns)

        # Step 1 — drop raw PII columns (phone, email, SSN, ...).
        drop_cols = [
            c for c in policy.get_columns_to_drop() if c in result
        ]
        for c in drop_cols:
            del result[c]
        if drop_cols:
            logger.info(
                "Dropped %d PII columns from '%s': %s",
                len(drop_cols), source_name, drop_cols,
            )

        # Step 2 — hash the remaining PII columns.
        col_domain_map = {
            c: d
            for c, d in policy.get_column_domain_map().items()
            if c in result
        }
        if col_domain_map:
            result = dict(
                self._encryptor.hash_columns(result, col_domain_map)
            )

        # Step 3 — convert hash bytes to stable INT32 indices.
        index_cols = [
            c
            for c in policy.get_columns_to_index()
            if f"{c}_hashed" in result
        ]
        if index_cols:
            index_map = {
                c: policy.policies[c].domain for c in index_cols
            }
            result = dict(self._indexer.index_columns(result, index_map))

        elapsed = time.time() - start
        audit = {
            "source": source_name,
            "rows": row_count,
            "dropped_columns": drop_cols,
            "hashed_columns": list(col_domain_map.keys()),
            "indexed_columns": index_cols,
            "duration_seconds": round(elapsed, 2),
        }
        self._audit.append(audit)
        if self._audit_store:
            self._audit_store.log_event("encryption", {"pk": source_name, **audit})

        logger.info(
            "Encryption complete for '%s': %d rows, %d dropped, "
            "%d hashed, %d indexed (%.2fs)",
            source_name, row_count, len(drop_cols),
            len(col_domain_map), len(index_cols), elapsed,
        )
        return result

    # ------------------------------------------------------------------
    # S3 convenience — DuckDB over Parquet (CLAUDE.md §3.3)
    # ------------------------------------------------------------------

    def process_s3_source(
        self,
        source_name: str,
        input_s3: str,
        output_s3: str,
    ) -> Dict[str, Any]:
        """Read a Parquet from S3, encrypt, write Parquet back — no pandas."""
        import io
        import pyarrow as pa
        import pyarrow.parquet as pq
        import boto3

        # Read via pyarrow (object-store friendly) rather than pandas.
        s3 = boto3.client("s3")
        bucket, key = input_s3.replace("s3://", "").split("/", 1)
        obj = s3.get_object(Bucket=bucket, Key=key)
        table = pq.read_table(io.BytesIO(obj["Body"].read()))

        # Convert Table → dict-of-lists (no pandas.DataFrame intermediate).
        columns = {name: table[name].to_pylist() for name in table.column_names}

        clean = self.process_source(source_name, columns)

        out_table = pa.Table.from_pydict(
            {k: pa.array(list(v)) for k, v in clean.items()}
        )
        buf = io.BytesIO()
        pq.write_table(out_table, buf)
        buf.seek(0)
        out_bucket, out_key = output_s3.replace("s3://", "").split("/", 1)
        s3.upload_fileobj(buf, out_bucket, out_key)

        return {
            "source": source_name,
            "input": input_s3,
            "output": output_s3,
            "rows": out_table.num_rows,
        }

    # ------------------------------------------------------------------
    # Admin
    # ------------------------------------------------------------------

    def save_indices(self) -> None:
        self._indexer.save_indices()

    def get_audit_report(self) -> List[Dict[str, Any]]:
        return list(self._audit)

    def get_index_stats(self) -> Dict[str, int]:
        return self._indexer.get_stats()
