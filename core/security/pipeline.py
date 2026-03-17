"""
Encryption Pipeline -- orchestrates PII detection -> hashing -> integer indexing.

Full flow for a data source:
  1. Load encryption policies (from schema or explicit config)
  2. Validate PII columns exist in DataFrame
  3. Hash PII columns with domain-specific salts (SHA256)
  4. Convert hashes to INT32 global indices
  5. Drop raw PII columns (phone, email, SSN)
  6. Save updated index tables
  7. Return clean DataFrame + audit report

Usage:
    salt_mgr = LocalSaltManager()
    indexer = PIIIntegerIndexer("s3://bucket/pii-indices/")
    policies = derive_from_schema(schema_registry)

    pipeline = EncryptionPipeline(salt_mgr, indexer, policies)
    clean_df = pipeline.process_source("customer_master", raw_df)
    pipeline.save_indices()
    report = pipeline.get_audit_report()
"""

import logging
import time
from typing import Any, Dict, List, Optional

import pandas as pd

from .domains import PIIDomain
from .salt_manager import SaltManager
from .encryptor import PIIEncryptor
from .integer_indexer import PIIIntegerIndexer
from .encryption_policy import SourceEncryptionPolicy

logger = logging.getLogger(__name__)


class EncryptionPipeline:
    """Top-level orchestrator that ties PII hashing and integer indexing together.

    Parameters
    ----------
    salt_manager : SaltManager
        Provides domain-specific salts (AWS Secrets Manager or local).
    indexer : PIIIntegerIndexer
        Maps hashed values to stable INT32 indices.
    policies : dict[str, SourceEncryptionPolicy]
        Per-source encryption policies (which columns to hash/drop/index).
    validator : DataValidator, optional
        If provided, run data-quality checks before encryption.
    """

    def __init__(
        self,
        salt_manager: SaltManager,
        indexer: PIIIntegerIndexer,
        policies: Dict[str, SourceEncryptionPolicy],
        validator=None,  # Optional DataValidator
        audit_store=None,
    ):
        self._encryptor = PIIEncryptor(salt_manager)
        self._indexer = indexer
        self._policies = policies
        self._validator = validator
        self._audit: List[Dict[str, Any]] = []
        self._audit_store = audit_store

    # ── Main entry point ──────────────────────────────────────────────

    def process_source(self, source_name: str, df: pd.DataFrame) -> pd.DataFrame:
        """Full encryption pipeline for one data source.

        Returns a clean DataFrame with:
        - PII columns hashed -> integer indexed -> '{col}_idx' columns
        - Contact/PersonalID columns dropped entirely
        - Original PII columns removed
        """
        start = time.time()
        policy = self._policies.get(source_name)

        if policy is None:
            logger.info(
                "No encryption policy for source '%s', passing through",
                source_name,
            )
            return df

        n_rows = len(df)
        result = df.copy()

        # Step 1: Drop columns marked for deletion (phone, email, SSN)
        drop_cols = [
            c for c in policy.get_columns_to_drop() if c in result.columns
        ]
        if drop_cols:
            result = result.drop(columns=drop_cols)
            logger.info(
                "Dropped %d PII columns from '%s': %s",
                len(drop_cols),
                source_name,
                drop_cols,
            )

        # Step 2: Hash PII columns
        col_domain_map = {
            c: d
            for c, d in policy.get_column_domain_map().items()
            if c in result.columns
        }
        if col_domain_map:
            result = self._encryptor.hash_dataframe(result, col_domain_map)

        # Step 3: Integer index hashed columns
        index_cols = [
            c
            for c in policy.get_columns_to_index()
            if f"{c}_hashed" in result.columns
        ]
        if index_cols:
            index_map = {c: policy.policies[c].domain for c in index_cols}
            result = self._indexer.index_dataframe(result, index_map)

        elapsed = time.time() - start

        # Audit record
        audit = {
            "source": source_name,
            "rows": n_rows,
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
            source_name,
            n_rows,
            len(drop_cols),
            len(col_domain_map),
            len(index_cols),
            elapsed,
        )
        return result

    # ── S3 convenience ────────────────────────────────────────────────

    def process_s3_source(
        self, source_name: str, input_s3: str, output_s3: str
    ) -> Dict:
        """S3-based: read Parquet, process, write Parquet."""
        import io

        try:
            import boto3
        except ImportError:
            raise ImportError("boto3 required for S3 operations")

        # Read
        s3 = boto3.client("s3")
        parts = input_s3.replace("s3://", "").split("/", 1)
        bucket, key = parts[0], parts[1]
        obj = s3.get_object(Bucket=bucket, Key=key)
        df = pd.read_parquet(io.BytesIO(obj["Body"].read()))

        # Process
        clean_df = self.process_source(source_name, df)

        # Write
        buf = io.BytesIO()
        clean_df.to_parquet(buf, index=False)
        buf.seek(0)
        out_parts = output_s3.replace("s3://", "").split("/", 1)
        s3.upload_fileobj(buf, out_parts[0], out_parts[1])

        return {
            "source": source_name,
            "input": input_s3,
            "output": output_s3,
            "rows": len(clean_df),
        }

    # ── Index persistence ─────────────────────────────────────────────

    def save_indices(self) -> None:
        """Persist updated index tables."""
        self._indexer.save_indices()

    # ── Audit & stats ─────────────────────────────────────────────────

    def get_audit_report(self) -> List[Dict[str, Any]]:
        """Return audit trail of all processed sources."""
        return list(self._audit)

    def get_index_stats(self) -> Dict[str, int]:
        """Return unique index count per domain."""
        return self._indexer.get_stats()
