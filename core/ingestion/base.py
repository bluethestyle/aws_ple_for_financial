"""
Domain Ingestion Base -- abstract base for all domain data ingestors.

Each domain (customer_master, account, card, transaction, etc.) implements
a concrete subclass that reads S3 Parquet, applies domain-specific transforms,
validates against SchemaRegistry, and optionally encrypts PII.

Pattern mirrors AbstractFeatureGenerator / FeatureGeneratorRegistry.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ======================================================================
# Result data class
# ======================================================================


@dataclass
class IngestionResult:
    """Outcome of a single domain ingestion run."""

    source_name: str
    row_count: int
    column_count: int
    output_path: str
    duration_seconds: float
    pii_columns_encrypted: int = 0
    pii_columns_dropped: int = 0
    validation_passed: bool = True
    validation_warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"IngestionResult(source={self.source_name!r}, "
            f"rows={self.row_count}, cols={self.column_count}, "
            f"valid={self.validation_passed}, "
            f"duration={self.duration_seconds:.2f}s)",
        ]
        if self.pii_columns_encrypted > 0:
            lines.append(f"  PII encrypted: {self.pii_columns_encrypted}")
        if self.pii_columns_dropped > 0:
            lines.append(f"  PII dropped: {self.pii_columns_dropped}")
        if self.validation_warnings:
            lines.append(f"  Warnings: {self.validation_warnings}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover
        return self.summary()


# ======================================================================
# Abstract base
# ======================================================================


class AbstractDomainIngestor(ABC):
    """Base class for domain data ingestors.

    A domain ingestor reads raw Parquet data from S3, applies
    domain-specific transforms (joins, aggregations, type casting),
    validates against the SchemaRegistry, and optionally encrypts PII
    columns via the EncryptionPipeline.

    Subclasses must implement:
        * ``source_name``      -- schema source key (in schema.yaml).
        * ``required_columns`` -- minimum required input columns.
        * ``ingest``           -- domain-specific transform logic.

    Subclasses may override:
        * ``pii_columns``      -- columns requiring encryption.
        * ``drop_columns``     -- columns to drop after ingestion.

    Attributes
    ----------
    name : str
        Registry name (set automatically by the decorator).
    """

    name: str = "base_ingestor"

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        schema_registry: Optional[Any] = None,
        encryption_pipeline: Optional[Any] = None,
    ) -> None:
        self._config = config or {}
        self._schema_registry = schema_registry
        self._encryption_pipeline = encryption_pipeline

    # -- Abstract properties / methods ------------------------------------

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Schema source name (key in schema.yaml)."""
        ...

    @property
    @abstractmethod
    def required_columns(self) -> List[str]:
        """Minimum required input columns."""
        ...

    @abstractmethod
    def ingest(self, df: Any) -> Any:
        """Domain-specific ingestion logic.

        Read raw data, apply transforms (joins, aggregations, type
        casting), return cleaned DataFrame.  PII encryption is handled
        by :meth:`run`.

        Parameters
        ----------
        df : DataFrame
            Raw input DataFrame (pandas or backend-native type).

        Returns
        -------
        DataFrame
            Cleaned and transformed DataFrame.
        """
        ...

    # -- Optional overrides -----------------------------------------------

    @property
    def pii_columns(self) -> List[str]:
        """Columns requiring PII encryption.

        Override in subclasses to specify which columns should be
        processed by the EncryptionPipeline.  If a SchemaRegistry is
        configured, PII columns are auto-detected from the schema.
        Defaults to an empty list.
        """
        if self._schema_registry is not None:
            try:
                return self._schema_registry.pii_columns(self.source_name)
            except KeyError:
                pass
        return []

    @property
    def drop_columns(self) -> List[str]:
        """Columns to drop after ingestion (e.g. intermediate join keys).

        Override in subclasses.  Defaults to an empty list.
        """
        return []

    # -- Full pipeline ----------------------------------------------------

    def run(
        self,
        input_path: str,
        output_path: str = "",
    ) -> IngestionResult:
        """Full pipeline: read -> validate -> ingest -> encrypt -> write.

        1. Read Parquet from *input_path* (S3 or local).
        2. Validate required columns exist.
        3. Call ``self.ingest(df)`` for domain-specific transforms.
        4. Apply encryption pipeline if configured.
        5. Write output Parquet.
        6. Return :class:`IngestionResult`.

        Parameters
        ----------
        input_path : str
            Path to input Parquet (local or ``s3://`` URI).
        output_path : str, optional
            Path to write output Parquet.  When empty, output is not
            written to disk (useful for testing / in-memory pipelines).

        Returns
        -------
        IngestionResult
        """
        start = time.time()
        warnings: List[str] = []

        # Step 1: Read
        logger.info(
            "Ingesting '%s' from %s", self.source_name, input_path,
        )
        df = self._read_input(input_path)
        logger.info(
            "Read %d rows x %d columns for '%s'",
            len(df), len(df.columns), self.source_name,
        )

        # Step 2: Validate required columns
        valid, val_warnings = self._validate_required(df)
        warnings.extend(val_warnings)
        if not valid:
            elapsed = time.time() - start
            return IngestionResult(
                source_name=self.source_name,
                row_count=len(df),
                column_count=len(df.columns),
                output_path=output_path,
                duration_seconds=round(elapsed, 3),
                validation_passed=False,
                validation_warnings=warnings,
            )

        # Step 3: Domain-specific ingest
        df = self.ingest(df)

        # Step 4: Schema validation (if registry available)
        schema_valid, schema_warnings = self._validate_schema(df)
        warnings.extend(schema_warnings)

        # Step 5: Encrypt PII
        pii_encrypted = 0
        pii_dropped = 0
        if self._encryption_pipeline is not None:
            pre_cols = set(df.columns)
            df = self._encryption_pipeline.process_source(
                self.source_name, df,
            )
            post_cols = set(df.columns)
            # Count new _idx columns as encrypted
            idx_cols = [c for c in post_cols - pre_cols if c.endswith("_idx")]
            pii_encrypted = len(idx_cols)
            dropped = pre_cols - post_cols
            pii_dropped = len(dropped)

        # Step 6: Drop columns
        drop = [c for c in self.drop_columns if c in df.columns]
        if drop:
            df = df.drop(columns=drop)

        # Step 7: Write output
        if output_path:
            self._write_output(df, output_path)

        elapsed = time.time() - start
        result = IngestionResult(
            source_name=self.source_name,
            row_count=len(df),
            column_count=len(df.columns),
            output_path=output_path,
            duration_seconds=round(elapsed, 3),
            pii_columns_encrypted=pii_encrypted,
            pii_columns_dropped=pii_dropped,
            validation_passed=schema_valid,
            validation_warnings=warnings,
        )

        logger.info(
            "Ingestion complete for '%s': %d rows, %d cols (%.2fs)",
            self.source_name, result.row_count,
            result.column_count, result.duration_seconds,
        )
        return result

    # -- I/O helpers ------------------------------------------------------

    def _read_input(self, path: str) -> Any:
        """Read Parquet from S3 or local path.

        Uses ``df_backend`` for unified I/O across DuckDB / cuDF / pandas.
        Falls back to pandas if df_backend is unavailable.
        """
        try:
            from core.data.dataframe import df_backend
            return df_backend.read_parquet(path)
        except Exception:
            pass

        # Fallback: direct pandas
        import pandas as pd

        if path.startswith("s3://"):
            try:
                import boto3
                import io

                parts = path.replace("s3://", "").split("/", 1)
                bucket, key = parts[0], parts[1]
                # Handle glob patterns
                s3 = boto3.client("s3")
                if "*" in key:
                    prefix = key.split("*")[0]
                    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
                    keys = [
                        o["Key"] for o in resp.get("Contents", [])
                        if o["Key"].endswith(".parquet")
                    ]
                    dfs = []
                    for k in keys:
                        obj = s3.get_object(Bucket=bucket, Key=k)
                        dfs.append(pd.read_parquet(io.BytesIO(obj["Body"].read())))
                    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
                else:
                    obj = s3.get_object(Bucket=bucket, Key=key)
                    return pd.read_parquet(io.BytesIO(obj["Body"].read()))
            except ImportError:
                raise ImportError("boto3 required for S3 reads")
        else:
            return pd.read_parquet(path)

    def _write_output(self, df: Any, path: str) -> None:
        """Write Parquet to S3 or local path."""
        try:
            from core.data.dataframe import df_backend
            df_backend.to_parquet(df, path)
            return
        except Exception:
            pass

        # Fallback: direct pandas
        if path.startswith("s3://"):
            try:
                import boto3
                import io

                buf = io.BytesIO()
                df.to_parquet(buf, index=False)
                buf.seek(0)
                parts = path.replace("s3://", "").split("/", 1)
                s3 = boto3.client("s3")
                s3.upload_fileobj(buf, parts[0], parts[1])
            except ImportError:
                raise ImportError("boto3 required for S3 writes")
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, index=False)

    # -- Validation helpers -----------------------------------------------

    def _validate_required(self, df: Any) -> Tuple[bool, List[str]]:
        """Check that all required columns exist in the DataFrame.

        Returns
        -------
        (passed, warnings) : tuple[bool, list[str]]
        """
        df_cols = set(df.columns)
        missing = [c for c in self.required_columns if c not in df_cols]
        if missing:
            msg = (
                f"[{self.source_name}] Missing required columns: {missing}"
            )
            logger.error(msg)
            return False, [msg]
        return True, []

    def _validate_schema(self, df: Any) -> Tuple[bool, List[str]]:
        """Validate DataFrame against SchemaRegistry if available.

        Returns
        -------
        (passed, warnings) : tuple[bool, list[str]]
        """
        if self._schema_registry is None:
            return True, []

        try:
            if not self._schema_registry.has(self.source_name):
                return True, [
                    f"No schema registered for '{self.source_name}'"
                ]
            valid, errors = self._schema_registry.validate_dataframe(
                self.source_name, df,
            )
            if not valid:
                logger.warning(
                    "Schema validation failed for '%s': %s",
                    self.source_name, errors,
                )
            return valid, errors
        except Exception as exc:
            msg = f"Schema validation error for '{self.source_name}': {exc}"
            logger.warning(msg)
            return True, [msg]

    # -- Params / repr ----------------------------------------------------

    def get_params(self) -> Dict[str, Any]:
        """Return ingestor parameters as a dictionary."""
        return {
            "name": self.name,
            "source_name": self.source_name,
            "required_columns": self.required_columns,
            "pii_columns": self.pii_columns,
            "drop_columns": self.drop_columns,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{type(self).__name__}("
            f"source={self.source_name!r}, "
            f"required_cols={len(self.required_columns)})"
        )
