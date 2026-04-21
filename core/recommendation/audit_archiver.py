"""
Recommendation Audit Archiver -- Parquet-based audit trail.

Stores full recommendation trace per customer per batch:
  - customer_id (encrypted integer index)
  - product_id, rank, score, score_components
  - reason_text, reason_layer (L1/L2a/L2b), reason_confidence
  - self_check_verdict, hallucination_flags
  - feature_importances (top-5)
  - task_name, segment_id, timestamp

Storage: S3 Parquet with daily partitions.
  s3://bucket/audit/recommendations/YYYY-MM-DD/batch_HHMMSS.parquet
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = ["RecommendationAuditArchiver", "RecommendationAuditRecord"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RecommendationAuditRecord:
    """A single recommendation audit trace record.

    Attributes
    ----------
    customer_id : int
        Encrypted integer index for the customer.
    product_id : str
        Recommended product identifier.
    task_name : str
        Prediction task name (e.g. ``"ctr"``, ``"cvr"``).
    rank : int
        1-based rank in the recommendation list.
    score : float
        Final priority score.
    score_components : dict
        Named scoring sub-components.
    reason_text : str
        Generated recommendation reason text.
    reason_layer : str
        Reason generation layer (``"L1"`` / ``"L2a"`` / ``"L2b"``).
    reason_confidence : float
        Confidence of the generated reason (0-1).
    self_check_verdict : str
        Self-checker verdict (``"pass"`` / ``"revise"`` / ``"reject"``).
    feature_importances : list of dict
        Top-5 feature importances, each ``{name, value, importance}``.
    segment_id : int
        Customer segment identifier (-1 if not assigned).
    timestamp : str
        ISO 8601 timestamp of the recommendation.
    batch_id : str
        Batch identifier for grouping records.
    """

    customer_id: int
    product_id: str
    task_name: str
    rank: int
    score: float
    score_components: Dict[str, float] = field(default_factory=dict)
    reason_text: str = ""
    reason_layer: str = "L1"
    reason_confidence: float = 0.0
    self_check_verdict: str = "pass"
    feature_importances: List[Dict[str, Any]] = field(default_factory=list)
    segment_id: int = -1
    timestamp: str = ""
    batch_id: str = ""
    # Sprint 3 M11 extensions (agentic + compliance trail)
    # All columns are nullable-by-default so existing Parquet readers keep
    # working on older rows.
    thinking_trace: str = ""                 # Agent reasoning chain (L1/L2a)
    hallucination_flags: List[str] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    critique_verdict: str = ""               # SelfChecker final call
    agent_tier: int = 0                      # 1 / 2 / 3 tier (0 = n/a)


# ---------------------------------------------------------------------------
# RecommendationAuditArchiver
# ---------------------------------------------------------------------------

class RecommendationAuditArchiver:
    """Parquet-based audit archiver for recommendation traces.

    Buffers records in memory and flushes to Parquet files on demand.
    Supports both local and S3 storage with daily partitions.

    Parameters
    ----------
    s3_base_path : str
        S3 base path for Parquet uploads (e.g.
        ``"s3://bucket/audit/recommendations"``). Empty disables S3.
    local_path : str
        Local directory for writing Parquet files.
    """

    def __init__(
        self,
        s3_base_path: str = "",
        local_path: str = "audit/recommendations",
    ) -> None:
        self._s3_base = s3_base_path.rstrip("/")
        self._local_path = local_path
        self._buffer: List[Dict[str, Any]] = []
        self._batch_id: str = ""

        self._s3_client = None
        if self._s3_base:
            try:
                import boto3

                region = os.environ.get("AWS_DEFAULT_REGION")
                self._s3_client = boto3.client("s3", region_name=region)
            except Exception as exc:
                logger.warning("S3 client init failed (audit archiver): %s", exc)

    # ------------------------------------------------------------------
    # Batch management
    # ------------------------------------------------------------------

    def start_batch(self, batch_id: str = "") -> None:
        """Start a new audit batch.

        Flushes any existing buffer before starting the new batch.

        Parameters
        ----------
        batch_id : str, optional
            Batch identifier. Auto-generated if empty.
        """
        if self._buffer:
            logger.warning(
                "Starting new batch with %d unflushed records; flushing first.",
                len(self._buffer),
            )
            self.flush()

        self._batch_id = batch_id or f"batch_{datetime.now(timezone.utc).strftime('%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self._buffer = []
        logger.info("Audit batch started: %s", self._batch_id)

    # ------------------------------------------------------------------
    # Record ingestion
    # ------------------------------------------------------------------

    def record(self, record: RecommendationAuditRecord) -> None:
        """Add a recommendation audit record to the current batch.

        Parameters
        ----------
        record : RecommendationAuditRecord
            The record to add.
        """
        if not record.timestamp:
            record.timestamp = datetime.now(timezone.utc).isoformat()
        if not record.batch_id:
            record.batch_id = self._batch_id

        row = asdict(record)
        # Serialize nested structures to JSON strings for Parquet compatibility
        row["score_components"] = json.dumps(row["score_components"], default=str)
        row["feature_importances"] = json.dumps(row["feature_importances"], default=str)
        # Sprint 3 M11 extensions — list columns flattened to JSON strings
        row["hallucination_flags"] = json.dumps(
            row.get("hallucination_flags", []), default=str
        )
        row["tools_used"] = json.dumps(
            row.get("tools_used", []), default=str
        )

        self._buffer.append(row)

    def record_from_result(
        self,
        result: Any,
        reason_result: Optional[Any] = None,
        check_result: Optional[Any] = None,
        feature_importances: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Build and add a record from pipeline output objects.

        Parameters
        ----------
        result : RecommendationItem
            A recommendation item from the pipeline.
        reason_result : object, optional
            Reason generation result with ``reason_text``, ``layer``,
            ``confidence`` attributes.
        check_result : CheckResult, optional
            Self-checker result with ``verdict`` attribute.
        feature_importances : list of dict, optional
            Top-5 feature importances.
        """
        # Extract reason fields
        reason_text = ""
        reason_layer = "L1"
        reason_confidence = 0.0

        if reason_result is not None:
            reason_text = getattr(reason_result, "reason_text", "")
            if not reason_text and hasattr(reason_result, "text"):
                reason_text = reason_result.text
            reason_layer = getattr(reason_result, "layer", "L1")
            reason_confidence = getattr(reason_result, "confidence", 0.0)
        elif hasattr(result, "reasons") and result.reasons:
            # Fall back to the first reason in the recommendation item
            first_reason = result.reasons[0] if result.reasons else {}
            reason_text = first_reason.get("text", "")
            reason_layer = first_reason.get("layer", "L1")
            reason_confidence = first_reason.get("confidence", 0.0)

        # Extract self-check verdict
        verdict = "pass"
        if check_result is not None:
            verdict = getattr(check_result, "verdict", "pass")
        elif hasattr(result, "check_result") and result.check_result is not None:
            verdict = getattr(result.check_result, "verdict", "pass")

        # Build customer_id as int
        cid = getattr(result, "customer_id", 0)
        if isinstance(cid, str):
            try:
                cid = int(cid)
            except (ValueError, TypeError):
                cid = hash(cid) & 0x7FFFFFFF  # positive 32-bit int

        audit_record = RecommendationAuditRecord(
            customer_id=cid,
            product_id=getattr(result, "item_id", ""),
            task_name=getattr(result, "metadata", {}).get("task_name", ""),
            rank=getattr(result, "rank", 0),
            score=getattr(result, "score", 0.0),
            score_components=getattr(result, "score_components", {}),
            reason_text=reason_text,
            reason_layer=reason_layer,
            reason_confidence=reason_confidence,
            self_check_verdict=verdict,
            feature_importances=feature_importances or [],
            segment_id=getattr(result, "metadata", {}).get("segment_id", -1),
        )
        self.record(audit_record)

    # ------------------------------------------------------------------
    # Flush / write
    # ------------------------------------------------------------------

    def flush(self) -> str:
        """Flush the buffer to a Parquet file.

        Writes to the local path and optionally uploads to S3.

        Returns
        -------
        str
            Path to the written Parquet file (local or S3).
        """
        if not self._buffer:
            logger.info("Nothing to flush; buffer is empty.")
            return ""

        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y-%m-%d")
        batch_id = self._batch_id or f"batch_{now.strftime('%H%M%S')}"

        # Build local path with daily partition
        local_dir = Path(self._local_path) / date_str
        local_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"{batch_id}.parquet"
        local_file = local_dir / file_name

        self._write_parquet(self._buffer, str(local_file))
        output_path = str(local_file)

        # Upload to S3 if configured
        if self._s3_base and self._s3_client:
            s3_key = f"{date_str}/{file_name}"
            self._upload_to_s3(str(local_file), s3_key)
            output_path = f"{self._s3_base}/{s3_key}"

        record_count = len(self._buffer)
        self._buffer = []
        logger.info(
            "Flushed %d records to %s", record_count, output_path,
        )
        return output_path

    def get_buffer_size(self) -> int:
        """Return the number of records in the current buffer."""
        return len(self._buffer)

    # ------------------------------------------------------------------
    # Parquet I/O
    # ------------------------------------------------------------------

    def _write_parquet(self, records: List[Dict[str, Any]], path: str) -> None:
        """Write records to a Parquet file.

        Uses PyArrow if available; falls back to JSON if not.

        Parameters
        ----------
        records : list of dict
            Flattened record dicts.
        path : str
            Output file path.
        """
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.Table.from_pylist(records)
            pq.write_table(table, path, compression="snappy")
            logger.debug("Wrote Parquet file: %s (%d rows)", path, len(records))
        except ImportError:
            # Fallback: write as JSON with .parquet extension (for environments
            # without PyArrow).  The file is still structured for later
            # conversion.
            logger.warning(
                "PyArrow not available; writing JSON fallback to %s", path,
            )
            fallback_path = path.replace(".parquet", ".json")
            with open(fallback_path, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2, default=str)

    def _upload_to_s3(self, local_path: str, s3_key: str) -> None:
        """Upload a local file to S3.

        Parameters
        ----------
        local_path : str
            Path to the local file.
        s3_key : str
            Relative key under the S3 base path.
        """
        if not self._s3_client:
            return

        # Parse bucket from s3_base if it includes s3:// prefix
        s3_base = self._s3_base
        if s3_base.startswith("s3://"):
            parts = s3_base[5:].split("/", 1)
            bucket = parts[0]
            prefix = parts[1] if len(parts) > 1 else ""
        else:
            bucket = os.environ.get("AUDIT_S3_BUCKET", "")
            prefix = s3_base

        full_key = f"{prefix}/{s3_key}" if prefix else s3_key

        try:
            with open(local_path, "rb") as f:
                self._s3_client.put_object(
                    Bucket=bucket,
                    Key=full_key,
                    Body=f.read(),
                    ContentType="application/octet-stream",
                )
            logger.debug("Uploaded to s3://%s/%s", bucket, full_key)
        except Exception as exc:
            logger.warning("Failed to upload %s to S3: %s", local_path, exc)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query_customer_audit(
        self,
        customer_id: int,
        date_range: Optional[Tuple[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Query audit records for a specific customer.

        Reads from local Parquet files (or JSON fallbacks).
        For production S3-based queries, use Athena or Spark.

        Parameters
        ----------
        customer_id : int
            Encrypted customer integer index.
        date_range : tuple of (start_date, end_date), optional
            Date strings in ``"YYYY-MM-DD"`` format to filter partitions.

        Returns
        -------
        list of dict
            Matching audit records.
        """
        results: List[Dict[str, Any]] = []
        base = Path(self._local_path)

        if not base.exists():
            logger.warning("Local audit path does not exist: %s", self._local_path)
            return results

        # Iterate over date partitions
        for date_dir in sorted(base.iterdir()):
            if not date_dir.is_dir():
                continue

            dir_name = date_dir.name  # "YYYY-MM-DD"
            if date_range:
                start_date, end_date = date_range
                if dir_name < start_date or dir_name > end_date:
                    continue

            # Read Parquet files in this partition
            for file_path in date_dir.iterdir():
                if not file_path.is_file():
                    continue

                records = self._read_file(str(file_path))
                for rec in records:
                    if rec.get("customer_id") == customer_id:
                        results.append(rec)

        logger.info(
            "Query returned %d records for customer_id=%d", len(results), customer_id,
        )
        return results

    @staticmethod
    def _read_file(path: str) -> List[Dict[str, Any]]:
        """Read records from a Parquet or JSON file.

        Parameters
        ----------
        path : str
            File path.

        Returns
        -------
        list of dict
        """
        if path.endswith(".parquet"):
            try:
                import pyarrow.parquet as pq

                table = pq.read_table(path)
                return table.to_pylist()
            except ImportError:
                logger.warning("PyArrow not available; cannot read %s", path)
                return []
            except Exception as exc:
                logger.warning("Failed to read Parquet file %s: %s", path, exc)
                return []
        elif path.endswith(".json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data if isinstance(data, list) else [data]
            except Exception as exc:
                logger.warning("Failed to read JSON file %s: %s", path, exc)
                return []
        return []
