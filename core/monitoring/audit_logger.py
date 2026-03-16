"""
Immutable Audit Logger with HMAC-SHA256 signing and hash-chain integrity.

Each audit entry contains:
- timestamp, operation, input_hash, output_hash, metadata
- HMAC-SHA256 signature (key from AWS SSM Parameter Store)
- SHA256 prev_hash linking consecutive entries (hash chain)

Storage backend:
- S3 with Object Lock (WORM) for immutability
- Daily JSONL files partitioned by YYYYMM/audit_YYYYMMDD.jsonl

Integrity verification:
- verify_chain() validates the full hash chain for a given log file
"""

from __future__ import annotations

import hashlib
import hmac as hmac_module
import json
import logging
import os
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HMAC secret management
# ---------------------------------------------------------------------------

_HMAC_SECRET_KEY: Optional[bytes] = None


def _get_hmac_secret() -> bytes:
    """Retrieve the HMAC signing key, preferring AWS SSM Parameter Store.

    Lookup order:
        1. AWS SSM Parameter Store  (parameter name from ``AUDIT_HMAC_SSM_PARAM``)
        2. Environment variable     (``AUDIT_HMAC_SECRET_KEY``)
        3. Hardcoded default        (development only -- logs a warning)

    Returns:
        The raw bytes of the HMAC key.
    """
    global _HMAC_SECRET_KEY
    if _HMAC_SECRET_KEY is not None:
        return _HMAC_SECRET_KEY

    ssm_param_name = os.environ.get("AUDIT_HMAC_SSM_PARAM")
    if ssm_param_name:
        try:
            import boto3

            region = os.environ.get("AWS_DEFAULT_REGION", "ap-northeast-2")
            ssm = boto3.client("ssm", region_name=region)
            response = ssm.get_parameter(Name=ssm_param_name, WithDecryption=True)
            _HMAC_SECRET_KEY = response["Parameter"]["Value"].encode("utf-8")
            logger.info("HMAC secret loaded from SSM Parameter Store: %s", ssm_param_name)
            return _HMAC_SECRET_KEY
        except Exception as exc:
            logger.warning("Failed to load HMAC secret from SSM (%s): %s", ssm_param_name, exc)

    env_key = os.environ.get("AUDIT_HMAC_SECRET_KEY")
    if env_key:
        _HMAC_SECRET_KEY = env_key.encode("utf-8")
        logger.info("HMAC secret loaded from environment variable.")
        return _HMAC_SECRET_KEY

    # Fallback -- development only
    _env = os.environ.get("ENVIRONMENT", os.environ.get("ENV", "development"))
    if _env.lower() in ("production", "prod", "staging"):
        logger.critical(
            "SECURITY WARNING: No HMAC secret configured in %s environment. "
            "Set AUDIT_HMAC_SSM_PARAM (preferred) or AUDIT_HMAC_SECRET_KEY.",
            _env,
        )
    else:
        logger.warning(
            "AUDIT_HMAC_SECRET_KEY not set -- using default key. "
            "Acceptable for development; must be configured in production."
        )
    _HMAC_SECRET_KEY = b"aws-ple-audit-default-key-CHANGE-ME"
    return _HMAC_SECRET_KEY


# ---------------------------------------------------------------------------
# Crypto helpers
# ---------------------------------------------------------------------------

def _compute_hmac(data_bytes: bytes) -> str:
    """Return HMAC-SHA256 hex digest for *data_bytes*."""
    return hmac_module.new(_get_hmac_secret(), data_bytes, hashlib.sha256).hexdigest()


def _compute_chain_hash(data_str: str) -> str:
    """Return SHA256 hex digest used for hash-chain linking."""
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# AuditLogger
# ---------------------------------------------------------------------------

class AuditLogger:
    """Append-only audit logger with HMAC signing and hash-chain integrity.

    Logs are written as JSONL to S3 (with Object Lock for WORM compliance)
    or to the local filesystem when S3 is unavailable / during development.

    Parameters
    ----------
    s3_bucket : str, optional
        S3 bucket name for audit log storage.  Falls back to the
        ``AUDIT_S3_BUCKET`` environment variable.
    s3_prefix : str
        Key prefix inside the bucket (default ``"audit_logs"``).
    local_fallback_dir : str, optional
        Local directory used when S3 is unavailable.
    object_lock_mode : str
        S3 Object Lock retention mode (``"GOVERNANCE"`` or ``"COMPLIANCE"``).
    object_lock_days : int
        Number of days to retain the object under Object Lock (default 2555 = ~7 years).
    """

    def __init__(
        self,
        s3_bucket: Optional[str] = None,
        s3_prefix: str = "audit_logs",
        local_fallback_dir: Optional[str] = None,
        object_lock_mode: str = "GOVERNANCE",
        object_lock_days: int = 2555,
    ) -> None:
        self.s3_bucket = s3_bucket or os.environ.get("AUDIT_S3_BUCKET", "")
        self.s3_prefix = s3_prefix.strip("/")
        self.object_lock_mode = object_lock_mode
        self.object_lock_days = object_lock_days
        self._local_dir = local_fallback_dir or os.environ.get(
            "AUDIT_LOG_DIR", "/tmp/audit_logs"
        )

        self._s3_client = None
        if self.s3_bucket:
            try:
                import boto3

                region = os.environ.get("AWS_DEFAULT_REGION", "ap-northeast-2")
                self._s3_client = boto3.client("s3", region_name=region)
            except Exception as exc:
                logger.warning("boto3 S3 client init failed; will use local fallback: %s", exc)

        # Hash chain state
        self._prev_hash: str = "GENESIS"

    # ------------------------------------------------------------------
    # Public logging methods
    # ------------------------------------------------------------------

    def log_operation(
        self,
        operation: str,
        input_data: Any = None,
        output_data: Any = None,
        user: str = "system",
        status: str = "SUCCESS",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Record an audit entry with HMAC signature and hash-chain link.

        Parameters
        ----------
        operation : str
            Name of the operation (e.g. ``"model_inference:cvr_v2"``).
        input_data : Any, optional
            Input payload (hashed, not stored raw).
        output_data : Any, optional
            Output payload (hashed, not stored raw).
        user : str
            Identity of the actor.
        status : str
            Outcome (``"SUCCESS"``, ``"FAILURE"``, etc.).
        metadata : dict, optional
            Arbitrary structured metadata attached to the entry.

        Returns
        -------
        dict or None
            The complete log entry dict, or ``None`` on failure.
        """
        try:
            now = datetime.now(timezone.utc)
            timestamp = now.isoformat()
            yyyymm = now.strftime("%Y%m")
            yyyymmdd = now.strftime("%Y%m%d")

            input_hash = self._compute_data_hash(input_data) if input_data is not None else None
            output_hash = self._compute_data_hash(output_data) if output_data is not None else None

            log_entry: Dict[str, Any] = {
                "timestamp": timestamp,
                "operation": operation,
                "input_hash": input_hash,
                "output_hash": output_hash,
                "user": user,
                "status": status,
                "metadata": metadata or {},
                "prev_hash": self._prev_hash,
            }

            # HMAC signature (computed before adding the hmac field itself)
            entry_bytes = json.dumps(log_entry, ensure_ascii=False, sort_keys=True).encode("utf-8")
            log_entry["hmac"] = _compute_hmac(entry_bytes)

            # Update hash chain
            full_entry_str = json.dumps(log_entry, ensure_ascii=False, sort_keys=True)
            self._prev_hash = _compute_chain_hash(full_entry_str)

            # Persist
            log_line = json.dumps(log_entry, ensure_ascii=False) + "\n"
            s3_key = f"{self.s3_prefix}/{yyyymm}/audit_{yyyymmdd}.jsonl"
            self._write_log_line(log_line, s3_key, yyyymm, yyyymmdd)

            logger.info("Audit log recorded: %s (%s)", operation, status)
            return log_entry
        except Exception as exc:
            logger.warning("Failed to record audit log: %s", exc)
            return None

    def log_model_inference(
        self,
        model_id: str,
        input_dim: int,
        output_dim: int,
        latency_ms: float,
        status: str = "SUCCESS",
        user: str = "system",
    ) -> Optional[Dict[str, Any]]:
        """Log a model inference event."""
        return self.log_operation(
            operation=f"model_inference:{model_id}",
            user=user,
            status=status,
            metadata={
                "model_id": model_id,
                "input_dim": input_dim,
                "output_dim": output_dim,
                "latency_ms": latency_ms,
                "operation_type": "inference",
            },
        )

    def log_data_access(
        self,
        table: str,
        columns: List[str],
        user: str,
        access_type: str = "READ",
        row_count: int = 0,
    ) -> Optional[Dict[str, Any]]:
        """Log a data access event."""
        return self.log_operation(
            operation=f"data_access:{table}",
            user=user,
            status="SUCCESS",
            metadata={
                "table": table,
                "columns": columns,
                "access_type": access_type,
                "row_count": row_count,
                "operation_type": "data_access",
            },
        )

    def log_dimension_change(
        self,
        old_dim: int,
        new_dim: int,
        component: str,
        reason: str,
        user: str = "system",
    ) -> Optional[Dict[str, Any]]:
        """Log a feature-dimension change event."""
        return self.log_operation(
            operation="dimension_change",
            metadata={
                "component": component,
                "reason": reason,
                "old_dim": old_dim,
                "new_dim": new_dim,
            },
            user=user,
            status="SUCCESS",
        )

    # ------------------------------------------------------------------
    # Chain verification
    # ------------------------------------------------------------------

    def verify_chain(self, log_lines: List[str]) -> bool:
        """Verify the integrity of a sequence of JSONL audit entries.

        Parameters
        ----------
        log_lines : list of str
            Raw JSONL lines (each a JSON object).

        Returns
        -------
        bool
            ``True`` if the hash chain is intact, ``False`` if tampered.
        """
        prev_hash = "GENESIS"
        for i, line in enumerate(log_lines):
            entry = json.loads(line.strip())
            if entry.get("prev_hash") != prev_hash:
                logger.warning(
                    "Hash chain broken at line %d: expected prev_hash=%s, got %s",
                    i + 1,
                    prev_hash,
                    entry.get("prev_hash"),
                )
                return False
            full_entry_str = json.dumps(entry, ensure_ascii=False, sort_keys=True)
            prev_hash = _compute_chain_hash(full_entry_str)
        return True

    def verify_chain_from_s3(self, s3_key: str) -> bool:
        """Download a JSONL file from S3 and verify its hash chain.

        Parameters
        ----------
        s3_key : str
            The S3 object key for the audit log file.

        Returns
        -------
        bool
            ``True`` if the chain is valid.
        """
        if not self._s3_client or not self.s3_bucket:
            logger.warning("S3 not configured; cannot verify chain from S3.")
            return False
        try:
            resp = self._s3_client.get_object(Bucket=self.s3_bucket, Key=s3_key)
            body = resp["Body"].read().decode("utf-8")
            lines = [ln for ln in body.strip().split("\n") if ln.strip()]
            return self.verify_chain(lines)
        except Exception as exc:
            logger.warning("Chain verification from S3 failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_log_line(
        self,
        log_line: str,
        s3_key: str,
        yyyymm: str,
        yyyymmdd: str,
    ) -> None:
        """Append a log line to S3 (with Object Lock) or local fallback."""
        if self._s3_client and self.s3_bucket:
            try:
                self._append_to_s3(log_line, s3_key)
                return
            except Exception as exc:
                logger.warning("S3 write failed, falling back to local: %s", exc)

        # Local fallback
        from pathlib import Path

        log_dir = Path(self._local_dir) / yyyymm
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"audit_{yyyymmdd}.jsonl"
        with open(log_file, "a", encoding="utf-8") as fh:
            fh.write(log_line)

    def _append_to_s3(self, log_line: str, s3_key: str) -> None:
        """Append *log_line* to an existing S3 object, or create a new one.

        When S3 Object Lock is enabled the bucket must have versioning and
        Object Lock configured at bucket creation time.  Each new object
        version receives the configured retention.
        """
        existing_body = b""
        try:
            resp = self._s3_client.get_object(Bucket=self.s3_bucket, Key=s3_key)
            existing_body = resp["Body"].read()
        except self._s3_client.exceptions.NoSuchKey:
            pass
        except Exception:
            # Object may not exist yet
            pass

        new_body = existing_body + log_line.encode("utf-8")
        put_kwargs: Dict[str, Any] = {
            "Bucket": self.s3_bucket,
            "Key": s3_key,
            "Body": new_body,
            "ContentType": "application/x-ndjson",
        }

        if self.object_lock_mode:
            from datetime import timedelta

            retain_until = datetime.now(timezone.utc) + timedelta(days=self.object_lock_days)
            put_kwargs["ObjectLockMode"] = self.object_lock_mode
            put_kwargs["ObjectLockRetainUntilDate"] = retain_until

        self._s3_client.put_object(**put_kwargs)

    @staticmethod
    def _compute_data_hash(data: Any) -> Optional[str]:
        """Compute a truncated SHA256 hash of arbitrary data.

        Supports: ``numpy.ndarray``, ``bytes``, ``str``, ``dict``, and
        pandas/PyArrow DataFrames.
        """
        try:
            if isinstance(data, np.ndarray):
                return hashlib.sha256(data.tobytes()).hexdigest()[:16]
            if isinstance(data, bytes):
                return hashlib.sha256(data).hexdigest()[:16]
            if isinstance(data, str):
                return hashlib.sha256(data.encode()).hexdigest()[:16]
            if isinstance(data, dict):
                dict_bytes = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
                return hashlib.sha256(dict_bytes).hexdigest()[:16]
            # pandas DataFrame
            if hasattr(data, "to_parquet"):
                buf = BytesIO()
                data.to_parquet(buf, index=False)
                return hashlib.sha256(buf.getvalue()).hexdigest()[:16]
        except Exception as exc:
            logger.warning("Hash computation failed: %s", exc)
        return None


__all__ = ["AuditLogger"]
