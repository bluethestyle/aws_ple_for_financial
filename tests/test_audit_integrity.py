"""Audit-logger integrity hardening (P2-③).

Covers three fail-closed properties that the design claims but the prior
implementation did not enforce:

1. verify_chain re-validates the HMAC, so an attacker who edits an entry and
   re-links the (keyless) hash chain is still detected.
2. The public default HMAC key is refused in production/staging — signing
   with it provides no tamper-evidence.
3. An S3 (WORM) write failure under fail_closed propagates instead of
   silently falling back to ephemeral local disk.

Run: pytest tests/test_audit_integrity.py -v
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

import core.monitoring.audit_logger as al_mod
from core.monitoring.audit_logger import (
    AuditIntegrityError,
    AuditLogger,
    _compute_chain_hash,
)


@pytest.fixture(autouse=True)
def _reset_hmac_cache():
    """Isolate the module-global HMAC key + env across tests."""
    saved_key = al_mod._HMAC_SECRET_KEY
    saved_env = {
        k: os.environ.get(k)
        for k in (
            "AUDIT_HMAC_SECRET_KEY", "AUDIT_HMAC_SSM_PARAM",
            "AUDIT_HMAC_ALLOW_INSECURE_DEFAULT", "ENVIRONMENT", "ENV",
            "AUDIT_S3_BUCKET", "AUDIT_FAIL_CLOSED",
        )
    }
    yield
    al_mod._HMAC_SECRET_KEY = saved_key
    for k, v in saved_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _logger(tmpdir: str) -> AuditLogger:
    os.environ.pop("AUDIT_HMAC_SSM_PARAM", None)
    os.environ["AUDIT_HMAC_SECRET_KEY"] = "pytest-integrity-key"
    al_mod._HMAC_SECRET_KEY = None  # force re-resolution with our key
    os.environ["AUDIT_LOG_DIR"] = tmpdir
    os.environ.pop("AUDIT_S3_BUCKET", None)
    return AuditLogger(s3_bucket=None, local_fallback_dir=tmpdir)


def _write_entries(logger: AuditLogger, n: int = 3) -> Path:
    for i in range(n):
        logger.log_operation(operation=f"op-{i}", input_data={"i": i})
    files = list(Path(os.environ["AUDIT_LOG_DIR"]).rglob("audit_*.jsonl"))
    assert files
    return files[0]


# ---------------------------------------------------------------------------
# 1. HMAC re-verification (the core regression)
# ---------------------------------------------------------------------------

class TestHmacReverification:
    def test_clean_chain_verifies(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = _logger(tmp)
            f = _write_entries(logger)
            lines = [ln for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
            assert logger.verify_chain(lines) is True

    def test_tamper_with_relinked_chain_still_detected(self):
        """Attacker edits entry then re-links the keyless chain — HMAC catches it.

        This is the property the chain hash alone could NOT provide: without
        HMAC re-verification, recomputing prev_hash after an edit would pass.
        """
        with tempfile.TemporaryDirectory() as tmp:
            logger = _logger(tmp)
            f = _write_entries(logger, n=3)
            entries = [json.loads(ln) for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]

            # Tamper the middle entry's content (hmac left stale).
            entries[1]["operation"] = "op-FORGED"

            # Re-link the chain from the tampered entry onward so prev_hash
            # linkage is internally consistent (what a naive attacker does).
            prev = "GENESIS"
            for e in entries:
                e["prev_hash"] = prev
                prev = _compute_chain_hash(
                    json.dumps(e, ensure_ascii=False, sort_keys=True)
                )

            lines = [json.dumps(e, ensure_ascii=False) for e in entries]
            # Chain linkage is intact, but entry[1]'s HMAC no longer matches.
            assert logger.verify_chain(lines) is False  # HMAC mismatch caught

    def test_chain_only_mode_misses_relinked_tamper(self):
        """Documents WHY HMAC matters: chain-only verification is fooled."""
        with tempfile.TemporaryDirectory() as tmp:
            logger = _logger(tmp)
            f = _write_entries(logger, n=3)
            entries = [json.loads(ln) for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
            entries[1]["operation"] = "op-FORGED"
            prev = "GENESIS"
            for e in entries:
                e["prev_hash"] = prev
                prev = _compute_chain_hash(
                    json.dumps(e, ensure_ascii=False, sort_keys=True)
                )
            lines = [json.dumps(e, ensure_ascii=False) for e in entries]
            # Chain-only: linkage recomputed, so the forge slips through.
            assert logger.verify_chain(lines, verify_hmac=False) is True
            # Full verification catches it.
            assert logger.verify_chain(lines, verify_hmac=True) is False


# ---------------------------------------------------------------------------
# 2. Production HMAC key fail-closed
# ---------------------------------------------------------------------------

class TestProductionKeyFailClosed:
    def test_production_without_key_raises(self):
        os.environ.pop("AUDIT_HMAC_SECRET_KEY", None)
        os.environ.pop("AUDIT_HMAC_SSM_PARAM", None)
        os.environ.pop("AUDIT_HMAC_ALLOW_INSECURE_DEFAULT", None)
        os.environ["ENVIRONMENT"] = "production"
        al_mod._HMAC_SECRET_KEY = None
        with pytest.raises(AuditIntegrityError):
            al_mod._get_hmac_secret()

    def test_insecure_override_allows_default(self):
        os.environ.pop("AUDIT_HMAC_SECRET_KEY", None)
        os.environ.pop("AUDIT_HMAC_SSM_PARAM", None)
        os.environ["ENVIRONMENT"] = "production"
        os.environ["AUDIT_HMAC_ALLOW_INSECURE_DEFAULT"] = "true"
        al_mod._HMAC_SECRET_KEY = None
        key = al_mod._get_hmac_secret()
        assert key == b"aws-ple-audit-default-key-CHANGE-ME"

    def test_development_uses_default_without_raising(self):
        os.environ.pop("AUDIT_HMAC_SECRET_KEY", None)
        os.environ.pop("AUDIT_HMAC_SSM_PARAM", None)
        os.environ["ENVIRONMENT"] = "development"
        al_mod._HMAC_SECRET_KEY = None
        key = al_mod._get_hmac_secret()
        assert key == b"aws-ple-audit-default-key-CHANGE-ME"

    def test_log_operation_propagates_integrity_error(self):
        """The fail-closed key error must not be swallowed by log_operation."""
        os.environ.pop("AUDIT_HMAC_SECRET_KEY", None)
        os.environ.pop("AUDIT_HMAC_SSM_PARAM", None)
        os.environ["ENVIRONMENT"] = "production"
        al_mod._HMAC_SECRET_KEY = None
        with tempfile.TemporaryDirectory() as tmp:
            logger = AuditLogger(s3_bucket=None, local_fallback_dir=tmp)
            with pytest.raises(AuditIntegrityError):
                logger.log_operation(operation="op", input_data={"x": 1})


# ---------------------------------------------------------------------------
# 3. S3 write fail-closed
# ---------------------------------------------------------------------------

class _FailingS3:
    class exceptions:
        class NoSuchKey(Exception):
            pass

    def get_object(self, **kwargs):
        raise self.exceptions.NoSuchKey()

    def put_object(self, **kwargs):
        raise RuntimeError("S3 unavailable")


class TestS3FailClosed:
    def _make(self, tmp, fail_closed):
        os.environ["AUDIT_HMAC_SECRET_KEY"] = "pytest-integrity-key"
        al_mod._HMAC_SECRET_KEY = None
        logger = AuditLogger(
            s3_bucket="dummy-bucket",
            local_fallback_dir=tmp,
            fail_closed=fail_closed,
        )
        logger._s3_client = _FailingS3()  # force S3 path to fail
        return logger

    def test_fallback_when_not_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = self._make(tmp, fail_closed=False)
            entry = logger.log_operation(operation="op", input_data={"x": 1})
            assert entry is not None  # wrote to local fallback
            assert list(Path(tmp).rglob("audit_*.jsonl"))

    def test_raises_when_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = self._make(tmp, fail_closed=True)
            with pytest.raises(AuditIntegrityError):
                logger.log_operation(operation="op", input_data={"x": 1})
