"""Tests for the CEH attribution -> AuditLogger integration (Paper 2 v2).

Covers:
  - ``CausalExpert.get_last_attribution`` returns a detached tensor of
    the expected shape after a forward pass, and ``None`` when CEH is
    disabled.
  - ``PLEModel.get_ceh_attribution`` (standalone unit; see
    ``test_ple_model.py`` for end-to-end PLE paths).
  - ``AuditLogger.log_attribution`` produces an HMAC-signed entry
    whose ``metadata`` carries the regulator-facing fields and whose
    hash chain survives verification; tampering rejects the chain.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import torch

from core.model.experts.causal import CausalExpert
from core.monitoring.audit_logger import AuditLogger


# ---------------------------------------------------------------------------
# CausalExpert.get_last_attribution
# ---------------------------------------------------------------------------

def test_get_last_attribution_shape_and_detached():
    expert = CausalExpert(
        input_dim=16,
        config={
            "output_dim": 8,
            "hidden_dim": 16,
            "n_causal_vars": 4,
            "ceh": {"enabled": True, "hidden_dim": 8, "target_mode": "demeaned"},
        },
    )
    expert.eval()
    x = torch.randn(3, 16)
    with torch.no_grad():
        _ = expert(x)
    attr = expert.get_last_attribution()
    assert attr is not None
    assert attr.shape == (3, 16)
    # Detached: grad_fn must be None
    assert attr.grad_fn is None


def test_get_last_attribution_none_when_ceh_disabled():
    expert = CausalExpert(
        input_dim=16,
        config={
            "output_dim": 8,
            "hidden_dim": 16,
            "n_causal_vars": 4,
            "ceh": {"enabled": False},
        },
    )
    expert.eval()
    _ = expert(torch.randn(2, 16))
    assert expert.get_last_attribution() is None


def test_get_last_attribution_none_before_forward():
    expert = CausalExpert(
        input_dim=16,
        config={
            "output_dim": 8,
            "hidden_dim": 16,
            "n_causal_vars": 4,
            "ceh": {"enabled": True, "hidden_dim": 8},
        },
    )
    assert expert.get_last_attribution() is None


# ---------------------------------------------------------------------------
# AuditLogger.log_attribution + hash chain
# ---------------------------------------------------------------------------

def _build_logger(tmpdir: str) -> AuditLogger:
    os.environ["AUDIT_LOG_DIR"] = tmpdir
    os.environ.pop("AUDIT_S3_BUCKET", None)
    os.environ.pop("AUDIT_HMAC_SSM_PARAM", None)
    os.environ["AUDIT_HMAC_SECRET_KEY"] = "pytest-key"
    return AuditLogger(s3_bucket=None, local_fallback_dir=tmpdir)


def test_log_attribution_writes_regulator_fields():
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = _build_logger(tmpdir)
        top = [{"feature": "income", "weight": 0.42},
               {"feature": "tenure", "weight": -0.18}]
        attr_hash = hashlib.sha256(b"dummy").hexdigest()
        entry = logger.log_attribution(
            model_id="m-1",
            sample_id="customer-007",
            top_features=top,
            attribution_hash=attr_hash,
            input_dim=102,
            user="unit-test",
        )
        assert entry is not None
        md = entry["metadata"]
        assert md["sample_id"] == "customer-007"
        assert md["top_features"] == top
        assert md["attribution_hash"] == attr_hash
        assert md["input_dim"] == 102
        assert md["operation_type"] == "attribution"
        # HMAC + chain linkage
        assert "hmac" in entry and len(entry["hmac"]) == 64
        assert entry["prev_hash"] == "GENESIS"


def test_log_attribution_chain_verifies_and_detects_tampering():
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = _build_logger(tmpdir)
        for i in range(3):
            logger.log_attribution(
                model_id="m-1",
                sample_id=f"customer-{i}",
                top_features=[{"feature": "f0", "weight": 0.1 * i}],
                attribution_hash=hashlib.sha256(f"{i}".encode()).hexdigest(),
                input_dim=102,
                user="unit-test",
            )
        log_files = list(Path(tmpdir).rglob("audit_*.jsonl"))
        assert log_files
        lines = [ln for ln in log_files[0].read_text().splitlines()
                 if ln.strip()]
        assert logger.verify_chain(lines) is True

        tampered = (log_files[0].read_text()
                    .replace("customer-1", "attacker"))
        log_files[0].write_text(tampered)
        tampered_lines = [ln for ln in log_files[0].read_text().splitlines()
                          if ln.strip()]
        assert logger.verify_chain(tampered_lines) is False
