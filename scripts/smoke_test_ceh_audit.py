"""Smoke test: CEH attribution -> AuditLogger.log_attribution pipeline.

End-to-end demonstration of the Paper 2 v2 audit-log integration:
1. Build a CEH-enabled CausalExpert.
2. Run an inference-mode forward.
3. Extract the full attribution vector via the public accessor.
4. Compute a SHA256 hash of the full vector + extract top-K features.
5. Emit ``log_attribution`` events through AuditLogger.
6. Verify the hash chain is intact on the resulting local JSONL.

All operations run on CPU, no network calls (AuditLogger falls back
to a local file store when AWS is unavailable). Runtime: <1 s.
"""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from pathlib import Path
from typing import List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.model.experts.causal import CausalExpert  # noqa: E402
from core.monitoring.audit_logger import AuditLogger  # noqa: E402


def build_expert(input_dim: int = 32) -> CausalExpert:
    expert = CausalExpert(
        input_dim=input_dim,
        config={
            "output_dim": 16,
            "hidden_dim": 32,
            "n_causal_vars": 8,
            "ceh": {
                "enabled": True,
                "hidden_dim": 16,
                "loss_weight": 0.1,
                "dropout": 0.0,
                "target_mode": "demeaned",
            },
        },
    )
    expert.eval()
    return expert


def attribution_to_audit_records(
    attribution: torch.Tensor,
    feature_names: List[str],
    sample_ids: List[str],
    top_k: int = 5,
) -> List[dict]:
    """Translate an attribution tensor into per-sample audit record inputs."""
    records = []
    for i, sid in enumerate(sample_ids):
        row = attribution[i].cpu().float().numpy()
        order = (-abs(row)).argsort()[:top_k]
        top_features = [
            {"feature": feature_names[j], "weight": float(row[j])}
            for j in order
        ]
        full_bytes = row.astype("float32").tobytes()
        records.append({
            "sample_id": sid,
            "top_features": top_features,
            "attribution_hash": hashlib.sha256(full_bytes).hexdigest(),
        })
    return records


def main() -> None:
    torch.manual_seed(0)
    input_dim = 32
    n_samples = 4
    feature_names = [f"f{i:02d}" for i in range(input_dim)]
    sample_ids = [f"customer-{i}" for i in range(n_samples)]

    expert = build_expert(input_dim=input_dim)
    x = torch.randn(n_samples, input_dim)
    with torch.no_grad():
        _ = expert(x)

    attribution = expert.get_last_attribution()
    assert attribution is not None, "attribution not populated"
    assert attribution.shape == (n_samples, input_dim), \
        f"unexpected shape {attribution.shape}"
    print(f"attribution shape: {tuple(attribution.shape)} (ok)")

    records = attribution_to_audit_records(
        attribution, feature_names, sample_ids, top_k=5,
    )
    print(f"built {len(records)} audit record payloads")

    # Route audit logs to a temp dir so the test is hermetic.
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["AUDIT_LOG_DIR"] = tmpdir
        # Force local fallback (don't touch S3 or SSM).
        os.environ.pop("AUDIT_S3_BUCKET", None)
        os.environ.pop("AUDIT_HMAC_SSM_PARAM", None)
        os.environ["AUDIT_HMAC_SECRET_KEY"] = "smoke-test-key"

        logger = AuditLogger(
            s3_bucket=None,
            local_fallback_dir=tmpdir,
        )

        emitted = 0
        for rec in records:
            entry = logger.log_attribution(
                model_id="ceh-smoke-test",
                sample_id=rec["sample_id"],
                top_features=rec["top_features"],
                attribution_hash=rec["attribution_hash"],
                input_dim=input_dim,
                user="smoke-test",
            )
            assert entry is not None, \
                f"log_attribution returned None for {rec['sample_id']}"
            assert "hmac" in entry and len(entry["hmac"]) == 64, \
                "missing or bad HMAC"
            assert "prev_hash" in entry, "missing prev_hash (chain link)"
            emitted += 1
        print(f"emitted {emitted} audit entries")

        log_files = list(Path(tmpdir).rglob("audit_*.jsonl"))
        assert log_files, f"no audit log file written in {tmpdir}"
        print(f"audit log file: {log_files[0]}")

        def _read_lines(path: Path) -> list:
            return [ln for ln in path.read_text().splitlines() if ln.strip()]

        ok = logger.verify_chain(_read_lines(log_files[0]))
        assert ok, "hash chain verification FAILED"
        print("hash chain verification: OK")

        # Sanity-check: tampering breaks the chain.
        tampered = log_files[0].read_text().replace("customer-0", "ATTACKER!")
        log_files[0].write_text(tampered)
        ok_after = logger.verify_chain(_read_lines(log_files[0]))
        assert not ok_after, "tampered file passed verification (bug)"
        print("tamper detection: OK (chain rejects modified record)")

    print()
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
