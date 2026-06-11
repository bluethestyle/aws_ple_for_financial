"""Tests for the s3://-aware Parquet I/O helper (core/monitoring/parquet_io.py).

Locks in the S7/S8 archive fix: the prior Path-based code mangled s3:// URIs
(``Path('s3://b/k')`` → ``s3:\\b\\k`` on Windows) so the fairness/drift
archive silently fell back to 0.5 in the PromotionGate. These tests verify
the local round-trip and that s3:// resolves to an S3 filesystem with the URI
intact (without needing a live bucket).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

from core.monitoring.parquet_io import (
    _resolve,
    parquet_exists,
    read_parquet_rows,
    write_parquet_rows,
)


class TestLocalRoundTrip:
    def test_write_then_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "sub" / "archive.parquet")
            rows = [{"disparate_impact": 0.8, "tag": "a"},
                    {"disparate_impact": 0.6, "tag": "b"}]
            assert write_parquet_rows(rows, path) is True
            assert parquet_exists(path) is True
            back = read_parquet_rows(path)
            assert back == rows

    def test_missing_file_reads_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "nope.parquet")
            assert parquet_exists(path) is False
            assert read_parquet_rows(path) == []

    def test_parent_dirs_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "a" / "b" / "c" / "x.parquet")
            assert write_parquet_rows([{"v": 1}], path) is True
            assert Path(path).exists()


class TestS3UriHandling:
    def test_s3_uri_not_mangled(self):
        """s3:// must resolve to an S3 filesystem with 'bucket/key' intact."""
        fs, normalized = _resolve("s3://my-bucket/compliance/fairness/m.parquet")
        # normalized drops the scheme but keeps the path, never backslashes
        assert normalized == "my-bucket/compliance/fairness/m.parquet"
        assert "\\" not in normalized
        assert "s3:" not in normalized
        assert type(fs).__name__ == "S3FileSystem"

    def test_local_uri_resolves_localfs(self):
        fs, normalized = _resolve("/tmp/x.parquet")
        assert type(fs).__name__ == "LocalFileSystem"
        assert normalized == "/tmp/x.parquet"


class TestFairnessArchiveIntegration:
    """End-to-end: FairnessMonitor write → archive source read on a local path."""

    def test_fairness_archive_round_trip(self):
        from core.compliance.metadata_aggregator import build_fairness_archive_source

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "fairness" / "metrics.parquet")
            # Write two archive rows directly via the helper (schema mirrors
            # FairnessMonitor entries: a disparate_impact field).
            write_parquet_rows(
                [{"disparate_impact": 0.9}, {"disparate_impact": 0.5}],
                path,
            )
            source = build_fairness_archive_source(path)
            out = source("model-v1")
            # worst-case DI 0.5 → min(0.5, 2.0)=0.5 surfaced (not the {}
            # fallback that the old Path-based s3 bug produced)
            assert "disparate_impact_min" in out
            assert out["disparate_impact_min"] == pytest.approx(0.5)


class TestDriftArchiveIntegration:
    """DriftDetector.archive_result must round-trip through the helper."""

    def test_drift_archive_round_trip(self):
        from core.monitoring.drift_detector import DriftDetector

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "drift" / "psi.parquet")
            det = DriftDetector()
            result = {
                "psi_scores": {"f0": 0.05, "f1": 0.30},
                "warning_features": [],
                "critical_features": ["f1"],
            }
            written = det.archive_result(result, path)
            assert written == 2
            rows = read_parquet_rows(path)
            assert {r["feature"] for r in rows} == {"f0", "f1"}
            sev = {r["feature"]: r["severity"] for r in rows}
            assert sev["f1"] == "critical"

    def test_drift_auto_archive_from_config(self):
        """config archive_parquet_path → detect_drift auto-archives."""
        import numpy as np

        from core.monitoring.drift_detector import DriftDetector

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "auto" / "psi.parquet")
            cfg = {"monitoring": {"drift": {"archive_parquet_path": path}}}
            det = DriftDetector(config=cfg)
            baseline = {"f0": np.zeros(100)}
            current = {"f0": np.ones(100)}
            det.detect_drift(baseline, current)
            assert read_parquet_rows(path)  # archive was written
