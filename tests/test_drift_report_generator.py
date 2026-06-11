"""Regression for the drift→auto-retrain chain TypeError (confirmed HIGH).

The drift Processing-Job inline script column_stacked the DuckDB
``fetchnumpy()`` dict into a 2D ndarray and passed it to
``DriftDetector.detect_drift``, which only accepts a column dict / pa.Table /
DataFrame → TypeError → drift_*.json never written → auto_retrain_trigger's
PSI check permanently False (drift could never trigger a retrain).

These tests verify (a) the inline script now passes a dict, and (b)
DriftDetector tolerates both a column dict and a 2D ndarray.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.monitoring.drift_detector import DriftDetector


def _load_build_inline_script():
    """Import _build_inline_script despite the 'lambda' reserved-word dir."""
    import importlib

    return importlib.import_module(
        "containers.lambda.drift_report_generator"
    )._build_inline_script


class TestInlineScript:
    def test_passes_column_dict_not_2d_array(self):
        build = _load_build_inline_script()
        script = build("s3://b/base.parquet", "s3://b/cur.parquet", "s3://b/out")
        # The fixed script builds per-column dicts and feeds them directly.
        assert "detect_drift(baseline_d, current_d)" in script
        # And must NOT column_stack into a 2D ndarray feeding detect_drift.
        assert "np.column_stack" not in script


class TestDetectDriftAcceptsBothShapes:
    def test_accepts_column_dict(self):
        det = DriftDetector()
        baseline = {"f0": np.zeros(200), "f1": np.zeros(200)}
        current = {"f0": np.ones(200), "f1": np.zeros(200)}
        result = det.detect_drift(baseline, current)
        assert "psi_scores" in result
        assert set(result["psi_scores"].keys()) == {"f0", "f1"}

    def test_accepts_2d_ndarray_defensively(self):
        det = DriftDetector()
        baseline = np.zeros((200, 3), dtype=np.float32)
        current = np.ones((200, 3), dtype=np.float32)
        # Must NOT raise TypeError (the original failure mode).
        result = det.detect_drift(baseline, current)
        assert "psi_scores" in result
        assert len(result["psi_scores"]) == 3
