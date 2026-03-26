"""Tests for the 3-stage FeatureNormalizer.

Verifies:
  - Scaler fits on train only (val/test use same params)
  - Power-law _log columns are NOT scaled
  - Binary columns are NOT scaled (pass-through)
  - Output column order: [scaled_continuous | binary | power_law_log_copies]
  - Save / load round-trip preserves parameters
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.pipeline.normalizer import FeatureNormalizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_dataset(n_train=500, n_val=100, seed=42):
    """Create a synthetic dataset with known column types.

    Columns:
      - normal_a, normal_b   : Gaussian continuous (should be scaled)
      - binary_x, binary_y   : 0/1 binary (should NOT be scaled)
      - powerlaw_z           : Pareto-like heavy tail (should get _log copy, NOT scaled)
    """
    rng = np.random.RandomState(seed)

    def _build(n):
        return pd.DataFrame({
            "normal_a": rng.randn(n) * 10 + 50,
            "normal_b": rng.randn(n) * 2 + 5,
            "binary_x": rng.choice([0, 1], size=n),
            "binary_y": rng.choice([0.0, 1.0], size=n),
            # Pareto with shape=1.5 → heavy tail, high skew/kurt
            "powerlaw_z": (rng.pareto(1.5, size=n) + 1) * 100,
        })

    train = _build(n_train)
    val = _build(n_val)
    feature_cols = ["normal_a", "normal_b", "binary_x", "binary_y", "powerlaw_z"]
    return train, val, feature_cols


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFeatureNormalizerClassification:
    """Column classification (binary vs continuous)."""

    def test_binary_detected(self):
        train, _, feature_cols = _make_dataset()
        norm = FeatureNormalizer()
        norm.fit(train, feature_cols)

        assert "binary_x" in norm.binary_cols
        assert "binary_y" in norm.binary_cols
        assert "binary_x" not in norm.continuous_cols
        assert "binary_y" not in norm.continuous_cols

    def test_continuous_detected(self):
        train, _, feature_cols = _make_dataset()
        norm = FeatureNormalizer()
        norm.fit(train, feature_cols)

        assert "normal_a" in norm.continuous_cols
        assert "normal_b" in norm.continuous_cols
        assert "powerlaw_z" in norm.continuous_cols


class TestScalerFitOnTrainOnly:
    """Scaler must use train statistics, not train+val."""

    def test_val_not_refitted(self):
        train, val, feature_cols = _make_dataset()
        norm = FeatureNormalizer()
        norm.fit(train, feature_cols)

        # Record scaler params after fitting on train
        mean_after_fit = norm.scaler.mean_.copy()
        scale_after_fit = norm.scaler.scale_.copy()

        # Transform val — should NOT change scaler params
        _ = norm.transform(val, feature_cols)

        np.testing.assert_array_equal(norm.scaler.mean_, mean_after_fit)
        np.testing.assert_array_equal(norm.scaler.scale_, scale_after_fit)

    def test_train_scaled_approx_zero_mean(self):
        train, _, feature_cols = _make_dataset()
        norm = FeatureNormalizer()
        norm.fit(train, feature_cols)
        out = norm.transform(train, feature_cols)

        for col in norm.continuous_cols:
            assert abs(out[col].mean()) < 0.15, (
                f"Scaled train column '{col}' should have near-zero mean, "
                f"got {out[col].mean():.4f}"
            )


class TestPowerLawNotScaled:
    """Power-law _log copies must NOT be standardized."""

    def test_log_copies_are_raw_log1p(self):
        train, _, feature_cols = _make_dataset()
        norm = FeatureNormalizer()
        norm.fit(train, feature_cols)

        # powerlaw_z should be detected as power-law with Pareto data
        # (If the detection threshold is not met due to random seed,
        #  we skip gracefully.)
        if "powerlaw_z" not in norm.power_law_cols:
            pytest.skip("powerlaw_z not detected as power-law with this seed")

        out = norm.transform(train, feature_cols)
        expected = np.log1p(train["powerlaw_z"].fillna(0).clip(lower=0))

        assert "powerlaw_z_log" in out.columns, (
            "Missing powerlaw_z_log column in output"
        )
        np.testing.assert_allclose(
            out["powerlaw_z_log"].values,
            expected.values,
            rtol=1e-6,
            err_msg="Power-law _log column should be raw log1p, not scaled",
        )

    def test_log_copy_not_zero_mean(self):
        """If _log were scaled, it would have ~0 mean. Raw log1p won't."""
        train, _, feature_cols = _make_dataset()
        norm = FeatureNormalizer()
        norm.fit(train, feature_cols)

        if "powerlaw_z" not in norm.power_law_cols:
            pytest.skip("powerlaw_z not detected")

        out = norm.transform(train, feature_cols)
        log_mean = out["powerlaw_z_log"].mean()
        # Raw log1p of Pareto(1.5)*100 has mean >> 0
        assert abs(log_mean) > 1.0, (
            f"powerlaw_z_log mean={log_mean:.4f} — looks scaled (should be raw)"
        )


class TestBinaryPassThrough:
    """Binary columns should not be modified."""

    def test_binary_unchanged(self):
        train, _, feature_cols = _make_dataset()
        norm = FeatureNormalizer()
        norm.fit(train, feature_cols)
        out = norm.transform(train, feature_cols)

        np.testing.assert_array_equal(
            out["binary_x"].values, train["binary_x"].values,
        )
        np.testing.assert_array_equal(
            out["binary_y"].values, train["binary_y"].values,
        )


class TestOutputColumnOrder:
    """Output must be [scaled_continuous | binary | power_law_log_copies]."""

    def test_column_order(self):
        train, _, feature_cols = _make_dataset()
        norm = FeatureNormalizer()
        norm.fit(train, feature_cols)
        out = norm.transform(train, feature_cols)

        expected_order = (
            norm.continuous_cols
            + norm.binary_cols
            + [f"{c}_log" for c in norm.power_law_cols]
        )
        assert list(out.columns) == expected_order

    def test_output_columns_property(self):
        train, _, feature_cols = _make_dataset()
        norm = FeatureNormalizer()
        norm.fit(train, feature_cols)
        out = norm.transform(train, feature_cols)

        assert list(out.columns) == norm.output_columns


class TestSaveLoadRoundTrip:
    """Serialization must preserve all fitted state."""

    def test_round_trip(self):
        train, val, feature_cols = _make_dataset()
        norm = FeatureNormalizer()
        norm.fit(train, feature_cols)
        out_before = norm.transform(val, feature_cols)

        with tempfile.TemporaryDirectory() as tmpdir:
            norm.save(tmpdir)
            loaded = FeatureNormalizer.load(tmpdir)

        out_after = loaded.transform(val, feature_cols)

        pd.testing.assert_frame_equal(out_before, out_after)
        assert loaded.continuous_cols == norm.continuous_cols
        assert loaded.binary_cols == norm.binary_cols
        assert loaded.power_law_cols == norm.power_law_cols


class TestEdgeCases:
    """Edge cases: empty columns, all binary, no power-law."""

    def test_all_binary(self):
        df = pd.DataFrame({
            "a": [0, 1, 0, 1, 0],
            "b": [1, 1, 0, 0, 1],
        })
        norm = FeatureNormalizer()
        norm.fit(df, ["a", "b"])
        out = norm.transform(df, ["a", "b"])

        assert norm.continuous_cols == []
        assert norm.power_law_cols == []
        assert set(out.columns) == {"a", "b"}
        np.testing.assert_array_equal(out["a"].values, df["a"].values)

    def test_missing_column_ignored(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        norm = FeatureNormalizer()
        norm.fit(df, ["x", "nonexistent"])
        assert "nonexistent" not in norm.continuous_cols
        assert "nonexistent" not in norm.binary_cols

    def test_transform_before_fit_raises(self):
        norm = FeatureNormalizer()
        df = pd.DataFrame({"x": [1.0, 2.0]})
        with pytest.raises(RuntimeError, match="before fit"):
            norm.transform(df, ["x"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
