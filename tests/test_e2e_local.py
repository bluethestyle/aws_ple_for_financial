"""
End-to-end local tests for the multi-task financial ML pipeline.

Validates the full flow: synthetic data generation -> schema validation ->
feature transform -> model forward pass -> loss computation -> evaluation.

All tests run WITHOUT AWS credentials and WITHOUT a GPU.  Data sizes are
kept small (100 samples) for speed.

Usage::

    pytest tests/test_e2e_local.py -v
    pytest tests/test_e2e_local.py -v -k champion
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest
import torch

from tests.fixtures.generate_financial_data import (
    FinancialDataGenerator,
    TASK_DEFINITIONS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SMALL_N = 100
SEED = 42
BATCH_SIZE = 32

# Subset of tasks for fast PLE tests (binary + regression)
FAST_TASKS = [
    {"name": "ctr",        "type": "binary",     "label_col": "label_ctr"},
    {"name": "churn",      "type": "binary",     "label_col": "label_churn"},
    {"name": "ltv",        "type": "regression", "label_col": "label_ltv"},
    {"name": "engagement", "type": "regression", "label_col": "label_engagement"},
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def generator() -> FinancialDataGenerator:
    """Create a reusable data generator with small sample size."""
    return FinancialDataGenerator(n_samples=SMALL_N, seed=SEED)


@pytest.fixture(scope="module")
def full_df(generator: FinancialDataGenerator) -> pd.DataFrame:
    """Generate the full synthetic DataFrame (module-scoped for reuse)."""
    return generator.generate()


@pytest.fixture(scope="module")
def split_dfs(generator: FinancialDataGenerator):
    """Generate train/val/test split DataFrames."""
    return generator.generate_split()


@pytest.fixture(scope="module")
def numeric_cols() -> List[str]:
    return FinancialDataGenerator.get_numeric_feature_names()


@pytest.fixture(scope="module")
def categorical_cols() -> List[str]:
    return FinancialDataGenerator.get_categorical_feature_names()


@pytest.fixture(scope="module")
def label_cols() -> List[str]:
    return FinancialDataGenerator.get_label_columns()


# ---------------------------------------------------------------------------
# 1. Data Generation Tests
# ---------------------------------------------------------------------------


class TestDataGeneration:
    """Verify synthetic data generator correctness."""

    def test_shape(self, full_df: pd.DataFrame, numeric_cols, categorical_cols, label_cols):
        """Output DataFrame has expected number of rows and columns."""
        assert len(full_df) == SMALL_N
        # customer_id + numeric + categorical + labels
        expected_cols = 1 + len(numeric_cols) + len(categorical_cols) + len(label_cols)
        assert len(full_df.columns) == expected_cols, (
            f"Expected {expected_cols} columns, got {len(full_df.columns)}"
        )

    def test_deterministic(self, generator: FinancialDataGenerator):
        """Two calls with the same seed produce identical results."""
        df1 = generator.generate()
        df2 = generator.generate()
        pd.testing.assert_frame_equal(df1, df2)

    def test_no_nulls(self, full_df: pd.DataFrame):
        """Synthetic data should contain no null values."""
        null_counts = full_df.isnull().sum()
        assert null_counts.sum() == 0, f"Found nulls:\n{null_counts[null_counts > 0]}"

    def test_numeric_features_are_finite(self, full_df: pd.DataFrame, numeric_cols):
        """All numeric features should be finite (no inf/nan)."""
        for col in numeric_cols:
            assert np.all(np.isfinite(full_df[col].values)), f"Non-finite values in {col}"

    def test_binary_labels_are_01(self, full_df: pd.DataFrame):
        """Binary task labels must be in {0, 1}."""
        binary_tasks = [t for t in TASK_DEFINITIONS if t["type"] == "binary"]
        for task in binary_tasks:
            col = task["label_col"]
            unique = set(full_df[col].unique())
            assert unique.issubset({0, 1}), f"{col} has values outside {{0, 1}}: {unique}"

    def test_multiclass_labels_range(self, full_df: pd.DataFrame):
        """Multiclass labels should be non-negative integers."""
        multiclass_tasks = [t for t in TASK_DEFINITIONS if t["type"] == "multiclass"]
        for task in multiclass_tasks:
            col = task["label_col"]
            values = full_df[col].values
            assert np.all(values >= 0), f"{col} has negative values"
            assert np.all(values == values.astype(int)), f"{col} has non-integer values"

    def test_regression_labels_finite(self, full_df: pd.DataFrame):
        """Regression labels should be finite."""
        regression_tasks = [t for t in TASK_DEFINITIONS if t["type"] == "regression"]
        for task in regression_tasks:
            col = task["label_col"]
            assert np.all(np.isfinite(full_df[col].values)), f"Non-finite in {col}"

    def test_label_correlations(self, full_df: pd.DataFrame):
        """Churn and retention should be negatively correlated."""
        corr = full_df["label_churn"].corr(full_df["label_retention"])
        assert corr < 0, f"Expected negative churn-retention correlation, got {corr:.3f}"

    def test_split_sizes(self, split_dfs):
        """Train/val/test split sizes should be approximately correct."""
        train_df, val_df, test_df = split_dfs
        total = len(train_df) + len(val_df) + len(test_df)
        assert total == SMALL_N
        assert len(train_df) == int(SMALL_N * 0.7)
        assert len(val_df) == int(SMALL_N * 0.15)

    def test_split_no_overlap(self, split_dfs):
        """Train/val/test splits should not overlap on customer_id."""
        train_df, val_df, test_df = split_dfs
        train_ids = set(train_df["customer_id"])
        val_ids = set(val_df["customer_id"])
        test_ids = set(test_df["customer_id"])
        assert len(train_ids & val_ids) == 0, "Train/val overlap"
        assert len(train_ids & test_ids) == 0, "Train/test overlap"
        assert len(val_ids & test_ids) == 0, "Val/test overlap"

    def test_save_parquet(self, generator: FinancialDataGenerator):
        """Saving to Parquet should produce readable files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generator.save_splits(tmpdir)
            assert set(paths.keys()) == {"train", "val", "test"}
            for name, path in paths.items():
                assert Path(path).exists(), f"{name} file missing: {path}"
                df = pd.read_parquet(path)
                assert len(df) > 0, f"{name} file is empty"


# ---------------------------------------------------------------------------
# 2. Schema Validation Tests
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    """Test data validation on synthetic data."""

    def test_null_ratio_check_passes(self, full_df: pd.DataFrame):
        """Synthetic data should pass null-ratio validation."""
        from core.data.validation import DataValidator, CheckStatus

        validator = DataValidator(null_ratio_threshold=0.5)
        result = validator.validate(
            "synthetic_financial", full_df, checks=["null_ratio"],
        )
        assert result.passed, f"Null ratio check failed:\n{result.summary()}"

    def test_pii_check_passes(self, full_df: pd.DataFrame):
        """Synthetic data should not contain PII patterns."""
        from core.data.validation import DataValidator, CheckStatus

        validator = DataValidator()
        result = validator.validate(
            "synthetic_financial", full_df, checks=["pii"],
        )
        # Check no FAIL (warnings for column name heuristic are OK)
        failures = [c for c in result.checks if c.status == CheckStatus.FAIL]
        assert len(failures) == 0, f"PII check failures:\n{result.summary()}"

    def test_drift_check_same_distribution(self, generator: FinancialDataGenerator):
        """Drift check between two samples from same seed should pass."""
        from core.data.validation import DataValidator

        df1 = generator.generate()
        gen2 = FinancialDataGenerator(n_samples=SMALL_N, seed=SEED + 100)
        df2 = gen2.generate()

        # Use only numeric columns for drift
        numeric = FinancialDataGenerator.get_numeric_feature_names()
        existing = [c for c in numeric if c in df1.columns and c in df2.columns]
        df1_num = df1[existing[:20]]  # subset for speed
        df2_num = df2[existing[:20]]

        validator = DataValidator(psi_threshold=0.5)
        result = validator.validate(
            "synthetic_financial", df1_num,
            reference_df=df2_num,
            checks=["drift"],
        )
        assert result.passed, f"Drift check failed:\n{result.summary()}"


# ---------------------------------------------------------------------------
# 3. Feature Transform Tests
# ---------------------------------------------------------------------------


class TestFeatureTransform:
    """Test that numeric features can be prepared for model input."""

    def test_numeric_to_tensor(self, full_df: pd.DataFrame, numeric_cols):
        """Numeric features should convert cleanly to a float32 tensor."""
        existing = [c for c in numeric_cols if c in full_df.columns]
        values = full_df[existing].values.astype(np.float32)
        tensor = torch.tensor(values, dtype=torch.float32)

        assert tensor.shape == (SMALL_N, len(existing))
        assert torch.all(torch.isfinite(tensor)), "Non-finite values in feature tensor"

    def test_feature_standardization(self, full_df: pd.DataFrame, numeric_cols):
        """Standardized features should have ~0 mean and ~1 std."""
        existing = [c for c in numeric_cols if c in full_df.columns]
        values = full_df[existing].values.astype(np.float64)

        mean = values.mean(axis=0)
        std = values.std(axis=0)

        # Not all features are standardized (some are counts, balances, etc.)
        # But at least the derived_ratios group should be close to standard normal
        ratio_cols = [c for c in existing if c.startswith("derived_ratios")]
        ratio_idx = [existing.index(c) for c in ratio_cols]
        if ratio_idx:
            ratio_means = mean[ratio_idx]
            ratio_stds = std[ratio_idx]
            assert np.abs(ratio_means).mean() < 1.0, "Derived ratios mean too far from 0"
            assert np.abs(ratio_stds - 1.0).mean() < 1.0, "Derived ratios std too far from 1"


# ---------------------------------------------------------------------------
# 4. Model Forward Pass Tests
# ---------------------------------------------------------------------------


class TestPLEModelForwardPass:
    """Test PLE model forward pass with synthetic financial features."""

    def _build_ple_config(self, input_dim: int, task_defs: List[dict]):
        """Build a minimal PLEConfig for testing."""
        from core.model.ple.config import (
            PLEConfig, ExpertConfig, TaskTowerConfig,
            CGCConfig, AdaTTConfig, LossWeightingConfig,
        )

        task_names = [t["name"] for t in task_defs]
        task_overrides: Dict[str, dict] = {}
        for t in task_defs:
            if t["type"] == "binary":
                task_overrides[t["name"]] = {
                    "task_type": "binary", "output_dim": 1, "activation": "sigmoid",
                }
            elif t["type"] == "regression":
                task_overrides[t["name"]] = {
                    "task_type": "regression", "output_dim": 1, "activation": None,
                }
            elif t["type"] == "multiclass":
                task_overrides[t["name"]] = {
                    "task_type": "multiclass", "output_dim": 5, "activation": "softmax",
                }
            elif t["type"] == "ranking":
                task_overrides[t["name"]] = {
                    "task_type": "regression", "output_dim": 1, "activation": "sigmoid",
                }

        return PLEConfig(
            input_dim=input_dim,
            task_names=task_names,
            num_shared_experts=2,
            num_task_experts_per_task=1,
            num_extraction_layers=1,
            task_expert_output_dim=16,
            shared_expert=ExpertConfig(hidden_dims=[32], output_dim=32, dropout=0.0),
            task_expert=ExpertConfig(hidden_dims=[32], output_dim=16, dropout=0.0),
            task_tower=TaskTowerConfig(hidden_dims=[16], dropout=0.0),
            cgc=CGCConfig(enabled=False),
            adatt=AdaTTConfig(enabled=False),
            loss_weighting=LossWeightingConfig(strategy="fixed"),
            dropout=0.0,
            task_overrides=task_overrides,
        )

    def _prepare_batch(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        task_defs: List[dict],
        batch_size: int = BATCH_SIZE,
    ):
        """Prepare a feature tensor and targets dict from a DataFrame."""
        existing = [c for c in numeric_cols if c in df.columns]
        features = torch.tensor(
            df[existing].iloc[:batch_size].values.astype(np.float32),
            dtype=torch.float32,
        )
        targets: Dict[str, torch.Tensor] = {}
        for t in task_defs:
            col = t["label_col"]
            if col in df.columns:
                targets[t["name"]] = torch.tensor(
                    df[col].iloc[:batch_size].values.astype(np.float32),
                    dtype=torch.float32,
                )
        return features, targets

    def test_forward_no_labels(self, full_df: pd.DataFrame, numeric_cols):
        """Forward pass without labels produces predictions, no loss."""
        from core.model.ple import PLEModel, PLEInput, PLEOutput

        existing = [c for c in numeric_cols if c in full_df.columns]
        input_dim = len(existing)
        config = self._build_ple_config(input_dim, FAST_TASKS)
        model = PLEModel(config)

        features, _ = self._prepare_batch(full_df, numeric_cols, FAST_TASKS)
        inputs = PLEInput(features=features)
        output = model(inputs, compute_loss=False)

        assert isinstance(output, PLEOutput)
        for t in FAST_TASKS:
            assert t["name"] in output.predictions, f"Missing prediction for {t['name']}"
            pred = output.predictions[t["name"]]
            assert pred.shape[0] == BATCH_SIZE, f"Wrong batch dim for {t['name']}"
        assert output.total_loss is None

    def test_forward_with_labels(self, full_df: pd.DataFrame, numeric_cols):
        """Forward pass with labels produces per-task losses."""
        from core.model.ple import PLEModel, PLEInput, PLEOutput

        existing = [c for c in numeric_cols if c in full_df.columns]
        input_dim = len(existing)
        config = self._build_ple_config(input_dim, FAST_TASKS)
        model = PLEModel(config)

        features, targets = self._prepare_batch(full_df, numeric_cols, FAST_TASKS)
        inputs = PLEInput(features=features, targets=targets)
        output = model(inputs)

        assert output.total_loss is not None
        assert output.total_loss.item() > 0
        assert output.task_losses is not None
        for t in FAST_TASKS:
            assert t["name"] in output.task_losses, f"Missing loss for {t['name']}"

    def test_backward_pass(self, full_df: pd.DataFrame, numeric_cols):
        """Loss should support backpropagation."""
        from core.model.ple import PLEModel, PLEInput

        existing = [c for c in numeric_cols if c in full_df.columns]
        input_dim = len(existing)
        config = self._build_ple_config(input_dim, FAST_TASKS)
        model = PLEModel(config)

        features, targets = self._prepare_batch(full_df, numeric_cols, FAST_TASKS)
        inputs = PLEInput(features=features, targets=targets)
        output = model(inputs)

        assert output.total_loss.requires_grad
        output.total_loss.backward()

        # At least some parameters should have gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad, "No parameter received a gradient"

    def test_prediction_shapes_binary(self, full_df: pd.DataFrame, numeric_cols):
        """Binary task predictions should be (batch, 1) in [0, 1]."""
        from core.model.ple import PLEModel, PLEInput

        existing = [c for c in numeric_cols if c in full_df.columns]
        config = self._build_ple_config(len(existing), FAST_TASKS)
        model = PLEModel(config)
        model.eval()

        features, _ = self._prepare_batch(full_df, numeric_cols, FAST_TASKS)
        with torch.no_grad():
            output = model(PLEInput(features=features), compute_loss=False)

        for t in FAST_TASKS:
            if t["type"] == "binary":
                pred = output.predictions[t["name"]]
                assert pred.shape == (BATCH_SIZE, 1), f"{t['name']} shape: {pred.shape}"
                assert pred.min() >= 0.0, f"{t['name']} has negative predictions"
                assert pred.max() <= 1.0, f"{t['name']} has predictions > 1"

    def test_all_16_tasks(self, full_df: pd.DataFrame, numeric_cols):
        """Model should handle all 16 financial tasks simultaneously."""
        from core.model.ple import PLEModel, PLEInput

        existing = [c for c in numeric_cols if c in full_df.columns]
        config = self._build_ple_config(len(existing), TASK_DEFINITIONS)
        model = PLEModel(config)

        features, targets = self._prepare_batch(
            full_df, numeric_cols, TASK_DEFINITIONS, batch_size=16,
        )
        inputs = PLEInput(features=features, targets=targets)
        output = model(inputs)

        assert output.total_loss is not None
        assert len(output.predictions) == 16


# ---------------------------------------------------------------------------
# 5. Evaluation / Metrics Tests
# ---------------------------------------------------------------------------


class TestMetrics:
    """Test that standard metrics can be computed on model predictions."""

    def test_binary_auc(self):
        """AUC should be computable on binary predictions."""
        from sklearn.metrics import roc_auc_score

        rng = np.random.default_rng(SEED)
        y_true = rng.integers(0, 2, 100)
        y_score = rng.uniform(0, 1, 100)
        auc = roc_auc_score(y_true, y_score)
        assert 0 <= auc <= 1

    def test_regression_metrics(self):
        """MAE and RMSE should be computable on regression predictions."""
        rng = np.random.default_rng(SEED)
        y_true = rng.normal(0, 1, 100)
        y_pred = y_true + rng.normal(0, 0.1, 100)

        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        assert mae > 0
        assert rmse > 0
        assert rmse >= mae  # RMSE >= MAE always


# ---------------------------------------------------------------------------
# 6. Champion vs. Challenger Tests
# ---------------------------------------------------------------------------


class TestChampionVsChallenger:
    """Test the champion-vs-challenger comparison flow."""

    def _train_model(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        task_defs: List[dict],
        seed: int,
        lr: float = 1e-3,
        steps: int = 5,
    ) -> Dict[str, float]:
        """Train a small PLE model for a few steps and return val metrics."""
        from core.model.ple import PLEModel, PLEInput
        from core.model.ple.config import (
            PLEConfig, ExpertConfig, TaskTowerConfig,
            CGCConfig, AdaTTConfig, LossWeightingConfig,
        )

        torch.manual_seed(seed)
        existing = [c for c in numeric_cols if c in df.columns]
        input_dim = len(existing)

        task_names = [t["name"] for t in task_defs]
        task_overrides = {}
        for t in task_defs:
            if t["type"] == "binary":
                task_overrides[t["name"]] = {
                    "task_type": "binary", "output_dim": 1, "activation": "sigmoid",
                }
            else:
                task_overrides[t["name"]] = {
                    "task_type": "regression", "output_dim": 1, "activation": None,
                }

        config = PLEConfig(
            input_dim=input_dim,
            task_names=task_names,
            num_shared_experts=2,
            num_task_experts_per_task=1,
            num_extraction_layers=1,
            task_expert_output_dim=16,
            shared_expert=ExpertConfig(hidden_dims=[32], output_dim=32, dropout=0.0),
            task_expert=ExpertConfig(hidden_dims=[32], output_dim=16, dropout=0.0),
            task_tower=TaskTowerConfig(hidden_dims=[16], dropout=0.0),
            cgc=CGCConfig(enabled=False),
            adatt=AdaTTConfig(enabled=False),
            loss_weighting=LossWeightingConfig(strategy="fixed"),
            dropout=0.0,
            task_overrides=task_overrides,
        )

        model = PLEModel(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        features = torch.tensor(
            df[existing].values.astype(np.float32), dtype=torch.float32,
        )
        targets = {}
        for t in task_defs:
            if t["label_col"] in df.columns:
                targets[t["name"]] = torch.tensor(
                    df[t["label_col"]].values.astype(np.float32),
                    dtype=torch.float32,
                )

        # Train
        model.train()
        losses = []
        for step in range(steps):
            optimizer.zero_grad()
            output = model(PLEInput(features=features, targets=targets))
            output.total_loss.backward()
            optimizer.step()
            losses.append(output.total_loss.item())

        return {"final_loss": losses[-1], "initial_loss": losses[0]}

    def test_champion_vs_challenger_comparison(
        self, full_df: pd.DataFrame, numeric_cols,
    ):
        """Two models with different configs should produce comparable metrics."""
        task_defs = FAST_TASKS[:2]  # Just binary tasks for speed

        champion = self._train_model(full_df, numeric_cols, task_defs, seed=42, lr=1e-3)
        challenger = self._train_model(full_df, numeric_cols, task_defs, seed=43, lr=2e-3)

        # Both should have trained (loss decreased or stayed reasonable)
        assert champion["final_loss"] < champion["initial_loss"] * 2, "Champion loss exploded"
        assert challenger["final_loss"] < challenger["initial_loss"] * 2, "Challenger loss exploded"

        # We can compare them (the actual winner doesn't matter for the test)
        champion_better = champion["final_loss"] < challenger["final_loss"]
        logger.info(
            "Champion loss=%.4f, Challenger loss=%.4f, champion_wins=%s",
            champion["final_loss"], challenger["final_loss"], champion_better,
        )

    def test_model_determinism(self, full_df: pd.DataFrame, numeric_cols):
        """Same seed should produce identical training results."""
        task_defs = FAST_TASKS[:2]

        run1 = self._train_model(full_df, numeric_cols, task_defs, seed=42)
        run2 = self._train_model(full_df, numeric_cols, task_defs, seed=42)

        assert abs(run1["final_loss"] - run2["final_loss"]) < 1e-5, (
            f"Non-deterministic: {run1['final_loss']:.6f} vs {run2['final_loss']:.6f}"
        )


# ---------------------------------------------------------------------------
# 7. Pipeline Config Integration Test
# ---------------------------------------------------------------------------


class TestPipelineConfigIntegration:
    """Test that PipelineConfig can be built from synthetic data parameters."""

    def test_build_pipeline_config(self, numeric_cols, categorical_cols):
        """PipelineConfig should accept synthetic data feature specs."""
        from core.pipeline.config import (
            PipelineConfig, TaskSpec, DataSpec, FeatureSpec,
            ModelSpec, TrainingSpec, AWSSpec,
        )

        existing_numeric = numeric_cols[:20]  # subset for speed
        tasks = [
            TaskSpec(name="ctr",   type="binary",     loss="focal", loss_weight=1.0, label_col="label_ctr"),
            TaskSpec(name="churn", type="binary",     loss="focal", loss_weight=1.0, label_col="label_churn"),
            TaskSpec(name="ltv",   type="regression", loss="mse",   loss_weight=0.5, label_col="label_ltv"),
        ]

        config = PipelineConfig(
            task_name="financial_e2e_test",
            tasks=tasks,
            data=DataSpec(source="dummy.parquet", format="parquet"),
            features=FeatureSpec(
                numeric=existing_numeric,
                categorical=categorical_cols,
            ),
            model=ModelSpec(architecture="ple"),
            training=TrainingSpec(epochs=2, batch_size=32, seed=42),
            aws=AWSSpec(),
        )

        assert config.task_name == "financial_e2e_test"
        assert len(config.tasks) == 3
        assert config.model.architecture == "ple"
        assert config.training.epochs == 2


# ---------------------------------------------------------------------------
# 8. Parquet I/O Round-trip Test
# ---------------------------------------------------------------------------


class TestParquetRoundTrip:
    """Test that data survives a Parquet write/read cycle."""

    def test_parquet_round_trip(self, full_df: pd.DataFrame):
        """DataFrame should be identical after write + read."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.parquet")
            full_df.to_parquet(path, index=False)
            loaded = pd.read_parquet(path)

            assert loaded.shape == full_df.shape
            # Check numeric columns
            numeric = full_df.select_dtypes(include=[np.number]).columns.tolist()
            for col in numeric:
                np.testing.assert_array_almost_equal(
                    loaded[col].values, full_df[col].values, decimal=5,
                    err_msg=f"Mismatch in {col}",
                )
            # Check categorical columns
            cats = full_df.select_dtypes(include=["object"]).columns.tolist()
            for col in cats:
                assert list(loaded[col]) == list(full_df[col]), f"Mismatch in {col}"
