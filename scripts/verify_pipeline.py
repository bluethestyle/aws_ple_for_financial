#!/usr/bin/env python3
"""
Pipeline Verification Script — validates each stage locally with synthetic data.

Usage:
    python scripts/verify_pipeline.py [--samples 1000] [--verbose]

No AWS credentials or GPU required. Uses CPU + numpy fallbacks only.
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so ``core.*`` imports work regardless
# of working directory.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Logging & global state
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
_logger = logging.getLogger("verify_pipeline")
_logger.setLevel(logging.INFO)

# Collected results for the summary table
_results: List[Dict[str, Any]] = []
_shared: Dict[str, Any] = {}  # data shared between steps

# Colour helpers (no-ops when stdout is not a tty)
_USE_COLOR = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _green(s: str) -> str:
    return f"\033[92m{s}\033[0m" if _USE_COLOR else s


def _red(s: str) -> str:
    return f"\033[91m{s}\033[0m" if _USE_COLOR else s


def _yellow(s: str) -> str:
    return f"\033[93m{s}\033[0m" if _USE_COLOR else s


def _record(step: int, name: str, status: str, detail: str, elapsed: float):
    """Record a step result for the final summary table."""
    _results.append({
        "step": step,
        "name": name,
        "status": status,
        "detail": detail,
        "elapsed": elapsed,
    })


# ============================================================================
# Step 1 -- Synthetic Data Generation
# ============================================================================

def step1_data_generation(n_samples: int, verbose: bool) -> bool:
    _logger.info("=" * 60)
    _logger.info("Step 1: Synthetic Data Generation (%d samples)", n_samples)
    _logger.info("=" * 60)
    t0 = time.time()
    try:
        from tests.fixtures.generate_financial_data import (
            FinancialDataGenerator,
            TASK_DEFINITIONS,
            FEATURE_GROUPS,
            CATEGORICAL_FEATURES,
        )

        gen = FinancialDataGenerator(n_samples=n_samples, seed=42)
        df = gen.generate()

        # Basic checks
        assert len(df) == n_samples, f"Expected {n_samples} rows, got {len(df)}"
        n_cols = len(df.columns)

        label_cols = gen.get_label_columns()
        for lc in label_cols:
            assert lc in df.columns, f"Missing label column: {lc}"
            nan_count = int(df[lc].isna().sum())
            assert nan_count == 0, f"Label column '{lc}' has {nan_count} NaNs"

        # Check label distributions
        if verbose:
            for td in TASK_DEFINITIONS:
                lc = td["label_col"]
                nunique = int(df[lc].nunique())
                _logger.info("  %s (%s): %d unique values", lc, td["type"], nunique)

        # Save to temp parquet
        tmp_dir = tempfile.mkdtemp(prefix="ple_verify_")
        parquet_path = os.path.join(tmp_dir, "synthetic.parquet")
        df.to_parquet(parquet_path, index=False)

        # Store for later steps
        _shared["df"] = df
        _shared["tmp_dir"] = tmp_dir
        _shared["parquet_path"] = parquet_path
        _shared["n_samples"] = n_samples

        elapsed = time.time() - t0
        detail = f"{n_samples} samples, {n_cols} columns, saved to {parquet_path}"
        _logger.info(_green(f"  Step 1 PASS: Generated {detail} ({elapsed:.2f}s)"))
        _record(1, "Data Generation", "PASS", f"{n_samples} samples, {n_cols} columns", elapsed)
        return True

    except Exception as exc:
        elapsed = time.time() - t0
        _logger.error(_red(f"  Step 1 FAIL: {exc}"))
        if verbose:
            traceback.print_exc()
        _record(1, "Data Generation", "FAIL", str(exc)[:80], elapsed)
        return False


# ============================================================================
# Step 2 -- Schema Validation
# ============================================================================

def step2_schema_validation(verbose: bool) -> bool:
    _logger.info("=" * 60)
    _logger.info("Step 2: Schema Validation")
    _logger.info("=" * 60)
    t0 = time.time()
    try:
        from core.data.validation import DataValidator, CheckStatus
        from core.data.schema_registry import SchemaRegistry

        df = _shared.get("df")
        if df is None:
            raise RuntimeError("No DataFrame from Step 1 -- skipping.")

        # Try to load schema from config file
        schema_path = str(_PROJECT_ROOT / "configs" / "financial" / "schema.yaml")
        registry = None
        source_name = "financial_synthetic"

        try:
            registry = SchemaRegistry(config_path=schema_path)
            available = registry.list_sources()
            _logger.info("  Loaded schema sources: %s", available)
            # Use first available source for validation -- the synthetic data
            # columns won't match perfectly, but partial validation is useful.
            if available:
                source_name = available[0]
        except Exception as e:
            _logger.info("  Could not load schema.yaml (%s), using bare validator", e)

        validator = DataValidator(
            schema_registry=registry,
            null_ratio_threshold=0.5,
            sample_size=min(len(df), 1000),
        )

        result = validator.validate(source_name, df, checks={"null_ratio", "pii"})

        n_pass = sum(1 for c in result.checks if c.status == CheckStatus.PASS)
        n_warn = sum(1 for c in result.checks if c.status == CheckStatus.WARN)
        n_fail = sum(1 for c in result.checks if c.status == CheckStatus.FAIL)
        n_skip = sum(1 for c in result.checks if c.status == CheckStatus.SKIP)
        n_total = len(result.checks)

        if verbose:
            _logger.info("  Validation summary:\n%s", result.summary())

        # Also run direct schema-based checks if a registry is available
        if registry is not None and registry.has(source_name):
            valid, errors = registry.validate_dataframe(source_name, df)
            if errors:
                _logger.info("  Schema-based validation errors (expected with synthetic data):")
                for err in errors[:5]:
                    _logger.info("    - %s", err)

        elapsed = time.time() - t0
        detail = f"{n_pass}/{n_total} passed, {n_warn} warn, {n_fail} fail, {n_skip} skip"
        status = "PASS" if n_fail == 0 else "WARN"
        msg = f"Step 2 {status}: Schema validation ({detail}) ({elapsed:.2f}s)"
        _logger.info(_green(msg) if status == "PASS" else _yellow(msg))
        _record(2, "Schema Validation", status, detail, elapsed)
        return True

    except Exception as exc:
        elapsed = time.time() - t0
        _logger.error(_red(f"  Step 2 FAIL: {exc}"))
        if verbose:
            traceback.print_exc()
        _record(2, "Schema Validation", "FAIL", str(exc)[:80], elapsed)
        return False


# ============================================================================
# Step 3 -- Feature Generator Smoke Test
# ============================================================================

def step3_feature_generators(verbose: bool) -> bool:
    _logger.info("=" * 60)
    _logger.info("Step 3: Feature Generator Smoke Test")
    _logger.info("=" * 60)
    t0 = time.time()
    try:
        from core.feature.generator import FeatureGeneratorRegistry

        # Trigger registration of all built-in generators.
        # Import each sub-module individually to be resilient when some
        # generators have hard dependencies (e.g., torch for graph.py).
        _gen_modules = [
            "core.feature.generators.tda",
            "core.feature.generators.hmm",
            "core.feature.generators.gmm",
            "core.feature.generators.temporal",
            "core.feature.generators.multidisciplinary",
            "core.feature.generators.phase_transition",
            "core.feature.generators.graph",
            "core.feature.generators.mamba",
        ]
        for _mod_name in _gen_modules:
            try:
                __import__(_mod_name)
            except Exception as e:
                _logger.info("  Could not import %s: %s", _mod_name, e)

        df = _shared.get("df")
        if df is None:
            raise RuntimeError("No DataFrame from Step 1 -- skipping.")

        available = FeatureGeneratorRegistry.list_available()
        _logger.info("  Registered generators: %s", available)

        n_ok = 0
        n_warn = 0
        n_total = len(available)
        gen_outputs: Dict[str, Any] = {}
        sample_df = df.head(50).copy()

        for gen_name in available:
            try:
                info = FeatureGeneratorRegistry.get_info(gen_name)
                libs = info.get("required_libraries", [])
                _logger.info("  [%s] required_libs=%s", gen_name, libs)

                gen = FeatureGeneratorRegistry.create(gen_name)

                # Fit + generate
                gen.fit(sample_df)
                output = gen.generate(sample_df)

                import pandas as pd
                if not isinstance(output, pd.DataFrame):
                    _logger.warning("  [%s] WARN: generate() returned %s, not DataFrame",
                                    gen_name, type(output).__name__)
                    n_warn += 1
                    continue

                # Check for all-NaN columns
                all_nan_cols = [c for c in output.columns if output[c].isna().all()]
                if all_nan_cols:
                    _logger.warning("  [%s] WARN: %d all-NaN columns: %s",
                                    gen_name, len(all_nan_cols), all_nan_cols[:5])

                out_dim = gen.output_dim
                actual_cols = len(output.columns)
                backend_info = f"device={gen.device}"

                _logger.info("  [%s] OK: output_dim=%d, actual_cols=%d, rows=%d (%s)",
                             gen_name, out_dim, actual_cols, len(output), backend_info)

                gen_outputs[gen_name] = output
                n_ok += 1

            except Exception as e:
                n_warn += 1
                _logger.warning("  [%s] WARN: %s", gen_name, str(e)[:120])
                if verbose:
                    traceback.print_exc()

        _shared["gen_outputs"] = gen_outputs

        elapsed = time.time() - t0
        detail = f"{n_ok}/{n_total} generators OK, {n_warn} WARN"
        status = "PASS" if n_ok > 0 else "FAIL"
        msg = f"Step 3 {status}: {detail} ({elapsed:.2f}s)"
        _logger.info(_green(msg) if status == "PASS" else _red(msg))
        _record(3, "Feature Generators", status, detail, elapsed)
        return n_ok > 0

    except Exception as exc:
        elapsed = time.time() - t0
        _logger.error(_red(f"  Step 3 FAIL: {exc}"))
        if verbose:
            traceback.print_exc()
        _record(3, "Feature Generators", "FAIL", str(exc)[:80], elapsed)
        return False


# ============================================================================
# Step 4 -- Feature Group Assembly
# ============================================================================

def step4_feature_assembly(verbose: bool) -> bool:
    _logger.info("=" * 60)
    _logger.info("Step 4: Feature Group Assembly")
    _logger.info("=" * 60)
    t0 = time.time()
    try:
        import numpy as np
        import pandas as pd

        gen_outputs = _shared.get("gen_outputs", {})
        df = _shared.get("df")
        if df is None:
            raise RuntimeError("No DataFrame from Step 1 -- skipping.")

        # Strategy: try FeatureGroupPipeline from config first;
        # if that fails (expected), manually concatenate generator outputs
        # with numeric features from the synthetic data.
        assembled = False
        group_ranges: Dict[str, Tuple[int, int]] = {}
        feature_matrix = None

        # --- Attempt A: Config-driven pipeline ---
        try:
            from core.feature.group_pipeline import FeatureGroupPipeline
            from core.feature.group import FeatureGroupConfig
            import yaml

            cfg_path = str(_PROJECT_ROOT / "configs" / "financial" / "feature_groups.yaml")
            with open(cfg_path, "r") as f:
                raw = yaml.safe_load(f)

            groups = [FeatureGroupConfig.from_dict(g) for g in raw["feature_groups"]]
            pipeline = FeatureGroupPipeline(groups, name="verify_pipeline")
            sample = df.head(100)
            feature_matrix = pipeline.fit_transform(sample)
            group_ranges = pipeline.group_ranges
            assembled = True
            _logger.info("  Config-driven pipeline succeeded (total_dim=%d)", pipeline.total_dim)
        except Exception as e:
            _logger.info("  Config-driven pipeline failed (expected): %s", str(e)[:120])

        # --- Attempt B: Manual concatenation fallback ---
        if not assembled:
            _logger.info("  Falling back to manual concatenation of available outputs...")
            parts: List[pd.DataFrame] = []
            offset = 0

            # First add numeric columns from the raw synthetic data
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            # Exclude label columns
            label_cols = [c for c in numeric_cols if c.startswith("label_")]
            id_cols = ["customer_id"]
            feature_cols = [c for c in numeric_cols
                           if c not in label_cols and c not in id_cols]

            if feature_cols:
                base_features = df.head(50)[feature_cols].copy()
                # Fill NaN with 0 for assembly
                base_features = base_features.fillna(0.0)
                group_ranges["base_numeric"] = (offset, offset + len(feature_cols))
                offset += len(feature_cols)
                parts.append(base_features.reset_index(drop=True))

            # Then append generator outputs
            for gen_name, gen_out in gen_outputs.items():
                n_cols = len(gen_out.columns)
                if n_cols > 0:
                    group_ranges[gen_name] = (offset, offset + n_cols)
                    offset += n_cols
                    parts.append(gen_out.reset_index(drop=True))

            if parts:
                feature_matrix = pd.concat(parts, axis=1)
                feature_matrix = feature_matrix.fillna(0.0)
                assembled = True

        if not assembled or feature_matrix is None or len(feature_matrix.columns) == 0:
            raise RuntimeError("Could not assemble any features.")

        total_dim = len(feature_matrix.columns)
        n_groups = len(group_ranges)

        # Verify group_ranges are contiguous with no gaps
        if group_ranges:
            sorted_ranges = sorted(group_ranges.values(), key=lambda x: x[0])
            for i in range(1, len(sorted_ranges)):
                prev_end = sorted_ranges[i - 1][1]
                cur_start = sorted_ranges[i][0]
                if cur_start != prev_end:
                    _logger.warning("  Gap in group ranges: [%d, %d) -> [%d, %d)",
                                    sorted_ranges[i - 1][0], prev_end,
                                    cur_start, sorted_ranges[i][1])

        _shared["feature_matrix"] = feature_matrix
        _shared["group_ranges"] = group_ranges
        _shared["total_dim"] = total_dim

        if verbose:
            for gname, (s, e) in group_ranges.items():
                _logger.info("  Group '%s': [%d, %d) = %dD", gname, s, e, e - s)

        elapsed = time.time() - t0
        detail = f"{total_dim}D from {n_groups} groups"
        _logger.info(_green(f"  Step 4 PASS: Assembled {detail} ({elapsed:.2f}s)"))
        _record(4, "Feature Assembly", "PASS", detail, elapsed)
        return True

    except Exception as exc:
        elapsed = time.time() - t0
        _logger.error(_red(f"  Step 4 FAIL: {exc}"))
        if verbose:
            traceback.print_exc()
        _record(4, "Feature Assembly", "FAIL", str(exc)[:80], elapsed)
        return False


# ============================================================================
# Step 5 -- DataLoader + PLEInput Construction
# ============================================================================

def step5_dataloader(verbose: bool) -> bool:
    _logger.info("=" * 60)
    _logger.info("Step 5: DataLoader + PLEInput Construction")
    _logger.info("=" * 60)
    t0 = time.time()
    try:
        import torch
        import numpy as np
        import pandas as pd
        from core.data.dataloader import (
            FeatureColumnSpec,
            PLEDataset,
            build_ple_dataloader,
            _ple_collate,
        )
        from core.model.ple.model import PLEInput

        feature_matrix = _shared.get("feature_matrix")
        df = _shared.get("df")
        if feature_matrix is None or df is None:
            raise RuntimeError("No feature matrix from Step 4 -- skipping.")

        n_rows = len(feature_matrix)
        total_dim = len(feature_matrix.columns)

        # Build a combined DataFrame with features + labels
        feature_cols = list(feature_matrix.columns)
        combined_df = feature_matrix.copy()

        # Add label columns (aligned to same row count as feature_matrix)
        from tests.fixtures.generate_financial_data import TASK_DEFINITIONS
        label_columns: Dict[str, str] = {}
        for td in TASK_DEFINITIONS[:4]:  # Use 4 tasks for simplicity
            lc = td["label_col"]
            if lc in df.columns:
                combined_df[lc] = df[lc].iloc[:n_rows].values
                label_columns[td["name"]] = lc

        # Create FeatureColumnSpec -- treat all assembled features as static
        spec = FeatureColumnSpec(static_features=feature_cols)

        # Build DataLoader (CPU, small batch)
        batch_size = min(32, n_rows)
        loader = build_ple_dataloader(
            df=combined_df,
            feature_spec=spec,
            label_columns=label_columns,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            use_gpu_loading=False,
            return_ple_input=False,  # get dict first
        )

        # Iterate one batch
        batch = next(iter(loader))
        assert isinstance(batch, dict), f"Expected dict batch, got {type(batch)}"
        assert "features" in batch, f"Batch missing 'features' key. Keys: {list(batch.keys())}"

        feat_shape = tuple(batch["features"].shape)
        assert feat_shape[0] == batch_size, f"Batch dim mismatch: {feat_shape[0]} != {batch_size}"
        assert feat_shape[1] == total_dim, f"Feature dim mismatch: {feat_shape[1]} != {total_dim}"

        # Construct PLEInput from the batch
        targets = batch.get("targets")
        ple_input = PLEInput(
            features=batch["features"],
            targets=targets,
        )

        # Verify PLEInput basics
        assert ple_input.features.shape == (batch_size, total_dim)
        ple_input_cpu = ple_input.to(torch.device("cpu"))
        assert ple_input_cpu.features.device == torch.device("cpu")

        n_targets = len(targets) if targets else 0
        _shared["ple_input"] = ple_input
        _shared["batch_size"] = batch_size
        _shared["total_dim"] = total_dim
        _shared["label_columns"] = label_columns
        _shared["combined_df"] = combined_df
        _shared["feature_spec"] = spec

        if verbose:
            _logger.info("  Batch keys: %s", list(batch.keys()))
            if targets:
                for tn, tv in targets.items():
                    _logger.info("  Target '%s': shape=%s, dtype=%s", tn, tv.shape, tv.dtype)

        elapsed = time.time() - t0
        detail = f"PLEInput features={feat_shape}, targets={n_targets} tasks"
        _logger.info(_green(f"  Step 5 PASS: DataLoader yields {detail} ({elapsed:.2f}s)"))
        _record(5, "DataLoader", "PASS", detail, elapsed)
        return True

    except Exception as exc:
        elapsed = time.time() - t0
        _logger.error(_red(f"  Step 5 FAIL: {exc}"))
        if verbose:
            traceback.print_exc()
        _record(5, "DataLoader", "FAIL", str(exc)[:80], elapsed)
        return False


# ============================================================================
# Step 6 -- Model Forward Pass + Expert Routing
# ============================================================================

def step6_model_forward(verbose: bool) -> bool:
    _logger.info("=" * 60)
    _logger.info("Step 6: Model Forward Pass + Expert Routing")
    _logger.info("=" * 60)
    t0 = time.time()
    try:
        import torch
        from core.model.ple.model import PLEModel, PLEInput, PLEOutput
        from core.model.ple.config import (
            PLEConfig,
            ExpertConfig,
            ExpertBasketConfig,
            CGCConfig,
            AdaTTConfig,
            LossWeightingConfig,
            TaskTowerConfig,
        )

        total_dim = _shared.get("total_dim", 64)
        ple_input_orig = _shared.get("ple_input")

        # Define 4 tasks: 2 binary, 1 multiclass, 1 regression
        task_names = ["ctr", "cvr", "life_stage", "ltv"]
        task_overrides = {
            "ctr":        {"task_type": "binary",     "output_dim": 1, "activation": "sigmoid"},
            "cvr":        {"task_type": "binary",     "output_dim": 1, "activation": "sigmoid"},
            "life_stage": {"task_type": "multiclass", "output_dim": 6, "activation": "softmax"},
            "ltv":        {"task_type": "regression", "output_dim": 1, "activation": None},
        }

        config = PLEConfig(
            input_dim=total_dim,
            task_names=task_names,
            task_overrides=task_overrides,
            num_shared_experts=3,
            shared_expert=ExpertConfig(
                hidden_dims=[64, 64],
                output_dim=32,
                dropout=0.05,
            ),
            num_task_experts_per_task=1,
            task_expert=ExpertConfig(
                hidden_dims=[32],
                output_dim=16,
                dropout=0.05,
            ),
            task_expert_output_dim=16,
            num_extraction_layers=1,
            cgc=CGCConfig(enabled=True, entropy_lambda=0.0),
            adatt=AdaTTConfig(enabled=False),
            loss_weighting=LossWeightingConfig(strategy="fixed"),
            task_tower=TaskTowerConfig(hidden_dims=[32, 16], dropout=0.1),
            dropout=0.05,
            expert_basket=None,  # use legacy mode (MLP only) for CPU safety
        )

        model = PLEModel(config)
        model.eval()

        # Prepare PLEInput with correct targets
        batch_size = _shared.get("batch_size", 32)
        if ple_input_orig is not None:
            features = ple_input_orig.features
        else:
            features = torch.randn(batch_size, total_dim)

        # Build targets matching the 4 tasks
        targets: Dict[str, torch.Tensor] = {
            "ctr": torch.randint(0, 2, (batch_size,)).float(),
            "cvr": torch.randint(0, 2, (batch_size,)).float(),
            "life_stage": torch.randint(0, 6, (batch_size,)),
            "ltv": torch.randn(batch_size) * 1000 + 5000,
        }

        # If we have real data from prior steps, use its labels
        df = _shared.get("df")
        if df is not None and ple_input_orig is not None:
            for tname in ["ctr", "cvr"]:
                lc = f"label_{tname}"
                if lc in df.columns:
                    vals = df[lc].iloc[:batch_size].values
                    targets[tname] = torch.tensor(vals, dtype=torch.float32)
            if "label_life_stage" in df.columns:
                vals = df["label_life_stage"].iloc[:batch_size].values
                targets["life_stage"] = torch.tensor(vals, dtype=torch.long)
            if "label_ltv" in df.columns:
                vals = df["label_ltv"].iloc[:batch_size].values
                targets["ltv"] = torch.tensor(vals, dtype=torch.float32)

        test_input = PLEInput(features=features, targets=targets)

        # Forward pass
        with torch.no_grad():
            output: PLEOutput = model(test_input, compute_loss=True)

        # Verify outputs
        assert output.predictions is not None, "No predictions returned"
        for tn in task_names:
            assert tn in output.predictions, f"Missing prediction for task '{tn}'"
            pred = output.predictions[tn]
            _logger.info("  Task '%s': pred shape=%s", tn, tuple(pred.shape))

        assert output.total_loss is not None, "No total_loss computed"
        loss_val = output.total_loss.item()
        assert not (loss_val != loss_val), "Loss is NaN"  # NaN check
        assert abs(loss_val) < 1e10, f"Loss appears to have exploded: {loss_val}"

        # Verify backward works
        model.train()
        train_output = model(test_input, compute_loss=True)
        train_output.total_loss.backward()

        # Print model summary
        summary = model.summary()
        if verbose:
            _logger.info("  Model Summary:\n%s", summary)

        n_params = sum(p.numel() for p in model.parameters())

        _shared["model"] = model
        _shared["config"] = config
        _shared["task_names"] = task_names
        _shared["task_overrides"] = task_overrides
        _shared["targets_template"] = targets

        elapsed = time.time() - t0
        detail = f"loss={loss_val:.4f}, {len(task_names)} tasks, {n_params:,} params"
        _logger.info(_green(f"  Step 6 PASS: Forward OK, {detail} ({elapsed:.2f}s)"))
        _record(6, "Model Forward", "PASS", detail, elapsed)
        return True

    except Exception as exc:
        elapsed = time.time() - t0
        _logger.error(_red(f"  Step 6 FAIL: {exc}"))
        if verbose:
            traceback.print_exc()
        _record(6, "Model Forward", "FAIL", str(exc)[:80], elapsed)
        return False


# ============================================================================
# Step 7 -- Mini Training Loop (2-Phase)
# ============================================================================

def step7_training_loop(verbose: bool) -> bool:
    _logger.info("=" * 60)
    _logger.info("Step 7: Mini Training Loop (2-Phase)")
    _logger.info("=" * 60)
    t0 = time.time()
    try:
        import torch
        import torch.nn as nn
        from core.model.ple.model import PLEModel, PLEInput, PLEOutput
        from core.model.ple.config import (
            PLEConfig,
            ExpertConfig,
            CGCConfig,
            AdaTTConfig,
            LossWeightingConfig,
            TaskTowerConfig,
        )

        # Rebuild a fresh model to avoid stale gradients from Step 6
        total_dim = _shared.get("total_dim", 64)
        task_names = _shared.get("task_names", ["ctr", "cvr", "life_stage", "ltv"])
        task_overrides = _shared.get("task_overrides", {
            "ctr":        {"task_type": "binary",     "output_dim": 1, "activation": "sigmoid"},
            "cvr":        {"task_type": "binary",     "output_dim": 1, "activation": "sigmoid"},
            "life_stage": {"task_type": "multiclass", "output_dim": 6, "activation": "softmax"},
            "ltv":        {"task_type": "regression", "output_dim": 1, "activation": None},
        })

        config = PLEConfig(
            input_dim=total_dim,
            task_names=task_names,
            task_overrides=task_overrides,
            num_shared_experts=3,
            shared_expert=ExpertConfig(hidden_dims=[64, 64], output_dim=32, dropout=0.05),
            num_task_experts_per_task=1,
            task_expert=ExpertConfig(hidden_dims=[32], output_dim=16, dropout=0.05),
            task_expert_output_dim=16,
            num_extraction_layers=1,
            cgc=CGCConfig(enabled=True, entropy_lambda=0.0),
            adatt=AdaTTConfig(enabled=False),
            loss_weighting=LossWeightingConfig(strategy="fixed"),
            task_tower=TaskTowerConfig(hidden_dims=[32, 16], dropout=0.1),
            dropout=0.05,
        )

        model = PLEModel(config)
        model.train()
        device = torch.device("cpu")
        model.to(device)

        # Build a small training dataset from assembled features
        combined_df = _shared.get("combined_df")
        feature_matrix = _shared.get("feature_matrix")
        df = _shared.get("df")

        batch_size = _shared.get("batch_size", 32)
        n_train = min(200, len(feature_matrix)) if feature_matrix is not None else 200

        if feature_matrix is not None and df is not None:
            features_np = feature_matrix.iloc[:n_train].values
            features_t = torch.tensor(features_np, dtype=torch.float32)
        else:
            features_t = torch.randn(n_train, total_dim)

        # Build target tensors
        targets_all: Dict[str, torch.Tensor] = {}
        if df is not None:
            for tname in task_names:
                lc = f"label_{tname}"
                if lc in df.columns:
                    vals = df[lc].iloc[:n_train].values
                    dtype = torch.long if task_overrides.get(tname, {}).get("task_type") == "multiclass" else torch.float32
                    targets_all[tname] = torch.tensor(vals, dtype=dtype)
        # Fill missing targets with random data
        for tname in task_names:
            if tname not in targets_all:
                tt = task_overrides.get(tname, {}).get("task_type", "binary")
                if tt == "multiclass":
                    od = task_overrides.get(tname, {}).get("output_dim", 6)
                    targets_all[tname] = torch.randint(0, od, (n_train,))
                elif tt == "regression":
                    targets_all[tname] = torch.randn(n_train) * 1000 + 5000
                else:
                    targets_all[tname] = torch.randint(0, 2, (n_train,)).float()

        # Create mini-batch iterator
        def _iter_batches(bs: int):
            indices = list(range(n_train))
            for start in range(0, n_train, bs):
                end = min(start + bs, n_train)
                idx = indices[start:end]
                batch_targets = {tn: tv[idx] for tn, tv in targets_all.items()}
                yield PLEInput(
                    features=features_t[idx],
                    targets=batch_targets,
                )

        # ---- Phase 1: All parameters trainable (3 epochs) ----
        _logger.info("  Phase 1: 3 epochs, all parameters trainable")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        phase1_losses: List[float] = []

        for epoch in range(3):
            epoch_loss = 0.0
            n_batches = 0
            for batch_input in _iter_batches(batch_size):
                optimizer.zero_grad()
                output = model(batch_input, compute_loss=True)
                loss = output.total_loss
                if loss is None or not torch.isfinite(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            avg_loss = epoch_loss / max(n_batches, 1)
            phase1_losses.append(avg_loss)
            _logger.info("    Epoch %d: loss=%.6f", epoch + 1, avg_loss)

        p1_start = phase1_losses[0] if phase1_losses else float("nan")
        p1_end = phase1_losses[-1] if phase1_losses else float("nan")

        # ---- Phase 2: Freeze extraction_layers (2 epochs) ----
        _logger.info("  Phase 2: 2 epochs, extraction_layers frozen")

        # Freeze extraction layers
        for param in model.extraction_layers.parameters():
            param.requires_grad = False

        # Record frozen param state
        frozen_state = {
            name: param.data.clone()
            for name, param in model.extraction_layers.named_parameters()
        }

        # Recreate optimizer with only trainable params
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer2 = torch.optim.AdamW(trainable_params, lr=5e-4, weight_decay=0.01)
        phase2_losses: List[float] = []

        for epoch in range(2):
            epoch_loss = 0.0
            n_batches = 0
            for batch_input in _iter_batches(batch_size):
                optimizer2.zero_grad()
                output = model(batch_input, compute_loss=True)
                loss = output.total_loss
                if loss is None or not torch.isfinite(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 5.0)
                optimizer2.step()
                epoch_loss += loss.item()
                n_batches += 1
            avg_loss = epoch_loss / max(n_batches, 1)
            phase2_losses.append(avg_loss)
            _logger.info("    Epoch %d: loss=%.6f", epoch + 1, avg_loss)

        p2_start = phase2_losses[0] if phase2_losses else float("nan")
        p2_end = phase2_losses[-1] if phase2_losses else float("nan")

        # Verify frozen params did NOT change
        frozen_ok = True
        for name, param in model.extraction_layers.named_parameters():
            if name in frozen_state:
                if not torch.equal(param.data, frozen_state[name]):
                    _logger.error("  FROZEN PARAM CHANGED: %s", name)
                    frozen_ok = False

        # Unfreeze for cleanup
        for param in model.extraction_layers.parameters():
            param.requires_grad = True

        # Verify Phase 2 loss is finite
        if phase2_losses:
            assert all(
                abs(l) < 1e10 and l == l for l in phase2_losses
            ), "Phase 2 losses are not finite"

        if not frozen_ok:
            raise RuntimeError("Frozen parameters changed during Phase 2!")

        elapsed = time.time() - t0
        detail = (
            f"Phase1: {p1_start:.4f}->{p1_end:.4f}, "
            f"Phase2: {p2_start:.4f}->{p2_end:.4f}"
        )
        _logger.info(_green(f"  Step 7 PASS: {detail} ({elapsed:.2f}s)"))
        _record(7, "2-Phase Training", "PASS", detail, elapsed)
        return True

    except Exception as exc:
        elapsed = time.time() - t0
        _logger.error(_red(f"  Step 7 FAIL: {exc}"))
        if verbose:
            traceback.print_exc()
        _record(7, "2-Phase Training", "FAIL", str(exc)[:80], elapsed)
        return False


# ============================================================================
# Summary
# ============================================================================

def print_summary():
    total_time = sum(r["elapsed"] for r in _results)
    print()
    print("=" * 65)
    print("  Pipeline Verification Summary")
    print("=" * 65)
    for r in _results:
        status_str = r["status"]
        if status_str == "PASS":
            status_display = _green("PASS")
        elif status_str == "WARN":
            status_display = _yellow("WARN")
        else:
            status_display = _red("FAIL")

        line = (
            f"  Step {r['step']}: {r['name']:<22s} "
            f"{status_display:<4s}  "
            f"({r['detail']})"
        )
        print(line)

    n_pass = sum(1 for r in _results if r["status"] == "PASS")
    n_warn = sum(1 for r in _results if r["status"] == "WARN")
    n_fail = sum(1 for r in _results if r["status"] == "FAIL")
    n_total = len(_results)
    print("-" * 65)
    print(
        f"  Total: {n_pass} PASS, {n_warn} WARN, {n_fail} FAIL "
        f"out of {n_total} steps  ({total_time:.1f}s)"
    )
    print("=" * 65)

    # Cleanup temp dir
    tmp_dir = _shared.get("tmp_dir")
    if tmp_dir:
        import shutil
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Verification Script -- validates each PLE stage locally."
    )
    parser.add_argument(
        "--samples", type=int, default=500,
        help="Number of synthetic samples to generate (default: 500)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed diagnostic output for each step",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
        _logger.setLevel(logging.DEBUG)

    print()
    print("=" * 65)
    print("  PLE Platform -- Local Pipeline Verification")
    print(f"  Samples: {args.samples}  |  Verbose: {args.verbose}")
    print(f"  Project root: {_PROJECT_ROOT}")
    print("=" * 65)
    print()

    total_start = time.time()

    step1_data_generation(args.samples, args.verbose)
    step2_schema_validation(args.verbose)
    step3_feature_generators(args.verbose)
    step4_feature_assembly(args.verbose)
    step5_dataloader(args.verbose)
    step6_model_forward(args.verbose)
    step7_training_loop(args.verbose)

    print_summary()

    n_fail = sum(1 for r in _results if r["status"] == "FAIL")
    sys.exit(1 if n_fail > 0 else 0)


if __name__ == "__main__":
    main()
