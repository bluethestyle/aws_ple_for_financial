#!/usr/bin/env python3
"""
Distillation Pipeline Entry Point.

Runs on SageMaker as a Processing Job or Training Job.
Expects teacher checkpoint + training data on S3.

Usage:
    python scripts/run_distillation.py \
        --teacher-checkpoint s3://bucket/model.pt \
        --data-path s3://bucket/data/train/ \
        --output-dir /opt/ml/model \
        --config configs/pipeline.yaml \
        --dataset configs/datasets/santander.yaml \
        --soft-label-path s3://bucket/soft_labels/

    # Or skip soft label generation (use pre-computed):
    python scripts/run_distillation.py \
        --soft-label-path s3://bucket/soft_labels/labels.parquet \
        --skip-soft-label-gen \
        --data-path s3://bucket/data/train/ \
        --output-dir /opt/ml/model
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("run_distillation")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="PLE -> LGBM Knowledge Distillation Pipeline"
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=str,
        required=False,
        default=None,
        help="S3 or local path to the teacher PLE checkpoint (.pt)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="S3 or local path to training data (parquet)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pipeline.yaml",
        help="Common pipeline YAML path (or legacy single-file path)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Dataset-specific YAML to deep-merge on top of --config "
             "(e.g. configs/datasets/santander.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/opt/ml/model",
        help="Directory to save student models",
    )
    parser.add_argument(
        "--soft-label-path",
        type=str,
        default="",
        help="Path to save/load soft labels parquet",
    )
    parser.add_argument(
        "--skip-soft-label-gen",
        action="store_true",
        help="Skip soft label generation; load from --soft-label-path instead",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Distillation temperature (higher = softer labels). Overrides YAML config.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Hard label weight (1-alpha = soft label weight). Overrides YAML config.",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Specific tasks to distill (default: all non-contrastive tasks)",
    )
    parser.add_argument(
        "--skip-fidelity-gate",
        action="store_true",
        help="Continue even if fidelity check fails (for testing)",
    )
    # LGBM hyperparameter CLI overrides (take precedence over YAML config)
    parser.add_argument(
        "--lgbm-num-leaves",
        type=int,
        default=None,
        help="LGBM max number of leaves per tree. Overrides YAML distillation.lgbm_params.num_leaves.",
    )
    parser.add_argument(
        "--lgbm-learning-rate",
        type=float,
        default=None,
        help="LGBM learning rate. Overrides YAML distillation.lgbm_params.learning_rate.",
    )
    parser.add_argument(
        "--lgbm-n-estimators",
        type=int,
        default=None,
        help="LGBM number of boosting rounds. Overrides YAML distillation.lgbm_params.n_estimators.",
    )
    return parser.parse_args()


def load_calibrated_model(task_dir: str):
    """Load a saved calibration model from a task directory.

    Args:
        task_dir: Path to the per-task output directory (e.g. ``output/churn/``).

    Returns:
        The deserialized calibrator object (``CalibratedClassifierCV`` for binary
        tasks, or a ``dict`` with ``"lgbm"`` and ``"bias_corrector"`` keys for
        regression tasks), or ``None`` if no calibrator was saved for this task.
    """
    calib_path = Path(task_dir) / "calibrator.joblib"
    if calib_path.exists():
        obj = joblib.load(calib_path)
        logger.info("Calibrator loaded from %s", calib_path)
        return obj
    return None


def main() -> None:
    """Run the distillation pipeline."""
    args = parse_args()

    # Import here to avoid import errors if torch is not available
    # in LGBM-only mode
    import yaml as _yaml
    from core.pipeline.config import load_config, load_merged_config
    from core.training.student_trainer import StudentConfig, StudentTrainer

    # Load pipeline config (supports split-config pattern).
    # Both load_config() and the raw dict use the same merged result so that
    # dataset-specific overrides are visible in the distillation section.
    _dataset_path = args.dataset if getattr(args, "dataset", "") else ""
    if _dataset_path and Path(_dataset_path).exists():
        pipeline_config = load_config(args.config, dataset_path=_dataset_path)
        _raw_yaml: dict = load_merged_config(args.config, _dataset_path)
        logger.info("Distillation config merged: %s + %s", args.config, _dataset_path)
    else:
        pipeline_config = load_config(args.config)
        with open(args.config, encoding="utf-8") as _f:
            _raw_yaml = _yaml.safe_load(_f) or {}
    _distillation_cfg: dict = _raw_yaml.get("distillation", {})

    # Load data via PyArrow (zero-copy, no pandas intermediate)
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    logger.info("Loading data from %s", args.data_path)
    data_path = Path(args.data_path)
    if data_path.is_dir():
        parquet_files = sorted(data_path.glob("*.parquet"))
        if not parquet_files:
            logger.error("No parquet files found in %s", args.data_path)
            sys.exit(1)
        table = pq.read_table(data_path)
        logger.info("Loaded %d files, %d rows total", len(parquet_files), table.num_rows)
    else:
        table = pq.read_table(str(data_path))
        logger.info("Loaded %d rows", table.num_rows)

    # Step 1.5: Quality gate — block pipeline on critical data issues
    # Passes PyArrow Table directly; no to_pandas() needed (CLAUDE.md 3.3)
    logger.info("Running quality gate on loaded data...")
    from core.data.quality_gate import QualityGate, QualityGateError

    quality_gate = QualityGate()
    try:
        gate_result = quality_gate.evaluate_and_block(table, source_name="distillation_train")
        logger.info(
            "Quality gate PASSED (verdict=%s, checks=%d)",
            gate_result.verdict.value, len(gate_result.checks),
        )
    except QualityGateError as exc:
        logger.error("Quality gate FAILED: %s", exc)
        # Save gate report for debugging before exit
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        gate_report = quality_gate.get_report(exc.result)
        with open(output_path / "quality_gate_report.json", "w") as f:
            json.dump(gate_report, f, indent=2, default=str)
        sys.exit(1)

    # Separate features and labels using PyArrow (no pandas)
    label_cols = {t.label_col for t in pipeline_config.tasks}
    id_cols = set(pipeline_config.features.id_cols)
    exclude_cols = label_cols | id_cols
    table_cols_set = set(table.column_names)

    # Use teacher checkpoint's feature schema to guarantee column order
    _teacher_schema_cols: list = []
    if args.teacher_checkpoint:
        _schema_path = Path(args.teacher_checkpoint).parent / "config.json"
        if _schema_path.exists():
            with open(_schema_path) as _sf:
                _schema = json.load(_sf)
            _teacher_schema_cols = _schema.get("feature_schema", {}).get("columns", [])
            logger.info(
                "Teacher schema loaded: %d features from %s",
                len(_teacher_schema_cols), _schema_path,
            )

    if _teacher_schema_cols:
        feature_cols = [c for c in _teacher_schema_cols if c in table_cols_set]
        missing = [c for c in _teacher_schema_cols if c not in table_cols_set]
        if missing:
            logger.warning("Missing %d columns from teacher schema (will fill 0)", len(missing))
    else:
        import pyarrow.types as pat
        feature_cols = [
            c for c in table.column_names
            if c not in exclude_cols
            and (pat.is_floating(table.schema.field(c).type)
                 or pat.is_integer(table.schema.field(c).type))
        ]

    logger.info(
        "Features: %d columns, Labels: %d tasks, IDs: %d columns",
        len(feature_cols), len(label_cols), len(id_cols),
    )

    # Extract features as numpy via PyArrow (zero-copy where possible)
    feature_arrays = []
    for c in feature_cols:
        col = table.column(c)
        arr = col.to_numpy(zero_copy_only=False).astype(np.float32)
        np.nan_to_num(arr, copy=False)
        feature_arrays.append(arr)
    features = np.column_stack(feature_arrays) if feature_arrays else np.empty((table.num_rows, 0), dtype=np.float32)
    del feature_arrays
    logger.info("Features array: %s, %.1f MB", features.shape, features.nbytes / 1024**2)

    # Extract hard labels via PyArrow
    hard_labels: dict[str, np.ndarray] = {}
    for t in pipeline_config.tasks:
        if t.label_col in table_cols_set:
            hard_labels[t.name] = table.column(t.label_col).to_numpy(zero_copy_only=False)

    # Release Arrow table
    del table

    # Build student config — YAML distillation section is the base,
    # CLI args are optional overrides (None means "not provided by user").
    # Priority: CLI arg > YAML distillation section > StudentConfig default.

    # 1. Start from the full distillation dict so that lgbm_params,
    #    task_lgbm_overrides, and any other fields are picked up from YAML.
    _student_dict = dict(_distillation_cfg)
    # YAML uses shorthand "lgbm:" but StudentConfig expects "lgbm_params:"
    if "lgbm" in _student_dict and "lgbm_params" not in _student_dict:
        _student_dict["lgbm_params"] = _student_dict.pop("lgbm")

    # 2. Mandatory fields that come from other CLI args / top-level config.
    _student_dict["teacher_checkpoint"] = args.teacher_checkpoint or _distillation_cfg.get(
        "teacher_checkpoint", ""
    )
    _student_dict["soft_label_path"] = args.soft_label_path or _distillation_cfg.get(
        "soft_label_path", ""
    )
    _student_dict["student_output_dir"] = args.output_dir

    # 3. CLI scalar overrides (only applied when the user explicitly passed them).
    if args.temperature is not None:
        _student_dict["temperature"] = args.temperature
    if args.alpha is not None:
        _student_dict["alpha"] = args.alpha
    if args.tasks is not None:
        _student_dict["enabled_tasks"] = args.tasks

    # 4. CLI LGBM param overrides — merge into the lgbm_params sub-dict so
    #    that params NOT specified on the CLI still come from YAML.
    _lgbm_overrides: dict = {}
    if args.lgbm_num_leaves is not None:
        _lgbm_overrides["num_leaves"] = args.lgbm_num_leaves
    if args.lgbm_learning_rate is not None:
        _lgbm_overrides["learning_rate"] = args.lgbm_learning_rate
    if args.lgbm_n_estimators is not None:
        _lgbm_overrides["n_estimators"] = args.lgbm_n_estimators

    if _lgbm_overrides:
        # Merge: YAML lgbm_params (if any) updated with CLI overrides.
        _base_lgbm = dict(_distillation_cfg.get("lgbm_params", {}))
        _base_lgbm.update(_lgbm_overrides)
        _student_dict["lgbm_params"] = _base_lgbm

    student_config = StudentConfig.from_dict(_student_dict)
    logger.info(
        "StudentConfig: temperature=%.1f alpha=%.2f lgbm_params=%s task_lgbm_overrides=%s",
        student_config.temperature,
        student_config.alpha,
        student_config.lgbm_params,
        list(student_config.task_lgbm_overrides.keys()) or "(none)",
    )

    # Create trainer
    trainer = StudentTrainer(
        config=student_config,
        task_specs=pipeline_config.tasks,
        feature_columns=feature_cols,
    )

    # Step 2: Generate or load soft labels
    if args.skip_soft_label_gen and args.soft_label_path:
        logger.info(
            "Loading pre-computed soft labels from %s", args.soft_label_path
        )
        trainer.load_soft_labels(args.soft_label_path)

    elif args.teacher_checkpoint:
        logger.info("Generating soft labels from teacher...")
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        # Load teacher with pipeline config fallback for expert basket reconstruction
        # (checkpoints may lack 'config' key — pipeline YAML fills the gap)
        trainer.load_teacher(
            checkpoint_path=args.teacher_checkpoint,
            pipeline_config=_raw_yaml,
        )

        # Build a simple DataLoader for teacher inference.
        # This yields raw tensor tuples; the StudentTrainer wraps them
        # into PLEInput-compatible dicts internally.
        features_tensor = torch.tensor(features, dtype=torch.float32)
        dataset = TensorDataset(features_tensor)
        loader = DataLoader(dataset, batch_size=4096, shuffle=False)

        trainer.generate_soft_labels(
            data_loader=loader,
            save_path=args.soft_label_path if args.soft_label_path else None,
        )
    else:
        logger.error(
            "Either --teacher-checkpoint or "
            "--skip-soft-label-gen + --soft-label-path required"
        )
        sys.exit(1)

    # ================================================================
    # Step 2.5: Teacher performance threshold — adaptive distillation
    #
    # Evaluate teacher soft labels against hard labels to determine
    # per-task distillation viability. Tasks where the teacher
    # performs near random are better served by direct hard-label
    # LGBM training (no distillation) — this is a Model Risk
    # Management safeguard.
    #
    # Threshold: teacher must exceed 2x random baseline.
    #   binary:     AUC > 0.60
    #   multiclass: F1_macro > 2/num_classes
    #   regression: R^2 > 0.05
    # ================================================================
    from sklearn.metrics import roc_auc_score, f1_score, r2_score

    soft_labels = trainer.get_soft_labels()
    distill_tasks: list[str] = []
    hardlabel_tasks: list[str] = []
    skip_tasks: list[str] = []  # floor threshold — no LGBM at all, Layer 3 only

    # Floor thresholds from config (below this → SKIP entirely)
    _thresh_cfg = _distillation_cfg.get("teacher_threshold", {})
    _floor_binary_auc = _thresh_cfg.get("floor_binary_auc", 0.52)
    _floor_mc_f1_ratio = _thresh_cfg.get("floor_multiclass_f1_ratio", 1.0)
    _floor_r2 = _thresh_cfg.get("floor_regression_r2", 0.01)

    logger.info("=" * 60)
    logger.info("Teacher threshold check (2x random baseline + floor)")
    logger.info("=" * 60)

    for t in pipeline_config.tasks:
        task_name = t.name
        task_type = t.type
        teacher_soft = soft_labels.get(task_name)
        hard = hard_labels.get(task_name)

        if teacher_soft is None or hard is None:
            logger.warning("  %s: no soft/hard labels — skipping", task_name)
            continue

        teacher_np = teacher_soft.numpy() if hasattr(teacher_soft, 'numpy') else np.array(teacher_soft)
        hard_np = hard if isinstance(hard, np.ndarray) else np.array(hard)

        viable = False
        metric_str = ""

        try:
            if task_type == "binary":
                preds = teacher_np.flatten()
                labels = hard_np.flatten()
                valid = (labels == 0) | (labels == 1)
                if valid.sum() > 10 and len(set(labels[valid].tolist())) == 2:
                    auc = roc_auc_score(labels[valid], preds[valid])
                    viable = auc > 0.60
                    metric_str = f"AUC={auc:.4f} (threshold=0.60)"

            elif task_type == "multiclass":
                n_classes = getattr(t, 'num_classes', None) or int(hard_np.max()) + 1
                threshold = 2.0 / n_classes
                if teacher_np.ndim > 1 and teacher_np.shape[-1] > 1:
                    pred_classes = teacher_np.argmax(axis=-1)
                else:
                    pred_classes = teacher_np.flatten().astype(int)
                true_classes = hard_np.flatten().astype(int)
                valid = true_classes >= 0
                if valid.sum() > 10:
                    f1 = f1_score(true_classes[valid], pred_classes[valid],
                                  average='macro', zero_division=0)
                    viable = f1 > threshold
                    metric_str = f"F1={f1:.4f} (threshold={threshold:.4f}, {n_classes}-class)"

            elif task_type == "regression":
                preds = teacher_np.flatten()
                labels = hard_np.flatten()
                if np.var(labels) > 1e-10:
                    r2 = r2_score(labels, preds)
                    viable = r2 > 0.05
                    metric_str = f"R2={r2:.4f} (threshold=0.05)"

        except Exception as e:
            logger.debug("  %s: threshold check error: %s", task_name, e)

        # 3-way routing: DISTILL / DIRECT / SKIP
        below_floor = False
        try:
            if task_type == "binary":
                _auc_val = float(metric_str.split("=")[1].split(" ")[0]) if "AUC=" in metric_str else 0.0
                below_floor = _auc_val <= _floor_binary_auc
            elif task_type == "multiclass":
                _f1_val = float(metric_str.split("=")[1].split(" ")[0]) if "F1=" in metric_str else 0.0
                n_classes = getattr(t, 'num_classes', None) or int(hard_np.max()) + 1
                below_floor = _f1_val <= (1.0 / n_classes) * _floor_mc_f1_ratio
            elif task_type == "regression":
                _r2_val = float(metric_str.split("=")[1].split(" ")[0]) if "R2=" in metric_str else 0.0
                below_floor = _r2_val <= _floor_r2
        except Exception:
            pass

        if viable:
            distill_tasks.append(task_name)
            logger.info("  [DISTILL] %s: %s", task_name, metric_str)
        elif below_floor:
            skip_tasks.append(task_name)
            logger.info("  [SKIP]    %s: %s — below floor, Layer 3 rule-only (no LGBM)",
                        task_name, metric_str)
        else:
            hardlabel_tasks.append(task_name)
            logger.info("  [DIRECT]  %s: %s — below threshold, using hard labels",
                        task_name, metric_str)

    logger.info("Distillation: %d tasks, Direct LGBM: %d tasks, SKIP: %d tasks",
                len(distill_tasks), len(hardlabel_tasks), len(skip_tasks))
    logger.info("=" * 60)

    # Step 3: Train LGBM students — adaptive per task
    logger.info("Training LGBM student models...")

    # 3a: Distilled tasks (soft + hard labels via StudentTrainer)
    if distill_tasks:
        logger.info("  [DISTILL] %d tasks: %s", len(distill_tasks), distill_tasks)
    students = trainer.train_students(features, hard_labels)

    # 3b: Hard-label-only tasks — retrain with alpha=1.0 (pure hard label)
    if hardlabel_tasks:
        logger.info("  [DIRECT] %d tasks: %s — retraining with hard labels only",
                    len(hardlabel_tasks), hardlabel_tasks)
        import lightgbm as lgb

        for task_name in hardlabel_tasks:
            t_spec = next((t for t in pipeline_config.tasks if t.name == task_name), None)
            if t_spec is None or task_name not in hard_labels:
                continue

            y = hard_labels[task_name]
            lgbm_params = dict(student_config.lgbm_params)
            task_overrides = student_config.task_lgbm_overrides.get(task_name, {})
            lgbm_params.update(task_overrides)

            if t_spec.type == "binary":
                lgbm_params.setdefault("objective", "binary")
                lgbm_params.setdefault("metric", "auc")
            elif t_spec.type == "multiclass":
                n_classes = getattr(t_spec, 'num_classes', None) or int(y.max()) + 1
                lgbm_params["objective"] = "multiclass"
                lgbm_params["num_class"] = n_classes
                lgbm_params.setdefault("metric", "multi_logloss")
            else:
                lgbm_params.setdefault("objective", "regression")
                lgbm_params.setdefault("metric", "mae")

            lgbm_params.setdefault("verbosity", -1)
            n_estimators = lgbm_params.pop("n_estimators", 300)

            ds = lgb.Dataset(features, label=y)
            model = lgb.train(
                lgbm_params, ds,
                num_boost_round=n_estimators,
            )
            students[task_name] = model
            logger.info("    %s: direct LGBM trained (%d rounds)", task_name, n_estimators)

    # ================================================================
    # Step 3.5: Calibration — Platt scaling for probability-critical tasks
    #
    # Tasks listed in distillation.calibration.tasks need accurate
    # probability outputs (not just ranking). Apply post-hoc calibration
    # using sklearn CalibratedClassifierCV on the validation split.
    # ================================================================
    _calib_cfg = _distillation_cfg.get("calibration", {})
    _calib_tasks = set(_calib_cfg.get("tasks", []))
    _calib_method = _calib_cfg.get("method", "platt")
    calibrated_models: dict[str, Any] = {}

    if _calib_cfg.get("enabled", False) and _calib_tasks:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.base import BaseEstimator, ClassifierMixin

        class _LGBMProbWrapper(BaseEstimator, ClassifierMixin):
            """Wrap a trained LGBM model for sklearn calibration."""
            def __init__(self, lgbm_model):
                self.lgbm_model = lgbm_model
                self.classes_ = np.array([0, 1])
            def fit(self, X, y):
                return self
            def predict_proba(self, X):
                raw = self.lgbm_model.predict(X)
                return np.column_stack([1 - raw, raw])

        # Use last 20% of data as calibration set (separate from training)
        n_total = len(features)
        calib_start = int(n_total * 0.8)
        X_calib = features[calib_start:]

        logger.info("=" * 60)
        logger.info("Calibration: %s method on %d tasks", _calib_method, len(_calib_tasks))
        logger.info("  Calibration set: %d samples (last 20%%)", len(X_calib))
        logger.info("=" * 60)

        _sklearn_method = "sigmoid" if _calib_method == "platt" else "isotonic"

        for task_name in _calib_tasks:
            if task_name not in students or task_name not in hard_labels:
                logger.warning("  %s: skipped (no model or labels)", task_name)
                continue

            t_spec = next((t for t in pipeline_config.tasks if t.name == task_name), None)
            if t_spec is None:
                continue

            y_calib = hard_labels[task_name][calib_start:]
            model = students[task_name]

            try:
                if t_spec.type == "binary":
                    wrapper = _LGBMProbWrapper(model)
                    calibrator = CalibratedClassifierCV(
                        wrapper, method=_sklearn_method, cv="prefit",
                    )
                    calibrator.fit(X_calib, y_calib)
                    calibrated_models[task_name] = calibrator
                    logger.info("  [CALIBRATED] %s: %s scaling applied", task_name, _calib_method)

                elif t_spec.type == "regression":
                    # For regression, calibration = bias correction via linear fit
                    raw_preds = model.predict(X_calib)
                    from sklearn.linear_model import LinearRegression
                    lr = LinearRegression()
                    lr.fit(raw_preds.reshape(-1, 1), y_calib)
                    calibrated_models[task_name] = {"lgbm": model, "bias_corrector": lr}
                    logger.info("  [CALIBRATED] %s: linear bias correction (slope=%.4f, intercept=%.4f)",
                                task_name, lr.coef_[0], lr.intercept_)

                else:
                    logger.info("  %s: multiclass calibration not implemented, skipped", task_name)

            except Exception as e:
                logger.warning("  %s: calibration failed: %s", task_name, e)

    # Step 4: Fidelity validation — teacher-student agreement check
    logger.info("Running fidelity validation (8 metrics per task)...")
    from core.training.distillation_validator import (
        DistillationValidator,
        ValidationCriteria,
    )

    # Build ValidationCriteria from config (distillation.fidelity section)
    _fidelity_cfg = _distillation_cfg.get("fidelity", {})
    _bin_cfg = _fidelity_cfg.get("binary", {})
    _mc_cfg = _fidelity_cfg.get("multiclass", {})
    _reg_cfg = _fidelity_cfg.get("regression", {})

    criteria = ValidationCriteria(
        max_auc_gap=_bin_cfg.get("max_auc_gap", 0.05),
        min_binary_agreement=_bin_cfg.get("min_agreement", 0.85),
        max_jsd=_bin_cfg.get("max_jsd", 0.10),
        min_ranking_corr=_bin_cfg.get("min_ranking_corr", 0.90),
        max_calibration_gap=_bin_cfg.get("max_calibration_gap", 0.05),
        min_multiclass_agreement=_mc_cfg.get("min_agreement", 0.70),
        max_f1_macro_gap=_mc_cfg.get("max_f1_macro_gap", 0.10),
        regression_quartile_agreement_min=_reg_cfg.get("min_quartile_agreement", 0.70),
        max_mae_gap=_reg_cfg.get("max_mae_gap", 0.05),
        max_rmse_gap=_reg_cfg.get("max_rmse_gap", 0.10),
    )
    validator = DistillationValidator(criteria=criteria)
    fidelity_results = []

    # Get teacher soft labels for comparison
    soft_labels = trainer.get_soft_labels()  # Dict[task_name, np.ndarray]

    for task_spec in pipeline_config.tasks:
        task_name = task_spec.name
        if task_name not in students:
            continue

        student_model = students[task_name]

        # Use calibrated model if available for this task
        if task_name in calibrated_models:
            calib = calibrated_models[task_name]
            if task_spec.type == "binary" and hasattr(calib, "predict_proba"):
                student_preds = calib.predict_proba(features)[:, 1]
            elif task_spec.type == "regression" and isinstance(calib, dict):
                raw = calib["lgbm"].predict(features)
                student_preds = calib["bias_corrector"].predict(raw.reshape(-1, 1))
            else:
                student_preds = student_model.predict(features)
        else:
            # Student predictions on the same training features.
            # Custom fobj -> predict() returns raw margins, not probabilities.
            # Apply sigmoid (binary) or softmax (multiclass) to match teacher scale.
            raw_preds = student_model.predict(features)

            if task_spec.type == "binary" and student_config.use_custom_objective:
                student_preds = 1.0 / (1.0 + np.exp(-raw_preds))
            elif task_spec.type == "multiclass" and student_config.use_custom_objective:
                n_classes = task_spec.num_classes
                raw_2d = raw_preds.reshape(-1, n_classes)
                exp_shifted = np.exp(raw_2d - raw_2d.max(axis=1, keepdims=True))
                student_preds = exp_shifted / exp_shifted.sum(axis=1, keepdims=True)
            else:
                student_preds = raw_preds

        # Teacher predictions (soft labels)
        teacher_preds = soft_labels.get(task_name)
        if teacher_preds is None:
            logger.warning("No soft labels for task %s, skipping fidelity", task_name)
            continue

        # Ground truth labels (for AUC, calibration)
        labels = hard_labels.get(task_name)

        try:
            result = validator.validate_task(
                task_name=task_name,
                task_type=task_spec.type,
                teacher_preds=teacher_preds,
                student_preds=student_preds,
                labels=labels,
            )
        except Exception as e:
            logger.warning("Fidelity validation failed for %s: %s", task_name, e)
            from core.training.distillation_validator import FidelityResult
            result = FidelityResult(
                task_name=task_name, task_type=task_spec.type, passed=False,
                metrics={}, failures=[str(e)],
            )
        fidelity_results.append(result)

        status = "PASS" if result.passed else "FAIL"
        logger.info(
            "  [%s] %s — metrics: %s%s",
            status, task_name,
            {k: round(v, 4) for k, v in result.metrics.items()},
            f" failures: {result.failures}" if result.failures else "",
        )

    # Check overall fidelity
    passed_count = sum(1 for r in fidelity_results if r.passed)
    failed_count = len(fidelity_results) - passed_count
    logger.info(
        "Fidelity summary: %d/%d tasks passed",
        passed_count, passed_count + failed_count,
    )

    if failed_count > 0:
        failed_tasks = [r.task_name for r in fidelity_results if not r.passed]
        logger.error(
            "Fidelity FAILED for %d task(s): %s — aborting pipeline",
            failed_count, failed_tasks,
        )
        # Save partial results for debugging before exit
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        fidelity_report = {
            "status": "FAILED",
            "passed": passed_count,
            "failed": failed_count,
            "failed_tasks": failed_tasks,
            "details": {
                r.task_name: {
                    "passed": r.passed,
                    "metrics": r.metrics,
                    "failures": r.failures,
                }
                for r in fidelity_results
            },
        }
        with open(output_path / "fidelity_report.json", "w") as f:
            json.dump(fidelity_report, f, indent=2, default=str)
        if not getattr(args, "skip_fidelity_gate", False):
            logger.error("Aborting pipeline due to fidelity failure.")
            sys.exit(1)
        else:
            logger.warning("--skip-fidelity-gate set, continuing despite failures.")

    # ================================================================
    # Step 4.5b: Drift monitoring — compare vs previous distillation
    #
    # SR 11-7 MRM safeguard: detect whether the newly distilled student
    # has shifted significantly from the previous version.  Alerts are
    # WARNING-only (non-blocking) so the pipeline never stops here.
    # ================================================================
    _drift_cfg = _distillation_cfg.get("drift_monitoring", {})
    if _drift_cfg.get("enabled", False):
        from core.training.distillation_drift import DistillationDriftMonitor

        logger.info("=" * 60)
        logger.info("Step 4.5b: Temporal drift monitoring (SR 11-7)")
        logger.info("=" * 60)

        _drift_monitor = DistillationDriftMonitor(config=_drift_cfg)
        _baseline_path = _drift_cfg.get("baseline_path", "outputs/distillation_baseline/")

        # Build current student prediction dict (reuse post-sigmoid/softmax predictions
        # consistent with fidelity validation for apples-to-apples comparison).
        _current_student_preds: dict = {}
        for _task_spec in pipeline_config.tasks:
            _tname = _task_spec.name
            if _tname not in students:
                continue
            _sm = students[_tname]
            if _tname in calibrated_models:
                _cal = calibrated_models[_tname]
                if _task_spec.type == "binary" and hasattr(_cal, "predict_proba"):
                    _current_student_preds[_tname] = _cal.predict_proba(features)[:, 1]
                elif _task_spec.type == "regression" and isinstance(_cal, dict):
                    _raw = _cal["lgbm"].predict(features)
                    _current_student_preds[_tname] = _cal["bias_corrector"].predict(
                        _raw.reshape(-1, 1)
                    )
                else:
                    _current_student_preds[_tname] = _sm.predict(features)
            else:
                _raw_preds = _sm.predict(features)
                if _task_spec.type == "binary" and student_config.use_custom_objective:
                    _current_student_preds[_tname] = 1.0 / (1.0 + np.exp(-_raw_preds))
                elif (
                    _task_spec.type == "multiclass"
                    and student_config.use_custom_objective
                ):
                    _nc = _task_spec.num_classes
                    _r2d = _raw_preds.reshape(-1, _nc)
                    _ex = np.exp(_r2d - _r2d.max(axis=1, keepdims=True))
                    # Scalar per sample: confidence of the argmax class
                    _current_student_preds[_tname] = (
                        _ex / _ex.sum(axis=1, keepdims=True)
                    ).max(axis=1)
                else:
                    _current_student_preds[_tname] = _raw_preds

        # Load previous baseline and run comparison
        _prev_preds = _drift_monitor.load_baseline(_baseline_path)

        if _prev_preds is not None:
            _drift_report = _drift_monitor.compare_versions(
                current_preds=_current_student_preds,
                previous_preds=_prev_preds,
                labels=hard_labels,
            )

            _n_alerts = len(_drift_report["alert_tasks"])
            if _drift_report["any_alert"]:
                logger.warning(
                    "Drift alerts on %d task(s): %s — MRM review recommended.",
                    _n_alerts,
                    _drift_report["alert_tasks"],
                )
            else:
                logger.info(
                    "Drift check PASSED: all %d tasks within thresholds.",
                    len(_drift_report["per_task"]),
                )

            # Persist drift report alongside other pipeline outputs
            _drift_output_path = Path(args.output_dir)
            _drift_output_path.mkdir(parents=True, exist_ok=True)
            with open(_drift_output_path / "drift_report.json", "w") as _drf:
                json.dump(
                    {
                        "any_alert": _drift_report["any_alert"],
                        "alert_tasks": _drift_report["alert_tasks"],
                        "thresholds": _drift_report["thresholds"],
                        "per_task": {
                            t: {
                                k: round(v, 6) if isinstance(v, float) else v
                                for k, v in m.items()
                            }
                            for t, m in _drift_report["per_task"].items()
                        },
                    },
                    _drf,
                    indent=2,
                    default=str,
                )
            logger.info("Drift report saved to %s/drift_report.json", args.output_dir)
        else:
            logger.info(
                "No previous baseline found — current predictions will become the first baseline."
            )

        # Always update the baseline with the current version's predictions
        _drift_monitor.save_baseline(_current_student_preds, _baseline_path)
        logger.info("Drift baseline updated at %s", _baseline_path)
    else:
        logger.info("Drift monitoring disabled (distillation.drift_monitoring.enabled=false).")

    # Step 4.5: Feature selection (IG dual-objective + LGBM gain pruning)
    # -----------------------------------------------------------------------
    # method is read from pipeline.yaml distillation.feature_selection.method:
    #   "ig_dual"   -- IG pred + explain scores, cumulative threshold selection
    #   "lgbm_gain" -- LGBM gain pruning only (original behaviour, default)
    #   "both"      -- ig_dual first, then LGBM zero-gain prune within selection
    # -----------------------------------------------------------------------
    logger.info("Running adaptive feature selection per task...")
    from core.training.feature_selector import (
        FeatureSelector,
        FeatureSelectionConfig,
        FeatureSelectionResult,
    )

    _fs_cfg_raw: dict = _distillation_cfg.get("feature_selection", {})
    _fs_method: str = _fs_cfg_raw.get("method", "lgbm_gain")
    _ig_alpha: float = float(_fs_cfg_raw.get("ig_alpha", 0.7))
    _cumulative_threshold: float = float(_fs_cfg_raw.get("cumulative_threshold", 0.95))
    _ig_sample_size: int = int(_fs_cfg_raw.get("ig_sample_size", 10000))

    logger.info(
        "Feature selection method=%s ig_alpha=%.2f cumulative_threshold=%.2f ig_sample_size=%d",
        _fs_method, _ig_alpha, _cumulative_threshold, _ig_sample_size,
    )

    fs_config = FeatureSelectionConfig(
        cumulative_threshold=_cumulative_threshold,
    )
    feature_selector = FeatureSelector(config=fs_config)

    # ------------------------------------------------------------------
    # Build explain-value map from feature_groups.yaml (config-driven).
    # Category scores can be overridden via
    # pipeline.yaml distillation.feature_selection.category_explain_scores.
    # ------------------------------------------------------------------
    _explain_scores: dict = {}
    if _fs_method in ("ig_dual", "both"):
        import yaml as _yaml_inner

        _fg_cfg_path = Path(args.config).parent / "feature_groups.yaml"
        _category_scores_from_cfg: dict = _fs_cfg_raw.get("category_explain_scores", {})

        _default_category_scores: dict = {
            "demographics":           1.0,
            "spending_pattern":       1.0,
            "transaction_behavior":   1.0,
            "temporal":               1.0,
            "economic_behavior":      1.0,
            "cross_domain":           0.9,
            "extended_services":      0.9,
            "domain_topology":        0.8,
            "customer_segment":       0.8,
            "temporal_state":         0.8,
            "merchant_structure":     0.9,
            "behavioral_state":       0.8,
            "graph_structure":        0.7,
            "model_insight":          0.3,
        }
        # YAML overrides take precedence over built-in defaults
        _category_scores: dict = {**_default_category_scores, **_category_scores_from_cfg}

        if _fg_cfg_path.exists():
            with open(_fg_cfg_path, encoding="utf-8") as _fgf:
                _fg_data: dict = _yaml_inner.safe_load(_fgf)
            for _grp in _fg_data.get("feature_groups", []):
                _interp = _grp.get("interpretation", {})
                _cat = _interp.get("category", "")
                _cat_score = _category_scores.get(_cat, 0.5)
                for _col in _grp.get("columns", []):
                    _explain_scores[_col] = _cat_score
            logger.info(
                "Explain-value map built: %d features from %s",
                len(_explain_scores), _fg_cfg_path,
            )
        else:
            logger.warning(
                "feature_groups.yaml not found at %s -- explain scores will be uniform",
                _fg_cfg_path,
            )

    # ------------------------------------------------------------------
    # Retrieve teacher model for IG (loaded by load_teacher above).
    # Falls back to None if teacher was not loaded in this run.
    # ------------------------------------------------------------------
    _teacher_model = getattr(trainer, "_teacher", None)

    feature_selections = {}
    _ig_raw_scores: dict = {}  # task_name -> top-50 dual scores, saved to summary

    for task_spec in pipeline_config.tasks:
        task_name = task_spec.name
        if task_name not in students:
            continue

        student_model = students[task_name]

        if _fs_method in ("ig_dual", "both") and _teacher_model is not None:
            # Stage 1: IG dual-objective selection
            try:
                ig_result = feature_selector.select_by_ig_dual(
                    model=_teacher_model,
                    features=features,
                    task_name=task_name,
                    feature_names=feature_cols,
                    n_samples=_ig_sample_size,
                    ig_alpha=_ig_alpha,
                    explain_scores=_explain_scores if _explain_scores else None,
                )
                _ig_raw_scores[task_name] = ig_result.feature_importances

                if _fs_method == "both":
                    # Stage 2: LGBM zero-gain pruning within IG selection
                    pruned_indices = feature_selector.prune_by_lgbm(
                        lgbm_model=student_model,
                        feature_names=feature_cols,
                        selected_indices=ig_result.selected_indices,
                    )
                    pruned_names = [feature_cols[i] for i in pruned_indices]
                    lgbm_gains = student_model.feature_importance(importance_type="gain")
                    selection_result = FeatureSelectionResult(
                        task_name=task_name,
                        original_count=len(feature_cols),
                        selected_count=len(pruned_indices),
                        reduction_pct=round(
                            (1 - len(pruned_indices) / len(feature_cols)) * 100, 1,
                        ),
                        cumulative_threshold_used=_cumulative_threshold,
                        selection_method="ig_dual+lgbm",
                        selected_indices=sorted(pruned_indices),
                        selected_names=pruned_names,
                        feature_importances={
                            pruned_names[i]: float(lgbm_gains[pruned_indices[i]])
                            for i in range(min(50, len(pruned_indices)))
                        },
                        mandatory_included=ig_result.mandatory_included,
                    )
                else:
                    selection_result = ig_result

            except Exception as _ig_exc:
                logger.warning(
                    "IG dual selection failed for task %s (%s) -- falling back to lgbm_gain",
                    task_name, _ig_exc,
                    exc_info=True,
                )
                pruned_indices = feature_selector.prune_by_lgbm(
                    lgbm_model=student_model,
                    feature_names=feature_cols,
                )
                pruned_names = [feature_cols[i] for i in pruned_indices]
                lgbm_gains = student_model.feature_importance(importance_type="gain")
                selection_result = FeatureSelectionResult(
                    task_name=task_name,
                    original_count=len(feature_cols),
                    selected_count=len(pruned_indices),
                    reduction_pct=round(
                        (1 - len(pruned_indices) / len(feature_cols)) * 100, 1,
                    ),
                    cumulative_threshold_used=0.0,
                    selection_method="lgbm_fallback",
                    selected_indices=sorted(pruned_indices),
                    selected_names=pruned_names,
                    feature_importances={
                        pruned_names[i]: float(lgbm_gains[pruned_indices[i]])
                        for i in range(min(50, len(pruned_indices)))
                    },
                    mandatory_included=[],
                )

        elif _fs_method in ("ig_dual", "both") and _teacher_model is None:
            logger.warning(
                "method=%s but no teacher model loaded -- falling back to lgbm_gain for task %s",
                _fs_method, task_name,
            )
            pruned_indices = feature_selector.prune_by_lgbm(
                lgbm_model=student_model,
                feature_names=feature_cols,
            )
            pruned_names = [feature_cols[i] for i in pruned_indices]
            lgbm_gains = student_model.feature_importance(importance_type="gain")
            selection_result = FeatureSelectionResult(
                task_name=task_name,
                original_count=len(feature_cols),
                selected_count=len(pruned_indices),
                reduction_pct=round(
                    (1 - len(pruned_indices) / len(feature_cols)) * 100, 1,
                ),
                cumulative_threshold_used=0.0,
                selection_method="lgbm_fallback",
                selected_indices=sorted(pruned_indices),
                selected_names=pruned_names,
                feature_importances={
                    pruned_names[i]: float(lgbm_gains[pruned_indices[i]])
                    for i in range(min(50, len(pruned_indices)))
                },
                mandatory_included=[],
            )

        else:
            # lgbm_gain only (default / original behaviour)
            pruned_indices = feature_selector.prune_by_lgbm(
                lgbm_model=student_model,
                feature_names=feature_cols,
            )
            pruned_names = [feature_cols[i] for i in pruned_indices]
            lgbm_gains = student_model.feature_importance(importance_type="gain")
            selection_result = FeatureSelectionResult(
                task_name=task_name,
                original_count=len(feature_cols),
                selected_count=len(pruned_indices),
                reduction_pct=round(
                    (1 - len(pruned_indices) / len(feature_cols)) * 100, 1,
                ),
                cumulative_threshold_used=0.0,
                selection_method="lgbm",
                selected_indices=sorted(pruned_indices),
                selected_names=pruned_names,
                feature_importances={
                    pruned_names[i]: float(lgbm_gains[pruned_indices[i]])
                    for i in range(min(50, len(pruned_indices)))
                },
                mandatory_included=[],
            )

        feature_selections[task_name] = selection_result

        logger.info(
            "  %s [%s]: %d/%d features selected (%.1f%% reduction)",
            task_name, selection_result.selection_method,
            selection_result.selected_count,
            selection_result.original_count, selection_result.reduction_pct,
        )

    # Step 5: Save with fidelity results and feature selections
    saved = trainer.save_students(
        args.output_dir,
        feature_selections=feature_selections,
        fidelity_results=fidelity_results,
    )

    # Step 5.5: Save calibration models alongside LGBM models
    saved_calibrators: dict[str, str] = {}
    if calibrated_models:
        logger.info("Saving calibration models for %d task(s)...", len(calibrated_models))
        for task_name, calib in calibrated_models.items():
            task_dir = Path(args.output_dir) / task_name
            task_dir.mkdir(parents=True, exist_ok=True)
            calib_path = task_dir / "calibrator.joblib"
            joblib.dump(calib, calib_path)
            saved_calibrators[task_name] = str(calib_path)
            logger.info("  [CALIBRATOR SAVED] %s -> %s", task_name, calib_path)
        logger.info("Calibration models saved: %d task(s)", len(saved_calibrators))

    # Summary
    logger.info("=" * 60)
    logger.info("Distillation complete! (all fidelity checks passed)")
    logger.info("  Tasks distilled: %d", len(saved))
    for task_name, path in saved.items():
        logger.info("  %s -> %s", task_name, path)
    logger.info("=" * 60)

    # Save summary JSON for SageMaker
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary = {
        "tasks_distilled": distill_tasks,
        "tasks_direct_hardlabel": hardlabel_tasks,
        "tasks_skipped_rule_only": skip_tasks,
        "num_students": len(saved),
        "adaptive_strategy": {
            "distill_count": len(distill_tasks),
            "direct_count": len(hardlabel_tasks),
            "skip_count": len(skip_tasks),
            "threshold_rule": "binary: AUC>0.60, multiclass: F1>2/K, regression: R2>0.05",
            "floor_rule": f"binary: AUC>{_floor_binary_auc}, multiclass: F1>{_floor_mc_f1_ratio}/K, regression: R2>{_floor_r2}",
        },
        "temperature": student_config.temperature,
        "alpha": student_config.alpha,
        "lgbm_params": student_config.lgbm_params,
        "task_lgbm_overrides": student_config.task_lgbm_overrides,
        "output_dir": args.output_dir,
        "feature_count": len(feature_cols),
        "sample_count": len(features),
        "fidelity": {
            "all_passed": failed_count == 0,
            "passed": passed_count,
            "failed": failed_count,
            "per_task": {
                r.task_name: {
                    "passed": r.passed,
                    "metrics": {k: round(v, 4) for k, v in r.metrics.items()},
                }
                for r in fidelity_results
            },
        },
        "feature_selection": {
            task_name: {
                "selected": sel.selected_count,
                "original": sel.original_count,
                "reduction_pct": sel.reduction_pct,
                "method": sel.selection_method,
            }
            for task_name, sel in feature_selections.items()
        },
        "ig_dual_scores": _ig_raw_scores,
        "feature_selection_config": {
            "method": _fs_method,
            "ig_alpha": _ig_alpha,
            "cumulative_threshold": _cumulative_threshold,
            "ig_sample_size": _ig_sample_size,
        },
        "calibration": {
            "enabled": bool(calibrated_models),
            "method": _calib_method if calibrated_models else None,
            "calibrated_tasks": list(saved_calibrators.keys()),
            "calibrator_paths": saved_calibrators,
        },
    }
    summary_path = output_path / "distillation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
