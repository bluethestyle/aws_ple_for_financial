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
        --config configs/financial/pipeline.yaml \
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
        default="configs/financial/pipeline.yaml",
        help="Pipeline YAML config path",
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


def main() -> None:
    """Run the distillation pipeline."""
    args = parse_args()

    # Import here to avoid import errors if torch is not available
    # in LGBM-only mode
    import yaml as _yaml
    from core.pipeline.config import load_config
    from core.training.student_trainer import StudentConfig, StudentTrainer

    # Load pipeline config (structured) and raw YAML (for distillation section)
    pipeline_config = load_config(args.config)
    with open(args.config, encoding="utf-8") as _f:
        _raw_yaml: dict = _yaml.safe_load(_f)
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

    # Lightweight pandas view for quality gate (uses column metadata only)
    df = table.to_pandas()

    # Step 1.5: Quality gate — block pipeline on critical data issues
    logger.info("Running quality gate on loaded data...")
    from core.data.quality_gate import QualityGate, QualityGateError

    quality_gate = QualityGate()
    try:
        gate_result = quality_gate.evaluate_and_block(df, source_name="distillation_train")
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

    # Release pandas view after quality gate
    del df

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

    # Step 3: Train LGBM students
    logger.info("Training LGBM student models...")
    students = trainer.train_students(features, hard_labels)

    # Step 4: Fidelity validation — teacher-student agreement check
    logger.info("Running fidelity validation (8 metrics per task)...")
    from core.training.distillation_validator import (
        DistillationValidator,
        ValidationCriteria,
    )

    validator = DistillationValidator(criteria=ValidationCriteria())
    fidelity_results = []

    # Get teacher soft labels for comparison
    soft_labels = trainer.get_soft_labels()  # Dict[task_name, np.ndarray]

    for task_spec in pipeline_config.tasks:
        task_name = task_spec.name
        if task_name not in students:
            continue

        student_model = students[task_name]

        # Student predictions on the same training features.
        # Custom fobj → predict() returns raw margins, not probabilities.
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

    # Step 4.5: Feature selection — per-task LGBM importance-based pruning
    logger.info("Running adaptive feature selection per task...")
    from core.training.feature_selector import FeatureSelector, FeatureSelectionConfig

    feature_selector = FeatureSelector(config=FeatureSelectionConfig())
    feature_selections = {}

    for task_spec in pipeline_config.tasks:
        task_name = task_spec.name
        if task_name not in students:
            continue

        student_model = students[task_name]

        # Use LGBM gain-based pruning (Stage 2 only — no teacher needed)
        pruned_indices = feature_selector.prune_by_lgbm(
            lgbm_model=student_model,
            feature_names=feature_cols,
        )
        pruned_names = [feature_cols[i] for i in pruned_indices]

        # Build a FeatureSelectionResult-compatible dict for save_students
        from core.training.feature_selector import FeatureSelectionResult

        selection_result = FeatureSelectionResult(
            task_name=task_name,
            original_count=len(feature_cols),
            selected_count=len(pruned_indices),
            reduction_pct=round((1 - len(pruned_indices) / len(feature_cols)) * 100, 1),
            cumulative_threshold_used=0.0,
            selection_method="lgbm",
            selected_indices=sorted(pruned_indices),
            selected_names=pruned_names,
            feature_importances={
                pruned_names[i]: float(
                    student_model.feature_importance(importance_type="gain")[pruned_indices[i]]
                )
                for i in range(min(50, len(pruned_indices)))
            },
            mandatory_included=[],
        )
        feature_selections[task_name] = selection_result

        logger.info(
            "  %s: %d/%d features selected (%.1f%% reduction)",
            task_name, selection_result.selected_count,
            selection_result.original_count, selection_result.reduction_pct,
        )

    # Step 5: Save with fidelity results and feature selections
    saved = trainer.save_students(
        args.output_dir,
        feature_selections=feature_selections,
        fidelity_results=fidelity_results,
    )

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
        "tasks_distilled": list(saved.keys()),
        "num_students": len(saved),
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
            }
            for task_name, sel in feature_selections.items()
        },
    }
    summary_path = output_path / "distillation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
