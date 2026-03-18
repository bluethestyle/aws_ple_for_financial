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
import pandas as pd

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
        default=5.0,
        help="Distillation temperature (higher = softer labels)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="Hard label weight (1-alpha = soft label weight)",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Specific tasks to distill (default: all non-contrastive tasks)",
    )
    return parser.parse_args()


def main() -> None:
    """Run the distillation pipeline."""
    args = parse_args()

    # Import here to avoid import errors if torch is not available
    # in LGBM-only mode
    from core.pipeline.config import load_config
    from core.training.student_trainer import StudentConfig, StudentTrainer

    # Load pipeline config
    pipeline_config = load_config(args.config)

    # Load data
    logger.info("Loading data from %s", args.data_path)
    data_path = Path(args.data_path)
    if data_path.is_dir():
        # Read all parquet files in directory
        parquet_files = sorted(data_path.glob("*.parquet"))
        if not parquet_files:
            logger.error("No parquet files found in %s", args.data_path)
            sys.exit(1)
        df = pd.concat(
            [pd.read_parquet(f) for f in parquet_files],
            ignore_index=True,
        )
        logger.info("Loaded %d files, %d rows total", len(parquet_files), len(df))
    else:
        df = pd.read_parquet(args.data_path)
        logger.info("Loaded %d rows", len(df))

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

    # Separate features and labels
    label_cols = {t.label_col for t in pipeline_config.tasks}
    id_cols = set(pipeline_config.features.id_cols)
    exclude_cols = label_cols | id_cols

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    logger.info(
        "Features: %d columns, Labels: %d tasks, IDs: %d columns",
        len(feature_cols),
        len(label_cols),
        len(id_cols),
    )

    features = df[feature_cols].values.astype(np.float32)
    hard_labels: dict[str, np.ndarray] = {}
    for t in pipeline_config.tasks:
        if t.label_col in df.columns:
            hard_labels[t.name] = df[t.label_col].values

    # Build student config
    student_config = StudentConfig(
        teacher_checkpoint=args.teacher_checkpoint or "",
        soft_label_path=args.soft_label_path,
        student_output_dir=args.output_dir,
        temperature=args.temperature,
        alpha=args.alpha,
        enabled_tasks=args.tasks,
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

        # Student predictions on the same training features
        student_preds = student_model.predict(features)

        # Teacher predictions (soft labels)
        teacher_preds = soft_labels.get(task_name)
        if teacher_preds is None:
            logger.warning("No soft labels for task %s, skipping fidelity", task_name)
            continue

        # Ground truth labels (for AUC, calibration)
        labels = hard_labels.get(task_name)

        result = validator.validate_task(
            task_name=task_name,
            task_type=task_spec.type,
            teacher_preds=teacher_preds,
            student_preds=student_preds,
            labels=labels,
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
        sys.exit(1)  # SageMaker Job fails → Step Functions catches

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
        "temperature": args.temperature,
        "alpha": args.alpha,
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
