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

    # Save
    saved = trainer.save_students(args.output_dir)

    # Summary
    logger.info("=" * 60)
    logger.info("Distillation complete!")
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
    }
    summary_path = output_path / "distillation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
