#!/usr/bin/env python3
"""
Pipeline Submission CLI — submit the full ML pipeline to AWS.

Supports three modes:
  1. full     — Feature Engineering → PLE Training → Distillation (default)
  2. training — PLE Training only (assumes features already on S3)
  3. distill  — Distillation only (assumes trained PLE on S3)

Usage:
    # Full pipeline
    python scripts/submit_pipeline.py --config configs/financial/pipeline.yaml

    # Training only (features already generated)
    python scripts/submit_pipeline.py --mode training --features-uri s3://bucket/features/

    # Distillation only (teacher already trained)
    python scripts/submit_pipeline.py --mode distill --teacher-uri s3://bucket/models/model.tar.gz

    # Step Functions execution
    python scripts/submit_pipeline.py --mode stepfunctions

    # Dry run (print what would be submitted)
    python scripts/submit_pipeline.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("submit_pipeline")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Submit ML pipeline to AWS SageMaker",
    )
    parser.add_argument(
        "--config", type=str, default="configs/financial/pipeline.yaml",
        help="Pipeline config YAML path",
    )
    parser.add_argument(
        "--mode", type=str, default="full",
        choices=["full", "training", "distill", "stepfunctions"],
        help="Execution mode",
    )
    parser.add_argument(
        "--features-uri", type=str, default="",
        help="S3 URI to pre-computed features (skips feature engineering)",
    )
    parser.add_argument(
        "--teacher-uri", type=str, default="",
        help="S3 URI to trained PLE model (for distillation-only mode)",
    )
    parser.add_argument(
        "--instance-type", type=str, default="",
        help="Override training instance type (e.g. ml.g4dn.xlarge)",
    )
    parser.add_argument(
        "--no-spot", action="store_true",
        help="Disable Spot instances (use on-demand)",
    )
    parser.add_argument(
        "--no-wait", action="store_true",
        help="Submit and return immediately (don't wait for completion)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be submitted without actually running",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load pipeline config
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from core.pipeline.config import load_config

    config = load_config(args.config)
    aws = config.aws
    s3_base = f"s3://{aws.s3_bucket}/{config.task_name}"
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    wait = not args.no_wait

    if args.instance_type:
        config.aws.instance_type = args.instance_type
    if args.no_spot:
        config.aws.use_spot = False

    logger.info("=" * 60)
    logger.info("Pipeline Submission")
    logger.info("  Config: %s", args.config)
    logger.info("  Mode: %s", args.mode)
    logger.info("  Region: %s", aws.region)
    logger.info("  S3 Bucket: %s", aws.s3_bucket)
    logger.info("  Instance: %s (Spot=%s)", aws.instance_type, aws.use_spot)
    logger.info("  Tasks: %d", len(config.tasks))
    logger.info("=" * 60)

    if args.mode == "stepfunctions":
        _run_stepfunctions(config, args, s3_base, ts)
    elif args.mode == "full":
        _run_full(config, args, s3_base, ts, wait)
    elif args.mode == "training":
        _run_training(config, args, s3_base, ts, wait)
    elif args.mode == "distill":
        _run_distill(config, args, s3_base, ts, wait)


def _run_stepfunctions(config, args, s3_base, ts):
    """Submit the full pipeline via Step Functions."""
    import boto3

    sfn_client = boto3.client("stepfunctions", region_name=config.aws.region)

    # Find the state machine ARN
    template_path = Path(__file__).resolve().parent.parent / \
        "aws/stepfunctions/templates/training_pipeline.json"

    execution_input = {
        "task_name": config.task_name,
        "raw_data_uri": config.data.source,
        "features_output_uri": f"{s3_base}/features/{ts}/",
        "model_output_uri": f"{s3_base}/models/{ts}/",
        "checkpoint_uri": f"{s3_base}/checkpoints/{ts}/",
        "processing_image_uri": f"{config.aws.s3_bucket}.dkr.ecr.{config.aws.region}.amazonaws.com/feature-gen:latest",
        "training_image_uri": f"{config.aws.s3_bucket}.dkr.ecr.{config.aws.region}.amazonaws.com/ple-training:latest",
        "instance_type": config.aws.instance_type,
        "use_spot": config.aws.use_spot,
        "role_arn": config.aws.role_arn,
        "hyperparameters": {
            "batch_size": str(config.training.batch_size),
            "epochs": str(config.training.epochs),
            "learning_rate": str(config.training.learning_rate),
            "seed": str(config.training.seed),
        },
    }

    if args.dry_run:
        logger.info("[DRY RUN] Step Functions execution input:")
        logger.info(json.dumps(execution_input, indent=2))
        logger.info("[DRY RUN] Template: %s", template_path)
        return

    # List state machines and find ours
    paginator = sfn_client.get_paginator("list_state_machines")
    sm_arn = None
    for page in paginator.paginate():
        for sm in page["stateMachines"]:
            if "training-pipeline" in sm["name"].lower():
                sm_arn = sm["stateMachineArn"]
                break

    if sm_arn is None:
        logger.error(
            "No Step Functions state machine found with 'training-pipeline' in name. "
            "Create one from aws/stepfunctions/templates/training_pipeline.json first."
        )
        sys.exit(1)

    execution_name = f"pipeline-{ts}"
    response = sfn_client.start_execution(
        stateMachineArn=sm_arn,
        name=execution_name,
        input=json.dumps(execution_input),
    )
    logger.info("Step Functions execution started: %s", response["executionArn"])
    logger.info("Console: https://%s.console.aws.amazon.com/states/home?region=%s",
                config.aws.region, config.aws.region)


def _run_full(config, args, s3_base, ts, wait):
    """Run the full pipeline: Feature Eng → Training → Distillation."""
    from aws.sagemaker.processing import SageMakerProcessingJob
    from aws.sagemaker.trainer import SageMakerTrainer

    features_uri = args.features_uri or f"{s3_base}/features/{ts}/"

    # Step 1: Feature Engineering (skip if features_uri provided)
    if not args.features_uri:
        logger.info("--- Step 1: Feature Engineering ---")
        proc = SageMakerProcessingJob(config)
        if args.dry_run:
            logger.info("[DRY RUN] Processing Job: %s → %s", config.data.source, features_uri)
        else:
            proc.run(
                script="entrypoint.py",
                source_dir="containers/generators/base/",
                input_s3=config.data.source,
                output_s3=features_uri,
                wait=wait,
            )
    else:
        logger.info("--- Step 1: Skipped (features_uri provided) ---")

    # Step 2: PLE Training
    logger.info("--- Step 2: PLE Training (2-phase) ---")
    trainer = SageMakerTrainer(config)
    if args.dry_run:
        logger.info("[DRY RUN] Training Job: Phase1 + Phase2")
        logger.info("[DRY RUN] Instance: %s, Spot: %s", config.aws.instance_type, config.aws.use_spot)
        model_uri = f"{s3_base}/models/{ts}/model.tar.gz"
    else:
        phase1_result = trainer.launch_phase1(wait=wait)
        logger.info("Phase 1 complete: %s", phase1_result.get("job_name"))

        model_uri = phase1_result.get("s3_model_uri", "")
        if wait and model_uri:
            phase2_result = trainer.launch_phase2(model_uri, wait=wait)
            model_uri = phase2_result.get("s3_model_uri", model_uri)
            logger.info("Phase 2 complete: %s", phase2_result.get("job_name"))

    # Step 3: Distillation
    logger.info("--- Step 3: Knowledge Distillation (PLE → LGBM) ---")
    if args.dry_run:
        logger.info("[DRY RUN] Distillation Job: teacher=%s", model_uri)
        logger.info("[DRY RUN] Output: %s/students/", s3_base)
    else:
        _submit_distillation_job(config, features_uri, model_uri, s3_base, ts, wait)

    logger.info("=" * 60)
    logger.info("Pipeline submission complete!")
    logger.info("  Features: %s", features_uri)
    logger.info("  Teacher model: %s", model_uri)
    logger.info("  Student models: %s/students/%s/", s3_base, ts)
    logger.info("=" * 60)


def _run_training(config, args, s3_base, ts, wait):
    """Training-only mode."""
    from aws.sagemaker.trainer import SageMakerTrainer

    if not args.features_uri:
        logger.error("--features-uri required for training mode")
        sys.exit(1)

    trainer = SageMakerTrainer(config)

    if args.dry_run:
        logger.info("[DRY RUN] Training: Phase1 → Phase2")
        return

    phase1 = trainer.launch_phase1(wait=wait)
    logger.info("Phase 1: %s", phase1.get("job_name"))

    if wait and phase1.get("s3_model_uri"):
        phase2 = trainer.launch_phase2(phase1["s3_model_uri"], wait=wait)
        logger.info("Phase 2: %s", phase2.get("job_name"))
        logger.info("Model: %s", phase2.get("s3_model_uri"))


def _run_distill(config, args, s3_base, ts, wait):
    """Distillation-only mode."""
    if not args.teacher_uri:
        logger.error("--teacher-uri required for distill mode")
        sys.exit(1)

    features_uri = args.features_uri or config.data.source

    if args.dry_run:
        logger.info("[DRY RUN] Distillation: teacher=%s", args.teacher_uri)
        return

    _submit_distillation_job(config, features_uri, args.teacher_uri, s3_base, ts, wait)


def _submit_distillation_job(config, features_uri, model_uri, s3_base, ts, wait):
    """Submit a SageMaker Processing Job for distillation."""
    import boto3
    import sagemaker
    from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor

    session = sagemaker.Session()
    aws = config.aws
    job_name = f"distill-{config.task_name}-{ts}"
    output_uri = f"{s3_base}/students/{ts}/"

    processor = ScriptProcessor(
        role=aws.role_arn,
        image_uri=sagemaker.image_uris.retrieve(
            "sklearn", aws.region, version="1.2-1",
        ),
        instance_count=1,
        instance_type="ml.m5.xlarge",
        command=["python3"],
        sagemaker_session=session,
    )

    processor.run(
        code="scripts/run_distillation.py",
        source_dir=".",
        inputs=[
            ProcessingInput(
                source=model_uri,
                destination="/opt/ml/processing/input/teacher",
            ),
            ProcessingInput(
                source=features_uri,
                destination="/opt/ml/processing/input/data",
            ),
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/output",
                destination=output_uri,
            ),
        ],
        arguments=[
            "--teacher-checkpoint", "/opt/ml/processing/input/teacher/model.pth",
            "--data-path", "/opt/ml/processing/input/data/",
            "--output-dir", "/opt/ml/processing/output",
            "--config", "configs/financial/pipeline.yaml",
            "--soft-label-path", "/opt/ml/processing/output/soft_labels.parquet",
        ],
        job_name=job_name,
        wait=wait,
    )

    logger.info("Distillation job submitted: %s", job_name)
    logger.info("Output: %s", output_uri)


if __name__ == "__main__":
    main()
