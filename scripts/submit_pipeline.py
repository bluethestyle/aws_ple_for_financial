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

# Force UTF-8 stdout/stderr. SageMaker SDK streams CloudWatch container
# logs straight to sys.stdout.print(); Windows' cp949 default raises
# UnicodeEncodeError on characters like em-dash (U+2014) that appear in
# progress bars and task descriptions, killing the orchestrator even
# though the Training Job itself is fine.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except (AttributeError, Exception):
    pass

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
    parser.add_argument(
        "--attach-phase1-job", type=str, default="",
        help=(
            "Attach to an already-running Phase 1 Training Job (by job name) "
            "instead of submitting a new one. Used to recover from an "
            "orchestrator crash without losing the billable clock."
        ),
    )
    parser.add_argument(
        "--attach-phase2-job", type=str, default="",
        help=(
            "Attach to an already-completed / running Phase 2 Training Job "
            "by name. Lets the orchestrator skip both teacher phases and "
            "resume at Distillation (used to recover after a distill crash)."
        ),
    )
    parser.add_argument(
        "--raw-data-uri", type=str, default="",
        help=(
            "S3 URI pointing at the ingestion parquet that Phase 0 should "
            "consume (e.g. s3://aiops-ple-financial/data/santander/"
            "santander_final.parquet). Required when neither --features-uri "
            "nor --attach-phase0-job is set — submit_pipeline will launch a "
            "Phase 0 SageMaker Training Job to produce the training-ready "
            "features."
        ),
    )
    parser.add_argument(
        "--attach-phase0-job", type=str, default="",
        help=(
            "Attach to an already-running Phase 0 Training Job by name. "
            "Lets the orchestrator skip feature engineering and jump "
            "straight to Phase 1 with the job's output prefix."
        ),
    )
    parser.add_argument(
        "--attach-distill-output", type=str, default="",
        help=(
            "Skip Distillation submission and use the given S3 prefix "
            "(SageMaker distill Job output) as the student artefacts "
            "source for Register + PromotionGate. Useful for re-running "
            "registration with updated scan logic without paying for a "
            "fresh distill job."
        ),
    )
    parser.add_argument(
        "--force-promote", action="store_true",
        help=(
            "Promote the new model to champion unconditionally, bypassing the "
            "Champion-Challenger offline competition gate. Use for bootstrap "
            "or emergency rollback."
        ),
    )
    parser.add_argument(
        "--epochs", type=int, default=0,
        help=(
            "Override ``training.epochs`` from the YAML config. The value "
            "is the *total* training budget; the trainer splits it as "
            "phase1 = epochs/3 (warm-up, shared layers) and phase2 = epochs "
            "(fine-tune). Use 10 to match the historical ablation standard "
            "(see outputs/ablation_v12/run_manifest.json: epochs=10, "
            "batch=5632, amp=true, lr=0.0005, seed=42). "
            "0 (default) keeps the YAML value."
        ),
    )
    parser.add_argument(
        "--phase0-only", action="store_true",
        help=(
            "Stop after Phase 0 completes. Skips PLE training, distillation, "
            "and registration so the operator can verify the new "
            "feature_schema / group_ranges / expert_routing before paying for "
            "GPU training. Combine with --raw-data-uri to launch a fresh "
            "Phase 0 job, or --attach-phase0-job to attach to a running one. "
            "Per CLAUDE.md §1.4 (pre-flight check) and the GPU-on-SageMaker "
            "policy: stage Phase 0 first, validate, then submit training."
        ),
    )
    parser.add_argument(
        "--phase0-split", action="store_true",
        help=(
            "Run Phase 0 as 3 sequential SageMaker Jobs (A: stages 1-2, "
            "B: stage 3 (feature gen), C: stages 4-9) sharing a single "
            "S3 checkpoint prefix. Bypasses the OOM that hits a single "
            "ml.m5.4xlarge during the contiguous 9-stage run on the new "
            "1285-D feature set. Implies --phase0-only is wired through "
            "(downstream phases use the Job-C model artefact)."
        ),
    )
    parser.add_argument(
        "--mamba-precompute", action="store_true",
        help=(
            "Run the Mamba temporal-embedding pre-compute job on a GPU "
            "SageMaker instance. Produces a per-customer "
            "embedding.parquet under s3://{bucket}/{task}/mamba/{ts}/. "
            "After it succeeds, set ``feature_groups.yaml::mamba_temporal."
            "generator_params.cached_embedding_uri`` to that path and "
            "flip ``enabled: true`` so the next Phase 0 run picks up "
            "the +50D mamba block. Job exits on completion; this flag "
            "does NOT chain into Phase 0 / training."
        ),
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
    if args.epochs and args.epochs > 0:
        prev = config.training.epochs
        config.training.epochs = args.epochs
        logger.info(
            "Overriding training.epochs: %s -> %d (phase1=%d, phase2=%d)",
            prev, args.epochs, max(1, args.epochs // 3), args.epochs,
        )

    logger.info("=" * 60)
    logger.info("Pipeline Submission")
    logger.info("  Config: %s", args.config)
    logger.info("  Mode: %s", args.mode)
    logger.info("  Region: %s", aws.region)
    logger.info("  S3 Bucket: %s", aws.s3_bucket)
    logger.info("  Instance: %s (Spot=%s)", aws.instance_type, aws.use_spot)
    logger.info("  Tasks: %d", len(config.tasks))
    logger.info("=" * 60)

    # The mamba-precompute side-track runs *before* every other mode
    # check because it's a standalone GPU job that doesn't chain into
    # Phase 0 / training. Operators run it once, copy the resulting S3
    # path into feature_groups.yaml, then resubmit a normal Phase 0.
    if args.mamba_precompute:
        _run_mamba_precompute(config, args, s3_base, ts, wait)
        return

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


def _build_staging_dir() -> str:
    """Build the shared SageMaker source staging dir once for every Job.

    CLAUDE.md §1.5: ``source 패키지는 1회만 빌드하고 모든 Job에서 재사용한다``.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from package_source import build_staging  # type: ignore[import]
    return build_staging()


def _run_mamba_precompute(config, args, s3_base, ts, wait):
    """Standalone GPU SageMaker job that produces the Mamba embedding parquet.

    Output (set on the estimator's ``output_path``):
      ``s3://{bucket}/{task}/mamba/{ts}/<job_name>/output/model.tar.gz``

    The tar contains ``embedding.parquet``. To consume it, set
    ``feature_groups.yaml::mamba_temporal.generator_params.
    cached_embedding_uri`` to a (post-extraction) S3 prefix and
    flip ``enabled: true``.
    """
    from aws.sagemaker.trainer import SageMakerTrainer

    if args.dry_run:
        logger.info("[DRY RUN] Mamba pre-compute would launch on %s",
                    getattr(config.aws, "gpu_instance_type", "ml.g4dn.xlarge"))
        return

    staging = _build_staging_dir()
    logger.info("Staging dir: %s", staging)

    raw_uri = args.raw_data_uri or (
        config.data.source if config.data.source.startswith("s3://") else ""
    )
    if not raw_uri:
        logger.error(
            "Mamba pre-compute needs --raw-data-uri s3://... or "
            "data.source pointing at an S3 parquet."
        )
        sys.exit(2)

    trainer = SageMakerTrainer(config)
    result = trainer.launch_mamba_precompute(
        staging_dir=staging,
        raw_s3_uri=raw_uri,
        wait=wait,
    )
    logger.info("=" * 60)
    logger.info("Mamba pre-compute complete: %s", result.get("job_name"))
    logger.info("  Model URI: %s", result.get("s3_model_uri"))
    logger.info("  Billable seconds: %s", result.get("billable_seconds"))
    logger.info("  Next: download model.tar.gz, extract embedding.parquet")
    logger.info("        upload to s3://{bucket}/{task}/mamba/<ts>/embedding.parquet")
    logger.info("        set feature_groups.yaml::mamba_temporal."
                "generator_params.cached_embedding_uri to that path,")
    logger.info("        flip enabled: true, resubmit Phase 0.")
    logger.info("=" * 60)


def _launch_phase0_split(
    trainer,
    staging_dir: str,
    raw_s3_uri: str,
    s3_base: str,
    ts: str,
    wait: bool,
) -> dict:
    """Run Phase 0 as 3 sequential SageMaker jobs.

    The 9-stage runner is split into:
      * Job A (stages 1-2): load + preprocess. Cheap on memory, writes
        ``post_stage2/main.parquet`` to the shared S3 checkpoint prefix.
      * Job B (stage 3):    feature engineering — the heaviest stage
        memory-wise (1285-D output via lag/rolling/multihot/SQL-native
        generators). Reads post_stage2, writes post_stage3.
      * Job C (stages 4-9): labels + split + normalize + leakage +
        sequences + save. Reads post_stage3 and writes the final
        Phase-0 model artefact (``features.parquet``, ``feature_schema``,
        scaler, etc).

    The shared checkpoint prefix is
    ``s3://{bucket}/{task}/phase0-ckpt/{ts}/`` and is mapped to
    ``/opt/ml/checkpoints`` on each job by SageMaker's checkpoint
    manager.

    Returns a dict with the model URI of Job C (used as
    ``output_path`` so the rest of the pipeline keeps working) plus
    per-job metadata under ``jobs``.
    """
    if not wait:
        # SageMaker auto-syncs checkpoints during training, but the
        # parent script still needs to wait for each upstream job to
        # complete before launching the next one — otherwise Job B
        # would race against Job A's checkpoint write.
        raise ValueError(
            "--phase0-split requires waiting between jobs (drop --no-wait)."
        )

    bucket = trainer.config.aws.s3_bucket
    task = trainer.config.task_name
    checkpoint_s3_uri = f"s3://{bucket}/{task}/phase0-ckpt/{ts}/"

    logger.info("Phase 0 split: shared checkpoint %s", checkpoint_s3_uri)

    # ---- Job A: stages 1-2 (load + preprocess) ----------------------
    logger.info("Phase 0 split [Job A] launching stages 1-2 …")
    job_a = trainer.launch_phase0(
        staging_dir=staging_dir,
        raw_s3_uri=raw_s3_uri,
        wait=True,
        start_stage=1,
        end_stage=2,
        checkpoint_s3_uri=checkpoint_s3_uri,
        job_label="a",
    )
    logger.info(
        "Phase 0 split [Job A] done: %s (%.1fs billable)",
        job_a["job_name"], job_a.get("billable_seconds", 0),
    )

    # ---- Job B: stage 3 (feature gen) -------------------------------
    logger.info("Phase 0 split [Job B] launching stage 3 …")
    job_b = trainer.launch_phase0(
        staging_dir=staging_dir,
        raw_s3_uri=raw_s3_uri,
        wait=True,
        start_stage=3,
        end_stage=3,
        checkpoint_s3_uri=checkpoint_s3_uri,
        job_label="b",
    )
    logger.info(
        "Phase 0 split [Job B] done: %s (%.1fs billable)",
        job_b["job_name"], job_b.get("billable_seconds", 0),
    )

    # ---- Job C: stages 4-9 (labels + split + norm + save) -----------
    logger.info("Phase 0 split [Job C] launching stages 4-9 …")
    job_c = trainer.launch_phase0(
        staging_dir=staging_dir,
        raw_s3_uri=raw_s3_uri,
        wait=True,
        start_stage=4,
        end_stage=9,
        checkpoint_s3_uri=checkpoint_s3_uri,
        job_label="c",
    )
    logger.info(
        "Phase 0 split [Job C] done: %s (%.1fs billable)",
        job_c["job_name"], job_c.get("billable_seconds", 0),
    )

    return {
        "jobs": {"a": job_a, "b": job_b, "c": job_c},
        "checkpoint_s3_uri": checkpoint_s3_uri,
        # The downstream pipeline reads ``output_path`` for the
        # Phase-0 features. Job C is the only one that produces a full
        # Phase-0 artefact; A/B only emit the inter-stage checkpoint.
        "output_path": job_c["output_path"],
        "job_name": job_c["job_name"],
        "status": job_c.get("status"),
        "s3_model_uri": job_c.get("s3_model_uri"),
        "billable_seconds": (
            job_a.get("billable_seconds", 0)
            + job_b.get("billable_seconds", 0)
            + job_c.get("billable_seconds", 0)
        ),
    }


def _run_full(config, args, s3_base, ts, wait):
    """Run the full pipeline: (Phase 0 →) Training → Distillation → Register.

    Phase 0 cloud submission is TODO — see note below. For now, callers
    must point ``--features-uri`` at a pre-built Phase 0 output on S3
    (e.g. ``s3://aiops-ple-financial/data/phase0_v12/``).
    """
    from aws.sagemaker.trainer import SageMakerTrainer

    # Step 0: Build source staging (reused by every Job in this run).
    if args.dry_run:
        staging = "<staging-dir>"
    else:
        staging = _build_staging_dir()
        logger.info("Staging dir: %s", staging)

    # Phase 0 decision. Precedence:
    #   1. --features-uri: re-use an existing Phase 0 output (cheapest).
    #   2. --attach-phase0-job: attach to a running Phase 0 Training Job.
    #   3. --raw-data-uri: submit a fresh Phase 0 Training Job
    #      (containers/phase0/entrypoint.py).
    trainer = SageMakerTrainer(config)
    if args.features_uri:
        logger.info(
            "--- Step 1: Phase 0 skipped (features_uri=%s) ---",
            args.features_uri,
        )
        config.data.source = args.features_uri
        features_uri = args.features_uri
    elif args.dry_run:
        logger.info(
            "--- Step 1: Phase 0 (dry-run placeholder on %s) ---",
            args.raw_data_uri or config.data.source,
        )
        features_uri = f"{s3_base}/features/{ts}/"
    else:
        if args.attach_phase0_job:
            logger.info(
                "--- Step 1: Attaching to Phase 0 job: %s ---",
                args.attach_phase0_job,
            )
            phase0 = trainer.attach_running_job(args.attach_phase0_job)
            if phase0.get("status") != "Completed":
                logger.error(
                    "Phase 0 %s ended with status %s; aborting.",
                    args.attach_phase0_job, phase0.get("status"),
                )
                sys.exit(2)
        else:
            raw_uri = args.raw_data_uri or (
                config.data.source
                if config.data.source.startswith("s3://") else ""
            )
            if not raw_uri:
                logger.error(
                    "Phase 0 needs a raw input. Pass --raw-data-uri s3://... "
                    "or --features-uri (to skip Phase 0) or --attach-phase0-job.",
                )
                sys.exit(2)
            if args.phase0_split:
                logger.info(
                    "--- Step 1: Phase 0 Feature Engineering "
                    "(3-job split) ---"
                )
                phase0 = _launch_phase0_split(
                    trainer=trainer,
                    staging_dir=staging,
                    raw_s3_uri=raw_uri,
                    s3_base=s3_base,
                    ts=ts,
                    wait=wait,
                )
                logger.info(
                    "Phase 0 split complete: A=%s  B=%s  C=%s",
                    phase0["jobs"]["a"]["job_name"],
                    phase0["jobs"]["b"]["job_name"],
                    phase0["jobs"]["c"]["job_name"],
                )
            else:
                logger.info(
                    "--- Step 1: Phase 0 Feature Engineering (cloud) ---"
                )
                phase0 = trainer.launch_phase0(
                    staging_dir=staging,
                    raw_s3_uri=raw_uri,
                    wait=wait,
                )
                logger.info("Phase 0 complete: %s", phase0.get("job_name"))
        # Trainer reads ``config.data.source`` for its TrainingInput —
        # point it at the Phase 0 output prefix (the training channel
        # will pick up the parquet + schemas produced by the runner).
        features_uri = phase0.get("output_path", "")
        if not features_uri:
            logger.error("Phase 0 did not return output_path; aborting.")
            sys.exit(2)
        config.data.source = features_uri.rstrip("/") + "/"

    # Early-return gate: --phase0-only stops the pipeline after Phase 0
    # so the operator can verify feature_schema / group_ranges /
    # expert_routing before paying for downstream GPU training. The
    # features URI is the contract for the next invocation
    # (``--features-uri <returned_uri>``).
    if args.phase0_only:
        logger.info("=" * 60)
        logger.info("--phase0-only: stopping after Phase 0")
        logger.info("  Features URI: %s", features_uri)
        logger.info("  Next: download artifacts, verify feature_schema.json,")
        logger.info("        then resubmit with --features-uri %s", features_uri)
        logger.info("=" * 60)
        return

    # Step 2: Teacher training (Phase 1 warm-up + Phase 2 fine-tune).
    logger.info("--- Step 2: PLE Training (2-phase) ---")
    if args.dry_run:
        logger.info(
            "[DRY RUN] Training Job: Phase1 + Phase2 on %s", features_uri,
        )
        logger.info(
            "[DRY RUN] Instance: %s, Spot: %s",
            config.aws.instance_type, config.aws.use_spot,
        )
        model_uri = f"{s3_base}/models/phase2/{ts}/model.tar.gz"
    else:
        if args.attach_phase2_job:
            # Recovery path: both teacher phases are already done on the
            # cluster. Skip submission, attach to Phase 2 by name, and go
            # straight to Distillation.
            logger.info(
                "Attaching to existing Phase 2 job: %s",
                args.attach_phase2_job,
            )
            phase2 = trainer.attach_running_job(args.attach_phase2_job)
            if phase2.get("status") != "Completed":
                logger.error(
                    "Existing Phase 2 %s ended with status %s; aborting.",
                    args.attach_phase2_job, phase2.get("status"),
                )
                sys.exit(2)
            model_uri = phase2.get("s3_model_uri", "")
        else:
            if args.attach_phase1_job:
                logger.info(
                    "Attaching to existing Phase 1 job: %s",
                    args.attach_phase1_job,
                )
                phase1 = trainer.attach_running_job(args.attach_phase1_job)
                if phase1.get("status") != "Completed":
                    logger.error(
                        "Existing Phase 1 %s ended with status %s; aborting.",
                        args.attach_phase1_job, phase1.get("status"),
                    )
                    sys.exit(2)
            else:
                phase1 = trainer.launch_phase1(
                    staging_dir=staging, wait=wait,
                )
                logger.info(
                    "Phase 1 complete: %s", phase1.get("job_name"),
                )
            phase1_uri = phase1.get("s3_model_uri", "")
            if not phase1_uri:
                logger.error(
                    "Phase 1 did not return an s3_model_uri; aborting.",
                )
                sys.exit(2)
            phase2 = trainer.launch_phase2(
                staging_dir=staging,
                phase1_model_uri=phase1_uri,
                wait=wait,
            )
            logger.info("Phase 2 complete: %s", phase2.get("job_name"))
            model_uri = phase2.get("s3_model_uri", phase1_uri)
        # distill_entry.py expects a directory of .pt/.pth files (best.pt +
        # epoch_*.pt), which the Spot checkpoint_s3_uri always holds; the
        # SageMaker output model.tar.gz, in contrast, is a single gzipped
        # archive that the distillation entrypoint cannot open without an
        # extra extraction step. Prefer the checkpoint dir when available.
        teacher_channel_uri = (
            phase2.get("checkpoint_s3_uri") or model_uri
        )

    # Step 3: Distillation (CPU instance, PyTorch estimator, 2 channels).
    logger.info("--- Step 3: Knowledge Distillation (PLE → LGBM, CPU) ---")
    if args.dry_run:
        logger.info("[DRY RUN] Distillation Job: teacher=%s", model_uri)
        student_uri = f"{s3_base}/students/{ts}/"
    elif args.attach_distill_output:
        logger.info(
            "Attaching existing distill output: %s", args.attach_distill_output,
        )
        student_uri = args.attach_distill_output.rstrip("/") + "/"
    else:
        distill = trainer.launch_distillation(
            staging_dir=staging,
            teacher_uri=teacher_channel_uri,
            wait=wait,
        )
        student_uri = distill.get("output_path", f"{s3_base}/students/{ts}/")
        logger.info("Distillation complete: %s", distill.get("job_name"))

    # Step 4: Register + PromotionGate + Audit + Tracker (local).
    version = f"v{ts.replace('-', '.').replace('T', '-')}"
    logger.info("--- Step 4: Register Model (version=%s) ---", version)
    if args.dry_run:
        logger.info("[DRY RUN] ModelRegistry.package(version=%s)", version)
        logger.info("[DRY RUN] artifacts: %s/artifacts/%s/", s3_base, version)
    elif wait:
        _register_model(
            config, s3_base, version, model_uri, student_uri,
            force_promote=args.force_promote,
        )

    # Step 5: Ops + Audit reports (post-hoc observability).
    # Wired to live AWS sources: SageMaker DescribeTrainingJob for the
    # Phase 0/1/2/Distill metrics, CloudWatch GetMetricData for Lambda
    # serving health, DynamoDB scan for reason_cache depth. Reports are
    # best-effort — failures log but never roll back the already-signed
    # promotion audit record from Step 4.
    if not args.dry_run and wait:
        try:
            from core.agent.pipeline_reports import (
                PipelineJobContext,
                run_pipeline_reports,
            )
            logger.info(
                "--- Step 5: Ops + Audit reports ---",
            )
            ctx = PipelineJobContext(
                region=config.aws.region,
                phase0_job_name=locals().get("phase0", {}).get("job_name", ""),
                phase1_job_name=locals().get("phase1", {}).get("job_name", ""),
                phase2_job_name=locals().get("phase2", {}).get("job_name", ""),
                distill_job_name=locals().get("distill", {}).get("job_name", ""),
            )
            report_artifacts = run_pipeline_reports(
                version=version,
                artifacts_dir=f"outputs/pipeline_reports/{version}",
                s3_prefix=f"{s3_base}/artifacts/{version}",
                enable_consensus=True,
                aws_context=ctx,
            )
            logger.info(
                "Ops status=%s | Audit risk=%s | Ops s3=%s | Audit s3=%s",
                report_artifacts.ops_status,
                report_artifacts.audit_risk_level,
                report_artifacts.ops_s3_uri,
                report_artifacts.audit_s3_uri,
            )
        except Exception:
            logger.exception("Pipeline reports failed (non-fatal)")

    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info("  Features:  %s", features_uri)
    logger.info("  Teacher:   %s", model_uri)
    logger.info("  Students:  %s", student_uri)
    logger.info("  Registry:  %s/artifacts/%s/", s3_base, version)
    logger.info("=" * 60)


def _run_training(config, args, s3_base, ts, wait):
    """Training-only mode (Phase 1 + Phase 2)."""
    from aws.sagemaker.trainer import SageMakerTrainer

    if not args.features_uri:
        logger.error("--features-uri required for training mode")
        sys.exit(1)
    config.data.source = args.features_uri

    if args.dry_run:
        logger.info("[DRY RUN] Training: Phase1 → Phase2 on %s", args.features_uri)
        return

    staging = _build_staging_dir()
    trainer = SageMakerTrainer(config)

    phase1 = trainer.launch_phase1(staging_dir=staging, wait=wait)
    logger.info("Phase 1: %s", phase1.get("job_name"))
    if wait and phase1.get("s3_model_uri"):
        phase2 = trainer.launch_phase2(
            staging_dir=staging,
            phase1_model_uri=phase1["s3_model_uri"],
            wait=wait,
        )
        logger.info("Phase 2: %s", phase2.get("job_name"))
        logger.info("Model: %s", phase2.get("s3_model_uri"))


def _run_distill(config, args, s3_base, ts, wait):
    """Distillation-only mode (CPU instance)."""
    from aws.sagemaker.trainer import SageMakerTrainer

    if not args.teacher_uri:
        logger.error("--teacher-uri required for distill mode")
        sys.exit(1)

    # Distillation reads features from config.data.source via TrainingInput.
    if args.features_uri:
        config.data.source = args.features_uri

    if args.dry_run:
        logger.info("[DRY RUN] Distillation: teacher=%s", args.teacher_uri)
        return

    staging = _build_staging_dir()
    trainer = SageMakerTrainer(config)
    distill = trainer.launch_distillation(
        staging_dir=staging,
        teacher_uri=args.teacher_uri,
        wait=wait,
    )
    logger.info("Distillation: %s", distill.get("job_name"))
    logger.info("Output: %s", distill.get("output_path"))


def _audit_promotion(**kwargs) -> None:
    """Best-effort audit log for a promotion decision.

    Failures are swallowed — promotion must not be blocked by audit
    logging.  The underlying AuditLogger already writes to a local
    fallback when S3 is unavailable.
    """
    try:
        from core.monitoring.audit_logger import AuditLogger
        AuditLogger().log_model_promotion(**kwargs)
    except Exception as exc:
        logger.warning("Audit log for promotion failed (non-fatal): %s", exc)


def _build_metadata_aggregator(
    pipeline_config: dict,
    registry: Optional[object] = None,
) -> Optional[object]:
    """Build a MetadataAggregator from the pipeline config.

    The aggregator merges real-metadata sources (lineage, fairness, LLM
    config, registry manifest, static overrides) into a
    ``(model_version) -> dict`` callable suitable for
    MetricsDerivedScoreProvider.

    Returns ``None`` if the compliance module cannot be imported (CI,
    truncated install) — the downstream gate factory will fall back to
    conservative 0.5 for every dimension in that case.
    """
    try:
        from core.compliance.metadata_aggregator import (
            MetadataAggregatorConfig,
            build_metadata_aggregator_from_config,
        )
    except Exception:
        logger.exception("Could not import metadata_aggregator module")
        return None

    gate_cfg = (
        (pipeline_config.get("compliance") or {}).get("promotion_gate") or {}
    )
    agg_cfg = (gate_cfg.get("providers") or {}).get("aggregator") or {}

    cfg = MetadataAggregatorConfig(
        cache_ttl_seconds=float(agg_cfg.get("cache_ttl_seconds", 300.0)),
        max_cache_entries=int(agg_cfg.get("max_cache_entries", 256)),
    )
    static_overrides = agg_cfg.get("static_overrides") or {}
    agent_slot_baseline = agg_cfg.get("agent_slot_baseline")

    # Archive paths — preferred in production because they survive a
    # fresh process start (live runtime objects are always empty when
    # submit_pipeline spawns a new orchestrator).
    sources_cfg = agg_cfg.get("sources") or {}
    lineage_yaml_path = sources_cfg.get("lineage_yaml_path")
    fairness_archive_path = sources_cfg.get("fairness_archive_parquet_path")

    # Falls back to monitoring.fairness.archive_parquet_path if the
    # aggregator block does not override it.
    if not fairness_archive_path:
        fairness_archive_path = (
            (pipeline_config.get("monitoring") or {})
            .get("fairness", {})
            .get("archive_parquet_path")
        )

    # Lineage / fairness / review-queue live instances — kept for
    # backward compatibility. The archive sources above already cover
    # fresh-start case, so the live instances are best-effort extras.
    lineage_tracker = None
    fairness_monitor = None
    try:
        from core.monitoring.lineage_tracker import DataLineageTracker
        lineage_tracker = DataLineageTracker()
    except Exception:
        logger.debug("Lineage tracker unavailable; pii_ratio will fallback")
    try:
        from core.monitoring.fairness_monitor import FairnessMonitor
        fairness_monitor = FairnessMonitor(config=pipeline_config)
    except Exception:
        logger.debug(
            "Fairness monitor unavailable; disparate_impact_min will fallback",
        )

    return build_metadata_aggregator_from_config(
        pipeline_config,
        lineage_tracker=lineage_tracker,
        fairness_monitor=fairness_monitor,
        model_registry=registry,
        review_queue=None,
        total_predictions_fn=None,
        static_overrides=static_overrides,
        aggregator_config=cfg,
        agent_slot_baseline=agent_slot_baseline,
        lineage_yaml_path=lineage_yaml_path,
        fairness_archive_path=fairness_archive_path,
    )


def _run_promotion_gate(
    new_version: str,
    pipeline_config: Optional[dict],
    registry: Optional[object] = None,
) -> Optional["object"]:
    """Optional Sprint 2 FRIA + AI Risk gate.

    Returns a :class:`GateVerdict` when the gate is enabled (via
    ``compliance.promotion_gate.enabled``), else ``None``. Failures in the
    gate itself are logged and treated as ``skip`` so a compliance-module
    error does not silently block unrelated promotions.
    """
    if not pipeline_config:
        return None
    gate_cfg = (
        (pipeline_config.get("compliance") or {}).get("promotion_gate") or {}
    )
    if not gate_cfg.get("enabled", False):
        return None
    try:
        from core.evaluation.promotion_gate import build_promotion_gate
        aggregator = _build_metadata_aggregator(pipeline_config, registry)
        gate = build_promotion_gate(
            pipeline_config, metadata_aggregator=aggregator,
        )
        verdict = gate.evaluate(model_version=new_version)
        logger.info(
            "Promotion gate verdict: %s - %s", verdict.decision, verdict.reason,
        )
        _track_promotion_gate_verdict(
            pipeline_config, new_version, verdict,
        )
        return verdict
    except Exception as exc:
        logger.warning(
            "Promotion gate raised (non-fatal, treating as skip): %s", exc,
        )
        return None


def _track_promotion_gate_verdict(
    pipeline_config: dict,
    model_version: str,
    verdict: object,
) -> None:
    """Best-effort SageMaker Experiments log for a gate verdict.

    Records a ``promotion_gate_verdict`` artifact tagged with the decision
    so auditors can pull every gate run from the Experiments stream alone
    (CLAUDE.md §1.14). Failures swallow — tracker outage must not block
    promotion decisions.
    """
    try:
        from core.compliance.sagemaker_compliance_tracker import (
            build_sagemaker_compliance_tracker,
        )
        tracker = build_sagemaker_compliance_tracker(pipeline_config)
        fria = getattr(verdict, "fria", None)
        ai_risk = getattr(verdict, "ai_risk", None)
        tracker.log_promotion_decision(
            model_version=model_version,
            decision=getattr(verdict, "decision", "unknown"),
            reason=getattr(verdict, "reason", ""),
            fria_result=fria,
            ai_risk_assessment=ai_risk,
        )
    except Exception as exc:
        logger.warning(
            "SageMaker compliance tracker failed for gate verdict "
            "(non-fatal): %s", exc,
        )


def _decide_promotion(
    registry,
    new_version: str,
    new_metrics: dict,
    fidelity_summary: dict,
    force_promote: bool,
    aws_region: str,
    pipeline_config: Optional[dict] = None,
) -> None:
    """Run the Champion-Challenger offline gate and act on its verdict.

    See :func:`_register_model` for the decision matrix.  Every outcome
    is recorded to the immutable audit log via
    :class:`core.monitoring.audit_logger.AuditLogger`.
    """
    champion_version = registry.get_promoted()

    # Force-promote: operator override. No comparison, no veto.
    if force_promote:
        registry.promote(new_version)
        logger.info(
            "Model %s force-promoted to champion (previous=%s)",
            new_version, champion_version or "<none>",
        )
        _audit_promotion(
            champion_version=champion_version,
            challenger_version=new_version,
            decision="force_promote",
            reason="Operator override via --force-promote",
            trigger="manual",
        )
        return

    # Bootstrap: first model, nothing to compete against.
    if champion_version is None:
        registry.promote(new_version)
        logger.info("Model %s bootstrap-promoted (no prior champion)", new_version)
        _audit_promotion(
            champion_version=None,
            challenger_version=new_version,
            decision="bootstrap",
            reason="No prior champion in registry",
            trigger="auto",
        )
        return

    # Safety floor: a challenger that fails fidelity must not be promoted,
    # even if its training metrics beat the champion.  This preserves the
    # teacher-student fidelity guarantee regardless of Competition verdict.
    fidelity_failed = int(fidelity_summary.get("failed", 0))
    if fidelity_failed > 0:
        logger.warning(
            "Model %s registered but NOT eligible for promotion: "
            "%d fidelity failures",
            new_version, fidelity_failed,
        )
        _audit_promotion(
            champion_version=champion_version,
            challenger_version=new_version,
            decision="reject",
            reason=f"{fidelity_failed} fidelity failures",
            trigger="auto",
        )
        return

    # Offline Champion-Challenger competition on recorded training metrics.
    from core.evaluation.model_competition import (
        ModelCandidate,
        ModelCompetition,
    )

    champion_manifest = None
    for v in registry.list_versions():
        if v.version == champion_version:
            champion_manifest = v
            break

    if champion_manifest is None:
        # Shouldn't happen — get_promoted returned this version — but guard
        # rather than crash the whole pipeline.
        logger.warning(
            "Champion %s manifest not found; registering challenger without promotion",
            champion_version,
        )
        _audit_promotion(
            champion_version=champion_version,
            challenger_version=new_version,
            decision="reject",
            reason=f"Champion manifest {champion_version} not loadable",
            trigger="auto",
        )
        return

    champion_candidate = ModelCandidate(
        model_id=champion_version,
        model_uri="",
        model_type="ple_teacher",
        version=champion_version,
        trained_at=champion_manifest.created_at,
        metrics=dict(champion_manifest.teacher_metrics or {}),
    )
    challenger_candidate = ModelCandidate(
        model_id=new_version,
        model_uri="",
        model_type="ple_teacher",
        version=new_version,
        trained_at=datetime.now().isoformat(),
        metrics=dict(new_metrics or {}),
    )

    # Sprint 2 S15: honour serving.competition.auto_promote (defaults false
    # in pipeline.yaml to comply with EU AI Act Art. 14). Falling back to
    # legacy default only if the yaml block is missing entirely.
    from core.evaluation.model_competition import CompetitionConfig

    comp_cfg = None
    if pipeline_config:
        comp_cfg = CompetitionConfig.from_dict(
            (pipeline_config.get("serving") or {}).get("competition")
        )
    competition = ModelCompetition(config=comp_cfg)
    result = competition.evaluate(champion_candidate, challenger_candidate)

    if result.promotion_approved:
        # Sprint 2 post-gate: FRIA + AI Risk (safety floor layered on top
        # of ModelCompetition). Disabled by default; enable via
        # compliance.promotion_gate.enabled in pipeline.yaml.
        gate_verdict = _run_promotion_gate(
            new_version, pipeline_config, registry=registry,
        )
        if gate_verdict is not None and gate_verdict.blocks_promotion:
            logger.warning(
                "Model %s rejected by promotion gate: %s",
                new_version, gate_verdict.reason,
            )
            _audit_promotion(
                champion_version=champion_version,
                challenger_version=new_version,
                decision="reject",
                reason=f"promotion_gate: {gate_verdict.reason}",
                comparison=result.comparison,
                significance=result.significance,
                trigger="auto",
                gate_details=dict(gate_verdict.details or {}),
            )
            return

        registry.promote(new_version)
        logger.info(
            "Model %s promoted to champion (previous=%s): %s",
            new_version, champion_version, result.decision_reason,
        )
        _audit_promotion(
            champion_version=champion_version,
            challenger_version=new_version,
            decision="promote",
            reason=result.decision_reason,
            comparison=result.comparison,
            significance=result.significance,
            trigger="auto",
            gate_details=(
                dict(gate_verdict.details or {})
                if gate_verdict is not None else None
            ),
        )
    else:
        logger.info(
            "Model %s registered (not promoted): %s",
            new_version, result.decision_reason,
        )
        _audit_promotion(
            champion_version=champion_version,
            challenger_version=new_version,
            decision="reject",
            reason=result.decision_reason,
            comparison=result.comparison,
            significance=result.significance,
            trigger="auto",
        )


def _register_model(
    config,
    s3_base,
    version,
    teacher_uri,
    student_uri,
    force_promote: bool = False,
):
    """Repackage SageMaker Job outputs into ModelRegistry versioned structure.

    Reads raw artifacts from SageMaker output paths, restructures into
    the registry format, and runs the Champion-Challenger offline gate
    before promoting.

    Promotion logic:
      - ``force_promote=True``                  -> always promote (operator override).
      - No current champion                     -> bootstrap promote.
      - ``ModelCompetition.evaluate()`` approves -> promote.
      - Otherwise                               -> register only (``promoted=False``).

    Every decision is written to the immutable audit log
    (:class:`core.monitoring.audit_logger.AuditLogger`) so that later
    reviewers can reconstruct why a version did or did not become champion.
    """
    from core.serving.model_registry import ModelRegistry

    registry = ModelRegistry(
        s3_base=f"{s3_base}/artifacts/",
        local_base="/tmp/model_registry/",
        region=config.aws.region,
    )

    # Download raw artifacts to temp dir for repackaging
    import tempfile
    import json
    tmp = tempfile.mkdtemp(prefix="register_")

    try:
        import boto3
        s3 = boto3.client("s3", region_name=config.aws.region)

        # Download teacher model.pth from model.tar.gz
        teacher_state_dict = None
        teacher_config = None
        training_metrics = None

        try:
            import tarfile, io, torch

            # Parse S3 URI
            parts = teacher_uri.replace("s3://", "").split("/", 1)
            bucket, key = parts[0], parts[1]

            # Download tar.gz
            tar_buf = io.BytesIO()
            s3.download_fileobj(bucket, key, tar_buf)
            tar_buf.seek(0)

            with tarfile.open(fileobj=tar_buf, mode="r:gz") as tar:
                tar.extractall(tmp)

            # Load state_dict
            model_path = Path(tmp) / "model.pth"
            if model_path.exists():
                teacher_state_dict = torch.load(str(model_path), map_location="cpu")

            # Load config
            config_path = Path(tmp) / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    teacher_config = json.load(f)

            # Load metrics
            metrics_path = Path(tmp) / "training_metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    training_metrics = json.load(f)

        except Exception as e:
            logger.warning("Failed to extract teacher artifacts: %s", e)
            training_metrics = {}

        # List student models from S3.
        #
        # SageMaker Training Jobs auto-tar /opt/ml/model into an
        # output/model.tar.gz artefact. distill_entry.py writes
        # {task}/model.lgbm files under /opt/ml/model, so the students'
        # S3 prefix contains a single model.tar.gz rather than the
        # flat {task}/model.lgbm files the registry expects. Detect the
        # tarball, extract it locally, and re-upload the flat layout to
        # a dedicated "extracted/" sub-prefix so both the scan below and
        # downstream Lambda predict code can pull individual model files.
        students: dict[str, str] = {}
        student_metadata: dict[str, dict] = {}
        parts = student_uri.replace("s3://", "").split("/", 1)
        bucket, prefix = parts[0], parts[1]

        scan_prefix = prefix
        try:
            import tarfile, io
            paginator = s3.get_paginator("list_objects_v2")
            tar_key: Optional[str] = None
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    if obj["Key"].endswith("/output/model.tar.gz"):
                        tar_key = obj["Key"]
                        break
                if tar_key:
                    break

            if tar_key:
                logger.info(
                    "Detected distillation tarball: s3://%s/%s — extracting",
                    bucket, tar_key,
                )
                extract_dir = Path(tmp) / "students_extract"
                extract_dir.mkdir(parents=True, exist_ok=True)
                tar_buf = io.BytesIO()
                s3.download_fileobj(bucket, tar_key, tar_buf)
                tar_buf.seek(0)
                with tarfile.open(fileobj=tar_buf, mode="r:gz") as tar:
                    tar.extractall(extract_dir)
                scan_prefix = f"{prefix.rstrip('/')}/extracted"
                uploaded = 0
                for fp in extract_dir.rglob("*"):
                    if not fp.is_file():
                        continue
                    rel = fp.relative_to(extract_dir).as_posix()
                    s3_key = f"{scan_prefix.rstrip('/')}/{rel}"
                    s3.upload_file(str(fp), bucket, s3_key)
                    uploaded += 1
                logger.info(
                    "Re-uploaded %d flat student artifacts under s3://%s/%s/",
                    uploaded, bucket, scan_prefix,
                )

                # Load LGBM Boosters from the extracted files. ModelRegistry.
                # package expects live Booster objects (it calls
                # model.save_model inside the version dir). The S3 prefix
                # above exists for downstream Lambda/predict code that pulls
                # flat files.
                import lightgbm as _lgb
                for task_dir in sorted(extract_dir.iterdir()):
                    if not task_dir.is_dir():
                        continue
                    lgbm_file = task_dir / "model.lgbm"
                    if not lgbm_file.exists():
                        continue
                    task_name = task_dir.name
                    try:
                        students[task_name] = _lgb.Booster(
                            model_file=str(lgbm_file),
                        )
                    except Exception as load_err:
                        logger.warning(
                            "Failed to load Booster for task '%s': %s",
                            task_name, load_err,
                        )
                        continue
                    meta_file = task_dir / "metadata.json"
                    if meta_file.exists():
                        try:
                            with open(meta_file, encoding="utf-8") as mf:
                                student_metadata[task_name] = json.load(mf)
                        except Exception:
                            logger.debug(
                                "Could not parse metadata for '%s'",
                                task_name, exc_info=True,
                            )
            else:
                # No tarball: treat scan_prefix as a flat layout and fetch
                # each .lgbm file locally before loading into a Booster.
                import lightgbm as _lgb
                flat_dir = Path(tmp) / "students_flat"
                flat_dir.mkdir(parents=True, exist_ok=True)
                for page in paginator.paginate(
                    Bucket=bucket, Prefix=scan_prefix,
                ):
                    for obj in page.get("Contents", []):
                        key = obj["Key"]
                        if key.endswith("/model.lgbm"):
                            task_name = key.split("/")[-2]
                            local = flat_dir / task_name / "model.lgbm"
                            local.parent.mkdir(parents=True, exist_ok=True)
                            s3.download_file(bucket, key, str(local))
                            try:
                                students[task_name] = _lgb.Booster(
                                    model_file=str(local),
                                )
                            except Exception as load_err:
                                logger.warning(
                                    "Failed to load Booster for '%s': %s",
                                    task_name, load_err,
                                )
                        elif key.endswith("/metadata.json"):
                            task_name = key.split("/")[-2]
                            meta_obj = s3.get_object(Bucket=bucket, Key=key)
                            student_metadata[task_name] = json.loads(
                                meta_obj["Body"].read().decode()
                            )
        except Exception as e:
            logger.warning("Failed to list student artifacts: %s", e)

        # Package into registry
        model_version = registry.package(
            version=version,
            teacher_state_dict=teacher_state_dict,
            teacher_config=teacher_config,
            training_metrics=training_metrics or {},
            students=students,
            student_metadata=student_metadata,
        )

        logger.info(
            "Model registered: %s (%d students, promoted=%s)",
            version, len(students), model_version.promoted,
        )

        # --- Champion-Challenger gate --------------------------------------
        # Forward the raw pipeline config so the Sprint 2 post-gate
        # (FRIA + AI Risk) can read `compliance.promotion_gate`.
        raw_pipeline_cfg: Optional[dict] = None
        try:
            import yaml
            cfg_path = Path("configs/pipeline.yaml")
            if cfg_path.exists():
                raw_pipeline_cfg = yaml.safe_load(
                    cfg_path.read_text(encoding="utf-8")
                )
        except Exception:
            logger.debug(
                "Could not load raw pipeline.yaml for promotion_gate; "
                "continuing with gate disabled",
                exc_info=True,
            )

        _decide_promotion(
            registry=registry,
            new_version=version,
            new_metrics=training_metrics or {},
            fidelity_summary=model_version.fidelity_summary,
            force_promote=force_promote,
            aws_region=config.aws.region,
            pipeline_config=raw_pipeline_cfg,
        )

    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
