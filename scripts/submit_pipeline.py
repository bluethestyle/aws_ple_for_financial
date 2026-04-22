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
        "--force-promote", action="store_true",
        help=(
            "Promote the new model to champion unconditionally, bypassing the "
            "Champion-Challenger offline competition gate. Use for bootstrap "
            "or emergency rollback."
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


def _build_staging_dir() -> str:
    """Build the shared SageMaker source staging dir once for every Job.

    CLAUDE.md §1.5: ``source 패키지는 1회만 빌드하고 모든 Job에서 재사용한다``.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from package_source import build_staging  # type: ignore[import]
    return build_staging()


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

    # Phase 0 decision.
    if args.features_uri:
        logger.info(
            "--- Step 1: Phase 0 skipped (features_uri=%s) ---",
            args.features_uri,
        )
        # Trainer reads ``config.data.source`` for its TrainingInput, so
        # override to the user-provided Phase 0 prefix.
        config.data.source = args.features_uri
        features_uri = args.features_uri
    else:
        # TODO(phase0-cloud): submit_pipeline does not yet produce a
        # fresh Phase 0 output on SageMaker. An end-to-end Phase 0
        # cloud path would need either
        #   (a) a unifying entrypoint (``containers/phase0/entrypoint.py``)
        #       running FeatureGroupPipeline.fit_transform inside a
        #       single Training Job, or
        #   (b) per-group Processing Jobs via
        #       SageMakerProcessingJob.submit_feature_groups() *plus* a
        #       downstream integration Job that concatenates the per-group
        #       parquet outputs into the unified feature matrix + schema
        #       the trainer expects (``feature_schema.json``,
        #       ``feature_stats.json``, ``normalizer/``).
        # Neither exists today. Until (a) lands, operators must supply
        # ``--features-uri`` pointing at an existing Phase 0 output.
        logger.error(
            "Phase 0 cloud submission not yet supported. Re-run with "
            "--features-uri s3://<bucket>/<phase0_prefix>/ pointing at an "
            "existing Phase 0 output.",
        )
        if not args.dry_run:
            sys.exit(2)
        features_uri = f"{s3_base}/features/{ts}/"  # dry-run placeholder

    # Step 2: Teacher training (Phase 1 warm-up + Phase 2 fine-tune).
    logger.info("--- Step 2: PLE Training (2-phase) ---")
    trainer = SageMakerTrainer(config)
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

        # List student models from S3
        students = {}
        student_metadata = {}
        parts = student_uri.replace("s3://", "").split("/", 1)
        bucket, prefix = parts[0], parts[1]

        try:
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if key.endswith("/model.lgbm"):
                        task_name = key.split("/")[-2]
                        students[task_name] = f"s3://{bucket}/{key}"
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
