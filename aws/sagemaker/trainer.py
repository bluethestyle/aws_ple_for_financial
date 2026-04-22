"""
SageMaker Training Job launcher.

Builds on the verified submit patterns from
``scripts/run_sagemaker_teacher.py`` and
``scripts/run_sagemaker_distillation.py`` — both scripts have produced
successful training jobs on the real cluster (S3 output trails dated
2026-04-14 through 2026-04-20 under
``s3://aiops-ple-financial/teacher-full-30ep-*/``).

Key differences from the previous scaffold that never ran:

* ``source_dir`` points at the full staging directory built by
  :func:`scripts.package_source.build_staging` so that the container has
  ``core/``, ``configs/``, and ``containers/`` available at
  ``/opt/ml/code``. The old scaffold set ``source_dir="containers/training/"``
  which fails ``from core.pipeline.config import load_config`` inside the
  container.
* Hyperparameters carry *paths* to config YAMLs
  (``"config": "configs/pipeline.yaml"``), not a serialised JSON dump.
  SageMaker enforces a per-HP length limit well below the size of the
  merged PipelineConfig, and the container's loader already knows how
  to read these paths directly (see ``containers/training/train.py::get_hyperparameters``).
* Job names are sanitised against the SageMaker regex
  ``[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}``. Underscores in ``task_name``
  (e.g. ``santander_ple``) are rewritten to hyphens; the trailing
  timestamp is ``%m%d-%H%M`` to stay under the 63-char cap.

Usage::

    config = load_config("configs/santander/pipeline.yaml")
    trainer = SageMakerTrainer(config)

    from scripts.package_source import build_staging
    staging = build_staging()

    phase1 = trainer.launch_phase1(staging_dir=staging, wait=True)
    phase2 = trainer.launch_phase2(
        staging_dir=staging,
        phase1_model_uri=phase1["s3_model_uri"],
        wait=True,
    )
    distill = trainer.launch_distillation(
        staging_dir=staging,
        teacher_uri=phase2["s3_model_uri"],
        wait=True,
    )
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from typing import Any

import boto3
import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.pytorch import PyTorch

from core.pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Container-internal paths (relative to staging_dir root which SageMaker
# extracts to /opt/ml/code). Must match the files packaged by
# scripts/package_source.py::build_staging.
CONTAINER_CONFIG = "configs/pipeline.yaml"
CONTAINER_DATASET_CONFIG_SANTANDER = "configs/datasets/santander.yaml"

# Default distillation instance: CPU is sufficient for LGBM training and
# SHAP. The scenario-tested value from run_sagemaker_distillation.py uses
# the same instance family.
DEFAULT_DISTILL_INSTANCE = "ml.m5.2xlarge"
DEFAULT_DISTILL_MAX_RUN_S = 10800   # 3h
DEFAULT_DISTILL_MAX_WAIT_S = 14400  # 4h (max_run + 1h per CLAUDE.md §1.5)

METRIC_DEFINITIONS = [
    {"Name": "train:loss",          "Regex": r"train_loss=([0-9\.e\-\+]+)"},
    {"Name": "val:loss",            "Regex": r"val_loss=([0-9\.e\-\+]+)"},
    {"Name": "val:avg_auc",         "Regex": r"avg_auc=([0-9\.]+)"},
    {"Name": "val:avg_f1_macro",    "Regex": r"avg_f1_macro=([0-9\.]+)"},
    {"Name": "val:avg_ndcg3",       "Regex": r"avg_ndcg@3=([0-9\.]+)"},
    {"Name": "val:avg_mae",         "Regex": r"avg_mae=([0-9\.]+)"},
    {"Name": "epoch",               "Regex": r"Epoch (\d+):"},
]

DISTILL_METRIC_DEFINITIONS = [
    {"Name": "distill:num_students",    "Regex": r"Tasks distilled: ([0-9]+)"},
    {"Name": "distill:passed_fidelity", "Regex": r"Fidelity summary: ([0-9]+)/"},
    {"Name": "distill:distill_tasks",   "Regex": r"Distillation: ([0-9]+) tasks"},
    {"Name": "distill:direct_tasks",    "Regex": r"Direct LGBM: ([0-9]+) tasks"},
]

_JOB_NAME_RE = re.compile(r"[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_job_name(raw: str, max_len: int = 63) -> str:
    """Return a SageMaker-safe Job name.

    SageMaker Job names must match ``[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}`` —
    underscores, dots, and slashes are rejected. This helper rewrites
    those to hyphens, collapses consecutive hyphens, strips leading /
    trailing hyphens, and caps the result at ``max_len``.
    """
    sub = re.sub(r"[^a-zA-Z0-9]+", "-", raw).strip("-")
    sub = re.sub(r"-+", "-", sub)
    if not sub:
        return "job"
    sub = sub[:max_len].rstrip("-")
    if not _JOB_NAME_RE.fullmatch(sub):
        # Should not happen after the regex rewrite, but guard anyway.
        raise ValueError(
            f"Could not sanitise SageMaker job name from input {raw!r} "
            f"(got {sub!r} which fails the SM regex)"
        )
    return sub


def _dataset_config_path(cfg: PipelineConfig) -> str | None:
    """Return the container-relative dataset YAML for the adapter, or None.

    Currently only the santander dataset has a dedicated split-config
    entry in ``configs/datasets/``. Other adapters fall back to the
    single-file pipeline YAML.
    """
    adapter = getattr(cfg, "adapter", "") or ""
    if adapter.lower() == "santander":
        return CONTAINER_DATASET_CONFIG_SANTANDER
    return None


# ---------------------------------------------------------------------------
# SageMakerTrainer
# ---------------------------------------------------------------------------

class SageMakerTrainer:
    """Launch SageMaker Training Jobs (teacher Phase 1/2 + distillation).

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration parsed from YAML. ``config.aws.*`` supplies
        region, bucket, role ARN, instance type, Spot flag, and max_run.
    session : sagemaker.Session, optional
        Existing SageMaker session (created if *None*).
    """

    def __init__(
        self,
        config: PipelineConfig,
        session: sagemaker.Session | None = None,
    ):
        self.config = config
        self.session = session or sagemaker.Session()
        self._sm_client = boto3.client(
            "sagemaker", region_name=config.aws.region,
        )
        # Cached per-run timestamp; all phases submitted within a single
        # SageMakerTrainer instance share this so that job names cluster
        # together in the console.
        self._run_stamp = time.strftime("%m%d-%H%M")

    # ------------------------------------------------------------------
    # Phase 1 (warm-up) / Phase 2 (fine-tune) / single-shot
    # ------------------------------------------------------------------

    def launch(self, staging_dir: str, wait: bool = True) -> dict[str, Any]:
        """Launch a single (non-phased) training job."""
        return self._launch_training(
            staging_dir=staging_dir,
            phase="full",
            extra_hyperparams={},
            wait=wait,
        )

    def launch_phase1(
        self, staging_dir: str, wait: bool = True,
    ) -> dict[str, Any]:
        """Phase 1 — warm-up: shared layers, frozen towers."""
        return self._launch_training(
            staging_dir=staging_dir,
            phase="phase1",
            extra_hyperparams={
                "phase": "1",
                "freeze_towers": "true",
                "epochs": str(max(1, self.config.training.epochs // 3)),
            },
            wait=wait,
        )

    def launch_phase2(
        self,
        staging_dir: str,
        phase1_model_uri: str,
        wait: bool = True,
    ) -> dict[str, Any]:
        """Phase 2 — fine-tune: unfreeze, lower LR."""
        lr = self.config.training.learning_rate
        return self._launch_training(
            staging_dir=staging_dir,
            phase="phase2",
            extra_hyperparams={
                "phase": "2",
                "freeze_towers": "false",
                "pretrained_model_uri": phase1_model_uri,
                "learning_rate": str(lr * 0.1),
                "epochs": str(self.config.training.epochs),
            },
            wait=wait,
        )

    # ------------------------------------------------------------------
    # Distillation (PLE → LGBM), CPU instance
    # ------------------------------------------------------------------

    def launch_distillation(
        self,
        staging_dir: str,
        teacher_uri: str,
        wait: bool = True,
        instance_type: str = DEFAULT_DISTILL_INSTANCE,
        max_run_s: int = DEFAULT_DISTILL_MAX_RUN_S,
        max_wait_s: int = DEFAULT_DISTILL_MAX_WAIT_S,
    ) -> dict[str, Any]:
        """Launch the PLE→LGBM distillation job on a CPU instance.

        Two :class:`TrainingInput` channels are wired up:

        * ``train`` — Phase 0 feature parquet + schemas.
        * ``model`` — directory containing the teacher checkpoint (S3
          prefix, not a single ``model.tar.gz`` file).
        """
        cfg = self.config
        aws = cfg.aws
        s3_base = f"s3://{aws.s3_bucket}/{cfg.task_name}"
        # Reuse the existing Phase 0 output by default. submit_pipeline
        # can override via ``config.data.source`` already being an S3 URI
        # produced by Phase 0.
        data_uri = cfg.data.source
        output_uri = f"{s3_base}/students/{self._run_stamp}"

        task_slug = _sanitize_job_name(cfg.task_name, max_len=20)
        job_name = _sanitize_job_name(
            f"{task_slug}-distill-{self._run_stamp}",
        )

        hyperparams: dict[str, str] = {
            "config": CONTAINER_CONFIG,
        }
        dataset_yaml = _dataset_config_path(cfg)
        if dataset_yaml:
            hyperparams["dataset_config"] = dataset_yaml
        hyperparams["skip_fidelity_gate"] = "false"
        hyperparams["scenario"] = "submit-pipeline"

        logger.info("Launching SageMaker Distillation Job: %s", job_name)
        logger.info("  Instance: %s (Spot=%s, CPU)", instance_type, aws.use_spot)
        logger.info("  Teacher: %s", teacher_uri)
        logger.info("  Data:    %s", data_uri)
        logger.info("  Output:  %s", output_uri)

        estimator = PyTorch(
            entry_point="containers/distillation/distill_entry.py",
            source_dir=staging_dir,
            role=aws.role_arn,
            instance_type=instance_type,
            instance_count=1,
            framework_version="2.1",
            py_version="py310",
            hyperparameters=hyperparams,
            use_spot_instances=aws.use_spot,
            max_run=max_run_s,
            max_wait=max_wait_s if aws.use_spot else None,
            output_path=output_uri,
            disable_profiler=True,
            environment={
                "PYTHONPATH": "/opt/ml/code",
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUNBUFFERED": "1",
            },
            sagemaker_session=self.session,
            base_job_name=_sanitize_job_name(f"{task_slug}-distill"),
            metric_definitions=DISTILL_METRIC_DEFINITIONS,
            tags=[
                {"Key": "Project", "Value": cfg.task_name},
                {"Key": "Phase", "Value": "distill"},
            ],
        )

        estimator.fit(
            inputs={
                "train": TrainingInput(
                    data_uri, content_type="application/x-parquet",
                ),
                "model": TrainingInput(
                    teacher_uri, content_type="application/octet-stream",
                ),
            },
            job_name=job_name,
            wait=False,
            logs="None",
        )
        logger.info("Submitted: %s (wait=%s)", job_name, wait)

        result: dict[str, Any] = {
            "job_name": job_name,
            "phase": "distill",
            "instance_type": instance_type,
            "spot": aws.use_spot,
            "output_path": output_uri,
            "teacher_uri": teacher_uri,
        }
        if wait:
            attached = self.attach_running_job(job_name)
            result.update(
                status=attached["status"],
                s3_output_uri=attached["s3_model_uri"],
                billable_seconds=attached["billable_seconds"],
            )
            if attached["status"] != "Completed":
                raise RuntimeError(
                    f"Distillation job {job_name} ended with status "
                    f"{attached['status']} — aborting pipeline.",
                )
        return result

    # ------------------------------------------------------------------
    # Job lifecycle helpers
    # ------------------------------------------------------------------

    def describe_job(self, job_name: str) -> dict:
        """Return the SageMaker DescribeTrainingJob response."""
        return self._sm_client.describe_training_job(
            TrainingJobName=job_name,
        )

    def wait_for_job(self, job_name: str) -> str:
        """Block until a running job completes. Returns terminal status.

        Uses the AWS-side waiter rather than the SDK's log streamer to
        avoid the Windows cp949 UnicodeEncodeError path that kills the
        orchestrator when container logs contain non-ASCII characters
        (e.g. em-dash, Korean).
        """
        waiter = self._sm_client.get_waiter(
            "training_job_completed_or_stopped",
        )
        logger.info("Waiting for job %s ...", job_name)
        waiter.wait(
            TrainingJobName=job_name,
            WaiterConfig={"Delay": 30, "MaxAttempts": 240},
        )
        desc = self.describe_job(job_name)
        status = desc["TrainingJobStatus"]
        logger.info("Job %s finished with status: %s", job_name, status)
        return status

    def attach_running_job(
        self, job_name: str, poll_interval_s: int = 30,
    ) -> dict[str, Any]:
        """Attach to a Training Job already running on the cluster.

        When ``submit_pipeline`` crashes (typically due to a local-side
        issue like the cp949 encoding bug) the underlying Training Job
        usually keeps running — SageMaker-side state is fully
        decoupled. ``attach_running_job`` polls the job until it reaches
        a terminal status and returns the same dict shape as
        :meth:`_launch_training`, so the orchestrator can continue with
        Phase 2 / Distillation without resubmitting Phase 1.

        Parameters
        ----------
        job_name : str
            SageMaker Training Job name (as returned by an earlier
            ``launch_*`` call).
        poll_interval_s : int
            Seconds between ``describe_training_job`` polls (default 30).

        Returns
        -------
        dict
            ``job_name``, ``status``, ``s3_model_uri`` (if Completed),
            ``billable_seconds``, ``output_path``, ``instance_type``.
        """
        logger.info(
            "Attaching to running job %s (poll every %ds)...",
            job_name, poll_interval_s,
        )
        terminal = {"Completed", "Failed", "Stopped"}
        while True:
            desc = self.describe_job(job_name)
            status = desc["TrainingJobStatus"]
            secondary = desc.get("SecondaryStatus", "")
            if status in terminal:
                break
            logger.info("  [%s] %s ...", status, secondary)
            time.sleep(poll_interval_s)

        billable = int(desc.get("BillableTimeInSeconds", 0))
        model_arts = (desc.get("ModelArtifacts") or {}).get(
            "S3ModelArtifacts", "",
        )
        output = (desc.get("OutputDataConfig") or {}).get("S3OutputPath", "")
        instance = (
            (desc.get("ResourceConfig") or {}).get("InstanceType", "")
        )
        checkpoint_uri = (
            (desc.get("CheckpointConfig") or {}).get("S3Uri", "")
        )
        logger.info(
            "Job %s finished: status=%s, billable=%ds, model=%s",
            job_name, status, billable, model_arts or "<none>",
        )
        return {
            "job_name": job_name,
            "status": status,
            "s3_model_uri": model_arts,
            "checkpoint_s3_uri": checkpoint_uri,
            "output_path": output,
            "billable_seconds": billable,
            "instance_type": instance,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _launch_training(
        self,
        staging_dir: str,
        phase: str,
        extra_hyperparams: dict[str, str],
        wait: bool,
    ) -> dict[str, Any]:
        """Create and fit a PyTorch estimator for one training phase."""
        cfg = self.config
        aws = cfg.aws

        # -- S3 paths --
        s3_base = f"s3://{aws.s3_bucket}/{cfg.task_name}"
        output_path = f"{s3_base}/models/{phase}/{self._run_stamp}"
        checkpoint_s3 = f"{s3_base}/checkpoints/{phase}/{self._run_stamp}"
        checkpoint_local = "/opt/ml/checkpoints"

        # -- Job name --
        task_slug = _sanitize_job_name(cfg.task_name, max_len=20)
        phase_slug = _sanitize_job_name(phase, max_len=12)
        job_name = _sanitize_job_name(
            f"{task_slug}-{phase_slug}-{self._run_stamp}",
        )

        # -- Hyperparameters: carry paths to YAMLs, not JSON blobs --
        hyperparams: dict[str, str] = {
            "config": CONTAINER_CONFIG,
            "task_name": cfg.task_name,
            "batch_size": str(cfg.training.batch_size),
            "epochs": str(cfg.training.epochs),
            "learning_rate": str(cfg.training.learning_rate),
            "early_stopping_patience": str(
                cfg.training.early_stopping_patience,
            ),
            "seed": str(cfg.training.seed),
            "amp": "true",
            "num_workers": "2",
            "use_adatt": "false",
            "use_grad_surgery": "false",
        }
        dataset_yaml = _dataset_config_path(cfg)
        if dataset_yaml:
            hyperparams["dataset_config"] = dataset_yaml
        hyperparams.update(extra_hyperparams)

        # -- Estimator --
        estimator = PyTorch(
            entry_point="containers/training/train.py",
            source_dir=staging_dir,
            role=aws.role_arn,
            instance_type=aws.instance_type,
            instance_count=1,
            framework_version="2.1",
            py_version="py310",
            hyperparameters=hyperparams,
            use_spot_instances=aws.use_spot,
            max_run=aws.max_run_seconds,
            max_wait=(
                aws.max_run_seconds + 3600
            ) if aws.use_spot else None,
            checkpoint_s3_uri=checkpoint_s3 if aws.use_spot else None,
            checkpoint_local_path=(
                checkpoint_local if aws.use_spot else None
            ),
            output_path=output_path,
            disable_profiler=True,
            environment={
                "PHASE": phase,
                "PYTHONPATH": "/opt/ml/code",
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUNBUFFERED": "1",
                "NCCL_DEBUG": "WARN",
            },
            sagemaker_session=self.session,
            base_job_name=_sanitize_job_name(f"{task_slug}-{phase_slug}"),
            metric_definitions=METRIC_DEFINITIONS,
            enable_sagemaker_metrics=True,
            tags=[
                {"Key": "Project", "Value": cfg.task_name},
                {"Key": "Phase", "Value": phase},
            ],
        )

        # -- Input channels --
        inputs = self._build_input_channels()

        logger.info("Launching SageMaker Training Job: %s", job_name)
        logger.info("  Phase: %s", phase)
        logger.info(
            "  Instance: %s (Spot=%s)",
            aws.instance_type, aws.use_spot,
        )
        logger.info("  Data: %s", cfg.data.source)
        logger.info("  Output: %s", output_path)
        if aws.use_spot:
            logger.info("  Checkpoint: %s", checkpoint_s3)

        # Submit but never stream CloudWatch logs — SageMaker's SDK
        # streamer prints raw container stdout to the local console,
        # which dies on Windows cp949 whenever a non-ASCII byte (e.g.
        # em-dash, Korean logging) lands in the stream. We wait via the
        # AWS waiter + describe_training_job polling instead.
        estimator.fit(
            inputs,
            job_name=job_name,
            wait=False,
            logs="None",
        )
        logger.info("Submitted: %s (wait=%s)", job_name, wait)

        result: dict[str, Any] = {
            "job_name": job_name,
            "phase": phase,
            "instance_type": aws.instance_type,
            "spot": aws.use_spot,
            "output_path": output_path,
            "checkpoint_s3_uri": checkpoint_s3 if aws.use_spot else "",
        }
        if wait:
            attached = self.attach_running_job(job_name)
            result.update(
                status=attached["status"],
                s3_model_uri=attached["s3_model_uri"],
                billable_seconds=attached["billable_seconds"],
            )
            if attached.get("checkpoint_s3_uri"):
                result["checkpoint_s3_uri"] = attached["checkpoint_s3_uri"]
            if attached["status"] != "Completed":
                raise RuntimeError(
                    f"Training job {job_name} ended with status "
                    f"{attached['status']} — aborting pipeline.",
                )
        return result

    def _build_input_channels(self) -> dict[str, TrainingInput]:
        """Construct S3 input channel mapping.

        Channels
        --------
        * ``train`` — training data (Parquet)
        * ``validation`` — validation data (Parquet), same source with
          distribution mode ``FullyReplicated`` so every instance gets
          the entire validation set.
        """
        source = self.config.data.source
        return {
            "train": TrainingInput(
                source,
                content_type="application/x-parquet",
                distribution="ShardedByS3Key",
            ),
            "validation": TrainingInput(
                source,
                content_type="application/x-parquet",
                distribution="FullyReplicated",
            ),
        }
