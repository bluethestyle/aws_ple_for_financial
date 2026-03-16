"""
SageMaker Training Job launcher with 2-phase training, Spot instance
checkpoint auto-resume, and CloudWatch metric streaming.

Phase 1 (warm-up):  Trains shared experts + CGC gating with frozen towers.
Phase 2 (fine-tune): Unfreezes everything, lowers LR, trains end-to-end.

Each phase is a separate SageMaker Training Job so that Spot interruptions
can resume from the latest checkpoint without re-running both phases.

Usage::

    config = load_config("configs/my_problem.yaml")
    trainer = SageMakerTrainer(config)

    # Single-phase (legacy)
    result = trainer.launch(wait=True)

    # Two-phase
    phase1 = trainer.launch_phase1(wait=True)
    phase2 = trainer.launch_phase2(phase1["s3_model_uri"], wait=True)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from typing import Any

import boto3
import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.pytorch import PyTorch

from ...core.pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CloudWatch metric definitions
# ---------------------------------------------------------------------------
METRIC_DEFINITIONS = [
    {"Name": "train:loss",      "Regex": r"train_loss=([0-9\.e\-\+]+)"},
    {"Name": "train:auc",       "Regex": r"train_auc=([0-9\.e\-\+]+)"},
    {"Name": "val:loss",        "Regex": r"val_loss=([0-9\.e\-\+]+)"},
    {"Name": "val:auc",         "Regex": r"val_auc=([0-9\.e\-\+]+)"},
    {"Name": "val:f1",          "Regex": r"val_f1=([0-9\.e\-\+]+)"},
    {"Name": "val:mae",         "Regex": r"val_mae=([0-9\.e\-\+]+)"},
    {"Name": "val:ndcg",        "Regex": r"val_ndcg=([0-9\.e\-\+]+)"},
    {"Name": "epoch",           "Regex": r"epoch=([0-9]+)"},
    {"Name": "learning_rate",   "Regex": r"learning_rate=([0-9\.e\-\+]+)"},
]


class SageMakerTrainer:
    """Launch SageMaker Training Jobs from a ``PipelineConfig``.

    Features
    --------
    * **2-phase training** — ``launch_phase1`` / ``launch_phase2`` for
      warm-up + fine-tune with separate hyperparameters.
    * **Spot instances** — ``use_spot=True`` (default) saves ~70% cost;
      checkpoint auto-resume is handled by ``checkpoint_s3_uri``.
    * **CloudWatch metrics** — stdout metric lines are captured by
      SageMaker and pushed to CloudWatch for dashboarding / alarms.
    * **Wait / no-wait** — ``wait=False`` launches asynchronously and
      returns the job name for later polling.

    Parameters
    ----------
    config : PipelineConfig
        Full pipeline configuration parsed from YAML.
    session : sagemaker.Session, optional
        Existing SageMaker session (created if *None*).

    Examples
    --------
    >>> config = load_config("configs/my_problem.yaml")
    >>> trainer = SageMakerTrainer(config)
    >>> result = trainer.launch(wait=True)
    >>> print(result["s3_model_uri"])
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def launch(self, wait: bool = True) -> dict[str, Any]:
        """Launch a single training job (backward-compatible entrypoint).

        Parameters
        ----------
        wait : bool
            Block until the job completes.

        Returns
        -------
        dict
            ``job_name``, ``s3_model_uri`` (if *wait*), ``instance_type``,
            ``spot``.
        """
        return self._launch_job(
            phase="single",
            extra_hyperparams={},
            wait=wait,
        )

    def launch_phase1(self, wait: bool = True) -> dict[str, Any]:
        """Phase 1 — warm-up: train shared layers, freeze towers.

        Passes ``phase=1`` and ``freeze_towers=true`` as hyperparameters
        so the container entry point can configure the model accordingly.

        Returns
        -------
        dict
            Same as :meth:`launch` with extra key ``phase``.
        """
        return self._launch_job(
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
        phase1_model_uri: str,
        wait: bool = True,
    ) -> dict[str, Any]:
        """Phase 2 — fine-tune: unfreeze everything, lower LR.

        Parameters
        ----------
        phase1_model_uri : str
            S3 URI of the ``model.tar.gz`` produced by phase 1.

        Returns
        -------
        dict
            Same as :meth:`launch` with extra key ``phase``.
        """
        lr = self.config.training.learning_rate
        return self._launch_job(
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

    def describe_job(self, job_name: str) -> dict:
        """Return the SageMaker DescribeTrainingJob response."""
        return self._sm_client.describe_training_job(
            TrainingJobName=job_name,
        )

    def wait_for_job(self, job_name: str) -> str:
        """Block until a running job completes.

        Returns
        -------
        str
            Final job status (``Completed``, ``Failed``, ``Stopped``).
        """
        waiter = self._sm_client.get_waiter("training_job_completed_or_stopped")
        logger.info(f"Waiting for job {job_name} ...")
        waiter.wait(
            TrainingJobName=job_name,
            WaiterConfig={"Delay": 30, "MaxAttempts": 240},
        )
        desc = self.describe_job(job_name)
        status = desc["TrainingJobStatus"]
        logger.info(f"Job {job_name} finished with status: {status}")
        return status

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _launch_job(
        self,
        phase: str,
        extra_hyperparams: dict[str, str],
        wait: bool,
    ) -> dict[str, Any]:
        """Create and fit a PyTorch estimator for one training phase."""
        cfg = self.config
        aws = cfg.aws
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        job_name = f"{cfg.task_name}-{phase}-{ts}"

        # -- S3 paths --
        s3_base = f"s3://{aws.s3_bucket}/{cfg.task_name}"
        output_path = f"{s3_base}/models/"
        checkpoint_s3 = f"{s3_base}/checkpoints/{phase}/"
        checkpoint_local = "/opt/ml/checkpoints"

        # -- Hyperparameters --
        hyperparams: dict[str, str] = {
            "config": json.dumps(self._config_to_dict()),
            "task_name": cfg.task_name,
            "batch_size": str(cfg.training.batch_size),
            "epochs": str(cfg.training.epochs),
            "learning_rate": str(cfg.training.learning_rate),
            "early_stopping_patience": str(cfg.training.early_stopping_patience),
            "seed": str(cfg.training.seed),
        }
        hyperparams.update(extra_hyperparams)

        # -- Estimator --
        estimator = PyTorch(
            entry_point="train.py",
            source_dir="containers/training/",
            role=aws.role_arn,
            instance_type=aws.instance_type,
            instance_count=1,
            framework_version="2.1",
            py_version="py310",
            output_path=output_path,
            # Spot instance settings — ~70% cost savings
            use_spot_instances=aws.use_spot,
            max_run=aws.max_run_seconds,
            max_wait=(aws.max_run_seconds + 3600) if aws.use_spot else None,
            # Checkpoint auto-resume on Spot interruption
            checkpoint_s3_uri=checkpoint_s3 if aws.use_spot else None,
            checkpoint_local_path=checkpoint_local if aws.use_spot else None,
            hyperparameters=hyperparams,
            metric_definitions=METRIC_DEFINITIONS,
            enable_sagemaker_metrics=True,
            # Environment variables
            environment={
                "PHASE": phase,
                "NCCL_DEBUG": "WARN",
            },
            tags=[
                {"Key": "Project", "Value": cfg.task_name},
                {"Key": "Phase", "Value": phase},
            ],
        )

        # -- Input channels --
        inputs = self._build_input_channels()

        logger.info(f"Launching SageMaker Training Job: {job_name}")
        logger.info(f"  Phase: {phase}")
        logger.info(f"  Instance: {aws.instance_type} (Spot={aws.use_spot})")
        logger.info(f"  Data: {cfg.data.source}")
        logger.info(f"  Output: {output_path}")
        if aws.use_spot:
            logger.info(f"  Checkpoint: {checkpoint_s3}")

        estimator.fit(
            inputs,
            job_name=job_name,
            wait=wait,
            logs="All" if wait else None,
        )

        result: dict[str, Any] = {
            "job_name": job_name,
            "phase": phase,
            "instance_type": aws.instance_type,
            "spot": aws.use_spot,
            "output_path": output_path,
        }
        if wait:
            result["s3_model_uri"] = estimator.model_data
        return result

    def _build_input_channels(self) -> dict[str, TrainingInput]:
        """Construct S3 input channel mapping.

        Channels
        --------
        * ``train`` — training data (Parquet)
        * ``validation`` — validation data (Parquet), same source with
          distribution mode ``FullyReplicated`` so every instance gets
          the entire validation set.

        Returns
        -------
        dict[str, TrainingInput]
        """
        source = self.config.data.source
        channels: dict[str, TrainingInput] = {
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
        return channels

    def _config_to_dict(self) -> dict:
        """Serialize ``PipelineConfig`` to a plain dict for JSON transport."""
        return asdict(self.config)
