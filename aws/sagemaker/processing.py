"""
SageMaker Processing Job launcher for Phase 0 feature engineering.

Two public entry points:

1. :meth:`SageMakerProcessingJob.submit_feature_groups` — the **production
   path** used by ``scripts/submit_pipeline.py::_run_full``. It loops
   over :class:`FeatureGroupConfig` objects and fires one Processing
   Job per group, each configured with ``FEATURE_GROUP_NAME``,
   ``FEATURE_GENERATOR``, and ``FEATURE_GENERATOR_PARAMS`` environment
   variables. This matches the env-var contract in
   ``containers/generators/base/entrypoint.py`` and mirrors the
   long-standing behaviour of
   :meth:`core.feature.group_pipeline.FeatureGroupPipeline._run_container_job`.

2. :meth:`SageMakerProcessingJob.run` — a single-job helper left in place
   for ad-hoc jobs (e.g. utilities). Uses a
   :class:`FrameworkProcessor(estimator_cls=SKLearn)` so that
   ``source_dir`` + ``requirements.txt`` auto-install works.

The verified submit patterns from
``scripts/run_sagemaker_teacher.py`` and
``scripts/run_sagemaker_distillation.py`` informed the design:

* Packaged ``source_dir`` built via
  :func:`scripts.package_source.build_staging` — the staging tree
  contains ``core/``, ``configs/``, ``containers/`` so every job has
  the imports it needs.
* Job names sanitised against SageMaker's regex.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import boto3
import sagemaker
from sagemaker.processing import (
    FrameworkProcessor,
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.pytorch.processing import PyTorchProcessor
from sagemaker.sklearn.estimator import SKLearn

from core.pipeline.config import PipelineConfig

from .trainer import _sanitize_job_name  # shared regex-safe helper

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SageMakerProcessingJob
# ---------------------------------------------------------------------------

class SageMakerProcessingJob:
    """Launch SageMaker Processing Jobs for feature engineering.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration with AWS and data settings.
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
        self._run_stamp = time.strftime("%m%d-%H%M")

    # ------------------------------------------------------------------
    # Phase 0: submit one Processing Job per feature group
    # ------------------------------------------------------------------

    def submit_feature_groups(
        self,
        staging_dir: str,
        input_s3: str,
        output_s3_base: str,
        groups: Iterable[dict[str, Any]] | None = None,
        wait: bool = True,
    ) -> list[dict[str, Any]]:
        """Launch one Processing Job per feature group.

        Contract of ``containers/generators/base/entrypoint.py``
        (unchanged):

        * Input Parquet at ``/opt/ml/processing/input/data`` (single file
          or directory).
        * Output Parquet at ``/opt/ml/processing/output``.
        * Environment variables ``FEATURE_GROUP_NAME``,
          ``FEATURE_GROUP_TYPE``, ``FEATURE_GENERATOR``, and
          ``FEATURE_GENERATOR_PARAMS`` (JSON) control which generator
          runs.

        Parameters
        ----------
        staging_dir : str
            Local path produced by ``scripts.package_source.build_staging``.
            Includes ``core/``, ``configs/``, ``containers/generators/``.
        input_s3 : str
            S3 URI to the ingestion output (hashed + indexed Parquet).
        output_s3_base : str
            S3 prefix under which each group's output is written:
            ``{output_s3_base}/{group_name}/``.
        groups : iterable of dict, optional
            Sequence of feature-group dicts. When ``None``, falls back to
            ``config.feature_groups`` (list form). Each dict must carry
            ``name``, ``group_type``, and optionally ``generator`` and
            ``generator_params``.
        wait : bool
            Block until every submitted job finishes. When False the
            caller is responsible for polling each returned job name.

        Returns
        -------
        list of dict
            One entry per submitted job with
            ``job_name``, ``group``, ``status``, ``output_s3``.
        """
        cfg = self.config
        groups_iter = groups if groups is not None else cfg.feature_groups
        if not groups_iter:
            raise ValueError(
                "No feature groups to submit — config.feature_groups is empty",
            )

        results: list[dict[str, Any]] = []
        for group in groups_iter:
            entry = self._submit_single_feature_group(
                group=group,
                staging_dir=staging_dir,
                input_s3=input_s3,
                output_s3_base=output_s3_base,
                wait=wait,
            )
            results.append(entry)
        return results

    def _submit_single_feature_group(
        self,
        group: dict[str, Any],
        staging_dir: str,
        input_s3: str,
        output_s3_base: str,
        wait: bool,
    ) -> dict[str, Any]:
        cfg = self.config
        aws = cfg.aws
        name = group["name"]
        group_type = group.get("group_type", "generate")
        generator = group.get("generator", "")
        generator_params = group.get("generator_params", {}) or {}

        task_slug = _sanitize_job_name(cfg.task_name, max_len=16)
        group_slug = _sanitize_job_name(name, max_len=20)
        job_name = _sanitize_job_name(
            f"{task_slug}-fg-{group_slug}-{self._run_stamp}",
        )
        output_s3 = f"{output_s3_base.rstrip('/')}/{name}/"

        env = {
            "FEATURE_GROUP_NAME": name,
            "FEATURE_GROUP_TYPE": group_type,
            "FEATURE_GENERATOR": generator,
            "FEATURE_GENERATOR_PARAMS": json.dumps(generator_params),
            "PYTHONPATH": "/opt/ml/code",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
        }
        processor = self._build_framework_processor(
            use_gpu=bool(
                (generator_params.get("prefer_gpu") is True)
                or (generator in {"mamba", "graph"})
            ),
            instance_type=aws.cpu_instance_type,
            environment=env,
        )

        inputs = [
            ProcessingInput(
                source=input_s3,
                destination="/opt/ml/processing/input/data",
                input_name="data",
            ),
        ]
        outputs = [
            ProcessingOutput(
                source="/opt/ml/processing/output",
                destination=output_s3,
                output_name="features",
            ),
        ]

        logger.info("Submitting Phase 0 feature group: %s", name)
        logger.info("  Job name: %s", job_name)
        logger.info("  Generator: %s (params=%s)", generator, generator_params)
        logger.info("  Instance: %s", aws.cpu_instance_type)
        logger.info("  Input:  %s", input_s3)
        logger.info("  Output: %s", output_s3)

        processor.run(
            code="containers/generators/base/entrypoint.py",
            source_dir=staging_dir,
            inputs=inputs,
            outputs=outputs,
            arguments=[],
            job_name=job_name,
            wait=wait,
            logs=wait,
        )

        return {
            "job_name": job_name,
            "group": name,
            "status": "Submitted" if not wait else "Completed",
            "output_s3": output_s3,
        }

    # ------------------------------------------------------------------
    # Legacy one-shot helper (unchanged contract, now safe for source_dir)
    # ------------------------------------------------------------------

    def run(
        self,
        script: str,
        source_dir: str | None = None,
        input_s3: str | None = None,
        output_s3: str | None = None,
        instance_type: str = "ml.m5.2xlarge",
        instance_count: int = 1,
        volume_size_gb: int = 100,
        max_runtime_seconds: int = 7200,
        extra_inputs: dict[str, str] | None = None,
        extra_args: list[str] | None = None,
        environment: dict[str, str] | None = None,
        wait: bool = True,
        use_gpu: bool = False,
        image_uri: str | None = None,
    ) -> dict[str, Any]:
        """Launch a single Processing Job (ad-hoc utility helper)."""
        cfg = self.config
        aws = cfg.aws

        input_s3 = input_s3 or cfg.data.source
        output_s3 = output_s3 or (
            f"s3://{aws.s3_bucket}/{cfg.task_name}/features/"
        )

        task_slug = _sanitize_job_name(cfg.task_name, max_len=20)
        job_name = _sanitize_job_name(
            f"{task_slug}-proc-{self._run_stamp}",
        )

        processor = self._build_framework_processor(
            image_uri=image_uri,
            use_gpu=use_gpu,
            instance_type=instance_type,
            instance_count=instance_count,
            volume_size_gb=volume_size_gb,
            max_runtime_seconds=max_runtime_seconds,
            environment=environment,
        )

        processing_inputs = [
            ProcessingInput(
                source=input_s3,
                destination="/opt/ml/processing/input/data",
                input_name="data",
            ),
        ]
        if extra_inputs:
            for name, s3_uri in extra_inputs.items():
                processing_inputs.append(
                    ProcessingInput(
                        source=s3_uri,
                        destination=f"/opt/ml/processing/input/{name}",
                        input_name=name,
                    )
                )

        processing_outputs = [
            ProcessingOutput(
                source="/opt/ml/processing/output/features",
                destination=output_s3,
                output_name="features",
            ),
            ProcessingOutput(
                source="/opt/ml/processing/output/metadata",
                destination=f"{output_s3.rstrip('/')}_metadata/",
                output_name="metadata",
            ),
        ]

        arguments = list(extra_args or [])

        logger.info("Launching Processing Job: %s", job_name)
        logger.info("  Script: %s", script)
        if source_dir and image_uri is None:
            logger.info("  Source dir: %s", source_dir)
        logger.info(
            "  Instance: %s x %d", instance_type, instance_count,
        )
        logger.info("  GPU mode: %s", use_gpu)
        logger.info("  Input:  %s", input_s3)
        logger.info("  Output: %s", output_s3)

        run_kwargs: dict[str, Any] = dict(
            code=script,
            inputs=processing_inputs,
            outputs=processing_outputs,
            arguments=arguments,
            job_name=job_name,
            wait=wait,
            logs=wait,
        )
        if image_uri is None and source_dir is not None:
            run_kwargs["source_dir"] = source_dir

        processor.run(**run_kwargs)

        result: dict[str, Any] = {
            "job_name": job_name,
            "output_s3": output_s3,
            "instance_type": instance_type,
        }
        if wait:
            desc = self._sm_client.describe_processing_job(
                ProcessingJobName=job_name,
            )
            result["status"] = desc.get("ProcessingJobStatus", "Unknown")
        return result

    def describe_job(self, job_name: str) -> dict:
        """Return the SageMaker DescribeProcessingJob response."""
        return self._sm_client.describe_processing_job(
            ProcessingJobName=job_name,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_framework_processor(
        self,
        *,
        image_uri: str | None = None,
        use_gpu: bool = False,
        instance_type: str = "ml.m5.2xlarge",
        instance_count: int = 1,
        volume_size_gb: int = 100,
        max_runtime_seconds: int = 7200,
        environment: dict[str, str] | None = None,
    ) -> ScriptProcessor | FrameworkProcessor | PyTorchProcessor:
        """Build a Processing instance of the appropriate type.

        * ``image_uri`` provided → :class:`ScriptProcessor` (custom Docker
          image, no ``source_dir`` auto-install).
        * ``use_gpu=True`` → :class:`PyTorchProcessor` (framework 2.1 /
          py310), supports ``source_dir``.
        * otherwise → :class:`FrameworkProcessor` wrapping
          :class:`SKLearn` 1.2-1, supports ``source_dir`` + auto
          ``requirements.txt`` install unlike the legacy
          ``SKLearnProcessor``.
        """
        aws = self.config.aws
        common: dict[str, Any] = dict(
            role=aws.role_arn,
            instance_type=instance_type,
            instance_count=instance_count,
            volume_size_in_gb=volume_size_gb,
            max_runtime_in_seconds=max_runtime_seconds,
            sagemaker_session=self.session,
            tags=[
                {"Key": "Project", "Value": self.config.task_name},
                {"Key": "JobType", "Value": "processing"},
            ],
        )
        if environment:
            common["env"] = dict(environment)

        if image_uri is not None:
            logger.info("Using custom Docker image: %s", image_uri)
            return ScriptProcessor(
                image_uri=image_uri,
                command=["python3"],
                **common,
            )

        if use_gpu:
            logger.info(
                "Using AWS-managed PyTorch 2.1 processor (GPU)",
            )
            return PyTorchProcessor(
                framework_version="2.1",
                py_version="py310",
                **common,
            )

        logger.info(
            "Using AWS-managed SKLearn 1.2-1 framework processor (CPU)",
        )
        return FrameworkProcessor(
            estimator_cls=SKLearn,
            framework_version="1.2-1",
            **common,
        )
