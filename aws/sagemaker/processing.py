"""
SageMaker Processing Job wrapper for feature engineering.

Runs feature engineering scripts inside SageMaker Processing containers.
Input/output channels are mapped to S3 URIs and the processing script
receives them as local paths under ``/opt/ml/processing/``.

By default, uses **AWS-managed images** (SKLearn for CPU, PyTorch for GPU)
with a ``requirements.txt`` that is pip-installed automatically when SageMaker
unpacks the ``source_dir``.  A custom Docker image can still be used by
passing ``image_uri`` explicitly (fallback mode).

Usage::

    config = load_config("configs/my_problem.yaml")
    proc = SageMakerProcessingJob(config)

    # CPU job (DuckDB, HMM, GMM, TDA generators)
    result = proc.run(
        script="entrypoint.py",
        source_dir="containers/generators/base/",
        input_s3="s3://bucket/raw/",
        output_s3="s3://bucket/features/",
    )

    # GPU job (Graph, Mamba generators)
    result = proc.run(
        script="entrypoint.py",
        source_dir="containers/generators/base/",
        input_s3="s3://bucket/raw/",
        output_s3="s3://bucket/features/",
        use_gpu=True,
        instance_type="ml.g4dn.xlarge",
    )

    # Custom Docker fallback
    result = proc.run(
        script="entrypoint.py",
        image_uri="123456789.dkr.ecr.us-east-1.amazonaws.com/my-image:latest",
        input_s3="s3://bucket/raw/",
    )
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from typing import Any

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

logger = logging.getLogger(__name__)


class SageMakerProcessingJob:
    """Launch SageMaker Processing Jobs for feature engineering.

    This wrapper handles:

    * AWS-managed images (SKLearn for CPU, PyTorch for GPU) with
      automatic ``requirements.txt`` installation via ``source_dir``
    * Custom Docker image fallback via ``image_uri``
    * S3 input/output channel management
    * Instance type selection
    * Wait/no-wait execution modes
    * Job tagging for cost attribution

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
        wait: bool = True,
        use_gpu: bool = False,
        image_uri: str | None = None,
    ) -> dict[str, Any]:
        """Launch a processing job.

        By default uses AWS-managed images with ``requirements.txt``
        auto-installed from ``source_dir``.  When ``image_uri`` is
        provided, falls back to a custom Docker container via
        :class:`ScriptProcessor`.

        Parameters
        ----------
        script : str
            Processing script filename (relative to *source_dir* when
            using managed images, or an absolute/relative path when
            using a custom ``image_uri``).
        source_dir : str, optional
            Local directory containing the processing script **and**
            ``requirements.txt``.  SageMaker uploads this directory to
            S3 and pip-installs ``requirements.txt`` at job start.
            Required when using managed images (``image_uri`` is None).
            Typically ``"containers/generators/base/"`` for feature
            generator jobs.
        input_s3 : str, optional
            S3 URI for the primary input data.  Defaults to
            ``config.data.source``.
        output_s3 : str, optional
            S3 URI for output data.  Defaults to
            ``s3://<bucket>/<task>/features/``.
        instance_type : str
            Processing instance type.  Defaults to ``ml.m5.2xlarge``
            (CPU).  For GPU jobs use e.g. ``ml.g4dn.xlarge``.
        instance_count : int
            Number of processing instances.
        volume_size_gb : int
            EBS volume size in GB.
        max_runtime_seconds : int
            Maximum job runtime.
        extra_inputs : dict[str, str], optional
            Additional S3 input channels ``{name: s3_uri}``.
        extra_args : list[str], optional
            Extra CLI arguments for the processing script.
        wait : bool
            Block until the job completes.
        use_gpu : bool
            When *True* and ``image_uri`` is *None*, use a PyTorch
            managed image (framework 2.1, py310) suitable for GPU
            workloads (graph, mamba generators).  When *False* (default),
            use an SKLearn managed image for CPU workloads (DuckDB, HMM,
            GMM, TDA).
        image_uri : str, optional
            **Fallback**: explicit Docker image URI.  When provided,
            ``source_dir`` is ignored and a plain :class:`ScriptProcessor`
            is used (legacy custom-container behaviour).

        Returns
        -------
        dict
            ``job_name``, ``status``, ``output_s3``, ``instance_type``.
        """
        cfg = self.config
        aws = cfg.aws

        input_s3 = input_s3 or cfg.data.source
        output_s3 = output_s3 or (
            f"s3://{aws.s3_bucket}/{cfg.task_name}/features/"
        )

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        job_name = f"{cfg.task_name}-processing-{ts}"

        # -- Build processor --
        processor = self._build_processor(
            image_uri=image_uri,
            use_gpu=use_gpu,
            instance_type=instance_type,
            instance_count=instance_count,
            volume_size_gb=volume_size_gb,
            max_runtime_seconds=max_runtime_seconds,
        )

        # -- Input channels --
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

        # -- Output channels --
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

        # -- Script arguments --
        arguments = [
            "--config", json.dumps(asdict(cfg)),
            "--input-dir", "/opt/ml/processing/input/data",
            "--output-dir", "/opt/ml/processing/output/features",
            "--metadata-dir", "/opt/ml/processing/output/metadata",
        ]
        if extra_args:
            arguments.extend(extra_args)

        logger.info(f"Launching Processing Job: {job_name}")
        logger.info(f"  Script: {script}")
        if source_dir and image_uri is None:
            logger.info(f"  Source dir: {source_dir}")
        logger.info(f"  Instance: {instance_type} x {instance_count}")
        logger.info(f"  GPU mode: {use_gpu}")
        logger.info(f"  Image: {image_uri or 'AWS managed'}")
        logger.info(f"  Input: {input_s3}")
        logger.info(f"  Output: {output_s3}")

        # -- Build run kwargs --
        run_kwargs: dict[str, Any] = dict(
            code=script,
            inputs=processing_inputs,
            outputs=processing_outputs,
            arguments=arguments,
            job_name=job_name,
            wait=wait,
            logs=wait,
        )
        # source_dir is supported by managed-image processors
        # (SKLearnProcessor / PyTorchProcessor) but not by the plain
        # ScriptProcessor fallback.
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
            result["status"] = desc["ProcessingJobStatus"]
        return result

    def run_duckdb_feature_engineering(
        self,
        sql_dir_s3: str | None = None,
        wait: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Convenience method for DuckDB-based feature engineering.

        Wraps :meth:`run` with the standard DuckDB entry point script
        and injects SQL template files as an extra input channel.
        Uses CPU managed image (SKLearn) by default since DuckDB is
        CPU-only.

        Parameters
        ----------
        sql_dir_s3 : str, optional
            S3 URI containing ``.sql`` templates for feature generation.
        wait : bool
            Block until the job completes.
        **kwargs
            Forwarded to :meth:`run`.

        Returns
        -------
        dict
            Same as :meth:`run`.
        """
        extra_inputs = kwargs.pop("extra_inputs", {}) or {}
        if sql_dir_s3:
            extra_inputs["sql_templates"] = sql_dir_s3

        extra_args = kwargs.pop("extra_args", []) or []
        extra_args.extend([
            "--engine", "duckdb",
            "--sql-dir", "/opt/ml/processing/input/sql_templates",
        ])

        # Default source_dir for managed-image mode
        kwargs.setdefault("source_dir", "containers/generators/base/")

        return self.run(
            script="entrypoint.py",
            extra_inputs=extra_inputs,
            extra_args=extra_args,
            wait=wait,
            **kwargs,
        )

    def describe_job(self, job_name: str) -> dict:
        """Return the SageMaker DescribeProcessingJob response."""
        return self._sm_client.describe_processing_job(
            ProcessingJobName=job_name,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_processor(
        self,
        *,
        image_uri: str | None,
        use_gpu: bool,
        instance_type: str,
        instance_count: int,
        volume_size_gb: int,
        max_runtime_seconds: int,
    ) -> ScriptProcessor | FrameworkProcessor | PyTorchProcessor:
        """Build the appropriate processor.

        * ``image_uri`` provided  -> :class:`ScriptProcessor` (custom
          Docker fallback).
        * ``use_gpu=True``        -> :class:`PyTorchProcessor` with
          framework 2.1 / py310.
        * ``use_gpu=False``       -> :class:`FrameworkProcessor` wrapping
          the SKLearn 1.2-1 estimator (supports ``source_dir`` +
          requirements.txt unlike the legacy ``SKLearnProcessor``).

        Returns
        -------
        ScriptProcessor | FrameworkProcessor | PyTorchProcessor
        """
        aws = self.config.aws
        common = dict(
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

        if image_uri is not None:
            # Fallback: custom Docker container
            logger.info("Using custom Docker image: %s", image_uri)
            return ScriptProcessor(
                image_uri=image_uri,
                command=["python3"],
                **common,
            )

        if use_gpu:
            # GPU workloads: Graph, Mamba generators
            logger.info(
                "Using AWS-managed PyTorch 2.1 image (GPU processing)"
            )
            return PyTorchProcessor(
                framework_version="2.1",
                py_version="py310",
                **common,
            )

        # CPU workloads: DuckDB, HMM, GMM, TDA generators.
        # FrameworkProcessor(estimator_cls=SKLearn) is used instead of the
        # legacy SKLearnProcessor because only the former supports the
        # ``source_dir`` + auto requirements.txt contract used by the
        # feature-engineering entrypoint.
        logger.info(
            "Using AWS-managed SKLearn 1.2-1 framework processor (CPU)"
        )
        return FrameworkProcessor(
            estimator_cls=SKLearn,
            framework_version="1.2-1",
            **common,
        )
