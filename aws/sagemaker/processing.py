"""
SageMaker Processing Job wrapper for feature engineering.

Runs a DuckDB-based feature engineering script inside a SageMaker
Processing container.  Input/output channels are mapped to S3 URIs
and the processing script receives them as local paths under
``/opt/ml/processing/``.

Usage::

    config = load_config("configs/my_problem.yaml")
    proc = SageMakerProcessingJob(config)
    result = proc.run(
        script="scripts/feature_engineering.py",
        input_s3="s3://bucket/raw/",
        output_s3="s3://bucket/features/",
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
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)

from ...core.pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


class SageMakerProcessingJob:
    """Launch SageMaker Processing Jobs for feature engineering.

    This wrapper handles:

    * DuckDB-based feature processing scripts
    * S3 input/output channel management
    * Instance type selection (CPU-optimized for feature engineering)
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
        input_s3: str | None = None,
        output_s3: str | None = None,
        instance_type: str = "ml.m5.2xlarge",
        instance_count: int = 1,
        volume_size_gb: int = 100,
        max_runtime_seconds: int = 7200,
        extra_inputs: dict[str, str] | None = None,
        extra_args: list[str] | None = None,
        wait: bool = True,
    ) -> dict[str, Any]:
        """Launch a processing job.

        Parameters
        ----------
        script : str
            Path to the processing script (e.g.
            ``"scripts/feature_engineering.py"``).
        input_s3 : str, optional
            S3 URI for the primary input data.  Defaults to
            ``config.data.source``.
        output_s3 : str, optional
            S3 URI for output data.  Defaults to
            ``s3://<bucket>/<task>/features/``.
        instance_type : str
            Processing instance type.  CPU is preferred for DuckDB.
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
        processor = ScriptProcessor(
            role=aws.role_arn,
            image_uri=self._get_processing_image_uri(),
            command=["python3"],
            instance_type=instance_type,
            instance_count=instance_count,
            volume_size_in_gb=volume_size_gb,
            max_runtime_in_seconds=max_runtime_seconds,
            sagemaker_session=self.session,
            tags=[
                {"Key": "Project", "Value": cfg.task_name},
                {"Key": "JobType", "Value": "processing"},
            ],
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
        logger.info(f"  Instance: {instance_type} x {instance_count}")
        logger.info(f"  Input: {input_s3}")
        logger.info(f"  Output: {output_s3}")

        processor.run(
            code=script,
            inputs=processing_inputs,
            outputs=processing_outputs,
            arguments=arguments,
            job_name=job_name,
            wait=wait,
            logs=wait,
        )

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

        return self.run(
            script="containers/processing/feature_engineering.py",
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

    def _get_processing_image_uri(self) -> str:
        """Resolve the processing container image URI.

        Falls back to the official SageMaker SKLearn processing image
        which includes DuckDB-compatible native libraries.
        """
        region = self.config.aws.region
        account_map = {
            "ap-northeast-2": "366743142698",
            "us-east-1": "683313688378",
            "us-west-2": "246618743249",
            "eu-west-1": "141502667606",
        }
        account = account_map.get(region, "683313688378")
        return (
            f"{account}.dkr.ecr.{region}.amazonaws.com/"
            f"sagemaker-scikit-learn:1.2-1-cpu-py3"
        )
