"""
SageMaker Training Job 래퍼.

PipelineConfig를 받아 SageMaker Training Job을 생성합니다.
Spot 인스턴스를 기본으로 사용하여 비용을 최소화합니다.
"""

import json
import logging
from datetime import datetime

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput

from ...core.pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


class SageMakerTrainer:
    """
    Example:
        config = load_config("configs/my_problem.yaml")
        trainer = SageMakerTrainer(config)
        result = trainer.launch()
        # → {"job_name": "...", "s3_model_uri": "s3://..."}
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.session = sagemaker.Session()

    def launch(self, wait: bool = True) -> dict:
        cfg = self.config
        aws = cfg.aws
        job_name = f"{cfg.task_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        estimator = PyTorch(
            entry_point="train.py",
            source_dir="containers/training/",
            role=aws.role_arn,
            instance_type=aws.instance_type,
            instance_count=1,
            framework_version="2.1",
            py_version="py310",
            output_path=f"s3://{aws.s3_bucket}/models/{cfg.task_name}/",
            # Spot 인스턴스 설정 — 비용 70% 절감
            use_spot_instances=aws.use_spot,
            max_run=aws.max_run_seconds,
            max_wait=aws.max_run_seconds + 3600 if aws.use_spot else None,
            hyperparameters={
                "config": json.dumps(self._config_to_dict()),
                "task_name": cfg.task_name,
            },
            metric_definitions=[
                {"Name": "val:loss", "Regex": r"val_loss=([0-9\.]+)"},
                {"Name": "val:auc",  "Regex": r"val_auc=([0-9\.]+)"},
            ],
            enable_sagemaker_metrics=True,
        )

        inputs = {"train": TrainingInput(cfg.data.source, content_type="application/x-parquet")}

        logger.info(f"Launching SageMaker Training Job: {job_name}")
        logger.info(f"  Instance: {aws.instance_type} (Spot={aws.use_spot})")
        logger.info(f"  Data: {cfg.data.source}")

        estimator.fit(inputs, job_name=job_name, wait=wait, logs="All" if wait else None)

        return {
            "job_name": job_name,
            "s3_model_uri": estimator.model_data if wait else None,
            "instance_type": aws.instance_type,
            "spot": aws.use_spot,
        }

    def _config_to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self.config)
