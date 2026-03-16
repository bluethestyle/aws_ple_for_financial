"""
SageMaker 서빙 — 실시간 엔드포인트 또는 배치 추론.
실시간 엔드포인트는 비용이 발생하므로, 기본은 배치 추론(Batch Transform)입니다.
"""

import logging

import sagemaker
from sagemaker.pytorch import PyTorchModel

logger = logging.getLogger(__name__)


class SageMakerServing:
    """
    Example:
        serving = SageMakerServing(model_uri="s3://...", role_arn="arn:aws:iam::...")
        # 배치 추론 (비용 효율적)
        serving.batch_transform(
            input_s3="s3://my-bucket/inference/input/",
            output_s3="s3://my-bucket/inference/output/",
        )
        # 실시간 엔드포인트 (테스트용 — 사용 후 반드시 삭제)
        predictor = serving.deploy_endpoint(instance_type="ml.g4dn.xlarge")
        serving.delete_endpoint(predictor)
    """

    def __init__(self, model_uri: str, role_arn: str, region: str = "ap-northeast-2"):
        self.model_uri = model_uri
        self.role_arn = role_arn
        self.session = sagemaker.Session()
        self.region = region

    def batch_transform(
        self,
        input_s3: str,
        output_s3: str,
        instance_type: str = "ml.g4dn.xlarge",
        instance_count: int = 1,
    ) -> str:
        model = PyTorchModel(
            model_data=self.model_uri,
            role=self.role_arn,
            entry_point="inference.py",
            source_dir="containers/inference/",
            framework_version="2.1",
            py_version="py310",
        )
        transformer = model.transformer(
            instance_count=instance_count,
            instance_type=instance_type,
            output_path=output_s3,
            assemble_with="Line",
            accept="application/json",
        )
        transformer.transform(input_s3, content_type="application/json", wait=True)
        logger.info(f"Batch transform complete. Output: {output_s3}")
        return output_s3

    def deploy_endpoint(
        self,
        instance_type: str = "ml.g4dn.xlarge",
        endpoint_name: str | None = None,
    ):
        """실시간 엔드포인트 배포. 테스트 후 delete_endpoint()로 반드시 삭제하세요."""
        from datetime import datetime
        name = endpoint_name or f"ple-endpoint-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        model = PyTorchModel(
            model_data=self.model_uri,
            role=self.role_arn,
            entry_point="inference.py",
            source_dir="containers/inference/",
            framework_version="2.1",
            py_version="py310",
        )
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=name,
        )
        logger.info(f"Endpoint deployed: {name}")
        logger.warning("Don't forget to delete the endpoint after testing!")
        return predictor

    def delete_endpoint(self, predictor) -> None:
        predictor.delete_endpoint()
        logger.info("Endpoint deleted.")
