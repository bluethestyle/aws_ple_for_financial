"""
PipelineRunner — config를 받아 로컬 또는 SageMaker에서 파이프라인을 실행합니다.
"""

import logging
from pathlib import Path

from .config import PipelineConfig

logger = logging.getLogger(__name__)


class PipelineRunner:
    """
    실행 환경(로컬 / SageMaker)에 따라 적절한 실행기를 선택합니다.

    Example:
        config = load_config("configs/examples/multitask.yaml")
        runner = PipelineRunner(config)
        runner.run(mode="local")       # 로컬 개발 테스트
        runner.run(mode="sagemaker")   # AWS Spot 인스턴스 학습
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(self, mode: str = "local", output_dir: str = "outputs/") -> dict:
        if mode == "local":
            return self._run_local(output_dir)
        elif mode == "sagemaker":
            return self._run_sagemaker()
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'local' or 'sagemaker'.")

    def _run_local(self, output_dir: str) -> dict:
        from ..model.registry import ModelRegistry
        from ..task.registry import TaskRegistry
        from ..task.base import TaskConfig
        from ..task.types import TaskType, LossType

        logger.info(f"[LOCAL] Starting pipeline: {self.config.task_name}")

        # 1. 데이터 로드
        df = self._load_data_local()

        # 2. 피처 파이프라인
        processed = self._build_features(df)

        # 3. 모델 빌드 & 학습
        results = self._train(processed, output_dir)

        logger.info(f"[LOCAL] Pipeline complete. Results saved to {output_dir}")
        return results

    def _run_sagemaker(self) -> dict:
        from ...aws.sagemaker.trainer import SageMakerTrainer
        trainer = SageMakerTrainer(self.config)
        return trainer.launch()

    def _load_data_local(self):
        import pandas as pd
        source = self.config.data.source
        fmt = self.config.data.format
        if fmt == "parquet":
            return pd.read_parquet(source)
        elif fmt == "csv":
            return pd.read_csv(source)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

    def _build_features(self, df):
        # 기본 구현: 숫자형 컬럼만 선택 후 null 채우기
        num_cols = self.config.features.numeric
        cat_cols = self.config.features.categorical
        label_cols = [t.label_col for t in self.config.tasks]
        keep = num_cols + cat_cols + label_cols
        return df[[c for c in keep if c in df.columns]].fillna(0)

    def _train(self, df, output_dir: str) -> dict:
        import numpy as np
        arch = self.config.model.architecture

        label_cols = [t.label_col for t in self.config.tasks]
        feat_cols = [c for c in df.columns if c not in label_cols]
        X = df[feat_cols].values
        y_dict = {t.name: df[t.label_col].values for t in self.config.tasks}

        if arch == "lgbm":
            from ..model.lgbm import LGBMModel, LGBMConfig
            cfg = LGBMConfig()
            tasks_meta = [{"name": t.name, "type": t.type} for t in self.config.tasks]
            model = LGBMModel(cfg, tasks_meta)
            model.fit(X, y_dict)
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            model.save(output_dir)
            return {"status": "success", "model": "lgbm", "output_dir": output_dir}

        elif arch == "ple":
            # PLE 학습 (간단 버전 — 실제 학습 루프는 examples/ 참고)
            logger.info("PLE training not yet implemented in runner. See examples/.")
            return {"status": "skipped", "model": "ple"}

        else:
            raise ValueError(f"Unknown architecture: {arch}")
