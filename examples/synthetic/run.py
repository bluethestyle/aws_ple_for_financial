"""
Synthetic 데이터로 E2E 파이프라인 시연.
AWS 없이 로컬에서 전체 흐름을 테스트할 수 있습니다.

Usage:
    python examples/synthetic/run.py
    python examples/synthetic/run.py --tasks binary regression --n 50000
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 패키지 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.pipeline.config import load_config, PipelineConfig, TaskSpec, DataSpec, FeatureSpec, ModelSpec, TrainingSpec, AWSSpec
from core.pipeline.runner import PipelineRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def generate_synthetic_data(n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """멀티태스크 학습용 합성 데이터 생성."""
    rng = np.random.default_rng(seed)

    # 피처
    user_age = rng.integers(18, 70, n)
    item_price = rng.exponential(50, n)
    item_popularity = rng.beta(2, 5, n)
    days_since_last_visit = rng.integers(0, 365, n)
    user_segment = rng.choice(["A", "B", "C", "D"], n)
    item_category = rng.choice(["electronics", "fashion", "food", "travel"], n)
    platform = rng.choice(["web", "mobile", "app"], n)

    # 레이블 (실제 데이터처럼 상관관계 있게 생성)
    base_score = (
        0.3 * (user_age / 70)
        + 0.2 * item_popularity
        - 0.1 * (item_price / 200)
        - 0.1 * (days_since_last_visit / 365)
        + rng.normal(0, 0.1, n)
    )
    clicked = (base_score + rng.normal(0, 0.2, n) > 0.3).astype(int)
    converted = ((base_score + rng.normal(0, 0.1, n) > 0.5) & (clicked == 1)).astype(int)

    return pd.DataFrame({
        "user_age": user_age,
        "item_price": item_price,
        "item_popularity": item_popularity,
        "days_since_last_visit": days_since_last_visit,
        "user_segment": user_segment,
        "item_category": item_category,
        "platform": platform,
        "clicked": clicked,
        "converted": converted,
    })


def make_local_config(data_path: str) -> PipelineConfig:
    return PipelineConfig(
        task_name="synthetic_multitask",
        tasks=[
            TaskSpec(name="click",   type="binary", loss="focal", loss_weight=1.0, label_col="clicked"),
            TaskSpec(name="convert", type="binary", loss="focal", loss_weight=1.5, label_col="converted"),
        ],
        data=DataSpec(source=data_path, format="parquet"),
        features=FeatureSpec(
            numeric=["user_age", "item_price", "item_popularity", "days_since_last_visit"],
            categorical=["user_segment", "item_category", "platform"],
        ),
        model=ModelSpec(architecture="lgbm"),
        training=TrainingSpec(epochs=20, seed=42),
        aws=AWSSpec(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10_000, help="샘플 수")
    parser.add_argument("--output", default="outputs/synthetic/", help="결과 저장 경로")
    args = parser.parse_args()

    # 1. 데이터 생성
    logger.info(f"Generating {args.n:,} synthetic samples...")
    df = generate_synthetic_data(n=args.n)

    data_path = "outputs/synthetic/data.parquet"
    Path(data_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(data_path, index=False)
    logger.info(f"Data saved: {data_path} | shape={df.shape}")
    logger.info(f"  click rate   : {df['clicked'].mean():.3f}")
    logger.info(f"  convert rate : {df['converted'].mean():.3f}")

    # 2. 파이프라인 실행
    config = make_local_config(data_path)
    runner = PipelineRunner(config)
    result = runner.run(mode="local", output_dir=args.output)

    logger.info(f"Pipeline result: {result}")


if __name__ == "__main__":
    main()
