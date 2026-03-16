from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class FeatureSchema:
    """데이터셋의 컬럼 타입을 선언합니다."""
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    sequence: list[str] = field(default_factory=list)   # 시퀀스 피처 (e.g., 구매 이력)
    timestamp: list[str] = field(default_factory=list)
    label_cols: list[str] = field(default_factory=list)

    @property
    def feature_cols(self) -> list[str]:
        return self.numeric + self.categorical + self.sequence + self.timestamp

    @property
    def input_dim(self) -> int:
        """임베딩 없이 단순 concat 시 차원 수 (numeric + categorical one-hot 제외)."""
        return len(self.numeric)


class AbstractFeatureTransformer(ABC):
    """
    피처 변환 플러그인의 기반 클래스.

    fit() → transform()의 sklearn 패턴을 따릅니다.
    S3 데이터 → 모델 입력 텐서까지의 변환 로직을 담습니다.

    Example:
        class LogTransformer(AbstractFeatureTransformer):
            def fit(self, df):
                self.cols = df.select_dtypes("number").columns.tolist()
                return self

            def transform(self, df):
                df = df.copy()
                df[self.cols] = np.log1p(df[self.cols].clip(lower=0))
                return df
    """

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "AbstractFeatureTransformer":
        """학습 데이터로 통계치(mean, std, vocab 등)를 계산합니다."""
        ...

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """DataFrame → 변환된 DataFrame."""
        ...

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def save(self, path: str) -> None:
        """변환 파라미터를 저장합니다 (pickle 또는 JSON)."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "AbstractFeatureTransformer":
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
