import pandas as pd

from .base import AbstractFeatureTransformer, FeatureSchema


class FeaturePipeline:
    """
    여러 FeatureTransformer를 순서대로 적용하는 파이프라인.

    Example:
        pipeline = FeaturePipeline(schema, [
            StandardScaler(cols=schema.numeric),
            CategoricalEncoder(cols=schema.categorical),
        ])
        train_df = pipeline.fit_transform(raw_df)
        test_df  = pipeline.transform(test_df)
    """

    def __init__(self, schema: FeatureSchema, transformers: list[AbstractFeatureTransformer]):
        self.schema = schema
        self.transformers = transformers
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        for t in self.transformers:
            df = t.fit(df).transform(df)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("FeaturePipeline must be fitted before calling transform().")
        for t in self.transformers:
            df = t.transform(df)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def save(self, dir_path: str) -> None:
        import json, pickle
        from pathlib import Path
        p = Path(dir_path)
        p.mkdir(parents=True, exist_ok=True)
        for i, t in enumerate(self.transformers):
            t.save(str(p / f"transformer_{i:02d}_{type(t).__name__}.pkl"))
        with open(p / "schema.json", "w") as f:
            import dataclasses
            json.dump(dataclasses.asdict(self.schema), f)

    @classmethod
    def load(cls, dir_path: str) -> "FeaturePipeline":
        import json, pickle
        from pathlib import Path
        from .base import FeatureSchema
        p = Path(dir_path)
        with open(p / "schema.json") as f:
            schema = FeatureSchema(**json.load(f))
        transformer_files = sorted(p.glob("transformer_*.pkl"))
        transformers = [AbstractFeatureTransformer.load(str(f)) for f in transformer_files]
        pipeline = cls(schema, transformers)
        pipeline._fitted = True
        return pipeline
