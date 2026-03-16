"""
Feature Pipeline — ordered chain of transformers with serialisation.

Applies a sequence of :class:`AbstractFeatureTransformer` instances
in order.  The pipeline tracks fitted state and can be serialised
to / loaded from disk so that the *exact same* transformation is
applied at training and serving time.

DuckDB integration: the pipeline can optionally materialise its
output as a Parquet file via :class:`~core.data.query_engine.QueryEngine`.

Example::

    pipeline = FeaturePipeline(schema, [
        NullFiller(strategy="median"),
        StandardScaler(columns=schema.numeric),
        LabelEncoder(columns=schema.categorical),
    ])
    train_df = pipeline.fit_transform(raw_train)
    test_df  = pipeline.transform(raw_test)
    pipeline.save("/models/feature_pipeline")
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from .base import AbstractFeatureTransformer, FeatureSchema

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    Ordered chain of feature transformers.

    Parameters
    ----------
    schema : FeatureSchema
        Column layout of the input data.
    transformers : list[AbstractFeatureTransformer]
        Transformers executed in order.
    name : str
        Human-readable pipeline name (for logging / MLflow).
    """

    def __init__(
        self,
        schema: FeatureSchema,
        transformers: Sequence[AbstractFeatureTransformer],
        name: str = "feature_pipeline",
    ) -> None:
        self.schema = schema
        self.transformers: List[AbstractFeatureTransformer] = list(transformers)
        self.name = name
        self._fitted = False
        self._fit_metadata: Dict[str, Any] = {}

    @property
    def fitted(self) -> bool:
        return self._fitted

    # ── Core API ───────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        """Fit every transformer in sequence.

        Each transformer's ``fit()`` is called, then its ``transform()``
        is applied so that subsequent transformers see the transformed
        output.

        Parameters
        ----------
        df : pd.DataFrame
            Training data.

        Returns
        -------
        FeaturePipeline
            ``self``, for chaining.
        """
        start = time.time()
        logger.info(
            "[%s] Fitting %d transformers on %d rows x %d cols",
            self.name, len(self.transformers), len(df), len(df.columns),
        )

        current = df
        for i, t in enumerate(self.transformers):
            t_name = getattr(t, "name", type(t).__name__)
            logger.debug("  [%d/%d] fitting %s", i + 1, len(self.transformers), t_name)
            current = t.fit(current).transform(current)

        elapsed = time.time() - start
        self._fitted = True
        self._fit_metadata = {
            "fit_rows": len(df),
            "fit_cols": len(df.columns),
            "fit_time_seconds": round(elapsed, 2),
            "output_cols": len(current.columns),
            "output_dim": len(current.select_dtypes(include=["number"]).columns),
        }

        logger.info(
            "[%s] Fit complete in %.2fs — output %d cols (%dD numeric)",
            self.name, elapsed,
            self._fit_metadata["output_cols"],
            self._fit_metadata["output_dim"],
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply every fitted transformer in order.

        Parameters
        ----------
        df : pd.DataFrame
            Data to transform (train, valid, or test).

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame.

        Raises
        ------
        RuntimeError
            If the pipeline has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError(
                "FeaturePipeline must be fitted before calling transform(). "
                "Call fit() or fit_transform() first."
            )
        current = df
        for t in self.transformers:
            current = t.transform(current)
        return current

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience: ``fit(df)`` then ``transform(df)``."""
        self.fit(df)
        # Re-transform from scratch so the output is consistent
        return self.transform(df)

    # ── DuckDB materialisation ─────────────────────────────────────────

    def transform_and_save(
        self,
        df: pd.DataFrame,
        output_path: str,
        *,
        compression: str = "SNAPPY",
        row_group_size: int = 500_000,
    ) -> Path:
        """Transform *df* and write the result as Parquet via DuckDB.

        Parameters
        ----------
        df : pd.DataFrame
            Input data.
        output_path : str
            Destination Parquet file path (local or S3).
        compression : str
            Parquet compression codec.
        row_group_size : int
            Parquet row-group size.

        Returns
        -------
        Path
            The written file path.
        """
        try:
            import duckdb
        except ImportError:
            raise ImportError("duckdb is required for transform_and_save()")

        transformed = self.transform(df)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        norm_path = str(out.resolve()).replace("\\", "/")

        conn = duckdb.connect(":memory:")
        try:
            conn.register("_df", transformed)
            conn.execute(
                f"COPY _df TO '{norm_path}' "
                f"(FORMAT PARQUET, COMPRESSION {compression}, "
                f"ROW_GROUP_SIZE {row_group_size})"
            )
        finally:
            conn.close()

        logger.info(
            "[%s] Saved %d rows to %s",
            self.name, len(transformed), output_path,
        )
        return out

    # ── Serialisation ──────────────────────────────────────────────────

    def save(self, dir_path: str) -> None:
        """Persist the pipeline (schema + all transformers) to disk.

        Layout::

            dir_path/
                schema.json
                metadata.json
                transformer_00_StandardScaler.pkl
                transformer_01_LabelEncoder.pkl
                ...

        Parameters
        ----------
        dir_path : str
            Directory to write into (created if needed).
        """
        p = Path(dir_path)
        p.mkdir(parents=True, exist_ok=True)

        # schema
        with open(p / "schema.json", "w", encoding="utf-8") as fh:
            json.dump(self.schema.to_dict(), fh, indent=2, ensure_ascii=False)

        # metadata
        meta = {
            "name": self.name,
            "fitted": self._fitted,
            "n_transformers": len(self.transformers),
            "transformer_names": [
                getattr(t, "name", type(t).__name__)
                for t in self.transformers
            ],
        }
        meta.update(self._fit_metadata)
        with open(p / "metadata.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)

        # transformers
        for i, t in enumerate(self.transformers):
            t_name = type(t).__name__
            t.save(str(p / f"transformer_{i:02d}_{t_name}.pkl"))

        logger.info("[%s] Pipeline saved to %s", self.name, dir_path)

    @classmethod
    def load(cls, dir_path: str) -> "FeaturePipeline":
        """Load a previously saved pipeline.

        Parameters
        ----------
        dir_path : str
            Directory produced by :meth:`save`.

        Returns
        -------
        FeaturePipeline
            A fitted pipeline ready for ``transform()``.
        """
        p = Path(dir_path)

        # schema
        with open(p / "schema.json", "r", encoding="utf-8") as fh:
            schema = FeatureSchema.from_dict(json.load(fh))

        # metadata
        meta: Dict[str, Any] = {}
        meta_path = p / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)

        # transformers
        transformer_files = sorted(p.glob("transformer_*.pkl"))
        transformers = [
            AbstractFeatureTransformer.load(str(f)) for f in transformer_files
        ]

        pipeline = cls(
            schema=schema,
            transformers=transformers,
            name=meta.get("name", "feature_pipeline"),
        )
        pipeline._fitted = meta.get("fitted", True)
        pipeline._fit_metadata = {
            k: v for k, v in meta.items()
            if k not in ("name", "fitted", "n_transformers", "transformer_names")
        }

        logger.info(
            "Loaded pipeline '%s' with %d transformers from %s",
            pipeline.name, len(transformers), dir_path,
        )
        return pipeline

    # ── Introspection ──────────────────────────────────────────────────

    @property
    def input_dim(self) -> int:
        """Dynamic input dimension from the schema."""
        return self.schema.input_dim

    def summary(self) -> str:
        """Human-readable summary of the pipeline."""
        lines = [
            f"FeaturePipeline '{self.name}' (fitted={self._fitted})",
            f"  Schema: {self.schema}",
            f"  Transformers ({len(self.transformers)}):",
        ]
        for i, t in enumerate(self.transformers):
            t_name = getattr(t, "name", type(t).__name__)
            lines.append(f"    [{i}] {t_name} — fitted={t.fitted}")
        if self._fit_metadata:
            lines.append(f"  Fit metadata: {self._fit_metadata}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover
        return self.summary()

    def __len__(self) -> int:
        return len(self.transformers)
