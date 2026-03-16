"""
Feature Pipeline Builder — config-driven pipeline assembly.

Reads a YAML configuration file and automatically constructs a
:class:`FeaturePipeline` with the correct transformer chain.

Example YAML config::

    feature_pipeline:
      name: main_pipeline
      schema:
        id_cols: [user_id]
        label_cols: [target]
        numeric: [amount, balance, tenure]
        categorical: [region, product_type]
      steps:
        - transformer: null_filler
          params:
            strategy: median
        - transformer: log_transformer
          params:
            columns: [amount, balance]
            add_raw_copy: true
        - transformer: standard_scaler
          params:
            columns: [amount, balance, tenure]
        - transformer: label_encoder
          params:
            columns: [region, product_type]
      power_law:
        auto_detect: true
        skewness_threshold: 2.0
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from .base import AbstractFeatureTransformer, FeatureSchema
from .pipeline import FeaturePipeline
from .registry import FeatureRegistry

logger = logging.getLogger(__name__)


class FeaturePipelineBuilder:
    """
    Build a :class:`FeaturePipeline` from a YAML configuration.

    The builder handles:

    * Parsing the YAML into a schema and an ordered list of
      transformer steps.
    * Resolving transformer names via :class:`FeatureRegistry`.
    * Grouping numeric columns by transformer type
      (standard / quantile / log1p).
    * Auto-detecting power-law columns and inserting a
      ``LogTransformer`` with ``add_raw_copy=True`` before the main
      scaler.
    * Including optional plugin features registered in the
      :class:`FeatureRegistry`.

    Parameters
    ----------
    config : dict or str or Path
        Either a parsed dict, a path to a YAML file, or a YAML string.
    """

    def __init__(self, config: Any) -> None:
        if isinstance(config, (str, Path)):
            p = Path(config)
            if p.is_file():
                with open(p, "r", encoding="utf-8") as fh:
                    self._raw = yaml.safe_load(fh)
            else:
                # try parsing as raw YAML string
                self._raw = yaml.safe_load(config)
        elif isinstance(config, dict):
            self._raw = config
        else:
            raise TypeError(
                f"config must be dict, str, or Path — got {type(config).__name__}"
            )

        # Allow the config to be nested under "feature_pipeline" or flat
        if "feature_pipeline" in self._raw:
            self._cfg = self._raw["feature_pipeline"]
        else:
            self._cfg = self._raw

    # ── Build API ──────────────────────────────────────────────────────

    def build(
        self,
        *,
        extra_transformers: Optional[Sequence[AbstractFeatureTransformer]] = None,
    ) -> FeaturePipeline:
        """Construct a :class:`FeaturePipeline` from the configuration.

        Parameters
        ----------
        extra_transformers : sequence, optional
            Additional transformers appended *after* the config-defined
            steps.

        Returns
        -------
        FeaturePipeline
        """
        schema = self._build_schema()
        transformers = self._build_transformers(schema)

        if extra_transformers:
            transformers.extend(extra_transformers)

        name = self._cfg.get("name", "feature_pipeline")

        logger.info(
            "Built pipeline '%s' with %d transformers, schema input_dim=%d",
            name, len(transformers), schema.input_dim,
        )
        return FeaturePipeline(
            schema=schema,
            transformers=transformers,
            name=name,
        )

    def build_and_fit(
        self,
        df: "pandas.DataFrame",  # noqa: F821
        **kwargs: Any,
    ) -> FeaturePipeline:
        """Build the pipeline and immediately fit it on *df*.

        Parameters
        ----------
        df : pd.DataFrame
            Training data.

        Returns
        -------
        FeaturePipeline
            A fitted pipeline.
        """
        pipeline = self.build(**kwargs)
        pipeline.fit(df)
        return pipeline

    # ── Schema construction ────────────────────────────────────────────

    def _build_schema(self) -> FeatureSchema:
        """Parse the ``schema`` section of the config."""
        schema_cfg = self._cfg.get("schema", {})
        return FeatureSchema(
            numeric=schema_cfg.get("numeric", []),
            categorical=schema_cfg.get("categorical", []),
            sequence=schema_cfg.get("sequence", []),
            timestamp=schema_cfg.get("timestamp", []),
            label_cols=schema_cfg.get("label_cols", []),
            id_cols=schema_cfg.get("id_cols", []),
            power_law_skewness_threshold=schema_cfg.get(
                "power_law_skewness_threshold",
                self._cfg.get("power_law", {}).get("skewness_threshold", 2.0),
            ),
        )

    # ── Transformer chain ──────────────────────────────────────────────

    def _build_transformers(
        self, schema: FeatureSchema
    ) -> List[AbstractFeatureTransformer]:
        """Resolve and instantiate transformers from the ``steps`` list."""
        steps_cfg: List[Dict[str, Any]] = self._cfg.get("steps", [])
        transformers: List[AbstractFeatureTransformer] = []

        for step in steps_cfg:
            t_name = step.get("transformer") or step.get("name")
            if t_name is None:
                raise ValueError(
                    f"Pipeline step missing 'transformer' key: {step}"
                )

            params: Dict[str, Any] = dict(step.get("params", {}))

            # resolve column references like "$numeric" or "$categorical"
            params = self._resolve_column_refs(params, schema)

            # handle "enabled" flag
            if not step.get("enabled", True):
                logger.debug("Skipping disabled step: %s", t_name)
                continue

            try:
                transformer = FeatureRegistry.build(t_name, **params)
                transformers.append(transformer)
                logger.debug("Added transformer: %s(%s)", t_name, params)
            except KeyError:
                logger.warning(
                    "Transformer '%s' not found in registry — skipping. "
                    "Available: %s",
                    t_name, FeatureRegistry.list_registered(),
                )

        # ── Power-law auto-detection ──────────────────────────────────
        power_law_cfg = self._cfg.get("power_law", {})
        if power_law_cfg.get("auto_detect", False):
            # Insert a LogTransformer placeholder — actual columns will
            # be determined at fit-time via schema.detect_power_law_columns
            logger.info(
                "Power-law auto-detection enabled "
                "(skewness_threshold=%.1f). "
                "LogTransformer will be added during fit.",
                schema.power_law_skewness_threshold,
            )
            # We add a special marker transformer that detects columns at fit time
            transformers.insert(
                0,
                _PowerLawAutoDetector(
                    skewness_threshold=schema.power_law_skewness_threshold,
                    add_raw_copy=power_law_cfg.get("add_raw_copy", True),
                ),
            )

        # ── Plugin features (optional extras from registry) ───────────
        plugins_cfg: List[str] = self._cfg.get("plugins", [])
        for plugin_name in plugins_cfg:
            try:
                transformer = FeatureRegistry.build(plugin_name)
                transformers.append(transformer)
                logger.info("Added plugin transformer: %s", plugin_name)
            except KeyError:
                logger.warning(
                    "Plugin '%s' not found in registry — skipping.", plugin_name
                )

        return transformers

    @staticmethod
    def _resolve_column_refs(
        params: Dict[str, Any], schema: FeatureSchema
    ) -> Dict[str, Any]:
        """Replace ``$numeric``, ``$categorical``, etc. with actual column lists."""
        resolved = dict(params)
        _REFS = {
            "$numeric": schema.numeric,
            "$categorical": schema.categorical,
            "$sequence": schema.sequence,
            "$timestamp": schema.timestamp,
            "$feature_cols": schema.feature_cols,
            "$label_cols": schema.label_cols,
            "$id_cols": schema.id_cols,
        }
        for key, value in resolved.items():
            if isinstance(value, str) and value in _REFS:
                resolved[key] = list(_REFS[value])
            elif isinstance(value, list):
                expanded: List[str] = []
                for item in value:
                    if isinstance(item, str) and item in _REFS:
                        expanded.extend(_REFS[item])
                    else:
                        expanded.append(item)
                resolved[key] = expanded
        return resolved


# ──────────────────────────────────────────────────────────────────────
# Internal: power-law auto-detection transformer
# ──────────────────────────────────────────────────────────────────────


class _PowerLawAutoDetector(AbstractFeatureTransformer):
    """Internal transformer that detects power-law columns at fit-time.

    At ``fit()`` time it scans all numeric columns for high skewness and
    applies ``log1p(max(x, 0))`` with an optional raw copy.  This is
    inserted at the *beginning* of the pipeline when
    ``power_law.auto_detect: true`` is set in the config.
    """

    name = "_power_law_auto_detector"

    def __init__(
        self,
        skewness_threshold: float = 2.0,
        add_raw_copy: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.skewness_threshold = skewness_threshold
        self.add_raw_copy = add_raw_copy
        self._power_law_cols: List[str] = []

    def fit(self, df: "pandas.DataFrame") -> "_PowerLawAutoDetector":
        import numpy as np

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        self._power_law_cols = []
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 10:
                continue
            skew = float(series.skew())
            if abs(skew) > self.skewness_threshold:
                self._power_law_cols.append(col)

        self._fitted = True
        if self._power_law_cols:
            logger.info(
                "Power-law auto-detected %d columns (|skew| > %.1f): %s",
                len(self._power_law_cols),
                self.skewness_threshold,
                self._power_law_cols[:5],
            )
        return self

    def transform(self, df: "pandas.DataFrame") -> "pandas.DataFrame":
        import numpy as np

        if not self._fitted:
            raise RuntimeError("_PowerLawAutoDetector must be fitted first.")
        df = df.copy()
        for col in self._power_law_cols:
            if col not in df.columns:
                continue
            if self.add_raw_copy:
                df[f"{col}_raw"] = df[col]
            df[col] = np.log1p(np.maximum(df[col].values.astype(np.float64), 0))
        return df
