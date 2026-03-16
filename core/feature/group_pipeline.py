"""
Feature Group Pipeline -- the main orchestrator for rich feature engineering.

Replaces :class:`FeaturePipeline` as the primary feature engineering
orchestrator.  Instead of a flat list of transformers, the
:class:`FeatureGroupPipeline` processes a list of
:class:`FeatureGroupConfig` objects, where each group either *generates*
new features or *transforms* existing columns.

The pipeline:

1. Reads the :class:`FeatureGroupConfig` list from the
   :class:`FeatureGroupRegistry`.
2. For each enabled group:
   - ``group_type="generate"``: instantiates the generator from
     :class:`FeatureGeneratorRegistry` and runs ``fit_generate`` /
     ``generate``.
   - ``group_type="transform"``: instantiates the transformer chain
     from :class:`FeatureRegistry` and runs ``fit_transform`` /
     ``transform``.
3. Concatenates all group outputs column-wise.
4. Tracks dimension ranges per group.
5. Provides ``expert_routing`` for :class:`PLEModel` consumption.
6. Provides ``interpretation_map`` for
   :class:`RecommendationPipeline` consumption.
7. Supports save / load for production serving.

Usage::

    groups = [
        FeatureGroupConfig(name="demographics", group_type="transform", ...),
        FeatureGroupConfig(name="tda_topology", group_type="generate", ...),
    ]
    pipeline = FeatureGroupPipeline(groups)
    train_df = pipeline.fit_transform(raw_train)
    test_df  = pipeline.transform(raw_test)

    # Wire to PLE model
    ple_config.input_dim = pipeline.total_dim
    # Expert routing: which feature indices go to which expert
    routing = pipeline.expert_routing
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .base import AbstractFeatureTransformer
from .generator import AbstractFeatureGenerator, FeatureGeneratorRegistry
from .group import (
    FeatureGroupConfig,
    FeatureGroupRegistry,
    FeatureInterpretationConfig,
)
from .registry import FeatureRegistry

logger = logging.getLogger(__name__)


class FeatureGroupPipeline:
    """Orchestrator that processes feature groups and produces a unified feature matrix.

    This is the primary entry point for the feature engineering layer.  It
    replaces the simpler :class:`FeaturePipeline` with a richer abstraction
    that supports both transformations and generation, expert routing,
    interpretation metadata, and distillation configuration.

    Parameters
    ----------
    groups : list[FeatureGroupConfig]
        Feature group definitions.  Only enabled groups are processed.
    name : str
        Human-readable pipeline name (for logging / MLflow).
    """

    def __init__(
        self,
        groups: List[FeatureGroupConfig],
        name: str = "feature_group_pipeline",
    ) -> None:
        self.name = name
        self._registry = FeatureGroupRegistry(groups)
        self._fitted = False
        self._fit_metadata: Dict[str, Any] = {}

        # Instantiated components (populated during fit)
        self._generators: Dict[str, AbstractFeatureGenerator] = {}
        self._transformer_chains: Dict[str, List[AbstractFeatureTransformer]] = {}

        # Dimension tracking (populated during fit/transform)
        self._group_ranges: Dict[str, Tuple[int, int]] = {}
        self._total_dim: int = 0
        self._output_columns: List[str] = []

        # Instantiate generators and transformer chains eagerly so
        # configuration errors surface immediately.
        self._instantiate_components()

    # -- Component instantiation ---------------------------------------

    def _instantiate_components(self) -> None:
        """Create generator and transformer instances from configs."""
        for group in self._registry.enabled_groups:
            if group.group_type == "generate":
                gen = FeatureGeneratorRegistry.build(
                    group.generator, **group.generator_params
                )
                self._generators[group.name] = gen
                logger.debug(
                    "Instantiated generator '%s' for group '%s'",
                    group.generator, group.name,
                )
            elif group.group_type == "transform":
                chain: List[AbstractFeatureTransformer] = []
                for t_name in group.transformers:
                    t_params = group.transformer_params.get(t_name, {})
                    # Inject column restriction from the group config
                    if group.columns and "columns" not in t_params:
                        t_params["columns"] = list(group.columns)
                    transformer = FeatureRegistry.build(t_name, **t_params)
                    chain.append(transformer)
                self._transformer_chains[group.name] = chain
                logger.debug(
                    "Instantiated %d transformers for group '%s': %s",
                    len(chain), group.name, group.transformers,
                )

    # -- Core API ------------------------------------------------------

    @property
    def fitted(self) -> bool:
        return self._fitted

    def fit(self, df: pd.DataFrame) -> "FeatureGroupPipeline":
        """Fit all feature groups on training data.

        Generators call ``fit()``; transformer chains call ``fit()`` on
        each transformer in sequence (output of transformer N feeds
        into transformer N+1).

        Parameters
        ----------
        df : pd.DataFrame
            Training data.

        Returns
        -------
        FeatureGroupPipeline
            ``self``, for chaining.
        """
        start = time.time()
        logger.info(
            "[%s] Fitting %d feature groups on %d rows x %d cols",
            self.name,
            len(self._registry.enabled_groups),
            len(df),
            len(df.columns),
        )

        for group in self._registry.enabled_groups:
            g_start = time.time()

            if group.group_type == "generate":
                gen = self._generators[group.name]
                gen.fit(df)
                # Update output_dim and output_columns from generator
                if group.output_dim == 0:
                    group.output_dim = gen.output_dim
                if not group.output_columns:
                    group.output_columns = list(gen.output_columns)
                logger.debug(
                    "  Fitted generator '%s' -> %d features in %.2fs",
                    group.name, gen.output_dim, time.time() - g_start,
                )

            elif group.group_type == "transform":
                chain = self._transformer_chains[group.name]
                current = df
                for t in chain:
                    current = t.fit(current).transform(current)
                # Determine output columns from the group's column list
                if not group.output_columns:
                    group.output_columns = list(group.columns)
                if group.output_dim == 0:
                    group.output_dim = len(group.output_columns)
                logger.debug(
                    "  Fitted transformer chain '%s' (%d steps) -> %d features in %.2fs",
                    group.name, len(chain), group.output_dim, time.time() - g_start,
                )

        # Compute dimension ranges
        self._compute_dim_ranges()

        elapsed = time.time() - start
        self._fitted = True
        self._fit_metadata = {
            "fit_rows": len(df),
            "fit_cols": len(df.columns),
            "fit_time_seconds": round(elapsed, 2),
            "total_dim": self._total_dim,
            "n_groups": len(self._registry.enabled_groups),
        }

        logger.info(
            "[%s] Fit complete in %.2fs -- total_dim=%d across %d groups",
            self.name, elapsed, self._total_dim,
            len(self._registry.enabled_groups),
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all fitted feature groups and concatenate results.

        Parameters
        ----------
        df : pd.DataFrame
            Input data.

        Returns
        -------
        pd.DataFrame
            Concatenated feature matrix with all group outputs.

        Raises
        ------
        RuntimeError
            If the pipeline has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError(
                "FeatureGroupPipeline must be fitted before transform(). "
                "Call fit() or fit_transform() first."
            )

        start = time.time()
        group_outputs: List[pd.DataFrame] = []

        for group in self._registry.enabled_groups:
            if group.group_type == "generate":
                gen = self._generators[group.name]
                generated = gen.generate(df)
                group_outputs.append(generated)

            elif group.group_type == "transform":
                chain = self._transformer_chains[group.name]
                current = df
                for t in chain:
                    current = t.transform(current)
                # Extract only the group's output columns
                out_cols = [c for c in group.output_columns if c in current.columns]
                if out_cols:
                    group_outputs.append(current[out_cols].copy())
                else:
                    logger.warning(
                        "Transform group '%s' produced no matching output columns",
                        group.name,
                    )

        if not group_outputs:
            logger.warning("[%s] No group outputs produced", self.name)
            return pd.DataFrame(index=df.index)

        result = pd.concat(group_outputs, axis=1)

        logger.debug(
            "[%s] Transform complete in %.2fs -- %d rows x %d cols",
            self.name, time.time() - start, len(result), len(result.columns),
        )
        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience: ``fit(df)`` then ``transform(df)``."""
        self.fit(df)
        return self.transform(df)

    # -- Dimension tracking --------------------------------------------

    def _compute_dim_ranges(self) -> None:
        """Compute per-group dimension ranges after fitting."""
        self._group_ranges = {}
        self._output_columns = []
        offset = 0

        for group in self._registry.enabled_groups:
            end = offset + group.output_dim
            self._group_ranges[group.name] = (offset, end)
            self._output_columns.extend(group.output_columns)
            offset = end

        self._total_dim = offset

    @property
    def total_dim(self) -> int:
        """Total feature dimension (sum of all enabled groups)."""
        return self._total_dim

    @property
    def group_ranges(self) -> Dict[str, Tuple[int, int]]:
        """Feature group name -> ``(start_idx, end_idx)`` in concatenated vector.

        Essential for Integrated Gradients attribution and per-group
        feature importance analysis.
        """
        return dict(self._group_ranges)

    # -- Downstream integration points ---------------------------------

    @property
    def expert_routing(self) -> Dict[str, List[int]]:
        """Expert name -> list of feature indices it should receive.

        Used by :class:`PLEModel` to route specific feature slices to
        specific expert networks.  If an expert is not listed in any
        group's ``target_experts``, it receives all features (broadcast).

        Returns
        -------
        dict[str, list[int]]
            Maps expert name to a sorted list of integer feature indices.
        """
        routing: Dict[str, List[int]] = {}

        for group in self._registry.enabled_groups:
            if not group.target_experts:
                continue
            start, end = self._group_ranges.get(group.name, (0, 0))
            indices = list(range(start, end))

            for expert in group.target_experts:
                if expert not in routing:
                    routing[expert] = []
                routing[expert].extend(indices)

        # Sort and deduplicate
        for expert in routing:
            routing[expert] = sorted(set(routing[expert]))

        return routing

    @property
    def interpretation_map(self) -> Dict[str, FeatureInterpretationConfig]:
        """Auto-generated interpretation mapping for reverse_mapper.

        Returns
        -------
        dict[str, FeatureInterpretationConfig]
            Maps feature group name to its interpretation config.
        """
        return self._registry.get_interpretation_mapping()

    @property
    def distillation_config(self) -> List[Dict[str, Any]]:
        """Distillation configuration for each group.

        Returns a list of dicts with group name, dimension range,
        and distillation weight -- ready for the distillation pipeline
        to consume.

        Returns
        -------
        list[dict]
        """
        configs = []
        for group in self._registry.get_distillation_groups():
            dim_range = self._group_ranges.get(group.name)
            configs.append({
                "name": group.name,
                "dim_range": dim_range,
                "weight": group.distill_weight,
                "output_dim": group.output_dim,
            })
        return configs

    @property
    def registry(self) -> FeatureGroupRegistry:
        """Access to the underlying group registry."""
        return self._registry

    # -- Serialisation -------------------------------------------------

    def save(self, dir_path: str) -> None:
        """Persist the pipeline to disk.

        Layout::

            dir_path/
                metadata.json
                groups.json
                generators/
                    {group_name}.pkl
                transformers/
                    {group_name}_{idx}_{class}.pkl

        Parameters
        ----------
        dir_path : str
            Directory to write into (created if needed).
        """
        p = Path(dir_path)
        p.mkdir(parents=True, exist_ok=True)

        # Metadata
        meta = {
            "name": self.name,
            "fitted": self._fitted,
            "total_dim": self._total_dim,
            "group_ranges": {
                k: list(v) for k, v in self._group_ranges.items()
            },
            "output_columns": self._output_columns,
        }
        meta.update(self._fit_metadata)
        with open(p / "metadata.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)

        # Group configs
        with open(p / "groups.json", "w", encoding="utf-8") as fh:
            json.dump(
                self._registry.to_dict_list(),
                fh, indent=2, ensure_ascii=False,
            )

        # Generators
        gen_dir = p / "generators"
        gen_dir.mkdir(exist_ok=True)
        for name, gen in self._generators.items():
            gen.save(str(gen_dir / f"{name}.pkl"))

        # Transformer chains
        t_dir = p / "transformers"
        t_dir.mkdir(exist_ok=True)
        for name, chain in self._transformer_chains.items():
            for i, t in enumerate(chain):
                t.save(str(t_dir / f"{name}_{i:02d}_{type(t).__name__}.pkl"))

        logger.info("[%s] Pipeline saved to %s", self.name, dir_path)

    @classmethod
    def load(cls, dir_path: str) -> "FeatureGroupPipeline":
        """Load a previously saved pipeline.

        Parameters
        ----------
        dir_path : str
            Directory produced by :meth:`save`.

        Returns
        -------
        FeatureGroupPipeline
            A fitted pipeline ready for ``transform()``.
        """
        p = Path(dir_path)

        # Metadata
        with open(p / "metadata.json", "r", encoding="utf-8") as fh:
            meta = json.load(fh)

        # Groups
        with open(p / "groups.json", "r", encoding="utf-8") as fh:
            group_dicts = json.load(fh)
        groups = [FeatureGroupConfig.from_dict(d) for d in group_dicts]

        # Create pipeline (this will try to instantiate components from
        # registries, which is fine for validation but we'll overwrite)
        pipeline = cls.__new__(cls)
        pipeline.name = meta.get("name", "feature_group_pipeline")
        pipeline._registry = FeatureGroupRegistry(groups)
        pipeline._fitted = meta.get("fitted", True)
        pipeline._fit_metadata = {
            k: v for k, v in meta.items()
            if k not in ("name", "fitted", "total_dim", "group_ranges", "output_columns")
        }
        pipeline._total_dim = meta.get("total_dim", 0)
        pipeline._group_ranges = {
            k: tuple(v) for k, v in meta.get("group_ranges", {}).items()
        }
        pipeline._output_columns = meta.get("output_columns", [])

        # Load generators
        pipeline._generators = {}
        gen_dir = p / "generators"
        if gen_dir.exists():
            for group in pipeline._registry.enabled_groups:
                if group.group_type == "generate":
                    pkl_path = gen_dir / f"{group.name}.pkl"
                    if pkl_path.exists():
                        pipeline._generators[group.name] = (
                            AbstractFeatureGenerator.load(str(pkl_path))
                        )

        # Load transformer chains
        pipeline._transformer_chains = {}
        t_dir = p / "transformers"
        if t_dir.exists():
            for group in pipeline._registry.enabled_groups:
                if group.group_type == "transform":
                    # Find all pkl files for this group
                    chain_files = sorted(t_dir.glob(f"{group.name}_*.pkl"))
                    if chain_files:
                        chain = [
                            AbstractFeatureTransformer.load(str(f))
                            for f in chain_files
                        ]
                        pipeline._transformer_chains[group.name] = chain

        logger.info(
            "Loaded FeatureGroupPipeline '%s' with %d groups "
            "(total_dim=%d) from %s",
            pipeline.name,
            len(pipeline._registry.enabled_groups),
            pipeline._total_dim,
            dir_path,
        )
        return pipeline

    # -- Introspection -------------------------------------------------

    def summary(self) -> str:
        """Human-readable summary of the pipeline."""
        lines = [
            f"FeatureGroupPipeline '{self.name}' (fitted={self._fitted})",
            f"  Total dim: {self._total_dim}",
            f"  Groups ({len(self._registry.enabled_groups)}):",
        ]

        for group in self._registry.enabled_groups:
            dim_range = self._group_ranges.get(group.name, (0, 0))
            expert_str = ", ".join(group.target_experts) or "broadcast"
            interp = group.interpretation.category

            if group.group_type == "generate":
                gen = self._generators.get(group.name)
                gen_fitted = gen.fitted if gen else False
                lines.append(
                    f"    [{group.name}] GENERATE via '{group.generator}' "
                    f"-> {group.output_dim}D [{dim_range[0]}:{dim_range[1]}] "
                    f"experts=[{expert_str}] interp={interp} "
                    f"fitted={gen_fitted}"
                )
            else:
                chain = self._transformer_chains.get(group.name, [])
                lines.append(
                    f"    [{group.name}] TRANSFORM {group.transformers} "
                    f"-> {group.output_dim}D [{dim_range[0]}:{dim_range[1]}] "
                    f"experts=[{expert_str}] interp={interp} "
                    f"chain_len={len(chain)}"
                )

        # Expert routing summary
        routing = self.expert_routing
        if routing:
            lines.append("  Expert routing:")
            for expert, indices in routing.items():
                lines.append(
                    f"    {expert}: {len(indices)} features "
                    f"[{indices[0]}..{indices[-1]}]"
                )

        # Distillation summary
        distill_groups = self._registry.get_distillation_groups()
        lines.append(
            f"  Distillation: {len(distill_groups)} groups "
            f"(total weight={sum(g.distill_weight for g in distill_groups):.1f})"
        )

        if self._fit_metadata:
            lines.append(f"  Fit metadata: {self._fit_metadata}")

        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover
        return self.summary()

    def __len__(self) -> int:
        return len(self._registry.enabled_groups)
