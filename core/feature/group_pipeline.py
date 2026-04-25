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
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.data.dataframe import df_backend

from .base import AbstractFeatureTransformer
from .generator import AbstractFeatureGenerator, FeatureGeneratorRegistry
from .group import (
    ContainerConfig,
    FeatureGroupConfig,
    FeatureGroupRegistry,
    FeatureInterpretationConfig,
)
from .registry import FeatureRegistry

# Trigger @register decorators for all built-in generators so that the
# FeatureGeneratorRegistry is populated before we try to build generators.
try:
    from . import generators as _generators  # noqa: F401
except Exception:
    pass

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

        # Resolved input columns per generator group (populated during fit)
        self._resolved_inputs: Dict[str, List[str]] = {}

        # Dimension tracking (populated during fit/transform)
        self._group_ranges: Dict[str, Tuple[int, int]] = {}
        self._total_dim: int = 0
        self._output_columns: List[str] = []

        # Instantiate generators and transformer chains eagerly so
        # configuration errors surface immediately.
        self._instantiate_components()

    @classmethod
    def from_yaml(cls, yaml_path: str, name: str = "feature_group_pipeline") -> "FeatureGroupPipeline":
        """Create a pipeline from a ``feature_groups.yaml`` file.

        Parameters
        ----------
        yaml_path : str
            Path to a YAML file with a top-level ``feature_groups`` key.
        name : str
            Pipeline name for logging / MLflow.

        Returns
        -------
        FeatureGroupPipeline
            Ready for ``fit()`` / ``fit_transform()``.
        """
        from .group import load_feature_groups

        groups = load_feature_groups(yaml_path)
        return cls(groups=groups, name=name)

    # -- Input column resolution ----------------------------------------

    def _resolve_input_columns(self, df: Any, group_config: FeatureGroupConfig) -> List[str]:
        """Resolve input columns for a generator from config.

        Priority:
        1. Explicit ``input_columns`` in generator_params.
        2. Filter-based resolution via ``input_filter`` in generator_params.
        3. Default: all numeric columns (excluding id columns).

        Parameters
        ----------
        df : DataFrame
            The input DataFrame (pandas).
        group_config : FeatureGroupConfig
            The group whose input columns to resolve.

        Returns
        -------
        list[str]
            Resolved column names.
        """
        gen_params = group_config.generator_params

        # Explicit columns take priority
        if "input_columns" in gen_params:
            return gen_params["input_columns"]

        # Filter-based resolution
        input_filter = gen_params.get("input_filter", {})
        if input_filter:
            return self._apply_column_filter(df, input_filter)

        # Default: all numeric columns
        return df.select_dtypes(include="number").columns.tolist()

    @staticmethod
    def _apply_column_filter(df: Any, filter_config: Dict[str, Any]) -> List[str]:
        """Apply column filter from config.

        Supports the following filter keys:

        - ``dtype``: ``"continuous"`` | ``"all_numeric"`` | ``"all"``
          (default ``"all_numeric"``).
        - ``exclude_binary``: bool -- drop columns whose unique non-null
          values are a subset of ``{0, 1}`` (default ``False``).
        - ``min_nunique``: int -- minimum number of unique values required
          (default ``0``).
        - ``include_prefix``: list[str] -- only keep columns whose name
          starts with one of these prefixes (default ``[]`` = no filter).
        - ``exclude_prefix``: list[str] -- drop columns whose name starts
          with any of these prefixes (default ``[]``).
        - ``exclude_columns``: list[str] -- explicit column names to
          exclude (default ``[]``).

        Parameters
        ----------
        df : DataFrame
            The input DataFrame.
        filter_config : dict
            Filter specification from ``generator_params.input_filter``.

        Returns
        -------
        list[str]
            Filtered column names.
        """
        exclude_binary = filter_config.get("exclude_binary", False)
        min_nunique = filter_config.get("min_nunique", 0)
        include_prefix = filter_config.get("include_prefix", [])
        exclude_prefix = filter_config.get("exclude_prefix", [])
        exclude_cols = set(filter_config.get("exclude_columns", []))

        # Start with numeric columns
        numeric = df.select_dtypes(include="number")
        cols: List[str] = []

        for col in numeric.columns:
            if col in exclude_cols:
                continue
            if exclude_binary:
                uniq = set(df[col].dropna().unique())
                if uniq.issubset({0, 0.0, 1, 1.0}):
                    continue
            if min_nunique > 0 and df[col].nunique() < min_nunique:
                continue
            if include_prefix and not any(col.startswith(p) for p in include_prefix):
                continue
            if exclude_prefix and any(col.startswith(p) for p in exclude_prefix):
                continue
            cols.append(col)

        return cols

    # -- Component instantiation ---------------------------------------

    def _instantiate_components(self) -> None:
        """Create generator and transformer instances from configs."""
        # Keys consumed by the pipeline, not forwarded to generators
        _pipeline_only_keys = {"input_filter", "input_groups"}

        for group in self._registry.enabled_groups:
            if group.group_type == "generate":
                # Strip pipeline-only keys before forwarding to generator
                gen_params = {
                    k: v for k, v in group.generator_params.items()
                    if k not in _pipeline_only_keys
                }
                gen = FeatureGeneratorRegistry.build(
                    group.generator, **gen_params
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

    def fit(self, df: Any) -> "FeatureGroupPipeline":
        """Fit all feature groups on training data.

        Generators call ``fit()``; transformer chains call ``fit()`` on
        each transformer in sequence (output of transformer N feeds
        into transformer N+1).

        For groups with ``runtime="container"``, fitting is deferred to
        the container execution (the container receives training data
        and returns both the fitted artefact and the generated features).

        Parameters
        ----------
        df : DataFrame
            Training data (pandas, cuDF, or any backend-native type).

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

            if group.runtime == "container":
                logger.info(
                    "  Group '%s' uses container runtime -- "
                    "fit deferred to container execution",
                    group.name,
                )
                # For container groups, we only record output metadata
                # here.  The actual fit happens during transform() via
                # _run_container_job().
                continue

            if group.group_type == "generate":
                gen = self._generators[group.name]
                pdf = df_backend.to_pandas(df)
                # Resolve input columns via input_filter / input_columns
                resolved_cols = self._resolve_input_columns(pdf, group)
                if resolved_cols:
                    self._resolved_inputs[group.name] = resolved_cols
                    # Inject resolved columns into the generator so it
                    # knows which columns to operate on.  Generators use
                    # different attribute names: feature_columns (graph,
                    # gmm, mamba), sequence_columns (hmm), input_columns.
                    for attr in ("feature_columns", "sequence_columns", "input_columns"):
                        if hasattr(gen, attr):
                            current_val = getattr(gen, attr, None)
                            # Only override if the generator does not
                            # already have explicit columns set.
                            if not current_val:
                                setattr(gen, attr, resolved_cols)
                    logger.debug(
                        "  Resolved %d input columns for '%s': %s...",
                        len(resolved_cols), group.name,
                        resolved_cols[:5],
                    )
                # SQL-native generators (CLAUDE.md §3.3) want a DuckDB
                # connection + table name in **context, not pandas. We
                # spin a scratch connection per fit() call (cheap).
                if getattr(gen, "supports_sql_native", False):
                    import duckdb as _duckdb
                    _con = _duckdb.connect()
                    try:
                        _con.register("_grp_fit", pdf)
                        gen.fit(None, duckdb_con=_con, source_table="_grp_fit")
                    finally:
                        try:
                            _con.unregister("_grp_fit")
                        except Exception:
                            pass
                        _con.close()
                else:
                    gen.fit(pdf)
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
                current = df_backend.to_pandas(df)
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

    def transform(self, df: Any) -> Any:
        """Apply all fitted feature groups and concatenate results.

        Parameters
        ----------
        df : DataFrame
            Input data (pandas, cuDF, or any backend-native type).

        Returns
        -------
        DataFrame
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
        group_outputs: List[Any] = []
        pdf = df_backend.to_pandas(df)
        n_input_rows = len(pdf)

        # Lazy DuckDB connection for sql_native generators (CLAUDE.md §3.3).
        # Only created when at least one generator advertises the
        # capability, so the transform path stays free of DuckDB import
        # weight when no SQL-native generator is configured.
        _sql_con = None
        _sql_table = "_grp_input"

        def _ensure_sql_context():
            nonlocal _sql_con
            if _sql_con is None:
                import duckdb as _duckdb
                _sql_con = _duckdb.connect()
                _sql_con.register(_sql_table, pdf)
            return _sql_con, _sql_table

        try:
            for group in self._registry.enabled_groups:
                if group.runtime == "container":
                    container_result = self._run_container_job(group, pdf)
                    if container_result is not None:
                        self._assert_row_invariant(group.name, container_result, n_input_rows)
                        group_outputs.append(container_result)
                    continue

                if group.group_type == "generate":
                    gen = self._generators[group.name]
                    if getattr(gen, "supports_sql_native", False):
                        con, src = _ensure_sql_context()
                        generated = gen.generate(
                            None, duckdb_con=con, source_table=src,
                        )
                        # Convert pyarrow.Table -> pandas for downstream
                        # concat / column slicing. zero-copy where possible.
                        if hasattr(generated, "to_pandas"):
                            generated = generated.to_pandas()
                    else:
                        generated = gen.generate(pdf)
                    self._assert_row_invariant(group.name, generated, n_input_rows)
                    group_outputs.append(generated)

                elif group.group_type == "transform":
                    chain = self._transformer_chains[group.name]
                    current = pdf
                    for t in chain:
                        current = t.transform(current)
                    # Extract only the group's output columns
                    out_cols = [c for c in group.output_columns if c in current.columns]
                    if out_cols:
                        sliced = current[out_cols].copy()
                        self._assert_row_invariant(group.name, sliced, n_input_rows)
                        group_outputs.append(sliced)
                    else:
                        logger.warning(
                            "Transform group '%s' produced no matching output columns",
                            group.name,
                        )
        finally:
            if _sql_con is not None:
                try:
                    _sql_con.unregister(_sql_table)
                except Exception:
                    pass
                _sql_con.close()

        if not group_outputs:
            logger.warning("[%s] No group outputs produced", self.name)
            return df_backend.empty()

        result = df_backend.concat(group_outputs, axis=1)

        # Hard invariant: 1 input row -> 1 output row.  This blocks the
        # multi-row inflation pattern flagged by the on-prem audit (B1/C1)
        # at the integrator boundary so it cannot propagate into training
        # data even if a generator silently produces a duplicated frame.
        if len(result) != n_input_rows:
            raise RuntimeError(
                f"[{self.name}] 1-customer-1-row invariant violated after concat: "
                f"input_rows={n_input_rows}, output_rows={len(result)}. "
                f"Check upstream generators for join inflation."
            )

        logger.debug(
            "[%s] Transform complete in %.2fs -- %d rows x %d cols",
            self.name, time.time() - start, len(result), len(result.columns),
        )
        return result

    @staticmethod
    def _assert_row_invariant(group_name: str, output: Any, expected_rows: int) -> None:
        """Hard assert: per-group output must have exactly ``expected_rows`` rows.

        Generators are contractually obligated to return the same row count
        as their input.  Failing this invariant indicates either a generator
        bug or upstream data inflation; either way training data is no
        longer aligned with the customer index.
        """
        actual = getattr(output, "shape", (None,))[0]
        if actual is None:
            try:
                actual = len(output)
            except TypeError:
                return  # opaque object; downstream concat will surface mismatch
        if actual != expected_rows:
            raise RuntimeError(
                f"Feature group '{group_name}' violated 1-customer-1-row invariant: "
                f"expected {expected_rows} rows, got {actual}. "
                f"Generators must preserve row count (input=output)."
            )

    def fit_transform(self, df: Any) -> Any:
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
    def expert_routing_by_groups(self) -> Dict[str, List[str]]:
        """Expert name -> list of feature group names it should receive.

        This is the group-level counterpart of :attr:`expert_routing`
        (which returns column indices).  Used when constructing
        ``PLEInput.expert_routing`` for the model.

        Returns
        -------
        dict[str, list[str]]
            Maps expert name to a list of feature group names.
        """
        routing: Dict[str, List[str]] = {}

        for group in self._registry.enabled_groups:
            if not group.target_experts:
                continue
            for expert in group.target_experts:
                if expert not in routing:
                    routing[expert] = []
                routing[expert].append(group.name)

        return routing

    def get_ple_input_metadata(self) -> Dict[str, Any]:
        """Return metadata needed for PLEInput construction.

        Provides the feature group ranges and expert routing information
        that the DataLoader / training loop needs when constructing
        ``PLEInput`` instances.  This bridges the feature engineering
        layer to the model layer.

        Returns
        -------
        dict[str, Any]
            Contains:
            - ``feature_group_ranges``: ``{group_name: (start, end)}``
              column index ranges per group in the concatenated feature
              tensor.
            - ``expert_routing``: ``{expert_name: [group_names]}``
              which feature groups route to which expert.
            - ``total_dim``: Total feature dimension.
            - ``output_columns``: Ordered list of all output column names.

        Raises
        ------
        RuntimeError
            If the pipeline has not been fitted yet.
        """
        if not self._fitted:
            raise RuntimeError(
                "Pipeline must be fitted before get_ple_input_metadata(). "
                "Call fit() or fit_transform() first."
            )

        return {
            "feature_group_ranges": dict(self._group_ranges),
            "expert_routing": self.expert_routing_by_groups,
            "expert_routing_indices": self.expert_routing,
            "total_dim": self._total_dim,
            "output_columns": list(self._output_columns),
            "resolved_inputs": dict(self._resolved_inputs),
        }

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

    # -- Container runtime support -------------------------------------

    def _run_container_job(
        self,
        group: FeatureGroupConfig,
        df: Any,
    ) -> Any:
        """Execute a feature group inside a SageMaker Processing Job.

        Workflow:
            1. Write input DataFrame to S3 as Parquet.
            2. Launch a SageMaker Processing Job with the group's
               container image.
            3. Wait for completion.
            4. Download the output Parquet from S3.
            5. Return the output DataFrame.

        Parameters
        ----------
        group : FeatureGroupConfig
            The feature group config with ``runtime="container"``.
        df : DataFrame
            Input data to send to the container.

        Returns
        -------
        DataFrame or None
            The container's output features, or ``None`` on failure.
        """
        import pandas as pd

        container_cfg = group.container
        job_id = f"{group.name}-{uuid.uuid4().hex[:8]}"
        s3_prefix = container_cfg.s3_staging_prefix.rstrip("/")
        s3_input = f"{s3_prefix}/{job_id}/input/data.parquet"
        s3_output = f"{s3_prefix}/{job_id}/output/"

        logger.info(
            "[container] Launching job '%s' for group '%s' "
            "(image=%s, instance=%s)",
            job_id, group.name,
            container_cfg.image, container_cfg.instance_type,
        )

        try:
            import boto3

            # 1. Upload input to S3
            self._upload_df_to_s3(df, s3_input)

            # 2. Create and run SageMaker Processing Job
            sm_client = boto3.client("sagemaker")
            processing_job_name = f"feat-{job_id}"

            env_vars = dict(container_cfg.env)
            env_vars["FEATURE_GROUP_NAME"] = group.name
            env_vars["FEATURE_GROUP_TYPE"] = group.group_type
            if group.generator:
                env_vars["FEATURE_GENERATOR"] = group.generator
                env_vars["FEATURE_GENERATOR_PARAMS"] = json.dumps(
                    group.generator_params
                )

            sm_client.create_processing_job(
                ProcessingJobName=processing_job_name,
                ProcessingResources={
                    "ClusterConfig": {
                        "InstanceCount": container_cfg.instance_count,
                        "InstanceType": container_cfg.instance_type,
                        "VolumeSizeInGB": container_cfg.volume_size_gb,
                    }
                },
                AppSpecification={
                    "ImageUri": container_cfg.image,
                },
                Environment=env_vars,
                ProcessingInputs=[
                    {
                        "InputName": "input-data",
                        "S3Input": {
                            "S3Uri": s3_input.rsplit("/", 1)[0] + "/",
                            "LocalPath": "/opt/ml/processing/input",
                            "S3DataType": "S3Prefix",
                            "S3InputMode": "File",
                        },
                    }
                ],
                ProcessingOutputConfig={
                    "Outputs": [
                        {
                            "OutputName": "output-features",
                            "S3Output": {
                                "S3Uri": s3_output,
                                "LocalPath": "/opt/ml/processing/output",
                                "S3UploadMode": "EndOfJob",
                            },
                        }
                    ]
                },
                StoppingCondition={
                    "MaxRuntimeInSeconds": container_cfg.max_runtime_seconds,
                },
            )

            # 3. Wait for completion
            waiter = sm_client.get_waiter("processing_job_completed_or_stopped")
            waiter.wait(
                ProcessingJobName=processing_job_name,
                WaiterConfig={"Delay": 15, "MaxAttempts": 240},
            )

            # 4. Check job status
            status = sm_client.describe_processing_job(
                ProcessingJobName=processing_job_name
            )
            if status["ProcessingJobStatus"] != "Completed":
                reason = status.get("FailureReason", "unknown")
                logger.error(
                    "[container] Job '%s' failed: %s",
                    processing_job_name, reason,
                )
                return None

            # 5. Download output
            output_path = f"{s3_output}features.parquet"
            result = df_backend.read_parquet(output_path)

            logger.info(
                "[container] Job '%s' completed: %d rows x %d cols",
                processing_job_name, len(result), len(result.columns),
            )

            # Update group metadata from container output
            if group.output_dim == 0:
                group.output_dim = len(result.columns)
            if not group.output_columns:
                group.output_columns = list(result.columns)

            return result

        except ImportError:
            logger.error(
                "[container] boto3 is required for container runtime. "
                "Install with: pip install boto3"
            )
            return None
        except Exception as exc:
            logger.error(
                "[container] Job '%s' failed with error: %s",
                job_id, exc,
            )
            return None

    @staticmethod
    def _upload_df_to_s3(df: Any, s3_uri: str) -> None:
        """Write a DataFrame to S3 as Parquet via boto3.

        Parameters
        ----------
        df : DataFrame
            Data to upload.
        s3_uri : str
            Full ``s3://bucket/key`` destination.
        """
        import io
        import boto3
        import pandas as pd

        # Ensure we have a pandas DataFrame for serialisation
        if hasattr(df, "to_pandas"):
            pdf = df.to_pandas()
        else:
            pdf = df

        buf = io.BytesIO()
        pdf.to_parquet(buf, index=False)
        buf.seek(0)

        # Parse S3 URI
        parts = s3_uri.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else "data.parquet"

        s3 = boto3.client("s3")
        s3.upload_fileobj(buf, bucket, key)
        logger.debug("Uploaded DataFrame (%d bytes) to %s", buf.tell(), s3_uri)

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
        pipeline._resolved_inputs = {}
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

            runtime_tag = (
                f" [CONTAINER:{group.container.image.split('/')[-1]}]"
                if group.runtime == "container"
                else ""
            )

            if group.group_type == "generate":
                gen = self._generators.get(group.name)
                gen_fitted = gen.fitted if gen else False
                lines.append(
                    f"    [{group.name}] GENERATE via '{group.generator}' "
                    f"-> {group.output_dim}D [{dim_range[0]}:{dim_range[1]}] "
                    f"experts=[{expert_str}] interp={interp} "
                    f"fitted={gen_fitted}{runtime_tag}"
                )
            else:
                chain = self._transformer_chains.get(group.name, [])
                lines.append(
                    f"    [{group.name}] TRANSFORM {group.transformers} "
                    f"-> {group.output_dim}D [{dim_range[0]}:{dim_range[1]}] "
                    f"experts=[{expert_str}] interp={interp} "
                    f"chain_len={len(chain)}{runtime_tag}"
                )

        # Expert routing summary
        routing = self.expert_routing
        if routing:
            lines.append("  Expert routing:")
            for expert, indices in routing.items():
                if indices:
                    lines.append(
                        f"    {expert}: {len(indices)} features "
                        f"[{indices[0]}..{indices[-1]}]"
                    )
                else:
                    lines.append(f"    {expert}: 0 features (pending fit)")

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
