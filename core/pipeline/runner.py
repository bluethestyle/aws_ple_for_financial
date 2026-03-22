"""
PipelineRunner -- 9-stage universal pipeline orchestrator.

Stages:
  1. Load raw data via DataAdapter
  2. Schema classification
  3. PII encryption (optional)
  4. Feature engineering + normalization via FeatureGroupPipeline
  5. Label derivation
  6. Sequence building (flat -> 3D tensors)
  7. Build DataLoaders (PLEDataset)
  8. Train teacher model (PLE)
  9. Knowledge distillation (teacher -> LGBM students)

Each stage is independently testable and logs entry/exit.

Usage::

    config = load_config("configs/examples/multitask.yaml")
    runner = PipelineRunner(config)
    results = runner.run(output_dir="outputs/")
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

from .adapter import DataAdapter
from .config import PipelineConfig

logger = logging.getLogger(__name__)


# ======================================================================
# Pipeline state tracker for stage-level checkpointing / resume
# ======================================================================

class _PipelineState:
    """Tracks completed stages for pipeline resume capability."""

    def __init__(self, output_dir: str) -> None:
        self.path = Path(output_dir) / "pipeline_state.json"
        self.state = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            with open(self.path) as f:
                return json.load(f)
        return {"completed_stages": [], "artifacts": {}, "start_time": None}

    def mark_complete(self, stage: str, artifacts: dict = None) -> None:
        if stage not in self.state["completed_stages"]:
            self.state["completed_stages"].append(stage)
        if artifacts:
            self.state["artifacts"][stage] = artifacts
        self._save()

    def mark_failed(self, stage: str, error: str) -> None:
        self.state["failed_stage"] = stage
        self.state["error"] = error
        self._save()

    def is_complete(self, stage: str) -> bool:
        return stage in self.state["completed_stages"]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.state, f, indent=2, default=str)


class PipelineRunner:
    """Execute a 9-stage training pipeline driven by :class:`PipelineConfig`.

    The runner wires together data loading (adapter), schema classification,
    encryption, feature engineering, label derivation, sequence building,
    DataLoader construction, and model training.

    Args:
        config: A fully populated :class:`PipelineConfig`.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, output_dir: str = "outputs/") -> dict:
        """Run the full 9-stage pipeline with checkpointing and resume.

        Each stage is wrapped with :class:`_PipelineState` tracking so that
        a failed run can be resumed from the last successful stage.
        Intermediate artifacts (parquet, npy) are saved under
        ``<output_dir>/checkpoints/`` for stages 4-6.

        Args:
            output_dir: Directory for model artifacts.

        Returns:
            A result dict with metadata from each stage.
        """
        import numpy as np

        results: Dict[str, Any] = {}
        pipeline_start = time.time()

        # Store output_dir for cross-stage artifact saving
        out = Path(output_dir)
        self._output_dir = out
        out.mkdir(parents=True, exist_ok=True)
        (out / "audit").mkdir(exist_ok=True)
        checkpoint_dir = out / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        state = _PipelineState(output_dir)
        if state.state["start_time"] is None:
            state.state["start_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            state._save()

        logger.info("=" * 60)
        logger.info("[PIPELINE] Starting 9-stage pipeline: %s", self.config.task_name)
        logger.info("=" * 60)

        # ----------------------------------------------------------
        # Stage 1: Load raw data
        # ----------------------------------------------------------
        if not state.is_complete("stage1"):
            try:
                adapter = self._build_adapter()
                raw_data = adapter.load_raw()
                results["stage1_metadata"] = adapter.metadata
                state.mark_complete("stage1", {"num_frames": len(raw_data)})
                logger.info(
                    "[PIPELINE] Stage 1 complete: loaded %d DataFrames",
                    len(raw_data),
                )
            except Exception as e:
                logger.error("[PIPELINE] Stage 1 failed: %s", e)
                state.mark_failed("stage1", str(e))
                raise
        else:
            logger.info("[PIPELINE] Stage 1 already complete, re-loading...")
            adapter = self._build_adapter()
            raw_data = adapter.load_raw()
            results["stage1_metadata"] = adapter.metadata

        # ----------------------------------------------------------
        # Stage 1.5: Temporal preparation
        # ----------------------------------------------------------
        if not state.is_complete("stage1_5"):
            try:
                raw_data["main"] = self._prepare_temporal(raw_data["main"])
                results["stage1_5_temporal_prep"] = {"applied": True}
                state.mark_complete("stage1_5")
                logger.info("[PIPELINE] Stage 1.5 complete: temporal preparation")
            except Exception as e:
                logger.error("[PIPELINE] Stage 1.5 failed: %s", e)
                state.mark_failed("stage1_5", str(e))
                raise
        else:
            logger.info("[PIPELINE] Stage 1.5 already complete, re-applying...")
            raw_data["main"] = self._prepare_temporal(raw_data["main"])
            results["stage1_5_temporal_prep"] = {"applied": True}

        # ----------------------------------------------------------
        # Stage 2: Schema classification
        # ----------------------------------------------------------
        if not state.is_complete("stage2"):
            try:
                schema = self._classify_schema(raw_data["main"])
                results["stage2_schema"] = schema
                state.mark_complete("stage2", {
                    k: len(v) for k, v in schema.items()
                })
                logger.info(
                    "[PIPELINE] Stage 2 complete: %d numeric, %d categorical, "
                    "%d sequence cols",
                    len(schema.get("numeric", [])),
                    len(schema.get("categorical", [])),
                    len(schema.get("sequence", [])),
                )
            except Exception as e:
                logger.error("[PIPELINE] Stage 2 failed: %s", e)
                state.mark_failed("stage2", str(e))
                raise
        else:
            logger.info("[PIPELINE] Stage 2 already complete, re-classifying...")
            schema = self._classify_schema(raw_data["main"])
            results["stage2_schema"] = schema

        # Save schema classification audit
        schema_path = out / "audit" / "schema_classification.json"
        with open(schema_path, "w") as f:
            json.dump(schema, f, indent=2, default=str)
        logger.info("[PIPELINE] Schema classification saved to %s", schema_path)

        # ----------------------------------------------------------
        # Stage 3: Encryption
        # ----------------------------------------------------------
        if not state.is_complete("stage3"):
            try:
                df_main = self._encrypt(raw_data["main"])
                results["stage3_encryption"] = {
                    "applied": df_main is not raw_data["main"],
                    "rows": len(df_main),
                    "cols": len(df_main.columns),
                }
                state.mark_complete("stage3", {
                    "rows": len(df_main), "cols": len(df_main.columns),
                })
                logger.info(
                    "[PIPELINE] Stage 3 complete: %d rows x %d cols after encryption",
                    len(df_main), len(df_main.columns),
                )
            except Exception as e:
                logger.error("[PIPELINE] Stage 3 failed: %s", e)
                state.mark_failed("stage3", str(e))
                raise
        else:
            logger.info("[PIPELINE] Stage 3 already complete, re-encrypting...")
            df_main = self._encrypt(raw_data["main"])
            results["stage3_encryption"] = {
                "applied": df_main is not raw_data["main"],
                "rows": len(df_main),
                "cols": len(df_main.columns),
            }

        # ----------------------------------------------------------
        # Stage 4: Feature engineering + normalization
        # ----------------------------------------------------------
        features_ckpt = checkpoint_dir / "features.parquet"
        if not state.is_complete("stage4"):
            try:
                feature_pipeline, df_features = self._engineer_features(
                    df_main, raw_data,
                )
                results["feature_metadata"] = (
                    feature_pipeline.get_ple_input_metadata()
                )

                # Save intermediate artifacts
                df_features.to_parquet(features_ckpt)
                feature_pipeline.save(str(checkpoint_dir / "feature_pipeline"))

                state.mark_complete("stage4", {
                    "shape": list(df_features.shape),
                })
                logger.info(
                    "[PIPELINE] Stage 4 complete: %d features across %d groups",
                    feature_pipeline.total_dim, len(feature_pipeline),
                )
            except Exception as e:
                logger.error("[PIPELINE] Stage 4 failed: %s", e)
                state.mark_failed("stage4", str(e))
                raise
        else:
            import pandas as pd
            logger.info(
                "[PIPELINE] Stage 4 already complete, loading checkpoint..."
            )
            df_features = pd.read_parquet(features_ckpt)
            feature_pipeline, _ = self._engineer_features(df_main, raw_data)
            results["feature_metadata"] = (
                feature_pipeline.get_ple_input_metadata()
            )

        # Save scaler/transformer parameters
        scaler_params: Dict[str, Any] = {}
        for group_name, chain in feature_pipeline._transformer_chains.items():
            scaler_params[group_name] = [t.get_params() for t in chain]
        scaler_path = out / "audit" / "scaler_params.json"
        with open(scaler_path, "w") as f:
            json.dump(scaler_params, f, indent=2, default=str)
        logger.info("[PIPELINE] Scaler parameters saved to %s", scaler_path)

        # ----------------------------------------------------------
        # Stage 5: Label derivation
        # ----------------------------------------------------------
        labels_ckpt = checkpoint_dir / "labels.parquet"
        if not state.is_complete("stage5"):
            try:
                df_labels = self._derive_labels(raw_data["main"])
                results["stage5_labels"] = {
                    "label_columns": list(df_labels.columns),
                    "rows": len(df_labels),
                }

                # Save intermediate artifact
                df_labels.to_parquet(labels_ckpt)

                state.mark_complete("stage5", {
                    "label_columns": list(df_labels.columns),
                    "rows": len(df_labels),
                })
                logger.info(
                    "[PIPELINE] Stage 5 complete: %d label columns",
                    len(df_labels.columns),
                )
            except Exception as e:
                logger.error("[PIPELINE] Stage 5 failed: %s", e)
                state.mark_failed("stage5", str(e))
                raise
        else:
            import pandas as pd
            logger.info(
                "[PIPELINE] Stage 5 already complete, loading checkpoint..."
            )
            df_labels = pd.read_parquet(labels_ckpt)
            results["stage5_labels"] = {
                "label_columns": list(df_labels.columns),
                "rows": len(df_labels),
            }

        # Save label distributions
        label_stats: Dict[str, Any] = {}
        for col in df_labels.columns:
            if df_labels[col].dtype in ("int64", "int32"):
                label_stats[col] = df_labels[col].value_counts().to_dict()
            else:
                label_stats[col] = {
                    "mean": float(df_labels[col].mean()),
                    "std": float(df_labels[col].std()),
                    "min": float(df_labels[col].min()),
                    "max": float(df_labels[col].max()),
                    "count": int(df_labels[col].count()),
                }
        stats_path = out / "audit" / "label_distributions.json"
        with open(stats_path, "w") as f:
            json.dump(label_stats, f, indent=2, default=str)
        logger.info("[PIPELINE] Label distributions saved to %s", stats_path)

        # ----------------------------------------------------------
        # Stage 5.5: Leakage validation
        # ----------------------------------------------------------
        if not state.is_complete("stage5_5"):
            try:
                leakage_result = self._validate_leakage(
                    df_features, df_labels, raw_data.get("main"),
                )
                results["stage5_5_leakage"] = {
                    "passed": leakage_result.passed,
                    "warnings": leakage_result.warnings,
                }
                state.mark_complete("stage5_5", {
                    "passed": leakage_result.passed,
                })
                logger.info(
                    "[PIPELINE] Stage 5.5 complete: leakage validation %s "
                    "(%d warnings)",
                    "PASSED" if leakage_result.passed else "FAILED",
                    len(leakage_result.warnings),
                )
            except Exception as e:
                logger.error("[PIPELINE] Stage 5.5 failed: %s", e)
                state.mark_failed("stage5_5", str(e))
                raise
        else:
            logger.info("[PIPELINE] Stage 5.5 already complete, skipping...")
            results["stage5_5_leakage"] = state.state["artifacts"].get(
                "stage5_5", {"passed": True},
            )
            # Still run validation for the audit report
            leakage_result = self._validate_leakage(
                df_features, df_labels, raw_data.get("main"),
            )

        # Save leakage report
        leakage_path = out / "audit" / "leakage_report.json"
        with open(leakage_path, "w") as f:
            json.dump(
                {
                    "passed": leakage_result.passed,
                    "warnings": leakage_result.warnings,
                    "details": leakage_result.details,
                },
                f, indent=2, default=str,
            )
        logger.info("[PIPELINE] Leakage report saved to %s", leakage_path)

        # ----------------------------------------------------------
        # Stage 6: Sequence building
        # ----------------------------------------------------------
        if not state.is_complete("stage6"):
            try:
                sequences = self._build_sequences(raw_data)
                results["stage6_sequences"] = {
                    "has_sequences": sequences is not None,
                }
                if sequences is not None:
                    results["stage6_sequences"]["keys"] = list(sequences.keys())
                    # Save intermediate artifacts
                    for name, arr in sequences.items():
                        np.save(checkpoint_dir / f"{name}.npy", arr)

                state.mark_complete("stage6", {
                    "has_sequences": sequences is not None,
                    "keys": list(sequences.keys()) if sequences else [],
                })
                logger.info(
                    "[PIPELINE] Stage 6 complete: sequences=%s",
                    "present" if sequences else "none",
                )
            except Exception as e:
                logger.error("[PIPELINE] Stage 6 failed: %s", e)
                state.mark_failed("stage6", str(e))
                raise
        else:
            logger.info(
                "[PIPELINE] Stage 6 already complete, loading checkpoint..."
            )
            saved_keys = state.state["artifacts"].get(
                "stage6", {},
            ).get("keys", [])
            if saved_keys:
                sequences = {}
                for name in saved_keys:
                    npy_path = checkpoint_dir / f"{name}.npy"
                    if npy_path.exists():
                        sequences[name] = np.load(
                            str(npy_path), allow_pickle=False,
                        )
            else:
                sequences = None
            results["stage6_sequences"] = {
                "has_sequences": sequences is not None,
            }
            if sequences is not None:
                results["stage6_sequences"]["keys"] = list(sequences.keys())

        # ----------------------------------------------------------
        # Stage 7: Build DataLoaders
        # ----------------------------------------------------------
        if not state.is_complete("stage7"):
            try:
                train_loader, val_loader = self._build_dataloaders(
                    df_features, df_labels, sequences, feature_pipeline,
                )
                results["stage7_dataloaders"] = {
                    "train_batches": len(train_loader),
                    "val_batches": len(val_loader),
                }
                state.mark_complete("stage7", {
                    "train_batches": len(train_loader),
                    "val_batches": len(val_loader),
                })
                logger.info(
                    "[PIPELINE] Stage 7 complete: train=%d batches, "
                    "val=%d batches",
                    len(train_loader), len(val_loader),
                )
            except Exception as e:
                logger.error("[PIPELINE] Stage 7 failed: %s", e)
                state.mark_failed("stage7", str(e))
                raise
        else:
            logger.info(
                "[PIPELINE] Stage 7 already complete, rebuilding loaders..."
            )
            train_loader, val_loader = self._build_dataloaders(
                df_features, df_labels, sequences, feature_pipeline,
            )
            results["stage7_dataloaders"] = {
                "train_batches": len(train_loader),
                "val_batches": len(val_loader),
            }

        # ----------------------------------------------------------
        # Stage 8: Train teacher model
        # ----------------------------------------------------------
        if not state.is_complete("stage8"):
            try:
                results["training"] = self._train(
                    train_loader, val_loader, output_dir,
                )
                feature_pipeline.save(str(out / "feature_pipeline"))
                state.mark_complete("stage8", {
                    "status": results["training"].get("status", "unknown"),
                })
            except Exception as e:
                logger.error("[PIPELINE] Stage 8 failed: %s", e)
                state.mark_failed("stage8", str(e))
                raise
        else:
            logger.info(
                "[PIPELINE] Stage 8 already complete, skipping training..."
            )
            results["training"] = state.state["artifacts"].get(
                "stage8", {"status": "resumed"},
            )
            feature_pipeline.save(str(out / "feature_pipeline"))

        # ----------------------------------------------------------
        # Stage 9: Knowledge distillation
        # ----------------------------------------------------------
        if not state.is_complete("stage9"):
            try:
                distill_cfg = self._config_to_dict().get("distillation", {})
                if distill_cfg.get("enabled", True):
                    results["distillation"] = self._distill(
                        teacher_checkpoint=str(out / "model.pth"),
                        feature_df=df_features,
                        label_df=df_labels,
                        output_dir=output_dir,
                    )
                else:
                    logger.info(
                        "[Stage 9] Distillation disabled in config, skipping."
                    )
                    results["distillation"] = {"status": "disabled"}
                state.mark_complete("stage9", {
                    "status": results.get("distillation", {}).get(
                        "status", "unknown",
                    ),
                })
            except Exception as e:
                logger.error("[PIPELINE] Stage 9 failed: %s", e)
                state.mark_failed("stage9", str(e))
                raise
        else:
            logger.info(
                "[PIPELINE] Stage 9 already complete, skipping distillation..."
            )
            results["distillation"] = state.state["artifacts"].get(
                "stage9", {"status": "resumed"},
            )

        # ----------------------------------------------------------
        # Finalize
        # ----------------------------------------------------------
        elapsed = time.time() - pipeline_start
        results["total_time_seconds"] = round(elapsed, 2)

        # Save pipeline manifest
        manifest_path = out / "pipeline_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("[PIPELINE] Manifest saved to %s", manifest_path)

        logger.info("=" * 60)
        logger.info(
            "[PIPELINE] Complete in %.1fs. Artifacts saved to %s",
            elapsed, output_dir,
        )
        logger.info("=" * 60)

        return results

    # ------------------------------------------------------------------
    # Stage 1: Build adapter and load raw data
    # ------------------------------------------------------------------

    def _build_adapter(self) -> Any:
        """Build a DataAdapter from config or fall back to GenericAdapter.

        Returns:
            A DataAdapter instance ready for ``load_raw()``.
        """
        from .adapter import AdapterRegistry, DataAdapter

        stage_start = time.time()
        logger.info("[Stage 1] Building data adapter...")

        adapter_name = getattr(self.config, "adapter", None)

        if adapter_name:
            # Use registered adapter
            logger.info("[Stage 1] Using registered adapter: %s", adapter_name)
            adapter = AdapterRegistry.build(adapter_name, self._config_to_dict())
        else:
            # Fall back to GenericAdapter that reads config.data.source
            logger.info("[Stage 1] No adapter specified, using GenericAdapter")
            adapter = _GenericAdapter(self._config_to_dict())

        logger.info("[Stage 1] Adapter built in %.2fs", time.time() - stage_start)
        return adapter

    # ------------------------------------------------------------------
    # Stage 2: Schema classification
    # ------------------------------------------------------------------

    def _classify_schema(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Classify DataFrame columns into numeric, categorical, and sequence.

        Uses config.features if available, otherwise auto-detects from
        DataFrame dtypes.

        Args:
            df: The main entity-level DataFrame.

        Returns:
            Dict with keys ``"numeric"``, ``"categorical"``, ``"sequence"``.
        """
        stage_start = time.time()
        logger.info("[Stage 2] Classifying schema...")

        # Try to import a dedicated SchemaClassifier
        try:
            from .schema_classifier import SchemaClassifier
            classifier = SchemaClassifier(self.config)
            schema = classifier.classify(df)
            logger.info("[Stage 2] Schema classified via SchemaClassifier in %.2fs",
                        time.time() - stage_start)
            return schema
        except (ImportError, ModuleNotFoundError):
            pass

        # Inline implementation: 5-axis schema matching SchemaClassifier
        schema: Dict[str, List[str]] = {
            "state": [],
            "snapshot": [],
            "timeseries": [],
            "hierarchy": [],
            "item": [],
        }

        if self.config.features.numeric or self.config.features.categorical:
            # Use config as source of truth, mapped to 5-axis keys
            schema["state"] = list(self.config.features.numeric)
            schema["item"] = list(self.config.features.categorical)
            schema["timeseries"] = list(self.config.features.sequence)
            logger.info("[Stage 2] Schema from config in %.2fs",
                        time.time() - stage_start)
            return schema

        # Auto-detect from dtypes
        import numpy as np

        for col in df.columns:
            dtype = df[col].dtype
            if np.issubdtype(dtype, np.number):
                # Binary 0/1 -> item, otherwise state
                nunique = df[col].nunique()
                if nunique <= 2:
                    unique_vals = set(df[col].dropna().unique())
                    if unique_vals.issubset({0, 1, 0.0, 1.0, True, False}):
                        schema["item"].append(col)
                        continue
                schema["state"].append(col)
            elif dtype == object or str(dtype) == "category":
                sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
                if isinstance(sample, (list, np.ndarray)):
                    schema["snapshot"].append(col)
                else:
                    schema["state"].append(col)
            else:
                schema["state"].append(col)

        logger.info("[Stage 2] Schema auto-detected in %.2fs",
                    time.time() - stage_start)
        return schema

    # ------------------------------------------------------------------
    # Stage 3: Encryption
    # ------------------------------------------------------------------

    def _encrypt(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply PII encryption if security is configured.

        Args:
            df: Raw main DataFrame.

        Returns:
            Encrypted DataFrame (or original if encryption is disabled).
        """
        stage_start = time.time()
        logger.info("[Stage 3] Checking encryption configuration...")

        # Check if security config exists and is enabled
        security_cfg = getattr(self.config, "security", None)
        if security_cfg is None or not getattr(security_cfg, "enabled", False):
            logger.info("[Stage 3] Encryption disabled or not configured, passing through")
            return df

        try:
            from ..security.pipeline import EncryptionPipeline
            from ..security.salt_manager import LocalSaltManager
            from ..security.integer_indexer import PIIIntegerIndexer
            from ..security.encryption_policy import derive_from_config

            salt_mgr = LocalSaltManager()
            indexer = PIIIntegerIndexer(
                getattr(security_cfg, "index_store", "pii-indices/")
            )
            policies = derive_from_config(security_cfg)

            pipeline = EncryptionPipeline(salt_mgr, indexer, policies)
            source_name = getattr(security_cfg, "source_name", self.config.task_name)
            result = pipeline.process_source(source_name, df)

            # Save encryption audit
            audit = pipeline.get_audit_report()
            audit_path = self._output_dir / "audit" / "encryption_audit.json"
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            with open(audit_path, "w") as f:
                json.dump(audit, f, indent=2, default=str)
            logger.info("[Stage 3] Encryption audit saved to %s", audit_path)

            # Save index tables
            pipeline.save_indices()

            logger.info("[Stage 3] Encryption complete in %.2fs", time.time() - stage_start)
            return result

        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(
                "[Stage 3] Security module not available (%s), skipping encryption", e,
            )
            return df
        except Exception as e:
            logger.error(
                "[Stage 3] Encryption failed (%s), passing through raw data", e,
            )
            return df

    # ------------------------------------------------------------------
    # Stage 4: Feature engineering + normalization
    # ------------------------------------------------------------------

    def _engineer_features(
        self,
        df: pd.DataFrame,
        raw_data: Dict[str, pd.DataFrame],
    ) -> Tuple[Any, pd.DataFrame]:
        """Build and fit FeatureGroupPipeline, then transform data.

        If ``feature_groups`` are defined in config, uses the new
        FeatureGroupPipeline.  Otherwise falls back to the legacy
        FeaturePipelineBuilder.

        Args:
            df: Main DataFrame (post-encryption).
            raw_data: All raw DataFrames from adapter (for multi-source features).

        Returns:
            ``(fitted_pipeline, transformed_df)``
        """
        stage_start = time.time()
        logger.info("[Stage 4] Engineering features...")

        feature_groups_cfg = self.config.feature_groups

        if feature_groups_cfg:
            return self._engineer_features_grouped(df, feature_groups_cfg)

        # Try to build FeatureGroupConfig from config.features
        return self._engineer_features_from_config(df)

    def _engineer_features_grouped(
        self,
        df: pd.DataFrame,
        groups_cfg: List[Dict[str, Any]],
    ) -> Tuple[Any, pd.DataFrame]:
        """Feature engineering via FeatureGroupPipeline with explicit group configs."""
        from ..feature.group_pipeline import FeatureGroupPipeline
        from ..feature.group import FeatureGroupConfig

        groups = [FeatureGroupConfig.from_dict(g) for g in groups_cfg]

        pipeline = FeatureGroupPipeline(
            groups=groups,
            name=f"{self.config.task_name}_features",
        )

        df_features = pipeline.fit_transform(df)

        logger.info(
            "[Stage 4] FeatureGroupPipeline '%s': %d groups, total_dim=%d, "
            "output %d cols in %.2fs",
            pipeline.name, len(pipeline), pipeline.total_dim,
            len(df_features.columns), time.time() - time.time(),
        )

        return pipeline, df_features

    def _engineer_features_from_config(
        self,
        df: pd.DataFrame,
    ) -> Tuple[Any, pd.DataFrame]:
        """Feature engineering via auto-built FeatureGroupPipeline from FeatureSpec.

        Constructs FeatureGroupConfig objects from the flat config.features
        specification (numeric, categorical, sequence) and wraps them in a
        FeatureGroupPipeline.  This provides the same get_ple_input_metadata()
        API as the explicit grouped path.
        """
        from ..feature.group_pipeline import FeatureGroupPipeline
        from ..feature.group import FeatureGroupConfig

        groups: List[FeatureGroupConfig] = []

        # Numeric features group
        if self.config.features.numeric:
            numeric_cols = list(self.config.features.numeric)
            groups.append(FeatureGroupConfig(
                name="numeric_features",
                group_type="transform",
                transformers=["standard_scaler"],
                columns=numeric_cols,
                output_columns=numeric_cols,
                output_dim=len(numeric_cols),
            ))

        # Categorical features group
        if self.config.features.categorical:
            cat_cols = list(self.config.features.categorical)
            groups.append(FeatureGroupConfig(
                name="categorical_features",
                group_type="transform",
                transformers=["label_encoder"],
                columns=cat_cols,
                output_columns=cat_cols,
                output_dim=len(cat_cols),
            ))

        if not groups:
            # Fallback: treat all non-label columns as numeric
            label_cols = {t.label_col for t in self.config.tasks}
            id_cols = set(getattr(self.config.features, "id_cols", []))
            exclude = label_cols | id_cols
            all_cols = [c for c in df.columns if c not in exclude]

            groups.append(FeatureGroupConfig(
                name="all_features",
                group_type="transform",
                transformers=["null_filler"],
                transformer_params={"null_filler": {"strategy": "median"}},
                columns=all_cols,
                output_columns=all_cols,
                output_dim=len(all_cols),
            ))

        pipeline = FeatureGroupPipeline(
            groups=groups,
            name=f"{self.config.task_name}_features",
        )
        df_features = pipeline.fit_transform(df)

        logger.info(
            "[Stage 4] Auto-built FeatureGroupPipeline: %d groups, total_dim=%d, "
            "output %d cols",
            len(pipeline), pipeline.total_dim, len(df_features.columns),
        )

        return pipeline, df_features

    # ------------------------------------------------------------------
    # Stage 5: Label derivation
    # ------------------------------------------------------------------

    def _derive_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive label columns from raw data.

        Tries to use a dedicated LabelDeriver if available.
        Falls back to extracting label columns directly from config.

        Args:
            df: Raw main DataFrame (pre-feature-engineering).

        Returns:
            DataFrame containing only label columns, aligned with df index.
        """
        import pandas as pd

        stage_start = time.time()
        logger.info("[Stage 5] Deriving labels...")

        # Try dedicated LabelDeriver
        try:
            from .label_deriver import LabelDeriver
            deriver = LabelDeriver(self.config)
            labels = deriver.derive(df)
            logger.info("[Stage 5] Labels derived via LabelDeriver in %.2fs",
                        time.time() - stage_start)
            return labels
        except (ImportError, ModuleNotFoundError):
            pass

        # Inline implementation: extract configured label columns
        label_cols = []
        for task in self.config.tasks:
            col = task.label_col
            if col in df.columns:
                label_cols.append(col)
            else:
                logger.warning(
                    "[Stage 5] Label column '%s' for task '%s' not found in DataFrame",
                    col, task.name,
                )

        if not label_cols:
            raise ValueError(
                f"No label columns found in DataFrame. "
                f"Expected: {[t.label_col for t in self.config.tasks]}, "
                f"Available: {list(df.columns[:20])}..."
            )

        df_labels = df[label_cols].copy()

        logger.info("[Stage 5] Extracted %d label columns in %.2fs",
                    len(label_cols), time.time() - stage_start)
        return df_labels

    # ------------------------------------------------------------------
    # Stage 6: Sequence building
    # ------------------------------------------------------------------

    def _build_sequences(
        self, raw_data: Dict[str, pd.DataFrame],
    ) -> Optional[Dict[str, Any]]:
        """Build sequence tensors from raw data.

        Tries a dedicated SequenceBuilder if available.  Falls back to
        loading .npy files or detecting list-like columns in raw_data.

        Args:
            raw_data: Dict of DataFrames from adapter.

        Returns:
            Dict of sequence arrays keyed by name, or None if no sequences.
        """
        import numpy as np

        stage_start = time.time()
        logger.info("[Stage 6] Building sequences...")

        # Try dedicated SequenceBuilder
        try:
            from .sequence_builder import SequenceBuilder
            builder = SequenceBuilder(self.config)
            sequences = builder.build(raw_data)
            logger.info("[Stage 6] Sequences built via SequenceBuilder in %.2fs",
                        time.time() - stage_start)
            return sequences
        except (ImportError, ModuleNotFoundError):
            pass

        # Inline: check for pre-computed .npy sequence files
        sequences: Dict[str, Any] = {}

        data_dir = Path(self.config.data.source).parent if self.config.data.source else None

        if data_dir and data_dir.exists():
            for npy_file in data_dir.glob("*_sequences.npy"):
                key = npy_file.stem.replace("_sequences", "")
                try:
                    arr = np.load(str(npy_file), allow_pickle=False)
                    sequences[key] = arr
                    logger.info(
                        "[Stage 6] Loaded sequence '%s' from %s: shape=%s",
                        key, npy_file, arr.shape,
                    )
                except Exception as e:
                    logger.warning("[Stage 6] Failed to load %s: %s", npy_file, e)

        # Check for list-like columns in raw_data["main"]
        if not sequences and "main" in raw_data:
            main_df = raw_data["main"]
            seq_cols = list(self.config.features.sequence)

            for col in seq_cols:
                if col in main_df.columns:
                    sample = main_df[col].dropna().iloc[0] if len(main_df[col].dropna()) > 0 else None
                    if isinstance(sample, (list, np.ndarray)):
                        try:
                            # Convert list column to padded 2D/3D array
                            max_len = max(len(x) for x in main_df[col].dropna())
                            padded = np.zeros((len(main_df), max_len), dtype=np.float32)
                            for i, val in enumerate(main_df[col]):
                                if val is not None and hasattr(val, "__len__"):
                                    length = min(len(val), max_len)
                                    padded[i, :length] = np.array(val[:length], dtype=np.float32)
                            sequences[col] = padded
                            logger.info(
                                "[Stage 6] Built sequence '%s' from list column: shape=%s",
                                col, padded.shape,
                            )
                        except Exception as e:
                            logger.warning(
                                "[Stage 6] Failed to convert column '%s' to sequence: %s",
                                col, e,
                            )

        if not sequences:
            logger.info("[Stage 6] No sequences found")
            return None

        logger.info("[Stage 6] Built %d sequence arrays in %.2fs",
                    len(sequences), time.time() - stage_start)
        return sequences

    # ------------------------------------------------------------------
    # Stage 7: Build DataLoaders
    # ------------------------------------------------------------------

    def _build_dataloaders(
        self,
        df_features: pd.DataFrame,
        df_labels: pd.DataFrame,
        sequences: Optional[Dict[str, Any]],
        feature_pipeline: Any,
    ) -> Tuple[Any, Any]:
        """Build train and validation DataLoaders.

        Uses build_ple_dataloader with FeatureColumnSpec constructed from
        the feature pipeline metadata.

        Args:
            df_features: Transformed feature DataFrame.
            df_labels: Label DataFrame (aligned with df_features).
            sequences: Optional dict of sequence arrays.
            feature_pipeline: Fitted FeatureGroupPipeline.

        Returns:
            ``(train_loader, val_loader)``
        """
        import numpy as np
        import pandas as pd

        stage_start = time.time()
        logger.info("[Stage 7] Building DataLoaders...")

        # Merge features and labels into a single DataFrame
        df_combined = pd.concat([df_features.reset_index(drop=True),
                                 df_labels.reset_index(drop=True)], axis=1)

        # Split into train / val -- use temporal split if configured
        temporal_cfg = self._get_temporal_split_config()

        if temporal_cfg is not None and temporal_cfg.get("enabled", False):
            train_df, val_df = self._temporal_split(
                df_combined, temporal_cfg
            )
            logger.info(
                "[Stage 7] Temporal split: train=%d rows, val=%d rows "
                "(gap_days=%d)",
                len(train_df), len(val_df),
                temporal_cfg.get("gap_days", 7),
            )
        else:
            # Fallback: deterministic random shuffle split
            train_frac = self.config.data.train_split
            seed = self.config.training.seed
            n_total = len(df_combined)
            n_train = int(n_total * train_frac)

            rng = np.random.RandomState(seed)
            indices = rng.permutation(n_total)

            train_idx = indices[:n_train]
            val_idx = indices[n_train:]

            train_df = df_combined.iloc[train_idx].reset_index(drop=True)
            val_df = df_combined.iloc[val_idx].reset_index(drop=True)

            logger.info(
                "[Stage 7] Random split: train=%d rows, val=%d rows (split=%.2f)",
                len(train_df), len(val_df), train_frac,
            )

        # Build FeatureColumnSpec from pipeline metadata
        ple_metadata = feature_pipeline.get_ple_input_metadata()
        feature_spec = self._build_feature_spec(ple_metadata)

        # Label column mapping: task_name -> column_name
        label_map = {
            task.name: task.label_col
            for task in self.config.tasks
            if task.label_col in df_labels.columns
        }

        # Build DataLoaders
        from ..data.dataloader import build_ple_dataloader

        batch_size = self.config.training.batch_size

        train_loader = build_ple_dataloader(
            df=train_df,
            feature_spec=feature_spec,
            label_columns=label_map,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
        )

        val_loader = build_ple_dataloader(
            df=val_df,
            feature_spec=feature_spec,
            label_columns=label_map,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )

        logger.info("[Stage 7] DataLoaders built in %.2fs", time.time() - stage_start)
        return train_loader, val_loader

    @staticmethod
    def _build_feature_spec(ple_metadata: Dict[str, Any]) -> Any:
        """Construct a FeatureColumnSpec from pipeline metadata.

        Args:
            ple_metadata: Dict from FeatureGroupPipeline.get_ple_input_metadata().

        Returns:
            A FeatureColumnSpec instance.
        """
        from ..data.dataloader import FeatureColumnSpec

        output_columns = ple_metadata.get("output_columns", [])
        expert_routing = ple_metadata.get("expert_routing", {})

        # Map feature group names to the FeatureColumnSpec fields
        spec_kwargs: Dict[str, Any] = {
            "static_features": list(output_columns),
        }

        # Route columns to specialized fields based on expert routing
        _EXPERT_FIELD_MAP = {
            "hyperbolic": "hyperbolic_columns",
            "tda": "tda_columns",
            "collaborative": "collaborative_columns",
            "hmm_journey": "hmm_journey_columns",
            "hmm_lifecycle": "hmm_lifecycle_columns",
            "hmm_behavior": "hmm_behavior_columns",
            "multidisciplinary": "multidisciplinary_columns",
            "coldstart": "coldstart_columns",
            "anonymous": "anonymous_columns",
        }

        group_ranges = ple_metadata.get("feature_group_ranges", {})

        for expert_name, group_names in expert_routing.items():
            field_name = _EXPERT_FIELD_MAP.get(expert_name)
            if field_name:
                expert_cols: List[str] = []
                for gname in group_names:
                    start, end = group_ranges.get(gname, (0, 0))
                    expert_cols.extend(output_columns[start:end])
                if expert_cols:
                    spec_kwargs[field_name] = expert_cols

        return FeatureColumnSpec(**spec_kwargs)

    # ------------------------------------------------------------------
    # Stage 8: Training
    # ------------------------------------------------------------------

    def _train(
        self,
        train_loader: Any,
        val_loader: Any,
        output_dir: str,
    ) -> dict:
        """Dispatch training to the configured architecture.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            output_dir: Directory for model artifacts.

        Returns:
            Training results dict.
        """
        arch = self.config.model.architecture

        stage_start = time.time()
        logger.info("[Stage 8] Starting training (architecture=%s)...", arch)

        if arch == "ple":
            results = self._train_ple(train_loader, val_loader, output_dir)
        elif arch == "lgbm":
            results = self._train_lgbm(train_loader, val_loader, output_dir)
        else:
            raise ValueError(f"Unknown architecture: {arch}")

        logger.info("[Stage 8] Training complete in %.2fs", time.time() - stage_start)
        return results

    def _train_ple(
        self,
        train_loader: Any,
        val_loader: Any,
        output_dir: str,
    ) -> dict:
        """Build a PLEModel and train with PLETrainer.

        Steps:
          1. Compute input_dim from the feature columns.
          2. Construct a PLEConfig from the pipeline config.
          3. Create a PLETrainer and run 2-phase training.
          4. Save the model checkpoint.
        """
        import torch

        from ..model.ple.config import (
            PLEConfig, ExpertConfig, TaskTowerConfig,
            ExpertBasketConfig, AdaTTConfig, GroupTaskExpertConfig,
        )
        from ..model.ple.model import PLEModel
        from ..training.trainer import PLETrainer
        from ..training.config import TrainingConfig

        # -- Determine input_dim from first batch or config ----------------
        input_dim = self.config.features.input_dim
        if input_dim == 0:
            # Peek at first batch to determine input_dim
            for batch in train_loader:
                if "features" in batch:
                    input_dim = batch["features"].shape[-1]
                break

        task_names = [t.name for t in self.config.tasks]
        logger.info("[PLE] Building model: input_dim=%d, tasks=%d",
                    input_dim, len(task_names))

        # -- Task overrides ------------------------------------------------
        task_overrides: Dict[str, Dict[str, Any]] = {}
        for t in self.config.tasks:
            override: Dict[str, Any] = {"task_type": t.type}
            if t.type == "contrastive":
                override["output_dim"] = t.num_classes
                override["activation"] = None
            elif t.type == "multiclass":
                override["output_dim"] = t.num_classes
                override["activation"] = "softmax"
            elif t.type == "regression":
                override["output_dim"] = 1
                override["activation"] = None
            else:
                override["output_dim"] = 1
                override["activation"] = "sigmoid"
            if t.tower_type:
                override["tower_type"] = t.tower_type
            if t.tower_dims:
                override["tower_dims"] = t.tower_dims
            task_overrides[t.name] = override

        expert_output_dim = max(self.config.model.expert_hidden_dim // 4, 64)

        # -- ExpertBasketConfig --------------------------------------------
        expert_basket_cfg: Optional[ExpertBasketConfig] = None
        if self.config.model.expert_basket is not None:
            eb = self.config.model.expert_basket
            expert_basket_cfg = ExpertBasketConfig(
                shared_experts=eb.get("shared_experts", []),
                task_experts=eb.get("task_experts", []),
                expert_configs=eb.get("expert_configs", {}),
            )

        # -- AdaTTConfig ---------------------------------------------------
        adatt_cfg: Optional[AdaTTConfig] = None
        if self.config.task_groups:
            adatt_cfg = AdaTTConfig.from_pipeline_groups(self.config.task_groups)
            logger.info("[PLE] adaTT: %d task groups", len(self.config.task_groups))

        # -- GroupTaskExpertConfig -----------------------------------------
        group_task_expert_cfg: Optional[GroupTaskExpertConfig] = None
        if self.config.model.group_task_expert is not None:
            gte = self.config.model.group_task_expert
            group_task_expert_cfg = GroupTaskExpertConfig(**gte)

        # -- task_group_map ------------------------------------------------
        task_group_map: Dict[str, str] = {}
        for tg in self.config.task_groups:
            for t in tg.tasks:
                task_group_map[t] = tg.name

        # -- Build PLEConfig -----------------------------------------------
        ple_config = PLEConfig(
            input_dim=input_dim,
            task_names=task_names,
            num_shared_experts=self.config.model.num_shared_experts,
            num_task_experts_per_task=self.config.model.num_task_experts,
            num_extraction_layers=self.config.model.num_layers,
            shared_expert=ExpertConfig(
                hidden_dims=[self.config.model.expert_hidden_dim],
                output_dim=expert_output_dim,
            ),
            task_tower=TaskTowerConfig(
                hidden_dims=list(self.config.model.tower_dims),
                dropout=self.config.model.dropout,
            ),
            dropout=self.config.model.dropout,
            task_overrides=task_overrides,
            expert_basket=expert_basket_cfg,
            task_group_map=task_group_map,
            **({"group_task_expert": group_task_expert_cfg} if group_task_expert_cfg is not None else {}),
            **({"adatt": adatt_cfg} if adatt_cfg is not None else {}),
        )

        model = PLEModel(ple_config)

        # -- TrainingConfig ------------------------------------------------
        total_epochs = self.config.training.epochs
        phase1_epochs = max(1, total_epochs * 3 // 5)
        phase2_epochs = max(1, total_epochs - phase1_epochs)

        training_config = TrainingConfig.from_dict({
            "batch_size": self.config.training.batch_size,
            "optimizer": {"learning_rate": self.config.training.learning_rate},
            "early_stopping": {
                "patience": self.config.training.early_stopping_patience,
            },
            "phase1": {"epochs": phase1_epochs},
            "phase2": {"epochs": phase2_epochs},
            "experiment_name": self.config.task_name,
        })

        # -- Train ---------------------------------------------------------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trainer = PLETrainer(model, training_config, device=device)
        results = trainer.train(train_loader, val_loader, phase="full")

        # -- Save ----------------------------------------------------------
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        checkpoint_path = out / "ple_model.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "ple_config": ple_config,
            "training_results": results,
            "task_names": task_names,
            "input_dim": input_dim,
        }, str(checkpoint_path))

        logger.info(
            "[PLE] best_val_loss=%.6f, saved to %s",
            results.get("best_val_loss", float("inf")),
            checkpoint_path,
        )
        return {"status": "success", "model": "ple", **results}

    def _train_lgbm(
        self,
        train_loader: Any,
        val_loader: Any,
        output_dir: str,
    ) -> dict:
        """Train per-task LGBM models.

        Extracts features and labels from DataLoaders to build flat
        numpy arrays for LightGBM.
        """
        import numpy as np
        from ..model.lgbm.model import LGBMModel
        from ..model.lgbm.config import LGBMConfig

        # Collect all data from loaders into flat arrays
        X_train_parts, y_train_parts = [], {}
        for batch in train_loader:
            features = batch.get("features")
            if features is not None:
                X_train_parts.append(features.numpy())
            targets = batch.get("targets", {})
            for task_name, vals in targets.items():
                y_train_parts.setdefault(task_name, []).append(vals.numpy())

        X_train = np.concatenate(X_train_parts, axis=0)
        y_train = {k: np.concatenate(v) for k, v in y_train_parts.items()}

        cfg = LGBMConfig(
            learning_rate=self.config.training.learning_rate,
            n_estimators=self.config.training.epochs * 25,
        )
        tasks_meta = [{"name": t.name, "type": t.type} for t in self.config.tasks]

        model = LGBMModel(cfg, tasks_meta)
        model.fit(X_train, y_train)

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        model.save(str(out))

        logger.info("[LGBM] Training complete. Model saved to %s", output_dir)
        return {"status": "success", "model": "lgbm"}

    # ------------------------------------------------------------------
    # Stage 1.5: Temporal preparation (leakage prevention)
    # ------------------------------------------------------------------

    def _prepare_temporal(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Truncate sequences and recompute product columns to prevent leakage.

        For datasets like Santander where seq_* columns include the label
        month, this truncates sequences to exclude the prediction window
        and recomputes prod_* from the truncated state.

        Controlled by ``data.temporal_split`` and
        ``preprocessing.leakage_prevention`` in the pipeline config YAML.

        Args:
            df: Raw main DataFrame with sequence and product columns.

        Returns:
            DataFrame with truncated sequences and recomputed products.
        """
        stage_start = time.time()
        logger.info("[Stage 1.5] Preparing temporal features...")

        # Check if leakage prevention is configured
        raw_cfg = self._config_to_dict()
        preproc = raw_cfg.get("preprocessing", {})
        leakage_cfg = preproc.get("leakage_prevention", {})

        if not leakage_cfg.get("recompute_prod_from_seq", False):
            # Also check sequences config for truncate_last
            seq_cfg = raw_cfg.get("sequences", {})
            has_truncation = any(
                s.get("truncate_last", 0) > 0
                for s in seq_cfg.values()
                if isinstance(s, dict)
            )
            if not has_truncation:
                logger.info(
                    "[Stage 1.5] No leakage prevention configured, "
                    "passing through"
                )
                return df

        try:
            from .temporal_split import TemporalSplitter

            splitter = TemporalSplitter()

            # Identify sequence columns (seq_* prefix)
            seq_cols = [c for c in df.columns if c.startswith("seq_")]

            if seq_cols:
                df = splitter.split_by_sequence_cutoff(
                    df,
                    seq_cols=seq_cols,
                    cutoff_offset=1,  # Drop last 1 month (the label month)
                    prod_col_prefix="prod_",
                    seq_col_prefix="seq_",
                )
                logger.info(
                    "[Stage 1.5] Temporal preparation complete in %.2fs: "
                    "truncated %d seq columns, recomputed prod columns",
                    time.time() - stage_start, len(seq_cols),
                )
            else:
                logger.info("[Stage 1.5] No seq_* columns found, skipping")

        except ImportError:
            logger.warning(
                "[Stage 1.5] TemporalSplitter not available, skipping"
            )

        return df

    # ------------------------------------------------------------------
    # Stage 5.5: Leakage validation
    # ------------------------------------------------------------------

    def _validate_leakage(
        self,
        df_features: "pd.DataFrame",
        df_labels: "pd.DataFrame",
        raw_df: Optional["pd.DataFrame"] = None,
    ) -> Any:
        """Run leakage validation checks between features and labels.

        Args:
            df_features: Transformed feature DataFrame.
            df_labels: Label DataFrame.
            raw_df: Optional raw DataFrame for sequence checks.

        Returns:
            ValidationResult from LeakageValidator.
        """
        stage_start = time.time()
        logger.info("[Stage 5.5] Validating for data leakage...")

        try:
            from .leakage_validator import LeakageValidator, ValidationResult
        except ImportError:
            logger.warning("[Stage 5.5] LeakageValidator not available, skipping")

            class _DummyResult:
                passed = True
                warnings: list = []
            return _DummyResult()

        validator = LeakageValidator(
            correlation_threshold=0.95,
            max_seq_len_expected=16,  # Santander: 17 months - 1 truncated
        )

        # Check feature-label correlation
        result = validator.validate(df_features, df_labels)

        # Check sequence leakage if raw data available
        if raw_df is not None:
            seq_cols = [c for c in raw_df.columns if c.startswith("seq_")]
            if seq_cols:
                validator.check_sequence_leakage(
                    raw_df, seq_cols, result=result
                )

            # Check product column leakage
            prod_cols = [c for c in raw_df.columns if c.startswith("prod_")]
            if prod_cols and seq_cols:
                validator.check_product_columns(
                    raw_df, prod_cols, seq_cols, result=result
                )

        logger.info(
            "[Stage 5.5] Leakage validation complete in %.2fs: %s",
            time.time() - stage_start,
            "PASSED" if result.passed else f"FAILED ({len(result.warnings)} warnings)",
        )

        return result

    # ------------------------------------------------------------------
    # Temporal split helper
    # ------------------------------------------------------------------

    def _get_temporal_split_config(self) -> Optional[Dict[str, Any]]:
        """Extract temporal split config from the raw YAML config.

        Returns None if temporal split is not configured.
        """
        raw_cfg = self._config_to_dict()
        data_cfg = raw_cfg.get("data", {})
        return data_cfg.get("temporal_split")

    def _temporal_split(
        self,
        df: "pd.DataFrame",
        temporal_cfg: Dict[str, Any],
    ) -> Tuple[Any, Any]:
        """Apply temporal split to a combined features+labels DataFrame.

        Args:
            df: Combined DataFrame.
            temporal_cfg: Temporal split configuration dict.

        Returns:
            ``(train_df, val_df)``
        """
        from .temporal_split import TemporalSplitter

        splitter = TemporalSplitter(
            train_ratio=temporal_cfg.get("train_ratio", 0.7),
            val_ratio=temporal_cfg.get("val_ratio", 0.15),
            gap_days=temporal_cfg.get("gap_days", 7),
        )

        date_col = temporal_cfg.get("date_col", "snapshot_date")

        train_df, val_df, test_df = splitter.split(df, date_col=date_col)

        # For training pipeline, merge test into val if test is non-empty
        # (we only need train/val for the training loop)
        if len(test_df) > 0:
            import pandas as pd
            val_df = pd.concat([val_df, test_df], ignore_index=True)
            logger.info(
                "[Stage 7] Merged test split (%d rows) into val for "
                "training (total val=%d)",
                len(test_df), len(val_df),
            )

        return train_df, val_df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _config_to_dict(self) -> dict:
        """Convert PipelineConfig to a plain dict for adapter consumption."""
        data_dict: Dict[str, Any] = {
            "source": self.config.data.source,
            "format": self.config.data.format,
            "train_split": self.config.data.train_split,
            "backend": self.config.data.backend,
            "train_path": self.config.data.train_path,
            "s3_path": self.config.data.s3_path,
            "parquet_file": self.config.data.parquet_file,
        }

        # Pass through temporal_split and preprocessing if configured
        if getattr(self.config.data, "temporal_split", None):
            data_dict["temporal_split"] = self.config.data.temporal_split
        if getattr(self.config.data, "preprocessing", None):
            data_dict["preprocessing"] = self.config.data.preprocessing

        return {
            "task_name": self.config.task_name,
            "data": data_dict,
            "preprocessing": getattr(self.config.data, "preprocessing", None) or {},
            "features": {
                "numeric": list(self.config.features.numeric),
                "categorical": list(self.config.features.categorical),
                "sequence": list(self.config.features.sequence),
            },
        }

    # ------------------------------------------------------------------
    # Stage 9: Knowledge Distillation
    # ------------------------------------------------------------------

    def _distill(
        self,
        teacher_checkpoint: str,
        feature_df: Any,
        label_df: Any,
        output_dir: str,
    ) -> dict:
        """Distill PLE teacher into per-task LGBM student models.

        Uses :class:`~core.training.student_trainer.StudentTrainer` to:
        1. Load the trained PLE teacher from checkpoint
        2. Generate soft labels (temperature-scaled predictions)
        3. Train per-task LGBM students on blended hard + soft targets
        4. Validate fidelity (teacher-student agreement)
        5. Save student models

        Returns:
            Dict with distillation results and per-task metrics.
        """
        import numpy as np
        stage_start = time.time()
        logger.info("[Stage 9] Starting knowledge distillation...")

        try:
            import torch
            from ..training.student_trainer import StudentTrainer, StudentConfig
            from ..pipeline.config import TaskSpec
        except ImportError as e:
            logger.warning("[Stage 9] Distillation skipped — missing dependency: %s", e)
            return {"status": "skipped", "reason": str(e)}

        # Build StudentConfig from pipeline config
        distill_cfg = self._config_to_dict().get("distillation", {})
        student_config = StudentConfig(
            teacher_checkpoint=teacher_checkpoint,
            temperature=distill_cfg.get("temperature", 5.0),
            alpha=distill_cfg.get("alpha", 0.5),
            lgbm_params=distill_cfg.get("lgbm_params", {}),
        )

        # Feature columns (exclude labels and IDs)
        id_cols = set(self.config.features.id_cols or ["user_id"])
        label_cols = set(label_df.columns) if hasattr(label_df, 'columns') else set()
        feature_columns = [
            c for c in feature_df.columns
            if c not in id_cols and c not in label_cols
        ]

        # Build task specs list
        task_specs = self.config.tasks

        trainer = StudentTrainer(
            config=student_config,
            task_specs=task_specs,
            feature_columns=feature_columns,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

        # Step 1: Load teacher
        teacher = trainer.load_teacher(teacher_checkpoint)
        logger.info("[Stage 9] Teacher loaded: %d params",
                    sum(p.numel() for p in teacher.parameters()))

        # Step 2: Generate soft labels
        # Build a simple DataLoader from feature_df for teacher inference
        import pandas as pd
        merged = pd.concat([feature_df, label_df], axis=1)
        from ..data.dataloader import build_ple_dataloader, FeatureColumnSpec
        spec = FeatureColumnSpec(static_features=feature_columns)
        label_map = {t.name: t.label_col for t in task_specs}
        loader = build_ple_dataloader(
            df=merged,
            feature_spec=spec,
            label_columns=label_map,
            batch_size=self.config.training.batch_size,
            shuffle=False,
        )
        soft_labels = trainer.generate_soft_labels(loader)
        logger.info("[Stage 9] Soft labels generated: %d tasks", len(soft_labels))

        # Step 3: Train students
        features_np = feature_df[feature_columns].values
        hard_labels = {}
        for t in task_specs:
            if t.label_col in label_df.columns:
                hard_labels[t.name] = label_df[t.label_col].values

        trainer.train_students(features_np, hard_labels)
        logger.info("[Stage 9] Students trained: %d tasks", len(trainer._students))

        # Step 4: Save students (fidelity_results passed after validation below)
        out = Path(output_dir) / "distillation"
        out.mkdir(parents=True, exist_ok=True)

        # Step 5: Fidelity validation (optional)
        fidelity_results = {}
        try:
            from ..training.distillation_validator import DistillationValidator
            validator = DistillationValidator()
            for task_name in trainer._students:
                t_spec = self.config.tasks[0]  # placeholder
                for t in self.config.tasks:
                    if t.name == task_name:
                        t_spec = t
                        break
                teacher_preds = soft_labels.get(task_name)
                student_preds = trainer.predict(
                    task_name, features_np,
                )
                if teacher_preds is not None and student_preds is not None:
                    try:
                        result = validator.validate_task(
                            task_name=task_name,
                            task_type=t_spec.type,
                            teacher_preds=teacher_preds,
                            student_preds=student_preds,
                            labels=hard_labels.get(task_name),
                        )
                        fidelity_results[task_name] = {
                            "passed": result.passed,
                            "metrics": result.metrics,
                        }
                    except Exception as e:
                        fidelity_results[task_name] = {
                            "passed": False,
                            "error": str(e),
                        }
        except ImportError:
            logger.info("[Stage 9] Fidelity validator not available, skipping.")

        # Save students with fidelity results
        trainer.save_students(str(out), fidelity_results=fidelity_results or None)

        # Save fidelity report separately for audit trail
        if fidelity_results:
            fidelity_path = out / "fidelity_report.json"
            with open(fidelity_path, "w") as f:
                json.dump(fidelity_results, f, indent=2, default=str)
            logger.info("[Stage 9] Fidelity report saved to %s", fidelity_path)

        elapsed = time.time() - stage_start
        logger.info(
            "[Stage 9] Distillation complete in %.1fs — %d students, artifacts at %s",
            elapsed, len(trainer._students), out,
        )

        return {
            "status": "completed",
            "num_students": len(trainer._students),
            "tasks": list(trainer._students.keys()),
            "fidelity": fidelity_results,
            "time_seconds": round(elapsed, 2),
            "output_dir": str(out),
        }


# ======================================================================
# GenericAdapter (fallback when no adapter is registered)
# ======================================================================

class _GenericAdapter(DataAdapter):
    """Fallback adapter that reads a single file from config.data.source.

    This adapter is used when no adapter name is specified in config.
    It reads the data source directly (CSV or Parquet) and returns it
    as the "main" DataFrame.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self._metadata = None

    def load_raw(self) -> Dict[str, pd.DataFrame]:
        """Load raw data from the configured source path.

        Returns:
            Dict with "main" key containing the loaded DataFrame.
        """
        import pandas as pd
        from .adapter import AdapterMetadata

        data_cfg = self.config.get("data", {})
        source = data_cfg.get("source", "")
        fmt = data_cfg.get("format", "parquet")

        if not source:
            # Try alternative path fields
            source = data_cfg.get("train_path") or data_cfg.get("s3_path") or ""

        if not source:
            raise ValueError(
                "No data source specified. Set data.source, data.train_path, "
                "or data.s3_path in config."
            )

        logger.info("[GenericAdapter] Loading '%s' (format=%s)", source, fmt)

        if fmt == "parquet":
            df = pd.read_parquet(source)
        elif fmt == "csv":
            df = pd.read_csv(source)
        else:
            raise ValueError(f"Unsupported data format: {fmt}")

        self._metadata = AdapterMetadata(
            num_entities=len(df),
            num_raw_rows=len(df),
            source_files=[source],
            backend_used="pandas",
        )

        logger.info("[GenericAdapter] Loaded %d rows x %d cols", len(df), len(df.columns))
        return {"main": df}

    @property
    def metadata(self) -> Any:
        if self._metadata is None:
            raise RuntimeError("Call load_raw() before accessing metadata")
        return self._metadata
