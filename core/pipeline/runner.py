"""
PipelineRunner -- orchestrates the full train pipeline locally or on SageMaker.

Integrates:
  - FeaturePipelineBuilder / FeaturePipeline for config-driven feature engineering.
  - PLEModel + PLETrainer for PLE training with 2-phase support.
  - LGBMModel for lightweight tree-based training.

Usage::

    config = load_config("configs/examples/multitask.yaml")
    runner = PipelineRunner(config)
    runner.run(mode="local")       # local dev/test
    runner.run(mode="sagemaker")   # AWS Spot training
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

from .config import PipelineConfig

logger = logging.getLogger(__name__)


class PipelineRunner:
    """Execute a training pipeline driven by :class:`PipelineConfig`.

    The runner selects an execution mode (local or SageMaker) and wires
    together data loading, feature engineering, model training, and
    artifact saving.

    Args:
        config: A fully populated :class:`PipelineConfig`.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, mode: str = "local", output_dir: str = "outputs/") -> dict:
        """Run the pipeline in the specified mode.

        Args:
            mode: ``"local"`` for single-machine training,
                  ``"sagemaker"`` for AWS SageMaker.
            output_dir: Directory for model artifacts (local mode only).

        Returns:
            A result dict containing at least ``"status"`` and ``"model"``.
        """
        if mode == "local":
            return self._run_local(output_dir)
        elif mode == "sagemaker":
            return self._run_sagemaker()
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'local' or 'sagemaker'.")

    # ------------------------------------------------------------------
    # Local execution
    # ------------------------------------------------------------------

    def _run_local(self, output_dir: str) -> dict:
        """Execute the full pipeline locally.

        Steps:
          1. Load raw data.
          2. Build and fit the feature pipeline, then transform the data.
          3. Split into train / validation sets.
          4. Train the model (PLE or LGBM).
          5. Save artifacts.
        """
        import numpy as np  # noqa: F811 -- deferred to keep module importable
        import pandas as pd  # noqa: F811

        logger.info("[LOCAL] Starting pipeline: %s", self.config.task_name)

        # 1. Data loading
        df = self._load_data_local()
        logger.info("[LOCAL] Loaded %d rows x %d columns", len(df), len(df.columns))

        # 2. Feature engineering via FeaturePipeline
        feature_pipeline, processed = self._build_features(df)

        # 3. Train/val split
        train_df, val_df = self._split_data(processed)

        # 4. Model training
        results = self._train(train_df, val_df, output_dir)

        # 5. Save feature pipeline alongside the model
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        feature_pipeline.save(str(out / "feature_pipeline"))
        logger.info(
            "[LOCAL] Pipeline complete. Artifacts saved to %s", output_dir,
        )

        results["output_dir"] = output_dir
        return results

    # ------------------------------------------------------------------
    # SageMaker execution
    # ------------------------------------------------------------------

    def _run_sagemaker(self) -> dict:
        """Launch training on SageMaker."""
        from ...aws.sagemaker.trainer import SageMakerTrainer  # type: ignore[import]

        trainer = SageMakerTrainer(self.config)
        return trainer.launch()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data_local(self) -> pd.DataFrame:
        """Load the raw dataset from a local path or S3."""
        import pandas as pd

        source = self.config.data.source
        fmt = self.config.data.format
        if fmt == "parquet":
            return pd.read_parquet(source)
        elif fmt == "csv":
            return pd.read_csv(source)
        else:
            raise ValueError(f"Unsupported data format: {fmt}")

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _build_features(
        self, df: pd.DataFrame
    ) -> Tuple[Any, pd.DataFrame]:
        """Build and fit a FeaturePipeline from the config, then transform *df*.

        Uses :class:`FeaturePipelineBuilder` so transformer steps, column
        mappings, and power-law auto-detection are all config-driven.

        Returns:
            ``(fitted_pipeline, transformed_df)``
        """
        from ..feature.pipeline_builder import FeaturePipelineBuilder
        from ..feature.pipeline import FeaturePipeline

        label_cols = [t.label_col for t in self.config.tasks]
        id_cols = getattr(self.config.features, "id_cols", [])

        # Build a config dict compatible with FeaturePipelineBuilder
        pipeline_cfg: Dict[str, Any] = {
            "feature_pipeline": {
                "name": f"{self.config.task_name}_features",
                "schema": {
                    "numeric": list(self.config.features.numeric),
                    "categorical": list(self.config.features.categorical),
                    "sequence": list(self.config.features.sequence),
                    "label_cols": label_cols,
                    "id_cols": id_cols,
                },
                "steps": self._build_transformer_steps(),
            }
        }

        builder = FeaturePipelineBuilder(pipeline_cfg)
        feature_pipeline: FeaturePipeline = builder.build_and_fit(df)
        transformed = feature_pipeline.transform(df)

        logger.info(
            "[FEATURES] Pipeline '%s': %d transformers, output %d cols",
            feature_pipeline.name,
            len(feature_pipeline),
            len(transformed.columns),
        )

        return feature_pipeline, transformed

    def _build_transformer_steps(self) -> list:
        """Convert the config's transformer list into FeaturePipelineBuilder steps.

        If no transformers are specified in the config, provide a sensible
        default chain (null_filler + standard_scaler + label_encoder).
        """
        raw_steps = self.config.features.transformers
        if raw_steps:
            # Each raw step may use "name" or "transformer"; normalise.
            steps = []
            for step in raw_steps:
                entry: Dict[str, Any] = {}
                entry["transformer"] = step.get("transformer") or step.get("name")
                if "params" in step:
                    entry["params"] = step["params"]
                elif "cols" in step:
                    # Shorthand used in example YAMLs
                    entry["params"] = {"columns": step["cols"]}
                steps.append(entry)
            return steps

        # Default transformer chain
        steps = [
            {
                "transformer": "null_filler",
                "params": {"strategy": "median"},
            },
        ]
        if self.config.features.numeric:
            steps.append({
                "transformer": "standard_scaler",
                "params": {"columns": list(self.config.features.numeric)},
            })
        if self.config.features.categorical:
            steps.append({
                "transformer": "label_encoder",
                "params": {"columns": list(self.config.features.categorical)},
            })
        return steps

    # ------------------------------------------------------------------
    # Data splitting
    # ------------------------------------------------------------------

    def _split_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split a DataFrame into train and validation sets.

        Uses ``data.train_split`` from the config. The remaining rows
        form the validation set.
        """
        train_frac = self.config.data.train_split
        n_train = int(len(df) * train_frac)

        # Deterministic shuffle
        seed = self.config.training.seed
        shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

        train_df = shuffled.iloc[:n_train]
        val_df = shuffled.iloc[n_train:]

        logger.info(
            "[SPLIT] train=%d rows, val=%d rows (split=%.2f)",
            len(train_df), len(val_df), train_frac,
        )
        return train_df, val_df

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        output_dir: str,
    ) -> dict:
        """Dispatch training to the configured architecture (PLE or LGBM)."""
        arch = self.config.model.architecture

        if arch == "lgbm":
            return self._train_lgbm(train_df, val_df, output_dir)
        elif arch == "ple":
            return self._train_ple(train_df, val_df, output_dir)
        else:
            raise ValueError(f"Unknown architecture: {arch}")

    # -- LGBM ----------------------------------------------------------

    def _train_lgbm(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        output_dir: str,
    ) -> dict:
        """Train per-task LGBM models."""
        from ..model.lgbm.model import LGBMModel
        from ..model.lgbm.config import LGBMConfig

        label_cols = [t.label_col for t in self.config.tasks]
        feat_cols = [c for c in train_df.columns if c not in label_cols]

        X_train = train_df[feat_cols].values
        y_train = {t.name: train_df[t.label_col].values for t in self.config.tasks}

        cfg = LGBMConfig(
            learning_rate=self.config.training.learning_rate,
            n_estimators=self.config.training.epochs * 25,  # rough mapping
        )
        tasks_meta = [{"name": t.name, "type": t.type} for t in self.config.tasks]

        model = LGBMModel(cfg, tasks_meta)
        model.fit(X_train, y_train)

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        model.save(str(out))

        logger.info("[LGBM] Training complete. Model saved to %s", output_dir)
        return {"status": "success", "model": "lgbm"}

    # -- PLE -----------------------------------------------------------

    def _train_ple(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        output_dir: str,
    ) -> dict:
        """Build a PLEModel and train it with PLETrainer.

        The method:
          1. Computes input_dim from the feature columns.
          2. Constructs a PLEConfig from the pipeline config.
          3. Builds train/val DataLoaders.
          4. Creates a PLETrainer and runs 2-phase training.
          5. Saves the model checkpoint.
        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        from ..model.ple.config import PLEConfig, ExpertConfig, TaskTowerConfig, ExpertBasketConfig, AdaTTConfig, GroupTaskExpertConfig
        from ..model.ple.model import PLEModel
        from ..training.trainer import PLETrainer
        from ..training.config import TrainingConfig

        label_cols = [t.label_col for t in self.config.tasks]
        feat_cols = [c for c in train_df.columns if c not in label_cols]
        input_dim = len(feat_cols)

        logger.info(
            "[PLE] Building model: input_dim=%d, tasks=%d",
            input_dim, len(self.config.tasks),
        )

        # -- PLEConfig -------------------------------------------------
        task_names = [t.name for t in self.config.tasks]
        task_overrides: Dict[str, Dict[str, Any]] = {}
        for t in self.config.tasks:
            override: Dict[str, Any] = {"task_type": t.type}
            if t.type == "multiclass":
                override["output_dim"] = t.num_classes
                override["activation"] = "softmax"
            elif t.type == "regression":
                override["output_dim"] = 1
                override["activation"] = None
            else:
                override["output_dim"] = 1
                override["activation"] = "sigmoid"
            task_overrides[t.name] = override

        expert_output_dim = max(self.config.model.expert_hidden_dim // 4, 64)

        # Build ExpertBasketConfig if expert_basket is defined in model spec
        expert_basket_cfg: Optional[ExpertBasketConfig] = None
        if self.config.model.expert_basket is not None:
            eb = self.config.model.expert_basket
            expert_basket_cfg = ExpertBasketConfig(
                shared_experts=eb.get("shared_experts", []),
                task_experts=eb.get("task_experts", []),
                expert_configs=eb.get("expert_configs", {}),
            )

        # -- adaTT config from pipeline task_groups (single source of truth) --
        adatt_cfg: Optional[AdaTTConfig] = None
        if self.config.task_groups:
            adatt_cfg = AdaTTConfig.from_pipeline_groups(self.config.task_groups)
            logger.info(
                "[PLE] adaTT task_groups from pipeline config: %d groups",
                len(self.config.task_groups),
            )

        # -- Build GroupTaskExpertConfig from pipeline model config --------------
        group_task_expert_cfg: Optional[GroupTaskExpertConfig] = None
        if self.config.model.group_task_expert is not None:
            gte = self.config.model.group_task_expert
            group_task_expert_cfg = GroupTaskExpertConfig(**gte)
            logger.info(
                "[PLE] GroupTaskExpertBasket: enabled=%s, output_dim=%d",
                gte.get("enabled", True),
                gte.get("task_output_dim", 32),
            )

        # -- Build task_group_map from pipeline task_groups -------------------
        task_group_map: Dict[str, str] = {}
        for tg in self.config.task_groups:
            for t in tg.tasks:
                task_group_map[t] = tg.name

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

        # -- DataLoaders -----------------------------------------------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_loader = self._build_dataloader(
            train_df, feat_cols, label_cols, task_names,
            batch_size=self.config.training.batch_size, shuffle=True,
        )
        val_loader = self._build_dataloader(
            val_df, feat_cols, label_cols, task_names,
            batch_size=self.config.training.batch_size, shuffle=False,
        )

        # -- TrainingConfig --------------------------------------------
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

        # -- Train -----------------------------------------------------
        trainer = PLETrainer(model, training_config, device=device)
        results = trainer.train(train_loader, val_loader, phase="full")

        # -- Save ------------------------------------------------------
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
            "[PLE] Training complete. best_val_loss=%.6f, saved to %s",
            results.get("best_val_loss", float("inf")),
            checkpoint_path,
        )
        return {"status": "success", "model": "ple", **results}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_dataloader(
        df: pd.DataFrame,
        feat_cols: list,
        label_cols: list,
        task_names: list,
        batch_size: int = 2048,
        shuffle: bool = True,
        feature_spec: Optional[Any] = None,
        use_gpu_loading: bool = False,
    ) -> Any:
        """Convert a DataFrame into a PyTorch DataLoader of dicts.

        Each batch is a ``dict`` with ``"features"`` and ``"targets"`` keys,
        compatible with :meth:`PLETrainer._prepare_inputs`.

        When *feature_spec* (a :class:`FeatureColumnSpec`) is provided, the
        new GPU-capable ``build_ple_dataloader`` is used.  Otherwise, the
        simple legacy path is preserved for backward compatibility.
        """
        if feature_spec is not None:
            from ..data.dataloader import build_ple_dataloader

            label_map = {
                task_name: label_col
                for task_name, label_col in zip(task_names, label_cols)
            }
            return build_ple_dataloader(
                df=df,
                feature_spec=feature_spec,
                label_columns=label_map,
                batch_size=batch_size,
                shuffle=shuffle,
                use_gpu_loading=use_gpu_loading,
                pin_memory=True,
            )

        # ---- Legacy simple path (backward compatible) ----
        import numpy as np
        import torch
        from torch.utils.data import DataLoader

        features = torch.tensor(
            df[feat_cols].values.astype(np.float32),
            dtype=torch.float32,
        )

        targets: Dict[str, torch.Tensor] = {}
        for task_name, label_col in zip(task_names, label_cols):
            if label_col in df.columns:
                targets[task_name] = torch.tensor(
                    df[label_col].values.astype(np.float32),
                    dtype=torch.float32,
                )

        class _DictDataset(torch.utils.data.Dataset):
            """Simple dataset that yields dict batches."""

            def __init__(self, feats, tgts):
                self.feats = feats
                self.tgts = tgts

            def __len__(self):
                return len(self.feats)

            def __getitem__(self, idx):
                item = {"features": self.feats[idx]}
                item["targets"] = {k: v[idx] for k, v in self.tgts.items()}
                return item

        def _collate(batch):
            feats = torch.stack([b["features"] for b in batch])
            tgts: Dict[str, torch.Tensor] = {}
            for key in batch[0]["targets"]:
                tgts[key] = torch.stack([b["targets"][key] for b in batch])
            return {"features": feats, "targets": tgts}

        dataset = _DictDataset(features, targets)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=_collate,
            drop_last=False,
        )
