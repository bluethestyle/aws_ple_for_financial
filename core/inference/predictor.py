"""
PLEPredictor — loads a trained PLE checkpoint and runs inference.

Used for:
  - Evaluating a trained model on validation/test data
  - Generating teacher soft labels for knowledge distillation

PLEConfig is reconstructed via core.model.config_builder (single source
of truth shared with train.py).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

logger = logging.getLogger(__name__)


class PLEPredictor:
    """Load a trained PLE checkpoint and run inference.

    Args:
        checkpoint_path: Path to .pt checkpoint file produced by train.py.
        config_path: Path to pipeline.yaml.
        feature_schema_path: Path to feature_schema.json (also used to
            locate label_schema.json in the same directory).
        device: "auto", "cuda", or "cpu".
        hp_overrides: Optional HP dict with the same keys as train.py
            (use_adatt, gate_type, shared_experts, use_ple, …).
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        feature_schema_path: str,
        device: str = "auto",
        hp_overrides: Optional[Dict] = None,
        dataset_config_path: Optional[str] = None,
    ):
        self._hp = hp_overrides or {}

        # -- Resolve device --
        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        # -- Load pipeline config (supports split-config pattern) --
        if dataset_config_path and Path(dataset_config_path).exists():
            from core.pipeline.config import load_merged_config
            self._pipeline_config: dict = load_merged_config(config_path, dataset_config_path)
            logger.info("PLEPredictor: config merged from %s + %s", config_path, dataset_config_path)
        else:
            with open(config_path, encoding="utf-8") as f:
                self._pipeline_config: dict = yaml.safe_load(f)

        # -- Load feature schema --
        schema_path = Path(feature_schema_path)
        with open(schema_path, encoding="utf-8") as f:
            self._feature_schema: dict = json.load(f)

        # -- Load label schema (expected alongside feature_schema.json) --
        label_schema_path = schema_path.parent / "label_schema.json"
        if label_schema_path.exists():
            with open(label_schema_path, encoding="utf-8") as f:
                self._label_schema: dict = json.load(f)
        else:
            from core.model.config_builder import build_label_schema_from_config
            logger.warning("label_schema.json not found next to feature_schema.json; "
                           "falling back to pipeline.yaml")
            self._label_schema = build_label_schema_from_config(self._pipeline_config)

        # -- Derive input_dim from schema --
        input_dim = self._feature_schema.get("num_features")
        if input_dim is None:
            columns = self._feature_schema.get("columns", [])
            input_dim = len(columns)
        if input_dim == 0:
            raise ValueError(
                "Cannot determine input_dim from feature_schema.json. "
                "Ensure 'num_features' or 'columns' is present."
            )

        # -- Build PLEConfig (single source of truth) --
        from core.model.config_builder import build_ple_config
        self._ple_config = build_ple_config(
            config=self._pipeline_config,
            feature_schema=self._feature_schema,
            label_schema=self._label_schema,
            input_dim=input_dim,
            hp=self._hp,
        )

        # -- Build model and load weights --
        from core.model.ple.model import PLEModel
        self._model = PLEModel(self._ple_config).to(self._device)

        ckpt = torch.load(checkpoint_path, map_location=self._device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        self._model.load_state_dict(state_dict, strict=True)
        self._model.eval()

        logger.info(
            "PLEPredictor ready: checkpoint=%s, device=%s, tasks=%d, input_dim=%d",
            Path(checkpoint_path).name,
            self._device,
            len(self._ple_config.task_names),
            input_dim,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def task_names(self) -> List[str]:
        return self._ple_config.task_names

    @property
    def model(self):
        from core.model.ple.model import PLEModel
        return self._model

    @property
    def config(self):
        from core.model.ple.config import PLEConfig
        return self._ple_config

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _prepare_inputs(self, batch: Any):
        """Convert batch to PLEInput and move to device."""
        from core.model.ple.model import PLEInput
        if isinstance(batch, PLEInput):
            return batch.to(self._device)
        if hasattr(batch, "to_ple_input"):
            return batch.to_ple_input().to(self._device)
        if isinstance(batch, dict):
            return PLEInput(
                features=batch["features"].to(self._device),
                cluster_ids=batch["cluster_ids"].to(self._device) if "cluster_ids" in batch else None,
                cluster_probs=batch["cluster_probs"].to(self._device) if "cluster_probs" in batch else None,
                targets=batch.get("targets"),
            )
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    def predict_batch(self, inputs: Any):
        """Run inference on a single batch.

        Args:
            inputs: PLEInput, dict batch, or any object with to_ple_input().

        Returns:
            PLEOutput with per-task predictions (no loss computed).
        """
        from core.model.ple.model import PLEOutput
        ple_inputs = self._prepare_inputs(inputs)
        use_amp = self._device.type == "cuda"
        with torch.no_grad():
            if use_amp:
                with torch.cuda.amp.autocast():
                    output = self._model(ple_inputs, compute_loss=False)
            else:
                output = self._model(ple_inputs, compute_loss=False)
        return output

    def predict(self, dataloader) -> Dict[str, torch.Tensor]:
        """Run inference on an entire dataloader.

        Args:
            dataloader: DataLoader yielding batches compatible with PLEInput.

        Returns:
            Dict mapping task_name -> concatenated prediction tensor
            (shape: [N, output_dim]).
        """
        all_preds: Dict[str, List[torch.Tensor]] = {name: [] for name in self.task_names}

        self._model.eval()
        use_amp = self._device.type == "cuda"

        with torch.no_grad():
            for batch in dataloader:
                ple_inputs = self._prepare_inputs(batch)
                if use_amp:
                    with torch.cuda.amp.autocast():
                        output = self._model(ple_inputs, compute_loss=False)
                else:
                    output = self._model(ple_inputs, compute_loss=False)

                for task_name, pred in output.predictions.items():
                    if task_name in all_preds:
                        all_preds[task_name].append(pred.cpu())

        return {
            task_name: torch.cat(tensors, dim=0)
            for task_name, tensors in all_preds.items()
            if tensors
        }
