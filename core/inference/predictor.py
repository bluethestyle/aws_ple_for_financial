"""
PLEPredictor — loads a trained PLE checkpoint and runs inference.

Used for:
  - Evaluating a trained model on validation/test data
  - Generating teacher soft labels for knowledge distillation

PLEConfig is reconstructed from pipeline.yaml + feature_schema.json +
label_schema (embedded in the same feature_schema dir or supplied
separately).  The checkpoint file does NOT contain ple_config.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

logger = logging.getLogger(__name__)


def _parse_bool_hp(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return json.loads(val)
    return bool(val)


def _build_ple_config(config: dict, feature_schema: dict, label_schema: dict,
                      input_dim: int, hp: dict):
    """Reconstruct PLEConfig identically to train.py build_model()."""
    from core.model.ple.config import (
        PLEConfig, ExpertConfig, ExpertBasketConfig,
        LossWeightingConfig, LogitTransferDef,
        GroupTaskExpertConfig, AdaTTConfig, TaskGroupDef,
        CGCConfig, ExpertInputConfig,
    )

    tasks = label_schema.get("tasks", [])
    task_names = [t["name"] for t in tasks]

    model_config = label_schema.get("model", {})
    ple_cfg = model_config.get("ple", {})
    expert_cfg = model_config.get("expert_config", {})
    tower_cfg = model_config.get("task_tower", {})

    mlp_cfg = expert_cfg.get("mlp", {})
    expert_hidden = mlp_cfg.get("hidden_dims", [input_dim * 2, input_dim])
    expert_output = ple_cfg.get("extraction_dim", 32)

    shared_expert = ExpertConfig(
        hidden_dims=expert_hidden, output_dim=expert_output,
        dropout=model_config.get("dropout", 0.1),
    )
    task_expert = ExpertConfig(
        hidden_dims=expert_hidden, output_dim=expert_output,
        dropout=model_config.get("dropout", 0.1),
    )

    num_extraction_layers = ple_cfg.get("num_layers", 2)
    hp_num_layers = hp.get("num_layers")
    if hp_num_layers is not None:
        num_extraction_layers = int(hp_num_layers)

    num_shared_experts = ple_cfg.get("num_shared_experts", 2)
    expert_basket = None

    shared_experts_raw = hp.get("shared_experts", "")
    if isinstance(shared_experts_raw, str) and shared_experts_raw:
        try:
            shared_experts_override = json.loads(shared_experts_raw)
        except (json.JSONDecodeError, ValueError):
            shared_experts_override = None
    elif isinstance(shared_experts_raw, list):
        shared_experts_override = shared_experts_raw
    else:
        shared_experts_override = None

    if shared_experts_override is not None:
        num_shared_experts = len(shared_experts_override)
        expert_basket_cfg = model_config.get("expert_basket", {})
        expert_basket = ExpertBasketConfig(
            shared_experts=shared_experts_override,
            task_experts=expert_basket_cfg.get("task", ["mlp"]),
            expert_configs={
                name: expert_cfg.get(name, {})
                for name in shared_experts_override if name in expert_cfg
            },
        )
    else:
        eb_cfg = model_config.get("expert_basket", {})
        if eb_cfg.get("shared"):
            expert_basket = ExpertBasketConfig(
                shared_experts=eb_cfg["shared"],
                task_experts=eb_cfg.get("task", ["mlp"]),
                expert_configs={
                    name: expert_cfg.get(name, {}) for name in eb_cfg["shared"] if name in expert_cfg
                },
            )

    if expert_basket is not None:
        _active_raw = hp.get("active_experts")
        _removed_raw = hp.get("removed_experts")

        _active_experts = None
        if _active_raw:
            _active_experts = json.loads(_active_raw) if isinstance(_active_raw, str) else _active_raw

        _removed_experts = None
        if _removed_raw:
            _removed_experts = json.loads(_removed_raw) if isinstance(_removed_raw, str) else _removed_raw

        if _active_experts is not None:
            filtered_shared = [e for e in expert_basket.shared_experts if e in _active_experts]
            filtered_task = [e for e in expert_basket.task_experts if e in _active_experts]
            filtered_configs = {k: v for k, v in expert_basket.expert_configs.items() if k in _active_experts}
            expert_basket = ExpertBasketConfig(
                shared_experts=filtered_shared,
                task_experts=filtered_task,
                expert_configs=filtered_configs,
            )
            num_shared_experts = len(filtered_shared)
        elif _removed_experts is not None:
            filtered_shared = [e for e in expert_basket.shared_experts if e not in _removed_experts]
            filtered_task = [e for e in expert_basket.task_experts if e not in _removed_experts]
            filtered_configs = {k: v for k, v in expert_basket.expert_configs.items() if k not in _removed_experts}
            expert_basket = ExpertBasketConfig(
                shared_experts=filtered_shared,
                task_experts=filtered_task,
                expert_configs=filtered_configs,
            )
            num_shared_experts = len(filtered_shared)

    # Structure ablation: PLE toggle
    use_ple_raw = hp.get("use_ple")
    if use_ple_raw is not None and not _parse_bool_hp(use_ple_raw):
        num_extraction_layers = 1

    # Structure ablation: adaTT toggle
    use_adatt_raw = hp.get("use_adatt")
    if use_adatt_raw is not None and not _parse_bool_hp(use_adatt_raw):
        model_config["adatt"] = {"enabled": False}

    # Structure ablation: CGC gate toggle
    use_cgc_gate_raw = hp.get("use_cgc_gate")
    if use_cgc_gate_raw is not None and not _parse_bool_hp(use_cgc_gate_raw):
        model_config.setdefault("cgc", {})["enabled"] = False

    # Structure ablation: GroupTaskExpert toggle
    use_gte_raw = hp.get("use_group_task_expert")
    if use_gte_raw is not None and not _parse_bool_hp(use_gte_raw):
        model_config.setdefault("group_task_expert", {})["enabled"] = False

    # Structure ablation: Logit transfer toggle
    use_lt_raw = hp.get("use_logit_transfer")
    if use_lt_raw is not None and not _parse_bool_hp(use_lt_raw):
        label_schema["task_relationships"] = []

    # Structure ablation: HMM projectors toggle
    use_hmm_raw = hp.get("use_hmm_projectors")
    if use_hmm_raw is not None and not _parse_bool_hp(use_hmm_raw):
        model_config["_disable_hmm_projectors"] = True

    # Loss weighting
    lw_cfg = model_config.get("loss_weighting", {})
    loss_weighting = LossWeightingConfig(
        strategy=lw_cfg.get("strategy", "fixed"),
        gradnorm_alpha=lw_cfg.get("gradnorm_alpha", 1.5),
        gradnorm_interval=lw_cfg.get("gradnorm_interval", 1),
        dwa_temperature=lw_cfg.get("dwa_temperature", 2.0),
        dwa_window_size=lw_cfg.get("dwa_window_size", 5),
    )

    # Build PLEConfig
    ple_config = PLEConfig(
        input_dim=input_dim,
        task_names=task_names,
        num_shared_experts=num_shared_experts,
        num_extraction_layers=num_extraction_layers,
        num_task_experts_per_task=ple_cfg.get("num_task_experts", 1),
        shared_expert=shared_expert,
        task_expert=task_expert,
        dropout=model_config.get("dropout", 0.1),
        expert_basket=expert_basket,
        loss_weighting=loss_weighting,
    )

    # Gate type
    gate_type_raw = hp.get("gate_type", "softmax")
    ple_config.gate_type = gate_type_raw

    # CGC enabled/disabled
    cgc_cfg_raw = model_config.get("cgc", {})
    if cgc_cfg_raw and not cgc_cfg_raw.get("enabled", True):
        ple_config.cgc = CGCConfig(enabled=False)

    # Per-expert input dimensions
    expert_input_dims_raw = model_config.get("expert_input_dims", {})
    if expert_input_dims_raw:
        ple_config.expert_input_dims = {k: int(v) for k, v in expert_input_dims_raw.items()}

    # Task loss weights
    ple_config.task_loss_weights = {t["name"]: t.get("loss_weight", 1.0) for t in tasks}

    # Task overrides (type + output_dim + loss)
    for t in tasks:
        task_override = {
            "task_type": t.get("type", "binary"),
            "output_dim": t.get("num_classes", 1),
        }
        if "loss" in t:
            task_override["loss"] = t["loss"]
        if "loss_params" in t:
            task_override["loss_params"] = t["loss_params"]
        if "topk_k" in t:
            task_override["topk_k"] = t["topk_k"]
        ple_config.task_overrides[t["name"]] = task_override

    # Logit transfers
    logit_transfers_raw = label_schema.get("task_relationships", [])
    if logit_transfers_raw:
        ple_config.logit_transfers = [
            LogitTransferDef(
                source=lt["source"], target=lt["target"],
                enabled=lt.get("enabled", True),
                transfer_method=lt.get("transfer_method", "residual"),
            )
            for lt in logit_transfers_raw
        ]
        ple_config.logit_transfer_strength = float(
            label_schema.get("logit_transfer_strength", 0.5)
        )

    # Feature group ranges
    group_ranges = feature_schema.get("feature_group_ranges",
                                      feature_schema.get("group_ranges", {}))
    if group_ranges:
        ple_config.feature_group_ranges = {k: tuple(v) for k, v in group_ranges.items()}
    col_ranges = feature_schema.get("group_ranges", {})

    # Inject feature_group_ranges into DeepFM expert config
    if col_ranges and ple_config.expert_basket is not None:
        deepfm_cfg = ple_config.expert_basket.expert_configs.get("deepfm")
        if deepfm_cfg is not None and deepfm_cfg.get("field_dims") == "auto":
            deepfm_cfg["feature_group_ranges"] = {
                k: tuple(v) for k, v in col_ranges.items()
            }

    # Expert routing
    expert_routing = feature_schema.get("expert_routing", {})
    if not expert_routing and group_ranges:
        fg_cfg = config.get("feature_groups", [])
        if not fg_cfg:
            _fg_path = config.get("feature_groups_file", "")
            if _fg_path:
                _fg_p = Path(_fg_path)
                if _fg_p.exists():
                    with open(_fg_p, encoding="utf-8") as _f_fg:
                        fg_cfg = yaml.safe_load(_f_fg).get("feature_groups", [])
        if fg_cfg:
            from collections import defaultdict
            _expert_to_groups: Dict[str, list] = defaultdict(list)
            for fg in fg_cfg:
                for exp in fg.get("target_experts", []):
                    if fg["name"] in group_ranges:
                        _expert_to_groups[exp].append(fg["name"])
            expert_routing = [
                {"expert_name": exp, "input_groups": grps}
                for exp, grps in _expert_to_groups.items()
            ]
    if expert_routing:
        ple_config.expert_input_routing = [
            ExpertInputConfig(**r) if isinstance(r, dict) else r
            for r in expert_routing
        ]

    # Task group map from label schema
    task_group_map = label_schema.get("task_group_map", {})
    if task_group_map:
        ple_config.task_group_map = task_group_map

    # adaTT task_groups
    raw_task_groups = label_schema.get("task_groups", [])
    if raw_task_groups:
        adatt_cfg_raw = model_config.get("adatt", {})
        if not adatt_cfg_raw:
            adatt_cfg_raw = label_schema.get("adatt", {})
        if not adatt_cfg_raw:
            adatt_cfg_raw = config.get("adatt", {})
        adatt_task_groups: Dict[str, TaskGroupDef] = {}
        for tg in raw_task_groups:
            tg_name = tg["name"] if isinstance(tg, dict) else tg.name
            tg_tasks = tg["tasks"] if isinstance(tg, dict) else tg.tasks
            tg_intra = (
                tg.get("adatt_intra_strength", 0.7) if isinstance(tg, dict)
                else getattr(tg, "adatt_intra_strength", 0.7)
            )
            adatt_task_groups[tg_name] = TaskGroupDef(
                members=list(tg_tasks),
                intra_strength=tg_intra,
            )
        _freeze = adatt_cfg_raw.get("freeze_epoch")
        if _freeze is not None:
            _freeze = int(_freeze)
        ple_config.adatt = AdaTTConfig(
            enabled=adatt_cfg_raw.get("enabled", True),
            task_groups=adatt_task_groups,
            inter_group_strength=adatt_cfg_raw.get("inter_group_strength", 0.3),
            transfer_lambda=adatt_cfg_raw.get("transfer_lambda", 0.1),
            temperature=adatt_cfg_raw.get("temperature", 1.0),
            warmup_epochs=adatt_cfg_raw.get("warmup_epochs", 3),
            freeze_epoch=_freeze,
            grad_interval=adatt_cfg_raw.get("grad_interval", 10),
        )

    # GroupTaskExpert config
    gte_cfg_raw = model_config.get("group_task_expert", {})
    if gte_cfg_raw:
        ple_config.group_task_expert = GroupTaskExpertConfig(
            enabled=gte_cfg_raw.get("enabled", True),
            group_hidden_dim=gte_cfg_raw.get("group_hidden_dim",
                                              gte_cfg_raw.get("group_hidden", 128)),
            group_output_dim=gte_cfg_raw.get("group_output_dim",
                                              gte_cfg_raw.get("group_output", 64)),
            cluster_embed_dim=gte_cfg_raw.get("cluster_embed_dim", 32),
            dropout=gte_cfg_raw.get("dropout", 0.2),
        )

    # HMM group-to-mode mapping
    hmm_gm_map = model_config.get("hmm_group_mode_map", {})
    if hmm_gm_map:
        ple_config.hmm_group_mode_map = {str(k): str(v) for k, v in hmm_gm_map.items()}

    # HMM projectors enabled/disabled
    if model_config.get("_disable_hmm_projectors"):
        ple_config.hmm_projectors_enabled = False

    # Multidisciplinary routing
    md_routing = model_config.get("multidisciplinary_routing", {})
    if md_routing:
        ple_config.multidisciplinary_routing = {str(k): list(v) for k, v in md_routing.items()}

    # Task tower dims
    default_tower_dims = tower_cfg.get("default_dims", [expert_output, expert_output // 2])
    ple_config.task_tower.default_dims = default_tower_dims

    return ple_config


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
    ):
        self._hp = hp_overrides or {}

        # -- Resolve device --
        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        # -- Load pipeline config --
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
            logger.warning("label_schema.json not found next to feature_schema.json; "
                           "falling back to pipeline.yaml")
            # Build label_schema from pipeline.yaml — must include model config,
            # task_relationships, task_groups, etc. for correct PLEConfig reconstruction
            self._label_schema = {
                "tasks": self._pipeline_config.get("tasks", []),
                "model": self._pipeline_config.get("model", {}),
                "task_relationships": self._pipeline_config.get("task_relationships", []),
                "task_groups": self._pipeline_config.get("task_groups", []),
                "task_group_map": self._pipeline_config.get("task_group_map", {}),
                "logit_transfer_strength": self._pipeline_config.get("logit_transfer_strength", 0.5),
                "adatt": self._pipeline_config.get("adatt", {}),
            }

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

        # -- Build PLEConfig --
        self._ple_config = _build_ple_config(
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
