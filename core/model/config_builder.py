"""
PLEConfig builder — single source of truth for model configuration.

Constructs a PLEConfig from pipeline.yaml, feature_schema.json,
and label_schema (tasks, model config, task_relationships, etc.).
Both train.py and PLEPredictor use this function to ensure identical
model architecture regardless of entry point.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


def parse_bool_hp(val: Any) -> bool:
    """Parse a hyperparameter value that may be bool, str, or int."""
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return json.loads(val)
    return bool(val)


def build_ple_config(
    config: dict,
    feature_schema: dict,
    label_schema: dict,
    input_dim: int,
    hp: dict,
) -> "PLEConfig":
    """Build a PLEConfig from pipeline config, feature schema, and label schema.

    This is the single source of truth for model configuration.
    All callers (train.py, PLEPredictor, eval_entry.py) must use this
    function to ensure identical model architecture.

    Args:
        config: Pipeline YAML config (full dict).
        feature_schema: Feature schema JSON (columns, group_ranges, expert_routing).
        label_schema: Label schema with tasks, model config, task_relationships.
            When label_schema.json is absent, built from pipeline.yaml sections.
        input_dim: Number of input features.
        hp: Hyperparameter overrides (from SageMaker HPs or CLI args).

    Returns:
        Fully configured PLEConfig.
    """
    from core.model.ple.config import (
        PLEConfig, ExpertConfig, ExpertBasketConfig,
        LossWeightingConfig, LogitTransferDef,
        GroupTaskExpertConfig, AdaTTConfig, AdaTTSPConfig,
        ResidualRecoveryConfig, ECEBConfig, TaskGroupDef,
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

    # Expert basket from HP override or model config
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

    # Expert ablation: active/removed
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

    # --- Structure ablation HP toggles ---
    use_ple_raw = hp.get("use_ple")
    if use_ple_raw is not None and not parse_bool_hp(use_ple_raw):
        num_extraction_layers = 1

    use_adatt_raw = hp.get("use_adatt")
    if use_adatt_raw is not None and not parse_bool_hp(use_adatt_raw):
        model_config["adatt"] = {"enabled": False}

    use_cgc_gate_raw = hp.get("use_cgc_gate")
    if use_cgc_gate_raw is not None and not parse_bool_hp(use_cgc_gate_raw):
        model_config.setdefault("cgc", {})["enabled"] = False

    use_gte_raw = hp.get("use_group_task_expert")
    if use_gte_raw is not None and not parse_bool_hp(use_gte_raw):
        model_config.setdefault("group_task_expert", {})["enabled"] = False

    use_lt_raw = hp.get("use_logit_transfer")
    if use_lt_raw is not None and not parse_bool_hp(use_lt_raw):
        label_schema["task_relationships"] = []

    use_hmm_raw = hp.get("use_hmm_projectors")
    if use_hmm_raw is not None and not parse_bool_hp(use_hmm_raw):
        model_config["_disable_hmm_projectors"] = True

    # AdaTT-sp (Li 2023 representation-level fusion at the CGC gate).
    # Off by default; enable via HP ``use_adatt_sp=true`` in ablation scenarios.
    use_adatt_sp_raw = hp.get("use_adatt_sp")
    if use_adatt_sp_raw is not None:
        model_config.setdefault("adatt_sp", {})["enabled"] = bool(
            parse_bool_hp(use_adatt_sp_raw)
        )

    # Residual recovery (Paper 3 intra-task recovery at the CGC gate).
    # Off by default; enable via HP ``use_residual_recovery=true`` and select
    # method via ``residual_method`` ("complement" / "orthogonal" / "dualgate").
    use_rr_raw = hp.get("use_residual_recovery")
    if use_rr_raw is not None:
        model_config.setdefault("residual_recovery", {})["enabled"] = bool(
            parse_bool_hp(use_rr_raw)
        )
    rr_method_raw = hp.get("residual_method")
    if rr_method_raw is not None:
        model_config.setdefault("residual_recovery", {})["method"] = str(rr_method_raw)

    # ECEB (Paper 3 MV: uncertainty-conditioned recovery at the CGC gate).
    # Off by default; enable via HP ``use_eceb=true`` in ablation scenarios.
    use_eceb_raw = hp.get("use_eceb")
    if use_eceb_raw is not None:
        model_config.setdefault("eceb", {})["enabled"] = bool(
            parse_bool_hp(use_eceb_raw)
        )

    # --- Loss weighting ---
    lw_cfg = model_config.get("loss_weighting", {})
    loss_weighting = LossWeightingConfig(
        strategy=lw_cfg.get("strategy", "fixed"),
        gradnorm_alpha=lw_cfg.get("gradnorm_alpha", 1.5),
        gradnorm_interval=lw_cfg.get("gradnorm_interval", 1),
        dwa_temperature=lw_cfg.get("dwa_temperature", 2.0),
        dwa_window_size=lw_cfg.get("dwa_window_size", 5),
    )

    # --- Build PLEConfig ---
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
    ple_config.gate_type = hp.get("gate_type", "softmax")

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

    # Task group map
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

    # AdaTT-sp (representation-level fusion, Li et al. KDD 2023).
    # Separate from the loss-level AdaTTConfig above; only the CGC gate
    # layer's fusion behaviour is affected.
    adatt_sp_raw = (
        model_config.get("adatt_sp")
        or label_schema.get("adatt_sp")
        or config.get("adatt_sp")
        or {}
    )
    ple_config.adatt_sp = AdaTTSPConfig(
        enabled=bool(adatt_sp_raw.get("enabled", False)),
        native_residual_weight_init=float(
            adatt_sp_raw.get("native_residual_weight_init", 1.0)
        ),
    )

    # Residual recovery (Paper 3 intra-task mechanisms: complement / orthogonal / dualgate).
    rr_raw = (
        model_config.get("residual_recovery")
        or label_schema.get("residual_recovery")
        or config.get("residual_recovery")
        or {}
    )
    ple_config.residual_recovery = ResidualRecoveryConfig(
        enabled=bool(rr_raw.get("enabled", False)),
        method=str(rr_raw.get("method", "complement")),
        weight_init=float(rr_raw.get("weight_init", 0.5)),
    )

    # ECEB (Paper 3 MV: uncertainty-conditioned recovery at the CGC gate).
    eceb_raw = (
        model_config.get("eceb")
        or label_schema.get("eceb")
        or config.get("eceb")
        or {}
    )
    ple_config.eceb = ECEBConfig(
        enabled=bool(eceb_raw.get("enabled", False)),
        uncertainty_source=str(eceb_raw.get("uncertainty_source", "gate_entropy")),
        recovery_source=str(eceb_raw.get("recovery_source", "uniform")),
        weight_init=float(eceb_raw.get("weight_init", 0.0)),
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

    # HMM projectors
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


def build_label_schema_from_config(pipeline_config: dict) -> dict:
    """Build a label_schema dict from pipeline.yaml when label_schema.json is absent.

    Extracts tasks, model config, task_relationships, task_groups, etc.
    from the pipeline config so that build_ple_config() receives all
    the information it needs.
    """
    return {
        "tasks": pipeline_config.get("tasks", []),
        "model": pipeline_config.get("model", {}),
        "task_relationships": pipeline_config.get("task_relationships", []),
        "task_groups": pipeline_config.get("task_groups", []),
        "task_group_map": pipeline_config.get("task_group_map", {}),
        "logit_transfer_strength": pipeline_config.get("logit_transfer_strength", 0.5),
        "adatt": pipeline_config.get("adatt", {}),
    }
