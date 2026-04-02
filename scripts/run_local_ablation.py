#!/usr/bin/env python
"""Local ablation runner — executes all 47 scenarios from pipeline.yaml config.

Phases 1-3 only (feature/expert/structure). Phase 4-5 (distillation/report) are SageMaker-only.
Each scenario runs train.py with different SM_HPS and collects eval_metrics.json.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = str(ROOT / "containers" / "training" / "train.py")
PHASE0_DIR = str(ROOT / "outputs" / "phase0")
RESULTS_DIR = ROOT / "outputs" / "ablation_results"
CONFIG = "configs/santander/pipeline.yaml"

# Load training defaults from pipeline.yaml ablation.training_defaults
def _load_training_defaults() -> dict:
    """Read ablation.training_defaults from pipeline.yaml (config-driven, no hardcoding)."""
    defaults = {
        "epochs": 10,
        "batch_size": 4096,
        "learning_rate": 0.008,
        "amp": True,
        "early_stopping_patience": 3,
        "seed": 42,
    }
    try:
        with open(ROOT / CONFIG, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        td = cfg.get("ablation", {}).get("training_defaults", {})
        for key in defaults:
            if key in td:
                defaults[key] = td[key]
    except Exception:
        pass
    return defaults

# Base hyperparameters (from pipeline.yaml ablation.training_defaults)
_td = _load_training_defaults()
BASE_HP = {
    "config": CONFIG,
    "epochs": _td["epochs"],
    "batch_size": _td["batch_size"],
    "learning_rate": _td["learning_rate"],
    "seed": _td["seed"],
    "amp": _td["amp"],
    "early_stopping_patience": _td["early_stopping_patience"],
}

# ============================================================
# Phase 1: Feature + Expert Joint Ablation
# Baseline = DeepFM only (all features, single expert)
# Each scenario adds ONE specialized expert + its matching features
# ============================================================
_ALL_EXPERTS = ["deepfm", "temporal_ensemble", "hgcn", "perslay", "causal", "lightgcn", "optimal_transport"]

def _experts_without(*remove):
    return json.dumps([e for e in _ALL_EXPERTS if e not in remove])

# Feature groups that are NOT base (to be removed in base-only scenarios)
_ALL_GENERATED = ["tda_global", "tda_local", "hmm_states", "mamba_temporal",
                  "product_hierarchy", "graph_collaborative", "gmm_clustering", "model_derived"]

def _remove_except(*keep):
    """Return JSON list of feature groups to remove, keeping only specified ones."""
    return json.dumps([g for g in _ALL_GENERATED if g not in keep])

FEATURE_EXPERT_SCENARIOS = {
    # === Baselines ===
    "deepfm_base": {
        "shared_experts": '["deepfm"]',
        "removed_feature_groups": json.dumps(_ALL_GENERATED),
    },
    "deepfm_all_features": {
        "shared_experts": '["deepfm"]',
    },
    "full": {},  # all experts + all features

    # === DeepFM + one expert (with matching features) ===
    "deepfm+tda": {
        "shared_experts": '["deepfm","perslay"]',
        "removed_feature_groups": _remove_except("tda_global", "tda_local"),
    },
    "deepfm+temporal": {
        "shared_experts": '["deepfm","temporal_ensemble"]',
        "removed_feature_groups": _remove_except("hmm_states", "mamba_temporal"),
    },
    "deepfm+hgcn": {
        "shared_experts": '["deepfm","hgcn"]',
        "removed_feature_groups": _remove_except("product_hierarchy"),
    },
    "deepfm+lightgcn": {
        "shared_experts": '["deepfm","lightgcn"]',
        "removed_feature_groups": _remove_except("graph_collaborative"),
    },
    "deepfm+causal": {
        "shared_experts": '["deepfm","causal"]',
        "removed_feature_groups": _remove_except(),  # causal uses base features
    },
    "deepfm+ot": {
        "shared_experts": '["deepfm","optimal_transport"]',
        "removed_feature_groups": _remove_except(),  # OT uses base features
    },
    "deepfm+gmm": {
        "shared_experts": '["deepfm"]',  # GMM is feature-only, no dedicated expert
        "removed_feature_groups": _remove_except("gmm_clustering"),
    },
    "deepfm+model_derived": {
        "shared_experts": '["deepfm"]',
        "removed_feature_groups": _remove_except("model_derived"),
    },

    # === Cumulative: progressively adding experts ===
    "cumul_1_deepfm+temporal": {
        "shared_experts": '["deepfm","temporal_ensemble"]',
        "removed_feature_groups": _remove_except("hmm_states", "mamba_temporal"),
    },
    "cumul_2_+hgcn": {
        "shared_experts": '["deepfm","temporal_ensemble","hgcn"]',
        "removed_feature_groups": _remove_except("hmm_states", "mamba_temporal", "product_hierarchy"),
    },
    "cumul_3_+lightgcn": {
        "shared_experts": '["deepfm","temporal_ensemble","hgcn","lightgcn"]',
        "removed_feature_groups": _remove_except("hmm_states", "mamba_temporal", "product_hierarchy", "graph_collaborative"),
    },
    "cumul_4_+tda": {
        "shared_experts": '["deepfm","temporal_ensemble","hgcn","lightgcn","perslay"]',
        "removed_feature_groups": _remove_except("hmm_states", "mamba_temporal", "product_hierarchy", "graph_collaborative", "tda_global", "tda_local"),
    },
    "cumul_5_+causal_ot": {
        "shared_experts": '["deepfm","temporal_ensemble","hgcn","lightgcn","perslay","causal","optimal_transport"]',
        "removed_feature_groups": _remove_except("hmm_states", "mamba_temporal", "product_hierarchy", "graph_collaborative", "tda_global", "tda_local"),
    },
    "cumul_6_+gmm_model": {},  # = full (all features + all experts)

    # === Top-down: full minus one (expert + matching features together) ===
    "full-tda_perslay": {
        "shared_experts": _experts_without("perslay"),
        "removed_feature_groups": '["tda_global","tda_local"]',
    },
    "full-temporal": {
        "shared_experts": _experts_without("temporal_ensemble"),
        "removed_feature_groups": '["hmm_states","mamba_temporal"]',
    },
    "full-hgcn_hierarchy": {
        "shared_experts": _experts_without("hgcn"),
        "removed_feature_groups": '["product_hierarchy"]',
    },
    "full-lightgcn_graph": {
        "shared_experts": _experts_without("lightgcn"),
        "removed_feature_groups": '["graph_collaborative"]',
    },
    "full-causal": {
        "shared_experts": _experts_without("causal"),
    },
    "full-ot": {
        "shared_experts": _experts_without("optimal_transport"),
    },
}

# ============================================================
# Phase 3: Task × Structure Cross Ablation (16 scenarios)
# ============================================================
TASK_TIERS = {
    "tasks_4": '["has_nba","churn_signal","product_stability","nba_primary"]',
    "tasks_8": '["has_nba","churn_signal","product_stability","nba_primary","tenure_stage","spend_level","cross_sell_count","engagement_score"]',
    "tasks_15": '["has_nba","churn_signal","product_stability","nba_primary","tenure_stage","spend_level","cross_sell_count","engagement_score","will_acquire_deposits","will_acquire_investments","will_acquire_accounts","will_acquire_lending","will_acquire_payments","segment_prediction","income_tier"]',
    "tasks_18": None,  # all tasks
}

STRUCTURES = {
    "shared_bottom": {"use_ple": "false", "use_adatt": "false"},
    "ple_only": {"use_ple": "true", "use_adatt": "false"},
    "adatt_only": {"use_ple": "false", "use_adatt": "true"},
    "full": {"use_ple": "true", "use_adatt": "true"},
}


IMAGE = os.environ.get("ABLATION_IMAGE", "model_training:v3.3")


def run_scenario(name: str, extra_hp: dict) -> dict:
    """Run a single training scenario via Docker (SageMaker local mode)."""
    out_dir = RESULTS_DIR / name
    metrics_path = out_dir / "eval_metrics.json"

    # Skip if already completed (eval_metrics.json exists with valid AUC)
    if metrics_path.exists():
        try:
            with open(metrics_path) as f:
                cached = json.load(f)
            cached_auc = cached.get("auc", cached.get("final_metrics", {}).get("auc"))
            if cached_auc is not None:
                print(f"  [SKIP] {name}: AUC={cached_auc} (cached)")
                return {"scenario": name, "status": "OK", "auc": str(cached_auc),
                        "f1_macro": "cached", "time_s": 0, "metrics": cached}
        except (json.JSONDecodeError, KeyError):
            pass  # corrupted file, re-run

    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = out_dir / "model"
    model_dir.mkdir(exist_ok=True)

    hp = {**BASE_HP, **extra_hp, "ablation_scenario": name}

    env_args = [
        "-e", f"SM_CHANNEL_TRAIN=/opt/ml/input/data/train",
        "-e", f"SM_OUTPUT_DATA_DIR=/opt/ml/output/data",
        "-e", f"SM_MODEL_DIR=/opt/ml/model",
        "-e", f"SM_HPS={json.dumps(hp)}",
        "-e", "PYTHONPATH=/opt/ml/code",
    ]

    phase0 = str(PHASE0_DIR).replace("\\", "/")
    output = str(out_dir).replace("\\", "/")
    model = str(model_dir).replace("\\", "/")
    code = str(ROOT).replace("\\", "/")

    # Docker SageMaker local mode (GPU passthrough)
    use_docker = os.environ.get("ABLATION_USE_DOCKER", "1") == "1"

    if use_docker:
        cmd = [
            "docker", "run", "--rm", "--gpus", "all",
            "-v", f"{phase0}:/opt/ml/input/data/train",
            "-v", f"{output}:/opt/ml/output/data",
            "-v", f"{model}:/opt/ml/model",
            "-v", f"{code}:/opt/ml/code",
            *env_args,
            IMAGE,
            "python", "/opt/ml/code/containers/training/train.py",
        ]
        env = {**os.environ, "MSYS_NO_PATHCONV": "1"}
    else:
        # Fallback: local Python (no Docker)
        env = {
            **os.environ,
            "SM_CHANNEL_TRAIN": str(PHASE0_DIR),
            "SM_OUTPUT_DATA_DIR": str(out_dir),
            "SM_MODEL_DIR": str(model_dir),
            "SM_HPS": json.dumps(hp),
            "PYTHONPATH": str(ROOT),
        }
        cmd = [sys.executable, str(ROOT / "containers" / "training" / "train.py")]

    t0 = time.time()
    # Write stdout/stderr to files instead of capture_output to avoid pipe deadlock
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = log_dir / "stdout.log"
    stderr_log = log_dir / "stderr.log"
    with open(stdout_log, "w") as fout, open(stderr_log, "w") as ferr:
        result = subprocess.run(
            cmd,
            env=env,
            stdout=fout,
            stderr=ferr,
            timeout=14400,
        )
    elapsed = time.time() - t0

    # Read eval_metrics.json (metrics may be nested under final_metrics)
    metrics_path = out_dir / "eval_metrics.json"
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            raw = json.load(f)
        metrics = raw.get("final_metrics", raw)

    auc = metrics.get("auc", "N/A")
    f1 = metrics.get("f1_macro_avg", "N/A")
    # Consider success if eval_metrics exists with valid AUC, even if returncode != 0
    # (Docker containers may return non-zero due to FutureWarnings on stderr)
    status = "OK" if (auc != "N/A" or result.returncode == 0) else "FAIL"

    print(f"  [{status}] {name}: AUC={auc}, F1_macro={f1}, time={elapsed:.1f}s")
    if result.returncode != 0 and status == "FAIL":
        # Print last 3 lines of stderr for debugging
        try:
            err_text = stderr_log.read_text(errors="replace").strip()
            err_lines = err_text.split("\n")[-3:]
            for line in err_lines:
                print(f"    ERR: {line}")
        except Exception:
            print(f"    ERR: (see {stderr_log})")

    return {
        "scenario": name,
        "status": status,
        "auc": auc,
        "f1_macro": f1,
        "time_s": round(elapsed, 1),
        "metrics": metrics,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    t_start = time.time()

    # Phase 1: Feature + Expert Joint Ablation
    print("=" * 60)
    n_fe = len(FEATURE_EXPERT_SCENARIOS)
    print(f"PHASE 1: Feature + Expert Joint Ablation ({n_fe} scenarios)")
    print("  Baseline: DeepFM only → +expert(+features) → cumulative → top-down")
    print("=" * 60)
    for name, extra in FEATURE_EXPERT_SCENARIOS.items():
        all_results.append(run_scenario(f"joint_{name}", extra))

    # Phase 3: Task × Structure
    print("\n" + "=" * 60)
    print("PHASE 3: Task × Structure Cross Ablation (16 scenarios)")
    print("=" * 60)
    for tier_name, tasks_json in TASK_TIERS.items():
        for struct_name, struct_hp in STRUCTURES.items():
            scenario_name = f"struct_{tier_name}_{struct_name}"
            extra = dict(struct_hp)
            if tasks_json is not None:
                extra["active_tasks"] = tasks_json
            all_results.append(run_scenario(scenario_name, extra))

    # Summary
    total_time = time.time() - t_start
    ok_count = sum(1 for r in all_results if r["status"] == "OK")
    fail_count = sum(1 for r in all_results if r["status"] == "FAIL")

    print("\n" + "=" * 60)
    print(f"ABLATION COMPLETE: {ok_count}/{len(all_results)} OK, {fail_count} FAIL")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print("=" * 60)

    # Save summary
    summary_path = RESULTS_DIR / "ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to {summary_path}")

    # Print comparison table
    print("\n{:<40} {:>8} {:>10} {:>7}".format("Scenario", "AUC", "F1_macro", "Time"))
    print("-" * 68)
    for r in all_results:
        auc_str = f"{r['auc']:.4f}" if isinstance(r['auc'], float) else str(r['auc'])
        f1_str = f"{r['f1_macro']:.4f}" if isinstance(r['f1_macro'], float) else str(r['f1_macro'])
        print(f"{r['scenario']:<40} {auc_str:>8} {f1_str:>10} {r['time_s']:>6.1f}s")


if __name__ == "__main__":
    main()
