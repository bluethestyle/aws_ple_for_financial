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

# Load batch_size from pipeline.yaml ablation.training_defaults (fallback 4096)
def _load_batch_size() -> int:
    try:
        with open(ROOT / CONFIG, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("ablation", {}).get("training_defaults", {}).get("batch_size", 4096)
    except Exception:
        return 4096

# Base hyperparameters (from pipeline.yaml ablation.training_defaults)
BASE_HP = {
    "config": CONFIG,
    "epochs": 3,
    "batch_size": _load_batch_size(),
    "learning_rate": 0.001,
    "seed": 42,
    "amp": True,
    "early_stopping_patience": 3,
}

# ============================================================
# Phase 1: Feature Group Ablation (16 scenarios)
# ============================================================
FEATURE_SCENARIOS = {
    # Baseline
    "full": {},
    "base_only": {"removed_feature_groups": '["tda_global","tda_local","hmm_states","mamba_temporal","product_hierarchy","graph_collaborative","gmm_clustering","model_derived"]'},
    # Bottom-up: base + one group
    "base+tda": {"removed_feature_groups": '["hmm_states","mamba_temporal","product_hierarchy","graph_collaborative","gmm_clustering","model_derived"]'},
    "base+hmm": {"removed_feature_groups": '["tda_global","tda_local","mamba_temporal","product_hierarchy","graph_collaborative","gmm_clustering","model_derived"]'},
    "base+mamba": {"removed_feature_groups": '["tda_global","tda_local","hmm_states","product_hierarchy","graph_collaborative","gmm_clustering","model_derived"]'},
    "base+graph": {"removed_feature_groups": '["tda_global","tda_local","hmm_states","mamba_temporal","product_hierarchy","gmm_clustering","model_derived"]'},
    "base+hierarchy": {"removed_feature_groups": '["tda_global","tda_local","hmm_states","mamba_temporal","graph_collaborative","gmm_clustering","model_derived"]'},
    "base+gmm": {"removed_feature_groups": '["tda_global","tda_local","hmm_states","mamba_temporal","product_hierarchy","graph_collaborative","model_derived"]'},
    "base+model_derived": {"removed_feature_groups": '["tda_global","tda_local","hmm_states","mamba_temporal","product_hierarchy","graph_collaborative","gmm_clustering"]'},
    # Top-down: full minus one group
    "full-tda": {"removed_feature_groups": '["tda_global","tda_local"]'},
    "full-hmm": {"removed_feature_groups": '["hmm_states"]'},
    "full-mamba": {"removed_feature_groups": '["mamba_temporal"]'},
    "full-graph": {"removed_feature_groups": '["graph_collaborative"]'},
    "full-hierarchy": {"removed_feature_groups": '["product_hierarchy"]'},
    "full-gmm": {"removed_feature_groups": '["gmm_clustering"]'},
    "full-model_derived": {"removed_feature_groups": '["model_derived"]'},
}

# ============================================================
# Phase 2: Expert Ablation (16 scenarios — full design)
# ============================================================
_ALL_EXPERTS = ["deepfm", "temporal_ensemble", "hgcn", "perslay", "causal", "lightgcn", "optimal_transport"]

def _experts_without(*remove):
    return json.dumps([e for e in _ALL_EXPERTS if e not in remove])

EXPERT_SCENARIOS = {
    # Bottom-up: deepfm + one expert
    "deepfm_only": {"shared_experts": '["deepfm"]'},
    "deepfm+temporal": {"shared_experts": '["deepfm","temporal_ensemble"]'},
    "deepfm+hgcn": {"shared_experts": '["deepfm","hgcn"]'},
    "deepfm+perslay": {"shared_experts": '["deepfm","perslay"]'},
    "deepfm+causal": {"shared_experts": '["deepfm","causal"]'},
    "deepfm+lightgcn": {"shared_experts": '["deepfm","lightgcn"]'},
    "deepfm+ot": {"shared_experts": '["deepfm","optimal_transport"]'},
    # Full basket
    "full_basket": {},
    # Top-down: full minus one expert
    "full-deepfm": {"shared_experts": _experts_without("deepfm")},
    "full-temporal": {"shared_experts": _experts_without("temporal_ensemble")},
    "full-hgcn": {"shared_experts": _experts_without("hgcn")},
    "full-perslay": {"shared_experts": _experts_without("perslay")},
    "full-causal": {"shared_experts": _experts_without("causal")},
    "full-lightgcn": {"shared_experts": _experts_without("lightgcn")},
    "full-ot": {"shared_experts": _experts_without("optimal_transport")},
    # Minimal baseline
    "mlp_only": {"shared_experts": '["mlp"]'},
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

    # Use local Python with GPU (Docker GPU passthrough unreliable on WSL)
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
    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        errors="replace",
    )
    elapsed = time.time() - t0

    # Read eval_metrics.json (metrics may be nested under final_metrics)
    metrics_path = out_dir / "eval_metrics.json"
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            raw = json.load(f)
        metrics = raw.get("final_metrics", raw)

    status = "OK" if result.returncode == 0 else "FAIL"
    auc = metrics.get("auc", "N/A")
    f1 = metrics.get("f1_macro_avg", "N/A")

    print(f"  [{status}] {name}: AUC={auc}, F1_macro={f1}, time={elapsed:.1f}s")
    if result.returncode != 0:
        # Print last 3 lines of stderr for debugging
        err_lines = result.stderr.strip().split("\n")[-3:]
        for line in err_lines:
            print(f"    ERR: {line}")

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

    # Phase 1: Feature Ablation
    print("=" * 60)
    print("PHASE 1: Feature Group Ablation (16 scenarios)")
    print("=" * 60)
    for name, extra in FEATURE_SCENARIOS.items():
        hp = {**extra, "ablation_scenario": name}
        all_results.append(run_scenario(f"feat_{name}", hp))

    # Phase 2: Expert Ablation
    print("\n" + "=" * 60)
    print("PHASE 2: Expert Ablation (8 scenarios)")
    print("=" * 60)
    for name, extra in EXPERT_SCENARIOS.items():
        hp = {**extra, "ablation_scenario": name}
        all_results.append(run_scenario(f"expert_{name}", hp))

    # Phase 3: Task × Structure
    print("\n" + "=" * 60)
    print("PHASE 3: Task × Structure Cross Ablation (16 scenarios)")
    print("=" * 60)
    for tier_name, tasks_json in TASK_TIERS.items():
        for struct_name, struct_hp in STRUCTURES.items():
            scenario_name = f"struct_{tier_name}_{struct_name}"
            hp = {**struct_hp}
            if tasks_json is not None:
                hp["active_tasks"] = tasks_json
            all_results.append(run_scenario(scenario_name, hp))

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
