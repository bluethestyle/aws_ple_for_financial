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

ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = str(ROOT / "containers" / "training" / "train.py")
PHASE0_DIR = str(ROOT / "outputs" / "phase0")
RESULTS_DIR = ROOT / "outputs" / "ablation_results"
CONFIG = "configs/santander/pipeline.yaml"

# Base hyperparameters (from pipeline.yaml ablation.training_defaults)
BASE_HP = {
    "config": CONFIG,
    "epochs": 3,
    "batch_size": 4096,
    "learning_rate": 0.001,
    "seed": 42,
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
# Phase 2: Expert Ablation (8 key scenarios, reduced from 16)
# ============================================================
EXPERT_SCENARIOS = {
    "deepfm_only": {"active_experts": '["deepfm"]'},
    "deepfm+temporal": {"active_experts": '["deepfm","temporal_ensemble"]'},
    "deepfm+perslay": {"active_experts": '["deepfm","perslay"]'},
    "deepfm+lightgcn": {"active_experts": '["deepfm","lightgcn"]'},
    "full_basket": {},  # all experts (same as full baseline)
    "full-temporal": {"removed_experts": '["temporal_ensemble"]'},
    "full-perslay": {"removed_experts": '["perslay"]'},
    "full-lightgcn": {"removed_experts": '["lightgcn"]'},
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


def run_scenario(name: str, extra_hp: dict) -> dict:
    """Run a single training scenario and return metrics."""
    out_dir = RESULTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    hp = {**BASE_HP, **extra_hp, "ablation_scenario": name}
    env = {
        **os.environ,
        "SM_CHANNEL_TRAIN": PHASE0_DIR,
        "SM_OUTPUT_DATA_DIR": str(out_dir),
        "SM_MODEL_DIR": str(out_dir / "model"),
        "SM_HPS": json.dumps(hp),
        "PYTHONPATH": str(ROOT),
    }

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, TRAIN_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
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
