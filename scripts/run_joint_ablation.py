"""
Joint ablation runner: 15 scenarios, PLE softmax (default), 10 epochs.
Runs train.py directly as subprocess — no orchestrator, no orphan issues.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Prevent orchestrator from grabbing GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Windows sleep prevention
if sys.platform == "win32":
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001 | 0x00000040)

PHASE0 = "outputs/phase0_v12"
OUTDIR = "outputs/ablation_v12"
CONFIG = "configs/santander/pipeline.yaml"

BASE_HPS = {
    "config": CONFIG,
    "epochs": 10,
    "batch_size": 5632,
    "learning_rate": 0.0005,
    "seed": 42,
    "amp": True,
    "early_stopping_patience": 10,
    "warmup_epochs": 3,
    "num_workers": 0,
    "use_adatt": "false",  # PLE softmax alone is best; adaTT degrades
}

SCENARIOS = [
    # Baselines
    {"name": "joint_full", "hp": {}},
    {"name": "joint_base_only", "hp": {
        "removed_feature_groups": json.dumps([
            "tda_global", "tda_local", "hmm_states", "mamba_temporal",
            "product_hierarchy", "merchant_hierarchy", "graph_collaborative",
            "gmm_clustering", "model_derived", "txn_behavior", "derived_temporal",
        ]),
    }},
    # Bottom-up: DeepFM + one expert
    {"name": "joint_deepfm_base", "hp": {"shared_experts": json.dumps(["deepfm"])}},
    {"name": "joint_deepfm+temporal", "hp": {"shared_experts": json.dumps(["deepfm", "temporal_ensemble"])}},
    {"name": "joint_deepfm+hgcn", "hp": {"shared_experts": json.dumps(["deepfm", "hgcn"])}},
    {"name": "joint_deepfm+tda", "hp": {"shared_experts": json.dumps(["deepfm", "perslay"])}},
    {"name": "joint_deepfm+lightgcn", "hp": {"shared_experts": json.dumps(["deepfm", "lightgcn"])}},
    {"name": "joint_deepfm+causal", "hp": {"shared_experts": json.dumps(["deepfm", "causal"])}},
    {"name": "joint_deepfm+ot", "hp": {"shared_experts": json.dumps(["deepfm", "optimal_transport"])}},
    # Top-down: Full minus one expert
    {"name": "joint_full-temporal", "hp": {"removed_experts": json.dumps(["temporal_ensemble"])}},
    {"name": "joint_full-hgcn", "hp": {"removed_experts": json.dumps(["hgcn"])}},
    {"name": "joint_full-tda", "hp": {"removed_experts": json.dumps(["perslay"])}},
    {"name": "joint_full-lightgcn", "hp": {"removed_experts": json.dumps(["lightgcn"])}},
    {"name": "joint_full-causal", "hp": {"removed_experts": json.dumps(["causal"])}},
    {"name": "joint_full-ot", "hp": {"removed_experts": json.dumps(["optimal_transport"])}},
]


def run_scenario(scenario):
    name = scenario["name"]
    out_dir = Path(OUTDIR) / name
    model_path = out_dir / "model" / "model.pth"

    if model_path.exists():
        print(f"[SKIP] {name}: already completed")
        return

    (out_dir / "model").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    hp = dict(BASE_HPS)
    hp.update(scenario.get("hp", {}))

    env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(Path.cwd())
    env["SM_CHANNEL_TRAIN"] = PHASE0
    env["SM_OUTPUT_DATA_DIR"] = str(out_dir)
    env["SM_MODEL_DIR"] = str(out_dir / "model")
    env["SM_HPS"] = json.dumps(hp)
    env["PYTHONUNBUFFERED"] = "1"

    stdout_path = out_dir / "logs" / "stdout.log"
    stderr_path = out_dir / "logs" / "stderr.log"

    start = time.time()
    print(f"[START] {name} ({time.strftime('%H:%M:%S')})")

    with open(stdout_path, "w") as fout, open(stderr_path, "w") as ferr:
        proc = subprocess.run(
            [sys.executable, "-u", "containers/training/train.py"],
            env=env,
            stdout=fout,
            stderr=ferr,
        )

    elapsed = time.time() - start
    status = "OK" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
    print(f"[{status}] {name} ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    print("=" * 60)
    print(f"Joint Ablation: {len(SCENARIOS)} scenarios x 10 epochs (PLE softmax)")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    for i, s in enumerate(SCENARIOS):
        run_scenario(s)
        # Brief pause between scenarios for GPU cleanup
        if i < len(SCENARIOS) - 1:
            time.sleep(5)

    print("=" * 60)
    print(f"Joint Ablation COMPLETE: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
