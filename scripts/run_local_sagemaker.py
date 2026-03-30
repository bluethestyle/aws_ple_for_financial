#!/usr/bin/env python
"""Run training via SageMaker Local Mode (Docker + GPU).

Simulates the exact SageMaker environment locally before cloud submission.
Uses the pre-built model_training Docker image with GPU support.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PHASE0_DIR = ROOT / "outputs" / "phase0"
OUTPUT_DIR = ROOT / "outputs" / "sagemaker_local"
CONFIG = "configs/santander/pipeline.yaml"
IMAGE = "model_training:v3.3"


def run_sagemaker_local(
    scenario_name: str = "baseline",
    extra_hp: dict = None,
    epochs: int = 3,
    batch_size: int = 4096,
    amp: bool = True,
):
    """Run a single training job via Docker (SageMaker-compatible paths)."""
    extra_hp = extra_hp or {}
    out_dir = OUTPUT_DIR / scenario_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = out_dir / "model"
    model_dir.mkdir(exist_ok=True)

    hp = {
        "config": CONFIG,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": 0.001,
        "seed": 42,
        "amp": amp,
        "ablation_scenario": scenario_name,
        **extra_hp,
    }

    # SageMaker environment variables
    env = {
        "SM_CHANNEL_TRAIN": "/opt/ml/input/data/train",
        "SM_OUTPUT_DATA_DIR": "/opt/ml/output/data",
        "SM_MODEL_DIR": "/opt/ml/model",
        "SM_HPS": json.dumps(hp),
        "PYTHONPATH": "/opt/ml/code",
    }

    env_args = []
    for k, v in env.items():
        env_args += ["-e", f"{k}={v}"]

    # Convert Windows paths to Docker-compatible paths
    phase0_path = str(PHASE0_DIR).replace("\\", "/")
    output_path = str(out_dir).replace("\\", "/")
    model_path = str(model_dir).replace("\\", "/")
    code_path = str(ROOT).replace("\\", "/")

    cmd = [
        "docker", "run", "--rm",
        "--gpus", "all",
        "-v", f"{phase0_path}:/opt/ml/input/data/train",
        "-v", f"{output_path}:/opt/ml/output/data",
        "-v", f"{model_path}:/opt/ml/model",
        "-v", f"{code_path}:/opt/ml/code",
        *env_args,
        IMAGE,
        "python", "/opt/ml/code/containers/training/train.py",
    ]

    print(f"[{scenario_name}] Starting Docker training...")
    print(f"  Image: {IMAGE}")
    print(f"  HP: {json.dumps(hp, indent=2)[:200]}...")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  [FAIL] {elapsed:.1f}s")
        err_lines = result.stderr.strip().split("\n")[-5:]
        for line in err_lines:
            print(f"    ERR: {line}")
        return None

    # Read metrics
    metrics_path = out_dir / "eval_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            raw = json.load(f)
        fm = raw.get("final_metrics", raw)
        auc = fm.get("auc", "N/A")
        f1 = fm.get("f1_macro_avg", "N/A")
        print(f"  [OK] AUC={auc}, F1={f1}, time={elapsed:.1f}s")
        return fm
    else:
        print(f"  [OK] No metrics file, time={elapsed:.1f}s")
        return {}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Quick smoke test: 1 epoch baseline
    print("=" * 60)
    print("SageMaker Local Mode Test")
    print("=" * 60)

    fm = run_sagemaker_local(
        scenario_name="baseline",
        epochs=3,
        batch_size=4096,
        amp=True,
    )

    if fm is not None:
        print("\nBaseline complete. Ready for full ablation.")
    else:
        print("\nBaseline FAILED. Check Docker/GPU setup.")


if __name__ == "__main__":
    main()
