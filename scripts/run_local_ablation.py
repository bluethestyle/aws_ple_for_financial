#!/usr/bin/env python
"""
Local Ablation Runner: 14 tasks, 10 epochs (warmup=3)
Delta measurement only — same epoch count for fair comparison.
No Docker, no SageMaker local mode. Pure local Python.

Usage:
    python scripts/run_local_ablation.py                # full 10-epoch ablation
    python scripts/run_local_ablation.py --dry-run      # 1-epoch verify settings
    python scripts/run_local_ablation.py --scenario NAME # single scenario
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PHASE0 = "outputs/phase0_v3"
RESULTS = "outputs/ablation_v3"
CONFIG = "configs/santander/pipeline.yaml"
EPOCHS = 10
WARMUP = 3
BATCH = 4096
LR = 0.0005
SEED = 42
AMP = True
PATIENCE = EPOCHS  # no early stop — run full epochs for fair delta comparison

BASE_HPS: Dict[str, Any] = {
    "config": CONFIG,
    "epochs": EPOCHS,
    "batch_size": BATCH,
    "learning_rate": LR,
    "seed": SEED,
    "amp": AMP,
    "early_stopping_patience": PATIENCE,
    "warmup_epochs": WARMUP,
    "num_workers": 0,
}

# ---------------------------------------------------------------------------
# Scenario definitions — json.dumps handles escaping correctly
# ---------------------------------------------------------------------------
SCENARIOS: List[Dict[str, Any]] = [
    # === Structure ablation (6) ===
    {"name": "struct_14_shared_bottom",
     "hp": {"use_ple": "false", "use_adatt": "false"}},
    {"name": "struct_14_ple_softmax",
     "hp": {"use_ple": "true", "use_adatt": "false", "gate_type": "softmax"}},
    {"name": "struct_14_ple_sigmoid",
     "hp": {"use_ple": "true", "use_adatt": "false", "gate_type": "sigmoid"}},
    {"name": "struct_14_ple_softmax_adatt",
     "hp": {"use_ple": "true", "use_adatt": "true", "gate_type": "softmax"}},
    {"name": "struct_14_ple_sigmoid_adatt",
     "hp": {"use_ple": "true", "use_adatt": "true", "gate_type": "sigmoid"}},
    {"name": "struct_14_adatt_only",
     "hp": {"use_ple": "false", "use_adatt": "true"}},

    # === Feature + Expert joint ablation ===
    # Full system baseline
    {"name": "joint_full", "hp": {}},
    # Base features only (demographics + product_holdings)
    {"name": "joint_base_only",
     "hp": {"removed_feature_groups": json.dumps([
         "tda_global", "tda_local", "hmm_states", "mamba_temporal",
         "product_hierarchy", "merchant_hierarchy", "graph_collaborative",
         "gmm_clustering", "model_derived", "txn_behavior", "derived_temporal",
     ])}},
    # DeepFM only
    {"name": "joint_deepfm_base",
     "hp": {"shared_experts": json.dumps(["deepfm"])}},
    # DeepFM + each advanced expert
    {"name": "joint_deepfm+temporal",
     "hp": {"shared_experts": json.dumps(["deepfm", "temporal_ensemble"])}},
    {"name": "joint_deepfm+hgcn",
     "hp": {"shared_experts": json.dumps(["deepfm", "hgcn"])}},
    {"name": "joint_deepfm+tda",
     "hp": {"shared_experts": json.dumps(["deepfm", "perslay"])}},
    {"name": "joint_deepfm+lightgcn",
     "hp": {"shared_experts": json.dumps(["deepfm", "lightgcn"])}},
    {"name": "joint_deepfm+causal",
     "hp": {"shared_experts": json.dumps(["deepfm", "causal"])}},
    {"name": "joint_deepfm+ot",
     "hp": {"shared_experts": json.dumps(["deepfm", "optimal_transport"])}},
    # Full minus one expert
    {"name": "joint_full-temporal",
     "hp": {"removed_experts": json.dumps(["temporal_ensemble"])}},
    {"name": "joint_full-hgcn",
     "hp": {"removed_experts": json.dumps(["hgcn"])}},
    {"name": "joint_full-tda",
     "hp": {"removed_experts": json.dumps(["perslay"])}},
    {"name": "joint_full-lightgcn",
     "hp": {"removed_experts": json.dumps(["lightgcn"])}},
    {"name": "joint_full-causal",
     "hp": {"removed_experts": json.dumps(["causal"])}},
    {"name": "joint_full-ot",
     "hp": {"removed_experts": json.dumps(["optimal_transport"])}},
]


def run_scenario(scenario: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
    """Run one ablation scenario."""
    name = scenario["name"]
    out_dir = Path(RESULTS) / name

    if (out_dir / "model" / "model.pth").exists() and not dry_run:
        logger.info("[SKIP] %s: already completed", name)
        return {"name": name, "status": "skipped"}

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model").mkdir(exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)

    hp = dict(BASE_HPS)
    hp.update(scenario.get("hp", {}))
    hp["ablation_scenario"] = name
    if dry_run:
        hp["epochs"] = 1
        hp["early_stopping_patience"] = 1
        hp["warmup_epochs"] = 0

    env = dict(os.environ)
    env["SM_CHANNEL_TRAIN"] = PHASE0
    env["SM_OUTPUT_DATA_DIR"] = str(out_dir)
    env["SM_MODEL_DIR"] = str(out_dir / "model")
    env["SM_HPS"] = json.dumps(hp)
    env["PYTHONPATH"] = str(Path.cwd())

    # Log only scenario-specific HP (not the base ones everyone shares)
    diff_hp = {k: v for k, v in hp.items() if k not in BASE_HPS}
    logger.info("[%s] %s %s", "DRY" if dry_run else "RUN", name, diff_hp or "(baseline)")

    start = time.time()
    stdout_path = out_dir / "logs" / "stdout.log"
    stderr_path = out_dir / "logs" / "stderr.log"

    with open(stdout_path, "w") as fout, open(stderr_path, "w") as ferr:
        proc = subprocess.run(
            [sys.executable, "-u", "containers/training/train.py"],
            env=env, stdout=fout, stderr=ferr,
        )

    elapsed = time.time() - start
    stdout_text = stdout_path.read_text(errors="replace")

    epoch_count = stdout_text.count("val_loss=")
    expected = 1 if dry_run else EPOCHS

    # Extract final metrics
    metrics: Dict[str, float] = {}
    for line in reversed(stdout_text.splitlines()):
        if "avg_auc=" in line:
            for token in line.replace(",", " ").split():
                if "=" in token:
                    k, _, v = token.partition("=")
                    if k in ("avg_auc", "avg_f1_macro", "avg_mae", "avg_accuracy"):
                        try:
                            metrics[k] = float(v)
                        except ValueError:
                            pass
            break

    # Verify scenario applied
    applied: List[str] = []
    for line in stdout_text.splitlines():
        if any(kw in line for kw in (
            "Ablation: shared experts",
            "Expert ablation",
            "Structure ablation",
            "Feature ablation",
        )):
            applied.append(line.strip().split("] ")[-1] if "] " in line else line.strip())

    success = proc.returncode == 0 and epoch_count >= expected

    logger.info(
        "[%s] %s: %d/%d epochs, %.0fs, AUC=%s F1=%s",
        "OK" if success else "FAIL", name, epoch_count, expected,
        elapsed, metrics.get("avg_auc", "?"), metrics.get("avg_f1_macro", "?"),
    )
    for a in applied:
        logger.info("  → %s", a)

    if not success and not dry_run:
        # Show last error
        stderr_text = stderr_path.read_text(errors="replace")
        for line in stderr_text.splitlines()[-5:]:
            if "FutureWarning" not in line and "UserWarning" not in line:
                logger.error("  stderr: %s", line.rstrip())

    return {
        "name": name, "status": "OK" if success else "FAIL",
        "epochs": epoch_count, "seconds": round(elapsed),
        "metrics": metrics, "applied": applied,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--scenario", type=str, default=None)
    args = parser.parse_args()

    scenarios = SCENARIOS
    if args.scenario:
        scenarios = [s for s in SCENARIOS if s["name"] == args.scenario]
        if not scenarios:
            logger.error("Unknown scenario '%s'. Available: %s",
                         args.scenario, [s["name"] for s in SCENARIOS])
            sys.exit(1)

    logger.info("=" * 60)
    logger.info("LOCAL ABLATION: %d scenarios x %d epochs", len(scenarios), 1 if args.dry_run else EPOCHS)
    logger.info("=" * 60)

    results = []
    for s in scenarios:
        results.append(run_scenario(s, dry_run=args.dry_run))

    # Summary
    logger.info("\n" + "=" * 60)
    ok = sum(1 for r in results if r["status"] == "OK")
    skip = sum(1 for r in results if r["status"] == "skipped")
    fail = sum(1 for r in results if r["status"] == "FAIL")
    logger.info("DONE: OK=%d SKIP=%d FAIL=%d", ok, skip, fail)
    for r in sorted(results, key=lambda x: x.get("metrics", {}).get("avg_auc", 0), reverse=True):
        if r["status"] == "skipped":
            continue
        m = r.get("metrics", {})
        logger.info("  %-35s AUC=%-7s F1=%-7s (%ds)",
                     r["name"], m.get("avg_auc", "?"), m.get("avg_f1_macro", "?"), r.get("seconds", 0))

    Path(RESULTS).mkdir(parents=True, exist_ok=True)
    with open(Path(RESULTS) / "ablation_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
