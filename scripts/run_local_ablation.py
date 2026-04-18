#!/usr/bin/env python
"""
Local Ablation Runner: 13 tasks, 20 epochs (warmup=7)
Delta measurement only — same epoch count for fair comparison.
No Docker, no SageMaker local mode. Pure local Python.

Usage:
    python scripts/run_local_ablation.py                # full 10-epoch ablation
    python scripts/run_local_ablation.py --dry-run      # 1-epoch verify settings
    python scripts/run_local_ablation.py --scenario NAME # single scenario
    python scripts/run_local_ablation.py --force-fresh  # archive old + start clean
"""
from __future__ import annotations

import argparse
import json
import logging
import os

# Block orchestrator from initialising a CUDA context.  On WDDM (Windows),
# a parent process that touches CUDA will hold shared GPU memory even if the
# actual work runs in a child subprocess.  Setting this BEFORE any library
# import prevents accidental torch/cupy init in the orchestrator.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import ctypes
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Prevent Windows from sleeping while ablation is running.
# ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
if sys.platform == "win32":
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001 | 0x00000040)
    logging.basicConfig(level=logging.INFO)  # ensure logger exists for this message
    logging.getLogger(__name__).info("Windows sleep prevention: SetThreadExecutionState enabled")
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Windows process isolation helpers
# ---------------------------------------------------------------------------
# On Windows, each child subprocess is launched in a NEW process group so that:
#   1. The child's Windows Job Object is independent from the parent's.
#   2. CUDA/GPU driver handles acquired by child processes are not inherited
#      back into the parent, preventing handle table exhaustion after ~4 runs.
#   3. STATUS_DATATYPE_MISALIGNMENT (0xC0000002) crashes caused by stale
#      inherited DLL/GPU state are avoided.
#
# CREATE_NEW_PROCESS_GROUP also prevents Ctrl-C from propagating to children
# so we must send CTRL_BREAK_EVENT explicitly if we ever need to kill them.
_IS_WINDOWS = sys.platform == "win32"
_SUBPROCESS_CREATION_FLAGS = subprocess.CREATE_NEW_PROCESS_GROUP if _IS_WINDOWS else 0

# CUDA environment variables that must NOT be inherited from the parent process.
# The parent never uses GPU; allowing children to inherit a dirty CUDA env from
# a previously-crashed child (via os.environ) can cause misalignment faults.
_CUDA_ENV_BLOCKLIST = {
    "CUDA_LAUNCH_BLOCKING",   # set by train.py at import time via os.environ.setdefault
    "CUDA_VISIBLE_DEVICES",   # may be set by a previous child crash handler
    "NCCL_DEBUG",
    "TORCH_DISTRIBUTED_DEBUG",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PHASE0 = "outputs/phase0_v12"
RESULTS = "outputs/ablation_v12"
CONFIG = "configs/pipeline.yaml"
DATASET_CONFIG = "configs/datasets/santander.yaml"
EPOCHS = 10             # reduced from 20 — plateau observed at epoch 5-6 across scenarios
WARMUP = 3              # 30% of epochs
BATCH = 5632          # orchestrator CUDA-blocked, no VRAM spillover
LR = 0.0005
SEED = 42
AMP = True
PATIENCE = EPOCHS       # no early stop — run full epochs for fair delta comparison

# Robustness: retry spurious failures (subprocess crashed/killed before producing output)
MIN_REAL_RUN_SEC = 60   # if elapsed < this, treat as spurious failure
MAX_RETRIES = 3
INTER_SCENARIO_DELAY_SEC = 5  # give GPU/system time to cleanup between scenarios

BASE_HPS: Dict[str, Any] = {
    "config": CONFIG,
    "dataset_config": DATASET_CONFIG,
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
    # === Structure ablation (8) — "add one component at a time" progression ===

    # TRUE shared_bottom: no CGC gate, no GroupTaskExpert, no logit transfer, no HMM projectors
    {"name": "struct_13_shared_bottom",
     "hp": {
         "use_ple": "false",
         "use_adatt": "false",
         "use_cgc_gate": "false",
         "use_group_task_expert": "false",
         "use_logit_transfer": "false",
         "use_hmm_projectors": "false",
     }},

    # CGC only: single PLE layer with CGC attention gate
    {"name": "struct_13_cgc_only",
     "hp": {
         "use_ple": "false",
         "use_adatt": "false",
         "use_cgc_gate": "true",
         "use_group_task_expert": "false",
         "use_logit_transfer": "false",
         "use_hmm_projectors": "false",
     }},

    # PLE softmax: multi-layer CGC with softmax gate
    {"name": "struct_13_ple_softmax",
     "hp": {
         "use_ple": "true",
         "use_adatt": "false",
         "gate_type": "softmax",
         "use_cgc_gate": "true",
         "use_group_task_expert": "false",
         "use_logit_transfer": "false",
         "use_hmm_projectors": "false",
     }},

    # PLE sigmoid: multi-layer CGC with sigmoid gate
    {"name": "struct_13_ple_sigmoid",
     "hp": {
         "use_ple": "true",
         "use_adatt": "false",
         "gate_type": "sigmoid",
         "use_cgc_gate": "true",
         "use_group_task_expert": "false",
         "use_logit_transfer": "false",
         "use_hmm_projectors": "false",
     }},

    # PLE sigmoid + adaTT (pure) — identical to struct_13_ple_sigmoid but with adaTT enabled.
    # Purpose: isolate the PLE+adaTT interference effect without GTE/LT/HMM confounds.
    {"name": "struct_13_ple_sigmoid_adatt",
     "hp": {
         "use_ple": "true",
         "use_adatt": "true",
         "gate_type": "sigmoid",
         "use_cgc_gate": "true",
         "use_group_task_expert": "false",
         "use_logit_transfer": "false",
         "use_hmm_projectors": "false",
     }},

    # PLE sigmoid + AdaTT-sp representation-level fusion (Li et al., KDD 2023).
    # Identical to struct_13_ple_sigmoid but with the CGC gate replaced by the
    # AdaTT-sp variant (native expert residual added on top of the gated sum).
    # Loss-level "adaTT" stays OFF to isolate the fusion-mechanism effect.
    {"name": "struct_13_adatt_sp",
     "hp": {
         "use_ple": "true",
         "use_adatt": "false",
         "use_adatt_sp": "true",
         "gate_type": "sigmoid",
         "use_cgc_gate": "true",
         "use_group_task_expert": "false",
         "use_logit_transfer": "false",
         "use_hmm_projectors": "false",
     }},

    # Paper 3 --- intra-task residual recovery via complementary gate (M1).
    # After the standard gated weighted sum, add a learnable-scalar-weighted
    # residual whose weights are the renormalised (1 - gate_weights). This
    # recovers signal from experts the gate down-weighted, *within the same
    # task* and without any cross-task mixing. Everything else matches
    # struct_13_ple_sigmoid.
    {"name": "struct_13_residual_complement",
     "hp": {
         "use_ple": "true",
         "use_adatt": "false",
         "use_adatt_sp": "false",
         "use_residual_recovery": "true",
         "residual_method": "complement",
         "gate_type": "sigmoid",
         "use_cgc_gate": "true",
         "use_group_task_expert": "false",
         "use_logit_transfer": "false",
         "use_hmm_projectors": "false",
     }},

    # PLE sigmoid + GroupTaskExpert (GroupEncoder + ClusterEmbedding)
    {"name": "struct_13_ple_sigmoid_gte",
     "hp": {
         "use_ple": "true",
         "use_adatt": "false",
         "gate_type": "sigmoid",
         "use_cgc_gate": "true",
         "use_group_task_expert": "true",
         "use_logit_transfer": "false",
         "use_hmm_projectors": "false",
     }},

    # PLE sigmoid + GTE + Logit Transfer
    {"name": "struct_13_ple_sigmoid_gte_lt",
     "hp": {
         "use_ple": "true",
         "use_adatt": "false",
         "gate_type": "sigmoid",
         "use_cgc_gate": "true",
         "use_group_task_expert": "true",
         "use_logit_transfer": "true",
         "use_hmm_projectors": "false",
     }},

    # Full PLE: all components except adaTT
    {"name": "struct_13_ple_full",
     "hp": {
         "use_ple": "true",
         "use_adatt": "false",
         "gate_type": "sigmoid",
         "use_cgc_gate": "true",
         "use_group_task_expert": "true",
         "use_logit_transfer": "true",
         "use_hmm_projectors": "true",
     }},

    # PLE softmax + adaTT: softmax isolates experts, adaTT selectively shares
    {"name": "struct_13_ple_softmax_adatt",
     "hp": {
         "use_ple": "true",
         "use_adatt": "true",
         "gate_type": "softmax",
         "use_cgc_gate": "true",
         "use_group_task_expert": "true",
         "use_logit_transfer": "true",
         "use_hmm_projectors": "true",
     }},

    # Full PLE (sigmoid) + adaTT
    {"name": "struct_13_ple_full_adatt",
     "hp": {
         "use_ple": "true",
         "use_adatt": "true",
         "gate_type": "sigmoid",
         "use_cgc_gate": "true",
         "use_group_task_expert": "true",
         "use_logit_transfer": "true",
         "use_hmm_projectors": "true",
     }},

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

    # === adaTT isolated ===
    # shared_bottom + adaTT only: isolates adaTT contribution without PLE/CGC
    {"name": "struct_13_shared_bottom_adatt",
     "hp": {
         "use_ple": "false",
         "use_adatt": "true",
         "use_cgc_gate": "false",
         "use_group_task_expert": "false",
         "use_logit_transfer": "false",
         "use_hmm_projectors": "false",
     }},

    # === GradSurgery scenarios ===
    # GradSurgery replaces adaTT loss-level transfer with gradient-level projection.
    # adaTT is disabled when GradSurgery is enabled — they address the same problem
    # (inter-task interference) at different levels.

    # PLE softmax + GradSurgery (no adaTT) — core hypothesis
    # batch_size=4096: retain_graph for GS increases VRAM ~2.4GB
    {"name": "struct_13_ple_softmax_gs",
     "hp": {
         "use_ple": "true",
         "use_adatt": "false",
         "gate_type": "softmax",
         "use_cgc_gate": "true",
         "use_group_task_expert": "true",
         "use_logit_transfer": "true",
         "use_hmm_projectors": "true",
         "use_grad_surgery": "true",
         "batch_size": 4096,
     }},

    # shared_bottom + GradSurgery — isolates GradSurgery contribution
    {"name": "struct_13_shared_bottom_gs",
     "hp": {
         "use_ple": "false",
         "use_adatt": "false",
         "use_cgc_gate": "false",
         "use_group_task_expert": "false",
         "use_logit_transfer": "false",
         "use_hmm_projectors": "false",
         "use_grad_surgery": "true",
         "batch_size": 4096,
     }},

    # PLE sigmoid + GradSurgery — sigmoid vs softmax under GradSurgery
    {"name": "struct_13_ple_sigmoid_gs",
     "hp": {
         "use_ple": "true",
         "use_adatt": "false",
         "gate_type": "sigmoid",
         "use_cgc_gate": "true",
         "use_group_task_expert": "true",
         "use_logit_transfer": "true",
         "use_hmm_projectors": "true",
         "use_grad_surgery": "true",
         "batch_size": 4096,
     }},
]


# ---------------------------------------------------------------------------
# Manifest / archive helpers
# ---------------------------------------------------------------------------
MANIFEST_FILENAME = "run_manifest.json"
# Keys compared when deciding if configs match on resume
_MATCH_KEYS = ("epochs", "warmup_epochs", "batch_size", "learning_rate", "seed", "amp")


def load_previous_manifest(results_dir: Path) -> Optional[Dict[str, Any]]:
    """Return the manifest saved in a previous run, or None if absent/corrupt."""
    path = results_dir / MANIFEST_FILENAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read previous manifest (%s): %s", path, exc)
        return None


def save_current_manifest(
    results_dir: Path,
    hps: Dict[str, Any],
    scenarios: List[Dict[str, Any]],
) -> None:
    """Write run_manifest.json BEFORE any scenario runs."""
    results_dir.mkdir(parents=True, exist_ok=True)

    # Try to capture git commit hash (best-effort)
    git_hash: Optional[str] = None
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        pass

    manifest = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "script_version": git_hash,
        "epochs": hps.get("epochs"),
        "warmup_epochs": hps.get("warmup_epochs"),
        "batch_size": hps.get("batch_size"),
        "learning_rate": hps.get("learning_rate"),
        "seed": hps.get("seed"),
        "amp": hps.get("amp"),
        "scenarios": [s["name"] for s in scenarios],
    }
    path = results_dir / MANIFEST_FILENAME
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Manifest saved → %s", path)


def archive_results(results_dir: Path) -> Path:
    """
    Atomically move results_dir to a timestamped archive sibling.
    Returns the archive path.  Raises on failure (disk full, etc.).
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = results_dir.parent / f"{results_dir.name}_archive_{ts}"
    logger.info("Archiving %s → %s", results_dir, archive_path)
    shutil.move(str(results_dir), str(archive_path))
    logger.info("Archive complete: %s", archive_path)
    return archive_path


def configs_match(prev: Dict[str, Any], curr: Dict[str, Any]) -> bool:
    """Return True only when all key training HPs are identical."""
    for key in _MATCH_KEYS:
        if prev.get(key) != curr.get(key):
            logger.debug("Config mismatch on '%s': prev=%s curr=%s",
                         key, prev.get(key), curr.get(key))
            return False
    return True


def _check_and_handle_existing_results(
    results_dir: Path,
    base_hps: Dict[str, Any],
    scenarios: List[Dict[str, Any]],
    force_fresh: bool,
) -> None:
    """
    Called once at startup.  Decides whether to archive, resume, or bail.

    Mutates nothing on disk except (possibly) archiving results_dir.
    After this call returns, results_dir either does not exist (fresh start)
    or contains a matching manifest (resume).
    """
    if not results_dir.exists():
        return  # nothing to check

    # Count any completed scenario directories
    completed = [
        d for d in results_dir.iterdir()
        if d.is_dir() and (d / "model" / "model.pth").exists()
    ]

    if not completed and not (results_dir / MANIFEST_FILENAME).exists():
        # Directory exists but is effectively empty — safe to continue
        return

    prev_manifest = load_previous_manifest(results_dir)

    if force_fresh:
        logger.warning(
            "--force-fresh requested: archiving %d completed scenario(s) in %s",
            len(completed), results_dir,
        )
        archive_results(results_dir)
        return

    if prev_manifest is None:
        # Has completed results but no manifest — legacy run
        logger.warning(
            "Found %d completed scenario(s) in %s but no manifest. "
            "Use --force-fresh to archive and start clean, or Ctrl-C to abort.",
            len(completed), results_dir,
        )
        # Give user 10 s to Ctrl-C; then continue (resume best-effort)
        for remaining in range(10, 0, -1):
            sys.stdout.write(f"\r  Resuming in {remaining}s (Ctrl-C to abort) ...")
            sys.stdout.flush()
            time.sleep(1)
        sys.stdout.write("\n")
        return

    curr_for_compare = {k: base_hps.get(k) for k in _MATCH_KEYS}
    if configs_match(prev_manifest, curr_for_compare):
        logger.info(
            "Config matches previous manifest (epochs=%s warmup=%s). "
            "Resuming — %d scenario(s) already completed.",
            prev_manifest.get("epochs"), prev_manifest.get("warmup_epochs"),
            len(completed),
        )
        return

    # Mismatch — show diff and archive
    mismatched = [
        k for k in _MATCH_KEYS
        if prev_manifest.get(k) != curr_for_compare.get(k)
    ]
    logger.warning(
        "Config mismatch vs previous run on: %s", mismatched,
    )
    for k in mismatched:
        logger.warning("  %s: was %s, now %s", k, prev_manifest.get(k), curr_for_compare.get(k))
    logger.warning(
        "Archiving previous results (%d scenario(s)) to preserve experiment data.",
        len(completed),
    )
    archive_results(results_dir)


def _build_child_env() -> Dict[str, str]:
    """
    Build a clean environment for child training subprocesses.

    Strategy:
    - Start from os.environ (to inherit PATH, TEMP, system vars).
    - Strip CUDA_ENV_BLOCKLIST entries so a dirty CUDA env from a previous
      child cannot pollute the next child's GPU initialisation.
    - Never inherit CUDA_VISIBLE_DEVICES from parent — let the child see all
      GPUs as intended by its own config.
    """
    env = {k: v for k, v in os.environ.items() if k not in _CUDA_ENV_BLOCKLIST}
    return env


def _execute_train(
    hp: Dict[str, Any],
    out_dir: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> tuple:
    """
    Execute train.py as an isolated subprocess.

    Windows isolation measures applied:
      1. CREATE_NEW_PROCESS_GROUP  — independent Job Object; GPU/kernel handles
         acquired by the child are not inherited by the parent or siblings.
      2. Clean CUDA env             — CUDA_ENV_BLOCKLIST vars stripped so stale
         driver state from a previous crashed child cannot cause the next child
         to fault at DLL-load time (STATUS_DATATYPE_MISALIGNMENT 0xC0000002).
      3. Explicit stdout/stderr     — files opened only for the duration of this
         call; handles are closed before we read the log back.

    Returns (elapsed_seconds, stdout_text, returncode).
    """
    env = _build_child_env()
    # Force UTF-8 encoding in child process to avoid cp949 UnicodeEncodeError
    # when log messages contain em-dash or other non-ASCII characters.
    env["PYTHONIOENCODING"] = "utf-8"
    env["SM_CHANNEL_TRAIN"] = PHASE0
    env["SM_OUTPUT_DATA_DIR"] = str(out_dir)
    env["SM_MODEL_DIR"] = str(out_dir / "model")
    env["SM_HPS"] = json.dumps(hp)
    env["PYTHONPATH"] = str(Path.cwd())
    env["PYTHONUNBUFFERED"] = "1"

    start = time.time()
    with open(stdout_path, "w") as fout, open(stderr_path, "w") as ferr:
        proc = subprocess.run(
            [sys.executable, "-u", "containers/training/train.py"],
            env=env,
            stdout=fout,
            stderr=ferr,
            creationflags=_SUBPROCESS_CREATION_FLAGS,
        )
    elapsed = time.time() - start
    stdout_text = stdout_path.read_text(errors="replace")
    return elapsed, stdout_text, proc.returncode


def run_scenario(scenario: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
    """Run one ablation scenario with automatic retry for spurious failures."""
    import gc

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

    diff_hp = {k: v for k, v in hp.items() if k not in BASE_HPS}
    logger.info("[%s] %s %s", "DRY" if dry_run else "RUN", name, diff_hp or "(baseline)")

    stdout_path = out_dir / "logs" / "stdout.log"
    stderr_path = out_dir / "logs" / "stderr.log"

    # Retry loop for spurious subprocess failures (e.g. Windows sleep, resource limit)
    elapsed = 0.0
    stdout_text = ""
    returncode = -1
    for attempt in range(MAX_RETRIES):
        elapsed, stdout_text, returncode = _execute_train(
            hp, out_dir, stdout_path, stderr_path,
        )
        # Detect spurious failure: elapsed < threshold and no output
        is_spurious = (
            elapsed < MIN_REAL_RUN_SEC
            and "val_loss=" not in stdout_text
            and not dry_run
        )
        if not is_spurious:
            break
        # Log Windows exit code in hex so 0xC0000002 etc. are immediately
        # recognisable without manual conversion.
        rc_hex = hex(returncode & 0xFFFFFFFF) if returncode < 0 else hex(returncode)
        logger.warning(
            "[RETRY] %s: spurious failure (elapsed=%.1fs, rc=%d / %s, empty output) — "
            "attempt %d/%d",
            name, elapsed, returncode, rc_hex, attempt + 1, MAX_RETRIES,
        )
        # Force cleanup before retry; give GPU driver time to release memory
        # from the crashed child before we spawn a new one.
        gc.collect()
        time.sleep(INTER_SCENARIO_DELAY_SEC * 4)

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

    success = returncode == 0 and epoch_count >= expected

    rc_info = (
        f"rc={returncode}"
        if returncode == 0
        else f"rc={returncode} ({hex(returncode & 0xFFFFFFFF)})"
    )
    logger.info(
        "[%s] %s: %d/%d epochs, %.0fs, %s, AUC=%s F1=%s",
        "OK" if success else "FAIL", name, epoch_count, expected,
        elapsed, rc_info,
        metrics.get("avg_auc", "?"), metrics.get("avg_f1_macro", "?"),
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
    parser.add_argument(
        "--force-fresh",
        action="store_true",
        help="Archive existing results and start a clean ablation run.",
    )
    args = parser.parse_args()

    scenarios = SCENARIOS
    if args.scenario:
        scenarios = [s for s in SCENARIOS if s["name"] == args.scenario]
        if not scenarios:
            logger.error("Unknown scenario '%s'. Available: %s",
                         args.scenario, [s["name"] for s in SCENARIOS])
            sys.exit(1)

    results_dir = Path(RESULTS)

    # --- Archive / resume decision (BEFORE any scenario runs) ---
    try:
        _check_and_handle_existing_results(
            results_dir=results_dir,
            base_hps=BASE_HPS,
            scenarios=scenarios,
            force_fresh=args.force_fresh,
        )
    except Exception as exc:
        logger.error(
            "Failed to handle existing results: %s — aborting to prevent mixed state.", exc
        )
        sys.exit(1)

    # Write manifest for this run (dry-run writes its own ephemeral manifest)
    if not args.dry_run:
        try:
            save_current_manifest(results_dir, BASE_HPS, scenarios)
        except Exception as exc:
            logger.error("Failed to write run manifest: %s — aborting.", exc)
            sys.exit(1)

    logger.info("=" * 60)
    logger.info("LOCAL ABLATION: %d scenarios x %d epochs", len(scenarios), 1 if args.dry_run else EPOCHS)
    logger.info("=" * 60)

    import gc
    results = []
    for i, s in enumerate(scenarios):
        results.append(run_scenario(s, dry_run=args.dry_run))
        # Force cleanup between scenarios to prevent resource accumulation
        gc.collect()
        if i < len(scenarios) - 1 and not args.dry_run:
            time.sleep(INTER_SCENARIO_DELAY_SEC)

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

    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "ablation_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
