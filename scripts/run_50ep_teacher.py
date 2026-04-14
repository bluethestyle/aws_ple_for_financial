"""
30-epoch teacher training: 3 key scenarios for Paper 2 distillation + comparison.
Runs train.py directly as subprocess — no orchestrator.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

if sys.platform == "win32":
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001 | 0x00000040)

PHASE0 = "outputs/phase0_v12"
OUTDIR = "outputs/ablation_v12_30ep"
CONFIG = "configs/santander/pipeline.yaml"

BASE_HPS = {
    "config": CONFIG,
    "epochs": 30,
    "batch_size": 5632,
    "learning_rate": 0.0005,
    "seed": 42,
    "amp": True,
    "early_stopping_patience": 30,  # no early stop
    "warmup_epochs": 5,             # 30ep -> warmup 5 (17%)
    "num_workers": 0,
    "use_adatt": "false",           # adaTT OFF — confirmed best
    "use_grad_surgery": "false",    # GradSurgery OFF — PLE softmax alone is best
}

SCENARIOS = [
    # Teacher model: PLE softmax, all 7 experts
    {"name": "teacher_full", "hp": {}},
    # Comparison: DeepFM only
    {"name": "teacher_deepfm_base", "hp": {"shared_experts": json.dumps(["deepfm"])}},
    # Comparison: Shared bottom (no PLE)
    {"name": "teacher_shared_bottom", "hp": {
        "use_ple": "false",
        "use_cgc_gate": "false",
        "use_group_task_expert": "false",
        "use_logit_transfer": "false",
        "use_hmm_projectors": "false",
    }},
]


def verify_config():
    """Print config summary for human verification BEFORE any training."""
    print("=" * 60)
    print("CONFIG VERIFICATION — CHECK BEFORE PROCEEDING")
    print("=" * 60)
    print(f"  Phase 0:       {PHASE0}")
    print(f"  Output:        {OUTDIR}")
    print(f"  Epochs:        {BASE_HPS['epochs']}")
    print(f"  Batch size:    {BASE_HPS['batch_size']}")
    print(f"  Warmup:        {BASE_HPS['warmup_epochs']}")
    print(f"  LR:            {BASE_HPS['learning_rate']}")
    print(f"  AMP:           {BASE_HPS['amp']}")
    print(f"  use_adatt:     {BASE_HPS['use_adatt']}")
    print(f"  use_grad_surgery: {BASE_HPS['use_grad_surgery']}")
    print(f"  Scenarios:     {len(SCENARIOS)}")
    for s in SCENARIOS:
        extra = {k: v for k, v in s['hp'].items()} if s['hp'] else "(default PLE softmax)"
        print(f"    - {s['name']}: {extra}")
    print("=" * 60)
    print()


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
            env=env, stdout=fout, stderr=ferr,
        )

    elapsed = time.time() - start
    status = "OK" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
    print(f"[{status}] {name} ({elapsed/60:.1f}min)")

    # Verify adaTT was actually disabled
    if stdout_path.exists():
        text = stdout_path.read_text(encoding="utf-8", errors="replace")
        if "adaTT: enabled" in text:
            print(f"  *** WARNING: adaTT was ENABLED in {name}! ***")
        elif "adaTT: disabled" in text:
            print(f"  [verified] adaTT disabled")
        if "grad_surgery=enabled" in text:
            print(f"  *** WARNING: GradSurgery was ENABLED in {name}! ***")


if __name__ == "__main__":
    verify_config()

    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    for i, s in enumerate(SCENARIOS):
        run_scenario(s)
        if i < len(SCENARIOS) - 1:
            time.sleep(5)

    print()
    print(f"COMPLETE: {time.strftime('%Y-%m-%d %H:%M:%S')}")
