#!/usr/bin/env python3
"""
SageMaker Pipeline ablation runner — local Docker mode.

Builds a SageMaker Pipeline with one TrainingStep per ablation scenario,
using LocalPipelineSession for local Docker execution.

Phases 1-3: Feature Group / Expert / Task x Structure ablation.
All scenario definitions and training defaults are read from pipeline.yaml
and feature_groups.yaml (config-driven, no hardcoding).

Usage::

    # Dry run — list all scenarios without executing
    python scripts/run_sagemaker_pipeline_ablation.py --dry-run

    # Run all scenarios
    python scripts/run_sagemaker_pipeline_ablation.py

    # Run specific scenarios only
    python scripts/run_sagemaker_pipeline_ablation.py --scenarios feat_full,feat_base_only

    # Parallel hint (for future cloud mode; local is always sequential)
    python scripts/run_sagemaker_pipeline_ablation.py --parallel 4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("sagemaker_pipeline_ablation")

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = os.environ.get(
    "ABLATION_CONFIG_PATH", "configs/santander/pipeline.yaml"
)
FEATURE_GROUPS_PATH = os.environ.get(
    "ABLATION_FEATURE_GROUPS_PATH", "configs/santander/feature_groups.yaml"
)
PHASE0_DIR = PROJECT_ROOT / "outputs" / "phase0"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "ablation_results"
DOCKER_IMAGE = os.environ.get("ABLATION_IMAGE", "model_training:v3.3")

# Container-internal config path (source_dir is project root)
CONTAINER_CONFIG_PATH = "configs/santander/pipeline.yaml"


# ---------------------------------------------------------------------------
# Config loading — ALL constants derived from YAML (CLAUDE.md 1.1)
# ---------------------------------------------------------------------------

def _load_pipeline_config() -> Dict[str, Any]:
    cfg_path = PROJECT_ROOT / CONFIG_PATH
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_feature_groups_config() -> Dict[str, Any]:
    fg_path = PROJECT_ROOT / FEATURE_GROUPS_PATH
    with open(fg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _extract_shared_experts(config: Dict[str, Any]) -> List[str]:
    return config.get("model", {}).get("expert_basket", {}).get("shared", [])


def _extract_training_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    td = config.get("ablation", {}).get("training_defaults", {})
    defaults = {
        "epochs": 10,
        "batch_size": 4096,
        "learning_rate": 0.008,
        "amp": True,
        "early_stopping_patience": 3,
        "seed": 42,
    }
    for key in defaults:
        if key in td:
            defaults[key] = td[key]
    return defaults


def _extract_task_tiers(config: Dict[str, Any]) -> Dict[str, Optional[List[str]]]:
    return config.get("ablation", {}).get("task_tiers", {})


def _extract_structure_variants(config: Dict[str, Any]) -> Dict[str, Dict[str, bool]]:
    return config.get("ablation", {}).get("structure_variants", {})


# ---------------------------------------------------------------------------
# Scenario builders (config-driven, mirrors run_santander_ablation.py)
# ---------------------------------------------------------------------------

def _build_feature_scenarios(
    fg_config: Dict[str, Any],
    base_group_names: List[str],
) -> List[Dict[str, Any]]:
    """Build feature group ablation scenarios from feature_groups.yaml."""
    all_groups = [
        g["name"]
        for g in fg_config.get("feature_groups", [])
        if g.get("enabled", True)
    ]
    advanced_groups = [g for g in all_groups if g not in base_group_names]

    scenarios: List[Dict[str, Any]] = []

    # Full baseline
    scenarios.append({"name": "feat_full", "hp": {}})

    # Base only (remove all advanced)
    scenarios.append({
        "name": "feat_base_only",
        "hp": {"removed_feature_groups": json.dumps(advanced_groups)},
    })

    # Bottom-up: base + one advanced group
    for group in advanced_groups:
        remove = [g for g in advanced_groups if g != group]
        scenarios.append({
            "name": f"feat_base+{group}",
            "hp": {"removed_feature_groups": json.dumps(remove)},
        })

    # Top-down: full minus one group
    for group in advanced_groups:
        scenarios.append({
            "name": f"feat_full-{group}",
            "hp": {"removed_feature_groups": json.dumps([group])},
        })

    return scenarios


def _build_expert_scenarios(
    all_experts: List[str],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Build expert ablation scenarios from model.expert_basket."""
    ablation_cfg = config.get("ablation", {})
    base_expert = ablation_cfg.get("base_expert", "deepfm")
    minimal_expert = ablation_cfg.get("minimal_expert", "mlp")

    scenarios: List[Dict[str, Any]] = []

    # Base expert alone
    scenarios.append({
        "name": f"expert_{base_expert}_only",
        "hp": {"shared_experts": json.dumps([base_expert])},
    })

    # Base + one other
    for expert in all_experts:
        if expert == base_expert:
            continue
        short = expert.replace("optimal_transport", "ot").replace(
            "temporal_ensemble", "temporal"
        )
        scenarios.append({
            "name": f"expert_{base_expert}+{short}",
            "hp": {"shared_experts": json.dumps([base_expert, expert])},
        })

    # Full basket
    scenarios.append({
        "name": "expert_full_basket",
        "hp": {},
    })

    # Top-down: full minus one
    for expert in all_experts:
        short = expert.replace("optimal_transport", "ot").replace(
            "temporal_ensemble", "temporal"
        )
        scenarios.append({
            "name": f"expert_full-{short}",
            "hp": {"shared_experts": json.dumps(
                [e for e in all_experts if e != expert]
            )},
        })

    # Minimal baseline
    scenarios.append({
        "name": f"expert_{minimal_expert}_only",
        "hp": {"shared_experts": json.dumps([minimal_expert])},
    })

    return scenarios


def _build_structure_scenarios(
    task_tiers: Dict[str, Optional[List[str]]],
    structure_variants: Dict[str, Dict[str, bool]],
) -> List[Dict[str, Any]]:
    """Build task x structure cross ablation scenarios."""
    scenarios: List[Dict[str, Any]] = []

    for tier_name, tasks in task_tiers.items():
        for struct_name, struct_flags in structure_variants.items():
            hp: Dict[str, Any] = {
                "use_ple": str(struct_flags.get("use_ple", True)).lower(),
                "use_adatt": str(struct_flags.get("use_adatt", True)).lower(),
            }
            if tasks is not None:
                hp["active_tasks"] = json.dumps(tasks)
            scenarios.append({
                "name": f"struct_{tier_name}_{struct_name}",
                "hp": hp,
            })

    return scenarios


def build_all_scenarios(
    pipeline_config: Dict[str, Any],
    fg_config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Build every ablation scenario from config."""
    ablation_cfg = pipeline_config.get("ablation", {})

    base_groups = ablation_cfg.get("base_groups", [])
    all_experts = _extract_shared_experts(pipeline_config)
    task_tiers = _extract_task_tiers(pipeline_config)
    structure_variants = _extract_structure_variants(pipeline_config)

    scenarios: List[Dict[str, Any]] = []

    # Phase 1: Feature ablation
    if ablation_cfg.get("feature_scenarios") == "auto":
        scenarios.extend(_build_feature_scenarios(fg_config, base_groups))
    else:
        for s in ablation_cfg.get("feature_scenarios", []):
            scenarios.append({"name": f"feat_{s['name']}", "hp": s.get("hp", {})})

    # Phase 2: Expert ablation
    if ablation_cfg.get("expert_scenarios") == "auto":
        scenarios.extend(_build_expert_scenarios(all_experts, pipeline_config))
    else:
        for s in ablation_cfg.get("expert_scenarios", []):
            scenarios.append({"name": f"expert_{s['name']}", "hp": s.get("hp", {})})

    # Phase 3: Task x Structure
    scenarios.extend(_build_structure_scenarios(task_tiers, structure_variants))

    return scenarios


# ---------------------------------------------------------------------------
# SageMaker Pipeline construction
# ---------------------------------------------------------------------------

def _create_training_step(
    scenario: Dict[str, Any],
    training_defaults: Dict[str, Any],
    source_dir: str,
    phase0_path: str,
    output_base: str,
    sagemaker_session: Any,
) -> Any:
    """Create a SageMaker TrainingStep for one ablation scenario.

    Returns (step, estimator) tuple for result collection.
    """
    from sagemaker.inputs import TrainingInput
    from sagemaker.pytorch import PyTorch
    from sagemaker.workflow.steps import TrainingStep

    name = scenario["name"]
    scenario_hp = scenario["hp"]

    # Build hyperparameters: training defaults + scenario overrides
    hyperparameters: Dict[str, str] = {
        "config": CONTAINER_CONFIG_PATH,
        "epochs": str(training_defaults["epochs"]),
        "batch_size": str(training_defaults["batch_size"]),
        "learning_rate": str(training_defaults["learning_rate"]),
        "seed": str(training_defaults["seed"]),
        "amp": str(training_defaults["amp"]).lower(),
        "early_stopping_patience": str(training_defaults["early_stopping_patience"]),
        "ablation_scenario": name,
    }
    # Merge scenario-specific HPs (removed_feature_groups, shared_experts, etc.)
    for k, v in scenario_hp.items():
        hyperparameters[k] = str(v)

    scenario_output = os.path.join(output_base, name)
    os.makedirs(scenario_output, exist_ok=True)

    estimator = PyTorch(
        entry_point="containers/training/train.py",
        source_dir=source_dir,
        role="arn:aws:iam::role/dummy-local",  # unused in local mode
        image_uri=DOCKER_IMAGE,
        instance_type="local_gpu",
        instance_count=1,
        output_path=f"file://{scenario_output}",
        hyperparameters=hyperparameters,
        sagemaker_session=sagemaker_session,
        environment={
            "PYTHONPATH": "/opt/ml/code",
        },
    )

    step = TrainingStep(
        name=name.replace("+", "_plus_").replace("-", "_minus_"),
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                f"file://{phase0_path}",
                content_type="application/x-parquet",
            ),
        },
    )

    return step, estimator


def build_pipeline(
    scenarios: List[Dict[str, Any]],
    training_defaults: Dict[str, Any],
    sagemaker_session: Any,
) -> Any:
    """Build a SageMaker Pipeline with one TrainingStep per scenario."""
    from sagemaker.workflow.pipeline import Pipeline

    source_dir = str(PROJECT_ROOT)
    phase0_path = str(PHASE0_DIR).replace("\\", "/")
    output_base = str(RESULTS_DIR).replace("\\", "/")

    steps = []
    estimators = {}

    for scenario in scenarios:
        step, est = _create_training_step(
            scenario=scenario,
            training_defaults=training_defaults,
            source_dir=source_dir,
            phase0_path=phase0_path,
            output_base=output_base,
            sagemaker_session=sagemaker_session,
        )
        steps.append(step)
        estimators[scenario["name"]] = est

    pipeline = Pipeline(
        name="ablation-study-local",
        steps=steps,
        sagemaker_session=sagemaker_session,
    )

    return pipeline, estimators


# ---------------------------------------------------------------------------
# Result collection
# ---------------------------------------------------------------------------

def _collect_result(scenario_name: str, elapsed: float) -> Dict[str, Any]:
    """Read eval_metrics.json from a completed scenario's output."""
    metrics_path = RESULTS_DIR / scenario_name / "eval_metrics.json"
    metrics: Dict[str, Any] = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            raw = json.load(f)
        metrics = raw.get("final_metrics", raw)

    auc = metrics.get("auc", "N/A")
    f1 = metrics.get("f1_macro_avg", "N/A")

    return {
        "scenario": scenario_name,
        "status": "OK" if metrics else "NO_METRICS",
        "auc": auc,
        "f1_macro": f1,
        "time_s": round(elapsed, 1),
        "metrics": metrics,
    }


def _print_summary(results: List[Dict[str, Any]], total_time: float) -> None:
    """Print final summary table."""
    ok_count = sum(1 for r in results if r["status"] == "OK")
    fail_count = sum(1 for r in results if r["status"] != "OK")

    print("\n" + "=" * 72)
    print(f"ABLATION COMPLETE: {ok_count}/{len(results)} OK, {fail_count} FAIL/NO_METRICS")
    print(f"Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print("=" * 72)

    print(f"\n{'Scenario':<45} {'AUC':>8} {'F1_macro':>10} {'Time':>7}")
    print("-" * 72)
    for r in results:
        auc_str = f"{r['auc']:.4f}" if isinstance(r["auc"], float) else str(r["auc"])
        f1_str = (
            f"{r['f1_macro']:.4f}"
            if isinstance(r["f1_macro"], float)
            else str(r["f1_macro"])
        )
        print(f"{r['scenario']:<45} {auc_str:>8} {f1_str:>10} {r['time_s']:>6.1f}s")


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SageMaker Pipeline ablation runner (local Docker mode)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print all scenarios without executing",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default="",
        help="Comma-separated list of scenario names to run (default: all)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Max parallel scenarios (local mode is sequential; prep for cloud)",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=CONFIG_PATH,
        help="Path to pipeline YAML config (default: %(default)s)",
    )
    parser.add_argument(
        "--feature-groups-path",
        type=str,
        default=FEATURE_GROUPS_PATH,
        help="Path to feature_groups YAML config (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Override config paths if provided
    global CONFIG_PATH, FEATURE_GROUPS_PATH
    CONFIG_PATH = args.config_path
    FEATURE_GROUPS_PATH = args.feature_groups_path

    # Load configs
    pipeline_config = _load_pipeline_config()
    fg_config = _load_feature_groups_config()
    training_defaults = _extract_training_defaults(pipeline_config)

    # Build all scenarios
    all_scenarios = build_all_scenarios(pipeline_config, fg_config)

    # Filter scenarios if --scenarios flag is provided
    if args.scenarios:
        requested = set(args.scenarios.split(","))
        filtered = [s for s in all_scenarios if s["name"] in requested]
        missing = requested - {s["name"] for s in filtered}
        if missing:
            logger.warning("Scenarios not found: %s", ", ".join(sorted(missing)))
        all_scenarios = filtered

    logger.info("Total scenarios: %d (parallel hint: %d)", len(all_scenarios), args.parallel)

    # Dry run: print scenarios and exit
    if args.dry_run:
        print(f"\n{'#':<4} {'Scenario':<45} {'Extra HPs'}")
        print("-" * 90)
        for i, s in enumerate(all_scenarios, 1):
            hp_summary = ", ".join(f"{k}={v}" for k, v in s["hp"].items()) or "(baseline)"
            print(f"{i:<4} {s['name']:<45} {hp_summary}")
        print(f"\nTotal: {len(all_scenarios)} scenarios")
        print("Training defaults:", json.dumps(training_defaults, indent=2))
        return

    # Import SageMaker (deferred to avoid import cost on --dry-run)
    try:
        from sagemaker.local import LocalSession
        from sagemaker.workflow.pipeline import Pipeline
    except ImportError:
        logger.error(
            "sagemaker package is required. Install with: pip install sagemaker"
        )
        sys.exit(1)

    # Use LocalSession for local Docker execution
    # LocalPipelineSession may not exist in all sagemaker versions;
    # fall back to LocalSession which supports local_gpu mode.
    try:
        from sagemaker.local.local_session import LocalPipelineSession
        sm_session = LocalPipelineSession()
        logger.info("Using LocalPipelineSession for pipeline execution")
    except ImportError:
        sm_session = LocalSession()
        logger.info("LocalPipelineSession not available; using LocalSession")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Build pipeline
    pipeline, estimators = build_pipeline(
        scenarios=all_scenarios,
        training_defaults=training_defaults,
        sagemaker_session=sm_session,
    )

    logger.info("Pipeline built with %d steps", len(pipeline.steps))

    # Execute: in local mode, run steps sequentially with timing
    results: List[Dict[str, Any]] = []
    t_global = time.time()

    for i, scenario in enumerate(all_scenarios, 1):
        name = scenario["name"]
        logger.info(
            "[%d/%d] Running scenario: %s", i, len(all_scenarios), name
        )
        t0 = time.time()

        try:
            est = estimators[name]
            from sagemaker.inputs import TrainingInput

            phase0_path = str(PHASE0_DIR).replace("\\", "/")
            est.fit(
                inputs={
                    "train": TrainingInput(
                        f"file://{phase0_path}",
                        content_type="application/x-parquet",
                    ),
                },
                wait=True,
                logs="All",
            )
            elapsed = time.time() - t0
            result = _collect_result(name, elapsed)
        except Exception as exc:
            elapsed = time.time() - t0
            logger.error("Scenario %s failed: %s", name, exc, exc_info=True)
            result = {
                "scenario": name,
                "status": "FAIL",
                "auc": "N/A",
                "f1_macro": "N/A",
                "time_s": round(elapsed, 1),
                "metrics": {},
                "error": str(exc),
            }

        results.append(result)
        status_icon = "OK" if result["status"] == "OK" else "FAIL"
        logger.info(
            "  [%s] %s: AUC=%s, time=%.1fs",
            status_icon, name, result["auc"], elapsed,
        )

    total_time = time.time() - t_global

    # Print summary
    _print_summary(results, total_time)

    # Save summary JSON
    summary_path = RESULTS_DIR / "pipeline_ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", summary_path)


if __name__ == "__main__":
    main()
