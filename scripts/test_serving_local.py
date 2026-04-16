"""
Local Serving Pipeline Test
============================

Tests the full serving pipeline up to (but NOT including) the LLM call
for recommendation reason generation.

Pipeline steps tested:
    1. Load distilled LGBM models from outputs/distillation_v2/model/
    2. Load 1 sample customer features from outputs/phase0_v12/ via DuckDB
    3. Run FallbackRouter to route each task to Layer 1/2/3
    4. Get predictions from LGBM models
    5. Extract contributing_features (top LGBM gain features per task)
    6. Run FactExtractor (Agent 1) — rule-based facts
    7. Run TemplateEngine L1 (Agent 2 first stage) — template-based reasons
    8. Run SelfChecker (Agent 3) — compliance validation
    9. STOP — does NOT call Bedrock/LLM

Run:
    PYTHONPATH=. python scripts/test_serving_local.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Windows: reconfigure stdout/stderr to UTF-8 so Korean/em-dash log messages
# from modules (fallback_router, etc.) do not raise UnicodeEncodeError on
# cp949 consoles.  Must be done before any logging.basicConfig call.
# ---------------------------------------------------------------------------
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except AttributeError:
        pass  # Python < 3.7 or non-TextIOWrapper stream
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except AttributeError:
        pass

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path (equivalent to PYTHONPATH=.)
# ---------------------------------------------------------------------------
_REPO_ROOT_CANDIDATE = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT_CANDIDATE) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_CANDIDATE))

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("test_serving_local")

# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_YAML = REPO_ROOT / "configs" / "pipeline.yaml"
DATASET_YAML = REPO_ROOT / "configs" / "datasets" / "santander.yaml"
DISTILLATION_DIR = REPO_ROOT / "outputs" / "distillation_v2" / "model"
PHASE0_PARQUET = REPO_ROOT / "outputs" / "phase0_v12" / "santander_final.parquet"
FACT_EXTRACTION_YAML = REPO_ROOT / "configs" / "financial" / "fact_extraction.yaml"


# ---------------------------------------------------------------------------
# Step 0: Load merged config
# ---------------------------------------------------------------------------

def load_config() -> Dict[str, Any]:
    logger.info("=" * 60)
    logger.info("STEP 0: Loading merged config")
    logger.info("  pipeline : %s", PIPELINE_YAML)
    logger.info("  dataset  : %s", DATASET_YAML)

    from core.pipeline.config import load_merged_config
    cfg = load_merged_config(PIPELINE_YAML, DATASET_YAML)

    tasks = cfg.get("tasks", [])
    logger.info("  tasks loaded: %d", len(tasks))
    for t in tasks:
        logger.info("    - %-30s type=%s", t["name"], t.get("type", "binary"))
    return cfg


# ---------------------------------------------------------------------------
# Step 1: Load LGBM models from distillation_v2
# ---------------------------------------------------------------------------

def load_lgbm_models() -> Tuple[
    Dict[str, Any],           # lgbm_models   {task_name: Booster}
    Dict[str, List[str]],     # feature_cols  {task_name: [col, ...]}
    Dict[str, Dict],          # fidelities    {task_name: fidelity_dict}
    Dict[str, Dict],          # teacher_metrics {task_name: metrics_dict}
]:
    logger.info("=" * 60)
    logger.info("STEP 1: Loading LGBM models from %s", DISTILLATION_DIR)

    try:
        import lightgbm as lgb
    except ImportError:
        logger.error("lightgbm not installed. Run: pip install lightgbm")
        sys.exit(1)

    # Load summary for teacher metrics
    summary_path = DISTILLATION_DIR / "distillation_summary.json"
    summary: Dict[str, Any] = {}
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        logger.info("  distillation_summary.json loaded")
        logger.info("  tasks_distilled       : %s", summary.get("tasks_distilled", []))
        logger.info("  tasks_direct_hardlabel: %s", summary.get("tasks_direct_hardlabel", []))
    else:
        logger.warning("  distillation_summary.json not found at %s", summary_path)

    lgbm_models: Dict[str, Any] = {}
    feature_cols: Dict[str, List[str]] = {}
    fidelities: Dict[str, Dict] = {}
    teacher_metrics: Dict[str, Dict] = {}

    # Iterate over task subdirectories
    for task_dir in sorted(DISTILLATION_DIR.iterdir()):
        if not task_dir.is_dir():
            continue
        task_name = task_dir.name
        model_path = task_dir / "model.lgbm"
        metadata_path = task_dir / "metadata.json"
        fidelity_path = task_dir / "fidelity.json"
        selected_features_path = task_dir / "selected_features.json"

        if not model_path.exists():
            logger.info("  %-35s — model.lgbm not found, skipping", task_name)
            continue

        try:
            booster = lgb.Booster(model_file=str(model_path))
            lgbm_models[task_name] = booster
        except Exception as exc:
            logger.error("  %-35s — failed to load model: %s", task_name, exc)
            continue

        # Load feature columns from metadata
        if metadata_path.exists():
            with open(metadata_path) as f:
                meta = json.load(f)
            cols = meta.get("feature_columns", [])
            feature_cols[task_name] = cols
            logger.info(
                "  %-35s — model loaded (%d trees, %d features)",
                task_name,
                meta.get("num_trees", "?"),
                len(cols),
            )
        else:
            # Fall back to selected_features indices if metadata missing
            feature_cols[task_name] = []
            logger.warning(
                "  %-35s — metadata.json not found, feature_cols empty", task_name
            )

        # Load fidelity
        if fidelity_path.exists():
            with open(fidelity_path) as f:
                fid = json.load(f)
            fidelities[task_name] = fid.get("metrics", fid)

        # Build teacher_metrics dict from fidelity or summary
        # For routing purposes we construct synthetic teacher metrics.
        # Real teacher metrics would come from teacher eval_metrics.json.
        # We use the student's fidelity to infer whether teacher was good.
        summary_fidelity = summary.get("fidelity", {}).get("per_task", {})
        task_fid = summary_fidelity.get(task_name, {})
        task_metrics = task_fid.get("metrics", {})

        # Reconstruct teacher quality proxies from available data
        task_type = None
        if metadata_path.exists():
            with open(metadata_path) as f_meta:
                meta2 = json.load(f_meta)
            task_type = meta2.get("task_type", "binary")

        if task_type == "binary":
            # student AUC ≈ teacher AUC - auc_gap
            student_auc = 1.0 - task_metrics.get("auc_gap", 0.05)  # rough proxy
            teacher_metrics[task_name] = {"auc_roc": max(0.5, student_auc)}
        elif task_type == "multiclass":
            # Use teacher_f1_macro from fidelity if available
            teacher_f1 = task_metrics.get("teacher_f1_macro", 0.3)
            # f1_ratio = teacher_f1 / (1/K) — we don't know K precisely, use 4
            teacher_metrics[task_name] = {"f1_ratio": teacher_f1 * 4}
        elif task_type == "regression":
            teacher_metrics[task_name] = {"r2": 0.1}  # safe default above threshold
        else:
            teacher_metrics[task_name] = {"auc_roc": 0.7}  # default

    logger.info("  Total models loaded: %d", len(lgbm_models))
    return lgbm_models, feature_cols, fidelities, teacher_metrics


# ---------------------------------------------------------------------------
# Step 2: Load 1 sample customer from phase0 parquet via DuckDB
# ---------------------------------------------------------------------------

def load_sample_customer(
    all_feature_cols: List[str],
) -> Tuple[str, Dict[str, Any], np.ndarray]:
    """Load one customer row from the phase0 parquet.

    Returns:
        (customer_id, feature_dict, feature_array)
    """
    logger.info("=" * 60)
    logger.info("STEP 2: Loading 1 sample customer from %s", PHASE0_PARQUET)

    if not PHASE0_PARQUET.exists():
        logger.error("Phase0 parquet not found: %s", PHASE0_PARQUET)
        sys.exit(1)

    try:
        import duckdb
    except ImportError:
        logger.error("duckdb not installed. Run: pip install duckdb")
        sys.exit(1)

    con = duckdb.connect()

    # Get column names from parquet
    available_cols_result = con.execute(
        f"SELECT * FROM read_parquet('{PHASE0_PARQUET.as_posix()}') LIMIT 0"
    ).description
    available_cols = [desc[0] for desc in available_cols_result]
    logger.info("  Parquet columns available: %d", len(available_cols))

    # Load one row
    row_result = con.execute(
        f"SELECT * FROM read_parquet('{PHASE0_PARQUET.as_posix()}') LIMIT 1"
    ).fetchdf()
    con.close()

    if row_result.empty:
        logger.error("Parquet is empty — cannot load sample")
        sys.exit(1)

    row = row_result.iloc[0]

    # Determine customer ID column (config-driven; try common names)
    cust_id_col = None
    for cand in ("ncodpers", "customer_id", "cust_id", "id"):
        if cand in available_cols:
            cust_id_col = cand
            break
    customer_id = str(int(row[cust_id_col])) if cust_id_col else "unknown_0"

    # Build feature dict from the row
    feature_dict: Dict[str, Any] = row.to_dict()

    logger.info("  customer_id : %s", customer_id)
    logger.info("  row shape   : %d columns", len(feature_dict))

    # Build feature array aligned to all_feature_cols for the LGBM models
    feat_array = np.array([
        float(feature_dict.get(c, 0.0) or 0.0) for c in all_feature_cols
    ], dtype=np.float32)
    logger.info("  feature_array shape: %s", feat_array.shape)

    return customer_id, feature_dict, feat_array


# ---------------------------------------------------------------------------
# Step 3: Run FallbackRouter
# ---------------------------------------------------------------------------

def run_fallback_router(
    cfg: Dict[str, Any],
    lgbm_models: Dict[str, Any],
    fidelities: Dict[str, Dict],
    teacher_metrics: Dict[str, Dict],
) -> Dict[str, int]:
    logger.info("=" * 60)
    logger.info("STEP 3: Running FallbackRouter")

    from core.recommendation.fallback_router import FallbackRouter

    router = FallbackRouter(cfg)

    task_names = [t["name"] for t in cfg.get("tasks", [])]
    routing = router.route_all(
        task_names=task_names,
        lgbm_models=lgbm_models,
        lgbm_fidelities=fidelities,
        teacher_metrics_all=teacher_metrics,
        rule_engine=None,  # Rule engine not instantiated in this test
    )

    logger.info("  Routing results:")
    for task_name in task_names:
        layer = routing.get(task_name, 3)
        explanation = router.explain(task_name, routing)
        logger.info("    %s", explanation)

    return routing


# ---------------------------------------------------------------------------
# Step 4: Get predictions from LGBM models
# ---------------------------------------------------------------------------

def get_predictions(
    cfg: Dict[str, Any],
    lgbm_models: Dict[str, Any],
    feature_cols: Dict[str, List[str]],
    all_feature_cols: List[str],
    feature_dict: Dict[str, Any],
    routing: Dict[str, int],
) -> Dict[str, Dict[str, Any]]:
    logger.info("=" * 60)
    logger.info("STEP 4: Getting predictions from LGBM models")

    # Build task_type lookup
    task_type_map: Dict[str, str] = {
        t["name"]: t.get("type", "binary")
        for t in cfg.get("tasks", [])
    }

    predictions: Dict[str, Dict[str, Any]] = {}

    for task_name, layer in routing.items():
        model = lgbm_models.get(task_name)
        if model is None or layer == 3:
            logger.info(
                "  %-35s → Layer %d  (no LGBM model or Layer 3 fallback)",
                task_name, layer,
            )
            predictions[task_name] = {
                "layer": layer,
                "prediction": None,
                "task_type": task_type_map.get(task_name, "binary"),
                "raw": None,
            }
            continue

        # Build feature vector aligned to this model's feature columns
        cols = feature_cols.get(task_name, all_feature_cols)
        x = np.array(
            [float(feature_dict.get(c, 0.0) or 0.0) for c in cols],
            dtype=np.float32,
        ).reshape(1, -1)

        try:
            raw_pred = model.predict(x)
        except Exception as exc:
            logger.error("  %-35s → predict() failed: %s", task_name, exc)
            predictions[task_name] = {
                "layer": layer,
                "prediction": None,
                "task_type": task_type_map.get(task_name, "binary"),
                "raw": None,
            }
            continue

        task_type = task_type_map.get(task_name, "binary")

        # Normalise output: custom objective outputs raw logits, apply sigmoid
        if task_type == "binary":
            raw_logit = float(raw_pred[0])
            pred_value = 1.0 / (1.0 + np.exp(-raw_logit))  # sigmoid
        elif task_type == "multiclass":
            # LGBM multiclass predict() returns class probabilities
            pred_value = raw_pred[0].tolist() if hasattr(raw_pred[0], "tolist") else list(raw_pred[0])
        else:
            # regression
            pred_value = float(raw_pred[0])

        predictions[task_name] = {
            "layer": layer,
            "prediction": pred_value,
            "task_type": task_type,
            "raw": raw_pred,
        }

        if task_type == "binary":
            logger.info(
                "  %-35s → Layer %d  prob=%.4f",
                task_name, layer, pred_value,
            )
        elif task_type == "multiclass":
            best_class = int(np.argmax(raw_pred[0]))
            logger.info(
                "  %-35s → Layer %d  best_class=%d  probs=%s",
                task_name, layer, best_class,
                [f"{p:.3f}" for p in (pred_value[:5] if isinstance(pred_value, list) else [])],
            )
        else:
            logger.info(
                "  %-35s → Layer %d  value=%.4f",
                task_name, layer, pred_value,
            )

    return predictions


# ---------------------------------------------------------------------------
# Step 5: Extract contributing features (top LGBM gain importance)
# ---------------------------------------------------------------------------

def extract_contributing_features(
    lgbm_models: Dict[str, Any],
    feature_cols: Dict[str, List[str]],
    top_k: int = 5,
) -> Dict[str, List[Tuple[str, float]]]:
    """Extract top-K gain importance features per task from LGBM models.

    Returns:
        {task_name: [(feature_name, importance_score), ...]} sorted descending
    """
    logger.info("=" * 60)
    logger.info("STEP 5: Extracting contributing features (top-%d LGBM gain)", top_k)

    contributing: Dict[str, List[Tuple[str, float]]] = {}

    for task_name, model in lgbm_models.items():
        cols = feature_cols.get(task_name, [])
        try:
            importances = model.feature_importance(importance_type="gain")
        except Exception as exc:
            logger.warning(
                "  %-35s — feature_importance() failed: %s", task_name, exc
            )
            contributing[task_name] = []
            continue

        if len(cols) != len(importances):
            # Mismatch — use model's internal feature names
            try:
                cols = model.feature_name()
            except Exception:
                cols = [f"f{i}" for i in range(len(importances))]

        # Sort by importance descending
        pairs = sorted(
            zip(cols, importances.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        top_pairs = pairs[:top_k]
        contributing[task_name] = top_pairs

        logger.info(
            "  %-35s — top features: %s",
            task_name,
            ", ".join(f"{n}({v:.1f})" for n, v in top_pairs),
        )

    return contributing


# ---------------------------------------------------------------------------
# Step 6: Run FactExtractor
# ---------------------------------------------------------------------------

def run_fact_extractor(
    feature_dict: Dict[str, Any],
) -> List[str]:
    logger.info("=" * 60)
    logger.info("STEP 6: Running FactExtractor (Agent 1)")
    logger.info("  config: %s", FACT_EXTRACTION_YAML)

    from core.recommendation.reason.fact_extractor import FactExtractor

    extractor = FactExtractor(str(FACT_EXTRACTION_YAML))
    logger.info("  Rules loaded: %d", extractor.rule_count)

    facts = extractor.extract(feature_dict)
    logger.info("  Facts extracted: %d", len(facts))
    for fact in facts:
        logger.info("    - %s", fact)

    if not facts:
        logger.info("  (No facts matched — feature names in sample may not overlap "
                    "with required_features in fact_extraction.yaml)")

    return facts


# ---------------------------------------------------------------------------
# Step 7: Run TemplateEngine L1
# ---------------------------------------------------------------------------

def run_template_engine(
    cfg: Dict[str, Any],
    customer_id: str,
    contributing: Dict[str, List[Tuple[str, float]]],
    predictions: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    logger.info("=" * 60)
    logger.info("STEP 7: Running TemplateEngine L1 (Agent 2 — first stage)")

    from core.recommendation.reason.template_engine import TemplateEngine

    engine = TemplateEngine(cfg)

    # Pick tasks that have an actual prediction (not Layer 3 fallback with None)
    scored_tasks = [
        (t_name, info)
        for t_name, info in predictions.items()
        if info.get("prediction") is not None
    ]

    if not scored_tasks:
        logger.warning("  No tasks with predictions — skipping TemplateEngine")
        return {}

    # Sort by prediction score for ranking (binary: by prob; multiclass: by max_prob)
    def sort_key(item):
        task_name, info = item
        pred = info["prediction"]
        if isinstance(pred, list):
            return max(pred)
        return float(pred) if pred is not None else 0.0

    scored_tasks.sort(key=sort_key, reverse=True)

    # Generate reason for the top-3 highest-scoring tasks
    reason_outputs: Dict[str, Dict[str, Any]] = {}
    for task_name, info in scored_tasks[:3]:
        task_type = info.get("task_type", "binary")
        ig_top = contributing.get(task_name, [])

        # Convert to (feature_name, ig_score) tuples
        ig_top_features = [(name, score) for name, score in ig_top]

        result = engine.generate_reason(
            customer_id=customer_id,
            item_id=task_name,
            ig_top_features=ig_top_features,
            segment="WARMSTART",
            task_type=task_type,
            task_name=task_name,
        )
        reason_outputs[task_name] = result

        reasons_text = [r.get("text", "") for r in result.get("reasons", [])]
        logger.info(
            "  %-35s — %d reason(s) generated",
            task_name, len(reasons_text),
        )
        for i, text in enumerate(reasons_text, 1):
            logger.info("    [%d] %s", i, text)

    return reason_outputs


# ---------------------------------------------------------------------------
# Step 8: Run SelfChecker
# ---------------------------------------------------------------------------

def run_self_checker(
    cfg: Dict[str, Any],
    reason_outputs: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    logger.info("=" * 60)
    logger.info("STEP 8: Running SelfChecker (Agent 3 — compliance validation)")
    logger.info("  NOTE: enable_llm_check is forced to False (no Bedrock call)")

    from core.recommendation.reason.self_checker import SelfChecker

    # Force LLM check off for local test (no Bedrock call)
    cfg_copy = dict(cfg)
    reason_copy = dict(cfg_copy.get("reason", {}))
    sc_copy = dict(reason_copy.get("self_checker", {}))
    sc_copy["enable_llm_check"] = False
    reason_copy["self_checker"] = sc_copy
    cfg_copy["reason"] = reason_copy

    checker = SelfChecker(cfg_copy, llm_provider=None)

    check_results: Dict[str, Any] = {}

    for task_name, reason_output in reason_outputs.items():
        reasons = reason_output.get("reasons", [])
        if not reasons:
            continue

        # Check each reason text
        task_checks = []
        for reason in reasons:
            text = reason.get("text", "")
            if not text:
                continue
            result = checker.check(reason_text=text, source_context=None)
            task_checks.append({
                "text": text,
                "verdict": result.verdict,
                "compliance_passed": result.compliance_passed,
                "injection_safe": result.injection_safe,
                "factuality_score": result.factuality_score,
                "violations": result.violations,
                "feedback": result.feedback,
            })

        check_results[task_name] = task_checks

        verdicts = [c["verdict"] for c in task_checks]
        all_pass = all(v == "pass" for v in verdicts)
        logger.info(
            "  %-35s — %d checks, all_pass=%s, verdicts=%s",
            task_name, len(verdicts), all_pass, verdicts,
        )
        if not all_pass:
            for c in task_checks:
                if c["verdict"] != "pass":
                    logger.warning(
                        "    FAIL text='%s...' feedback='%s'",
                        c["text"][:60], c["feedback"],
                    )

    return check_results


# ---------------------------------------------------------------------------
# Final: Print structured summary
# ---------------------------------------------------------------------------

def print_summary(
    customer_id: str,
    routing: Dict[str, int],
    predictions: Dict[str, Dict[str, Any]],
    contributing: Dict[str, List[Tuple[str, float]]],
    facts: List[str],
    reason_outputs: Dict[str, Dict[str, Any]],
    check_results: Dict[str, Any],
) -> None:
    sep = "=" * 60
    logger.info(sep)
    logger.info("FINAL SUMMARY")
    logger.info(sep)
    logger.info("Customer ID : %s", customer_id)
    logger.info("")

    logger.info("--- Routing ---")
    for task_name, layer in sorted(routing.items()):
        pred = predictions.get(task_name, {})
        pred_val = pred.get("prediction")
        pred_str = f"{pred_val:.4f}" if isinstance(pred_val, float) else str(pred_val)
        logger.info("  %-35s  Layer=%d  pred=%s", task_name, layer, pred_str)

    logger.info("")
    logger.info("--- Top Contributing Features (per task) ---")
    for task_name, feats in contributing.items():
        top = ", ".join(f"{n}({v:.0f})" for n, v in feats[:3])
        logger.info("  %-35s  %s", task_name, top)

    logger.info("")
    logger.info("--- FactExtractor Output (Agent 1) ---")
    if facts:
        for f in facts:
            logger.info("  - %s", f)
    else:
        logger.info("  (no facts matched)")

    logger.info("")
    logger.info("--- TemplateEngine L1 Output (Agent 2) ---")
    for task_name, ro in reason_outputs.items():
        for r in ro.get("reasons", []):
            logger.info(
                "  [%s rank=%d] %s",
                task_name, r.get("rank", "?"), r.get("text", ""),
            )

    logger.info("")
    logger.info("--- SelfChecker Results (Agent 3) ---")
    for task_name, checks in check_results.items():
        for c in checks:
            status = "PASS" if c["verdict"] == "pass" else f"!! {c['verdict'].upper()}"
            logger.info(
                "  [%s] %s — '%s...'",
                status, task_name, c["text"][:50],
            )

    logger.info("")
    logger.info(sep)
    logger.info("STOP: AsyncReasonOrchestrator L2a (Bedrock LLM rewrite) NOT called.")
    logger.info(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("Local Serving Pipeline Test -- starts")
    logger.info("Repo root: %s", REPO_ROOT)

    # Step 0: Config
    cfg = load_config()

    # Derive the union of all feature columns across all tasks
    # (used to align feature_dict → feature_array for models without metadata)
    all_feature_cols: List[str] = []
    seen: set = set()
    for t in cfg.get("tasks", []):
        pass  # placeholder; filled by Step 1 metadata

    # Step 1: Load models
    lgbm_models, feature_cols, fidelities, teacher_metrics = load_lgbm_models()

    # Build union of all feature columns in the order of the first model encountered
    for cols in feature_cols.values():
        for c in cols:
            if c not in seen:
                seen.add(c)
                all_feature_cols.append(c)
    logger.info("Union of feature columns across all models: %d", len(all_feature_cols))

    # Step 2: Load sample customer
    customer_id, feature_dict, feat_array = load_sample_customer(all_feature_cols)

    # Step 3: FallbackRouter
    routing = run_fallback_router(cfg, lgbm_models, fidelities, teacher_metrics)

    # Step 4: Predictions
    predictions = get_predictions(
        cfg, lgbm_models, feature_cols, all_feature_cols, feature_dict, routing
    )

    # Step 5: Contributing features
    contributing = extract_contributing_features(lgbm_models, feature_cols, top_k=5)

    # Step 6: FactExtractor
    facts = run_fact_extractor(feature_dict)

    # Step 7: TemplateEngine L1
    reason_outputs = run_template_engine(cfg, customer_id, contributing, predictions)

    # Step 8: SelfChecker
    check_results = run_self_checker(cfg, reason_outputs)

    # Final summary
    print_summary(
        customer_id=customer_id,
        routing=routing,
        predictions=predictions,
        contributing=contributing,
        facts=facts,
        reason_outputs=reason_outputs,
        check_results=check_results,
    )

    logger.info("Local serving pipeline test complete.")


if __name__ == "__main__":
    main()
