"""
Lambda 200-Customer Serving Simulation
=======================================

Simulates Lambda serving for 200 sampled customers by invoking the local
Lambda handler pipeline directly (no actual AWS Lambda deployment).

Pipeline per customer:
    1. Load LGBM models (once at startup)
    2. FallbackRouter.route_all()
    3. LGBM predict with sigmoid for binary tasks
    4. Contributing features extraction (top-5 LGBM gain)
    5. FactExtractor (Agent 1)
    6. TemplateEngine L1 (Agent 2)
    7. SelfChecker (Agent 3) — enable_llm_check=False
    8. Bedrock L2a reason rewrite — calls BedrockProvider from llm_provider.py
       Falls back to L1 text if Bedrock call fails.

Output:
    outputs/lambda_test_200_results.json

Run:
    PYTHONPATH=. python scripts/test_lambda_200.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Windows: reconfigure stdout/stderr to UTF-8 for Korean text
# ---------------------------------------------------------------------------
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except AttributeError:
        pass
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except AttributeError:
        pass

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("test_lambda_200")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = _REPO_ROOT
PIPELINE_YAML = REPO_ROOT / "configs" / "pipeline.yaml"
DATASET_YAML = REPO_ROOT / "configs" / "datasets" / "santander.yaml"
DISTILLATION_DIR = REPO_ROOT / "outputs" / "distillation_v2" / "model"
PHASE0_PARQUET = REPO_ROOT / "outputs" / "phase0_v12" / "santander_final.parquet"
FACT_EXTRACTION_YAML = REPO_ROOT / "configs" / "financial" / "fact_extraction.yaml"
SAMPLE_JSON = REPO_ROOT / "outputs" / "test_sample_5.json"
RESULTS_JSON = REPO_ROOT / "outputs" / "lambda_test_200_results.json"


# ---------------------------------------------------------------------------
# Step 0: Load config
# ---------------------------------------------------------------------------

def load_config() -> Dict[str, Any]:
    logger.info("STEP 0: Loading merged config")
    from core.pipeline.config import load_merged_config
    cfg = load_merged_config(PIPELINE_YAML, DATASET_YAML)
    tasks = cfg.get("tasks", [])
    logger.info("  tasks loaded: %d", len(tasks))
    return cfg


# ---------------------------------------------------------------------------
# Step 1: Load LGBM models (once at startup)
# ---------------------------------------------------------------------------

def load_lgbm_models() -> Tuple[
    Dict[str, Any],
    Dict[str, List[str]],
    Dict[str, Dict],
    Dict[str, Dict],
]:
    """Returns (lgbm_models, feature_cols, fidelities, teacher_metrics)."""
    logger.info("STEP 1: Loading LGBM models from %s", DISTILLATION_DIR)

    try:
        import lightgbm as lgb
    except ImportError:
        logger.error("lightgbm not installed. Run: pip install lightgbm")
        sys.exit(1)

    summary_path = DISTILLATION_DIR / "distillation_summary.json"
    summary: Dict[str, Any] = {}
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        logger.info("  distillation_summary.json loaded")

    lgbm_models: Dict[str, Any] = {}
    feature_cols: Dict[str, List[str]] = {}
    fidelities: Dict[str, Dict] = {}
    teacher_metrics: Dict[str, Dict] = {}

    for task_dir in sorted(DISTILLATION_DIR.iterdir()):
        if not task_dir.is_dir():
            continue
        task_name = task_dir.name
        model_path = task_dir / "model.lgbm"
        metadata_path = task_dir / "metadata.json"
        fidelity_path = task_dir / "fidelity.json"

        if not model_path.exists():
            logger.debug("  %-35s — model.lgbm not found, skipping", task_name)
            continue

        try:
            booster = lgb.Booster(model_file=str(model_path))
            lgbm_models[task_name] = booster
        except Exception as exc:
            logger.error("  %-35s — failed to load model: %s", task_name, exc)
            continue

        if metadata_path.exists():
            with open(metadata_path) as f:
                meta = json.load(f)
            cols = meta.get("feature_columns", [])
            feature_cols[task_name] = cols
            logger.info(
                "  %-35s — loaded (%d trees, %d features)",
                task_name, meta.get("num_trees", "?"), len(cols),
            )
        else:
            feature_cols[task_name] = []

        if fidelity_path.exists():
            with open(fidelity_path) as f:
                fid = json.load(f)
            fidelities[task_name] = fid.get("metrics", fid)

        # Reconstruct teacher quality proxies
        summary_fidelity = summary.get("fidelity", {}).get("per_task", {})
        task_fid = summary_fidelity.get(task_name, {})
        task_metrics = task_fid.get("metrics", {})

        task_type = None
        if metadata_path.exists():
            with open(metadata_path) as f_meta:
                meta2 = json.load(f_meta)
            task_type = meta2.get("task_type", "binary")

        if task_type == "binary":
            student_auc = 1.0 - task_metrics.get("auc_gap", 0.05)
            teacher_metrics[task_name] = {"auc_roc": max(0.5, student_auc)}
        elif task_type == "multiclass":
            teacher_f1 = task_metrics.get("teacher_f1_macro", 0.3)
            teacher_metrics[task_name] = {"f1_ratio": teacher_f1 * 4}
        elif task_type == "regression":
            teacher_metrics[task_name] = {"r2": 0.1}
        else:
            teacher_metrics[task_name] = {"auc_roc": 0.7}

    logger.info("  Total models loaded: %d", len(lgbm_models))
    return lgbm_models, feature_cols, fidelities, teacher_metrics


# ---------------------------------------------------------------------------
# Step 2: Load 200 customers from parquet via DuckDB
# ---------------------------------------------------------------------------

def load_customers_duckdb(
    customer_ids: List[int],
    sample_json: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Dict[str, Any]], Dict[str, str]]:
    """Load features for all 200 customers using DuckDB.

    Returns:
        (ordered_customer_ids, feature_dicts, cluster_map)
        where cluster_map = {customer_id_str: cluster_id_str}
    """
    logger.info("STEP 2: Loading %d customers from parquet via DuckDB", len(customer_ids))

    if not PHASE0_PARQUET.exists():
        logger.error("Phase0 parquet not found: %s", PHASE0_PARQUET)
        sys.exit(1)

    try:
        import duckdb
    except ImportError:
        logger.error("duckdb not installed. Run: pip install duckdb")
        sys.exit(1)

    # Build cluster_map: {customer_id_str: cluster_id_str}
    cluster_map: Dict[str, str] = {}
    for cluster_id, cluster_data in sample_json.get("per_cluster", {}).items():
        for cid in cluster_data.get("customer_ids", []):
            cluster_map[str(cid)] = str(cluster_id)

    con = duckdb.connect()
    parquet_path = PHASE0_PARQUET.as_posix()

    # Get available columns
    available_cols_result = con.execute(
        f"SELECT * FROM read_parquet('{parquet_path}') LIMIT 0"
    ).description
    available_cols = [desc[0] for desc in available_cols_result]
    logger.info("  Parquet columns available: %d", len(available_cols))

    # Detect customer ID column
    cust_id_col = None
    for cand in ("ncodpers", "customer_id", "cust_id", "id"):
        if cand in available_cols:
            cust_id_col = cand
            break
    if cust_id_col is None:
        logger.error("Cannot find customer ID column in parquet")
        sys.exit(1)
    logger.info("  Customer ID column: %s", cust_id_col)

    # Load rows for all 200 customer IDs via DuckDB SQL
    ids_str = ", ".join(str(cid) for cid in customer_ids)
    query = (
        f"SELECT * FROM read_parquet('{parquet_path}') "
        f"WHERE {cust_id_col} IN ({ids_str})"
    )
    result_df = con.execute(query).fetchdf()
    con.close()

    logger.info("  Rows fetched from parquet: %d", len(result_df))

    # Build feature dicts
    ordered_ids: List[str] = []
    feature_dicts: Dict[str, Dict[str, Any]] = {}

    for _, row in result_df.iterrows():
        cid_val = row[cust_id_col]
        cid_str = str(int(cid_val)) if not isinstance(cid_val, str) else cid_val
        ordered_ids.append(cid_str)
        feature_dicts[cid_str] = row.to_dict()

    # Warn if any requested IDs were not found
    found_set = set(ordered_ids)
    missing = [str(cid) for cid in customer_ids if str(cid) not in found_set]
    if missing:
        logger.warning("  %d customer IDs not found in parquet: %s", len(missing), missing[:10])

    logger.info("  Customer feature dicts built: %d", len(feature_dicts))
    return ordered_ids, feature_dicts, cluster_map


# ---------------------------------------------------------------------------
# Step 3: FallbackRouter (once, routing is same for all customers)
# ---------------------------------------------------------------------------

def build_routing(
    cfg: Dict[str, Any],
    lgbm_models: Dict[str, Any],
    fidelities: Dict[str, Dict],
    teacher_metrics: Dict[str, Dict],
) -> Dict[str, int]:
    logger.info("STEP 3: Building FallbackRouter routing (once for all customers)")
    from core.recommendation.fallback_router import FallbackRouter

    router = FallbackRouter(cfg)
    task_names = [t["name"] for t in cfg.get("tasks", [])]
    routing = router.route_all(
        task_names=task_names,
        lgbm_models=lgbm_models,
        lgbm_fidelities=fidelities,
        teacher_metrics_all=teacher_metrics,
        rule_engine=None,
    )

    for task_name in task_names:
        layer = routing.get(task_name, 3)
        logger.info("    %-35s → Layer %d", task_name, layer)

    return routing


# ---------------------------------------------------------------------------
# Step 4: Extract contributing features (once per model, not per customer)
# ---------------------------------------------------------------------------

def build_contributing_features(
    lgbm_models: Dict[str, Any],
    feature_cols: Dict[str, List[str]],
    top_k: int = 5,
) -> Dict[str, List[Tuple[str, float]]]:
    """Top-K LGBM gain importance per task (model-level, not customer-level)."""
    logger.info("STEP 4: Extracting top-%d contributing features per task", top_k)
    contributing: Dict[str, List[Tuple[str, float]]] = {}

    for task_name, model in lgbm_models.items():
        cols = feature_cols.get(task_name, [])
        try:
            importances = model.feature_importance(importance_type="gain")
        except Exception as exc:
            logger.warning("  %-35s — feature_importance() failed: %s", task_name, exc)
            contributing[task_name] = []
            continue

        if len(cols) != len(importances):
            try:
                cols = model.feature_name()
            except Exception:
                cols = [f"f{i}" for i in range(len(importances))]

        pairs = sorted(
            zip(cols, importances.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        contributing[task_name] = pairs[:top_k]

    return contributing


# ---------------------------------------------------------------------------
# BedrockProvider initialisation (once at startup)
# ---------------------------------------------------------------------------

def init_bedrock_provider(cfg: Dict[str, Any]):
    """Initialise BedrockProvider from config. Returns provider or None."""
    from core.recommendation.reason.llm_provider import BedrockProvider

    llm_cfg = cfg.get("llm_provider", {}).get("bedrock", {})
    models_cfg = llm_cfg.get("models", {})
    reason_cfg = models_cfg.get(
        "reason_generation", llm_cfg.get("default", {})
    )
    bedrock_config = {**llm_cfg, **reason_cfg}

    try:
        provider = BedrockProvider(bedrock_config)
        logger.info(
            "BedrockProvider initialised: model_id=%s region=%s",
            provider.model_id, provider.region,
        )
        return provider
    except Exception as exc:
        logger.warning("BedrockProvider init failed: %s — L2a will be skipped", exc)
        return None


# ---------------------------------------------------------------------------
# Per-customer processing
# ---------------------------------------------------------------------------

def predict_for_customer(
    customer_id: str,
    feature_dict: Dict[str, Any],
    cfg: Dict[str, Any],
    lgbm_models: Dict[str, Any],
    feature_cols: Dict[str, List[str]],
    routing: Dict[str, int],
) -> Tuple[Dict[str, Dict[str, Any]], float]:
    """Run LGBM predict for all tasks for one customer.

    Returns (predictions, elapsed_ms).
    """
    t0 = time.perf_counter()
    task_type_map: Dict[str, str] = {
        t["name"]: t.get("type", "binary") for t in cfg.get("tasks", [])
    }
    predictions: Dict[str, Dict[str, Any]] = {}

    for task_name, layer in routing.items():
        model = lgbm_models.get(task_name)
        if model is None or layer == 3:
            predictions[task_name] = {
                "layer": layer,
                "prediction": None,
                "task_type": task_type_map.get(task_name, "binary"),
            }
            continue

        cols = feature_cols.get(task_name, [])
        x = np.array(
            [float(feature_dict.get(c, 0.0) or 0.0) for c in cols],
            dtype=np.float32,
        ).reshape(1, -1)

        try:
            raw_pred = model.predict(x)
        except Exception as exc:
            logger.debug("customer=%s task=%s predict failed: %s", customer_id, task_name, exc)
            predictions[task_name] = {
                "layer": layer,
                "prediction": None,
                "task_type": task_type_map.get(task_name, "binary"),
            }
            continue

        task_type = task_type_map.get(task_name, "binary")
        if task_type == "binary":
            raw_logit = float(raw_pred[0])
            pred_value = 1.0 / (1.0 + np.exp(-raw_logit))
        elif task_type == "multiclass":
            pred_value = (
                raw_pred[0].tolist()
                if hasattr(raw_pred[0], "tolist")
                else list(raw_pred[0])
            )
        else:
            pred_value = float(raw_pred[0])

        predictions[task_name] = {
            "layer": layer,
            "prediction": pred_value,
            "task_type": task_type,
        }

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return predictions, elapsed_ms


def run_reason_pipeline_for_customer(
    customer_id: str,
    feature_dict: Dict[str, Any],
    predictions: Dict[str, Dict[str, Any]],
    contributing: Dict[str, List[Tuple[str, float]]],
    cfg: Dict[str, Any],
    fact_extractor,
    template_engine,
    self_checker,
    bedrock_provider,
    scorer=None,
    critique_provider=None,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Run Agent 1/2/3 + L2a Bedrock + L2a critique for one customer.

    Returns (reason_result, latency_breakdown_ms).
    """
    latency: Dict[str, float] = {
        "reason_l1": 0.0,
        "reason_l2a": 0.0,
        "reason_critique": 0.0,
        "selfcheck": 0.0,
    }

    # --- Agent 1: FactExtractor ---
    facts = fact_extractor.extract(feature_dict)

    # --- Select top tasks via FD-TVS scoring (or raw prediction fallback) ---
    pred_dict = {
        t_name: (float(info["prediction"]) if not isinstance(info["prediction"], list)
                 else max(info["prediction"]))
        for t_name, info in predictions.items()
        if info.get("prediction") is not None
    }

    if scorer is not None:
        # FD-TVS: compute customer-level score once, then rank tasks by
        # (dynamic_weight * prediction) to get personalized task ordering.
        context = {
            "modifier_segment": str(feature_dict.get("segment", "UNKNOWN")),
            "churn_prob": pred_dict.get("churn_signal", 0.0),
            "engagement_score": float(feature_dict.get("is_active", 0)),
            "income": float(feature_dict.get("income", 0) or 0),
            "product_diversity": float(feature_dict.get("product_diversity", 0) or 0),
            "n_messages_7d": 0,
            "channel": "app_push",
        }
        # Get dynamic weights for this customer's segment + behavior
        eff_weights = scorer._compute_dynamic_weights(context)
        # Rank tasks by weight * prediction (personalized per customer)
        task_scores = {
            t: eff_weights.get(t, 0.0) * pred_dict[t]
            for t in pred_dict
        }
        scored_tasks = sorted(task_scores.items(), key=lambda x: -x[1])
        top_tasks = [
            (t_name, predictions[t_name]) for t_name, _ in scored_tasks[:3]
        ]
    else:
        # Fallback: raw prediction sort
        scored_tasks = [
            (t_name, info)
            for t_name, info in predictions.items()
            if info.get("prediction") is not None
        ]
        scored_tasks.sort(
            key=lambda x: float(x[1]["prediction"]) if not isinstance(x[1]["prediction"], list) else max(x[1]["prediction"]),
            reverse=True,
        )
        top_tasks = scored_tasks[:3]

    # --- Agent 2: TemplateEngine L1 ---
    t1 = time.perf_counter()
    reason_outputs: Dict[str, Dict[str, Any]] = {}
    for task_name, info in top_tasks:
        task_type = info.get("task_type", "binary")
        ig_top = contributing.get(task_name, [])
        ig_top_features = [(n, s) for n, s in ig_top]
        result = template_engine.generate_reason(
            customer_id=customer_id,
            item_id=task_name,
            ig_top_features=ig_top_features,
            segment="WARMSTART",
            task_type=task_type,
            task_name=task_name,
        )
        reason_outputs[task_name] = result
    latency["reason_l1"] = (time.perf_counter() - t1) * 1000.0

    # --- Agent 2 L2a: Bedrock reason rewrite ---
    t2 = time.perf_counter()
    l2a_outputs: Dict[str, str] = {}
    bedrock_success = 0
    bedrock_attempts = 0

    if bedrock_provider is not None:
        facts_str = "; ".join(facts) if facts else "N/A"
        for task_name, reason_output in reason_outputs.items():
            reasons = reason_output.get("reasons", [])
            if not reasons:
                continue
            l1_text = reasons[0].get("text", "")
            if not l1_text:
                continue

            prompt = (
                f"원래 사유: {l1_text}\n"
                f"고객 특성: {facts_str}\n\n"
                "위 추천사유를 다듬어서 출력하세요."
            )
            system_msg = (
                "당신은 금융 상품 추천사유 작성자입니다. "
                "입력된 추천사유를 고객에게 전달할 자연스러운 한국어 1~2문장으로 다듬어 출력하세요. "
                "분석, 검토, 마크다운, 제목, 목록 없이 오직 다듬어진 문장만 출력하세요."
            )

            bedrock_attempts += 1
            try:
                rewritten = bedrock_provider.generate(prompt, system=system_msg)
                if rewritten:
                    l2a_outputs[task_name] = rewritten.strip()
                    bedrock_success += 1
                else:
                    l2a_outputs[task_name] = l1_text
                    logger.warning(
                        "customer=%s task=%s: Bedrock returned empty, falling back to L1",
                        customer_id, task_name,
                    )
            except Exception as exc:
                l2a_outputs[task_name] = l1_text
                logger.warning(
                    "customer=%s task=%s: Bedrock call failed (%s), falling back to L1",
                    customer_id, task_name, exc,
                )
    else:
        # No Bedrock provider — use L1 text
        for task_name, reason_output in reason_outputs.items():
            reasons = reason_output.get("reasons", [])
            l1_text = reasons[0].get("text", "") if reasons else ""
            l2a_outputs[task_name] = l1_text

    latency["reason_l2a"] = (time.perf_counter() - t2) * 1000.0

    # --- L2a Self-Critique (Sonnet) ---
    t2c = time.perf_counter()
    critique_outputs: Dict[str, str] = {}
    critique_success = 0
    critique_attempts = 0

    if critique_provider is not None:
        for task_name, l2a_text in l2a_outputs.items():
            if not l2a_text:
                continue
            critique_prompt = f"추천사유: {l2a_text}"
            critique_system = (
                "당신은 금융 규제 준수 검토자입니다. "
                "추천사유를 검토하고, 문제가 없으면 '통과'만 출력하세요. "
                "문제가 있으면 수정된 문장 1~2줄만 출력하세요. "
                "검토 기준: 자연스러운 한국어, 과장 표현 없음, 금소법 적합성 원칙 준수. "
                "분석 보고서를 쓰지 마세요."
            )
            critique_attempts += 1
            try:
                critique_result = critique_provider.generate(
                    critique_prompt, system=critique_system,
                )
                if critique_result:
                    critique_outputs[task_name] = critique_result.strip()
                    critique_success += 1
                    # If critique suggests revision, use it
                    if "통과" not in critique_result:
                        l2a_outputs[task_name] = critique_result.strip()
            except Exception as exc:
                logger.warning(
                    "customer=%s task=%s: critique failed (%s)",
                    customer_id, task_name, exc,
                )

    latency["reason_critique"] = (time.perf_counter() - t2c) * 1000.0

    # --- Agent 3: SelfChecker (factuality check via Haiku) ---
    t3 = time.perf_counter()

    selfcheck_results: Dict[str, Any] = {}
    for task_name in l2a_outputs:
        text = l2a_outputs.get(task_name, "")
        if not text:
            continue
        check = self_checker.check(reason_text=text, source_context=None)
        selfcheck_results[task_name] = {
            "verdict": check.verdict,
            "compliance_passed": check.compliance_passed,
            "injection_safe": check.injection_safe,
            "factuality_score": check.factuality_score,
            "violations": check.violations,
        }

    latency["selfcheck"] = (time.perf_counter() - t3) * 1000.0

    # --- Assemble result ---
    # Determine overall layer_used (most common non-3 layer, else 3)
    layers_used = [
        info["layer"]
        for info in predictions.values()
        if info.get("prediction") is not None
    ]
    layer_used = min(layers_used) if layers_used else 3

    # Serialize predictions (convert numpy to Python natives)
    serialized_preds: Dict[str, Any] = {}
    for task_name, info in predictions.items():
        pred = info.get("prediction")
        if isinstance(pred, (np.floating, np.integer)):
            pred = float(pred)
        elif isinstance(pred, np.ndarray):
            pred = pred.tolist()
        serialized_preds[task_name] = {
            "layer": info["layer"],
            "task_type": info["task_type"],
            "prediction": pred,
        }

    # Top task L1 and L2a reasons
    top_task = top_tasks[0][0] if top_tasks else None
    l1_reason = ""
    if top_task and top_task in reason_outputs:
        reasons = reason_outputs[top_task].get("reasons", [])
        l1_reason = reasons[0].get("text", "") if reasons else ""
    l2a_reason = l2a_outputs.get(top_task, "") if top_task else ""
    critique_out = critique_outputs.get(top_task, "") if top_task else ""

    # SelfChecker verdict summary
    all_verdicts = [r["verdict"] for r in selfcheck_results.values()]
    selfcheck_verdict = "pass" if all(v == "pass" for v in all_verdicts) else "fail"

    return {
        "customer_id": customer_id,
        "facts": facts,
        "layer_used": layer_used,
        "predictions": serialized_preds,
        "l1_reason": l1_reason,
        "l2a_reason": l2a_reason,
        "critique_output": critique_out,
        "selfcheck_verdict": selfcheck_verdict,
        "selfcheck_details": selfcheck_results,
        "bedrock_attempts": bedrock_attempts + critique_attempts,
        "bedrock_success": bedrock_success + critique_success,
    }, latency


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=" * 70)
    logger.info("Lambda 200-Customer Serving Simulation")
    logger.info("Repo root: %s", REPO_ROOT)
    logger.info("=" * 70)

    # --- Load sample IDs ---
    if not SAMPLE_JSON.exists():
        logger.error("Sample JSON not found: %s", SAMPLE_JSON)
        sys.exit(1)

    with open(SAMPLE_JSON, encoding="utf-8") as f:
        sample_data = json.load(f)

    all_customer_ids: List[int] = sample_data["all_customer_ids"]
    logger.info("Loaded %d customer IDs from %s", len(all_customer_ids), SAMPLE_JSON)

    # --- Step 0: Config ---
    cfg = load_config()

    # --- Step 1: LGBM models (once) ---
    lgbm_models, feature_cols, fidelities, teacher_metrics = load_lgbm_models()

    # Union of all feature columns
    all_feature_cols: List[str] = []
    seen_fc: set = set()
    for cols in feature_cols.values():
        for c in cols:
            if c not in seen_fc:
                seen_fc.add(c)
                all_feature_cols.append(c)
    logger.info("Union of feature columns across all models: %d", len(all_feature_cols))

    # --- Step 2: Load customers via DuckDB ---
    ordered_ids, feature_dicts, cluster_map = load_customers_duckdb(
        all_customer_ids, sample_data
    )

    # --- Step 3: FallbackRouter (once) ---
    routing = build_routing(cfg, lgbm_models, fidelities, teacher_metrics)

    # --- Step 4: Contributing features (once, model-level) ---
    contributing = build_contributing_features(lgbm_models, feature_cols, top_k=5)

    # --- Initialise agents (once at startup) ---
    logger.info("Initialising agents (once at startup)...")

    from core.recommendation.reason.fact_extractor import FactExtractor
    from core.recommendation.reason.template_engine import TemplateEngine
    from core.recommendation.reason.self_checker import SelfChecker

    fact_extractor = FactExtractor(str(FACT_EXTRACTION_YAML))
    template_engine = TemplateEngine(cfg)

    # SelfChecker with enable_llm_check=False always
    cfg_sc = dict(cfg)
    reason_sc = dict(cfg_sc.get("reason", {}))
    sc_block = dict(reason_sc.get("self_checker", {}))
    sc_block["enable_llm_check"] = False
    reason_sc["self_checker"] = sc_block
    cfg_sc["reason"] = reason_sc
    self_checker = SelfChecker(cfg_sc, llm_provider=None)

    # BedrockProvider for L2a (reason_generation)
    bedrock_provider = init_bedrock_provider(cfg)

    # BedrockProvider for L2a self-critique (reason_critique)
    critique_provider = None
    try:
        from core.recommendation.reason.llm_provider import BedrockProvider
        llm_cfg = cfg.get("llm_provider", {}).get("bedrock", {})
        models_cfg = llm_cfg.get("models", {})
        critique_cfg = models_cfg.get("reason_critique", llm_cfg.get("default", {}))
        critique_config = {**llm_cfg, **critique_cfg}
        critique_provider = BedrockProvider(critique_config)
        logger.info("CritiqueProvider initialised: model_id=%s", critique_provider.model_id)
    except Exception as exc:
        logger.warning("CritiqueProvider init failed: %s", exc)

    # SelfChecker with LLM factuality check (Haiku)
    factcheck_provider = None
    try:
        from core.recommendation.reason.llm_provider import BedrockProvider as _BP
        fc_cfg = models_cfg.get("factuality_check", llm_cfg.get("default", {}))
        fc_config = {**llm_cfg, **fc_cfg}
        factcheck_provider = _BP(fc_config)
        logger.info("FactCheckProvider initialised: model_id=%s", factcheck_provider.model_id)
        # Re-init SelfChecker with LLM enabled
        cfg_sc2 = dict(cfg)
        reason_sc2 = dict(cfg_sc2.get("reason", {}))
        sc_block2 = dict(reason_sc2.get("self_checker", {}))
        sc_block2["enable_llm_check"] = True
        reason_sc2["self_checker"] = sc_block2
        cfg_sc2["reason"] = reason_sc2
        self_checker = SelfChecker(cfg_sc2, llm_provider=factcheck_provider)
        logger.info("SelfChecker re-initialised with LLM factuality check (Haiku)")
    except Exception as exc:
        logger.warning("FactCheckProvider init failed: %s — SelfChecker stays rule-only", exc)

    # FD-TVS Scorer for personalized task ranking
    scorer = None
    try:
        from core.recommendation.scorer import ScorerRegistry
        scoring_cfg = cfg.get("scoring", {})
        scorer_name = scoring_cfg.get("method", "weighted_sum")
        # Bridge scoring config to scorer registry format
        scorer_sub: Dict[str, Any] = {}
        if scoring_cfg.get("weights"):
            scorer_sub["task_weights"] = scoring_cfg["weights"]
        if scoring_cfg.get("dna_modifier", {}).get("segment_weights"):
            scorer_sub["modifier_map"] = scoring_cfg["dna_modifier"]["segment_weights"]
        if scoring_cfg.get("fatigue", {}).get("decay_rate"):
            scorer_sub["fatigue_base_decay"] = scoring_cfg["fatigue"]["decay_rate"]
        if scoring_cfg.get("risk_penalty"):
            rp = scoring_cfg["risk_penalty"]
            scorer_sub["risk_threshold_churn"] = rp.get("max_churn_prob", 0.3)
        cfg.setdefault("scorer", {})[scorer_name] = scorer_sub
        scorer = ScorerRegistry.create(scorer_name, cfg)
        logger.info("FD-TVS scorer initialised: method=%s, task_weights=%s",
                     scorer_name, list(scorer_sub.get("task_weights", {}).keys()))
    except Exception as exc:
        logger.warning("Scorer init failed (%s), using raw prediction ranking", exc)

    # --- Process 200 customers sequentially ---
    logger.info("=" * 70)
    logger.info("Processing %d customers sequentially...", len(ordered_ids))
    logger.info("=" * 70)

    per_customer_results: List[Dict[str, Any]] = []
    total_latencies_ms: List[float] = []
    predict_latencies_ms: List[float] = []
    l1_latencies_ms: List[float] = []
    l2a_latencies_ms: List[float] = []
    selfcheck_latencies_ms: List[float] = []

    layer_counts: Dict[str, int] = {"L1": 0, "L2": 0, "L3": 0}
    selfcheck_pass_count = 0
    bedrock_total_attempts = 0
    bedrock_total_success = 0
    error_count = 0

    t_run_start = time.perf_counter()

    for idx, customer_id in enumerate(ordered_ids):
        feature_dict = feature_dicts.get(customer_id)
        if feature_dict is None:
            logger.warning("[%d/%d] customer_id=%s not in feature_dicts — skipping",
                           idx + 1, len(ordered_ids), customer_id)
            error_count += 1
            continue

        cluster_id = cluster_map.get(customer_id, "unknown")

        t_total_start = time.perf_counter()

        try:
            # Predict
            predictions, predict_ms = predict_for_customer(
                customer_id, feature_dict, cfg, lgbm_models, feature_cols, routing
            )

            # Reason pipeline
            result, latency = run_reason_pipeline_for_customer(
                customer_id=customer_id,
                feature_dict=feature_dict,
                predictions=predictions,
                contributing=contributing,
                cfg=cfg,
                fact_extractor=fact_extractor,
                template_engine=template_engine,
                self_checker=self_checker,
                bedrock_provider=bedrock_provider,
                scorer=scorer,
                critique_provider=critique_provider,
            )

        except Exception as exc:
            logger.error(
                "[%d/%d] customer_id=%s — UNHANDLED ERROR: %s",
                idx + 1, len(ordered_ids), customer_id, exc,
                exc_info=True,
            )
            error_count += 1
            continue

        total_ms = (time.perf_counter() - t_total_start) * 1000.0

        # Accumulate latencies
        total_latencies_ms.append(total_ms)
        predict_latencies_ms.append(predict_ms)
        l1_latencies_ms.append(latency["reason_l1"])
        l2a_latencies_ms.append(latency["reason_l2a"])
        selfcheck_latencies_ms.append(latency["selfcheck"])

        # Layer distribution
        layer_used = result["layer_used"]
        if layer_used == 1:
            layer_counts["L1"] += 1
        elif layer_used == 2:
            layer_counts["L2"] += 1
        else:
            layer_counts["L3"] += 1

        # Selfcheck pass
        if result["selfcheck_verdict"] == "pass":
            selfcheck_pass_count += 1

        # Bedrock stats
        bedrock_total_attempts += result["bedrock_attempts"]
        bedrock_total_success += result["bedrock_success"]

        # Assemble per-customer output record
        per_customer_results.append({
            "customer_id": customer_id,
            "cluster_id": cluster_id,
            "predictions": result["predictions"],
            "layer_used": layer_used,
            "l1_reason": result["l1_reason"],
            "l2a_reason": result["l2a_reason"],
            "critique_output": result.get("critique_output", ""),
            "selfcheck_verdict": result["selfcheck_verdict"],
            "latency_ms": {
                "total": round(total_ms, 2),
                "predict": round(predict_ms, 2),
                "reason_l1": round(latency["reason_l1"], 2),
                "reason_l2a": round(latency["reason_l2a"], 2),
                "reason_critique": round(latency.get("reason_critique", 0), 2),
                "selfcheck": round(latency["selfcheck"], 2),
            },
        })

        # Progress every 10 customers
        if (idx + 1) % 10 == 0 or (idx + 1) == len(ordered_ids):
            elapsed_run = (time.perf_counter() - t_run_start)
            logger.info(
                "[%3d/%d] customer_id=%-10s cluster=%-3s layer=%d "
                "total=%.1fms predict=%.1fms l1=%.1fms l2a=%.1fms crit=%.1fms sc=%.1fms "
                "| run_elapsed=%.1fs",
                idx + 1, len(ordered_ids), customer_id, cluster_id, layer_used,
                total_ms, predict_ms, latency["reason_l1"],
                latency["reason_l2a"], latency.get("reason_critique", 0),
                latency["selfcheck"],
                elapsed_run,
            )

    # --- Aggregate stats ---
    def pct(arr: List[float], p: float) -> float:
        if not arr:
            return 0.0
        return float(np.percentile(arr, p))

    def mean_(arr: List[float]) -> float:
        return float(np.mean(arr)) if arr else 0.0

    n_processed = len(per_customer_results)
    aggregate = {
        "n_requested": len(ordered_ids),
        "n_processed": n_processed,
        "n_errors": error_count,
        "latency_ms": {
            "total": {
                "mean": round(mean_(total_latencies_ms), 2),
                "p50": round(pct(total_latencies_ms, 50), 2),
                "p95": round(pct(total_latencies_ms, 95), 2),
            },
            "predict": {
                "mean": round(mean_(predict_latencies_ms), 2),
                "p50": round(pct(predict_latencies_ms, 50), 2),
                "p95": round(pct(predict_latencies_ms, 95), 2),
            },
            "reason_l1": {
                "mean": round(mean_(l1_latencies_ms), 2),
                "p50": round(pct(l1_latencies_ms, 50), 2),
                "p95": round(pct(l1_latencies_ms, 95), 2),
            },
            "reason_l2a": {
                "mean": round(mean_(l2a_latencies_ms), 2),
                "p50": round(pct(l2a_latencies_ms, 50), 2),
                "p95": round(pct(l2a_latencies_ms, 95), 2),
            },
            "selfcheck": {
                "mean": round(mean_(selfcheck_latencies_ms), 2),
                "p50": round(pct(selfcheck_latencies_ms, 50), 2),
                "p95": round(pct(selfcheck_latencies_ms, 95), 2),
            },
        },
        "layer_distribution": layer_counts,
        "selfcheck_pass_rate": round(selfcheck_pass_count / n_processed, 4) if n_processed else 0.0,
        "bedrock_success_rate": (
            round(bedrock_total_success / bedrock_total_attempts, 4)
            if bedrock_total_attempts > 0 else None
        ),
        "bedrock_attempts_total": bedrock_total_attempts,
        "bedrock_success_total": bedrock_total_success,
    }

    # --- Save results ---
    output = {
        "aggregate": aggregate,
        "per_customer": per_customer_results,
    }

    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    logger.info("=" * 70)
    logger.info("Results saved to: %s", RESULTS_JSON)
    logger.info("=" * 70)
    logger.info("AGGREGATE SUMMARY")
    logger.info("  Processed       : %d / %d (errors=%d)",
                n_processed, len(ordered_ids), error_count)
    logger.info("  Total latency   : mean=%.1fms  p50=%.1fms  p95=%.1fms",
                aggregate["latency_ms"]["total"]["mean"],
                aggregate["latency_ms"]["total"]["p50"],
                aggregate["latency_ms"]["total"]["p95"])
    logger.info("  Predict latency : mean=%.1fms  p50=%.1fms  p95=%.1fms",
                aggregate["latency_ms"]["predict"]["mean"],
                aggregate["latency_ms"]["predict"]["p50"],
                aggregate["latency_ms"]["predict"]["p95"])
    logger.info("  L1 reason       : mean=%.1fms  p50=%.1fms  p95=%.1fms",
                aggregate["latency_ms"]["reason_l1"]["mean"],
                aggregate["latency_ms"]["reason_l1"]["p50"],
                aggregate["latency_ms"]["reason_l1"]["p95"])
    logger.info("  L2a reason      : mean=%.1fms  p50=%.1fms  p95=%.1fms",
                aggregate["latency_ms"]["reason_l2a"]["mean"],
                aggregate["latency_ms"]["reason_l2a"]["p50"],
                aggregate["latency_ms"]["reason_l2a"]["p95"])
    logger.info("  SelfCheck       : mean=%.1fms  p50=%.1fms  p95=%.1fms",
                aggregate["latency_ms"]["selfcheck"]["mean"],
                aggregate["latency_ms"]["selfcheck"]["p50"],
                aggregate["latency_ms"]["selfcheck"]["p95"])
    logger.info("  Layer dist      : L1=%d  L2=%d  L3=%d",
                layer_counts["L1"], layer_counts["L2"], layer_counts["L3"])
    logger.info("  SelfCheck pass  : %.1f%%", aggregate["selfcheck_pass_rate"] * 100)
    if aggregate["bedrock_success_rate"] is not None:
        logger.info("  Bedrock success : %.1f%% (%d/%d)",
                    aggregate["bedrock_success_rate"] * 100,
                    bedrock_total_success, bedrock_total_attempts)
    else:
        logger.info("  Bedrock success : N/A (no Bedrock provider)")
    logger.info("=" * 70)
    logger.info("Lambda 200-customer simulation complete.")


if __name__ == "__main__":
    main()
