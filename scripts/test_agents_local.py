"""
Local test script for OpsAgent (CP1-CP7) and AuditAgent components.

Run with:
    PYTHONPATH=. python scripts/test_agents_local.py

Uses local file paths only - no Bedrock or S3 calls.
Reports saved to:
    outputs/ops_report.json
    outputs/audit_report.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("test_agents_local")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
OUTPUTS = BASE_DIR / "outputs"

PHASE0_DIR        = OUTPUTS / "phase0_v12"
TRAINING_DIR      = OUTPUTS / "ablation_v12" / "joint_full"
DISTILLATION_DIR  = OUTPUTS / "distillation_v2"
LAMBDA_RESULTS    = OUTPUTS / "lambda_test_200_results.json"

OPS_REPORT_PATH   = OUTPUTS / "ops_report.json"
AUDIT_REPORT_PATH = OUTPUTS / "audit_report.json"


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _load_json(path: Path, *, encoding: str = "utf-8") -> Any:
    """Load JSON from a local path. Returns None on error."""
    try:
        with open(path, encoding=encoding) as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("File not found: %s", path)
        return None
    except Exception as e:
        logger.warning("Failed to load %s: %s", path, e)
        return None


def _compute_feature_stats_summary(raw_stats: Dict) -> Dict:
    """Convert per-column feature_stats.json into aggregate summary dict."""
    if not raw_stats:
        return {}
    null_pcts = [v.get("null_pct", 0.0) for v in raw_stats.values() if isinstance(v, dict)]
    stds = [v.get("std", None) for v in raw_stats.values() if isinstance(v, dict)]
    zero_var = sum(1 for s in stds if s is not None and s == 0.0)
    nan_ratio_max = max(null_pcts) if null_pcts else 0.0
    return {
        "total_features": len(raw_stats),
        "zero_variance_count": zero_var,
        "nan_ratio_max": nan_ratio_max,
    }


def _build_fidelity_summary(fidelity_report: Dict) -> Dict:
    """Extract key fields from distillation fidelity_report.json."""
    if not fidelity_report:
        return {}
    details = fidelity_report.get("details", {})
    task_fidelity = {}
    max_gap = 0.0
    tasks_above = []
    for task, info in details.items():
        metrics = info.get("metrics", {})
        # Use calibration_gap if available, else auc_gap, else f1_macro_gap
        gap = metrics.get("calibration_gap",
              metrics.get("auc_gap",
              metrics.get("f1_macro_gap", 0.0))) or 0.0
        task_fidelity[task] = round(1.0 - gap, 4)
        if gap > max_gap:
            max_gap = gap
        if gap > 0.05:
            tasks_above.append(task)
    return {
        "task_fidelity": task_fidelity,
        "max_fidelity_gap": round(max_gap, 4),
        "tasks_above_threshold": tasks_above,
    }


def _extract_training_metrics(eval_metrics: Dict) -> Dict:
    """Extract training metrics from eval_metrics.json."""
    if not eval_metrics:
        return {}
    final = eval_metrics.get("final_metrics", {})
    # Collect AUC values for binary tasks
    auc_values = [v for k, v in final.items() if k.startswith("auc_") and isinstance(v, float)]
    best_val_auc = max(auc_values) if auc_values else None
    return {
        "final_loss": final.get("loss"),
        "best_val_auc": best_val_auc,
        "epochs_completed": eval_metrics.get("epochs_trained", 0),
        "grad_norm_max": None,   # not stored in eval_metrics, leave as None
        "nan_loss_count": 0,     # would require log scanning
    }


def _extract_latency_summary(lambda_results: Dict) -> Dict:
    """Extract latency/request summary from lambda_test results."""
    if not lambda_results:
        return {}
    agg = lambda_results.get("aggregate", {})
    latency = agg.get("latency_ms", {})
    total_lat = latency.get("total", {})
    return {
        "p50_latency_ms": total_lat.get("p50"),
        "p95_latency_ms": total_lat.get("p95"),
        "filter_pass_rate": agg.get("selfcheck_pass_rate"),
        "total_requests": agg.get("n_processed", 0),
    }


# ---------------------------------------------------------------------------
# Build a ToolRegistry wired to local data
# ---------------------------------------------------------------------------

def build_local_registry() -> "ToolRegistry":
    """Build a ToolRegistry with local-file-backed tool callables."""
    from core.agent.tool_registry import ToolRegistry

    # Pre-load data
    adapter_meta   = _load_json(PHASE0_DIR / "adapter_metadata.json") or {}
    raw_feat_stats = _load_json(PHASE0_DIR / "feature_stats.json") or {}
    feat_stats     = _compute_feature_stats_summary(raw_feat_stats)
    eval_metrics   = _load_json(TRAINING_DIR / "eval_metrics.json") or {}
    fidelity_raw   = _load_json(DISTILLATION_DIR / "output" / "fidelity_report.json") or {}
    fidelity       = _build_fidelity_summary(fidelity_raw)
    lambda_results = _load_json(LAMBDA_RESULTS) or {}
    latency        = _extract_latency_summary(lambda_results)
    training_met   = _extract_training_metrics(eval_metrics)

    registry = ToolRegistry(agent_id="test_local")

    # CP1: ingestion manifest
    def read_ingestion_manifest():
        return {
            "total_domains":       1,
            "domains_passed":      1,
            "domains_failed":      0,
            "total_rows":          adapter_meta.get("num_raw_rows", 0),
            "total_pii_encrypted": 0,
            "total_duration_seconds": 0,
        }

    # CP2: pipeline state + feature stats
    def read_pipeline_state():
        return {
            "completed_stages": ["ingestion", "feature_gen", "normalization", "tensor_save"],
        }

    def read_feature_stats():
        return feat_stats

    # CP3: training metrics
    def read_experiment_metrics():
        return training_met

    # CP4: distillation fidelity
    def read_distillation_fidelity():
        return fidelity

    # CP5: serving health (local Lambda stub, healthy based on lambda_results)
    def check_feature_store_health():
        agg = lambda_results.get("aggregate", {})
        healthy = agg.get("n_errors", 1) == 0
        return {
            "healthy":      healthy,
            "backend":      "local_lgbm",
            "record_count": agg.get("n_processed", 0),
        }

    # CP6: recommendation audit archive (from lambda_test results)
    def read_audit_archive():
        return latency

    # CP7: A/B test (no active experiment locally)
    def query_cloudwatch_metrics():
        return {
            "active_experiment": None,
            "variant_metrics":   {},
            "significance_test": {},
        }

    for name, fn in [
        ("read_ingestion_manifest",   read_ingestion_manifest),
        ("read_pipeline_state",       read_pipeline_state),
        ("read_feature_stats",        read_feature_stats),
        ("read_experiment_metrics",   read_experiment_metrics),
        ("read_distillation_fidelity", read_distillation_fidelity),
        ("check_feature_store_health", check_feature_store_health),
        ("read_audit_archive",        read_audit_archive),
        ("query_cloudwatch_metrics",  query_cloudwatch_metrics),
    ]:
        registry.register(name, fn, description=f"Local stub: {name}", category="query")

    logger.info("ToolRegistry built with %d tools", registry.tool_count["total"])
    return registry


# ---------------------------------------------------------------------------
# OpsAgent test
# ---------------------------------------------------------------------------

def run_ops_agent() -> Dict:
    """Run OpsCollector -> OpsDiagnoser -> OpsReporter and return the report dict."""
    logger.info("=" * 60)
    logger.info("OpsAgent Test - CP1 through CP7")
    logger.info("=" * 60)

    from core.agent.ops.collector import OpsCollector
    from core.agent.ops.diagnoser import OpsDiagnoser
    from core.agent.ops.reporter import OpsReporter

    registry = build_local_registry()

    ops_config = {
        "latency_sla_ms":       300,
        "min_val_auc":          0.55,
        "fidelity_gap_threshold": 0.05,
        "grad_norm_warning":    100,
    }

    # --- Step 1: Collect ---
    collector = OpsCollector(registry=registry, config=ops_config)
    logger.info("Running collect_all() ...")
    checkpoints = collector.collect_all()

    logger.info("Checkpoint results:")
    for cp in checkpoints:
        status_str = f"[{cp.status}]"
        err_str = f"  ERROR: {cp.error}" if cp.error else ""
        logger.info("  %s  %s - %s%s", status_str, cp.checkpoint, cp.name, err_str)
        if cp.anomalies:
            for a in cp.anomalies:
                logger.info("      anomaly: %s", a)

    # --- Step 2: Diagnose ---
    diagnoser = OpsDiagnoser(config=ops_config)
    logger.info("\nRunning OpsDiagnoser.diagnose() ...")
    diagnoses = diagnoser.diagnose(checkpoints)

    if diagnoses:
        logger.info("Diagnoses (%d found):", len(diagnoses))
        for d in diagnoses:
            logger.info("  [%s] %s - %s", d.severity, d.rule_id, d.finding)
            logger.info("    Cause:  %s", d.likely_cause)
            logger.info("    Action: %s", d.suggested_action)
    else:
        logger.info("No cross-checkpoint diagnoses triggered.")

    # --- Step 3: Report ---
    reporter = OpsReporter()
    report = reporter.generate(checkpoints, diagnoses, period="daily")
    report.save(str(OPS_REPORT_PATH))

    logger.info("\nOps Report Summary:")
    logger.info("  Overall status:       %s", report.status)
    logger.info("  Attention items:      %d", len(report.attention_required))
    logger.info("  Saved to:             %s", OPS_REPORT_PATH)

    return report.to_dict()


# ---------------------------------------------------------------------------
# AuditAgent test
# ---------------------------------------------------------------------------

def _build_reason_records(lambda_results: Dict) -> List[Dict]:
    """Build reason records from lambda_test_200_results.json for sampling/grounding."""
    records = []
    per_customer = lambda_results.get("per_customer", [])
    for pc in per_customer:
        cid = str(pc.get("customer_id", ""))
        layer_used = pc.get("layer_used", 1)
        layer_str = f"L{layer_used}" if layer_used in (1, 2, 3) else "L1"

        reason_text = pc.get("l2a_reason") or pc.get("l1_reason") or ""
        # Decode if bytes-like junk (garbled UTF-8 from cp949 decode)
        if isinstance(reason_text, bytes):
            try:
                reason_text = reason_text.decode("utf-8", errors="replace")
            except Exception:
                reason_text = ""

        verdict   = pc.get("selfcheck_verdict", "pass")
        selfcheck = 0.9 if verdict == "pass" else 0.5

        predictions = pc.get("predictions", {})
        # Map task types to classify this record
        binary_tasks = [t for t, v in predictions.items() if v.get("task_type") == "binary"]
        multi_tasks  = [t for t, v in predictions.items() if v.get("task_type") == "multiclass"]
        task_type = "binary" if binary_tasks else ("multiclass" if multi_tasks else "regression")

        # Synthetic ig_top_features based on prediction tasks (no real IG in lambda output)
        ig_feats = [
            {"name": t, "text": f"{t} 예측 기여 피처", "ig_score": 0.3}
            for t in list(predictions.keys())[:3]
        ]

        records.append({
            "customer_id":          cid,
            "item_id":              "prod_generic",
            "task_type":            task_type,
            "customer_segment":     "mass",
            "reason_layer":         layer_str,
            "reason_text":          reason_text,
            "ig_top_features":      ig_feats,
            "selfcheck_confidence": selfcheck,
            "selfcheck_verdict":    verdict,
            "human_review_flagged": False,
            "metadata":             {"layer_used": layer_used},
        })
    return records


def run_audit_agent() -> Dict:
    """Run GroundingValidator + StratifiedReasonSampler + AuditReporter."""
    logger.info("\n" + "=" * 60)
    logger.info("AuditAgent Test - Grounding + Sampling + Report")
    logger.info("=" * 60)

    from core.agent.audit.grounding_validator import GroundingValidator
    from core.agent.audit.reason_sampler import StratifiedReasonSampler
    from core.agent.audit.bias_stage_attributor import BiasStageAttributor
    from core.agent.audit.reporter import AuditReporter

    lambda_results = _load_json(LAMBDA_RESULTS) or {}

    # --- Step 1: Build reason records ---
    reason_records = _build_reason_records(lambda_results)
    logger.info("Built %d reason records from lambda test results", len(reason_records))

    # --- Step 2: Grounding Validation ---
    feature_glossary = {
        "churn_signal":            "해지 예측",
        "will_acquire_deposits":   "예금 취득 가능성",
        "will_acquire_investments": "투자 상품 취득 가능성",
        "will_acquire_accounts":   "계좌 취득 가능성",
        "will_acquire_lending":    "대출 취득 가능성",
        "will_acquire_payments":   "결제 상품 취득 가능성",
        "top_mcc_shift":           "주요 가맹점 변화",
        "nba_primary":             "핵심 추천 상품",
        "product_stability":       "보유 상품 안정성",
        "cross_sell_count":        "교차 판매 수",
    }

    validator = GroundingValidator(
        feature_glossary=feature_glossary,
        config={"min_grounding_score": 0.3, "max_sentence_length": 100},
    )

    grounding_results = []
    quality_scores = []

    for rec in reason_records:
        text = rec.get("reason_text", "")
        ig_feats = rec.get("ig_top_features", [])

        if not text or not ig_feats:
            continue

        gr = validator.validate(text, ig_feats)
        grounding_results.append(gr)

        qs = validator.compute_quality_score(
            reason_text=text,
            ig_top_features=ig_feats,
            faithfulness=0.7,  # placeholder (no XAI evaluator locally)
            compliance=1.0 if rec.get("selfcheck_verdict") == "pass" else 0.0,
        )
        quality_scores.append(qs)

    avg_grounding = (
        sum(r.grounding_score for r in grounding_results) / len(grounding_results)
        if grounding_results else 0.0
    )
    avg_overall = (
        sum(q.overall for q in quality_scores) / len(quality_scores)
        if quality_scores else 0.0
    )
    avg_readability = (
        sum(q.readability for q in quality_scores) / len(quality_scores)
        if quality_scores else 0.0
    )

    logger.info("Grounding Validation Results (%d reasons):", len(grounding_results))
    logger.info("  Avg grounding score:   %.4f", avg_grounding)
    logger.info("  Avg overall quality:   %.4f", avg_overall)
    logger.info("  Avg readability score: %.4f", avg_readability)

    reason_quality = {
        "tier1": {
            "total_validated":   len(grounding_results),
            "avg_grounding":     round(avg_grounding, 4),
            "avg_overall":       round(avg_overall, 4),
            "avg_readability":   round(avg_readability, 4),
        },
        "tier2": {},
        "tier3": {},
    }

    # --- Step 3: Stratified Reason Sampling ---
    sampler = StratifiedReasonSampler(config={
        "samples_per_stratum": 5,
        "task_types":         ["binary", "multiclass", "regression"],
        "customer_segments":  ["mass", "affluent", "vip"],
        "reason_layers":      ["L1", "L2a", "L2b"],
    })

    sampled_cases = sampler.sample(reason_records)
    sampling_stats = sampler.get_sampling_stats(sampled_cases)
    logger.info("Stratified Sampling Stats:")
    logger.info("  Total sampled:         %d", sampling_stats["total_sampled"])
    logger.info("  By task type:          %s", sampling_stats["by_task_type"])
    logger.info("  By reason layer:       %s", sampling_stats["by_reason_layer"])
    logger.info("  Borderline cases:      %d", sampling_stats["borderline_cases"])

    # --- Step 4: Bias Stage Attribution (synthetic demo) ---
    attributor = BiasStageAttributor(config={"di_threshold": 0.80, "min_sample_size": 3})

    per_cust = lambda_results.get("per_customer", [])
    customers = [{"customer_id": str(p.get("customer_id", "")), "age_group": "adult"} for p in per_cust]

    # Synthetic stage data: model considers all customers recommended
    model_scores   = [{"customer_id": str(p.get("customer_id", "")), "recommended": True} for p in per_cust]
    post_filter    = model_scores  # no filtering stub
    post_selection = model_scores

    try:
        bias_attr = attributor.attribute(
            attribute="age_group",
            group_value="adult",
            customers=customers,
            model_scores=model_scores,
            post_filter=post_filter,
            post_selection=post_selection,
        )
        logger.info("Bias Attribution (age_group=adult):")
        for stage in bias_attr.stages:
            logger.info("  Stage %-20s DI=%.4f  sample_size=%d",
                        stage.stage_label, stage.di_value, stage.sample_size)
        bias_summary = bias_attr.to_dict()
    except Exception as e:
        logger.warning("BiasStageAttributor failed: %s", e)
        bias_summary = {}

    # --- Step 5: Generate Audit Report ---
    reporter = AuditReporter()
    audit_report = reporter.generate(
        focus_areas=None,
        regulatory_results={
            "domestic":  {"status": "compliant", "checked_rules": 5},
            "eu_ai_act": {"status": "partial",   "risk_category": "limited"},
            "fria":      {"status": "pending"},
        },
        reason_quality=reason_quality,
        period="weekly",
    )
    audit_report.metadata["sampling_stats"] = sampling_stats
    audit_report.metadata["bias_attribution"] = bias_summary
    audit_report.save(str(AUDIT_REPORT_PATH))

    logger.info("\nAudit Report Summary:")
    logger.info("  Risk level:            %s", audit_report.risk_level)
    logger.info("  Focus areas:           %d", len(audit_report.focus_areas))
    logger.info("  Saved to:              %s", AUDIT_REPORT_PATH)

    return audit_report.to_dict()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("Starting local agent test - OpsAgent + AuditAgent")
    logger.info("Base dir: %s", BASE_DIR)

    errors: List[str] = []

    # OpsAgent
    try:
        ops_report = run_ops_agent()
        logger.info("\n[OPS] PASS - overall status: %s",
                    ops_report.get("ops_report", {}).get("status", "UNKNOWN"))
    except Exception:
        logger.exception("[OPS] FAILED with unhandled exception")
        errors.append("OpsAgent")

    # AuditAgent
    try:
        audit_report = run_audit_agent()
        logger.info("\n[AUDIT] PASS - risk level: %s",
                    audit_report.get("audit_report", {}).get("risk_level", "UNKNOWN"))
    except Exception:
        logger.exception("[AUDIT] FAILED with unhandled exception")
        errors.append("AuditAgent")

    # Summary
    logger.info("\n" + "=" * 60)
    if errors:
        logger.error("Test FAILED - errors in: %s", ", ".join(errors))
        sys.exit(1)
    else:
        logger.info("All tests PASSED")
        logger.info("  Ops report:   %s", OPS_REPORT_PATH)
        logger.info("  Audit report: %s", AUDIT_REPORT_PATH)


if __name__ == "__main__":
    main()
