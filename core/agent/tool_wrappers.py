"""
Tool Wrapper Functions for Agent ToolRegistry
================================================

Python callables that bridge ToolRegistry → existing monitoring/pipeline components.
Each function is registered via ``registry.register(name, func)``.

Usage::

    from core.agent.tool_wrappers import register_all_tools
    registry = ToolRegistry(config_path="configs/financial/agent_tools.yaml")
    register_all_tools(registry)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["register_all_tools"]


# ======================================================================
# Category 1: Infrastructure Query tools (read files/metrics)
# ======================================================================

def read_pipeline_state(output_dir: str = "outputs") -> Dict[str, Any]:
    """Read pipeline_state.json — stage completion, timing, artifacts."""
    path = Path(output_dir) / "pipeline_state.json"
    if not path.exists():
        return {"error": f"Not found: {path}"}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_feature_stats(output_dir: str = "outputs") -> Dict[str, Any]:
    """Read feature_stats.json — per-feature mean/std/null/zero-variance."""
    path = Path(output_dir) / "feature_stats.json"
    if not path.exists():
        return {"error": f"Not found: {path}"}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # Compute summary
    zero_var = sum(1 for v in data.values() if isinstance(v, dict) and v.get("zero_variance", False))
    nan_ratios = [v.get("nan_ratio", 0) for v in data.values() if isinstance(v, dict)]
    return {
        "total_features": len(data),
        "zero_variance_count": zero_var,
        "nan_ratio_max": max(nan_ratios) if nan_ratios else 0.0,
        "nan_ratio_mean": sum(nan_ratios) / len(nan_ratios) if nan_ratios else 0.0,
        "raw": data,
    }


def read_experiment_metrics(output_dir: str = "outputs") -> Dict[str, Any]:
    """Read metrics.jsonl — epoch-level loss, val_auc, grad_norm."""
    # Try multiple possible paths
    for subdir in ["experiments", "outputs", "."]:
        pattern = Path(subdir).glob("**/metrics.jsonl")
        for path in pattern:
            metrics = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        metrics.append(json.loads(line))
            if not metrics:
                continue
            # Extract summary
            losses = [m["value"] for m in metrics if m.get("key") == "loss"]
            aucs = [m["value"] for m in metrics if m.get("key", "").startswith("val_auc")]
            grad_norms = [m["value"] for m in metrics if m.get("key") == "grad_norm"]
            return {
                "metrics_path": str(path),
                "total_entries": len(metrics),
                "final_loss": losses[-1] if losses else None,
                "best_val_auc": max(aucs) if aucs else None,
                "epochs_completed": max((m.get("step", 0) for m in metrics), default=0),
                "grad_norm_max": max(grad_norms) if grad_norms else None,
                "nan_loss_count": sum(1 for v in losses if v != v),  # NaN check
            }
    return {"error": "metrics.jsonl not found"}


def read_ingestion_manifest(manifest_path: str = "outputs/manifests/latest.json") -> Dict[str, Any]:
    """Read ingestion manifest — domain row counts, PII, validation."""
    path = Path(manifest_path)
    if not path.exists():
        # Try glob
        candidates = sorted(Path("outputs/manifests").glob("*.json"), reverse=True) if Path("outputs/manifests").exists() else []
        if not candidates:
            return {"error": "No manifest found"}
        path = candidates[0]
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_leakage_report(output_dir: str = "outputs") -> Dict[str, Any]:
    """Read audit/leakage_report.json — leakage validation result."""
    path = Path(output_dir) / "audit" / "leakage_report.json"
    if not path.exists():
        return {"error": f"Not found: {path}"}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_distillation_fidelity(output_dir: str = "outputs") -> Dict[str, Any]:
    """Read distillation fidelity metrics."""
    path = Path(output_dir) / "distillation" / "fidelity_metrics.json"
    if not path.exists():
        # Try pipeline_manifest
        manifest_path = Path(output_dir) / "pipeline_manifest.json"
        if manifest_path.exists():
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            distill = manifest.get("stage_distill", {})
            return distill if distill else {"error": "No distillation data in manifest"}
        return {"error": f"Not found: {path}"}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_audit_archive(date: Optional[str] = None, metric: Optional[str] = None) -> Dict[str, Any]:
    """Read audit_archiver Parquet — recommendation trace statistics."""
    # Simplified: read from latest archive
    try:
        import duckdb
        archive_path = "outputs/audit_archive"
        if not Path(archive_path).exists():
            return {"error": "Audit archive not found"}
        files = sorted(Path(archive_path).glob("*.parquet"), reverse=True)
        if not files:
            return {"error": "No parquet files in archive"}
        target = str(files[0])
        conn = duckdb.connect()
        result = conn.execute(f"""
            SELECT
                count(*) as total_requests,
                avg(elapsed_ms) as avg_latency_ms,
                percentile_cont(0.50) WITHIN GROUP (ORDER BY elapsed_ms) as p50_latency_ms,
                percentile_cont(0.95) WITHIN GROUP (ORDER BY elapsed_ms) as p95_latency_ms
            FROM '{target}'
        """).fetchone()
        conn.close()
        if result:
            return {
                "total_requests": result[0],
                "avg_latency_ms": round(result[1], 2) if result[1] else None,
                "p50_latency_ms": round(result[2], 2) if result[2] else None,
                "p95_latency_ms": round(result[3], 2) if result[3] else None,
            }
        return {"total_requests": 0}
    except ImportError:
        return {"error": "duckdb not available"}
    except Exception as e:
        return {"error": str(e)}


def read_checklist_config(config_path: str = "configs/financial/checklist.yaml") -> Dict[str, Any]:
    """Read checklist YAML config."""
    import yaml
    path = Path(config_path)
    if not path.exists():
        return {"error": f"Not found: {path}"}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_git_diff(repo_root: str = ".") -> Dict[str, Any]:
    """Read git diff — changed files and affected pipeline parts."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD~1"],
            capture_output=True, text=True, cwd=repo_root, timeout=10,
        )
        files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
        return {"changed_files": files, "count": len(files)}
    except Exception as e:
        return {"error": str(e)}


# ======================================================================
# Category 2: Monitoring Query tools (wrap monitoring components)
# ======================================================================

def _lazy_import_monitoring():
    """Lazy import to avoid circular dependencies at module load time."""
    from core.monitoring import (
        DriftDetector, FairnessMonitor, HerdingDetector, AuditLogger,
    )
    from core.data.quality_gate import QualityGate
    return DriftDetector, FairnessMonitor, HerdingDetector, AuditLogger, QualityGate


def detect_drift(baseline_path: str = "", current_path: str = "", **kwargs) -> Dict[str, Any]:
    """Detect feature drift via PSI."""
    DriftDetector = _lazy_import_monitoring()[0]
    detector = DriftDetector()
    # Simplified — real impl would load parquet data
    return {"status": "requires_data", "note": "Call with baseline/current DataFrames"}


def evaluate_fairness(recommendations: Optional[List[Dict]] = None, attributes: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
    """Evaluate fairness metrics (DI/SPD/EOD) — wraps FairnessMonitor with auto_incident=False."""
    FairnessMonitor = _lazy_import_monitoring()[1]
    monitor = FairnessMonitor(auto_incident=False)
    if not recommendations:
        return {"error": "No recommendations provided"}
    attrs = attributes or ["age_group", "income_tier", "gender"]
    # Build default group pairs
    group_pairs = {a: [("privileged", "unprivileged")] for a in attrs}
    results = monitor.evaluate_all_attributes(recommendations, group_pairs)
    return {a: monitor.to_dict(m) for a, m in results.items()}


def detect_herding(recommendations: Optional[List[Dict]] = None, **kwargs) -> Dict[str, Any]:
    """Detect recommendation concentration — wraps HerdingDetector."""
    HerdingDetector = _lazy_import_monitoring()[2]
    detector = HerdingDetector()
    if not recommendations:
        return {"error": "No recommendations provided"}
    return detector.detect_herding(recommendations)


def check_feature_store_health(**kwargs) -> Dict[str, Any]:
    """Check feature store health."""
    try:
        from core.serving.feature_store import FeatureStoreFactory
        from core.serving.config import ServingConfig
        config = ServingConfig()
        store = FeatureStoreFactory.create(config)
        return store.health_check()
    except Exception as e:
        return {"healthy": False, "error": str(e)}


def evaluate_data_quality(source_name: str = "", **kwargs) -> Dict[str, Any]:
    """Evaluate data quality via QualityGate."""
    QualityGate = _lazy_import_monitoring()[4]
    gate = QualityGate()
    return {"status": "requires_dataframe", "note": "Call with DataFrame and source_name"}


def verify_audit_chain(log_dir: str = "audit_logs", **kwargs) -> Dict[str, Any]:
    """Verify HMAC audit log hash chain integrity."""
    AuditLogger = _lazy_import_monitoring()[3]
    logger_inst = AuditLogger()
    try:
        from pathlib import Path
        log_path = sorted(Path(log_dir).glob("**/*.jsonl"), reverse=True)
        if not log_path:
            return {"verified": False, "error": "No log files found"}
        with open(log_path[0], encoding="utf-8") as f:
            lines = f.readlines()
        result = logger_inst.verify_chain(lines)
        return {"verified": result, "file": str(log_path[0]), "entries": len(lines)}
    except Exception as e:
        return {"verified": False, "error": str(e)}


# ======================================================================
# Category 3: Regulatory/Quality Query tools
# ======================================================================

def run_regulatory_checks(**kwargs) -> Dict[str, Any]:
    """Run domestic regulatory compliance checks (20 items)."""
    try:
        from core.compliance.regulatory_checker import RegulatoryComplianceChecker
        checker = RegulatoryComplianceChecker()
        results = checker.run_all_checks()
        return checker.get_summary(results)
    except Exception as e:
        return {"error": str(e)}


def run_compliance_check(**kwargs) -> Dict[str, Any]:
    """Run infrastructure compliance check (9 items)."""
    try:
        from core.monitoring.compliance_checker import ComplianceChecker
        checker = ComplianceChecker()
        return checker.run_full_check()
    except Exception as e:
        return {"error": str(e)}


def evaluate_eu_ai_act(**kwargs) -> Dict[str, Any]:
    """Evaluate EU AI Act compliance."""
    try:
        from core.monitoring.eu_ai_act_mapper import EUAIActMapper
        mapper = EUAIActMapper()
        report = mapper.generate_report()
        return mapper.to_dict(report)
    except Exception as e:
        return {"error": str(e)}


def evaluate_fria(**kwargs) -> Dict[str, Any]:
    """Evaluate Financial Risk Impact Assessment."""
    try:
        from core.monitoring.fria_evaluator import FRIAEvaluator
        evaluator = FRIAEvaluator()
        report = evaluator.generate_report()
        return evaluator.to_dict(report)
    except Exception as e:
        return {"error": str(e)}


def check_reason_quality(reason_text: str = "", **kwargs) -> Dict[str, Any]:
    """Check single reason quality via SelfChecker."""
    try:
        from core.recommendation.reason.self_checker import SelfChecker
        checker = SelfChecker(config={})
        result = checker.check(reason_text)
        return {
            "verdict": result.verdict,
            "compliance_passed": result.compliance_passed,
            "injection_safe": result.injection_safe,
            "factuality_score": result.factuality_score,
            "violations": result.violations,
        }
    except Exception as e:
        return {"error": str(e)}


def evaluate_xai_quality(task_name: str = "", **kwargs) -> Dict[str, Any]:
    """Evaluate XAI explanation quality."""
    try:
        from core.monitoring.xai_quality_evaluator import XAIQualityEvaluator
        evaluator = XAIQualityEvaluator()
        metrics = evaluator.evaluate_task(task_name, **kwargs)
        return {
            "task_name": metrics.task_name,
            "overall_quality": metrics.overall_quality,
            "meets_threshold": metrics.meets_threshold,
        }
    except Exception as e:
        return {"error": str(e)}


def trace_feature_lineage(feature_names: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
    """Trace features to source data."""
    try:
        from core.monitoring.lineage_tracker import DataLineageTracker
        tracker = DataLineageTracker()
        if not feature_names:
            return {"error": "No feature names provided"}
        results = tracker.trace_features_batch(feature_names)
        return {"traces": results, "total": len(results)}
    except Exception as e:
        return {"error": str(e)}


def generate_lineage_report(batch_date: str = "", **kwargs) -> Dict[str, Any]:
    """Generate data lineage report."""
    try:
        from core.monitoring.lineage_tracker import DataLineageTracker
        tracker = DataLineageTracker()
        return tracker.generate_lineage_report(batch_date)
    except Exception as e:
        return {"error": str(e)}


# ======================================================================
# Category 4: Action tools
# ======================================================================

def create_incident(event_type: str = "", metrics: Optional[Dict] = None, source_module: str = "agent", **kwargs) -> Dict[str, Any]:
    """Create an incident — wraps IncidentReporter."""
    try:
        from core.monitoring.incident_reporter import IncidentReporter
        reporter = IncidentReporter()
        record = reporter.create_incident(event_type, metrics or {}, source_module)
        return reporter.generate_report(record)
    except Exception as e:
        return {"error": str(e)}


def log_audit_event(operation: str = "", **kwargs) -> Dict[str, Any]:
    """Log an audit event — wraps AuditLogger."""
    try:
        from core.monitoring.audit_logger import AuditLogger
        al = AuditLogger()
        result = al.log_operation(operation, **kwargs)
        return result or {"status": "logged"}
    except Exception as e:
        return {"error": str(e)}


def generate_governance_report(period: str = "monthly", **kwargs) -> Dict[str, Any]:
    """Generate governance report."""
    try:
        from core.monitoring.governance_report import GovernanceReportGenerator
        gen = GovernanceReportGenerator()
        report = gen.generate_report(period=period, **kwargs)
        return gen.to_dict(report)
    except Exception as e:
        return {"error": str(e)}


def archive_governance_report(report: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
    """Archive governance report to S3."""
    try:
        from core.monitoring.governance_report import GovernanceReportGenerator, GovernanceReport
        gen = GovernanceReportGenerator()
        uri = gen.archive_report(report)
        return {"archived": True, "uri": uri}
    except Exception as e:
        return {"error": str(e)}


def save_compliance_report(report: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
    """Save compliance report to S3."""
    try:
        from core.monitoring.compliance_checker import ComplianceChecker
        checker = ComplianceChecker()
        uri = checker.save_report(report or {})
        return {"saved": True, "uri": uri}
    except Exception as e:
        return {"error": str(e)}


def save_lineage(lineage: Optional[Dict] = None, lineage_type: str = "single", **kwargs) -> Dict[str, Any]:
    """Save data lineage to S3."""
    try:
        from core.monitoring.lineage_tracker import DataLineageTracker
        tracker = DataLineageTracker()
        record = tracker.save_lineage(lineage or {}, lineage_type)
        return {"saved": True, "execution_id": record.execution_id}
    except Exception as e:
        return {"error": str(e)}


def send_notification(subject: str = "", body: Optional[Dict] = None, channels: Optional[List[str]] = None, severity: str = "INFO", **kwargs) -> Dict[str, Any]:
    """Send notification via Slack/SNS."""
    try:
        from core.agent.notification import NotificationService
        service = NotificationService(kwargs.get("config", {}))
        return service.send(subject, body or {}, channels, severity)
    except Exception as e:
        return {"error": str(e)}


# ======================================================================
# Category 2 additional: Monitoring Query
# ======================================================================

def query_cloudwatch_metrics(namespace: str = "PLE/ABTest", metric_names: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
    """Query CloudWatch metrics — A/B test, latency (AWS only)."""
    try:
        import boto3
        client = boto3.client("cloudwatch", region_name=kwargs.get("region", "ap-northeast-2"))
        # Simplified — return latest datapoints for requested metrics
        results = {}
        for metric in (metric_names or ["CTR", "Latency_p95"]):
            resp = client.get_metric_statistics(
                Namespace=namespace,
                MetricName=metric,
                Period=3600,
                Statistics=["Average"],
                StartTime=kwargs.get("start_time", "2026-01-01"),
                EndTime=kwargs.get("end_time", "2026-12-31"),
            )
            datapoints = resp.get("Datapoints", [])
            if datapoints:
                latest = sorted(datapoints, key=lambda d: d["Timestamp"])[-1]
                results[metric] = latest.get("Average")
        return results
    except Exception as e:
        return {"error": str(e)}


def get_consecutive_drift_days(monitoring_dir: str = "outputs/monitoring", **kwargs) -> Dict[str, Any]:
    """Get consecutive critical drift days."""
    try:
        from core.monitoring.drift_detector import ConsecutiveDriftTracker
        tracker = ConsecutiveDriftTracker(monitoring_dir=monitoring_dir)
        return tracker.get_consecutive_critical_days()
    except Exception as e:
        return {"error": str(e)}


def detect_task_herding(task_contribution_map: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
    """Detect per-task contribution herding."""
    HerdingDetector = _lazy_import_monitoring()[2]
    detector = HerdingDetector()
    if not task_contribution_map:
        return {"error": "No task_contribution_map provided"}
    return detector.detect_task_herding(task_contribution_map)


def check_explanation_consistency(task_name: str = "", attributions_a: Optional[Any] = None, attributions_b: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
    """Check SHAP vs IG explanation consistency."""
    try:
        from core.monitoring.xai_quality_evaluator import XAIQualityEvaluator
        evaluator = XAIQualityEvaluator()
        if attributions_a is None or attributions_b is None:
            return {"error": "Both attributions_a and attributions_b required"}
        result = evaluator.check_explanation_consistency(task_name, attributions_a, attributions_b)
        return {
            "rank_correlation": result.rank_correlation,
            "top_k_overlap": result.top_k_overlap,
            "is_consistent": result.is_consistent,
        }
    except Exception as e:
        return {"error": str(e)}


# ======================================================================
# Category: Case Store tools
# ======================================================================

# Note: These require a DiagnosticCaseStore instance to be passed via kwargs
# or a singleton pattern. The wrapper creates a default instance.

def _get_case_store(**kwargs):
    """Get or create DiagnosticCaseStore instance."""
    from core.agent.case_store import DiagnosticCaseStore
    store_path = kwargs.get("store_path", "outputs/diagnostic_cases")
    return DiagnosticCaseStore(store_path=store_path)


def _embed_finding(finding: str, dim: int = 384) -> "np.ndarray":
    """Return a unit-normalised embedding vector for *finding*.

    Strategy (in order of preference):
    1. TF-IDF bag-of-words via sklearn (produces meaningful cosine distances).
    2. Hash-based fallback when sklearn is unavailable (warns once).

    The store uses cosine / L2 similarity so the exact dimensionality does not
    need to match a pre-trained model — relative distances are what matters for
    case retrieval.
    """
    import numpy as np
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        # Single-document vectoriser: project to *dim* via hashing trick on
        # TF-IDF term indices so the output is always length *dim*.
        from sklearn.feature_extraction.text import HashingVectorizer
        vec_model = HashingVectorizer(
            n_features=dim, norm="l2", alternate_sign=False, token_pattern=r"(?u)\b\w+\b"
        )
        mat = vec_model.transform([finding])
        vec = mat.toarray()[0].astype(np.float32)
    except ImportError:
        logger.warning(
            "sklearn not available — falling back to hash-based embedding for "
            "search_similar_cases. Install scikit-learn for meaningful similarity."
        )
        vec = np.array(
            [hash(finding + str(i)) % 1000 / 1000.0 for i in range(dim)],
            dtype=np.float32,
        )
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def search_similar_cases(finding: str = "", pipeline_part: Optional[str] = None, k: int = 5, **kwargs) -> Dict[str, Any]:
    """Search similar diagnostic cases by text similarity."""
    try:
        store = _get_case_store(**kwargs)
        import numpy as np  # noqa: F401 — keep import local; _embed_finding also imports it
        vec = _embed_finding(finding)
        results = store.search_similar(vec, k=k, pipeline_part=pipeline_part)
        return {
            "similar_cases": [{"case": case, "score": score} for case, score in results],
            "count": len(results),
        }
    except Exception as e:
        return {"error": str(e)}


def get_case_statistics(pipeline_part: Optional[str] = None, check_item: Optional[str] = None, period_days: Optional[int] = None, **kwargs) -> Dict[str, Any]:
    """Get diagnostic case statistics."""
    try:
        store = _get_case_store(**kwargs)
        return store.get_statistics(pipeline_part, check_item, period_days)
    except Exception as e:
        return {"error": str(e)}


def save_case(case: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
    """Save a diagnostic case to the case store."""
    try:
        store = _get_case_store(**kwargs)
        case_id = store.save_case(case or {})
        return {"saved": True, "case_id": case_id}
    except Exception as e:
        return {"error": str(e)}


def update_case_resolution(case_id: str = "", resolution: str = "", resolved_at: Optional[str] = None, post_resolution_verdict: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """Update case resolution information."""
    try:
        store = _get_case_store(**kwargs)
        success = store.update_resolution(case_id, resolution, resolved_at, post_resolution_verdict)
        return {"updated": success, "case_id": case_id}
    except Exception as e:
        return {"error": str(e)}


# ======================================================================
# Registration
# ======================================================================

def register_all_tools(registry) -> int:
    """Register all tool wrapper functions in the given ToolRegistry.

    Returns:
        Number of tools registered.
    """
    tools = {
        # Category 1: Infrastructure Query
        "read_pipeline_state": read_pipeline_state,
        "read_feature_stats": read_feature_stats,
        "read_experiment_metrics": read_experiment_metrics,
        "read_ingestion_manifest": read_ingestion_manifest,
        "read_leakage_report": read_leakage_report,
        "read_distillation_fidelity": read_distillation_fidelity,
        "read_audit_archive": read_audit_archive,
        "read_checklist_config": read_checklist_config,
        "read_git_diff": read_git_diff,
        "query_cloudwatch_metrics": query_cloudwatch_metrics,
        # Category 2: Monitoring Query
        "detect_drift": detect_drift,
        "get_consecutive_drift_days": get_consecutive_drift_days,
        "evaluate_fairness": evaluate_fairness,
        "detect_herding": detect_herding,
        "detect_task_herding": detect_task_herding,
        "check_feature_store_health": check_feature_store_health,
        "evaluate_data_quality": evaluate_data_quality,
        "verify_audit_chain": verify_audit_chain,
        "check_explanation_consistency": check_explanation_consistency,
        # Category 3: Regulatory/Quality Query
        "run_regulatory_checks": run_regulatory_checks,
        "run_compliance_check": run_compliance_check,
        "evaluate_eu_ai_act": evaluate_eu_ai_act,
        "evaluate_fria": evaluate_fria,
        "check_reason_quality": check_reason_quality,
        "evaluate_xai_quality": evaluate_xai_quality,
        "trace_feature_lineage": trace_feature_lineage,
        "generate_lineage_report": generate_lineage_report,
        # Category 4: Case Store
        "search_similar_cases": search_similar_cases,
        "get_case_statistics": get_case_statistics,
        "save_case": save_case,
        "update_case_resolution": update_case_resolution,
        # Category 5: Action
        "create_incident": create_incident,
        "log_audit_event": log_audit_event,
        "generate_governance_report": generate_governance_report,
        "archive_governance_report": archive_governance_report,
        "save_compliance_report": save_compliance_report,
        "save_lineage": save_lineage,
        "send_notification": send_notification,
    }

    count = 0
    for name, func in tools.items():
        registry.register(name, func)
        count += 1

    logger.info("Registered %d tool wrappers", count)
    return count
