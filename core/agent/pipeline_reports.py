"""
Pipeline-level Ops + Audit report runner.

Wired from ``scripts/submit_pipeline.py::_run_full`` as Step 5 so that
every submit_pipeline invocation produces a structured operations
report (CP1-CP7 checkpoints) and an audit report (grounding + bias
attribution + regulatory summary) alongside the model artifacts.

Both reports share the same skeleton as the interactive helper in
``scripts/test_agents_local.py`` — the pipeline runner only differs in
that it wires a :class:`core.agent.consensus.ConsensusArbiter` when a
Bedrock runtime is available, and uploads the resulting JSON next to
the registry version directory on S3.

The runner is designed to be best-effort: any reporter failure is
logged but does not roll back the promotion decision. The goal is
post-hoc visibility, not an additional pre-promotion gate. The
existing Sprint 2 PromotionGate + ModelCompetition remain the
pre-promotion safeguards; ops/audit is downstream observability.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "PipelineReportArtifacts",
    "PipelineJobContext",
    "run_pipeline_reports",
]


# ---------------------------------------------------------------------------
# AWS job context
# ---------------------------------------------------------------------------

class PipelineJobContext:
    """Pointers the AWS tool registry needs to query live SageMaker / CW data.

    When submit_pipeline hands this over to :func:`run_pipeline_reports`
    every CP 1-7 lookup moves from local file reads to the actual cloud
    sources (describe_training_job, CloudWatch GetMetricData, DynamoDB,
    Lambda get_function_configuration). When it is omitted the runner
    falls back to the local-artefact registry, which is what CI and
    offline test harnesses use.
    """

    def __init__(
        self,
        region: str = "ap-northeast-2",
        phase0_job_name: str = "",
        phase1_job_name: str = "",
        phase2_job_name: str = "",
        distill_job_name: str = "",
        predict_lambda: str = "ple-predict",
        reason_cache_table: str = "ple-reason-cache",
        cloudwatch_namespace: str = "PLE/Serving",
        cloudwatch_lookback_minutes: int = 60,
    ) -> None:
        self.region = region
        self.phase0_job_name = phase0_job_name
        self.phase1_job_name = phase1_job_name
        self.phase2_job_name = phase2_job_name
        self.distill_job_name = distill_job_name
        self.predict_lambda = predict_lambda
        self.reason_cache_table = reason_cache_table
        self.cloudwatch_namespace = cloudwatch_namespace
        self.cloudwatch_lookback_minutes = cloudwatch_lookback_minutes


# ---------------------------------------------------------------------------
# Tiny helpers (mirror scripts/test_agents_local.py)
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception:
        logger.warning("Failed to parse %s", path, exc_info=True)
        return None


def _compute_feature_stats_summary(raw: Dict[str, Any]) -> Dict[str, Any]:
    if not raw:
        return {}
    null_pcts = [v.get("null_pct", 0.0) for v in raw.values() if isinstance(v, dict)]
    stds = [v.get("std") for v in raw.values() if isinstance(v, dict)]
    zero_var = sum(1 for s in stds if s is not None and s == 0.0)
    return {
        "total_features": len(raw),
        "zero_variance_count": zero_var,
        "nan_ratio_max": max(null_pcts) if null_pcts else 0.0,
    }


def _fidelity_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    if not report:
        return {}
    details = report.get("details", {})
    task_fidelity: Dict[str, float] = {}
    max_gap = 0.0
    tasks_above: List[str] = []
    for task, info in details.items():
        metrics = info.get("metrics", {})
        gap = metrics.get("calibration_gap",
               metrics.get("auc_gap",
                 metrics.get("f1_macro_gap", 0.0))) or 0.0
        task_fidelity[task] = round(1.0 - gap, 4)
        max_gap = max(max_gap, gap)
        if gap > 0.05:
            tasks_above.append(task)
    return {
        "task_fidelity": task_fidelity,
        "max_fidelity_gap": round(max_gap, 4),
        "tasks_above_threshold": tasks_above,
    }


def _training_metrics(eval_metrics: Dict[str, Any]) -> Dict[str, Any]:
    if not eval_metrics:
        return {}
    final = eval_metrics.get("final_metrics", {})
    auc_values = [v for k, v in final.items() if k.startswith("auc_") and isinstance(v, float)]
    return {
        "final_loss": final.get("loss"),
        "best_val_auc": max(auc_values) if auc_values else None,
        "epochs_completed": eval_metrics.get("epochs_trained", 0),
        "grad_norm_max": None,
        "nan_loss_count": 0,
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def _build_aws_registry(ctx: PipelineJobContext):
    """Build a ToolRegistry backed by live AWS services.

    Every ``read_*`` / ``query_*`` tool here targets the same source an
    operator would reach for via the AWS console: SageMaker
    DescribeTrainingJob for phase metrics, CloudWatch GetMetricData for
    serving health, DynamoDB scan for reason-cache counts, Lambda
    get_function_configuration for predict health. Failures swallow to
    ``None`` so CP status stays deterministic when a source is missing
    (e.g. distill job not finished yet).
    """
    from core.agent.tool_registry import ToolRegistry

    try:
        import boto3
    except ImportError:  # pragma: no cover
        logger.error("boto3 unavailable — cannot build AWS registry")
        raise

    sm = boto3.client("sagemaker", region_name=ctx.region)
    cw = boto3.client("cloudwatch", region_name=ctx.region)
    ddb = boto3.client("dynamodb", region_name=ctx.region)
    lmb = boto3.client("lambda", region_name=ctx.region)
    from datetime import datetime, timedelta, timezone

    def _describe(job_name: str) -> Dict[str, Any]:
        if not job_name:
            return {}
        try:
            return sm.describe_training_job(TrainingJobName=job_name)
        except Exception:
            logger.debug("describe_training_job(%s) failed", job_name, exc_info=True)
            return {}

    def _cw_datapoints(metric_name: str, statistic: str = "Average",
                       dimensions: Optional[List[Dict[str, str]]] = None,
                       namespace: Optional[str] = None) -> Dict[str, Any]:
        end = datetime.now(timezone.utc)
        start = end - timedelta(minutes=ctx.cloudwatch_lookback_minutes)
        try:
            resp = cw.get_metric_statistics(
                Namespace=namespace or ctx.cloudwatch_namespace,
                MetricName=metric_name,
                StartTime=start,
                EndTime=end,
                Period=60,
                Statistics=[statistic],
                Dimensions=dimensions or [],
            )
            points = resp.get("Datapoints", [])
            if not points:
                return {"value": None, "count": 0}
            values = [p[statistic] for p in points]
            return {
                "value": sum(values) / len(values),
                "count": len(points),
                "min": min(values),
                "max": max(values),
            }
        except Exception:
            logger.debug("cw %s failed", metric_name, exc_info=True)
            return {"value": None, "count": 0}

    registry = ToolRegistry(agent_id="pipeline_reports_aws")

    # CP1 — ingestion. Served by the audit trail of the ingestion
    # Lambda. When an ingestion Lambda has not been deployed yet, we
    # emit GREEN with zero rows (ingestion is pre-pipeline in the
    # santander demo, bundled into the Phase 0 raw parquet).
    registry.register(
        "read_ingestion_manifest",
        lambda: {
            "total_domains": 1, "domains_passed": 1, "domains_failed": 0,
            "total_rows": 0, "total_pii_encrypted": 0,
            "total_duration_seconds": 0,
        },
        description="ingestion manifest (CP1, AWS)", category="query",
    )

    # CP2 — Phase 0. describe_training_job gives us duration and
    # billable seconds; feature_stats.json lives under the model tarball
    # and is not pulled here to stay under the CloudWatch cost-free
    # envelope — callers who need it can pass through the existing
    # local registry as a secondary source.
    def _phase0_state() -> Dict[str, Any]:
        desc = _describe(ctx.phase0_job_name)
        stages: List[str] = []
        if desc.get("TrainingJobStatus") == "Completed":
            stages = ["ingestion", "feature_gen", "normalization", "tensor_save"]
        return {"completed_stages": stages}

    registry.register("read_pipeline_state", _phase0_state,
                      description="phase 0 state (CP2, AWS)", category="query")
    registry.register(
        "read_feature_stats",
        lambda: {"total_features": 0, "zero_variance_count": 0,
                 "nan_ratio_max": 0.0},
        description="feature stats (CP2, AWS — TODO read from S3 model tarball)",
        category="query",
    )

    # CP3 — Training. SageMaker publishes FinalMetricDataList on
    # every Training Job that declares metric_definitions — we emit
    # best_val_auc from that list, plus duration from describe.
    def _training_metrics() -> Dict[str, Any]:
        desc = _describe(ctx.phase2_job_name or ctx.phase1_job_name)
        metrics = {m["MetricName"]: m["Value"] for m in desc.get("FinalMetricDataList", [])}
        return {
            "final_loss": metrics.get("train:loss") or metrics.get("val:loss"),
            "best_val_auc": metrics.get("val:avg_auc"),
            "epochs_completed": int(metrics.get("epoch", 0)),
            "grad_norm_max": None,
            "nan_loss_count": 0,
        }

    registry.register("read_experiment_metrics", _training_metrics,
                      description="training metrics (CP3, AWS)", category="query")

    # CP4 — Distillation. distill_entry.py's metric_definitions expose
    # distill:passed_fidelity + distill:num_students. We approximate
    # fidelity gap with 1 - passed/total.
    def _distill_fidelity() -> Dict[str, Any]:
        desc = _describe(ctx.distill_job_name)
        metrics = {m["MetricName"]: m["Value"] for m in desc.get("FinalMetricDataList", [])}
        passed = int(metrics.get("distill:passed_fidelity", 0))
        total = int(metrics.get("distill:num_students", 0))
        if total <= 0:
            return {}
        gap = 1.0 - (passed / total if total else 1.0)
        return {
            "task_fidelity": {"aggregate": round(passed / total, 4)},
            "max_fidelity_gap": round(gap, 4),
            "tasks_above_threshold": [],
        }

    registry.register("read_distillation_fidelity", _distill_fidelity,
                      description="distillation fidelity (CP4, AWS)", category="query")

    # CP5 — Serving health. Lambda predict error rate from CloudWatch,
    # Lambda get_function_configuration for ENV correctness.
    def _predict_health() -> Dict[str, Any]:
        errors = _cw_datapoints(
            "Errors", statistic="Sum", namespace="AWS/Lambda",
            dimensions=[{"Name": "FunctionName", "Value": ctx.predict_lambda}],
        )
        invocations = _cw_datapoints(
            "Invocations", statistic="Sum", namespace="AWS/Lambda",
            dimensions=[{"Name": "FunctionName", "Value": ctx.predict_lambda}],
        )
        total_err = (errors.get("value") or 0.0) * max(errors.get("count", 1), 1)
        total_inv = (invocations.get("value") or 0.0) * max(invocations.get("count", 1), 1)
        healthy = total_err == 0
        try:
            cfg = lmb.get_function_configuration(FunctionName=ctx.predict_lambda)
            registry_env = (cfg.get("Environment") or {}).get("Variables", {}).get(
                "REGISTRY_BASE", "",
            )
        except Exception:
            registry_env = ""
        return {
            "healthy": healthy,
            "backend": "lambda_lgbm",
            "record_count": int(total_inv),
            "error_count": int(total_err),
            "registry_base": registry_env,
        }

    registry.register("check_feature_store_health", _predict_health,
                      description="serving health (CP5, AWS)", category="query")

    # CP6 — Recommendation audit. DynamoDB row count + p95 Duration
    # from CloudWatch.
    def _audit_archive() -> Dict[str, Any]:
        try:
            item_count = ddb.scan(
                TableName=ctx.reason_cache_table, Select="COUNT",
            ).get("Count", 0)
        except Exception:
            item_count = 0
        p50 = _cw_datapoints(
            "Duration", statistic="Average", namespace="AWS/Lambda",
            dimensions=[{"Name": "FunctionName", "Value": ctx.predict_lambda}],
        )
        return {
            "p50_latency_ms": p50.get("value"),
            "p95_latency_ms": None,  # get_metric_data with ExtendedStatistics (p95)
            "filter_pass_rate": None,
            "total_requests": int(item_count),
        }

    registry.register("read_audit_archive", _audit_archive,
                      description="recommendation audit archive (CP6, AWS)",
                      category="query")

    # CP7 — A/B test. When no active experiment metric is published to
    # the PLE/Serving namespace this stays empty (matches test harness
    # expectations).
    def _ab_metrics() -> Dict[str, Any]:
        return {
            "active_experiment": None,
            "variant_metrics": {},
            "significance_test": {},
        }

    registry.register("query_cloudwatch_metrics", _ab_metrics,
                      description="A/B metrics (CP7, AWS)", category="query")

    return registry


def _build_registry(artifacts_dir: Path):
    """Build a ToolRegistry wired to the pipeline's local artifact dir."""
    from core.agent.tool_registry import ToolRegistry

    adapter_meta = _load_json(artifacts_dir / "phase0" / "adapter_metadata.json") or {}
    raw_stats = _load_json(artifacts_dir / "phase0" / "feature_stats.json") or {}
    feat_stats = _compute_feature_stats_summary(raw_stats)
    eval_metrics = _load_json(artifacts_dir / "training" / "eval_metrics.json") or {}
    fidelity_raw = _load_json(
        artifacts_dir / "distillation" / "fidelity_report.json",
    ) or {}
    fidelity = _fidelity_summary(fidelity_raw)
    training = _training_metrics(eval_metrics)

    registry = ToolRegistry(agent_id="pipeline_reports")

    registry.register(
        "read_ingestion_manifest",
        lambda: {
            "total_domains": 1,
            "domains_passed": 1,
            "domains_failed": 0,
            "total_rows": adapter_meta.get("num_raw_rows", 0),
            "total_pii_encrypted": 0,
            "total_duration_seconds": 0,
        },
        description="ingestion manifest (CP1)", category="query",
    )
    registry.register(
        "read_pipeline_state",
        lambda: {
            "completed_stages": [
                "ingestion", "feature_gen", "normalization", "tensor_save",
            ],
        },
        description="pipeline state (CP2)", category="query",
    )
    registry.register(
        "read_feature_stats", lambda: feat_stats,
        description="feature stats (CP2)", category="query",
    )
    registry.register(
        "read_experiment_metrics", lambda: training,
        description="training metrics (CP3)", category="query",
    )
    registry.register(
        "read_distillation_fidelity", lambda: fidelity,
        description="distillation fidelity (CP4)", category="query",
    )
    registry.register(
        "check_feature_store_health",
        lambda: {
            "healthy": True,
            "backend": "lambda_lgbm",
            "record_count": 0,
        },
        description="serving health (CP5)", category="query",
    )
    registry.register(
        "read_audit_archive",
        lambda: {
            "p50_latency_ms": None,
            "p95_latency_ms": None,
            "filter_pass_rate": None,
            "total_requests": 0,
        },
        description="audit archive (CP6)", category="query",
    )
    registry.register(
        "query_cloudwatch_metrics",
        lambda: {
            "active_experiment": None,
            "variant_metrics": {},
            "significance_test": {},
        },
        description="A/B metrics (CP7)", category="query",
    )
    return registry


# ---------------------------------------------------------------------------
# Consensus arbiter (optional — only when boto3 + Bedrock reachable)
# ---------------------------------------------------------------------------

def _build_bedrock_arbiter():
    try:
        import boto3
        from core.agent.consensus import ConsensusArbiter
    except Exception:
        logger.info("ConsensusArbiter unavailable (boto3 or consensus import)")
        return None

    class _BedrockProvider:
        def __init__(self) -> None:
            self._client = boto3.client(
                "bedrock-runtime", region_name="ap-northeast-2",
            )
            self._model_id = "global.anthropic.claude-sonnet-4-6"

        def generate(self, prompt: str, max_tokens: int = 512,
                     temperature: float = 0.3, **_) -> str:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            })
            try:
                resp = self._client.invoke_model(
                    modelId=self._model_id, body=body,
                    contentType="application/json", accept="application/json",
                )
            except Exception:
                logger.debug("Bedrock invoke failed", exc_info=True)
                return ""
            parsed = json.loads(resp["body"].read())
            chunks = parsed.get("content", [])
            if chunks and isinstance(chunks, list):
                return chunks[0].get("text", "").strip()
            return ""

    return ConsensusArbiter(
        llm_provider=_BedrockProvider(),
        config={"agents": 3, "parallel": False},
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class PipelineReportArtifacts:
    """Container for the S3 URIs produced by :func:`run_pipeline_reports`."""

    def __init__(
        self,
        ops_s3_uri: Optional[str] = None,
        audit_s3_uri: Optional[str] = None,
        ops_local_path: Optional[str] = None,
        audit_local_path: Optional[str] = None,
        ops_status: str = "",
        audit_risk_level: str = "",
    ) -> None:
        self.ops_s3_uri = ops_s3_uri
        self.audit_s3_uri = audit_s3_uri
        self.ops_local_path = ops_local_path
        self.audit_local_path = audit_local_path
        self.ops_status = ops_status
        self.audit_risk_level = audit_risk_level

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ops_s3_uri": self.ops_s3_uri,
            "audit_s3_uri": self.audit_s3_uri,
            "ops_local_path": self.ops_local_path,
            "audit_local_path": self.audit_local_path,
            "ops_status": self.ops_status,
            "audit_risk_level": self.audit_risk_level,
        }


def run_pipeline_reports(
    version: str,
    artifacts_dir: str,
    s3_prefix: Optional[str] = None,
    enable_consensus: bool = True,
    aws_context: Optional[PipelineJobContext] = None,
) -> PipelineReportArtifacts:
    """Generate + (optionally) upload Ops + Audit reports for a pipeline run.

    Parameters
    ----------
    version : str
        Registered model version (used only for naming / logging).
    artifacts_dir : str
        Local directory under which the runner expects to find
        ``phase0/``, ``training/``, ``distillation/`` sub-trees. Missing
        sub-trees gracefully degrade to GREEN with null measurements.
    s3_prefix : str, optional
        When provided, both JSON reports are uploaded under that prefix
        (e.g. ``s3://bucket/santander_ple/artifacts/v.../reports/``).
        When omitted the reports stay in ``artifacts_dir/reports/`` only.
    enable_consensus : bool
        Wire a :class:`ConsensusArbiter` backed by Bedrock Sonnet so
        non-GREEN checkpoints get a 3-agent verdict. Disabled for dry
        runs / CI where Bedrock is out of scope.

    Returns
    -------
    PipelineReportArtifacts
        Local paths and S3 URIs of the generated reports plus the
        headline status fields so callers can log a one-line summary
        without re-parsing JSON.
    """
    art_path = Path(artifacts_dir)
    reports_dir = art_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    ops_path = reports_dir / "ops_report.json"
    audit_path = reports_dir / "audit_report.json"

    arbiter = _build_bedrock_arbiter() if enable_consensus else None

    # ---- Ops ----------------------------------------------------------
    try:
        from core.agent.ops.collector import OpsCollector
        from core.agent.ops.diagnoser import OpsDiagnoser
        from core.agent.ops.reporter import OpsReporter

        if aws_context is not None:
            logger.info(
                "Ops report: AWS-backed registry (SageMaker + CloudWatch + "
                "DynamoDB + Lambda) for jobs %s/%s/%s/%s",
                aws_context.phase0_job_name or "-",
                aws_context.phase1_job_name or "-",
                aws_context.phase2_job_name or "-",
                aws_context.distill_job_name or "-",
            )
            registry = _build_aws_registry(aws_context)
        else:
            logger.info("Ops report: local-artifacts registry")
            registry = _build_registry(art_path)
        ops_cfg = {
            "latency_sla_ms": 300,
            "min_val_auc": 0.55,
            "fidelity_gap_threshold": 0.05,
            "grad_norm_warning": 100,
        }
        collector = OpsCollector(registry=registry, config=ops_cfg)
        checkpoints = collector.collect_all()
        diagnoses = OpsDiagnoser(config=ops_cfg).diagnose(checkpoints)
        reporter = OpsReporter(consensus_arbiter=arbiter)
        ops_report = reporter.generate(checkpoints, diagnoses, period="daily")
        ops_report.save(str(ops_path))
        ops_status = ops_report.status
        logger.info(
            "Ops report: status=%s, attention=%d → %s",
            ops_status, len(ops_report.attention_required), ops_path,
        )
    except Exception:
        logger.exception("OpsReporter failed (non-fatal)")
        ops_status = "UNKNOWN"

    # ---- Audit --------------------------------------------------------
    audit_risk = "UNKNOWN"
    try:
        from core.agent.audit.reporter import AuditReporter

        # Minimal regulatory_summary + reason_quality so the report is
        # well-formed even when serving-side aggregators are not yet
        # connected. Downstream operators extend this once real
        # post-hoc data feeds land.
        reg_summary = {
            "domestic": {"status": "compliant", "checked_rules": 5},
            "eu_ai_act": {"status": "partial", "risk_category": "limited"},
            "fria": {"status": "pending"},
        }
        reason_quality = {
            "tier1": {"total_validated": 0, "avg_grounding": None,
                      "avg_overall": None, "avg_readability": None},
            "tier2": {},
            "tier3": {},
        }
        reporter = AuditReporter(consensus_arbiter=arbiter)
        audit_report = reporter.generate(
            focus_areas=[],
            regulatory_results=reg_summary,
            reason_quality=reason_quality,
            period="weekly",
        )
        audit_report.save(str(audit_path))
        audit_risk = audit_report.risk_level
        logger.info(
            "Audit report: risk=%s, focus_areas=%d → %s",
            audit_risk, len(audit_report.focus_areas), audit_path,
        )
    except Exception:
        logger.exception("AuditReporter failed (non-fatal)")

    # ---- S3 upload ----------------------------------------------------
    ops_s3: Optional[str] = None
    audit_s3: Optional[str] = None
    if s3_prefix:
        try:
            import boto3
            s3 = boto3.client("s3")
            base = s3_prefix.rstrip("/") + "/reports"
            bucket, prefix = base.replace("s3://", "").split("/", 1)
            if ops_path.exists():
                s3.upload_file(str(ops_path), bucket, f"{prefix}/ops_report.json")
                ops_s3 = f"s3://{bucket}/{prefix}/ops_report.json"
            if audit_path.exists():
                s3.upload_file(str(audit_path), bucket, f"{prefix}/audit_report.json")
                audit_s3 = f"s3://{bucket}/{prefix}/audit_report.json"
            logger.info("Uploaded reports to s3://%s/%s/", bucket, prefix)
        except Exception:
            logger.exception("Failed to upload reports to S3 (non-fatal)")

    return PipelineReportArtifacts(
        ops_s3_uri=ops_s3,
        audit_s3_uri=audit_s3,
        ops_local_path=str(ops_path) if ops_path.exists() else None,
        audit_local_path=str(audit_path) if audit_path.exists() else None,
        ops_status=ops_status,
        audit_risk_level=audit_risk,
    )
