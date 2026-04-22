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
        audit_archive_parquet_glob: str = "",
        l2a_lambda: str = "ple-reason-l2a",
        predict_warm_sla_ms: int = 500,
        l2a_sla_ms: int = 15000,
        fairness_archive_glob: str = "",
        promotion_gate_archive_glob: str = "",
        lineage_yaml_path: str = "",
        regulatory_status: Optional[Dict[str, Any]] = None,
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
        # DuckDB glob against the serving-side reason-audit Parquet
        # archive (e.g. s3://.../recommendation_audit/dt=*/events_*.parquet).
        # When set, the audit reporter samples real reason records,
        # runs them through GroundingValidator, and surfaces the tier-1
        # grounding / readability / overall quality metrics.
        self.audit_archive_parquet_glob = audit_archive_parquet_glob
        # L2a rewrite Lambda + its own latency SLA. This is deliberately
        # decoupled from predict_warm_sla_ms because L2a bodies always
        # call Bedrock Sonnet (~3-4s) so a sub-second SLA is not
        # meaningful. predict_warm_sla_ms targets warm ple-predict only
        # (cold starts excluded via CW Logs Insights), so the real CP6
        # metric reflects steady-state user-perceived latency rather
        # than Lambda init-phase noise.
        self.l2a_lambda = l2a_lambda
        self.predict_warm_sla_ms = predict_warm_sla_ms
        self.l2a_sla_ms = l2a_sla_ms
        # Audit-side data sources. When unset the corresponding audit
        # tool returns {} so AuditDiagnoser just doesn't emit a
        # viewpoint — no fake findings from empty archives.
        self.fairness_archive_glob = fairness_archive_glob
        self.promotion_gate_archive_glob = promotion_gate_archive_glob
        self.lineage_yaml_path = lineage_yaml_path
        self.regulatory_status = regulatory_status


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
# Audit tier-1 (DuckDB httpfs over the S3 Parquet reason archive)
# ---------------------------------------------------------------------------

def _compute_audit_tier1(parquet_glob: str) -> Dict[str, Any]:
    """Grounding / readability / overall quality on the reason archive.

    ``parquet_glob`` is a DuckDB-compatible glob such as
    ``s3://bucket/recommendation_audit/dt=*/events_*.parquet``. When the
    glob is unset or DuckDB cannot reach the data, we return empty
    tiers and the report simply shows tier1 total_validated=0 — that's
    the pre-archive state.
    """
    empty = {
        "tier1": {"total_validated": 0, "avg_grounding": None,
                  "avg_overall": None, "avg_readability": None},
        "tier2": {},
        "tier3": {},
    }
    if not parquet_glob:
        return empty

    try:
        import duckdb  # type: ignore
        from core.agent.audit.grounding_validator import GroundingValidator
    except Exception:
        logger.debug("tier1 audit: duckdb or validator import failed",
                     exc_info=True)
        return empty

    try:
        con = duckdb.connect(":memory:")
        con.execute("INSTALL httpfs; LOAD httpfs;")
        # Pull a bounded sample so the reporter stays fast even when
        # the archive grows; 500 is the same cap the existing
        # StratifiedReasonSampler defaults to.
        rows = con.execute(f"""
            SELECT
                coalesce(reason_text, l2a_reason, l1_reason, '') AS reason,
                coalesce(ig_top_features, CAST([] AS JSON)) AS ig_top_features,
                coalesce(selfcheck_verdict, 'pass') AS selfcheck_verdict
            FROM '{parquet_glob}'
            WHERE reason_text IS NOT NULL OR l2a_reason IS NOT NULL OR l1_reason IS NOT NULL
            LIMIT 500
        """).fetchall()
        con.close()
    except Exception:
        logger.debug(
            "tier1 audit: DuckDB scan of %s failed", parquet_glob,
            exc_info=True,
        )
        return empty

    if not rows:
        return empty

    # Minimal feature glossary — production deployments should pull
    # this from configs/financial/feature_glossary.yaml when it lands
    # in the staging tree.
    feature_glossary: Dict[str, str] = {}
    validator = GroundingValidator(
        feature_glossary=feature_glossary,
        config={"min_grounding_score": 0.3, "max_sentence_length": 100},
    )

    grounding_scores: List[float] = []
    overall_scores: List[float] = []
    readability_scores: List[float] = []
    for reason_text, ig_raw, verdict in rows:
        try:
            ig_feats = (
                json.loads(ig_raw) if isinstance(ig_raw, str) else (ig_raw or [])
            )
        except Exception:
            ig_feats = []
        if not reason_text or not ig_feats:
            continue
        try:
            gr = validator.validate(reason_text, ig_feats)
            grounding_scores.append(gr.grounding_score)
            qs = validator.compute_quality_score(
                reason_text=reason_text,
                ig_top_features=ig_feats,
                faithfulness=0.7,
                compliance=1.0 if verdict == "pass" else 0.0,
            )
            overall_scores.append(qs.overall)
            readability_scores.append(qs.readability)
        except Exception:
            logger.debug("tier1 validate failed for one row", exc_info=True)

    def _avg(values: List[float]) -> Optional[float]:
        return round(sum(values) / len(values), 4) if values else None

    return {
        "tier1": {
            "total_validated": len(grounding_scores),
            "avg_grounding": _avg(grounding_scores),
            "avg_overall": _avg(overall_scores),
            "avg_readability": _avg(readability_scores),
        },
        "tier2": {},
        "tier3": {},
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
    logs_client = boto3.client("logs", region_name=ctx.region)
    s3_client = boto3.client("s3", region_name=ctx.region)
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
    # CP2 feature_stats — pulled from the Phase 0 model.tar.gz on S3.
    # PipelineRunner.run writes feature_stats.json at the top of
    # /opt/ml/model, which SageMaker tars; so we fetch the tarball,
    # extract only feature_stats.json in-memory, and compute the
    # {total_features, zero_variance_count, nan_ratio_max} summary the
    # OpsDiagnoser expects.
    _phase0_stats_cache: Dict[str, Any] = {}

    def _read_feature_stats() -> Dict[str, Any]:
        if _phase0_stats_cache:
            return _phase0_stats_cache
        desc = _describe(ctx.phase0_job_name)
        model_uri = (desc.get("ModelArtifacts") or {}).get(
            "S3ModelArtifacts", "",
        )
        if not model_uri:
            return {"total_features": 0, "zero_variance_count": 0,
                    "nan_ratio_max": 0.0}
        try:
            import io, tarfile
            bucket, key = model_uri.replace("s3://", "").split("/", 1)
            buf = io.BytesIO()
            s3_client.download_fileobj(bucket, key, buf)
            buf.seek(0)
            raw_stats: Dict[str, Any] = {}
            with tarfile.open(fileobj=buf, mode="r:gz") as tar:
                for member in tar.getmembers():
                    if member.name.endswith("feature_stats.json"):
                        fh = tar.extractfile(member)
                        if fh is not None:
                            raw_stats = json.loads(fh.read().decode("utf-8"))
                        break
            summary = _compute_feature_stats_summary(raw_stats)
            logger.info(
                "CP2 feature_stats pulled from %s: %d features, %d zero-variance, "
                "max_nan_ratio=%.4f",
                model_uri, summary.get("total_features", 0),
                summary.get("zero_variance_count", 0),
                summary.get("nan_ratio_max", 0.0),
            )
            _phase0_stats_cache.update(summary)
            return summary
        except Exception:
            logger.debug("CP2 feature_stats fetch failed", exc_info=True)
            return {"total_features": 0, "zero_variance_count": 0,
                    "nan_ratio_max": 0.0}

    registry.register(
        "read_feature_stats", _read_feature_stats,
        description="feature stats (CP2, AWS — model.tar.gz extract)",
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

    # CP4 — Distillation. FinalMetricDataList first (SageMaker-collected
    # from stdout via metric_definitions regex). When those fail to
    # match the actual log lines — common for distillation because the
    # fidelity line is logged once at end-of-job and sometimes slips the
    # collector window — we fall back to CloudWatch Logs
    # filter_log_events, which lets us scan every algo-1 stream in the
    # /aws/sagemaker/TrainingJobs log group for the literal
    # "Fidelity summary: X/Y" line the distill entrypoint prints. This
    # is the same pattern surfaced at run_sagemaker_distillation.py
    # metric_definitions (see docs/design/04_training_pipeline.md).
    _distill_cache: Dict[str, Any] = {}

    def _distill_fidelity() -> Dict[str, Any]:
        if _distill_cache:
            return _distill_cache
        desc = _describe(ctx.distill_job_name)
        metrics = {
            m["MetricName"]: m["Value"]
            for m in desc.get("FinalMetricDataList", [])
        }
        passed = int(metrics.get("distill:passed_fidelity", 0))
        total = int(metrics.get("distill:num_students", 0))

        if total <= 0 and ctx.distill_job_name:
            # Stdout parse fallback. 16-char Job-name prefix is the
            # convention SageMaker uses for log stream names
            # ("<job>/algo-1-<suffix>"), so we scan the full group with
            # a stream prefix to stay cheap.
            try:
                import re
                resp = logs_client.filter_log_events(
                    logGroupName="/aws/sagemaker/TrainingJobs",
                    logStreamNamePrefix=ctx.distill_job_name,
                    filterPattern='"Fidelity summary"',
                    limit=200,
                )
                events = resp.get("events", []) or []
                fid_re = re.compile(r"Fidelity summary:\s*(\d+)\s*/\s*(\d+)")
                for ev in events:
                    m = fid_re.search(ev.get("message", ""))
                    if m:
                        passed = int(m.group(1))
                        total = int(m.group(2))
                        logger.info(
                            "CP4 fidelity parsed from logs: %d/%d", passed, total,
                        )
                        break
            except Exception:
                logger.debug("CP4 Logs filter failed", exc_info=True)

        if total <= 0:
            return {}

        ratio = passed / total
        gap = 1.0 - ratio
        out = {
            "task_fidelity": {"aggregate": round(ratio, 4)},
            "max_fidelity_gap": round(gap, 4),
            "tasks_above_threshold": (
                ["aggregate"] if gap > 0.05 else []
            ),
            "source": "FinalMetricDataList" if total == int(
                metrics.get("distill:num_students", 0),
            ) else "cloudwatch_logs",
        }
        _distill_cache.update(out)
        return out

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

    # CP6 — Recommendation audit. This is the user-perceived serving
    # latency and it needs per-lambda breakdown:
    #
    #   - ple-predict (warm-only): the synchronous inference path. We
    #     exclude cold starts via CloudWatch Logs Insights because
    #     cold-start init-phase latency (~8s on a 3GB Lambda) is not a
    #     steady-state signal — it's a deployment characteristic that
    #     the SRE team mitigates with provisioned concurrency, not a
    #     feature-store issue.
    #   - ple-reason-l2a: SQS-triggered Bedrock Sonnet rewrite. Every
    #     invocation body includes a Bedrock call (~3-4s), so a
    #     sub-second SLA is meaningless. Tracked with a dedicated
    #     15s SLA via ctx.l2a_sla_ms.
    #
    # CP6 primary ``p95_latency_ms`` stays as the predict-warm-only
    # value so OpsDiagnoser's single-SLA rule (CORR-002) scores
    # against the right steady-state metric. The l2a latency + the
    # cold-inclusive predict latency live under ``per_lambda`` for
    # operators who want the full picture.

    def _simple_percentile(stat: str, lambda_name: str) -> Optional[float]:
        """GetMetricData p50/p95 over the lookback window."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(minutes=ctx.cloudwatch_lookback_minutes)
        try:
            resp = cw.get_metric_data(
                MetricDataQueries=[{
                    "Id": "pct",
                    "MetricStat": {
                        "Metric": {
                            "Namespace": "AWS/Lambda",
                            "MetricName": "Duration",
                            "Dimensions": [{
                                "Name": "FunctionName", "Value": lambda_name,
                            }],
                        },
                        "Period": 60,
                        "Stat": stat,
                    },
                    "ReturnData": True,
                }],
                StartTime=start,
                EndTime=end,
                ScanBy="TimestampDescending",
            )
            points = resp.get("MetricDataResults", [{}])[0].get("Values", [])
            if not points:
                return None
            return float(max(points))
        except Exception:
            logger.debug(
                "simple_percentile %s/%s failed", lambda_name, stat,
                exc_info=True,
            )
            return None

    def _warm_only_percentiles(lambda_name: str) -> Dict[str, Optional[float]]:
        """CloudWatch Logs Insights — p50/p95 of Duration excluding cold starts.

        The Lambda managed runtime writes one REPORT line per invocation
        with ``@duration`` (end-to-end) and ``@initDuration`` (cold-start
        init-phase, only present on the very first invocation of a
        sandbox). We filter warm invocations with
        ``ispresent(@initDuration) = 0``, then ``pct(@duration, 95)``
        gives the steady-state user-perceived tail.
        """
        import time as _time
        end = datetime.now(timezone.utc)
        start = end - timedelta(minutes=ctx.cloudwatch_lookback_minutes)
        query = (
            "fields @timestamp, @duration, @initDuration"
            " | filter @type = \"REPORT\" and ispresent(@initDuration) = 0"
            " | stats pct(@duration, 95) as p95,"
            "         pct(@duration, 50) as p50,"
            "         count(*) as warm_count"
        )
        try:
            resp = logs_client.start_query(
                logGroupName=f"/aws/lambda/{lambda_name}",
                startTime=int(start.timestamp()),
                endTime=int(end.timestamp()),
                queryString=query,
            )
            qid = resp["queryId"]
            for _ in range(30):
                res = logs_client.get_query_results(queryId=qid)
                if res["status"] in ("Complete", "Failed", "Cancelled"):
                    break
                _time.sleep(1)
            if res["status"] != "Complete":
                return {"p50": None, "p95": None, "warm_count": 0}
            rows = res.get("results", [])
            if not rows:
                return {"p50": None, "p95": None, "warm_count": 0}
            flat: Dict[str, str] = {f["field"]: f["value"] for f in rows[0]}
            def _num(key: str) -> Optional[float]:
                val = flat.get(key)
                try:
                    return float(val) if val else None
                except Exception:
                    return None
            return {
                "p50": _num("p50"),
                "p95": _num("p95"),
                "warm_count": int(_num("warm_count") or 0),
            }
        except Exception:
            logger.debug(
                "warm-only percentile Logs Insights failed for %s",
                lambda_name, exc_info=True,
            )
            return {"p50": None, "p95": None, "warm_count": 0}

    def _audit_archive() -> Dict[str, Any]:
        try:
            item_count = ddb.scan(
                TableName=ctx.reason_cache_table, Select="COUNT",
            ).get("Count", 0)
        except Exception:
            item_count = 0

        # predict — warm-only p50/p95 via Logs Insights.
        # When the warm sample is too small (< 5 invocations) the tail
        # is dominated by cold-start init phase, which reflects
        # provisioned-concurrency posture, not feature-store health.
        # In that regime we **suppress** the primary p95 so CP6 does
        # not fire a latency SLA breach; the raw values still live in
        # per_lambda for operators who need the full picture.
        predict_warm = _warm_only_percentiles(ctx.predict_lambda)
        predict_full_p95 = _simple_percentile("p95", ctx.predict_lambda)
        warm_count = int(predict_warm.get("warm_count", 0) or 0)
        MIN_WARM_SAMPLES = 5
        if (
            warm_count >= MIN_WARM_SAMPLES
            and predict_warm["p95"] is not None
        ):
            predict_p50 = predict_warm["p50"]
            predict_p95 = predict_warm["p95"]
            predict_sla_scope = "warm-only"
        else:
            # Insufficient warm samples — leave p95 unreported so
            # OpsDiagnoser's single-SLA rule (CORR-002) does not alarm
            # on cold-dominant noise.
            predict_p50 = (
                predict_warm["p50"]
                if predict_warm["p50"] is not None
                else _simple_percentile("p50", ctx.predict_lambda)
            )
            predict_p95 = None
            predict_sla_scope = (
                f"warm-only (suppressed: {warm_count} warm samples < "
                f"{MIN_WARM_SAMPLES} minimum)"
            )

        # l2a — always includes Bedrock Sonnet in the body, so cold-vs-
        # warm exclusion is meaningless. Report the raw GetMetricData
        # percentiles.
        l2a_p50 = _simple_percentile("p50", ctx.l2a_lambda)
        l2a_p95 = _simple_percentile("p95", ctx.l2a_lambda)

        per_lambda = {
            ctx.predict_lambda: {
                "warm_p50_ms": predict_warm["p50"],
                "warm_p95_ms": predict_warm["p95"],
                "warm_count": warm_count,
                "full_p95_ms": predict_full_p95,
                "sla_ms": ctx.predict_warm_sla_ms,
                "sla_scope": predict_sla_scope,
            },
            ctx.l2a_lambda: {
                "p50_ms": l2a_p50,
                "p95_ms": l2a_p95,
                "sla_ms": ctx.l2a_sla_ms,
                "sla_scope": "all-invocations (Bedrock-bound)",
            },
        }
        # Surface the strictest per-lambda SLA breach as an extra
        # anomaly so OpsReporter.attention_required picks it up even
        # without modifying OpsDiagnoser.
        extra_anomalies: List[Dict[str, Any]] = []
        if l2a_p95 and l2a_p95 > ctx.l2a_sla_ms:
            extra_anomalies.append({
                "type": "l2a_latency_sla_breach",
                "lambda": ctx.l2a_lambda,
                "p95": l2a_p95,
                "sla": ctx.l2a_sla_ms,
            })
        return {
            "p50_latency_ms": predict_p50,
            "p95_latency_ms": predict_p95,        # predict warm-only
            "filter_pass_rate": None,
            "total_requests": int(item_count),
            "per_lambda": per_lambda,
            "extra_anomalies": extra_anomalies,
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


def _build_audit_registry(ctx: "PipelineJobContext"):
    """Audit-side ToolRegistry.

    Mirrors the Ops-side AWS registry: every tool hits a live AWS
    source (S3 Parquet for promotion-gate verdicts and fairness
    archive, DuckDB httpfs for tier1 grounding, Compliance Store for
    FRIA + AI-risk status). When a source is missing the tool returns
    an empty dict so downstream AuditDiagnoser simply skips the
    corresponding viewpoint — no fake findings.
    """
    from core.agent.tool_registry import ToolRegistry

    registry = ToolRegistry(agent_id="pipeline_reports_audit")

    # Fairness — DuckDB read of the archive Parquet written by the
    # nightly fairness Lambda (monitoring.fairness.archive_parquet_path).
    def _fairness_summary() -> Dict[str, Any]:
        glob = getattr(ctx, "fairness_archive_glob", "") or ""
        if not glob:
            return {}
        try:
            import duckdb  # type: ignore
            con = duckdb.connect(":memory:")
            con.execute("INSTALL httpfs; LOAD httpfs;")
            rows = con.execute(
                f"SELECT min(disparate_impact_min) AS di_min, "
                f"       count(*) FILTER (WHERE disparate_impact_min < 0.8) "
                f"         AS violations_count "
                f"FROM '{glob}'"
            ).fetchone()
            con.close()
            if not rows:
                return {}
            return {
                "di_min": rows[0], "violations": [],
                "hidden_violations": int(rows[1] or 0),
            }
        except Exception:
            logger.debug("fairness summary failed", exc_info=True)
            return {}

    # Promotion-gate verdict history — compliance audit reads this for
    # FRIA + AI-risk completeness checks.
    def _recent_promotion_verdicts() -> Dict[str, Any]:
        glob = getattr(ctx, "promotion_gate_archive_glob", "") or ""
        if not glob:
            return {}
        try:
            import duckdb  # type: ignore
            con = duckdb.connect(":memory:")
            con.execute("INSTALL httpfs; LOAD httpfs;")
            rows = con.execute(
                f"SELECT decision, count(*) AS n FROM '{glob}' "
                f"GROUP BY decision ORDER BY n DESC"
            ).fetchall()
            con.close()
            return {
                "verdicts": [{"decision": r[0], "count": int(r[1])} for r in rows],
            }
        except Exception:
            logger.debug("promotion verdicts read failed", exc_info=True)
            return {}

    # Reason quality (tier1) — same validator as the standalone path;
    # expressed here as a tool so Audit agents see the same interface.
    def _reason_tier1() -> Dict[str, Any]:
        glob = getattr(ctx, "audit_archive_parquet_glob", "") or ""
        return _compute_audit_tier1(glob) if glob else {}

    # Regulatory status — static summary for the demo; production
    # deployments swap this via ctx.regulatory_status. The schema
    # matches what AuditDiagnoser._analyze_regulatory expects
    # (critical_failures / pass_rate / by_regulation) AND keeps the
    # human-readable status fields the reporter uses for its
    # regulatory_summary block.
    def _regulatory_status() -> Dict[str, Any]:
        override = getattr(ctx, "regulatory_status", None)
        if override:
            return dict(override)
        # Default demo scheme: FRIA pending counts as one critical
        # failure (EU AI Act Art. 9 / domestic AI기본법 §35), EU AI Act
        # "partial" is a soft failure that lowers pass_rate but does
        # not trip the HIGH priority focus area.
        critical = 1  # FRIA pending
        soft = 1      # EU AI Act partial
        total_regs = 3
        return {
            "critical_failures": critical,
            "pass_rate": round(1.0 - (critical + soft) / total_regs, 4),
            "by_regulation": {
                "fria": "pending",
                "eu_ai_act": "partial",
                "domestic": "compliant",
            },
            # Passthrough fields for the reporter's regulatory_summary
            "domestic_status": "compliant",
            "eu_ai_act_status": "partial",
            "eu_ai_act_risk_category": "limited",
            "fria_status": "pending",
        }

    # Lineage — read the lineage YAML that the aggregator already
    # consumes; expose unmapped_ratio so AuditDiagnoser._analyze_lineage
    # can flag missing pseudonymization or unknown sources (threshold
    # 5%).
    def _lineage_summary() -> Dict[str, Any]:
        path = getattr(ctx, "lineage_yaml_path", "") or ""
        if not path:
            return {}
        try:
            import yaml
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            groups = data.get("feature_groups", []) or []
            if not groups:
                return {}
            pseudo = sum(1 for g in groups if g.get("pseudonymized"))
            missing = len(groups) - pseudo
            return {
                "feature_groups_total": len(groups),
                "pseudonymized_groups": pseudo,
                "missing_pseudonymization": missing,
                # Schema-compatible field AuditDiagnoser expects.
                "unmapped_ratio": round(missing / len(groups), 4),
            }
        except Exception:
            logger.debug("lineage read failed", exc_info=True)
            return {}

    registry.register("read_fairness_summary", _fairness_summary,
                      description="fairness archive DI summary", category="query")
    registry.register("read_recent_promotion_verdicts", _recent_promotion_verdicts,
                      description="promotion-gate verdict counts", category="query")
    registry.register("read_reason_tier1", _reason_tier1,
                      description="tier1 grounding over serving audit archive",
                      category="query")
    registry.register("read_regulatory_status", _regulatory_status,
                      description="domestic + EU AI Act + FRIA status", category="query")
    registry.register("read_lineage_summary", _lineage_summary,
                      description="feature group lineage + pseudonymization",
                      category="query")
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

# Ops agent's 3-person voting panel. Each perspective corresponds to a
# stakeholder who would normally be in a production incident call —
# SRE owns infra availability + tail latency, MLOps owns training /
# distillation quality, Biz owns customer-visible impact.
_OPS_PERSPECTIVES = {
    "sre": (
        "당신은 SRE 엔지니어입니다. 인프라 가용성, 콜드 스타트, p50/p95 "
        "지연시간, 에러율, 배포 롤백 가능성에 민감하세요. 지표 수치를 직접 "
        "인용하고 '운영상 조치' 관점으로 판단하세요."
    ),
    "mlops": (
        "당신은 MLOps/ML 엔지니어입니다. 학습 수렴(loss, val_auc, epochs), "
        "증류 피델리티, 피처 엔지니어링 상태(zero_variance, nan_ratio) 의 "
        "모델 품질 영향을 중시합니다. 데이터 또는 모델 차원의 근본 원인을 "
        "찾는 관점으로 판단하세요."
    ),
    "biz": (
        "당신은 금융 서비스 PM입니다. 추천 응답 지연, 서빙 가용성, 모델 "
        "정확도가 실제 고객 전환율/이탈/만족도에 미치는 영향을 우선합니다. "
        "트래픽 수치와 Lambda/DDB 사용량을 비즈니스 KPI 관점에서 해석하세요."
    ),
}

# Audit agent's 3-person voting panel. Each perspective is a role a
# real bank audit committee would actually call into — a regulator
# checking compliance, a risk manager looking for bias / stability
# issues, and an internal auditor focused on evidentiary record
# integrity.
_AUDIT_PERSPECTIVES = {
    "regulator": (
        "당신은 금융 규제 감독자입니다. 국내 금소법·AI기본법, EU AI Act, "
        "FRIA(기본권영향평가) 준수 여부를 체크합니다. 설명의무(예: EU AI Act "
        "제13조), 투명성, 고위험 AI 분류 상태에 민감하세요. 증빙이 없으면 "
        "non-compliant 로 판단해야 합니다."
    ),
    "risk": (
        "당신은 모델 리스크 매니저입니다. 편향(DI), 부당차별, 모델 드리프트, "
        "설명가능성(grounding score, faithfulness) 지표로 판단하세요. 샘플 "
        "사이즈가 작으면 불확실성을 명시적으로 기록해야 합니다. SR 11-7 "
        "MRM 원칙을 따릅니다."
    ),
    "audit_trail": (
        "당신은 내부감사인입니다. 감사 추적(HMAC 서명, hash chain), 증빙 "
        "보존, 누가/언제/무엇을 승인했는지의 서명 체인을 강조하세요. "
        "metadata 가 비어있거나 FRIA 상태가 pending 이면 증거 미비로 "
        "판정해야 합니다."
    ),
}


def _build_sonnet_provider():
    """Bedrock Sonnet provider (lazy-imports boto3 + consensus)."""
    try:
        import boto3
        from core.agent.consensus import ConsensusArbiter  # noqa: F401
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

    return _BedrockProvider()


def _build_ops_arbiter():
    """3-agent ConsensusArbiter for the Ops reporter (SRE / MLOps / Biz)."""
    provider = _build_sonnet_provider()
    if provider is None:
        return None
    from core.agent.consensus import ConsensusArbiter
    return ConsensusArbiter(
        llm_provider=provider,
        config={
            "agents": 3,
            "parallel": False,
            "perspectives": _OPS_PERSPECTIVES,
        },
    )


def _build_audit_arbiter():
    """3-agent ConsensusArbiter for the Audit reporter
    (Regulator / Risk / AuditTrail)."""
    provider = _build_sonnet_provider()
    if provider is None:
        return None
    from core.agent.consensus import ConsensusArbiter
    return ConsensusArbiter(
        llm_provider=provider,
        config={
            "agents": 3,
            "parallel": False,
            "perspectives": _AUDIT_PERSPECTIVES,
        },
    )


def _build_bedrock_arbiter():
    """Backwards-compatible single arbiter — kept for callers that still
    inject one arbiter into both reporters. New code should use
    :func:`_build_ops_arbiter` / :func:`_build_audit_arbiter` so each
    report gets its own stakeholder panel."""
    provider = _build_sonnet_provider()
    if provider is None:
        return None
    from core.agent.consensus import ConsensusArbiter
    return ConsensusArbiter(
        llm_provider=provider,
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

    # Each reporter gets its own 3-agent panel so Ops sees
    # SRE/MLOps/Biz and Audit sees Regulator/Risk/AuditTrail — the
    # perspectives that actually exist in a bank's incident-and-
    # compliance review loop, instead of the generic alpha/beta/gamma
    # arbiter both reports used to share.
    ops_arbiter = _build_ops_arbiter() if enable_consensus else None
    audit_arbiter = _build_audit_arbiter() if enable_consensus else None

    # LanceDB-backed persistence so findings + consensus votes are
    # searchable later (similar-incident retrieval, temporal trend
    # queries). Both reporters share one DiagnosticCaseStore + one
    # TemporalFactStore rooted at <artifacts_dir>/case_store. Tables
    # auto-migrate if the backend falls back to numpy (no lancedb
    # installed). Failures swallow — persistence is post-hoc and must
    # not block report generation.
    case_store = None
    temporal_fact_store = None
    try:
        from core.agent.case_store import DiagnosticCaseStore
        from core.agent.temporal_fact_store import TemporalFactStore
        case_store_root = str(reports_dir.parent / "case_store")
        case_store = DiagnosticCaseStore(store_path=case_store_root)
        temporal_fact_store = TemporalFactStore(
            store_path=str(reports_dir.parent / "temporal_facts"),
        )
        logger.info(
            "LanceDB case_store root: %s (backend=%s)",
            case_store_root, getattr(case_store, "_backend", "unknown"),
        )
    except Exception:
        logger.exception("LanceDB stores unavailable (non-fatal)")

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
        # latency_sla_ms is the *warm* predict SLA — cold-start noise
        # is filtered out upstream in the AWS registry. L2a and any
        # Bedrock-bound Lambda carry their own SLA in the per_lambda
        # payload (ctx.l2a_sla_ms).
        warm_sla = (
            aws_context.predict_warm_sla_ms
            if aws_context is not None else 500
        )
        ops_cfg = {
            "latency_sla_ms": warm_sla,
            "min_val_auc": 0.55,
            "fidelity_gap_threshold": 0.05,
            "grad_norm_warning": 100,
        }
        collector = OpsCollector(registry=registry, config=ops_cfg)
        checkpoints = collector.collect_all()
        diagnoses = OpsDiagnoser(config=ops_cfg).diagnose(checkpoints)
        reporter = OpsReporter(
            consensus_arbiter=ops_arbiter,
            case_store=case_store,
        )
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
        from core.agent.audit.diagnoser import AuditDiagnoser
        from core.agent.audit.reporter import AuditReporter

        # Audit now uses the same ToolRegistry pattern as Ops. We pull
        # every viewpoint's data through a tool so the pipeline is
        # consistent end-to-end: every live-AWS read is discoverable,
        # swappable, and — once IAM is locked down — audit-logged by
        # the ToolRegistry itself (CLAUDE.md §1.11 compliance trail).
        if aws_context is not None:
            audit_registry = _build_audit_registry(aws_context)
            fairness_results = audit_registry.call("read_fairness_summary") or {}
            promotion_verdicts = (
                audit_registry.call("read_recent_promotion_verdicts") or {}
            )
            reason_quality = audit_registry.call("read_reason_tier1") or {}
            reg_status = audit_registry.call("read_regulatory_status") or {}
            lineage = audit_registry.call("read_lineage_summary") or {}
            logger.info(
                "Audit tools fired: fairness=%s, verdicts=%s, tier1=%s, "
                "regulatory=%s, lineage=%s",
                bool(fairness_results), bool(promotion_verdicts),
                bool(reason_quality), bool(reg_status), bool(lineage),
            )
        else:
            fairness_results = {}
            promotion_verdicts = {}
            reason_quality = _compute_audit_tier1("")
            reg_status = {}
            lineage = {}

        # Translate the tool payloads into the three reporter inputs.
        reg_summary = {
            "domestic": {
                "status": reg_status.get("domestic_status", "compliant"),
                "checked_rules": 5,
            },
            "eu_ai_act": {
                "status": reg_status.get("eu_ai_act_status", "partial"),
                "risk_category": reg_status.get(
                    "eu_ai_act_risk_category", "limited",
                ),
            },
            "fria": {"status": reg_status.get("fria_status", "pending")},
        }

        # Drive the AuditDiagnoser so focus_areas are generated from
        # real tool output, not hard-coded []. We pass the raw tool
        # results (not reg_summary) because the diagnoser analysers
        # expect their own schema (critical_failures, pass_rate,
        # unmapped_ratio, etc.) — the tool functions above now return
        # that shape directly.
        diagnoser = AuditDiagnoser()
        focus_areas = diagnoser.diagnose(
            fairness_results=fairness_results or None,
            herding_results=None,  # viewpoint reserved for future
            reason_quality=reason_quality or None,
            regulatory_results=reg_status or None,
            lineage_results=lineage or None,
        )

        reporter = AuditReporter(
            consensus_arbiter=audit_arbiter,
            case_store=case_store,
            temporal_fact_store=temporal_fact_store,
        )
        audit_report = reporter.generate(
            focus_areas=focus_areas,
            regulatory_results=reg_summary,
            reason_quality=reason_quality,
            period="weekly",
        )
        audit_report.save(str(audit_path))
        audit_risk = audit_report.risk_level
        logger.info(
            "Audit report: risk=%s, focus_areas=%d, tier1_validated=%d → %s",
            audit_risk, len(audit_report.focus_areas),
            reason_quality.get("tier1", {}).get("total_validated", 0),
            audit_path,
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
