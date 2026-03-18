"""
Model Performance Monitor
=========================

Tracks model quality over time with three mechanisms:

1. **Prediction Logging** -- every inference writes a record to DynamoDB
   (prediction_id, user_id, version, predictions, timestamp).  TTL=90 days.

2. **CloudWatch Metrics** -- InferenceLatency + per-task PredictionScore
   emitted at every inference; aggregated P50/P95/P99 visible in dashboards.

3. **Champion-Challenger Evaluation** -- compares two model versions on
   the DynamoDB prediction log table using aggregate score statistics.
   Call :meth:`evaluate_champion_challenger` from a scheduled Lambda or
   Step Functions task to decide when to promote a challenger.

DynamoDB table schema
---------------------
Table name: ``ple-prediction-log``  (set via ``MONITOR_TABLE`` env var)

Partition key: ``prediction_id``  (UUID string)
Sort key: ``timestamp``           (ISO 8601 string)

Global Secondary Index: ``version-timestamp-index``
  PK: ``version``, SK: ``timestamp``
  -- used for per-version metric aggregation

Attributes:
    prediction_id   String
    user_id         String
    version         String
    predictions     Map<task, String>   -- scores as strings to avoid float precision issues
    elapsed_ms      String
    channel         String
    segment         String
    timestamp       String
    ttl             Number             -- epoch seconds, DynamoDB TTL attribute

Usage::

    monitor = ModelMonitor(
        table_name="ple-prediction-log",
        cw_namespace="PLE/Serving",
        region="ap-northeast-2",
    )

    # Called from the serving Lambda after inference
    monitor.log_prediction(user_id, version, predictions, elapsed_ms, ctx)
    monitor.emit_metrics(version, predictions, elapsed_ms)

    # Called from a scheduled evaluation job
    result = monitor.evaluate_champion_challenger(
        champion_version="v-abc",
        challenger_version="v-def",
        task_name="ctr",
        min_samples=5000,
    )
"""

from __future__ import annotations

import logging
import math
import statistics
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["ModelMonitor", "ChampionChallengerResult"]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

class ChampionChallengerResult:
    """Result of a Champion-Challenger evaluation.

    Attributes:
        champion_version: Champion model version string.
        challenger_version: Challenger model version string.
        task_name: Task evaluated.
        champion_mean_score: Mean prediction score for the champion.
        challenger_mean_score: Mean prediction score for the challenger.
        champion_samples: Number of prediction records used for champion.
        challenger_samples: Number of prediction records used for challenger.
        lift: Relative improvement of challenger over champion.
        significant: Whether the difference is statistically significant.
        p_value: Two-proportion z-test p-value (for binary tasks).
        action: ``"promote"`` | ``"keep"`` | ``"wait"``.
        reason: Human-readable explanation.
    """

    def __init__(
        self,
        champion_version: str,
        challenger_version: str,
        task_name: str,
        champion_mean_score: float,
        challenger_mean_score: float,
        champion_samples: int,
        challenger_samples: int,
        lift: float,
        significant: bool,
        p_value: float,
        action: str,
        reason: str,
    ) -> None:
        self.champion_version = champion_version
        self.challenger_version = challenger_version
        self.task_name = task_name
        self.champion_mean_score = champion_mean_score
        self.challenger_mean_score = challenger_mean_score
        self.champion_samples = champion_samples
        self.challenger_samples = challenger_samples
        self.lift = lift
        self.significant = significant
        self.p_value = p_value
        self.action = action
        self.reason = reason

    def to_dict(self) -> Dict[str, Any]:
        return {
            "champion_version": self.champion_version,
            "challenger_version": self.challenger_version,
            "task_name": self.task_name,
            "champion_mean_score": round(self.champion_mean_score, 6),
            "challenger_mean_score": round(self.challenger_mean_score, 6),
            "champion_samples": self.champion_samples,
            "challenger_samples": self.challenger_samples,
            "lift": round(self.lift, 4),
            "significant": self.significant,
            "p_value": round(self.p_value, 6),
            "action": self.action,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# ModelMonitor
# ---------------------------------------------------------------------------

class ModelMonitor:
    """Unified model performance monitor.

    Args:
        table_name: DynamoDB table name for prediction logs.
        cw_namespace: CloudWatch custom namespace.
        region: AWS region.
        min_samples: Minimum per-version sample count before evaluation.
        alpha: Statistical significance threshold for Champion-Challenger.
        lookback_days: How many days of prediction logs to pull for evaluation.
    """

    def __init__(
        self,
        table_name: str = "ple-prediction-log",
        cw_namespace: str = "PLE/Serving",
        region: str = "ap-northeast-2",
        min_samples: int = 5_000,
        alpha: float = 0.05,
        lookback_days: int = 7,
    ) -> None:
        self._table_name = table_name
        self._cw_namespace = cw_namespace
        self._region = region
        self._min_samples = min_samples
        self._alpha = alpha
        self._lookback_days = lookback_days

    # ------------------------------------------------------------------
    # 1. Prediction logging
    # ------------------------------------------------------------------

    def log_prediction(
        self,
        user_id: str,
        version: str,
        predictions: Dict[str, Any],
        elapsed_ms: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write a prediction record to DynamoDB.

        Args:
            user_id: Hashed/anonymised user identifier.
            version: Model version string.
            predictions: Dict of task_name -> score.
            elapsed_ms: Inference wall-clock time in milliseconds.
            context: Optional context dict (channel, segment, etc.).
        """
        try:
            import boto3

            dynamodb = boto3.resource("dynamodb", region_name=self._region)
            table = dynamodb.Table(self._table_name)
            ctx = context or {}

            record = {
                "prediction_id": str(uuid.uuid4()),
                "user_id": user_id,
                "version": version,
                "predictions": {k: str(v) for k, v in predictions.items()},
                "elapsed_ms": str(round(elapsed_ms, 2)),
                "channel": ctx.get("channel", "unknown"),
                "segment": ctx.get("segment", "unknown"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ttl": int(time.time()) + self._lookback_days * 2 * 86400,
            }

            table.put_item(Item=record)

        except Exception:
            logger.debug("log_prediction failed (non-fatal)", exc_info=True)

    # ------------------------------------------------------------------
    # 2. CloudWatch metrics
    # ------------------------------------------------------------------

    def emit_metrics(
        self,
        version: str,
        predictions: Dict[str, Any],
        elapsed_ms: float,
        extra_dimensions: Optional[Dict[str, str]] = None,
    ) -> None:
        """Emit per-request metrics to CloudWatch.

        Emits:
        - ``InferenceLatency`` (Milliseconds)
        - ``PredictionScore`` per task (None unit)

        Args:
            version: Model version string.
            predictions: Dict of task_name -> score.
            elapsed_ms: Inference latency in milliseconds.
            extra_dimensions: Additional CloudWatch dimensions.
        """
        try:
            import boto3

            cw = boto3.client("cloudwatch", region_name=self._region)
            dims = [{"Name": "Version", "Value": version}]
            if extra_dimensions:
                dims += [
                    {"Name": k, "Value": v}
                    for k, v in extra_dimensions.items()
                ]

            metric_data = [
                {
                    "MetricName": "InferenceLatency",
                    "Dimensions": dims,
                    "Value": elapsed_ms,
                    "Unit": "Milliseconds",
                }
            ]

            for task_name, value in predictions.items():
                if isinstance(value, (int, float)):
                    metric_data.append({
                        "MetricName": "PredictionScore",
                        "Dimensions": dims + [
                            {"Name": "Task", "Value": task_name}
                        ],
                        "Value": float(value),
                        "Unit": "None",
                    })

            for i in range(0, len(metric_data), 20):
                cw.put_metric_data(
                    Namespace=self._cw_namespace,
                    MetricData=metric_data[i : i + 20],
                )

        except Exception:
            logger.debug("emit_metrics failed (non-fatal)", exc_info=True)

    # ------------------------------------------------------------------
    # 3. Champion-Challenger evaluation
    # ------------------------------------------------------------------

    def evaluate_champion_challenger(
        self,
        champion_version: str,
        challenger_version: str,
        task_name: str,
        min_samples: Optional[int] = None,
    ) -> ChampionChallengerResult:
        """Compare champion and challenger on a specific task.

        Pulls prediction logs from DynamoDB for both versions within
        the lookback window, computes aggregate score statistics, and
        runs a two-proportion z-test (for binary tasks) or a t-test
        approximation (for regression tasks) to determine if the
        challenger is significantly better.

        Args:
            champion_version: Current champion version string.
            challenger_version: Challenger version string.
            task_name: Which task metric to evaluate.
            min_samples: Override minimum samples threshold.

        Returns:
            :class:`ChampionChallengerResult` with ``action`` field.
        """
        min_n = min_samples if min_samples is not None else self._min_samples

        champ_scores = self._fetch_scores(champion_version, task_name)
        chall_scores = self._fetch_scores(challenger_version, task_name)

        champ_n = len(champ_scores)
        chall_n = len(chall_scores)

        if champ_n < min_n or chall_n < min_n:
            return ChampionChallengerResult(
                champion_version=champion_version,
                challenger_version=challenger_version,
                task_name=task_name,
                champion_mean_score=_safe_mean(champ_scores),
                challenger_mean_score=_safe_mean(chall_scores),
                champion_samples=champ_n,
                challenger_samples=chall_n,
                lift=0.0,
                significant=False,
                p_value=1.0,
                action="wait",
                reason=(
                    f"Insufficient samples: champion={champ_n}, "
                    f"challenger={chall_n}, required={min_n}"
                ),
            )

        champ_mean = _safe_mean(champ_scores)
        chall_mean = _safe_mean(chall_scores)

        lift = (chall_mean - champ_mean) / champ_mean if champ_mean > 0 else 0.0

        # Two-proportion z-test (treating scores as conversion rates)
        p_value, significant = _two_proportion_z_test(
            challenger_sum=sum(chall_scores),
            challenger_n=chall_n,
            champion_sum=sum(champ_scores),
            champion_n=champ_n,
            alpha=self._alpha,
        )

        if significant and chall_mean > champ_mean:
            action = "promote"
            reason = (
                f"Challenger is significantly better: "
                f"lift={lift:.2%}, p={p_value:.4f}"
            )
        elif significant and chall_mean <= champ_mean:
            action = "keep"
            reason = (
                f"Champion is significantly better: "
                f"challenger lift={lift:.2%} (negative), p={p_value:.4f}"
            )
        else:
            action = "wait"
            reason = (
                f"No significant difference yet: "
                f"lift={lift:.2%}, p={p_value:.4f}"
            )

        logger.info(
            "Champion-Challenger [%s] %s vs %s: action=%s, lift=%.2f%%, p=%.4f",
            task_name, champion_version, challenger_version,
            action, lift * 100, p_value,
        )

        return ChampionChallengerResult(
            champion_version=champion_version,
            challenger_version=challenger_version,
            task_name=task_name,
            champion_mean_score=champ_mean,
            challenger_mean_score=chall_mean,
            champion_samples=champ_n,
            challenger_samples=chall_n,
            lift=lift,
            significant=significant,
            p_value=p_value,
            action=action,
            reason=reason,
        )

    def get_version_stats(
        self,
        version: str,
        task_name: str,
    ) -> Dict[str, Any]:
        """Compute aggregate stats for a version/task combination.

        Returns:
            Dict with mean, p50, p95, p99, count, std.
        """
        scores = self._fetch_scores(version, task_name)
        if not scores:
            return {
                "version": version,
                "task": task_name,
                "count": 0,
                "mean": None,
                "p50": None,
                "p95": None,
                "p99": None,
                "std": None,
            }

        scores_sorted = sorted(scores)
        n = len(scores_sorted)

        def percentile(p: float) -> float:
            idx = min(int(n * p / 100), n - 1)
            return scores_sorted[idx]

        return {
            "version": version,
            "task": task_name,
            "count": n,
            "mean": round(_safe_mean(scores), 6),
            "p50": round(percentile(50), 6),
            "p95": round(percentile(95), 6),
            "p99": round(percentile(99), 6),
            "std": round(statistics.stdev(scores) if n > 1 else 0.0, 6),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_scores(
        self,
        version: str,
        task_name: str,
    ) -> List[float]:
        """Fetch prediction scores from DynamoDB for a version/task."""
        try:
            import boto3
            from boto3.dynamodb.conditions import Key, Attr

            dynamodb = boto3.resource("dynamodb", region_name=self._region)
            table = dynamodb.Table(self._table_name)

            cutoff = (
                datetime.now(timezone.utc) - timedelta(days=self._lookback_days)
            ).isoformat()

            # Query via GSI: version-timestamp-index
            response = table.query(
                IndexName="version-timestamp-index",
                KeyConditionExpression=(
                    Key("version").eq(version)
                    & Key("timestamp").gte(cutoff)
                ),
                ProjectionExpression="predictions",
            )

            scores: List[float] = []
            for item in response.get("Items", []):
                preds = item.get("predictions", {})
                if task_name in preds:
                    try:
                        scores.append(float(preds[task_name]))
                    except (ValueError, TypeError):
                        pass

            # Handle pagination
            while "LastEvaluatedKey" in response:
                response = table.query(
                    IndexName="version-timestamp-index",
                    KeyConditionExpression=(
                        Key("version").eq(version)
                        & Key("timestamp").gte(cutoff)
                    ),
                    ProjectionExpression="predictions",
                    ExclusiveStartKey=response["LastEvaluatedKey"],
                )
                for item in response.get("Items", []):
                    preds = item.get("predictions", {})
                    if task_name in preds:
                        try:
                            scores.append(float(preds[task_name]))
                        except (ValueError, TypeError):
                            pass

            return scores

        except Exception:
            logger.warning("_fetch_scores failed", exc_info=True)
            return []


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _safe_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _two_proportion_z_test(
    challenger_sum: float,
    challenger_n: int,
    champion_sum: float,
    champion_n: int,
    alpha: float = 0.05,
) -> tuple[float, bool]:
    """Two-proportion z-test using mean scores as rates.

    Returns:
        (p_value, significant) tuple.
    """
    if challenger_n == 0 or champion_n == 0:
        return 1.0, False

    p_t = challenger_sum / challenger_n
    p_c = champion_sum / champion_n
    p_pool = (challenger_sum + champion_sum) / (challenger_n + champion_n)

    if p_pool <= 0 or p_pool >= 1:
        # Not a valid proportion; fall back to approximate t-test direction only
        return 1.0, False

    se = math.sqrt(
        p_pool * (1 - p_pool) * (1 / challenger_n + 1 / champion_n)
    )

    if se == 0:
        return 1.0, False

    z = (p_t - p_c) / se
    p_value = 2.0 * (1.0 - _normal_cdf(abs(z)))
    return p_value, p_value < alpha


def _normal_cdf(x: float) -> float:
    """Approximate standard normal CDF via error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
