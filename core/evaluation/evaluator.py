"""PLEEvaluator — per-task metric computation extracted from trainer.py.

Handles binary (AUC), multiclass (accuracy / F1-macro / top-K / NDCG@K),
and regression (MAE / RMSE / R²) tasks.  Aggregated metrics follow the
task-type separation rule from CLAUDE.md §1.7:

  avg_auc        → binary tasks only
  avg_f1_macro   → multiclass tasks only
  avg_mae        → regression tasks only
  avg_ndcg@3     → multiclass tasks that declare topk_k
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class PLEEvaluator:
    """Compute comprehensive per-task metrics and save results as JSON.

    Args:
        task_configs: List of task dicts from pipeline.yaml.  Each dict must
            contain at minimum:

            * ``name``       (str)
            * ``task_type``  (str) — ``"binary"``, ``"multiclass"``, or
              ``"regression"``

            Optional keys: ``num_classes``, ``topk_k`` (list[int] or int),
            ``loss``, ``loss_params``.

        task_val_masks: Optional mapping from task name to a boolean
            ``np.ndarray``.  When provided the mask is applied to both
            predictions and labels before computing metrics, enabling
            per-task validation subsets (e.g. tasks that are only defined
            for a small positive subset of the validation split).
    """

    def __init__(
        self,
        task_configs: List[Dict[str, Any]],
        task_val_masks: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        self.task_configs: Dict[str, Dict[str, Any]] = {
            t["name"]: t for t in task_configs
        }
        self.task_val_masks = task_val_masks or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        predictions: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """Compute all metrics for every task present in both dicts.

        Predictions and labels can be either:

        * Accumulated lists of tensors (``List[torch.Tensor]``) — will be
          concatenated automatically.
        * Already-concatenated tensors.

        Returns:
            Flat ``dict`` with:

            * Per-task keys: ``{task}_auc``, ``{task}_f1_macro``,
              ``{task}_f1_weighted``, ``{task}_mae``, ``{task}_rmse``,
              ``{task}_r2``, ``{task}_accuracy``,
              ``{task}_hit@K``, ``{task}_precision@K``, ``{task}_recall@K``,
              ``{task}_ndcg@K``.
            * Aggregated keys: ``avg_auc``, ``avg_accuracy`` (binary),
              ``avg_f1_macro`` (multiclass), ``avg_mae`` (regression),
              ``avg_ndcg@3`` (multiclass tasks with topk_k).
            * Meta keys: ``n_binary_tasks``, ``n_multiclass_tasks``,
              ``n_regression_tasks``, ``binary_task_names``,
              ``multiclass_task_names``, ``regression_task_names``.
        """
        try:
            from sklearn.metrics import (
                accuracy_score,
                confusion_matrix,
                f1_score,
                mean_absolute_error,
                r2_score,
                roc_auc_score,
            )
        except ImportError:
            logger.warning("sklearn not available — returning empty metrics.")
            return {}

        metrics: Dict[str, Any] = {}

        # Per-type accumulators
        binary_aucs: List[float] = []
        binary_tasks_seen: List[str] = []

        multiclass_accuracies: List[float] = []
        multiclass_f1s: List[float] = []
        multiclass_tasks_seen: List[str] = []
        topk_ndcg3_values: List[float] = []

        regression_maes: List[float] = []
        regression_tasks_seen: List[str] = []

        for task_name, pred_raw in predictions.items():
            label_raw = labels.get(task_name)
            if label_raw is None:
                continue

            # Support both accumulated lists and single tensors
            preds_t = (
                torch.cat(pred_raw) if isinstance(pred_raw, list) else pred_raw
            )
            labs_t = (
                torch.cat(label_raw) if isinstance(label_raw, list) else label_raw
            )

            if torch.isnan(preds_t).any() or torch.isnan(labs_t).any():
                logger.warning(
                    "NaN detected in %s predictions/labels — skipping.", task_name
                )
                continue

            preds_np: np.ndarray = preds_t.float().numpy()
            labs_np: np.ndarray = labs_t.float().numpy()

            # Apply per-task validation mask
            mask = self.task_val_masks.get(task_name)
            if mask is not None:
                n = min(len(mask), len(preds_np))
                m = mask[:n]
                preds_np = preds_np[m]
                labs_np = labs_np[m]
                if len(preds_np) == 0:
                    logger.debug(
                        "task_val_mask[%s]: 0 samples after masking — skipping.",
                        task_name,
                    )
                    continue

            task_cfg = self.task_configs.get(task_name, {})
            task_type = task_cfg.get("task_type", "binary")

            try:
                if task_type == "regression":
                    preds_f = preds_np.flatten()
                    labs_f = labs_np.flatten()
                    mae = float(mean_absolute_error(labs_f, preds_f))
                    rmse = float(np.sqrt(np.mean((preds_f - labs_f) ** 2)))
                    r2 = (
                        float(r2_score(labs_f, preds_f))
                        if np.var(labs_f) > 1e-10
                        else 0.0
                    )
                    metrics[f"{task_name}_mae"] = mae
                    metrics[f"{task_name}_rmse"] = rmse
                    metrics[f"{task_name}_r2"] = r2
                    regression_maes.append(mae)
                    regression_tasks_seen.append(task_name)

                elif task_type == "multiclass":
                    pred_classes = np.argmax(preds_np, axis=-1)
                    true_classes = labs_np.flatten().astype(int)
                    valid = true_classes >= 0
                    if valid.sum() < 2:
                        logger.debug(
                            "task %s: fewer than 2 valid samples — skipping.",
                            task_name,
                        )
                        continue

                    acc = float(
                        accuracy_score(true_classes[valid], pred_classes[valid])
                    )
                    f1_macro = float(
                        f1_score(
                            true_classes[valid],
                            pred_classes[valid],
                            average="macro",
                            zero_division=0,
                        )
                    )
                    f1_weighted = float(
                        f1_score(
                            true_classes[valid],
                            pred_classes[valid],
                            average="weighted",
                            zero_division=0,
                        )
                    )
                    cm = confusion_matrix(
                        true_classes[valid], pred_classes[valid]
                    ).tolist()

                    metrics[f"{task_name}_accuracy"] = acc
                    metrics[f"{task_name}_f1_macro"] = f1_macro
                    metrics[f"{task_name}_f1_weighted"] = f1_weighted
                    metrics[f"{task_name}_confusion_matrix"] = cm
                    multiclass_accuracies.append(acc)
                    multiclass_f1s.append(f1_macro)
                    multiclass_tasks_seen.append(task_name)

                    # Top-K metrics (opt-in via topk_k in task config)
                    k_values = self._get_topk_k(task_cfg)
                    if k_values:
                        topk_m = self.compute_topk_metrics(
                            preds_np[valid], true_classes[valid], k_values
                        )
                        for metric_key, metric_val in topk_m.items():
                            metrics[f"{task_name}_{metric_key}"] = metric_val
                        if "ndcg@3" in topk_m:
                            topk_ndcg3_values.append(topk_m["ndcg@3"])

                else:  # binary
                    unique = set(labs_np.flatten().tolist())
                    if unique <= {0.0, 1.0} and len(unique) == 2:
                        auc = float(roc_auc_score(labs_np, preds_np))
                        pred_bin = (preds_np.flatten() >= 0.5).astype(int)
                        labs_bin = labs_np.flatten().astype(int)
                        acc = float(accuracy_score(labs_bin, pred_bin))
                        f1 = float(
                            f1_score(labs_bin, pred_bin, zero_division=0)
                        )
                        cm = confusion_matrix(labs_bin, pred_bin).tolist()

                        metrics[f"{task_name}_auc"] = auc
                        metrics[f"{task_name}_accuracy"] = acc
                        metrics[f"{task_name}_f1"] = f1
                        metrics[f"{task_name}_confusion_matrix"] = cm
                        binary_aucs.append(auc)
                        binary_tasks_seen.append(task_name)
                    else:
                        logger.debug(
                            "task %s: labels not in {0,1} or single class "
                            "(unique=%s) — skipping AUC.",
                            task_name,
                            unique,
                        )

            except Exception:
                logger.debug(
                    "Metric computation failed for task %s.", task_name, exc_info=True
                )

        # ------------------------------------------------------------------
        # Aggregated metrics — task-type separated (CLAUDE.md §1.7)
        # ------------------------------------------------------------------
        if binary_aucs:
            metrics["avg_auc"] = float(np.mean(binary_aucs))
            metrics["avg_accuracy"] = float(
                np.mean(
                    [
                        metrics[f"{t}_accuracy"]
                        for t in binary_tasks_seen
                        if f"{t}_accuracy" in metrics
                    ]
                )
            )
        if multiclass_f1s:
            metrics["avg_f1_macro"] = float(np.mean(multiclass_f1s))
        if multiclass_accuracies:
            metrics["avg_multiclass_accuracy"] = float(np.mean(multiclass_accuracies))
        if topk_ndcg3_values:
            metrics["avg_ndcg@3"] = float(np.mean(topk_ndcg3_values))
        if regression_maes:
            metrics["avg_mae"] = float(np.mean(regression_maes))

        # Task-type count and name summary (pipe-delimited to fit flat dict)
        metrics["n_binary_tasks"] = float(len(binary_tasks_seen))
        metrics["n_multiclass_tasks"] = float(len(multiclass_tasks_seen))
        metrics["n_regression_tasks"] = float(len(regression_tasks_seen))
        metrics["binary_task_names"] = "|".join(binary_tasks_seen)
        metrics["multiclass_task_names"] = "|".join(multiclass_tasks_seen)
        metrics["regression_task_names"] = "|".join(regression_tasks_seen)

        return metrics

    def save(self, metrics: Dict[str, Any], path: str) -> None:
        """Serialise *metrics* to a JSON file at *path*.

        Creates parent directories as needed.  Values that are not
        JSON-serialisable (e.g. ``np.float32``) are cast to Python floats.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        def _default(obj: Any) -> Any:
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2, default=_default)
        logger.info("Evaluation metrics saved to %s", path)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compute_topk_metrics(
        logits: np.ndarray,
        labels: np.ndarray,
        k_values: List[int],
    ) -> Dict[str, float]:
        """Compute top-K ranking metrics for single-label multiclass tasks.

        Assumes one relevant item per sample (single-label ground truth), which
        is the case for ``nba_primary`` and ``next_mcc`` style tasks.

        Args:
            logits: Raw class scores, shape ``(n_samples, n_classes)``.
            labels: Integer class indices, shape ``(n_samples,)``.
            k_values: K values to evaluate, e.g. ``[1, 3, 5]``.

        Returns:
            Dict with the following keys for each K:

            * ``accuracy@K``  — fraction of samples whose true label is in top-K
            * ``hit@K``       — same as accuracy@K (recsys alias)
            * ``recall@K``    — same as hit@K for single-label tasks (|relevant|=1)
            * ``precision@K`` — hit@K / K
            * ``ndcg@K``      — binary-relevance NDCG (IDCG=1 per sample)
        """
        result: Dict[str, float] = {}
        if len(logits) == 0 or logits.ndim < 2:
            return result

        n_samples, n_classes = logits.shape
        max_k = min(max(k_values), n_classes)

        # Descending argsort; materialise only the columns we need.
        topk_preds = np.argsort(-logits, axis=1)[:, :max_k]  # (n, max_k)
        labels_col = labels.reshape(-1, 1)                    # (n, 1)

        # Binary match matrix: matches[i, j] == 1 iff topk_preds[i, j] == labels[i]
        matches = (topk_preds == labels_col).astype(np.float32)  # (n, max_k)

        for k in k_values:
            k_eff = min(k, n_classes)
            hit = matches[:, :k_eff].sum(axis=1)  # (n,) — 0 or 1
            hit_mean = float(hit.mean())

            result[f"accuracy@{k}"] = hit_mean
            result[f"hit@{k}"] = hit_mean
            result[f"recall@{k}"] = hit_mean          # single-label: recall == hit
            result[f"precision@{k}"] = hit_mean / k   # single-label: precision == hit/K

            # NDCG@K: binary relevance; IDCG = 1 (single relevant item)
            discounts = 1.0 / np.log2(
                np.arange(2, k_eff + 2, dtype=np.float32)
            )  # (k_eff,)
            dcg = (matches[:, :k_eff] * discounts).sum(axis=1)  # (n,)
            result[f"ndcg@{k}"] = float(dcg.mean())

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_topk_k(task_cfg: Dict[str, Any]) -> Optional[List[int]]:
        """Return the topk_k list for a task config, or None if not set."""
        val = task_cfg.get("topk_k")
        if val is None:
            return None
        if isinstance(val, int):
            return [val]
        if isinstance(val, (list, tuple)) and len(val) > 0:
            return [int(v) for v in val]
        return None
