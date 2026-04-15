"""
3-Layer Serving Fallback Router
=================================

Routes prediction requests through the 3-layer fallback architecture:

    Layer 1 — PLE → LGBM distillation  (best quality, requires teacher)
    Layer 2 — LGBM direct hard-label   (teacher quality below threshold)
    Layer 3 — Rule-based fallback       (LGBM unavailable or fidelity fail)

Routing decisions are fully config-driven (distillation.teacher_threshold,
distillation.fidelity, rule_engine.enabled).  No thresholds are hardcoded.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .rule_engine import RuleBasedRecommender

logger = logging.getLogger(__name__)

__all__ = ["FallbackRouter"]


class FallbackRouter:
    """Routes predictions through the 3-layer fallback architecture.

    For each task the router checks:
    1. Is a distilled LGBM available AND teacher quality above threshold?
       → Layer 1
    2. Is a direct LGBM available AND fidelity within acceptable bounds?
       → Layer 2
    3. Otherwise → Layer 3 (rule-based)

    Args:
        config: Merged pipeline config.  Keys consumed:

            ``distillation.teacher_threshold``
                binary_min_auc, multiclass_min_f1_ratio, regression_min_r2

            ``distillation.fidelity``
                binary / multiclass / regression sub-dicts with quality gates

            ``rule_engine.enabled``
                Whether Layer 3 is available (default True)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

        # Teacher quality thresholds (Layer 1 gate)
        thresh_cfg = config.get("distillation", {}).get("teacher_threshold", {})
        self.binary_min_auc: float = thresh_cfg.get("binary_min_auc", 0.60)
        self.multiclass_min_f1_ratio: float = thresh_cfg.get(
            "multiclass_min_f1_ratio", 2.0
        )
        self.regression_min_r2: float = thresh_cfg.get("regression_min_r2", 0.05)

        # Student fidelity thresholds (Layer 2 gate)
        fidelity_cfg = config.get("distillation", {}).get("fidelity", {})
        self.fidelity_binary: Dict[str, float] = fidelity_cfg.get("binary", {})
        self.fidelity_multiclass: Dict[str, float] = fidelity_cfg.get(
            "multiclass", {}
        )
        self.fidelity_regression: Dict[str, float] = fidelity_cfg.get(
            "regression", {}
        )

        # Layer 3 availability
        self.rule_engine_enabled: bool = config.get("rule_engine", {}).get(
            "enabled", True
        )

        # Task type lookup from tasks list
        self._task_type: Dict[str, str] = {
            t["name"]: t.get("type", "binary")
            for t in config.get("tasks", [])
        }

        logger.info(
            "FallbackRouter initialised: binary_min_auc=%.2f "
            "multiclass_min_f1_ratio=%.2f regression_min_r2=%.2f "
            "rule_engine_enabled=%s",
            self.binary_min_auc,
            self.multiclass_min_f1_ratio,
            self.regression_min_r2,
            self.rule_engine_enabled,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(
        self,
        task_name: str,
        lgbm_model=None,
        lgbm_fidelity: Optional[Dict[str, Any]] = None,
        teacher_metrics: Optional[Dict[str, float]] = None,
        rule_engine: Optional["RuleBasedRecommender"] = None,
    ) -> int:
        """Determine which layer to use for a single task.

        Args:
            task_name: Task identifier.
            lgbm_model: Fitted LGBM model (or None if not available).
            lgbm_fidelity: Fidelity validation results for the LGBM model.
                           Expected keys match distillation.fidelity schema.
            teacher_metrics: Teacher eval metrics for Layer 1 gate.
                             E.g. ``{"auc_roc": 0.72}`` for a binary task.
            rule_engine: RuleBasedRecommender instance for Layer 3.

        Returns:
            1 (distilled LGBM), 2 (direct LGBM), or 3 (rule-based).
        """
        task_type = self._task_type.get(task_name, "binary")

        # --- Layer 1 check: distilled LGBM with teacher quality gate ---
        if lgbm_model is not None and teacher_metrics is not None:
            teacher_ok = self._check_teacher_quality(task_type, teacher_metrics)
            fidelity_ok = self._check_fidelity(
                task_type, lgbm_fidelity or {}, layer=1
            )
            if teacher_ok and fidelity_ok:
                logger.debug(
                    "task=%s → Layer 1 (distilled LGBM)", task_name
                )
                return 1

        # --- Layer 2 check: direct LGBM with fidelity gate ---
        if lgbm_model is not None:
            fidelity_ok = self._check_fidelity(
                task_type, lgbm_fidelity or {}, layer=2
            )
            if fidelity_ok:
                logger.debug(
                    "task=%s → Layer 2 (direct LGBM)", task_name
                )
                return 2

        # --- Layer 3: rule-based fallback ---
        rule_available = (
            self.rule_engine_enabled
            and rule_engine is not None
            and task_name in (rule_engine.get_available_rules()
                              if rule_engine is not None else [])
        )
        if rule_available:
            logger.debug("task=%s → Layer 3 (rule-based)", task_name)
            return 3

        # Last resort: Layer 3 even if rule_engine is None (caller handles None)
        logger.warning(
            "task=%s: all layers unavailable, defaulting to Layer 3", task_name
        )
        return 3

    def route_all(
        self,
        task_names: List[str],
        lgbm_models: Optional[Dict[str, Any]] = None,
        lgbm_fidelities: Optional[Dict[str, Dict[str, Any]]] = None,
        teacher_metrics_all: Optional[Dict[str, Dict[str, float]]] = None,
        rule_engine: Optional["RuleBasedRecommender"] = None,
    ) -> Dict[str, int]:
        """Route all tasks at once.

        Args:
            task_names: List of task names to route.
            lgbm_models: ``{task_name: lgbm_model}`` mapping.
            lgbm_fidelities: ``{task_name: fidelity_dict}`` mapping.
            teacher_metrics_all: ``{task_name: metrics_dict}`` mapping.
            rule_engine: Shared rule engine instance.

        Returns:
            ``{task_name: layer_number}`` for every task in *task_names*.
        """
        lgbm_models = lgbm_models or {}
        lgbm_fidelities = lgbm_fidelities or {}
        teacher_metrics_all = teacher_metrics_all or {}

        routing: Dict[str, int] = {}
        layer_counts: Dict[int, int] = {1: 0, 2: 0, 3: 0}

        for task_name in task_names:
            layer = self.route(
                task_name=task_name,
                lgbm_model=lgbm_models.get(task_name),
                lgbm_fidelity=lgbm_fidelities.get(task_name),
                teacher_metrics=teacher_metrics_all.get(task_name),
                rule_engine=rule_engine,
            )
            routing[task_name] = layer
            layer_counts[layer] = layer_counts.get(layer, 0) + 1

        logger.info(
            "route_all: %d tasks → L1=%d L2=%d L3=%d",
            len(task_names),
            layer_counts[1],
            layer_counts[2],
            layer_counts[3],
        )
        return routing

    def explain(self, task_name: str, routing: Dict[str, int]) -> str:
        """Return a human-readable explanation of the routing decision.

        Args:
            task_name: Task name.
            routing: Output of :meth:`route_all`.

        Returns:
            Explanation string.
        """
        layer = routing.get(task_name)
        if layer == 1:
            return (
                f"[{task_name}] Layer 1 — Distilled LGBM: teacher quality "
                "above threshold and student fidelity within bounds."
            )
        elif layer == 2:
            return (
                f"[{task_name}] Layer 2 — Direct LGBM: teacher quality "
                "below threshold but student fidelity acceptable."
            )
        else:
            return (
                f"[{task_name}] Layer 3 — Rule-based fallback: LGBM "
                "unavailable or fidelity failed."
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_teacher_quality(
        self,
        task_type: str,
        teacher_metrics: Dict[str, float],
    ) -> bool:
        """Return True if teacher quality exceeds the configured minimum."""
        if task_type == "binary":
            auc = teacher_metrics.get("auc_roc", 0.0)
            ok = auc >= self.binary_min_auc
            if not ok:
                logger.debug(
                    "Teacher quality gate FAIL: binary auc_roc=%.4f < %.4f",
                    auc, self.binary_min_auc,
                )
            return ok

        elif task_type == "multiclass":
            # F1 > 2 / K where K = num_classes → stored as ratio in teacher_metrics
            f1_ratio = teacher_metrics.get("f1_ratio", 0.0)
            ok = f1_ratio >= self.multiclass_min_f1_ratio
            if not ok:
                logger.debug(
                    "Teacher quality gate FAIL: multiclass f1_ratio=%.4f < %.4f",
                    f1_ratio, self.multiclass_min_f1_ratio,
                )
            return ok

        elif task_type == "regression":
            r2 = teacher_metrics.get("r2", -999.0)
            ok = r2 >= self.regression_min_r2
            if not ok:
                logger.debug(
                    "Teacher quality gate FAIL: regression r2=%.4f < %.4f",
                    r2, self.regression_min_r2,
                )
            return ok

        # Unknown task type → allow through (no gate)
        logger.warning(
            "Unknown task_type '%s' for teacher quality gate, allowing.",
            task_type,
        )
        return True

    def _check_fidelity(
        self,
        task_type: str,
        fidelity: Dict[str, Any],
        layer: int,
    ) -> bool:
        """Return True if LGBM fidelity metrics are within configured bounds.

        For Layer 1 the bar is higher (all thresholds must pass).
        For Layer 2 we only require the primary metric to pass.
        """
        if not fidelity:
            # No fidelity data → assume acceptable for Layer 2, reject for Layer 1
            return layer == 2

        if task_type == "binary":
            cfg = self.fidelity_binary
            checks = []
            if "auc_gap" in fidelity and "max_auc_gap" in cfg:
                checks.append(
                    fidelity["auc_gap"] <= cfg["max_auc_gap"]
                )
            if "agreement" in fidelity and "min_agreement" in cfg:
                checks.append(
                    fidelity["agreement"] >= cfg["min_agreement"]
                )
            if "jsd" in fidelity and "max_jsd" in cfg:
                checks.append(
                    fidelity["jsd"] <= cfg["max_jsd"]
                )
            if "ranking_corr" in fidelity and "min_ranking_corr" in cfg:
                checks.append(
                    fidelity["ranking_corr"] >= cfg["min_ranking_corr"]
                )

        elif task_type == "multiclass":
            cfg = self.fidelity_multiclass
            checks = []
            if "agreement" in fidelity and "min_agreement" in cfg:
                checks.append(
                    fidelity["agreement"] >= cfg["min_agreement"]
                )
            if "jsd" in fidelity and "max_jsd" in cfg:
                checks.append(
                    fidelity["jsd"] <= cfg["max_jsd"]
                )
            if "f1_macro_gap" in fidelity and "max_f1_macro_gap" in cfg:
                checks.append(
                    fidelity["f1_macro_gap"] <= cfg["max_f1_macro_gap"]
                )

        elif task_type == "regression":
            cfg = self.fidelity_regression
            checks = []
            if "ranking_corr" in fidelity and "min_ranking_corr" in cfg:
                checks.append(
                    fidelity["ranking_corr"] >= cfg["min_ranking_corr"]
                )
            if "quartile_agreement" in fidelity and "min_quartile_agreement" in cfg:
                checks.append(
                    fidelity["quartile_agreement"]
                    >= cfg["min_quartile_agreement"]
                )
            if "mae_gap" in fidelity and "max_mae_gap" in cfg:
                checks.append(
                    fidelity["mae_gap"] <= cfg["max_mae_gap"]
                )
        else:
            checks = []

        if not checks:
            # No applicable checks → accept
            return True

        if layer == 1:
            # All checks must pass for Layer 1
            result = all(checks)
        else:
            # At least one check must pass for Layer 2
            result = any(checks)

        if not result:
            logger.debug(
                "Fidelity gate FAIL: task_type=%s layer=%d "
                "checks=%s fidelity=%s",
                task_type, layer, checks, fidelity,
            )
        return result
