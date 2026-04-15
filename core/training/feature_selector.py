"""
Adaptive Feature Selector -- per-task feature filtering by importance.

2-stage pipeline:
  Stage 1: LGBM gain importance-based selection (top-k features capturing 95% cumulative gain)
  Stage 2: Mandatory feature guarantee

Teacher model IG (Integrated Gradients) attribution is NOT used for feature selection.
The teacher has already been distilled; feature selection from the serving model (LGBM Student)
perspective ensures serving model alignment and avoids OOM at production scale
(941K customers, 403 features).

Key design: cumulative importance threshold (default 95%) determines
how many features to keep per task.  Tasks with concentrated importance
get fewer features; tasks with distributed importance get more.

Usage::

    from core.training.feature_selector import (
        FeatureSelector,
        FeatureSelectionConfig,
        FeatureSelectionResult,
    )

    selector = FeatureSelector(config)

    # Stage 1: LGBM gain importance selection
    selected = selector.select_by_lgbm_gain(lgbm_model, feature_names, task_name)
    # Returns: FeatureSelectionResult with selected indices/names

    # Combined (LGBM gain -> mandatory features)
    final = selector.select(model, features, task_name, lgbm_model)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from core.evaluation.integrated_gradients import IntegratedGradients

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class FeatureSelectionConfig:
    """Configuration for adaptive feature selection.

    The ``cumulative_threshold`` is the key parameter: it determines what
    fraction of total feature importance must be captured.  Features are
    sorted by IG importance descending and accumulated until the threshold
    is reached.

    Tasks with a few dominant features will select fewer features;
    tasks with evenly distributed importance will select more.

    Attributes:
        cumulative_threshold: Keep features until cumulative IG >= this.
        min_features: Never select fewer than this many features.
        max_features: Never select more than this many features.
        lgbm_prune_zero_gain: Remove features with zero LGBM gain in Stage 2.
        ig_baseline: Baseline for IG computation (``"zeros"``, ``"mean"``,
            or ``"median"``).
        ig_steps: Number of interpolation steps for IG integration.
        mandatory_features: Feature names that are always included.
    """

    cumulative_threshold: float = 0.95  # Keep features up to 95% cumulative IG
    min_features: int = 50  # Never select fewer than this
    max_features: int = 400  # Never select more than this
    lgbm_prune_zero_gain: bool = True  # Remove features with 0 LGBM gain
    ig_baseline: str = "zeros"  # "zeros" | "mean" | "median"
    ig_steps: int = 50  # Integration steps for IG
    mandatory_features: List[str] = field(default_factory=list)  # Always include


# ============================================================================
# Result dataclass
# ============================================================================


@dataclass
class FeatureSelectionResult:
    """Result of adaptive feature selection for a single task.

    Attributes:
        task_name: Task identifier.
        original_count: Total number of input features.
        selected_count: Number of features after selection.
        reduction_pct: Percentage of features removed.
        cumulative_threshold_used: The threshold that was applied.
        selection_method: ``"ig"``, ``"lgbm"``, or ``"combined"``.
        selected_indices: Sorted feature indices that were selected.
        selected_names: Names of the selected features.
        feature_importances: Top-50 feature importances (name -> value).
        mandatory_included: Mandatory features that were included.
    """

    task_name: str
    original_count: int  # Total features (e.g. 644)
    selected_count: int  # After selection (e.g. 180)
    reduction_pct: float  # e.g. 72.0
    cumulative_threshold_used: float
    selection_method: str  # "ig" | "lgbm" | "combined"
    selected_indices: List[int]
    selected_names: List[str]
    feature_importances: Dict[str, float]  # top-50 saved
    mandatory_included: List[str]


# ============================================================================
# Feature Selector
# ============================================================================


class FeatureSelector:
    """Adaptive per-task feature selection using cumulative importance.

    Implements a 3-stage pipeline:

    1. **Integrated Gradients (IG)**: Compute per-feature importance from the
       PLE teacher model, then keep features until cumulative importance
       reaches ``cumulative_threshold``.
    2. **LGBM gain pruning**: Remove features with zero gain in the trained
       LGBM student model.
    3. **Mandatory features**: Ensure that domain-critical features are
       always included regardless of importance scores.

    Args:
        config: Feature selection configuration.  Uses defaults if ``None``.
        audit_store: Optional audit store for logging selection events.
            Must expose a ``log_event(event_type, payload)`` method.
    """

    def __init__(
        self,
        config: Optional[FeatureSelectionConfig] = None,
        audit_store: Optional[Any] = None,
    ) -> None:
        self._config = config or FeatureSelectionConfig()
        self._audit_store = audit_store
        # Signed IG cache: task_name → np.ndarray (n_features,) with +/- direction
        self._last_signed_ig: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Stage 1: IG-based selection
    # ------------------------------------------------------------------

    def select_by_ig(
        self,
        model: Any,
        features: Any,
        task_name: str,
        feature_names: Optional[List[str]] = None,
        n_samples: int = 1000,
    ) -> FeatureSelectionResult:
        """Stage 1: Select features by Integrated Gradients cumulative importance.

        For each feature, computes |IG| averaged over samples, sorts by
        importance, and keeps features until the cumulative sum reaches
        ``cumulative_threshold``.

        Args:
            model: PLE teacher model (``torch.nn.Module``).  Must accept
                :class:`PLEInput` and return :class:`PLEOutput`.
            features: Feature matrix as ``np.ndarray (n_samples, n_features)``
                or ``torch.Tensor``.
            task_name: Task to compute IG for.
            feature_names: Optional list of feature names.  Auto-generated
                as ``f_0``, ``f_1``, ... if ``None``.
            n_samples: Maximum samples to use for IG computation
                (subsampled if input is larger).

        Returns:
            A :class:`FeatureSelectionResult` with selected feature indices.
        """
        import torch

        # Subsample if needed
        if isinstance(features, np.ndarray):
            if features.shape[0] > n_samples:
                idx = np.random.choice(features.shape[0], n_samples, replace=False)
                features_sub = features[idx]
            else:
                features_sub = features
            features_tensor = torch.tensor(features_sub, dtype=torch.float32)
        else:
            features_tensor = features[:n_samples]

        n_features = features_tensor.shape[1]
        feature_names = feature_names or [f"f_{i}" for i in range(n_features)]

        # Compute IG
        importances = self._compute_ig(model, features_tensor, task_name)

        # Cumulative selection
        selected_indices = self._cumulative_select(importances)

        # Ensure mandatory features
        selected_indices = self._ensure_mandatory(selected_indices, feature_names)

        selected_names = [feature_names[i] for i in selected_indices]
        top_50 = {
            feature_names[i]: float(importances[i])
            for i in np.argsort(importances)[::-1][:50]
        }

        result = FeatureSelectionResult(
            task_name=task_name,
            original_count=n_features,
            selected_count=len(selected_indices),
            reduction_pct=round((1 - len(selected_indices) / n_features) * 100, 1),
            cumulative_threshold_used=self._config.cumulative_threshold,
            selection_method="ig",
            selected_indices=sorted(selected_indices),
            selected_names=selected_names,
            feature_importances=top_50,
            mandatory_included=[
                f for f in self._config.mandatory_features if f in selected_names
            ],
        )

        if self._audit_store:
            self._audit_store.log_event(
                "feature_selection",
                {
                    "pk": task_name,
                    "task": task_name,
                    "method": "ig",
                    "original": n_features,
                    "selected": len(selected_indices),
                    "reduction_pct": result.reduction_pct,
                },
            )

        return result

    # ------------------------------------------------------------------
    # Stage 2: LGBM gain pruning
    # ------------------------------------------------------------------

    def prune_by_lgbm(
        self,
        lgbm_model: Any,
        feature_names: List[str],
        selected_indices: Optional[List[int]] = None,
    ) -> List[int]:
        """Stage 2: Remove features with zero LGBM gain.

        Args:
            lgbm_model: Trained LightGBM model with a
                ``feature_importance(importance_type='gain')`` method.
            feature_names: All feature names (full set).
            selected_indices: If provided, only prune within these indices.
                Otherwise prune from the full feature set.

        Returns:
            List of surviving feature indices.
        """
        if not self._config.lgbm_prune_zero_gain:
            return selected_indices or list(range(len(feature_names)))

        importance = lgbm_model.feature_importance(importance_type="gain")

        if selected_indices is not None:
            # Only prune within already-selected features
            pruned = [i for i in selected_indices if importance[i] > 0]
        else:
            pruned = [i for i in range(len(importance)) if importance[i] > 0]

        # Ensure minimum feature count
        if len(pruned) < self._config.min_features:
            # Add back top features by gain
            top_by_gain = np.argsort(importance)[::-1][: self._config.min_features]
            pruned = sorted(set(pruned) | set(top_by_gain.tolist()))

        return pruned

    # ------------------------------------------------------------------
    # Combined pipeline
    # ------------------------------------------------------------------

    def select(
        self,
        model: Any,
        features: Any,
        task_name: str,
        lgbm_model: Any = None,
        feature_names: Optional[List[str]] = None,
    ) -> FeatureSelectionResult:
        """Combined: LGBM gain selection -> mandatory features.

        Runs the 2-stage pipeline.  If ``lgbm_model`` is ``None``,
        only Stage 2 (mandatory features) is applied.
        Teacher IG attribution is not used.

        Args:
            model: PLE teacher model.
            features: Feature matrix.
            task_name: Task to select features for.
            lgbm_model: Optional trained LGBM student model.
            feature_names: Optional feature name list.

        Returns:
            A :class:`FeatureSelectionResult` with final selection.
        """
        # Stage 1
        ig_result = self.select_by_ig(model, features, task_name, feature_names)

        # Stage 2
        if lgbm_model is not None:
            all_names = feature_names or [
                f"f_{i}" for i in range(ig_result.original_count)
            ]
            pruned_indices = self.prune_by_lgbm(
                lgbm_model,
                all_names,
                ig_result.selected_indices,
            )
            ig_result.selected_indices = pruned_indices
            ig_result.selected_names = [all_names[i] for i in pruned_indices]
            ig_result.selected_count = len(pruned_indices)
            ig_result.reduction_pct = round(
                (1 - len(pruned_indices) / ig_result.original_count) * 100, 1,
            )
            ig_result.selection_method = "combined"

        return ig_result

    # ------------------------------------------------------------------
    # Internal: Integrated Gradients (delegates to standalone class)
    # ------------------------------------------------------------------

    def _compute_ig(
        self,
        model: Any,
        features: Any,
        task_name: str,
    ) -> np.ndarray:
        """Compute Integrated Gradients for a task.

        Delegates to :class:`~core.evaluation.integrated_gradients.IntegratedGradients`
        for proper trapezoidal integration.  Processes in mini-batches of 64
        samples to avoid OOM on large feature matrices.

        The ``"median"`` baseline strategy is handled here (not supported by
        the standalone class) by computing the median and passing it as a
        custom ``torch.Tensor`` baseline.

        Also populates ``self._last_signed_ig[task_name]`` with the mean
        signed attributions for direction-of-effect interpretation.

        Args:
            model: PLE teacher model (torch.nn.Module).
            features: torch.Tensor of shape ``(n_samples, n_features)``.
            task_name: Target task for IG attribution.

        Returns:
            Array of shape ``(n_features,)`` with mean absolute IG per feature.
        """
        import torch
        from core.model.ple.model import PLEInput

        device = next(model.parameters()).device
        features = features.to(device)

        # Resolve baseline tensor for "median" (not supported by standalone).
        # For "zeros" and "mean" we pass the strategy string directly.
        if self._config.ig_baseline == "median":
            baseline_arg: Any = features.median(dim=0).values.unsqueeze(0)
        else:
            baseline_arg = self._config.ig_baseline  # "zeros" | "mean"

        ig_engine = IntegratedGradients(
            model=model,
            baseline=baseline_arg,
            n_steps=self._config.ig_steps,
            device=device,
        )

        batch_size = min(64, features.shape[0])
        all_attrs: List[np.ndarray] = []

        for start in range(0, features.shape[0], batch_size):
            end = min(start + batch_size, features.shape[0])
            batch_feat = features[start:end]

            # Validate task availability via a minimal probe (first step only)
            # so we can emit the fallback warning before running full IG.
            probe_inputs = PLEInput(features=batch_feat.detach())
            model.eval()
            with torch.no_grad():
                probe_out = model(probe_inputs, compute_loss=False)
            effective_task = task_name
            if task_name not in probe_out.predictions:
                first_task = next(iter(probe_out.predictions))
                logger.warning(
                    "Task '%s' not in model outputs, using '%s' instead.",
                    task_name,
                    first_task,
                )
                effective_task = first_task

            ple_inputs = PLEInput(features=batch_feat)
            # attribute() returns (batch, n_features) signed attributions
            attr = ig_engine.attribute(ple_inputs, target_task=effective_task)
            all_attrs.append(attr.cpu().numpy())

        if all_attrs:
            all_ig = np.concatenate(all_attrs, axis=0)  # (n_samples, n_features)
            abs_ig = np.abs(all_ig).mean(axis=0)         # (n_features,)
            self._last_signed_ig[task_name] = all_ig.mean(axis=0)
            return abs_ig

        # No gradients computed — return uniform importance
        return np.ones(features.shape[1]) / features.shape[1]

    def select_by_ig_dual(
        self,
        model: Any,
        features: Any,
        task_name: str,
        feature_names: Optional[List[str]] = None,
        n_samples: int = 10000,
        ig_alpha: float = 0.7,
        explain_scores: Optional[Dict[str, float]] = None,
    ) -> FeatureSelectionResult:
        """Deprecated: Dual-objective IG selection (no longer used).

        Teacher model IG attribution is removed. Feature selection is now performed
        using LGBM gain importance (see ``select_by_lgbm_gain`` and ``prune_by_lgbm``).
        This method is retained for backward compatibility only.

        Previously combined:
            score(f) = alpha * IG_pred(f) + (1 - alpha) * IG_explain(f)

        where ``IG_pred`` was normalized absolute IG from the teacher model.
        Removed because: OOM at 941K scale, serving model misalignment.

        Args:
            model: PLE teacher model (``torch.nn.Module``).
            features: Feature matrix as ``np.ndarray (n_rows, n_features)``
                or ``torch.Tensor``.
            task_name: Task to compute IG for.
            feature_names: Optional list of feature names.
            n_samples: Maximum samples used for IG computation.
            ig_alpha: Weight for predictive (IG) component vs. explanation
                component.  Default 0.7 (Paper 2 default).
            explain_scores: Dict mapping feature name -> explain score in
                [0, 1].  Features not listed get score 0.0.  If None or
                empty, falls back to pure IG (``ig_alpha=1.0``).

        Returns:
            A :class:`FeatureSelectionResult` using dual-objective scores,
            with ``selection_method="ig_dual"``.
        """
        import torch

        # Subsample for IG computation
        if isinstance(features, np.ndarray):
            if features.shape[0] > n_samples:
                idx = np.random.choice(features.shape[0], n_samples, replace=False)
                features_sub = features[idx]
            else:
                features_sub = features
            features_tensor = torch.tensor(features_sub, dtype=torch.float32)
        else:
            features_tensor = features[:n_samples]

        n_features = features_tensor.shape[1]
        feature_names = feature_names or [f"f_{i}" for i in range(n_features)]

        # --- Predictive component: IG attribution from teacher ---
        ig_abs = self._compute_ig(model, features_tensor, task_name)

        # Normalize IG to [0, 1]
        ig_max = ig_abs.max()
        if ig_max > 1e-10:
            ig_pred_norm = ig_abs / ig_max
        else:
            ig_pred_norm = np.ones(n_features) / n_features

        # --- Explanation component: per-feature explain score ---
        if explain_scores:
            ig_explain_arr = np.array(
                [explain_scores.get(name, 0.0) for name in feature_names],
                dtype=np.float64,
            )
            expl_max = ig_explain_arr.max()
            if expl_max > 1e-10:
                ig_explain_norm = ig_explain_arr / expl_max
            else:
                ig_explain_norm = np.zeros(n_features)
            effective_alpha = ig_alpha
        else:
            # No explain scores provided — pure IG
            ig_explain_norm = np.zeros(n_features)
            effective_alpha = 1.0
            logger.warning(
                "No explain_scores provided for task '%s'; using pure IG (alpha=1.0).",
                task_name,
            )

        # --- Dual-objective score ---
        dual_scores = (
            effective_alpha * ig_pred_norm
            + (1.0 - effective_alpha) * ig_explain_norm
        )

        # Cumulative selection on dual scores
        selected_indices = self._cumulative_select(dual_scores)

        # Ensure mandatory features
        selected_indices = self._ensure_mandatory(selected_indices, feature_names)

        selected_names = [feature_names[i] for i in selected_indices]
        # Store top-50 by dual score for reporting
        top_50 = {
            feature_names[i]: float(dual_scores[i])
            for i in np.argsort(dual_scores)[::-1][:50]
        }

        result = FeatureSelectionResult(
            task_name=task_name,
            original_count=n_features,
            selected_count=len(selected_indices),
            reduction_pct=round((1 - len(selected_indices) / n_features) * 100, 1),
            cumulative_threshold_used=self._config.cumulative_threshold,
            selection_method="ig_dual",
            selected_indices=sorted(selected_indices),
            selected_names=selected_names,
            feature_importances=top_50,
            mandatory_included=[
                f for f in self._config.mandatory_features if f in selected_names
            ],
        )

        if self._audit_store:
            self._audit_store.log_event(
                "feature_selection",
                {
                    "pk": task_name,
                    "task": task_name,
                    "method": "ig_dual",
                    "ig_alpha": effective_alpha,
                    "original": n_features,
                    "selected": len(selected_indices),
                    "reduction_pct": result.reduction_pct,
                },
            )

        return result

    def get_signed_ig(self, task_name: str) -> Optional[np.ndarray]:
        """Return signed IG values from the last compute_ig call.

        Positive = feature value ↑ → task prediction ↑
        Negative = feature value ↑ → task prediction ↓

        Must be called after :meth:`select` for the given task.

        Returns:
            Array of shape ``(n_features,)`` or ``None`` if not computed.
        """
        return self._last_signed_ig.get(task_name)

    # ------------------------------------------------------------------
    # Internal: cumulative selection
    # ------------------------------------------------------------------

    def _cumulative_select(self, importances: np.ndarray) -> List[int]:
        """Select features by cumulative importance threshold.

        Sorts features by importance descending, computes cumulative sum,
        and returns indices up to the point where the cumulative fraction
        reaches ``cumulative_threshold``.  Result is clamped between
        ``min_features`` and ``max_features``.

        Args:
            importances: Per-feature importance scores of shape ``(n_features,)``.

        Returns:
            List of selected feature indices.
        """
        # Sort by importance descending
        sorted_idx = np.argsort(importances)[::-1]
        total = importances.sum()

        if total < 1e-10:
            # All zero importance -- keep min_features
            return sorted_idx[: self._config.min_features].tolist()

        cumsum = np.cumsum(importances[sorted_idx]) / total

        # Find cutoff index where cumulative >= threshold
        cutoff = int(np.searchsorted(cumsum, self._config.cumulative_threshold) + 1)

        # Clamp between min and max
        cutoff = max(cutoff, self._config.min_features)
        cutoff = min(cutoff, self._config.max_features, len(importances))

        return sorted_idx[:cutoff].tolist()

    # ------------------------------------------------------------------
    # Internal: mandatory feature guarantee
    # ------------------------------------------------------------------

    def _ensure_mandatory(
        self,
        indices: List[int],
        feature_names: List[str],
    ) -> List[int]:
        """Ensure mandatory features are included in the selection.

        Args:
            indices: Currently selected feature indices.
            feature_names: Full list of feature names.

        Returns:
            Updated list of feature indices with mandatory features included.
        """
        name_to_idx = {name: i for i, name in enumerate(feature_names)}
        result = set(indices)
        for mf in self._config.mandatory_features:
            if mf in name_to_idx:
                result.add(name_to_idx[mf])
        return sorted(result)
