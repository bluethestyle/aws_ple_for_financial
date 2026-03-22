"""
Integrated Gradients feature attribution for PLE models.

Computes IG(x) = (x - x') * integral(grad(F(x' + alpha*(x-x')), x), alpha=0..1)

This is a standalone, reusable module for computing feature attributions
using the Integrated Gradients method (Sundararajan et al., 2017).

Reference implementation:
    gotothemoon/workspace/code/src/distillation/feature_selector.py

Usage::

    ig = IntegratedGradients(model, device=device)
    attributions = ig.attribute(inputs, target_task="churn_signal")
    importance = ig.feature_importance(dataloader, target_task="churn_signal")
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class IntegratedGradients:
    """Integrated Gradients feature attribution for PLE models.

    Computes per-feature attribution scores by integrating gradients
    along the straight-line path from a baseline to the input.

    Args:
        model: A PLEModel (or any nn.Module that accepts PLEInput and
            returns PLEOutput with ``predictions[task_name]``).
        baseline: Baseline strategy -- ``"zeros"`` (default), ``"mean"``
            (computed from first batch), or a custom ``torch.Tensor``.
        n_steps: Number of interpolation steps for the integral
            approximation. Higher = more accurate but slower.
        device: Device to run on. If None, inferred from model parameters.
    """

    def __init__(
        self,
        model: nn.Module,
        baseline: Union[str, torch.Tensor] = "zeros",
        n_steps: int = 50,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.baseline_strategy = baseline
        self.n_steps = n_steps

        if device is not None:
            self.device = device
        else:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")

        self._mean_baseline: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def attribute(
        self,
        inputs: Any,
        target_task: str,
    ) -> torch.Tensor:
        """Compute per-sample attributions for a single batch.

        IG operates on ``inputs.features`` (the flat feature tensor).
        The model is put into eval mode and gradients are computed via
        ``torch.autograd.grad`` with ``create_graph=False``.

        Interpolation steps are batched together for GPU efficiency
        rather than looped one-by-one.

        Args:
            inputs: A ``PLEInput`` instance.
            target_task: Name of the task to compute attributions for.

        Returns:
            ``(batch_size, input_dim)`` tensor of per-feature attributions.
        """
        self.model.eval()

        features = inputs.features  # (batch, input_dim)
        batch_size, input_dim = features.shape

        # 1. Create baseline
        baseline = self._get_baseline(features)

        # 2. Build interpolation alphas: (n_steps+1,)
        alphas = torch.linspace(0, 1, self.n_steps + 1, device=self.device)

        # 3. Batch all interpolated inputs together for efficiency
        # Shape: (n_steps+1, batch_size, input_dim) -> ((n_steps+1)*batch_size, input_dim)
        diff = features - baseline  # (batch, input_dim)

        # Collect gradients at each interpolation point
        all_gradients = torch.zeros(
            self.n_steps + 1, batch_size, input_dim,
            device=self.device, dtype=features.dtype,
        )

        # Process in chunks to avoid OOM on large batches
        # Each step: interpolated = baseline + alpha * diff
        for step_idx in range(self.n_steps + 1):
            alpha = alphas[step_idx]
            interpolated = baseline + alpha * diff
            interpolated = interpolated.detach().requires_grad_(True)

            # Build a modified PLEInput with the interpolated features
            modified_inputs = self._replace_features(inputs, interpolated)

            # Forward pass
            output = self.model(modified_inputs, compute_loss=False)
            task_pred = output.predictions[target_task]

            # Reduce to scalar per sample, then sum for backward
            if task_pred.dim() > 1:
                task_pred = task_pred.squeeze(-1)
            scalar_output = task_pred.sum()

            # Compute gradients w.r.t. interpolated features
            grads = torch.autograd.grad(
                outputs=scalar_output,
                inputs=interpolated,
                create_graph=False,
            )[0]

            all_gradients[step_idx] = grads

        # 4. Trapezoidal integration: (1/n_steps) * sum of
        #    (g[0]/2 + g[1] + g[2] + ... + g[n-1] + g[n]/2)
        trapez = (
            all_gradients[0] / 2.0
            + all_gradients[1:-1].sum(dim=0)
            + all_gradients[-1] / 2.0
        ) / self.n_steps

        # 5. Multiply by (input - baseline)
        attributions = diff * trapez

        return attributions.detach()

    def feature_importance(
        self,
        dataloader: Any,
        target_task: str,
        max_batches: int = 50,
    ) -> Dict[int, float]:
        """Compute mean absolute attribution per feature across the dataset.

        Iterates over the dataloader (up to ``max_batches`` batches),
        computes IG attributions, and averages the absolute values per
        feature dimension.

        Args:
            dataloader: An iterable yielding PLEInput-compatible batches.
            target_task: Task name.
            max_batches: Maximum number of batches to process.

        Returns:
            ``{feature_index: mean_abs_attribution}`` sorted descending
            by importance.
        """
        total_abs_attr = None
        n_samples = 0

        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            inputs = self._to_ple_input(batch)
            if inputs is None:
                continue

            inputs = self._move_to_device(inputs)
            attr = self.attribute(inputs, target_task)  # (batch, input_dim)
            abs_attr = attr.abs().sum(dim=0)  # (input_dim,)

            if total_abs_attr is None:
                total_abs_attr = abs_attr
            else:
                total_abs_attr = total_abs_attr + abs_attr

            n_samples += attr.shape[0]

        if total_abs_attr is None or n_samples == 0:
            logger.warning("No samples processed for IG feature importance")
            return {}

        mean_abs_attr = (total_abs_attr / n_samples).cpu().numpy()

        # Build sorted dict
        importance = {
            int(idx): float(val)
            for idx, val in sorted(
                enumerate(mean_abs_attr),
                key=lambda x: -x[1],
            )
        }

        logger.info(
            "[IG] Feature importance computed: %d features, %d samples, "
            "top-5 indices: %s",
            len(importance), n_samples,
            list(importance.keys())[:5],
        )

        return importance

    def top_k_features(
        self,
        dataloader: Any,
        target_task: str,
        k: int = 20,
        max_batches: int = 50,
    ) -> List[int]:
        """Return top-K most important feature indices for a task.

        Args:
            dataloader: An iterable yielding PLEInput-compatible batches.
            target_task: Task name.
            k: Number of top features to return.
            max_batches: Maximum batches to process.

        Returns:
            List of top-K feature indices sorted by importance descending.
        """
        importance = self.feature_importance(
            dataloader, target_task, max_batches=max_batches,
        )
        return list(importance.keys())[:k]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_baseline(self, features: torch.Tensor) -> torch.Tensor:
        """Create baseline tensor matching features shape."""
        if isinstance(self.baseline_strategy, torch.Tensor):
            baseline = self.baseline_strategy
            if baseline.shape != features.shape:
                baseline = baseline.expand_as(features)
            return baseline.to(self.device)

        if self.baseline_strategy == "zeros":
            return torch.zeros_like(features)

        if self.baseline_strategy == "mean":
            if self._mean_baseline is None:
                self._mean_baseline = features.mean(dim=0, keepdim=True)
            return self._mean_baseline.expand_as(features).to(self.device)

        # Default: zeros
        return torch.zeros_like(features)

    @staticmethod
    def _replace_features(inputs: Any, new_features: torch.Tensor) -> Any:
        """Create a shallow copy of PLEInput with replaced features.

        This avoids mutating the original input object. We use
        ``dataclasses.replace`` if available, otherwise construct
        a new PLEInput manually.
        """
        from dataclasses import fields, replace

        try:
            return replace(inputs, features=new_features)
        except Exception:
            # Fallback: manual copy for non-dataclass inputs
            import copy
            new_inputs = copy.copy(inputs)
            new_inputs.features = new_features
            return new_inputs

    def _to_ple_input(self, batch: Any) -> Any:
        """Convert a dataloader batch to PLEInput if needed.

        Handles both dict-style batches (from PLEDataset) and direct
        PLEInput objects.
        """
        # Already a PLEInput
        if hasattr(batch, "features") and hasattr(batch, "targets"):
            return batch

        # Dict-style batch from PLEDataset
        if isinstance(batch, dict) and "features" in batch:
            from core.model.ple.model import PLEInput
            kwargs = {"features": batch["features"]}

            # Map common dict keys to PLEInput fields
            field_map = {
                "cluster_ids": "cluster_ids",
                "cluster_probs": "cluster_probs",
                "targets": "targets",
                "hyperbolic_features": "hyperbolic_features",
                "tda_features": "tda_features",
                "tda_diagrams": "tda_diagrams",
                "tda_diagram_mask": "tda_diagram_mask",
                "collaborative_features": "collaborative_features",
                "hmm_journey": "hmm_journey",
                "hmm_lifecycle": "hmm_lifecycle",
                "hmm_behavior": "hmm_behavior",
                "event_sequences": "event_sequences",
                "session_sequences": "session_sequences",
                "event_time_delta": "event_time_delta",
                "session_time_delta": "session_time_delta",
                "sequence_lengths": "sequence_lengths",
                "multidisciplinary_features": "multidisciplinary_features",
                "coldstart_features": "coldstart_features",
                "anonymous_features": "anonymous_features",
                "edge_index": "edge_index",
                "edge_weight": "edge_weight",
                "sample_weights": "sample_weights",
                "feature_group_ranges": "feature_group_ranges",
                "expert_routing": "expert_routing",
            }
            for dict_key, field_name in field_map.items():
                if dict_key in batch:
                    kwargs[field_name] = batch[dict_key]

            return PLEInput(**kwargs)

        logger.debug("Cannot convert batch to PLEInput: type=%s", type(batch))
        return None

    def _move_to_device(self, inputs: Any) -> Any:
        """Move PLEInput to the configured device."""
        if hasattr(inputs, "to"):
            return inputs.to(self.device)
        return inputs
