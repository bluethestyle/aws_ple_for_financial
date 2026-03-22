"""
CGC Gate Weight Analyzer.

Extracts and analyzes per-task CGC attention weights from a trained
PLEModel, showing which shared experts each task relies on most.

Usage::

    from core.evaluation.gate_analyzer import GateAnalyzer

    analyzer = GateAnalyzer(model)
    result = analyzer.analyze(val_loader, max_batches=20)
    print(result.dominant_expert_per_task)
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class GateAnalysisResult:
    """Result container for CGC gate weight analysis.

    Attributes:
        task_expert_weights: Per-task mean attention over experts.
            ``{task_name: {expert_name: mean_weight}}``.
        expert_utilization: Average attention weight per expert across
            all tasks. ``{expert_name: avg_weight}``.
        dominant_expert_per_task: The most-attended expert for each task.
            ``{task_name: expert_name}``.
        entropy_per_task: Shannon entropy of the gate distribution per
            task (higher = more uniform / diverse expert usage).
    """

    task_expert_weights: Dict[str, Dict[str, float]]
    expert_utilization: Dict[str, float]
    dominant_expert_per_task: Dict[str, str]
    entropy_per_task: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "task_expert_weights": self.task_expert_weights,
            "expert_utilization": self.expert_utilization,
            "dominant_expert_per_task": self.dominant_expert_per_task,
            "entropy_per_task": self.entropy_per_task,
        }


class GateAnalyzer:
    """Extract and analyze CGC gate attention weights per task.

    Shows which experts each task relies on most.

    Args:
        model: A trained PLEModel instance.
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        self._result: Optional[GateAnalysisResult] = None

    def analyze(
        self,
        dataloader: Any,
        max_batches: int = 20,
    ) -> GateAnalysisResult:
        """Collect gate weights across batches and compute statistics.

        Runs inference on up to ``max_batches`` batches, collects
        ``PLEOutput.cgc_attention_weights``, and averages across all
        samples to produce per-task expert attention distributions.

        Args:
            dataloader: Validation DataLoader yielding dicts or PLEInput.
            max_batches: Maximum number of batches to process.

        Returns:
            :class:`GateAnalysisResult` with per-task expert attention
            statistics.
        """
        from core.model.ple.model import PLEInput

        self.model.eval()
        device = next(self.model.parameters()).device

        # Accumulators: {task_name: list of (n_experts,) tensors}
        weight_accum: Dict[str, List[torch.Tensor]] = defaultdict(list)
        num_samples = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break

                # Convert batch dict to PLEInput if needed
                if isinstance(batch, dict):
                    ple_input = PLEInput(
                        features=batch["features"].to(device),
                        targets=None,
                    )
                    # Forward optional fields
                    for fld in [
                        "hyperbolic_features", "tda_features",
                        "collaborative_features", "event_sequences",
                        "session_sequences",
                    ]:
                        if fld in batch and batch[fld] is not None:
                            object.__setattr__(
                                ple_input, fld, batch[fld].to(device),
                            )
                elif hasattr(batch, "features"):
                    ple_input = batch.to(device) if hasattr(batch, "to") else batch
                else:
                    continue

                output = self.model(ple_input)
                cgc_weights = output.cgc_attention_weights

                if cgc_weights is None:
                    logger.warning(
                        "GateAnalyzer: PLEOutput.cgc_attention_weights is None. "
                        "Ensure CGC attention is enabled and the model is in eval mode."
                    )
                    break

                batch_size = next(iter(cgc_weights.values())).size(0)
                num_samples += batch_size

                for task_name, weights in cgc_weights.items():
                    # weights: (batch, n_experts) -- average over batch
                    weight_accum[task_name].append(weights.sum(dim=0).cpu())

        if num_samples == 0 or not weight_accum:
            logger.warning("GateAnalyzer: no samples collected.")
            return GateAnalysisResult(
                task_expert_weights={},
                expert_utilization={},
                dominant_expert_per_task={},
                entropy_per_task={},
            )

        # Resolve expert names from the model's CGC attention module
        expert_names = self._get_expert_names()

        # Compute mean weights per task
        task_expert_weights: Dict[str, Dict[str, float]] = {}
        entropy_per_task: Dict[str, float] = {}

        for task_name, weight_list in weight_accum.items():
            mean_w = torch.stack(weight_list).sum(dim=0) / num_samples
            n_experts = mean_w.size(0)

            names = expert_names[:n_experts] if len(expert_names) >= n_experts else [
                f"expert_{i}" for i in range(n_experts)
            ]

            task_expert_weights[task_name] = {
                name: round(mean_w[i].item(), 6) for i, name in enumerate(names)
            }

            # Shannon entropy: -sum(p * log(p))
            p = mean_w.clamp(min=1e-8)
            entropy = -(p * p.log()).sum().item()
            entropy_per_task[task_name] = round(entropy, 6)

        # Expert utilization: average weight across all tasks
        all_expert_names = set()
        for tw in task_expert_weights.values():
            all_expert_names.update(tw.keys())

        expert_utilization: Dict[str, float] = {}
        for ename in sorted(all_expert_names):
            vals = [
                tw.get(ename, 0.0)
                for tw in task_expert_weights.values()
            ]
            expert_utilization[ename] = round(
                sum(vals) / max(len(vals), 1), 6
            )

        # Dominant expert per task
        dominant_expert_per_task: Dict[str, str] = {}
        for task_name, tw in task_expert_weights.items():
            dominant_expert_per_task[task_name] = max(tw, key=tw.get)

        self._result = GateAnalysisResult(
            task_expert_weights=task_expert_weights,
            expert_utilization=expert_utilization,
            dominant_expert_per_task=dominant_expert_per_task,
            entropy_per_task=entropy_per_task,
        )

        logger.info(
            "GateAnalyzer: analyzed %d samples, %d tasks, %d experts",
            num_samples, len(task_expert_weights), len(expert_utilization),
        )
        for task_name, dominant in dominant_expert_per_task.items():
            logger.info(
                "  %s -> dominant expert: %s (entropy=%.4f)",
                task_name, dominant, entropy_per_task.get(task_name, 0.0),
            )

        return self._result

    def dominant_experts(self) -> Dict[str, str]:
        """Return the most-attended expert per task.

        Requires :meth:`analyze` to have been called first.
        """
        if self._result is None:
            raise RuntimeError("Call analyze() before dominant_experts().")
        return self._result.dominant_expert_per_task

    def _get_expert_names(self) -> List[str]:
        """Resolve expert names from the model's CGC attention module."""
        if hasattr(self.model, "cgc_attention") and self.model.cgc_attention is not None:
            return list(self.model.cgc_attention._expert_names)
        if hasattr(self.model, "expert_basket") and self.model.expert_basket is not None:
            return self.model.expert_basket.shared_expert_names
        return []
