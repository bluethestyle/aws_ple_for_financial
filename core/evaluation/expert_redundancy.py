"""
Expert Redundancy Analysis via Canonical Correlation Analysis (CCA).

Measures representation overlap between shared experts in a PLE model
to detect redundancy. High canonical correlation between two experts
indicates they encode similar information and one may be prunable.

Reference implementation:
    gotothemoon/workspace/code/src/evaluation/expert_redundancy.py

Uses SVD-based CCA computation (no sklearn dependency):
    1. Center X and Y (subtract mean)
    2. Compute covariance matrices with regularization
    3. Cxx^{-1/2} @ Cxy @ Cyy^{-1/2} via eigendecomposition
    4. SVD -> singular values = canonical correlations
    5. Mean of top-n components = redundancy score

Usage::

    analyzer = ExpertRedundancyAnalyzer(model)
    result = analyzer.analyze(dataloader, max_batches=20)
    print(result.redundancy_matrix)  # [n_experts, n_experts]
    print(result.summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ======================================================================
# Result container
# ======================================================================

@dataclass
class RedundancyResult:
    """CCA redundancy analysis result.

    Attributes:
        expert_names: Ordered list of expert names (matrix index order).
        redundancy_matrix: ``[n_experts, n_experts]`` symmetric matrix
            where entry ``(i, j)`` is the mean canonical correlation
            between experts ``i`` and ``j``. Diagonal is 1.0.
        pairwise_correlations: ``{"expert_i__expert_j": correlation}``
            mapping for each unique pair.
        classifications: ``{"expert_i__expert_j": "HIGH"/"MID"/"LOW"}``
            based on correlation thresholds (>0.7=HIGH, >0.4=MID).
        n_samples: Number of samples used in the analysis.
    """

    expert_names: List[str] = field(default_factory=list)
    redundancy_matrix: Optional[np.ndarray] = None
    pairwise_correlations: Dict[str, float] = field(default_factory=dict)
    classifications: Dict[str, str] = field(default_factory=dict)
    n_samples: int = 0

    def summary(self) -> str:
        """Human-readable summary of redundancy analysis."""
        lines = [
            f"Expert Redundancy Analysis (N={self.n_samples}, "
            f"{len(self.expert_names)} experts, "
            f"{len(self.pairwise_correlations)} pairs)",
            "",
        ]

        # Sort pairs by correlation descending
        sorted_pairs = sorted(
            self.pairwise_correlations.items(),
            key=lambda x: -x[1],
        )

        for pair_key, corr in sorted_pairs:
            level = self.classifications.get(pair_key, "???")
            names = pair_key.replace("__", " <-> ")
            lines.append(f"  [{level:4s}] {names}: mean_rho={corr:.3f}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-safe dict."""
        result: Dict[str, Any] = {
            "expert_names": self.expert_names,
            "n_samples": self.n_samples,
            "pairwise_correlations": self.pairwise_correlations,
            "classifications": self.classifications,
        }
        if self.redundancy_matrix is not None:
            result["redundancy_matrix"] = self.redundancy_matrix.tolist()
        return result


# ======================================================================
# Analyzer
# ======================================================================

class ExpertRedundancyAnalyzer:
    """Canonical Correlation Analysis between shared expert outputs.

    Registers forward hooks on the shared experts in a PLE model's
    first CGC layer to capture their outputs, then runs CCA on all
    unique expert pairs.

    Args:
        model: A PLEModel with ``extraction_layers`` containing
            ``CGCLayer`` instances with ``shared_experts``.
        n_components: Number of canonical components to compute
            (capped at min expert output dim).
        min_samples: Minimum number of samples required for analysis.
    """

    def __init__(
        self,
        model: nn.Module,
        n_components: int = 10,
        min_samples: int = 256,
    ) -> None:
        self.model = model
        self.n_components = n_components
        self.min_samples = min_samples
        self._hooks: List[Any] = []

    def analyze(
        self,
        dataloader: Any,
        max_batches: int = 20,
    ) -> Optional[RedundancyResult]:
        """Run CCA analysis on expert outputs.

        1. Register forward hooks on shared experts to capture outputs
        2. Run dataloader through model (eval mode, no grad)
        3. For each pair of experts, compute canonical correlations
        4. Build redundancy matrix

        Args:
            dataloader: Iterable yielding PLEInput-compatible batches.
            max_batches: Maximum batches to process.

        Returns:
            RedundancyResult, or None if insufficient data / experts.
        """
        self.model.eval()

        # Discover shared experts from the first extraction layer
        expert_names, expert_modules = self._get_shared_experts()
        if len(expert_modules) < 2:
            logger.warning(
                "CCA analysis requires at least 2 shared experts, found %d",
                len(expert_modules),
            )
            return None

        # Buffers to accumulate expert outputs
        expert_outputs: Dict[str, List[torch.Tensor]] = {
            name: [] for name in expert_names
        }

        # Register forward hooks
        self._register_hooks(expert_names, expert_modules, expert_outputs)

        try:
            # Forward pass through the model to collect expert outputs
            n_samples = 0
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    if batch_idx >= max_batches:
                        break

                    inputs = self._to_ple_input(batch)
                    if inputs is None:
                        continue

                    inputs = self._move_to_device(inputs)
                    self.model(inputs, compute_loss=False)
                    n_samples += inputs.features.shape[0]

        finally:
            # Always remove hooks
            self._remove_hooks()

        if n_samples < self.min_samples:
            logger.warning(
                "Insufficient samples for CCA: %d < %d",
                n_samples, self.min_samples,
            )
            return None

        # Concatenate accumulated outputs -> numpy
        expert_data: Dict[str, np.ndarray] = {}
        for name, tensors in expert_outputs.items():
            if not tensors:
                logger.warning("No outputs captured for expert '%s'", name)
                continue
            cat = torch.cat(tensors, dim=0).cpu().numpy()
            # Flatten to 2D if needed (e.g. sequence experts)
            if cat.ndim > 2:
                cat = cat.reshape(cat.shape[0], -1)
            expert_data[name] = cat

        if len(expert_data) < 2:
            logger.warning("Fewer than 2 experts produced outputs")
            return None

        # Run pairwise CCA
        sorted_names = sorted(expert_data.keys())
        n_experts = len(sorted_names)
        redundancy_matrix = np.eye(n_experts)
        pairwise_correlations: Dict[str, float] = {}
        classifications: Dict[str, str] = {}

        for i, j in combinations(range(n_experts), 2):
            name_i, name_j = sorted_names[i], sorted_names[j]
            X, Y = expert_data[name_i], expert_data[name_j]

            n_comp = min(self.n_components, X.shape[1], Y.shape[1])
            corrs = self._compute_cca(X, Y, n_comp)

            pair_key = f"{name_i}__{name_j}"
            if corrs is not None:
                mean_corr = float(np.mean(corrs))
                redundancy_matrix[i, j] = mean_corr
                redundancy_matrix[j, i] = mean_corr
                pairwise_correlations[pair_key] = round(mean_corr, 4)

                if mean_corr > 0.7:
                    level = "HIGH"
                elif mean_corr > 0.4:
                    level = "MID"
                else:
                    level = "LOW"
                classifications[pair_key] = level
            else:
                pairwise_correlations[pair_key] = 0.0
                classifications[pair_key] = "FAIL"

        result = RedundancyResult(
            expert_names=sorted_names,
            redundancy_matrix=redundancy_matrix,
            pairwise_correlations=pairwise_correlations,
            classifications=classifications,
            n_samples=n_samples,
        )

        logger.info(result.summary())
        return result

    # ------------------------------------------------------------------
    # CCA computation (SVD-based, no sklearn)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_cca(
        X: np.ndarray,
        Y: np.ndarray,
        n_components: int,
    ) -> Optional[List[float]]:
        """SVD-based Canonical Correlation Analysis.

        Computes canonical correlations without sklearn dependency.

        Steps:
            1. Center X, Y
            2. Regularized covariance: Cxx, Cyy, Cxy
            3. Cxx^{-1/2} via eigendecomposition
            4. Cyy^{-1/2} via eigendecomposition
            5. SVD(Cxx^{-1/2} @ Cxy @ Cyy^{-1/2}) -> singular values
            6. Top-n singular values = canonical correlations

        Args:
            X: ``(n_samples, dim_x)`` array.
            Y: ``(n_samples, dim_y)`` array.
            n_components: Number of canonical components.

        Returns:
            List of canonical correlations (clipped to [0, 1]), or None
            on numerical failure.
        """
        n = X.shape[0]

        # Center
        X = X - X.mean(axis=0, keepdims=True)
        Y = Y - Y.mean(axis=0, keepdims=True)

        # Regularized covariance matrices
        reg = 1e-4
        Cxx = (X.T @ X) / n + reg * np.eye(X.shape[1])
        Cyy = (Y.T @ Y) / n + reg * np.eye(Y.shape[1])
        Cxy = (X.T @ Y) / n

        try:
            # Cxx^{-1/2} via eigendecomposition
            eigvals_x, eigvecs_x = np.linalg.eigh(Cxx)
            eigvals_x = np.maximum(eigvals_x, 1e-8)
            Cxx_inv_sqrt = (
                eigvecs_x
                @ np.diag(1.0 / np.sqrt(eigvals_x))
                @ eigvecs_x.T
            )

            # Cyy^{-1/2} via eigendecomposition
            eigvals_y, eigvecs_y = np.linalg.eigh(Cyy)
            eigvals_y = np.maximum(eigvals_y, 1e-8)
            Cyy_inv_sqrt = (
                eigvecs_y
                @ np.diag(1.0 / np.sqrt(eigvals_y))
                @ eigvecs_y.T
            )

            # Canonical correlations = SVD singular values
            M = Cxx_inv_sqrt @ Cxy @ Cyy_inv_sqrt
            _, s, _ = np.linalg.svd(M, full_matrices=False)

            corrs = np.clip(s[:n_components], 0.0, 1.0).tolist()
            return corrs

        except np.linalg.LinAlgError as e:
            logger.debug("CCA SVD failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def _get_shared_experts(self) -> Tuple[List[str], List[nn.Module]]:
        """Extract shared expert names and modules from the model.

        Looks for:
            model.extraction_layers[0].shared_experts (CGCLayer)
            model.extraction_layers[0].shared_expert_names
        """
        names: List[str] = []
        modules: List[nn.Module] = []

        if not hasattr(self.model, "extraction_layers"):
            logger.warning("Model has no extraction_layers attribute")
            return names, modules

        layers = self.model.extraction_layers
        if len(layers) == 0:
            return names, modules

        first_layer = layers[0]
        shared = getattr(first_layer, "shared_experts", None)
        if shared is None:
            return names, modules

        expert_names = getattr(
            first_layer, "shared_expert_names",
            [f"shared_{i}" for i in range(len(shared))],
        )

        for i, expert in enumerate(shared):
            name = expert_names[i] if i < len(expert_names) else f"shared_{i}"
            names.append(name)
            modules.append(expert)

        return names, modules

    def _register_hooks(
        self,
        expert_names: List[str],
        expert_modules: List[nn.Module],
        output_buffers: Dict[str, List[torch.Tensor]],
    ) -> None:
        """Register forward hooks to capture expert outputs."""
        for name, module in zip(expert_names, expert_modules):

            def _make_hook(expert_name: str) -> Callable:
                def hook_fn(mod: nn.Module, inp: Any, out: torch.Tensor) -> None:
                    # Detach and store on CPU to save GPU memory
                    if isinstance(out, torch.Tensor):
                        output_buffers[expert_name].append(out.detach().cpu())
                    elif isinstance(out, (tuple, list)):
                        output_buffers[expert_name].append(out[0].detach().cpu())
                return hook_fn

            handle = module.register_forward_hook(_make_hook(name))
            self._hooks.append(handle)

    def _remove_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Input conversion helpers
    # ------------------------------------------------------------------

    def _to_ple_input(self, batch: Any) -> Any:
        """Convert a dataloader batch to PLEInput if needed."""
        if hasattr(batch, "features") and hasattr(batch, "targets"):
            return batch

        if isinstance(batch, dict) and "features" in batch:
            from core.model.ple.model import PLEInput
            kwargs = {"features": batch["features"]}

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

        return None

    def _move_to_device(self, inputs: Any) -> Any:
        """Move PLEInput to the model device."""
        device = self.device if hasattr(self, "device") else torch.device("cpu")
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            pass
        if hasattr(inputs, "to"):
            return inputs.to(device)
        return inputs

    @property
    def device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")
