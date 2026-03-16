"""
Feature Router for Expert-Specific Input Routing.

Routes feature subsets to specific experts based on configuration. Given a
concatenated feature tensor ``(batch, total_dim)`` and group ranges, slices
the appropriate features for each expert.

Backward compatibility:
  - If no routing config is provided, all experts receive the full feature
    tensor (identical to the pre-routing behavior).
  - If ``feature_group_ranges`` is ``None`` at forward time, routing is
    bypassed entirely.

Example usage::

    router = FeatureRouter(
        expert_names=["perslay", "hgcn", "mamba", "mlp_shared"],
        routing_config=[
            ExpertInputConfig(expert_name="perslay", input_groups=["tda_topology"]),
            ExpertInputConfig(expert_name="hgcn",    input_groups=["graph_features"]),
            ExpertInputConfig(expert_name="mamba",   input_groups=["temporal_seq"]),
            # mlp_shared has no config entry -> receives ALL features
        ],
        group_ranges={
            "base_profile":   (0, 64),
            "tda_topology":   (64, 128),
            "graph_features": (128, 200),
            "temporal_seq":   (200, 280),
        },
    )
    x_for_perslay = router.route(x, "perslay")   # (batch, 64)
    x_for_hgcn    = router.route(x, "hgcn")      # (batch, 72)
    x_for_mamba   = router.route(x, "mamba")      # (batch, 80)
    x_for_mlp     = router.route(x, "mlp_shared") # (batch, 280) -- full
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .config import ExpertInputConfig

logger = logging.getLogger(__name__)


class FeatureRouter(nn.Module):
    """Routes feature subsets to specific experts based on config.

    Given a concatenated feature tensor ``(batch, total_dim)`` and group
    ranges, slices the appropriate features for each expert.

    If no routing is configured for an expert, that expert receives the full
    feature tensor (backward compatible).

    Args:
        expert_names: Ordered list of expert name strings.
        routing_config: List of ``ExpertInputConfig`` defining per-expert
            feature group assignments. Experts not mentioned receive ALL
            features.
        group_ranges: ``{group_name: (start_idx, end_idx)}`` mapping from
            feature group names to their column ranges in the concatenated
            feature tensor.
    """

    def __init__(
        self,
        expert_names: List[str],
        routing_config: List[ExpertInputConfig],
        group_ranges: Dict[str, Tuple[int, int]],
    ) -> None:
        super().__init__()
        self.expert_names = list(expert_names)
        self.group_ranges = dict(group_ranges)

        # Build a lookup: expert_name -> list of (start, end) slices
        # Experts with no routing entry get None (= full tensor)
        self._expert_slices: Dict[str, Optional[List[Tuple[int, int]]]] = {}

        routing_by_name = {rc.expert_name: rc for rc in routing_config}

        for expert_name in self.expert_names:
            rc = routing_by_name.get(expert_name)
            if rc is None or not rc.input_groups:
                # No routing config or empty groups -> full features
                self._expert_slices[expert_name] = None
            else:
                slices: List[Tuple[int, int]] = []
                for group_name in rc.input_groups:
                    if group_name not in self.group_ranges:
                        raise ValueError(
                            f"Expert '{expert_name}' references unknown feature "
                            f"group '{group_name}'. Available groups: "
                            f"{list(self.group_ranges.keys())}"
                        )
                    slices.append(self.group_ranges[group_name])
                # Sort by start index for deterministic concatenation order
                slices.sort(key=lambda s: s[0])
                self._expert_slices[expert_name] = slices

        # Pre-compute expert input dimensions
        total_dim = max(end for _, end in group_ranges.values()) if group_ranges else 0
        self._expert_input_dims: Dict[str, int] = {}
        for expert_name in self.expert_names:
            slices = self._expert_slices.get(expert_name)
            if slices is None:
                self._expert_input_dims[expert_name] = total_dim
            else:
                self._expert_input_dims[expert_name] = sum(
                    end - start for start, end in slices
                )

        # Register index tensors as buffers for efficient slicing on GPU
        # We store them as flat index tensors: one per expert that has routing
        for expert_name in self.expert_names:
            slices = self._expert_slices.get(expert_name)
            if slices is not None:
                indices = []
                for start, end in slices:
                    indices.extend(range(start, end))
                idx_tensor = torch.tensor(indices, dtype=torch.long)
                # Register as buffer so it moves with .to(device) automatically
                self.register_buffer(
                    f"_idx_{expert_name}",
                    idx_tensor,
                    persistent=False,
                )

        # Log routing summary
        for expert_name in self.expert_names:
            slices = self._expert_slices.get(expert_name)
            dim = self._expert_input_dims[expert_name]
            if slices is None:
                logger.info(
                    f"FeatureRouter: expert '{expert_name}' -> ALL features "
                    f"(dim={dim})"
                )
            else:
                groups = routing_by_name[expert_name].input_groups
                logger.info(
                    f"FeatureRouter: expert '{expert_name}' -> "
                    f"groups {groups} (dim={dim})"
                )

    def route(self, features: torch.Tensor, expert_name: str) -> torch.Tensor:
        """Return the feature slice for a specific expert.

        Args:
            features: ``(batch, total_dim)`` concatenated feature tensor.
            expert_name: Name of the expert to route features for.

        Returns:
            ``(batch, expert_input_dim)`` tensor. If no routing is configured
            for this expert, returns ``features`` unchanged.

        Raises:
            KeyError: If ``expert_name`` is not in the router's expert list.
        """
        if expert_name not in self._expert_slices:
            raise KeyError(
                f"Unknown expert '{expert_name}'. "
                f"Known experts: {self.expert_names}"
            )

        slices = self._expert_slices[expert_name]
        if slices is None:
            return features

        # Use pre-registered index buffer for efficient GPU slicing
        idx = getattr(self, f"_idx_{expert_name}")
        return features[:, idx]

    def get_expert_input_dim(self, expert_name: str) -> int:
        """Return the input dimension for a specific expert.

        This is the sum of all feature group dimensions assigned to the
        expert, or ``total_dim`` if the expert receives all features.

        Args:
            expert_name: Name of the expert.

        Returns:
            Integer input dimension.

        Raises:
            KeyError: If ``expert_name`` is not in the router.
        """
        if expert_name not in self._expert_input_dims:
            raise KeyError(
                f"Unknown expert '{expert_name}'. "
                f"Known experts: {self.expert_names}"
            )
        return self._expert_input_dims[expert_name]

    def has_routing(self, expert_name: str) -> bool:
        """Return True if the expert has explicit routing (not full tensor).

        Args:
            expert_name: Name of the expert.
        """
        return (
            expert_name in self._expert_slices
            and self._expert_slices[expert_name] is not None
        )

    def __repr__(self) -> str:
        parts = [f"FeatureRouter(experts={self.expert_names}"]
        for name in self.expert_names:
            dim = self._expert_input_dims.get(name, "?")
            routed = "routed" if self.has_routing(name) else "full"
            parts.append(f"  {name}: dim={dim} ({routed})")
        parts.append(")")
        return "\n".join(parts)
