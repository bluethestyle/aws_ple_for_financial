"""
Feature-group-aware knowledge distillation.

Extracted from distillation.py to keep each module under 500 lines.
Public API is unchanged — import FeatureGroupDistillation from here
or from core.training.distillation (re-exported for backwards compatibility).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from core.feature.group_config import FeatureGroupConfig

logger = logging.getLogger(__name__)


class FeatureGroupDistillation(nn.Module):
    """Feature-group-aware distillation with configurable per-group weights.

    Some feature groups (e.g. TDA topology, graph embeddings) are inherently
    harder to replicate in lightweight LGBM student models.  This class
    computes a weighted MSE loss across feature groups, where each group's
    contribution is controlled by its ``distill_weight`` in the
    :class:`FeatureGroupConfig`.

    Groups with ``distill=False`` are excluded entirely (e.g. HMM states
    that cannot be meaningfully distilled to tree-based models).

    This class reads its configuration directly from FeatureGroupConfig
    objects -- the single source of truth -- so no manual weight
    duplication is needed.

    Args:
        groups: List of FeatureGroupConfig instances.  Only groups with
                ``enabled=True`` and ``distill=True`` participate.

    Example::

        from core.feature.group_config import load_feature_groups
        from core.training.feature_group_distillation import FeatureGroupDistillation

        groups = load_feature_groups("configs/feature_groups.yaml")
        fg_distill = FeatureGroupDistillation(groups)
        ranges = FeatureGroupConfig.compute_group_ranges(groups)

        loss = fg_distill(teacher_features, student_features, ranges)
    """

    def __init__(self, groups: List["FeatureGroupConfig"]) -> None:
        super().__init__()
        self.distill_groups: List["FeatureGroupConfig"] = [
            g for g in groups if g.enabled and g.distill
        ]
        self.weights: Dict[str, float] = {
            g.name: g.distill_weight for g in self.distill_groups
        }

        # Validate
        if not self.distill_groups:
            logger.warning(
                "FeatureGroupDistillation: no distillable groups found. "
                "Loss will always be zero."
            )
        else:
            logger.info(
                "FeatureGroupDistillation: %d distillable groups — %s",
                len(self.distill_groups),
                {g.name: g.distill_weight for g in self.distill_groups},
            )

    def forward(
        self,
        teacher_features: torch.Tensor,
        student_features: torch.Tensor,
        group_ranges: Dict[str, Tuple[int, int]],
    ) -> Dict[str, torch.Tensor]:
        """Compute weighted MSE loss per feature group.

        Args:
            teacher_features: Teacher intermediate representations,
                              shape ``(batch, total_dim)``.
            student_features: Student intermediate representations,
                              shape ``(batch, total_dim)``.  Must have the
                              same total_dim as the teacher (use a projection
                              layer if dimensions differ).
            group_ranges: Dict mapping group name to ``(start, end)`` index
                          tuple in the concatenated feature vector.  Obtain
                          via ``FeatureGroupConfig.compute_group_ranges()``.

        Returns:
            Dict with:
                * ``"total"``: Scalar weighted sum of per-group losses.
                * ``"per_group"``: Dict of ``{group_name: scalar_loss}``.
                * ``"weights_used"``: Dict of ``{group_name: weight}``.
        """
        device = teacher_features.device
        total_loss = torch.tensor(0.0, device=device)
        per_group: Dict[str, torch.Tensor] = {}

        teacher_features = teacher_features.detach()

        for group in self.distill_groups:
            if group.name not in group_ranges:
                logger.warning(
                    "FeatureGroupDistillation: group '%s' not found in "
                    "group_ranges — skipping.",
                    group.name,
                )
                continue

            start, end = group_ranges[group.name]

            # Bounds checking
            if end > teacher_features.size(1) or end > student_features.size(1):
                logger.warning(
                    "FeatureGroupDistillation: group '%s' range [%d, %d) "
                    "exceeds feature dimension (teacher=%d, student=%d) "
                    "— skipping.",
                    group.name,
                    start,
                    end,
                    teacher_features.size(1),
                    student_features.size(1),
                )
                continue

            teacher_slice = teacher_features[:, start:end]
            student_slice = student_features[:, start:end]

            group_loss = F.mse_loss(student_slice, teacher_slice)
            weight = self.weights[group.name]

            per_group[group.name] = group_loss
            total_loss = total_loss + weight * group_loss

        return {
            "total": total_loss,
            "per_group": per_group,
            "weights_used": dict(self.weights),
        }

    @classmethod
    def from_feature_groups(
        cls, groups: List["FeatureGroupConfig"]
    ) -> "FeatureGroupDistillation":
        """Convenience constructor matching the pattern used by other modules.

        Identical to ``cls(groups)`` but provides a consistent API surface
        alongside ``ReverseMapper.from_feature_groups()`` and
        ``TemplateEngine.from_feature_groups()``.

        Args:
            groups: List of FeatureGroupConfig instances.

        Returns:
            A configured FeatureGroupDistillation instance.
        """
        return cls(groups)
