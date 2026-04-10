"""
Bias Stage Attributor for Audit Agent AV1
============================================

Measures Disparate Impact (DI) at each stage of the recommendation pipeline
to identify where bias is introduced or amplified:

    Stage 1: Model output logit → model's inherent bias
    Stage 2: ConstraintEngine filtering → business rule impact
    Stage 3: TopK selection → diversity method impact

Provides stage-by-stage DI delta analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["BiasStageAttributor", "StageAttribution"]


@dataclass
class StageDI:
    """DI measurement at a single pipeline stage."""
    stage: str
    stage_label: str
    di_value: float
    positive_rate_subgroup: float
    positive_rate_overall: float
    sample_size: int


@dataclass
class StageAttribution:
    """Full stage-by-stage bias attribution for one protected group."""
    attribute: str
    group_value: str
    stages: List[StageDI] = field(default_factory=list)
    worst_stage: str = ""
    worst_delta: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        stage_data = []
        for s in self.stages:
            stage_data.append({
                "stage": s.stage,
                "label": s.stage_label,
                "di": round(s.di_value, 4),
                "sample_size": s.sample_size,
            })

        return {
            "attribute": self.attribute,
            "group_value": self.group_value,
            "stages": stage_data,
            "worst_stage": self.worst_stage,
            "worst_delta": round(self.worst_delta, 4),
        }


class BiasStageAttributor:
    """Attributes bias to specific pipeline stages.

    Requires intermediate results from each pipeline stage:
    - model_scores: raw model output scores per customer
    - post_filter: candidates after ConstraintEngine
    - post_selection: final top-K selected items

    Args:
        config: Config dict with keys:
            - di_threshold: float (default 0.80)
            - min_sample_size: int (default 30)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self.di_threshold = cfg.get("di_threshold", 0.80)
        self.min_sample_size = cfg.get("min_sample_size", 30)

    def attribute(
        self,
        attribute: str,
        group_value: str,
        customers: List[Dict[str, Any]],
        model_scores: List[Dict[str, Any]],
        post_filter: List[Dict[str, Any]],
        post_selection: List[Dict[str, Any]],
    ) -> StageAttribution:
        """Compute DI at each stage for a specific protected group.

        Args:
            attribute: Protected attribute name (e.g., "age_group").
            group_value: Protected group value (e.g., "elderly").
            customers: Customer attribute dicts with at least {customer_id, <attribute>}.
            model_scores: Dicts with {customer_id, recommended: bool} from model stage.
            post_filter: Dicts with {customer_id, recommended: bool} after filtering.
            post_selection: Dicts with {customer_id, recommended: bool} after top-K.
        """
        # Build customer attribute lookup
        attr_lookup = {c.get("customer_id"): c.get(attribute) for c in customers}

        stages_data = [
            ("model_output", "Model Output Logit", model_scores),
            ("constraint_engine", "Constraint Engine", post_filter),
            ("topk_selection", "Top-K Selection", post_selection),
        ]

        stage_results = []
        for stage_name, stage_label, stage_data in stages_data:
            di = self._compute_stage_di(attr_lookup, group_value, stage_data)
            stage_results.append(StageDI(
                stage=stage_name,
                stage_label=stage_label,
                di_value=di["di"],
                positive_rate_subgroup=di["subgroup_rate"],
                positive_rate_overall=di["overall_rate"],
                sample_size=di["subgroup_size"],
            ))

        # Find worst DI delta between consecutive stages
        worst_stage = ""
        worst_delta = 0.0
        for i in range(1, len(stage_results)):
            delta = stage_results[i].di_value - stage_results[i - 1].di_value
            if delta < worst_delta:
                worst_delta = delta
                worst_stage = stage_results[i].stage

        result = StageAttribution(
            attribute=attribute,
            group_value=group_value,
            stages=stage_results,
            worst_stage=worst_stage,
            worst_delta=worst_delta,
        )

        if worst_delta < -0.1:
            logger.warning(
                "Bias amplification detected at %s: DI delta %.4f for %s=%s",
                worst_stage, worst_delta, attribute, group_value,
            )

        return result

    def _compute_stage_di(
        self,
        attr_lookup: Dict[str, Any],
        group_value: str,
        stage_data: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Compute DI for a protected group at one stage."""
        if not stage_data:
            return {"di": 0.0, "subgroup_rate": 0.0, "overall_rate": 0.0, "subgroup_size": 0}

        subgroup = []
        complement = []

        for item in stage_data:
            cid = item.get("customer_id", "")
            attr_val = attr_lookup.get(cid)
            if attr_val == group_value:
                subgroup.append(item)
            elif attr_val is not None:
                complement.append(item)

        if len(subgroup) < self.min_sample_size:
            return {"di": 1.0, "subgroup_rate": 0.0, "overall_rate": 0.0, "subgroup_size": len(subgroup)}

        all_items = subgroup + complement
        overall_positive = sum(1 for r in all_items if r.get("recommended", False))
        overall_rate = overall_positive / len(all_items) if all_items else 0

        subgroup_positive = sum(1 for r in subgroup if r.get("recommended", False))
        subgroup_rate = subgroup_positive / len(subgroup) if subgroup else 0

        di = subgroup_rate / overall_rate if overall_rate > 0 else 0.0

        return {
            "di": round(di, 4),
            "subgroup_rate": round(subgroup_rate, 4),
            "overall_rate": round(overall_rate, 4),
            "subgroup_size": len(subgroup),
        }

    def batch_attribute(
        self,
        attributes_groups: List[Dict[str, str]],
        customers: List[Dict[str, Any]],
        model_scores: List[Dict[str, Any]],
        post_filter: List[Dict[str, Any]],
        post_selection: List[Dict[str, Any]],
    ) -> List[StageAttribution]:
        """Run attribution for multiple attribute-group combinations.

        Args:
            attributes_groups: List of {"attribute": "age_group", "group_value": "elderly"}.
        """
        return [
            self.attribute(
                ag["attribute"], ag["group_value"],
                customers, model_scores, post_filter, post_selection,
            )
            for ag in attributes_groups
        ]
