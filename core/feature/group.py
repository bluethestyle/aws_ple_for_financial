"""
Feature Group system -- the core abstraction for organic feature engineering.

A :class:`FeatureGroupConfig` is the **single source of truth** for a group of
related features.  It declares:

* **What** features exist (columns, dimensions).
* **How** they are created (generator or transformer chain).
* **Which experts** receive them (expert routing for PLE).
* **How to interpret** them (templates for recommendation explanations).
* **Whether** to include them in knowledge distillation.

:class:`FeatureGroupRegistry` collects all enabled groups and provides
efficient lookups that downstream systems consume:

* ``get_expert_feature_mapping``  -- used by PLE expert gating.
* ``get_interpretation_mapping``  -- used by ReverseMapper / ReasonGenerator.
* ``get_distillation_groups``     -- used by distillation pipeline.
* ``get_group_dim_ranges``        -- used by Integrated Gradients attribution.

Design principle: **one config object, many consumers**.  Adding a new feature
group requires editing only the FeatureGroupConfig list; all downstream
systems discover it automatically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# ======================================================================
# Interpretation config
# ======================================================================


@dataclass
class FeatureInterpretationConfig:
    """How to interpret a feature group for recommendation explanations.

    This config propagates automatically to the reverse-mapper and
    reason-generator so that every generated feature carries a
    human-readable interpretation template.

    Parameters
    ----------
    category : str
        Semantic category for grouping in explanations (e.g.
        ``"demographics"``, ``"domain_topology"``, ``"temporal"``).
    template : str
        Python format-string template for rendering a single feature
        contribution.  Available placeholders: ``{feature}``,
        ``{value}``, ``{direction}`` (positive/negative).
        Example: ``"{feature} is {value}, indicating {direction}"``.
    narrative_lens : str
        Narrative lens for the LLM reason generator.  Controls the
        *perspective* from which the feature is described (e.g.
        ``"engagement"``, ``"lifecycle"``, ``"value"``).
    primary_tasks : list[str]
        Task names for which this feature group is most relevant.
        Used by the task-aware interpretation pipeline to weight
        feature importance appropriately.
    """

    category: str = "general"
    template: str = "{feature} is {value}, indicating {direction}"
    narrative_lens: str = "engagement"
    primary_tasks: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary (JSON-safe)."""
        return {
            "category": self.category,
            "template": self.template,
            "narrative_lens": self.narrative_lens,
            "primary_tasks": list(self.primary_tasks),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeatureInterpretationConfig":
        """Deserialise from a dictionary."""
        return cls(
            category=d.get("category", "general"),
            template=d.get("template", "{feature} is {value}, indicating {direction}"),
            narrative_lens=d.get("narrative_lens", "engagement"),
            primary_tasks=d.get("primary_tasks", []),
        )


# ======================================================================
# Container config (for runtime isolation)
# ======================================================================


@dataclass
class ContainerConfig:
    """Configuration for container-isolated execution of a feature group.

    When a feature group specifies ``runtime="container"``, the pipeline
    uploads input data to S3, launches a SageMaker Processing Job with
    the specified container image, and downloads the output.

    Parameters
    ----------
    image : str
        ECR image URI (e.g. ``"123456789.dkr.ecr.ap-northeast-2.amazonaws.com/feature-tda:latest"``).
        Required when ``runtime="container"``.
    instance_type : str
        SageMaker instance type for the processing job.
    instance_count : int
        Number of processing instances.
    volume_size_gb : int
        EBS volume size in GB attached to each processing instance.
    max_runtime_seconds : int
        Maximum job duration before timeout.
    requirements : list[str]
        Python packages to install at container startup (in addition
        to what is baked into the image).
    env : dict[str, str]
        Environment variables injected into the container.
    s3_staging_prefix : str
        S3 prefix for staging input/output data during container
        execution.
    """

    image: str = ""
    instance_type: str = "ml.m5.xlarge"
    instance_count: int = 1
    volume_size_gb: int = 30
    max_runtime_seconds: int = 3600
    requirements: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    s3_staging_prefix: str = "s3://sagemaker-default/feature-pipeline/staging"

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary (JSON-safe)."""
        return {
            "image": self.image,
            "instance_type": self.instance_type,
            "instance_count": self.instance_count,
            "volume_size_gb": self.volume_size_gb,
            "max_runtime_seconds": self.max_runtime_seconds,
            "requirements": list(self.requirements),
            "env": dict(self.env),
            "s3_staging_prefix": self.s3_staging_prefix,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ContainerConfig":
        """Deserialise from a dictionary."""
        return cls(
            image=d.get("image", ""),
            instance_type=d.get("instance_type", "ml.m5.xlarge"),
            instance_count=d.get("instance_count", 1),
            volume_size_gb=d.get("volume_size_gb", 30),
            max_runtime_seconds=d.get("max_runtime_seconds", 3600),
            requirements=d.get("requirements", []),
            env=d.get("env", {}),
            s3_staging_prefix=d.get(
                "s3_staging_prefix",
                "s3://sagemaker-default/feature-pipeline/staging",
            ),
        )


# ======================================================================
# Feature Group Config
# ======================================================================


@dataclass
class FeatureGroupConfig:
    """Single feature group definition -- the unit of organic connection.

    This dataclass is the **single source of truth** for a group of
    related features.  It tells every downstream system what it needs:
    the feature pipeline (generator or transformer chain), expert routing,
    interpretation, and distillation.

    Parameters
    ----------
    name : str
        Unique group identifier (e.g. ``"tda_topology"``).
    group_type : str
        ``"generate"`` for feature generators, ``"transform"`` for
        transformer chains on existing columns.
    generator : str or None
        Registry name of the feature generator (when
        ``group_type="generate"``).
    generator_params : dict
        Keyword arguments forwarded to the generator constructor.
    transformers : list[str]
        Ordered list of transformer registry names (when
        ``group_type="transform"``).
    transformer_params : dict
        ``{transformer_name: {param: value}}`` overrides.
    columns : list[str]
        Input columns for transform-type groups.
    output_dim : int
        Dimension of this group's output feature vector.
    output_columns : list[str]
        Explicit names for generated columns.  For generators, these
        are auto-populated from the generator's ``output_columns``
        property if left empty.
    target_experts : list[str]
        Names of PLE experts that should receive this feature group
        (e.g. ``["deepfm", "temporal"]``).
    interpretation : FeatureInterpretationConfig
        How to interpret features in this group for explanations.
    runtime : str
        Execution mode for this feature group:

        * ``"local"`` (default) -- run the generator/transformer chain
          in the current process.
        * ``"container"`` -- upload input to S3, run a SageMaker
          Processing Job with the specified container image, and
          download the output.  Useful when generators have conflicting
          Python dependencies (e.g. TDA needs ``ripser``, graph needs
          ``torch-geometric``).

    container : ContainerConfig
        Container configuration (only used when ``runtime="container"``).
    distill : bool
        Whether to include this group in knowledge distillation.
    distill_weight : float
        Relative importance weight in distillation loss.
    enabled : bool
        Master toggle.  Disabled groups are skipped entirely.
    """

    name: str = ""
    group_type: str = "transform"  # "transform" | "generate"

    # -- Generator config (group_type="generate") ----------------------
    generator: Optional[str] = None
    generator_params: Dict[str, Any] = field(default_factory=dict)

    # -- Transformer config (group_type="transform") -------------------
    transformers: List[str] = field(default_factory=list)
    transformer_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    columns: List[str] = field(default_factory=list)

    # -- Output --------------------------------------------------------
    output_dim: int = 0
    output_columns: List[str] = field(default_factory=list)

    # -- Expert routing ------------------------------------------------
    target_experts: List[str] = field(default_factory=list)

    # -- Interpretation ------------------------------------------------
    interpretation: FeatureInterpretationConfig = field(
        default_factory=FeatureInterpretationConfig
    )

    # -- Runtime isolation ---------------------------------------------
    runtime: str = "local"  # "local" | "container"
    container: ContainerConfig = field(default_factory=ContainerConfig)

    # -- Distillation --------------------------------------------------
    distill: bool = True
    distill_weight: float = 1.0

    # -- Toggle --------------------------------------------------------
    enabled: bool = True

    # -- Validation ----------------------------------------------------

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("FeatureGroupConfig.name must not be empty")
        if self.group_type not in ("transform", "generate"):
            raise ValueError(
                f"group_type must be 'transform' or 'generate', "
                f"got '{self.group_type}'"
            )
        if self.group_type == "generate" and not self.generator:
            raise ValueError(
                f"FeatureGroupConfig '{self.name}' has group_type='generate' "
                f"but no generator specified"
            )
        if self.group_type == "transform" and not self.transformers:
            raise ValueError(
                f"FeatureGroupConfig '{self.name}' has group_type='transform' "
                f"but no transformers specified"
            )
        if self.runtime not in ("local", "container"):
            raise ValueError(
                f"runtime must be 'local' or 'container', "
                f"got '{self.runtime}'"
            )
        if self.runtime == "container" and not self.container.image:
            raise ValueError(
                f"FeatureGroupConfig '{self.name}' has runtime='container' "
                f"but no container.image specified"
            )

    # -- Serialisation -------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary (JSON/YAML-safe)."""
        return {
            "name": self.name,
            "group_type": self.group_type,
            "generator": self.generator,
            "generator_params": dict(self.generator_params),
            "transformers": list(self.transformers),
            "transformer_params": {
                k: dict(v) for k, v in self.transformer_params.items()
            },
            "columns": list(self.columns),
            "output_dim": self.output_dim,
            "output_columns": list(self.output_columns),
            "target_experts": list(self.target_experts),
            "interpretation": self.interpretation.to_dict(),
            "runtime": self.runtime,
            "container": self.container.to_dict(),
            "distill": self.distill,
            "distill_weight": self.distill_weight,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeatureGroupConfig":
        """Deserialise from a dictionary."""
        interp_raw = d.get("interpretation", {})
        if isinstance(interp_raw, dict):
            interpretation = FeatureInterpretationConfig.from_dict(interp_raw)
        else:
            interpretation = interp_raw

        container_raw = d.get("container", {})
        if isinstance(container_raw, dict):
            container = ContainerConfig.from_dict(container_raw)
        else:
            container = container_raw

        return cls(
            name=d["name"],
            group_type=d.get("group_type", "transform"),
            generator=d.get("generator"),
            generator_params=d.get("generator_params", {}),
            transformers=d.get("transformers", []),
            transformer_params=d.get("transformer_params", {}),
            columns=d.get("columns", []),
            output_dim=d.get("output_dim", 0),
            output_columns=d.get("output_columns", []),
            target_experts=d.get("target_experts", []),
            interpretation=interpretation,
            runtime=d.get("runtime", "local"),
            container=container,
            distill=d.get("distill", True),
            distill_weight=d.get("distill_weight", 1.0),
            enabled=d.get("enabled", True),
        )


# ======================================================================
# Feature Group Registry
# ======================================================================


class FeatureGroupRegistry:
    """Manages all feature groups and provides lookups for downstream systems.

    This is not a *class-level* registry like FeatureRegistry (which
    registers transformer *classes*).  Instead, this is an *instance*
    that holds a concrete list of :class:`FeatureGroupConfig` objects
    representing the feature groups configured for a specific run.

    Parameters
    ----------
    groups : list[FeatureGroupConfig]
        All feature group configurations.  Disabled groups
        (``enabled=False``) are tracked but excluded from most lookups.
    """

    def __init__(self, groups: List[FeatureGroupConfig]) -> None:
        self._all_groups = list(groups)
        self._enabled_groups = [g for g in self._all_groups if g.enabled]
        self._by_name: Dict[str, FeatureGroupConfig] = {
            g.name: g for g in self._all_groups
        }

        # Validate uniqueness
        names = [g.name for g in self._all_groups]
        if len(names) != len(set(names)):
            dupes = [n for n in names if names.count(n) > 1]
            raise ValueError(f"Duplicate feature group names: {set(dupes)}")

        logger.info(
            "FeatureGroupRegistry: %d groups (%d enabled), total_dim=%d",
            len(self._all_groups),
            len(self._enabled_groups),
            self.get_total_dim(),
        )

    # -- Accessors -----------------------------------------------------

    @property
    def enabled_groups(self) -> List[FeatureGroupConfig]:
        """All enabled feature groups, in definition order."""
        return list(self._enabled_groups)

    @property
    def all_groups(self) -> List[FeatureGroupConfig]:
        """All feature groups including disabled ones."""
        return list(self._all_groups)

    def get_group(self, name: str) -> FeatureGroupConfig:
        """Retrieve a group by name.

        Raises
        ------
        KeyError
            If *name* is not registered.
        """
        if name not in self._by_name:
            raise KeyError(
                f"Unknown feature group '{name}'. "
                f"Available: {list(self._by_name.keys())}"
            )
        return self._by_name[name]

    # -- Lookups used by downstream systems ----------------------------

    def get_expert_feature_mapping(self) -> Dict[str, List[str]]:
        """Expert name -> list of feature group names it receives.

        Used by PLE model construction to wire feature groups to the
        correct expert networks.

        Returns
        -------
        dict[str, list[str]]
            Maps each expert name to the feature group names whose
            output is routed to that expert.
        """
        mapping: Dict[str, List[str]] = {}
        for group in self._enabled_groups:
            for expert in group.target_experts:
                mapping.setdefault(expert, []).append(group.name)
        return mapping

    def get_interpretation_mapping(self) -> Dict[str, FeatureInterpretationConfig]:
        """Feature group name -> interpretation config.

        Used by the reverse-mapper and reason-generator to produce
        human-readable explanations of feature contributions.

        Returns
        -------
        dict[str, FeatureInterpretationConfig]
        """
        return {
            group.name: group.interpretation
            for group in self._enabled_groups
        }

    def get_distillation_groups(self) -> List[FeatureGroupConfig]:
        """Return groups included in knowledge distillation.

        Returns
        -------
        list[FeatureGroupConfig]
            Enabled groups with ``distill=True``.
        """
        return [g for g in self._enabled_groups if g.distill]

    def get_total_dim(self) -> int:
        """Sum of all enabled groups' ``output_dim``.

        Returns
        -------
        int
            Total feature dimension after all groups are concatenated.
        """
        return sum(g.output_dim for g in self._enabled_groups)

    def get_group_dim_ranges(self) -> Dict[str, Tuple[int, int]]:
        """Feature group name -> ``(start_idx, end_idx)`` in the concatenated vector.

        The ranges are contiguous and non-overlapping, ordered by group
        definition order.  This mapping is essential for Integrated
        Gradients attribution (to slice the gradient vector per group).

        Returns
        -------
        dict[str, tuple[int, int]]
            Maps group name to ``(start, end)`` index range (exclusive end).
        """
        ranges: Dict[str, Tuple[int, int]] = {}
        offset = 0
        for group in self._enabled_groups:
            end = offset + group.output_dim
            ranges[group.name] = (offset, end)
            offset = end
        return ranges

    def get_groups_for_task(self, task_name: str) -> List[FeatureGroupConfig]:
        """Return groups whose interpretation lists *task_name* as primary.

        Parameters
        ----------
        task_name : str
            PLE task name.

        Returns
        -------
        list[FeatureGroupConfig]
        """
        return [
            g for g in self._enabled_groups
            if task_name in g.interpretation.primary_tasks
        ]

    def get_groups_by_type(self, group_type: str) -> List[FeatureGroupConfig]:
        """Return enabled groups filtered by ``group_type``.

        Parameters
        ----------
        group_type : str
            ``"generate"`` or ``"transform"``.

        Returns
        -------
        list[FeatureGroupConfig]
        """
        return [g for g in self._enabled_groups if g.group_type == group_type]

    # -- Serialisation -------------------------------------------------

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Serialise all groups to a list of dictionaries."""
        return [g.to_dict() for g in self._all_groups]

    @classmethod
    def from_dict_list(cls, dicts: List[Dict[str, Any]]) -> "FeatureGroupRegistry":
        """Deserialise from a list of dictionaries."""
        groups = [FeatureGroupConfig.from_dict(d) for d in dicts]
        return cls(groups)

    # -- Repr ----------------------------------------------------------

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"FeatureGroupRegistry: {len(self._enabled_groups)} enabled "
            f"/ {len(self._all_groups)} total groups, "
            f"total_dim={self.get_total_dim()}",
        ]
        for g in self._enabled_groups:
            expert_str = ", ".join(g.target_experts) if g.target_experts else "all"
            lines.append(
                f"  [{g.name}] type={g.group_type}, "
                f"dim={g.output_dim}, "
                f"experts=[{expert_str}], "
                f"distill={'Y' if g.distill else 'N'}"
            )
        disabled = [g for g in self._all_groups if not g.enabled]
        if disabled:
            lines.append(f"  (disabled: {[g.name for g in disabled]})")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover
        return self.summary()
