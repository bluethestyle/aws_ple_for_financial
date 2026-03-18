"""
Interpretation Registry — cascading feature × task interpretation resolution.
=============================================================================

Connects feature definitions (feature_groups.yaml) and task definitions
(pipeline.yaml) into a single lookup interface for grounding context.

The 644 features × 16 tasks = 10,304 possible combinations are managed
via a 3-level cascade (similar to CSS specificity):

    Level 3 (Feature × Task):    sparse manual overrides (~20 entries)
    Level 2 (FeatureGroup × Task): medium specificity (~50 entries)
    Level 1 (FeatureGroup × TaskGroup): auto-generated defaults (52 entries)
    Fallback: glossary base template with {value} substitution

Resolution order: Level 3 → Level 2 → Level 1 → Fallback

This means:
  - Adding a new feature group → auto-gets Level 1 interpretations for all tasks
  - Adding a new task → auto-gets Level 1 interpretations for all feature groups
  - Critical feature×task combos → manually override at Level 2 or 3

Config sources (single responsibility per file):
  - feature_groups.yaml: feature group definitions (names, dims, target_experts)
  - pipeline.yaml: task definitions (name, type, group)
  - feature_glossary.yaml: per-feature Korean text + task_interpretations (Level 3)
  - interpretation_matrix.yaml (new): Level 1 + Level 2 definitions

Usage::

    registry = InterpretationRegistry.from_configs(
        feature_groups_path="configs/financial/feature_groups.yaml",
        pipeline_path="configs/financial/pipeline.yaml",
        glossary_path="configs/financial/feature_glossary.yaml",
        matrix_path="configs/financial/interpretation_matrix.yaml",
    )

    # Single entry point for all interpretation needs
    text = registry.interpret("tda_short_001", value=0.8, task="churn")
    # → Level 3 hit: "최근 소비 패턴에 급격한 변화가 감지되어 이탈 위험 신호"

    text = registry.interpret("tda_short_099", value=0.5, task="churn")
    # → Level 3 miss → Level 2 hit: "소비 구조의 위상적 변화 → 이탈 관련 신호"

    text = registry.interpret("new_feature_001", value=0.3, task="new_task")
    # → Level 3 miss → Level 2 miss → Level 1 auto: group×taskgroup default
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = ["InterpretationRegistry"]


# ---------------------------------------------------------------------------
# Feature group ↔ Task group mapping
# ---------------------------------------------------------------------------

# Default task-to-taskgroup mapping (from pipeline.yaml task_groups)
_DEFAULT_TASK_TO_GROUP: Dict[str, str] = {
    "ctr": "engagement",
    "cvr": "engagement",
    "churn": "lifecycle",
    "retention": "lifecycle",
    "life_stage": "lifecycle",
    "ltv": "lifecycle",
    "balance_util": "value",
    "engagement": "value",
    "channel": "value",
    "timing": "value",
    "nba": "consumption",
    "spending_category": "consumption",
    "consumption_cycle": "consumption",
    "spending_bucket": "consumption",
    "merchant_affinity": "consumption",
    "brand_prediction": "consumption",
}

# Default feature-name-prefix to feature group mapping
_DEFAULT_PREFIX_TO_GROUP: Dict[str, str] = {
    "rfm_": "base_rfm",
    "category_": "base_category",
    "txn_": "base_txn_stats",
    "temporal_": "base_temporal",
    "multi_": "multi_source",
    "ext_": "extended_source",
    "tda_": "tda_topology",
    "gmm_": "gmm_clustering",
    "mamba_": "mamba_temporal",
    "income_": "economics",
    "financial_": "economics",
    "chemical_": "multidisciplinary",
    "epidemic_": "multidisciplinary",
    "interference_": "multidisciplinary",
    "crime_": "multidisciplinary",
    "model_": "model_derived",
    "hmm_": "hmm_states",
    "graph_": "graph_embeddings",
    "merchant_": "merchant_hierarchy",
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class InterpretationRegistry:
    """Cascading feature × task interpretation resolver.

    Manages the full interpretation chain from feature+task to Korean text,
    using a 3-level cascade to avoid the 10,304-cell explosion problem.

    Args:
        level1: FeatureGroup × TaskGroup → text.
            Auto-generated from group descriptions + task group purpose.
        level2: FeatureGroup × Task → text.
            Manual overrides for important group×task combos.
        level3: Feature × Task → text.
            Sparse manual overrides for critical individual features.
        glossary: Feature → base template.
            Fallback when no level matches.
        task_to_group: Task name → task group name mapping.
        prefix_to_group: Feature name prefix → feature group name mapping.
    """

    def __init__(
        self,
        level1: Optional[Dict[str, Dict[str, str]]] = None,
        level2: Optional[Dict[str, Dict[str, str]]] = None,
        level3: Optional[Dict[str, Dict[str, str]]] = None,
        glossary: Optional[Dict[str, Dict[str, Any]]] = None,
        task_to_group: Optional[Dict[str, str]] = None,
        prefix_to_group: Optional[Dict[str, str]] = None,
    ) -> None:
        # Level 1: feature_group × task_group → text
        self._level1: Dict[str, Dict[str, str]] = level1 or {}
        # Level 2: feature_group × task → text
        self._level2: Dict[str, Dict[str, str]] = level2 or {}
        # Level 3: feature_name × task → text
        self._level3: Dict[str, Dict[str, str]] = level3 or {}
        # Fallback: feature → {template, name, ...}
        self._glossary: Dict[str, Dict[str, Any]] = glossary or {}

        self._task_to_group = task_to_group or dict(_DEFAULT_TASK_TO_GROUP)
        self._prefix_to_group = prefix_to_group or dict(_DEFAULT_PREFIX_TO_GROUP)

        logger.info(
            "InterpretationRegistry: L1=%d groups, L2=%d entries, "
            "L3=%d entries, glossary=%d features",
            len(self._level1), len(self._level2),
            len(self._level3), len(self._glossary),
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def interpret(
        self,
        feature_name: str,
        value: float = 0.0,
        task: str = "",
    ) -> str:
        """Resolve the best interpretation for a feature × task combination.

        Resolution cascade:
            Level 3 (exact feature × task)
            → Level 2 (feature group × task)
            → Level 1 (feature group × task group)
            → Fallback (glossary template with {value})

        Args:
            feature_name: Feature identifier (e.g. ``"tda_short_001"``).
            value: Feature value for template substitution.
            task: Task name (e.g. ``"churn"``). Empty → skip L1-L3.

        Returns:
            Korean text interpretation string.
        """
        if task:
            # Level 3: exact feature × task
            text = self._lookup_level3(feature_name, task)
            if text:
                return text

            # Level 2: feature_group × task
            feature_group = self._resolve_feature_group(feature_name)
            if feature_group:
                text = self._lookup_level2(feature_group, task)
                if text:
                    return text

                # Level 1: feature_group × task_group
                task_group = self._task_to_group.get(task, "")
                if task_group:
                    text = self._lookup_level1(feature_group, task_group)
                    if text:
                        return text

        # Fallback: glossary template
        return self._glossary_fallback(feature_name, value)

    def interpret_batch(
        self,
        features: List[Tuple[str, float]],
        task: str = "",
    ) -> List[Dict[str, Any]]:
        """Interpret a batch of (feature_name, value) pairs.

        Convenience method for interpreting top-K features at once.

        Returns:
            List of dicts with ``name``, ``value``, ``text``, ``level``.
        """
        results = []
        for feat_name, feat_value in features:
            text = self.interpret(feat_name, feat_value, task)
            level = self._resolve_level(feat_name, task)
            results.append({
                "name": feat_name,
                "value": feat_value,
                "text": text,
                "level": level,
                "feature_group": self._resolve_feature_group(feat_name) or "unknown",
            })
        return results

    # ------------------------------------------------------------------
    # Registration (runtime)
    # ------------------------------------------------------------------

    def register_task(
        self,
        task_name: str,
        task_group: str,
    ) -> None:
        """Register a new task at runtime (e.g. credit_risk → lifecycle).

        Level 1 interpretations are auto-available via group × taskgroup.
        """
        self._task_to_group[task_name] = task_group
        logger.info(
            "InterpretationRegistry: registered task '%s' → group '%s'",
            task_name, task_group,
        )

    def register_feature_group(
        self,
        prefix: str,
        group_name: str,
    ) -> None:
        """Register a new feature prefix → group mapping."""
        self._prefix_to_group[prefix] = group_name

    def set_level2(
        self,
        feature_group: str,
        task: str,
        text: str,
    ) -> None:
        """Set a Level 2 override."""
        if feature_group not in self._level2:
            self._level2[feature_group] = {}
        self._level2[feature_group][task] = text

    def set_level3(
        self,
        feature_name: str,
        task: str,
        text: str,
    ) -> None:
        """Set a Level 3 override."""
        if feature_name not in self._level3:
            self._level3[feature_name] = {}
        self._level3[feature_name][task] = text

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_configs(
        cls,
        feature_groups_path: str = "",
        pipeline_path: str = "",
        glossary_path: str = "",
        matrix_path: str = "",
    ) -> "InterpretationRegistry":
        """Build from config files.

        Args:
            feature_groups_path: Path to feature_groups.yaml.
            pipeline_path: Path to pipeline.yaml.
            glossary_path: Path to feature_glossary.yaml.
            matrix_path: Path to interpretation_matrix.yaml (Level 1 + 2).

        Missing files are silently skipped — the registry works with
        whatever is available.
        """
        try:
            import yaml
        except ImportError:
            logger.warning("PyYAML not available; returning empty registry")
            return cls()

        level1: Dict[str, Dict[str, str]] = {}
        level2: Dict[str, Dict[str, str]] = {}
        level3: Dict[str, Dict[str, str]] = {}
        glossary: Dict[str, Dict[str, Any]] = {}
        task_to_group: Dict[str, str] = dict(_DEFAULT_TASK_TO_GROUP)
        prefix_to_group: Dict[str, str] = dict(_DEFAULT_PREFIX_TO_GROUP)

        # 1. Pipeline: task → task_group mapping
        if pipeline_path:
            try:
                with open(pipeline_path, "r", encoding="utf-8") as f:
                    pipe_data = yaml.safe_load(f) or {}
                for tg in pipe_data.get("task_groups", []):
                    group_name = tg.get("name", "")
                    for task_name in tg.get("tasks", []):
                        task_to_group[task_name] = group_name
            except Exception as e:
                logger.warning("Failed to load pipeline config: %s", e)

        # 2. Glossary: features + Level 3 (task_interpretations)
        if glossary_path:
            try:
                with open(glossary_path, "r", encoding="utf-8") as f:
                    gloss_data = yaml.safe_load(f) or {}

                # Parse feature definitions
                for group_name, group_def in gloss_data.get("feature_groups", {}).items():
                    for feat_id, feat_def in group_def.get("features", {}).items():
                        glossary[feat_id] = {
                            "name": feat_def.get("name", feat_id),
                            "template": feat_def.get("template", "{value}"),
                            "group": group_name,
                        }

                # Parse Level 3: task_interpretations
                for feat_id, task_map in gloss_data.get("task_interpretations", {}).items():
                    if isinstance(task_map, dict):
                        level3[feat_id] = task_map

            except Exception as e:
                logger.warning("Failed to load glossary: %s", e)

        # 3. Interpretation matrix: Level 1 + Level 2
        if matrix_path:
            try:
                with open(matrix_path, "r", encoding="utf-8") as f:
                    matrix_data = yaml.safe_load(f) or {}

                level1 = matrix_data.get("group_x_taskgroup", {})
                level2 = matrix_data.get("group_x_task", {})

            except Exception as e:
                logger.warning("Failed to load interpretation matrix: %s", e)

        # 4. Auto-generate Level 1 defaults for any missing group × taskgroup
        if not level1:
            level1 = cls._auto_generate_level1(
                prefix_to_group, task_to_group,
            )

        return cls(
            level1=level1,
            level2=level2,
            level3=level3,
            glossary=glossary,
            task_to_group=task_to_group,
            prefix_to_group=prefix_to_group,
        )

    @staticmethod
    def _auto_generate_level1(
        prefix_to_group: Dict[str, str],
        task_to_group: Dict[str, str],
    ) -> Dict[str, Dict[str, str]]:
        """Auto-generate Level 1 interpretations from group semantics.

        Uses the feature group name and task group purpose to produce
        generic but meaningful interpretations.  These are the "CSS
        defaults" that cover all combinations without manual work.
        """
        # Feature group → semantic description
        group_semantics: Dict[str, str] = {
            "base_rfm": "고객 거래 빈도/금액 패턴",
            "base_category": "카테고리별 소비 비중",
            "base_txn_stats": "거래 통계 지표",
            "base_temporal": "시간대별 거래 패턴",
            "multi_source": "다채널 금융 활동",
            "extended_source": "확장 금융 서비스 이용",
            "tda_topology": "소비 구조의 위상적 패턴",
            "gmm_clustering": "고객 세그먼트 소속 특성",
            "mamba_temporal": "시계열 거래 동향",
            "economics": "경제 행동 및 재무 상태",
            "multidisciplinary": "다학제 소비 동태 분석",
            "model_derived": "모델 기반 파생 지표",
            "hmm_states": "행동 상태 전이 패턴",
            "graph_embeddings": "거래 네트워크 위치",
            "merchant_hierarchy": "가맹점 이용 구조",
        }

        # Task group → interpretation perspective
        taskgroup_perspectives: Dict[str, str] = {
            "engagement": "고객 관심도 및 참여 활성화 관점에서 해석됩니다",
            "lifecycle": "고객 생애주기 관리 관점에서 해석됩니다",
            "value": "고객 가치 및 활용도 최적화 관점에서 해석됩니다",
            "consumption": "소비 행동 및 니즈 매칭 관점에서 해석됩니다",
        }

        level1: Dict[str, Dict[str, str]] = {}
        unique_groups = set(prefix_to_group.values())
        unique_taskgroups = set(task_to_group.values())

        for fg in unique_groups:
            fg_desc = group_semantics.get(fg, fg)
            level1[fg] = {}
            for tg in unique_taskgroups:
                tg_perspective = taskgroup_perspectives.get(tg, tg)
                level1[fg][tg] = f"{fg_desc}이(가) {tg_perspective}"

        return level1

    # ------------------------------------------------------------------
    # Internal lookups
    # ------------------------------------------------------------------

    def _lookup_level3(self, feature_name: str, task: str) -> Optional[str]:
        """Level 3: exact feature × task."""
        entry = self._level3.get(feature_name, {})
        if task in entry:
            return entry[task]
        # Prefix match
        for key, task_map in self._level3.items():
            if feature_name.startswith(key) and task in task_map:
                return task_map[task]
        return None

    def _lookup_level2(self, feature_group: str, task: str) -> Optional[str]:
        """Level 2: feature_group × task."""
        return self._level2.get(feature_group, {}).get(task)

    def _lookup_level1(self, feature_group: str, task_group: str) -> Optional[str]:
        """Level 1: feature_group × task_group."""
        return self._level1.get(feature_group, {}).get(task_group)

    def _resolve_feature_group(self, feature_name: str) -> Optional[str]:
        """Determine which feature group a feature belongs to."""
        # Check glossary first
        entry = self._glossary.get(feature_name)
        if entry:
            return entry.get("group")

        # Prefix matching
        for prefix, group in self._prefix_to_group.items():
            if feature_name.startswith(prefix):
                return group

        return None

    def _resolve_level(self, feature_name: str, task: str) -> str:
        """Determine which level resolved for diagnostics."""
        if not task:
            return "fallback"
        if self._lookup_level3(feature_name, task):
            return "L3"
        fg = self._resolve_feature_group(feature_name)
        if fg and self._lookup_level2(fg, task):
            return "L2"
        if fg:
            tg = self._task_to_group.get(task, "")
            if tg and self._lookup_level1(fg, tg):
                return "L1"
        return "fallback"

    def _glossary_fallback(self, feature_name: str, value: float) -> str:
        """Fallback: glossary template with value substitution."""
        entry = self._glossary.get(feature_name)
        if entry:
            template = entry.get("template", "{value}")
            try:
                if isinstance(value, float):
                    if value == int(value):
                        dv = str(int(value))
                    elif abs(value) >= 100:
                        dv = f"{value:,.0f}"
                    elif abs(value) >= 1:
                        dv = f"{value:.1f}"
                    else:
                        dv = f"{value:.2f}"
                else:
                    dv = str(value)
                return template.format(value=dv)
            except (KeyError, IndexError, ValueError):
                return f"{entry.get('name', feature_name)}: {value}"
        return f"{feature_name}: {value}"

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def coverage_report(self) -> Dict[str, Any]:
        """Return a coverage report showing how many combinations are
        covered at each level.

        Useful for identifying gaps in interpretation coverage.
        """
        all_feature_groups = set(self._prefix_to_group.values())
        all_tasks = set(self._task_to_group.keys())
        all_task_groups = set(self._task_to_group.values())

        total_combos = len(all_feature_groups) * len(all_tasks)

        l1_coverage = sum(
            1 for fg in all_feature_groups
            for tg in all_task_groups
            if self._level1.get(fg, {}).get(tg)
        )
        l2_coverage = sum(
            1 for fg in all_feature_groups
            for t in all_tasks
            if self._level2.get(fg, {}).get(t)
        )
        l3_entries = sum(len(v) for v in self._level3.values())

        return {
            "total_feature_groups": len(all_feature_groups),
            "total_tasks": len(all_tasks),
            "total_task_groups": len(all_task_groups),
            "total_possible_combos": total_combos,
            "level1_coverage": l1_coverage,
            "level2_overrides": l2_coverage,
            "level3_overrides": l3_entries,
            "glossary_features": len(self._glossary),
            "effective_coverage_pct": round(
                min(100.0, (l1_coverage + l2_coverage + l3_entries) / max(total_combos, 1) * 100),
                1,
            ),
        }
