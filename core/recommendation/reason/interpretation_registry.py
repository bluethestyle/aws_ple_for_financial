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

    # Tasks where IG auto-interpretation is unreliable (multiclass with
    # ambiguous class-summed gradients).  These fall through to L3→L2→L1.
    _IG_SKIP_TASKS: set = {
        "life_stage", "channel", "timing", "nba",
        "spending_category", "consumption_cycle", "brand_prediction",
    }

    def __init__(
        self,
        level1: Optional[Dict[str, Dict[str, str]]] = None,
        level2: Optional[Dict[str, Dict[str, str]]] = None,
        level3: Optional[Dict[str, Dict[str, str]]] = None,
        glossary: Optional[Dict[str, Dict[str, Any]]] = None,
        task_to_group: Optional[Dict[str, str]] = None,
        prefix_to_group: Optional[Dict[str, str]] = None,
        feature_medians: Optional[Dict[str, float]] = None,
        ig_sign_blacklist: Optional[Dict[str, set]] = None,
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

        # Per-feature median from training data (for high/low classification)
        # Loaded from DatasetRegistry.feature_stats or passed directly.
        self._feature_medians: Dict[str, float] = feature_medians or {}

        # Features whose IG sign flipped between model versions.
        # {task_name: {feature_name, ...}}  — IG interpretation disabled for these.
        self._ig_sign_blacklist: Dict[str, set] = ig_sign_blacklist or {}

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
        signed_ig: Optional[float] = None,
    ) -> str:
        """Resolve the best interpretation for a feature × task combination.

        Resolution cascade (5 levels):
            Level IG (signed IG auto-interpretation)
            → Level 3 (exact feature × task)
            → Level 2 (feature group × task)
            → Level 1 (feature group × task group)
            → Fallback (glossary template with {value})

        When ``signed_ig`` is provided, an automatic direction-aware
        interpretation is generated from the IG sign, feature glossary
        name, and task semantics.  This eliminates the need for most
        manual Level 2/3 entries.

        Args:
            feature_name: Feature identifier (e.g. ``"tda_short_001"``).
            value: Feature value for template substitution.
            task: Task name (e.g. ``"churn"``). Empty → skip L1-L3.
            signed_ig: Signed Integrated Gradient value.  Positive means
                feature value ↑ → task prediction ↑.

        Returns:
            Korean text interpretation string.
        """
        if task:
            # Level IG: auto-interpretation from signed gradient direction
            if signed_ig is not None:
                text = self._interpret_from_ig(feature_name, value, task, signed_ig)
                if text:
                    return text

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

    # ------------------------------------------------------------------
    # IG-based auto interpretation
    # ------------------------------------------------------------------

    # Task prediction semantics: what does "prediction goes up" mean?
    _TASK_PREDICTION_SEMANTICS: Dict[str, Dict[str, str]] = {
        # binary: prediction ↑ = probability of positive class ↑
        "ctr":       {"up": "클릭 가능성이 높아집니다",
                      "down": "클릭 가능성이 낮아집니다"},
        "cvr":       {"up": "전환 가능성이 높아집니다",
                      "down": "전환 가능성이 낮아집니다"},
        "churn":     {"up": "이탈 위험이 높아집니다",
                      "down": "이탈 위험이 낮아집니다"},
        "retention": {"up": "유지 가능성이 높아집니다",
                      "down": "유지 가능성이 낮아집니다"},
        # regression
        "ltv":       {"up": "고객 생애 가치가 높아집니다",
                      "down": "고객 생애 가치가 낮아집니다"},
        "balance_util": {"up": "잔액 활용도가 높아집니다",
                         "down": "잔액 활용도가 낮아집니다"},
        "engagement":   {"up": "참여도가 높아집니다",
                         "down": "참여도가 낮아집니다"},
        "spending_bucket": {"up": "지출 규모가 커집니다",
                            "down": "지출 규모가 작아집니다"},
        "merchant_affinity": {"up": "가맹점 선호도가 높아집니다",
                              "down": "가맹점 선호도가 낮아집니다"},
        # multiclass — direction is less meaningful, but still useful
        "nba":       {"up": "해당 행동 추천 확률이 높아집니다",
                      "down": "해당 행동 추천 확률이 낮아집니다"},
        "life_stage": {"up": "해당 생애 단계 분류 확률이 높아집니다",
                       "down": "해당 생애 단계 분류 확률이 낮아집니다"},
    }

    # Feature value direction semantics (from glossary "direction" field)
    _VALUE_DIRECTION_TEMPLATES: Dict[str, Dict[str, str]] = {
        "higher_is_better": {
            "high": "{name}이(가) 우수한 수준이어서",
            "low":  "{name}이(가) 낮은 수준이어서",
        },
        "lower_is_better": {
            "high": "{name}이(가) 높은 수준이어서",
            "low":  "{name}이(가) 안정적인 수준이어서",
        },
        "": {  # neutral
            "high": "{name}이(가) 높은 편이어서",
            "low":  "{name}이(가) 낮은 편이어서",
        },
    }

    def _interpret_from_ig(
        self,
        feature_name: str,
        value: float,
        task: str,
        signed_ig: float,
    ) -> Optional[str]:
        """Auto-generate interpretation from signed IG direction.

        Guards:
            - Multiclass tasks (_IG_SKIP_TASKS) → returns None (fallback to L3-L1)
            - Blacklisted feature×task (IG sign flipped between versions) → None
            - Missing glossary entry or task semantics → None

        Logic:
            1. Get feature Korean name from glossary
            2. Determine value level (high/low) using per-feature median
               (from training data statistics) instead of fixed 0.5
            3. Get feature value direction (higher_is_better etc.)
            4. Get task prediction semantic (churn ↑ = 이탈 위험 ↑)
            5. Combine: "{feature}이 {high/low}여서 {task prediction} {up/down}"

        Example:
            IG("tda_short_001", churn) = -0.15, value = 0.8, median = 0.45
            → value > median → "high"
            → "단기 토폴로지 패턴이 높은 편이어서 이탈 위험이 낮아집니다"

        Returns:
            Korean text, or None to fall through to L3→L2→L1.
        """
        # Guard 1: Skip multiclass tasks where IG direction is ambiguous
        if task in self._IG_SKIP_TASKS:
            return None

        # Guard 2: Skip features whose IG sign flipped between model versions
        if feature_name in self._ig_sign_blacklist.get(task, set()):
            return None

        # Get feature info from glossary
        entry = self._glossary.get(feature_name)
        if not entry:
            return None

        feat_name = entry.get("name", feature_name)
        direction = entry.get("direction", "")

        # Task prediction semantics
        task_sem = self._TASK_PREDICTION_SEMANTICS.get(task)
        if not task_sem:
            return None

        # Determine if feature value is high or low
        # Use per-feature median from training data when available,
        # otherwise fall back to 0.5
        median = self._feature_medians.get(feature_name, 0.5)
        value_level = "high" if value >= median else "low"

        # Feature value description
        dir_templates = self._VALUE_DIRECTION_TEMPLATES.get(
            direction, self._VALUE_DIRECTION_TEMPLATES[""]
        )
        value_desc = dir_templates[value_level].format(name=feat_name)

        # IG direction determines task prediction direction
        # signed_ig > 0: feature value ↑ → prediction ↑
        # signed_ig < 0: feature value ↑ → prediction ↓
        if value_level == "high":
            pred_desc = task_sem["up"] if signed_ig > 0 else task_sem["down"]
        else:
            pred_desc = task_sem["down"] if signed_ig > 0 else task_sem["up"]

        return f"{value_desc} {pred_desc}"

    # ------------------------------------------------------------------
    # IG sign stability check (called at model registration time)
    # ------------------------------------------------------------------

    def check_ig_sign_stability(
        self,
        prev_signed_ig: Dict[str, Dict[str, float]],
        curr_signed_ig: Dict[str, Dict[str, float]],
        flip_threshold: float = 0.01,
    ) -> Dict[str, set]:
        """Compare IG signs between two model versions.

        Features whose IG sign flips (crosses zero beyond threshold)
        are added to the blacklist — their IG auto-interpretation is
        disabled and falls back to Level 3/2/1.

        Args:
            prev_signed_ig: ``{task: {feature: signed_ig_value}}`` from
                previous model version.
            curr_signed_ig: Same structure from current version.
            flip_threshold: Minimum absolute IG value to consider a sign
                meaningful.  Features with ``|ig| < threshold`` in either
                version are ignored (too weak to be reliable).

        Returns:
            ``{task: {feature_names_that_flipped}}`` dict.
            Also updates ``self._ig_sign_blacklist`` in-place.
        """
        flipped: Dict[str, set] = {}

        for task in set(prev_signed_ig.keys()) | set(curr_signed_ig.keys()):
            prev = prev_signed_ig.get(task, {})
            curr = curr_signed_ig.get(task, {})
            task_flipped: set = set()

            for feat in set(prev.keys()) & set(curr.keys()):
                prev_val = prev[feat]
                curr_val = curr[feat]

                # Only check features with meaningful IG in both versions
                if abs(prev_val) < flip_threshold or abs(curr_val) < flip_threshold:
                    continue

                # Sign flip: positive → negative or vice versa
                if (prev_val > 0) != (curr_val > 0):
                    task_flipped.add(feat)

            if task_flipped:
                flipped[task] = task_flipped
                # Update blacklist
                if task not in self._ig_sign_blacklist:
                    self._ig_sign_blacklist[task] = set()
                self._ig_sign_blacklist[task] |= task_flipped

                logger.warning(
                    "IG sign flipped for %d features in task '%s': %s. "
                    "IG auto-interpretation disabled for these (falling back to L3/L2/L1).",
                    len(task_flipped), task,
                    list(task_flipped)[:5],
                )

        return flipped

    def load_feature_medians(
        self, stats: Dict[str, Dict[str, float]],
    ) -> None:
        """Load per-feature medians from DatasetRegistry feature_stats.

        Args:
            stats: ``{feature_name: {"mean": ..., "std": ..., "null_pct": ...}}``
                from :meth:`DatasetRegistry.register` output.
                Uses ``mean`` as proxy for median when median is not available.
        """
        for feat_name, feat_stats in stats.items():
            # Prefer median if available, fall back to mean
            median = feat_stats.get("median", feat_stats.get("mean", 0.5))
            self._feature_medians[feat_name] = float(median)

        logger.info(
            "Loaded %d feature medians for high/low classification",
            len(self._feature_medians),
        )

    def interpret_batch(
        self,
        features: List[Tuple[str, float]],
        task: str = "",
        signed_ig_values: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Interpret a batch of (feature_name, value) pairs.

        When ``signed_ig_values`` is provided (from
        :meth:`FeatureSelector.get_signed_ig`), the IG sign is used for
        automatic direction-aware interpretation, eliminating the need
        for most manual Level 2/3 entries.

        Args:
            features: List of ``(feature_name, value)`` tuples.
            task: Task name for task-specific interpretation.
            signed_ig_values: Optional dict of ``{feature_name: signed_ig_float}``.
                Positive = feature ↑ → task prediction ↑.

        Returns:
            List of dicts with ``name``, ``value``, ``text``, ``level``,
            ``feature_group``, and optionally ``ig_direction``.
        """
        ig_map = signed_ig_values or {}
        results = []
        for feat_name, feat_value in features:
            sig_ig = ig_map.get(feat_name)
            text = self.interpret(feat_name, feat_value, task, signed_ig=sig_ig)
            level = self._resolve_level(feat_name, task, signed_ig=sig_ig)
            entry = {
                "name": feat_name,
                "value": feat_value,
                "text": text,
                "level": level,
                "feature_group": self._resolve_feature_group(feat_name) or "unknown",
            }
            if sig_ig is not None:
                entry["ig_direction"] = "positive" if sig_ig > 0 else "negative"
                entry["ig_value"] = round(sig_ig, 4)
            results.append(entry)
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
                            "direction": feat_def.get("direction", ""),
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

    def _resolve_level(
        self, feature_name: str, task: str, signed_ig: Optional[float] = None,
    ) -> str:
        """Determine which level resolved for diagnostics."""
        if not task:
            return "fallback"
        if signed_ig is not None:
            entry = self._glossary.get(feature_name)
            task_sem = self._TASK_PREDICTION_SEMANTICS.get(task)
            if entry and task_sem:
                return "IG"
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
