"""Serving-side Integrated-Gradients artifact helpers (#14 설명가능성).

``run_full`` Stage 8.5 computes per-task Integrated-Gradients feature
importances as ``{feature_index: mean_abs_attribution}``. This module converts
that into a serving-friendly, name-keyed top-feature map and loads it back at
serving time, so :class:`RecommendationService` can surface *model-faithful*
attribution instead of the Layer-3 rule proxy.

Artifact format (``ig_top_features.json`` in the model/analysis bundle)::

    {"task_a": [["income", 0.41], ["dsr", 0.22], ...], ...}
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

IG_TOP_FEATURES_FILE = "ig_top_features.json"


def build_serving_ig_top_features(
    ig_results: Dict[str, Any],
    feature_names: Optional[List[str]] = None,
    top_n: int = 10,
) -> Dict[str, List[List[Any]]]:
    """Convert Stage 8.5 IG output into a name-keyed top-feature map.

    Args:
        ig_results: ``{task_name: {feature_index: importance}}`` as produced by
            ``IntegratedGradients.feature_importance``. Entries that are not a
            mapping (e.g. ``{"error": ...}``) are skipped.
        feature_names: Optional ordered feature-name list for index→name
            resolution. When absent (or an index is out of range) a stable
            ``feature_{index}`` name is used.
        top_n: Number of top features to keep per task.

    Returns:
        ``{task_name: [[name, value], ...]}`` sorted by descending importance.
    """
    out: Dict[str, List[List[Any]]] = {}
    for task_name, importance in (ig_results or {}).items():
        if not isinstance(importance, dict):
            continue
        ranked: List[List[Any]] = []
        for idx, value in importance.items():
            try:
                fval = float(value)
            except (TypeError, ValueError):
                continue
            try:
                i = int(idx)
            except (TypeError, ValueError):
                name = str(idx)
            else:
                if feature_names is not None and 0 <= i < len(feature_names):
                    name = feature_names[i]
                else:
                    name = f"feature_{i}"
            ranked.append([name, fval])
        ranked.sort(key=lambda e: -e[1])
        if ranked:
            out[task_name] = ranked[:top_n]
    return out


def load_ig_attributions(bundle_dir: Any) -> Optional[Dict[str, List]]:
    """Load ``ig_top_features.json`` from a model/analysis bundle directory.

    Returns the parsed ``{task: [[name, value], ...]}`` map, or ``None`` when
    the artifact is absent or unreadable (serving then falls back to the
    Layer-3 rule proxy — non-fatal).
    """
    if not bundle_dir:
        return None
    base = os.fspath(bundle_dir)
    candidates = [
        os.path.join(base, IG_TOP_FEATURES_FILE),
        os.path.join(base, "analysis", IG_TOP_FEATURES_FILE),
    ]
    path = next((c for c in candidates if os.path.exists(c)), None)
    if path is None:
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
        logger.warning("ig_artifacts: %s is not a dict; ignoring", path)
    except Exception as e:  # noqa: BLE001 - serving must not fail on this
        logger.warning("ig_artifacts: failed to load %s: %s", path, e)
    return None
