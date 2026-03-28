"""MCC (Merchant Category Code) hierarchy lookup.

Loads configs/mcc_hierarchy.yaml and provides:
- mcc_to_l1(code) -> L1 category index (0-9)
- mcc_to_l2(code) -> L2 sub-category index (0-29)
- Vectorized versions for DuckDB/numpy arrays
"""
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Standard MCC range-based fallback (when code not in YAML)
_MCC_RANGE_L1 = [
    (1500, 2999, "home_construction"),
    (3000, 3299, "travel_entertainment"),  # airlines
    (3300, 3499, "travel_entertainment"),  # car rental
    (3500, 3999, "travel_entertainment"),  # hotels
    (4000, 4799, "transportation"),
    (4800, 4999, "services"),  # utilities/telecom
    (5000, 5599, "retail"),
    (5600, 5699, "retail"),  # clothing
    (5700, 5799, "retail"),  # home furnishings
    (5800, 5899, "food_beverage"),  # restaurants
    (5900, 5999, "retail"),  # drug stores etc
    (6000, 6999, "financial"),
    (7000, 7299, "services"),
    (7300, 7529, "services"),
    (7530, 7549, "services"),  # auto repair
    (7600, 7999, "entertainment"),
    (8000, 8999, "services"),  # professional
    (9000, 9999, "government"),
]


@lru_cache(maxsize=1)
def _load_hierarchy() -> Dict[int, Tuple[str, str]]:
    """Load MCC hierarchy from YAML config."""
    yaml_path = Path(__file__).resolve().parents[2] / "configs" / "mcc_hierarchy.yaml"
    if not yaml_path.exists():
        logger.warning("mcc_hierarchy.yaml not found at %s", yaml_path)
        return {}

    import yaml
    with open(yaml_path, encoding="utf-8") as f:
        h = yaml.safe_load(f)

    lookup = {}
    for l1_name, l2_dict in h.get("hierarchy", {}).items():
        for l2_name, spec in l2_dict.items():
            for code in spec.get("codes", []):
                lookup[code] = (l1_name, l2_name)
    return lookup


@lru_cache(maxsize=1)
def get_l1_categories() -> List[str]:
    """Return ordered L1 category names."""
    return [
        "travel_entertainment", "food_beverage", "retail",
        "transportation", "services", "financial",
        "entertainment", "home_construction", "government", "education",
    ]


@lru_cache(maxsize=1)
def get_l2_categories() -> List[str]:
    """Return ordered L2 sub-category names."""
    lookup = _load_hierarchy()
    l2s = sorted(set(v[1] for v in lookup.values()))
    return l2s


def _fallback_l1(mcc: int) -> str:
    """Range-based L1 lookup when code not in YAML."""
    for lo, hi, l1 in _MCC_RANGE_L1:
        if lo <= mcc <= hi:
            return l1
    return "other"


def mcc_to_l1(code: int) -> int:
    """Map MCC code to L1 category index (0-based)."""
    lookup = _load_hierarchy()
    l1_cats = get_l1_categories()
    if code in lookup:
        l1_name = lookup[code][0]
    else:
        l1_name = _fallback_l1(code)
    try:
        return l1_cats.index(l1_name)
    except ValueError:
        return len(l1_cats) - 1  # "other" or last


def mcc_to_l2(code: int) -> int:
    """Map MCC code to L2 sub-category index (0-based)."""
    lookup = _load_hierarchy()
    l2_cats = get_l2_categories()
    if code in lookup:
        l2_name = lookup[code][1]
        try:
            return l2_cats.index(l2_name)
        except ValueError:
            return 0
    return 0


def vectorized_l1(mcc_array: np.ndarray) -> np.ndarray:
    """Vectorized MCC -> L1 index mapping for numpy arrays."""
    lookup = _load_hierarchy()
    l1_cats = get_l1_categories()
    l1_map = {}
    for code, (l1_name, _) in lookup.items():
        try:
            l1_map[code] = l1_cats.index(l1_name)
        except ValueError:
            l1_map[code] = len(l1_cats) - 1

    result = np.zeros(len(mcc_array), dtype=np.int32)
    for i, mcc in enumerate(mcc_array):
        if mcc in l1_map:
            result[i] = l1_map[mcc]
        else:
            result[i] = l1_cats.index(_fallback_l1(int(mcc)))
    return result


def build_duckdb_case_sql(column: str = "mcc", level: str = "l1") -> str:
    """Generate DuckDB CASE WHEN SQL for MCC -> L1/L2 mapping.

    Returns SQL expression that can be used in SELECT/WHERE clauses.
    """
    lookup = _load_hierarchy()
    cats = get_l1_categories() if level == "l1" else get_l2_categories()

    groups = {}
    for code, (l1, l2) in lookup.items():
        key = l1 if level == "l1" else l2
        groups.setdefault(key, []).append(code)

    cases = []
    for cat_name, codes in groups.items():
        try:
            idx = cats.index(cat_name)
        except ValueError:
            continue
        code_list = ", ".join(str(c) for c in sorted(codes))
        cases.append(f"WHEN {column} IN ({code_list}) THEN {idx}")

    # Range-based fallback for uncovered codes
    if level == "l1":
        for lo, hi, l1_name in _MCC_RANGE_L1:
            try:
                idx = cats.index(l1_name)
            except ValueError:
                continue
            cases.append(f"WHEN {column} BETWEEN {lo} AND {hi} THEN {idx}")

    return f"CASE {' '.join(cases)} ELSE 0 END"
