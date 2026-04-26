"""
Config-driven Label Derivation Engine
======================================

Generic, YAML-driven label deriver.  Reads the ``labels`` section from
pipeline.yaml and applies derivation rules declaratively — no dataset-
specific hardcoding required.

Supported derivation types
--------------------------
* **direct**               — copy an existing column as-is
* **bucket**               — bin a numeric column using explicit boundaries
* **string_map**           — map string values to integers via a dict
* **list_first**           — first element of a list column
* **list_length**          — length of a list column
* **list_intersect**       — 1 if any of ``indices`` appear in a list column
  (alias: ``nba_group_check``)
* **weighted_sum**         — weighted combination of multiple columns
* **sequence_last**        — last element of a sequence, with optional top-k
  remapping
* **sequence_diversity_trend** — unique-count ratio between halves of a
  sequence
* **sequence_mode_shift**  — 1 if the mode changed between halves

Legacy aliases (for backward compatibility):
  ``first_from_list`` → ``list_first``,
  ``len_of_list``     → ``list_length``,
  ``nba_group_check`` → ``list_intersect``

Usage::

    from core.pipeline.label_deriver import LabelDeriver

    # --- Option A: pass YAML labels dict directly ---
    deriver = LabelDeriver()
    labels_df = deriver.derive(df, label_configs=cfg["labels"])

    # --- Option B: pass full pipeline config (runner style) ---
    deriver = LabelDeriver(config)
    labels_df = deriver.derive(df)
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

try:
    import duckdb
    _HAS_DUCKDB = True
except ImportError:
    _HAS_DUCKDB = False

logger = logging.getLogger(__name__)

__all__ = ["LabelDeriver", "LabelConfig"]


# ============================================================================
# Pure derivation functions — each takes (df, cfg_dict) → pd.Series
# ============================================================================

def _derive_direct(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.Series:
    """Copy an existing column directly."""
    col = cfg["source"]
    if col not in df.columns:
        logger.warning("direct: source column '%s' not found, filling NaN", col)
        return pd.Series(np.nan, index=df.index)
    return df[col].copy()


def _derive_bucket(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.Series:
    """Bin a numeric column using explicit boundaries.

    Config keys:
        source (str): column name
        boundaries (list[float]): cut-points
        sentinel (optional): value to treat as NaN
        sentinel_class (optional int): class to assign to sentinel rows
    """
    col = df[cfg["source"]].copy().astype(float)

    # Handle sentinel values (e.g. -999999 for unknown)
    sentinel = cfg.get("sentinel") or cfg.get("sentinel_value")
    if sentinel is not None:
        col = col.replace(float(sentinel), np.nan)

    boundaries = cfg["boundaries"]
    result = pd.cut(
        col,
        bins=[-np.inf] + list(boundaries) + [np.inf],
        labels=False,
    )

    sentinel_class = cfg.get("sentinel_class")
    if sentinel_class is not None:
        result = result.fillna(int(sentinel_class))
    else:
        result = result.fillna(0)

    return result.astype(int)


def _derive_string_map(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.Series:
    """Map string column values to integers via an explicit mapping dict.

    Config keys:
        source (str): column name
        mapping (dict): {string_value: int_class}
    """
    mapping = cfg["mapping"]
    col = cfg["source"]
    if col not in df.columns:
        logger.warning("string_map: source '%s' not found, filling 0", col)
        return pd.Series(0, index=df.index)

    # If already numeric (pre-encoded), clip to max class
    try:
        is_numeric = np.issubdtype(df[col].dtype, np.number)
    except TypeError:
        is_numeric = False
    if is_numeric:
        max_class = max(mapping.values())
        return df[col].clip(upper=max_class).astype(int)

    default_class = max(mapping.values())  # unknown → highest class
    return df[col].apply(lambda x: mapping.get(str(x), default_class))


def _derive_list_first(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.Series:
    """First element of a list column, or *default* if empty/missing.

    Config keys:
        source (str): column containing lists
        default (int): fallback value (default -1)
    """
    col = cfg["source"]
    default = cfg.get("default", -1)
    if col not in df.columns:
        return pd.Series(default, index=df.index)

    def _extract(x):
        if isinstance(x, (list, np.ndarray)) and len(x) > 0:
            return int(x[0])
        return default

    return df[col].apply(_extract)


def _derive_list_length(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.Series:
    """Length of a list column.

    Config keys:
        source (str): column containing lists
    """
    col = cfg["source"]
    if col not in df.columns:
        return pd.Series(0, index=df.index)

    return df[col].apply(
        lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0
    )


def _derive_list_intersect(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.Series:
    """1 if *any* of ``indices`` appear in the list column, else 0.

    Config keys:
        source (str): column containing lists
        indices (list[int]): target indices to check
    """
    col = cfg["source"]
    indices = set(cfg["indices"])
    if col not in df.columns:
        return pd.Series(0, index=df.index)

    def _check(x):
        if isinstance(x, (list, np.ndarray)) and len(x) > 0:
            return 1 if indices.intersection(int(i) for i in x) else 0
        return 0

    return df[col].apply(_check)


def _derive_weighted_sum(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.Series:
    """Weighted sum of multiple columns, with optional normalisation.

    Config keys:
        source (list[str]) or sources (list[str]): column names
        weights (list[float]): per-column weights
        normalize (bool): if True, normalise each column to [0,1] first
    """
    columns = cfg.get("source") or cfg.get("sources") or cfg.get("columns", [])
    weights = cfg["weights"]
    normalize = cfg.get("normalize", False)

    score = np.zeros(len(df))
    for col_name, weight in zip(columns, weights):
        if col_name not in df.columns:
            logger.warning("weighted_sum: column '%s' not found, skipping", col_name)
            continue
        vals = np.where(np.isnan(df[col_name].values.astype(float)), 0.0,
                        df[col_name].values.astype(float))
        if normalize:
            max_val = vals.max()
            if max_val > 0:
                vals = vals / max_val
        score += weight * vals

    return pd.Series(score, index=df.index)


def _derive_sequence_last(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.Series:
    """Last element of a sequence column, with optional top-k remapping.

    If ``top_k`` (or ``cap``) is set, the most-frequent *top_k* values are
    mapped to indices 0..top_k-1 and everything else maps to ``default``.

    Config keys:
        source (str): column containing sequences (lists)
        top_k / cap (int, optional): remap to top-K vocabulary
        default (int): fallback value (default -1)
    """
    col = cfg["source"]
    default = cfg.get("default", -1)
    top_k = cfg.get("top_k") or cfg.get("cap")

    if col not in df.columns:
        return pd.Series(default, index=df.index)

    # Extract last element
    raw_last = df[col].apply(
        lambda x: int(x[-1]) if isinstance(x, (list, np.ndarray)) and len(x) > 0 else None
    )

    if top_k is not None:
        # Build top-K vocabulary from last elements
        counts = raw_last.dropna().astype(int).value_counts()
        top_values = list(counts.index[:int(top_k)])
        val_to_idx = {v: idx for idx, v in enumerate(top_values)}
        logger.info(
            "sequence_last(%s): built top-%d vocabulary from %d unique values",
            col, top_k, len(counts),
        )
        return raw_last.apply(
            lambda x: val_to_idx.get(int(x), default) if x is not None else default
        )

    return raw_last.fillna(default).astype(int)


def _derive_sequence_diversity_trend(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.Series:
    """Diversity trend: unique(second_half) / unique(first_half) - 1.0.

    Config keys:
        source (str): column containing sequences
    """
    col = cfg["source"]
    if col not in df.columns:
        return pd.Series(0.0, index=df.index)

    def _trend(x):
        if not isinstance(x, (list, np.ndarray)) or len(x) < 4:
            return 0.0
        mid = len(x) // 2
        first_unique = len(set(x[:mid]))
        second_unique = len(set(x[mid:]))
        if first_unique == 0:
            return 0.0
        return (second_unique / first_unique) - 1.0

    return df[col].apply(_trend)


def _derive_sequence_mode_shift(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.Series:
    """1 if the mode of the sequence changed between first and second half.

    Config keys:
        source (str): column containing sequences
    """
    col = cfg["source"]
    if col not in df.columns:
        return pd.Series(0, index=df.index)

    def _shift(x):
        if not isinstance(x, (list, np.ndarray)) or len(x) < 4:
            return 0
        mid = len(x) // 2
        mode_first = Counter(x[:mid]).most_common(1)[0][0]
        mode_second = Counter(x[mid:]).most_common(1)[0][0]
        return int(mode_first != mode_second)

    return df[col].apply(_shift)


# ============================================================================
# DuckDB-accelerated derivation functions
# ============================================================================

def _derive_direct_duckdb(con, table: str, cfg: Dict[str, Any]) -> pd.Series:
    """Copy an existing column directly via DuckDB."""
    col = cfg["source"]
    sql = f'SELECT "{col}" AS result FROM {table}'
    return con.execute(sql).df()["result"]


def _derive_weighted_sum_duckdb(con, table: str, cfg: Dict[str, Any]) -> pd.Series:
    """Weighted sum of multiple columns via DuckDB SQL."""
    columns = cfg.get("source") or cfg.get("sources") or cfg.get("columns", [])
    weights = cfg["weights"]
    normalize = cfg.get("normalize", False)

    if not columns:
        return None

    # Check which columns actually exist
    available = set(con.execute(f"SELECT * FROM {table} LIMIT 0").df().columns)
    terms = []
    for col_name, weight in zip(columns, weights):
        if col_name not in available:
            logger.warning("weighted_sum: column '%s' not found, skipping", col_name)
            continue
        coalesced = f'COALESCE("{col_name}"::DOUBLE, 0.0)'
        if normalize:
            # Normalize to [0,1] by dividing by MAX; use a subquery
            terms.append(
                f'{weight} * (CASE WHEN (SELECT MAX("{col_name}"::DOUBLE) FROM {table}) > 0 '
                f'THEN {coalesced} / (SELECT MAX("{col_name}"::DOUBLE) FROM {table}) '
                f'ELSE 0.0 END)'
            )
        else:
            terms.append(f'{weight} * {coalesced}')

    if not terms:
        return None

    expr = " + ".join(terms)
    sql = f"SELECT ({expr})::DOUBLE AS result FROM {table}"
    return con.execute(sql).df()["result"]


def _derive_bucket_duckdb(con, table: str, cfg: Dict[str, Any]) -> pd.Series:
    """Bin a numeric column using CASE WHEN instead of pd.cut()."""
    col = cfg["source"]
    boundaries = cfg["boundaries"]
    sentinel = cfg.get("sentinel") or cfg.get("sentinel_value")
    sentinel_class = cfg.get("sentinel_class")

    # Build CASE WHEN for boundaries: bucket 0 = < boundaries[0], etc.
    cases = []
    for i, b in enumerate(boundaries):
        cases.append(f'WHEN val < {b} THEN {i}')
    cases.append(f'ELSE {len(boundaries)}')
    case_sql = " ".join(cases)

    if sentinel is not None:
        fill = int(sentinel_class) if sentinel_class is not None else 0
        sql = f"""
            WITH src AS (
                SELECT CASE WHEN "{col}"::DOUBLE = {float(sentinel)}
                            THEN NULL
                            ELSE "{col}"::DOUBLE END AS val
                FROM {table}
            )
            SELECT COALESCE(CASE {case_sql} END, {fill})::INTEGER AS result
            FROM src
        """
    else:
        fill = int(sentinel_class) if sentinel_class is not None else 0
        sql = f"""
            WITH src AS (
                SELECT "{col}"::DOUBLE AS val FROM {table}
            )
            SELECT COALESCE(CASE {case_sql} END, {fill})::INTEGER AS result
            FROM src
        """

    return con.execute(sql).df()["result"]


def _derive_string_map_duckdb(con, table: str, cfg: Dict[str, Any]) -> pd.Series:
    """Map string values to integers via CASE WHEN."""
    col = cfg["source"]
    mapping = cfg["mapping"]
    default_class = max(mapping.values())

    # Check if column is numeric — fall back to pandas for that path
    dtype_sql = f"""
        SELECT typeof("{col}") AS t FROM {table} LIMIT 1
    """
    try:
        col_type = con.execute(dtype_sql).fetchone()[0]
    except Exception:
        col_type = "VARCHAR"

    if col_type in ("INTEGER", "BIGINT", "FLOAT", "DOUBLE", "HUGEINT",
                     "SMALLINT", "TINYINT", "DECIMAL"):
        max_class = max(mapping.values())
        sql = f"""
            SELECT LEAST("{col}"::INTEGER, {max_class})::INTEGER AS result
            FROM {table}
        """
        return con.execute(sql).df()["result"]

    cases = " ".join(
        f"WHEN \"{col}\"::VARCHAR = '{k}' THEN {v}" for k, v in mapping.items()
    )
    sql = f"""
        SELECT CASE {cases} ELSE {default_class} END::INTEGER AS result
        FROM {table}
    """
    return con.execute(sql).df()["result"]


def _derive_list_first_duckdb(con, table: str, cfg: Dict[str, Any]) -> pd.Series:
    """First element of a list column via DuckDB list indexing."""
    col = cfg["source"]
    default = cfg.get("default", -1)
    sql = f"""
        SELECT CASE WHEN "{col}" IS NOT NULL AND len("{col}") > 0
                    THEN "{col}"[1]::INTEGER
                    ELSE {default} END AS result
        FROM {table}
    """
    return con.execute(sql).df()["result"]


def _derive_list_length_duckdb(con, table: str, cfg: Dict[str, Any]) -> pd.Series:
    """Length of a list column via DuckDB len()."""
    col = cfg["source"]
    sql = f"""
        SELECT COALESCE(len("{col}"), 0)::INTEGER AS result
        FROM {table}
    """
    return con.execute(sql).df()["result"]


def _derive_list_intersect_duckdb(con, table: str, cfg: Dict[str, Any]) -> pd.Series:
    """Check if any of `indices` appear in the list column."""
    col = cfg["source"]
    indices = cfg["indices"]
    checks = " OR ".join(f'list_contains("{col}", {idx})' for idx in indices)
    sql = f"""
        SELECT CASE WHEN "{col}" IS NOT NULL AND len("{col}") > 0
                         AND ({checks})
                    THEN 1 ELSE 0 END::INTEGER AS result
        FROM {table}
    """
    return con.execute(sql).df()["result"]


def _derive_sequence_last_duckdb(con, table: str, cfg: Dict[str, Any]) -> pd.Series:
    """Last element of a sequence column via DuckDB negative indexing."""
    col = cfg["source"]
    default = cfg.get("default", -1)
    top_k = cfg.get("top_k") or cfg.get("cap")

    if top_k is None:
        sql = f"""
            SELECT COALESCE(
                CASE WHEN "{col}" IS NOT NULL AND len("{col}") > 0
                     THEN "{col}"[len("{col}")]::INTEGER
                     ELSE NULL END,
                {default}
            )::INTEGER AS result
            FROM {table}
        """
        return con.execute(sql).df()["result"]

    # top_k remapping: build vocabulary entirely in DuckDB
    vocab_sql = f"""
        SELECT raw_last, COUNT(*) AS cnt FROM (
            SELECT CASE WHEN "{col}" IS NOT NULL AND len("{col}") > 0
                        THEN "{col}"[len("{col}")]::INTEGER
                        ELSE NULL END AS raw_last
            FROM {table}
        ) sub
        WHERE raw_last IS NOT NULL
        GROUP BY raw_last
        ORDER BY cnt DESC
        LIMIT {int(top_k)}
    """
    vocab_rows = con.execute(vocab_sql).fetchall()
    n_unique_sql = f"""
        SELECT COUNT(DISTINCT raw_last) AS n FROM (
            SELECT CASE WHEN "{col}" IS NOT NULL AND len("{col}") > 0
                        THEN "{col}"[len("{col}")]::INTEGER
                        ELSE NULL END AS raw_last
            FROM {table}
        ) sub WHERE raw_last IS NOT NULL
    """
    n_unique = con.execute(n_unique_sql).fetchone()[0]
    top_values = [row[0] for row in vocab_rows]
    logger.info(
        "sequence_last(%s): built top-%d vocabulary from %d unique values",
        col, top_k, n_unique,
    )

    if not top_values:
        n_rows = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return pd.Series(default, index=range(n_rows))

    # Build CASE WHEN for remapping via SQL against the original table
    cases = " ".join(
        f"WHEN raw_last = {v} THEN {idx}" for idx, v in enumerate(top_values)
    )
    remap_sql = f"""
        SELECT CASE {cases} ELSE {default} END::INTEGER AS result
        FROM (
            SELECT CASE WHEN "{col}" IS NOT NULL AND len("{col}") > 0
                        THEN "{col}"[len("{col}")]::INTEGER
                        ELSE NULL END AS raw_last
            FROM {table}
        ) sub
    """
    return con.execute(remap_sql).df()["result"]


def _derive_sequence_diversity_trend_duckdb(
    con, table: str, cfg: Dict[str, Any]
) -> pd.Series:
    """Diversity trend via DuckDB list slicing and list_distinct.

    unique(second_half) / unique(first_half) - 1.0
    """
    col = cfg["source"]
    # DuckDB list slicing: list[start:end] is 1-based inclusive
    # first_half = list[1 : mid], second_half = list[mid+1 : ]
    sql = f"""
        SELECT CASE
            WHEN "{col}" IS NULL OR len("{col}") < 4 THEN 0.0
            ELSE (
                len(list_distinct(list_slice("{col}", (len("{col}") / 2) + 1, len("{col}"))))::DOUBLE
                / GREATEST(len(list_distinct(list_slice("{col}", 1, len("{col}") / 2)))::DOUBLE, 1.0)
                - 1.0
            )
        END AS result
        FROM {table}
    """
    return con.execute(sql).df()["result"]


def _derive_sequence_mode_shift_duckdb(
    con, table: str, cfg: Dict[str, Any]
) -> pd.Series:
    """Mode shift between halves — DuckDB list + mode() via lateral unnest.

    Uses LATERAL UNNEST to compute the mode of each half per row,
    then compares them.
    """
    col = cfg["source"]
    sql = f"""
        WITH numbered AS (
            SELECT row_number() OVER () AS _rid, "{col}" AS seq
            FROM {table}
        ),
        mode_first AS (
            SELECT n._rid, u.val AS mode_val
            FROM numbered n, LATERAL (
                SELECT unnest(list_slice(n.seq, 1, len(n.seq) / 2)) AS val
            ) u
            WHERE n.seq IS NOT NULL AND len(n.seq) >= 4
            GROUP BY n._rid, u.val
            QUALIFY row_number() OVER (
                PARTITION BY n._rid ORDER BY count(*) DESC, u.val
            ) = 1
        ),
        mode_second AS (
            SELECT n._rid, u.val AS mode_val
            FROM numbered n, LATERAL (
                SELECT unnest(list_slice(n.seq, (len(n.seq) / 2) + 1, len(n.seq))) AS val
            ) u
            WHERE n.seq IS NOT NULL AND len(n.seq) >= 4
            GROUP BY n._rid, u.val
            QUALIFY row_number() OVER (
                PARTITION BY n._rid ORDER BY count(*) DESC, u.val
            ) = 1
        )
        SELECT CASE
            WHEN mf.mode_val IS NULL OR ms.mode_val IS NULL THEN 0
            WHEN mf.mode_val != ms.mode_val THEN 1
            ELSE 0
        END::INTEGER AS result
        FROM numbered n
        LEFT JOIN mode_first mf ON n._rid = mf._rid
        LEFT JOIN mode_second ms ON n._rid = ms._rid
        ORDER BY n._rid
    """
    return con.execute(sql).df()["result"]


# Map of DuckDB-accelerated derivation types
_DUCKDB_DERIVE_METHODS = {
    "direct": _derive_direct_duckdb,
    "bucket": _derive_bucket_duckdb,
    "string_map": _derive_string_map_duckdb,
    "list_first": _derive_list_first_duckdb,
    "list_length": _derive_list_length_duckdb,
    "list_intersect": _derive_list_intersect_duckdb,
    "weighted_sum": _derive_weighted_sum_duckdb,
    "sequence_last": _derive_sequence_last_duckdb,
    "sequence_diversity_trend": _derive_sequence_diversity_trend_duckdb,
    "sequence_mode_shift": _derive_sequence_mode_shift_duckdb,

    # Legacy aliases
    "first_from_list": _derive_list_first_duckdb,
    "len_of_list": _derive_list_length_duckdb,
    "nba_group_check": _derive_list_intersect_duckdb,
}


# ============================================================================
# Dispatch table — maps type strings to derivation functions
# ============================================================================

_DERIVE_METHODS = {
    # Canonical names
    "direct": _derive_direct,
    "bucket": _derive_bucket,
    "string_map": _derive_string_map,
    "list_first": _derive_list_first,
    "list_length": _derive_list_length,
    "list_intersect": _derive_list_intersect,
    "weighted_sum": _derive_weighted_sum,
    "sequence_last": _derive_sequence_last,
    "sequence_diversity_trend": _derive_sequence_diversity_trend,
    "sequence_mode_shift": _derive_sequence_mode_shift,

    # Legacy aliases (backward compatibility)
    "first_from_list": _derive_list_first,
    "len_of_list": _derive_list_length,
    "nba_group_check": _derive_list_intersect,
}


# ============================================================================
# LabelConfig — kept for backward compatibility with train.py callers
# ============================================================================

class LabelConfig:
    """Thin wrapper so callers that build LabelConfig objects still work.

    Internally the engine only uses plain dicts, but this class lets old
    code like ``LabelConfig(name=..., source=..., type=...)`` keep working.
    """

    def __init__(self, *, name: str = "", **kwargs):
        self.name = name
        self._cfg = dict(kwargs)
        # Ensure 'type' key exists (old callers may pass it as the derive type)
        if "type" not in self._cfg:
            self._cfg["type"] = "direct"

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._cfg)

    # Allow attribute access for backward compat
    def __getattr__(self, key):
        if key.startswith("_") or key == "name":
            raise AttributeError(key)
        return self._cfg.get(key)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LabelConfig":
        name = d.pop("name", "")
        return cls(name=name, **d)


# ============================================================================
# LabelDeriver — the main engine
# ============================================================================

class LabelDeriver:
    """Generic label derivation engine driven by YAML config.

    Supports two calling conventions:

    1. **Explicit** (train.py style)::

           deriver = LabelDeriver()
           labels_df = deriver.derive(df, label_configs=cfg["labels"])

    2. **Config-object** (runner.py style)::

           deriver = LabelDeriver(pipeline_config)
           labels_df = deriver.derive(df)

    ``label_configs`` may be:
      - a ``dict[str, dict]`` (the ``labels:`` section from pipeline.yaml)
      - a ``list[LabelConfig]`` (legacy)
    """

    DERIVE_METHODS = _DERIVE_METHODS

    def __init__(self, config=None):
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def derive(
        self,
        df: pd.DataFrame,
        label_configs: Optional[Union[Dict[str, dict], List[LabelConfig]]] = None,
        con: Any = None,
        source_table: Optional[str] = None,
    ) -> pd.DataFrame:
        """Derive all configured label columns.

        Uses DuckDB-accelerated SQL when available (avoids row-wise
        ``.apply()`` on 941K+ rows).  Falls back to pandas per-label
        if a DuckDB derivation fails or returns None.

        Parameters
        ----------
        df : pd.DataFrame
            Raw input data.
        label_configs : dict | list[LabelConfig] | None
            Label definitions.  If *None*, falls back to ``self._config``.
        con : duckdb.DuckDBPyConnection, optional
            Caller-supplied DuckDB connection. When given together with
            ``source_table`` we run the SQL derivation paths against
            that table directly — no scratch ``duckdb.connect()``, no
            ``con.register(df)`` materialisation. CLAUDE.md §3.3.
        source_table : str, optional
            Table name on ``con`` that holds the post-Stage-3 feature
            matrix. Required when ``con`` is supplied.

        Returns
        -------
        pd.DataFrame
            One column per successfully derived label, indexed like *df*.
        """
        cfgs = self._resolve_configs(label_configs)
        results: Dict[str, pd.Series] = {}

        # ---- DuckDB connection: caller-supplied (preferred) or scratch ----
        own_con = False
        table_name = source_table or "_label_src"
        if con is None and _HAS_DUCKDB:
            try:
                con = duckdb.connect()
                con.register(table_name, df)
                own_con = True
                logger.debug("LabelDeriver: scratch DuckDB connection opened for %d rows", len(df))
            except Exception:
                logger.debug("LabelDeriver: DuckDB init failed, using pandas", exc_info=True)
                con = None

        for label_name, cfg in cfgs.items():
            derive_type = cfg.get("type", "direct")

            # Check source column exists (skip early for missing columns)
            source_col = cfg.get("source")
            if source_col and not isinstance(source_col, list) and source_col not in df.columns and derive_type != "weighted_sum":
                logger.warning(
                    "LabelDeriver: source '%s' not in df for label '%s', using default",
                    source_col, label_name,
                )
                # Let the pandas fallback handle defaults gracefully
                method = self.DERIVE_METHODS.get(derive_type)
                if method is not None:
                    try:
                        results[label_name] = method(df, cfg)
                    except Exception:
                        logger.exception(
                            "LabelDeriver: failed to derive '%s' (type=%s)",
                            label_name, derive_type,
                        )
                continue

            # Attempt DuckDB derivation
            duckdb_done = False
            if con is not None:
                duckdb_method = _DUCKDB_DERIVE_METHODS.get(derive_type)
                if duckdb_method is not None:
                    try:
                        series = duckdb_method(con, table_name, cfg)
                        if series is not None:
                            series.index = df.index
                            results[label_name] = series
                            duckdb_done = True
                            logger.debug(
                                "LabelDeriver: derived '%s' via DuckDB (type=%s, non-null=%d/%d)",
                                label_name, derive_type,
                                series.notna().sum(), len(series),
                            )
                    except Exception:
                        logger.debug(
                            "LabelDeriver: DuckDB failed for '%s' (type=%s), falling back to pandas",
                            label_name, derive_type, exc_info=True,
                        )

            # Pandas fallback
            if not duckdb_done:
                method = self.DERIVE_METHODS.get(derive_type)
                if method is None:
                    logger.warning(
                        "LabelDeriver: unknown type '%s' for label '%s', skipping",
                        derive_type, label_name,
                    )
                    continue
                try:
                    series = method(df, cfg)
                    results[label_name] = series
                    logger.debug(
                        "LabelDeriver: derived '%s' via pandas (type=%s, non-null=%d/%d)",
                        label_name, derive_type,
                        series.notna().sum(), len(series),
                    )
                except Exception:
                    logger.exception(
                        "LabelDeriver: failed to derive '%s' (type=%s)",
                        label_name, derive_type,
                    )

        # Cleanup DuckDB — only the scratch connection we owned
        if con is not None and own_con:
            try:
                con.unregister(table_name)
                con.close()
            except Exception:
                pass

        out = pd.DataFrame(results, index=df.index)
        logger.info(
            "LabelDeriver: derived %d/%d labels: %s",
            len(results), len(cfgs), sorted(results.keys()),
        )
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_configs(
        self,
        label_configs: Optional[Union[Dict[str, dict], List[LabelConfig]]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Normalise whatever the caller passed into ``{name: cfg_dict}``."""

        # 1. Explicit dict (preferred path)
        if isinstance(label_configs, dict):
            return {
                name: (cfg if isinstance(cfg, dict) else {"type": "direct", "source": name})
                for name, cfg in label_configs.items()
            }

        # 2. List of LabelConfig objects (legacy train.py path)
        if isinstance(label_configs, (list, tuple)):
            out = {}
            for lc in label_configs:
                if isinstance(lc, LabelConfig):
                    out[lc.name] = lc.to_dict()
                elif isinstance(lc, dict):
                    name = lc.pop("name", lc.get("label_col", "unknown"))
                    out[name] = lc
                else:
                    logger.warning("LabelDeriver: skipping unrecognised config: %s", lc)
            return out

        # 3. Fall back to self._config (runner.py path)
        if self._config is not None:
            # Try config.labels (dict attribute)
            if hasattr(self._config, "labels") and isinstance(self._config.labels, dict):
                return self._config.labels
            # Try config["labels"]
            if hasattr(self._config, "__getitem__"):
                try:
                    labels = self._config["labels"]
                    if isinstance(labels, dict):
                        return labels
                except (KeyError, TypeError):
                    pass
            # Try extracting from config.raw (raw YAML dict)
            raw = getattr(self._config, "raw", None) or getattr(self._config, "_raw", None)
            if isinstance(raw, dict) and "labels" in raw:
                return raw["labels"]

        raise ValueError(
            "LabelDeriver: no label configs provided and none found in "
            "pipeline config.  Pass label_configs= or init with config."
        )
