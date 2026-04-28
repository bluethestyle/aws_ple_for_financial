"""3-stage feature normalization pipeline.

Matches the on-prem design with proper train-only fitting and
power-law raw copies that are NOT scaled.

Stages
------
1. **Amount-column log1p pre-transform** — applied in-place to
   power-law columns before scaling so the scaler sees the log-space
   values (this is handled implicitly: we detect power-law columns,
   and their raw values are used to create unscaled log copies).
2. **Z-score normalization** — mean/std computed on *training split
   only* using CuPy (GPU) when available, numpy otherwise.  Applied
   to continuous columns.  Binary columns are passed through as-is.
3. **Power-law raw copies** — ``log1p`` of the original (pre-scaled)
   values, appended as ``{col}_log`` columns.  These are **never**
   scaled so the model can see raw magnitude.

Output column order: ``[scaled_continuous | binary | power_law_log_copies]``
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import cupy as cp  # GPU-accelerated array ops

    _HAS_CUPY = True
except ImportError:  # pragma: no cover
    cp = None
    _HAS_CUPY = False

logger = logging.getLogger(__name__)


def _xp():
    """Return CuPy if available, else numpy."""
    return cp if _HAS_CUPY else np


def _to_numpy(arr) -> np.ndarray:
    """Ensure *arr* is a plain numpy array (move off GPU if needed)."""
    if _HAS_CUPY and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)

__all__ = ["FeatureNormalizer"]


class FeatureNormalizer:
    """3-stage normalization pipeline.

    Usage::

        normalizer = FeatureNormalizer()
        normalizer.fit(train_df, feature_cols)

        train_normed = normalizer.transform(train_df, feature_cols)
        val_normed   = normalizer.transform(val_df, feature_cols)
        test_normed  = normalizer.transform(test_df, feature_cols)

    The returned DataFrames have columns in a deterministic order:
    ``[scaled_continuous | binary | power_law_log_copies]``.
    """

    # Power-law detection thresholds (2-stage: fast filter + log-log R²)
    # These are class-level defaults; instance-level values come from config.
    SKEW_THRESH: float = 2.0
    KURT_THRESH: float = 6.0
    R2_THRESH: float = 0.9

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        self.config = cfg
        # Allow config overrides for detection thresholds
        self.SKEW_THRESH = cfg.get("skew_threshold", self.__class__.SKEW_THRESH)
        self.KURT_THRESH = cfg.get("kurtosis_threshold", self.__class__.KURT_THRESH)
        self.R2_THRESH = cfg.get("loglog_r2_threshold", self.__class__.R2_THRESH)
        self._min_nunique: int = cfg.get("min_nunique", 20)
        self._min_samples: int = cfg.get("min_samples_loglog", 50)
        self._mean: Optional[np.ndarray] = None  # per-column means
        self._std: Optional[np.ndarray] = None   # per-column stds
        self.power_law_cols: List[str] = []
        self.continuous_cols: List[str] = []
        self.binary_cols: List[str] = []
        self.categorical_int_cols: List[str] = []  # integer IDs excluded from scaler
        self.probability_cols: List[str] = []       # already 0~1, excluded from scaler
        self.power_law_details: Dict[str, Dict] = {}
        self._fitted: bool = False

        # Patterns for columns that should NOT be StandardScaled.
        # Suffix-based: legacy convention (cluster_id, state_id, ...).
        # Prefix-based: nominal-int columns produced by lag-style flatteners
        #   (e.g. txn_lag_mcc_001..200 — MCC IDs encoded as integers, must
        #   not get ordinal distance from a scaler) and cyclical-int columns
        #   (e.g. txn_lag_hour_001..200 — 0..23 wraps, scaler distorts).
        self._categorical_id_suffixes: List[str] = cfg.get(
            "categorical_id_suffixes",
            ["_cluster_id", "_state_id", "_segment_id", "_group_id"],
        )
        self._categorical_id_prefixes: List[str] = cfg.get(
            "categorical_id_prefixes",
            [],
        )
        self._probability_prefixes: List[str] = cfg.get(
            "probability_prefixes",
            ["gmm_clustering_cluster_prob_", "model_derived_gmm_prob_"],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, feature_cols: List[str]) -> "FeatureNormalizer":
        """Fit on **training data only**.

        Parameters
        ----------
        df : pd.DataFrame
            Training split DataFrame.
        feature_cols : list[str]
            Columns to normalise (numeric feature columns).

        Returns
        -------
        self
        """
        # Only consider columns actually present in df
        feature_cols = [c for c in feature_cols if c in df.columns]

        # --- Classify columns ---
        self.binary_cols = [
            c for c in feature_cols
            if set(df[c].dropna().unique()).issubset({0, 0.0, 1, 1.0})
        ]

        # Categorical integer columns (cluster_id, state_id, etc.)
        # These are nominal — scaler would impose ordinal distance semantics.
        # Match either a suffix (legacy) OR a configured prefix (lag-style
        # flatteners that emit nominal/cyclical int columns).
        self.categorical_int_cols = [
            c for c in feature_cols
            if c not in self.binary_cols
            and (any(c.endswith(sfx) for sfx in self._categorical_id_suffixes)
                 or any(c.startswith(pfx) for pfx in self._categorical_id_prefixes))
        ]

        # Probability columns already in [0, 1] — scaler destroys interpretability.
        self.probability_cols = [
            c for c in feature_cols
            if c not in self.binary_cols
            and c not in self.categorical_int_cols
            and any(c.startswith(pfx) for pfx in self._probability_prefixes)
        ]

        _exclude = set(self.binary_cols) | set(self.categorical_int_cols) | set(self.probability_cols)
        self.continuous_cols = [
            c for c in feature_cols if c not in _exclude
        ]

        if self.categorical_int_cols:
            logger.info(
                "Scaler-excluded categorical IDs: %d cols %s",
                len(self.categorical_int_cols), self.categorical_int_cols,
            )
        if self.probability_cols:
            logger.info(
                "Scaler-excluded probabilities: %d cols",
                len(self.probability_cols),
            )

        # --- Stage 1: Detect power-law columns ---
        self.power_law_cols, self.power_law_details = self._detect_power_law(df)
        if self.power_law_cols:
            logger.info(
                "Power-law detected: %d columns confirmed (R²>%.1f): %s",
                len(self.power_law_cols),
                self.R2_THRESH,
                self.power_law_cols,
            )

        # --- Stage 2: Compute mean / std (CuPy when available) ---
        self._mean = None
        self._std = None
        if self.continuous_cols:
            xp = _xp()
            raw = df[self.continuous_cols].fillna(0).values.astype(np.float64)
            arr = xp.asarray(raw)
            self._mean = _to_numpy(xp.mean(arr, axis=0))
            std = _to_numpy(xp.std(arr, axis=0))
            # Guard against zero-variance columns (mirror sklearn behaviour)
            std[std < 1e-10] = 1.0
            self._std = std

        self._fitted = True
        logger.info(
            "FeatureNormalizer fitted: %d continuous, %d binary, %d power-law",
            len(self.continuous_cols),
            len(self.binary_cols),
            len(self.power_law_cols),
        )
        return self

    def transform(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
    ) -> pd.DataFrame:
        """Transform a split using fitted parameters.

        Call this on train, val, and test splits separately.  The scaler
        parameters come from the training fit — val/test are *not* re-fit.

        Returns a new DataFrame whose column order matches the input
        ``feature_cols`` order, followed by power-law ``_log`` copies.
        Preserves the invariant
            "feature_groups.yaml order = concat order = parquet order
             = sequential group_ranges"
        so downstream FeatureRouter slicing stays contiguous per group
        (see transform_sql() for the full rationale).
        """
        if not self._fitted:
            raise RuntimeError("FeatureNormalizer.transform() called before fit()")

        # Pre-compute the scaled continuous block once (vectorised
        # across all continuous cols), then pull individual columns
        # out of it as needed below.
        scaled_lookup: Dict[str, "pd.Series"] = {}
        if self.continuous_cols and self._mean is not None and self._std is not None:
            xp = _xp()
            raw = df[self.continuous_cols].fillna(0).values.astype(np.float64)
            arr = xp.asarray(raw)
            mean = xp.asarray(self._mean)
            std = xp.asarray(self._std)
            scaled = _to_numpy((arr - mean) / std)
            scaled_df = pd.DataFrame(
                scaled,
                columns=self.continuous_cols,
                index=df.index,
            )
            scaled_lookup = {c: scaled_df[c] for c in self.continuous_cols}

        binary_set = set(self.binary_cols)
        cat_int_set = set(self.categorical_int_cols)
        prob_set = set(self.probability_cols)

        # Build the result in input order — emit each col's transformed
        # value in place rather than concat'ing per-bucket blocks.
        cols_in_order: List["pd.Series"] = []
        col_names: List[str] = []
        for c in feature_cols:
            if c in scaled_lookup:
                cols_in_order.append(scaled_lookup[c])
                col_names.append(c)
            elif c in binary_set or c in cat_int_set or c in prob_set:
                if c in df.columns:
                    cols_in_order.append(df[c])
                    col_names.append(c)
            else:
                # Unknown bucket — pass through (see transform_sql).
                if c in df.columns:
                    cols_in_order.append(df[c])
                    col_names.append(c)

        if cols_in_order:
            result = pd.concat(cols_in_order, axis=1)
            result.columns = col_names
        else:
            result = pd.DataFrame(index=df.index)

        # Power-law log copies are NEW columns appended at the tail —
        # additions outside the original feature_cols, so they don't
        # disturb the order invariant.
        if self.power_law_cols:
            log_df = pd.DataFrame(index=df.index)
            for col in self.power_law_cols:
                log_df[f"{col}_log"] = np.log1p(
                    df[col].fillna(0).clip(lower=0)
                )
            result = pd.concat([result, log_df], axis=1)

        return result

    def fit_transform(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
    ) -> pd.DataFrame:
        """Convenience: fit on *df* then transform it."""
        return self.fit(df, feature_cols).transform(df, feature_cols)

    # ------------------------------------------------------------------
    # SQL-native API (CLAUDE.md §3.3)
    # ------------------------------------------------------------------
    # Pandas fit/transform above materialise a 1M × ~1285D matrix to
    # numpy via ``df[cols].values.astype(float64)`` — the dominant
    # Phase-0 memory hot-spot. The SQL versions below replace that with
    # DuckDB column aggregates and a SELECT-time projection: nothing
    # leaves the database except the per-column scaler parameters and
    # an output table reference.

    _NUMERIC_DUCKDB_TYPES = (
        "TINYINT", "SMALLINT", "INTEGER", "BIGINT",
        "FLOAT", "DOUBLE", "DECIMAL", "HUGEINT",
    )

    def fit_sql(
        self,
        con: Any,
        source_table: str,
        feature_cols: List[str],
        train_predicate: Optional[str] = None,
    ) -> "FeatureNormalizer":
        """SQL-native fit on a DuckDB table or view.

        Parameters
        ----------
        con : duckdb.DuckDBPyConnection
            Connection that owns ``source_table``.
        source_table : str
            Table or view that contains all of ``feature_cols``.
        feature_cols : list[str]
            Columns to consider for normalisation.
        train_predicate : str, optional
            Optional ``WHERE`` clause to limit the fit to the training
            split (e.g. ``"_split = 'train'"``). Without it the entire
            table is used.

        Notes
        -----
        Stage 1 (column classification): one ``DESCRIBE`` + one
        per-column ``COUNT(DISTINCT)`` SQL. Cheap.
        Stage 2 (power-law detection): one batched aggregate using
        DuckDB's ``SKEWNESS`` / ``KURTOSIS`` / ``MIN`` / ``COUNT(DISTINCT)``;
        then a log-log R² check on candidates only (small loop).
        Stage 3 (mean / std): single SELECT with all ``AVG`` / ``STDDEV_SAMP``
        aggregates. Returns one row → numpy-cheap.
        """
        where_sql = f"WHERE {train_predicate}" if train_predicate else ""

        # --- Stage 1: classify columns (binary / cat / probability / cont) ---
        # Numeric type filter via DESCRIBE → which feature_cols are present
        type_rows = con.execute(
            f"SELECT column_name, column_type FROM (DESCRIBE {source_table})"
        ).fetchall()
        type_map = {r[0]: (r[1] or "").upper() for r in type_rows}
        present = [c for c in feature_cols if c in type_map]

        # Binary detection: all distinct non-null values are subset of {0, 1}.
        # Done in a single SQL by counting distinct values per column.
        # Skip non-numeric types (binary lookup is meaningless on VARCHAR etc.)
        # AND skip LIST/ARRAY columns (e.g. txn_amount_seq is "DOUBLE[]" in
        # DESCRIBE; the substring 'DOUBLE' would otherwise pull it into the
        # numeric set and break the IN (0, 1) cast).
        def _is_scalar_numeric(t: str) -> bool:
            return (
                any(p in t for p in self._NUMERIC_DUCKDB_TYPES)
                and "[" not in t
                and "STRUCT" not in t
                and "MAP" not in t
            )
        numeric_present = [c for c in present if _is_scalar_numeric(type_map[c])]

        binary_cols: List[str] = []
        if numeric_present:
            # Per-column 'all values in {0,1}' check via boolean aggregate
            checks = ", ".join(
                f"BOOL_AND(\"{c}\" IS NULL OR \"{c}\" IN (0, 1)) AS \"_bin_{i}\""
                for i, c in enumerate(numeric_present)
            )
            row = con.execute(
                f"SELECT {checks} FROM {source_table} {where_sql}"
            ).fetchone() or ()
            for i, c in enumerate(numeric_present):
                if i < len(row) and bool(row[i]):
                    binary_cols.append(c)
        self.binary_cols = binary_cols

        # Categorical / probability classification (suffix / prefix only,
        # type-independent → no extra SQL needed).
        self.categorical_int_cols = [
            c for c in present
            if c not in self.binary_cols
            and (any(c.endswith(s) for s in self._categorical_id_suffixes)
                 or any(c.startswith(p) for p in self._categorical_id_prefixes))
        ]
        self.probability_cols = [
            c for c in present
            if c not in self.binary_cols
            and c not in self.categorical_int_cols
            and any(c.startswith(p) for p in self._probability_prefixes)
        ]
        excluded = set(self.binary_cols) | set(self.categorical_int_cols) | set(self.probability_cols)
        self.continuous_cols = [c for c in present if c not in excluded
                                and _is_scalar_numeric(type_map[c])]

        if self.categorical_int_cols:
            logger.info("Scaler-excluded categorical IDs: %d cols",
                        len(self.categorical_int_cols))
        if self.probability_cols:
            logger.info("Scaler-excluded probabilities: %d cols",
                        len(self.probability_cols))

        # --- Stage 2: power-law detection (DuckDB SKEWNESS / KURTOSIS) ---
        self.power_law_cols, self.power_law_details = self._detect_power_law_sql(
            con, source_table, where_sql,
        )
        if self.power_law_cols:
            logger.info(
                "Power-law detected: %d columns confirmed (R²>%.1f): %s",
                len(self.power_law_cols), self.R2_THRESH,
                self.power_law_cols[:10],
            )

        # --- Stage 3: mean / std (single SQL, one row) ---
        self._mean = None
        self._std = None
        if self.continuous_cols:
            # Match the pandas path: NULLs become 0 *for the scaler
            # statistics* so the same value substitution downstream
            # (transform_sql also wraps each continuous column in
            # COALESCE("c", 0)) produces consistent (x - mean) / std.
            # We split the aggregate across batches so a single column
            # overflow doesn't abort the whole fit.
            n = len(self.continuous_cols)
            mean = np.zeros(n, dtype=np.float64)
            std = np.ones(n, dtype=np.float64)
            BATCH = 64
            for start in range(0, n, BATCH):
                stop = min(start + BATCH, n)
                agg_terms = []
                for i in range(start, stop):
                    c = self.continuous_cols[i]
                    agg_terms.append(f'AVG(COALESCE("{c}", 0)) AS "_m{i}"')
                    agg_terms.append(f'STDDEV_SAMP(COALESCE("{c}", 0)) AS "_s{i}"')
                try:
                    row = con.execute(
                        f"SELECT {', '.join(agg_terms)} FROM {source_table} {where_sql}"
                    ).fetchone() or ()
                except Exception as exc:
                    logger.debug(
                        "mean/std batch %d-%d failed (%s), per-col fallback",
                        start, stop, exc,
                    )
                    row = None
                if row:
                    for i in range(start, stop):
                        idx = (i - start) * 2
                        if idx < len(row) and row[idx] is not None:
                            mean[i] = float(row[idx])
                        if idx + 1 < len(row) and row[idx + 1] is not None:
                            s = float(row[idx + 1])
                            std[i] = s if s > 1e-10 else 1.0
                else:
                    # per-column fallback
                    for i in range(start, stop):
                        c = self.continuous_cols[i]
                        try:
                            r = con.execute(
                                f'SELECT AVG(COALESCE("{c}", 0)), '
                                f'STDDEV_SAMP(COALESCE("{c}", 0)) '
                                f'FROM {source_table} {where_sql}'
                            ).fetchone()
                            if r:
                                if r[0] is not None:
                                    mean[i] = float(r[0])
                                if r[1] is not None:
                                    s = float(r[1])
                                    std[i] = s if s > 1e-10 else 1.0
                        except Exception as exc:
                            logger.debug("mean/std skipped for %s: %s", c, exc)
            self._mean = mean
            self._std = std

        self._fitted = True
        logger.info(
            "FeatureNormalizer.fit_sql: %d continuous, %d binary, "
            "%d categorical_int, %d probability, %d power-law",
            len(self.continuous_cols), len(self.binary_cols),
            len(self.categorical_int_cols), len(self.probability_cols),
            len(self.power_law_cols),
        )
        return self

    def transform_sql(
        self,
        con: Any,
        source_table: str,
        feature_cols: List[str],
        output_table: str,
        extra_pass_through: Optional[List[str]] = None,
    ) -> str:
        """SQL-native transform — emits ``output_table`` with the
        normalised columns plus ``extra_pass_through`` (id_cols, label
        cols, lazy LIST columns, etc.) untouched.

        Returns the output table name for chaining. The full 1M × ~1285D
        matrix never leaves DuckDB; the caller can pull only the slices
        it needs (e.g. label_df, train_idx) at the boundary downstream.
        """
        if not self._fitted:
            raise RuntimeError("FeatureNormalizer.transform_sql() called before fit/fit_sql")

        feature_cols = [c for c in feature_cols if c]
        proj: List[str] = []

        # Look up which bucket each column belongs to — these four sets
        # are mutually exclusive and together cover every column in
        # ``feature_cols`` (see fit_sql: continuous_cols is computed as
        # feature_cols MINUS the union of the other three).
        cont_idx = {c: i for i, c in enumerate(self.continuous_cols)}
        binary_set = set(self.binary_cols)
        cat_int_set = set(self.categorical_int_cols)
        prob_set = set(self.probability_cols)

        have_cont_stats = (
            self._mean is not None and self._std is not None
        )

        # Iterate ``feature_cols`` in input order so the OUTPUT table's
        # column order matches the Stage-3 concatenation order. The
        # earlier implementation emitted continuous → binary →
        # categorical_int → probability blocks regardless of input
        # order, which broke the invariant
        #   "feature_groups.yaml order = concat order = parquet order
        #    = sequential group_ranges"
        # and forced Stage 6 to call _rebuild_group_ranges_post_normalization,
        # which then degraded to ``longest contiguous block`` for any
        # group that mixed binary + continuous cols (e.g. demographics:
        # gender / is_active ended up at idx 503-504 while the rest sat
        # at idx 1-9). Preserving order eliminates the orphan-cols
        # problem at the source.
        #
        # CAST sources to DOUBLE either way so DECIMAL(P,S) inputs
        # don't constrain the output domain (a DECIMAL(18,17) source
        # can't hold any standardised value outside ±9.99…) and
        # Stage 9 features.parquet stays an even-typed numeric matrix.
        for c in feature_cols:
            if c in cont_idx and have_cont_stats:
                i = cont_idx[c]
                m = float(self._mean[i])
                s = float(self._std[i])
                proj.append(
                    f'(CAST(COALESCE("{c}", 0) AS DOUBLE) - {repr(m)}) '
                    f'/ {repr(s)} AS "{c}"'
                )
            elif c in binary_set or c in cat_int_set or c in prob_set:
                proj.append(f'CAST("{c}" AS DOUBLE) AS "{c}"')
            else:
                # Should not reach here because the fit-time bucketing
                # is exhaustive, but pass through unchanged so a
                # mismatch surfaces as data, not a missing column.
                logger.warning(
                    "FeatureNormalizer.transform_sql: column %r is in "
                    "feature_cols but not in any normalizer bucket — "
                    "passing through untransformed", c,
                )
                proj.append(f'CAST("{c}" AS DOUBLE) AS "{c}"')

        # Power-law log copies are NEW columns (not present in input),
        # so appending them at the tail does not break the order
        # invariant — they're additions, not relocations. Stage 6's
        # rebuild already attaches each ``{col}_log`` to its parent
        # group via the ``log_cols_created`` parameter.
        for c in self.power_law_cols:
            proj.append(
                f'LN(GREATEST(CAST(COALESCE("{c}", 0) AS DOUBLE), 0) + 1) '
                f'AS "{c}_log"'
            )

        # Caller-requested pass-through (ids, label cols, LIST cols)
        # — also additions outside the feature matrix proper, so the
        # tail position is fine.
        if extra_pass_through:
            seen = {p.split(' AS ')[-1].strip(' "') for p in proj}
            for c in extra_pass_through:
                if c and c not in seen:
                    proj.append(f'"{c}"')

        select_sql = ",\n    ".join(proj) if proj else "*"
        con.execute(
            f'CREATE OR REPLACE TABLE {output_table} AS\n'
            f'SELECT {select_sql}\nFROM {source_table}'
        )
        n_out_cols = con.execute(
            f"SELECT COUNT(*) FROM (DESCRIBE {output_table})"
        ).fetchone()[0]
        logger.info(
            "FeatureNormalizer.transform_sql: -> %s (%d cols)",
            output_table, n_out_cols,
        )
        return output_table

    def _detect_power_law_sql(
        self,
        con: Any,
        source_table: str,
        where_sql: str,
    ) -> tuple[List[str], Dict[str, Dict]]:
        """SQL version of ``_detect_power_law``: one batched aggregate
        for skew/kurt/min/n_unique on every continuous column, then a
        log-log R² check on the small set of candidates (still SQL —
        no full-column pandas materialisation)."""
        if not self.continuous_cols:
            return [], {}

        # Per-column aggregates with try/except. SKEWNESS / KURTOSIS can
        # overflow on degenerate columns (constant + a single outlier);
        # we want to skip those without aborting the whole batch. NULL
        # values stay NULL — coalescing to 0 distorts moments and was
        # the original cause of the OutOfRange skew error.
        candidates: List[tuple[str, float, float]] = []
        for c in self.continuous_cols:
            try:
                row = con.execute(
                    f'SELECT SKEWNESS("{c}"), KURTOSIS("{c}"), '
                    f'MIN("{c}"), COUNT(DISTINCT "{c}") '
                    f'FROM {source_table} {where_sql}'
                ).fetchone()
            except Exception as exc:
                logger.debug("skew/kurt skipped for %s: %s", c, exc)
                continue
            if not row:
                continue
            skew_v, kurt_v, min_v, nuniq = row
            try:
                if (skew_v is not None and kurt_v is not None
                    and min_v is not None and nuniq is not None
                    and abs(float(skew_v)) > self.SKEW_THRESH
                    and float(kurt_v) > self.KURT_THRESH
                    and float(min_v) >= 0.0
                    and int(nuniq) > self._min_nunique):
                    candidates.append((c, float(skew_v), float(kurt_v)))
            except (TypeError, ValueError):
                pass

        if not candidates:
            return [], {}

        # Log-log R² check per candidate (still SQL — fetch only the
        # small log-frequency table, not the full column).
        confirmed: List[str] = []
        details: Dict[str, Dict] = {}
        for col, skew, kurt in candidates:
            try:
                # Build histogram via SQL: log10(value+1) vs log10(rank).
                # We use a simple value-frequency table (small, distinct
                # values <= n_unique cap) and fit log-log on rank-frequency.
                hist = con.execute(
                    f'SELECT "{col}" AS v, COUNT(*) AS c '
                    f'FROM {source_table} {where_sql} '
                    f'WHERE "{col}" IS NOT NULL AND "{col}" > 0 '
                    f'GROUP BY "{col}" ORDER BY c DESC LIMIT 1000'
                ).fetchall()
                if len(hist) < self._min_samples:
                    continue
                ranks = np.arange(1, len(hist) + 1, dtype=np.float64)
                freqs = np.array([h[1] for h in hist], dtype=np.float64)
                log_r = np.log10(ranks)
                log_f = np.log10(freqs)
                # R² of linear fit
                slope, intercept = np.polyfit(log_r, log_f, 1)
                pred = slope * log_r + intercept
                ss_res = float(np.sum((log_f - pred) ** 2))
                ss_tot = float(np.sum((log_f - log_f.mean()) ** 2)) or 1e-9
                r2 = 1.0 - ss_res / ss_tot
                if r2 >= self.R2_THRESH:
                    confirmed.append(col)
                    details[col] = {
                        "skew": round(skew, 2),
                        "kurt": round(kurt, 2),
                        "loglog_r2": round(r2, 4),
                    }
            except Exception as exc:
                logger.debug("loglog R² skipped for %s: %s", col, exc)
        return confirmed, details

    # ------------------------------------------------------------------
    # Column introspection
    # ------------------------------------------------------------------

    @property
    def output_columns(self) -> List[str]:
        """Return the ordered list of output column names."""
        cols = list(self.continuous_cols)
        cols.extend(self.binary_cols)
        cols.extend(f"{c}_log" for c in self.power_law_cols)
        return cols

    # ------------------------------------------------------------------
    # Power-law detection (private)
    # ------------------------------------------------------------------

    def _detect_power_law(self, df: pd.DataFrame):
        """2-stage detection: skew+kurt filter then log-log R² confirmation.

        Returns
        -------
        confirmed : list[str]
        details : dict[str, dict]
        """
        candidates = []
        for col in self.continuous_cols:
            try:
                skew = float(df[col].skew())
                kurt = float(df[col].kurtosis())
                nunique = int(df[col].nunique())
                if (
                    abs(skew) > self.SKEW_THRESH
                    and kurt > self.KURT_THRESH
                    and df[col].min() >= 0
                    and nunique > self._min_nunique
                ):
                    candidates.append((col, skew, kurt))
            except (TypeError, ValueError):
                pass

        confirmed = []
        details = {}
        for col, skew, kurt in candidates:
            r2 = self._loglog_r2(df[col])
            if r2 >= self.R2_THRESH:
                confirmed.append(col)
                details[col] = {
                    "skew": round(skew, 2),
                    "kurt": round(kurt, 2),
                    "loglog_r2": round(r2, 4),
                }
            else:
                logger.debug(
                    "Power-law rejected '%s': skew=%.1f, kurt=%.1f, "
                    "loglog_R²=%.3f < %.1f",
                    col, skew, kurt, r2, self.R2_THRESH,
                )

        return confirmed, details

    def _loglog_r2(self, series: pd.Series) -> float:
        """Log-log rank-frequency R²."""
        min_samples = self._min_samples
        vals = series.dropna()
        vals = vals[vals > 0].sort_values(ascending=False).values
        if len(vals) < min_samples:
            return 0.0
        n = max(min_samples, len(vals) // 2)
        vals = vals[:n]
        log_rank = np.log(1 + np.arange(n))
        log_val = np.log(vals)
        if log_val.std() < 1e-10:
            return 0.0
        corr = np.corrcoef(log_rank, log_val)[0, 1]
        return corr ** 2

    def _detect_power_law_from_numpy(
        self, data: dict, continuous_cols: list
    ) -> None:
        """Power-law detection from numpy dict (DuckDB fetchnumpy output).

        Sets self.power_law_cols and self.power_law_details.
        No pandas dependency.
        """
        candidates = []
        for col in continuous_cols:
            arr = data.get(col)
            if arr is None:
                continue
            arr = arr[~np.isnan(arr)] if np.issubdtype(arr.dtype, np.floating) else arr
            if len(arr) < self._min_samples:
                continue
            skew = float(np.mean(((arr - arr.mean()) / max(arr.std(), 1e-10)) ** 3))
            kurt = float(np.mean(((arr - arr.mean()) / max(arr.std(), 1e-10)) ** 4) - 3)
            nunique = len(np.unique(arr))
            if abs(skew) > self.SKEW_THRESH and kurt > self.KURT_THRESH and arr.min() >= 0 and nunique > self._min_nunique:
                candidates.append((col, skew, kurt))

        self.power_law_cols = []
        self.power_law_details = {}
        for col, skew, kurt in candidates:
            arr = data[col]
            arr = arr[~np.isnan(arr)] if np.issubdtype(arr.dtype, np.floating) else arr
            vals = arr[arr > 0]
            vals = np.sort(vals)[::-1]
            n = max(self._min_samples, len(vals) // 2)
            vals = vals[:n]
            if len(vals) < self._min_samples:
                continue
            log_rank = np.log(1 + np.arange(len(vals)))
            log_val = np.log(vals.astype(np.float64))
            if log_val.std() < 1e-10:
                continue
            corr = np.corrcoef(log_rank, log_val)[0, 1]
            r2 = corr ** 2
            if r2 >= self.R2_THRESH:
                self.power_law_cols.append(col)
                self.power_law_details[col] = {"skew": round(skew, 2), "kurt": round(kurt, 2), "loglog_r2": round(r2, 4)}

        if self.power_law_cols:
            logger.info("Power-law detected: %d columns confirmed (R²>%.1f): %s",
                        len(self.power_law_cols), self.R2_THRESH, self.power_law_cols)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save fitted normalizer to *path* directory."""
        if not self._fitted:
            raise RuntimeError("Cannot save unfitted FeatureNormalizer")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self._mean is not None:
            np.savez(
                path / "scaler_params.npz",
                mean=np.asarray(self._mean),
                std=np.asarray(self._std),
            )

        meta = {
            "continuous_cols": self.continuous_cols,
            "binary_cols": self.binary_cols,
            "categorical_int_cols": self.categorical_int_cols,
            "probability_cols": self.probability_cols,
            "power_law_cols": self.power_law_cols,
            "power_law_details": self.power_law_details,
        }
        with open(path / "normalizer_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("FeatureNormalizer saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "FeatureNormalizer":
        """Load a previously saved normalizer."""
        path = Path(path)

        obj = cls()
        params_path = path / "scaler_params.npz"
        if params_path.exists():
            data = np.load(params_path)
            obj._mean = data["mean"]
            obj._std = data["std"]

        with open(path / "normalizer_meta.json", "r") as f:
            meta = json.load(f)

        obj.continuous_cols = meta["continuous_cols"]
        obj.binary_cols = meta["binary_cols"]
        obj.categorical_int_cols = meta.get("categorical_int_cols", [])
        obj.probability_cols = meta.get("probability_cols", [])
        obj.power_law_cols = meta["power_law_cols"]
        obj.power_law_details = meta.get("power_law_details", {})
        obj._fitted = True

        logger.info("FeatureNormalizer loaded from %s", path)
        return obj
