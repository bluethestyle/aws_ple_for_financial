# -*- coding: utf-8 -*-
"""
calibrate_real_distributions.py -- Extract statistical parameters from real data
================================================================================
Reads real financial users, transactions, Santander demographics, and
pre-built windowed samples to produce calibration parameters that drive
the benchmark data generator.

Outputs:  configs/santander/calibration_params.yaml

Steps:
  A. Persona discovery (6-component GMM on user-level behaviour vectors)
  B. Demographic distributions per persona (TruncNorm, LogNorm, Gamma)
  C. Transaction patterns per persona (MCC, amount, hour, AR(1), FFT)
  D. Signal-to-noise calibration (XGBoost ceilings on windowed data)
  E. YAML serialisation

All data loading via DuckDB.  No direct pandas usage.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import argparse
import logging
import time
from typing import Any, Dict, List, Tuple

import duckdb
import numpy as np
import yaml

from scipy import stats as sp_stats
from sklearn.mixture import GaussianMixture

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("calibrate")


# =========================================================================
# Config helpers
# =========================================================================

def _load_pipeline_config() -> Dict[str, Any]:
    """Load pipeline.yaml for config-driven parameters."""
    for candidate in [
        os.path.join(PROJECT_ROOT, "configs", "santander", "pipeline.yaml"),
        os.path.join(PROJECT_ROOT, "configs", "financial", "pipeline.yaml"),
    ]:
        if os.path.exists(candidate):
            with open(candidate, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
    return {}


def _default_config() -> Dict[str, Any]:
    """Return default calibration configuration.  Pipeline YAML overrides."""
    return {
        "n_personas": 6,
        "persona_names": [
            "conservative_saver",
            "active_spender",
            "young_digital",
            "high_value",
            "occasional_user",
            "diversified",
        ],
        "txn_filter": {
            "period_start": "2019-03-01",
            "period_end": "2020-02-28",
        },
        "gmm_random_state": 42,
        "fft_top_harmonics": 2,
        "xgb_params": {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        },
    }


def _merge_config(pipeline_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Merge pipeline.yaml calibration section into defaults."""
    cfg = _default_config()
    cal = pipeline_cfg.get("calibration", {})
    for k, v in cal.items():
        if isinstance(v, dict) and k in cfg and isinstance(cfg[k], dict):
            cfg[k].update(v)
        else:
            cfg[k] = v

    # Pull Santander column names from pipeline data config (avoid hardcoding)
    data_cfg = pipeline_cfg.get("data", {})
    if "id_col" in data_cfg:
        cfg.setdefault("sant_id_col", data_cfg["id_col"])
    temporal = data_cfg.get("temporal_split", {})
    if "date_col" in temporal:
        cfg.setdefault("sant_date_col", temporal["date_col"])
    # Numeric feature names from pipeline features section
    features_cfg = pipeline_cfg.get("features", {})
    for num_entry in features_cfg.get("numeric", []):
        col_name = num_entry if isinstance(num_entry, str) else num_entry.get("name", "")
        if col_name == "age":
            cfg.setdefault("sant_age_col", col_name)
        elif col_name == "income":
            cfg.setdefault("sant_income_col", col_name)
        elif col_name == "tenure_months":
            cfg.setdefault("sant_tenure_col", col_name)

    return cfg


# =========================================================================
# Path resolution
# =========================================================================

def _resolve_paths(pipeline_cfg: Dict[str, Any]) -> Dict[str, str]:
    """Resolve data file paths, preferring pipeline.yaml values."""
    cal_paths = pipeline_cfg.get("calibration", {}).get("paths", {})
    data_dir = os.path.join(PROJECT_ROOT, "data")
    raw_dir = os.path.join(data_dir, "\uc0c8 \ud3f4\ub354")  # data/새 폴더

    return {
        "real_users": cal_paths.get(
            "real_users",
            os.path.join(raw_dir, "01_financial_users.parquet"),
        ),
        "real_txns": cal_paths.get(
            "real_txns",
            os.path.join(raw_dir, "01_financial_transactions.parquet"),
        ),
        "santander": cal_paths.get(
            "santander",
            os.path.join(data_dir, "santander_linked.parquet"),
        ),
        "windows": cal_paths.get(
            "windows",
            os.path.join(data_dir, "real_2k_windows.parquet"),
        ),
    }


# =========================================================================
# A. Persona discovery
# =========================================================================

def _build_user_behaviour_matrix(
    con: duckdb.DuckDBPyConnection,
    paths: Dict[str, str],
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Per-user behaviour vector: MCC frequency (N_mcc dims) + avg_amount +
    txn_count + unique_merchants.

    Returns:
        X        -- (n_users, feature_dim) float64
        user_ids -- (n_users,) int
        mcc_list -- sorted list of unique MCCs
    """
    users_path = paths["real_users"].replace("\\", "/")
    txn_path = paths["real_txns"].replace("\\", "/")
    period_start = cfg["txn_filter"]["period_start"]
    period_end = cfg["txn_filter"]["period_end"]

    log.info("  Building user-level behaviour matrix (period %s .. %s)...",
             period_start, period_end)

    # Load users as id mapping
    con.execute(f"""
        CREATE OR REPLACE TABLE cal_users AS
        SELECT
            row_number() OVER () - 1 AS user_id,
            "Person"                 AS person_label
        FROM read_parquet('{users_path}')
    """)

    # Filter transactions to measurement period
    con.execute(f"""
        CREATE OR REPLACE TABLE cal_txns AS
        SELECT
            user_id,
            mcc::INT        AS mcc,
            amount,
            merchant_id
        FROM read_parquet('{txn_path}')
        WHERE make_date(CAST(year AS INT), CAST(month AS INT), CAST(day AS INT))
              BETWEEN DATE '{period_start}' AND DATE '{period_end}'
    """)

    # Discover unique MCCs
    mcc_rows = con.execute("""
        SELECT DISTINCT mcc FROM cal_txns ORDER BY mcc
    """).fetchnumpy()
    mcc_list = mcc_rows["mcc"].tolist()
    n_mcc = len(mcc_list)
    log.info("  Unique MCCs in period: %d", n_mcc)

    # MCC frequency pivot per user
    # Build a CASE WHEN pivot dynamically
    pivot_cols = []
    for i, mcc_code in enumerate(mcc_list):
        pivot_cols.append(
            f"COALESCE(SUM(CASE WHEN mcc = {mcc_code} THEN 1 ELSE 0 END), 0) AS mcc_{i}"
        )

    pivot_sql = ", ".join(pivot_cols)
    con.execute(f"""
        CREATE OR REPLACE TABLE cal_user_agg AS
        SELECT
            user_id,
            {pivot_sql},
            COALESCE(AVG(amount), 0.0)              AS avg_amount,
            COUNT(*)                                 AS txn_count,
            COUNT(DISTINCT merchant_id)              AS unique_merchants
        FROM cal_txns
        GROUP BY user_id
    """)

    # Fetch as numpy
    col_names = [f"mcc_{i}" for i in range(n_mcc)] + [
        "avg_amount", "txn_count", "unique_merchants",
    ]
    select_str = ", ".join(col_names)
    result = con.execute(f"""
        SELECT user_id, {select_str}
        FROM cal_user_agg
        ORDER BY user_id
    """).fetchnumpy()

    user_ids = result["user_id"]
    n_users = len(user_ids)
    feature_dim = n_mcc + 3
    X = np.zeros((n_users, feature_dim), dtype=np.float64)

    for i, col in enumerate(col_names):
        X[:, i] = result[col].astype(np.float64)

    # Normalise MCC frequencies to proportions per user
    mcc_sums = X[:, :n_mcc].sum(axis=1, keepdims=True)
    mcc_sums = np.where(mcc_sums == 0, 1.0, mcc_sums)
    X[:, :n_mcc] /= mcc_sums

    # Log-transform skewed columns
    X[:, n_mcc] = np.log1p(X[:, n_mcc])         # avg_amount
    X[:, n_mcc + 1] = np.log1p(X[:, n_mcc + 1]) # txn_count
    X[:, n_mcc + 2] = np.log1p(X[:, n_mcc + 2]) # unique_merchants

    log.info("  Behaviour matrix shape: %s", X.shape)
    return X, user_ids, mcc_list


def _fit_gmm(
    X: np.ndarray,
    cfg: Dict[str, Any],
) -> Tuple[GaussianMixture, np.ndarray]:
    """Fit GMM and return model + per-user persona labels."""
    n_comp = cfg["n_personas"]
    seed = cfg["gmm_random_state"]

    log.info("  Fitting GMM with %d components (seed=%d)...", n_comp, seed)
    gmm = GaussianMixture(
        n_components=n_comp,
        covariance_type="full",
        random_state=seed,
        max_iter=300,
        n_init=3,
    )
    gmm.fit(X)
    labels = gmm.predict(X)
    log.info("  GMM converged=%s, BIC=%.1f", gmm.converged_, gmm.bic(X))

    # Log cluster sizes
    for c in range(n_comp):
        log.info("    Persona %d: %d users (%.1f%%)",
                 c, (labels == c).sum(),
                 100.0 * (labels == c).sum() / len(labels))

    return gmm, labels


# =========================================================================
# B. Demographic distributions per persona
# =========================================================================

def _fit_demographics(
    con: duckdb.DuckDBPyConnection,
    paths: Dict[str, str],
    persona_labels: np.ndarray,
    user_ids: np.ndarray,
    cfg: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Fit age/income/tenure distributions per persona from Santander data.

    Since Santander users and real users are different populations, we:
    1. Load Santander demographics
    2. Build age/income/tenure arrays
    3. Assign Santander users to personas via nearest-centroid on
       (age_group, income_group) -- approximate mapping
    4. Fit parametric distributions per persona
    """
    log.info("  Fitting demographic distributions per persona...")

    sant_path = paths["santander"].replace("\\", "/")

    # Column names from pipeline config (no hardcoding)
    id_col = cfg.get("sant_id_col", "customer_id")
    date_col = cfg.get("sant_date_col", "snapshot_date")
    age_col = cfg.get("sant_age_col", "age")
    income_col = cfg.get("sant_income_col", "income")
    tenure_col = cfg.get("sant_tenure_col", "tenure_months")

    # Load Santander demographics (deduplicated by id, latest snapshot)
    result = con.execute(f"""
        WITH ranked AS (
            SELECT
                {age_col},
                {income_col},
                {tenure_col},
                ROW_NUMBER() OVER (PARTITION BY {id_col}
                                   ORDER BY {date_col} DESC) AS rn
            FROM read_parquet('{sant_path}')
            WHERE {age_col} > 0
        )
        SELECT {age_col}, {income_col}, {tenure_col}
        FROM ranked WHERE rn = 1
    """).fetchnumpy()

    sant_age = result[age_col].astype(np.float64)
    sant_income = result[income_col].astype(np.float64)
    sant_tenure = result[tenure_col].astype(np.float64)

    # Simple persona assignment for Santander users:
    # Map (age_group, income_group) -> persona via majority vote from real users
    # First, get real user age/income for users that appear in the GMM
    real_users_path = paths["real_users"].replace("\\", "/")

    # Insert user_ids into DuckDB for filtering
    uid_list = user_ids.astype(int).tolist()
    con.execute("CREATE OR REPLACE TABLE cal_uid_filter (user_id INT)")
    con.executemany("INSERT INTO cal_uid_filter VALUES (?)",
                    [(u,) for u in uid_list])

    real_demo = con.execute(f"""
        SELECT
            u.user_id,
            u.age,
            u.yearly_income
        FROM (
            SELECT
                row_number() OVER () - 1    AS user_id,
                "Current Age"               AS age,
                "Yearly Income - Person"    AS yearly_income
            FROM read_parquet('{real_users_path}')
        ) u
        JOIN cal_uid_filter f ON u.user_id = f.user_id
        ORDER BY u.user_id
    """).fetchnumpy()

    n_personas = cfg["n_personas"]
    persona_names = cfg["persona_names"]

    # Build persona centroids from real users (aligned with persona_labels)
    real_age = real_demo["age"].astype(np.float64)
    real_income = real_demo["yearly_income"].astype(np.float64)

    centroids_age = np.zeros(n_personas)
    centroids_income = np.zeros(n_personas)
    for p in range(n_personas):
        mask = persona_labels == p
        if mask.sum() > 0:
            centroids_age[p] = np.median(real_age[mask])
            centroids_income[p] = np.median(np.log1p(real_income[mask]))

    # Assign each Santander user to nearest centroid (age + log_income)
    sant_log_income = np.log1p(np.clip(sant_income, 0, None))
    # Normalise dimensions
    age_scale = max(np.std(centroids_age), 1.0)
    income_scale = max(np.std(centroids_income), 1.0)

    sant_persona = np.zeros(len(sant_age), dtype=np.int32)
    for i in range(len(sant_age)):
        dists = ((sant_age[i] - centroids_age) / age_scale) ** 2 + \
                ((sant_log_income[i] - centroids_income) / income_scale) ** 2
        sant_persona[i] = int(np.argmin(dists))

    # Fit parametric distributions per persona
    demographics: Dict[str, Dict[str, Any]] = {}

    for p in range(n_personas):
        mask = sant_persona == p
        count = int(mask.sum())
        if count < 10:
            log.warning("    Persona %d (%s): only %d Santander users, skipping fit",
                        p, persona_names[p], count)
            demographics[persona_names[p]] = {"count": count}
            continue

        p_age = sant_age[mask]
        p_income = sant_income[mask]
        p_tenure = sant_tenure[mask]

        # Age: TruncatedNormal (18..100)
        lo_age, hi_age = 18.0, 100.0
        age_mu = float(np.mean(p_age))
        age_sigma = max(float(np.std(p_age)), 1.0)
        a_clip = (lo_age - age_mu) / age_sigma
        b_clip = (hi_age - age_mu) / age_sigma
        # Verify fit
        try:
            sp_stats.truncnorm.fit(p_age, a_clip, b_clip, loc=age_mu, scale=age_sigma)
        except Exception:
            pass  # use empirical estimates

        # Income: LogNormal (positive values only)
        p_income_pos = p_income[p_income > 0]
        if len(p_income_pos) > 10:
            log_inc = np.log(p_income_pos)
            inc_mu = float(np.mean(log_inc))
            inc_sigma = max(float(np.std(log_inc)), 0.01)
        else:
            inc_mu = 10.0
            inc_sigma = 1.0
        income_zero_rate = float(1.0 - len(p_income_pos) / max(count, 1))

        # Tenure: Gamma (positive values only)
        p_tenure_pos = p_tenure[p_tenure > 0]
        if len(p_tenure_pos) > 10:
            tenure_mean = float(np.mean(p_tenure_pos))
            tenure_var = max(float(np.var(p_tenure_pos)), 1.0)
            # Method of moments: shape = mean^2/var, scale = var/mean
            gamma_shape = tenure_mean ** 2 / tenure_var
            gamma_scale = tenure_var / tenure_mean
        else:
            gamma_shape = 2.0
            gamma_scale = 30.0

        demographics[persona_names[p]] = {
            "count": count,
            "age": {
                "dist": "truncnorm",
                "mu": round(age_mu, 2),
                "sigma": round(age_sigma, 2),
                "lo": lo_age,
                "hi": hi_age,
            },
            "income": {
                "dist": "lognorm",
                "mu": round(inc_mu, 4),
                "sigma": round(inc_sigma, 4),
                "zero_rate": round(income_zero_rate, 4),
            },
            "tenure_months": {
                "dist": "gamma",
                "shape": round(gamma_shape, 4),
                "scale": round(gamma_scale, 4),
            },
        }
        log.info("    Persona %d (%s): n=%d, age=%.1f+/-%.1f, "
                 "income_mu=%.2f, tenure_shape=%.2f",
                 p, persona_names[p], count, age_mu, age_sigma,
                 inc_mu, gamma_shape)

    return demographics


# =========================================================================
# C. Transaction patterns per persona
# =========================================================================

def _fit_transaction_patterns(
    con: duckdb.DuckDBPyConnection,
    paths: Dict[str, str],
    persona_labels: np.ndarray,
    user_ids: np.ndarray,
    mcc_list: List[int],
    cfg: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Per-persona: MCC probs, amount params, hour dist, AR(1), seasonality."""
    log.info("  Fitting transaction patterns per persona...")

    txn_path = paths["real_txns"].replace("\\", "/")
    period_start = cfg["txn_filter"]["period_start"]
    period_end = cfg["txn_filter"]["period_end"]
    n_personas = cfg["n_personas"]
    persona_names = cfg["persona_names"]
    top_harmonics = cfg["fft_top_harmonics"]

    # Build persona mapping table in DuckDB
    persona_data = list(zip(
        user_ids.astype(int).tolist(),
        persona_labels.astype(int).tolist(),
    ))
    con.execute("CREATE OR REPLACE TABLE cal_persona_map (user_id INT, persona INT)")
    con.executemany("INSERT INTO cal_persona_map VALUES (?, ?)", persona_data)

    # Load filtered transactions with persona assignment
    con.execute(f"""
        CREATE OR REPLACE TABLE cal_ptxns AS
        SELECT
            t.user_id,
            p.persona,
            t.mcc::INT              AS mcc,
            t.amount,
            EXTRACT(HOUR FROM t.time)::INT AS hour,
            make_date(CAST(t.year AS INT), CAST(t.month AS INT), CAST(t.day AS INT)) AS txn_date,
            CAST(t.year AS INT) * 12 + CAST(t.month AS INT) AS year_month
        FROM read_parquet('{txn_path}') t
        JOIN cal_persona_map p ON t.user_id = p.user_id
        WHERE make_date(CAST(t.year AS INT), CAST(t.month AS INT), CAST(t.day AS INT))
              BETWEEN DATE '{period_start}' AND DATE '{period_end}'
    """)

    patterns: Dict[str, Dict[str, Any]] = {}

    for p in range(n_personas):
        pname = persona_names[p]

        # --- MCC distribution ---
        mcc_counts = con.execute(f"""
            SELECT mcc, COUNT(*) AS cnt
            FROM cal_ptxns
            WHERE persona = {p}
            GROUP BY mcc
            ORDER BY cnt DESC
        """).fetchnumpy()

        total_txns = int(mcc_counts["cnt"].sum()) if len(mcc_counts["cnt"]) > 0 else 1
        mcc_probs: Dict[int, float] = {}
        for mcc_val, cnt in zip(mcc_counts["mcc"].tolist(), mcc_counts["cnt"].tolist()):
            mcc_probs[int(mcc_val)] = round(float(cnt) / total_txns, 6)

        # --- Amount per MCC (lognormal params) ---
        amount_params: Dict[int, Dict[str, float]] = {}
        # Only fit for MCCs with enough data
        top_mccs = sorted(mcc_probs.keys(), key=lambda m: mcc_probs[m], reverse=True)[:30]

        for mcc_code in top_mccs:
            amt_data = con.execute(f"""
                SELECT amount FROM cal_ptxns
                WHERE persona = {p} AND mcc = {mcc_code} AND amount > 0
            """).fetchnumpy()

            if len(amt_data["amount"]) < 5:
                continue

            log_amounts = np.log(amt_data["amount"].astype(np.float64))
            amt_mu = float(np.mean(log_amounts))
            amt_sigma = max(float(np.std(log_amounts)), 0.01)
            amount_params[int(mcc_code)] = {
                "mu": round(amt_mu, 4),
                "sigma": round(amt_sigma, 4),
            }

        # --- Hour distribution ---
        hour_data = con.execute(f"""
            SELECT hour, COUNT(*) AS cnt
            FROM cal_ptxns
            WHERE persona = {p}
            GROUP BY hour
            ORDER BY hour
        """).fetchnumpy()

        hour_dist: Dict[int, float] = {}
        hour_total = int(hour_data["cnt"].sum()) if len(hour_data["cnt"]) > 0 else 1
        for h, cnt in zip(hour_data["hour"].tolist(), hour_data["cnt"].tolist()):
            hour_dist[int(h)] = round(float(cnt) / hour_total, 4)

        # --- AR(1) coefficient on monthly spending per user ---
        monthly_spend = con.execute(f"""
            SELECT user_id, year_month, SUM(amount) AS monthly_total
            FROM cal_ptxns
            WHERE persona = {p}
            GROUP BY user_id, year_month
            ORDER BY user_id, year_month
        """).fetchnumpy()

        ar1_coeffs: List[float] = []
        if len(monthly_spend["user_id"]) > 0:
            # Group by user_id in numpy
            uids = monthly_spend["user_id"]
            totals = monthly_spend["monthly_total"].astype(np.float64)
            unique_uids = np.unique(uids)

            for uid in unique_uids:
                mask = uids == uid
                series = totals[mask]
                if len(series) >= 4:
                    # AR(1) = corr(y[t], y[t-1])
                    y = series[1:]
                    y_lag = series[:-1]
                    if np.std(y) > 0 and np.std(y_lag) > 0:
                        corr = float(np.corrcoef(y, y_lag)[0, 1])
                        if np.isfinite(corr):
                            ar1_coeffs.append(corr)

        ar1_median = round(float(np.median(ar1_coeffs)), 4) if ar1_coeffs else 0.5

        # --- Seasonality: FFT on persona-level monthly aggregate ---
        persona_monthly = con.execute(f"""
            SELECT year_month, SUM(amount) AS monthly_total
            FROM cal_ptxns
            WHERE persona = {p}
            GROUP BY year_month
            ORDER BY year_month
        """).fetchnumpy()

        harmonics: List[Dict[str, float]] = []
        if len(persona_monthly["monthly_total"]) >= 6:
            signal = persona_monthly["monthly_total"].astype(np.float64)
            # Remove mean
            signal = signal - np.mean(signal)
            fft_vals = np.fft.rfft(signal)
            magnitudes = np.abs(fft_vals)
            phases = np.angle(fft_vals)
            # Skip DC component (index 0), get top harmonics
            if len(magnitudes) > 1:
                indices = np.argsort(magnitudes[1:])[::-1][:top_harmonics] + 1
                for idx in indices:
                    harmonics.append({
                        "frequency_idx": int(idx),
                        "magnitude": round(float(magnitudes[idx]), 4),
                        "phase": round(float(phases[idx]), 4),
                    })

        patterns[pname] = {
            "total_txns": total_txns,
            "mcc_probs": mcc_probs,
            "amount_params": amount_params,
            "hour_dist": hour_dist,
            "ar1_median": ar1_median,
            "seasonality_harmonics": harmonics,
        }
        log.info("    Persona %d (%s): %d txns, %d MCCs, AR(1)=%.3f, %d harmonics",
                 p, pname, total_txns, len(mcc_probs), ar1_median, len(harmonics))

    return patterns


# =========================================================================
# D. Signal-to-noise calibration
# =========================================================================

def _calibrate_label_ceilings(
    con: duckdb.DuckDBPyConnection,
    paths: Dict[str, str],
    cfg: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    """Train XGBoost on windowed observable features to measure empirical ceiling."""
    log.info("  Calibrating label ceilings with XGBoost...")

    try:
        from xgboost import XGBClassifier, XGBRegressor
        from sklearn.model_selection import cross_val_score
    except ImportError:
        log.warning("  xgboost or sklearn not available -- skipping ceiling calibration")
        return {}

    win_path = paths["windows"].replace("\\", "/")

    # Load windowed data via DuckDB -> numpy arrays
    schema = con.execute(f"""
        SELECT column_name, column_type
        FROM (DESCRIBE SELECT * FROM read_parquet('{win_path}'))
    """).fetchall()

    # Identify feature columns (exclude ids, dates, sequences, labels)
    exclude_prefixes = ("user_id", "win_id", "win_start", "win_end",
                        "label_start", "label_end", "txn_amount_seq",
                        "txn_mcc_seq", "txn_date_seq", "seq_length")
    label_cols = {
        "has_new_merchant": "binary",
        "spend_change": "regression",
        "label_total_spend": "regression",
        "label_txn_count": "regression",
    }

    feature_cols = []
    for col_name, col_type in schema:
        if col_name in label_cols:
            continue
        if col_name in exclude_prefixes:
            continue
        if col_type in ("DOUBLE", "FLOAT", "INTEGER", "TINYINT",
                        "SMALLINT", "BIGINT"):
            feature_cols.append(col_name)

    if not feature_cols:
        log.warning("  No suitable feature columns found in windowed data")
        return {}

    feat_str = ", ".join(feature_cols)
    all_cols = feature_cols + list(label_cols.keys())
    all_str = ", ".join(all_cols)

    data = con.execute(f"""
        SELECT {all_str}
        FROM read_parquet('{win_path}')
        WHERE txn_count > 0
    """).fetchnumpy()

    n_samples = len(data[feature_cols[0]])
    log.info("    Loaded %d windowed samples, %d features", n_samples, len(feature_cols))

    X = np.column_stack([data[c].astype(np.float64) for c in feature_cols])
    # Replace NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    xgb_params = cfg["xgb_params"]
    ceilings: Dict[str, Dict[str, float]] = {}

    for label_name, label_type in label_cols.items():
        y = data[label_name].astype(np.float64)
        valid = np.isfinite(y)
        if valid.sum() < 100:
            log.info("    %s: too few valid samples (%d), skipping", label_name, valid.sum())
            continue

        X_valid = X[valid]
        y_valid = y[valid]

        try:
            if label_type == "binary":
                model = XGBClassifier(
                    n_estimators=xgb_params["n_estimators"],
                    max_depth=xgb_params["max_depth"],
                    learning_rate=xgb_params["learning_rate"],
                    subsample=xgb_params["subsample"],
                    colsample_bytree=xgb_params["colsample_bytree"],
                    random_state=xgb_params["random_state"],
                    use_label_encoder=False,
                    eval_metric="logloss",
                    verbosity=0,
                )
                scores = cross_val_score(
                    model, X_valid, y_valid.astype(int),
                    cv=5, scoring="roc_auc",
                )
                metric_name = "xgb_auc"
                metric_val = round(float(np.mean(scores)), 4)
                metric_std = round(float(np.std(scores)), 4)
            else:
                model = XGBRegressor(
                    n_estimators=xgb_params["n_estimators"],
                    max_depth=xgb_params["max_depth"],
                    learning_rate=xgb_params["learning_rate"],
                    subsample=xgb_params["subsample"],
                    colsample_bytree=xgb_params["colsample_bytree"],
                    random_state=xgb_params["random_state"],
                    verbosity=0,
                )
                scores = cross_val_score(
                    model, X_valid, y_valid,
                    cv=5, scoring="r2",
                )
                metric_name = "xgb_r2"
                metric_val = round(float(np.mean(scores)), 4)
                metric_std = round(float(np.std(scores)), 4)

            ceilings[label_name] = {
                metric_name: metric_val,
                f"{metric_name}_std": metric_std,
                "n_samples": int(valid.sum()),
            }
            log.info("    %s: %s = %.4f (+/- %.4f), n=%d",
                     label_name, metric_name, metric_val, metric_std, valid.sum())

        except Exception as exc:
            log.warning("    %s: XGBoost failed: %s", label_name, exc, exc_info=True)

    return ceilings


# =========================================================================
# E. YAML serialisation
# =========================================================================

def _numpy_to_python(obj: Any) -> Any:
    """Recursively convert numpy types to native Python for YAML."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {_numpy_to_python(k): _numpy_to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_numpy_to_python(x) for x in obj]
    return obj


def _save_calibration_yaml(
    output_path: str,
    gmm: GaussianMixture,
    cfg: Dict[str, Any],
    mcc_list: List[int],
    demographics: Dict[str, Dict[str, Any]],
    patterns: Dict[str, Dict[str, Any]],
    ceilings: Dict[str, Dict[str, float]],
) -> None:
    """Write calibration_params.yaml."""
    persona_names = cfg["persona_names"]

    result = {
        "personas": {
            "n_components": int(gmm.n_components),
            "weights": [round(float(w), 6) for w in gmm.weights_],
            "means": gmm.means_.tolist(),
            "covariances": gmm.covariances_.tolist(),
            "names": persona_names,
            "mcc_codes": [int(m) for m in mcc_list],
            "feature_order": (
                [f"mcc_freq_{m}" for m in mcc_list]
                + ["log_avg_amount", "log_txn_count", "log_unique_merchants"]
            ),
        },
        "demographics": _numpy_to_python(demographics),
        "transactions": _numpy_to_python(patterns),
        "label_ceilings": _numpy_to_python(ceilings),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(
            result,
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            width=120,
        )

    # Verify roundtrip
    with open(output_path, "r", encoding="utf-8") as f:
        yaml.safe_load(f)

    file_size_kb = os.path.getsize(output_path) / 1024
    log.info("  Saved calibration to %s (%.1f KB)", output_path, file_size_kb)


# =========================================================================
# Main
# =========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Calibrate real data distributions for benchmark generator",
    )
    p.add_argument(
        "--output",
        type=str,
        default=os.path.join(
            PROJECT_ROOT, "configs", "santander", "calibration_params.yaml",
        ),
        help="Output YAML path",
    )
    p.add_argument(
        "--n-personas",
        type=int,
        default=None,
        help="Override number of GMM components (default: from config)",
    )
    p.add_argument(
        "--skip-xgb",
        action="store_true",
        help="Skip XGBoost label ceiling calibration (faster)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    log.info("=" * 60)
    log.info("Calibrate Real Distributions for Benchmark Generator")
    log.info("=" * 60)

    # Load config
    pipeline_cfg = _load_pipeline_config()
    cfg = _merge_config(pipeline_cfg)

    if args.n_personas is not None:
        cfg["n_personas"] = args.n_personas
        # Extend or truncate persona names
        while len(cfg["persona_names"]) < cfg["n_personas"]:
            cfg["persona_names"].append(f"persona_{len(cfg['persona_names'])}")
        cfg["persona_names"] = cfg["persona_names"][: cfg["n_personas"]]

    paths = _resolve_paths(pipeline_cfg)

    # Validate all input files exist
    for name, path in paths.items():
        if not os.path.exists(path):
            log.error("Missing input file: %s -> %s", name, path)
            sys.exit(1)
        log.info("  Input: %s -> %s", name, path)

    # DuckDB connection
    con = duckdb.connect()
    con.execute("SET memory_limit='8GB'")
    con.execute("SET threads=4")

    # ---- A. Persona discovery ----
    log.info("[A] Persona discovery")
    X, user_ids, mcc_list = _build_user_behaviour_matrix(con, paths, cfg)
    gmm, persona_labels = _fit_gmm(X, cfg)

    # ---- B. Demographic distributions ----
    log.info("[B] Demographic distributions per persona")
    demographics = _fit_demographics(con, paths, persona_labels, user_ids, cfg)

    # ---- C. Transaction patterns ----
    log.info("[C] Transaction patterns per persona")
    patterns = _fit_transaction_patterns(
        con, paths, persona_labels, user_ids, mcc_list, cfg,
    )

    # ---- D. Label ceilings ----
    if args.skip_xgb:
        log.info("[D] Skipping XGBoost ceiling calibration (--skip-xgb)")
        ceilings: Dict[str, Dict[str, float]] = {}
    else:
        log.info("[D] Signal-to-noise calibration (XGBoost ceilings)")
        ceilings = _calibrate_label_ceilings(con, paths, cfg)

    # ---- E. Save ----
    log.info("[E] Saving calibration parameters")
    _save_calibration_yaml(
        args.output, gmm, cfg, mcc_list, demographics, patterns, ceilings,
    )

    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info("Calibration complete in %.1f seconds", elapsed)
    log.info("  Output: %s", args.output)
    log.info("  Personas: %d components", cfg["n_personas"])
    log.info("  MCCs: %d unique codes", len(mcc_list))
    log.info("  Label ceilings: %d labels", len(ceilings))
    log.info("=" * 60)

    con.close()


if __name__ == "__main__":
    main()
