# -*- coding: utf-8 -*-
"""
build_real_sequences.py — Time-based Sliding Window Training Data
=================================================================
Creates sliding-window feature/label rows from the real 2,000-user
financial transaction dataset using DuckDB for all heavy computation.

Input:
  data/새 폴더/01_financial_transactions.parquet   24.4M txns
  data/새 폴더/01_financial_users.parquet           2,000 users
  data/새 폴더/01_financial_cards.parquet            6,146 cards

Output:
  data/real_2k_windows.parquet
  ~9 windows per user, ~18,000 rows total

Window spec:
  Measurement period : 2019-03-01 .. 2020-02-28
  window_days=90, stride_days=30
  Label period       : next 30 days after each window
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import time
import argparse
import duckdb

# ── paths ────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "새 폴더")

TXN_PATH = os.path.join(DATA_DIR, "01_financial_transactions.parquet").replace("\\", "/")
USR_PATH = os.path.join(DATA_DIR, "01_financial_users.parquet").replace("\\", "/")
CRD_PATH = os.path.join(DATA_DIR, "01_financial_cards.parquet").replace("\\", "/")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "real_2k_windows.parquet").replace("\\", "/")

# ── parameters ───────────────────────────────────────────────────
PERIOD_START = "2019-03-01"
PERIOD_END = "2020-02-28"
WINDOW_DAYS = 90
STRIDE_DAYS = 30
LABEL_DAYS = 30


def parse_args():
    p = argparse.ArgumentParser(description="Build sliding-window training data")
    p.add_argument("--window-days", type=int, default=WINDOW_DAYS)
    p.add_argument("--stride-days", type=int, default=STRIDE_DAYS)
    p.add_argument("--label-days", type=int, default=LABEL_DAYS)
    p.add_argument("--output", type=str, default=OUTPUT_PATH)
    return p.parse_args()


def main():
    args = parse_args()
    window_days = args.window_days
    stride_days = args.stride_days
    label_days = args.label_days
    output_path = args.output

    t_total = time.time()
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    # allow large intermediate results
    con.execute("SET memory_limit = '4GB'")

    # ================================================================
    # Phase 1 — Register source tables
    # ================================================================
    print("=" * 64)
    print("build_real_sequences.py — Sliding Window Builder")
    print("=" * 64)

    print("\n[Phase 1] Registering source tables ...")
    t0 = time.time()

    con.execute(f"""
        CREATE TABLE txn AS
        SELECT
            user_id,
            MAKE_DATE(year::INTEGER, month::INTEGER, day::INTEGER) AS txn_date,
            amount,
            mcc,
            merchant_id,
            is_fraud
        FROM read_parquet('{TXN_PATH}')
        WHERE (year = 2019 AND month >= 3)
           OR (year = 2020 AND month <= 3)
    """)
    txn_cnt = con.execute("SELECT COUNT(*) FROM txn").fetchone()[0]
    print(f"  txn rows in period: {txn_cnt:,}")

    # Users: row_number()-1 gives the 0-based user_id matching txn.user_id
    con.execute(f"""
        CREATE TABLE usr AS
        SELECT
            (row_number() OVER () - 1)::INTEGER AS user_id,
            "Current Age"                       AS age,
            "Yearly Income - Person"            AS yearly_income,
            "FICO Score"                        AS fico_score,
            "Gender"                            AS gender
        FROM read_parquet('{USR_PATH}')
    """)

    # Cards: aggregate per user
    con.execute(f"""
        CREATE TABLE card_agg AS
        SELECT
            "User"::INTEGER            AS user_id,
            COUNT(*)::INTEGER          AS n_cards,
            SUM("Credit Limit")        AS total_credit_limit,
            MAX("Credit Limit")        AS max_credit_limit
        FROM read_parquet('{CRD_PATH}')
        GROUP BY "User"
    """)

    # Demographics (users + cards)
    con.execute("""
        CREATE TABLE demo AS
        SELECT
            u.user_id,
            u.age,
            u.yearly_income,
            u.fico_score,
            CASE WHEN u.gender = 'Male' THEN 1 ELSE 0 END AS gender_male,
            COALESCE(c.n_cards, 0)              AS n_cards,
            COALESCE(c.total_credit_limit, 0.0) AS total_credit_limit,
            COALESCE(c.max_credit_limit, 0.0)   AS max_credit_limit
        FROM usr u
        LEFT JOIN card_agg c ON u.user_id = c.user_id
    """)
    user_cnt = con.execute("SELECT COUNT(*) FROM demo").fetchone()[0]
    print(f"  users with demographics: {user_cnt:,}")
    print(f"  ({time.time() - t0:.1f}s)")

    # ================================================================
    # Phase 2 — Generate window boundaries
    # ================================================================
    print("\n[Phase 2] Generating window boundaries ...")
    t0 = time.time()

    con.execute(f"""
        CREATE TABLE windows AS
        WITH RECURSIVE gen(win_id, win_start) AS (
            SELECT 0, DATE '{PERIOD_START}'
            UNION ALL
            SELECT win_id + 1, win_start + INTERVAL '{stride_days}' DAY
            FROM gen
            WHERE win_start + INTERVAL '{stride_days}' DAY
                  + INTERVAL '{window_days}' DAY
                  + INTERVAL '{label_days}' DAY
                  <= DATE '{PERIOD_END}' + INTERVAL '1' DAY
        )
        SELECT
            win_id,
            win_start::DATE                                            AS win_start,
            (win_start + INTERVAL '{window_days}' DAY - INTERVAL '1' DAY)::DATE
                                                                       AS win_end,
            (win_start + INTERVAL '{window_days}' DAY)::DATE           AS label_start,
            (win_start + INTERVAL '{window_days}' DAY
                       + INTERVAL '{label_days}' DAY - INTERVAL '1' DAY)::DATE
                                                                       AS label_end
        FROM gen
    """)
    win_cnt = con.execute("SELECT COUNT(*) FROM windows").fetchone()[0]
    print(f"  windows generated: {win_cnt}")
    bounds = con.execute("SELECT * FROM windows ORDER BY win_id").fetchall()
    for b in bounds:
        print(f"    win {b[0]}: [{b[1]} .. {b[2]}] -> label [{b[3]} .. {b[4]}]")
    print(f"  ({time.time() - t0:.1f}s)")

    # ================================================================
    # Phase 3 — Feature computation per window
    # ================================================================
    print("\n[Phase 3] Computing per-window features (DuckDB SQL) ...")
    t0 = time.time()

    # 3a. Aggregate features within each window
    con.execute("""
        CREATE TABLE win_feats AS
        SELECT
            d.user_id,
            w.win_id,
            w.win_start,
            w.win_end,
            w.label_start,
            w.label_end,
            -- demographics (static per user)
            d.age,
            d.yearly_income,
            d.fico_score,
            d.gender_male,
            d.n_cards,
            d.total_credit_limit,
            d.max_credit_limit,
            -- aggregate features
            COALESCE(agg.txn_count, 0)                  AS txn_count,
            COALESCE(agg.total_spend, 0.0)              AS total_spend,
            COALESCE(agg.avg_amount, 0.0)               AS avg_amount,
            COALESCE(agg.std_amount, 0.0)               AS std_amount,
            COALESCE(agg.min_amount, 0.0)               AS min_amount,
            COALESCE(agg.max_amount, 0.0)               AS max_amount,
            COALESCE(agg.median_amount, 0.0)            AS median_amount,
            COALESCE(agg.unique_mcc, 0)                 AS unique_mcc,
            COALESCE(agg.unique_merchant, 0)             AS unique_merchant,
            COALESCE(agg.fraud_count, 0)                AS fraud_count,
            COALESCE(agg.monthly_spend, 0.0)            AS monthly_spend,
            COALESCE(agg.txn_days, 0)                   AS active_days,
            COALESCE(agg.max_daily_spend, 0.0)          AS max_daily_spend,
            COALESCE(agg.avg_daily_spend, 0.0)          AS avg_daily_spend
        FROM demo d
        CROSS JOIN windows w
        LEFT JOIN (
            SELECT
                t.user_id,
                w2.win_id,
                COUNT(*)::INTEGER                       AS txn_count,
                SUM(t.amount)                           AS total_spend,
                AVG(t.amount)                           AS avg_amount,
                STDDEV_SAMP(t.amount)                   AS std_amount,
                MIN(t.amount)                           AS min_amount,
                MAX(t.amount)                           AS max_amount,
                MEDIAN(t.amount)                        AS median_amount,
                COUNT(DISTINCT t.mcc)::INTEGER          AS unique_mcc,
                COUNT(DISTINCT t.merchant_id)::INTEGER  AS unique_merchant,
                SUM(t.is_fraud)::INTEGER                AS fraud_count,
                SUM(t.amount) / 3.0                     AS monthly_spend,
                COUNT(DISTINCT t.txn_date)::INTEGER     AS txn_days,
                MAX(daily.daily_sum)                    AS max_daily_spend,
                AVG(daily.daily_sum)                    AS avg_daily_spend
            FROM txn t
            JOIN windows w2
              ON t.txn_date BETWEEN w2.win_start AND w2.win_end
            LEFT JOIN (
                SELECT user_id, txn_date, SUM(amount) AS daily_sum
                FROM txn
                GROUP BY user_id, txn_date
            ) daily ON t.user_id = daily.user_id AND t.txn_date = daily.txn_date
            GROUP BY t.user_id, w2.win_id
        ) agg ON d.user_id = agg.user_id AND w.win_id = agg.win_id
    """)
    feat_cnt = con.execute("SELECT COUNT(*) FROM win_feats").fetchone()[0]
    print(f"  win_feats rows: {feat_cnt:,}")
    print(f"  ({time.time() - t0:.1f}s)")

    # ================================================================
    # Phase 4 — Sequence features per window
    # ================================================================
    print("\n[Phase 4] Building sequence columns ...")
    t0 = time.time()

    con.execute("""
        CREATE TABLE win_seqs AS
        SELECT
            t.user_id,
            w.win_id,
            LIST(t.amount ORDER BY t.txn_date, t.amount)                           AS txn_amount_seq,
            LIST(t.mcc ORDER BY t.txn_date, t.amount)                              AS txn_mcc_seq,
            LIST((YEAR(t.txn_date)*10000 + MONTH(t.txn_date)*100 + DAY(t.txn_date))::INTEGER
                 ORDER BY t.txn_date, t.amount)                                    AS txn_date_seq
        FROM txn t
        JOIN windows w ON t.txn_date BETWEEN w.win_start AND w.win_end
        GROUP BY t.user_id, w.win_id
    """)
    seq_cnt = con.execute("SELECT COUNT(*) FROM win_seqs").fetchone()[0]
    print(f"  sequence rows: {seq_cnt:,}")
    print(f"  ({time.time() - t0:.1f}s)")

    # ================================================================
    # Phase 5 — Labels from the next-30-day period
    # ================================================================
    print("\n[Phase 5] Computing labels ...")
    t0 = time.time()

    # 5a. MCC set in the window (for has_new_merchant detection)
    con.execute("""
        CREATE TABLE win_mcc_set AS
        SELECT
            t.user_id,
            w.win_id,
            LIST(DISTINCT t.mcc) AS mcc_set
        FROM txn t
        JOIN windows w ON t.txn_date BETWEEN w.win_start AND w.win_end
        GROUP BY t.user_id, w.win_id
    """)

    # 5b. Label-period aggregates
    con.execute("""
        CREATE TABLE label_base AS
        SELECT
            t.user_id,
            w.win_id,
            SUM(t.amount)                   AS label_spend,
            COUNT(*)::INTEGER               AS label_txn_count,
            LIST(DISTINCT t.mcc)            AS label_mcc_set
        FROM txn t
        JOIN windows w ON t.txn_date BETWEEN w.label_start AND w.label_end
        GROUP BY t.user_id, w.win_id
    """)

    # 5c. Top MCC per user/window in label period (most frequent)
    con.execute("""
        CREATE TABLE label_top AS
        SELECT user_id, win_id, mcc AS label_top_mcc
        FROM (
            SELECT
                t.user_id,
                w.win_id,
                t.mcc,
                COUNT(*) AS freq,
                ROW_NUMBER() OVER (
                    PARTITION BY t.user_id, w.win_id
                    ORDER BY COUNT(*) DESC, t.mcc
                ) AS rn
            FROM txn t
            JOIN windows w ON t.txn_date BETWEEN w.label_start AND w.label_end
            GROUP BY t.user_id, w.win_id, t.mcc
        ) ranked
        WHERE rn = 1
    """)

    # 5d. Combine label tables
    con.execute("""
        CREATE TABLE label_agg AS
        SELECT
            lb.user_id,
            lb.win_id,
            lb.label_spend,
            lb.label_txn_count,
            lb.label_mcc_set,
            lt.label_top_mcc
        FROM label_base lb
        LEFT JOIN label_top lt
          ON lb.user_id = lt.user_id AND lb.win_id = lt.win_id
    """)

    # 5c. Compute has_new_merchant: any MCC in label period not in window period
    con.execute("""
        CREATE TABLE labels AS
        SELECT
            la.user_id,
            la.win_id,
            la.label_spend,
            la.label_txn_count,
            la.label_top_mcc,
            -- has_new_merchant: 1 if label period has an MCC not seen in window
            CASE WHEN EXISTS (
                SELECT 1
                FROM UNNEST(la.label_mcc_set) AS lm(mcc_val)
                WHERE lm.mcc_val NOT IN (
                    SELECT UNNEST(wm.mcc_set) FROM win_mcc_set wm
                    WHERE wm.user_id = la.user_id AND wm.win_id = la.win_id
                )
            ) THEN 1 ELSE 0 END             AS has_new_merchant
        FROM label_agg la
    """)
    label_cnt = con.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
    print(f"  label rows: {label_cnt:,}")
    print(f"  ({time.time() - t0:.1f}s)")

    # ================================================================
    # Phase 6 — Spend change (ratio vs previous window)
    # ================================================================
    print("\n[Phase 6] Computing spend_change ...")
    t0 = time.time()

    con.execute("""
        CREATE TABLE spend_prev AS
        SELECT
            f.user_id,
            f.win_id,
            f.total_spend,
            LAG(f.total_spend) OVER (
                PARTITION BY f.user_id ORDER BY f.win_id
            ) AS prev_spend
        FROM win_feats f
    """)
    print(f"  ({time.time() - t0:.1f}s)")

    # ================================================================
    # Phase 7 — Final join and output
    # ================================================================
    print("\n[Phase 7] Assembling final table ...")
    t0 = time.time()

    con.execute(f"""
        COPY (
            SELECT
                -- identifiers
                f.user_id,
                f.win_id,
                f.win_start,
                f.win_end,
                f.label_start,
                f.label_end,

                -- demographics
                f.age,
                f.yearly_income,
                f.fico_score,
                f.gender_male,
                f.n_cards,
                f.total_credit_limit,
                f.max_credit_limit,

                -- aggregate features
                f.txn_count,
                f.total_spend,
                f.avg_amount,
                f.std_amount,
                f.min_amount,
                f.max_amount,
                f.median_amount,
                f.unique_mcc,
                f.unique_merchant,
                f.fraud_count,
                f.monthly_spend,
                f.active_days,
                f.max_daily_spend,
                f.avg_daily_spend,

                -- sequence features
                s.txn_amount_seq,
                s.txn_mcc_seq,
                s.txn_date_seq,
                COALESCE(LEN(s.txn_amount_seq), 0)::INTEGER AS seq_length,

                -- labels
                COALESCE(lb.has_new_merchant, 0)::INTEGER   AS has_new_merchant,
                CASE
                    WHEN sp.prev_spend IS NULL OR sp.prev_spend = 0
                        THEN 0.0
                    ELSE (COALESCE(lb.label_spend, 0.0) - sp.prev_spend)
                         / sp.prev_spend
                END                                         AS spend_change,
                COALESCE(lb.label_top_mcc, -1)::INTEGER     AS top_mcc,
                COALESCE(lb.label_spend, 0.0)               AS label_total_spend,
                COALESCE(lb.label_txn_count, 0)::INTEGER    AS label_txn_count

            FROM win_feats f
            LEFT JOIN win_seqs s
              ON f.user_id = s.user_id AND f.win_id = s.win_id
            LEFT JOIN labels lb
              ON f.user_id = lb.user_id AND f.win_id = lb.win_id
            LEFT JOIN spend_prev sp
              ON f.user_id = sp.user_id AND f.win_id = sp.win_id
            ORDER BY f.user_id, f.win_id
        )
        TO '{output_path}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 50000)
    """)

    # ── summary ──────────────────────────────────────────────────
    final = con.execute(f"""
        SELECT COUNT(*) AS rows,
               COUNT(DISTINCT user_id) AS users,
               COUNT(DISTINCT win_id) AS windows
        FROM read_parquet('{output_path}')
    """).fetchone()

    cols = con.execute(f"""
        SELECT column_name, column_type
        FROM (DESCRIBE SELECT * FROM read_parquet('{output_path}'))
    """).fetchall()

    fsize = os.path.getsize(output_path) / (1024 * 1024)

    print(f"\n{'=' * 64}")
    print(f"  Output : {output_path}")
    print(f"  Size   : {fsize:.1f} MB")
    print(f"  Rows   : {final[0]:,}  ({final[1]:,} users x {final[2]} windows)")
    print(f"  Columns: {len(cols)}")
    print(f"{'=' * 64}")
    print(f"  Schema:")
    for c in cols:
        print(f"    {c[0]:30s}  {c[1]}")
    print(f"{'=' * 64}")

    # label distribution
    dist = con.execute(f"""
        SELECT
            AVG(has_new_merchant)       AS new_merchant_rate,
            AVG(spend_change)           AS avg_spend_change,
            STDDEV(spend_change)        AS std_spend_change,
            COUNT(DISTINCT top_mcc)     AS distinct_top_mcc,
            AVG(txn_count)              AS avg_txn_count,
            AVG(seq_length)             AS avg_seq_length
        FROM read_parquet('{output_path}')
    """).fetchone()
    print(f"  Label stats:")
    print(f"    has_new_merchant rate : {dist[0]:.4f}")
    print(f"    spend_change mean    : {dist[1]:.4f} (std {dist[2]:.4f})")
    print(f"    distinct top_mcc     : {dist[3]}")
    print(f"    avg txn_count/window : {dist[4]:.1f}")
    print(f"    avg seq_length       : {dist[5]:.1f}")

    con.close()
    elapsed = time.time() - t_total
    print(f"\n  Total elapsed: {elapsed:.1f}s")
    print("  Done.")


if __name__ == "__main__":
    main()
