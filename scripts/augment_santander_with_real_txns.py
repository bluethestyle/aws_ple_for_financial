# -*- coding: utf-8 -*-
"""
Augment Santander 941K customers with real transaction patterns
───────────────────────────────────────────────────────────────
Segment-based pooling: match each Santander customer to a real user
from the 2,000-user ealtman dataset by (age_group, income_group,
activity_level), then copy+perturb their transaction history.

All heavy computation runs inside DuckDB.

Usage:
    python scripts/augment_santander_with_real_txns.py
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import time
import logging
import duckdb

# ============================================================
# Paths
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_USERS_PATH = os.path.join(BASE_DIR, "data", "새 폴더", "01_financial_users.parquet")
REAL_TXN_PATH = os.path.join(BASE_DIR, "data", "새 폴더", "01_financial_transactions.parquet")
SANTANDER_PATH = os.path.join(BASE_DIR, "data", "santander_linked.parquet")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "santander_augmented.parquet")

# ============================================================
# Config -- noise parameters (override from pipeline.yaml if available)
# ============================================================
NOISE_CFG = {
    "seed": 42,
    "amount_noise_std": 0.15,       # +/-15% normal noise on amounts
    "mcc_swap_prob": 0.10,          # 10% chance to swap MCC to neighbor
    "gap_noise_std": 0.20,          # +/-20% noise on inter-txn gaps
    "real_txn_year": 2019,          # last full year in real data
}

def _try_load_yaml_config():
    """Attempt to load noise params from pipeline.yaml."""
    try:
        import yaml
        for candidate in [
            os.path.join(BASE_DIR, "configs", "santander", "pipeline.yaml"),
            os.path.join(BASE_DIR, "configs", "financial", "pipeline.yaml"),
        ]:
            if os.path.exists(candidate):
                with open(candidate, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                aug = cfg.get("augmentation", {})
                if aug:
                    for k in NOISE_CFG:
                        if k in aug:
                            NOISE_CFG[k] = aug[k]
                    logging.info("Loaded augmentation config from %s", candidate)
                    return
    except ImportError:
        pass

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("augment")

# ============================================================
# Main
# ============================================================
def main():
    _try_load_yaml_config()

    seed = NOISE_CFG["seed"]
    amount_std = NOISE_CFG["amount_noise_std"]
    mcc_swap_prob = NOISE_CFG["mcc_swap_prob"]
    gap_std = NOISE_CFG["gap_noise_std"]
    real_year = NOISE_CFG["real_txn_year"]

    log.info("=" * 60)
    log.info("Augment Santander with Real Transaction Patterns")
    log.info("  seed=%d  amount_noise=+/-%.0f%%  mcc_swap=%.0f%%  gap_noise=+/-%.0f%%",
             seed, amount_std * 100, mcc_swap_prob * 100, gap_std * 100)
    log.info("=" * 60)

    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'")
    con.execute("SET threads=4")
    con.execute("SET enable_progress_bar=true")
    con.execute(f"SELECT setseed({seed / 2**31:.10f})")

    t0 = time.time()

    # ------------------------------------------------------------------
    # Step 1: Load real users and build their segments
    # ------------------------------------------------------------------
    log.info("[1/7] Loading real users and segmenting...")
    con.execute(f"""
        CREATE TABLE real_users AS
        SELECT
            row_number() OVER () - 1   AS user_id,
            "Current Age"               AS age,
            "Yearly Income - Person"    AS yearly_income
        FROM read_parquet('{REAL_USERS_PATH}')
    """)

    con.execute("""
        CREATE TABLE real_income_quantiles AS
        SELECT
            QUANTILE_CONT(yearly_income, 0.25) AS q25,
            QUANTILE_CONT(yearly_income, 0.50) AS q50,
            QUANTILE_CONT(yearly_income, 0.75) AS q75
        FROM real_users
    """)

    # ------------------------------------------------------------------
    # Step 2: Aggregate real transactions for last full year per user
    #         Pre-compute day offsets here (only 2000 users, cheap)
    # ------------------------------------------------------------------
    log.info("[2/7] Aggregating real transactions for year %d...", real_year)

    con.execute(f"""
        CREATE TABLE real_txns_year AS
        SELECT
            user_id,
            EXTRACT(HOUR FROM time)::INT AS hour,
            amount,
            mcc::INT AS mcc,
            make_date(CAST(year AS INT), CAST(month AS INT), CAST(day AS INT)) AS txn_date
        FROM read_parquet('{REAL_TXN_PATH}')
        WHERE year = {real_year}
    """)

    con.execute("""
        CREATE TABLE real_user_txn_counts AS
        SELECT user_id, COUNT(*) AS txn_count
        FROM real_txns_year
        GROUP BY user_id
    """)

    con.execute("""
        CREATE TABLE real_activity_quantiles AS
        SELECT
            QUANTILE_CONT(txn_count, 0.25) AS q25,
            QUANTILE_CONT(txn_count, 0.50) AS q50,
            QUANTILE_CONT(txn_count, 0.75) AS q75
        FROM real_user_txn_counts
    """)

    # Build per-user sequences with pre-computed day offsets from max date
    # day_offset = days between this txn and the user's last txn (integer >= 0)
    # This is done ONCE for 2000 users, so the expensive date math is cheap
    con.execute("""
        CREATE TABLE real_user_sequences AS
        WITH per_user AS (
            SELECT
                user_id,
                txn_date,
                hour,
                amount,
                mcc,
                MAX(txn_date) OVER (PARTITION BY user_id) AS max_txn_date
            FROM real_txns_year
        )
        SELECT
            user_id,
            LIST(amount ORDER BY txn_date, hour)   AS amounts,
            LIST(mcc ORDER BY txn_date, hour)       AS mccs,
            LIST(hour ORDER BY txn_date, hour)      AS hours,
            LIST(datediff('day', txn_date, max_txn_date) ORDER BY txn_date, hour)
                AS day_offsets,
            COUNT(*)                                 AS seq_len
        FROM per_user
        GROUP BY user_id
    """)

    # ------------------------------------------------------------------
    # Step 3: Build segment labels for real users
    # ------------------------------------------------------------------
    log.info("[3/7] Building segment labels for real users...")
    con.execute("""
        CREATE TABLE real_user_segments AS
        SELECT
            ru.user_id,
            CASE
                WHEN ru.age < 25 THEN 1
                WHEN ru.age < 35 THEN 2
                WHEN ru.age < 45 THEN 3
                WHEN ru.age < 55 THEN 4
                WHEN ru.age < 65 THEN 5
                ELSE 6
            END AS seg_age,
            CASE
                WHEN ru.yearly_income < riq.q25 THEN 1
                WHEN ru.yearly_income < riq.q50 THEN 2
                WHEN ru.yearly_income < riq.q75 THEN 3
                ELSE 4
            END AS seg_income,
            CASE
                WHEN COALESCE(tc.txn_count, 0) < raq.q25 THEN 1
                WHEN COALESCE(tc.txn_count, 0) < raq.q50 THEN 2
                WHEN COALESCE(tc.txn_count, 0) < raq.q75 THEN 3
                ELSE 4
            END AS seg_activity,
            COALESCE(tc.txn_count, 0) AS txn_count
        FROM real_users ru
        CROSS JOIN real_income_quantiles riq
        CROSS JOIN real_activity_quantiles raq
        LEFT JOIN real_user_txn_counts tc ON ru.user_id = tc.user_id
    """)

    seg_age_labels = {1: '0-25', 2: '25-35', 3: '35-45', 4: '45-55', 5: '55-65', 6: '65+'}
    seg_inc_labels = {1: 'inc_low', 2: 'inc_mid_low', 3: 'inc_mid_high', 4: 'inc_high'}
    seg_act_labels = {1: 'low', 2: 'medium', 3: 'high', 4: 'very_high'}

    pool_info = con.execute("""
        SELECT seg_age, seg_income, seg_activity, COUNT(*) AS pool_size
        FROM real_user_segments
        GROUP BY seg_age, seg_income, seg_activity
        ORDER BY pool_size
    """).fetchall()
    total_segments = len(pool_info)
    small_segments = sum(1 for r in pool_info if r[3] < 5)
    log.info("  Total segments: %d  |  Segments with <5 users: %d", total_segments, small_segments)
    for row in pool_info:
        if row[3] < 5:
            log.warning("  SMALL POOL: (%s, %s, %s) -> %d users",
                        seg_age_labels[row[0]], seg_inc_labels[row[1]],
                        seg_act_labels[row[2]], row[3])

    # ------------------------------------------------------------------
    # Step 4: Build segment labels for Santander customers
    # ------------------------------------------------------------------
    log.info("[4/7] Loading Santander customers and segmenting...")

    con.execute(f"""
        CREATE TABLE santander AS
        SELECT *
        FROM read_parquet('{SANTANDER_PATH}')
    """)

    con.execute("""
        CREATE TABLE sant_income_quantiles AS
        SELECT
            QUANTILE_CONT(income, 0.25) AS q25,
            QUANTILE_CONT(income, 0.50) AS q50,
            QUANTILE_CONT(income, 0.75) AS q75
        FROM santander
        WHERE income IS NOT NULL AND income > 0
    """)

    con.execute("""
        CREATE TABLE sant_activity_quantiles AS
        SELECT
            QUANTILE_CONT(synth_monthly_txns, 0.25) AS q25,
            QUANTILE_CONT(synth_monthly_txns, 0.50) AS q50,
            QUANTILE_CONT(synth_monthly_txns, 0.75) AS q75
        FROM santander
        WHERE synth_monthly_txns IS NOT NULL
    """)

    con.execute("""
        CREATE TABLE sant_segments AS
        SELECT
            s.customer_id,
            s.snapshot_date,
            s.synth_monthly_txns,
            CASE
                WHEN s.age < 25 THEN 1
                WHEN s.age < 35 THEN 2
                WHEN s.age < 45 THEN 3
                WHEN s.age < 55 THEN 4
                WHEN s.age < 65 THEN 5
                ELSE 6
            END AS seg_age,
            CASE
                WHEN s.income IS NULL OR s.income <= 0 THEN 1
                WHEN s.income < siq.q25 THEN 1
                WHEN s.income < siq.q50 THEN 2
                WHEN s.income < siq.q75 THEN 3
                ELSE 4
            END AS seg_income,
            CASE
                WHEN s.synth_monthly_txns < saq.q25 THEN 1
                WHEN s.synth_monthly_txns < saq.q50 THEN 2
                WHEN s.synth_monthly_txns < saq.q75 THEN 3
                ELSE 4
            END AS seg_activity
        FROM santander s
        CROSS JOIN sant_income_quantiles siq
        CROSS JOIN sant_activity_quantiles saq
    """)

    sant_count = con.execute("SELECT COUNT(*) FROM sant_segments").fetchone()[0]
    log.info("  Santander customers segmented: %d", sant_count)

    # ------------------------------------------------------------------
    # Step 5: Build real user pool with segment numbering
    # ------------------------------------------------------------------
    log.info("[5/7] Building real user pool with segment indices...")

    con.execute("""
        CREATE TABLE real_pool AS
        SELECT
            rus.user_id,
            rs.seg_age,
            rs.seg_income,
            rs.seg_activity,
            row_number() OVER (
                PARTITION BY rs.seg_age, rs.seg_income, rs.seg_activity
                ORDER BY rus.user_id
            ) - 1 AS idx_in_seg,
            COUNT(*) OVER (
                PARTITION BY rs.seg_age, rs.seg_income, rs.seg_activity
            ) AS seg_size
        FROM real_user_sequences rus
        JOIN real_user_segments rs ON rus.user_id = rs.user_id
    """)

    # Fallback pools (only IDs and segment keys, lightweight)
    con.execute("""
        CREATE TABLE real_pool_fb1 AS
        SELECT
            user_id, seg_age, seg_income,
            row_number() OVER (PARTITION BY seg_age, seg_income ORDER BY user_id) - 1 AS idx_fb1,
            COUNT(*) OVER (PARTITION BY seg_age, seg_income) AS size_fb1
        FROM real_user_segments
    """)

    con.execute("""
        CREATE TABLE real_pool_fb2 AS
        SELECT
            user_id, seg_age,
            row_number() OVER (PARTITION BY seg_age ORDER BY user_id) - 1 AS idx_fb2,
            COUNT(*) OVER (PARTITION BY seg_age) AS size_fb2
        FROM real_user_segments
    """)

    # ------------------------------------------------------------------
    # Step 6: Match each Santander customer to a donor
    # ------------------------------------------------------------------
    log.info("[6/7] Matching Santander customers to real user donors...")
    t_match = time.time()

    # Exact segment match
    con.execute(f"""
        CREATE TABLE sant_donor_exact AS
        SELECT
            ss.customer_id,
            rp.user_id AS donor_user_id
        FROM sant_segments ss
        JOIN real_pool rp
            ON  ss.seg_age      = rp.seg_age
            AND ss.seg_income   = rp.seg_income
            AND ss.seg_activity = rp.seg_activity
            AND rp.idx_in_seg   = ABS(hash(ss.customer_id + {seed})) % rp.seg_size
    """)
    exact_ct = con.execute("SELECT COUNT(*) FROM sant_donor_exact").fetchone()[0]
    log.info("  Exact segment match: %d / %d", exact_ct, sant_count)

    # Fallback 1: age + income
    con.execute(f"""
        CREATE TABLE sant_donor_fb1 AS
        SELECT
            ss.customer_id,
            rp.user_id AS donor_user_id
        FROM sant_segments ss
        LEFT JOIN sant_donor_exact e ON ss.customer_id = e.customer_id
        JOIN real_pool_fb1 rp
            ON  ss.seg_age    = rp.seg_age
            AND ss.seg_income = rp.seg_income
            AND rp.idx_fb1    = ABS(hash(ss.customer_id + {seed})) % rp.size_fb1
        WHERE e.customer_id IS NULL
    """)
    fb1_ct = con.execute("SELECT COUNT(*) FROM sant_donor_fb1").fetchone()[0]
    log.info("  Fallback age+income match: %d", fb1_ct)

    # Fallback 2: age only
    con.execute(f"""
        CREATE TABLE sant_donor_fb2 AS
        SELECT
            ss.customer_id,
            rp.user_id AS donor_user_id
        FROM sant_segments ss
        LEFT JOIN sant_donor_exact e ON ss.customer_id = e.customer_id
        LEFT JOIN sant_donor_fb1 f1 ON ss.customer_id = f1.customer_id
        JOIN real_pool_fb2 rp
            ON  ss.seg_age = rp.seg_age
            AND rp.idx_fb2 = ABS(hash(ss.customer_id + {seed})) % rp.size_fb2
        WHERE e.customer_id IS NULL AND f1.customer_id IS NULL
    """)
    fb2_ct = con.execute("SELECT COUNT(*) FROM sant_donor_fb2").fetchone()[0]
    log.info("  Fallback age-only match: %d", fb2_ct)

    # Combine
    con.execute("""
        CREATE TABLE sant_donor AS
        SELECT customer_id, donor_user_id FROM sant_donor_exact
        UNION ALL
        SELECT customer_id, donor_user_id FROM sant_donor_fb1
        UNION ALL
        SELECT customer_id, donor_user_id FROM sant_donor_fb2
    """)

    matched_count = con.execute("SELECT COUNT(*) FROM sant_donor").fetchone()[0]
    log.info("  Total matched: %d / %d (%.1f%%)", matched_count, sant_count,
             100 * matched_count / sant_count)

    # Global fallback for any remaining
    unmatched = sant_count - matched_count
    if unmatched > 0:
        log.warning("  %d unmatched -- assigning random global donors", unmatched)
        total_pool = con.execute("SELECT COUNT(*) FROM real_user_sequences").fetchone()[0]
        con.execute(f"""
            INSERT INTO sant_donor
            SELECT
                ss.customer_id,
                (ABS(hash(ss.customer_id + {seed})) % {total_pool})::INT
            FROM sant_segments ss
            LEFT JOIN sant_donor sd ON ss.customer_id = sd.customer_id
            WHERE sd.customer_id IS NULL
        """)

    log.info("  Matching took %.1fs", time.time() - t_match)

    # ------------------------------------------------------------------
    # Step 6b: Join donors with sequences and apply noise via list_transform
    #          Key optimization: day_offsets are pre-computed integers,
    #          so the lambda only does integer/float arithmetic (no date parsing)
    # ------------------------------------------------------------------
    log.info("  Applying noise and frequency scaling via list transforms...")
    t_noise = time.time()

    # Join: 941K customers -> their donor's pre-computed sequences
    con.execute("""
        CREATE TABLE augmented_raw AS
        SELECT
            sd.customer_id,
            CAST(ss.snapshot_date AS DATE) AS snap_date,
            ss.synth_monthly_txns,
            rus.amounts   AS d_amt,
            rus.mccs      AS d_mcc,
            rus.hours     AS d_hr,
            rus.day_offsets AS d_off,
            rus.seq_len   AS d_len
        FROM sant_donor sd
        JOIN sant_segments ss ON sd.customer_id = ss.customer_id
        JOIN real_user_sequences rus ON sd.donor_user_id = rus.user_id
    """)

    aug_count = con.execute("SELECT COUNT(*) FROM augmented_raw").fetchone()[0]
    log.info("  Augmented sequences joined: %d", aug_count)

    # Process in batches to avoid DuckDB list size limits (2^32 elements)
    BATCH_SIZE = 100000
    con.execute("CREATE TABLE final_seqs (customer_id BIGINT, txn_amount_seq DOUBLE[], txn_mcc_seq INT[], txn_hour_seq INT[], txn_date_seq INT[])")

    # Add a batch key via hash-based bucketing on customer_id
    n_batches = (aug_count // BATCH_SIZE) + 1
    con.execute(f"""
        ALTER TABLE augmented_raw ADD COLUMN batch_id INT
    """)
    con.execute(f"""
        UPDATE augmented_raw SET batch_id = ABS(hash(customer_id)) % {n_batches}
    """)
    log.info("  Processing %d batches of ~%d customers...", n_batches, BATCH_SIZE)

    for b in range(n_batches):
        con.execute(f"""
            INSERT INTO final_seqs
            WITH src AS (
                SELECT * FROM augmented_raw WHERE batch_id = {b}
            ),
            zipped AS (
                SELECT
                    customer_id,
                    snap_date,
                    d_len,
                    GREATEST(synth_monthly_txns * 12, 1) AS target_annual,
                    list_zip(
                        generate_series(1, d_len),
                        d_amt, d_mcc, d_hr, d_off
                    ) AS zdata
                FROM src
            ),
            filtered AS (
                SELECT
                    customer_id,
                    snap_date,
                    d_len,
                    target_annual,
                    list_filter(
                        zdata,
                        el -> (target_annual >= d_len)
                            OR (ABS(hash(customer_id * 100000 + el[1] + {seed})) % 10000
                                < (LEAST(target_annual * 10000.0
                                         / GREATEST(d_len, 1), 10000))::INT)
                    ) AS kept
                FROM zipped
            )
            SELECT
                customer_id,

                -- Amount noise: amt * (1 + std * box_muller)
                list_transform(
                    kept,
                    el -> ROUND(
                        el[2] * (1.0 + {amount_std} * (
                            SQRT(-2.0 * LN(GREATEST(
                                (ABS(hash(customer_id::BIGINT * 1000003 + el[1] * 7 + {seed}))
                                 % 10000) / 10000.0, 0.0001)))
                            * COS(2.0 * PI() * (
                                (ABS(hash(customer_id::BIGINT * 1000003 + el[1] * 13 + {seed + 1}))
                                 % 10000) / 10000.0))
                        )), 2)
                ) AS txn_amount_seq,

                -- MCC swap: 10% chance to shift MCC by small offset
                list_transform(
                    kept,
                    el -> CASE
                        WHEN (ABS(hash(customer_id::BIGINT * 1000003 + el[1] * 17 + {seed + 2}))
                              % 100) < {int(mcc_swap_prob * 100)}
                        THEN CASE
                            WHEN (ABS(hash(customer_id::BIGINT * 1000003 + el[1] * 19 + {seed + 5}))
                                  % 2) = 0
                            THEN el[3] + (ABS(hash(el[3]::BIGINT * 31 + {seed + 6})) % 5 + 1)::INT
                            ELSE GREATEST(
                                el[3] - (ABS(hash(el[3]::BIGINT * 37 + {seed + 7})) % 5 + 1)::INT,
                                1000)
                        END
                        ELSE el[3]
                    END
                ) AS txn_mcc_seq,

                -- Hours: no noise
                list_transform(kept, el -> el[4]) AS txn_hour_seq,

                -- Date shift: snap_date - noised_offset -> YYYYMMDD int
                list_transform(
                    kept,
                    el -> (STRFTIME(
                        snap_date - (
                            GREATEST(0, LEAST(ROUND(
                                COALESCE(el[5], 0) * (1.0 + {gap_std} * (
                                    SQRT(-2.0 * LN(GREATEST(
                                        (ABS(hash(customer_id::BIGINT * 1000003 + el[1] * 23
                                                  + {seed + 3})) % 10000) / 10000.0,
                                        0.0001)))
                                    * SIN(2.0 * PI() * (
                                        (ABS(hash(customer_id::BIGINT * 1000003 + el[1] * 29
                                                  + {seed + 4})) % 10000) / 10000.0))
                                )), 3650))::INT
                        ) * INTERVAL '1 DAY',
                        '%Y%m%d'
                    ))::INT
                ) AS txn_date_seq

            FROM filtered
        """)
        if (b + 1) % 3 == 0 or b == n_batches - 1:
            log.info("    Batch %d/%d done", b + 1, n_batches)

    log.info("  Noise + transforms took %.1fs", time.time() - t_noise)

    # ------------------------------------------------------------------
    # Step 7: Join back to original Santander data and write output
    # ------------------------------------------------------------------
    log.info("[7/7] Joining augmented sequences and writing output...")

    con.execute(f"""
        COPY (
            SELECT
                s.* EXCLUDE (txn_amount_seq, txn_mcc_seq, txn_hour_seq),
                COALESCE(fs.txn_amount_seq, s.txn_amount_seq) AS txn_amount_seq,
                COALESCE(fs.txn_mcc_seq, s.txn_mcc_seq)      AS txn_mcc_seq,
                COALESCE(fs.txn_hour_seq, s.txn_hour_seq)     AS txn_hour_seq,
                fs.txn_date_seq                                AS txn_date_seq
            FROM santander s
            LEFT JOIN final_seqs fs ON s.customer_id = fs.customer_id
            ORDER BY s.customer_id
        ) TO '{OUTPUT_PATH}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info("Done in %.1fs", elapsed)
    log.info("Output: %s", OUTPUT_PATH)

    # Quick validation stats
    stats = con.execute(f"""
        SELECT
            COUNT(*)                    AS total_rows,
            COUNT(txn_date_seq)         AS rows_with_date_seq,
            AVG(len(txn_amount_seq))    AS avg_seq_len,
            MIN(len(txn_amount_seq))    AS min_seq_len,
            MAX(len(txn_amount_seq))    AS max_seq_len
        FROM read_parquet('{OUTPUT_PATH}')
    """).fetchone()
    log.info("  total_rows=%d  rows_with_date_seq=%d", stats[0], stats[1])
    log.info("  seq_len: avg=%.1f  min=%d  max=%d", stats[2], stats[3], stats[4])

    con.close()


if __name__ == "__main__":
    main()
