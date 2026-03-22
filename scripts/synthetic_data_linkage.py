# -*- coding: utf-8 -*-
"""
Synthetic Data Linkage v2 — 극강 최적화
────────────────────────────────────────
메모리 1.5GB + E드라이브 스필, 월별 분할 스캔으로 130만 고객 전체 확보
Santander (프로필 + 24상품) × ealtman (거래 시계열) → Lagged Tensor Parquet
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import duckdb
import os
import time

# ============================================================
# 경로
# ============================================================
BASE = 'E:/AIOps_project_AWS/data'
SANTANDER_CSV = f'{BASE}/raw/santander/train_ver2.csv'
EALTMAN_TXN = f'{BASE}/raw/ealtman2019credit-card-transactions/credit_card_transactions-ibm_v2.csv'
EALTMAN_USERS = f'{BASE}/raw/ealtman2019credit-card-transactions/sd254_users.csv'
OUTPUT_DIR = f'{BASE}/synthetic'
SPILL_DIR = 'E:/duckdb_spill'
DB_PATH = f'{OUTPUT_DIR}/linkage_v2.duckdb'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SPILL_DIR, exist_ok=True)

# 이전 DB 삭제 (클린 스타트)
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

con = duckdb.connect(DB_PATH)
con.execute(f"SET temp_directory='{SPILL_DIR}'")
con.execute("SET memory_limit='1500MB'")
con.execute("SET threads=2")
con.execute("SET preserve_insertion_order=false")
con.execute("SET enable_progress_bar=false")  # 프로그레스 바 끄기 (메모리 절약)

print("=" * 60)
print("Synthetic Data Linkage v2 — 극강 최적화")
print("  메모리: 1.5GB | 스필: E:/duckdb_spill | 스레드: 2")
print("=" * 60)

# ============================================================
# Phase 1: Santander 월별 분할 스캔 → 고객별 최신 스냅샷
# ============================================================
print("\n[Phase 1] Santander 130만 고객 추출 (월별 분할)...")
t0 = time.time()

# 1-a: 어떤 월이 있는지 먼저 확인 (가벼운 DISTINCT)
months = con.execute(f"""
    SELECT DISTINCT fecha_dato
    FROM read_csv_auto('{SANTANDER_CSV}', sample_size=2000)
    ORDER BY fecha_dato
""").fetchall()
months = [m[0] for m in months]
print(f"  발견된 월: {len(months)}개 ({months[0]} ~ {months[-1]})")

# 1-b: 빈 테이블 생성
con.execute("CREATE TABLE IF NOT EXISTS santander_best (customer_id BIGINT PRIMARY KEY, snapshot_date VARCHAR)")

# 1-c: 역순(최신→과거)으로 월별 스캔, 이미 있는 고객은 스킵
for i, m in enumerate(reversed(months)):
    con.execute(f"""
        INSERT OR IGNORE INTO santander_best
        SELECT ncodpers, '{m}'
        FROM read_csv_auto('{SANTANDER_CSV}', sample_size=2000)
        WHERE fecha_dato = '{m}'
          AND ncodpers IS NOT NULL
    """)
    cnt = con.execute("SELECT COUNT(*) FROM santander_best").fetchone()[0]
    if (i + 1) % 5 == 0 or i == 0:
        print(f"  {m} 처리 완료 — 누적 고객: {cnt:,}")

total_customers = con.execute("SELECT COUNT(*) FROM santander_best").fetchone()[0]
print(f"  전체 고유 고객: {total_customers:,} ({time.time()-t0:.1f}s)")

# 1-d: 최신 스냅샷 기반 프로필 추출 (월별 분할 조인)
print("\n[Phase 2] 프로필 + 상품 보유 추출...")
t0 = time.time()

PRODUCT_COLS = [
    ('ind_ahor_fin_ult1', 'prod_saving'), ('ind_aval_fin_ult1', 'prod_guarantee'),
    ('ind_cco_fin_ult1', 'prod_checking'), ('ind_cder_fin_ult1', 'prod_derivados'),
    ('ind_cno_fin_ult1', 'prod_payroll_acct'), ('ind_ctju_fin_ult1', 'prod_junior_acct'),
    ('ind_ctma_fin_ult1', 'prod_particular_acct'), ('ind_ctop_fin_ult1', 'prod_particular_plus'),
    ('ind_ctpp_fin_ult1', 'prod_short_deposit'), ('ind_deco_fin_ult1', 'prod_medium_deposit'),
    ('ind_deme_fin_ult1', 'prod_long_deposit'), ('ind_dela_fin_ult1', 'prod_e_account'),
    ('ind_ecue_fin_ult1', 'prod_funds'), ('ind_fond_fin_ult1', 'prod_mortgage'),
    ('ind_hip_fin_ult1', 'prod_pension_plan'), ('ind_plan_fin_ult1', 'prod_loans'),
    ('ind_pres_fin_ult1', 'prod_taxes'), ('ind_reca_fin_ult1', 'prod_credit_card'),
    ('ind_tjcr_fin_ult1', 'prod_securities'), ('ind_valo_fin_ult1', 'prod_home_acct'),
    ('ind_viv_fin_ult1', 'prod_payroll'), ('ind_nomina_ult1', 'prod_pension_deposit'),
    ('ind_nom_pens_ult1', 'prod_direct_debit'), ('ind_recibo_ult1', 'prod_auto_debit'),
]
prod_select = ', '.join(f"COALESCE(TRY_CAST(r.{src} AS INTEGER), 0) as {dst}" for src, dst in PRODUCT_COLS)
prod_sum = ' + '.join(dst for _, dst in PRODUCT_COLS)

# 월별로 분할 INSERT (메모리 폭발 방지)
con.execute(f"""
    CREATE TABLE customers (
        customer_id BIGINT PRIMARY KEY,
        snapshot_date VARCHAR,
        gender VARCHAR,
        age INTEGER,
        income DOUBLE,
        segment VARCHAR,
        country VARCHAR,
        channel VARCHAR,
        tenure_months INTEGER,
        is_active INTEGER,
        {', '.join(f'{dst} INTEGER' for _, dst in PRODUCT_COLS)},
        age_group VARCHAR,
        income_group VARCHAR,
        num_products INTEGER
    )
""")

distinct_months = con.execute("SELECT DISTINCT snapshot_date FROM santander_best ORDER BY snapshot_date").fetchall()
for m_row in distinct_months:
    m = m_row[0]
    con.execute(f"""
        INSERT OR IGNORE INTO customers
        SELECT
            r.ncodpers,
            '{m}',
            CASE WHEN r.sexo = 'H' THEN 'M' ELSE 'F' END,
            TRY_CAST(r.age AS INTEGER),
            COALESCE(TRY_CAST(r.renta AS DOUBLE), 0),
            COALESCE(r.segmento, 'UNKNOWN'),
            COALESCE(r.pais_residencia, 'ES'),
            COALESCE(r.canal_entrada, 'UNK'),
            TRY_CAST(r.antiguedad AS INTEGER),
            TRY_CAST(r.ind_actividad_cliente AS INTEGER),
            {prod_select},
            CASE
                WHEN TRY_CAST(r.age AS INTEGER) < 25 THEN 'young'
                WHEN TRY_CAST(r.age AS INTEGER) < 35 THEN 'adult'
                WHEN TRY_CAST(r.age AS INTEGER) < 50 THEN 'middle'
                WHEN TRY_CAST(r.age AS INTEGER) < 65 THEN 'senior'
                ELSE 'elderly'
            END,
            CASE
                WHEN COALESCE(TRY_CAST(r.renta AS DOUBLE), 0) <= 0 THEN 'unknown'
                WHEN TRY_CAST(r.renta AS DOUBLE) < 50000 THEN 'low'
                WHEN TRY_CAST(r.renta AS DOUBLE) < 100000 THEN 'mid'
                WHEN TRY_CAST(r.renta AS DOUBLE) < 200000 THEN 'high'
                ELSE 'very_high'
            END,
            {prod_sum}
        FROM read_csv_auto('{SANTANDER_CSV}', sample_size=2000) r
        WHERE r.fecha_dato = '{m}'
          AND r.ncodpers IN (SELECT customer_id FROM santander_best WHERE snapshot_date = '{m}')
          AND r.age IS NOT NULL
          AND TRY_CAST(r.age AS INTEGER) BETWEEN 18 AND 100
    """)

# santander_best 테이블 삭제 (메모리 확보)
con.execute("DROP TABLE santander_best")

cust_count = con.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
seg_dist = con.execute("""
    SELECT age_group, income_group, COUNT(*) as cnt
    FROM customers GROUP BY age_group, income_group ORDER BY cnt DESC LIMIT 8
""").fetchdf()
print(f"  최종 고객 수: {cust_count:,} ({time.time()-t0:.1f}s)")
for _, r in seg_dist.iterrows():
    print(f"    {r['age_group']:8s} × {r['income_group']:10s} : {r['cnt']:>8,}")

# ============================================================
# Phase 3: ealtman 거래 패턴 학습
# ============================================================
print("\n[Phase 3] ealtman 거래 패턴 학습...")
t0 = time.time()

con.execute(f"""
    CREATE TABLE ealtman_user_seg AS
    SELECT
        ROW_NUMBER() OVER (ORDER BY "Person") - 1 as user_id,
        CASE
            WHEN "Current Age" < 25 THEN 'young'
            WHEN "Current Age" < 35 THEN 'adult'
            WHEN "Current Age" < 50 THEN 'middle'
            WHEN "Current Age" < 65 THEN 'senior'
            ELSE 'elderly'
        END as age_group,
        CASE
            WHEN TRY_CAST(REPLACE(REPLACE("Yearly Income - Person",'$',''),',','') AS DOUBLE) IS NULL THEN 'unknown'
            WHEN TRY_CAST(REPLACE(REPLACE("Yearly Income - Person",'$',''),',','') AS DOUBLE) < 50000 THEN 'low'
            WHEN TRY_CAST(REPLACE(REPLACE("Yearly Income - Person",'$',''),',','') AS DOUBLE) < 100000 THEN 'mid'
            WHEN TRY_CAST(REPLACE(REPLACE("Yearly Income - Person",'$',''),',','') AS DOUBLE) < 200000 THEN 'high'
            ELSE 'very_high'
        END as income_group
    FROM read_csv_auto('{EALTMAN_USERS}')
""")

# 월별 통계
con.execute(f"""
    CREATE TABLE monthly_stats AS
    SELECT
        t."User" as user_id,
        t."Year" * 100 + t."Month" as ym,
        COUNT(*) as txn_count,
        AVG(TRY_CAST(REPLACE(REPLACE(t."Amount",'$',''),',','') AS DOUBLE)) as avg_amt,
        STDDEV(TRY_CAST(REPLACE(REPLACE(t."Amount",'$',''),',','') AS DOUBLE)) as std_amt,
        SUM(TRY_CAST(REPLACE(REPLACE(t."Amount",'$',''),',','') AS DOUBLE)) as total_amt,
        COUNT(DISTINCT t."MCC") as uniq_mcc,
        COUNT(DISTINCT t."Merchant Name") as uniq_merchant,
        SUM(CASE WHEN EXTRACT(HOUR FROM t."Time") BETWEEN 6 AND 11 THEN 1 ELSE 0 END) as am_txn,
        SUM(CASE WHEN EXTRACT(HOUR FROM t."Time") BETWEEN 12 AND 17 THEN 1 ELSE 0 END) as pm_txn,
        SUM(CASE WHEN EXTRACT(HOUR FROM t."Time") BETWEEN 18 AND 23 THEN 1 ELSE 0 END) as ev_txn,
        SUM(CASE WHEN EXTRACT(HOUR FROM t."Time") BETWEEN 0 AND 5 THEN 1 ELSE 0 END) as nt_txn,
        SUM(CASE WHEN t."Is Fraud?" = 'Yes' THEN 1 ELSE 0 END) as fraud_cnt
    FROM read_csv_auto('{EALTMAN_TXN}', sample_size=5000) t
    GROUP BY t."User", t."Year", t."Month"
""")
print(f"  월별 통계: {con.execute('SELECT COUNT(*) FROM monthly_stats').fetchone()[0]:,}행 ({time.time()-t0:.1f}s)")

# 세그먼트별 프로필
con.execute("""
    CREATE TABLE seg_profile AS
    SELECT
        u.age_group, u.income_group,
        COUNT(DISTINCT s.user_id) as n_users,
        AVG(s.txn_count) as mu_txns, STDDEV(s.txn_count) as sd_txns,
        AVG(s.avg_amt) as mu_amt, STDDEV(s.avg_amt) as sd_amt,
        AVG(s.total_amt) as mu_spend, STDDEV(s.total_amt) as sd_spend,
        AVG(s.uniq_mcc) as mu_mcc, AVG(s.uniq_merchant) as mu_merch,
        AVG(s.am_txn*1.0/NULLIF(s.txn_count,0)) as r_am,
        AVG(s.pm_txn*1.0/NULLIF(s.txn_count,0)) as r_pm,
        AVG(s.ev_txn*1.0/NULLIF(s.txn_count,0)) as r_ev,
        AVG(s.nt_txn*1.0/NULLIF(s.txn_count,0)) as r_nt,
        AVG(s.fraud_cnt*1.0/NULLIF(s.txn_count,0)) as r_fraud
    FROM monthly_stats s
    JOIN ealtman_user_seg u ON s.user_id = u.user_id
    GROUP BY u.age_group, u.income_group
""")
print(f"  세그먼트 프로필: {con.execute('SELECT COUNT(*) FROM seg_profile').fetchone()[0]}개")

# ============================================================
# Phase 4: Lagged Tensor — 세그먼트별 거래 시퀀스 (LIST)
# ============================================================
print("\n[Phase 4] Lagged Tensor 시퀀스 구축...")
t0 = time.time()

con.execute(f"""
    CREATE TABLE seg_sequences AS
    WITH txn AS (
        SELECT
            t."User" as uid,
            TRY_CAST(REPLACE(REPLACE(t."Amount",'$',''),',','') AS DOUBLE) as amt,
            CAST(t."MCC" AS INTEGER) as mcc,
            CAST(EXTRACT(HOUR FROM t."Time") AS INTEGER) as hr,
            t."Year"*10000 + t."Month"*100 + t."Day" as dt
        FROM read_csv_auto('{EALTMAN_TXN}', sample_size=5000) t
    ),
    txn_seg AS (
        SELECT tx.*, u.age_group, u.income_group
        FROM txn tx JOIN ealtman_user_seg u ON tx.uid = u.user_id
    )
    SELECT
        age_group, income_group,
        LIST(amt ORDER BY dt DESC)[:90] as amt_pool,
        LIST(mcc ORDER BY dt DESC)[:90] as mcc_pool,
        LIST(hr ORDER BY dt DESC)[:90] as hr_pool
    FROM txn_seg
    GROUP BY age_group, income_group
""")

seq_info = con.execute("SELECT age_group, income_group, LENGTH(amt_pool) as n FROM seg_sequences ORDER BY n DESC LIMIT 5").fetchdf()
print(f"  시퀀스 풀 생성 완료 ({time.time()-t0:.1f}s)")
for _, r in seq_info.iterrows():
    print(f"    {r['age_group']:8s} × {r['income_group']:10s} : {r['n']} 거래")

# 불필요 테이블 정리
con.execute("DROP TABLE monthly_stats")
con.execute("DROP TABLE ealtman_user_seg")

# ============================================================
# Phase 5: 합성 결합 — 집계 피처 + Lagged Tensor
# ============================================================
print("\n[Phase 5] 합성 결합...")
t0 = time.time()

con.execute("""
    CREATE TABLE result AS
    SELECT
        c.customer_id, c.snapshot_date, c.gender, c.age, c.income,
        c.segment, c.country, c.channel, c.tenure_months, c.is_active,
        c.age_group, c.income_group, c.num_products,
        -- 24 상품
        c.prod_saving, c.prod_guarantee, c.prod_checking, c.prod_derivados,
        c.prod_payroll_acct, c.prod_junior_acct, c.prod_particular_acct,
        c.prod_particular_plus, c.prod_short_deposit, c.prod_medium_deposit,
        c.prod_long_deposit, c.prod_e_account, c.prod_funds, c.prod_mortgage,
        c.prod_pension_plan, c.prod_loans, c.prod_taxes, c.prod_credit_card,
        c.prod_securities, c.prod_home_acct, c.prod_payroll,
        c.prod_pension_deposit, c.prod_direct_debit, c.prod_auto_debit,

        -- 합성 집계 피처
        GREATEST(1, ROUND(COALESCE(p.mu_txns,30) + (RANDOM()-0.5)*2*COALESCE(p.sd_txns,10)))::INTEGER as synth_monthly_txns,
        GREATEST(1.0, ROUND(COALESCE(p.mu_amt,50) + (RANDOM()-0.5)*2*COALESCE(p.sd_amt,30), 2)) as synth_avg_amount,
        GREATEST(1.0, ROUND(COALESCE(p.mu_spend,1500) + (RANDOM()-0.5)*2*COALESCE(p.sd_spend,500), 2)) as synth_monthly_spend,
        GREATEST(1, ROUND(COALESCE(p.mu_mcc,8) + (RANDOM()-0.5)*4))::INTEGER as synth_unique_mcc,
        GREATEST(1, ROUND(COALESCE(p.mu_merch,15) + (RANDOM()-0.5)*8))::INTEGER as synth_unique_merchants,
        COALESCE(p.r_am, 0.25) as synth_morning_ratio,
        COALESCE(p.r_pm, 0.35) as synth_afternoon_ratio,
        COALESCE(p.r_ev, 0.30) as synth_evening_ratio,
        COALESCE(p.r_nt, 0.10) as synth_night_ratio,
        ROUND(30.0 / GREATEST(1, COALESCE(p.mu_txns,30)), 1) as synth_recency_days,
        GREATEST(1, ROUND(COALESCE(p.mu_txns,30) + (RANDOM()-0.5)*10))::INTEGER as synth_frequency,
        GREATEST(1.0, ROUND(COALESCE(p.mu_spend,1500) + (RANDOM()-0.5)*500, 2)) as synth_monetary,
        CASE WHEN COALESCE(p.sd_amt,0) > 0 THEN ROUND(COALESCE(p.mu_amt,50)/p.sd_amt, 3) ELSE 1.0 END as synth_stability,
        COALESCE(p.r_fraud, 0.001) as synth_fraud_ratio,

        -- Lagged Tensor (LIST) — 세그먼트 시퀀스에서 고객별 랜덤 오프셋 슬라이싱
        COALESCE(
            sq.amt_pool[1 + ABS(HASH(c.customer_id)) % GREATEST(1, LENGTH(sq.amt_pool)-59) :
                        ABS(HASH(c.customer_id)) % GREATEST(1, LENGTH(sq.amt_pool)-59) + 60],
            [0.0]
        ) as txn_amount_seq,
        COALESCE(
            sq.mcc_pool[1 + ABS(HASH(c.customer_id)) % GREATEST(1, LENGTH(sq.mcc_pool)-59) :
                        ABS(HASH(c.customer_id)) % GREATEST(1, LENGTH(sq.mcc_pool)-59) + 60],
            [0]
        ) as txn_mcc_seq,
        COALESCE(
            sq.hr_pool[1 + ABS(HASH(c.customer_id)) % GREATEST(1, LENGTH(sq.hr_pool)-59) :
                       ABS(HASH(c.customer_id)) % GREATEST(1, LENGTH(sq.hr_pool)-59) + 60],
            [0]
        ) as txn_hour_seq

    FROM customers c
    LEFT JOIN seg_profile p ON c.age_group = p.age_group AND c.income_group = p.income_group
    LEFT JOIN seg_sequences sq ON c.age_group = sq.age_group AND c.income_group = sq.income_group
""")

result_count = con.execute("SELECT COUNT(*) FROM result").fetchone()[0]
print(f"  결합 완료: {result_count:,}행 ({time.time()-t0:.1f}s)")

# 정리
con.execute("DROP TABLE customers")
con.execute("DROP TABLE seg_profile")
con.execute("DROP TABLE seg_sequences")

# ============================================================
# Phase 6: Parquet 출력
# ============================================================
print("\n[Phase 6] Parquet 저장...")
t0 = time.time()

out_path = f'{OUTPUT_DIR}/santander_linked.parquet'
con.execute(f"COPY result TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)")

fsize = os.path.getsize(out_path) / (1024*1024)
ncols = len(con.execute("SELECT * FROM result LIMIT 1").description)
print(f"  {out_path}")
print(f"  {fsize:.1f}MB | {result_count:,}행 | {ncols}컬럼 ({time.time()-t0:.1f}s)")

# ============================================================
# 요약
# ============================================================
print("\n" + "=" * 60)
print("합성 데이터 최종 요약")
print("=" * 60)

s = con.execute("""
    SELECT
        COUNT(*) as n,
        AVG(num_products) as avg_prod,
        AVG(synth_monthly_txns) as avg_txns,
        AVG(synth_avg_amount) as avg_amt,
        AVG(synth_monthly_spend) as avg_spend,
        AVG(LENGTH(txn_amount_seq)) as avg_seq_len
    FROM result
""").fetchone()

print(f"  총 고객:        {s[0]:,}")
print(f"  평균 보유 상품:  {s[1]:.1f}개")
print(f"  평균 월 거래:    {s[2]:.1f}건")
print(f"  평균 거래 금액:  ${s[3]:.1f}")
print(f"  평균 월 소비:    ${s[4]:.1f}")
print(f"  평균 시퀀스 길이: {s[5]:.0f}건")

print(f"\n  컬럼 구성:")
cols = con.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name='result' ORDER BY ordinal_position").fetchdf()
profile = [c for c in cols['column_name'] if not c.startswith('prod_') and not c.startswith('synth_') and not c.startswith('txn_')]
products = [c for c in cols['column_name'] if c.startswith('prod_')]
synth = [c for c in cols['column_name'] if c.startswith('synth_')]
tensor = [c for c in cols['column_name'] if c.startswith('txn_')]

print(f"    프로필:        {len(profile)}개")
print(f"    상품 보유:     {len(products)}개")
print(f"    합성 집계:     {len(synth)}개")
print(f"    Lagged Tensor: {len(tensor)}개 (LIST)")
for t in tensor:
    dtype = cols[cols['column_name']==t]['data_type'].values[0]
    print(f"      - {t} ({dtype})")

con.close()

# DB 파일 삭제 (Parquet만 남기기)
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)
# WAL 파일도 정리
for ext in ['.wal']:
    p = DB_PATH + ext
    if os.path.exists(p):
        os.remove(p)

print("\n완료!")
