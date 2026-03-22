# -*- coding: utf-8 -*-
"""
Temporal Feature Extraction v4 — 하이브리드 전략
──────────────────────────────────────────────────
DuckDB로 월별 CSV 스캔 → Python dict에 고객별 시퀀스 누적 → Parquet 출력
GROUP BY 없이 메모리 효율적으로 94만 고객 × 17개월 처리

핵심: DuckDB는 CSV 읽기 엔진으로만 사용, 집계는 Python dict
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import os
import time

BASE = 'E:/AIOps_project_AWS/data'
SANTANDER_CSV = f'{BASE}/raw/santander/train_ver2.csv'
EXISTING_PARQUET = f'{BASE}/synthetic/santander_linked.parquet'
OUTPUT_PATH = f'{BASE}/synthetic/santander_linked_v2.parquet'

PRODUCTS = [
    'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
    'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
    'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
    'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
    'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
    'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
    'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
    'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1',
]
PROD_ALIASES = [
    'saving', 'guarantee', 'checking', 'derivados', 'payroll_acct',
    'junior_acct', 'particular_acct', 'particular_plus', 'short_deposit',
    'medium_deposit', 'long_deposit', 'e_account', 'funds', 'mortgage',
    'pension_plan', 'loans', 'taxes', 'credit_card', 'securities',
    'home_acct', 'payroll', 'pension_deposit', 'direct_debit', 'auto_debit',
]

print("=" * 60)
print("Temporal Feature Extraction v4 — 하이브리드")
print("  DuckDB(CSV 읽기) + Python dict(집계) + PyArrow(저장)")
print("=" * 60)

# ============================================================
# Phase 1: 대상 고객 ID 집합 로드
# ============================================================
print("\n[Phase 1] 대상 고객 ID 로드...")
t0 = time.time()

con = duckdb.connect()
con.execute("SET enable_progress_bar=false")
customer_ids = set(
    con.execute(f"SELECT customer_id FROM read_parquet('{EXISTING_PARQUET}')").fetchnumpy()['customer_id']
)
print(f"  대상 고객: {len(customer_ids):,} ({time.time()-t0:.1f}s)")

# ============================================================
# Phase 2: 월별 스캔 → Python dict에 시퀀스 누적
# ============================================================
print("\n[Phase 2] 17개월 상품 보유 시퀀스 수집...")
t0 = time.time()

# 월 목록
months = con.execute(f"""
    SELECT DISTINCT fecha_dato FROM read_csv_auto('{SANTANDER_CSV}', sample_size=2000)
    ORDER BY fecha_dato
""").fetchall()
months = [m[0] for m in months]
print(f"  월: {len(months)}개 ({months[0]} ~ {months[-1]})")

# 고객별 시퀀스 딕셔너리 초기화
# key: customer_id, value: list of (month_idx, num_prods, [p0..p23])
# 메모리 추정: 94만 × 17개월 × 25 int ≈ 400M int × 4B ≈ 1.6GB → 너무 큼
# → 대신 numpy array로 사전 할당

# 고객 ID → 인덱스 매핑
id_list = sorted(customer_ids)
id_to_idx = {cid: i for i, cid in enumerate(id_list)}
n = len(id_list)
n_months = len(months)

# 사전 할당: (n_customers, n_months, 24+1) — 상품24 + 보유수1
# int8로 충분 (0 or 1), -1은 미관측
data = np.full((n, n_months, 25), -1, dtype=np.int8)

prod_select = ', '.join(f"COALESCE(TRY_CAST({p} AS INTEGER), 0)" for p in PRODUCTS)
prod_sum = ' + '.join(f"COALESCE(TRY_CAST({p} AS INTEGER), 0)" for p in PRODUCTS)

for mi, m in enumerate(months):
    # DuckDB로 해당 월 데이터만 읽기 (대상 고객만)
    chunk = con.execute(f"""
        SELECT
            ncodpers,
            {prod_sum} as num_prods,
            {prod_select}
        FROM read_csv_auto('{SANTANDER_CSV}', sample_size=2000)
        WHERE fecha_dato = '{m}'
    """).fetchnumpy()

    ids = chunk['ncodpers']
    # numpy 배열에 채우기
    matched = 0
    for row_i in range(len(ids)):
        cid = int(ids[row_i])
        if cid in id_to_idx:
            idx = id_to_idx[cid]
            data[idx, mi, 0] = int(chunk['num_prods'][row_i])
            for pi in range(24):
                col_key = list(chunk.keys())[pi + 2]
                val = chunk[col_key][row_i]
                data[idx, mi, pi + 1] = int(val) if val is not None and not np.isnan(val) else 0
            matched += 1

    if (mi + 1) % 5 == 0 or mi == n_months - 1:
        print(f"  {m} ({mi+1}/{n_months}) — 매칭: {matched:,}")

print(f"  수집 완료 ({time.time()-t0:.1f}s)")
print(f"  numpy 배열: {data.nbytes / 1024 / 1024:.1f}MB")

# DuckDB 연결 해제 (메모리 확보)
con.close()

# ============================================================
# Phase 3: 파생 피처 계산
# ============================================================
print("\n[Phase 3] 파생 피처 계산...")
t0 = time.time()

# 관측 월 수
observed = (data[:, :, 0] != -1).sum(axis=1).astype(np.int16)

# 월별 보유 상품 수 시퀀스 (관측된 것만)
monthly_prods_list = []
for i in range(n):
    mask = data[i, :, 0] != -1
    monthly_prods_list.append(data[i, mask, 0].astype(np.int32).tolist())

# 24개 상품별 상태 시퀀스
state_seqs = {}
for pi, alias in enumerate(PROD_ALIASES):
    seq_list = []
    for i in range(n):
        mask = data[i, :, 0] != -1
        seq_list.append(data[i, mask, pi + 1].astype(np.int32).tolist())
    state_seqs[alias] = seq_list

# NBA 레이블: 마지막 관측 월에 0→1 전환
nba = {}
for pi, alias in enumerate(PROD_ALIASES):
    nba_arr = np.zeros(n, dtype=np.int8)
    for i in range(n):
        seq = state_seqs[alias][i]
        if len(seq) >= 2 and seq[-1] == 1 and seq[-2] == 0:
            nba_arr[i] = 1
    nba[alias] = nba_arr

# 집계 피처
first_prods = np.array([s[0] if len(s) > 0 else 0 for s in monthly_prods_list], dtype=np.int16)
last_prods = np.array([s[-1] if len(s) > 0 else 0 for s in monthly_prods_list], dtype=np.int16)
avg_prods = np.array([np.mean(s) if len(s) > 0 else 0 for s in monthly_prods_list], dtype=np.float32)
growth = np.where(first_prods > 0, (last_prods - first_prods) / first_prods, 0.0).astype(np.float32)
nba_count = sum(nba[a] for a in PROD_ALIASES).astype(np.int16)

# 상태 전이 횟수 (전체 상품 합산)
total_transitions = np.zeros(n, dtype=np.int16)
for alias in PROD_ALIASES:
    for i in range(n):
        seq = state_seqs[alias][i]
        for j in range(1, len(seq)):
            if seq[j] != seq[j-1]:
                total_transitions[i] += 1

# 안정성 (전이 없을수록 안정)
stability = np.where(observed > 1, 1.0 - total_transitions / (observed * 24), 1.0).astype(np.float32)

print(f"  NBA 있는 고객: {(nba_count > 0).sum():,} ({(nba_count > 0).sum()/n*100:.1f}%)")
print(f"  평균 관측 월수: {observed.mean():.1f}")
print(f"  평균 전이 횟수: {total_transitions.mean():.1f}")
print(f"  ({time.time()-t0:.1f}s)")

# numpy 원본 배열 해제
del data

# ============================================================
# Phase 4: 기존 parquet 읽고 합병 → 새 parquet 출력
# ============================================================
print("\n[Phase 4] 기존 데이터와 합병...")
t0 = time.time()

existing_table = pq.read_table(EXISTING_PARQUET)
existing_ids = existing_table.column('customer_id').to_numpy()

# 기존 parquet의 고객 순서에 맞춰 인덱스 매핑
idx_map = np.array([id_to_idx[int(cid)] for cid in existing_ids])

# 새 컬럼 추가
new_columns = {
    'n_months_observed': pa.array(observed[idx_map], type=pa.int16()),
    'avg_monthly_products': pa.array(avg_prods[idx_map], type=pa.float32()),
    'first_month_products': pa.array(first_prods[idx_map], type=pa.int16()),
    'last_month_products': pa.array(last_prods[idx_map], type=pa.int16()),
    'product_growth_rate': pa.array(growth[idx_map], type=pa.float32()),
    'product_stability': pa.array(stability[idx_map], type=pa.float32()),
    'total_transitions': pa.array(total_transitions[idx_map], type=pa.int16()),
    'nba_count': pa.array(nba_count[idx_map], type=pa.int16()),
    'monthly_num_products': pa.array([monthly_prods_list[i] for i in idx_map], type=pa.list_(pa.int32())),
}

# NBA 레이블 24개
for alias in PROD_ALIASES:
    new_columns[f'nba_{alias}'] = pa.array(nba[alias][idx_map], type=pa.int8())

# 상품별 상태 시퀀스 24개
for alias in PROD_ALIASES:
    new_columns[f'state_seq_{alias}'] = pa.array(
        [state_seqs[alias][i] for i in idx_map],
        type=pa.list_(pa.int32())
    )

# 기존 테이블에 새 컬럼 추가
for col_name, col_data in new_columns.items():
    existing_table = existing_table.append_column(col_name, col_data)

print(f"  합병: {existing_table.num_rows:,}행 × {existing_table.num_columns}컬럼")

# Parquet 저장
print("\n[Phase 5] Parquet 저장...")
pq.write_table(
    existing_table, OUTPUT_PATH,
    compression='zstd',
    row_group_size=100000,
)

fsize = os.path.getsize(OUTPUT_PATH) / (1024*1024)
print(f"  {OUTPUT_PATH}")
print(f"  {fsize:.1f}MB | {existing_table.num_rows:,}행 | {existing_table.num_columns}컬럼")

# ============================================================
# 요약
# ============================================================
print("\n" + "=" * 60)
print("최종 데이터셋 v2 요약")
print("=" * 60)

col_cats = {'profile': 0, 'product': 0, 'synth': 0, 'txn_tensor': 0,
            'temporal': 0, 'nba': 0, 'state_seq': 0}
for c in existing_table.column_names:
    if c.startswith('prod_'): col_cats['product'] += 1
    elif c.startswith('synth_'): col_cats['synth'] += 1
    elif c.startswith('txn_'): col_cats['txn_tensor'] += 1
    elif c.startswith('nba_'): col_cats['nba'] += 1
    elif c.startswith('state_seq_'): col_cats['state_seq'] += 1
    elif c in ('n_months_observed','avg_monthly_products','first_month_products',
               'last_month_products','product_growth_rate','product_stability',
               'total_transitions','monthly_num_products'):
        col_cats['temporal'] += 1
    else: col_cats['profile'] += 1

print(f"  프로필:              {col_cats['profile']}개")
print(f"  상품 보유 (현재):     {col_cats['product']}개")
print(f"  합성 거래 집계:       {col_cats['synth']}개")
print(f"  거래 시퀀스 (LIST):   {col_cats['txn_tensor']}개")
print(f"  시계열 집계:          {col_cats['temporal']}개")
print(f"  NBA 레이블:           {col_cats['nba']}개 (+ nba_count)")
print(f"  상품 상태 시퀀스:     {col_cats['state_seq']}개 (LIST)")
print(f"  ────────────────────────────────")
print(f"  총 컬럼:             {existing_table.num_columns}개")

print("\n완료!")
