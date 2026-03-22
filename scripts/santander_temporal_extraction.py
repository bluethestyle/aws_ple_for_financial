# -*- coding: utf-8 -*-
"""
Santander 17개월 시계열 추출 — SageMaker Processing Job용
원본 CSV에서 고객별 상품 보유 변화를 LIST(lagged tensor)로 저장
"""
import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

def main():
    # SageMaker Processing 경로
    input_dir = os.environ.get("SM_INPUT_DIR", "/opt/ml/processing/input")
    output_dir = os.environ.get("SM_OUTPUT_DIR", "/opt/ml/processing/output")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(input_dir, "train_ver2.csv")
    print(f"Reading: {csv_path}")
    print(f"File size: {os.path.getsize(csv_path) / 1024 / 1024:.0f} MB")

    # 프로필 + 상품 컬럼 정의
    profile_cols = [
        'fecha_dato', 'ncodpers', 'ind_empleado', 'pais_residencia', 'sexo',
        'age', 'fecha_alta', 'ind_nuevo', 'antiguedad', 'indrel',
        'ult_fec_cli_1t', 'indrel_1mes', 'tiprel_1mes', 'indresi', 'indext',
        'conyuemp', 'canal_entrada', 'indfall', 'tipodom', 'cod_prov',
        'nomprov', 'ind_actividad_cliente', 'renta', 'segmento'
    ]

    product_cols = [
        'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
        'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
        'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
        'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
        'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
        'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
        'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
        'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1'
    ]

    product_names = [
        'saving', 'guarantee', 'checking', 'derivados', 'payroll_acct',
        'junior_acct', 'particular_acct', 'particular_plus', 'short_deposit',
        'medium_deposit', 'long_deposit', 'e_account', 'funds', 'mortgage',
        'pension_plan', 'loans', 'taxes', 'credit_card', 'securities',
        'home_acct', 'payroll', 'pension_deposit', 'direct_debit', 'auto_debit'  # 수정: 24개 맞춤
    ]

    # =========================================================
    # Step 1: 월별 청크로 읽으면서 고객별 상품 시퀀스 구축
    # =========================================================
    print("\n=== Step 1: 월별 상품 보유 시퀀스 구축 ===")

    # 필요한 컬럼만 읽기 (메모리 절약)
    use_cols = ['fecha_dato', 'ncodpers', 'sexo', 'age', 'antiguedad',
                'ind_actividad_cliente', 'renta', 'segmento', 'canal_entrada',
                'pais_residencia', 'indrel'] + product_cols

    # 청크로 읽어서 월별로 분리
    chunk_size = 500_000
    monthly_data = {}  # {fecha_dato: DataFrame}

    for i, chunk in enumerate(pd.read_csv(csv_path, usecols=use_cols, chunksize=chunk_size)):
        for col in product_cols:
            chunk[col] = pd.to_numeric(chunk[col], errors='coerce').fillna(0).astype(np.int8)

        for fecha, group in chunk.groupby('fecha_dato'):
            if fecha not in monthly_data:
                monthly_data[fecha] = []
            monthly_data[fecha].append(group)

        if (i + 1) % 5 == 0:
            print(f"  Chunk {i+1} processed ({(i+1)*chunk_size:,} rows)")

    # 월별 합치기
    months_sorted = sorted(monthly_data.keys())
    print(f"  발견된 월: {len(months_sorted)}개 ({months_sorted[0]} ~ {months_sorted[-1]})")

    for m in months_sorted:
        monthly_data[m] = pd.concat(monthly_data[m], ignore_index=True)
        print(f"    {m}: {len(monthly_data[m]):,} 고객")

    # =========================================================
    # Step 2: 고객별 상품 시퀀스 (LIST) 생성
    # =========================================================
    print("\n=== Step 2: 고객별 상품 시퀀스 LIST 생성 ===")

    # 전체 고객 ID 수집
    all_customers = set()
    for m in months_sorted:
        all_customers.update(monthly_data[m]['ncodpers'].values)
    print(f"  전체 고유 고객: {len(all_customers):,}")

    # 최신 월 프로필을 기준으로
    latest_month = months_sorted[-1]
    latest_df = monthly_data[latest_month].set_index('ncodpers')

    # 고객별 상품 보유 시퀀스 구축
    # {customer_id: {product: [0,1,1,1,0,...], ...}}
    # 메모리 효율을 위해 numpy 배열 사용
    n_months = len(months_sorted)
    n_products = len(product_cols)

    # 고객 ID → index 매핑
    customer_list = sorted(all_customers)
    cust_to_idx = {c: i for i, c in enumerate(customer_list)}
    n_customers = len(customer_list)

    print(f"  매트릭스 크기: {n_customers:,} x {n_months} x {n_products}")
    print(f"  예상 메모리: {n_customers * n_months * n_products / 1024 / 1024:.0f} MB")

    # 3D 배열: (고객, 월, 상품) — int8로 메모리 절약
    product_tensor = np.full((n_customers, n_months, n_products), -1, dtype=np.int8)  # -1 = 미존재

    for m_idx, m in enumerate(months_sorted):
        df_m = monthly_data[m]
        for _, row in df_m.iterrows():
            c_idx = cust_to_idx[row['ncodpers']]
            for p_idx, pcol in enumerate(product_cols):
                product_tensor[c_idx, m_idx, p_idx] = row[pcol]

        if (m_idx + 1) % 3 == 0:
            print(f"    월 {m_idx+1}/{n_months} ({m}) 완료")

    # monthly_data 해제
    del monthly_data
    import gc
    gc.collect()

    # =========================================================
    # Step 3: 파생 피처 생성
    # =========================================================
    print("\n=== Step 3: 파생 피처 생성 ===")

    results = []

    for c_idx in range(n_customers):
        cust_id = customer_list[c_idx]
        seq = product_tensor[c_idx]  # (n_months, n_products)

        # 존재하는 월만 (값이 -1이 아닌 월)
        valid_mask = seq[:, 0] >= 0
        valid_months = np.where(valid_mask)[0]

        if len(valid_months) == 0:
            continue

        row = {'customer_id': cust_id}

        # 1) 상품별 보유 시퀀스 (LIST)
        for p_idx, pname in enumerate(product_names):
            pname_safe = pname if p_idx < len(product_names) else f'prod_{p_idx}'
            row[f'seq_{pname_safe}'] = seq[valid_months, p_idx].tolist()

        # 2) 월별 보유 상품 수 시퀀스
        monthly_counts = seq[valid_months].clip(0).sum(axis=1).tolist()
        row['seq_num_products'] = monthly_counts

        # 3) 신규 가입 이벤트 (0→1 전이)
        acquisitions = []
        churns = []
        for m in range(1, len(valid_months)):
            prev = seq[valid_months[m-1]].clip(0)
            curr = seq[valid_months[m]].clip(0)
            acq = int((curr - prev).clip(0).sum())  # 0→1 개수
            churn = int((prev - curr).clip(0).sum())  # 1→0 개수
            acquisitions.append(acq)
            churns.append(churn)

        row['seq_acquisitions'] = acquisitions if acquisitions else [0]
        row['seq_churns'] = churns if churns else [0]

        # 4) 집계 피처
        row['total_acquisitions'] = sum(acquisitions)
        row['total_churns'] = sum(churns)
        row['months_observed'] = len(valid_months)
        row['product_diversity'] = int(seq[valid_months].clip(0).max(axis=0).sum())

        # 5) NBA 레이블 (마지막 월에 새로 가입한 상품)
        if len(valid_months) >= 2:
            prev = seq[valid_months[-2]].clip(0)
            curr = seq[valid_months[-1]].clip(0)
            new_products = np.where((curr - prev) > 0)[0]
            row['nba_label'] = new_products.tolist() if len(new_products) > 0 else []
            row['has_nba'] = 1 if len(new_products) > 0 else 0
        else:
            row['nba_label'] = []
            row['has_nba'] = 0

        # 6) Churn 시그널 (마지막 월 상품 수가 이전 평균 대비 감소)
        if len(monthly_counts) >= 3:
            avg_prev = np.mean(monthly_counts[:-1])
            row['churn_signal'] = 1 if monthly_counts[-1] < avg_prev * 0.7 else 0
        else:
            row['churn_signal'] = 0

        # 7) 상품 보유 안정성 (변동 계수)
        if len(monthly_counts) >= 2 and np.mean(monthly_counts) > 0:
            row['product_stability'] = 1.0 - min(np.std(monthly_counts) / np.mean(monthly_counts), 1.0)
        else:
            row['product_stability'] = 1.0

        results.append(row)

        if (c_idx + 1) % 200_000 == 0:
            print(f"    {c_idx+1:,}/{n_customers:,} 고객 처리 완료")

    print(f"  총 {len(results):,} 고객 처리 완료")

    # =========================================================
    # Step 4: DataFrame 변환 및 저장
    # =========================================================
    print("\n=== Step 4: Parquet 저장 ===")

    df_result = pd.DataFrame(results)

    # 기본 통계
    print(f"  행: {len(df_result):,}")
    print(f"  컬럼: {len(df_result.columns)}")
    print(f"  NBA 있는 고객: {df_result['has_nba'].sum():,} ({df_result['has_nba'].mean()*100:.1f}%)")
    print(f"  Churn 시그널: {df_result['churn_signal'].sum():,} ({df_result['churn_signal'].mean()*100:.1f}%)")
    print(f"  평균 관찰 기간: {df_result['months_observed'].mean():.1f}개월")
    print(f"  평균 총 가입: {df_result['total_acquisitions'].mean():.1f}")
    print(f"  평균 총 해지: {df_result['total_churns'].mean():.1f}")

    out_path = os.path.join(output_dir, "santander_temporal.parquet")
    df_result.to_parquet(out_path, engine='pyarrow', index=False)
    print(f"  저장: {out_path} ({os.path.getsize(out_path)/1024/1024:.1f} MB)")

    # 메타데이터 저장
    meta = {
        'total_customers': len(df_result),
        'months': months_sorted,
        'n_months': n_months,
        'product_names': product_names,
        'columns': list(df_result.columns),
        'nba_rate': float(df_result['has_nba'].mean()),
        'churn_rate': float(df_result['churn_signal'].mean()),
    }
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  메타데이터: {meta_path}")

    print("\n완료!")


if __name__ == '__main__':
    main()
