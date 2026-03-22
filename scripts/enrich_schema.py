# -*- coding: utf-8 -*-
import sys, json
sys.stdout.reconfigure(encoding='utf-8')

with open('E:/AIOps_project_AWS/data/synthetic/santander_final_schema.json', 'r', encoding='utf-8') as f:
    schema = json.load(f)

col_meta = {
    'customer_id':       {'desc': '고객 고유 식별자 (Santander ncodpers)', 'nullable': False, 'valid': 'BIGINT, 15889~1553689'},
    'snapshot_date':     {'desc': '프로필 스냅샷 기준일 (고객별 최신 월)', 'nullable': False, 'valid': '2015-07-28 ~ 2016-05-28 (11개 값)'},
    'gender':            {'desc': '성별', 'nullable': False, 'valid': 'F(여), M(남)'},
    'age':               {'desc': '나이 (18세 미만 제외)', 'nullable': False, 'valid': '18~100'},
    'income':            {'desc': '연 소득 (EUR). 0=미기입 (25.4%)', 'nullable': True, 'valid': '0~28,894,395'},
    'segment':           {'desc': '고객 세그먼트', 'nullable': False, 'valid': '01-TOP, 02-PARTICULARES, 03-UNIVERSITARIO, UNKNOWN'},
    'country':           {'desc': '거주 국가 코드 (ISO 2자리)', 'nullable': False, 'valid': '118개국 (ES 최다 96%)'},
    'channel':           {'desc': '최초 가입 채널 코드', 'nullable': False, 'valid': '163개 (KHE 최다)'},
    'tenure_months':     {'desc': '거래 기간 (개월). -999999=미상', 'nullable': False, 'valid': '-999999~256'},
    'is_active':         {'desc': '활성 고객 여부', 'nullable': False, 'valid': '0(비활성), 1(활성). 활성률 42%'},
    'age_group':         {'desc': '나이 구간 (합성 분류)', 'nullable': False, 'valid': 'young, adult, middle, senior, elderly'},
    'income_group':      {'desc': '소득 구간 (합성 분류)', 'nullable': False, 'valid': 'low, mid, high, very_high, unknown'},
    'num_products':      {'desc': '현재 보유 상품 수', 'nullable': False, 'valid': '0~15, 평균 1.3'},
    'prod_saving':       {'desc': '저축 계좌 보유', 'nullable': False, 'valid': '0/1', 'rate': '0.3%'},
    'prod_guarantee':    {'desc': '보증 계좌 보유', 'nullable': False, 'valid': '0/1', 'rate': '0.1%'},
    'prod_checking':     {'desc': '당좌 계좌 보유', 'nullable': False, 'valid': '0/1', 'rate': '60%'},
    'prod_derivados':    {'desc': '파생상품 보유', 'nullable': False, 'valid': '0/1', 'rate': '0.2%'},
    'prod_payroll_acct': {'desc': '급여 계좌 보유', 'nullable': False, 'valid': '0/1', 'rate': '8%'},
    'prod_junior_acct':  {'desc': '주니어(청소년) 계좌 보유', 'nullable': False, 'valid': '0/1', 'rate': '0.4%'},
    'prod_particular_acct': {'desc': '개인 계좌 보유', 'nullable': False, 'valid': '0/1', 'rate': '1%'},
    'prod_particular_plus': {'desc': '개인 플러스 계좌 보유', 'nullable': False, 'valid': '0/1', 'rate': '11%'},
    'prod_short_deposit':   {'desc': '단기 예금 보유', 'nullable': False, 'valid': '0/1', 'rate': '4%'},
    'prod_medium_deposit':  {'desc': '중기 예금 보유', 'nullable': False, 'valid': '0/1', 'rate': '0.1%'},
    'prod_long_deposit':    {'desc': '장기 예금 보유', 'nullable': False, 'valid': '0/1', 'rate': '0.1%'},
    'prod_e_account':    {'desc': '전자 계좌 보유', 'nullable': False, 'valid': '0/1', 'rate': '3%'},
    'prod_funds':        {'desc': '펀드 보유', 'nullable': False, 'valid': '0/1', 'rate': '8%'},
    'prod_mortgage':     {'desc': '주택담보대출 보유', 'nullable': False, 'valid': '0/1', 'rate': '2%'},
    'prod_pension_plan': {'desc': '연금 플랜 보유', 'nullable': False, 'valid': '0/1', 'rate': '0.3%'},
    'prod_loans':        {'desc': '대출 보유', 'nullable': False, 'valid': '0/1', 'rate': '1%'},
    'prod_taxes':        {'desc': '세금 계좌 보유', 'nullable': False, 'valid': '0/1', 'rate': '0.2%'},
    'prod_credit_card':  {'desc': '신용카드 보유', 'nullable': False, 'valid': '0/1', 'rate': '5%'},
    'prod_securities':   {'desc': '증권 계좌 보유', 'nullable': False, 'valid': '0/1', 'rate': '4%'},
    'prod_home_acct':    {'desc': '주택 계좌 보유', 'nullable': False, 'valid': '0/1', 'rate': '2%'},
    'prod_payroll':      {'desc': '급여 이체 등록', 'nullable': False, 'valid': '0/1', 'rate': '0.1%'},
    'prod_pension_deposit': {'desc': '연금 예금 보유', 'nullable': False, 'valid': '0/1', 'rate': '5%'},
    'prod_direct_debit': {'desc': '자동이체 등록', 'nullable': False, 'valid': '0/1', 'rate': '6%'},
    'prod_auto_debit':   {'desc': '자동 결제 등록', 'nullable': False, 'valid': '0/1', 'rate': '12%'},
    'synth_monthly_txns':    {'desc': '월 거래 건수 (합성)', 'nullable': False, 'valid': '20~157, 평균 69'},
    'synth_avg_amount':      {'desc': '건당 평균 거래금액 USD (합성)', 'nullable': False, 'valid': '1~218, 평균 49'},
    'synth_monthly_spend':   {'desc': '월 총 소비금액 USD (합성)', 'nullable': False, 'valid': '536~7542, 평균 3146'},
    'synth_unique_mcc':      {'desc': '고유 업종(MCC) 수 (합성)', 'nullable': False, 'valid': '6~23, 평균 16'},
    'synth_unique_merchants': {'desc': '고유 가맹점 수 (합성)', 'nullable': False, 'valid': '11~33, 평균 24'},
    'synth_morning_ratio':   {'desc': '오전(6~12시) 거래 비율 (합성)', 'nullable': False, 'valid': '0.2~0.65'},
    'synth_afternoon_ratio': {'desc': '오후(12~18시) 거래 비율 (합성)', 'nullable': False, 'valid': '0.23~0.57'},
    'synth_evening_ratio':   {'desc': '저녁(18~22시) 거래 비율 (합성)', 'nullable': False, 'valid': '0.08~0.30'},
    'synth_night_ratio':     {'desc': '야간(22~6시) 거래 비율 (합성)', 'nullable': False, 'valid': '0.02~0.14'},
    'synth_recency_days':    {'desc': 'RFM Recency 정규화 (합성)', 'nullable': False, 'valid': '0.2~1.0'},
    'synth_frequency':       {'desc': 'RFM Frequency 거래 빈도 (합성)', 'nullable': False, 'valid': '25~131'},
    'synth_monetary':        {'desc': 'RFM Monetary 총 소비 USD (합성)', 'nullable': False, 'valid': '1250~5430'},
    'synth_stability':       {'desc': '소비 안정성 지수 (합성, CV 기반)', 'nullable': False, 'valid': '0.58~1.76 (1=안정)'},
    'synth_fraud_ratio':     {'desc': '사기 거래 비율 (합성)', 'nullable': False, 'valid': '0.0 (전부 정상)'},
    'txn_amount_seq':    {'desc': '거래금액 시퀀스 (합성, ealtman 기반)', 'nullable': False, 'valid': 'DOUBLE[], len 1~60 (avg 41)', 'tensor_dim': '(N,)', 'unit': 'USD'},
    'txn_mcc_seq':       {'desc': '업종코드(MCC) 시퀀스 (합성)', 'nullable': False, 'valid': 'INTEGER[], len 1~60', 'tensor_dim': '(N,)'},
    'txn_hour_seq':      {'desc': '거래시간(시) 시퀀스 (합성)', 'nullable': False, 'valid': 'INTEGER[], len 1~60, 값 0~23', 'tensor_dim': '(N,)'},
    'seq_saving':        {'desc': '저축계좌 보유 시퀀스 (17개월)', 'nullable': False, 'valid': 'BIGINT[], len 1~17, 값 0/1/-1(-1=미존재)', 'tensor_dim': '(T,)'},
    'seq_guarantee':     {'desc': '보증계좌 보유 시퀀스', 'nullable': False, 'valid': 'BIGINT[], len 1~17, 값 0/1/-1'},
    'seq_checking':      {'desc': '당좌계좌 보유 시퀀스', 'nullable': False, 'valid': 'BIGINT[], len 1~17, 값 0/1/-1'},
    'seq_derivados':     {'desc': '파생상품 보유 시퀀스', 'nullable': False, 'valid': 'BIGINT[], len 1~17, 값 0/1/-1'},
    'seq_payroll_acct':  {'desc': '급여계좌 보유 시퀀스', 'nullable': False, 'valid': 'BIGINT[], len 1~17, 값 0/1/-1'},
    'seq_junior_acct':   {'desc': '주니어계좌 보유 시퀀스', 'nullable': False, 'valid': 'BIGINT[], len 1~17, 값 0/1/-1'},
    'seq_particular_acct': {'desc': '개인계좌 보유 시퀀스', 'nullable': False, 'valid': 'BIGINT[], len 1~17, 값 0/1/-1'},
    'seq_particular_plus': {'desc': '개인플러스계좌 보유 시퀀스', 'nullable': False, 'valid': 'BIGINT[], len 1~17, 값 0/1/-1'},
    'seq_short_deposit':   {'desc': '단기예금 보유 시퀀스', 'nullable': False, 'valid': 'BIGINT[], len 1~17, 값 0/1/-1'},
    'seq_medium_deposit':  {'desc': '중기예금 보유 시퀀스', 'nullable': False, 'valid': 'BIGINT[], len 1~17, 값 0/1/-1'},
    'seq_long_deposit':    {'desc': '장기예금 보유 시퀀스', 'nullable': False, 'valid': 'BIGINT[], len 1~17, 값 0/1/-1'},
    'seq_e_account':     {'desc': '전자계좌 보유 시퀀스', 'nullable': False, 'valid': 'BIGINT[], len 1~17, 값 0/1/-1'},
    'seq_funds':         {'desc': '펀드 보유 시퀀스', 'nullable': False, 'valid': 'BIGINT[], len 1~17, 값 0/1/-1'},
    'seq_mortgage':      {'desc': '주택담보대출 보유 시퀀스', 'nullable': False, 'valid': 'BIGINT[], len 1~17, 값 0/1/-1'},
    'seq_pension_plan':  {'desc': '연금플랜 보유 시퀀스', 'nullable': False, 'valid': 'BIGINT[], len 1~17, 값 0/1/-1'},
    'seq_loans':         {'desc': '대출 보유 시퀀스', 'nullable': False, 'valid': 'BIGINT[], len 1~17, 값 0/1/-1'},
    'seq_taxes':         {'desc': '세금계좌 보유 시퀀스', 'nullable': False, 'valid': 'BIGINT[], len 1~17, 값 0/1/-1'},
    'seq_credit_card':   {'desc': '신용카드 보유 시퀀스', 'nullable': False, 'valid': 'BIGINT[], len 1~17, 값 0/1/-1'},
    'seq_securities':    {'desc': '증권계좌 보유 시퀀스', 'nullable': False, 'valid': 'BIGINT[], len 1~17, 값 0/1/-1'},
    'seq_home_acct':     {'desc': '주택계좌 보유 시퀀스', 'nullable': False, 'valid': 'BIGINT[], len 1~17, 값 0/1/-1'},
    'seq_payroll':       {'desc': '급여이체 등록 시퀀스', 'nullable': False, 'valid': 'BIGINT[], len 1~17, 값 0/1/-1'},
    'seq_pension_deposit': {'desc': '연금예금 보유 시퀀스', 'nullable': False, 'valid': 'BIGINT[], len 1~17, 값 0/1/-1'},
    'seq_direct_debit':  {'desc': '자동이체 등록 시퀀스', 'nullable': False, 'valid': 'BIGINT[], len 1~17, 값 0/1/-1'},
    'seq_auto_debit':    {'desc': '자동결제 등록 시퀀스', 'nullable': False, 'valid': 'BIGINT[], len 1~17, 값 0/1/-1'},
    'seq_num_products':  {'desc': '월별 보유 상품 수 시퀀스', 'nullable': False, 'valid': 'BIGINT[], len 1~17, 값 0~15'},
    'seq_acquisitions':  {'desc': '월별 신규 가입 수 시퀀스 (0->1 전이)', 'nullable': False, 'valid': 'BIGINT[], len 1~16'},
    'seq_churns':        {'desc': '월별 해지 수 시퀀스 (1->0 전이)', 'nullable': False, 'valid': 'BIGINT[], len 1~16'},
    'total_acquisitions': {'desc': '전체 기간 총 신규 가입 횟수', 'nullable': False, 'valid': '0~22, 평균 0.6'},
    'total_churns':       {'desc': '전체 기간 총 해지 횟수', 'nullable': False, 'valid': '0~22, 평균 0.56'},
    'months_observed':    {'desc': '관찰 기간 (개월 수)', 'nullable': False, 'valid': '1~17, 평균 14.3'},
    'product_diversity':  {'desc': '보유한 적 있는 상품 종류 수', 'nullable': False, 'valid': '0~15, 평균 1.56'},
    'nba_label':          {'desc': 'NBA 레이블: 마지막 월 신규 가입 상품 인덱스', 'nullable': False, 'valid': 'BIGINT[], 상품인덱스 0~23, 빈리스트=없음'},
    'has_nba':            {'desc': 'NBA 이벤트 존재 여부', 'nullable': False, 'valid': '0/1, 양성 2.98%'},
    'churn_signal':       {'desc': '이탈 시그널 (상품수 < 이전평균×0.7)', 'nullable': False, 'valid': '0/1, 양성 5.1%'},
    'product_stability':  {'desc': '상품 보유 안정성 (1-CV). 1=매우 안정', 'nullable': True, 'valid': '0~1, 평균 0.92'},
}

for col in schema['columns']:
    name = col['name']
    if name in col_meta:
        meta = col_meta[name]
        col['description'] = meta['desc']
        col['nullable'] = meta['nullable']
        col['valid_values'] = meta['valid']
        if 'rate' in meta:
            col['positive_rate'] = meta['rate']
        if 'tensor_dim' in meta:
            col['tensor_dim'] = meta['tensor_dim']
        if 'unit' in meta:
            col['unit'] = meta['unit']

with open('E:/AIOps_project_AWS/data/synthetic/santander_final_schema.json', 'w', encoding='utf-8') as f:
    json.dump(schema, f, indent=2, ensure_ascii=False)

print(f"스키마 보강 완료: {len(schema['columns'])}개 컬럼 x (description + nullable + valid_values)")
