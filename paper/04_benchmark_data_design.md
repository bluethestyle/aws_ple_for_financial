# Benchmark Data Design

## 1. 합성 데이터 사용 이유

- Santander 공개 데이터: 레이블 제한적, 시나리오별 의미있는 AUC 델타 측정 어려움
- 합성 데이터: 재현 가능한 실험 환경 + AUC ceiling 통제 가능
- 논문 재현성: seed 고정 시 동일 결과 보장

## 2. 4-Layer Generative Model

### Layer 1: Latent Personas
- 6 GMM-fitted personas (calibration_params.yaml에서 실데이터 기반 fitting)
- 5D continuous latent vector (wealth_propensity, activity_level, risk_tolerance, digital_affinity, loyalty)
- Latent 디커플링: 70% persona-conditioned + 30% independent noise
  → 모델이 관측 변수에서 latent를 완벽히 복원하지 못하게 함

### Layer 2: Gaussian Copula Demographics
- 6 persona별 상관 구조 유지하며 demographics 생성
- age, income, tenure_months 등 인구통계 변수
- Copula로 변수 간 현실적 상관 유지

### Layer 3: Vectorized Transactions
- 고객별 거래 시퀀스를 LIST 컬럼으로 생성 (row explosion 없음)
- MCC hierarchy: 10 L1 / 30 L2 / 109 MCC codes
- AR(1) spending persistence, FFT seasonality

### Layer 4: Variance Budget Labels
- 각 레이블의 예측 난이도를 variance budget으로 통제
- 공식: logit = sqrt(obs_frac) * obs + sqrt(lat_frac) * latent + sqrt(noise_frac) * noise

| Tier | 레이블 | obs_frac | lat_frac | noise_frac | label_noise | XGB AUC ceiling |
|------|--------|----------|----------|------------|-------------|-----------------|
| Easy | segment, income_tier, tenure_stage | 결정론적 | - | - | - | 0.95-1.0 |
| Core | has_nba, churn_signal | 0.04 | 0.28 | 0.68 | 6% | 0.58-0.65 |
| Hard | will_acquire_* (5개) | 0.03 | 0.25 | 0.72 | 8% | 0.50-0.56 |
| Regression | product_stability, engagement_score | 연속 | - | 직접 가산 | - | R2 0.0-1.0 |
| Very Hard | next_mcc, top_mcc_shift | 0.02 | 0.20 | 0.78 | 5% | 0.50-0.51 |

### Label Noise (Post-hoc Flipping)
- 목적: AUC ceiling을 강제로 제한 (feature-latent 상관 통한 우회 방지)
- has_nba/churn_signal: 6% 랜덤 플립
- will_acquire_*: 8% 랜덤 플립
- top_mcc_shift: 5% 랜덤 플립

## 3. Data Leakage 방지 설계

### Generator 입력에서 label 제외
- 문제: GMM/model_derived generator가 label 컬럼을 입력으로 사용 → AUC 1.0
- 해결: run_generators_duckdb()에서 label_cols 자동 제외
- 검증: XGBoost AUC가 1.0 → 0.60으로 정상화

### 검증 수치 (수정 후)
```
XGBoost AUC (all 316 features):
  has_nba:      0.6081
  churn_signal: 0.6531

Per-group AUC (has_nba):
  demographics:        0.5932
  txn_behavior:        0.6051
  hmm_states:          0.5803
  gmm_clustering:      0.5999  (수정 전: 1.0000)
  model_derived:       0.5956  (수정 전: 0.9986)
  tda_global/local:    0.5000
  mamba_temporal:       0.5013
  product_hierarchy:   0.4997
  graph_collaborative: 0.5003
```

## 4. 데이터 스펙

- 고객 수: 1,000,000
- 컬럼 수: 106 (raw) → 316 (Phase 0 후 features)
- 레이블: 18 tasks
- 파일 크기: ~1.1GB (raw), ~1.2GB (Phase 0 output)
- 시퀀스: LIST 컬럼 (ragged tensor), 최대 12개월 × 30 거래
