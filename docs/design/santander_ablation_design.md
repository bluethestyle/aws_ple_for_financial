# Santander Ablation Study 설계 문서

> **프로젝트**: AIOps PLE Financial — Santander Customer Product Recommendation
> **데이터셋**: 941,132 users x 89 columns x 18 tasks x 7 shared experts
> **최종 갱신**: 2026-03-30
> **Config 경로**: `configs/santander/pipeline.yaml`, `configs/santander/feature_groups.yaml`, `configs/santander/item_universe.yaml`
> **Orchestrator**: `scripts/run_santander_ablation.py`

---

## 목차

1. [파이프라인 아키텍처](#1-파이프라인-아키텍처)
2. [Santander 데이터셋 설계](#2-santander-데이터셋-설계)
3. [태스크 설계 (18개)](#3-태스크-설계-18개)
4. [모델 아키텍처](#4-모델-아키텍처)
5. [4차원 Ablation 설계](#5-4차원-ablation-설계)
6. [비용 최적화](#6-비용-최적화)
7. [해석/분석 모듈](#7-해석분석-모듈)
8. [스코어링 및 제약조건](#8-스코어링-및-제약조건)
9. [알려진 이슈 및 향후 과제](#9-알려진-이슈-및-향후-과제)

---

## 1. 파이프라인 아키텍처

### 1.1 10-Stage 범용 파이프라인

`core/pipeline/runner.py`의 `PipelineRunner`가 10개 스테이지를 순차 실행한다. 각 스테이지는 독립적으로 테스트 가능하며, `_PipelineState`에 의해 체크포인트/resume을 지원한다.

```
Stage 1    데이터 로드        DataAdapter.load_raw() -> Dict[str, DataFrame]
Stage 1.5  Temporal 준비      시퀀스 truncation + prod_* 재계산 (리키지 방지)
Stage 2    스키마 분류        numeric / categorical / sequence 자동 분류
Stage 3    PII 암호화        (선택적) 개인식별정보 암호화
Stage 4    피처 엔지니어링    FeatureGroupPipeline: 12개 그룹 -> ~500D (총합; 각 Expert는 FeatureRouter를 통해 지정된 서브셋만 수신)
Stage 5    레이블 파생        LabelDeriver: 18개 태스크 레이블 생성
Stage 5.5  리키지 검증        LeakageValidator: 상관/시퀀스/시간/prod 4개 검증
Stage 6    시퀀스 빌드        Time-based + sliding window → 3D tensor (txn_day_offset_seq, prod_seq)
Stage 7    DataLoader 구성    PLEDataset -> train_loader / val_loader
Stage 8    학습 (Teacher)     PLE 모델 학습 (Phase 1 joint + Phase 2 freeze)
Stage 8.5  모델 분석          IG attribution + Expert Redundancy CCA + CGC Gate
Stage 9    지식 증류          Teacher -> LGBM student (temperature=5, alpha=0.3)
Stage 9.5  서빙 준비          Context Vector Store + Reason Generation (config-gated)
```

### 1.2 Config 연결 구조

| Config 파일 | 역할 | 연결 스테이지 |
|---|---|---|
| `pipeline.yaml` | 태스크, 모델, 학습, 증류, 평가, AWS 설정 | 전체 |
| `feature_groups.yaml` | 12개 피처 그룹 정의 (5-axis + derived) | Stage 4 |
| `item_universe.yaml` | 24개 상품 계층, 대체재/보완재, 상품 티어 | Stage 4 (hierarchy), Stage 8 |

### 1.3 Stage 간 데이터 흐름

```
raw_data["main"] (941K x 89)
  |
  +-- Stage 1.5 --> seq_* truncated, prod_* recomputed
  |
  +-- Stage 4 --> df_features (941K x ~500D, 총합)
  |                 checkpoints/features.parquet
  |                 ※ FeatureRouter: 각 Expert는 지정된 피처 그룹의 서브셋만 수신
  |                    (316D 라우팅 기준: deepfm=109D, temporal_ensemble=129D, hgcn=34D,
  |                     perslay=32D, causal=103D, lightgcn=66D, optimal_transport=69D)
  |
  +-- Stage 5 --> df_labels (941K x 18 label cols)
  |                 checkpoints/labels.parquet
  |
  +-- Stage 5.5 --> leakage_report.json (pass/fail + warnings)
  |
  +-- Stage 6 --> sequences dict (txn: 3 x [N,60], prod: 27 x [N,16])
  |                 checkpoints/*.npy
  |
  +-- Stage 7 --> train_loader, val_loader (PLEDataset)
  |
  +-- Stage 8 --> model.pth + eval_metrics.json
  |
  +-- Stage 8.5 --> ig_attributions.json, expert_redundancy.json
  |
  +-- Stage 9 --> lgbm_models/*.pkl + distill_metrics.json
```

### 1.4 체크포인트 / Resume 구조

`_PipelineState`가 `outputs/pipeline_state.json`을 관리한다:

```json
{
  "completed_stages": ["stage1", "stage1_5", "stage2", ...],
  "artifacts": {
    "stage4": {"shape": [941132, 500]},
    "stage6": {"keys": ["txn_amount_seq", ...]}
  },
  "start_time": "2026-03-25T10:00:00"
}
```

- 각 스테이지 완료 시 `mark_complete()` 호출
- 실패 시 `mark_failed()` 호출하고 예외 전파
- Resume 시 이미 완료된 스테이지는 아티팩트를 로드하여 건너뜀
- Stage 4/5/6는 중간 결과를 `checkpoints/` 디렉토리에 parquet/npy로 저장

---

## 2. Santander 데이터셋 설계

### 2.1 데이터 개요

| 항목 | 값 |
|---|---|
| 총 사용자 수 | 941,132 |
| 총 컬럼 수 | 89 (원본) + augmented txn sequences |
| 데이터 형식 | Parquet |
| Real txn 소스 | ealtman2019: 2,000 users, 24.4M transactions |
| Augmentation | Segment pooling: (age_group, income_group, activity_level) 매칭 |
| MCC Hierarchy | 109 L3 codes, ~30 L2 subcategories, 10 L1 groups (ISO 18245) |
| 시퀀스 시간 | txn_day_offset_seq (상대 일수 오프셋, YYYYMMDD 아님) |
| S3 경로 | `s3://aiops-ple-financial/data/santander/` |
| 백엔드 우선순위 | cudf -> duckdb -> pandas |
| Cold start | `is_cold_start` flag + sequence-derived feature zeroing |

### 2.2 5개 컬럼 그룹

| 그룹 | 컬럼 수 | 예시 |
|---|---|---|
| **Profile** (인구통계) | 11 | age, income, tenure_months, gender, segment, country, channel, age_group, income_group, is_active, num_products |
| **Product** (상품 보유) | 24 | prod_saving, prod_checking, ..., prod_auto_debit (binary 0/1) |
| **Synth** (거래 합성) | 14 | synth_monthly_txns, synth_avg_amount, synth_monthly_spend, synth_unique_mcc, synth_*_ratio, synth_recency/frequency/monetary, synth_stability, synth_fraud_ratio |
| **Txn Sequence** | 3 | txn_amount_seq (float[]), txn_mcc_seq (int[]), txn_day_offset_seq (int[], snap_date 기준 상대 일수) — max_len=60 |
| **Product Sequence** | 27 | seq_saving, ..., seq_auto_debit, seq_num_products, seq_acquisitions, seq_churns — max_len=17 (truncate to 16) |

### 2.3 데이터 전처리 규칙

| 컬럼 | 이슈 | 처리 방법 |
|---|---|---|
| `income` | 25.4% 결측 (값=0이 결측 표시) | median imputation |
| `tenure_months` | -999999 센티넬 (unknown) | clip [0, 256] + tenure_unknown 플래그 컬럼 |
| `synth_fraud_ratio` | 전체 0.0 (zero variance) | 포함하되 파이프라인이 자동 제거 가능 |

### 2.4 데이터 리키지 방지

Santander 데이터의 핵심 리키지 위험과 방지 전략 3가지:

#### (1) 시퀀스 Truncation

`core/pipeline/temporal_split.py`의 `split_by_sequence_cutoff()`:

- **문제**: `seq_*` 시퀀스가 17개월 전체 포함 (month 17 = 레이블 월)
- **해결**: `truncate_last=1` 설정으로 month 17 제거 -> max_len=16
- **검증**: `LeakageValidator.check_sequence_leakage()`가 max_len > 16이면 CRITICAL 실패

#### (2) prod_* 컬럼 재계산

- **문제**: `prod_*` 24개 컬럼이 month 17 (레이블 상태) 반영
- **해결**: `recompute_prod_from_seq: true` 설정으로 truncated seq의 마지막 원소 (month 16)에서 재계산
- **검증**: `LeakageValidator.check_product_columns()`가 prod_*와 seq_*[-1] 매치율 > 90%이면 CRITICAL 실패

#### (3) Temporal Split

`TemporalSplitter.split()`:

- **설정**: `date_col=snapshot_date`, `gap_days=30`, `train_ratio=0.7`, `val_ratio=0.15`
- **방법**: snapshot_date 기준 시간순 정렬 후 train/val/test 분할, 분할 간 30일 gap
- **검증**: `LeakageValidator.check_temporal_integrity()`가 train_max >= val_min이면 CRITICAL 실패

#### (4) Feature-Label 상관 검증

- threshold=0.95 이상 Pearson 상관이면 CRITICAL 경고
- label 컬럼이 feature에 포함되면 CRITICAL 실패

---

## 3. 태스크 설계 (18개)

### 3.1 4-Tier 태스크 구조

```
Tier 1 — Core (4 tasks)       직접 레이블. 데이터셋에서 바로 사용 가능
Tier 2 — Derived (4 tasks)    프로필/합성 컬럼에서 규칙으로 파생
Tier 3 — Product Group (7)    nba_label 인덱스에서 그룹별 이진 분류
Tier 5 — Txn NBA (3 tasks)    txn_mcc_seq 거래 시퀀스에서 파생
```

> Tier 4 (개별 상품 태스크)는 Tier 3 그룹과 중복으로 제외됨

### 3.2 전체 18개 태스크 상세

#### Tier 1 — Core (직접 레이블)

| 태스크 | 유형 | Loss | Weight | 설명 |
|---|---|---|---|---|
| `has_nba` | binary | Focal(a=0.90, g=2.0) | 2.5 | 신규 상품 가입 여부 (2.98% positive) |
| `churn_signal` | binary | Focal(a=0.85, g=2.0) | 2.0 | 이탈 신호: 상품수 < 이전평균 x 0.7 (5.1%) |
| `product_stability` | regression | Huber | 1.5 | 상품 보유 안정성 (1-CV), nullable |
| `nba_primary` | multiclass(24) | CE | 2.0 | 어떤 상품 추천? (nba_label 첫 번째 원소) |

#### Tier 2 — Derived (규칙 파생)

| 태스크 | 유형 | Loss | Weight | 파생 방법 |
|---|---|---|---|---|
| `tenure_stage` | multiclass(5) | CE | 0.8 | tenure_months bucket [6,24,60,120] -> 5 클래스 |
| `spend_level` | multiclass(4) | CE | 0.8 | synth_monthly_spend bucket [1500,3000,5000] |
| `cross_sell_count` | regression | Huber | 1.0 | len(nba_label) |
| `engagement_score` | regression | Huber | 1.0 | is_active*0.3 + synth_frequency/max*0.4 + num_products/max*0.3 |

#### Tier 3 — Product Group + Segmentation

| 태스크 | 유형 | Loss | Weight | 파생: nba_label 인덱스 |
|---|---|---|---|---|
| `will_acquire_deposits` | binary | Focal(a=0.95) | 1.5 | [8,9,10] short/medium/long deposit |
| `will_acquire_investments` | binary | Focal(a=0.90) | 1.5 | [12,18] funds, securities |
| `will_acquire_accounts` | binary | Focal(a=0.85) | 1.2 | [2,5,6,7,11,19] 계좌 유형 |
| `will_acquire_lending` | binary | Focal(a=0.90) | 1.5 | [13,15] mortgage, loans |
| `will_acquire_payments` | binary | Focal(a=0.85) | 1.2 | [4,17,20,22,23] 결제 유형 |
| `segment_prediction` | multiclass(4) | CE | 1.0 | segment string_map -> 4 클래스 |
| `income_tier` | multiclass(4) | CE | 0.8 | income bucket [30k,80k,200k] |

#### Tier 5 — Transaction-based NBA

| 태스크 | 유형 | Loss | Weight | 파생 방법 |
|---|---|---|---|---|
| `next_mcc` | multiclass(50) | CE | 1.2 | txn_mcc_seq의 마지막 원소 (top-50 MCC) |
| `mcc_diversity_trend` | regression | Huber | 0.8 | unique(후반부)/unique(전반부) - 1.0 |
| `top_mcc_shift` | binary | Focal(a=0.70) | 1.0 | 최빈 MCC가 전반/후반 사이에 변경 여부 |

### 3.3 4개 의미적 그룹 (adaTT Transfer Unit)

```
engagement  "고객이 반응하는가"    [has_nba, engagement_score, next_mcc, top_mcc_shift]
            intra=0.8, inter=0.3

lifecycle   "고객이 어디에 있는가"  [churn_signal, product_stability, tenure_stage, segment_prediction]
            intra=0.7, inter=0.3

value       "언제/어디서 가치를"    [spend_level, income_tier, mcc_diversity_trend]
            intra=0.6, inter=0.3

consumption "무엇을 소비하는가"    [nba_primary, cross_sell_count, will_acquire_*5]
            intra=0.7, inter=0.3
```

그룹 간 adaTT inter_group_strength = 0.3

### 3.4 5개 로짓 전이 관계

```
engagement_score  --[sequential/output_concat]--> has_nba        "활성도 -> 가입 (선행지표)"
has_nba           --[sequential/output_concat]--> nba_primary    "가입 여부 -> 어떤 상품"
churn_signal      --[causal/output_concat]------> product_stability "이탈 -> 상품 안정성"
spend_level       --[causal/output_concat]------> cross_sell_count  "소비수준 -> 교차판매"
next_mcc          --[sequential/output_concat]--> nba_primary    "다음 업종 -> 다음 상품"
```

logit_transfer_strength = 0.5

---

## 4. 모델 아키텍처

### 4.1 PLE + adaTT + 7 Shared Experts

핵심 아키텍처는 **Progressive Layered Extraction (PLE)** 에 **adaptive Task-Transfer (adaTT)** 를 결합한 멀티태스크 학습 모델이다.

```
Input (~500D, 총 316D 활성 피처)
  |
  v
[FeatureRouter] ── Expert별 지정 피처 그룹 서브셋만 라우팅 (전체 브로드캐스트 아님)
  |                  deepfm          → 109D  (demographics + product + synth 교차)
  |                  temporal_ens    → 129D  (txn_behavior + temporal_pattern + synth)
  |                  hgcn            → 34D   (product_hierarchy + product_holdings)
  |                  perslay         → 32D   (tda_global)
  |                  causal          → 103D  (demographics + synth + txn_behavior)
  |                  lightgcn        → 66D   (product_holdings + co-purchase graph)
  |                  optimal_transport→ 69D  (demographics + synth + distribution)
  |                  ※ 파라미터: 4.77M → ~2.8M (감소)
  v
[CGC Layer 1] ── shared experts x 7 + task expert x 1 per task
  |                gating: softmax(W * concat(input, task_embedding))
  v
[CGC Layer 2] ── dim_normalize: true
  |
  v
[CGC Layer 3] ── extraction_dim=64, num_layers=3
  |
  v
[Task Towers] ── default_dims=[64,32], dropout=0.1
  |                18개 태스크별 독립 tower
  v
[Predictions] ── binary: sigmoid, multiclass: softmax, regression: linear
```

### 4.2 7 Shared Expert 구성

| Expert | 역할 | 라우팅 입력 차원 | 핵심 파라미터 |
|---|---|---|---|
| **DeepFM** | 피처 교차 상호작용 | **109D** | emb_dim=8, hidden=[256,128], dropout=0.1 |
| **Temporal Ensemble** | 시계열 패턴 (Mamba+Transformer+LNN) | **129D** | mamba_d=64, n_layers=2, transformer_heads=4 |
| **HGCN** | 쌍곡 기하학 기반 상품 계층 | **34D** | hyperbolic_dim=20, product_dim=24, refine=128 |
| **PersLay** | TDA 위상 특징 처리 | **32D** | phi_hidden=64, rho_hidden=64 |
| **Causal** | DAG 기반 인과관계 | **103D** | dag_hidden=[128,64], lambda_dag=0.01 |
| **LightGCN** | 협업 필터링 그래프 | **66D** | emb_dim=64, n_layers=3 |
| **Optimal Transport** | 분포 매칭 | **69D** | hidden=[128,64], sinkhorn_iter=50 |

> **FeatureRouter 활성화**: 전체 ~500D 피처를 모든 Expert에 브로드캐스트하지 않고, 각 Expert의 설계 목적에 맞는 피처 그룹만 라우팅한다. 총 모델 파라미터: 4.77M → ~2.8M (감소). 라우팅 입력 차원의 합이 총 피처 수(~500D)를 초과하는 것은 일부 피처 그룹이 복수 Expert에 공유되기 때문이다 (현재 활성 316D 기준).

Task expert: MLP (hidden=[256,128], dropout=0.1) -- 각 태스크마다 1개

### 4.3 Evidential Deep Learning

```yaml
evidential:
  enabled: true
  kl_lambda: 0.01
  annealing_epochs: 10
```

- Regression 태스크에 Normal Inverse-Gamma prior 적용
- 예측값과 함께 인식적 불확실성 (epistemic uncertainty) 출력
- KL divergence regularization으로 과확신 방지
- annealing: 처음 10 에폭 동안 kl_lambda를 0에서 0.01까지 선형 증가

### 4.4 Sparse Autoencoder (SAE)

```yaml
sae:
  enabled: true
  weight: 0.01
  expansion_factor: 4
  l1_lambda: 0.001
```

- 해석 가능성을 위한 sparse representation 학습
- Expert 출력을 expansion_factor=4로 확장 후 L1 sparsity 적용
- SAE loss가 전체 loss에 weight=0.01로 추가됨

### 4.5 AMP FP32 Loss Computation

AMP (Mixed Precision) 환경에서 loss 안정성:
- Tower output은 FP16으로 계산
- Loss 계산 시 FP32로 cast (`torch.cuda.amp.autocast(enabled=False)`)
- FP16 range (+/-65504) overflow 방지

### 4.6 Logit Transfer 3-Method Dispatch

`task_relationships`에 정의된 5개 관계에 대해 `output_concat` 방식으로 소스 태스크의 logit을 타겟 태스크 tower 입력에 연결(concatenate)한다.

- `output_concat`: 소스 태스크 예측값을 타겟 tower 입력에 concat
- `hidden_concat`: 소스 pre-tower hidden과 tower input concat
- `residual`: 소스 output → Linear → tower_dim, 잔차 합산
- 전이 강도: logit_transfer_strength = 0.5

### 4.6 Multidisciplinary Per-Task Routing

24D multidisciplinary feature를 4개 태스크 그룹에 6D씩 라우팅:

```yaml
engagement:  [0,1,2,3,4,5]      # chemical_kinetics 관점
lifecycle:   [6,7,8,9,10,11]    # epidemic_diffusion 관점
value:       [12,13,14,15,16,17] # crime_pattern 관점
consumption: [18,19,20,21,22,23] # interference 관점
```

### 4.7 Loss 전략

| 태스크 유형 | Loss 함수 | 추가 전략 |
|---|---|---|
| Binary | Focal Loss | alpha, gamma per task; 극단적 불균형 (has_nba 2.98%) 대응 |
| Multiclass | Cross Entropy | nba_primary (24 classes), next_mcc (50 classes) |
| Regression | Huber Loss | 이상치에 로버스트 |

- **Uncertainty Weighting**: `loss_weighting.strategy: uncertainty` -- 각 태스크의 homoscedastic uncertainty를 학습 가능 파라미터로 설정하여 자동 가중
- **Per-task loss_weight**: 수동 설정된 가중치와 uncertainty weighting이 결합
- Evidential KL loss + SAE L1 loss가 추가

---

## 5. 4차원 Ablation 설계

### 5.1 개요

`scripts/run_santander_ablation.py`가 6-Phase, **48 시나리오** ablation study를 오케스트레이션한다. 모든 시나리오는 config에서 동적 생성된다 (`ablation.feature_scenarios: auto`, `ablation.expert_scenarios: auto`).

```
Phase 0   데이터 준비          Processing Job (Stage 1-6)
Phase 1   Feature Group Ablation   16 시나리오 (full + base_only + 7 bottom-up + 7 top-down)
Phase 2   Expert Ablation         16 시나리오 (deepfm baseline + 7 bottom-up + 7 top-down + mlp_only)
Phase 3   Task x Structure Cross   16 시나리오 (4 tiers x 4 structures)
Phase 4   Best Config Teacher + Distillation
Phase 5   Analysis + HTML Report
```

Docker-based 실행: `containers/training/Dockerfile`로 로컬 GPU에서도 동일 환경 재현 가능.

### 5.2 Dimension 1: Feature Group Ablation (16 시나리오)

학계 표준 bottom-up + top-down 설계:

**Advanced 피처 그룹 (base 제외 10개)**:
txn_behavior, derived_temporal, tda_global, tda_local, graph_collaborative, product_hierarchy, hmm_states, mamba_temporal, gmm_clustering, model_derived

**Base**: demographics + product_holdings (항상 포함)

| 유형 | 시나리오 | 제거 대상 | 측정 목적 |
|---|---|---|---|
| Baseline | `full` | 없음 | 전체 성능 기준선 |
| Base | `base_only` | 10개 advanced 전부 | base만의 성능 |
| Bottom-up | `base-txn` | base + txn_behavior만 | txn_behavior의 pairwise 기여 |
| Bottom-up | `base-tda` | base + tda_global + tda_local | TDA의 pairwise 기여 |
| Bottom-up | `base-graph` | base + graph_collaborative | 그래프 협업 필터링 기여 |
| Bottom-up | `base-hierarchy` | base + product_hierarchy | 상품 계층 기여 |
| Bottom-up | `base-hmm` | base + hmm_states | HMM 상태 기여 |
| Bottom-up | `base-mamba` | base + mamba_temporal | Mamba 시계열 기여 |
| Bottom-up | `base-gmm` | base + gmm_clustering | GMM 클러스터링 기여 |
| Top-down | `full-txn` | txn_behavior 제거 | txn_behavior의 irreplaceability |
| Top-down | `full-tda` | tda_global + tda_local 제거 | TDA의 irreplaceability |
| Top-down | `full-graph` | graph_collaborative 제거 | 그래프 CF의 irreplaceability |
| Top-down | `full-hierarchy` | product_hierarchy 제거 | 계층의 irreplaceability |
| Top-down | `full-hmm` | hmm_states 제거 | HMM의 irreplaceability |
| Top-down | `full-mamba` | mamba_temporal 제거 | Mamba의 irreplaceability |
| Top-down | `full-gmm` | gmm_clustering 제거 | GMM의 irreplaceability |

**해석 프레임워크**:
- Bottom-up 기여 = `base-X` 성능 - `base_only` 성능 (해당 그룹의 독립 기여)
- Top-down irreplaceability = `full` 성능 - `full-X` 성능 (다른 그룹이 대체 불가한 정보)
- 기여 > irreplaceability: 해당 그룹은 다른 그룹과 정보 중복
- 기여 < irreplaceability: 해당 그룹은 고유한 정보 제공

### 5.3 Dimension 2: Expert Ablation (16 시나리오)

DeepFM을 기준선으로 한 bottom-up + top-down 설계:

| 유형 | 시나리오 | 전문가 구성 |
|---|---|---|
| Minimal | `deepfm_only` | [deepfm] |
| Bottom-up | `deepfm+temporal` | [deepfm, temporal_ensemble] |
| Bottom-up | `deepfm+hgcn` | [deepfm, hgcn] |
| Bottom-up | `deepfm+perslay` | [deepfm, perslay] |
| Bottom-up | `deepfm+causal` | [deepfm, causal] |
| Bottom-up | `deepfm+lightgcn` | [deepfm, lightgcn] |
| Bottom-up | `deepfm+ot` | [deepfm, optimal_transport] |
| Top-down | `full-deepfm` | 7개 중 deepfm 제외 |
| Top-down | `full-temporal` | 7개 중 temporal_ensemble 제외 |
| Top-down | `full-hgcn` | 7개 중 hgcn 제외 |
| Top-down | `full-perslay` | 7개 중 perslay 제외 |
| Top-down | `full-causal` | 7개 중 causal 제외 |
| Top-down | `full-lightgcn` | 7개 중 lightgcn 제외 |
| Top-down | `full-ot` | 7개 중 optimal_transport 제외 |
| Absolute min | `mlp_only` | [mlp] (shared expert 없이 task expert만) |

> Phase 1의 `full` 결과를 Phase 2의 `full_basket` baseline으로 재사용 (-1 job)

> **FeatureRouter 활성화에 따른 해석 유의사항**: Expert 제거(ablation)는 단순히 해당 Expert의 연산을 제거하는 것이 아니라, 그 Expert에게만 라우팅되던 **피처 경로(feature routing path)도 함께 제거**한다. 예를 들어 `full-perslay` 시나리오에서는 32D TDA 피처 경로가 완전히 차단된다. 따라서 Expert Ablation 결과는 "Expert 구조의 기여"와 "해당 피처 그룹의 기여"를 동시에 반영하므로, Dim 1 Feature Ablation 결과와 교차 비교하여 해석해야 한다.

### 5.4 Dimension 3: Task x Structure Cross (16 시나리오)

4개 태스크 티어 x 4개 구조 변형 = **16 시나리오**.

**태스크 티어**:

| 티어 | 태스크 수 | 포함 범위 |
|---|---|---|
| `tasks_4` | 4 | Tier 1 Core only |
| `tasks_8` | 8 | + Tier 2 Derived |
| `tasks_15` | 15 | + Tier 3 Product Group + Segmentation |
| `tasks_18` | 18 | + Tier 5 Txn NBA |

**구조 변형**:

| 변형 | use_ple | use_adatt | 설명 |
|---|---|---|---|
| `shared_bottom` | false | false | 기본 Shared-Bottom MTL |
| `ple_only` | true | false | PLE gating만 |
| `adatt_only` | false | true | adaTT 전이만 |
| `full` | true | true | PLE + adaTT 전체 |

**핵심 질문**: "태스크를 추가할수록 성능이 향상되는가?" + "PLE/adaTT가 태스크 수에 따라 어떤 차이를 만드는가?"

### 5.5 Phase 4: Best Config Teacher + Distillation

Phase 1-3의 `eval_metrics.json`에서 `aggregate_score`가 가장 높은 설정을 자동 선택:

1. Phase 1 최고 feature 시나리오
2. Phase 2 최고 expert 시나리오
3. Phase 3 최고 task x structure 시나리오

선택된 best config로:
- **Step 4a**: Teacher 모델을 full epochs (30+20)로 학습
- **Step 4b**: Knowledge Distillation -> LGBM student
  - temperature=5.0, alpha=0.3
  - LGBM: num_leaves=127, lr=0.05, n_estimators=500

### 5.6 Phase 5: Analysis + HTML Report

모든 Phase의 결과를 종합하여:
- 4차원 ablation 히트맵
- Feature importance (IG attribution)
- Expert redundancy matrix (CCA)
- 태스크별 성능 비교표
- HTML 리포트 생성

---

## 6. 비용 최적화

### 6.1 학습 하이퍼파라미터 최적화

Ablation 시나리오별 학습 설정 (Phase 1-3):

| 파라미터 | 값 | 근거 |
|---|---|---|
| batch_size | 4096 | 941K / 4096 = ~230 steps/epoch, GPU utilization 최대화 |
| AMP | true | Mixed precision으로 GPU 메모리 50% 절감, 속도 1.5x |
| learning_rate | 0.008 | 큰 batch에 맞춰 상향 (linear scaling rule) |
| epochs | 10 | Ablation에는 수렴 불필요, 상대 비교만 필요 |
| early_stopping_patience | 3 | 빠른 중단으로 비용 절감 |
| gradient_accumulation_steps | 2 | effective batch = 8192 |

> Phase 4 Teacher: epochs=30(phase1)+20(phase2), patience=7, lr=0.001

### 6.2 SageMaker 비용 최적화

| 최적화 | 효과 |
|---|---|
| **ProfilerReport 비활성화** | ~$1.50/job 절감 (Processing Job 제거) |
| **Spot instance 활용** | `use_spot: true`, 최대 70% 비용 절감 |
| **max_parallel=4** | 4대 spot 인스턴스 동시 실행 (같은 AZ 경쟁 방지) |
| **Spot/On-demand 교대** | spot 인스턴스 기본, capacity 부족 시 on-demand fallback |
| **S3 Resume** | 완료된 시나리오 자동 감지 (`eval_metrics.json` 존재 여부 확인) |
| **Checkpoint S3 sync** | `/opt/ml/checkpoints/` -> S3 자동 동기화, spot 중단 시 resume |
| **Phase 1 full 재사용** | Phase 2/3에서 full baseline을 Phase 1 결과로 재사용 (-2 jobs, ~3시간 절감) |

### 6.3 인스턴스 타입

| 용도 | 인스턴스 | 비용 (on-demand) |
|---|---|---|
| GPU 학습 | ml.g4dn.xlarge | ~$0.73/hr |
| CPU 처리 (Phase 0, 5) | ml.m5.xlarge | ~$0.23/hr |
| max_run_seconds | 28800 (8시간) | safety margin |
| max_wait (spot) | 36000 (10시간) | spot capacity 대기 |

---

## 7. 해석/분석 모듈

### 7.1 Integrated Gradients (IG)

`core/evaluation/integrated_gradients.py`

- **알고리즘**: Sundararajan et al. (2017) -- baseline에서 입력까지의 gradient 적분
- **Baseline**: zeros (기본) 또는 mean
- **Integration**: 사다리꼴 적분, n_steps=50
- **효율성**: 각 interpolation step을 batch 단위로 처리 (OOM 방지)
- **API**:
  - `attribute(inputs, target_task)` -> `(batch, input_dim)` per-sample attribution
  - `feature_importance(dataloader, target_task)` -> `{feature_idx: mean_abs_attr}`
  - `top_k_features(dataloader, target_task, k=20)` -> top-K feature indices

### 7.2 Expert Redundancy CCA

`core/evaluation/expert_redundancy.py`

- **알고리즘**: SVD-based Canonical Correlation Analysis (sklearn 무의존)
- **방법**:
  1. Forward hook으로 첫 번째 CGC layer의 shared expert 출력 캡처
  2. 모든 expert 쌍에 대해 CCA 수행
  3. Mean canonical correlation -> redundancy score
- **분류**: HIGH (>0.7), MID (>0.4), LOW (<=0.4)
- **출력**: `RedundancyResult` -- `[n_experts, n_experts]` 대칭 행렬 + pairwise 상관계수
- **파라미터**: n_components=10, min_samples=256, max_batches=20

### 7.3 CGC Gate Analysis

- 각 CGC layer의 gating network이 태스크별로 어떤 expert를 선택하는지 분석
- Gate activation 분포, expert utilization rate, dead expert 탐지

### 7.4 HGCN Interpretable

- 쌍곡 공간에서의 상품 임베딩 시각화
- 상품 간 쌍곡 거리 기반 유사도/대체재 관계 해석

### 7.5 추가 해석 모듈 (config-gated)

| 모듈 | 설명 | 활성화 |
|---|---|---|
| **Template Reason Engine** | L1 batch reason 생성 (IG + rule template) | `serving_prep.reason_generation.l1_batch: true` |
| **XAI Quality Evaluator** | attribution faithfulness/stability 평가 | `analysis.xai_quality.enabled: true` |
| **Model Card Generator** | 자동 모델 카드 생성 | `analysis.model_card.enabled: true` |

### 7.6 Stage C Stubs (미래)

| 모듈 | 역할 | 상태 |
|---|---|---|
| **Context Vector Store** | feature pipeline 출력을 벡터 저장소에 저장 | `serving_prep.context_store.backend: auto` (lancedb/numpy) |
| **CPE** | Context-aware Prediction Engine | Stage 9.5 stub |
| **Agentic Orchestrator** | LLM 기반 추천 이유 생성 (L2a rewrite) | `l2a_rewrite: false` (LLM 필요) |

---

## 8. 스코어링 및 제약조건

### 8.1 FD-TVS Scoring

FD-TVS (Financial Decision - Total Value Score) 복합 스코어링:

```yaml
scoring:
  method: fd_tvs
  weights:
    has_nba: 0.20       # 가입 확률
    nba_primary: 0.30   # 상품 매칭 확신도
    cross_sell_count: 0.15  # 교차판매 기회
    churn_signal: 0.15  # 이탈 위험 (역수)
    product_stability: 0.10  # 안정성
    engagement_score: 0.10   # 활성도
```

**DNA Modifier** (세그먼트별 가중치):

| 세그먼트 | 가중치 |
|---|---|
| 01-TOP | 1.3 |
| 02-PARTICULARES | 1.0 |
| 03-UNIVERSITARIO | 0.8 |
| UNKNOWN | 0.7 |

### 8.2 Constraint Engine

3개 필터 타입:

| 필터 | 규칙 |
|---|---|
| **Fatigue** | 최근 7일 메시지 수 <= 5, decay_rate=0.85 |
| **Eligibility** | min_score >= 0.05, max_churn_prob <= 0.6 |
| **Owned Product** | `prod_*` 컬럼으로 이미 보유 중인 상품 제외 |

### 8.3 Product Eligibility Tiers

`item_universe.yaml`에서 정의된 5개 상품 등급:

| 등급 | min_tenure_months | 상품 예시 |
|---|---|---|
| entry | 0 | junior_acct |
| core | 0 | saving, checking, credit_card, direct_debit, auto_debit |
| standard | 3 | particular_acct, particular_plus, e_account, home_acct, taxes, payroll |
| growth | 6 | short/medium/long deposit, funds, loans, guarantee |
| premium | 12 | mortgage, securities |

### 8.4 Top-K MMR Diversity Selection

```yaml
top_k:
  k: 5
  diversity_method: mmr
  diversity_lambda: 0.5
```

- Maximal Marginal Relevance로 추천 상품 다양성 보장
- lambda=0.5: 관련성과 다양성의 균형
- 상품 간 유사도는 Poincare 임베딩 거리 기반

### 8.5 상품 관계 (대체재/보완재)

**대체재** (같은 니즈 경쟁):
- short/medium/long deposit
- saving vs checking
- particular_acct vs particular_plus
- pension_plan vs pension_deposit
- funds vs securities

**보완재** (함께 구매):
- checking + credit_card
- checking + direct_debit / auto_debit
- payroll_acct + checking
- funds + securities
- pension_plan + pension_deposit

---

## 9. 알려진 이슈 및 향후 과제

### 9.1 범주형 인코딩 (해결됨)

- **현황**: PipelineRunner Stage 2에서 LabelEncoder 기반 categorical encoding 처리
- **train.py**: PyArrow로 로드하므로 이미 numeric encoding된 상태로 전달
- **상태**: 해결됨 (Phase 0에서 encoding → train.py는 ZERO preprocessing)

### 9.2 LabelDeriver 위치 (해결됨)

- **현황**: 모든 label 파생이 Phase 0 (PipelineRunner Stage 5)에서 완료됨
- **train.py**: labels.parquet를 PyArrow로 그대로 로드 (label derivation 없음)
- **Label dedup**: features와 labels에 중복 컬럼이 있으면 features에서 DROP before merge
- **상태**: 해결됨

### 9.3 Phase 0 → Training Job 데이터 흐름 (확립됨)

- **현황**: Phase 0 artifacts → S3 → train.py `load_ready_data()` (PyArrow zero-copy)
- **구조**: features.parquet, labels.parquet, feature_schema.json, label_schema.json, split_indices.json
- **LeakageValidator**: train.py에서 학습 전 자동 호출, >0.95 상관 피처 auto-drop
- **상태**: 확립됨

### 9.4 피처 그룹 생성기 구현 상태

8개 핵심 Generator 모두 구현 완료. cuDF primary / pandas fallback 패턴:

| Generator | 파일 | 상태 | GPU 가속 | 비고 |
|---|---|---|---|---|
| `tda` (global/local) | `core/feature/generators/tda.py` | 구현됨 | cuPY + ripser | Persistence Diagram |
| `hmm` | `core/feature/generators/hmm.py` | 구현됨 | - | hmmlearn Triple-Mode |
| `mamba` | `core/feature/generators/mamba.py` | 구현됨 | GPU (mamba-ssm) | Selective SSM |
| `graph` | `core/feature/generators/graph.py` | 구현됨 | - | Poincare ball model |
| `gmm` | `core/feature/generators/gmm.py` | 구현됨 | cuML (optional) | **GMM soft labels** (not KMeans) |
| `model_derived` | `core/feature/generators/model_features.py` | 구현됨 | - | GMM soft probs(5D) + Bandit(4D) + LNN(18D) |
| `economics` | `core/feature/generators/economics.py` | 구현됨 | - | Income decomp(8D) + Fin behavior(9D) |
| `merchant_hierarchy` | `core/feature/generators/merchant_hierarchy.py` | 구현됨 | - | MCC L1/L2/L3 + Brand SVD |

### 9.5 평가 메트릭 체계

| 태스크 유형 | Primary | 전체 메트릭 |
|---|---|---|
| binary | auc_roc | auc_roc, auc_pr, f1, precision, recall, accuracy, log_loss |
| multiclass | f1_macro | accuracy, f1_macro, f1_weighted, top_k_accuracy |
| regression | rmse | rmse, mae, r2, mape |

### 9.6 향후 과제

1. **2차 Ablation**: PLE/adaTT 구조 + 피처-전문가 연동 + Loss weighting (project_ablation_round2.md 참조)
2. **Serving Pipeline**: Stage C (CPE, Agentic Orchestrator, Vector Store) 구현
3. **Real-time Inference**: SageMaker Endpoint + Lambda 파이프라인
4. **A/B Testing**: SageMaker Experiments 기반 온라인 평가
5. **데이터 갱신**: 주기적 재학습 파이프라인 (SageMaker Pipelines)
6. **모니터링**: Model Monitor + Data Quality 자동 경고

---

## 부록 A: 피처 그룹 상세 (12개 그룹, ~500D)

### State Axis (정적 속성)

| # | 그룹명 | 유형 | 출력 차원 | 대상 Expert | Distill |
|---|---|---|---|---|---|
| 1 | demographics | transform | 38D | deepfm, mlp | O (1.0) |
| 2 | product_holdings | transform | 24D | deepfm, mlp | O (1.0) |
| 3 | txn_behavior | transform | 14D | deepfm, temporal_ensemble | O (1.0) |
| 4 | derived_temporal | transform | 4D | deepfm, causal | O (0.9) |

### Snapshot Axis (장기 요약)

| # | 그룹명 | 유형 | 출력 차원 | 대상 Expert | Distill |
|---|---|---|---|---|---|
| 5 | tda_global | generate | 36D | perslay | O (0.5) |
| 6 | tda_local | generate | 24D | perslay | O (0.5) |
| 7 | hmm_states | generate | 48D | temporal_ensemble | X |

### Timeseries Axis (단기 시계열)

| # | 그룹명 | 유형 | 출력 차원 | 대상 Expert | Distill |
|---|---|---|---|---|---|
| 8 | mamba_temporal | generate | 50D | temporal_ensemble | X |

### Hierarchy Axis (상품 계층)

| # | 그룹명 | 유형 | 출력 차원 | 대상 Expert | Distill |
|---|---|---|---|---|---|
| 9 | product_hierarchy | generate | 32D | hgcn | X |

### Item Axis (협업 필터링)

| # | 그룹명 | 유형 | 출력 차원 | 대상 Expert | Distill |
|---|---|---|---|---|---|
| 10 | graph_collaborative | generate | 64D | lightgcn | X |

### Derived (파생 그룹)

| # | 그룹명 | 유형 | 출력 차원 | 대상 Expert | Distill |
|---|---|---|---|---|---|
| 11 | gmm_clustering | generate | 22D | deepfm, mlp | O (0.8) |
| 12 | model_derived | generate | 27D | temporal_ensemble, deepfm | X | GMM soft probs(5D) + Bandit(4D) + LNN(18D) |

**합계**: ~383D (transform 80D + generate ~303D) + categorical embeddings -> ~500D

---

## 부록 B: 상품 계층 (24개 상품 x 6 L1 그룹)

```
Deposits (4)
  ├── saving [0]           core    0.3%
  ├── short_deposit [8]    growth  4%
  ├── medium_deposit [9]   growth  0.1%
  └── long_deposit [10]    growth  0.1%

Accounts (8)
  ├── checking [2]         core    60%
  ├── derivados [3]        standard 0.2%
  ├── payroll_acct [4]     core    8%
  ├── junior_acct [5]      entry   0.4%
  ├── particular_acct [6]  standard 1%
  ├── particular_plus [7]  standard 11%
  ├── e_account [11]       standard 3%
  └── home_acct [19]       standard 2%

Investments (4)
  ├── funds [12]           growth  8%
  ├── pension_plan [14]    core    0.3%
  ├── securities [18]      premium 4%
  └── pension_deposit [21] core    5%

Loans (2)
  ├── mortgage [13]        premium 2%
  └── loans [15]           growth  1%

Cards & Payments (3)
  ├── credit_card [17]     core    5%
  ├── direct_debit [22]    core    6%
  └── auto_debit [23]      core    12%

Other Services (3)
  ├── guarantee [1]        growth  0.1%
  ├── taxes [16]           standard 0.2%
  └── payroll [20]         standard 0.1%
```

`[index]` = nba_label 인덱스, tier, positive_rate 순

---

## 부록 C: Ablation Orchestrator CLI

```bash
# 전체 실행
python scripts/run_santander_ablation.py --phase all

# 단일 Phase
python scripts/run_santander_ablation.py --phase 3

# Dry run (제출 없이 설정 출력)
python scripts/run_santander_ablation.py --phase all --dry-run

# 기존 teacher 모델로 Phase 4만
python scripts/run_santander_ablation.py --phase 4 --model-uri s3://bucket/model.tar.gz

# 비동기 실행
python scripts/run_santander_ablation.py --phase 1 --no-wait
```

**환경변수**:
- `SANTANDER_S3_BASE`: S3 base path 설정 시 완료된 시나리오 자동 스킵 (resume)
