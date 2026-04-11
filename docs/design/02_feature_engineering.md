# 02. Feature Engineering — 5-Axis 피처 매핑, Generator, 정규화, LabelDeriver

## 10+ Stage 파이프라인에서의 위치

Feature Engineering은 Stage 4 ~ Stage 6을 담당한다:

```
Stage 4:   FeatureGroupPipeline + Normalization (per axis generators + PowerLawAwareScaler)
Stage 5:   LabelDeriver (14 tasks, config-driven derivation)
Stage 5.5: LeakageValidator (sequence/correlation/product/temporal)
Stage 6:   SequenceBuilder (time-based + sliding window → sequences.npy, seq_lengths.npy)
```

---

## FeatureGroupPipeline — 피처 엔지니어링 오케스트레이터

`core/pipeline/features.py`의 `FeatureGroupPipeline`이 Stage 4 피처 엔지니어링의 주 진입점이다.

### Expert Routing — Feature Group 단위 원칙 (2026-04-11)

**expert_routing은 개별 컬럼 이름이 아닌 feature group 이름을 기준으로 해야 한다.** Phase 0의 3-stage 정규화는 power-law 컬럼마다 `_log` 접미사 복사본을 추가하므로, 컬럼 이름 기반 routing은 이 재배열 이후 잘못된 슬라이스를 Expert에 전달한다.

`feature_group_ranges`는 연속된 컬럼 블록으로 저장해야 한다 (contiguous blocks). min~max index 방식은 정규화 후 삽입된 `_log` 컬럼이 블록 경계를 넘어갈 때 인접 group의 컬럼을 포함하는 오류를 일으킨다. 비연속 매칭 발생 시 가장 긴 연속 블록을 사용하거나 컬럼 이름 리스트로 range를 저장한다.

### FeatureGroupPipeline vs Adapter 역할 분리

| 역할 | DataAdapter (Stage 1) | FeatureGroupPipeline (Stage 4) |
|------|----------------------|-------------------------------|
| DuckDB 집계 | O (24M → 2,000 user) | X |
| RFM 기본 피처 | O (base aggregation) | X |
| TDA (global/local) | X | O — Generator 호출 |
| HMM 상태 추정 | X | O — Generator 호출 |
| Mamba SSM | X | O — Generator 호출 |
| Graph 임베딩 | X | O — Generator 호출 |
| GMM 클러스터링 | X | O — Generator 호출 |
| Normalization | X | O — PowerLawAwareScaler |

### TDA Global/Local 구분

TDA 피처는 두 개의 별도 Generator 호출로 분리된다:

- **TDA Global** (Snapshot 축, 36D+10D): 12개월 장기 윈도우 Persistence Diagram
- **TDA Local** (Timeseries 축, 24D): 90일 단기 윈도우 Persistence Diagram

---

## 5-Axis Feature Group 매핑 (Santander 데이터)

### 전체 구조

```
5-Axis Classification
    ↓
    ├── State (정적)
    │   ├── base_rfm (34D) — 인구통계, RFM 프로필
    │   ├── base_category (64D) — 업종별 소비 패턴
    │   ├── multi_source (91D) — 예금/카드/멤버십/투자/디지털
    │   ├── extended_source (84D) — 보험/상담/캠페인/해외결제
    │   ├── economics (17D) — MPC, 소득 탄력성, 항상소득 가설
    │   └── multidisciplinary (24D) — 화학동역학/전염병확산/간섭/범죄패턴
    │
    ├── Snapshot (장기)
    │   ├── tda_topology [global] (36D+10D) — 12개월 Persistence Diagram + Phase Transition
    │   ├── hmm_states (48D) — HMM Triple-Mode 상태 전이 (journey/lifecycle/behavior)
    │   ├── gmm_clustering (22D) — 장기 고객 세그먼트 (20 soft + entropy + dominant)
    │   ├── model_derived [HMM/LNN] (23D) — HMM summary(5) + LNN(18)
    │   └── 상품 트렌드 — 월별 상품 보유 변화 패턴
    │
    ├── Timeseries (단기)
    │   ├── tda_topology [local] (24D) — 90일 Persistence Diagram (short window)
    │   ├── base_temporal (60D) — 시계열 집계 + 주기 인코딩
    │   ├── mamba_temporal (50D) — Mamba SSM 시퀀스 출력
    │   ├── base_txn_stats (80D) — 최근 기간 거래 통계
    │   └── PatchTST — Patch 기반 시계열 트랜스포머
    │
    ├── Hierarchy (구조)
    │   ├── merchant_hierarchy (27D) — MCC 계층의 Poincaré 쌍곡 임베딩 (27D). HGCN 라우팅
    │   ├── graph_embeddings (20D) — Poincare 쌍곡 임베딩 (MCC/상품/지역 계층)
    │   └── product_hierarchy — 상품 카테고리 트리 (24개 금융 상품)
    │
    └── Item (관계)
        ├── customer_product_bipartite — 고객×상품 상호작용 그래프
        └── lightgcn_collaborative (64D) — LightGCN 협업 필터링 임베딩
```

### 차원 요약

| Axis | Feature Groups | Total Dim | Target Experts |
|------|---------------|-----------|----------------|
| **State** | 6 groups | ~314D | DeepFM, MLP, AutoInt |
| **Snapshot** | 4-5 groups | ~139D | PersLay, Causal, OT |
| **Timeseries** | 4-5 groups | ~214D | Temporal Ensemble, Mamba |
| **Hierarchy** | 2-3 groups | ~41D | HGCN |
| **Item** | 2 groups | ~64D | LightGCN |
| **합계** | | ~772D+ | 11종 Expert Pool |

---

## Axis별 Feature Generator 상세

### State 축 — 정적 속성

```yaml
- name: base_rfm
  axis: state
  group_type: transform
  output_dim: 34
  target_experts: [deepfm, mlp]

- name: economics
  axis: state
  group_type: generate
  generator: economics
  output_dim: 17
  target_experts: [deepfm]

- name: multidisciplinary
  axis: state
  group_type: generate
  generator: multidisciplinary
  output_dim: 24
  target_experts: [deepfm]
  # 4 subgroups (6D each): chemical_kinetics, epidemic, crime, interference
  # Per-task-group routing: engagement←chemical, lifecycle←epidemic, etc.
```

### Snapshot 축 — 장기 패턴

```yaml
- name: tda_global
  axis: snapshot
  generator: tda
  generator_params:
    window_days: 365
    output_dim: 46        # long(36) + phase_transition(10)
  target_experts: [perslay]

- name: hmm_states
  axis: snapshot
  generator: hmm
  generator_params:
    modes: [journey, lifecycle, behavior]   # Triple-Mode
    state_dim: 16
  output_dim: 48          # 3 modes x 16D
  target_experts: [temporal_ensemble]
  # HMM Triple-Mode → task group routing:
  #   journey  → value, consumption groups
  #   lifecycle → lifecycle group
  #   behavior  → engagement group

- name: gmm_clustering
  axis: snapshot
  generator: gmm
  generator_params:
    n_components: 20       # K=20 clusters
    covariance_type: full
    # BIC-based validation: warns if K is suboptimal
    # Cold-start fallback: uniform distribution for small data
  output_dim: 22           # K soft probs(20) + entropy(1) + dominant(1)
  target_experts: [deepfm, mlp]
  # NOTE: GMM soft labels (not KMeans) — posterior probabilities + Shannon entropy
```

### Timeseries 축 — 단기 시퀀스

```yaml
- name: tda_local
  axis: timeseries
  generator: tda
  generator_params:
    window_days: 90
    output_dim: 24
  target_experts: [temporal_ensemble]

- name: mamba_temporal
  axis: timeseries
  generator: mamba
  generator_params:
    d_model: 128
    d_state: 16
    expand: 2
    output_dim: 50
  target_experts: [temporal_ensemble]
```

### Hierarchy 축 — 구조적 계층

```yaml
- name: graph_embeddings
  axis: hierarchy
  generator: graph
  generator_params:
    hierarchy_sources: [mcc, product, region]
    curvature: 1.0
  output_dim: 20
  target_experts: [hgcn]

- name: merchant_hierarchy
  axis: hierarchy
  generator: merchant_hierarchy
  generator_params:
    mcc_hierarchy_path: configs/mcc_hierarchy.yaml   # ISO 18245, L1/L2/L3
    n_svd_components: 8       # Brand SVD embedding dim
    poincare_dim: 27          # Poincaré hyperbolic embedding of MCC hierarchy
  output_dim: 27              # Poincaré 27D embedding (쌍곡 공간, MCC 계층 구조 인코딩)
  target_experts: [hgcn]
  # MCC Hierarchy: 10 L1 groups, ~30 L2 subcategories, 109 L3 codes in dataset
  # NOTE: txn_mcc_seq는 generator 호출 전에 마지막 1원소 제거 (next_mcc 누수 방지)

- name: product_hierarchy
  axis: hierarchy
  generator: product_hierarchy
  generator_params:
    hierarchy_depth: 2        # Level 1: 예금/대출/투자/보험/카드/기타
    embedding_dim: 16         # Level 2: 24개 세부 상품
  output_dim: 16
  target_experts: [hgcn]
```

### Item 축 — 관계적 상호작용

```yaml
- name: customer_product_bipartite
  axis: item
  generator: bipartite_graph
  generator_params:
    customer_col: customer_id_idx
    product_cols_prefix: "ind_"
    edge_weight: holding_duration
  output_dim: 64
  target_experts: [lightgcn]

- name: lightgcn_collab
  axis: item
  generator: lightgcn
  generator_params:
    num_layers: 3
    embedding_dim: 64
  output_dim: 64
  target_experts: [lightgcn]
```

---

## Feature Generator Registry

### 8 구현 완료 Generator

모든 Generator는 `core/feature/generators/` 디렉토리에 구현되어 ���으며, `FeatureGeneratorRegistry`에 ���록된다. cuDF primary / pandas fallback 패턴을 따른다. Generator output은 cuDF DataFrame 또는 pandas DataFrame이다.

| # | 등록 이름 | 파일 | Axis | Output | GPU 가속 | 설명 |
|---|-----------|------|------|--------|---------|------|
| 1 | `tda` | `core/feature/generators/tda.py` | Snapshot/Timeseries | 70D | cuPY + ripser | Persistence Diagram (short+long) |
| 2 | `hmm` | `core/feature/generators/hmm.py` | Snapshot | 48D | hmmlearn | HMM Triple-Mode 상태 |
| 3 | `mamba` | `core/feature/generators/mamba.py` | Timeseries | 50D | GPU (mamba-ssm) | Mamba SSM 시퀀스 |
| 4 | `graph` | `core/feature/generators/graph.py` | Hierarchy | 20D | - | Poincare 쌍곡 임베딩 |
| 5 | `gmm` | `core/feature/generators/gmm.py` | Snapshot | 22D | cuML (optional) | GMM soft labels (K=20, not KMeans) |
| 6 | `model_derived` | `core/feature/generators/model_features.py` | Snapshot | 27D | - | GMM soft probs(5D) + Bandit(4D) + LNN(18D) |
| 7 | `economics` | `core/feature/generators/economics.py` | State | 17D | - | Income decomposition(8D) + Financial behavior(9D) |
| 8 | `merchant_hierarchy` | `core/feature/generators/merchant_hierarchy.py` | Hierarchy | 27D | - | MCC 계층의 Poincaré 쌍곡 임베딩 (27D). txn_mcc_seq는 next_mcc 누수 방지를 위해 마지막 원소 제거 후 전달 |

### 추가 Generator (보조)

| # | 등록 이름 | 파일 | Axis | Output | 설명 |
|---|-----------|------|------|--------|------|
| 9 | `temporal_pattern` | `core/feature/generators/temporal.py` | Timeseries | 가변 | 시계열 집계 + 주기 인코딩 |
| 10 | `multidisciplinary` | `core/feature/generators/multidisciplinary.py` | State | 24D | 화학/전염병/간섭/범죄 |
| 11 | `phase_transition` | `core/feature/generators/phase_transition.py` | Snapshot | 10D | Phase transition features |

### GPU Utility Layer

`core/feature/generators/gpu_utils.py`가 모든 Generator에 공통 GPU 유틸리티를 제공:
- Device detection (`get_device(prefer_gpu=True)`)
- Adaptive batch sizing
- OOM-retry decorator (GPU 메모리 부족 시 자동 CPU fallback)
- `has_cudf()`, `has_cuml()`, `has_cupy()` lazy import 체크

---

## Stage 4: Feature Integration + Normalization

### Power-law 자동 감지 + StandardScaler

```
Numeric Features (all axes merged)
    ↓
[Skewness 검사]─── skewness > 2.0 ──▶ log1p 변환 (power-law 자동 감지)
    ↓                                      ↓
    │ (원본 유지)                      (log1p 변환본)
    ↓                                      ↓
    └──── log1p + raw 병렬 결합 ──────────┘
                   ↓
          StandardScaler (z-score: mean=0, std=1)
                   ↓
        scaler_params.json 저장 (mean, std, log1p_cols)
```

### 구현

```python
# core/feature/normalizer.py
class PowerLawAwareScaler:
    """
    Power-law 자동 감지 + StandardScaler.
    1. fit 시 각 컬럼의 skewness 계산
    2. skewness > 2.0인 컬럼에 log1p 변환 (원본은 유지)
    3. log1p 변환본 + 원본을 병렬로 결합
    4. 전체에 StandardScaler 적용 (z-score)
    5. get_params() → scaler_params.json 저장
    cuPY 가속: cupy 설치 시 skewness/mean/std 계산을 GPU에서 수행.
    """
```

---

## Santander 데이터 피처 상세 (configs/santander/)

### Categorical Embeddings

| 컬럼 | Cardinality | Embedding Dim | 비고 |
|------|-------------|---------------|------|
| `gender` | 2 | 4 | F, M |
| `segment` | 4 | 4 | 01-TOP, 02-PARTICULARES, 03-UNIVERSITARIO, UNKNOWN |
| `country` | 118 | 16 | 118개국 (ES 96%) |
| `channel` | 163 | 16 | 163개 채널 코드 |
| `age_group` | 5 | 4 | young/adult/middle/senior/elderly |
| `income_group` | 5 | 4 | low/mid/high/very_high/unknown |

### Numeric Features

| 컬럼 | 범위 | 특이사항 |
|------|------|---------|
| `age` | 18-100 | 정상 범위 |
| `income` | 0-28.8M EUR | 25.4% missing (0=결측) |
| `tenure_months` | -999999~256 | sentinel 처리 필요 |
| `is_active` | 0/1 | 42% active |
| `num_products` | 0-15 | avg 1.3 |

---

## 14-Task Label Architecture

### 4-Tier 태스크 구조

| Tier | 설명 | 태스크 수 | 예시 |
|------|------|----------|------|
| **Tier 1** | Core targets (직접 레이블) | 4 | has_nba, churn_signal, product_stability, nba_primary |
| **Tier 2** | Derived targets (규칙 유도) | 1 | cross_sell_count |
| **Tier 3** | Product group + segmentation | 6 | will_acquire_{deposits,investments,accounts,lending,payments}, segment_prediction |
| **Tier 5** | Transaction-based NBA | 3 | next_mcc, mcc_diversity_trend, top_mcc_shift |

### Per-task Focal Alpha Calibration

Binary 태스크의 `focal_alpha`는 positive rate에 따라 calibrated:

| 태스크 | Positive Rate | focal_alpha | 근거 |
|--------|-------------|-------------|------|
| `has_nba` | 2.98% | 0.90 | 극도로 불균형 |
| `churn_signal` | 5.1% | 0.85 | 매우 불균형 |
| `will_acquire_deposits` | ~1% | 0.95 | 극단적 불균형 |
| `will_acquire_investments` | ~2% | 0.90 | 매우 불균형 |
| `will_acquire_accounts` | ~5% | 0.85 | 불균형 |
| `will_acquire_lending` | ~1% | 0.90 | 극단적 |
| `will_acquire_payments` | ~3% | 0.85 | 매우 불균형 |
| `top_mcc_shift` | ~10% | 0.70 | 경도 불균형 |

### 4 Semantic Groups (adaTT Transfer Units)

```yaml
task_groups:
  engagement:   [has_nba, next_mcc, top_mcc_shift]
  lifecycle:    [churn_signal, product_stability, segment_prediction]
  value:        [mcc_diversity_trend, cross_sell_count]
  consumption:  [nba_primary, will_acquire_deposits,
                 will_acquire_investments, will_acquire_accounts,
                 will_acquire_lending, will_acquire_payments]
```

---

## Item Universe + Product Hierarchy

### Item Universe (24 금융 상품)

```
┌──────────────────────────────────────────────────┐
│ 예금: saving, guarantee, checking, derivados     │
│ 계좌: payroll_acct, junior_acct, particular_acct │
│       particular_plus, e_account, home_acct      │
│ 예탁: short_deposit, medium_deposit, long_deposit│
│ 투자: funds, securities, pension_plan,           │
│       pension_deposit                             │
│ 대출: mortgage, loans                            │
│ 결제: credit_card, direct_debit, auto_debit      │
│ 세금: taxes                                      │
│ 보수: payroll                                    │
└──────────────────────────────────────────────────┘
```

### Product Hierarchy Config

```json
{
  "hierarchy": {
    "level_1": {
      "deposits": ["saving", "short_deposit", "medium_deposit", "long_deposit"],
      "accounts": ["checking", "payroll_acct", "junior_acct", "particular_acct", "..."],
      "investments": ["funds", "securities", "pension_plan", "pension_deposit"],
      "lending": ["mortgage", "loans"],
      "payments": ["credit_card", "direct_debit", "auto_debit"]
    }
  }
}
```

---

## 피처 차원 관리 — 동적 계산

```python
# core/feature/schema.py
class FeatureSchema:
    @property
    def input_dim(self) -> int:
        return sum(g.output_dim for g in self.groups if g.enabled)

    @property
    def axis_dims(self) -> dict[str, int]:
        dims = defaultdict(int)
        for g in self.groups:
            if g.enabled:
                dims[g.axis] += g.output_dim
        return dict(dims)
        # → {"state": 314, "snapshot": 139, "timeseries": 214, "hierarchy": 41, "item": 64}
```

---

## SageMaker Processing에서 실행

```
Step Functions
    ↓
SageMaker Processing Job
    ├── 입력: s3://bucket/data/encrypted/
    ├── 코드: core/feature/pipeline_builder.py
    ├── config: configs/santander/feature_groups.yaml
    ├── 엔진: DuckDB (인프로세스)
    ├── GPU: cuPY/cuDF optional (ml.g4dn.xlarge)
    └── 출력: s3://bucket/features/v{version}/
         ├── features.parquet
         ├── labels.parquet
         ├── sequences.npy
         ├── seq_lengths.npy
         ├── feature_schema.json
         ├── label_schema.json
         ├── split_indices.json
         ├── scaler_params.json
         ├── label_transforms.json
         ├── item_universe/
         │   ├── product_hierarchy.json
         │   └── customer_product_graph.parquet
         └── transformers/
             └── transformer_00_PowerLawAwareScaler.pkl
```

---

## 현재 vs AWS — 피처 엔지니어링 비교

| 항목 | 현재 (On-Prem) | AWS (설계) | 변경 이유 |
|------|---------------|-----------|----------|
| 피처 분류 | 없음 (flat 목록) | **5-Axis** (State/Snapshot/Timeseries/Hierarchy/Item) | Expert 라우팅 명시적 기반 |
| 피처 정의 | 코드 내 하드코딩 | YAML 선언형 (feature_groups.yaml) | 코드 변경 없이 피처 추가/삭제 |
| 차원 관리 | 644D, 734D 하드코딩 | `FeatureSchema.input_dim` 동적 계산 + axis_dims | 피처 변경 시 자동 반영 |
| 정규화 | QuantileTransform + Raw Power-Law | **StandardScaler + power-law 자동 감지** (skew>2.0 → log1p+raw) | 단일 경로, 분포 보존 |
| TDA | 단일 윈도우 | **Global(365일)/Local(90일) 분리** → Snapshot/Timeseries 축 | 장기/단기 위상 분리 |
| HMM | 코드에 직접 구현 | Generator Registry (hmm_triple_mode) | 선택적 활성화 |
| Graph | HGCN만 | **Poincare + LightGCN** (Hierarchy + Item 축) | 계층+협업 분리 |
| Item Universe | 없음 | **고객x상품 bipartite graph** (24 금융 상품) | 상품 추천 핵심 |
| 레이블 생성 | 코드 하드코딩 | **LabelDeriver** (config-driven 14 tasks) | 선언적, 재현 가능 |
| 누수 방지 | 없음 | **LeakageValidator** (4-check) + temporal split | 자동 누수 감지 |
| 피처 버전 | 없음 | features/v{version}/ | 재현성, 롤백 |
| Cold Start | 없음 | **is_cold_start flag + sequence-derived feature zeroing** | cold start 고객 대응 |
| GPU 가속 | 없음 | cuDF primary (generators), cuPY (TDA, scaler), cuML (GMM) | 5-10x 가속 |
