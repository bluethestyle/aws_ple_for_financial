# 01. Data Layer — DataAdapter, TemporalPrep, 암호화, 5-Axis, Temporal Split, LeakageValidator

## 10+ Stage 파이프라인에서의 위치

Data Layer는 Stage 1 ~ Stage 7을 담당한다:

```
Stage 1:   DataAdapter (Raw Data Load + Schema Validation)
Stage 1.5: TemporalPrep (Leakage Prevention — seq truncation, prod recompute)
Stage 2:   SchemaClassifier (5-axis)
Stage 3:   EncryptionPipeline (PII → SHA256 salt → INT32)
Stage 4:   FeatureGroupPipeline + Normalization
Stage 5:   LabelDeriver (14 tasks)
Stage 5.5: LeakageValidator (sequence/correlation/product/temporal)
Stage 6:   SequenceBuilder (flat → 3D tensors)
Stage 7:   DataLoader (temporal split with gap_days=30)
```

---

## DataAdapter ABC 및 AdapterRegistry (Stage 1)

Stage 1의 데이터 로딩은 `DataAdapter` 추상 클래스를 통해 표준화된다.

### DataAdapter 계약

```python
# core/pipeline/adapter.py
class DataAdapter(ABC):
    """데이터셋별 원시 데이터 로딩.
    - load_raw() → Dict[str, pd.DataFrame] 반환
    - 최소 "main" 키 필수 (entity-level DataFrame)
    - 피처 엔지니어링 수행 금지 (FeatureGroupPipeline 담당)
    """

class AdapterMetadata:
    id_col: str           # "user_id"
    timestamp_col: str    # optional
    entity_granularity: str  # "user" | "transaction"
    num_entities: int
    num_raw_rows: int
    source_files: List[str]
    backend_used: str     # "cudf" | "duckdb" | "pandas"
```

### AdapterRegistry

```python
class AdapterRegistry:
    @classmethod
    def register(cls, name: str): ...    # 데코레이터
    @classmethod
    def build(cls, name: str, config: dict) -> DataAdapter: ...
    @classmethod
    def list_registered(cls) -> List[str]: ...
```

### 백엔드 선택 (cuDF → DuckDB → Pandas)

`DataAdapter._select_backend()`는 config `data.backend` 리스트를 순회하며 사용 가능한 첫 번째 백엔드를 선택한다. 기본 순서: `["cudf", "duckdb", "pandas"]`.

### 현재 등록된 어댑터

| 이름 | 파일 | 데이터 | 특성 |
|------|------|--------|------|
| `ealtman2019` | `adapters/ealtman2019_adapter.py` | 24M 신용카드 거래 (2K users, 6,146 cards) | DuckDB → ~469D features + 16 labels, 180-step sequence tensor |
| `santander` | `adapters/santander_adapter.py` | 941K 사용자 × 89 컬럼 + real txn data | DuckDB-native pipeline, cold start 처리 |

### Real Transaction Data 통합

`scripts/augment_santander_with_real_txns.py`가 segment-based pooling으로 실거래 데이터를 Santander 고객에 매칭한다:

| 항목 | 값 |
|------|------|
| Real 데이터 | ealtman2019: 2,000 users, 24.4M transactions |
| 매칭 기준 | (age_group, income_group, activity_level) segment pooling |
| Augmented 결과 | 941K 고객 x augmented txn sequences |
| Noise 파라미터 | amount +/-15%, MCC swap 10%, gap +/-20% (config-driven) |
| 처리 엔진 | 전체 DuckDB SQL (pandas 없음) |

### MCC Hierarchy (configs/mcc_hierarchy.yaml)

ISO 18245 기반 3-level 계층 구조:
- **L1**: 10개 Major category (travel_entertainment, food_beverage, retail, ...)
- **L2**: ~30개 Sub-category (airlines, grocery, restaurants, ...)
- **L3**: 109개 Individual MCC code (데이터셋 내 고유 코드)

### Cold Start 고객 처리

`santander_adapter.py`의 Phase 0 `__main__`에서:
1. `is_cold_start` 플래그 컬럼 추가 (NULL sequence or length <= min_txn_count)
2. Cold start 고객의 sequence-derived feature를 0으로 zeroing (synth_*, temporal 접두사)
3. Config-driven: `pipeline.yaml > cold_start > min_txn_count, zero_prefixes`

---

## Stage 1.5: TemporalPrep (Leakage Prevention)

`core/pipeline/temporal_split.py`의 `TemporalSplitter`가 시퀀스 데이터의 누수를 방지한다.

### 시퀀스 절단 (Sequence Truncation)

Santander 데이터의 17개월 상품 보유 시퀀스에서 month 17(레이블 월)을 제거:

```
원본: [m1, m2, ..., m16, m17]  ← m17이 레이블
절단: [m1, m2, ..., m16]       ← 피처에서 제거
```

Config:
```yaml
product_sequences:
  max_len: 16
  truncate_last: 1    # drop last 1 month
```

### 제품 컬럼 재계산 (Product Recompute)

`prod_*` 스냅샷 컬럼은 month 17(레이블 상태)이 아닌 month 16(피처 상태)에서 재계산:

```yaml
leakage_prevention:
  recompute_prod_from_seq: true   # prod_* ← seq_* month 16
  validate_after_split: true      # LeakageValidator 자동 실행
```

---

## Stage 2: SchemaClassifier (5-Axis)

모든 피처를 5개 축으로 자동 분류:

| Axis | 시간 의존성 | 분류 키워드 |
|------|-----------|------------|
| **State** | 없음 (정적) | 기본값 (매칭 없을 때) |
| **Snapshot** | 장기 (월/분기) | tda_long, hmm, snapshot, trend |
| **Timeseries** | 단기 (일/주) | temporal, sequence, mamba, tda_short |
| **Hierarchy** | 구조적 | mcc, hierarchy, poincare |
| **Item** | 관계적 | graph, bipartite, lightgcn |

### 5-Axis 분류 체계

```
┌─────────┬────────────────────┬──────────────┬──────────────────────────┐
│ Axis    │ 특성               │ 시간 의존성   │ Santander 데이터 예시     │
├─────────┼────────────────────┼──────────────┼──────────────────────────┤
│ State   │ 정적 속성, 느린 변화│ 없음/연 단위  │ 나이, 성별, 가입일,      │
│         │                    │              │ 지역, RFM 기본 프로필    │
├─────────┼────────────────────┼──────────────┼──────────────────────────┤
│ Snapshot│ 장기 집계/요약      │ 월/분기 단위  │ 12개월 TDA 위상, HMM     │
│         │                    │              │ 상태 전이, 상품 트렌드    │
├─────────┼────────────────────┼──────────────┼──────────────────────────┤
│Timeseries│단기 시퀀스 패턴    │ 일/주 단위   │ 최근 90일 거래 시퀀스,    │
│         │                    │              │ 단기 TDA, Mamba 출력     │
├─────────┼────────────────────┼──────────────┼──────────────────────────┤
│Hierarchy│ 구조적 계층 관계    │ 없음 (구조적) │ MCC L1/L2 코드,          │
│         │                    │              │ 상품 카테고리 트리        │
├─────────┼────────────────────┼──────────────┼──────────────────────────┤
│ Item    │ 관계적 상호작용     │ 중간 (갱신)  │ 고객×상품 bipartite,     │
│         │                    │              │ 협업 필터링 임베딩        │
└─────────┴────────────────────┴──────────────┴──────────────────────────┘
```

---

## Stage 3: EncryptionPipeline (SHA256 → INT32)

`core/security/` 모듈이 PII 보호를 담당한다. 스키마의 `pii: true` 마킹에서 자동으로 암호화 정책을 유도한다.

### 전체 흐름

```
Raw DataFrame (PII 포함)
    │
    ▼
EncryptionPipeline.process_source()
    │
    ├── Step 1: Drop (contact/personal_id → phone, email, SSN 삭제)
    │
    ├── Step 2: SHA256 Hash
    │   ├── PIIEncryptor.hash_dataframe()
    │   ├── domain-specific salt (PIIDomain별 고유 salt)
    │   ├── SHA256(salt + str(value)) → 32-byte digest
    │   └── '{col}_hashed' 컬럼 생성, 원본 삭제
    │
    ├── Step 3: Integer Index
    │   ├── PIIIntegerIndexer.index_dataframe()
    │   ├── hash BLOB → INT32 global index (append-only)
    │   ├── '{col}_idx' 컬럼 생성, hashed 삭제
    │   └── -1: null sentinel, 0: reserved, 1~: valid
    │
    └── Step 4: Audit report (source, rows, dropped/hashed/indexed counts)
    │
    ▼
Clean DataFrame (PII 제거, INT32 인덱스만)
    + Index tables (S3 Parquet 영속)
    + Audit log
```

### PIIDomain 체계

```python
class PIIDomain(Enum):
    CUSTOMER = "customer"       # customer_id, cust_no, csno
    ACCOUNT = "account"         # account_no, actno, deps_actno
    CARD = "card"               # card_no, chk_cdno
    MERCHANT = "merchant"       # merchant_id, frcs_no
    TRANSACTION = "transaction"
    INSURANCE = "insurance"     # insurance_policy, insr_plcy_no
    CONSULTATION = "consultation"
    CAMPAIGN = "campaign"
    MARKETING = "marketing"
    OPEN_BANKING = "open_banking"
    E_FINANCE = "e_finance"
    MEMBERSHIP = "membership"
    FUND_FOREX = "fund_forex"
    CONTACT = "contact"         # phone, email → DROP (해싱도 안 함)
    PERSONAL_ID = "personal_id" # SSN, passport → DROP
    DEFAULT = "default"
```

---

## Stage 5: LabelDeriver (14 Tasks)

`core/pipeline/label_deriver.py`가 config-driven으로 14개 레이블을 생성한다.

### 지원 derivation 방법

| Method | 설명 | 예시 |
|--------|------|------|
| `direct` | 기존 컬럼 그대로 사용 | `has_nba`, `churn_signal` |
| `bucket` | 연속값 → 구간 분류 | `tenure_months` → 5 class |
| `weighted_sum` | 가중합 + normalize | `engagement_score` |
| `product_group_acquisition` | 상품 그룹 보유 변화 감지 | `will_acquire_deposits` |
| `categorical_encode` | 범주형 → 정수 인코딩 | `segment_prediction` |
| `list_first` | 리스트의 첫 번째 항목 | `nba_primary` |
| `sequence_next` | 시퀀스 다음 아이템 예측 | `next_mcc` |
| `sequence_diversity_trend` | 시퀀스 다양성 변화 | `mcc_diversity_trend` |
| `sequence_mode_shift` | 시퀀스 최빈값 변화 | `top_mcc_shift` |

### 회귀 레이블 변환

```json
{
  "product_stability": {"clip_value": 0.99, "transform": "none"},
  "cross_sell_count":  {"clip_percentile": 99.5, "transform": "log1p"},
  "mcc_diversity_trend": {"clip_percentile": 99.5, "transform": "none"}
}
```

---

## Stage 5.5: LeakageValidator

`core/pipeline/leakage_validator.py`가 4가지 누수 검증을 수행한다.

### 검증 항목

| 검증 | 설명 | 실패 시 |
|------|------|---------|
| **Sequence Leakage** | 시퀀스가 예측 윈도우(month 17)에 침투하는지 확인 | CRITICAL fail |
| **Feature-Label Correlation** | Pearson > threshold인 피처 플래그 | WARNING |
| **Temporal Leakage** | Train set에 val/test 분할 경계 이후 데이터 포함 여부 | CRITICAL fail |
| **Product Column Leakage** | prod_* 컬럼이 month 16 상태인지 확인 (month 17 아닌지) | CRITICAL fail |

```python
validator = LeakageValidator()
result = validator.validate(features_df, labels_df, config)
if not result.passed:
    for warning in result.warnings:
        logger.error(f"LEAKAGE: {warning}")
```

---

## Stage 7: DataLoader (Auto-detect Split Strategy)

`containers/training/train.py`의 `main()`이 데이터 특성에 따라 자동으로 split 전략을 결정한다.

### Cross-sectional Auto-detect

```python
# train.py: split strategy selection
if >80% of rows share the same date:
    → Cross-sectional data → random split (seeded)
else:
    → Multi-date data → temporal split (DuckDB SQL)
```

### Temporal Split Config (multi-date 데이터일 경우)

```yaml
temporal_split:
  enabled: true
  date_col: snapshot_date
  gap_days: 30          # 30일 갭 (누수 방지)
  train_ratio: 0.7
  val_ratio: 0.15       # test = 1 - 0.7 - 0.15 = 0.15
```

```
Time ──────────────────────────────────────────────────▶
     │ Train (70%)  │ gap │ Val (15%) │ gap │ Test (15%) │
                     30d                 30d
```

### Per-task Validation Split Strategy

`data.split_strategy`에서 task group별 val_method를 지정할 수 있다:
- `random`: 전체 val set 사용 (기본)
- `temporal_latest`: val set 중 최신 snapshot_date 행만 평가

```yaml
data:
  split_strategy:
    lifecycle:
      val_method: temporal_latest
      tasks: [churn_signal, product_stability]
    engagement:
      val_method: random
```

---

## Preprocessing (Stage 내 DuckDB-only)

DuckDB는 단일 머신에서 **수백 GB까지 처리 가능**하다. Pandas fallback 경로는 **완전 제거**.

### Santander 전처리 특수 사항

| 항목 | 처리 | 이유 |
|------|------|------|
| `income` (25.4% missing) | null_indicator=0 → median 대체 | 0이 결측 의미 |
| `tenure_months` (-999999 sentinel) | clip_and_flag (0~256) + `tenure_unknown` 플래그 | sentinel 정규화 |
| `prod_*` 재계산 | seq_* month 16에서 recompute | 누수 방지 |

---

## 시퀀스 데이터 (Stage 6: SequenceBuilder)

`core/pipeline/sequence_builder.py`가 flat DataFrame을 3D 텐서로 변환한다.

### SequenceBuilder 모드

| 모드 | 설명 | 설정 |
|------|------|------|
| **count_based** (legacy) | 마지막 `max_len` 항목 슬라이싱 | `mode: count_based` |
| **time_based** | date-range window 필터링 (sliding window 지원) | `mode: time_based, window_days: 90` |

Time-based 모드는 `timestamp_col`이 데이터에 존재하면 auto-detect된다.

### Sliding Window Bootstrapping

`stride_days > 0` 설정 시 entity당 여러 overlapping window sample 생성:

```yaml
txn_sequences:
  mode: time_based
  window_days: 90
  stride_days: 30        # 30일 간격 sliding window
  max_len: 200           # safety cap
  timestamp_col: txn_date
```

### 거래 시퀀스 (ealtman 기반)

```yaml
txn_sequences:
  max_len: 60
  columns:
    txn_amount_seq: {feat_dim: 1, dtype: float}
    txn_mcc_seq: {feat_dim: 1, dtype: int}
    txn_day_offset_seq: {feat_dim: 1, dtype: int}   # 날짜 대신 일수 오프셋 (snap_date 기준)
```

**txn_day_offset_seq**: YYYYMMDD 절대 날짜 대신 `snap_date` 기준 상대 일수 오프셋을 사용한다 (augment 스크립트에서 생성). 이로써 시계열 모델이 절대 날짜가 아닌 시간 간격 패턴을 학습한다.

### 상품 보유 시퀀스 (Santander 17개월)

```yaml
product_sequences:
  max_len: 16           # months 1-16 (month 17 = label)
  truncate_last: 1      # LEAKAGE FIX: drop month 17
  columns:
    seq_saving: {feat_dim: 1, dtype: int}
    seq_checking: {feat_dim: 1, dtype: int}
    # ... 24개 상품 + num_products + acquisitions + churns = 27 seq cols
```

출력: `sequences.npy` (3D: batch x seq_len x feat_dim), `seq_lengths.npy` (1D: 실제 시퀀스 길이)

---

## 저장소 구조 (S3)

```
s3://aiops-ple-financial/
├── data/
│   ├── raw/                          ← Stage 1 입력
│   │   ├── transactions/dt=2024-01-01/
│   │   ├── user_profiles/
│   │   └── product_catalog/
│   ├── santander/                    ← Santander 데이터
│   │   └── santander_final.parquet
│   ├── encrypted/                    ← Stage 3 출력
│   ├── processed/                    ← 전처리 완료
│   └── validated/                    ← 검증 통과
├── features/                         ← Stage 4 출력
│   ├── v1.0/
│   │   ├── features.parquet
│   │   ├── labels.parquet
│   │   ├── sequences.npy
│   │   ├── seq_lengths.npy
│   │   ├── feature_schema.json
│   │   ├── label_schema.json
│   │   ├── split_indices.json
│   │   ├── scaler_params.json
│   │   ├── label_transforms.json
│   │   └── item_universe/
│   └── latest -> v1.0
├── pii-indices/                      ← Stage 3 인덱스
├── models/
│   ├── ple-santander-20260320/
│   └── lgbm-distill-20260320/
├── analysis/                         ← Stage 8.5 출력
│   ├── ig/
│   ├── cca/
│   ├── gate/
│   └── model_card/
├── serving/                          ← Stage 9.5-10 출력
│   ├── cpe/
│   ├── reasons/
│   └── context_store/
├── audit/                            ← 감사 아티팩트
│   ├── schema/
│   ├── encryption/
│   ├── leakage/
│   └── fidelity/
└── experiments/
    └── santander-ablation/
```

---

## 데이터 흐름 다이어그램 (10+ Stage 통합)

```
[외부 소스]     Stage 1          Stage 1.5         Stage 2
CSV/DB/API     ┌──────────┐    ┌──────────────┐   ┌──────────┐
    │          │DataAdapter│    │TemporalPrep  │   │ 5-Axis   │
    │ S3       │ load_raw()│───▶│ seq truncate │──▶│ Feature  │
    ├─────────▶│ Schema    │    │ prod recomp  │   │ Classify │
    │          └──────────┘    └──────────────┘   └────┬─────┘
    │                                                   │
                Stage 3          Stage 4                │
               ┌──────────┐    ┌──────────────┐        │
               │Encryption│    │FeatureGroup  │◀───────┘
               │ Pipeline │───▶│ + PowerLaw   │
               │ SHA256→  │    │ Scaler       │
               │ INT32    │    └──────┬───────┘
               └──────────┘           │
                                      ▼
                Stage 5          Stage 5.5         Stage 6
               ┌──────────┐    ┌──────────────┐   ┌──────────┐
               │LabelDeriv│    │ Leakage      │   │ Sequence │
               │ 14 tasks │───▶│ Validator    │──▶│ Builder  │
               │ config-  │    │ 4-check      │   │ flat→3D  │
               │ driven   │    └──────────────┘   └────┬─────┘
               └──────────┘                             │
                                                        ▼
                Stage 7          Stage 8           Stage 8.5
               ┌──────────┐    ┌──────────────┐   ┌──────────────┐
               │DataLoader │    │PLETrainer    │   │Model Analysis│
               │ temporal  │───▶│ 2-phase      │──▶│ IG,CCA,Gate  │
               │ split     │    │ Evidential   │   │ Multi,Tmpl   │
               │ gap=30d   │    │ SAE          │   │ XAI,Card     │
               └──────────┘    └──────────────┘   └──────┬───────┘
                                                          │
                                                          ▼
                Stage 9          Stage 9.5         Stage 10
               ┌──────────┐    ┌──────────────┐   ┌──────────────┐
               │Student   │    │Context Vector│   │CPE + Agentic │
               │Trainer   │───▶│ Store (RAG)  │──▶│ Orchestrator │
               │ PLE→LGBM │    │              │   │ FD-TVS,DNA   │
               │ fidelity │    └──────────────┘   │ Constraints  │
               └──────────┘                        └──────────────┘
```

---

## 데이터 검증 게이트

```
Raw Data → [Schema Validation] → [TemporalPrep] → [Quality Check] → [PII Scan] → [Encryption]
              ↓ 실패                                   ↓ 실패            ↓ 발견         ↓ 실패
           SNS 알림                                 SNS 알림         자동 처리       롤백 + 알림

Features → [LeakageValidator] → [Temporal Split] → Training
              ↓ 실패
           CRITICAL 중단 + 알림
```

---

## 현재 vs AWS — 핵심 변경점

| 항목 | 현재 (On-Prem) | AWS (설계) | 변경 이유 |
|------|---------------|-----------|----------|
| 스키마 관리 | 4개 YAML 분산 | 단일 schema.yaml + Registry | 일관성, 자동 검증 |
| 데이터 그룹 | G1-G10 하드코딩 | 5-Axis 분류 (State/Snapshot/Timeseries/Hierarchy/Item) | Expert 라우팅 명시적 기반 |
| 쿼리 엔진 | DuckDB + Pandas fallback | **DuckDB-native** (cuDF 선택적 가속, pandas only at generator boundary) | 단일 경로, GPU 가속 옵션 |
| 데이터 분할 | Random split | **Auto-detect** (cross-sectional → random / multi-date → temporal split + gap_days) | 자동 감지 + 누수 방지 |
| 시퀀스 빌드 | Count-based 고정 | **Time-based + sliding window bootstrapping** (stride_days) | 가변 길이, data augmentation |
| Training 로딩 | pd.read_parquet | **PyArrow zero-copy parquet** (pandas 없음 hot path) | 메모리 효율, 속도 |
| Cold Start | 없음 | **is_cold_start flag + sequence-derived feature zeroing** | cold start 고객 대응 |
| 시퀀스 처리 | 전체 시퀀스 사용 | **Truncate last month + prod recompute** | 레이블 누수 제거 |
| 누수 검증 | 없음 | **LeakageValidator 4-check** | 자동 누수 감지 |
| 레이블 생성 | 코드 내 하드코딩 | **LabelDeriver (config-driven, 14 tasks)** | 선언적, 재현 가능 |
| 암호화 | encryption_config.yaml 별도 | `core/security/` 통합 (스키마 pii 자동 연동) | Stage 3 자동 처리 |
| 저장소 | 로컬 Parquet + GCS | S3 (버전관리 + 파티셔닝) | 내구성, IAM, 비용 |
| GPU 가속 | 없음 | cuDF/cuPY optional (Stage 3/4) | 대규모 데이터 처리 가속 |
