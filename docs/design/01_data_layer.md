# 01. Data Layer — DuckDB-only, 암호화 파이프라인, 5-Axis 분류, cuDF 가속

## 9-Stage 파이프라인에서의 위치

Data Layer는 Stage 1~4를 담당한다:

```
Stage 1: Raw Data Load + Schema Validation
Stage 2: Feature Classification (5-axis)
Stage 3: Preprocessing (type conversion, nulls, outliers)
Stage 4: Encryption + Integer Indexing (PII → SHA256 salt → INT32)
```

---

## Stage 1: Raw Data Load + Schema Validation

### 스키마 레지스트리

```
configs/schemas/
├── schema.yaml          ← 단일 진실의 원천 (Single Source of Truth)
├── schema_registry.py   ← 스키마 로드/검증/진화 관리
└── examples/
    ├── financial.yaml   ← 금융 도메인 예시 (Santander)
    └── ecommerce.yaml   ← 이커머스 도메인 예시
```

#### 스키마 정의 포맷

```yaml
# configs/schemas/schema.yaml
schema:
  version: "1.0"

  sources:
    - name: transactions
      location: s3://bucket/data/raw/transactions/
      format: parquet
      partition_keys: [dt, region]
      columns:
        - {name: user_id, type: string, nullable: false, pii: true, pii_domain: customer}
        - {name: amount, type: float64, nullable: false, pii: false}
        - {name: category, type: string, nullable: true, pii: false}
        - {name: timestamp, type: timestamp, nullable: false, pii: false}
        - {name: phone_number, type: string, nullable: true, pii: true, pii_domain: contact}
        - {name: mcc_code, type: string, nullable: true, pii: false}
        - {name: merchant_id, type: string, nullable: true, pii: true, pii_domain: merchant}

    - name: user_profiles
      location: s3://bucket/data/raw/users/
      format: parquet
      columns:
        - {name: user_id, type: string, nullable: false, pii: true, pii_domain: customer}
        - {name: age, type: int32, nullable: true, pii: false}
        - {name: segment, type: string, nullable: true, pii: false}

    - name: product_catalog
      location: s3://bucket/data/raw/products/
      format: parquet
      columns:
        - {name: product_id, type: string, nullable: false, pii: false}
        - {name: category_l1, type: string, nullable: false, pii: false}
        - {name: category_l2, type: string, nullable: true, pii: false}
        - {name: is_active, type: bool, nullable: false, pii: false}

  # 피처 정의
  features:
    numeric:
      - {name: total_spend, source: transactions, agg: sum(amount)}
      - {name: avg_amount, source: transactions, agg: mean(amount)}
      - {name: tx_count, source: transactions, agg: count(*)}
      - {name: age, source: user_profiles, agg: first(age)}
    categorical:
      - {name: top_category, source: transactions, agg: mode(category)}
      - {name: segment, source: user_profiles, agg: first(segment)}

  # 레이블 정의
  labels:
    - {name: clicked, source: events, column: is_click, type: binary}
    - {name: purchased, source: events, column: is_purchase, type: binary}
    - {name: revenue_30d, source: targets, column: revenue, type: continuous}
```

#### 스키마 레지스트리 클래스

```python
# core/data/schema_registry.py
class SchemaRegistry:
    """
    스키마 로드, 검증, 진화를 관리합니다.

    - 스키마 변경 시 하위 파이프라인에 영향 분석
    - 버전 관리: S3에 이력 저장
    - 호환성 검증: 새 스키마가 기존 데이터와 호환되는지 확인
    - PII 컬럼 자동 추출: pii: true 마킹 → EncryptionPipeline에 전달
    """

    def load(self, path: str) -> Schema
    def validate_data(self, df, schema) -> ValidationResult
    def check_compatibility(self, old_schema, new_schema) -> CompatibilityReport
    def evolve(self, schema, changes) -> Schema
    def all_pii_columns(self) -> dict[str, list[str]]  # source → pii cols
```

---

## Stage 2: Feature Classification (5-Axis)

모든 피처를 5개 축으로 분류한다. 이 분류가 Feature Engineering (Stage 5)과 Expert 라우팅 (Stage 9)의 기반이 된다.

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

### 분류 규칙

```python
# core/feature/classifier.py
class FeatureAxisClassifier:
    """
    스키마와 feature_groups.yaml을 기반으로 각 피처 그룹의 축을 결정.

    분류 로직:
    1. feature_groups.yaml에 axis: 명시적 지정 → 그대로 사용
    2. 미지정 시 자동 추론:
       - temporal/sequence 키워드 → Timeseries
       - tda_long, hmm, snapshot 키워드 → Snapshot
       - mcc, hierarchy, poincare 키워드 → Hierarchy
       - graph, bipartite, lightgcn 키워드 → Item
       - 나머지 → State
    """

    AXIS_KEYWORDS = {
        "timeseries": ["temporal", "sequence", "mamba", "tda_short", "patch"],
        "snapshot": ["tda_long", "hmm", "snapshot", "trend", "lifecycle"],
        "hierarchy": ["mcc", "hierarchy", "poincare", "category_tree"],
        "item": ["graph", "bipartite", "lightgcn", "collaborative"],
    }
```

### Santander 데이터 5-Axis 매핑

| Axis | Feature Group | Output Dim | 설명 |
|------|--------------|------------|------|
| **State** | base_rfm | 34D | 인구통계 + RFM 프로필 |
| **State** | base_category | 64D | 업종별 소비 패턴 |
| **State** | multi_source | 91D | 예금/카드/멤버십/투자 |
| **State** | extended_source | 84D | 보험/상담/캠페인 |
| **State** | economics | 17D | MPC, 소득 탄력성 |
| **State** | multidisciplinary | 24D | 화학동역학/전염병/간섭/범죄 |
| **Snapshot** | tda_topology (long) | 36D+10D | 12개월 TDA global + phase transition |
| **Snapshot** | hmm_states | 48D | HMM Triple-Mode 상태 전이 |
| **Snapshot** | gmm_clustering | 22D | 장기 고객 세그먼트 |
| **Snapshot** | model_derived (HMM/LNN) | 23D | HMM summary + LNN 패턴 |
| **Timeseries** | tda_topology (short) | 24D | 90일 TDA local |
| **Timeseries** | base_temporal | 60D | 시계열 집계 + 주기 인코딩 |
| **Timeseries** | mamba_temporal | 50D | Mamba SSM 출력 |
| **Timeseries** | base_txn_stats | 80D | 거래 통계 (최근 기간) |
| **Hierarchy** | merchant_hierarchy | 21D | MCC L1/L2, 브랜드 임베딩 |
| **Hierarchy** | graph_embeddings | 20D | Poincare 쌍곡 임베딩 |
| **Item** | (item_universe) | 가변 | 고객×상품 bipartite graph |
| **Item** | (lightgcn_collab) | 64D | LightGCN 협업 필터링 임베딩 |

---

## Stage 3: Preprocessing — DuckDB-only

DuckDB는 단일 머신에서 **수백 GB까지 처리 가능**하다. 디스크 스필이 자동으로 되므로 메모리보다 큰 데이터도 처리된다.

> **핵심 결정**: Pandas fallback 경로를 완전 제거하고 **DuckDB-only**로 단일화. 모든 데이터 로드/조인/집계가 DuckDB SQL로 처리된다.

### 쿼리 엔진

```python
# core/data/query_engine.py
class QueryEngine:
    """
    DuckDB 기반 쿼리 엔진 (Pandas fallback 없음).
    S3의 Parquet 파일을 httpfs 확장으로 직접 쿼리합니다.

    GPU 가속: cuDF가 설치되어 있으면 대규모 sort/groupby에 cuDF 경로를 사용.
    cuDF 미설치 시 DuckDB CPU 경로로 자동 폴백.
    """

    def __init__(self, config):
        self.backend = config.get("query_engine", "duckdb")
        self._use_cudf = self._check_cudf_available()

    def query(self, sql: str) -> "duckdb.DuckDBPyRelation":
        if self.backend == "duckdb":
            return self._duckdb_query(sql)
        elif self.backend == "athena":
            return self._athena_query(sql)

    def _duckdb_query(self, sql: str):
        import duckdb
        conn = duckdb.connect()
        conn.execute("INSTALL httpfs; LOAD httpfs;")
        conn.execute("SET s3_region='ap-northeast-2';")
        return conn.execute(sql)

    def to_dataframe(self, result) -> "pd.DataFrame | cudf.DataFrame":
        """결과를 DataFrame으로 변환. cuDF 사용 가능 시 GPU DataFrame 반환."""
        if self._use_cudf:
            return cudf.DataFrame.from_arrow(result.arrow())
        return result.fetchdf()
```

| 데이터 규모 | 엔진 | 이유 |
|------------|------|------|
| ~수백 GB | DuckDB | 단일 머신 최강, 비용 0 |
| TB급 (분산 필요) | Athena (옵션) | 분산 스캔 필요 시만 |

### Preprocessing 상세

```python
# core/data/preprocessor.py
class DuckDBPreprocessor:
    """
    DuckDB SQL로 전처리를 수행합니다. Pandas 사용하지 않음.

    처리 항목:
    1. 타입 변환: 스키마 정의에 따라 cast
    2. Null 처리: 컬럼별 전략 (median/mode/0/drop)
    3. Outlier clip: numeric 컬럼 99.5 percentile clip
    4. 중복 제거: id_cols 기준 dedup
    """

    def preprocess(self, conn, table_name: str, schema: Schema):
        # 모든 처리가 DuckDB SQL로 수행됨
        self._cast_types(conn, table_name, schema)
        self._handle_nulls(conn, table_name, schema)
        self._clip_outliers(conn, table_name, schema)
        self._dedup(conn, table_name, schema)
```

### cuDF 가속 포인트

Stage 3에서 cuDF를 선택적으로 사용할 수 있는 지점:

| 연산 | CPU (DuckDB) | GPU (cuDF) | 조건 |
|------|-------------|-----------|------|
| GroupBy aggregation | DuckDB SQL | cudf.groupby() | 100M+ rows |
| Sort | DuckDB SQL | cudf.sort_values() | 50M+ rows |
| Join | DuckDB SQL | cudf.merge() | 양쪽 100M+ |
| Null 처리 | DuckDB SQL | cudf.fillna() | 선택적 |

---

## Stage 4: Encryption + Integer Indexing

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

### 스키마 연동

```python
# 스키마에서 자동으로 암호화 정책 생성
from core.security.encryption_policy import derive_from_schema

policies = derive_from_schema(schema_registry)
# → {source_name: SourceEncryptionPolicy(column→domain→action)}
#   action: "hash_and_index" | "hash_only" | "drop"
#   contact/personal_id → drop (해싱 없이 즉시 삭제)
#   customer/account/card/merchant → hash_and_index

pipeline = EncryptionPipeline(salt_mgr, indexer, policies)
clean_df = pipeline.process_source("transactions", raw_df)
```

---

## 저장소 구조 (S3)

```
s3://aiops-ple-financial/
├── data/
│   ├── raw/                          ← 원본 데이터 (Stage 1 입력)
│   │   ├── transactions/dt=2024-01-01/
│   │   ├── user_profiles/
│   │   ├── product_catalog/          ← 상품 카탈로그 (Item Universe 기반)
│   │   └── events/
│   ├── encrypted/                    ← Stage 4 출력 (PII → INT32)
│   │   └── transactions/dt=2024-01-01/
│   ├── processed/                    ← Stage 3 출력 (전처리 완료)
│   │   └── transactions/dt=2024-01-01/
│   └── validated/                    ← Stage 1 검증 통과
│       └── transactions/dt=2024-01-01/
├── features/                         ← Stage 5-6 출력
│   ├── v1.0/
│   │   ├── train/
│   │   ├── val/
│   │   ├── test/
│   │   ├── scaler_params.json       ← StandardScaler mean/std + log1p 컬럼
│   │   ├── label_transforms.json    ← 회귀 레이블 clip/log1p 파라미터
│   │   └── item_universe/            ← Stage 8 출력
│   │       ├── product_hierarchy.json
│   │       └── customer_product_graph.parquet
│   └── latest -> v1.0
├── pii-indices/                      ← Stage 4 인덱스 테이블
│   ├── customer_index.parquet
│   ├── account_index.parquet
│   ├── card_index.parquet
│   └── merchant_index.parquet
├── models/
│   ├── ple-multitask-20240101/
│   └── lgbm-distill-20240101/
├── experiments/
│   └── experiment-001/
├── schemas/                          ← 스키마 이력
│   ├── v1.0.yaml
│   └── v1.1.yaml
└── audit/
    └── 2024/01/
```

### S3 파티셔닝 전략

| 데이터 유형 | 파티션 키 | 이유 |
|------------|----------|------|
| raw 트랜잭션 | `dt` (날짜) | 날짜별 증분 수집 |
| 피처 테이블 | `version` | 피처 스키마 변경 시 공존 |
| PII 인덱스 | `domain` | 도메인별 독립 관리 |
| 모델 아티팩트 | `job_name` | 실험별 격리 |

---

## 데이터 검증 게이트

```
Raw Data → [Schema Validation] → [Quality Check] → [PII Scan] → [Encryption] → Clean Data
              ↓ 실패                ↓ 실패            ↓ 발견         ↓ 실패
           SNS 알림            SNS 알림         자동 처리       롤백 + 알림
```

```python
# core/data/validation.py
class DataValidator:
    """
    스키마 기반 자동 검증.

    검증 항목:
    1. 컬럼 존재/타입 일치 (스키마 기반)
    2. Null 비율 임계값 (컬럼별 설정 가능)
    3. 값 범위 (numeric: min/max, categorical: allowed values)
    4. 분포 드리프트 (PSI, 이전 배치 대비)
    5. PII 자동 탐지 (정규표현식 + 패턴)
    """
```

---

## 레이블 변환 (Stage 7)

회귀 레이블(LTV, spending_amount 등 금액 계열)은 Adapter 단계에서 다음 변환을 적용한다:

1. **clip(99.5 percentile)**: 극단 이상치 제거
2. **log1p**: 우편향(right-skewed) 분포를 정규분포에 가깝게 변환

변환 파라미터는 `label_transforms.json` 사이드카 파일로 출력 디렉터리에 저장되어 추론 시 역변환(`expm1`)에 사용된다.

```json
{
  "spending_amount": {"clip_value": 15234.50, "transform": "log1p"},
  "ltv_365d":       {"clip_value": 89120.00, "transform": "log1p"}
}
```

---

## 데이터 흐름 다이어그램 (9-Stage 통합)

```
[외부 소스]        Stage 1              Stage 2             Stage 3
CSV/DB/API        ┌──────────┐        ┌──────────┐       ┌──────────┐
    │             │ Schema   │        │ 5-Axis   │       │ DuckDB   │
    │ S3 업로드    │ Registry │        │ Feature  │       │ Preproc  │
    ├────────────▶│ Validate │───────▶│ Classify │──────▶│ (cuDF    │
    │             └──────────┘        └──────────┘       │  option) │
    │                  │ 실패                              └─────┬────┘
    │               SNS 알림                                    │
    │                                                           ▼
                  Stage 4              Stage 5             Stage 6
                 ┌──────────┐        ┌──────────┐       ┌──────────┐
                 │Encryption│        │ Per-Axis  │       │ Power-law│
                 │ Pipeline │        │ Feature   │       │ Auto-det │
                 │ SHA256 → │───────▶│ Generators│──────▶│ + Std    │
                 │ INT32    │        │ TDA/HMM/  │       │ Scaler   │
                 └──────────┘        │ Mamba/Graph│      └─────┬────┘
                                     └──────────┘              │
                                                               ▼
                  Stage 7              Stage 8             Stage 9
                 ┌──────────┐        ┌──────────┐       ┌──────────┐
                 │ Label    │        │ Item     │       │ PLE +    │
                 │ clip +   │        │ Universe │       │ adaTT    │
                 │ log1p    │───────▶│ Product  │──────▶│ Training │
                 │ (regr.)  │        │ Hierarchy│       │ (GPU)    │
                 └──────────┘        └──────────┘       └──────────┘
```

---

## 현재 vs AWS — 핵심 변경점

| 항목 | 현재 (On-Prem) | AWS (설계) | 변경 이유 |
|------|---------------|-----------|----------|
| 스키마 관리 | 4개 YAML 분산 | 단일 schema.yaml + Registry | 일관성, 자동 검증 |
| 데이터 그룹 | G1-G10 하드코딩 | 5-Axis 분류 (State/Snapshot/Timeseries/Hierarchy/Item) | Expert 라우팅 명시적 기반 |
| 쿼리 엔진 | DuckDB + Pandas fallback | **DuckDB-only** (cuDF 선택적 가속) | 단일 경로, GPU 가속 옵션 |
| 암호화 | encryption_config.yaml 별도 | `core/security/` 통합 (스키마 pii 자동 연동) | Stage 4 자동 처리 |
| PII 처리 | 수동 마스킹 | SHA256 + INT32 자동 파이프라인 | 재현성, 감사 추적 |
| 저장소 | 로컬 Parquet + GCS | S3 (버전관리 + 파티셔닝) | 내구성, IAM, 비용 |
| PII 인덱스 | 없음 | domain별 Parquet (append-only) | 안정적 ID 매핑, 역추적 가능 |
| 피처 버전 | 없음 (덮어쓰기) | features/v1.0, v1.1 | 재현성, 롤백 |
| GPU 가속 | 없음 | cuDF/cuPY optional (Stage 3/5/6) | 대규모 데이터 처리 가속 |
