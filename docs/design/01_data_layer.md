# 01. Data Layer — 스키마 관리, 데이터 흐름, 저장소

## 현재 (On-Prem) 분석

### 문제점
1. **스키마 관리 분산**: raw_schema_spec.yaml, clean_schema_spec.yaml, clean_table_mapping.yaml, feature_schema.yaml, 그리고 코드 내 하드코딩이 혼재
2. **그룹 구조 고정**: G1-G10 체계가 금융 도메인에 종속 (예금, 카드, 앱로그 등)
3. **쿼리 엔진 단일**: DuckDB 전용 — 소규모에 최적이지만 확장성 제한
4. **경로 하드코딩**: `D:/storage`, `C:/storage` 등 절대 경로 사용
5. **소스 우선순위 복잡**: `master → encrypted → raw` 3단계 폴백 로직이 yaml에 혼재

### 유지할 패턴
- **DuckDB 기반 처리**: 메모리 효율적, SQL 표현력 높음 → 그대로 유지
- **YAML 기반 스키마 정의**: 선언적 방식 → 확장하여 단일 진실의 원천으로
- **Parquet 포맷**: 컬럼형, 압축, 파티셔닝 → S3와 최적 조합

---

## AWS 설계

### 스키마 레지스트리

```
configs/schemas/
├── schema.yaml          ← 단일 진실의 원천 (Single Source of Truth)
├── schema_registry.py   ← 스키마 로드/검증/진화 관리
└── examples/
    ├── financial.yaml   ← 금융 도메인 예시
    └── ecommerce.yaml   ← 이커머스 도메인 예시
```

#### 스키마 정의 포맷

```yaml
# configs/schemas/schema.yaml
schema:
  version: "1.0"

  # 데이터 소스 정의 (그룹 구조 대신 범용 소스 개념)
  sources:
    - name: transactions
      location: s3://bucket/data/raw/transactions/
      format: parquet
      partition_keys: [dt, region]
      columns:
        - {name: user_id, type: string, nullable: false, pii: false}
        - {name: amount, type: float64, nullable: false, pii: false}
        - {name: category, type: string, nullable: true, pii: false}
        - {name: timestamp, type: timestamp, nullable: false, pii: false}
        - {name: phone_number, type: string, nullable: true, pii: true}  # PII 마킹

    - name: user_profiles
      location: s3://bucket/data/raw/users/
      format: parquet
      columns:
        - {name: user_id, type: string, nullable: false, pii: false}
        - {name: age, type: int32, nullable: true, pii: false}
        - {name: segment, type: string, nullable: true, pii: false}

  # 피처 정의 (어떤 소스에서 어떤 피처를 추출하는지)
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
    """

    def load(self, path: str) -> Schema
    def validate_data(self, df, schema) -> ValidationResult
    def check_compatibility(self, old_schema, new_schema) -> CompatibilityReport
    def evolve(self, schema, changes) -> Schema  # 스키마 진화 (컬럼 추가 등)
```

### 저장소 구조 (S3)

```
s3://aiops-ple-financial/
├── data/
│   ├── raw/                          ← 원본 데이터 (파티셔닝)
│   │   ├── transactions/dt=2024-01-01/
│   │   ├── user_profiles/
│   │   └── events/
│   ├── processed/                    ← 전처리 완료 (PII 제거, 타입 정규화)
│   │   └── transactions/dt=2024-01-01/
│   └── validated/                    ← 검증 통과 데이터만
│       └── transactions/dt=2024-01-01/
├── features/                         ← 학습용 피처 테이블
│   ├── v1.0/                         ← 피처 버전별 관리
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── latest -> v1.0               ← 심볼릭 링크 (최신)
├── models/                           ← 학습된 모델
│   ├── ple-multitask-20240101/
│   └── lgbm-distill-20240101/
├── experiments/                      ← 실험 메타데이터
│   └── experiment-001/
├── schemas/                          ← 스키마 이력
│   ├── v1.0.yaml
│   └── v1.1.yaml
└── audit/                            ← 감사 로그
    └── 2024/01/
```

#### S3 파티셔닝 전략

| 데이터 유형 | 파티션 키 | 이유 |
|------------|----------|------|
| raw 트랜잭션 | `dt` (날짜) | 날짜별 증분 수집 |
| 피처 테이블 | `version` | 피처 스키마 변경 시 공존 |
| 모델 아티팩트 | `job_name` | 실험별 격리 |

### 쿼리 엔진 — 이중 백엔드

```python
# core/data/query_engine.py
class QueryEngine:
    """
    DuckDB (기본) / Athena (대규모) 자동 선택.

    사용자는 SQL만 작성하면 됩니다.
    데이터 크기에 따라 엔진이 자동으로 결정됩니다.
    """

    def __init__(self, config):
        self.backend = config.get("query_engine", "duckdb")
        self.auto_switch_threshold_gb = config.get("auto_switch_gb", 50)

    def query(self, sql: str, source: str = None) -> pd.DataFrame:
        if self.backend == "auto":
            size = self._estimate_scan_size(source)
            backend = "athena" if size > self.auto_switch_threshold_gb else "duckdb"
        else:
            backend = self.backend

        if backend == "duckdb":
            return self._duckdb_query(sql)  # S3 httpfs 직접 쿼리
        else:
            return self._athena_query(sql)  # Athena 서버리스 쿼리
```

### 데이터 검증 게이트

```
Raw Data → [Schema Validation] → [Quality Check] → [PII Scan] → Validated Data
              ↓ 실패                ↓ 실패            ↓ 발견
           SNS 알림            SNS 알림         자동 마스킹 또는 차단
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

    def validate(self, df, schema) -> ValidationResult:
        results = []
        results += self._check_schema(df, schema)
        results += self._check_nulls(df, schema)
        results += self._check_ranges(df, schema)
        results += self._check_drift(df, schema)
        results += self._check_pii(df, schema)
        return ValidationResult(results)
```

---

## 데이터 흐름 다이어그램 (상세)

```
[외부 소스]                    [S3 Data Lake]               [피처 스토어]

CSV/DB/API                     ┌─────────────┐              ┌─────────────┐
    │                          │ data/raw/   │              │ features/   │
    │ ① 업로드                  │  └── tx/    │              │  └── v1.0/  │
    ├─────────────────────────▶│  └── users/ │              │     ├─ train│
    │                          └──────┬──────┘              │     ├─ val  │
    │                                 │                     │     └─ test │
    │                          ② 스키마 검증                 └──────▲──────┘
    │                                 │                            │
    │                          ┌──────▼──────┐              ⑤ 피처 저장
    │                          │ data/       │                     │
    │                          │ validated/  │              ┌──────┴──────┐
    │                          └──────┬──────┘              │ SageMaker   │
    │                                 │                     │ Processing  │
    │                          ③ 전처리 (PII 제거,           │ (DuckDB)    │
    │                             타입 정규화)               └─────────────┘
    │                                 │                            ▲
    │                          ┌──────▼──────┐                     │
    │                          │ data/       │──── ④ 피처 추출 ────┘
    │                          │ processed/  │
    │                          └─────────────┘
```

### 단계별 상세

| 단계 | 처리 | 도구 | 실패 시 |
|------|------|------|---------|
| ① 업로드 | S3 sync, API → Lambda → S3 | aws s3 sync / Lambda | 재시도 3회 |
| ② 스키마 검증 | 컬럼, 타입, PII 마킹 확인 | SchemaRegistry | SNS 알림 → 차단 |
| ③ 전처리 | Null 처리, 타입 변환, PII 마스킹 | SageMaker Processing (DuckDB) | 롤백 |
| ④ 피처 추출 | FeaturePipeline 실행 | SageMaker Processing (DuckDB) | 이전 버전 유지 |
| ⑤ 피처 저장 | 버전별 Parquet 저장 | S3 + 메타데이터 | 이전 버전 유지 |

---

## 현재 vs AWS — 핵심 변경점

| 항목 | 현재 (On-Prem) | AWS (설계) | 변경 이유 |
|------|---------------|-----------|----------|
| 스키마 관리 | 4개 YAML 분산 | 단일 schema.yaml + Registry | 일관성, 자동 검증 |
| 데이터 그룹 | G1-G10 하드코딩 | sources: 배열 (동적) | 도메인 무관 |
| 저장소 | 로컬 Parquet + GCS | S3 (버전관리 + 파티셔닝) | 내구성, IAM, 비용 |
| 쿼리 | DuckDB only | DuckDB + Athena (auto) | 확장성 |
| PII 처리 | encryption_config.yaml 별도 | 스키마 내 pii: true 선언 | 단일 진실의 원천 |
| 데이터 검증 | GX 수동 설정 | 스키마 기반 자동 생성 | 설정 줄이고 일관성 |
| 피처 버전 | 없음 (덮어쓰기) | features/v1.0, v1.1 | 재현성, 롤백 |
