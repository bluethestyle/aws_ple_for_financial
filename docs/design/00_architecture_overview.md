# 00. Architecture Overview — AWS PLE Platform

## 설계 철학

### 핵심 원칙
1. **Config-Driven**: YAML 하나로 데이터/태스크/모델/인프라를 정의 — 코드 변경 없이 새 문제 적용
2. **Registry Pattern**: Expert, Task, Feature, Model 모두 플러그인 방식 등록 — 확장 시 기존 코드 수정 불필요
3. **Pay-as-you-go**: 상시 서버 없음 — 학습/추론 시에만 리소스 할당, 완료 후 자동 해제
4. **Schema-First**: 데이터 스키마가 파이프라인 전체를 결정 — 스키마 변경 시 하위 파이프라인 자동 조정
5. **Audit by Design**: 모든 단계에서 데이터 리니지/실험 이력/결정 근거 기록
6. **Scale-Aware**: 트래픽/데이터 규모에 따라 config 한 줄로 인프라 전환 (Lambda↔ECS, Memory↔DynamoDB)
7. **5-Axis Feature Classification**: 모든 피처를 State/Snapshot/Timeseries/Hierarchy/Item 5축으로 분류 — Expert 라우팅의 기반
8. **Pool/Basket/Runtime 3계층**: 코드(Pool) → Config(Basket) → 학습(Runtime) 분리 — 도메인 전환 시 코드 수정 0

---

## 9-Stage End-to-End 파이프라인

```
Stage 1          Stage 2              Stage 3              Stage 4
Raw Data Load    Feature              Preprocessing        Encryption +
+ Schema Valid.  Classification       (type, null,         Integer Indexing
                 (5-axis)             outlier)             (PII→SHA256→INT32)
┌──────────┐    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ S3 Raw   │───▶│ 5-Axis       │────▶│ DuckDB-only  │────▶│ core/        │
│ Parquet  │    │ Classifier   │     │ Processing   │     │ security/    │
│ + Schema │    │              │     │ cuDF/cuPY    │     │ pipeline.py  │
│ Registry │    │ state        │     │ accelerated  │     │              │
└──────────┘    │ snapshot     │     └──────────────┘     └──────┬───────┘
     │          │ timeseries   │                                  │
  검증 실패      │ hierarchy    │                                  │
  → SNS 알림    │ item         │                                  │
                └──────────────┘                                  │
                                                                  ▼
Stage 5              Stage 6              Stage 7              Stage 8-9
Feature Eng.         Feature Integration  Label Generation     Training +
(per axis)           + Normalization      + Transforms         Distillation
┌──────────────┐    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ State→RFM,   │───▶│ Power-law    │────▶│ clip(99.5p)  │────▶│ Product      │
│   Demographics│    │ auto-detect  │     │ + log1p      │     │ Hierarchy    │
│ Snapshot→TDA │    │ (skew>2→     │     │ (regression) │     │ Config       │
│   global,HMM │    │  log1p+raw)  │     │              │     │              │
│ Timeseries→  │    │ StandardScaler│    │ label_       │     │ Item Universe│
│   TDA local, │    │              │     │ transforms   │     │ (customer×   │
│   Mamba,Patch│    └──────────────┘     │ .json        │     │  product     │
│ Hierarchy→   │                          └──────────────┘     │  bipartite)  │
│   Poincaré,  │                                               └──────┬───────┘
│   MCC L1/L2  │                                                      │
│ Item→Graph,  │                                                      │
│   LightGCN   │                                                      │
└──────────────┘                                                      ▼
                                                               Stage 9
                                                               Training
                                                               ┌──────────────┐
                                                               │ PLE + adaTT  │
                                                               │ Uncertainty  │
                                                               │ Weighting    │
                                                               │ Per-task Loss│
                                                               │ Dispatch     │
                                                               │ PLETrainer   │
                                                               └──────────────┘
```

### Stage별 상세

| Stage | 이름 | 설명 | 구현 위치 | GPU 가속 |
|-------|------|------|-----------|----------|
| **1** | Raw Data Load + Schema Validation | S3 Parquet 로드, SchemaRegistry 기반 컬럼/타입/PII 검증 | `core/data/schema_registry.py` | - |
| **2** | Feature Classification (5-axis) | 모든 컬럼을 State/Snapshot/Timeseries/Hierarchy/Item 축으로 분류 | `core/feature/classifier.py` | - |
| **3** | Preprocessing | DuckDB-only 타입 변환, null 처리, outlier clip. **Pandas fallback 없음** | `core/data/preprocessor.py` | cuDF optional |
| **4** | Encryption + Integer Indexing | PII 컬럼 → SHA256 domain-specific salt → INT32 global index | `core/security/pipeline.py` | - |
| **5** | Feature Engineering (per axis) | 5축 각각에 대응하는 Generator 실행 (TDA, HMM, Mamba, Graph 등) | `core/feature/generators/` | cuPY for TDA |
| **6** | Feature Integration + Normalization | Power-law 자동 감지 (skew>2.0) → log1p+raw 병렬 → StandardScaler | `core/feature/normalizer.py` | cuPY for scaler |
| **7** | Label Generation + Transforms | 회귀 레이블 clip(99.5p) + log1p, label_transforms.json 저장 | Adapter | - |
| **8** | Item Universe + Product Hierarchy | 고객×상품 bipartite graph 구성, MCC/상품 계층 config 로드 | `core/feature/item_universe.py` | - |
| **9** | Training | PLE + adaTT, uncertainty weighting, per-task loss dispatch, PLETrainer 단일 경로 | `core/training/trainer.py` | GPU required |

---

## On-Prem (현재) vs AWS (목표) 아키텍처 대비

```
[On-Prem]                              [AWS]
──────────────────────                 ──────────────────────
로컬 파일 / GCS                        S3 Data Lake
    ↓                                      ↓
DuckDB (인프로세스)                     DuckDB (전 구간 사용, cuDF 가속)
    ↓                                      ↓
Airflow 86 DAGs (Docker)               Step Functions 5개
    ↓                                      ↓
로컬 GPU 학습                          SageMaker Training (Spot)
    ↓                                      ↓
MLflow (Docker)                        SageMaker Experiments
    ↓                                      ↓
FastAPI + Docker Compose               Lambda (기본) / ECS (대규모)
    ↓                                      ↓
Great Expectations (로컬)              CloudWatch + SageMaker Monitor
```

## 서비스별 매핑 상세

| 관점 | On-Prem (현재) | AWS (목표) | 전환 근거 |
|------|---------------|-----------|----------|
| **저장소** | 로컬 Parquet + GCS | S3 (버전 관리 활성화) | 내구성, 비용, IAM 통합 |
| **쿼리 엔진** | DuckDB 전용 | DuckDB 전용 (단일 머신 최강) | 수백 GB까지 DuckDB로 충분. TB급 필요 시 Athena 옵션 |
| **오케스트레이션** | Airflow 86 DAGs (상시 가동) | Step Functions 5개 (실행당 과금) | $300/월 → $0/월 |
| **학습** | 로컬 GPU (전용 머신) | SageMaker Training Job (Spot) | 70% 비용 절감, 자동 체크포인트 |
| **실험 관리** | MLflow (Docker) | SageMaker Experiments | 서버 유지 비용 제거 |
| **서빙** | FastAPI + Docker Compose (상시) | Lambda (기본) / ECS Fargate (대규모) | 규모별 자동 전환 |
| **피처 스토어** | 로컬 파일 | Lambda 메모리 (기본) / DynamoDB (대규모) | 규모별 자동 전환 |
| **데이터 검증** | Great Expectations (로컬) | SageMaker Processing + GX | 동일 로직, 실행 환경만 변경 |
| **감사/리니지** | 커스텀 audit_logger + DVC | CloudTrail + S3 버전관리 + SageMaker Lineage | AWS 네이티브 통합 |
| **모니터링** | 커스텀 drift_monitor | SageMaker Model Monitor + CloudWatch | 관리형 서비스, 알림 자동화 |
| **암호화** | encryption_config.yaml 별도 | `core/security/` 통합 파이프라인 (SHA256 + INT32) | Stage 4에서 자동 처리 |

---

## 5-Axis Feature Classification 체계

파이프라인의 핵심 설계 결정 — 모든 피처를 5개 축으로 분류하고, 각 축에 대응하는 Feature Generator와 Expert를 매핑한다.

```
            ┌────────────────────────────────────────────────────────┐
            │              5-Axis Feature Classification             │
            ├──────────┬──────────┬──────────┬──────────┬───────────┤
            │  State   │ Snapshot │Timeseries│Hierarchy │   Item    │
            │ (정적)    │ (장기)   │ (단기)    │ (구조)   │ (관계)    │
            ├──────────┼──────────┼──────────┼──────────┼───────────┤
 Features   │ 인구통계 │ TDA글로벌│ TDA로컬  │ Poincaré │ 고객×상품 │
            │ RFM      │ HMM 전이 │ Mamba SSM│ MCC L1/L2│ bipartite │
            │          │ 상품트렌드│ PatchTST │ 상품계층  │ LightGCN  │
            ├──────────┼──────────┼──────────┼──────────┼───────────┤
 Experts    │ DeepFM   │ PersLay  │ Temporal │ HGCN     │ LightGCN  │
            │ MLP      │ (global) │ Ensemble │          │           │
            │ AutoInt  │ Causal   │ Mamba    │          │           │
            ├──────────┼──────────┼──────────┼──────────┼───────────┤
 GPU 가속   │ -        │ cuPY     │ GPU      │ GPU      │ GPU       │
            │          │ (TDA)    │ (Mamba)  │ (HGCN)   │ (GCN)     │
            └──────────┴──────────┴──────────┴──────────┴───────────┘
```

### 축별 특성

| 축 | 시간 의존성 | 변화 속도 | 대표 데이터 | 처리 방식 |
|----|-----------|-----------|------------|----------|
| **State** | 없음 (정적) | 거의 변하지 않음 | 나이, 성별, 가입일 | 피처 상호작용 (FM) |
| **Snapshot** | 장기 (월/분기) | 느림 | 12개월 거래 위상, HMM 상태 | 장기 패턴 추출 |
| **Timeseries** | 단기 (일/주) | 빠름 | 최근 90일 거래 시퀀스 | 시퀀스 모델링 (SSM) |
| **Hierarchy** | 없음 (구조적) | 느림 | MCC 코드 계층, 상품 카테고리 | 쌍곡 임베딩 |
| **Item** | 없음 (관계적) | 중간 | 고객-상품 상호작용 | 그래프 협업 필터링 |

---

## 암호화 파이프라인 통합 (Stage 4)

`core/security/` 모듈이 Stage 4를 담당한다. 스키마의 `pii: true` 마킹에서 자동으로 암호화 정책을 유도한다.

```
Schema (pii: true)
    ↓ derive_from_schema()
EncryptionPolicy (per source, per column)
    ↓
EncryptionPipeline.process_source()
    ├── Step 1: Drop (phone, email, SSN 등 contact/personal_id)
    ├── Step 2: SHA256 Hash (domain-specific salt)
    │           PIIDomain: CUSTOMER, ACCOUNT, CARD, MERCHANT, ...
    │           SaltManager: AWS Secrets Manager or local
    ├── Step 3: Integer Index (hash BLOB → INT32 global index)
    │           PIIIntegerIndexer: append-only, Parquet 저장
    └── Step 4: Audit report
```

| 컴포넌트 | 파일 | 역할 |
|---------|------|------|
| `PIIDomain` | `core/security/domains.py` | 16개 PII 도메인 정의 + 컬럼 자동 매핑 |
| `SaltManager` | `core/security/salt_manager.py` | 도메인별 salt 관리 (Secrets Manager / 로컬) |
| `PIIEncryptor` | `core/security/encryptor.py` | SHA256(salt + value) 해싱 |
| `PIIIntegerIndexer` | `core/security/integer_indexer.py` | Hash → INT32 매핑 (S3 Parquet 영속) |
| `EncryptionPipeline` | `core/security/pipeline.py` | 전체 오케스트레이션 |

---

## cuDF/cuPY GPU 가속 포인트

| Stage | 대상 | CPU 경로 | GPU 경로 | 가속 효과 |
|-------|------|---------|---------|----------|
| 3 | Preprocessing (대규모 DataFrame) | DuckDB | cuDF (DuckDB fallback) | 10-50x on sort/groupby |
| 5 | TDA persistence diagram 계산 | ripser (NumPy) | cuPY + ripser | 5-10x |
| 6 | StandardScaler (mean/std) | NumPy | cuPY | 3-5x on 100M+ rows |
| 9 | Model training | PyTorch CPU | PyTorch CUDA | Required |

GPU 가속은 선택적이며, cuDF/cuPY 미설치 시 CPU 경로로 자동 폴백한다.

---

## 모듈형 설계 — 유연성 포인트

### 데이터 유연성 (어떤 데이터든)
```yaml
# 금융 데이터 — Santander 트랜잭션
data:
  source: s3://bucket/financial/
  schema: configs/schemas/financial.yaml

# 이커머스 데이터
data:
  source: s3://bucket/ecommerce/
  schema: configs/schemas/ecommerce.yaml

# 어떤 도메인이든 스키마만 정의하면 동일 파이프라인 사용
```

### 태스크 유연성 (어떤 조합이든)
```yaml
tasks:
  - {name: click, type: binary, loss: focal}           # CTR — build_loss("focal")
  - {name: purchase, type: binary, loss: focal}         # CVR — build_loss("focal")
  - {name: revenue, type: regression, loss: huber}      # LTV — build_loss("huber")
  - {name: category, type: multiclass, num_classes: 50} # 카테고리 — build_loss("ce") + auto class_weights
  # 태스크 추가/제거는 이 목록만 수정
  # Per-task loss는 build_loss() 팩토리로 디스패치 (focal/huber/mse/ce/infonce)
  # Uncertainty weighting (Kendall et al.)으로 태스크 간 자동 밸런싱
```

### Expert 유연성 (5축 매핑)
```yaml
experts:
  shared:
    - {type: deepfm, enabled: true, axis: state}       # State 축 전담
    - {type: temporal_ensemble, enabled: true, axis: timeseries}  # Timeseries 축
    - {type: hgcn, enabled: true, axis: hierarchy}      # Hierarchy 축
    - {type: lightgcn, enabled: true, axis: item}       # Item 축
    - {type: perslay, enabled: true, axis: snapshot}     # Snapshot 축
  # 커스텀 Expert는 @ExpertRegistry.register("my_expert")로 추가
```

### 서빙/인프라 유연성 (규모에 따라 자동 전환)
```yaml
serving:
  mode: auto              # auto | lambda | ecs
  feature_store: auto     # auto | memory | dynamodb
  # 월 1억 건 이하: Lambda + 메모리 → $0-1/월
  # 월 1억 건 이상: ECS + DynamoDB → $360/월
  # config 한 줄로 전환, 추론 코드 동일
```

---

## 핵심 설계 결정 요약

| 결정 | 선택 | 근거 |
|------|------|------|
| 쿼리 엔진 | DuckDB 단일 (Athena는 옵션) | 단일 머신 최강, 수백 GB까지 충분 |
| Pandas fallback | **제거** (DuckDB-only) | 이중 경로 유지보수 비용 > 이점 |
| 피처 분류 | 5-Axis (State/Snapshot/Timeseries/Hierarchy/Item) | Expert 라우팅의 명시적 기반 |
| 암호화 | SHA256 + INT32 (core/security/) | PII 보호 + 모델 입력 호환 |
| 정규화 | Power-law 자동감지 → log1p + raw 병렬 → StandardScaler | 단일 경로, 분포 보존 |
| 오케스트레이션 | Step Functions | Airflow 대비 비용 0, 복잡도 감소 |
| 학습 | SageMaker Spot | 70% 절감, 체크포인트 자동 재개 |
| 실시간 추론 | LGBM (매 요청 추론) | ~5ms, 지식 증류로 PLE 품질 유지 |
| Loss 전략 | Per-task dispatch (build_loss) + Uncertainty weighting | config 선언적, 자동 밸런싱 |
| 서빙 기본 | Lambda + 메모리 피처 로드 | 서버리스, 유휴 $0, ~5ms |
| 서빙 확장 | ECS + DynamoDB | 월 1억 건 이상 시 비용 역전 |
| GPU 가속 | cuDF/cuPY optional | Stage 3/5/6에서 선택적 가속 |

---

## PipelineRunner 통합 아키텍처 (Step 12-14)

### 8-Stage PipelineRunner

`core/pipeline/runner.py`의 `PipelineRunner`가 전체 파이프라인을 단일 진입점으로 통합한다.

```
PipelineRunner.run()
    │
    ├── Stage 1: DataAdapter.load_raw()
    │   └── AdapterRegistry.build("ealtman2019", config)
    │       └── DuckDB 24M txn → 2,000 user-level aggregation
    │
    ├── Stage 2: SchemaClassifier (5-axis)
    │
    ├── Stage 3: Preprocessing (DuckDB-only, cuDF optional)
    │
    ├── Stage 4: EncryptionPipeline (SHA256 → INT32)
    │
    ├── Stage 5: FeatureGroupPipeline
    │   └── TDA, HMM, Mamba, Graph generators (adapter가 아닌 여기서 실행)
    │
    ├── Stage 6: Normalization (PowerLawAwareScaler)
    │
    ├── Stage 7: Label derivation + transforms
    │
    ├── Stage 8: Training (PLETrainer)
    │   └── containers/training/train.py::main_pipeline()
    │
    └── Stage 9: Knowledge Distillation (StudentTrainer)
        └── PLE teacher → LGBM students (soft label + fidelity validation)
```

### DataAdapter 패턴

```python
# core/pipeline/adapter.py
class DataAdapter(ABC):
    """데이터셋별 원시 데이터 로딩 계약.

    각 데이터셋은 load_raw()를 구현하여 entity-level DataFrame을 반환.
    피처 엔지니어링은 수행하지 않음 — FeatureGroupPipeline 담당.
    """
    def load_raw(self) -> Dict[str, pd.DataFrame]: ...

class AdapterRegistry:
    """이름 기반 어댑터 등록/조회."""
    @classmethod
    def register(cls, name: str): ...
    @classmethod
    def build(cls, name: str, config: dict) -> DataAdapter: ...
```

**현재 등록된 어댑터:**

| 이름 | 파일 | 데이터 | 집계 방식 |
|------|------|--------|----------|
| `ealtman2019` | `adapters/ealtman2019_adapter.py` | 24M 신용카드 거래 | DuckDB → 2,000 user |

### 백엔드 선택 체인 (cuDF → DuckDB → Pandas)

`DataAdapter._select_backend()`가 config의 `data.backend` 리스트를 순회하며 사용 가능한 최적 백엔드를 선택한다:

```
cuDF (GPU DataFrame)  →  DuckDB (in-process SQL)  →  Pandas (fallback)
       │ ImportError           │ ImportError                │
       └──────────────────────▶└──────────────────────────▶ OK
```

### Training 이중 진입점

`containers/training/train.py`는 두 가지 경로를 지원한다:

| 경로 | 진입점 | 용도 |
|------|--------|------|
| Legacy | `main()` | SageMaker Training Job 직접 호출 (기존 호환) |
| Pipeline | `main_pipeline(config)` | `--pipeline config.yaml` 플래그로 PipelineRunner 경유 |

---

## 설계서 구성

| 문서 | 내용 |
|------|------|
| [01_data_layer](01_data_layer.md) | DuckDB-only, 암호화 파이프라인, 5-axis 분류, cuDF 가속 |
| [02_feature_engineering](02_feature_engineering.md) | 5축별 피처 매핑, TDA/HMM/Mamba/Graph/LightGCN, 정규화 |
| [03_model_architecture](03_model_architecture.md) | Per-task loss dispatch, uncertainty weighting, Expert→5축 라우팅 |
| [04_training_pipeline](04_training_pipeline.md) | PLETrainer 단일 경로, 4-dimension ablation 설계 |
| [05_serving_and_testing](05_serving_and_testing.md) | Lambda↔ECS 자동 전환, 실시간 추론, A/B 테스트 |
| [06_orchestration_and_audit](06_orchestration_and_audit.md) | Step Functions 5개, 3계층 감사, E2E 리니지 |
| [07_cost_analysis](07_cost_analysis.md) | 규모별 비용 시뮬레이션, 손익분기점 분석 |
| [08_recommendation_intelligence](08_recommendation_intelligence.md) | 스코어링, 추천 사유 3계층, 역매핑, 규제 준수 |
| [09_compliance_governance](09_compliance_governance.md) | 감사 불변성, 36항목 레지스트리, 공정성, 쏠림, 인시던트, 킬스위치 |
| [10_pool_basket_architecture](10_pool_basket_architecture.md) | Pool/Basket/Runtime 3계층, Expert 11종, Feature Generator, Task Group |
