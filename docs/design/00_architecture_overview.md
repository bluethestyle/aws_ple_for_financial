# 00. Architecture Overview — AWS PLE Platform

## 설계 철학

### 핵심 원칙
1. **Config-Driven**: YAML 하나로 데이터/태스크/모델/인프라를 정의 — 코드 변경 없이 새 문제 적용
2. **Registry Pattern**: Expert, Task, Feature, Model 모두 플러그인 방식 등록 — 확장 시 기존 코드 수정 불필요
3. **Pay-as-you-go**: 상시 서버 없음 — 학습/추론 시에만 리소스 할당, 완료 후 자동 해제
4. **Schema-First**: 데이터 스키마가 파이프라인 전체를 결정 — 스키마 변경 시 하위 파이프라인 자동 조정
5. **Audit by Design**: 모든 단계에서 데이터 리니지/실험 이력/결정 근거 기록
6. **Scale-Aware**: 트래픽/데이터 규모에 따라 config 한 줄로 인프라 전환 (Lambda↔ECS, Memory↔DynamoDB)

---

## On-Prem (현재) vs AWS (목표) 아키텍처 대비

```
[On-Prem]                              [AWS]
──────────────────────                 ──────────────────────
로컬 파일 / GCS                        S3 Data Lake
    ↓                                      ↓
DuckDB (인프로세스)                     DuckDB (전 구간 사용)
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

---

## End-to-End 데이터 흐름

```
① 데이터 등록                    ② 스키마 검증                ③ 피처 엔지니어링
┌─────────────┐               ┌──────────────┐            ┌───────────────┐
│ S3 Raw Data │──────────────▶│ Schema       │───────────▶│ SageMaker     │
│ (Parquet)   │   스키마 선언  │ Registry     │  검증 통과  │ Processing    │
│             │   (YAML)      │              │            │ (DuckDB 내장)  │
└─────────────┘               └──────────────┘            └───────┬───────┘
                                     │                            │
                                실패 시 알림                       ▼
                                (SNS/Slack)              ④ 피처 저장 (S3)
                                                                  │
                                                                  ▼
⑧ 온라인 서빙                  ⑦ 모델 등록                ⑤ 모델 학습
┌──────────────┐              ┌──────────────┐            ┌───────────────┐
│ Lambda/ECS   │◀─────────────│ Model        │◀───────────│ SageMaker     │
│ (LGBM 실시간 │   배포 승인   │ Registry     │  최적 모델  │ Training      │
│  추론)       │              │ (S3 + meta)  │            │ (Spot GPU)    │
│ + A/B Test   │              └──────────────┘            └───────┬───────┘
└──────┬───────┘                     │                            │
       │                             ▼                     ⑥ 실험 기록
       │                      감사 로그                   ┌───────────────┐
       ▼                      (CloudTrail +               │ SageMaker     │
⑨ 모니터링                     S3 Versioning)             │ Experiments   │
┌─────────────┐                                           └───────────────┘
│ CloudWatch  │
│ + 비용 알림  │
└─────────────┘
```

---

## 모듈형 설계 — 유연성 포인트

### 데이터 유연성 (어떤 데이터든)
```yaml
# 금융 데이터
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
  - {name: click, type: binary, loss: focal}           # CTR
  - {name: purchase, type: binary, loss: focal}         # CVR
  - {name: revenue, type: regression, loss: huber}      # LTV
  - {name: category, type: multiclass, num_classes: 50} # 카테고리 예측
  # 태스크 추가/제거는 이 목록만 수정
```

### Expert 유연성 (어떤 구조든)
```yaml
experts:
  shared:
    - {type: deepfm, enabled: true, output_dim: 64}
    - {type: temporal, enabled: true, output_dim: 64}
    - {type: hgcn, enabled: false}  # 그래프 데이터 없으면 비활성화
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
| 오케스트레이션 | Step Functions | Airflow 대비 비용 0, 복잡도 감소 |
| 학습 | SageMaker Spot | 70% 절감, 체크포인트 자동 재개 |
| 실시간 추론 | LGBM (매 요청 추론) | ~5ms, 지식 증류로 PLE 품질 유지 |
| 서빙 기본 | Lambda + 메모리 피처 로드 | 서버리스, 유휴 $0, ~5ms |
| 서빙 확장 | ECS + DynamoDB | 월 1억 건 이상 시 비용 역전 |
| 피처 스토어 | Lambda 메모리 → DynamoDB | 유저 500만 이하는 메모리 충분 |
| A/B 테스트 | API Gateway 가중 라우팅 | Lambda/ECS 모두 호환 |
| 실험 관리 | SageMaker Experiments | MLflow 서버 유지비 0 |
| 감사 | CloudTrail + S3 Versioning + 커스텀 리니지 | 3계층 자동 |

---

## 설계서 구성

| 문서 | 내용 |
|------|------|
| [01_data_layer](01_data_layer.md) | 스키마 관리, 데이터 흐름, 저장소, DuckDB |
| [02_feature_engineering](02_feature_engineering.md) | 피처 파이프라인, 트랜스포머, 동적 차원 계산 |
| [03_model_architecture](03_model_architecture.md) | PLE, Expert/Task Registry, adaTT, 증류 |
| [04_training_pipeline](04_training_pipeline.md) | SageMaker Spot 2-Phase, Champion/Challenger |
| [05_serving_and_testing](05_serving_and_testing.md) | Lambda↔ECS 자동 전환, 실시간 추론, A/B 테스트 |
| [06_orchestration_and_audit](06_orchestration_and_audit.md) | Step Functions 5개, 3계층 감사, E2E 리니지 |
| [07_cost_analysis](07_cost_analysis.md) | 규모별 비용 시뮬레이션, 손익분기점 분석 |
| [08_recommendation_intelligence](08_recommendation_intelligence.md) | 스코어링, 추천 사유 3계층, 역매핑, 규제 준수 |
