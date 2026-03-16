# 00. Architecture Overview — AWS PLE Platform

## 설계 철학

### 핵심 원칙
1. **Config-Driven**: YAML 하나로 데이터/태스크/모델/인프라를 정의 — 코드 변경 없이 새 문제 적용
2. **Registry Pattern**: Expert, Task, Feature, Model 모두 플러그인 방식 등록 — 확장 시 기존 코드 수정 불필요
3. **Pay-as-you-go**: 상시 서버 없음 — 학습/추론 시에만 리소스 할당, 완료 후 자동 해제
4. **Schema-First**: 데이터 스키마가 파이프라인 전체를 결정 — 스키마 변경 시 하위 파이프라인 자동 조정
5. **Audit by Design**: 모든 단계에서 데이터 리니지/실험 이력/결정 근거 기록

---

## On-Prem (현재) vs AWS (목표) 아키텍처 대비

```
[On-Prem]                              [AWS]
──────────────────────                 ──────────────────────
로컬 파일 / GCS                        S3 Data Lake
    ↓                                      ↓
DuckDB (인프로세스)                     DuckDB (SageMaker 내)
    ↓                                  + Athena (대규모 시)
Airflow 86 DAGs (Docker)                   ↓
    ↓                                  Step Functions
로컬 GPU 학습                              ↓
    ↓                                  SageMaker Training (Spot)
MLflow (Docker)                            ↓
    ↓                                  SageMaker Experiments
FastAPI + Docker Compose                   ↓
    ↓                                  ECS Fargate / Batch Transform
Great Expectations (로컬)                  ↓
    ↓                                  CloudWatch + SageMaker Monitor
```

## 서비스별 매핑 상세

| 관점 | On-Prem (현재) | AWS (목표) | 전환 근거 |
|------|---------------|-----------|----------|
| **저장소** | 로컬 Parquet + GCS | S3 (버전 관리 활성화) | 내구성, 비용, IAM 통합 |
| **쿼리 엔진** | DuckDB 전용 | DuckDB (기본) + Athena (확장) | 소규모는 DuckDB, TB급은 Athena 자동 전환 |
| **오케스트레이션** | Airflow 86 DAGs (상시 가동) | Step Functions (실행당 과금) | $250/월 → $0.025/1000회 |
| **학습** | 로컬 GPU (전용 머신) | SageMaker Training Job (Spot) | 70% 비용 절감, 자동 체크포인트 |
| **실험 관리** | MLflow (Docker) | SageMaker Experiments | 서버 유지 비용 제거 |
| **서빙** | FastAPI + Docker Compose | ECS Fargate (실시간) / Batch Transform (배치) | 자동 스케일링, 0 트래픽 시 0 비용 |
| **데이터 검증** | Great Expectations (로컬) | SageMaker Processing + GX | 동일 로직, 실행 환경만 변경 |
| **감사/린리지** | 커스텀 audit_logger + DVC | CloudTrail + S3 버전관리 + SageMaker Lineage | AWS 네이티브 통합 |
| **모니터링** | 커스텀 drift_monitor | SageMaker Model Monitor + CloudWatch | 관리형 서비스, 알림 자동화 |
| **컨테이너** | Docker Compose (4+ 서비스) | ECR + ECS/SageMaker | 이미지 관리 단순화 |

---

## End-to-End 데이터 흐름

```
① 데이터 등록                    ② 스키마 검증                ③ 피처 엔지니어링
┌─────────────┐               ┌──────────────┐            ┌───────────────┐
│ S3 Raw Data │──────────────▶│ Schema       │───────────▶│ SageMaker     │
│ (Parquet)   │   스키마 선언  │ Registry     │  검증 통과  │ Processing    │
│             │   (YAML)      │ (S3 + DDB)   │            │ (DuckDB 내장)  │
└─────────────┘               └──────────────┘            └───────┬───────┘
                                     │                            │
                                실패 시 알림                       ▼
                                (SNS/Slack)              ④ 피처 저장 (S3)
                                                                  │
                                                                  ▼
⑧ 온라인 서빙                  ⑦ 모델 등록                ⑤ 모델 학습
┌─────────────┐               ┌──────────────┐            ┌───────────────┐
│ ECS Fargate │◀──────────────│ Model        │◀───────────│ SageMaker     │
│ (FastAPI)   │   배포 승인    │ Registry     │  최적 모델  │ Training      │
│ + A/B Test  │               │ (S3 + meta)  │            │ (Spot GPU)    │
└─────────────┘               └──────────────┘            └───────┬───────┘
       │                             │                            │
       ▼                             ▼                     ⑥ 실험 기록
⑨ 모니터링                    감사 로그                   ┌───────────────┐
┌─────────────┐               (CloudTrail +               │ SageMaker     │
│ CloudWatch  │                S3 Versioning)             │ Experiments   │
│ + SageMaker │                                           └───────────────┘
│ Monitor     │
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

---

## 설계서 구성

| 문서 | 내용 |
|------|------|
| [01_data_layer](01_data_layer.md) | 스키마 관리, 데이터 흐름, 저장소, 쿼리 엔진 |
| [02_feature_engineering](02_feature_engineering.md) | 피처 파이프라인, 트랜스포머, 정규화 |
| [03_model_architecture](03_model_architecture.md) | PLE, Expert, Task, adaTT 모듈화 |
| [04_training_pipeline](04_training_pipeline.md) | SageMaker Training, 2-Phase, 실험 관리 |
| [05_serving_and_testing](05_serving_and_testing.md) | 배치/실시간 서빙, A/B 테스트, 카나리 배포 |
| [06_orchestration_and_audit](06_orchestration_and_audit.md) | Step Functions, 리니지, 감사, 규제 |
