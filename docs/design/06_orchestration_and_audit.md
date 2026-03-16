# 06. Orchestration & Audit — Step Functions, 리니지, 감사, 규제

## 현재 (On-Prem) 분석

### 오케스트레이션
- **Airflow**: 86 DAGs (Docker Compose, Postgres 백엔드)
- **구조**: Master DAG → 하위 DAG 트리거 (TriggerDagRunOperator)
- **센서**: ExternalTaskSensor로 Cross-DAG 의존성
- **스케줄**: 매일 03:00 전처리 시작

### 감사/린리지
- **audit_logger.py**: 이벤트 로깅 (데이터 접근, 모델 변경)
- **audit_access_controller.py**: GDPR 접근 제어
- **compliance_dvc_tracker.py**: 데이터 버전 관리 (DVC)
- **data_lineage_tracker.py**: 변환 추적
- **audit_package_builder.py**: 감사 증거 패키지

### 문제점
1. **Airflow 상시 비용**: 4개 컨테이너 (webserver, scheduler, triggerer, postgres) 상시 가동
2. **DAG 복잡도**: 86개 DAG 중 다수가 도메인 특화 (G1-G10별 개별 DAG)
3. **ExternalTaskSensor 이슈**: 일부 DAG ID를 task ID로 잘못 참조
4. **감사 추적 분산**: audit_logger, DVC, MLflow가 별도 시스템
5. **리니지 불완전**: 데이터 → 피처 → 모델 전체 연결이 암묵적

### 유지할 패턴
- **감사 로깅**: 모든 데이터 접근/모델 변경 기록 → AWS 환경에서 확장
- **DAG 구조의 논리적 단계**: 수집 → 전처리 → 피처 → 학습 → 서빙 → 순서 유지

---

## AWS 설계 — 오케스트레이션

### Airflow 86 DAGs → Step Functions 5개 상태 머신

현재 86개 DAG를 기능별로 **5개 Step Functions 상태 머신**으로 통합합니다.

```
[On-Prem 86 DAGs]                    [AWS 5 State Machines]
──────────────────                   ──────────────────────
G1-G10 수집 DAGs (10+)      ──▶     ① ingestion_pipeline
전처리 DAGs (10+)            ──▶     ② preprocessing_pipeline
피처 추출/통합 DAGs (10+)    ──▶     ③ feature_pipeline
학습/증류 DAGs (5+)          ──▶     ④ training_pipeline
서빙/모니터링 DAGs (5+)      ──▶     ⑤ serving_pipeline
나머지 유틸리티 DAGs         ──▶     Lambda 함수 또는 제거
```

### 전체 오케스트레이션 흐름

```
EventBridge (스케줄 또는 S3 이벤트 트리거)
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ ① ingestion_pipeline                                           │
│                                                                 │
│   [데이터 소스 스캔] → [변경 감지] → [S3 업로드]                   │
│         │                              │                        │
│         └── 변경 없음 → 종료            └── S3 이벤트 발생        │
└─────────────────────────────────────────────────────────────────┘
    ↓ (S3 이벤트)
┌─────────────────────────────────────────────────────────────────┐
│ ② preprocessing_pipeline                                       │
│                                                                 │
│   [스키마 검증] ─── 실패 → [알림 (SNS)] → 종료                    │
│        │ 통과                                                    │
│        ▼                                                        │
│   [데이터 품질 검사] ─── 실패 → [알림] → 종료                      │
│        │ 통과                                                    │
│        ▼                                                        │
│   [PII 처리] → [타입 정규화] → [S3 processed/ 저장]               │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ ③ feature_pipeline                                             │
│                                                                 │
│   SageMaker Processing Job                                     │
│   [피처 추출] → [정규화] → [통합] → [S3 features/v{n}/ 저장]      │
│        │                                                        │
│        └── fit된 transformer → S3 features/v{n}/transformers/   │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ ④ training_pipeline                                            │
│                                                                 │
│   [Phase 1: Shared Expert]                                     │
│        ↓ 체크포인트 → S3                                        │
│   [Phase 2: Task Head]                                         │
│        ↓ 최종 모델 → S3                                         │
│   [평가: Champion/Challenger]                                   │
│        ├── 합격 → [Model Registry 등록]                          │
│        └── 불합격 → [알림] → 종료                                 │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ ⑤ serving_pipeline (선택적 — 자동 배포 활성화 시)                  │
│                                                                 │
│   [ECR 이미지 빌드] → [카나리 배포 5%] → [메트릭 확인]             │
│        ├── 정상 → [50%] → [100%] → 완료                         │
│        └── 이상 → [롤백] → [알림]                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Step Functions 상태 머신 설계

```json
// ④ training_pipeline 예시 (간략)
{
  "StartAt": "CheckFeatureReady",
  "States": {
    "CheckFeatureReady": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:...:check-s3-exists",
      "Next": "Phase1Training"
    },
    "Phase1Training": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sagemaker:createTrainingJob.sync",
      "Parameters": {
        "EnableManagedSpotTraining": true,
        "HyperParameters": {"phase": "1"}
      },
      "Catch": [{"ErrorEquals": ["States.ALL"], "Next": "NotifyFailure"}],
      "Next": "Phase2Training"
    },
    "Phase2Training": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sagemaker:createTrainingJob.sync",
      "Parameters": {
        "HyperParameters": {"phase": "2"},
        "InputDataConfig": [{"ChannelName": "checkpoint"}]
      },
      "Catch": [{"ErrorEquals": ["States.ALL"], "Next": "NotifyFailure"}],
      "Next": "Evaluate"
    },
    "Evaluate": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sagemaker:createProcessingJob.sync",
      "Next": "ChampionChallenger"
    },
    "ChampionChallenger": {
      "Type": "Choice",
      "Choices": [{
        "Variable": "$.evaluation.passed",
        "BooleanEquals": true,
        "Next": "RegisterModel"
      }],
      "Default": "NotifyReview"
    },
    "RegisterModel": {"Type": "Succeed"},
    "NotifyFailure": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sns:publish",
      "Next": "FailState"
    },
    "NotifyReview": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sns:publish",
      "End": true
    },
    "FailState": {"Type": "Fail"}
  }
}
```

### 트리거 방식

| 트리거 | 메커니즘 | 대상 |
|--------|---------|------|
| 신규 데이터 업로드 | S3 이벤트 → EventBridge | ② preprocessing |
| 정기 학습 | EventBridge 스케줄 (주 1회) | ④ training |
| 수동 실행 | AWS Console / CLI | 어떤 파이프라인이든 |
| 모델 등록 | ④ 완료 이벤트 | ⑤ serving (선택적) |

### 비용 비교

```
Airflow (현재):
  4 컨테이너 상시 → ~$200-300/월 (EC2 기준)

Step Functions (AWS):
  5 상태 머신 × 주 2회 실행 × 평균 6 transition
  = 60 transition/월
  = $0.0015/월  ← 사실상 무료
```

---

## AWS 설계 — 감사 & 리니지

### 3계층 감사 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│ Layer 1: AWS 네이티브 (자동)                              │
│                                                         │
│ CloudTrail      ← 모든 AWS API 호출 자동 기록            │
│ S3 Versioning   ← 데이터/모델 변경 이력 자동 보존         │
│ SageMaker       ← 학습/추론 Job 메타데이터 자동 기록      │
│   Lineage                                               │
└─────────────────────────────────────────────────────────┘
        +
┌─────────────────────────────────────────────────────────┐
│ Layer 2: 플랫폼 레벨 (반자동)                             │
│                                                         │
│ ExperimentTracker ← 하이퍼파라미터/메트릭 기록             │
│ SchemaRegistry    ← 스키마 버전 이력                      │
│ FeaturePipeline   ← fit된 transformer 버전별 저장         │
│ ModelRegistry     ← 모델 버전 + 평가 결과                 │
└─────────────────────────────────────────────────────────┘
        +
┌─────────────────────────────────────────────────────────┐
│ Layer 3: 비즈니스 레벨 (명시적)                            │
│                                                         │
│ AuditLogger       ← 의사결정 근거 기록                    │
│ ComplianceChecker ← 규제 준수 검증                       │
│ AccessController  ← 데이터 접근 권한 관리                  │
└─────────────────────────────────────────────────────────┘
```

### 데이터 리니지 — 전체 추적 경로

```
Raw Data (S3, 버전 v1)
    │ 변환: preprocessing_pipeline (execution_id: abc123)
    │ 시간: 2024-01-01 03:15:00
    ▼
Processed Data (S3, 버전 v1)
    │ 변환: feature_pipeline (execution_id: def456)
    │ config: features_v1.0.yaml (sha256: 7a3b...)
    │ transformers: s3://bucket/features/v1.0/transformers/
    ▼
Features v1.0 (S3)
    │ 학습: training_pipeline (execution_id: ghi789)
    │ config: training_v1.yaml (sha256: 9c2d...)
    │ 환경: ml.g4dn.xlarge, Spot, PyTorch 2.1
    ▼
Model v1.0 (S3)
    │ 평가: AUC=0.85, MAE=0.12
    │ 비교: Champion v0.9 (AUC=0.83) → 승격
    ▼
Serving (ECS, image sha256: ef12...)
    │ A/B: 10% traffic → 메트릭 양호 → 100%
    ▼
Production
```

### 리니지 메타데이터 구조

```python
# core/audit/lineage.py
@dataclass
class LineageRecord:
    """모든 파이프라인 단계에서 자동 생성되는 리니지 기록."""
    execution_id: str           # Step Functions execution ID
    stage: str                  # preprocessing | feature | training | serving
    timestamp: str
    inputs: list[str]           # S3 URI 목록
    outputs: list[str]          # S3 URI 목록
    config_snapshot: str        # 사용된 config의 S3 URI (또는 SHA)
    code_version: str           # git commit hash
    environment: dict           # instance_type, framework_version 등
    metrics: dict | None        # 평가 메트릭 (학습/서빙 단계)
    parent_execution: str | None # 이전 단계의 execution_id
```

```
저장 위치:
s3://bucket/audit/lineage/
├── 2024/01/01/
│   ├── preprocessing_abc123.json
│   ├── feature_def456.json
│   └── training_ghi789.json
```

### PII 처리 — 스키마 기반 자동화

```yaml
# 스키마에서 PII 마킹
sources:
  - name: transactions
    columns:
      - {name: phone, type: string, pii: true}    # PII 마킹
      - {name: email, type: string, pii: true}
      - {name: amount, type: float64, pii: false}
```

```python
# core/data/pii_handler.py
class PIIHandler:
    """
    스키마의 pii: true 컬럼을 자동 처리합니다.

    전략:
    - hash: SHA256 해싱 (조인 가능, 복원 불가)
    - mask: 부분 마스킹 (010-****-1234)
    - drop: 완전 제거
    - encrypt: AES 암호화 (키 관리: AWS KMS)
    """

    def process(self, df, schema, strategy="hash"):
        pii_cols = [c.name for c in schema.columns if c.pii]
        for col in pii_cols:
            if strategy == "hash":
                df[col] = df[col].apply(lambda x: hashlib.sha256(str(x).encode()).hexdigest())
            elif strategy == "drop":
                df = df.drop(columns=[col])
            # KMS 암호화는 aws/kms.py에서 처리
        return df
```

### 모니터링 — CloudWatch 통합

```yaml
# configs/monitoring.yaml
monitoring:
  # 데이터 드리프트
  drift:
    metric: psi                # PSI (Population Stability Index)
    warning_threshold: 0.1
    critical_threshold: 0.25
    check_frequency: daily
    action_on_critical: trigger_retraining  # 3일 연속 시

  # 모델 성능
  model:
    metrics: [auc, mae, f1_macro]
    degradation_threshold: 0.05  # 5% 이상 성능 저하 시
    action: notify               # notify | auto_retrain

  # 인프라
  infra:
    cpu_alarm: 80%
    memory_alarm: 85%
    cost_alarm: $10/month
```

---

## 현재 vs AWS — 오케스트레이션 & 감사 비교

| 항목 | 현재 (On-Prem) | AWS (설계) | 변경 이유 |
|------|---------------|-----------|----------|
| 오케스트레이션 | Airflow 86 DAGs | Step Functions 5개 | 비용 $300→$0, 복잡도 감소 |
| API 호출 감사 | 없음 | CloudTrail (자동) | 전체 API 기록 |
| 데이터 버전 | DVC (설정 불완전) | S3 Versioning (자동) | 추가 도구 불필요 |
| 실험 리니지 | MLflow (부분) | SageMaker Lineage + 커스텀 | 완전한 E2E 추적 |
| PII 처리 | encryption_config.yaml 별도 | 스키마 내 pii 선언 → 자동 | 누락 방지 |
| 비용 모니터링 | 없음 | CloudWatch + Budgets | 비용 폭탄 방지 |
| DAG 의존성 | ExternalTaskSensor (일부 오류) | Step Functions 내장 흐름 | 의존성 명시적 |
