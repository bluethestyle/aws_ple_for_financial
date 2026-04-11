# AWS Deployment Guide

This guide covers deploying the PLE platform on AWS, from S3 bucket setup
through SageMaker training to production serving.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [S3 Bucket Setup](#s3-bucket-setup)
3. [SageMaker Training](#sagemaker-training)
4. [Serving Modes: Lambda vs ECS](#serving-modes)
5. [Feature Store](#feature-store)
6. [A/B Testing](#ab-testing)
7. [Kill Switch](#kill-switch)
8. [Monitoring and Alerts](#monitoring-and-alerts)
9. [운영/감사 에이전트 배포](#운영감사-에이전트-배포)

---

## Architecture Overview

```
                            +-------------------+
                            |  S3 Bucket        |
                            |  - Training data  |
                            |  - Model artifacts|
                            |  - Feature store  |
                            +--------+----------+
                                     |
              +----------------------+------------------------+
              |                      |                        |
    +---------v----------+  +--------v---------+    +---------v--------+
    | SageMaker Training |  | Serving Layer    |    | Monitoring       |
    | (Spot instances)   |  | Lambda or ECS    |    | CloudWatch       |
    | PLE -> Distill     |  | LGBM inference   |    | Fairness / Drift |
    | -> LGBM student    |  | + Recommendation |    | Incidents / SNS  |
    +--------------------+  |   Pipeline       |    +------------------+
                            +--------+---------+
                                     |
                            +--------v---------+
                            | DynamoDB         |
                            | - Feature store  |
                            | - Kill switch    |
                            | - A/B config     |
                            +------------------+
```

---

## S3 Bucket Setup

### Recommended bucket structure

```
s3://my-ple-bucket/
  data/
    train/                    # Training data (Parquet)
    validation/               # Validation data
    baselines/                # Drift detection baselines
  models/
    ple/
      v1/                     # PLE teacher model
      latest -> v1/
    lgbm/
      v1/                     # LGBM student model
      latest -> v1/
  features/
    store/                    # Precomputed feature vectors
    staging/                  # Container pipeline staging
  monitoring/
    reports/                  # Governance reports
    drift_results/            # Drift tracking
  incidents/                  # Incident archives
```

### IAM policy

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket",
        "s3:DeleteObject"
      ],
      "Resource": [
        "arn:aws:s3:::my-ple-bucket",
        "arn:aws:s3:::my-ple-bucket/*"
      ]
    }
  ]
}
```

---

## SageMaker Training

### Spot Instances

The platform supports SageMaker Managed Spot Training, which can reduce training
cost by up to 90%.

```yaml
aws:
  region: ap-northeast-2
  s3_bucket: my-ple-bucket
  instance_type: ml.g4dn.xlarge       # GPU instance for PLE
  use_spot: true                       # Enable Spot instances
  max_run_seconds: 7200                # 2-hour timeout
  max_wait_seconds: 14400              # 4-hour max wait for spot
  role_arn: arn:aws:iam::123456789:role/SageMakerRole
```

### Instance type recommendations

| Workload | Instance Type | vCPU | GPU | Memory | Cost/hr (Spot) |
|---|---|---|---|---|---|
| Small PLE (<100K rows) | `ml.g4dn.xlarge` | 4 | 1x T4 | 16 GB | ~$0.16 |
| Medium PLE (100K-1M) | `ml.g4dn.2xlarge` | 8 | 1x T4 | 32 GB | ~$0.24 |
| Large PLE (>1M) | `ml.g5.2xlarge` | 8 | 1x A10G | 32 GB | ~$0.38 |
| LGBM distillation | `ml.m5.xlarge` | 4 | None | 16 GB | ~$0.06 |
| Feature generation (container) | `ml.m5.xlarge` | 4 | None | 16 GB | ~$0.06 |

### Training pipeline

```python
from core.pipeline.config import PipelineConfig
from core.pipeline.runner import PipelineRunner

config = PipelineConfig.from_yaml("configs/examples/multitask_binary.yaml")
runner = PipelineRunner(config)

# Run on SageMaker
result = runner.run(
    mode="sagemaker",
    output_dir="s3://my-ple-bucket/models/ple/v2/",
)
```

### Training container

The training container entry point is `containers/training/train.py`. It:

1. Downloads data from S3
2. Runs the feature pipeline
3. Trains the PLE model (2-phase)
4. Distills to LGBM
5. Uploads model artifacts to S3

```bash
# Build the training container
docker build -t ple-training:latest -f containers/training/Dockerfile .

# Push to ECR
aws ecr get-login-password --region ap-northeast-2 | \
  docker login --username AWS --password-stdin 123456789.dkr.ecr.ap-northeast-2.amazonaws.com
docker tag ple-training:latest 123456789.dkr.ecr.ap-northeast-2.amazonaws.com/ple-training:latest
docker push 123456789.dkr.ecr.ap-northeast-2.amazonaws.com/ple-training:latest
```

---

## Serving Modes

The platform supports two serving backends: **AWS Lambda** for low-traffic
workloads and **ECS Fargate** for high-traffic workloads. An `auto` mode
selects between them based on traffic volume.

### Decision logic

```
                    Monthly Requests
                         |
            < 150M       |        >= 150M
                |                    |
            Lambda               ECS Fargate
         (pay-per-request)     (always-on containers)
```

### Lambda Serving

**File:** `containers/inference/lambda_handler.py`

Best for: < 150M monthly requests, bursty traffic, cost-sensitive workloads.

```yaml
serving:
  mode: lambda
  lambda:
    memory_mb: 1024                 # Lambda function memory
    timeout_seconds: 30             # Request timeout
    reserved_concurrency: 100       # Max concurrent executions
```

**Cost analysis (150M requests/month):**

| Component | Cost |
|---|---|
| Lambda invocations | $30.00 |
| Lambda compute (1024 MB, 100ms avg) | $200.00 |
| DynamoDB reads (feature store) | $37.50 |
| **Total** | **~$267.50/month** |

### ECS Fargate Serving

**File:** `containers/inference/app.py` (FastAPI)

Best for: >= 150M monthly requests, latency-sensitive, consistent traffic.

```yaml
serving:
  mode: ecs
  ecs:
    cpu: 1024                       # 1 vCPU
    memory: 2048                    # 2 GB
    min_tasks: 2                    # Minimum running tasks
    max_tasks: 10                   # Auto-scaling ceiling
    target_cpu_pct: 70              # Auto-scale target CPU
```

**Cost analysis (500M requests/month):**

| Component | Cost |
|---|---|
| ECS Fargate (4 tasks avg) | $116.00 |
| ALB | $22.00 |
| DynamoDB reads | $125.00 |
| **Total** | **~$263.00/month** |

### Auto mode

```yaml
serving:
  mode: auto
  auto_threshold: 150000000          # Switch to ECS above 150M requests/month
```

```python
from core.serving.config import ServingConfig

config = ServingConfig.from_dict(yaml_config)
resolved = config.resolve_compute_mode(monthly_requests=200_000_000)
# ServingMode.ECS
```

### Inference flow

Both Lambda and ECS use the same `RecommendationService` class:

```
Request (user_id, context)
    |
    v
1. Kill switch check          --> blocked? return fallback
    |
2. A/B variant selection      --> assign user to variant
    |
3. Feature store lookup        --> get precomputed features
    |
4. Context enrichment          --> merge request context
    |
5. LGBM multi-task inference   --> predict per task
    |
6. Output normalisation        --> sigmoid/softmax/identity per task type
    |
7. Recommendation pipeline     --> scoring + filtering + reasons (optional)
    |
    v
Response (predictions, recommendations, variant, metadata)
```

---

## Feature Store

Two backends are available: in-memory (for development and small user bases)
and DynamoDB (for production).

### In-memory feature store

```yaml
serving:
  feature_store: memory
  feature_store_config:
    s3_uri: s3://my-ple-bucket/features/store/
```

Features are loaded from S3 at startup and kept in a Python dict. Fast but
limited by Lambda/ECS memory.

### DynamoDB feature store

```yaml
serving:
  feature_store: dynamodb
  feature_store_config:
    table_name: ple-feature-store
    region: ap-northeast-2
    read_capacity: 100               # On-demand recommended for variable traffic
```

DynamoDB table schema:

| Attribute | Type | Description |
|---|---|---|
| `user_id` (PK) | `S` | User identifier |
| `features` | `M` | Map of feature name to value |
| `updated_at` | `N` | Unix timestamp of last update |

### Auto mode

```yaml
serving:
  feature_store: auto
  auto_feature_threshold: 5000000     # Switch to DynamoDB above 5M users
```

---

## A/B Testing

Built-in A/B testing support for model variant comparison.

### Configuration

```yaml
serving:
  ab_test:
    enabled: true
    variants:
      - name: control
        model_path: s3://my-ple-bucket/models/lgbm/v1/
        weight: 0.8                   # 80% of traffic
      - name: challenger
        model_path: s3://my-ple-bucket/models/lgbm/v2/
        weight: 0.2                   # 20% of traffic
```

### How it works

1. Each user is deterministically assigned to a variant based on a hash of
   their `user_id`.
2. The hash ensures consistent assignment (same user always sees same variant).
3. The `weight` controls the traffic split.
4. Latency and prediction metrics are tracked per variant.
5. The variant name is included in the response for downstream analysis.

### Reading A/B results

```python
response = service.predict("user_123")
print(response.variant)      # "control" or "challenger"
print(response.metadata)     # {"ab_variant": "challenger", "ab_hash": 42}
```

---

## Kill Switch

Emergency circuit breaker that immediately halts model inference and returns
a fallback response. Backed by DynamoDB for instant propagation.

### Configuration

```yaml
serving:
  kill_switch:
    table_name: ple-kill-switch
    fallback_strategy: rule_based     # What to do when kill switch fires
```

### DynamoDB table schema

| Attribute | Type | Description |
|---|---|---|
| `scope` (PK) | `S` | `"global"`, `"task:{name}"`, or `"cluster:{id}"` |
| `active` | `BOOL` | Whether the kill switch is engaged |
| `reason` | `S` | Human-readable explanation |
| `activated_by` | `S` | Who activated it |
| `activated_at` | `N` | Unix timestamp |
| `fallback_strategy` | `S` | `"rule_based"`, `"cached"`, `"default"` |

### Activating the kill switch

```python
# Via the KillSwitch API
from core.serving.kill_switch import KillSwitch

ks = KillSwitch(table_name="ple-kill-switch")
ks.activate(
    scope="global",
    reason="Model drift detected, rolling back",
    fallback_strategy="rule_based",
)
```

```bash
# Via AWS CLI
aws dynamodb put-item \
  --table-name ple-kill-switch \
  --item '{
    "scope": {"S": "global"},
    "active": {"BOOL": true},
    "reason": {"S": "Emergency rollback"},
    "activated_by": {"S": "oncall-engineer"},
    "fallback_strategy": {"S": "rule_based"}
  }'
```

### Fallback strategies

| Strategy | Behaviour |
|---|---|
| `rule_based` | Use hand-crafted business rules instead of model |
| `cached` | Return the last successful prediction for this user |
| `default` | Return a static default response |

---

## Monitoring and Alerts

The monitoring subsystem runs alongside the serving layer and detects issues
before they impact users.

### Fairness monitoring

Detects bias across protected attributes (age, gender, region, income).

```yaml
fairness:
  thresholds:
    di_lower: 0.8                    # Disparate Impact lower bound
    di_upper: 1.25
    spd_max: 0.1                     # Statistical Parity Difference
    eod_max: 0.1                     # Equal Opportunity Difference
  protected_attributes: [age_group, gender, region_type, income_tier]
  auto_incident: true                # Create incident on violation
```

### Drift detection

Monitors feature distribution shifts using Population Stability Index (PSI).

```yaml
drift:
  psi_threshold_warning: 0.1
  psi_threshold_critical: 0.25
  consecutive_tracker:
    consecutive_threshold: 3          # Retrain after 3 consecutive critical days
    critical_feature_threshold: 5
```

### Herding detection

Monitors recommendation concentration (are we recommending the same items to
everyone?).

```yaml
herding:
  hhi_warning: 0.15                  # Herfindahl-Hirschman Index warning
  hhi_critical: 0.25
  top_k_concentration_warning: 0.5
  top_k_concentration_critical: 0.7
```

### Incident management

Classifies incidents by severity and routes them to SNS for notification.

```yaml
incidents:
  severity_criteria:
    critical:
      conditions: [kill_switch_activation, di_below_0.6, security_breach]
      response_time: "1 hour"
      escalate: true
    major:
      conditions: [di_below_0.8, herding_critical, model_rollback]
      response_time: "4 hours"
      escalate: true
    minor:
      conditions: [drift_warning, quality_drop]
      response_time: "24 hours"
      escalate: false
  sns_topic_arn: arn:aws:sns:ap-northeast-2:123456789:ple-incidents
```

### CloudWatch integration

Key metrics to monitor:

| Metric | Source | Alarm Threshold |
|---|---|---|
| Inference latency (p99) | Lambda/ECS | > 500ms |
| Kill switch activations | DynamoDB Streams | > 0 |
| Feature store miss rate | Feature store | > 5% |
| PSI critical features | Drift detector | > 5 features |
| Fairness DI violations | Fairness monitor | Any violation |
| Recommendation HHI | Herding detector | > 0.25 |

### Governance reports

Weekly automated reports combining all monitoring signals:

```yaml
general:
  governance_report:
    enabled: true
    frequency: weekly
    include_fairness: true
    include_drift: true
    include_incidents: true
    include_herding: true
    output_format: json
```

---

## 운영/감사 에이전트 배포

### 사전 요구사항

- Bedrock 모델 접근 권한: Claude Sonnet, Claude Haiku, Titan Embeddings V2
- Bedrock Marketplace: Upstage Solar Pro (선택, L2a 사유 생성용)
- IAM 역할: `bedrock:InvokeModel`, `s3:GetObject/PutObject`, `dynamodb:*`, `sns:Publish`

### 에이전트 설정

- `configs/financial/agent.yaml`: 에이전트 설정 (모델, 스케줄, 합의)
- `configs/financial/checklist.yaml`: 48개 체크리스트 항목

### 실행 모드

1. **이벤트 기반**: 파이프라인 스테이지 완료 시 자동 트리거
   - `_PipelineState.mark_complete()` 콜백 → ChangeDetector → 에이전트 실행
2. **주기적**: CloudWatch Events / EventBridge 스케줄
   - CP5 서빙 헬스: 5분 주기
   - CP6 추천 응답: 1시간 집계
   - AV4 규제 적합성: 주 1회

### Bedrock 합의 메커니즘

- 3개 Sonnet 세션 병렬 호출 (독립 투표)
- WARN/FAIL 항목에만 적용 (비용 효율)
- 마이너리티 리포트 보존

### 케이스 스토어

- LanceDB on S3 (`DiagnosticCaseStore`)
- 진단 이력 영구 보존 — 유사 케이스 검색, 통계, 대응 효과 추적

### 알림

- SNS: CRITICAL/FAIL 자동 에스컬레이션
- Slack: 일일 `ops_report` / 주간 `audit_report` 전달

---

### 메모리 프레임워크 컴포넌트 (2026-04 추가)

PaperClip과 메모리 프레임워크 연구에서 차용한 선택적 기능:

**PaperClip 3종 (운영 효율성):**
- `core/agent/heartbeat.py` — 에이전트 주기 실행 (CP5 5분, CP6 1시간 등)
- `core/agent/budget.py` — 에이전트별 토큰 예산 (80% 경고, 100% 하드스톱)
- `core/agent/tracer.py` — 모든 `ToolRegistry.call()` 자동 추적

**메모리 프레임워크 4종:**
- `core/agent/temporal_fact_store.py` — 시간적 지식 그래프 (Zep/Graphiti)
- `core/agent/dialog_recall.py` — 담당자 대화 DynamoDB 저장 (Letta)
- `core/recommendation/reason/fact_extractor.py` — 룰 기반 고객 팩트 추출 (Mem0)
- `core/agent/case_store.py` — 기존 파일에 시간 decay 추가 (SuperLocalMemory)

**설정 추가:**
- `configs/financial/agent.yaml`: `heartbeat`, `budget`, `tracer` 섹션
- `configs/financial/fact_extraction.yaml`: 15개 룰 기본 제공

**의존성:** 모든 차용은 기존 LanceDB + DynamoDB 스택을 재사용. 신규 의존성 0.

**활성화:** 전부 opt-in — `agent.yaml`의 해당 섹션을 설정하지 않으면 기존 동작 그대로.

---

### 온프레미스 배포

온프레미스에서는 Bedrock/SageMaker 대신 로컬 환경을 사용:
- **학습**: 로컬 GPU (RTX 4070) + PyTorch
- **서빙**: Docker 컨테이너 + vLLM
- **모델**: Exaone 3.5 7.8B (사유) + Qwen 2.5 14B Q4 (에이전트)
- **에이전트**: 동일한 룰 엔진 + 2-Round 합의 (Sonnet 대신 Qwen)
- **알림**: SNS 대신 이메일/Slack

상세: `docs/design/11_ops_audit_agent_onprem_handoff.md`
