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

### Config pattern for SageMaker

SageMaker Training Jobs receive config paths as hyperparameters. The platform
supports two patterns:

**Split-config (recommended):** `pipeline.yaml` holds common defaults;
`datasets/<name>.yaml` holds dataset-specific tasks, labels, and adapter.
The two files are deep-merged at job startup.

```python
hyperparameters = {
    "config":         "configs/pipeline.yaml",
    "dataset_config": "configs/datasets/santander.yaml",
    # All other HP overrides (batch_size, epochs, etc.)
    "batch_size": 5632,
    "epochs": 10,
}
```

**Single-file (backward compatible):** omit `dataset_config` and pass a
self-contained YAML as `config`.

```python
hyperparameters = {
    "config": "configs/santander/pipeline.yaml",  # legacy single file
}
```

### Training pipeline

```python
from core.pipeline.config import PipelineConfig
from core.pipeline.runner import PipelineRunner

config = PipelineConfig.from_yaml("configs/pipeline.yaml",
                                   dataset_config="configs/datasets/santander.yaml")
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
3. Trains the PLE model (2-phase, with checkpoint resume support)
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

### SageMaker 제출 스크립트

| 스크립트 | 역할 |
|---------|------|
| `scripts/package_source.py` | source 디렉토리 스테이징 → tarball 생성 → S3 업로드 (모든 Job에서 재사용) |
| `scripts/run_sagemaker_teacher.py` | 3-시나리오(baseline/gradsurgery/adaTT) Spot 학습 Job 병렬 제출 |
| `scripts/run_sagemaker_eval.py` | 평가 Job 제출 (`containers/evaluation/eval_entry.py` 진입점 사용) |

```bash
# 1. source 패키징 (한 번만 실행)
python scripts/package_source.py --config configs/pipeline.yaml

# 2. teacher 학습 Job 제출 (3-시나리오 병렬)
#    --dataset 로 dataset config를 지정한다 (split-config 패턴)
python scripts/run_sagemaker_teacher.py \
  --config configs/pipeline.yaml \
  --dataset configs/datasets/santander.yaml

# 3. 평가 Job 제출
python scripts/run_sagemaker_eval.py \
  --config configs/pipeline.yaml \
  --dataset configs/datasets/santander.yaml \
  --model-s3 s3://my-ple-bucket/models/ple/v2/
```

### 평가 컨테이너

**파일:** `containers/evaluation/eval_entry.py`

SageMaker Processing Job 진입점으로, S3에서 모델 아티팩트와 피처 데이터를 내려받아 평가 메트릭(`eval_metrics.json`)을 산출하고 다시 S3에 저장한다.

```
s3://my-ple-bucket/
  models/
    ple/
      v{n}/
        eval_metrics.json   ← 평가 결과 (run_sagemaker_eval.py가 존재 여부 확인 후 스킵)
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

## Ops / Audit Agent Deployment

### Prerequisites

- Bedrock model access: Claude Sonnet, Claude Haiku, Titan Embeddings V2
- Bedrock Marketplace: Upstage Solar Pro (optional, for L2a reason generation)
- IAM role: `bedrock:InvokeModel`, `s3:GetObject/PutObject`, `dynamodb:*`, `sns:Publish`

### Agent Configuration

- `configs/financial/agent.yaml`: Agent settings (models, schedules, consensus)
- `configs/financial/checklist.yaml`: 48 checklist items

### Execution Modes

1. **Event-driven**: Automatically triggered on pipeline stage completion
   - `_PipelineState.mark_complete()` callback -> ChangeDetector -> agent execution
2. **Periodic**: CloudWatch Events / EventBridge schedule
   - CP5 serving health: every 5 minutes
   - CP6 recommendation response: hourly aggregation
   - AV4 regulatory compliance: weekly

### Bedrock Consensus Mechanism

- 3 Sonnet sessions called in parallel (independent voting)
- Applied only to WARN/FAIL items (cost-efficient)
- Minority report preserved

### Case Store

- LanceDB on S3 (`DiagnosticCaseStore`)
- Permanent diagnostic history -- similar case retrieval, statistics, response effectiveness tracking

### Alerts

- SNS: Automatic escalation for CRITICAL/FAIL
- Slack: Daily `ops_report` / weekly `audit_report` delivery

---

### Memory Framework Components (added 2026-04)

Optional features borrowed from PaperClip and memory framework research:

**PaperClip (3 components for operational efficiency):**
- `core/agent/heartbeat.py` -- Periodic agent execution (CP5 every 5 min, CP6 every 1 hour, etc.)
- `core/agent/budget.py` -- Per-agent token budget (80% warning, 100% hard stop)
- `core/agent/tracer.py` -- Automatic tracing of all `ToolRegistry.call()` invocations

**Memory Framework (4 components):**
- `core/agent/temporal_fact_store.py` -- Temporal knowledge graph (Zep/Graphiti)
- `core/agent/dialog_recall.py` -- Advisor dialog DynamoDB storage (Letta)
- `core/recommendation/reason/fact_extractor.py` -- Rule-based customer fact extraction (Mem0)
- `core/agent/case_store.py` -- Time decay added to existing file (SuperLocalMemory)

**Configuration additions:**
- `configs/financial/agent.yaml`: `heartbeat`, `budget`, `tracer` sections
- `configs/financial/fact_extraction.yaml`: 15 default rules provided

**Dependencies:** All borrowed components reuse the existing LanceDB + DynamoDB stack. Zero new dependencies.

**Activation:** All opt-in -- if the corresponding section in `agent.yaml` is not configured, existing behavior is unchanged.

---

### On-Premises Deployment

On-premises deployments use local infrastructure instead of Bedrock/SageMaker:
- **Training**: Local GPU (RTX 4070) + PyTorch
- **Serving**: Docker containers + vLLM
- **Models**: Exaone 3.5 7.8B (reason generation) + Qwen 2.5 14B Q4 (agents)
- **Agents**: Same rule engine + 2-round consensus (Qwen instead of Sonnet)
- **Alerts**: Email/Slack instead of SNS

Details: `docs/design/11_ops_audit_agent_onprem_handoff.md`
