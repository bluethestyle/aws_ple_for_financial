# 05. Serving & Online Testing — 배치/실시간 서빙, A/B 테스트

## 현재 (On-Prem) 분석

### 서빙 구조
- **model_server**: FastAPI + Docker (PLE 추론)
- **conda_lgbm**: FastAPI + Docker (경량 LGBM 추론)
- **엔드포인트**: `/inference/predict`, `/inference/batch`, `/inference/interpret`
- **모델 로딩**: model_manager.py에서 체크포인트 로드/캐싱

### 문제점
1. **Docker Compose 상시 가동**: 트래픽 없어도 리소스 점유
2. **스케일링 불가**: 단일 서버, 수평 확장 없음
3. **A/B 테스트 없음**: 새 모델 배포 시 전량 교체
4. **출력 정규화 복잡**: 태스크별 스케일링이 코드에 흩어져 있음

### 유지할 패턴
- **FastAPI 기반 API**: 인터페이스 유지, 배포 환경만 변경
- **태스크별 출력 정규화**: binary→sigmoid, multiclass→softmax, regression→clipping
- **Expert weight 해석**: 추론 시 Expert 기여도 반환

---

## AWS 설계

### 서빙 모드 3가지

```
┌─────────────────────────────────────────────────────────┐
│                    서빙 전략 선택                         │
│                                                         │
│  ① 배치 추론 (비용 최소)                                  │
│     SageMaker Batch Transform                           │
│     - S3에 입력 업로드 → 결과 S3에 저장                    │
│     - 인스턴스 자동 종료                                   │
│     - 월 1-2회 대량 추론에 최적                            │
│                                                         │
│  ② 서버리스 추론 (중간)                                   │
│     Lambda + API Gateway                                │
│     - 요청 당 과금, 유휴 시 0원                            │
│     - Cold start 주의 (모델 로딩 시간)                     │
│     - LGBM 경량 모델에 적합                               │
│                                                         │
│  ③ 실시간 추론 (포트폴리오 시연)                           │
│     ECS Fargate + ALB                                   │
│     - FastAPI 컨테이너 그대로 배포                         │
│     - Auto-scaling (min=0, 트래픽 시 스케일)               │
│     - A/B 테스트, 카나리 배포 가능                          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### ① 배치 추론 (기본 모드)

```yaml
# configs/serving/batch.yaml
serving:
  mode: batch
  input: s3://bucket/inference/input/
  output: s3://bucket/inference/output/
  model_path: s3://bucket/models/ple-latest/
  instance_type: ml.g4dn.xlarge
  instance_count: 1
```

```
S3 입력 데이터
    ↓
SageMaker Batch Transform
    ├── 모델 로드 (S3)
    ├── 배치별 추론
    └── 결과 Parquet → S3
    ↓
인스턴스 자동 종료
```

### ② 서버리스 추론 (LGBM)

```python
# containers/inference/lambda_handler.py
import json
import pickle
import numpy as np

# Lambda 초기화 시 모델 로드 (Cold start에 포함)
model = pickle.load(open("/opt/ml/model/lgbm.pkl", "rb"))

def handler(event, context):
    features = np.array(event["features"])
    predictions = {}
    for task_name, task_model in model.items():
        predictions[task_name] = task_model.predict_proba(features).tolist()
    return {"predictions": predictions}
```

### ③ 실시간 추론 (ECS Fargate)

```
클라이언트 요청
    ↓
API Gateway
    ↓
ALB (Application Load Balancer)
    ├── Rule: /v1/* → Target Group A (Model v1, weight: 90%)
    └── Rule: /v1/* → Target Group B (Model v2, weight: 10%)  ← A/B Test
    ↓
ECS Fargate
    ├── Task: FastAPI 컨테이너 (ECR 이미지)
    ├── Auto-scaling: CPU > 70% → scale up
    └── 모델: S3에서 시작 시 로드 → 메모리 캐싱
```

```python
# containers/inference/main.py (FastAPI — 현재 model_server 구조 유지)
from fastapi import FastAPI
from core.pipeline.config import load_config
from core.model.registry import ModelRegistry

app = FastAPI()

@app.on_event("startup")
async def load_model():
    """S3에서 모델 로드 → 메모리 캐싱"""
    global model, config
    config = load_config(os.environ["CONFIG_PATH"])
    model = ModelRegistry.load(os.environ["MODEL_S3_URI"])

@app.post("/v1/predict")
async def predict(request: PredictRequest):
    outputs = model(request.features)
    return normalize_outputs(outputs, config.tasks)

@app.post("/v1/interpret")
async def interpret(request: PredictRequest):
    """Expert 기여도 + 피처 중요도 반환."""
    outputs = model(request.features, return_expert_weights=True)
    return {
        "predictions": outputs.predictions,
        "expert_weights": outputs.expert_weights,
        "feature_importance": outputs.feature_importance,
    }
```

### A/B 테스트

```yaml
# configs/serving/ab_test.yaml
ab_test:
  enabled: true
  variants:
    - name: control
      model: s3://bucket/models/ple-v1/
      weight: 90              # 트래픽 90%
    - name: treatment
      model: s3://bucket/models/ple-v2/
      weight: 10              # 트래픽 10%

  metrics:
    primary: click_through_rate
    secondary: [conversion_rate, revenue_per_user]
    min_sample_size: 10000
    significance_level: 0.05

  auto_promote:
    enabled: true
    # treatment이 control 대비 유의미하게 좋으면 자동 전환
    min_improvement: 0.02      # 2% 이상 개선
```

```
ALB 가중 라우팅
    ├── 90% → ECS Task (Model v1) → CloudWatch 메트릭 기록
    └── 10% → ECS Task (Model v2) → CloudWatch 메트릭 기록
    ↓
Lambda (통계 분석)
    ├── 일별 메트릭 비교
    ├── t-test / 베이지안 A/B
    └── 유의미 → 자동 전환 (또는 알림)
```

### 카나리 배포

```
배포 단계:
  1. 새 모델 → ECR 이미지 빌드
  2. 카나리: 5% 트래픽으로 시작
  3. 메트릭 모니터링 (5분 간격)
  4. 이상 없으면: 25% → 50% → 100%
  5. 이상 발견: 즉시 롤백 (이전 이미지로)
```

### 출력 정규화 (태스크 타입별)

```python
# core/serving/normalizer.py
class OutputNormalizer:
    """
    태스크 타입에 따라 모델 출력을 정규화합니다.
    On-Prem의 recommendation.py 로직을 범용화.
    """

    STRATEGIES = {
        "binary": lambda logits: torch.sigmoid(logits),         # [0, 1]
        "multiclass": lambda logits: torch.softmax(logits, -1), # [0, 1] sum=1
        "regression": lambda logits: logits,                     # raw value
        "contrastive": lambda logits: F.normalize(logits, dim=-1), # unit vector
        "ranking": lambda logits: logits,                        # raw score
    }

    def normalize(self, outputs: dict, task_configs: list) -> dict:
        result = {}
        for task in task_configs:
            strategy = self.STRATEGIES[task.type]
            result[task.name] = strategy(outputs[task.name].logits)
        return result
```

---

## 서빙 비용 비교

| 모드 | 월 비용 (추정) | 적합한 상황 |
|------|--------------|------------|
| 배치 (Batch Transform) | ~$1-3 (월 2-3회 실행) | 정기 대량 추론 |
| 서버리스 (Lambda) | ~$0-1 (1만 요청 이하) | 간헐적 소량 요청 |
| 실시간 (ECS Fargate) | ~$5-30 (가동 시간 비례) | 데모, 포트폴리오 시연 |

---

## 현재 vs AWS — 서빙 비교

| 항목 | 현재 (On-Prem) | AWS (설계) | 변경 이유 |
|------|---------------|-----------|----------|
| 배포 | Docker Compose (상시) | ECS Fargate (필요 시) | 유휴 비용 0 |
| 스케일링 | 불가 (단일 서버) | Auto-scaling (0~N) | 트래픽 대응 |
| A/B 테스트 | 없음 | ALB 가중 라우팅 | 안전한 모델 교체 |
| 카나리 | 없음 | 점진적 트래픽 전환 | 롤백 안전성 |
| 배치 추론 | API 호출 루프 | Batch Transform | 인프라 자동 관리 |
| 해석 API | /interpret 엔드포인트 | 동일 유지 | FastAPI 구조 재사용 |
