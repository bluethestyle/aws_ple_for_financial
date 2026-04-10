# 05. Serving & Online Testing — 실시간 추론, 규모별 자동 전환, A/B 테스트

## 현재 (On-Prem) 분석

### 서빙 구조
- **model_server**: FastAPI + Docker (PLE 추론)
- **conda_lgbm**: FastAPI + Docker (경량 LGBM 추론)
- **지식 증류**: PLE → LGBM으로 모델 압축 (추론 시간 단축 목적)

### 문제점
1. **Docker Compose 상시 가동**: 트래픽 없어도 리소스 점유
2. **스케일링 불가**: 단일 서버, 수평 확장 없음
3. **A/B 테스트 없음**: 새 모델 배포 시 전량 교체

### 유지할 패턴
- **LGBM 실시간 추론**: 매 요청마다 추론 (~5ms), 충분히 빠름
- **태스크별 출력 정규화**: binary→sigmoid, multiclass→softmax
- **지식 증류 파이프라인**: PLE(teacher) → LGBM(student)

---

## AWS 설계 — 핵심 원칙

### 실시간 추론 전략

**매 요청마다 LGBM 추론**합니다. 사전 계산 + 캐시 방식이 아닙니다.

```
요청 (user_id, context)
    ↓
① 피처 조회 (메모리 또는 DynamoDB)    ~0.01ms 또는 ~5ms
    ↓
② 실시간 컨텍스트 결합                ~0.1ms
   (현재 시간, 세션 정보, 요청 컨텍스트)
    ↓
③ LGBM 멀티태스크 추론               ~5ms
    ↓
④ 출력 정규화 + 응답                 ~0.1ms
    ↓
총: ~5-10ms
```

이 구조를 선택한 이유:
- LGBM 추론은 ~5ms로 충분히 빠름 (추천 업계 기준 50-200ms 대비 최상위)
- 실시간 컨텍스트 반영 가능 (시간대, 세션 행동 등)
- 사전 계산 방식의 stale 문제 없음

---

## 규모별 자동 전환 아키텍처

### 전체 구조

```yaml
# configs/serving.yaml
serving:
  mode: auto              # auto | lambda | ecs
  auto_threshold: 100000000  # 월 1억 건 이상이면 ECS 전환 알림

  feature_store: auto     # auto | memory | dynamodb
  auto_feature_threshold: 5000000  # 유저 500만 이상이면 DynamoDB 전환

  lambda:
    memory_mb: 1024
    timeout: 30
  ecs:
    cpu: 4096
    memory: 16384
    min_tasks: 2
    embedded_stores:       # 대규모 시 활성화
      redis: false
      rocksdb: false
      lancedb: false
```

### 3단계 확장 경로

```
┌─────────────────────────────────────────────────────────────────────┐
│                        규모별 서빙 아키텍처                           │
│                                                                     │
│  [1단계] 포트폴리오/소규모 (~100만 건/월)                             │
│  ┌──────────────────────────────────────┐                           │
│  │ API Gateway → Lambda                 │                           │
│  │   ├── 피처: 메모리 로드 (S3 Parquet)  │  비용: $0-1/월            │
│  │   ├── 모델: LGBM (메모리 내장)        │  지연: ~5ms              │
│  │   └── 응답                            │  서버리스: 완전           │
│  └──────────────────────────────────────┘                           │
│                         │                                           │
│                    월 1억 건 돌파                                     │
│                         ▼                                           │
│  [2단계] 중규모 (~1-3억 건/월)                                       │
│  ┌──────────────────────────────────────┐                           │
│  │ API Gateway → Lambda                 │                           │
│  │   ├── 피처: DynamoDB 조회             │  비용: $100-400/월        │
│  │   ├── 모델: LGBM (메모리 내장)        │  지연: ~10ms             │
│  │   └── 응답                            │  서버리스: 완전           │
│  └──────────────────────────────────────┘                           │
│                         │                                           │
│                    비용 역전 지점                                     │
│                         ▼                                           │
│  [3단계] 대규모 (3억+ 건/월, 금융사 앱 수준)                          │
│  ┌──────────────────────────────────────┐                           │
│  │ ALB → ECS Fargate                    │                           │
│  │   ├── Redis (실시간 피처 캐시)         │  비용: ~$360/월           │
│  │   ├── RocksDB (전체 유저 피처)        │  지연: ~5-8ms            │
│  │   ├── LanceDB (벡터 검색)             │  서버리스: 아님           │
│  │   ├── LGBM 추론                      │                           │
│  │   └── 실시간 스트리밍 피처 수신        │                           │
│  │       (Kinesis → Redis 갱신)          │                           │
│  └──────────────────────────────────────┘                           │
└─────────────────────────────────────────────────────────────────────┘
```

### 추론 코드는 동일 (환경만 다름)

```python
# core/serving/predict.py — Lambda든 ECS든 이 코드 공유
class RecommendationService:
    def __init__(self, config):
        self.feature_store = FeatureStoreFactory.create(config)  # memory | dynamodb | redis
        self.model = load_lgbm_model(config.model_path)
        self.tasks = config.tasks

    def predict(self, user_id: str, context: dict) -> dict:
        # ① 피처 조회
        features = self.feature_store.get(user_id)

        # ② 실시간 컨텍스트 결합
        features = self._enrich_with_context(features, context)

        # ③ LGBM 멀티태스크 추론
        predictions = {}
        for task in self.tasks:
            raw = self.model[task.name].predict(features)
            predictions[task.name] = self._normalize(raw, task.type)

        return predictions

    def _normalize(self, raw, task_type):
        if task_type == "binary":
            return float(raw[0][1])              # 양성 클래스 확률
        elif task_type == "multiclass":
            return raw[0].tolist()               # 클래스별 확률
        else:
            return float(raw[0])                 # 회귀값


# Lambda 핸들러
def lambda_handler(event, context):
    service = RecommendationService(config)       # cold start 시 초기화
    return service.predict(event["user_id"], event.get("context", {}))

# ECS FastAPI
@app.post("/v1/recommend")
async def recommend(request: RecommendRequest):
    return service.predict(request.user_id, request.context)
```

### 피처 스토어 추상화

```python
# core/serving/feature_store.py
class FeatureStoreFactory:
    @staticmethod
    def create(config) -> AbstractFeatureStore:
        mode = config.feature_store
        if mode == "memory":
            return MemoryFeatureStore(config.feature_path)   # S3 Parquet → dict
        elif mode == "dynamodb":
            return DynamoDBFeatureStore(config.dynamodb_table)
        elif mode == "redis":
            return RedisFeatureStore(config.redis_endpoint)
        elif mode == "auto":
            user_count = estimate_user_count(config)
            if user_count < 5_000_000:
                return MemoryFeatureStore(config.feature_path)
            else:
                return DynamoDBFeatureStore(config.dynamodb_table)


class MemoryFeatureStore(AbstractFeatureStore):
    """S3의 피처 Parquet을 메모리에 dict로 로드. 유저 500만 이하에 적합."""

    def __init__(self, path: str):
        import pandas as pd
        df = pd.read_parquet(path)
        self._store = {row["user_id"]: row.drop("user_id").values for _, row in df.iterrows()}

    def get(self, user_id: str) -> np.ndarray:
        return self._store.get(user_id)


class DynamoDBFeatureStore(AbstractFeatureStore):
    """DynamoDB에서 유저 피처 조회. 대규모 유저에 적합. ~3-5ms."""
    ...
```

---

## A/B 테스트

Lambda와 ECS 모두 API Gateway의 **스테이지 변수 + 가중 라우팅**으로 A/B 테스트를 지원합니다.

```yaml
# configs/serving/ab_test.yaml
ab_test:
  enabled: true
  variants:
    - name: control
      model: s3://bucket/models/lgbm-v1/
      weight: 90              # 트래픽 90%
    - name: treatment
      model: s3://bucket/models/lgbm-v2/
      weight: 10              # 트래픽 10%

  evaluation:
    primary_metric: click_through_rate
    secondary: [conversion_rate, revenue_per_user]
    min_sample_size: 10000
    significance_level: 0.05

  auto_promote:
    enabled: true
    min_improvement: 0.02     # 2% 이상 개선 시 자동 전환
```

```
API Gateway
    ├── 90% → Lambda/ECS (Model v1) → CloudWatch 메트릭 기록
    └── 10% → Lambda/ECS (Model v2) → CloudWatch 메트릭 기록
    ↓
Lambda (일일 통계 분석)
    ├── t-test / 베이지안 A/B 검정
    └── 유의미 → 자동 전환 or 알림
```

### 카나리 배포

```
새 모델 배포 시:
  1. 새 Lambda 버전 배포 (또는 ECS Task Definition)
  2. 5% 트래픽 → 새 버전
  3. CloudWatch 메트릭 5분 간격 모니터링
  4. 이상 없으면: 25% → 50% → 100%
  5. 이상 발견: 즉시 이전 버전으로 롤백
```

---

## 출력 정규화

```python
# core/serving/normalizer.py
class OutputNormalizer:
    """태스크 타입에 따라 모델 출력을 정규화합니다."""

    STRATEGIES = {
        "binary": lambda logits: sigmoid(logits),           # [0, 1] 확률
        "multiclass": lambda logits: softmax(logits),       # [0, 1] 클래스별
        "regression": lambda logits: logits,                 # raw value
        "contrastive": lambda logits: normalize(logits),     # unit vector
        "ranking": lambda logits: logits,                    # raw score
    }
```

---

## 3단계 확장 시 추가 구성 (대규모)

월 3억 건 이상의 금융사 앱 수준으로 확장할 때는 ECS + 임베디드 스토어를 활성화합니다:

```
[스트리밍 파이프라인 — 실시간 피처 갱신]

이벤트 소스 (클릭, 구매, 세션 등)
    ↓
Kinesis Data Streams
    ↓
Lambda (피처 업데이트 계산)
    ↓
Redis (ECS 컨테이너 내) 실시간 갱신
    ├── 최근 30분 클릭 수
    ├── 현재 세션 관심 카테고리
    └── 실시간 행동 시그널
```

```
[ECS Fargate 컨테이너 구성]

FastAPI
├── Redis    (핫 유저 피처 캐시, 실시간 스트리밍 수신)    <0.5ms
├── RocksDB  (전체 유저 배치 피처, 로컬 디스크)           <3ms
├── LanceDB  (임베딩 벡터 인덱스, 콜드스타트 유사 유저)   <5ms
└── LGBM     (실시간 추론)                               <5ms

총: <8ms (네트워크 홉 0, 모든 데이터 로컬)
```

이 구성은 **config에서 `mode: ecs`로 전환**할 때만 활성화되며, 추론 코드(`core/serving/predict.py`)는 변경 없이 그대로 사용합니다.

---

## 현재 vs AWS — 서빙 비교

| 항목 | 현재 (On-Prem) | AWS (설계) | 변경 이유 |
|------|---------------|-----------|----------|
| 기본 서빙 | Docker Compose (상시) | Lambda (서버리스) | 유휴 비용 $0 |
| 대규모 서빙 | 해당 없음 | ECS + 임베디드 스토어 | config 전환으로 확장 |
| 추론 방식 | 매 요청 추론 | 매 요청 추론 (동일) | LGBM ~5ms 충분 |
| 피처 조회 | 로컬 파일 | 메모리 → DynamoDB → Redis | 규모별 자동 선택 |
| A/B 테스트 | 없음 | API Gateway 가중 라우팅 | 안전한 모델 교체 |
| 카나리 배포 | 없음 | 점진적 트래픽 전환 | 롤백 안전성 |

### 운영/감사 에이전트 연계

서빙 파이프라인의 주요 지표(p50/p95 latency, filter 통과율, A/B variant CTR)는 OpsAgent의 CP5(서빙 헬스), CP6(추천 응답), CP7(A/B 테스트) 체크포인트에서 자동 모니터링된다. SLA 초과나 A/B 유의미한 결과 발생 시 정형 리포트로 담당자에게 전달되며, AuditAgent가 추천 결과의 공정성 및 집중도를 감사한다.

상세 설계: `docs/design/11_ops_audit_agent.md`
| 스케일링 | 불가 | Lambda 자동 / ECS Auto-scaling | 트래픽 대응 |

### 온프레미스 서빙

온프레미스에서는 Lambda/ECS 대신 Docker 컨테이너로 서빙하며, 추천사유 L2a는 Exaone 3.5 7.8B로 배치 생성한다. A/B 테스트는 동일한 해시 기반 분배를 사용하되, CloudWatch 대신 로컬 메트릭 파일로 수집한다.
