# AWS 쪽 구축 계획서 (Build Plan)

**대상**: `docs/aws_work_plan.md` 의 Must (M1~M12) 중심 상세 실행 계획.
**기준 시점**: 2026-04-28
**전제**: AWS 는 온프렘의 클라우드 확장 버전. 온프렘 → AWS 이식 시 AWS 의 **config-중심 / 데코레이터 Registry 패턴** 으로 재구성.

> **진행 상태 배너 (2026-04-28)** — 본 문서는 Sprint 계획 단계에서 작성된 실행 명세로, 아래 항목은 **이미 완료**되어 merge 된 상태입니다:
> - **Phase 1 Must M1~M12** — 전량 완료
> - **Phase 2 Should S1~S15** — 전량 완료 (S5 는 SageMaker Experiments 네이티브, S6 는 DuckDB httpfs 로 재설계 — CLAUDE.md §1.14/§1.15)
> - **Phase 3 Could** — C1/C3/C4/C5 완료, C2 는 Won't (SageMaker managed orchestration 네이티브)
> - **PR #1~#3 PromotionGate Live Wiring (2026-04-21)** — `core/compliance/metadata_aggregator.py` + 6 evidence source + audit trail + SageMakerComplianceTracker artifact 연결. `compliance.promotion_gate.enabled: true` 가 pipeline.yaml 기본값으로 전환됨 (commit `51149f3`/`9426162`/`ec8587b`).
> - **Phase 0 schema audit + Mamba precompute (2026-04-26~28, local main, 미푸시)** — Phase 0 invariant 위반 6종 차단 + Mamba GPU precompute 분리. 상세는 §11.
> - **현재 테스트**: 620/620 PASS
>
> 실시간 진행 현황과 최근 3~7일 변경사항은 `docs/aws_work_plan.md` 상단 "진행 현황" 블록 및 `docs/pipeline_comparison_matrix.md §5.10` 을 참조하십시오. 본 문서는 **역사적 Build Plan** 으로 유지되며 Sprint 설계 원칙을 추적하고자 할 때 사용하십시오.
>
> **FRIA 모듈 파일 경로 주의**: M7 은 **EU AI Act Art. 9** 기반이며 구현 위치는 `core/monitoring/fria_evaluator.py::FRIAEvaluator` (5-차원) 입니다. 한국 AI기본법 §35 FRIA 는 별도 class `core/compliance/fria_assessment.py::KoreanFRIAAssessor` (7-차원, 5-년 retention) 로 구현되어 있으며 법적 기반이 달라 저장도 분리합니다 (CLAUDE.md §1.11). 본 문서 M7 섹션에서 `fria_evaluator.py` 로 표기된 부분은 EU 경로이며, 한국 경로 구현은 별도 진행되었습니다.

**관련 문서**:
- `docs/aws_work_plan.md` — Must/Should/Could 항목 리스트 + 법적 근거 (최신)
- `docs/pipeline_comparison_matrix.md` — 4-레이어 gap 분석 + §5.10 운영 전환 체크리스트
- `docs/onprem_work_plan.md` — 반대 방향 계획

---

## 0. 아키텍처 결정

### 0.1 모듈 배치 원칙

| 역할 | 배치 위치 |
|---|---|
| 규제/권리 관리 | `core/compliance/` (기존 확장) |
| 실운영 라이프사이클 (queue, switch, universe) | `core/serving/` 또는 `core/recommendation/` |
| 감사 로그 확장 | `core/monitoring/` 또는 `core/recommendation/audit_archiver.py` |
| 추론 reason/LLM 관련 | `core/recommendation/reason/` |

### 0.2 공통 패턴 — 강제 준수

모든 신규/이식 모듈은:

1. **Dataclass config**: `@dataclass` + `from_yaml` + 검증 로직
2. **Registry 데코레이터**: 가능한 경우 `@SomethingRegistry.register("name")`
3. **Storage backend 추상화**: 인메모리 테스트 + S3/DynamoDB 프로덕션 swap 가능
4. **HMAC/Audit 통합**: 규제 관련 모든 기록은 `AuditLogger.log_operation()` 경로로 진입
5. **단일 진입점**: 설정은 `pipeline.yaml` 에 블록으로, HP 는 `config_builder.py` 에서 주입

### 0.3 신규 subdirectory 결정

| 신규 위치 | 이유 |
|---|---|
| `core/compliance/rights/` | M4, M5, M6 가 "권리 관리" 라이프사이클 공유 → 서브모듈화 |
| `core/serving/review/` | M1 Human Review Queue 는 serving 측 큐 |
| `core/recommendation/universe/` | M10 Item Universe 는 추천 대상 풀 관리 (신규) |

나머지는 기존 경로에 파일 단위로 추가.

---

## 1. Shared Foundation (Sprint 0, 0.5~1주)

**M1~M12 모두가 의존하는 공통 요소.** 이것부터 먼저.

### F1. `ComplianceRequest` + `ComplianceEvent` base types

**위치**: `core/compliance/types.py` (신규)

```python
@dataclass
class ComplianceRequest:
    request_id: str        # UUID
    user_id: str
    request_type: str      # "consent_update" | "opt_out" | "profiling_query" | ...
    submitted_at: datetime
    sla_deadline: datetime
    status: str            # "pending" | "processed" | "expired"
    metadata: Dict[str, Any]

@dataclass
class ComplianceEvent:
    event_id: str
    user_id: str
    event_type: str
    timestamp: datetime
    payload: Dict[str, Any]
```

- `from_yaml` 불필요 (런타임 생성만)
- JSON 직렬화/역직렬화 지원 (S3 저장)

### F2. `ComplianceStore` 추상 인터페이스

**위치**: `core/compliance/store.py` (기존 `audit_store.py` 확장 or 별도 파일)

```python
class ComplianceStore(ABC):
    @abstractmethod
    def put_request(self, req: ComplianceRequest) -> None: ...
    @abstractmethod
    def get_request(self, request_id: str) -> Optional[ComplianceRequest]: ...
    @abstractmethod
    def list_pending(self, user_id: Optional[str] = None) -> List[ComplianceRequest]: ...
    @abstractmethod
    def put_event(self, evt: ComplianceEvent) -> None: ...
    @abstractmethod
    def query_events(self, user_id: str, since: datetime) -> List[ComplianceEvent]: ...

class DynamoDBComplianceStore(ComplianceStore): ...     # 프로덕션
class InMemoryComplianceStore(ComplianceStore): ...     # 테스트
class S3ParquetComplianceStore(ComplianceStore): ...    # 배치/아카이브
```

### F3. `SLATracker` 공통 클래스

**위치**: `core/compliance/sla_tracker.py` (신규)

- `check_deadline()`, `list_approaching_deadline()`, `mark_processed()`
- M6 (Explanation SLA) 의 기반이 되지만 다른 SLA 에도 재사용

### F4. `pipeline.yaml` 내 `compliance:` 블록 확장

```yaml
compliance:
  store:
    backend: "dynamodb"   # | "s3_parquet" | "in_memory"
    dynamodb_table: "ple-compliance-events"
    s3_bucket: "aiops-ple-financial"
    s3_prefix: "compliance"
  retention:
    default_days: 1825    # 5년 (AI기본법)
    rights_request_days: 2555  # 7년 (개보법)
  sla:
    explanation_response_days: 10   # 개보법 시행령 §44의2~4
    opt_out_response_days: 30
```

### F5. Unit test infrastructure

**위치**: `tests/test_compliance_foundation.py` (신규)

- `InMemoryComplianceStore` 기반 round-trip 테스트
- SLATracker 데드라인 계산 테스트
- 공통 fixture 정의

**Sprint 0 예상**: 3~5일 (혼자 기준).

---

## 2. Must 개별 구축 Spec

### M1. Human Review Queue

**위치**: `core/serving/review/human_review_queue.py`

**의존**: F1 (ComplianceRequest), F2 (store)

**핵심 설계**:

```python
@dataclass
class ReviewConfig:
    tier_1_sample_rate: float = 0.05  # 5% 사후 샘플
    tier_2_review_required: bool = True  # 100% 상담원 확인
    tier_3_human_fallback: bool = True   # HumanFallbackRouter 필수
    queue_backend: str = "dynamodb"

@dataclass
class ReviewItem:
    review_id: str
    user_id: str
    recommendation_id: str
    tier: int                 # 1/2/3
    state: str                # "pending"/"approved"/"rejected"/"modified"
    created_at: datetime
    reviewer_id: Optional[str]
    disposition: Optional[str]
    modifications: Optional[Dict]

class HumanReviewQueue:
    def enqueue(self, rec: RecommendationResult, tier: int) -> ReviewItem: ...
    def dequeue(self, reviewer_id: str, tier: int) -> Optional[ReviewItem]: ...
    def approve(self, review_id: str, reviewer_id: str) -> None: ...
    def reject(self, review_id: str, reviewer_id: str, reason: str) -> None: ...
    def modify(self, review_id: str, reviewer_id: str, new_rec: dict) -> None: ...
```

**통합**: `core/serving/predict.py` 에 `if tier > 1: review_queue.enqueue(...)` 주입.

**테스트**: `InMemoryComplianceStore` 로 enqueue/dequeue/dispose 사이클.

**공수**: 2일.

---

### M2. Kill Switch 로직 확장

**위치**: `core/serving/kill_switch.py` (기존 파일 확장)

**의존**: AWS 기존 DynamoDB 기반 `kill_switch.py` — 온프렘의 로직만 이식.

**핵심 설계**:

- 온프렘 `src/recommendation/kill_switch.py` 의 3단계 (global / task / cluster) 상태 머신 이식
- AWS 는 이미 DynamoDB 백엔드 보유 → 스키마 확장만 필요
- CloudWatch 감사 로그 연동 유지

**공수**: 1일.

---

### M3. Marketing Consent

**위치**: `core/compliance/consent_manager.py` (기존 확장)

**의존**: F1, F2, F4 (consent 블록)

**핵심 설계**:

- 4채널 (SMS / EMAIL / APP_PUSH / PHONE / MAIL) 각각 독립 동의
- 야간 발송 제한 (21:00~08:00 KST, 금소법)
- DNC 레지스트리 (전화 수신거부)
- 동의 만료 자동 취소
- `pipeline.yaml::compliance.consent.channels` 로 채널 정의

```python
@dataclass
class ConsentConfig:
    channels: List[str]
    night_hours: Tuple[int, int] = (21, 8)  # (start, end_next_day) KST
    default_retention_days: int = 365

@dataclass
class ConsentRecord:
    user_id: str
    channel: str
    state: str                 # "granted"/"revoked"/"expired"
    granted_at: datetime
    expires_at: Optional[datetime]
    legal_basis: str           # "개보법 §22" | "정통망 §50" | ...

class ConsentManager:
    def is_consented(self, user_id: str, channel: str, at: datetime) -> bool: ...
    def is_night_hour(self, at: datetime) -> bool: ...
    def grant(self, user_id: str, channel: str, expires_at: Optional[datetime], legal_basis: str) -> None: ...
    def revoke(self, user_id: str, channel: str, reason: str) -> None: ...
```

**통합**: `core/serving/predict.py` 에서 현재 `consent_manager.is_consented(user_id)` 호출을 **채널별 체크** 로 확장.

**공수**: 1.5일.

---

### M4. AI Decision Opt-out

**위치**: `core/compliance/rights/opt_out.py` (신규 subdirectory)

**의존**: F1, F2, F3 (SLA)

**핵심 설계**:

- 개보법 §37의2 "자동화된 결정의 거부권" + 설명요구권
- AI기본법 §31 (AI 거부권)
- Opt-out 상태 → fallback_type 매핑: `rule_based` / `human_review` / `disable`
- 설명요구권: 10일 SLA (개보법 시행령)

```python
@dataclass
class OptOutConfig:
    default_fallback: str = "rule_based"  # | "human_review" | "disable"
    explanation_sla_days: int = 10

class OptOutManager:
    def opt_out(self, user_id: str, fallback_type: str, reason: str) -> ComplianceRequest: ...
    def is_opted_out(self, user_id: str) -> bool: ...
    def request_explanation(self, user_id: str, recommendation_id: str) -> ComplianceRequest: ...
    def mark_explanation_provided(self, request_id: str, explanation: str) -> None: ...
```

**통합**: `predict.py` 에서 `if opt_out_manager.is_opted_out(user_id): return fallback(...)` 주입.

**공수**: 1일.

---

### M5. Profiling Rights Manager

**위치**: `core/compliance/rights/profiling.py` (신규)

**의존**: F1, F2, F3

**핵심 설계**:

- 신정법 §36의2 "프로파일링에 대한 권리"
- 열람권 / 정정권 / 삭제권 3종
- 각 권리 요청은 `ComplianceRequest` 로 생성 + SLA 추적

```python
class ProfilingRightsManager:
    def request_access(self, user_id: str) -> ComplianceRequest: ...
    def request_correction(self, user_id: str, field: str, new_value: Any) -> ComplianceRequest: ...
    def request_deletion(self, user_id: str, scope: str) -> ComplianceRequest: ...
    def fulfill_access(self, request_id: str) -> Dict[str, Any]: ...
    # 정정/삭제도 유사
```

**기존 `core/compliance/profiling_rights.py` 가 있으면 확장.**

**공수**: 1일.

---

### M6. Explanation SLA Tracker

**위치**: `core/compliance/rights/explanation_sla.py` (신규)

**의존**: F3 (SLATracker)

**핵심 설계**:

- 개보법 시행령 §44의2~4 10일 SLA 강제
- 요청 ↔ 답변 매핑 + 초과 경고
- 주간/월간 SLA 준수율 리포트 생성

```python
class ExplanationSLATracker(SLATracker):
    SLA_DAYS = 10
    def track_request(self, req: ComplianceRequest) -> None: ...
    def get_sla_breaches(self, since: datetime) -> List[ComplianceRequest]: ...
    def generate_monthly_report(self) -> Dict[str, Any]: ...
```

**공수**: 1일.

---

### M7. FRIA Evaluator

**위치**: `core/compliance/fria_evaluator.py` (기존 있으면 확장, 없으면 신규)

**의존**: F2 (store), 설정 블록

**핵심 설계**:

- AI기본법 §35②③ (국가기관 강화)
- 시행령 §27 의 7개 평가 차원
- 5년 보존 + retention_expiry 자동 산출
- 점수 집계 → `UNACCEPTABLE` / `HIGH` / `LIMITED` / `MINIMAL` 분류
- Operator type = "국가기관등 (우정사업본부)" 메타데이터

```python
@dataclass
class FRIAResult:
    assessment_id: str
    operator_type: str        # "국가기관등" | "민간"
    assessed_at: datetime
    retention_expiry: datetime  # +5년
    dimensions: Dict[str, float]   # 7차원 점수
    total_score: float
    risk_category: str
    mitigation_plan: Optional[str]

class FRIAEvaluator:
    DIMENSIONS = ("data_sensitivity", "automation_level", "scope_of_impact",
                  "model_complexity", "external_dependency", "fairness_risk",
                  "explainability_gap")
    def evaluate(self, model_version: str, context: Dict) -> FRIAResult: ...
    def archive(self, result: FRIAResult) -> None: ...  # 5년 WORM
    def list_expiring(self, within_days: int) -> List[FRIAResult]: ...
```

**통합**: Champion-Challenger 승격 전 FRIA 게이트 추가 (`ModelCompetition._decide_promotion`).

**공수**: 2일.

---

### M8. 36-항목 Compliance Registry

**위치**: `core/compliance/regulatory_checker.py` (기존 확장)

**의존**: F2

**핵심 설계**:

- A-group 18개 + GAP 18개 항목
- 각 항목 = (항목 ID, 법적 근거, 검증 방법, 현재 상태)
- 검증 방법: `file_exists` / `config_range` / `module_exists` / `endpoint_alive` / `custom_check`
- 분기별 리포트 생성

```python
@dataclass
class ComplianceItem:
    item_id: str            # e.g. "A-05", "GAP-12"
    group: str              # "A" | "GAP"
    description: str
    legal_basis: List[str]  # ["AI기본법 §33", "금감원 원칙 5"]
    check_type: str
    check_params: Dict[str, Any]
    status: str             # "compliant" | "non_compliant" | "not_applicable"
    last_checked_at: datetime
    evidence_path: Optional[str]

class ComplianceRegistry:
    ITEMS = [...]  # 36 항목 정의
    def check_all(self) -> Dict[str, ComplianceItem]: ...
    def check_single(self, item_id: str) -> ComplianceItem: ...
    def generate_quarterly_report(self, quarter: str) -> str: ...  # markdown
```

**공수**: 2~3일 (36항목 정의가 주요 작업).

---

### M9. AI Risk Classifier (금감원 6-차원 RMF)

**위치**: `core/compliance/ai_risk_classifier.py` (신규)

**의존**: F2

**핵심 설계**:

- 금감원 AI RMF 6차원: 데이터민감도 / 자동화수준 / 영향범위 / 모델복잡도 / 외부의존도 / 공정성리스크
- 가중 합산 → 고/중/저 3등급
- 이전 등급 대비 변경 감지 + 이력 추적

```python
@dataclass
class AIRiskAssessment:
    model_version: str
    assessed_at: datetime
    dimensions: Dict[str, float]   # 6차원 점수 (0.0~1.0)
    weights: Dict[str, float]
    total_score: float
    grade: str                      # "high" | "medium" | "low"
    prev_grade: Optional[str]
    grade_change: bool

class AIRiskClassifier:
    DEFAULT_WEIGHTS = {
        "data_sensitivity": 0.25,
        "automation_level": 0.20,
        "scope_of_impact": 0.20,
        "model_complexity": 0.15,
        "external_dependency": 0.10,
        "fairness_risk": 0.10,
    }
    def classify(self, model_version: str, context: Dict) -> AIRiskAssessment: ...
    def archive(self, assessment: AIRiskAssessment) -> None: ...
```

**통합**: 모델 승격 시 `classify` 호출 → 결과를 `log_model_promotion` metadata 에 포함.

**공수**: 1.5일.

---

### M10. Dynamic Item Universe Loader

**위치**: `core/recommendation/universe/dynamic_loader.py` (신규 subdirectory)

**의존**: S3 Parquet 조회

**핵심 설계**:

- G3 (캠페인) + G6 (상품) 테이블을 배치 시점에 Parquet 쿼리 (온프렘 DuckDB 대응, AWS 는 S3 Parquet + DuckDB 쿼리 엔진)
- 캠페인 상태 enum 라이프사이클: 기획 → 승인 → 수행중 → 완료 → 취소
- "오늘 승인난 캠페인" 자동 포함

```python
class CampaignStatus(str, Enum):
    PLANNING = "planning"
    APPROVED = "approved"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELED = "canceled"

@dataclass
class ItemUniverseConfig:
    campaign_parquet: str   # s3://.../campaigns.parquet
    product_parquet: str    # s3://.../products.parquet
    active_statuses: List[CampaignStatus] = (CampaignStatus.APPROVED, CampaignStatus.RUNNING)

class DynamicItemUniverseLoader:
    def load(self, as_of: datetime) -> List[Item]: ...
    def get_active_campaigns(self, as_of: datetime) -> List[Campaign]: ...
    def get_product_pool(self) -> List[Product]: ...
```

**통합**: `core/recommendation/pipeline.py` 에서 `candidate_items` 를 caller 주입 대신 `loader.load()` 로 대체.

**공수**: 2일.

---

### M11. Audit Archive 확장 컬럼

**위치**: `core/recommendation/audit_archiver.py` (기존 확장)

**의존**: 없음 (스키마만 확장)

**핵심 설계**:

현재 AWS 스키마:
```
user_id_hash, rec_id, task, score, layer, feature_top5_names, feature_top5_values,
verdict, timestamp
```

추가 컬럼:
```
thinking_trace        VARCHAR       # Agent reasoning chain (L1/L2a)
hallucination_flags   VARCHAR[]     # Safety gate detected patterns
tools_used            VARCHAR[]     # Agent invoked tools
critique_verdict      VARCHAR       # SelfChecker final call
agent_tier            INTEGER       # 1/2/3 tier
```

Parquet schema evolution: 기존 행 호환 (nullable).

**공수**: 0.5일.

---

### M12. LLM Generation Marker 자동 삽입

**위치**: `core/recommendation/reason/template_engine.py` (확장)

**의존**: 없음

**핵심 설계**:

- AI기본법 §31 (AI 생성 표시 의무), §34 (고지 의무)
- 모든 L2a rewrite 결과에 marker 자동 추가
- Config 로 marker 문구/위치 제어

```python
LLM_GENERATION_MARKER = "※ 본 추천 사유는 AI가 생성하였습니다. (AI기본법 §31)"

def apply_marker(reason_text: str, config: ReasonConfig) -> str:
    if not config.add_ai_marker:
        return reason_text
    return f"{reason_text}\n\n{config.marker_text or LLM_GENERATION_MARKER}"
```

**통합**: `llm_provider.py` 의 output post-processing.

**공수**: 0.5일.

---

## 3. Sprint 일정

### Sprint 0 (Week 1 전반, 0.5~1주) — Foundation

**Deliverables**:
- F1~F5 (types, store, SLATracker, config block, test infra)
- `DynamoDBComplianceStore` 실구현 + `InMemoryComplianceStore` 테스트

**Exit criteria**:
- `tests/test_compliance_foundation.py` 통과
- `pipeline.yaml` 에 `compliance:` 블록 추가 + 읽기 경로 검증

### Sprint 1 (Week 1 후반 ~ Week 2, 약 1주) — Rights & Consent

**Deliverables**:
- M3 (Marketing Consent 4-채널)
- M4 (AI Decision Opt-out)
- M5 (Profiling Rights Manager)
- M6 (Explanation SLA Tracker)

**Exit criteria**:
- 각 모듈 unit test 통과
- `predict.py` 에 opt-out / consent 채널별 체크 통합
- SLA 브리치 탐지 smoke test

### Sprint 2 (Week 2 후반 ~ Week 3, 약 1주) — Registry & Assessments

**Deliverables**:
- M7 (FRIA Evaluator)
- M8 (36-항목 Compliance Registry)
- M9 (AI Risk Classifier 6-차원)
- Champion-Challenger 승격에 FRIA + Risk 게이트 통합

**Exit criteria**:
- 36개 항목 정의 완료 + 최초 분기 리포트 생성
- FRIA 5년 보존 라이프사이클 작동
- Risk classification 이 `log_model_promotion` metadata 에 기록

### Sprint 3 (Week 3 후반, 약 3~4일) — Serving & Audit

**Deliverables**:
- M1 (Human Review Queue)
- M2 (Kill Switch 확장)
- M10 (Dynamic Item Universe Loader)
- M11 (Audit Archive 확장 컬럼)
- M12 (LLM Generation Marker)

**Exit criteria**:
- Serving 파이프라인 end-to-end smoke test (tier-based review queue, dynamic universe, LLM marker)
- Audit Archive 새 컬럼이 실제 serving 로그에 기록

### Sprint 4 (Week 4 초반, 2~3일) — 통합 + 문서

**Deliverables**:
- `core/serving/predict.py` 11단계 pipeline 에 신규 체크 순서 재설계
- `paper/typst/paper2.typ` 규제 섹션 업데이트 (이식된 모듈들 명시)
- `docs/pipeline_comparison_matrix.md` 갱신 (양쪽 구현 으로 이동)
- `CLAUDE.md` 에 신규 compliance 모듈 정책 추가

**Exit criteria**:
- 1M 샘플 배치 추론이 신규 compliance 체크 포함한 채로 완주
- Paper 2 v2 초안에 실제 코드 경로 인용 가능

**Phase 1 (Must) 총 기간**: **2.5~3.5주** (Sprint 0 + 1 + 2 + 3 + 4).

---

## 4. 주요 통합 지점

### 4.1 `core/serving/predict.py` 11단계 pipeline 재설계

현재 순서 (존재 순):
1. Kill switch
2. Consent
3. Opt-out
4. Suitability
5. A/B
6. Feature
7. LGBM predict
8. Normalize
9. Calibrate
10. Pipeline
11. Audit

이식 후 순서 (제안):
1. **Kill switch** (global / task / cluster) ← M2 확장
2. **Opt-out** (if opted_out → fallback) ← M4
3. **Consent** (채널별) ← M3
4. **SLA check** (if explanation pending) ← M6
5. **Profiling rights** (if access/correction/deletion pending) ← M5
6. **AI risk classification** (현재 모델 RMF 등급) ← M9
7. Suitability / A/B (기존)
8. Feature / LGBM predict / Normalize / Calibrate (기존)
9. **Dynamic item universe** (후보 상품 풀) ← M10
10. Pipeline (scoring, ranking, constraints)
11. **Human review triage** (tier 결정 + queue) ← M1
12. Reason generation (L1/L2a)
13. **LLM marker 자동 삽입** ← M12
14. **Audit log** (확장 스키마 포함) ← M11

### 4.2 `core/evaluation/model_competition.py` 승격 게이트 확장

현재 `_decide_promotion` 순서:
1. `--force-promote` → 무조건 승격
2. champion 없음 → bootstrap
3. fidelity_summary.failed > 0 → reject
4. `ModelCompetition.evaluate()` → promotion_approved

이식 후 추가:
5. **FRIA 평가** ← M7 (risk_category == UNACCEPTABLE → reject)
6. **AI Risk Classifier** ← M9 (grade 변경 감지, 'high' 로 상승 시 추가 승인 필요)

모든 결정은 기존 `log_model_promotion` 에 FRIA + Risk metadata 포함해서 기록.

### 4.3 `AuditLogger.log_operation` 호출 지점 증가

새로 추가되는 감사 entry types:
- `compliance_request:consent_grant` / `consent_revoke`
- `compliance_request:opt_out` / `opt_out_revoke`
- `compliance_request:profiling_access` / `correction` / `deletion`
- `compliance_request:explanation`
- `fria_assessment:{UNACCEPTABLE/HIGH/LIMITED/MINIMAL}`
- `ai_risk_assessment:{high/medium/low}`
- `human_review:enqueue` / `approve` / `reject` / `modify`

---

## 5. 의존성 그래프

```
[F1 types]
  ├─► [F2 store]
  ├─► [F3 SLATracker]
  │     └─► [M6 Explanation SLA]
  ├─► [M4 Opt-out] ─► uses F3
  ├─► [M5 Profiling Rights] ─► uses F3
  └─► [M3 Consent]

[F2 store]
  ├─► [M7 FRIA] ─► integrate into ModelCompetition
  ├─► [M8 Compliance Registry]
  ├─► [M9 AI Risk Classifier] ─► integrate into ModelCompetition

[M1 Human Review Queue] — independent (DynamoDB only)
[M2 Kill Switch] — extend existing
[M10 Item Universe] — independent (S3 Parquet only)
[M11 Audit schema] — independent
[M12 LLM marker] — independent

Integration:
  predict.py 11-step pipeline ← M1, M2, M3, M4, M5, M6, M9, M10, M12, M11
  ModelCompetition ← M7, M9
  Paper 2 v2 섹션 ← 모든 M1~M12
```

---

## 6. 리스크 및 검증

### 6.1 리스크 레지스터

| # | 리스크 | 영향 | 완화 |
|---|---|---|---|
| R1 | 36-항목 정의 시간 | Sprint 2 지연 | 온프렘 `regulatory_compliance_checker.py` 의 항목 정의 직접 복사 + 번역만 (법적 근거는 한국법이라 영문화 최소) |
| R2 | DynamoDB 스키마 충돌 | 기존 데이터 영향 | 신규 테이블 별도 생성 (`ple-compliance-*`) + 기존 `ple-predictions` 는 불변 |
| R3 | predict.py 11→14단계 re-sequence | 기존 E2E 테스트 깨짐 | 단계별 feature flag 로 점진 활성화. default=off → 검증 후 on |
| R4 | FRIA 5년 보존이 S3 Object Lock 이력과 충돌 | 저장 비용 증가 | FRIA 전용 S3 prefix + `FRIA_RETENTION_DAYS` config. Existing audit store 는 2555일 유지 |
| R5 | LLM Marker 가 이미 삽입된 L2a 결과에 중복 삽입 | 사용자 경험 저하 | 삽입 전 marker 문구 존재 여부 체크 idempotent 보장 |
| R6 | Paper 2 v2 섹션 업데이트 누락 | 논문 제출 시 코드-문서 괴리 | Sprint 4 exit criteria 에 paper2.typ 업데이트 강제 |

### 6.2 검증 체크리스트

Phase 1 종료 시:

- [ ] Sprint 0: Foundation 5개 파일 + 테스트 통과
- [ ] Sprint 1: M3~M6 통합 + predict.py 체크 순서 재설계 검증
- [ ] Sprint 2: M7~M9 통합 + ModelCompetition 게이트 추가 검증
- [ ] Sprint 3: M1, M2, M10~M12 serving 통합 검증
- [ ] Sprint 4: 1M 샘플 배치 추론 end-to-end smoke test 완료
- [ ] paper/typst/paper2.typ 의 "규제 준수" 관련 subsection 들이 실제 코드 경로 (파일:라인) 인용 포함
- [ ] `docs/pipeline_comparison_matrix.md` 의 Must 12 항목이 "양쪽 구현" 으로 이동
- [ ] `docs/aws_work_plan.md` 의 Must 체크박스 12개 모두 완료

---

## 7. Should 후속 (Phase 2 스케치)

Must 완료 후 2~3주 추가로:
- S1 Human Fallback Router 통합
- S7~S9 Monitoring 3종 (Fairness 영속화, Drift DuckDB, Lineage 확장)
- S10 EU AI Act Annex IV 매핑
- S11 L2a Safety Gate 3-layer
- S14 Counterfactual C-C + S15 auto_promote=False

구체 spec 은 Phase 1 완료 후 별도 build plan 섹션 추가.

---

## 8. Could 후속 (Phase 3 스케치)

전체 Sprint 가 끝나 여유 되면:
- C1 Uplift T-Learner (Pearl Rung 2 보강)
- C3 LiquidNeuralNetwork Expert

Paper 2 v2 필수는 아니므로 스킵 가능.

---

## 9. 한눈에 보는 Sprint 마일스톤

| Sprint | 기간 | 주요 Deliverable | v2 Paper 2 기여 |
|---|---|---|---|
| S0 | 3~5일 | Foundation (types, store, SLA tracker, config) | — |
| S1 | 5~7일 | Rights & Consent (M3~M6) | 섹션 "사용자 권리 관리" 코드 근거 |
| S2 | 5~7일 | Registry & Assessments (M7~M9) | 섹션 "FRIA + RMF + 36-항목 준수" 코드 근거 |
| S3 | 3~4일 | Serving & Audit (M1, M2, M10~M12) | 섹션 "운영 라이프사이클 + 감사" 코드 근거 |
| S4 | 2~3일 | 통합 + 문서 | Paper 2 v2 규제 섹션 완성, main merge |
| **총계** | **2.5~3.5주** | **Phase 1 완료** | **v2 Paper 2 규제 근거 확보** |

---

## 10. 즉시 시작 가능한 작업 (우선순위)

Sprint 0 착수 전 **지금 이 세션에서 할 수 있는 것**:

- [ ] `core/compliance/types.py` 스켈레톤 작성
- [ ] `core/compliance/store.py` 에서 `InMemoryComplianceStore` 구현 (테스트 전용)
- [ ] `pipeline.yaml` 에 `compliance:` 블록 추가
- [ ] `tests/test_compliance_foundation.py` 최소 3개 테스트 (put/get/list)

이 정도만 해도 Sprint 0 의 50% 이상 선행. 지금 착수할지 여부는 사용자 지시 대기.

---

## 11. Phase 0 Schema Audit + Mamba Precompute (2026-04-26~28)

본 섹션은 본 문서가 다루는 Sprint 범위 (compliance M1~M12) 와는 별개의 **데이터/피처 파이프라인 인프라 보강**으로, 같은 시기에 main 브랜치에 누적된 변경사항을 추적용으로 정리합니다 (로컬 commit, 아직 origin push 안 됨).

### 11.1 Phase 0 Schema Audit — invariant 위반 6종 차단

Phase 0 출력 (`features.parquet`, `feature_schema.json`) 을 정밀 audit 한 결과, FeatureRouter / config_builder 가 silently 잘못된 슬라이스를 잡아낼 수 있는 invariant 위반 6종 발견 + 차단:

| # | 위반 | 수정 |
|---|---|---|
| 1 | `feature_groups.yaml` 의 `output_columns` 가 placeholder 로 남아 generator 실제 출력과 불일치 | `core/feature/group_pipeline.py` — fit 후 `output_columns` 강제 overwrite (CLAUDE.md §1.7 group range contiguity 의 사전 조건) |
| 2 | Stage 6 정규화가 binary 컬럼을 reorder 하면서 group range 가 비연속 블록으로 깨짐 | `core/preprocessing/normalizer.py::transform_sql` — 입력 컬럼 순서 보존 |
| 3 | Stage 6 출력 후 `feature_cols` 가 group registry 순서와 일치하지 않음 | Stage 6 종료 시 feature group registry 순서로 재정렬 |
| 4 | FeatureRouter / config_builder 가 OOB 슬라이스에서 silently 빈 텐서 반환 | 양쪽 모두 defensive OOB guard 추가 |
| 5 | `feature_groups.yaml::output_dim` 이 generator 실제 출력 차원과 불일치 | 6개 그룹 정렬: demographics 38→11, tda_global 36→16, tda_local 24→16, hmm_states 48→25, product_hierarchy 32→34, graph_collaborative 64→66 |
| 6 | `FeatureSpec.meta_cols` / `TaskSpec.derive` 가 dataclass 에 없어 dead config 였음. `snapshot_date`, `has_nba` 가 `features.parquet` 으로 leak | dataclass 에 필드 추가 + Phase 0 에서 meta_cols 분리 보장 |

**검증**: Phase 0 v3 (commit `88f7a7b`) 이 1211 features / 17 groups / 모두 sequential back-to-back / 마지막 range 가 1205 에서 종료 / trailing 6 passthrough cols (auto-encoded categoricals) 로 통과 확인.

### 11.2 Mamba precompute — 별도 SageMaker Job 으로 분리

Mamba SSM 의 GPU 의존성 (`causal_conv1d` + `mamba_ssm`) 이 stock SageMaker PyTorch 컨테이너에 빌드되지 않아 Phase 0 본 잡과 분리된 precompute 잡으로 운영합니다.

- **이미지 빌드**: `bash scripts/build_mamba_image_codebuild.sh` — AWS CodeBuild 에서 cu122-torch2.1 wheel (causal_conv1d 1.2.0.post2 + mamba_ssm 1.2.0.post1) 사전 빌드 후 ECR 푸시.
- **이미지**: `795833413857.dkr.ecr.ap-northeast-2.amazonaws.com/ple-mamba-precompute:1908a1e3`
- **트리거**: `python scripts/submit_pipeline.py --mamba-precompute`
- **출력**: `s3://aiops-ple-financial/santander_ple/mamba/embedding.parquet` (192 MB, 941K customers × 50D)
- **실측 비용/시간**: g4dn.xlarge spot, **8.6 분 / $0.026/run** (Phase 0 본 잡과 별도 과금)

### 11.3 비용 실측 업데이트

§1.5 (비용 관리) 의 추정치를 본 시기 실측으로 보정:

| 잡 | 추정 → 실측 |
|---|---|
| Phase 0 single-job (CPU instance) | ~7 분 / **$0.04/run (spot)** |
| Mamba precompute (g4dn.xlarge spot) | 8.6 분 / **$0.026/run** |

**4월 누적 비용**: ~$22.55 (CodeBuild $2.49 / 16회 attempts, SageMaker $12+ Phase 0 + Mamba precompute, S3 $1.22, etc.). 비용 비대 항목은 CodeBuild 이미지 빌드 시도 (mamba wheel 빌드 실패 → wheel 사전 빌드 전략 전환) 였고, 1908a1e3 태그 안정화 이후 1회 빌드 / 다회 재사용 구조로 전환.
