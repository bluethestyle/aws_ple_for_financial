# 08. Recommendation Intelligence — 추천 사유, 스코어링, 해석, 운영 규칙

## 개요

모델 추론(05장)은 "어떤 태스크에서 몇 점인지"를 계산합니다.
이 장은 그 **이후의 전체 추천 파이프라인** — 점수를 비즈니스 추천 리스트로 변환하고,
추천 사유를 생성하고, 규제 준수를 검증하는 과정을 다룹니다.

```
모델 추론 (LGBM/PLE)
    ↓ 태스크별 raw score
스코어링 (FD-TVS)
    ↓ 비즈니스 가중 점수
제약조건 필터링
    ↓ 적격 추천 목록
Top-K 선택 + 다양성
    ↓ 최종 추천 리스트
추천 사유 생성 (3계층)
    ↓ 비즈니스 언어의 설명
셀프 검증 + 규제 준수
    ↓ 검증된 추천 + 사유
서빙 응답
```

---

## 현재 (On-Prem) 구현 요약

### 1. FD-TVS 스코어링 (`fd_tvs_scorer.py`)
```
FD-TVS = S_task × W_DNA × V_TDA × (1 - R_penalty) × fatigue_decay × engagement_boost
```
- Stage 1: 태스크 가중합 (CTR×0.3 + CVR×0.4 + NBA×0.2 + LTV×0.1)
- Stage 2: 소득 안정성 DNA 보정 (Permanent 1.2 / Mixed 1.0 / Transitory 0.8)
- Stage 3: TDA 행동 가속도 반영 (1 + γ × Flare지표)
- Stage 4: 리스크 패널티 (한도소진율, 메시지빈도, 이탈확률)
- 추가: 피로도 감쇠, 참여도 부스트

### 2. 제약조건 추천 엔진 (`constraint_aware_engine.py`)
5단계 파이프라인:
1. 모델 예측값 수집
2. 고객 적격성 필터 (피로도, 이탈위험, 한도소진)
3. 상품 필터 (보유상품 제외, 적합성, 캠페인 제외)
4. 점수 조정 (FD-TVS, DPP 다양성, 비즈니스 가중치)
5. Top-K 선택

### 3. 추천 사유 3계층 (`grounding/` 패키지)
- **L1 (템플릿)**: IG top-3 피처 → 역매핑 → 30개 템플릿 (6카테고리 × 5변형)
  - 120만 고객, ~20분, LLM 호출 없음
- **L2a (LLM 리라이트)**: L1 결과를 LLM(Qwen3-8B-AWQ)으로 자연어 보강
  - 주간 32.5만 고객, 우선순위 큐 (rich → moderate)
  - 3겹 안전 게이트 (파싱 → 규제 → 품질)
- **L2b (품질 검증)**: L1 0.4% + L2a 5% 샘플링 검증

### 4. LanceDB 벡터 스토어 (`context_vector_store.py`)
- **customer_context 테이블**: 644D 피처 해석 임베딩 + 상담 이력 + 세그먼트
- **generated_reasons 테이블**: 생성된 추천 사유 캐시 (서빙용)
- 유사 고객 검색: 콜드스타트 시 ANN 검색 (k=5)
- 하이브리드 검색: 벡터 + 풀텍스트

### 5. 피처 역매핑 (`feature_reverse_mapper.py`)
644D 피처를 7개 범위로 분류 → 비즈니스 언어로 변환:
- Profile (0-238D): 인구통계, RFM, 재무요약
- Multi-Source (238-329D): 거래통계, 행동패턴
- Domain (329-488D): 쌍곡 MCC, TDA, HMM, GMM
- Multidisciplinary (488-512D): 화학동역학, 전파확산, 간섭패턴, 범죄패턴
- Model-Derived (512-539D): HMM, MAB, LNN
- Extended (539-623D): 보험, 상담, 해외결제
- Merchant (623-644D): MCC/브랜드 계층

### 6. 셀프 검증 (`self_critique_agent.py`, `ai_security_checker.py`)
- 규칙 기반 규제 준수 (금소법, 금감원: "확정수익", "원금보장" 등 차단)
- LLM 기반 사실성 검증 (factual_score, compliance_score)
- 프롬프트 인젝션 방어 (7개 패턴)
- AI 공시 의무 (AI 기본법 제31·34조)

---

## AWS 설계 — 모듈형 재설계

### 핵심 변경 방향

현재 코드는 **금융 도메인에 종속된 규칙**이 코드에 하드코딩되어 있습니다.
AWS 버전에서는 이를 **config + 플러그인**으로 분리하여, 도메인을 바꾸면 규칙도 바뀌게 합니다.

```yaml
# configs/recommendation.yaml — 추천 전체 파이프라인 설정
recommendation:
  scoring:
    strategy: fd_tvs           # fd_tvs | weighted_sum | custom
    task_weights:
      click: 0.3
      purchase: 0.4
      category: 0.2
      revenue: 0.1

  constraints:
    fatigue:
      enabled: true
      max_messages_7d: 5
      decay_rate: 0.15
    eligibility:
      min_score: 0.3
      exclude_owned_products: true

  diversity:
    method: dpp                # dpp | mmr | none
    lambda: 0.3                # 다양성 vs 관련성 균형

  top_k: 10

  reason_generation:
    strategy: template_llm     # template_only | template_llm | llm_only
    llm_provider: bedrock      # bedrock | openai | local_vllm
    llm_model: anthropic.claude-3-haiku
    self_check: true
    compliance_rules: configs/compliance_rules.yaml
```

---

### 스코어링 모듈

```python
# core/recommendation/scorer.py
class ScorerRegistry:
    """스코어링 전략 플러그인 등록."""
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(scorer_cls):
            cls._registry[name] = scorer_cls
            return scorer_cls
        return decorator

    @classmethod
    def build(cls, config) -> AbstractScorer:
        return cls._registry[config.strategy](config)


class AbstractScorer(ABC):
    @abstractmethod
    def score(self, predictions: dict, user_context: dict) -> float:
        """태스크별 예측값 + 유저 컨텍스트 → 최종 추천 점수"""
        ...


@ScorerRegistry.register("weighted_sum")
class WeightedSumScorer(AbstractScorer):
    """단순 태스크 가중합. 가장 기본적인 스코어링."""
    def score(self, predictions, user_context):
        total = sum(predictions[t] * self.weights[t] for t in self.weights)
        return total


@ScorerRegistry.register("fd_tvs")
class FDTVSScorer(AbstractScorer):
    """
    Financial DNA-based Target Value Score.
    4단계: 태스크가중합 × DNA보정 × TDA가속도 × (1-리스크패널티)

    금융 도메인 전용. 다른 도메인에서는 weighted_sum 또는 커스텀 사용.
    """
    def score(self, predictions, user_context):
        s_task = self._task_weighted_sum(predictions)
        w_dna = self._dna_modifier(user_context)
        v_tda = self._behavioral_velocity(user_context)
        r_penalty = self._risk_penalty(user_context)
        fatigue = self._fatigue_decay(user_context)
        engagement = self._engagement_boost(user_context)
        return s_task * w_dna * v_tda * (1 - r_penalty) * fatigue * engagement
```

### 제약조건 필터링

```python
# core/recommendation/constraint_engine.py
class ConstraintEngine:
    """
    5단계 추천 필터링 파이프라인.
    각 단계는 config에서 on/off 가능합니다.
    """

    def __init__(self, config):
        self.filters = []
        if config.constraints.fatigue.enabled:
            self.filters.append(FatigueFilter(config.constraints.fatigue))
        if config.constraints.eligibility.exclude_owned_products:
            self.filters.append(OwnedProductFilter())
        # 커스텀 필터 추가 가능
        for custom in config.constraints.get("custom_filters", []):
            self.filters.append(FilterRegistry.build(custom))

    def apply(self, candidates: list[dict], user_context: dict) -> list[dict]:
        for f in self.filters:
            candidates = f.filter(candidates, user_context)
        return candidates
```

### Top-K + 다양성

```python
# core/recommendation/selector.py
class TopKSelector:
    """Top-K 선택 + 다양성 필터."""

    def select(self, scored_candidates, k, diversity_config) -> list:
        if diversity_config.method == "dpp":
            return self._dpp_select(scored_candidates, k, diversity_config.lambda_)
        elif diversity_config.method == "mmr":
            return self._mmr_select(scored_candidates, k, diversity_config.lambda_)
        else:
            return sorted(scored_candidates, key=lambda x: x["score"], reverse=True)[:k]

    def _dpp_select(self, candidates, k, lambda_):
        """Determinantal Point Process — 점수 높으면서 서로 다른 아이템 선택."""
        ...
```

---

### 추천 사유 생성 — 3계층 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│ L1: 템플릿 엔진 (전체 유저, LLM 호출 없음, ~20분)                 │
│                                                                 │
│ IG top-3 피처 → 역매핑 → 카테고리 분류 → 템플릿 선택              │
│                                                                 │
│ 6 카테고리:                                                      │
│   spending_pattern | frequency_pattern | life_stage              │
│   benefit_match | popularity | minimum_safe                     │
│ × 5 변형 = 30 템플릿                                             │
│                                                                 │
│ 태스크별 프레임 적용 (Phase 5):                                   │
│   churn → "이탈 방지 관점"                                       │
│   ltv → "고객 가치 극대화 관점"                                   │
│   nba → "다음 최적 행동 관점"                                    │
└─────────────────────────────┬───────────────────────────────────┘
                              │ 우선순위 큐 (rich → moderate)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ L2a: LLM 리라이트 (우선 유저만, 주간 배치)                        │
│                                                                 │
│ L1 템플릿 + 고객 컨텍스트 + 상담 이력                              │
│   → LLM (자연어 보강)                                            │
│   → 3겹 안전 게이트:                                             │
│     Gate 1: 파싱 검증 (비어있거나 깨진 텍스트)                     │
│     Gate 2: 규제 준수 (금소법 위반 키워드)                         │
│     Gate 3: 품질 (길이 30-200자, 한국어 비율 ≥80%)               │
│   → 실패 시: L1 원본 유지                                        │
└─────────────────────────────┬───────────────────────────────────┘
                              │ 샘플링 (L1 0.4% + L2a 5%)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ L2b: 품질 검증 (샘플 기반)                                       │
│                                                                 │
│ 검증 항목:                                                       │
│   - 길이 분포                                                    │
│   - 한국어 비율 통계                                              │
│   - 규제 위반 건수                                                │
│   - 템플릿 변형 분포 (특정 변형 편중 감지)                          │
│                                                                 │
│ → 품질 보고서 생성 → S3 저장                                      │
└─────────────────────────────────────────────────────────────────┘
```

### AWS에서의 추천 사유 생성

```yaml
# configs/recommendation.yaml (reason 부분)
reason_generation:
  # L1: 템플릿 (항상 실행)
  l1:
    enabled: true
    template_categories: 6
    template_variants_per_category: 5

  # L2a: LLM 리라이트 (선택적)
  l2a:
    enabled: true
    strategy: template_llm
    llm_provider: bedrock          # AWS Bedrock (관리형)
    llm_model: anthropic.claude-3-haiku
    # 또는 local_vllm (SageMaker Endpoint에 배포)
    batch_size: 1000
    priority: [rich, moderate]     # sparse는 L1만

  # L2b: 품질 검증 (항상 실행)
  l2b:
    l1_sample_rate: 0.004
    l2a_sample_rate: 0.05

  # 셀프 검증
  self_check:
    compliance_rules: configs/compliance_rules.yaml
    prompt_injection_check: true
    ai_disclosure: true
```

### LLM 프로바이더 — AWS Bedrock 통합

```
On-Prem:  vLLM (Qwen3-8B-AWQ) → GPU 서버 상시 가동
AWS:      Bedrock (Claude Haiku) → 요청당 과금, 서버 없음

비용 비교:
  vLLM GPU 상시: ~$200/월
  Bedrock Haiku: 325K 요청/주 × $0.00025/건 = ~$80/월
  → 50% 이상 절감 + 서버 관리 불필요
```

```python
# core/recommendation/reason/llm_provider.py
class LLMProviderFactory:
    """LLM 프로바이더 추상화. config로 전환."""

    @staticmethod
    def create(config) -> AbstractLLMProvider:
        if config.llm_provider == "bedrock":
            return BedrockProvider(config.llm_model)
        elif config.llm_provider == "openai":
            return OpenAIProvider(config.llm_model)
        elif config.llm_provider == "local_vllm":
            return VLLMProvider(config.vllm_endpoint)
        else:
            return DummyProvider()  # 테스트/개발용
```

---

### LanceDB — AWS 환경에서의 배치

```
On-Prem:  로컬 디스크에 LanceDB 파일
AWS:      2가지 옵션

Option A (서버리스 — Lambda):
  LanceDB 파일을 S3에 저장
  → Lambda 시작 시 /tmp에 다운로드 (~5-10초, cold start)
  → 이후 쿼리는 로컬 속도
  → Provisioned Concurrency로 cold start 제거 가능

Option B (ECS — 대규모):
  LanceDB 파일을 EFS에 마운트
  → ECS 컨테이너에서 직접 접근
  → 네트워크 홉 없음
```

```python
# core/recommendation/reason/vector_store.py
class VectorStoreFactory:
    @staticmethod
    def create(config) -> AbstractVectorStore:
        if config.vector_store == "lancedb":
            return LanceDBStore(config.lancedb_path)
        elif config.vector_store == "dynamodb":
            # 벡터 검색 없이 key-value만 필요할 때
            return DynamoDBStore(config.dynamodb_table)
        elif config.vector_store == "memory":
            return InMemoryStore()  # 테스트/소규모
```

---

### 피처 역매핑 — 도메인 무관 설계

현재는 644D 피처를 7개 금융 범위로 하드코딩했지만, AWS 버전에서는 **스키마에서 자동 생성**합니다.

```yaml
# configs/feature_interpretation.yaml
interpretation:
  # 피처 범위 → 비즈니스 카테고리 매핑
  ranges:
    - name: demographics
      features: [age, income, tenure]
      template: "{feature_name}이(가) {value}로, {direction} 고객입니다."
    - name: transaction_behavior
      features: [total_spend, tx_count, avg_amount]
      template: "{period} 동안 {feature_name} 기준 {percentile}% 수준입니다."
    - name: engagement
      features: [click_rate, session_count, app_usage_days]
      template: "최근 {feature_name}이(가) {trend} 추세입니다."

  # 다학제 해석 (선택적 플러그인)
  multidisciplinary:
    chemical_kinetics:
      enabled: false    # 금융 도메인이 아니면 비활성화
      description: "소비 변화 속도"
    epidemic_diffusion:
      enabled: false
      description: "카테고리 탐색 패턴"

  # 태스크별 해석 프레임 (Phase 5)
  task_frames:
    click:
      narrative_lens: engagement
      explanation_frame: "클릭 관여도 요인"
      primary_features: [frequency_pattern, benefit_match]
    churn:
      narrative_lens: lifecycle
      explanation_frame: "이탈 위험 요인"
      primary_features: [tenure, recency, engagement_trend]
    revenue:
      narrative_lens: value
      explanation_frame: "수익 기여 요인"
      primary_features: [total_spend, purchase_frequency]
```

```python
# core/recommendation/interpretation/reverse_mapper.py
class FeatureReverseMapper:
    """
    피처 인덱스 → 비즈니스 언어 변환.
    config에서 범위/템플릿을 정의하므로 도메인 교체 가능.
    """

    def __init__(self, config):
        self.ranges = config.interpretation.ranges
        self.task_frames = config.interpretation.task_frames

    def interpret_top_k(self, feature_importances: np.ndarray, k: int = 3,
                         task: str = None) -> list[str]:
        top_indices = np.argsort(feature_importances)[-k:][::-1]
        interpretations = []
        for idx in top_indices:
            range_info = self._find_range(idx)
            text = range_info.template.format(
                feature_name=range_info.features[idx - range_info.start],
                value=feature_importances[idx],
                direction=self._direction(feature_importances[idx]),
            )
            if task:
                frame = self.task_frames.get(task, {})
                text = f"[{frame.get('narrative_lens', '')}] {text}"
            interpretations.append(text)
        return interpretations
```

---

### 셀프 검증 & 규제 준수

```yaml
# configs/compliance_rules.yaml
compliance:
  # 규칙 기반 차단 (즉시 거부)
  blacklist_patterns:
    critical:
      - pattern: "확정.*수익"
        reason: "확정수익 보장 표현 금지"
      - pattern: "원금.*보장"
        reason: "원금보장 표현 금지 (원본손실 가능)"
      - pattern: "손실.*불가능"
        reason: "손실불가능 표현 금지"
    high:
      - pattern: "반드시.*수익"
        reason: "수익 단정 표현 금지"

  # 프롬프트 인젝션 방어
  injection_patterns:
    - "ignore previous instructions"
    - "<system>"
    - "이전 지시 무시"
    - "역할을 바꿔"
    - "you are now"
    - "너는 이제"

  # AI 공시 의무
  ai_disclosure:
    enabled: true
    template: "본 추천은 AI 알고리즘에 의해 생성되었으며, 투자 판단의 최종 책임은 고객에게 있습니다."
    regulations:
      - "AI 기본법 제31조 (AI 생성물 표시)"
      - "AI 기본법 제34조 (이용자 보호)"
      - "금소법 제19조 (설명의무)"
```

```python
# core/recommendation/reason/self_checker.py
class SelfChecker:
    """
    추천 사유 검증 파이프라인.
    규칙 기반 (즉시) + LLM 기반 (선택적) 2중 검증.
    """

    def check(self, reason_text: str, config) -> CheckResult:
        # Gate 1: 규칙 기반 (빠름, 필수)
        rule_result = self._rule_check(reason_text, config.compliance)
        if rule_result.blocked:
            return CheckResult(passed=False, reason=rule_result.reason)

        # Gate 2: 프롬프트 인젝션 (빠름, 필수)
        injection_result = self._injection_check(reason_text, config.compliance)
        if injection_result.detected:
            return CheckResult(passed=False, reason="prompt_injection_detected")

        # Gate 3: LLM 사실성 검증 (선택적, 비용 발생)
        if config.self_check_llm:
            llm_result = self._llm_factuality_check(reason_text)
            if llm_result.score < 0.7:
                return CheckResult(passed=False, reason="low_factuality")

        return CheckResult(passed=True)
```

---

### 전체 추천 파이프라인 (통합)

```
요청 (user_id, context)
    ↓
① 피처 조회 (Feature Store)
    ↓
② LGBM 멀티태스크 추론 (~5ms)
    ↓ {click: 0.82, purchase: 0.45, revenue: 120, category: [0.3, 0.2, ...]}
    ↓
③ FD-TVS 스코어링 (config.scoring)
    ↓ 비즈니스 가중 점수
    ↓
④ 제약조건 필터링 (config.constraints)
    ├── 피로도 체크
    ├── 보유상품 제외
    └── 적격성 검증
    ↓
⑤ Top-K + 다양성 (config.diversity)
    ↓ 최종 추천 리스트 (k=10)
    ↓
⑥ 추천 사유 생성 (config.reason_generation)
    ├── L1: IG top-3 → 역매핑 → 템플릿
    ├── L2a: LLM 리라이트 (캐시 있으면 스킵)
    └── 셀프 검증 (규제 준수 + 프롬프트 인젝션)
    ↓
⑦ LanceDB 사유 캐시 저장
    ↓
⑧ 응답
    {
      "recommendations": [...],
      "reasons": {"click": "...", "purchase": "..."},
      "ai_disclosure": "..."
    }
```

---

## 현재 vs AWS — 추천 인텔리전스 비교

| 항목 | 현재 (On-Prem) | AWS (설계) | 변경 이유 |
|------|---------------|-----------|----------|
| 스코어링 | FD-TVS 하드코딩 | ScorerRegistry (플러그인) | 도메인별 전략 교체 |
| 제약조건 | 금융 규칙 코드 내장 | config YAML + FilterRegistry | 선언적, 도메인 무관 |
| 다양성 | DPP 하드코딩 | DPP/MMR/없음 config 전환 | 유연성 |
| LLM | vLLM 자체 호스팅 ($200/월) | Bedrock ($80/월) 또는 vLLM | 비용 절감 + 서버 제거 |
| 벡터 스토어 | LanceDB 로컬 | LanceDB (Lambda/tmp 또는 ECS/EFS) | 동일 엔진, 배포만 변경 |
| 역매핑 | 644D 범위 하드코딩 | config 기반 범위 정의 | 피처 구조 변경 시 자동 반영 |
| 규제 준수 | 금소법 패턴 코드 내장 | compliance_rules.yaml | 규제 변경 시 코드 수정 불필요 |
| 태스크 해석 | Phase 5 코드 내장 | task_frames config | 태스크 추가 시 해석도 추가 |
