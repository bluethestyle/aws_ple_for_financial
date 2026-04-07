# Architecture Positioning & Design Rationale

## 0. 최상위 메시지

AI 추천의 최종 산출물은 확률(0.73)이 아니라 **고객이 납득할 수 있는 이유**이다.

설득의 대상은 항상 사람이다:
- **고객**: "왜 이 상품인가" → 신뢰 → 전환
- **행원**: "왜 이 고객에게 이걸 권하는가" → 영업 근거
- **금감원**: "왜 이런 결정을 내렸는가" → 규제 준수

사람은 확률이 아니라 이야기로 설득된다. 따라서 모델 아키텍처 자체가
설명 가능한 구조여야 하며, 사후적 설명(post-hoc SHAP/LIME)이 아닌
**구조적 설명(inherent explainability)**이 필요하다.

기존 SHAP/LIME의 한계:
- 사후적: 모델과 분리되어 설명이 내부 동작과 괴리될 수 있음
- 불안정: 입력 약간 변해도 설명이 크게 바뀜
- 비용: 추론마다 별도 계산 → 서빙 latency 수 배 증가

이 아키텍처가 지향하는 3가지 구조적 특성:
1. **견고한 설명**: 모델 구조(gate, evidential, contrastive) 자체에서
   설명이 산출. forward pass 한 번에 추론 + 설명 동시 생성.
2. **안정적 내결함성**: expert 하나가 쓸모없더라도 나머지가 gate 재분배로
   자연스럽게 지탱 (graceful degradation). ablation으로 증명.
3. **유연한 확장성**: 새 피처/태스크 추가 시 config만 변경.
   기존 구조 수정 없이 adaTT가 새 관계를 자동 학습.
4. **통합 관리 가능성**: 전체 파이프라인(피처 생성 → 학습 → 증류 → 서빙 → 추천사유)이
   하나의 config 체계(pipeline.yaml + feature_groups.yaml)로 통합 관리.
   기존 방식(모델 N개 × 설명 모듈 × 앙상블 로직 = 관리 포인트 N개)과 대비.
   금융사 ML 운영 인력 1-2명 현실에서 도입 가능성을 결정짓는 요소.

기존 추천 모델의 한계:
- 통계적 상관만으로 추천 → "비슷한 사람들이 샀으니까" (협업 필터링)
- MLP/GBM이 피처를 흔들어서 확률을 출력 → 인과적 근거 없음
- 상관 ≠ 인과: "왜 나한테 이걸 추천하는가"에 답 불가

이종 전문가 구조의 해결:
- 각 expert가 다른 종류의 "왜"에 답한다
  - Temporal: "소비 패턴이 이렇게 변하고 있어서" (시간적 인과)
  - Causal: "이 행동이 이 결과로 이어지는 구조라서" (인과 DAG)
  - HGCN: "상품 카테고리 구조상 자연스러운 다음 단계라서" (계층적 논리)
  - DeepFM: "소득과 보유 상품의 조합 패턴이라서" (교차 근거)
- 단순 상관이 아닌 시간적/구조적/인과적 근거를 가진 설명 생성 가능

이것이 이종 전문가 PLE+adaTT를 설계한 근본 동기이며,
이 논문의 모든 기술적 결정은 이 원칙에서 출발한다.

## 1. 논문 핵심 주장

금융 상품 추천에서 PLE+adaTT 멀티태스크 아키텍처가:
- 이종 전문가(TDA, HMM, Mamba, GNN, GMM 등) 각각이 실질적으로 기여한다 (ablation 증명)
- EU AI Act / 금감원 가이드라인을 충족하는 설명 가능한 추천이 가능하다
- 학습 → 증류 → 설명 가능한 서빙까지 end-to-end 파이프라인을 제공한다
- 불필요한 인프라 복잡성 없이 서버리스 아키텍처로 프로덕션 배포 가능하다

## 2. 아키텍처 구성

### 학습 파이프라인
```
Phase 0: Data Ingestion + Feature Engineering (DuckDB/cuDF)
  ├── Adapter: raw data → standardized format
  ├── 10 Generators: TDA, HMM, Mamba, Graph, GMM, Model-derived 등
  ├── 3-stage Normalization: power-law → StandardScaler → raw copy
  └── Leakage Validation

Phase 1-3: Ablation Study (48 scenarios)
  ├── Feature Group Ablation (16): 각 피처 그룹 기여도
  ├── Expert Ablation (16): 각 전문가 네트워크 기여도
  └── Task × Structure Cross (16): PLE/adaTT 구조 효과

Phase 4: Knowledge Distillation
  ├── PLE 18-task teacher → LGBM student
  ├── IG 기반 피처 선택 (top-k features)
  └── Soft label distillation

Phase 5: Deployment
  ├── LGBM 모델 → Lambda 서빙
  ├── 추천사유 생성 (SHAP/IG attribution → 자연어 템플릿)
  └── API Gateway + Lambda (서버리스)
```

### 데이터 5축 분류 → Expert/Feature 매핑

데이터의 본질적 구조에 따라 5축으로 분류하고, 각 축에 최적화된 expert/feature를 대응:

| 데이터 축 | 예시 | 최적 Expert | 최적 Feature Generator |
|-----------|------|------------|----------------------|
| 상태 정보 | is_active, 보유상품 | DeepFM | base (교차 패턴) |
| 스냅샷 정보 | 월말 잔액, 인구통계 | GMM, DeepFM | GMM clustering |
| 단기 시계열 | 최근 거래 시퀀스 | Transformer | temporal (attention) |
| 장기 시계열 | 월별 소비 트렌드 | Mamba | mamba temporal |
| 단절 시계열 | 휴면→복귀 패턴 | LNN | model derived (LNN velocity) |
| 계층 정보 | 상품 카테고리 트리 | HGCN | product hierarchy |
| 관계 정보 | 고객-상품 그래프 | LightGCN | graph collaborative |
| 위상 정보 | 행동 패턴의 형태 | PersLay | TDA global/local |
| 인과 정보 | 행동 간 인과 방향 | Causal | causal features |

계층 정보에 쌍곡 모델(HGCN)을 도입한 이유:
- 금융 상품은 본질적으로 트리 구조 (예금→단기/중기/장기, 투자→펀드/증권/연금 등)
- 유클리드 공간: 트리 표현에 차원이 기하급수적으로 필요
- 쌍곡 공간: 지수적으로 넓어지는 특성 → 저차원으로 계층 구조를 왜곡 없이 임베딩
- "예금→투자"보다 "예금→대출"이 더 먼 거리라는 관계가 자연스럽게 보존

### 모델 구조
- **PLE (Progressive Layered Extraction)**: 18 tasks, 7 heterogeneous experts, **~2.8M params**
- **FeatureRouter**: feature_groups.yaml의 target_experts 설정에 따라 각 expert에 해당 피처 그룹만 라우팅
  - deepfm=109D, temporal_ensemble=129D, hgcn=34D, perslay=32D, causal=103D, lightgcn=66D, optimal_transport=69D
  - 전체 316 피처 중 expert별로 귀납적 편향에 부합하는 부분집합만 수신 (config-driven)
- **adaTT (Adaptive Task Transfer)**: 4 task groups 간 knowledge transfer
  - engagement: has_nba, engagement_score, next_mcc, top_mcc_shift
  - lifecycle: churn_signal, tenure_stage, segment_prediction
  - value: income_tier, spend_level, cross_sell_count, product_stability
  - consumption: will_acquire_* (5개), nba_primary
- **Expert Basket**: deepfm, temporal_ensemble, hgcn, perslay, causal, lightgcn, optimal_transport
- **Feature Groups**: 12 groups (4 base + 8 generated), 316 features total → FeatureRouter가 expert별 부분집합으로 분배

## 3. 아키텍처 기여 (Contribution)

### Heterogeneous Shared Expert Basket + FeatureRouter
기존 PLE (Tencent, 2020)의 한계:
- shared expert = 동종 MLP × N개 (같은 구조, 다른 초기화)
- 모든 expert가 동일한 전체 피처를 입력받음 → 구조적 차이가 없어 파라미터 낭비
- 파라미터를 늘려야 표현력이 올라감 → 경량 모델에서 표현력 부족

본 논문의 접근 — **이종 아키텍처 × 이종 입력**:
- shared expert = 이종 전문가 7개 (DeepFM, Temporal, HGCN, PersLay, Causal, LightGCN, OT)
- **FeatureRouter가 각 expert에 귀납적 편향에 부합하는 피처 부분집합만 라우팅**:
  - PersLay는 TDA 위상 피처(32D)만 수신, LightGCN은 그래프 임베딩(66D)만 수신
  - HGCN은 상품 계층 피처(34D)만, Temporal Ensemble은 시계열 피처(129D)만 수신
  - 라우팅은 feature_groups.yaml의 target_experts에서 선언적으로 정의 (코드 수정 불필요)
- 기존 PLE는 "다른 구조가 같은 데이터를 다른 관점에서 처리"하는 수준이었으나,
  본 구조는 **"다른 구조가 다른 데이터를 처리"**하는 더 강한 이종성을 실현
- 이로 인해 expert 간 입력 중복이 제거되어 파라미터 효율이 향상: 4.77M → ~2.8M (감소)
- CGC gate가 "이 태스크에는 어떤 관점이 유용한가"를 학습
- 결과: 단일 대형 MLP보다 가벼우면서 표현력은 더 높음. 파라미터 대비 표현력 비율이 극대화됨

### Inherent Explainability (구조적 설명 가능성)
이종 전문가 구조의 부수적이지만 핵심적인 이점: 모델 자체가 설명력을 내재.

동종 MLP expert의 경우:
- "MLP 3번이 기여했습니다" → 비즈니스 의미 없음
- 사후적으로 SHAP/LIME을 적용해야 설명 가능 → 추가 비용

이종 expert의 경우:
- CGC gate weight 자체가 설명: "Temporal(0.35), DeepFM(0.28), HGCN(0.22)"
- expert 이름이 곧 비즈니스 맥락: "시계열 패턴이 35% 기여, 피처 교차 28%, 상품 계층 22%"
- 별도 설명 모듈 없이도 금감원 "왜 이 추천인가"에 답변 가능

이것이 Black-Litterman에서 "각 전문가(모델) 의견의 불확실성 기반 비중"으로 하려던 것을,
해석 가능한 gate weight로, 그리고 데이터 기반 학습으로 구현한 셈이다.

이 설계의 동기 (하드웨어 제약):
- 금융사 현실: GPU 1-4장 수준의 제한된 리소스
- MLP 파라미터 확장 불가 → 구조적 편향(inductive bias)으로 표현력 확보
- 각 expert가 수만 파라미터로, MLP가 수백만 파라미터로 하는 패턴 인식을 대체

| Expert | Inductive Bias | 적은 파라미터로 잡는 패턴 |
|--------|---------------|----------------------|
| DeepFM | feature interaction | 피처 간 2차 교차 효과 |
| **Temporal Ensemble** | **복합 시계열** | **장/단기/단절 시계열 통합** (아래 상세) |
| HGCN | 계층 구조 | 상품 카테고리 트리 |
| PersLay | 위상 구조 | TDA persistence 형상 |
| LightGCN | 그래프 관계 | 고객-상품 협업 필터링 |
| Causal | 인과 관계 | 인과 방향 (DAG 제약) |
| OT | 분포 매칭 | 고객 세그먼트 간 분포 차이 |

### Temporal Ensemble: Expert 내부 앙상블
금융 시계열은 단일 모델로 포착할 수 없는 복합 구조:
- 스냅샷 (월 1회 잔액/보유 상품), 일/주 단위 거래, 고빈도 실시간 거래
- 단절 (3개월 휴면 후 갑자기 활동 재개)

따라서 Temporal Expert 자체를 3개 모델의 내부 앙상블로 설계:

```
Temporal Ensemble Expert
  ├── Mamba (SSM)       — 장기 시계열: 월/분기 단위 트렌드, O(n) 효율
  ├── LNN (Liquid NN)   — 단절 시계열: 적응적 시간 상수, 불규칙 간격에 강함
  └── Transformer       — 단기 맥락: 최근 거래 시퀀스 내 attention 패턴
```

| 모델 | 강점 | 약점 |
|------|------|------|
| Mamba | 장기 의존성, O(n) 효율 | 불규칙 간격에 약함 |
| LNN | 단절/불규칙 적응 | 긴 맥락 요약 약함 |
| Transformer | 단기 맥락 추출 | O(n²), 긴 시퀀스 비효율 |

PLE 전체의 "이종 expert" 철학이 Temporal Expert 내부에서도 동일하게 적용.
서로의 약점을 보완하는 조합으로, 금융 시계열의 복합 구조를 포착.

### 출력 쪽 인과설명 아키텍처

이종 expert + gate weight만으로는 "어떤 관점이 기여했는가"는 설명되지만,
"이 추천이 얼마나 확실한가", "왜 이것이고 저것은 아닌가"에는 답이 부족.
출력 쪽에 3가지 인과/설명 구조를 추가:

**1. Contrastive Learning (InfoNCE)**
- 대조 학습으로 "왜 이 상품이고, 왜 저 상품이 아닌가" 설명
- 양성/음성 쌍을 대조하여, 추천 상품과 비추천 상품의 차이를 명시적으로 학습
- 추천사유 생성 시 "A 상품은 적합하지만 B 상품은 적합하지 않은 이유" 제공 가능

**2. Evidential Deep Learning (NIG 분포)**
- 예측의 불확실성을 정량화: Normal-Inverse-Gamma 분포로 (mu, v, alpha, beta) 출력
- "이 추천은 확신도가 높다/낮다"를 수치로 제공
- 불확실성이 높으면: 행원에게 "추가 상담 필요" 플래그, 고객에게 공격적 추천 자제
- 금감원 관점: 모델이 "모른다"고 말할 수 있는 능력 = 안전장치

**3. SAE (Sparse Autoencoder) Regularization**
- shared expert의 concat 출력에 sparse autoencoder 적용
- 내부 표현의 희소성을 강제 → 해석 가능한 뉴런 활성화
- 특정 뉴런이 특정 비즈니스 개념에 대응 → mechanistic interpretability

이 3가지가 로짓 전이, gate weight와 함께 작동하여:
- gate weight: "어떤 관점이 기여" (expert-level 설명)
- contrastive: "왜 이것이고 저것은 아닌가" (대조 설명)
- evidential: "얼마나 확실한가" (불확실성 정량화)
- SAE: "내부적으로 어떤 개념이 활성화됐는가" (뉴런-level 설명)

## 4. 금융 도메인 특화 설계

### 성능보다 안정성/설명 가능성 우선
- 빅테크: "AUC 0.01 올리면 매출 N억" → 공격적 실험
- 금융사: "모델 이상 작동 시 금감원 제재" → 보수적 운영, graceful degradation
- Ablation으로 "어떤 expert를 빼도 급격한 성능 저하 없음"을 보여줌

### Cold Start 처리
- is_cold_start flag + sequence feature zeroing (거래 3건 미만 고객)
- 신규 고객 대응: demographics + product holdings만으로 추천 가능

### Data Leakage 방지
- Scaler는 TRAIN split에서만 fit
- Temporal split gap_days 설정
- Generator 입력에서 label 컬럼 자동 제외
- LeakageValidator 학습 전 필수 호출
