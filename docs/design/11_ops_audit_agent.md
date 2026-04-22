# 11. 운영 에이전트 & 감사 에이전트 — 파이프라인 자율 진단 체계

## 개요

09장의 감사/거버넌스 인프라와 08장의 추천사유 생성은 **개별 컴포넌트**로 존재한다.
이 장은 그 컴포넌트들을 **두 개의 자율 진단 에이전트**로 묶어,
파이프라인 전체를 관점별로 관찰하고 담당자에게 "어디를 봐야 하는지" 진단하는 체계를 설계한다.

```
자율 진단 체계
├── 운영 에이전트 (Ops Agent)
│   └── "파이프라인이 잘 돌아가고 있는가" — 성능, 안정성, 비용
├── 감사 에이전트 (Audit Agent)
│   └── "규정을 준수하고 있는가" — 공정성, 설명가능성, 추천사유 품질
└── 공유 인프라
    ├── Event Bus (EventBridge/SNS)
    ├── GovernanceReportGenerator (월간 통합)
    └── IncidentReporter (긴급 에스컬레이션)
```

### 추천사유 파이프라인과의 관계

추천사유 생성(`AsyncReasonOrchestrator`)은 서빙 경로의 일부로, 고객 응답에 직접 영향을 주는
latency-critical 경로이다. 운영/감사 에이전트는 이 경로와 **비동기로 분리**되어 다음을 보장한다:

- **레이턴시 디커플링**: 감사 로깅이나 fairness 계산 장애가 추천 응답을 블로킹하지 않음
- **규제 독립성**: 감사 정책 변경 시 추천 파이프라인 재배포 불필요
- **장애 격리**: 감사 로그 유실 방지를 위한 별도 재시도/DLQ 전략 적용 가능

---

## 공통 아키텍처: Collect → Diagnose → Report

두 에이전트 모두 동일한 3단계 루프를 따른다:

```
Collect  (체크포인트별 측정값 수집)
  → Diagnose  (임계값 · 추세 · 상관관계 기반 진단)
    → Report  (담당자에게 "어디를 봐야 하는지" 전달)
```

차이는 **어떤 체크포인트를 보는가**, **어떤 기준으로 진단하는가**이다.

### 실행 계층: 온프렘 기본 + AWS Bedrock 확장

에이전트의 **본질적 가치**(자동 진단 + 체크리스트 판정 + 리포팅)는
**결정론적 Python 룰 엔진**만으로 완결된다. LLM 없이 온프렘에서 100% 작동하는 것이 기본이다.

AWS 환경에서는 Bedrock을 통해 **담당자 편의 기능**을 추가한다.
퍼블릭 클라우드로 추천 서비스를 구성하는 기업들에게 실용적 레퍼런스 모델을 제시하는 것이 목적이다.

```
┌─────────────────────────────────────────────────────┐
│         온프렘 — 기본 (Baseline)                      │
│                                                      │
│  Collect ──→ Diagnose ──→ Report                     │
│  (측정값 수집)  (룰 엔진 판정)  (템플릿 리포트)          │
│                                                      │
│  LLM 없음 · 결정론적 · 감사 추적 100% 재현 가능        │
│  48개 체크리스트 자동 판정 · 비용 0                     │
└─────────────────────┬───────────────────────────────┘
                      │ 온프렘 엔진 그대로 +
┌─────────────────────▼───────────────────────────────┐
│         AWS — Bedrock 확장 (담당자 편의 기능)          │
│                                                      │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │
│  │ Interpret &  │ │ Impact       │ │ Deep Audit   │ │
│  │ Discuss      │ │ Review       │ │ (분기 1회)    │ │
│  │ (Sonnet)     │ │ (Sonnet)     │ │ (Opus)       │ │
│  │              │ │              │ │              │ │
│  │ 진단 결과    │ │ 변경 영향도  │ │ 다중 규제    │ │
│  │ 해석 + 대화  │ │ 추론         │ │ 트레이드오프 │ │
│  └──────────────┘ └──────────────┘ └──────────────┘ │
│                                                      │
│  Bedrock 호출 · 대화당 ~$0.01                         │
└──────────────────────────────────────────────────────┘
```

#### 환경별 기능 매트릭스

| 기능 | 온프렘 | AWS | 비고 |
|---|---|---|---|
| 체크리스트 자동 판정 (48항목) | O | O | Python 룰 엔진 |
| 연쇄 영향 분석 (cross-checkpoint) | O | O | 사전 정의 룰 테이블 |
| 정형 리포트 (finding/cause/action) | O | O | 템플릿 + 수치 삽입 |
| 상호 트리거 (Ops ↔ Audit) | O | O | 이벤트 기반 |
| 거버넌스 리포트 통합 | O | O | GovernanceReportGenerator |
| 인시던트 에스컬레이션 | O | O | SNS / 이메일 / Slack |
| 진단 결과 해석 대화 | -- | O | Sonnet via Bedrock |
| 변경 영향도 리뷰 | -- | O | Sonnet via Bedrock |
| 분기 심층 감사 리뷰 | -- | O | Opus via Bedrock (선택) |

#### 왜 이 구조인가

> **온프렘이 기본인 이유**: (1) 결정론성 — 같은 입력이면 같은 판정, 감사 추적에서 본질적으로 중요. (2) 독립성 — 외부 API 의존 없이 운영, 네트워크/보안/비용 제약에 무관. (3) 충분성 — 48개 체크리스트 + 연쇄 영향 룰 + 정형 리포트면 대부분의 상황을 커버.

> **AWS Bedrock 확장의 가치**: (1) 대화 — "이 DI가 필터 문제인지 모수 문제인지"를 에이전트와 논의, 의사결정 가속. (2) 영향도 리뷰 — 코드 변경의 하류 영향을 추론, 리뷰 누락 감소. (3) 레퍼런스 — 퍼블릭 클라우드 추천 서비스 운영 기업에게 "Bedrock으로 운영 에이전트를 이렇게 강화할 수 있다"는 실용적 모델 제시.

---

### 3-에이전트 합의 메커니즘 (AWS Bedrock)

LLM 기반 해석에는 할루시네이션 리스크가 있다. 이를 구조적으로 완화하기 위해 **3개의 독립 에이전트 세션**을 병렬 실행하고 **합의 수준**에 따라 결과를 분류한다.

```
Agent α (Sonnet)     Agent β (Sonnet)     Agent γ (Sonnet)
"WARN: 필터 문제"    "WARN: 모수 문제"    "PASS: 유의성 부족"
      │                    │                    │
      └────────┬───────────┘                    │
               ▼                                ▼
        최우선 리뷰 (2/3)              마이너리티 리포트 (1/3)
```

#### 합의 판정 규칙

| α | β | γ | 판정 |
|---|---|---|---|
| PASS | PASS | PASS | **Consensus Pass** — 안전 |
| WARN | WARN | WARN | **Consensus WARN** — 확정, 최우선 리뷰 |
| WARN | WARN | PASS | **최우선 리뷰** — 2/3 이상 징후 |
| WARN | PASS | PASS | **마이너리티 리포트** — 1/3 소수 의견, 2순위 리뷰 |
| FAIL | * | * | **즉시 에스컬레이션** — FAIL은 합의와 무관 |

#### 마이너리티 리포트 (Minority Report)

소수 의견(1/3)이라고 무시하지 않는다. "3명의 프리콕 중 1명만 다른 예측을 했다"는 그 자체로 중요한 신호.

포함 정보:
- 소수 의견 에이전트의 finding + likely_cause + 근거
- 다수 의견과의 구체적 차이점
- 과거 마이너리티 리포트 중 사후에 맞았던 비율 (케이스 스토어 통계)

> **마이너리티 리포트의 가치**: 새로운 유형의 문제는 기존 패턴에 익숙한 다수가 놓치고 다른 관점의 소수가 먼저 포착할 수 있다. 케이스 스토어에 `consensus_type: "minority"` 태그로 저장하여 "마이너리티가 맞았던 비율"을 추적.

#### 환경별 합의 방식: 독립 투표 vs 순차 심의

AWS와 온프렘에서 합의 메커니즘의 **방식 자체가 다르다**:

**AWS: 독립 병렬 투표 (배심원)**
- Sonnet 3개 × 병렬 실행, 서로의 출력을 보지 않음
- 관점 변주: α 보수적 / β 통계적 / γ 비즈니스
- ~5초 (병렬), 비용 3×

**온프렘: 2-Round 하이브리드 (독립 투표 → 순차 심의)**

순수 델파이(순차 심의)는 수렴 편향이 있다 — 뒤로 갈수록 앞 의견에 끌려 소수 의견이 사라진다. 운영/감사에서는 **"놓치는 것"이 "오탐"보다 훨씬 위험**하므로 2-Round로 분리:

```
[Round 1: 독립 투표 — 마이너리티 보존]
  ① → "필터 문제"      (독립, 서로 안 봄)
  ② → "모수 문제"      (독립)
  ③ → "PASS"          (독립)
  ④ → "필터 문제"      (독립)
  ⑤ → "계절 패턴"      (독립)

  → 집계: 필터 2, 모수 1, PASS 1, 계절 1
  → 마이너리티 확정: ③(PASS), ⑤(계절) — 이후 삭제 불가

[Round 2: 순차 심의 — 논거 보강 (마이너리티 삭제 불가)]
  ⑥ → Round 1 전체를 보고 다수 논거 정리 + 소수 타당성 평가
  ⑦ → ⑥을 보고 종합 판정 + 각 의견 근거 보강/반박 정리
```

> **핵심 원칙**: Round 1에서 확정된 마이너리티는 삭제되지 않는다. Round 2에서 ⑥이 "계절 패턴은 타당성 낮다"고 해도, "⑥이 타당성 낮다고 평가 — 근거: ... / 원 의견(⑤) 보존"으로 기록. 최종 판단은 사람이 한다.

2-Round가 순수 델파이보다 나은 이유:
- **마이너리티 보존**: 독립 투표로 확보, 이후 삭제 불가
- **논거 품질**: Round 2에서 보강 (델파이와 동일 효과)
- **약한 모델 적합성**: Round 1은 독립이라 conformity bias 없음
- **감사 적합성**: 모든 의견 보존, "왜 소수 의견이 사라졌나" 설명 가능

| 환경 | 모델 | R1 | R2 | 합계 | 항목당 소요 |
|---|---|---|---|---|---|
| AWS | Sonnet | 3 (병렬) | -- | 3 | ~5초 |
| 온프렘 기본 | 14B Q4 | 5×15초 | 2×20초 | 7 | ~2분 |
| 온프렘 고위험 | 14B Q4 | 7×15초 | 2×20초 | 9 | ~2.5분 |

입력 ~1,000 토큰, 출력 500~800 토큰(논거는 풍부할수록 좋다). 14B Q4 on RTX 4070에서 건당 30~40초.
WARN/FAIL만 합의 실행 (보통 5~10개) → **~45분이면 끝**. 점검 직후 바로 실행 가능.

출력은 구조화 판정(`verdict`+`confidence`, ~10토큰)과 자유 논거(`reasoning` 300~600토큰 + `recommendation` 50~100토큰)로 분리. 풍부한 논거가 케이스 스토어 유사 검색 품질을 높인다.

#### 최종 분류 (양 환경 공통)

| 분류 | AWS (3명) | 온프렘 (5명) | 처리 |
|---|---|---|---|
| **Consensus** (만장일치) | 3/3 일치 | 5/5 일치 | 통과 또는 확정 WARN |
| **최우선 리뷰** (다수) | 2/3 이상 | 3/5 이상 | 즉시 담당자 리뷰 |
| **마이너리티 리포트** (소수) | 1/3 이상 | 1~2/5 이상 | 2순위 리뷰 리스트 |

#### 적용 범위 (고위험 판정에만)

| 상황 | AWS | 온프렘 | 이유 |
|---|---|---|---|
| 체크리스트 WARN/FAIL 해석 | O | O | 잘못된 해석 → 잘못된 대응 |
| 변경 영향도 리뷰 | O | O | 누락 시 장애 위험 |
| 규제 적합성 판단 | O | O | 오판 시 규제 리스크 |
| 일상 대화 (질문 응답) | -- | N/A | 저위험, 비용 불필요 |

---

## 파이프라인 파트 분류 및 점검 체크리스트

에이전트가 파이프라인을 체계적으로 점검하기 위해 6개 파트로 분류한다.

### 6개 파트

```
P1 인제스천 → P2 피처 엔지니어링 → P3 학습 & 증류 → P4 서빙 & 추천 → P5 추천사유 생성
                                                                          ↑
                                        P6 모니터링 & 거버넌스 (전 파트 감시) ─┘
```

| 파트 | 이름 | 범위 | 주요 입력 | 주요 출력 |
|---|---|---|---|---|
| P1 | 인제스천 | IngestionRunner | 원천 데이터 (S3/DB) | 정제된 Parquet + manifest |
| P2 | 피처 엔지니어링 | Phase 0 Stage 1~9 | 정제된 Parquet | features/labels.parquet + schema |
| P3 | 학습 & 증류 | 교사 학습 → 분석 → 증류 | 학습 데이터 + config | ple_model.pt + LGBM 학생 |
| P4 | 서빙 & 추천 | FeatureStore → Scorer → TopK | 고객 ID + 모델 | 추천 결과 (top-K) |
| P5 | 추천사유 생성 | Template → LLM → SelfChecker | 추천 + IG 기여도 | 사유 텍스트 |
| P6 | 모니터링 & 거버넌스 | 드리프트/공정성/규제/감사 | 전 파트 측정값 | 진단 리포트 |

### P1. 인제스천 체크리스트

| # | 에이전트 | 점검 항목 | 판정 |
|---|---|---|---|
| 1.1 | Ops | 도메인별 row count가 이전 배치 대비 ±20% 이내 | PASS/WARN |
| 1.2 | Ops | validation 경고 (스키마 불일치, null 비율 초과) 없음 | PASS/FAIL |
| 1.3 | Ops | 인제스천 소요시간이 이전 대비 2× 미만 | PASS/WARN |
| 1.4 | Ops | 전체 도메인 성공 로드 (누락 없음) | PASS/FAIL |
| 1.5 | Audit | PII 컬럼 암호화/삭제 처리 완료 | PASS/FAIL |
| 1.6 | Audit | 데이터 보존 기간 정책 준수 | PASS/WARN |
| 1.7 | Audit | 감사 로그에 인제스천 이벤트 기록 | PASS/FAIL |

### P2. 피처 엔지니어링 체크리스트

| # | 에이전트 | 점검 항목 | 판정 |
|---|---|---|---|
| 2.1 | Ops | zero-variance 컬럼 0개 | PASS/WARN |
| 2.2 | Ops | NaN 비율이 threshold 미만 | PASS/WARN |
| 2.3 | Ops | 피처 수가 config 기준 예상 범위 내 | PASS/WARN |
| 2.4 | Ops | 정규화가 TRAIN split에서만 fit (리키지 검증) | PASS/FAIL |
| 2.5 | Ops | leakage_report.json PASS | PASS/FAIL |
| 2.6 | Ops | 스테이지별 소요시간 정상 범위 | PASS/WARN |
| 2.7 | Audit | 피처 드리프트 (PSI) critical 미만 | PASS/WARN/FAIL |
| 2.8 | Audit | 멱법칙 피처 log1p 처리 정상 | PASS/WARN |
| 2.9 | Audit | 데이터 계보 — 피처→원천 추적 가능 (미매핑 < 5%) | PASS/WARN |

### P3. 학습 & 증류 체크리스트

| # | 에이전트 | 점검 항목 | 판정 |
|---|---|---|---|
| 3.1 | Ops | 전 태스크 val metric 수렴 (최근 3 epoch 변동 < 1%) — binary: AUC, multiclass: F1-macro, regression: MAE (태스크 유형별 지표 집계, 13개 태스크) | PASS/WARN |
| 3.2 | Ops | NaN/Inf loss 미발생 | PASS/FAIL |
| 3.3 | Ops | grad norm이 clip 임계값 10× 미만 | PASS/WARN |
| 3.4 | Ops | 증류 fidelity gap 태스크별 5% 미만 | PASS/WARN |
| 3.5 | Ops | GPU 메모리 OOM 위험 없음 | PASS/WARN |
| 3.6 | Ops | 학습 비용 예산 가드 이내 | PASS/WARN |
| 3.7 | Audit | 리트레이닝 전후 공정성 지표 악화 없음 | PASS/WARN/FAIL |
| 3.8 | Audit | 증류 후 설명 재료 피처 보존 (IG top-K 중복률) | PASS/WARN |
| 3.9 | Audit | 실험 파라미터 감사 로그 기록 | PASS/FAIL |

### P4. 서빙 & 추천 체크리스트

| # | 에이전트 | 점검 항목 | 판정 |
|---|---|---|---|
| 4.1 | Ops | feature store health_check 정상 | PASS/FAIL |
| 4.2 | Ops | 추천 p95 latency SLA 이내 | PASS/WARN/FAIL |
| 4.3 | Ops | filter 통과율 정상 범위 | PASS/WARN |
| 4.4 | Ops | kill switch 비활성 | PASS/FAIL |
| 4.5 | Ops | A/B variant 할당 균등 | PASS/WARN |
| 4.6 | Audit | 단일 보호속성 DI/SPD/EOD 임계값 이내 | PASS/WARN/FAIL |
| 4.7 | Audit | 교차 보호속성 DI 임계값 이내 | PASS/WARN/FAIL |
| 4.8 | Audit | 추천 집중도 (HHI/Gini) herding 미만 | PASS/WARN |
| 4.9 | Audit | 편향 Stage Attribution (정보) | 정보 |
| 4.10 | Audit | 추천 결과 감사 아카이브 기록 | PASS/FAIL |

### P5. 추천사유 생성 체크리스트

| # | 에이전트 | 점검 항목 | 판정 |
|---|---|---|---|
| 5.1 | Ops | 사유 생성 latency가 전체 SLA 미초과 | PASS/WARN |
| 5.2 | Ops | L2a SQS 큐 깊이 정상 (백로그 없음) | PASS/WARN |
| 5.3 | Ops | DynamoDB reason_cache hit rate 정상 | PASS/WARN |
| 5.4 | Audit | Tier 1: SelfChecker pass rate ≥ 95% | PASS/WARN |
| 5.5 | Audit | Tier 1: reject/revise 비율 추이 악화 없음 | PASS/WARN |
| 5.6 | Audit | Tier 2: 품질 점수 (faithfulness, grounding) 추이 | PASS/WARN |
| 5.7 | Audit | Tier 2: cross-method 일관성 (SHAP vs IG) | PASS/WARN |
| 5.8 | Audit | Tier 3: 전문가 리뷰 대기 건수 적정 | PASS/WARN |
| 5.9 | Audit | 금지 패턴 (부적절 조언, injection) 없음 | PASS/FAIL |
| 5.10 | Audit | AI 공시 문구 포함 | PASS/FAIL |

### P6. 모니터링 & 거버넌스 체크리스트

| # | 에이전트 | 점검 항목 | 판정 |
|---|---|---|---|
| 6.1 | Ops | 감사 로그 해시 체인 무결 (verify_chain) | PASS/FAIL |
| 6.2 | Ops | 인시던트 알림 (SNS) 정상 작동 | PASS/FAIL |
| 6.3 | Ops | 에이전트 자체 스케줄 실행 (watchdog) | PASS/FAIL |
| 6.4 | Audit | 국내 규제 **36항목** (A-group 18 + GAP-group 18, CLAUDE.md §1.11) critical failure 없음 | PASS/FAIL |
| 6.5 | Audit | EU AI Act compliance rate 목표 이상 | PASS/WARN |
| 6.6 | Audit | FRIA 종합 리스크 HIGH 미만 | PASS/WARN/FAIL |
| 6.7 | Audit | 거버넌스 리포트 주기 생성 | PASS/WARN |
| 6.8 | Audit | 감사 패키지 외부 제출 요건 충족 | PASS/WARN |

> **체크리스트 운영**: YAML config로 관리하여 항목 추가/수정/비활성화 가능. 룰 엔진이 자동 판정하고, WARN/FAIL인 항목만 Sonnet 대화 인터페이스에 전달하여 담당자와 해석/대응을 논의.

### 변경 영향도 리뷰 (Impact Review)

코드/설정/모델/데이터 소스 변경 시 파이프라인 영향을 추론하고 담당자와 논의하는 기능.
이 기능이 **Sonnet이 필요한 핵심 이유**이다.

#### 변경 감지 메커니즘

영향도를 리뷰하려면 먼저 변경 발생 사실을 감지해야 한다. 두 가지 채널:

**Push 채널 (이벤트 기반, 즉시 감지)**
- 코드/설정 변경: `git post-commit` hook → 변경 파일 + diff 추출
- 파이프라인 완료: `_PipelineState.mark_complete()` 콜백 → 이벤트 발행 — **학습(train), 평가(eval), 증류(distill), 서빙(serving), 추천사유(reason) 모든 스테이지에서 발행**
- 인제스천 완료: `IngestionRunner` 종료 → manifest 발행

**Pull 채널 (주기적 비교, 지연 감지)**
- 데이터 소스 변동: 인제스천 manifest 이전 배치 대비 diff
- 서빙 지표 변동: CloudWatch/audit_archive 주기 폴링
- 상태 변화 탐지: 체크리스트 정기 실행 → 이전 판정과 비교 (PASS→WARN 전이)

| 변경 유형 | 채널 | 감지 소스 | 전달 정보 |
|---|---|---|---|
| 코드 변경 | Push | `git post-commit` hook | 변경 파일, diff, 영향 파트(P1~P6) |
| 설정 변경 | Push | `git post-commit` (YAML) | 변경 config 키, 이전→신규값 |
| 모델 변경 | Push | `_PipelineState` 완료 이벤트 | val metric, 이전 모델 대비 변동 |
| 데이터 소스 변경 | Pull | 인제스천 manifest diff | 스키마 변경, 볼륨 변동 |
| 서빙 지표 변동 | Pull | CloudWatch/audit_archive | latency 추세, CTR 변동 |
| 규제 변경 | 수동 | 담당자 입력 | 변경 규제 항목, 신규 요구사항 |

**온프렘 vs AWS 변경사항 관리의 구조적 차이**

| 변경 대상 | 온프렘 | AWS |
|---|---|---|
| 코드/설정 | 로컬 git → 사내 서버, `post-commit` hook | GitHub/CodeCommit + CloudTrail API 추적 |
| 모델 | 로컬 체크포인트, **버전 관리 수동** | SageMaker → S3, **Model Registry 자동 버전 관리** |
| 데이터 | DuckDB 파일, **manifest 수동 비교** | S3 + **버전관리/이벤트 알림 자동**, Glue Catalog |
| 배포 | Docker 직접 배포, **이력은 사내 CI/CD만** | SageMaker Endpoint, **CloudTrail 자동 추적** |

핵심 차이는 **추적 가능성(traceability)**: AWS는 자동으로 이력이 남지만, 온프렘은 의도적으로 기록하지 않으면 누락.

**온프렘 보완 전략**: 모델/배포 등 자동 추적이 불가능한 영역은 에이전트의 Action 도구(`log_audit_event`)를 학습/증류 완료 시, 배포 스크립트에 내장하여 변경 이력을 기록. AWS처럼 *자동*이 아니라 *규약 기반*.

> **변경 감지 → 영향도 리뷰 연결**: 온프렘은 변경 감지 시 영향 받는 파트의 체크리스트만 재실행. AWS는 동일한 재판정 + Sonnet이 diff를 읽고 맥락 설명 + 대화.

| 변경 유형 | 예시 | 잠재 영향 범위 |
|---|---|---|
| 코드 변경 | constraint_engine 필터 추가 | 공정성(DI), 추천 다양성, 사유 품질, 규제 |
| 설정 변경 | 태스크 가중치, 그룹 변경 | 모델 성능, 피처 라우팅, 증류 fidelity |
| 모델 변경 | 리트레이닝, 학생 모델 교체 | 서빙 지표, 사유 품질(IG 변동), 편향 |
| 데이터 소스 변경 | 인제스천 스키마 변경 | 피처 엔지니어링, 정규화, 모델 입력 |
| 규제 변경 | 금감원 가이드라인 개정 | compliance_rules, 감사 항목 |

---

## 1. 운영 에이전트 (Ops Agent)

### 1.1 관찰 지점 — 7개 체크포인트

| # | 체크포인트 | 수집 소스 | 측정 항목 |
|---|---|---|---|
| CP1 | 인제스천 완료 | `IngestionRunner.generate_manifest()` | 도메인별 row count 변동률, PII 암호화 누락, validation 경고 |
| CP2 | Phase 0 완료 | `pipeline_state.json` + `feature_stats.json` | 스테이지별 소요시간, zero-variance 컬럼 수, NaN 비율 분포 |
| CP3 | 학습 완료 | `ExperimentTracker` (metrics.jsonl) | loss 수렴 여부, grad norm 이상, epoch별 val metric 추이 |
| CP4 | 증류 완료 | `DistillationValidator` fidelity 결과 | 태스크별 teacher-student AUC gap, fidelity 임계값 위반 |
| CP5 | 서빙 헬스 | `FeatureStore.health_check()` + CloudWatch | feature store 응답시간, 레코드 수 정합성, kill switch 상태 |
| CP6 | 추천 응답 | `audit_archiver` (Parquet traces) | p50/p95 latency, filter 통과율, top-K 다양성 지표 |
| CP7 | A/B 테스트 | `ABTestManager` CloudWatch metrics | variant별 CTR/CVR, significance test 결과, auto-promote 판단 |

### 1.2 진단 로직

#### 시계열 이상 탐지

```
인제스천 row count가 이전 대비 ±20%
  → "데이터 소스 이상 의심"

Phase 0 소요시간이 이전 대비 2x
  → "데이터 볼륨 급증 또는 generator 병목"

서빙 p95 latency > SLA (예: 200ms)
  → "feature store 또는 reason 생성 병목"
```

#### 연쇄 영향 분석 (Cross-Checkpoint Correlation)

운영 에이전트의 핵심 가치는 단일 지표 알림이 아니라 **체크포인트 간 상관관계 분석**에 있다:

```
zero-variance 컬럼 증가 (CP2) + 특정 태스크 AUC 하락 (CP3)
  → "피처 품질 저하가 모델 성능에 영향"

drift PSI critical 3일 연속 (CP2) + 서빙 CTR 하락 (CP7)
  → "리트레이닝 필요"

teacher-student fidelity gap > 5% (CP4) + 해당 태스크 서빙 지표 하락 (CP6)
  → "증류 품질 문제"

인제스천 도메인 row count 급감 (CP1) + 해당 도메인 유래 피처 NaN 증가 (CP2)
  → "업스트림 데이터 소스 장애"
```

#### 비용 감시

```
SageMaker billable time vs 예산 가드 (pipeline.yaml ablation.budget_limit)
Spot 중단율 추적 → 연속 중단 시 on-demand 전환 권고
Phase 0 CPU 인스턴스 vs GPU 인스턴스 오사용 감지
```

### 1.3 리포트 포맷

```yaml
ops_report:
  generated_at: "2026-04-10T09:00:00Z"
  period: "daily"  # daily | weekly
  
  status: YELLOW  # GREEN / YELLOW / RED
  
  # 담당자가 봐야 할 것만, 우선순위 순
  attention_required:
    - checkpoint: CP3
      severity: WARNING
      finding: >
        churn_signal 태스크 val_auc가 3일 연속 하락 (0.82→0.79→0.76)
      likely_cause: >
        CP2에서 tenure_months 컬럼 NaN 비율이 12%→23% 증가한 것과 상관
      suggested_action: >
        인제스천 도메인 customer_master의 tenure 필드 품질 확인
      
    - checkpoint: CP6
      severity: INFO
      finding: >
        추천 응답 p95 latency 180ms→240ms (SLA 300ms 이내이나 추세 상승)
      likely_cause: >
        context_store 벡터 수 50K→80K 증가
      suggested_action: >
        context_store 인덱스 재빌드 또는 top-K retrieval 수 축소 고려
  
  # 전체 현황 요약
  all_checkpoints:
    CP1: {status: GREEN, rows: 941132, delta: "+0.2%"}
    CP2: {status: GREEN, duration: "4m32s", zero_var: 0}
    CP3: {status: YELLOW, detail: "1/13 tasks degrading"}
    CP4: {status: GREEN, max_fidelity_gap: "2.1%"}
    CP5: {status: GREEN, latency_p50: "12ms"}
    CP6: {status: YELLOW, detail: "p95 trending up"}
    CP7: {status: GREEN, detail: "no active experiment"}
```

핵심: **finding + likely_cause + suggested_action** 세트.
단순 알림이 아니라 체크포인트 간 상관관계를 분석해서 원인 추정까지 제공.

### 1.4 실행 주기

| 체크포인트 | 트리거 | 주기 |
|---|---|---|
| CP1~CP4 | 이벤트 기반 (각 스테이지 완료 시) | 배치당 1회 |
| CP5 서빙 헬스 | 주기적 | 5분 |
| CP6 추천 응답 | 주기적 (1시간 집계) | 상시 |
| CP7 A/B 테스트 | 주기적 | 일 1회 |

---

## 2. 감사 에이전트 (Audit Agent)

### 2.1 관찰 지점 — 5개 관점

| # | 관점 | 수집 소스 | 측정 항목 |
|---|---|---|---|
| AV1 | 공정성 | `FairnessMonitor` | 보호속성별 DI/SPD/EOD, 위반 추이 |
| AV2 | 집중도 | `HerdingDetector` | HHI/Gini/Entropy, 태스크별 기여도 편중 |
| AV3 | 추천사유 품질 | `SelfChecker` + `XAIQualityEvaluator` | 사유 통과율, faithfulness, stability, 샘플 심층검토 |
| AV4 | 규제 적합성 | `RegulatoryComplianceChecker` + `KoreanFRIAAssessor` (AI기본법 §35, 7-차원) + `FRIAEvaluator` (EU AI Act Art.9, 5-차원) + `AnnexIVMapper` (Art.11 tech doc) + `SuitabilityFilter` (KFCPA §17) + `AISecurityChecker` + `PromptSanitizer` + `wrap_provider` + `ComplianceSQLHelper` (S6 DuckDB httpfs) + `SageMakerComplianceTracker` (S5) | **36개 국내 규제 (A-18 + GAP-18)** + EU AI Act Art.9/11/13/14 + PIPA §37의2 + KFCPA §17 + Credit Info Act §36의2 + SR 11-7 준수율. 한국 FRIA 와 EU FRIA 는 법적 기반이 달라 별도 class·저장 (CLAUDE.md §1.11) |
| AV5 | 데이터 계보 | `DataLineageTracker` | 추천→피처→원천 추적 가능 여부, 미매핑 피처 비율 |

### 2.2 추천사유 품질 검증 전략 (AV3 상세)

전수조사가 불가능하므로 **3-Tier 샘플링** 전략을 사용한다.

#### Tier 1: 전수 자동검증 (100% — 모든 추천에 적용)

```
├── SelfChecker 통과율 모니터링 (reject / revise / pass 비율 추이)
├── 금지 패턴 탐지 (compliance_rules 위반)
└── prompt injection 탐지
```

기존 `SelfChecker`가 이미 모든 추천사유에 대해 실행되므로, 여기서는 **집계와 추이 분석**만 추가.

#### Tier 2: 통계적 샘플링 + 자동 심층검증 (일 1~5%)

```
├── 층화추출: 태스크유형 × 고객세그먼트 × 사유레이어(L1/L2a/L2b) 조합별 균등
├── XAI 품질 평가 (faithfulness, stability, comprehensibility)
├── Grounding 검증: 사유 텍스트가 IG top features와 실제로 일치하는지
└── Cross-method 일관성: SHAP vs IG 상위 피처 rank correlation
```

**층화추출 설계:**

```yaml
stratification_axes:
  task_type: [binary, multiclass, regression]     # 3가지
  customer_segment: [mass, affluent, vip]          # 3가지
  reason_layer: [L1, L2a, L2b]                    # 3가지
  # → 27개 스트라텀, 각 10~20건 = 일 270~540건
```

**우선 추출 조건 (경계선 + 고위험):**

```yaml
priority_sampling:
  - condition: "selfcheck_confidence BETWEEN 0.6 AND 0.8"
    weight: 3x   # 과표집 — 경계선 사례
  - condition: "customer_segment IN ('elderly', 'low_income')"
    weight: 2x   # 과표집 — 보호계층
  - condition: "reason_layer = 'L2b' AND human_review_flagged = True"
    weight: ALL  # 전수 — L2b 플래그 건
```

#### Tier 3: 주기적 전문가 리뷰 (월 50~100건)

```
├── Tier 2에서 경계선 사례 (confidence 0.6~0.8) 우선 추출
├── 고위험 세그먼트 과표집 (elderly, low-income 등)
├── human_review_flagged=True인 L2b 결과 전수 포함
├── 리뷰 결과를 피드백 루프로 Tier 1/2 규칙 업데이트
```

#### 품질 점수 체계

```
품질 = 0.30 × Faithfulness
     + 0.25 × Grounding
     + 0.25 × Compliance
     + 0.20 × Readability
```

| 차원 | 측정 방법 | 소스 |
|---|---|---|
| Faithfulness | IG attribution과 perturbation 결과의 상관 | `XAIQualityEvaluator` |
| Grounding | 사유 텍스트에 언급된 피처가 IG top-K에 포함되는 비율 | `ReverseMapper` + IG |
| Compliance | SelfChecker 통과 + 금지패턴 미검출 | `SelfChecker` |
| Readability | 문장 길이, 전문용어 비율, 모호한 표현 비율 | 규칙 기반 |

#### 피드백 루프

- 전문가 reject인데 Tier 1/2에서 pass → 새 `compliance_rule` 추가
- 전문가 accept인데 Tier 1/2에서 revise → 과도한 규칙 완화 검토
- 전문가 간 일치율(inter-rater agreement) 추적 → 규칙 신뢰도 지표

### 2.3 편향 감지 심화 (AV1 상세)

운영 에이전트가 보지 않는 세 가지 분석:

#### 교차 보호속성 분석

단일 속성은 통과하지만 교차(intersection)에서 위반 발생하는 케이스 탐지:

```
예: age_group별 DI = 0.85 (통과)
    income_tier별 DI = 0.88 (통과)
    age_group=elderly ∩ income_tier=low → DI = 0.62 (위반)
```

#### 편향 발생 단계 분리 (Stage Attribution)

```
Stage 1: 모델 output logit → 모델 자체의 편향
Stage 2: constraint_engine 필터링 후 → 비즈니스 룰 영향
Stage 3: top-K 선택 후 → diversity method 영향
→ 어느 단계에서 편향이 발생/증폭되는지 분리 진단
```

#### 시계열 편향 추이

- 공정성 지표의 추세 (악화 중인가, 개선 중인가)
- 리트레이닝 전후 편향 변화 (의도치 않은 악화 감지)
- 계절성 패턴 (특정 시기에 반복되는 편향)

### 2.4 감사 리포트 포맷

```yaml
audit_report:
  generated_at: "2026-04-10T09:00:00Z"
  period: "weekly"
  risk_level: MEDIUM

  focus_areas:
    - area: "추천사유 품질"
      priority: HIGH
      finding: |
        L2b 사유 중 grounding < 0.7 비율이 18%→24%로 증가.
        nba_primary 태스크에서 IG top-3 불일치 집중 발생.
      evidence:
        tier2_sample_size: 540
        grounding_below_threshold: 129
        worst_task: "nba_primary"
      recommended_review: |
        nba_primary의 L2b 사유 중 grounding < 0.5인 32건을
        Tier 3 전문가 리뷰 대상에 추가 권고.

    - area: "교차속성 공정성"
      priority: MEDIUM
      finding: |
        elderly ∩ low_income 그룹의 DI = 0.68 (기준 0.80).
        단일 속성은 모두 통과, 교차에서 위반.
      evidence:
        subgroup: "elderly ∩ low_income"
        pipeline_stage_attribution: "constraint_engine에서 악화"
      recommended_review: |
        constraint_engine eligibility 필터의 비례성 점검 권고.

  regulatory_summary:
    domestic:
      financial_consumer_protection: {pass: 4, fail: 1}
      personal_info_protection: {pass: 6, fail: 0}
      ai_basic_act: {pass: 8, fail: 1}
    eu_ai_act:
      risk_classification: "HIGH"
      compliance_rate: 0.82

  reason_quality_dashboard:
    tier1_auto:
      total_checked: 142000
      pass_rate: 0.96
    tier2_sample:
      sample_size: 540
      avg_quality_score: 0.74
      trend: "grounding declining"
    tier3_expert:
      pending_review: 82
      agreement_with_auto: 0.89
```

### 2.5 실행 주기

| 관점 | 트리거 | 주기 |
|---|---|---|
| AV1 공정성 | 주기적 + Ops 트리거 | 일 1회 (drift 시 즉시) |
| AV2 집중도 | 주기적 | 일 1회 |
| AV3 추천사유 | Tier 1: 실시간, Tier 2: 일 1회, Tier 3: 월 1회 | 혼합 |
| AV4 규제 적합성 | 주기적 | 주 1회 (분기 1회 전체) |
| AV5 데이터 계보 | 이벤트 기반 (모델 변경 시) | 변경당 1회 |

---

## 3. 두 에이전트 간 관계

### 3.1 상호 트리거 (EventBridge 기반)

AWS 환경에서 Ops ↔ Audit 에이전트 간 통신은 **EventBridge**를 통해 이루어진다. 각 에이전트는 Lambda로 배포되며 상시 서버 없이 이벤트 발생 시에만 실행된다.

| 조건 | 발신 | 수신 | 트리거 내용 |
|---|---|---|---|
| drift PSI critical 3일 연속 | Ops | Audit | 모델 성능 저하 구간의 추천사유 품질 집중 점검 |
| 특정 세그먼트 편향 발견 | Audit | Ops | 해당 세그먼트 서빙 지표 모니터링 강화 |
| 서빙 latency SLA 초과 | Ops | Audit | latency 문제 구간의 사유 생성 skip 여부 확인 |
| 규제 critical failure | Audit | Ops | 해당 규제 관련 파이프라인 스테이지 상태 즉시 확인 |

### 3.2 거버넌스 리포트 통합

기존 `GovernanceReportGenerator`의 9개 섹션에 두 에이전트 결과를 공급:

| 거버넌스 리포트 섹션 | 데이터 소스 |
|---|---|
| fairness_summary | Audit AV1 |
| drift_summary | Ops CP2 |
| incident_summary | 양쪽 공유 (IncidentReporter) |
| model_changes | Ops CP3, CP4 |
| kill_switch_history | Ops CP5 |
| recommendation_quality | Audit AV3 |
| herding_summary | Audit AV2 |
| audit_summary | Audit AV4 |
| executive_summary | 양쪽 attention_required / focus_areas 통합 |

### 3.3 AWS 서버리스 배포 아키텍처

에이전트는 **상시 가동 서버 없이** Lambda로 배포된다. EventBridge 스케줄러가 주기적으로 호출하고, 에이전트간 통신도 EventBridge를 통한다.

#### CloudFormation 스택 구성 요소

**EventBridge 스케줄**

| 스케줄 이름 | 주기 | 대상 Lambda |
|---|---|---|
| `ple-ops-daily` | 일 1회 (09:00 KST) | `OpsAgent` 전체 체크리스트 실행 |
| `ple-audit-daily` | 일 1회 (09:30 KST) | `AuditAgent` 전체 체크리스트 실행 |
| `ple-heartbeat-5min` | 5분 | `OpsAgent` 헬스 확인 (CP5 서빙 헬스만) |

**Lambda 헬스 엔드포인트**: OpsAgent Lambda는 `{"action": "heartbeat"}` 페이로드를 받으면 추론 없이 헬스 체크만 수행하고 반환한다. 5분 주기 EventBridge가 이 페이로드로 호출하여 Lambda 자체의 가용성을 확인한다.

```json
// 헬스 확인 요청
{"action": "heartbeat"}

// 헬스 확인 응답
{"status": "ok", "timestamp": "2026-04-15T09:00:00Z", "agent": "ops"}
```

**CloudWatch 알람**

| 알람 이름 | 지표 | 임계값 | 알림 대상 |
|---|---|---|---|
| `ple-ops-error-rate` | Lambda 에러율 | > 5% (5분 윈도우) | `ple-ops-alerts` SNS |
| `ple-audit-error-rate` | Lambda 에러율 | > 5% (5분 윈도우) | `ple-audit-alerts` SNS |
| `ple-ops-duration` | Lambda 실행시간 | > 13분 (max 15분) | `ple-ops-alerts` SNS |
| `ple-heartbeat-missing` | Lambda 호출 횟수 | < 1 (10분 윈도우) | `ple-heartbeat-alerts` SNS |
| `ple-consecutive-errors` | 연속 에러 횟수 | >= 3 | `ple-ops-alerts` SNS |

**SNS 토픽**

| 토픽 | 구독 | 용도 |
|---|---|---|
| `ple-ops-alerts` | Email/Slack | 운영 에이전트 에러, 성능 저하 알림 |
| `ple-audit-alerts` | Email/Slack | 감사 에이전트 에러, 규제 위반 알림 |
| `ple-heartbeat-alerts` | PagerDuty/Email | Lambda 응답 없음 (긴급) |

**DynamoDB 규제 준수 테이블**

| 테이블 이름 | 키 | 용도 |
|---|---|---|
| `ple-audit-log` | `audit_id` (파티션), `timestamp` (정렬) | 감사 이벤트 로그 (7년 보존) |
| `ple-consent-records` | `customer_id`, `consent_type` | AI 추천 동의 여부 기록 |
| `ple-opt-out` | `customer_id` | 고객 거부 의사 기록 (즉시 반영) |
| `ple-profiling-rights` | `customer_id`, `request_id` | 프로파일링 열람/정정 요청 추적 |

#### 체크포인트-에이전트 분담

OpsAgent와 AuditAgent는 체크포인트를 분담하여 수집한다:

| 체크포인트 | 수집 에이전트 | 근거 |
|---|---|---|
| CP1 인제스천 | **OpsAgent** | 운영 지표 (row count, 소요시간, PII 처리) |
| CP2 Phase 0 | **OpsAgent** | 운영 지표 (피처 통계, 정규화, 리키지) |
| CP3 학습 | **OpsAgent** | 운영 지표 (loss, grad norm, GPU 메모리) |
| CP4 증류 | **OpsAgent** | 운영 지표 (fidelity gap, 비용) |
| CP5 서빙 헬스 | **OpsAgent** | 운영 지표 (latency, kill switch, feature store) |
| CP6 추천 응답 품질 | **AuditAgent** | 감사 지표 (공정성 DI/SPD/EOD, 추천사유 품질) |
| CP7 A/B 테스트 | **OpsAgent** | 운영 지표 (CTR/CVR, significance) |

#### 진단 및 리포트 컴포넌트

- **Diagnoser** (OpsAgent/AuditAgent 각각 보유): 수집된 체크포인트 측정값을 룰 테이블로 교차 상관 분석. LLM 없이 결정론적으로 동작. CP2 NaN 증가 + CP3 AUC 하락 → "피처 품질 저하" 등 연쇄 인과 패턴을 매핑.
- **Reporter** (OpsAgent/AuditAgent 각각 보유): Diagnoser 결과를 `finding + likely_cause + suggested_action` 구조의 JSON 리포트로 생성. SNS/Slack/Email 전달.
- **BedrockDialogSession**: 담당자가 진단 리포트를 두고 Claude Sonnet과 대화하는 인터페이스. "이 DI 위반이 필터 문제인지 모수 문제인지"를 에이전트와 논의. 세션당 ~$0.01.
- **BudgetTracker**: BedrockDialogSession 및 ConsensusArbiter의 LLM 호출 비용을 토큰 단위로 추적. 월간 소프트 경고(80%) 및 하드 정지(100%) 임계값. 예산 초과 시 LLM 호출만 차단하고 룰 엔진은 계속 작동.

---

## 4. 기존 컴포넌트 재사용 매핑

### 4.1 재사용 (기존)

| 컴포넌트 | 에이전트 | 역할 |
|---|---|---|
| `DriftDetector` / `PSICalculator` | Ops | 피처 드리프트 감지 |
| `FairnessMonitor` | Audit | 단일 속성 공정성 |
| `HerdingDetector` | Audit | 추천 집중도 |
| `SelfChecker` | Audit | 사유 자동검증 (Tier 1) |
| `XAIQualityEvaluator` | Audit | 설명 품질 평가 (Tier 2) |
| `RegulatoryComplianceChecker` | Audit | 국내 규제 **36항목** (A-18 + GAP-18, CLAUDE.md §1.11) |
| `KoreanFRIAAssessor` | Audit | **AI기본법 §35** FRIA (7-차원, 5-년 retention) — `core/compliance/fria_assessment.py` |
| `FRIAEvaluator` | Audit | **EU AI Act Art.9** FRIA (5-차원) — `core/monitoring/fria_evaluator.py`. 한국 FRIA 와 별도 class·저장 |
| `AnnexIVMapper` | Audit | **EU AI Act Art.11** 12 섹션 tech doc 증거 resolve + coverage 자동 추적 |
| `SuitabilityFilter` | Audit | **KFCPA §17** 적합성 원칙, ≥65세 / <30M KRW hard cap |
| `AISecurityChecker` + `PromptSanitizer` + `wrap_provider` | Audit | prompt injection 14 패턴 + output leak 8 패턴 catalog, L2aSafetyGate 와 결합하여 LLM 경로 4중 방어 (CLAUDE.md §1.17) |
| `ComplianceSQLHelper` (DuckDB httpfs) | Audit | S3 Parquet 배치 SQL 조회 — Athena 비용 회피 (S6, CLAUDE.md §1.15) |
| `SageMakerComplianceTracker` | Audit | 4개 규제 산출물 (FRIA / AI Risk / Registry Sweep / Promotion Gate) TrialComponent 자동 기록 (S5, CLAUDE.md §1.14) |
| `MetadataAggregator` + PromotionGate | Ops | 6 evidence source composite, PromotionGate verdict per-dimension trail |
| `DataLineageTracker` | Audit | 데이터 계보 |
| `IncidentReporter` | 양쪽 | 긴급 에스컬레이션 |
| `GovernanceReportGenerator` | 양쪽 | 월간 통합 리포트 |
| `AuditPackageBuilder` | Audit | 외부 감사 패키지 |

### 4.2 신규 개발 필요

| 컴포넌트 | 에이전트 | 설명 |
|---|---|---|
| `OpsCollector` | Ops | 7개 체크포인트 측정값 수집기 |
| `OpsDiagnoser` | Ops | 연쇄 영향 분석 (cross-checkpoint) |
| `OpsReporter` | Ops | 리포트 생성 + 전달 (Slack/Email/SNS) |
| `StratifiedReasonSampler` | Audit | Tier 2 층화추출 + 우선순위 샘플링 |
| `GroundingValidator` | Audit | 사유 텍스트 ↔ IG top-K 일치율 검증 |
| `IntersectionalFairnessAnalyzer` | Audit | 교차 보호속성 분석 |
| `BiasStageAttributor` | Audit | 편향 발생/증폭 단계 분리 |
| `AuditDiagnoser` | Audit | focus_areas 생성 + 우선순위 판단 |
| `AuditReporter` | Audit | 리포트 생성 + 전달 |
| `AgentEventBridge` | 양쪽 | 상호 트리거 + GovernanceReport 연동 |
| `DiagnosticCaseStore` | 양쪽 | LanceDB 기반 진단 케이스 저장/검색/통계 |
| `ConsensusArbiter` | 양쪽 | 3-에이전트 합의 판정 + 마이너리티 분류 |
| `ChangeDetector` | 양쪽 | 변경 감지 Push/Pull 채널 + 표준 이벤트 포맷 |
| `ToolRegistry` | 양쪽 | 도구 정의(JSON Schema) 관리 + 호출 메커니즘 |
| `SendNotification` | 양쪽 | Slack/Email/SNS 리포트 전달 |

---

## 도구 호출 체계 (Tool Calling)

에이전트가 파이프라인을 점검하려면 각 컴포넌트를 **도구(tool)**로 호출할 수 있어야 한다.
온프렘에서는 Python 직접 호출, AWS에서는 Bedrock Tool Use 포맷으로 동일한 도구를 노출한다.

### 설계 원칙

- **Query/Action 분리**: 읽기 도구와 쓰기 도구를 명확히 구분. 감사 에이전트가 실수로 상태를 변경하는 것을 구조적으로 방지. Action은 명시적 승인 후 실행.
- **단일 인터페이스**: 온프렘 `ToolRegistry.call("tool_name", params)` → Python 직접 호출. AWS Bedrock `tool_use` 블록 → Lambda/직접 호출. 도구 정의(JSON Schema)는 동일.
- **최소 권한**: 각 에이전트는 필요한 도구만 접근.

### 범주 1: 인프라 도구 (Query) — 신규 개발

| 도구 이름 | 설명 | Ops | Audit |
|---|---|---|---|
| `read_pipeline_state` | pipeline_state.json — 스테이지별 완료/소요시간 | O | O |
| `read_feature_stats` | feature_stats.json — 피처별 mean/std/null/zero-var | O | O |
| `read_experiment_metrics` | metrics.jsonl — epoch별 loss, val_auc, grad_norm | O | -- |
| `read_ingestion_manifest` | 인제스천 manifest — 도메인별 row count, PII, validation | O | O |
| `read_leakage_report` | leakage_report.json — 리키지 검증 결과 | O | -- |
| `read_distillation_fidelity` | 태스크별 teacher-student fidelity gap | O | O |
| `query_cloudwatch_metrics` | CloudWatch — latency, A/B CTR (AWS 전용) | O | -- |
| `read_audit_archive` | audit_archiver Parquet — 추천 통계 집계 | O | O |
| `read_git_diff` | git diff — 변경 파일, 영향 파트 (AWS Bedrock 전용) | O | O |
| `read_checklist_config` | 체크리스트 YAML — 활성 항목, 임계값 | O | O |

### 범주 2: 모니터링 도구 (Query) — 기존 컴포넌트 래핑

| 도구 이름 | 래핑 대상 | Ops | Audit |
|---|---|---|---|
| `detect_drift` | `DriftDetector.detect_drift()` | O | O |
| `get_consecutive_drift_days` | `ConsecutiveDriftTracker` | O | O |
| `evaluate_fairness` | `FairnessMonitor.evaluate_all_attributes()` | -- | O |
| `detect_herding` | `HerdingDetector.detect_herding()` | -- | O |
| `detect_task_herding` | `HerdingDetector.detect_task_herding()` | -- | O |
| `check_feature_store_health` | `FeatureStore.health_check()` | O | -- |
| `evaluate_data_quality` | `QualityGate.evaluate()` | O | O |

### 범주 3: 규제 · 품질 도구 (Query) — 기존 컴포넌트 래핑

| 도구 이름 | 래핑 대상 | Ops | Audit |
|---|---|---|---|
| `run_regulatory_checks` | `RegulatoryComplianceChecker.run_all_checks()` | -- | O |
| `run_compliance_check` | `ComplianceChecker.run_full_check()` | -- | O |
| `evaluate_eu_ai_act` | `EUAIActMapper.generate_report()` | -- | O |
| `evaluate_fria` | `FRIAEvaluator.generate_report()` | -- | O |
| `check_reason_quality` | `SelfChecker.check()` | -- | O |
| `evaluate_xai_quality` | `XAIQualityEvaluator.evaluate_task()` | -- | O |
| `check_explanation_consistency` | `XAIQualityEvaluator.check_explanation_consistency()` | -- | O |
| `trace_feature_lineage` | `DataLineageTracker.trace_features_batch()` | -- | O |
| `generate_lineage_report` | `DataLineageTracker.generate_lineage_report()` | -- | O |
| `verify_audit_chain` | `AuditLogger.verify_chain()` | O | O |

### 범주 4: Action 도구 (상태 변경) — 승인 후 실행

| 도구 이름 | 래핑 대상 | Ops | Audit |
|---|---|---|---|
| `create_incident` | `IncidentReporter.create_incident()` | O | O |
| `log_audit_event` | `AuditLogger.log_operation()` | O | O |
| `archive_governance_report` | `GovernanceReportGenerator.archive_report()` | -- | O |
| `save_compliance_report` | `ComplianceChecker.save_report()` | -- | O |
| `save_lineage` | `DataLineageTracker.save_lineage()` | -- | O |
| `generate_governance_report` | `GovernanceReportGenerator.generate_report()` | O | O |
| `send_notification` | Slack/Email/SNS 전달 (신규) | O | O |

### 도구 정의 형식 (JSON Schema)

모든 도구는 동일한 JSON Schema로 정의되어 온프렘/AWS에서 공유:

```json
{
  "name": "evaluate_fairness",
  "description": "보호속성별 공정성 지표(DI/SPD/EOD) 평가",
  "category": "query",
  "agents": ["audit"],
  "parameters": {
    "type": "object",
    "properties": {
      "recommendations": {"type": "string", "description": "추천 결과 경로 또는 날짜"},
      "attributes": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["recommendations"]
  },
  "returns": "FairnessMetrics (속성별 DI/SPD/EOD, 위반 목록)"
}
```

### 체크리스트 ↔ 도구 매핑

| 체크 항목 | 호출 도구 | 판정 로직 |
|---|---|---|
| 1.1 row count 변동 | `read_ingestion_manifest` | `abs(delta) / prev < 0.20` |
| 2.5 리키지 PASS | `read_leakage_report` | `result["passed"] == True` |
| 3.4 fidelity gap | `read_distillation_fidelity` | `max(gap) < 0.05` |
| 4.2 p95 latency | `query_cloudwatch_metrics` | `p95_ms < sla` |
| 4.6 공정성 DI | `evaluate_fairness` | `all(di >= 0.80)` |
| 5.4 사유 pass rate | `read_audit_archive` (집계) | `pass / total >= 0.95` |
| 6.1 해시 체인 | `verify_audit_chain` | `result == True` |

### 온프렘 vs AWS 호출 방식

**온프렘**: 룰 엔진이 체크리스트를 순회하며 `ToolRegistry.call()` 직접 호출
```python
for item in checklist:
    result = registry.call(item.tool, item.params)
    verdict = judge(result, item.threshold)
```

**AWS**: 온프렘 엔진 그대로 + Sonnet이 대화 중 도구를 선택적 호출
```
담당자: "elderly ∩ low_income 실제 영향 봐줘"
Sonnet: [evaluate_fairness(attributes=["age_group","income_tier"])]
        → 결과 해석 → "DI 0.62, 모수 47건이라 필터 1개에도 크게 흔들림"
```

### 도구 수량 요약

| 범주 | Query | Action | Ops | Audit |
|---|---|---|---|---|
| 인프라 | 10 | 0 | 9 | 7 |
| 모니터링 | 7 | 0 | 4 | 5 |
| 규제·품질 | 10 | 0 | 1 | 10 |
| 케이스 스토어 | 2 | 2 | 2 | 2 |
| Action (기타) | 0 | 7 | 4 | 6 |
| **합계** | **29** | **9** | **20** | **30** |

> **부작용 처리**: `FairnessMonitor.evaluate_fairness()`와 `HerdingDetector.detect_herding()`은 `auto_incident=True`일 때 인시던트를 자동 생성한다. 에이전트 도구로 래핑할 때는 `auto_incident=False`로 고정하고, 인시던트 생성은 별도 Action 도구로만 수행하게 하여 Query/Action 경계를 유지.

---

## 진단 케이스 스토어 (Diagnostic Case Store)

점검 리포트는 일회성 산출물이 아니다. 누적되면 **운영 지식 베이스**가 되어 유사 케이스 참조, 통계 분석, 대응 효과 추적이 가능해진다. 추천사유의 `ContextVectorStore`(LanceDB)와 동일한 패턴을 적용.

### 케이스 스키마

```json
{
  "case_id": "OPS-2026-04-10-001",
  "timestamp": "2026-04-10T09:00:00Z",
  "agent": "ops",
  "pipeline_part": "P3",
  "check_item": "3.1",
  "verdict": "WARN",
  "severity": "WARNING",
  "finding": "churn_signal val_auc 3일 연속 하락 (0.82→0.79→0.76)",
  "likely_cause": "CP2 tenure_months NaN 비율 12%→23% 증가",
  "suggested_action": "인제스천 customer_master tenure 필드 품질 확인",
  "metrics": {"val_auc": 0.76, "nan_ratio_tenure": 0.23},
  "consensus_type": "majority",
  "consensus_detail": {"round1_votes": {"WARN": 4, "PASS": 1}, "minority_agents": ["③"]},
  "resolution": null,
  "resolved_at": null,
  "post_resolution_verdict": null,
  "vector": [0.12, -0.34, ...]
}
```

### 3가지 활용

**1. 유사 케이스 검색**: finding+cause를 임베딩하여 코사인 유사도 검색. 과거 대응 이력/해결 기간 참조.

**2. 통계 분석**: 구조화된 메타데이터로 파트별 빈번 WARN, 공정성 위반 추이, 평균 해결 시간 등 집계.

**3. 대응 효과 추적**: (문제, 대응, 후속 판정) 3-tuple로 케이스 완결. 비효과적 대응 패턴 식별.

### 케이스 스토어 도구 (추가 4개)

| 도구 이름 | 설명 | Query/Action |
|---|---|---|
| `search_similar_cases` | finding으로 유사 케이스 벡터 검색 | Query |
| `get_case_statistics` | 파트별/항목별/기간별 통계 집계 | Query |
| `save_case` | 진단 결과를 케이스로 저장 | Action |
| `update_case_resolution` | resolution/resolved_at 갱신 | Action |

### 규제기관 관점의 가치

케이스 스토어는 규제기관 감사 대응에도 직접 활용된다:

- **감사 증적**: "이 시스템은 공정성 위반을 몇 건 감지했고, 평균 N일 내 해결했다"를 정량적으로 제시
- **지속적 개선 증명**: 대응 효과 추적으로 "같은 유형의 문제가 반복 감소하고 있다" 추세 제시
- **AuditPackageBuilder 연동**: 분기 감사 패키지에 케이스 통계 섹션 자동 포함

### 히스토리 관리 체계

| 계층 | 저장소 | 보존 기간 | 용도 |
|---|---|---|---|
| 실시간 | LanceDB (DiagnosticCaseStore) | 영구 | 유사 검색, 통계, 대응 추적 |
| 감사 로그 | AuditLogger (HMAC chain, S3 Object Lock) | 7년 | 규제 감사 증적, 위변조 방지 |
| 거버넌스 리포트 | GovernanceReportGenerator (S3) | 영구 | 월/분기 거버넌스 위원회 보고 |
| 인시던트 | IncidentReporter (S3 + DynamoDB) | 영구 | 인시던트 이력, PIR |
| 체크리스트 판정 이력 | 케이스 스토어 메타데이터 | 영구 | PASS→WARN 전이 추적 |

**온프렘**: LanceDB 파일 + 로컬 감사 로그 (JSONL). 백업은 사내 정책.
**AWS**: LanceDB on S3 + DynamoDB + S3 Object Lock 7년. CloudTrail 자동 보존.

### 구현: `DiagnosticCaseStore`

| 구성 요소 | 설명 |
|---|---|
| 백엔드 | LanceDB (선호) / numpy fallback |
| 임베딩 | 온프렘: sentence-transformers (all-MiniLM-L6-v2), AWS: Bedrock Titan Embeddings V2 |
| 메타데이터 | case_id, timestamp, agent, pipeline_part, check_item, verdict, severity, metrics, resolution |
| 보존 | 전체 보존 (삭제 없음) — 감사 요구사항 |

---

## 5. 구현 우선순위

### 상세 구현 계획

#### 태스크 목록 (총 ~3,700 LOC / 신규 ~20개 파일 + 수정 ~5개 파일)

**Pre-req: 추천사유 4개 갭 수정 (~138 LOC)**

| ID | 태스크 | LOC | 의존 | 리스크 |
|---|---|---|---|---|
| P-1 | GAP 4: `reverse_mapper.py` 한국어 fallback | ~30 | 없음 | LOW |
| P-2 | GAP 3: `interpretation_registry.py` + YAML prefix 확장 | ~55 | 없음 | LOW |
| P-3 | GAP 2: InterpretationRegistry에 ReverseMapper fallback 통합 | ~30 | P-1 | MED |
| P-4 | GAP 1: `generate_l1()` InterpretationRegistry 3-tuple 연결 | ~23 | P-2,3 | LOW |

**Phase 0: 기반 인프라 (~950 LOC)**

| ID | 태스크 | LOC | 의존 | 리스크 |
|---|---|---|---|---|
| 0-1 | `ToolRegistry` — 38개 도구 정의 + 래퍼 + Bedrock 내보내기 | ~450 | 없음 | LOW |
| 0-2 | `ChangeDetector` + `_PipelineState` 콜백 + git hook | ~250 | 없음 | LOW |
| 0-3 | `BaseAgent` + `agent.yaml` + `checklist.yaml` (48항목) | ~250 | 0-1 | LOW |

**Phase 1: 추천사유 품질 Audit AV3 (~450 LOC)**

| ID | 태스크 | LOC | 의존 | 리스크 |
|---|---|---|---|---|
| 1-1 | `StratifiedReasonSampler` — 27개 스트라텀, 과표집 | ~180 | P-4, 0-3 | LOW |
| 1-2 | `GroundingValidator` — 사유↔IG top-K 정합성 + 품질 점수 | ~150 | 1-1 | MED |
| 1-3 | `Tier1Aggregator` — SelfChecker 결과 추이 집계 | ~120 | 0-3 | LOW |

**Phase 2: 편향 심화 Audit AV1 (~380 LOC)** — Phase 1과 병렬 가능

| ID | 태스크 | LOC | 의존 | 리스크 |
|---|---|---|---|---|
| 2-1 | `IntersectionalFairnessAnalyzer` — 교차 보호속성 DI | ~200 | 0-3 | MED |
| 2-2 | `BiasStageAttributor` — 단계별 DI + 증폭 식별 | ~180 | 0-3 | MED |

**Phase 3: 운영 에이전트 + 합의 (~1,100 LOC)** — Phase 4와 병렬 가능

| ID | 태스크 | LOC | 의존 | 리스크 |
|---|---|---|---|---|
| 3-1 | `OpsCollector` — 7개 체크포인트 수집 | ~250 | 0-1, 0-3 | LOW |
| 3-2 | `OpsDiagnoser` — 연쇄 영향 룰 테이블 | ~200 | 3-1 | LOW |
| 3-3 | `OpsReporter` — 템플릿 기반 리포트 | ~150 | 3-2 | LOW |
| 3-4 | `ConsensusArbiter` — AWS 독립투표 + 온프렘 2-Round | ~350 | 3-5 | **HIGH** |
| 3-5 | LLM Provider 확장 — LocalLLM(Qwen/Exaone) | ~150 | 없음 | MED |

**Phase 4: 진단 케이스 스토어 (~300 LOC)**

| ID | 태스크 | LOC | 의존 | 리스크 |
|---|---|---|---|---|
| 4-1 | `DiagnosticCaseStore` — LanceDB + 4개 도구 | ~300 | 0-1 | LOW |

**Phase 5: 통합 + AWS 확장 (~520 LOC)**

| ID | 태스크 | LOC | 의존 | 리스크 |
|---|---|---|---|---|
| 5-1 | `AgentEventBridge` — 상호 트리거 | ~120 | Phase 3 | LOW |
| 5-2 | `GovernanceReportGenerator` 확장 | ~80 | Phase 1-3 | LOW |
| 5-3 | `SendNotification` — Slack/Email/SNS | ~120 | 0-1 | LOW |
| 5-4 | `BedrockDialogSession` — Tool Use 대화 | ~200 | 0-1, 3-5 | MED |

#### 의존관계 및 병렬 가능성

```
Pre-req (P-1~P-4, 순차)
  → Phase 0 (0-1, 0-2 병렬 → 0-3)
    → Phase 1 + Phase 2 (병렬)
      → Phase 3 + Phase 4 (병렬)
        → Phase 5
```

#### 디렉토리 구조

```
core/agent/                     # 신규
    __init__.py, base.py, tool_registry.py, change_detector.py,
    consensus.py, case_store.py, event_bridge.py, notification.py,
    bedrock_dialog.py
    ops/  collector.py, diagnoser.py, reporter.py
    audit/ reason_sampler.py, grounding_validator.py, tier1_aggregator.py,
           intersectional_fairness.py, bias_stage_attributor.py,
           diagnoser.py, reporter.py
configs/financial/              # 신규
    agent.yaml, agent_tools.yaml, checklist.yaml
scripts/hooks/                  # 신규
    post_commit.py
```

---

## PaperClip 선택적 차용

PaperClip (@dotta, 2026.3, GitHub 30K stars)은 에이전트를 "직원"으로 조직화하는 오픈소스 프레임워크. "zero-human company" 철학은 우리의 "AI가 분석하고 사람이 판단한다" 원칙과 충돌하므로 전면 도입은 부적합하지만, 3가지 메커니즘을 선택적으로 차용.

### 차용 1: Heartbeat 패턴

PaperClip: 매 30분마다 에이전트를 깨워 `HEARTBEAT.md` 체크리스트 실행. 조치 불필요 시 `HEARTBEAT_OK` → 무동작.

우리 적용: CP5(5분), CP6(1시간), AV1~5(일/주) 주기 점검에 적용. 정상이면 skip, 이상 시에만 체크리스트 실행. **자율성이 아니라 효율성**을 위한 차용 — 고정된 체크리스트만 실행.

### 차용 2: 에이전트별 예산 캡

PaperClip "선불 직불카드" 모델: 에이전트마다 월간 토큰 예산, 80% 소프트 경고, 100% 하드 정지.

우리 적용:
```yaml
budget:
  ops: {monthly_token_limit: 500000, soft_warning_pct: 0.80, hard_stop_pct: 1.00}
  audit: {monthly_token_limit: 800000, soft_warning_pct: 0.80, hard_stop_pct: 1.00}
  consensus: {per_session_limit: 10000, daily_limit: 50000}
```

> **예산 초과 시 graceful degradation**: LLM 호출만 차단, 룰 엔진은 계속 동작. 예산 초과 = 임시 온프렘 모드.

### 차용 3: 전체 도구 호출 추적 (Full Trace)

PaperClip: 모든 instruction/response/tool call이 불변 감사 로그에 기록.

우리 적용: `ToolRegistry.call()` 호출 전후 자동 로깅. 도구명, 파라미터, 결과 요약, 토큰 비용을 추적. 에이전트 활동 로그로 "이 진단이 어떤 도구 호출을 거쳐 생성되었는가"를 완전 재현 가능.

### 차용하지 않는 것

| PaperClip 메커니즘 | 미차용 이유 | 우리의 대안 |
|---|---|---|
| 에이전트가 에이전트를 고용 | 감사 관점에서 자율 생성 위험 | 고정 2개 에이전트 + YAML 체크리스트 |
| SOUL.md 페르소나 | "성격" 부여 시 일관성 깨짐 | 시스템 프롬프트로 관점 정의 |
| 자율 의사결정 | EU AI Act Art.14 인간 감독 위반 | 에이전트는 권고만, 사람이 결정 |
| Node.js 서버 | Python 생태계 불일치 | Python 네이티브 구현 |
| PARA 메모리 | 파일 기반은 구조화 검색 어려움 | LanceDB DiagnosticCaseStore |

---

## 메모리 프레임워크 선택적 차용

2026년 초 여러 에이전트 메모리 프레임워크(Mem0, Zep/Graphiti, Letta, SuperLocalMemory, LangMem)가 발표. 프레임워크 전체 도입이 아닌 **핵심 패턴만 차용**.

### 이미 해결된 것 vs 실제 갭

| 기능 | 현재 상태 | 실제 갭 |
|---|---|---|
| 케이스 축적 | DiagnosticCaseStore (LanceDB) 구현됨 | 시간 decay 없음 |
| 피처 해석 | InterpretationRegistry 5-level cascade | 고객 서술적 프로파일 없음 |
| 감사 추적 | HMAC 해시체인 + S3 Object Lock 7년 | 시점 T 스냅샷 복원 비효율 |
| 담당자 대화 | BedrockDialogSession 세션 기반 | 세션 간 대화 이력 단절 |

### 차용 결정 (우선순위 순)

| 순위 | 프레임워크 | 차용 내용 | 우선순위 |
|---|---|---|---|
| 1 | Zep/Graphiti | 시간적 지식 그래프 — `(entity, attribute, value, valid_from, valid_to)` 스키마 | HIGH |
| 2 | SuperLocalMemory | 수학적 decay `exp(-age/τ)` — 원본 보존, 검색 가중치만 조정 | HIGH |
| 3 | Mem0 | 팩트 압축 레이어 — 룰 기반으로 LLM 없이 구현 | MEDIUM |
| 4 | Letta (MemGPT) | Recall memory 패턴 — 세션 간 대화 이력 유지 | MEDIUM |
| -- | LangMem | **차용 안 함** — 프롬프트 자기개선은 감사 관점 위험 | SKIP |
| -- | Succession/ALE | **차용 안 함** — 현재 불필요 | SKIP |

### 차용 1: 시간적 지식 그래프 (Zep/Graphiti)

감사 질의 "2026-03-15 시점에 고객 A에게 펀드 X 추천한 근거는?"을 단일 필터 쿼리로 해결.

```python
class TemporalFactStore:
    """같은 LanceDB 백엔드 재사용 (DiagnosticCaseStore와 공유)"""
    schema = {
        "fact_id": str,
        "entity_type": str,   # customer, model, recommendation
        "entity_id": str,
        "attribute": str,
        "value": str,         # JSON
        "valid_from": datetime,
        "valid_to": datetime, # None = 현재 유효
        "vector": List[float],
    }
```

대부분의 감사 쿼리가 단일 엔티티의 시점 복원 → LanceDB 네이티브 필터로 충분. JOIN 불필요.

### 차용 2: 수학적 Decay (SuperLocalMemory)

```python
# search_similar()에 추가
age_days = (now - case.timestamp).days
decay = math.exp(-age_days / decay_half_life_days)
adjusted_score = cosine_similarity * decay
```

**중요**: 원본 삭제 아님, 검색 가중치만 조정. 감사 추적성 유지.

### 차용 3: 고객 팩트 압축 (Mem0)

```python
class FactExtractor:
    """룰 기반 — LLM 호출 없음"""
    def extract(self, customer_features: Dict) -> List[str]:
        facts = []
        if features.get("deposit_balance_ratio", 0) > 0.6:
            facts.append("예적금 중심 포트폴리오")
        if features.get("fund_view_count_3m", 0) > 5:
            facts.append("최근 3개월 펀드 관심 증가")
        # ... config 기반
        return facts
```

Phase 0 배치에서 추출 → LanceDB에 저장 → 서빙 타임 조회만 → L2a 프롬프트 강화.

### 차용 4: Dialog Recall Memory (Letta)

BedrockDialogSession에 DynamoDB 기반 대화 이력 저장. 이전 세션의 관련 대화를 임베딩 검색으로 조회하여 시스템 프롬프트에 첨부.

### 구현 우선순위

| ID | 태스크 | LOC | Phase |
|---|---|---|---|
| M-1 | DiagnosticCaseStore에 decay 추가 | ~30 | 즉시 |
| M-2 | TemporalFactStore (시간 그래프) | ~120 | Phase 4 확장 |
| M-3 | FactExtractor (룰 기반 고객 팩트) | ~150 | Phase 1 확장 |
| M-4 | DialogRecallMemory + BedrockDialog 통합 | ~80 | Phase 5 확장 |

**총 ~380 LOC**. 기존 아키텍처 증분 추가.

> **LanceDB 단일 백엔드 원칙**: M-1~M-3은 기존 DiagnosticCaseStore와 같은 LanceDB 인스턴스 공유. DuckDB 등 새 의존성 추가 없음. M-4만 기존 DynamoDB(reason_cache) 스택 재사용.

---

## 6. 미결 설계 과제

- **Tier 3 전문가 리뷰 UI/UX**: 리뷰 인터페이스를 어디에 만들 것인가 (Retool, 사내 도구, 등)
- **샘플링 비율 자동 조정**: Tier 2 품질 저하 시 샘플 비율을 자동으로 5%→10%로 올릴 것인가
- **교차속성 조합 폭발**: 보호속성 5개의 2-way 조합 = 10개, 3-way = 10개 — 어디까지 볼 것인가
- **진단 정확도 검증**: "likely_cause" 추정이 맞았는지 사후 검증하는 메커니즘
- **비용 제약**: Tier 2 XAI 평가 (perturbation 기반)의 GPU 비용 vs 감사 가치 trade-off
- **에이전트 자체 모니터링**: 에이전트가 정상 작동하지 않을 때의 감시 (watchdog)
- **FeatureStore latency 계측**: `health_check()`에 per-request latency 없음. DynamoDB는 CloudWatch로 대체 가능하나 Memory 백엔드는 계측 추가 필요
- **Git hook 인프라 구축**: 현재 샘플 훅만 존재. `post-commit` → ChangeDetector 이벤트 전달 구조 신규 구축 필요
- **`_PipelineState` 이벤트 발행**: `mark_complete()`에 콜백 메커니즘 추가 필요 (EASY, 기존 코드 영향 없음)
- **`verify_chain()` 증분 검증**: 현재 GENESIS부터 전체만 검증. `start_hash` 파라미터 추가 시 증분 가능

---

## 추천사유 생성 파이프라인 연동 — 갭 분석 및 구현 계획

에이전트 설계와 추천사유 생성 코드의 연동을 점검한 결과, **각 컴포넌트는 프로덕션 수준이지만 컴포넌트 간 연결에 4개 갭**이 발견되었다.

### 현재 흐름과 갭

```
IG 점수 배열 [0.35, 0.22, -0.15, ...]
    │
    ├─ InterpretationRegistry (5-level, IG방향, 한국어) ← 호출 안 됨 (GAP 1)
    ├─ ReverseMapper (glossary 템플릿, 값 대입)        ← 별도 체계 (GAP 2)
    │
    ▼ 2-tuple (name, score)만 전달
TemplateEngine ← prefix 누락 (GAP 3) + 영문 fallback (GAP 4)
    ▼
SelfChecker (금지 패턴만) + Grounding (숫자만)
    → 피처-사유 정합성 미검증 → 에이전트 Tier 2가 커버
```

### 4개 갭

| GAP | 심각도 | 내용 | 수정 파일 |
|---|---|---|---|
| 1 | HIGH | `generate_l1()`이 `InterpretationRegistry` 미호출. 2-tuple만 전달. | `async_orchestrator.py` |
| 2 | HIGH | ReverseMapper와 InterpretationRegistry가 병렬 독립 체계 | `interpretation_registry.py`, `pipeline.py` |
| 3 | MEDIUM | `_DEFAULT_PREFIX_TO_GROUP`에 12개+ prefix 누락 | `interpretation_registry.py`, `feature_groups.yaml` |
| 4 | MEDIUM | ReverseMapper 매핑 실패 시 영문 fallback 출력 | `reverse_mapper.py` |

### 구현 순서 (의존관계 기반)

```
Step 1: GAP 4 — ReverseMapper 한국어 fallback (자체 완결, GAP 2 전제조건)
  ▼
Step 2: GAP 3 — prefix 커버리지 확장 (자체 완결, GAP 2 효과 증대)
  ▼
Step 3: GAP 2 — ReverseMapper를 InterpretationRegistry fallback으로 통합
  ▼
Step 4: GAP 1 — generate_l1()에 InterpretationRegistry 연결 (최종 배선)
```

**Step 1 (GAP 4)**: `reverse_mapper.py`의 기본 `interpretation_templates`를 한국어로 교체. `"Unknown"` → `"미분류 그룹"`. Config 오버라이드 유지.

**Step 2 (GAP 3)**: `feature_groups.yaml`에 `prefix_to_group` 섹션 추가 (config-driven). `from_configs()`에서 로드하여 defaults에 병합. Longest-first 정렬.

**Step 3 (GAP 2)**: `InterpretationRegistry.interpret()` cascade에 Level RM 추가 (L1과 glossary 사이). `reverse_mapper` 파라미터 Optional 주입.

**Step 4 (GAP 1)**: `AsyncReasonOrchestrator.__init__()`에 `interpretation_registry` 파라미터 추가. `generate_l1()`에서 `interpret_batch()` → 3-tuple enrichment → template engine 전달. None이면 기존 동작.

### 인터페이스 계약

| 생산자 | 출력 | 소비자 | 기대 입력 |
|---|---|---|---|
| `InterpretationRegistry.interpret_batch()` | `List[Dict]` (name, value, text) | `generate_l1()` | 3-tuple 변환 |
| 3-tuple | `(str, float, str)` | `TemplateEngine._ig_based_reasons()` | `len(entry) >= 3` (이미 구현) |
| `ReverseMapper.interpret_financial()` | `str` (한국어) | `InterpretationRegistry` Level RM | 비어있지 않은 문자열 |
| `feature_groups.yaml` `prefix_to_group` | `Dict[str, str]` | `from_configs()` | defaults에 병합 |

### 리스크

| GAP | 리스크 | 완화 |
|---|---|---|
| 1 | LOW | `interpretation_registry=None`이면 기존 동작 |
| 2 | MEDIUM | GAP 4 먼저 구현 → ReverseMapper가 항상 한국어 |
| 3 | LOW | Longest-first 정렬로 모호성 방지 |
| 4 | MEDIUM | 구현 전 영문 패턴 grep 스캔 |

---

## Bedrock 인프라 공유 및 태스크별 모델 선택

추천사유 생성(L2a)과 운영/감사 에이전트가 **동일한 Bedrock 인프라**를 공유한다. 태스크 특성에 따라 최적 모델을 분리 배정.

### 태스크별 모델 배정

| 태스크 | 모델 | 선택 이유 | 입출력 토큰 | 건당 비용 |
|---|---|---|---|---|
| L2a 사유 생성 | **Claude Sonnet** | Bedrock 네이티브, 한국어 자연스러움 | 입 ~600 / 출 ~80 | ~$0.002 |
| L2b self-critique | **Claude Sonnet** | 생성 모델 ≤ 크리틱 모델. 한국어 뉘앙스 판단 필요 | 입 ~800 / 출 ~100 | ~$0.003 |
| SelfChecker factuality | Claude Haiku | 수치 판정(0~1), 논리력 중시 | 입 ~800 / 출 ~50 | ~$0.001 |
| 에이전트 대화 | Claude Sonnet | 맥락 추론 + 도메인 판단 | 입 ~2K / 출 ~500 | ~$0.02 |
| 3-에이전트 합의 | Claude Sonnet × 3 | 독립 관점, 풍부한 reasoning | 입 ~1K / 출 ~600 | ~$0.03 |
| 분기 심층 리뷰 | Claude Opus | 다중 규제 교차 분석 | 입 ~5K / 출 ~2K | ~$0.30 |
| 벡터 임베딩 | Titan Embeddings V2 | ContextVectorStore + CaseStore 공유 | 입 ~200 | ~$0.0001 |

### 한국어 추천사유에 Claude Sonnet을 선택하는 이유

추천사유 L2a의 핵심: "고객이 납득할 만한 자연스러운 한국어 1~2문장"

| 기준 | Claude Sonnet | Claude Haiku | Llama 3.1 |
|---|---|---|---|
| 한국어 자연스러움 | **상** (범용 + 다국어) | 상 (범용) | 중하 (영어 중심) |
| 금융 존댓말 톤 | **상** | 상 | 하 |
| 비용 | 중 | 저 | 최저 |
| Bedrock 가용성 | **네이티브** | 네이티브 | 네이티브 |

> **Claude Sonnet 선택 이유**: Claude Sonnet(claude-sonnet-4-20250514)은 Bedrock 네이티브로 Marketplace 온보딩 없이 즉시 사용 가능. `llm_provider.py` 팩토리 패턴으로 config 변경만으로 전환 가능.

### L2a 비동기 처리 아키텍처

```
실시간 서빙                 비동기 L2a                   다음 서빙
L1 템플릿 즉시 반환    +    SQS → 워커 → Claude Sonnet    →    캐시에서 L2a 반환
(LLM 불필요, 0ms)          → DynamoDB 캐시 저장              (캐시 미스 시 L1)
```

| 구성 | 대상 | 소요시간 | 비용 |
|---|---|---|---|
| 전체 941K 중 L2a 대상 (5%) | ~47K건 | -- | -- |
| Sonnet 워커 1대 (순차) | 47K × 50ms | ~40분 | ~$0.21 |
| Sonnet 워커 5대 (병렬) | 47K / 5 | **~8분** | ~$0.21 |
| Bedrock Batch Inference | 일괄 제출 | **~수분** | ~$0.21 |

### Bedrock 공유 시 고려사항

| 고려사항 | 문제 | 대응 |
|---|---|---|
| 쿼터 경쟁 | L2a 배치 + 에이전트 합의 동시 | 시간대 분리 (L2a 야간, 에이전트 점검 후) |
| 비용 통합 | 추천사유 + 에이전트 합산 | CloudWatch `usage_type` 태그로 분리 추적 |
| fallback | Bedrock 장애 시 양쪽 영향 | 추천사유: L1 fallback, 에이전트: 룰 엔진만 (양쪽 이미 설계) |
| 모델 관리 | Sonnet+Haiku+Opus+Titan 4개 | `llm_provider.py` 팩토리로 config별 프로바이더 주입 |

### 온프렘 모델 선택: Exaone 3.5

온프렘에서는 LG AI Research의 **Exaone 3.5** (Apache 2.0 오픈소스) 한국어 특화 모델을 사용.

| 모델 | 파라미터 | VRAM | 한국어 | 비고 |
|---|---|---|---|---|
| **Exaone 3.5 7.8B** | 7.8B | ~8GB | **상** | RTX 4070에 여유, 한국어 특화 |
| Exaone 3.5 2.4B | 2.4B | ~3GB | 중상 | 초경량, 속도 우선 시 |
| Qwen 2.5 14B Q4 | 14B Q4 | ~9GB | 중상 | 범용, 한국어는 Exaone보다 약간 부자연 |
| Llama 3.1 8B | 8B | ~8GB | 중하 | 영어 중심 |

온프렘 용도별 배정:
- **추천사유 L2a 생성/critique**: Exaone 3.5 7.8B (~8GB) — *한국어 자연스러움*이 핵심
- **에이전트 2-Round 합의**: Qwen 2.5 14B Q4 (~9GB) — *논리력/추론력*이 핵심. 한국어 자연스러움 불필요. 파라미터 2×로 추론력 높음
- **임베딩**: sentence-transformers (all-MiniLM-L6-v2)

> **VRAM 관리**: Exaone 8GB + Qwen 9GB = 17GB > 12GB VRAM. 동시 로딩 불가하나 배치 순차 실행으로 해결 — (1) 룰 엔진 (GPU 불필요) → (2) Qwen 로드 → 합의 → 언로드 → (3) Exaone 로드 → L2a → 언로드

> **Exaone 생태계**: K-Exaone (236B MoE)이 글로벌 벤치마크 7위로 Qwen3/GPT 상회. Exaone 4.5 (멀티모달)도 출시. 향후 K-Exaone 오픈소스화 시 온프렘 품질 상승 기대. 현재 RTX 4070 기준 3.5 7.8B가 최선.

---

## 7. 규제 충족 매핑

이 시스템의 핵심 설계 원칙: **"AI가 분석하고, 사람이 판단한다."**
에이전트의 역할은 의사결정이 아니라 **의사결정 보조**로 명확히 한정된다.

### EU AI Act

| 조항 | 요구사항 | 충족 방식 | 근거 구성요소 |
|---|---|---|---|
| Art.9 | 리스크 관리 | 3-에이전트 합의 + 마이너리티 보존 | 합의 메커니즘 |
| Art.12 | 기록 보관 | 케이스 스토어 + HMAC 감사 로그 (7년) + reasoning 전문 보존 | DiagnosticCaseStore + AuditLogger |
| Art.13 | 투명성·설명 | reasoning 300~600 토큰, 판단 근거 명시적 기록 | 에이전트 출력 사양 |
| Art.14 | 인간 감독 | 에이전트는 권고만, 자동 조치 없음, Action은 승인 후 실행 | Query/Action 분리 |
| Art.15 | 정확성·견고성 | 독립 투표 + 룰 엔진(결정론적) + LLM(확률적) 이중 구조 | 2-Round 하이브리드 |

### 한국 금감원 AI 가이드라인

| 요구사항 | 충족 방식 | 근거 구성요소 |
|---|---|---|
| 설명 가능성 | reasoning + 마이너리티까지 "모든 관점 검토" 증명 | 에이전트 출력 + 마이너리티 리포트 |
| 공정성 모니터링 | 체크리스트 4.6~4.9 정기 점검 + 교차 보호속성 분석 | 체크리스트 + IntersectionalFairnessAnalyzer |
| 감사 추적 | HMAC 불변 로그 + 케이스 스토어 + 거버넌스 리포트 | AuditLogger + CaseStore + GovernanceReport |
| 인간 개입 | 합의 결과도 리뷰 우선순위일 뿐, 최종 판정은 사람 | 전체 아키텍처 원칙 |
| 모델 리스크 관리 | 3-에이전트 합의 + 대응 효과 추적 + 마이너리티 적중률 검증 | 합의 + 케이스 스토어 피드백 |

### 한국 AI 기본법

| 요구사항 | 충족 방식 | 근거 구성요소 |
|---|---|---|
| 고영향 AI 대응 | FRIA 5차원 리스크 자동 평가 (AV4) | FRIAEvaluator + 체크리스트 6.6 |
| AI 이용자 권리 | opt-out 이력 관리 + AI 공시 문구 확인 | opt_out_audit + 체크리스트 5.10 |
| 킬스위치/인간 감독 | 킬스위치 상태 모니터링, 활성화는 사람이 수행 | 체크리스트 4.4 + Action 승인 |

### 규제 충족 흐름

```
에이전트 (룰 엔진 + LLM + 합의 + 마이너리티 보존)
  ▼ 권고 (자동 조치 없음)
리포트 산출 (최우선 리뷰 + 마이너리티 리포트 + reasoning + 유사 케이스)
  ▼ 사람이 읽고 판단
인간 의사결정 (담당자가 검토 · 최종 판단 · 조치 · 승인)
  ▼ 전 과정 기록
감사 증적 (케이스 스토어 + HMAC 로그 + 거버넌스 리포트 → 규제기관 제출)
```

> **이 구조가 규제적으로 안전한 이유**: 에이전트는 (1) 자동 점검, (2) 이상 징후 해석, (3) 다중 관점 합의, (4) 소수 의견 보존까지만 책임지고, 최종 의사결정과 조치는 반드시 사람이 수행한다. 이 경계가 명확하기 때문에 EU AI Act "인간 감독", 금감원 "인간 개입", AI 기본법 "킬스위치" 요건을 모두 충족한다.
