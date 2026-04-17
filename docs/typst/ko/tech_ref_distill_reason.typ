// ============================================================
// 지식 증류 · 추천 사유 생성 기술 참조서
// AIOps PLE for Financial Recommendation
// Anthropic Design System
// ============================================================

#let anthropic-bg = rgb("#F0EFEA")
#let anthropic-text = rgb("#141413")
#let anthropic-accent = rgb("#CC785C")
#let anthropic-muted = rgb("#6B7280")
#let anthropic-rule = rgb("#D1D5DB")

#set document(
  title: "지식 증류 · 추천 사유 생성 기술 참조서",
  author: ("Author 1", "Author 2"),
)

#set page(
  paper: "a4",
  margin: (top: 2.5cm, bottom: 2.5cm, left: 2.5cm, right: 2.5cm),
  fill: anthropic-bg,
  header: context {
    if counter(page).get().first() > 1 [
      #set text(size: 7.5pt, font: ("Pretendard", "New Computer Modern"), fill: anthropic-muted, tracking: 0.12em)
      #smallcaps[Tech Reference]
      #h(1fr)
      #smallcaps[Knowledge Distillation & Recommendation Reasoning]
      #v(4pt)
      #line(length: 100%, stroke: 0.4pt + anthropic-rule)
    ]
  },
  footer: context {
    let pg = counter(page).get().first()
    if pg > 1 [
      #line(length: 100%, stroke: 0.3pt + anthropic-rule)
      #v(4pt)
      #set text(size: 8pt, font: ("Pretendard", "New Computer Modern"), fill: anthropic-muted)
      #h(1fr)
      — #pg —
      #h(1fr)
    ]
  },
)

#set text(font: ("Pretendard", "New Computer Modern"), size: 10pt, fill: anthropic-text)
#set par(justify: true, leading: 0.8em, spacing: 1.5em)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")

#show heading.where(level: 1): it => {
  v(0.6cm)
  set par(first-line-indent: 0pt)
  block(width: 100%)[
    #text(size: 20pt, fill: anthropic-text, weight: "bold")[#it.body]
    #v(4pt)
    #line(length: 100%, stroke: 1pt + anthropic-accent)
  ]
  v(0.4cm)
}

#show heading.where(level: 2): it => {
  v(0.4cm)
  set par(first-line-indent: 0pt)
  block[
    #text(size: 14pt, fill: anthropic-text, weight: "bold")[#it.body]
  ]
  v(0.15cm)
}

#show heading.where(level: 3): it => {
  v(0.2cm)
  set par(first-line-indent: 0pt)
  block[
    #text(size: 10pt, fill: anthropic-text, weight: "bold")[#it.body]
  ]
  v(0.1cm)
}

// ============================================================
// Title
// ============================================================
#set page(header: none, footer: none)

#v(3cm)
#align(center)[
  #text(
    size: 10pt,
    fill: anthropic-muted,
    tracking: 0.5em,
    weight: "regular",
  )[#upper[Tech Reference]]
  #v(0.5cm)

  #text(size: 26pt, fill: anthropic-text, weight: "bold")[
    지식 증류 · 추천 사유 생성
  ]

  #v(0.3em)

  #text(size: 14pt, fill: anthropic-muted)[
    기술 참조서: Temperature Scaling, FD-TVS, Grounding, LLM Safety
  ]

  #v(1em)

  #text(size: 11pt, fill: anthropic-text)[
    Author 1#super[1], Author 2#super[1]
  ]

  #v(0.3em)

  #text(size: 9pt, fill: anthropic-muted, style: "italic")[
    #super[1]Organization Name \
    contact\@org.com
  ]

  #v(0.6cm)
  #line(length: 30%, stroke: 0.5pt + anthropic-rule)
]

#v(1fr)
#pagebreak()

#set page(
  header: context {
    if counter(page).get().first() > 1 [
      #set text(size: 7.5pt, font: ("Pretendard", "New Computer Modern"), fill: anthropic-muted, tracking: 0.12em)
      #smallcaps[Tech Reference]
      #h(1fr)
      #smallcaps[Knowledge Distillation & Recommendation Reasoning]
      #v(4pt)
      #line(length: 100%, stroke: 0.4pt + anthropic-rule)
    ]
  },
  footer: context {
    let pg = counter(page).get().first()
    if pg > 1 [
      #line(length: 100%, stroke: 0.3pt + anthropic-rule)
      #v(4pt)
      #set text(size: 8pt, font: ("Pretendard", "New Computer Modern"), fill: anthropic-muted)
      #h(1fr)
      — #pg —
      #h(1fr)
    ]
  },
)

// ============================================================
// Abstract
// ============================================================
#block(
  width: 100%,
  inset: (left: 14pt, right: 14pt, top: 10pt, bottom: 10pt),
  stroke: (left: 2pt + anthropic-accent),
)[
  #text(weight: "bold")[범위(Scope).]
  - 지식 증류: PLE Teacher → LGBM Student (Temperature Scaling, custom objective, calibration)
  - 피처 선택: LGBM gain importance 기반 403D → task별 ~280D
  - FD-TVS 스코어링: dynamic(segment × behavior) task 가중치 적용
  - 추천 사유 생성: L1 Template → L2a LLM rewrite → L2b 3-Agent 검증 + SelfChecker
  - Layer 분리: 7 distilled LGBM + 3 direct LGBM + 3 Layer-3 rule-based
  - 서빙: Lambda Container Image + LanceDB(recommendation cases) + DynamoDB(cache·audit)
]

#v(1em)

#outline(indent: 1.5em, depth: 3)

#pagebreak()

#block(
  width: 100%,
  inset: (x: 1em, y: 0.8em),
  stroke: (left: 3pt + rgb("#d97706")),
  fill: rgb("#fffbeb"),
)[
  #text(weight: "bold", fill: rgb("#92400e"))[설계 vs 구현 차원 안내] \
  본 문서는 *풀뱅크 설계(734D)*를 기준으로 작성되었습니다. 현재 Santander 벤치마크 구현은 *~349D 입력 / Phase 0 후 403D (13 feature groups)*입니다. 본문의 차원 수치(734D, 200D, 140D 등)는 풀뱅크 설계 기준이며, 실제 Santander 구현에서는 중간 차원이 달라질 수 있습니다. 실제 구현의 차원 명세는 `outputs/phase0/feature_schema.json`을 참조하십시오.
]

#v(0.5em)

// ============================================================
= 지식 증류 (Knowledge Distillation)
// ============================================================

== 설계 철학 및 동기

=== 핵심 문제

PLE Teacher 모델은 약 2.6M 파라미터(AWS Santander 벤치마크 기준)이며 g4dn.xlarge
GPU에서 배치 1,024건 기준 약 50ms의 추론 지연을 보인다. 수백만 고객에 대한 일일 배치
추론에서 GPU 인스턴스를 상시 운용하는 비용을 제거하고, 실시간 추천 SLA(10ms 이내)를
CPU-only Lambda에서 충족하기 위해 LightGBM Student로 증류한다.

=== 해결 전략

Knowledge Distillation (Hinton et al., 2015)을 통해 Teacher의 출력 분포에 담긴
_암묵적 지식(dark knowledge)_ 을 LightGBM Student로 전이한다.
Student는 8GB RAM CPU에서 약 5ms/1K batch로 추론하여 10배 속도 향상을 달성하면서도
성능 손실을 3%p 이내로 유지한다.

=== Teacher--Student 비교

#table(
  columns: (1fr, 1fr, 1fr),
  inset: 8pt,
  stroke: 0.4pt + luma(200),
  [*속성*], [*PLE Teacher*], [*LGBM Student*],
  [모델 구조], [PLE-adaTT + Cluster Head], [LightGBM (태스크별 독립)],
  [파라미터], [$tilde$2.6M (AWS Santander)], [$tilde$500 trees/task],
  [메모리], [$tilde$8GB VRAM (g4dn.xlarge GPU)], [$tilde$1GB RAM (Lambda CPU)],
  [추론 속도], [$tilde$50ms / 1K batch], [$tilde$5ms / 1K batch],
  [피쳐 차원], [734D(설계) / ~349D 입력·403D 후처리(구현)], [200D (IG 선택 후, 설계 기준)],
  [학습 데이터], [원본 피쳐 + 라벨], [원본 피쳐 + Hard Label + Soft Label],
)

_Cross-architecture distillation_ (DNN $arrow$ GBDT)이 가능한 이유는
지식 증류가 파라미터가 아닌 출력 분포를 통해 지식을 전달하기 때문이다.
"딥러닝으로 학습, GBDT로 서빙"은 추천, 금융, 광고 도메인의 사실상 표준이다
(Borisov et al., NeurIPS 2022).


== Temperature Scaling

=== 수학적 정의

표준 softmax에 Temperature 파라미터 $T$를 도입한다.

$ p_i^T = frac(exp(z_i \/ T), sum_j exp(z_j \/ T)) $ <temp-scaling>

- $T = 1$: 표준 softmax. 최대 로짓에 확률이 집중된다.
- $T = 5$ (기본값): 클래스 간 상대적 관계가 드러나는 평활 분포.
- $T arrow infinity$: 균등 분포. 정보가 소실된다.

=== 볼츠만 분포와의 대응

이 수식은 통계역학의 볼츠만 분포 $P(E_i) = exp(-E_i \/ (k_B T)) \/ Z$와
수학적으로 동치(isomorphic)이다. 로짓 $z_i$는 (음의) 에너지, $T$는 절대 온도에
대응하며, 두 분야가 동일한 지수 족(exponential family) 분포를 공유한다.

=== Temperature 범위 설정

$T in [3, 7]$로 제한한다.
- $T = 3$: 이진 태스크 (CTR, CVR, Churn, Retention)
- $T = 5$: 기본값
- $T = 7$: 다중 클래스 태스크 (NBA 12-class, Timing 28-class)
- $T > 10$: 과도한 정보 손실 위험

#block(
  stroke: (left: 2pt + rgb("#6B7280")),
  inset: (left: 8pt, right: 8pt, top: 6pt, bottom: 6pt),
  width: 100%,
)[
  *LGBM Student의 Temperature* \
  LGBM Student에게 soft label을 전달할 때는 $T = 1$ (표준 softmax)를 사용한다. LGBM의 custom objective는 확률 벡터를 직접 수신하므로, GBDT 내부 최적화 위에 $T > 1$ 평활화를 추가 적용하면 오히려 캘리브레이션이 저하될 수 있다. $T = 3$–$5$ 범위는 신경망 간(neural-to-neural) 증류에 적합하며, DNN→GBDT 증류에는 해당하지 않는다.
]


== Dark Knowledge: Soft Label의 정보량

=== Hard Label의 한계

Hard Label (one-hot)은 $log_2(C)$ 비트의 정보만 인코딩한다.
NBA 12-class에서 약 3.6비트이며, 차선택 클래스 간 관계 정보를 전혀 포함하지 않는다.

=== Soft Label의 풍부한 구조

Teacher의 softmax 출력은 $(C - 1)$개의 연속 확률값을 담고 있어
$log_2(C)$보다 훨씬 많은 정보를 제공한다.

$ p_"teacher" = [0.72, 0.14, 0.08, 0.03, 0.01, ...] $

이 분포에서 주요 답변(72%), 차선택 구조(B 14% > C 8%), 클래스 간 유사성,
불확실성 수준까지 읽어낼 수 있다. 이것이 Hinton이 명명한 _Dark Knowledge_ 이다.

=== 라벨 스무딩 효과

Soft Label 학습은 Label Smoothing과 유사한 정규화 효과를 가진다.
Hard Label이 확률 0 또는 1의 극단으로 모델을 밀어붙이는 반면(과적합 유도),
Soft Label은 합리적인 확률 분포를 출력하도록 유도하여 일반화를 촉진한다.


== Unified Distillation Loss

=== 통합 손실 함수

$ cal(L)_"distill" = alpha dot cal(L)_"hard" + (1 - alpha) dot T^2 dot cal(L)_"soft" $ <unified-loss>

- $alpha$: Hard/Soft 비율 (기본값 0.3, 즉 30% ground truth + 70% Teacher 의견)
- $T^2$: 그래디언트 크기 보정 스케일링

=== $T^2$ 보정의 수학적 유도

Chain rule로부터 $partial hat(y) \/ partial z = (1\/T) sigma(z\/T)(1 - sigma(z\/T))$이다.
$1\/T$ 인자가 누적되어 전체 그래디언트가 $1\/T^2$로 축소된다.
$T^2$를 곱함으로써 원래 스케일을 복원한다. $T = 5$일 때 그래디언트는
$1\/25$로 축소되므로 $T^2 = 25$를 곱하여 보정한다.

=== 태스크별 손실 함수

*이진 분류 (CTR, CVR, Churn, Retention):*
$ cal(L)_"binary" = alpha dot "BCE"(hat(y), y) + (1 - alpha) dot T^2 dot D_"KL"(p_t || p_s) $
여기서 $p_t = sigma(z_t \/ T)$, $p_s = sigma(z_s \/ T)$이다.

*다중 클래스 (NBA 12-class, Life-stage 6-class, Timing 28-class):*
$ cal(L)_"multi" = alpha dot "CE"(z_s, y) + (1 - alpha) dot T^2 dot D_"KL"("softmax"(z_t \/ T) || "softmax"(z_s \/ T)) $

*회귀 (LTV, Engagement):*
$ cal(L)_"reg" = alpha dot "MSE"(hat(y)_s, y) + (1 - alpha) dot "MSE"(hat(y)_s, hat(y)_t) $
회귀에서는 Temperature Scaling이 무의미하므로 $T^2$ 보정을 적용하지 않는다.

=== KL-Divergence

$ D_"KL"(q || p) = sum_i q_i log frac(q_i, p_i) = underbrace(-H(q), "상수") + underbrace(H(q, p), "교차 엔트로피") $

Forward KL $D_"KL"("Teacher" || "Student")$를 사용한다. 이는 mean-seeking 특성으로
Student가 Teacher의 모든 중요 모드를 커버하도록 강제한다.
Reverse KL은 mode-seeking 특성으로 일부 모드를 무시할 위험이 있어 증류에 부적합하다.


== LGBM 피쳐 중요도 기반 피쳐 선택

피쳐 선택은 학습된 LGBM Student 모델의 *gain importance*를 기반으로 수행된다.
교사 모델 IG(Integrated Gradients) 귀인은 사용하지 않는다 — 교사는 이미 증류 완료된 상태이며,
서빙 모델(LGBM Student)의 관점에서 중요한 피쳐를 선택하는 것이 설계 목적과 일치한다.

=== 설계 근거

*서빙 모델 정렬*: 배포된 모델은 LGBM Student이다. Gain importance는 LGBM이 실제로 사용하는 피쳐를 반영한다.
교사(PLE) IG 귀인은 심층 신경망과 그래디언트 부스팅 간 아키텍처 차이로 인해 Student 중요도와 크게 다를 수 있다.

*운영 안정성*: LGBM gain 계산은 학습된 Student에 대한 빠른 사후 처리 단계이다.
교사 IG는 941K 고객 × 403D 피쳐 스케일에서 전체 PLE 그래프에 대한 역전파를 요구하여 OOM 장애를 유발한다.

*해석 가능성 정렬*: LGBM Student에서 도출된 SHAP/gain 설명은 추천을 생성한 모델에 직접 근거하여,
EU AI Act 제13조(실제 결정 메커니즘 반영 설명) 요건을 충족한다.

=== 2단계 파이프라인

_아래 차원 수치는 풀뱅크 설계(734D) 기준입니다. Santander 구현(~349D 입력 / 403D Phase 0 후)에서는 중간 차원이 달라질 수 있습니다._

*풀뱅크 설계 기준:* 734D(설계) / ~349D 입력·403D 후처리(구현) $arrow$ ~140D(Stage 1) $arrow$ 최종 LGBM 입력

*Stage 1 -- LGBM Gain Importance 필터:*
누적 gain importance 기준 상위 95%를 포착하는 상위-$k$ 피쳐를 태스크별로 유지한다.
결과 피쳐 집합은 일반적으로 태스크당 40~80개 피쳐이다(403D에서 축소).

*Stage 2 -- 필수 피쳐 보존:*
LGBM importance와 무관하게 항상 포함하는 7개 피쳐:
- TDA: `persistence_entropy`, `landscape_peak`
- Economics: `mpc`, `income_elasticity`, `permanent_income_ratio`
- FinEng: `sharpe_ratio`, `volatility`


== LightGBM Custom Objective

=== 구현 구조

`DistillationLossNumpy`가 LightGBM의 `fobj`에 gradient/hessian을 제공한다.

$ "grad" = alpha dot "grad"_"hard" + (1 - alpha) dot T^2 dot "grad"_"soft" $
$ "hess" = alpha dot "hess"_"hard" + (1 - alpha) dot T^2 dot "hess"_"soft" $

=== Soft Label 전달 기법

LightGBM Dataset은 `label`과 `weight`만 지원하므로,
Hard Label은 `get_label()`, Soft Label은 `get_weight()`를 통해 전달한다.

=== 증류 성능 비교

#table(
  columns: (1fr, 1fr, 1fr, 1fr),
  inset: 8pt,
  stroke: 0.4pt + luma(200),
  [*방법*], [*CTR AUC*], [*NBA Accuracy*], [*LTV RMSE*],
  [LGBM (Hard Label only)], [0.812], [0.634], [1.247],
  [LGBM (Distilled, $T = 5$)], [0.841], [0.698], [1.089],
  [PLE Teacher (원본)], [0.856], [0.723], [1.021],
)


== 10단계 DAG 오케스트레이션

전체 증류 파이프라인은 `distillation_entrypoint.py`에서 10단계 DAG로 실행된다.

#table(
  columns: (auto, 1fr),
  inset: 8pt,
  stroke: 0.4pt + luma(200),
  [*단계*], [*내용*],
  [1], [Teacher 추론 (전체 데이터에 대한 로짓 생성)],
  [2], [Soft Label 생성 (Temperature Scaling 적용)],
  [3], [LGBM gain importance 피쳐 선택 (~403D $arrow$ ~140D)],
  [4], [Student 학습 (태스크별 독립 LGBM)],
  [5], [검증 (5개 기준: AUC, RMSE, Accuracy 등; 메트릭 집계는 태스크 타입별 — 이진 태스크: AUC, 회귀 태스크: RMSE, 다중 클래스: top-k Accuracy)],
  [6], [MLflow 모델 레지스트리 등록],
  [7], [ONNX 변환 (ZipMap 제거 포함)],
  [8], [Triton 패키징 (config.pbtxt 생성)],
  [9], [통합 검증 (ONNX-PyTorch 수치 동치성)],
  [10], [배포 아티팩트 업로드],
)

== 적응형 증류 임계값

증류된 LGBM은 무조건 수용되지 않습니다. 수용을 위해서는 태스크별로 *랜덤 기준선의 2배* 이상의 적응형 임계값을 충족해야 합니다. 이 하한선 미만이면 Student는 SKIP으로 분류되고, 증류 라운드는 폐기되며 Teacher 직접 추론 경로가 활성화됩니다.

#table(
  columns: (auto, 1fr, 1fr),
  inset: 8pt,
  stroke: 0.4pt + luma(200),
  [*태스크 유형*], [*하한 임계값*], [*SKIP 정책*],
  [이진 (CTR/CVR/Churn)], [AUC $>$ 0.5 + $delta_"min"$ (랜덤 2배)], [Student 폐기, 2계층 (Teacher 직접) 활성화],
  [다중 클래스 (NBA)], [Top-1 정확도 $>$ 1/C + $delta_"min"$], [Student 폐기, 인기도 규칙으로 폴백],
  [회귀 (LTV)], [RMSE $<$ 기준선 평균 예측기], [Student 폐기, 전역 평균 사용],
)

이는 단순 기준선보다 나쁜 증류 모델의 배포를 방지하여 금감원 AI RMF C-1(출시 전 위험경감 조치 검증)을 충족합니다.

== 캘리브레이션: Platt Scaling

FD-TVS Stage 1은 모든 태스크 확률이 공통 $[0, 1]$ 스케일에 있을 것을 요구합니다. 확률이 중요한 태스크(이탈, product_stability, cross_sell_count)에는 증류 후 *Platt scaling*을 적용합니다.

$ p_"calibrated" = sigma(a dot f(bold(x)) + b) $

여기서 $f(bold(x))$는 LGBM 원시 점수이고, $a, b$는 홀드아웃 캘리브레이션 세트(학습 데이터 제외)에서 적합됩니다. 캘리브레이션은 ECE(Expected Calibration Error)를 줄이고, 예측 확률 0.8이 실제 양성률 약 80%에 대응하도록 보장합니다.

== 시간적 드리프트 모니터링

모집단 수준의 PSI 외에도 세 가지 지표가 증류 LGBM 입력 피쳐와 예측 출력의 분포 이동을 시계열로 감시합니다.

#table(
  columns: (auto, 1fr, auto),
  inset: 8pt,
  stroke: 0.4pt + luma(200),
  [*지표*], [*감지 대상*], [*임계값*],
  [PSI (모집단 안정성 지수)], [전체 입력 분포 이동], [> 0.25 critical],
  [JSD (Jensen-Shannon 발산)], [대칭적 분포 거리; KL보다 근-영 셀에 안정적], [> 0.10 alert],
  [순위 상관 (Spearman)], [재학습 주기 간 피쳐 중요도 순위 안정성], [< 0.70 alert],
)

세 가지 모두 피쳐별로 일간 계산됩니다. `ConsecutiveDriftTracker`는 PSI > 0.25가 3일 연속 지속될 때 자동 재학습을 트리거합니다. JSD 및 순위 상관 경고는 OpsAgent에 전달되어 서술적 분석이 이루어집니다.

#pagebreak()


// ============================================================
= FD-TVS 스코어링 엔진
// ============================================================

== 설계 철학

FD-TVS (Financial DNA-based Target Value Score)는 4단계 복합 스코어링 엔진으로,
모델 예측과 고객 수준 비즈니스 맥락을 결합한다.
핵심 설계 원칙은 *곱셈적 결합(multiplicative combination)*이다.
어떤 단일 요소라도 0에 접근하면 전체 점수가 거부(veto)되어
"리스크 우선(risk-first)" 원칙을 강제한다.

단순 합산 $sum p_i$는 윈도우 쇼퍼 (CTR=0.9, CVR=0.1)와
실구매자 (CTR=0.5, CVR=0.5)를 구분하지 못한다 --- 둘 다 합이 1.0이다.
곱셈 구조는 각 차원에 거부권(veto power)을 부여한다.

== Master Formula

$ "FD-TVS" = underbrace(S_"task", "What") times underbrace(W_"DNA", "Who") times underbrace(V_"TDA", "When") times underbrace((1 - R), "Safe?") times underbrace(f dot e, "Appropriate?") $ <fdtvs-master>

#table(
  columns: (auto, 1fr, auto),
  inset: 8pt,
  stroke: 0.4pt + luma(200),
  [*구성 요소*], [*설명*], [*범위*],
  [$S_"task"$], [Task Weighted Sum], [$[0, 1]$],
  [$W_"DNA"$], [Financial DNA Modifier], [$\{0.8, 1.0, 1.2\}$],
  [$V_"TDA"$], [Behavioral Velocity], [$[1.0, 1.15]$],
  [$R$], [Risk Penalty], [$[0, 1]$],
  [$f$], [Fatigue Decay], [$[0, 1]$],
  [$e$], [Engagement Boost], [$[0.85, 1.15]$],
)


== Stage 1: Task-Weighted Sum

$ S_"task" = beta_"CTR" dot p_"CTR" + beta_"CVR" dot p_"CVR" + beta_"NBA" dot p_"NBA" + beta_"LTV" dot p_"LTV" $ <stage1>

기본 가중치: CVR=0.4 (최고, 전환이 매출에 직결), CTR=0.3, NBA=0.2, LTV=0.1.
$sum beta_i = 1$이고 모든 $p_i in [0, 1]$이면 $S_"task" in [0, 1]$이 보장되는
WSM (Weighted Sum Model, Fishburn 1967) 볼록 결합(convex combination)이다.


== Stage 2: Financial DNA Modifier

Friedman의 항상소득가설(Permanent Income Hypothesis, 1957)에 기초한다.
소비자는 일시적 소득 변동이 아닌 항상(안정) 소득에 기반하여 소비를 결정한다.

$ W_"DNA" = cases(
  1.2 & "if CV" < 0.2 quad "(Permanent -- 안정 소득)",
  1.0 & "if" 0.2 <= "CV" < 0.5 quad "(Mixed)",
  0.8 & "if CV" >= 0.5 quad "(Transitory -- 불안정)"
) $ <dna-modifier>

여기서 $"CV" = sigma_"income" \/ mu_"income"$ (변동 계수, 무차원)이다.
안정 소득 고객은 장기 금융 상품(연금, 정기예금)에 적합하므로 추천 점수를 20% 상향한다.


== Stage 3: TDA Behavioral Velocity

$ V_"TDA" = 1.0 + gamma_"flare" dot bb(1)["flare"_"detected"] $ <tda-velocity>

$gamma_"flare" = 0.15$이다. TDA flare 감지는 행동 변화 가속을 나타내며,
점수를 최대 15% 상향한다.


== Stage 4: Risk Penalty

$ R = 0.2 dot I_"limit" + 0.3 dot I_"fatigue" + 0.5 dot I_"churn" $ <risk-penalty>

*비대칭 가중치의 근거 -- 비가역성(irreversibility):*
- 신용한도 소진 ($lambda_1 = 0.2$): 가역적 (상환으로 한도 복원)
- 메시지 피로 ($lambda_2 = 0.3$): 부분 가역 (시간이 지나면 회복)
- 고객 이탈 ($lambda_3 = 0.5$): 거의 비가역 (재획득 비용이 신규 대비 5--7배)

$(1 - R)$은 곱셈적 거부권을 행사한다.
$R arrow 1$이면 다른 요소와 무관하게 점수가 0으로 수렴한다.
로그 공간에서 $ln(1 - R) arrow -infinity$ as $R arrow 1$이다.


== Fatigue Decay

$ f(n) = e^(-lambda n) $ <fatigue>

지수적 감쇠로 *일정 비율 감소(constant fractional decay)*를 구현한다.
$f(n+1) \/ f(n) = e^(-lambda)$ (상수 비율).

반감기: $n_(1\/2) = ln 2 \/ lambda$. App Push ($lambda = 0.4$): 반감기 $approx 1.73$회.
Email ($lambda = 0.15$): 반감기 $approx 4.62$회.


== Confidence Formula

$ "confidence" = |p - 0.5| times 2 $ <confidence>

결정 경계(0.5)로부터의 거리를 $[0, 1]$로 정규화한다.
낮은 confidence 예측은 추천 품질 필터에서 후순위로 처리한다.

#pagebreak()


// ============================================================
= 추천 사유 생성
// ============================================================

== 2-Layer 아키텍처 (v3.0.0)

설계 철학: "전 고객에게 동등한 사유를, 컨텍스트 풍부 고객에게 LLM 강화 사유를."

#table(
  columns: (auto, auto, 1fr, auto, auto),
  inset: 8pt,
  stroke: 0.4pt + luma(200),
  [*Layer*], [*대상*], [*방식*], [*LLM 호출*], [*처리량*],
  [L1], [1,200만 전량], [3-에이전트 파이프라인 (FactExtractor → InterpretationRegistry → TemplateEngine → SelfChecker)], [0], [$tilde$20분],
  [L2a], [$tilde$50만/주], [LLM rewrite (Bedrock Claude Haiku, 3-layer safety gate)], [1], [$tilde$1.0초/건],
  [L2b], [$tilde$6.7만 샘플링], [품질 검증 (Bedrock Claude Haiku; 사실성, 관련성, 자연스러움)], [1], [--],
)

=== 3-에이전트 사유 파이프라인 (L1)

L1 사유 생성은 결정론적 4단계 에이전트 파이프라인으로 실행됩니다 --- LLM 호출 없음.

+ *FactExtractor:* YAML 설정 Python 표현식을 통해 피쳐 값에서 고객 수준 서술적 팩트를 추출합니다(예: "예적금 중심 포트폴리오", "리스크 회피 성향"). 샌드박스 환경에서 완전 결정론적으로 동작합니다.
+ *InterpretationRegistry:* 사전 등록된 3-tuple 풍부화(피쳐 → display_name, 방향, 설명)를 통해 IG 귀인 상위 5개 피쳐를 금융 언어로 매핑합니다. 항목이 없으면 `ReverseMapper`(Level RM)로 폴백합니다.
+ *TemplateEngine:* $"hash"("customer"_"id" : "category") mod 5$로 템플릿 변형을 선택하고 FactExtractor 출력 + IG 해석을 주입합니다.
+ *SelfChecker:* 3-stage 검증 게이트 --- (1) compliance: 금지 패턴(비교 광고, 수익 보장 표현 등) 규칙 매칭, (2) injection: 프롬프트 인젝션 패턴 탐지, (3) factuality: LLM 기반 사실성 검증. 세 단계 모두 통과한 사유만 release.

*규제 근거:* 금융소비자보호법 제19조는 전 고객에 대한 동등한 설명의무를 요구한다.
L1이 1,200만 전량에 비용 0의 템플릿을 제공하여 이를 충족한다.

*비용 효율:* 전량 LLM 처리 시 $tilde$1,000 GPU-hours 소요.
2-Layer 설계는 $tilde$162 GPU-hours로 달성한다.


== L1 Template 생성

=== 구조

6 categories $times$ 5 variants = 30개 템플릿.

*결정론적 변형 선택:*
$ "variant"_"index" = "hash"("customer"_"id" : "category") mod 5 $

동일 고객은 항상 동일 변형을 수신한다 (일관성 + 감사 재현성).

=== 세그먼트 인식

- *WARMSTART*: IG Top-3 역매핑 기반 사유
- *COLDSTART*: 인기도 + 혜택 기반 사유
- *ANONYMOUS*: 일반적 인기도 기반 사유

규칙 기반 컴플라이언스 검사 후 AI 생성 고지 문구를 자동 부착한다.


== L2a LLM Rewrite

우선순위 큐: rich 먼저, moderate 다음, sparse 제외.
3-Layer Safety Gate를 통과한 후 rewrite가 적용된다.
*AWS 배포:* Bedrock Claude Haiku (\$0.25/1M input tokens; VPC PrivateLink; Anthropic으로 데이터 미전송).
*온프레미스:* RTX 4070 (12GB VRAM)에서 vLLM Qwen3-8B-AWQ 구동.

=== 프롬프트 4계층 구조

+ *System 프롬프트*: 역할 정의 (금융 추천 사유 전문가) + 금소법 위반 금지 규칙
+ *Few-shot 예제*: 세그먼트별 톤·형식 가이드
+ *Context 주입*: 고객 피쳐, IG 기여도, 상담 이력 $arrow$ 자연어 변환
+ *출력 형식*: JSON 스키마 (`{"reasons": [...], "summary": "..."}`)

=== 디코딩 전략

- 사유 생성: $tau = 0.3$ (사실 보존 + 약간의 다양성)
- Critique: $tau = 0.1$ (거의 결정론적, 일관된 품질 평가)
- L2a Rewrite: $tau = 0.3$ (원본 사실 유지, 표현 윤색)


== Self-Critique 판정

$ "verdict" = cases(
  "pass" & "if" f >= 0.8 "and" c >= 1.0,
  "revise" & "if" f >= 0.5 "and" c >= 1.0,
  "reject" & "otherwise"
) $ <self-critique>

- $f$: 사실성 점수 (연속값)
- $c$: 컴플라이언스 점수 (이진: 1.0 = 위반 없음)

*컴플라이언스 절대 우선:* 규제 위반($c < 1.0$)이 있으면 사실성과 무관하게 즉시 거부.
*최대 1회 수정:* 수정 후에도 "revise"이면 안전 템플릿으로 폴백 (무한 루프 방지, LLM 호출 최대 3회).


== L2b 3축 품질 검증

$ "verdict" = cases(
  "pass" & "if" f >= 0.7 "and" r >= 0.7 "and" n >= 0.7,
  "needs"_"improvement" & "if any score" in [0.5, 0.7),
  "fail" & "if any score" < 0.5
) $ <l2b-validation>

- $f$: 사실성, $r$: 관련성, $n$: 자연스러움
- 임계값 0.7 (Self-Critique의 0.8보다 낮음): L2b는 사후 모니터링이며 실시간 게이트키퍼가 아니므로

#pagebreak()


// ============================================================
= 그라운딩 + 피쳐 역매핑
// ============================================================

== 핵심 문제

PLE-adaTT는 734D 피쳐 벡터를 소비하여 확률 점수를 출력하지만
_왜_ 그 추천을 했는지 설명하지 못한다.
AI 기본법 제31조·제34조, 금융소비자보호법 제19조는 유의미한 설명을 요구한다.

== Grounding 함수

$ f: bb(R)^(644) times cal(I) arrow.r cal(L) $ <grounding-fn>

여기서 $bb(R)^(644)$은 정규화 피쳐 벡터 공간, $cal(I)$는 IG 귀인 정보,
$cal(L)$은 자연어 설명 공간이다.


== 734D 피쳐 벡터 구조

#table(
  columns: (auto, auto, 1fr),
  inset: 8pt,
  stroke: 0.4pt + luma(200),
  [*범위*], [*차원*], [*내용*],
  [profile], [0--238], [Demographics (100D) + RFM (50D) + Financial Summary (88D)],
  [multi\_source], [238--329], [Transaction stats (40D) + Behavioral patterns (51D)],
  [extended\_source], [329--413], [보험, 상담, STT, 캠페인, 해외, 오픈뱅킹],
  [domain], [413--572], [TDA (70D) + GMM (22D) + Mamba (50D) + Economics (17D)],
  [model\_derived], [572--599], [HMM summary, Bandit/MAB, LNN],
  [multidisciplinary], [599--623], [전환 역학, 채택 역학, 교차 패턴, 루틴 분석],
  [merchant\_hierarchy], [623--644], [MCC levels, brand embeddings, 통계, radius],
)

총 644D 정규화 + 90D raw power-law = 734D 모델 입력.


== Integrated Gradients 귀인

$ "IG"_i (bold(x)) = (x_i - x'_i) times integral_0^1 frac(partial F(bold(x)' + alpha (bold(x) - bold(x)')), partial x_i) d alpha $

*IG가 SHAP보다 적합한 이유:*
- SHAP: $2^(734)$ 부분집합 평가 필요 (불가능)
- IG: 50-step 사다리꼴 근사로 선형 시간 계산
- *완전성 공리:* $sum_i "IG"_i (bold(x)) = F(bold(x)) - F(bold(x)')$ (벡터 해석학의 그래디언트 정리에 의해 보장)
- *Baseline:* 영벡터 (정규화된 피쳐에서 "평균 고객"에 해당)

== Reverse Mapping 아키텍처

$ "ReverseMap": (bold(x) in bb(R)^d, bold(a) in bb(R)^d) arrow.r {(r_k, s_k, t_k)}_(k=1)^K $

- $bold(x)$: 피쳐 벡터, $bold(a)$: IG 귀인 벡터
- $r_k$: 피쳐 범위 이름, $s_k$: 요약 점수, $t_k$: 금융 언어 텍스트

*서브레인지 슬라이싱:* $t_k = cal(M)_k (g(bold(x)[s_k : e_k]))$
여기서 $g$는 집계 함수 (mean, argmax, threshold 비교),
$cal(M)_k$는 도메인 전문가가 설계한 매핑 딕셔너리 (수치 $arrow$ 텍스트)이다.

== 모듈 구성

- *FeatureReverseMapper:* 644D 벡터 $arrow$ 금융 언어 텍스트 (계층적 범위 슬라이싱)
- *MultidisciplinaryInterpreter:* 24D 다학제 피쳐 $arrow$ 비즈니스 해석 (4개 서브도메인)
- *LanceContextVectorStore:* LanceDB 기반 고객 컨텍스트 저장/검색 (768D 임베딩 L2 거리)
- *ContextAssemblyAgent:* IG 기반 도구 선택 + 다중 소스 컨텍스트 조립
- *ConsultationContextExtractor:* STT 상담 이력 추출 + 요약

== Trust Loop

모델 예측 $arrow$ LGBM gain 귀인 $arrow$ 역매핑 $arrow$ 컨텍스트 조립 $arrow$ LLM 사유 생성
$arrow$ 상담원 전달 $arrow$ 고객 설득 $arrow$ 전환/피드백 $arrow$ 모델 개선.

역매핑과 컨텍스트 조립 없이는 모델 예측과 상담원 전달 사이에 해석가능성 간극이
발생하여 이 Trust Loop가 단절된다.

== 3중 Grounding

+ *피쳐 Grounding:* IG Top-5 귀인을 프롬프트에 주입 $arrow$ LLM이 실제 모델 판단 근거에 기반하여 사유 생성
+ *고객 Grounding:* 세그먼트, 거래 패턴, 상담 이력 주입 $arrow$ 환각(hallucination) 억제
+ *규정 Grounding:* 시스템 프롬프트 금지 규칙 + Rule-based Self-Critique $arrow$ 컴플라이언스 강제

#pagebreak()


// ============================================================
= Safety Gate
// ============================================================

== 다층 방어 아키텍처

금소법 및 AI 기본법 준수를 위한 6계층 안전 장치이다.

#table(
  columns: (auto, 1fr, 1fr),
  inset: 8pt,
  stroke: 0.4pt + luma(200),
  [*계층*], [*메커니즘*], [*대상 규제*],
  [1], [System 프롬프트: 불변 규제 금지 규칙], [AI 기본법 제31조·제34조],
  [2], [Self-Critique: 실시간 게이트키퍼 ($f >= 0.8$, $c = 1.0$)], [금소법 제19조·제21조],
  [3], [3-Layer Safety Gate: 프롬프트 인젝션 탐지 + 사실성 + 규제 준수], [금소법 제22조],
  [4], [L2b Quality Validation: 사후 모니터링 3축 평가], [내부 품질 기준],
  [5], [AI Security Checker: 인젝션 탐지 + 컴플라이언스 검증], [AI 기본법 제34조],
  [6], [Audit Archiver: 불변 Parquet 레코드 (DuckDB 기반 조회)], [금융감독원 사후 검사],
)

== 관련 규제

- *AI 기본법 제31조* (AI 사용 고지): 모든 추천 사유에 AI 생성 고지 문구 자동 부착
- *AI 기본법 제34조* (위험관리): Safety Gate + 감사 추적
- *금융소비자보호법 제19조* (적합성 원칙 + 설명의무): L1이 전량 커버리지 보장
- *금융소비자보호법 제21조* (광고 규제): 금지 패턴 탐지
- *금융소비자보호법 제22조* (불공정행위 금지): 컴플라이언스 점수 검증

== 감사 아카이빙

`RecommendationAuditArchiver`가 모든 추천 건을 Parquet으로 영속 저장한다.
- IG 귀인 점수, L1 사유, L2a rewrite 결과, L2b 검증 결과, 처리 시간
- DuckDB 기반 효율적 소급 조회
- 금융감독원 사후 검사 시 개별 추천 건의 전체 의사결정 경로 추적 가능

#pagebreak()


// ============================================================
= 서빙 아키텍처
// ============================================================

== End-to-End 파이프라인

```
PLE-adaTT Teacher (학습)
  |-> Knowledge Distillation (T=5, alpha=0.3) + 적응형 임계값 (랜덤 2배 하한)
    |-> LGBM Student (태스크별, 200D 피쳐) + Platt 캘리브레이션 (확률 중요 태스크)
      |-> ONNX Export (ZipMap 제거)
        |-> Lambda FallbackRouter
            |-> 1계층: LGBM ONNX (기본)
            |-> 2계층: PLE SageMaker Endpoint (폴오버)
            |-> 3계층: 규칙 엔진 (13개 태스크, Financial DNA 라우팅)
          |-> FD-TVS Scoring (4-Stage)
            |-> Feature Grounding (IG -> Reverse Mapping -> Context Assembly)
              |-> Recommendation Reason
                  L1: FactExtractor -> InterpretationRegistry -> TemplateEngine -> SelfChecker
                  L2a: Bedrock Claude Haiku (rewrite)
                  L2b: Bedrock Claude Haiku (검증)
                |-> Audit Archive (Parquet, ComplianceAuditStore)
```

== LGBM $arrow$ ONNX 변환

=== ZipMap 제거 (필수)

LightGBM의 ONNX 변환은 ZipMap 연산자를 추가하여 딕셔너리 출력을 생성한다.
Triton은 텐서 출력만 지원하므로 ONNX 그래프에서 ZipMap 노드를 우회하여 제거해야 한다.

=== 변환 사양

- Opset 13: LightGBM 연산자 전체 지원
- 2단계 검증: (1) `onnx.checker.check_model` 스펙 적합성, (2) 더미 추론 수치 동치성 테스트

== Triton Inference Server 구성

=== 모델 배치

15개 ONNX 모델 (태스크별) + 1 전처리기 + 1 후처리기 + 15 앙상블 스케줄러 = *32개 모델 설정*.

=== Dynamic Batching

- Preferred batch sizes: [256, 512, 1024]
- Max queue delay: 100$mu$s
- 전처리기: CPU $times$ 4 인스턴스 (CPU-bound JSON 파싱)
- ONNX 모델: GPU $times$ 2 인스턴스

=== 배치 + 실시간 하이브리드

일일 배치로 전체 고객 기본 점수를 산출하고, 실시간 거래 발생 시
Redis 캐시의 실시간 피쳐를 반영하여 FD-TVS 점수를 즉시 재계산한다.
Triton Dynamic Batching이 개별 실시간 요청을 큐에 모아 마이크로배치로 처리하여
GPU 활용률을 유지한다.

== Training-Serving Skew 방지

*Feature Serving Spec*이 학습과 서빙을 연결한다.
- `feature_selector`가 학습 시 `selected_features_{task}.json`을 출력
- `FeatureServingSpec`이 배포 시점에 이를 로드하여 동일한 피쳐 순서를 보장
- 7개 필수 피쳐 (TDA, Economics, FinEng)는 항상 포함

== Calibration 고려사항

FD-TVS Stage 1은 모든 태스크 확률이 공통 스케일 $[0, 1]$에 있을 것을 요구한다.
확률이 중요한 태스크(이탈, product_stability, cross_sell_count)는 증류 후 Platt scaling을 적용한다(§1 캘리브레이션 참조). 나머지 태스크에는 Temperature Scaling(Guo et al., ICML 2017)을 경량 대안으로 사용한다.

== 3계층 폴백 아키텍처

Lambda 서빙 레이어는 가용성과 신뢰도에 따라 세 계층 중 하나를 선택하는 `FallbackRouter`를 구현한다.

#table(
  columns: (auto, 1fr, 1fr),
  inset: 8pt,
  stroke: 0.4pt + luma(200),
  [*계층*], [*메커니즘*], [*생성 출력*],
  [1계층: 증류 LGBM], [Lambda를 통한 ONNX 모델; 태스크별 Platt 캘리브레이션], [`scores`, `contributing_features` (IG top-5)],
  [2계층: PLE 직접 추론], [SageMaker Endpoint; LGBM SKIP 또는 Lambda cold-start 실패 시 활성화], [`scores`, `contributing_features` (IG top-5)],
  [3계층: 규칙 엔진], [Python 규칙 셋: 13개 태스크별 규칙 + Financial DNA 피쳐 라우팅], [`scores` (휴리스틱), `contributing_features` (규칙 기반)],
)

세 계층 모두 `contributing_features`를 생성하여 성능 저하 서빙 시에도 AI 기본법 제34조 및 금소법 제19조에 따른 설명 컴플라이언스를 유지한다.

=== 규칙 엔진 설계

규칙 엔진은 Financial DNA 피쳐 라우팅을 중심으로 구성된 13개 태스크별 규칙을 구현한다.

- *DNA 라우팅:* 항상소득 고객($"CV" < 0.2$) → 장기 상품 규칙; 일시소득 고객($"CV" >= 0.5$) → 단기·저관여 규칙
- *태스크 규칙:* 13개 태스크 각각에 3--5개 해석 가능한 피쳐 기반 휴리스틱 점수 함수 (예: 이탈 규칙은 최신성 + 빈도 + 고객지원 통화 수 사용)
- *contributing_features:* 규칙이 규칙을 트리거한 상위 3개 피쳐를 선택하여 사유 생성 파이프라인에 `contributing_features`로 반환

== LLM 증류: Gemini Teacher $arrow$ Qwen Student (QLoRA)

=== 두 가지 증류의 구분

#table(
  columns: (auto, 1fr, 1fr),
  inset: 8pt,
  stroke: 0.4pt + luma(200),
  [*측면*], [*예측 모델 증류*], [*LLM 증류*],
  [목적], [예측 정확도], [텍스트 생성 품질],
  [Teacher], [PLE-Cluster-adaTT], [Google Gemini],
  [Student], [LightGBM], [Qwen3-8B (온프레미스)],
  [전달 대상], [Soft labels (logits/probs)], [텍스트 출력 (추천 사유)],
  [손실 함수], [KL Divergence + CE], [Cross-Entropy (SFT)],
  [학습 방법], [Soft label learning], [QLoRA fine-tuning],
)

*AWS 서빙 참고:* 프로덕션 AWS Lambda에서 L2a rewrite와 L2b 검증은 *Bedrock Claude Haiku*를 사용합니다(Qwen3-8B 아님). Qwen3-8B QLoRA는 온프레미스 대안입니다. 두 경로 모두 동일한 PromptSanitizer와 3-Layer Safety Gate를 공유합니다.

=== QLoRA: LoRA 수학적 기초

$ W' = W_0 + Delta W = W_0 + B A $ <lora>

- $W_0 in bb(R)^(d times k)$: 원본 사전학습 가중치 (동결)
- $B in bb(R)^(d times r)$: Down-projection (학습 가능)
- $A in bb(R)^(r times k)$: Up-projection (학습 가능)
- $r << min(d, k)$: Rank ($r = 16$에서 원본 대비 0.78%)

압축률: $r times (d + k) \/ (d times k)$.
Qwen3-8B ($d = k = 4096$, $r = 16$): $131,072 \/ 16,777,216 approx 0.78%$.

=== NF4 양자화

분포 인식(distribution-aware) 양자화로 표준정규분포의 분위수에 양자화 수준을 배치한다.

$ q_i = Phi^(-1)(i \/ (2^k + 1)) $

각 수준이 동일한 확률 질량을 커버하여 Lloyd-Max 최적 조건을 충족한다.

=== QLoRA 메모리 분석

- Full FT (FP16): 가중치 16GB + 옵티마이저 32GB + 그래디언트 16GB = 64GB+ (RTX 4070 불가)
- QLoRA: 베이스 모델 4GB (NF4) + LoRA 어댑터 $tilde$40MB = *6GB로 학습 가능*

=== Self-Consistency 학습 데이터 필터링

$ "consistency"(s_1, s_2, s_3) = min_(i != j) "BERTScore"(s_i, s_j) $ <consistency>

Gemini Teacher로 동일 입력에 대해 3회 출력을 생성하고,
최소 쌍별 BERTScore를 보수적 일관성 척도로 사용한다.
일관된 출력만 학습 데이터에 포함하여 Teacher 환각을 필터링한다.

#pagebreak()


// ============================================================
= 비용 효율 및 운영 요약
// ============================================================

== 비용 효율 요약

#table(
  columns: (1fr, 1fr, 1fr),
  inset: 8pt,
  stroke: 0.4pt + luma(200),
  [*구성 요소*], [*설계 선택*], [*효과*],
  [2-Layer 사유 생성], [L1 Template + L2 LLM], [162 vs 1,000 GPU-hours],
  [LGBM Student], [CPU 추론], [GPU Teacher 대비 1/10 비용],
  [QLoRA], [NF4 + LoRA], [6GB vs 64GB+ 학습],
  [Triton Dynamic Batching], [마이크로배치 큐잉], [GPU 활용률 극대화],
)

== 교차 관심사

=== 금융 도메인 특화

- *증류:* TDA, Economics, FinEng 필수 피쳐를 IG importance와 무관하게 보존
- *스코어링:* Friedman 항상소득가설 기반 DNA modifier, 비가역성 기반 비대칭 리스크 가중
- *그라운딩:* 도메인 전문가 설계 매핑 딕셔너리로 금융 언어 번역
- *사유 생성:* 규정 우선 설계 (규제가 품질을 우선)
- *LLM 증류:* 금융 용어, 컴플라이언스 인식 톤, 상품 특화 지식 전이

=== 전체 파이프라인 정합성

증류 $arrow$ 서빙 $arrow$ 스코어링 $arrow$ 그라운딩 $arrow$ 사유 생성 $arrow$ 감사의
각 단계는 Feature Serving Spec, IG 귀인, 역매핑 딕셔너리라는 공유 계약을 통해
End-to-End 정합성을 유지한다. 어떤 단계에서든 이 계약이 깨지면 하류 단계 전체에
오류가 전파되므로, 단계별 검증이 필수적이다.

// ============================================================
= 운영/감사 에이전트와의 연계

추천사유 품질은 AuditAgent의 AV3 관점에서 3-Tier 체계로 모니터링된다:
- *Tier 1* (전수): SelfChecker pass/revise/reject 비율 추이
- *Tier 2* (샘플): 27개 스트라텀 층화추출 → GroundingValidator(사유↔IG 정합성)
- *Tier 3* (전문가): 월 50~100건 수동 리뷰 → 피드백 루프

InterpretationRegistry → 3-tuple enrichment → TemplateEngine 연결이 완료되어 L1 사유에도 한국어 IG 해석이 반영된다. ReverseMapper가 InterpretationRegistry의 Level RM fallback으로 통합되어 피처 해석 커버리지가 확대되었다.

상세 설계: Design Document 11 (`docs/design/11_ops_audit_agent.typ`)

== 팩트 추출 레이어 (신규, 2026-04)

`FactExtractor`가 피처 값으로부터 *고객 서술적 팩트*를 추출한다.
`InterpretationRegistry`가 피처 레벨 해석을 제공한다면, `FactExtractor`는
고객 레벨 프로파일("예적금 중심 포트폴리오", "리스크 회피 성향")을 제공한다.

=== 룰 기반, LLM 없음

YAML config (`configs/financial/fact_extraction.yaml`)에 정의된 Python 표현식을
샌드박스(`__builtins__` 차단) 환경에서 평가한다. LLM 호출 0회로 결정론적으로 동작한다.
Phase 0 배치 시점에 추출되어 `ContextVectorStore`에 저장되며, 서빙 타임에는
조회만 하여 L2a 프롬프트에 주입한다.

=== L2a 프롬프트 통합

`AsyncReasonOrchestrator.generate_l1()`과 `get_best_reason()` 두 경로에서
`context_store.get_context(customer_id).get("customer_facts", [])`를 조회하여
`context["customer_facts"]`로 SQS 메시지에 포함한다. `_build_llm_prompt()`가
"\#\# 고객 특성 팩트" 섹션으로 최종 프롬프트에 주입한다.
