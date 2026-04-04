// =============================================================================
//  Temporal Ensemble Expert 기술 참조서 — AIOps PLE for Financial
//  v1.0 · 2026-04-01
// =============================================================================

#let anthropic-bg = rgb("#F0EFEA")
#let anthropic-text = rgb("#141413")
#let anthropic-accent = rgb("#CC785C")
#let anthropic-muted = rgb("#6B7280")
#let anthropic-rule = rgb("#D1D5DB")

#set page(
  paper: "a4",
  margin: (top: 2.5cm, bottom: 2.5cm, left: 2.5cm, right: 2.5cm),
  fill: anthropic-bg,
  header: context {
    if counter(page).get().first() > 1 [
      #set text(size: 7.5pt, font: "New Computer Modern", fill: anthropic-muted, tracking: 0.12em)
      #smallcaps[Temporal Ensemble Expert]
      #h(1fr)
      #smallcaps[AIOps PLE for Financial]
      #v(4pt)
      #line(length: 100%, stroke: 0.4pt + anthropic-rule)
    ]
  },
  footer: context {
    let pg = counter(page).get().first()
    if pg > 1 [
      #line(length: 100%, stroke: 0.3pt + anthropic-rule)
      #v(4pt)
      #set text(size: 8pt, font: "New Computer Modern", fill: anthropic-muted)
      #h(1fr)
      — #pg —
      #h(1fr)
    ]
  },
)

#set text(font: "New Computer Modern", size: 10pt, fill: anthropic-text, lang: "ko")
#set par(justify: true, leading: 0.8em, spacing: 1.5em)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")

// ─────────────────── 색상 팔레트 ────────────────────
// Legacy aliases for component compatibility
#let navy   = anthropic-text
#let teal   = anthropic-accent
#let amber  = anthropic-accent
#let indigo = anthropic-accent
#let rose   = anthropic-accent
#let slate  = anthropic-muted

// ─────────────────── 제목 스타일 ────────────────────
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

// ─────────────────── 코드 블록 스타일 ────────────────
#show raw.where(block: true): it => {
  block(
    width: 100%,
    inset: 10pt,
    radius: 4pt,
    fill: rgb("#f8fafc"),
    stroke: 0.5pt + anthropic-rule,
    text(size: 8.5pt, it),
  )
}
#show raw.where(block: false): box.with(
  fill: rgb("#f1f5f9"),
  inset: (x: 3pt, y: 1pt),
  outset: (y: 2pt),
  radius: 2pt,
)

// ─────────────────── 수식 블록 여백 ─────────────────
#show math.equation.where(block: true): it => {
  v(3pt)
  it
  v(3pt)
}

// ─────────────────── 커스텀 컴포넌트 ────────────────
#let note-box(title, body, accent: anthropic-accent) = {
  block(
    width: 100%,
    inset: (left: 14pt, right: 14pt, top: 10pt, bottom: 10pt),
    stroke: (left: 2pt + accent),
    [
      #text(weight: "bold", size: 9.5pt, fill: accent)[#title]
      #v(2pt)
      #text(size: 9.5pt)[#body]
    ],
  )
}

#let warn-box(title, body) = note-box(title, body, accent: anthropic-accent)

#let eq-box(body) = {
  block(
    width: 100%,
    inset: 12pt,
    stroke: (left: 2pt + anthropic-accent),
    body,
  )
}

#let dim(body) = text(size: 8.5pt, fill: anthropic-muted)[#body]

// ─────────────── 테이블 스타일 함수 ─────────────────
#let stbl(cols, ..args) = {
  set text(size: 9pt)
  table(
    columns: cols,
    inset: 7pt,
    stroke: 0.4pt + anthropic-rule,
    fill: (_, y) => if y == 0 { anthropic-accent.lighten(88%) } else if calc.odd(y) { luma(252) },
    ..args,
  )
}

// =================================================================
//  표지
// =================================================================
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
    Temporal Ensemble Expert\
    기술 참조서
  ]

  #v(0.3cm)
  #line(length: 30%, stroke: 0.5pt + anthropic-rule)
  #v(0.3cm)

  #text(size: 12pt, fill: anthropic-muted)[
    Mamba SSM · Liquid Neural Network · PatchTST · Ensemble Gating
  ]

  #v(2cm)

  #text(size: 10pt, fill: anthropic-text)[
    AIOps PLE for Financial Recommendation
  ]

  #v(0.5em)

  #text(size: 9pt, fill: anthropic-muted)[v1.0 · 2026-04-01]

  #v(2cm)

  #block(width: 85%, stroke: (left: 2pt + anthropic-accent), inset: (left: 14pt, right: 14pt, top: 8pt, bottom: 8pt))[
    #text(size: 9.5pt, fill: anthropic-muted)[
      본 문서는 Temporal Ensemble Expert를 구성하는 세 모델 --- Mamba (Selective State Space Model),
      Liquid Neural Network, PatchTST (Patch Time Series Transformer) --- 의 수학적 구조,
      앙상블 게이팅 메커니즘, 그리고 프로젝트 구현 사양을 기술한다.
    ]
  ]
]

#v(1fr)
#pagebreak()

#set page(
  header: context {
    if counter(page).get().first() > 1 [
      #set text(size: 7.5pt, font: "New Computer Modern", fill: anthropic-muted, tracking: 0.12em)
      #smallcaps[Temporal Ensemble Expert]
      #h(1fr)
      #smallcaps[AIOps PLE for Financial]
      #v(4pt)
      #line(length: 100%, stroke: 0.4pt + anthropic-rule)
    ]
  },
  footer: context {
    let pg = counter(page).get().first()
    if pg > 1 [
      #line(length: 100%, stroke: 0.3pt + anthropic-rule)
      #v(4pt)
      #set text(size: 8pt, font: "New Computer Modern", fill: anthropic-muted)
      #h(1fr)
      — #pg —
      #h(1fr)
    ]
  },
)

#outline(indent: 1.5em, depth: 3)

// =================================================================
= 금융 시계열의 복합 구조
// =================================================================

== 정적 피처 vs 시간적 피처

전통적 추천 시스템에서 사용자 표현은 _정적(static)_ 벡터이다.
나이, 직업, 선호 카테고리 등 시점에 무관한 속성으로 구성된다.
그러나 실제 사용자 행동은 시간 축을 따라 끊임없이 변화하며,
단일 스냅샷으로 축약하면 _주기성_, _트렌드_, _계절성_ 같은 핵심 행동 신호가 소멸한다.

#stbl(
  (1.5fr, 2.5fr, 2.5fr),
  table.header[*관점*][*정적 피처*][*시간적 피처*],
  [표현 형태], [고정 벡터 $bold(x) in RR^d$], [시퀀스 $bold(X) in RR^(T times d)$],
  [정보 손실], [시간 축 평균화 $arrow.r$ 패턴 소멸], [순서, 간격, 추세 보존],
  [모델 요구사항], [MLP, Embedding 테이블], [SSM, ODE, Transformer 등 시퀀스 모델],
)

Temporal Expert는 거래와 세션 데이터를 시퀀스로 유지한 채,
시간 차원에 내재된 패턴을 학습하여 64D 표현으로 압축한다.

== 시계열의 세 가지 구성 요소

모든 시계열 $y(t)$는 근본적으로 세 가지 성분의 합으로 분해된다:

#eq-box[
  $ y(t) = T(t) + S(t) + R(t) $

  #dim[
    $T(t)$: 트렌드 (Trend) --- 장기적 방향성 \
    $S(t)$: 계절성 (Seasonality) --- 반복적 주기 패턴 (요일, 월별, 연별) \
    $R(t)$: 잔차 (Residual) --- 트렌드와 계절성으로 설명되지 않는 불규칙 변동
  ]
]

Temporal Expert의 세 모델은 이 분해에 대응한다:
- *Mamba*: 트렌드 (장기 의존성, 선택적 기억을 통한 방향성 포착)
- *PatchTST*: 계절성 (패치 간 어텐션으로 글로벌 주기 포착)
- *LNN*: 잔차 (적응적 시간 상수로 불규칙 변동 처리)

== 금융 거래 데이터의 네 가지 특성

금융 시계열은 일반 시계열과 구별되는 고유한 특성을 갖는다:

#stbl(
  (1.5fr, 3fr, 2fr),
  table.header[*특성*][*설명*][*처리 모델*],
  [스냅샷 불연속], [월말 잔액, 분기 평가 등 이산적 관측], [Mamba 선택적 기억],
  [일별 거래 다발], [하루 수십 건의 결제, 이체], [PatchTST 패치 집계],
  [불규칙 간격], [거래일 vs 휴일, 활동기 vs 휴면기], [LNN 적응적 $tau$],
  [고빈도 이벤트], [카드 승인 실시간 스트림], [Mamba $O(L)$ 선형 처리],
)

== 윈도우 기반 vs 이벤트 기반 하이브리드

프로젝트에서는 거래 데이터를 일별 집계하여 _윈도우 기반 시퀀스_(180 타임스텝)로
구성한다.
이는 Mamba와 PatchTST에 입력된다.
동시에 거래 간 _실제 시간 간격_ 정보가 LNN에 전달되어, 이벤트 기반의 시간 인식 보정을 수행한다.

#stbl(
  (1.5fr, 2.5fr, 2.5fr),
  table.header[*접근법*][*윈도우 기반 (고정 간격)*][*이벤트 기반 (불규칙 간격)*],
  [시간 모델], [$Delta t = "const"$], [$Delta t_i eq.not Delta t_j$],
  [데이터 형태], [$bold(X) in RR^(T times d)$], [${(t_i, bold(x)_i)}$ 이벤트 스트림],
  [적합 모델], [Mamba, PatchTST], [LNN (ODE 기반)],
)

// =================================================================
= Mamba: Selective State Space Model
// =================================================================

== 이론적 배경

#note-box(accent: indigo)[참조 논문][
  Gu & Dao, _"Mamba: Linear-Time Sequence Modeling with Selective State Spaces"_ (NeurIPS 2023). \
  계보: HiPPO (Gu et al., NeurIPS 2020) $arrow.r$ S4 (ICLR 2022) $arrow.r$ Mamba (2023).
]

SSM(State Space Model)을 입력 의존적 선택 메커니즘(S6)으로 확장하여,
RNN의 선형 시간 복잡도와 Transformer의 콘텐츠 인식 능력을 결합한 아키텍처이다.

== 연속 상태 공간 모델

선형 시불변(LTI) 시스템의 기본 형태:

#eq-box[
  $ frac(d bold(x), d t) = bold(A) bold(x) + bold(B) u, quad y = bold(C) bold(x) + bold(D) u $

  #dim[
    $bold(x) in RR^N$: 은닉 상태, $u in RR$: 입력 신호, $y in RR$: 출력 신호 \
    $bold(A) in RR^(N times N)$: 상태 전이 행렬,
    $bold(B) in RR^(N times 1)$: 입력 행렬,
    $bold(C) in RR^(1 times N)$: 출력 행렬
  ]
]

== ZOH 이산화

연속계를 이산 시간 스텝 $Delta$로 변환한다:

$ macron(bold(A)) = exp(Delta dot bold(A)), quad macron(bold(B)) approx Delta dot bold(B) $ <eq-zoh>

$macron(bold(A))$는 "연속 시간 $Delta$만큼 정확히 전진시킨 상태 전이"를 의미한다.
1차 근사 $exp(Delta bold(A)) approx bold(I) + Delta bold(A)$가 Euler method이며,
정확한 행렬 지수는 무한 차수까지 고려한 것이다.
$bold(A)$가 대각 행렬이면 $exp(bold(A) t) = "diag"(e^(a_1 t), dots, e^(a_N t))$로
간소화되며, Mamba가 $bold(A)$를 대각으로 제한하는 이유이다.

== 이산 점화식

$ bold(h)_t = macron(bold(A)) dot bold(h)_(t-1) + macron(bold(B)) dot bold(x)_t, quad bold(y)_t = bold(C)_t dot bold(h)_t $ <eq-recurrence>

직관적으로 "오늘의 상태 = 어제의 기억 $times$ 감쇠율 + 오늘의 새 정보"이다.

== S6 선택 메커니즘

기존 LTI SSM에서 $bold(A), bold(B), bold(C)$는 입력과 무관한 상수이다.
Mamba의 S6는 $Delta, bold(B), bold(C)$를 _입력 의존적으로_ 생성하여 콘텐츠 인식 처리를 구현한다.

#eq-box[
  $ Delta = "softplus"(bold(W)_Delta dot bold(x) + bold(b)_Delta) $ <eq-delta>
  $ bold(B) = bold(W)_B dot bold(x), quad bold(C) = bold(W)_C dot bold(x) $ <eq-bc>

  #dim[
    $bold(W)_Delta in RR^(D times r)$: $Delta$ 프로젝션 ($r = ceil(D \/ 16)$, dt\_rank) \
    softplus 보장: $Delta > 0$ (시간 스텝은 양수)
  ]
]

입력에 따라 $Delta$가 커지면 해당 타임스텝의 정보를 강하게 기억(인코딩)하고,
$Delta$가 작아지면 이전 상태를 유지하며 현재 입력을 무시(망각)한다.

=== 금융 도메인에서의 선택 메커니즘

- 대형 거래 $arrow.r$ 큰 $Delta$ $arrow.r$ 은닉 상태에 강하게 기록 (선택적 기억)
- 소액 일상 거래 $arrow.r$ 작은 $Delta$ $arrow.r$ 이전 상태 유지, 배경 처리 (선택적 망각)
- 이 메커니즘은 금융 이벤트의 _이질적 중요도_를 자연스럽게 모델링한다

== 선택적 순차 스캔

각 타임스텝마다 $Delta_t, bold(B)_t, bold(C)_t$가 달라지므로:

$ bold(h)_t = underbrace(exp(Delta_t dot bold(A)), macron(bold(A))_t) dot bold(h)_(t-1) + underbrace(Delta_t dot bold(B)_t, macron(bold(B))_t) dot bold(x)_t, quad bold(y)_t = bold(C)_t^top bold(h)_t $ <eq-s6-scan>

전체 시퀀스에 대해 $O(L)$ 선형 시간 복잡도로 처리된다.

== Gated MLP + Causal Conv1d 아키텍처

MambaBlock은 SSM을 Gated MLP와 1D 인과 컨볼루션으로 감싸는 구조이다:

+ *Input Projection*: `d_input` $arrow.r$ `d_model`
+ *LayerNorm* $arrow.r$ *in\_proj*: `d_model` $arrow.r$ `2 * d_inner` (두 경로로 분기)
+ *SSM 경로*: Causal Conv1d (kernel=4) $arrow.r$ SiLU $arrow.r$ SelectiveSSM (S6)
+ *Gate 경로*: SiLU 활성화
+ *Element-wise multiply* (SSM 출력 $circle.small$ Gate 출력)
+ *out\_proj*: `d_inner` $arrow.r$ `d_model` + Residual connection

인과 Conv1d (kernel=4)는 시점 $t$에서 $t-3, t-2, t-1, t$만 참조하여
미래 정보 누출 없이 로컬 컨텍스트를 혼합한다.

== A 행렬 초기화와 안정성

$bold(A)$를 $log$ 공간에 저장하고, forward 시 $-exp("A"_"log")$로 복원한다.
이는 모든 고유값이 _항상 음수_임을 보장하여 시스템의 안정성을 확보한다.

초기값은 HiPPO 스타일의 대각 $[1, 2, dots, N]$이다.
더 큰 인덱스의 상태는 더 높은 주파수의 다항식 성분에 대응하므로,
대각 원소 $-1, -2, dots, -N$이 되어 고주파 성분일수록 빠르게 감쇠한다.

== 프로젝트 구현 사양

=== 거래용 Mamba (Transaction)

#stbl(
  (1fr, 1fr, 2.5fr),
  table.header[*파라미터*][*값*][*설명*],
  [`d_model`], [128], [은닉 차원],
  [`d_input`], [16], [card (8D) + deposit (8D) 거래 피처],
  [`d_inner`], [256], [$128 times 2$ (expand=2)],
  [`d_state`], [16], [SSM 상태 벡터 차원],
  [`d_conv`], [4], [1D 인과 컨볼루션 커널 크기],
  [`dt_rank`], [8], [$ceil(128 \/ 16) = 8$],
  [`seq_len`], [180], [180일 거래 시퀀스],
  [출력], [`[B, 128]`], [마지막 타임스텝 `[:, -1, :]`],
)

=== 세션용 Mamba (Session)

#stbl(
  (1fr, 1fr, 2.5fr),
  table.header[*파라미터*][*값*][*설명*],
  [`d_model`], [64], [거래 대비 절반 차원],
  [`d_input`], [8], [세션 피처 차원],
  [`d_inner`], [128], [$64 times 2$ (expand=2)],
  [`seq_len`], [90], [90일 세션 시퀀스],
  [출력], [`[B, 64]`], [마지막 타임스텝],
)

Mamba 총 출력 차원: $128 + 64 = 192$D (txn + session concat).

== 복잡도 분석

#stbl(
  (1.5fr, 1.5fr, 2.5fr),
  table.header[*연산*][*복잡도*][*비고*],
  [순차 스캔], [$O(L dot D dot N)$], [$L$: 시퀀스 길이, $D$: 모델 차원, $N$: 상태 차원],
  [Transformer 대비], [$O(L)$ vs $O(L^2)$], [시퀀스 길이에 선형],
  [메모리], [$O(D dot N)$], [은닉 상태만 유지 (시퀀스 길이 무관)],
)

// =================================================================
= Liquid Neural Network (LNN)
// =================================================================

== 이론적 배경

#note-box(accent: indigo)[참조 논문][
  Hasani et al., _"Liquid Time-constant Networks"_ (AAAI 2021). \
  계보: Neural ODE (Chen et al., NeurIPS 2018) + 생물학적 뉴런 감쇠 모델 + Liquid State Machine (Maass et al., 2002).
]

Neural ODE를 기반으로 하되, 시간 상수 $tau$가 입력에 따라 적응적으로 변하는
연속 시간 신경망이다.
불규칙 시간 간격(금융 거래의 본질적 특성)을 자연스럽게 처리할 수 있다.

== 핵심 ODE

#eq-box[
  $ frac(d bold(h), d t) = frac(-bold(h) + f(bold(x), bold(h)), tau(bold(x), bold(h))) $ <eq-lnn-ode>

  #dim[
    $bold(h) in RR^H$: 은닉 상태, $bold(x)$: 외부 입력 \
    $f(dot, dot)$: 상태 업데이트 함수 (목표 상태),
    $tau(dot, dot) > 0$: 적응적 시간 상수
  ]
]

=== 각 항의 물리적 의미

#stbl(
  (1fr, 3.5fr),
  table.header[*항*][*의미*],
  [$-bold(h)$], [감쇠 (leak): 입력 없이 은닉 상태가 0으로 수렴. 오래된 정보를 자연스럽게 잊음],
  [$f(bold(x), bold(h))$], [구동력 (driving force): 새 입력과 현재 상태에 기반한 목표 상태. $f = tanh(bold(W)_f [bold(x); bold(h)] + bold(b)_f)$],
  [$tau(bold(x), bold(h))$], [시간 상수: 클수록 변화가 느림 (관성, 상태 보존), 작을수록 빠르게 반응. $tau = "Softplus"("MLP"([bold(x); bold(h)])) + 0.1$],
)

== Euler 이산화

연속 ODE를 이산 타임스텝으로 변환:

$ bold(h)_(t+1) = bold(h)_t + Delta t dot frac(-bold(h)_t + f(bold(x)_t, bold(h)_t), tau(bold(x)_t, bold(h)_t)) $ <eq-euler>

$Delta t$는 _실제 시간 간격_(일 단위)으로, 거래 간격이 불규칙하더라도 ODE가 자연스럽게 처리한다.

== 적응적 시간 상수의 금융 도메인 해석

금융 거래 간격은 극도로 불규칙하다:

#stbl(
  (2fr, 1.5fr, 2.5fr),
  table.header[*상황*][*$Delta t$*][*$tau$의 역할*],
  [당일 복수 거래], [$tilde 0.01$ 일], [작은 $tau$: 빠른 반응, 각 거래를 즉시 반영],
  [주말 공백], [$2$ 일], [중간 $tau$: 상태 완만하게 유지],
  [장기 휴면], [$> 30$ 일], [큰 $tau$: 상태 보존, 느린 감쇠],
)

고정 $tau$를 사용하는 RNN/LSTM은 모든 간격을 동일하게 처리한다.
적응적 $tau$는 활발한 거래 시기에는 빠르게 반응하고,
휴면 시기에는 상태를 보존하여 _고객 행동 리듬_에 자동 적응한다.

== SingleStep 모드 설계

프로젝트에서 LNN은 *SingleStep 모드*로 운용된다.
전체 시퀀스를 ODE로 처리하지 않고, Mamba의 최종 은닉 상태에 대해
_단일 ODE 스텝_만 적용한다.

=== 설계 근거

- Mamba가 이미 $O(L)$로 전체 시퀀스 패턴을 포착
- LNN은 _시간 스케일 보정_ 역할만 수행 (중복 시퀀스 처리 회피)
- 계산 비용: $O(1)$ 단일 스텝 vs $O(L)$ 전체 시퀀스 ODE

#note-box[Mamba $arrow.r$ LNN 직렬 구조][
  Mamba가 시퀀스 패턴을 학습한 후,
  LNN이 최종 상태에 시간 인식 보정을 적용하는 직렬 구조이다.
  이는 "트렌드 추출 $arrow.r$ 잔차 보정"의 분해에 대응한다.
]

== 프로젝트 구현 사양

#stbl(
  (1.5fr, 1.5fr, 1.5fr, 1.5fr),
  table.header[*파라미터*][*LNN txn*][*LNN session*][*설명*],
  [`input_dim`], [128], [64], [Mamba 출력 차원],
  [`hidden_dim`], [64], [32], [LNN 은닉 차원],
  [출력], [`[B, 64]`], [`[B, 32]`], [SingleStep 출력],
)

LNN 총 출력 차원: $64 + 32 = 96$D (txn + session concat).

// =================================================================
= PatchTST: Patch Time Series Transformer
// =================================================================

== 이론적 배경

#note-box(accent: indigo)[참조 논문][
  Nie et al., _"A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"_ (ICLR 2023). \
  ViT (Dosovitskiy et al., ICLR 2021)의 패치 개념을 시계열에 적용.
]

시계열을 고정 크기 패치로 분할한 뒤,
각 패치를 토큰으로 취급하여 Self-Attention을 적용한다.
이는 어텐션 비용을 $O(L^2)$에서 $O((L\/P)^2)$로 축소하면서도
다중 스케일 주기 패턴을 효과적으로 포착한다.

== 패치 임베딩

$ bold(p)_i = bold(W)_"proj" dot "flatten"(bold(x)_[((i-1)P+1) : (i P)]) + bold(b)_"proj" $ <eq-patch>

패치 크기 $P = 16$에서 180 타임스텝 거래 시퀀스는 $floor(180 \/ 16) = 11$개 패치로 변환된다.
어텐션 비용: $O(180^2) = 32400$ $arrow.r$ $O(11^2) = 121$.

=== 패치 크기의 금융 도메인 해석

패치 크기 16은 약 2주에 해당하며, 급여 주기 (2주/4주)의 기본 단위와 자연스럽게 정렬된다.
각 패치 내부에서 로컬 패턴 (일별 지출)을 집약하고,
패치 간 어텐션으로 글로벌 주기성 (월간 급여, 분기 보너스)을 포착한다.

== Multi-Head Self-Attention

$ "Attention"(bold(Q), bold(K), bold(V)) = "softmax"(frac(bold(Q) bold(K)^top, sqrt(d_k))) bold(V) $ <eq-attention>

$bold(Q) bold(K)^top$의 $(i,j)$ 원소는 두 패치 벡터의 내적으로, _코사인 유사도_에 비례한다.
Softmax는 이를 확률 분포로 변환하여, 관련성 높은 패치의 정보를 가중합한다.
$sqrt(d_k)$로 나누는 것은 내적 분산을 정규화하여 softmax gradient의 건전성을 유지하기 위함이다.

== 위치 인코딩

정현파(sinusoidal) 위치 인코딩을 사용한다.
6--12개 패치 수준에서는 학습 가능 위치 인코딩 대비 충분한 성능을 제공하며,
시퀀스 길이 일반화에 유리하다.

== 프로젝트 구현 사양

#stbl(
  (1.5fr, 1.5fr, 1.5fr, 2fr),
  table.header[*파라미터*][*PatchTST txn*][*PatchTST session*][*설명*],
  [`d_model`], [64], [32], [Transformer 은닉 차원],
  [`nhead`], [4], [2], [멀티헤드 어텐션 수],
  [`num_layers`], [2], [2], [Transformer 인코더 레이어 수],
  [`patch_size`], [16], [16], [패치 크기 (약 2주)],
  [패치 수], [11], [5], [$floor(L \/ P)$],
  [출력], [`[B, 64]`], [`[B, 32]`], [AdaptiveAvgPool1d],
)

PatchTST 총 출력 차원: $64 + 32 = 96$D (txn + session concat).

== 복잡도 비교

#stbl(
  (2fr, 1.5fr, 2fr),
  table.header[*모델*][*복잡도*][*L=180 기준*],
  [Vanilla Transformer], [$O(L^2)$], [$32,400$],
  [PatchTST ($P=16$)], [$O((L\/P)^2)$], [$121$],
  [Mamba (SSM)], [$O(L)$], [$180$],
)

// =================================================================
= 앙상블 게이팅
// =================================================================

== 게이트 아키텍처

세 모델의 출력을 연결(concat)한 뒤, 학습 가능한 2-layer 게이트 네트워크가
각 모델의 기여도를 동적으로 결정한다:

#eq-box[
  $ bold(g) = "Softmax"(bold(W)_2 dot "ReLU"(bold(W)_1 dot bold(z)_"cat" + bold(b)_1) + bold(b)_2) in RR^3 $ <eq-gate>
  $ bold(y) = sum_(i=1)^3 g_i dot "Proj"_i (bold(z)_i) in RR^(64) $ <eq-ensemble>

  #dim[
    $bold(z)_"cat"$: 세 모델 출력의 연결 (192 + 96 + 96 = 384D) \
    $"Proj"_i$: 각 모델 출력을 공통 64D 공간으로 프로젝션 \
    $g_i$: 모델 $i$의 가중치 ($sum g_i = 1$)
  ]
]

== 입력 분리를 통한 앙상블 다양성

#stbl(
  (1.5fr, 2fr, 2.5fr),
  table.header[*모델*][*입력 경로*][*다양성 확보 근거*],
  [Mamba], [원본 시퀀스 직접 입력], [순차 상태 공간 관점],
  [LNN], [Mamba 최종 상태 (직렬)], [시간 스케일 보정 관점],
  [PatchTST], [원본 시퀀스 독립 입력], [글로벌 어텐션 관점],
)

Mamba $arrow.r$ LNN 직렬 + PatchTST 독립 구조는
입력 분리를 통해 앙상블 다양성을 확보한다.
동일 입력을 사용하면 게이팅 차별화 효과가 감소하므로,
독립 경로 설계가 필수적이다.

== 게이트 엔트로피 모니터링

게이트 분포의 건전성을 Shannon 엔트로피로 감시한다:

$ H(bold(g)) = -sum_(i=1)^3 g_i log_2(g_i) $ <eq-entropy>

#stbl(
  (2fr, 1.5fr, 3fr),
  table.header[*상태*][*엔트로피*][*해석*],
  [균일 분포], [$log_2(3) approx 1.585$ bits], [세 모델이 균등하게 기여 (최대 엔트로피)],
  [건전한 편중], [$0.5 tilde 1.2$ bits], [입력에 따른 적응적 가중치 (정상 작동)],
  [게이트 붕괴], [$< 0.3$ bits], [하나의 모델이 지배, 나머지 학습 중단 (경고)],
)

#warn-box[게이트 붕괴 (Gate Collapse)][
  $H < 0.3$ bits이면 하나의 모델이 거의 모든 가중치를 독점한다.
  나머지 두 모델은 gradient가 차단되어 학습이 중단되므로
  앙상블의 의미가 사라진다.
  학습 중 엔트로피를 주기적으로 로깅하고,
  붕괴 감지 시 게이트 온도 스케일링 또는 엔트로피 보너스 정규화를 적용한다.
]

== 최종 출력 흐름 요약

$
"txn\_seq" [B, 180, 16] &arrow.r "Mamba"_"txn" arrow.r [B, 128] \
"sess\_seq" [B, 90, 8] &arrow.r "Mamba"_"sess" arrow.r [B, 64] \
"Mamba"_"txn" &arrow.r "LNN"_"txn" arrow.r [B, 64] \
"Mamba"_"sess" &arrow.r "LNN"_"sess" arrow.r [B, 32] \
"txn\_seq" &arrow.r "PatchTST"_"txn" arrow.r [B, 64] \
"sess\_seq" &arrow.r "PatchTST"_"sess" arrow.r [B, 32] \
$

Concat: $[192 + 96 + 96 = 384"D"]$ $arrow.r$ Gate $arrow.r$ Weighted sum $arrow.r$ *64D* 최종 출력.

// =================================================================
= 입출력 사양 및 구현 참고사항
// =================================================================

== 전체 입출력 사양

=== 입력

#stbl(
  (2fr, 1.5fr, 3fr),
  table.header[*입력*][*Shape*][*설명*],
  [`txn_seq`], [`[B, 180, 16]`], [180일 거래 시퀀스, card 8D + deposit 8D],
  [`session_seq`], [`[B, 90, 8]`], [90일 세션 시퀀스, 8D 세션 피처],
)

=== 중간 표현

#stbl(
  (2fr, 1fr, 1fr, 1fr),
  table.header[*모델*][*txn 출력*][*session 출력*][*합계*],
  [Mamba (SSM)], [128D], [64D], [192D],
  [LNN (ODE)], [64D], [32D], [96D],
  [PatchTST (Attn)], [64D], [32D], [96D],
  [*Concat*], [], [], [*384D*],
)

=== 출력

#stbl(
  (2fr, 1.5fr, 3fr),
  table.header[*출력*][*Shape*][*설명*],
  [Temporal Expert 출력], [`[B, 64]`], [게이팅 가중합 후 최종 표현],
  [Gate weights], [`[B, 3]`], [Mamba, LNN, PatchTST 기여도 (모니터링용)],
)

출력 64D 벡터는 PLE의 CGC Gate Attention으로 전달되어,
다른 6개 Expert (PersLay, DeepFM, LightGCN, Unified H-GCN, Causal, OT)의 출력과
태스크별로 동적 결합된다.

== 세 모델의 역할 비교 요약

#stbl(
  (1.2fr, 2fr, 2fr, 1.5fr),
  table.header[*모델*][*포착하는 시간 패턴*][*메커니즘*][*복잡도*],
  [Mamba], [장기 순차 의존성 (트렌드)], [Selective State Space (S6)], [$O(L)$ 선형],
  [LNN], [불규칙 시간 간격 (잔차)], [적응적 시간 상수 ODE], [$O(1)$ 단일 스텝],
  [PatchTST], [글로벌 주기성 (계절성)], [패치 단위 Self-Attention], [$O((L\/P)^2)$],
)

== 시계열 분석 세대별 발전과 본 프로젝트의 위치

#stbl(
  (0.8fr, 2fr, 2fr, 1.5fr),
  table.header[*세대*][*접근법*][*한계*][*극복 모델*],
  [1세대], [ARIMA, Exponential Smoothing], [선형 가정, 수작업 차분], [LSTM, GRU],
  [2세대], [LSTM, GRU (게이트 기반 RNN)], [$O(L)$ 순차 병목, vanishing gradient], [Transformer],
  [3세대], [Transformer (Self-Attention)], [$O(L^2)$ 복잡도, 순서 정보 약함], [SSM, PatchTST],
  [4세대], [SSM + ODE + Patch Transformer], [모델 복잡도, 게이트 붕괴 위험], [본 프로젝트],
)

== 구현 시 유의사항

=== 데이터 리키지 방지

- 시퀀스 데이터의 마지막 타임스텝이 레이블 산출 기간과 겹치지 않도록 `gap_days` 설정 필수 (최소 7일)
- Mamba의 인과 Conv1d는 미래 정보를 참조하지 않음 (left-padding)
- PatchTST의 어텐션은 causal mask 없이 전체 시퀀스를 참조하나, 입력 시퀀스 자체가 레이블 기간 이전으로 절단되어 있으므로 리키지 위험 없음

=== 수치 안정성

- Mamba $bold(A)$ 행렬: $-exp("A"_"log")$로 음수 보장 (발산 방지)
- Mamba $Delta$: softplus로 양수 보장 ($Delta > 0$)
- LNN $tau$: $"Softplus"(dot) + 0.1$ 하한 (영점 방지)
- 게이트 엔트로피 $< 0.3$ bits 시 경고 로깅

=== 성능 최적화

- Mamba sequential scan: 프로토타입은 Python 루프, 프로덕션은 `mamba-ssm` CUDA 커널 권장
- AMP (Mixed Precision) 활성화: g4dn T4 GPU에서 약 2배 속도 향상
- 배치 크기: VRAM과 데이터 규모에 따라 조정 (대규모 데이터 4096 권장)

=== 참고 문헌

#set text(size: 9pt)

#table(
  columns: (2.5fr, 3.5fr, 1.5fr),
  inset: 6pt,
  stroke: 0.4pt + rgb("#e2e8f0"),
  fill: (_, y) => if y == 0 { navy.lighten(88%) } else if calc.odd(y) { luma(252) },
  table.header[*구성 요소*][*논문*][*발표*],
  [Mamba SSM], [Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"], [NeurIPS 2023],
  [LNN], [Hasani et al., "Liquid Time-constant Networks"], [AAAI 2021],
  [PatchTST], [Nie et al., "A Time Series is Worth 64 Words"], [ICLR 2023],
  [Neural ODE], [Chen et al., "Neural Ordinary Differential Equations"], [NeurIPS 2018],
  [S4 (SSM)], [Gu et al., "Efficiently Modeling Long Sequences with Structured State Spaces"], [ICLR 2022],
  [HiPPO], [Gu et al., "HiPPO: Recurrent Memory with Optimal Polynomial Projections"], [NeurIPS 2020],
  [ViT], [Dosovitskiy et al., "An Image is Worth 16x16 Words"], [ICLR 2021],
  [CfC], [Hasani et al., "Closed-form Continuous-depth Models"], [Nature MI 2022],
)
