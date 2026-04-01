// =============================================================================
//  피처 엔지니어링 기술 참조서 — AWS PLE for Financial
//  v1.0 · 2026-04-01
// =============================================================================

// ─────────────────────────── 색상 팔레트 ───────────────────────────
#let navy   = rgb("#1e3a5f")
#let teal   = rgb("#0d9488")
#let amber  = rgb("#d97706")
#let indigo = rgb("#4f46e5")
#let rose   = rgb("#e11d48")
#let slate  = rgb("#64748b")

// ─────────────────────────── 페이지 설정 ───────────────────────────
#set page(
  paper: "a4",
  margin: (top: 72pt, bottom: 56pt, left: 56pt, right: 48pt),
  header: context {
    if counter(page).get().first() > 1 {
      set text(8pt, fill: slate)
      grid(
        columns: (1fr, 1fr),
        align(left)[피처 엔지니어링 기술 참조서],
        align(right)[AWS PLE for Financial · v1.0],
      )
      v(-4pt)
      line(length: 100%, stroke: 0.3pt + slate.lighten(60%))
    }
  },
  footer: context {
    set text(8pt, fill: slate)
    let pg = counter(page).get().first()
    if pg > 1 {
      line(length: 100%, stroke: 0.3pt + slate.lighten(60%))
      v(4pt)
      align(center)[#pg]
    }
  },
)

// ─────────────────────────── 기본 텍스트 ──────────────────────────
#set text(size: 10pt, lang: "ko")
#set par(justify: true, leading: 0.72em)

// ─────────────────────────── 코드 블록 ───────────────────────────
#show raw.where(block: true): it => {
  block(
    width: 100%,
    inset: 12pt,
    radius: 5pt,
    fill: rgb("#f8fafc"),
    stroke: 0.5pt + rgb("#e2e8f0"),
    text(size: 8.5pt, it),
  )
}
#show raw.where(block: false): box.with(
  fill: rgb("#f1f5f9"),
  inset: (x: 3pt, y: 1pt),
  outset: (y: 2pt),
  radius: 2pt,
)

// ─────────────────────── 수식 블록 여백 ──────────────────────────
#show math.equation.where(block: true): it => {
  v(4pt)
  it
  v(4pt)
}

// ─────────────────────────── 제목 스타일 ──────────────────────────
#show heading.where(level: 1): it => {
  pagebreak(weak: true)
  v(8pt)
  block(width: 100%)[
    #text(18pt, weight: "bold", fill: navy)[#it.body]
    #v(-2pt)
    #line(length: 100%, stroke: 2pt + teal)
  ]
  v(8pt)
}
#show heading.where(level: 2): it => {
  v(14pt)
  block(width: 100%)[
    #grid(
      columns: (4pt, 1fr),
      gutter: 10pt,
      rect(width: 4pt, height: 16pt, fill: teal, radius: 2pt),
      text(13pt, weight: "bold", fill: navy.lighten(15%))[#it.body],
    )
  ]
  v(6pt)
}
#show heading.where(level: 3): it => {
  v(10pt)
  text(11pt, weight: "bold", fill: slate.darken(30%))[#it.body]
  v(4pt)
}

// ───────────────────── 커스텀 컴포넌트 ────────────────────────────
#let note(title, body, accent: teal) = {
  block(
    width: 100%,
    inset: (left: 16pt, rest: 12pt),
    radius: 5pt,
    fill: accent.lighten(93%),
    stroke: (left: 3pt + accent, rest: 0.5pt + accent.lighten(70%)),
    [
      #text(weight: "bold", size: 9.5pt, fill: accent.darken(15%))[#title]
      #v(2pt)
      #text(size: 9.5pt)[#body]
    ],
  )
}
#let warn(title, body) = note(title, body, accent: amber)
#let eq-highlight(body) = {
  block(
    width: 100%,
    inset: 14pt,
    radius: 5pt,
    fill: rgb("#eef2ff"),
    stroke: 0.5pt + indigo.lighten(60%),
    body,
  )
}
#let chip(label, color: teal) = {
  box(
    inset: (x: 6pt, y: 2pt),
    radius: 3pt,
    fill: color.lighten(88%),
    text(8pt, weight: "bold", fill: color.darken(10%))[#label],
  )
}
#let dim-label(body) = text(size: 8.5pt, fill: slate)[#body]

// ───────────────── 테이블 기본 스타일 함수 ────────────────────────
#let styled-table(cols, ..args) = {
  set text(size: 9pt)
  table(
    columns: cols,
    inset: 8pt,
    stroke: 0.4pt + rgb("#e2e8f0"),
    fill: (_, y) => if y == 0 { navy.lighten(88%) } else if calc.odd(y) { luma(252) },
    ..args,
  )
}


// =====================================================================
//  표지
// =====================================================================

#v(60pt)
#align(center)[
  #block(width: 85%)[
    #rect(
      width: 100%,
      inset: 0pt,
      stroke: none,
      fill: gradient.linear(navy, teal, angle: 135deg),
      radius: 10pt,
    )[
      #v(40pt)
      #align(center)[
        #text(26pt, weight: "bold", fill: white, tracking: 1pt)[
          피처 엔지니어링\
          기술 참조서
        ]
        #v(12pt)
        #line(length: 40%, stroke: 1pt + white.transparentize(50%))
        #v(12pt)
        #text(11pt, fill: white.transparentize(20%))[
          11개 학제 · 경제학 · 화학 · 역학 · 범죄학 · 파동 · TDA · HMM · GMM · Mamba · Graph · 3-Stage 정규화
        ]
      ]
      #v(40pt)
    ]
    #v(30pt)
    #grid(
      columns: (1fr, 1fr),
      align(left)[
        #text(10pt, fill: slate)[프로젝트] \
        #text(12pt, weight: "bold")[AWS PLE for Financial] \
        #text(9pt, fill: slate)[PLE-adaTT Multi-Task Recommendation]
      ],
      align(right)[
        #text(10pt, fill: slate)[버전] \
        #text(12pt, weight: "bold")[v1.0] \
        #text(9pt, fill: slate)[2026-04-01]
      ],
    )
    #v(24pt)
    #line(length: 100%, stroke: 0.5pt + slate.lighten(60%))
    #v(12pt)
    #text(9.5pt, fill: slate)[
      이 문서는 AWS PLE for Financial 프로젝트의 피처 엔지니어링 파이프라인 전체를 기술한다.
      11개 학제 간 피처 생성기(Economics, Chemical Kinetics, SIR, Crime Pattern, Wave Interference,
      TDA, HMM, GMM, Mamba, Graph, Base)의 이론적 근거, 수학적 정의, 출력 피처 명세,
      그리고 3-Stage 정규화 파이프라인을 포함한다.
    ]
    #v(16pt)
  ]
]

#pagebreak()

// =====================================================================
//  목차
// =====================================================================
#outline(title: [목차], indent: 1.5em, depth: 3)

#v(12pt)
#warn[설계 vs 구현 차원 안내][
  본 문서는 *풀뱅크 설계(734D)*를 기준으로 작성되었습니다. 현재 Santander 벤치마크 구현은 *316D (12 feature groups)*입니다. 실제 구현의 차원 명세는 `outputs/phase0/feature_schema.json`을 참조하십시오. 부록 "설계 vs 구현 차원 매핑"에서 각 그룹별 차이를 확인할 수 있습니다.
]

// =====================================================================
//  1. 피처 설계 철학
// =====================================================================
= 피처 설계 철학

== 다학제 접근의 근거

전통적 피처 엔지니어링은 통계학이라는 단일 렌즈로 데이터를 바라본다. 평균, 분산, 빈도, 상관계수는 강력하지만, 데이터가 가진 구조의 일부만을 드러낸다. 본 시스템은 11개 학문 분야의 수학적 프레임워크를 동시에 적용하여 데이터의 서로 다른 측면을 추출한다.

#styled-table(
  (1fr, 1.2fr, 1.5fr),
  table.header([*학문 분야*], [*포착하는 패턴*], [*통계학만으로 보이지 않는 것*]),
  [경제학], [소득 구조, 탄력성, 시간 선호], [항상소득 vs 일시소득 분해, 인과적 소비 방향성],
  [화학 반응속도론], [변환의 속도, 장벽, 촉매], [상태 전환의 에너지 장벽, 2차 도함수(가속도)],
  [역학 (SIR)], [채택 확산, 면역/이탈], [집단 수준 동역학, 전파 임계값 $R_0$],
  [범죄학], [루틴, 이탈, 이상 패턴], [시간의 원형적 성격, Burstiness],
  [파동 물리학], [주기 분해, 위상 동기화], [FFT 주파수 영역, PLV 위상 결합],
  [TDA], [위상적 형태, 구멍, 연결성], [좌표 불변 구조, 다중 해상도 관찰],
  [HMM], [숨겨진 상태 전이], [관측 불가능한 잠재 단계의 확률적 추론],
  [GMM], [소프트 클러스터링], [확률적 유형 할당, 불확실성 정량화],
  [Mamba SSM], [시계열 장기 의존성], [선택적 상태 공간의 비선형 시간 표현],
  [Graph (LightGCN)], [협업 필터링 신호], [다중 홉 간접 선호, 계층 구조],
  [Base 통계], [인구통계, 보유, 거래], [RFM, 카테고리 분포, 채널 다양성],
)

== 구조적 동형사상

다학제 피처 엔지니어링에서 "아날로지"는 단순한 비유가 아니다. *구조적 동형사상(structural isomorphism)*에 기반한다. 두 시스템의 표면적 대상(분자 vs 소비자)은 다르지만, 그 대상들 사이의 _관계 구조_가 수학적으로 동일할 때 구조적 동형사상이 성립한다.

#note[핵심 원리][
  화학에서 "반응물 농도가 절반으로 줄어드는 시간"과 금융에서 "거래 빈도가 절반으로 줄어드는 시간"은 모두 지수 감쇠(exponential decay)라는 동일한 수학적 구조를 공유한다. 수식이 동일하다면, 해당 수식이 포착하는 패턴도 동일하다. 대상이 무엇이냐는 수식의 유효성에 영향을 주지 않는다.
]

== 피처의 이중 역할

본 시스템의 피처는 두 가지 역할을 동시에 수행한다.

+ *예측 입력*: PLE-adaTT 모델의 입력 텐서로서 18개 태스크의 예측에 기여한다.
+ *전문가 라우팅 신호*: GMM의 소프트 할당 확률($gamma_(n k)$)은 GroupTaskExpertBasket의 20개 서브헤드를 가중 결합하는 라우팅 신호로 사용되고, HMM 48D는 별도 입력 경로(separate input)를 통해 전용 Projector에 공급된다.

== 전체 피처 텐서 구성

#styled-table(
  (1.2fr, 0.6fr, 2fr),
  table.header([*피처 블록*], [*차원*], [*구성 요소*]),
  [Base], [238D], [RFM(34) + Category(64) + Transaction Stats(76) + Product Diversity(12) + ...],
  [Multi-Source], [91D], [Deposit + Credit + Investment + Digital 등],
  [Extended-Source], [84D], [Insurance + Refund + Consultation + STT 등],
  [Domain], [159D], [TDA(70) + GMM(22) + Mamba(50) + Economics(17)],
  [Model-Derived], [27D], [HMM summary(5) + Bandit/MAB(4) + LNN(18)],
  [Multidisciplinary], [24D], [Chemical(6) + SIR(5) + Crime(5) + Wave(8)],
  [Merchant Hierarchy], [21D], [MCC 계층 좌표 + Brand 임베딩],
  [*합계 (normalized)*], [*644D*], [],
  [Raw power-law copy], [90D], [멱법칙 컬럼의 log1p 원본 (스케일링 미적용)],
  [*Main Tensor 총합*], [*734D*], [644D normalized + 90D raw power-law],
)

#v(4pt)
#dim-label[별도 입력: HMM Triple-Mode 48D + Hyperbolic 20D = 68D (separate input path)]


// =====================================================================
//  2. 경제학 피처 (17D)
// =====================================================================
= 경제학 피처 (Economics, 17D)

#chip[Domain Features] #chip(color: indigo)[Friedman PIH] #chip(color: amber)[17D / 734D]

== 이론적 근거: Friedman 항상소득가설

Friedman(1957)의 항상소득가설(Permanent Income Hypothesis, PIH)은 관측 소득을 항상 성분과 일시 성분으로 분해한다.

#eq-highlight[
  $ Y_t = Y_t^P + Y_t^T $

  여기서 $Y_t^P$는 항상소득(장기 안정), $Y_t^T$는 일시소득(임시 변동)이다. 소비 함수는 다음을 따른다:
  $ C_t = k(r, w, u) dot Y_t^P $

  핵심 함의: 소비자는 항상소득에 비례하여 소비하며, 일시소득은 저축/투자로 향한다.
]

=== 소득 분해 추정 방법

#styled-table(
  (1fr, 2fr, 0.8fr),
  table.header([*방법*], [*수식*], [*복잡도*]),
  [이동평균], [$hat(Y)_t^P = 1/L sum_(i=0)^(L-1) Y_(t-i)$, $L=12$], [Low],
  [HP Filter], [$min_tau {sum_t (Y_t - tau_t)^2 + lambda sum_t [(tau_(t+1)-tau_t) - (tau_t - tau_(t-1))]^2}$], [Medium],
  [Kalman Filter], [State: $Y_(t+1)^P = Y_t^P + eta_t$, Obs: $Y_t = Y_t^P + epsilon_t$], [High],
)

HP Filter의 1차 조건은 $(I + lambda D^top D)tau = Y$로, 양정치 띠 행렬 시스템이며 Cholesky 분해로 $O(T)$에 풀린다. Ravn-Uhlig(2002) 표준에 따라 월간 데이터에 $lambda = 14400$을 사용한다.

== 소득 분해 출력 (8D)

#styled-table(
  (1.5fr, 2fr, 2fr),
  table.header([*피처*], [*수식*], [*금융 해석*]),
  [`permanent_income_avg`], [$"mean"(hat(Y)^P)$], [장기 안정 소득 수준],
  [`permanent_income_stability`], [$sigma(hat(Y)^P) / mu(hat(Y)^P)$], [항상소득의 CV; 낮으면 안정 직업],
  [`permanent_income_growth`], [$(hat(Y)_T^P - hat(Y)_1^P) / hat(Y)_1^P$], [소득 궤적 (카드 등급 업그레이드 지표)],
  [`permanent_income_trend`], [REGR_SLOPE / polyfit], [강건한 장기 성장 방향],
  [`transitory_income_avg`], [$"mean"(hat(Y)^T)$], [보너스 빈도 지표 (이론적으로 $approx 0$)],
  [`transitory_income_volatility`], [$sigma(hat(Y)^T)$], [소득 불확실성 크기],
  [`transitory_income_max`], [$max(hat(Y)^T)$], [최대 보너스 이벤트 프록시],
  [`bonus_frequency`], [$"count"(hat(Y)^T > 0.5 hat(Y)^P) / N$], [대규모 보너스 발생 비율],
)

== 미시경제 행동 피처 (9D)

=== 소득 탄력성

#eq-highlight[
  $ epsilon_Y = frac(partial Q, partial Y) dot frac(Y, Q) = frac(d ln Q, d ln Y) $

  - $epsilon_Y > 1$: 사치재 소비 성향 (소득보다 소비가 빠르게 성장)
  - $0 < epsilon_Y < 1$: 필수재 소비 성향
  - $epsilon_Y < 0$: 열등재 (소득 증가 시 소비 감소)
]

이산 호탄력성으로 구현한다: $hat(epsilon)_Y = 1/T sum_(t=1)^T (Delta S_t / S_(t-1)) / (Delta Y_t / Y_(t-1))$

=== 소비 평활화

Hall(1978)은 PIH와 합리적 기대를 결합하여 최적 소비가 랜덤 워크를 따름을 보였다.

$ C_t = C_(t-1) + epsilon_t, quad epsilon_t tilde "WN"(0, sigma^2) $

`consumption_smoothing` $= mu / sigma$ (역CV, 소비의 Sharpe ratio). 높을수록 이론적 최적에 근접.

=== 시간 할인

$ V_0 = sum_(t=0)^T beta^t u(C_t), quad 0 < beta < 1 $

`discount_rate_proxy` $=$ 전반기 소비 / 후반기 소비. 값이 클수록 $beta$가 작으며 현재 편향이 강하다. 즉시 혜택(당일 캐시백)을 선호하는 고객 식별.

=== 기타 행동 피처

#styled-table(
  (1.5fr, 2fr, 2fr),
  table.header([*피처*], [*수식*], [*해석*]),
  [`spending_diversification`], [$H = -sum_i s_i ln(s_i)$], [Shannon 엔트로피; 카테고리 다양성],
  [`category_hhi`], [$"HHI" = sum_i s_i^2$], [허핀달 지수; 소비 집중도],
  [`savings_propensity`], [$(Y - C) / Y$], [저축 성향; 미래 한계효용 기대 반영],
  [`price_sensitivity`], [환불 비율 프록시], [가격 탄력성의 행동적 대리],
  [`cross_category_elasticity`], [카테고리 수 시간 변동], [교차 탄력성 프록시],
)

Shannon 엔트로피와 HHI는 Renyi 엔트로피 $H_alpha = 1/(1-alpha) ln(sum s_i^alpha)$의 특수 사례이다: Shannon은 $alpha -> 1$, HHI는 $alpha = 2$ ($H_2 = -ln("HHI")$).


// =====================================================================
//  3. 화학 반응속도론 피처 (6D)
// =====================================================================
= 화학 반응속도론 피처 (Chemical Kinetics, 6D)

#chip[Multidisciplinary] #chip(color: indigo)[Arrhenius Equation] #chip(color: amber)[6D / 24D]

== 이론적 근거

물리화학의 세계관에서 모든 변환(반응)은 에너지 지형(energy landscape) 위에서 일어난다. 시스템이 한 상태에서 다른 상태로 이동하려면 에너지 장벽을 넘어야 하며, 장벽의 높이가 전환의 난이도를, 넘는 빈도가 전환의 속도를 결정한다.

=== Arrhenius 방정식

#eq-highlight[
  $ k = A e^(-E_a \/ R T) $

  카테고리 전환 빈도($k$)는 진입 장벽($E_a$)이 감소하거나 소비자 활동성($T$)이 증가할 때 지수적으로 증가한다.
]

=== 반감기

1차 반응의 반감기 $T_(1\/2) = ln 2 \/ k$는 거래 간격의 중앙값에 대응한다. 반감기가 짧은 고객은 빠른 거래 주기를 가진다.

=== 2차 유한차분

소비 가속도(spending acceleration)는 이산 2차 도함수이다:

$ f''(t) approx f(t+Delta t) - 2f(t) + f(t-Delta t) $

Taylor 전개에서 $f'$ 항이 상쇄되어 유도되며, 가속(양)과 감속(음)을 구분한다.

== 출력 피처 (6D)

#styled-table(
  (1.5fr, 2fr, 2fr),
  table.header([*피처*], [*정의*], [*금융 해석*]),
  [`new_category_activation_rate`], [최근 30일 신규 MCC / 활성 MCC], [역활성화 에너지 프록시],
  [`spending_half_life`], [거래 간격 중앙값 (일)], [화학 반감기 $T_(1\/2) = ln 2 / k$],
  [`spending_acceleration`], [$f(t+1) - 2f(t) + f(t-1)$], [2차 유한차분: 가속(+) 또는 감속(-)],
  [`dormancy_reactivation_rate`], [W1 존재, W2 부재, W3 재출현 MCC], [촉매적 재활성화율],
  [`catalyst_sensitivity`], [월초 일평균 소비 / 월말 일평균 소비], [급여일 촉매 탄력성],
  [`saturation_proximity`], [$"max" / ("avg" + "std")$], [소비 천장 근접도],
)


// =====================================================================
//  4. SIR 역학 피처 (5D)
// =====================================================================
= SIR 역학 피처 (Epidemiology, 5D)

#chip[Multidisciplinary] #chip(color: indigo)[Compartmental Model] #chip(color: amber)[5D / 24D]

== 이론적 근거: Kermack-McKendrick 모델

#eq-highlight[
  $ frac(d S, d t) = -beta S I, quad frac(d I, d t) = beta S I - gamma I, quad frac(d R, d t) = gamma I $

  기본 재생산수: $R_0 = beta / gamma$. $R_0 > 1$이면 채택이 자기강화적으로 확산, $R_0 < 1$이면 자연 소멸.
]

== 금융 도메인 매핑

#styled-table(
  (0.8fr, 1fr, 2fr),
  table.header([*구획*], [*역학*], [*금융 해석*]),
  [S (감수성)], [미감염], [인구 Top-15 MCC 중 미사용 카테고리],
  [I (감염)], [현재 전파 중], [최근 30일 일평균 빈도 > 이전 기간],
  [R (회복)], [면역/회복], [과거 사용했으나 최근 30일 비활성 카테고리],
)

감염 분류 기준은 기간 길이 보정 계수 $30/(L-30)$을 적용한다.

== 출력 피처 (5D)

#styled-table(
  (1.5fr, 2.5fr),
  table.header([*피처*], [*설명*]),
  [`sir_susceptible`], [S 비율. 높으면 추천 기회 공간이 넓음 (exploration-ready)],
  [`sir_infected`], [I 비율. 높으면 활발한 채택 중 (cross-sell 최적 타이밍)],
  [`sir_recovered`], [R 비율. 높으면 수축 중 (retention 캠페인 대상)],
  [`sir_r0`], [기본 재생산수. 채택 확산의 자기강화 정도],
  [`sir_infection_rate`], [채택 속도. $beta S$에 대응],
)

== 고객 프로파일링

#styled-table(
  (1fr, 0.5fr, 0.5fr, 0.5fr, 2fr),
  table.header([*유형*], [*S*], [*I*], [*R*], [*해석*]),
  [탐색 준비], [High], [Low], [Low], [추천 기회 공간 최대],
  [활발 채택], [Mid], [High], [Low], [교차 판매 최적 시점],
  [안정 사용], [Low], [Low], [Low], [충성 고객],
  [수축], [Low], [Low], [High], [이탈 방지 캠페인 대상],
)


// =====================================================================
//  5. 범죄학 피처 (5D)
// =====================================================================
= 범죄학 피처 (Crime Pattern / Routine Activity, 5D)

#chip[Multidisciplinary] #chip(color: indigo)[Routine Activity Theory] #chip(color: amber)[5D / 24D]

== 이론적 근거

Cohen & Felson(1979)의 일상활동이론에서 행동은 일상 루틴에 의해 결정되며, 루틴의 이탈이 비정상 사건의 기회를 만든다.

=== Burstiness (Barabasi, 2005)

#eq-highlight[
  $ B = frac(sigma_tau - mu_tau, sigma_tau + mu_tau) in [-1, 1] $

  - $B = -1$: 완벽히 규칙적 간격 (정기 결제)
  - $B = 0$: 포아송 과정 (무작위)
  - $B = +1$: 극단적 군집 (폭발적 쇼핑)
]

인간 행동은 포아송 과정이 아니라, 짧은 폭발(burst) 후 긴 휴지(rest)가 반복되는 중후꼬리 분포를 따른다.

=== 원형 통계 (Circular Statistics)

거래 시각 분석에 필수적이다. 23시와 1시 사이의 거리는 22시간이 아니라 2시간이다.

#eq-highlight[
  $ overline(bold(R)) = (1/n sum cos theta_i, 1/n sum sin theta_i), quad "CV" = 1 - |overline(bold(R))| $

  여기서 $theta = 2 pi h \/ 24$. 유클리드 거리는 원형 시간을 왜곡하지만, 원형 통계는 올바르게 집중도를 측정한다.
]

== 출력 피처 (5D)

#styled-table(
  (1.5fr, 2.5fr),
  table.header([*피처*], [*설명*]),
  [`txn_burstiness`], [거래 간격의 Burstiness. 소비 리듬의 규칙성/폭발성],
  [`time_circular_variance`], [거래 시각의 원형 분산. 시간대 집중도],
  [`routine_stability`], [루틴 안정성. 요일별 패턴의 일관성],
  [`breakpoint_count`], [변조점 수. 소비 루틴이 깨진 횟수],
  [`anomaly_score`], [이상 패턴 점수. 루틴 이탈의 강도],
)


// =====================================================================
//  6. 파동 간섭 피처 (8D)
// =====================================================================
= 파동 간섭 피처 (Wave Interference, 8D)

#chip[Multidisciplinary] #chip(color: indigo)[Spectral Analysis] #chip(color: amber)[8D / 24D]

== 이론적 근거

파동 물리학의 세계관에서 여러 파동이 중첩될 때, 위상 관계에 따라 보강(진폭 증가) 또는 상쇄(진폭 감소)가 일어난다.

=== KL 발산

#eq-highlight[
  $ D_"KL"(P||Q) = sum P(x) ln frac(P(x), Q(x)) $

  주중/주말 소비 분포의 KL 발산은 두 맥락에서의 행동이 얼마나 다른지를 _비트 단위_로 정량화한다. Gibbs 부등식(Jensen 부등식의 결과)에 의해 항상 비음수이다.
]

=== Phase Locking Value (PLV)

신경과학의 기능적 연결성에서 차용한 개념이다.

$ "PLV" = frac(1, T) |sum_(t=1)^T e^(j(phi_x (t) - phi_y (t)))| $

Hilbert 변환으로 추출한 두 카테고리 소비 리듬의 위상 차이 일관성을 측정한다.

=== Cross-spectral Coherence

$ C_(x y)(f) = frac(|S_(x y)(f)|^2, S_(x x)(f) dot S_(y y)(f)) $

주파수 분해능 상관으로, 어떤 주기에서 카테고리가 동기화되는지 식별한다.

== 출력 피처 (8D)

#styled-table(
  (1.5fr, 2.5fr),
  table.header([*피처*], [*설명*]),
  [`weekday_weekend_kl`], [주중/주말 소비 분포 KL 발산],
  [`spectral_entropy`], [파워 스펙트럼의 정규화 Shannon 엔트로피. 소비 주기성의 예측 가능성],
  [`dominant_period`], [FFT에서 가장 강한 주파수 성분의 주기 (일)],
  [`phase_locking_value`], [PLV. 카테고리 간 소비 리듬의 위상 동기화],
  [`cross_coherence`], [교차 스펙트럼 코히어런스. 주파수별 카테고리 상관],
  [`hhi_shift`], [HHI 변화량. 소비 집중화/분산화 추세],
  [`category_sync_ratio`], [동기화 비율. 같은 위상의 카테고리 쌍 비율],
  [`interference_strength`], [간섭 강도. 카테고리 간 보강/상쇄 정도],
)

== 정보이론적 정당성

네 다학제 모듈은 거의 직교하는 투영을 포착한다:
- *화학 반응속도론*: 시간의 _미분 구조_ (1차, 2차 도함수)
- *역학 확산*: _상태 공간 전이 구조_ (S -> I -> R)
- *범죄 패턴*: 시계열의 _통계적 질감_ (주기성, 군집성, 분산)
- *파동 간섭*: _주파수 영역 스펙트럼 구조_ (FFT, 코히어런스, 위상)

교차 모듈 조합이 개별 모듈에서 보이지 않는 패턴을 드러낸다: 예를 들어 높은 `catalyst_sensitivity` + 높은 `burstiness` = 급여일 폭발 소비자 (월초 타겟 프로모션 최적).


// =====================================================================
//  7. TDA 피처 (70D)
// =====================================================================
= TDA 피처 (Topological Data Analysis, 70D)

#chip[Domain Features] #chip(color: indigo)[Persistent Homology] #chip(color: amber)[70D / 159D]

== 핵심 아이디어

#note[TDA의 핵심 원리][
  데이터 포인트에 점점 커지는 공을 씌우면서, 연결 성분/구멍/빈 공간 같은 위상적 특성이 _언제 나타나고 언제 사라지는지_를 추적한다. 오래 지속되는 특성은 신호이고, 금방 사라지는 특성은 노이즈이다.
]

TDA의 강점:
+ *좌표 불변성*: 데이터를 회전하거나 연속 변형해도 위상적 특성 보존. 피처 스케일링에 강건.
+ *다중 해상도 관찰*: 단일 임계값이 아닌 모든 스케일을 동시에 고려.
+ *노이즈 내성*: 오래 지속되는 특성만 취하여 허위 구조를 자연스럽게 필터링.

== Betti 수와 호몰로지

#eq-highlight[
  $ beta_k = "rank"(H_k (X)) $

  $H_k (X)$는 공간 $X$의 $k$-차원 호몰로지 그룹이다.

  - $beta_0$: 연결 성분 수 (점들이 몇 덩어리로 나뉘는가)
  - $beta_1$: 1차원 구멍 수 (고리/루프가 몇 개인가)
  - $beta_2$: 2차원 구멍 수 (빈 공동이 몇 개인가)
]

호몰로지는 "경계가 없는 체인 중에서 다른 것의 경계가 아닌 것"을 찾는 과정이다:

$ H_k = "Ker"(partial_k) slash "Im"(partial_(k+1)) $

=== 금융 도메인 해석

- *$H_0$ 큰 값*: 소비가 여러 분리된 클러스터 (생활비 vs 여행비 vs 교육비)
- *$H_1$ 존재*: 주기적 순환 패턴 (월초 카드값 -> 월중 식비 -> 월말 여가 -> 반복)
- *$H_2$ 존재*: 3차원 구조에서 "속이 빈" 패턴 (금액-카테고리-시간 축에서 중앙 공백)

== Vietoris-Rips 복합체와 지속 호몰로지

본 시스템은 Vietoris-Rips 복합체를 사용한다:

$ sigma = {x_0, ..., x_k} in "VR"_epsilon (X) quad "iff" quad d(x_i, x_j) <= epsilon, forall i,j $

Cech 복합체의 "2배 반지름" 근사이지만 계산이 훨씬 효율적이다 (쌍별 거리 비교만 필요).

지속 호몰로지(Persistent Homology)는 _모든 스케일을 동시에_ 관찰하여 단일 임계값 민감성 문제를 해결한다. Persistence Diagram에서 대각선과 먼 점이 강한 신호이다.

=== 안정성 정리

$ d_B ("Dgm"(f), "Dgm"(g)) <= ||f - g||_infinity $

입력 섭동이 filtration 함수의 최대 변화량으로 경계된다. 위상 피처의 노이즈 강건성을 수학적으로 보장.

== 출력 피처 (70D)

#styled-table(
  (1fr, 0.6fr, 2fr),
  table.header([*서브그룹*], [*차원*], [*설명*]),
  [`tda_short`], [24D], [90일 앱 로그 기반 단기 위상 패턴],
  [`tda_long`], [36D], [12개월 금융 거래 기반 장기 위상 패턴],
  [`phase_transition`], [10D], [시간 윈도우 간 위상 변화 감지],
)

#note[PersLay Expert와의 관계][
  본 문서의 TDA 피처(오프라인 70D)와 PersLay Expert(온라인 64D)는 모두 Persistent Homology를 활용하지만 역할이 다르다. 70D는 734D 메인 텐서의 일부이며 배치 전처리에서 사전 계산된다. PersLay Expert는 PLE 내부에서 Persistence Diagram을 end-to-end로 학습한다.
]


// =====================================================================
//  8. HMM 피처 (25D = 5D summary + 별도 48D → 시스템 총 53D)
// =====================================================================
= HMM 피처 (Hidden Markov Model, 5D summary + 48D separate)

#chip[Model-Derived + Separate Input] #chip(color: indigo)[Triple-Mode HMM] #chip(color: amber)[53D total]

== 핵심 아이디어: 숨겨진 상태 추론

카드 거래 데이터에서 직접 관측할 수 있는 것은 거래 금액, 횟수, 카테고리 다양성뿐이다. 그러나 같은 월 10만원 소비라도 "새 서비스를 탐색하는 고객"과 "이탈 직전의 마지막 소비"는 전혀 다른 상태이다. HMM은 관측 데이터로부터 관측 불가능한 잠재 상태를 *확률적으로 추론*한다.

=== 마르코프 성질

$ P(q_(t+1) | q_t, q_(t-1), ..., q_1) = P(q_(t+1) | q_t) $

다음 상태는 현재 상태에만 의존한다. 이 단순화 덕분에 파라미터 수가 $O(N^2)$로 제한되어 수십만 고객에서도 안정적 학습이 가능하다.

== Triple-Mode 구성

#styled-table(
  (1fr, 0.8fr, 0.8fr, 2fr),
  table.header([*모드*], [*상태 수*], [*출력*], [*포착하는 패턴*]),
  [Journey], [5], [16D], [일/주 단위 고객 여정 (AWARENESS -> CONSIDERATION -> PURCHASE -> ...)],
  [Lifecycle], [5], [16D], [월/년 단위 생애주기 (NEW -> GROWING -> MATURE -> AT_RISK -> ...)],
  [Behavior], [6], [16D], [월별 행동 패턴 유형 (절약형, 투자형, 소비형 등)],
)

세 모드의 48D는 별도 입력(separate input)으로 PLE의 HMM Triple-Mode Projector에 공급된다.

== 세 가지 핵심 알고리즘

#styled-table(
  (0.8fr, 1.5fr, 1fr, 1.5fr),
  table.header([*문제*], [*질문*], [*알고리즘*], [*용도*]),
  [평가], [시퀀스의 우도는?], [Forward], [모델 품질 검증, 이상치 탐지],
  [디코딩], [최적 상태 시퀀스는?], [Viterbi], [상태 시퀀스 추출 -> 메타 피처],
  [학습], [최적 파라미터는?], [Baum-Welch (EM)], [전이/방출 분포 최적화],
)

=== Forward-Backward

$ alpha_t (i) = P(o_1, ..., o_t, q_t = S_i | lambda), quad gamma_t (i) = P(q_t = S_i | bold(O), lambda) $

$gamma_t(i)$가 소프트 상태 할당 확률이며, 이것이 피처의 핵심이다. "80% 확률로 활성, 15% 확률로 성장, 5% 확률로 위험"이라는 확률 벡터 자체가 풍부한 피처가 된다.

== HMM Summary (메인 텐서 5D)

#styled-table(
  (1.5fr, 2.5fr),
  table.header([*피처*], [*설명*]),
  [`hmm_dominant_state`], [3개 모드 통합 시 최고 $gamma_t(i)$ 상태 ID],
  [`hmm_state_duration`], [현재 지배 상태에 머문 연속 기간 (월 수)],
  [`hmm_transition_stability`], [최근 N개월 상태 전이 빈도의 역수],
  [`hmm_transition_entropy`], [$H = -sum_j a_(i j) log a_(i j)$. 높으면 다음 상태 예측 어려움],
  [`hmm_state_change_rate`], [상태 변화 횟수 / 전체 기간],
)

#note[HMM vs GMM][
  HMM은 _시계열_ 상태 추론이다. "이 고객이 현재 어떤 _단계_에 있고, 다음에 어디로 이동하는가?" GMM은 _횡단면_ 클러스터링이다. "이 고객이 어떤 _유형_인가?" 전이 행렬 $bold(A)$가 시간 역학을 포착하므로, 같은 소비 패턴이라도 "성장 중"인지 "이탈 직전"인지를 구분한다. 두 모듈은 서로 다른 입력 경로로 PLE에 공급되어 상호 보완한다.
]


// =====================================================================
//  9. GMM 피처 (22D)
// =====================================================================
= GMM 피처 (Gaussian Mixture Model, 22D)

#chip[Domain Features] #chip(color: indigo)[EM Algorithm + BIC] #chip(color: amber)[22D / 159D]

== 이론적 근거

#eq-highlight[
  $ p(bold(x)) = sum_(k=1)^K pi_k cal(N)(bold(x) | bold(mu)_k, bold(Sigma)_k) $

  여기서 $pi_k >= 0$, $sum_k pi_k = 1$이고, 각 성분은 다변량 가우시안이다.
  구성: $K = 20$ 클러스터, $D = 40$ 입력 차원, `covariance_type = "full"`.
]

핵심 가정: "관측 데이터는 $K$개 하위 집단에서 생성되었으며, 각 데이터 포인트의 소속은 관측되지 않는다(잠재 변수)."

== EM 알고리즘

=== E-Step (사후 책임도)

$ gamma_(n k) = frac(pi_k cal(N)(bold(x)_n | bold(mu)_k, bold(Sigma)_k), sum_j pi_j cal(N)(bold(x)_n | bold(mu)_j, bold(Sigma)_j)) $

직접적인 베이즈 정리 적용: 사전 확률 $pi_k$와 우도 $cal(N)_k$를 결합하여 사후 확률 $gamma_(n k)$를 얻는다.

=== M-Step

$gamma_(n k)$를 가중치로 사용하여 $bold(mu)_k$, $bold(Sigma)_k$, $pi_k$를 갱신한다.

=== 수렴 보장

Jensen 부등식에 의해 EM은 ELBO를 단조 비감소로 구성한다. 전역 최적 보장은 없으므로 `n_init=10`으로 완화.

== 모델 선택: BIC

$ "BIC" = -2 ln hat(L) + k ln(n) $

수십만 데이터에서 AIC($-2 ln hat(L) + 2k$)보다 BIC의 $k ln(n)$ 패널티가 과적합을 더 효과적으로 방지한다.

== 출력 피처 (22D)

#styled-table(
  (1.5fr, 0.6fr, 2.5fr),
  table.header([*피처*], [*차원*], [*설명*]),
  [`cluster_prob_00` -- `19`], [20D], [소프트 할당 확률 $gamma_(n k)$ (합 = 1.0)],
  [`cluster_id`], [1D], [하드 할당 $arg max_k gamma_(n k)$],
  [`cluster_entropy`], [1D], [$H_n = -sum_k gamma_(n k) ln(gamma_(n k) + epsilon)$],
)

엔트로피 해석:
- $H approx 0$: 명확한 행동 원형 (확신 있는 분류)
- $H = ln(20) approx 2.996$: 균등 분포 (콜드스타트/분류 불능)

== GMM vs K-Means: 왜 소프트 할당인가

#styled-table(
  (1fr, 1.5fr, 1.5fr),
  table.header([*관점*], [*K-Means*], [*GMM*]),
  [할당], [하드: one-hot (1 bit)], [소프트: 확률 벡터 (~4.32 bits max)],
  [경계 고객], [자의적 할당, 불안정], [인접 클러스터에 확률 분산],
  [클러스터 형태], [구형 (유클리드)], [타원형 (Mahalanobis, full covariance)],
  [불확실성], [없음], [엔트로피 기반 신뢰도],
  [PLE 역할], [단일 서브헤드 활성], [$gamma_(n k)$ 가중 앙상블로 20개 서브헤드 결합],
)

Mahalanobis 거리 $d_M = sqrt((bold(x)-bold(mu))^top bold(Sigma)^(-1)(bold(x)-bold(mu)))$는 피처 상관과 스케일 차이를 반영하여 유클리드 거리($bold(Sigma) = bold(I)$)를 일반화한다.


// =====================================================================
//  10. Mamba 시계열 피처 (50D)
// =====================================================================
= Mamba 시계열 피처 (Selective State Space, 50D)

#chip[Domain Features] #chip(color: indigo)[Mamba SSM + PCA] #chip(color: amber)[50D / 159D]

== 핵심 아이디어

단순 집계 통계(합계, 평균, 표준편차)는 시계열의 *순서 정보(temporal ordering)*를 완전히 파괴한다. 값들을 무작위로 섞어도 합계와 평균은 변하지 않는다. Mamba SSM은 선택적 상태 공간 메커니즘으로 시계열의 비선형 장기 의존성을 포착한다.

== Selective State Space Model

Mamba는 연속 상태 공간 모델(SSM)의 이산화에 기반한다:

#eq-highlight[
  연속: $h'(t) = bold(A) h(t) + bold(B) x(t), quad y(t) = bold(C) h(t)$

  이산화: $overline(bold(A)) = exp(Delta bold(A)), quad overline(bold(B)) = (Delta bold(A))^(-1)(exp(Delta bold(A)) - bold(I)) dot Delta bold(B)$

  핵심: $Delta_k$가 입력에 따라 *동적으로* 변하므로, 비정상 시계열에서도 적응적으로 상태를 업데이트한다.
]

=== 기존 방법 대비 강점

#styled-table(
  (1fr, 1fr, 1fr, 1fr),
  table.header([*관점*], [*RNN*], [*Transformer*], [*Mamba*]),
  [장기 의존성], [소실 문제], [Self-attention으로 해결], [선택적 SSM으로 해결],
  [시퀀스 길이], [$O(T)$ 순차], [$O(T^2)$ 메모리], [$O(T)$ 선형],
  [비정상성], [적응 어려움], [위치 인코딩 의존], [$Delta_k$ 동적 적응],
)

== 파이프라인

+ MCC 15D $times$ 180일 입력 시퀀스
+ Mamba 인코더 (`d_model=256`)로 숨겨진 표현 추출
+ PCA $arrow.r$ 50D 압축
+ 734D 메인 텐서의 Domain 블록에 통합

#warn[온라인 Mamba Expert와의 구분][
  본 피처(오프라인 50D)는 `d_model=256`, 입력 MCC 15D $times$ 180일이며 PLE 학습 _이전_에 사전 계산된다. Temporal Expert의 Mamba는 PLE 내부 `d_model=128`, 입력 card 16D $times$ 180 steps로 end-to-end 학습된다. 가중치 공유 없음.
]

== 시계열 분석의 4가지 관점

Mamba 50D는 아래 네 관점의 정보를 비선형으로 통합한 _학습된 표현_이다:

#styled-table(
  (1fr, 1.5fr, 1.5fr),
  table.header([*관점*], [*핵심 질문*], [*기법*]),
  [시간 도메인], [값이 시간에 따라 어떻게 변하는가?], [자기상관, 변환점, 이동평균],
  [주파수 도메인], [어떤 주기로 반복되는가?], [FFT, 스펙트럼 분석],
  [분포/형상], [값의 분포는 어떤 모양인가?], [왜도, 첨도, 꼬리 확률],
  [정보이론], [얼마나 복잡하고 예측 가능한가?], [엔트로피, 순열 엔트로피],
)


// =====================================================================
//  11. Graph 피처 (LightGCN + Hyperbolic, 66D 임베딩)
// =====================================================================
= Graph 피처 (LightGCN + Hyperbolic GCN)

#chip[Offline Precomputed] #chip(color: indigo)[Graph Neural Network] #chip(color: amber)[64D + H-GCN aggregation]

== LightGCN: 협업 필터링

=== Message Passing

#eq-highlight[
  $ bold(e)_u^((k+1)) = sum_(i in cal(N)_u) frac(1, sqrt(|cal(N)_u|) dot sqrt(|cal(N)_i|)) dot bold(e)_i^((k)) $

  대칭 정규화 $tilde(A) = D^(-1\/2) A D^(-1\/2)$는 인기 아이템(sender)과 수신자 영향을 동시에 억제한다.
]

=== Layer Combination

$ bold(e)_u^"final" = frac(1, L+1) sum_(k=0)^L bold(e)_u^((k)) $

0-hop 자기 자신 + 1,2,3-hop 이웃의 균등 평균. 학습 가능한 attention 가중치 없이도 실증적으로 우수하며 과적합을 방지한다.

=== BPR Loss

$ cal(L)_"BPR" = -sum_((u, i^+, i^-)) log sigma(hat(y)_(u i^+) - hat(y)_(u i^-)) + lambda ||Theta||^2 $

쌍별 랭킹 손실로 절대 점수가 아닌 상대 순위를 최적화한다.

== Hyperbolic GCN: 계층 구조

=== Poincare Ball 모델

#eq-highlight[
  $ BB_c^d = { bold(x) in RR^d : c||bold(x)||^2 < 1 } $

  *지수 맵*: $exp_bold(0)(bold(v)) = tanh(sqrt(c)||bold(v)||) dot bold(v) / (sqrt(c)||bold(v)||)$

  *Poincare 거리*: $d_(BB)(bold(x), bold(y)) = 1/sqrt(c) "arccosh"(1 + frac(2c||bold(x)-bold(y)||^2, (1-c||bold(x)||^2)(1-c||bold(y)||^2)))$
]

경계 근처에서 분모 -> 0이므로 유클리드 거리가 작아도 쌍곡 거리가 폭발한다. 이것이 계층적 깊이를 자연스럽게 인코딩하는 메커니즘이다.

=== 금융 도메인 정당성

MCC 분류 체계(Root -> L1(8) -> L2(~100) -> Brand(~50K) -> Branch(~500K))는 본질적으로 트리 구조이다. 쌍곡 공간은 지수적 체적 성장으로 트리 분기를 매칭하여 8D Poincare Ball로 ~550K 노드를 임베딩한다. Nickel & Kiela(2017) 결과: 5D 쌍곡 > 200D 유클리드 (WordNet 계층).

== Dual GCN 구성

#styled-table(
  (1fr, 1.5fr, 1.5fr),
  table.header([*속성*], [*LightGCN*], [*H-GCN*]),
  [노드], [고객 + 가맹점 (이분 그래프)], [가맹점만 (MCC 트리)],
  [엣지], [고객-가맹점 거래], [부모-자식 계층 + 브랜드 co-visitation],
  [공간], [$RR^(64)$ 유클리드], [$BB^8$ Poincare Ball],
  [학습 목적], ["누가 무엇을 좋아하는가"], ["가맹점이 구조적으로 어떻게 관련되는가"],
  [출력], [고객 임베딩 64D (직접)], [가맹점 임베딩 -> 고객별 집계 (간접)],
)

*2-Stage 파이프라인*: Stage 1(오프라인) 그래프 학습 -> 임베딩 Parquet 저장. Stage 2(온라인) lookup + 경량 MLP 적응. 추론 시 그래프 전파 없음 -- 단일 GPU에서 VRAM 친화적.


// =====================================================================
//  12. 기타 (Base + Product + Transaction Behavior)
// =====================================================================
= 기타 피처 (Base Demographics, Product Holdings, Transaction Behavior)

#chip[Base Features] #chip(color: indigo)[238D + 91D + 84D] #chip(color: amber)[413D / 644D]

== Base 피처 (238D)

전통적 금융 ML의 기반이 되는 피처 블록이다.

#styled-table(
  (1.2fr, 0.6fr, 2fr),
  table.header([*서브그룹*], [*차원*], [*설명*]),
  [RFM (Recency, Frequency, Monetary)], [34D], [거래 기간, 빈도, 금액의 다차원 분해],
  [Category], [64D], [MCC 카테고리별 소비 비율/금액/빈도],
  [Transaction Stats], [76D], [거래 통계: 평균/중위/표준편차/왜도/첨도 등],
  [Product Diversity], [12D], [보유 상품 다양성, 교차 보유 패턴],
  [Channel Behavior], [18D], [온라인/오프라인/모바일 채널 비율],
  [Temporal Pattern], [22D], [요일별/시간대별 거래 분포],
  [Demographics], [12D], [연령, 성별, 지역 등 인구통계],
)

== Multi-Source 피처 (91D)

#styled-table(
  (1.2fr, 0.6fr, 2fr),
  table.header([*서브그룹*], [*차원*], [*설명*]),
  [Deposit], [~25D], [예금 잔액, 입출금 패턴, 금리 민감도],
  [Credit], [~20D], [신용 등급, 한도 사용률, 연체 이력],
  [Investment], [~15D], [투자 상품 보유, 위험 선호도],
  [Digital Engagement], [~31D], [앱 로그, 로그인 빈도, 기능 사용],
)

== Extended-Source 피처 (84D)

#styled-table(
  (1.2fr, 0.6fr, 2fr),
  table.header([*서브그룹*], [*차원*], [*설명*]),
  [Insurance], [~20D], [보험 상품 보유, 가입 이력],
  [Refund/Cancellation], [~15D], [환불 빈도, 취소 패턴],
  [Consultation/STT], [~25D], [상담 이력, 음성 인식 분석],
  [External Signals], [~24D], [외부 데이터 연동 신호],
)

== Model-Derived 피처 (27D)

#styled-table(
  (1.2fr, 0.6fr, 2fr),
  table.header([*서브그룹*], [*차원*], [*설명*]),
  [HMM Summary], [5D], [Triple-Mode 48D의 압축 표현 (dominant state, duration, stability, entropy, change rate)],
  [Bandit/MAB], [4D], [Multi-Armed Bandit 기반 탐색-활용 균형 지표],
  [LNN Statistics], [18D], [이동평균, 변동성, 자기상관 등 수작업 통계 피처],
)

== Merchant Hierarchy 피처 (21D)

MCC 계층 구조(Root -> L1 -> L2 -> Brand)를 반영한 좌표 및 임베딩. H-GCN에서 추출한 쌍곡 좌표와 co-visitation 기반 브랜드 유사도를 포함한다.


// =====================================================================
//  13. 3-Stage 정규화
// =====================================================================
= 3-Stage 정규화 파이프라인

#chip[Preprocessing] #chip(color: indigo)[Power-law -> Scaler -> Raw Copy] #chip(color: amber)[644D + 90D]

== 설계 원칙

정규화 파이프라인은 세 단계로 구성되며, *멱법칙(power-law) 분포를 가진 피처의 원본 크기 정보를 보존*하면서도 모델 학습에 적합한 스케일로 변환하는 것이 핵심 목표이다.

== Stage 1: 멱법칙 감지 + log1p 복사본 생성

#eq-highlight[
  *멱법칙 감지 기준*:
  + 왜도(skewness) + 첨도(kurtosis) 기반 후보 선별
  + log-log 회귀의 $R^2$로 멱법칙 여부 판정

  *멱법칙 확인 시*: 해당 컬럼의 `log1p` 복사본을 생성하여 별도 보존
  $ x_"raw" = log(1 + x) $
]

멱법칙 분포(거래 금액, 잔액 등)는 StandardScaler 적용 시 이상치에 의해 대부분의 값이 0 근처에 압축되어 정보가 손실된다. log1p 변환은 이 문제를 완화한다.

== Stage 2: StandardScaler (TRAIN fit only)

#eq-highlight[
  $ z = frac(x - mu_"train", sigma_"train") $

  *적용 대상*: continuous 컬럼만 (binary 컬럼 제외)
  *핵심 규칙*: Scaler는 반드시 TRAIN split에서만 fit. val/test는 train에서 fit된 scaler로 transform만 수행.
]

#warn[데이터 리키지 방지][
  val/test 데이터로 scaler를 fit하면 미래 정보가 학습에 유입되어 실제 성능보다 과대 추정된다. 이것은 가장 흔한 리키지 패턴이므로 LeakageValidator가 학습 전에 반드시 검증한다.
]

== Stage 3: Raw power-law 보존

멱법칙 컬럼의 `_log` 복사본(Stage 1에서 생성)은 *스케일링하지 않고* 원본 크기(raw magnitude)를 그대로 보존한다.

#note[왜 raw copy를 보존하는가][
  StandardScaler는 정보를 재배치하지만, *절대적 크기 정보*를 파괴한다. 예를 들어 "월 소비 100만원"과 "월 소비 1000만원"의 차이는 z-score로 변환 후 상대적 위치만 남는다. 그러나 일부 태스크(예: LTV 예측, spending_bucket 분류)에서는 절대 금액 정보가 핵심이다. Raw power-law 90D 복사본은 이 정보를 모델에 직접 제공한다.
]

== 최종 텐서 구성

#styled-table(
  (1fr, 0.8fr, 2fr),
  table.header([*구간*], [*차원*], [*처리*]),
  [Normalized continuous], [~554D], [Stage 2 StandardScaler 적용],
  [Binary (pass-through)], [~90D], [스케일링 미적용 (0/1 보존)],
  [Raw power-law copy], [90D], [Stage 1 log1p만 적용, Stage 2 미적용],
  [*합계*], [*734D*], [644D normalized + 90D raw],
)

== 정규화 파이프라인 흐름 요약

```
원본 피처 → [Stage 1] 멱법칙 감지 (skew+kurt → log-log R²)
                       ↓ 멱법칙 → log1p 복사본 생성 (90D raw)
         → [Stage 2] StandardScaler fit on TRAIN only
                       ↓ continuous만 적용, binary 제외
         → [Stage 3] raw power-law 복사본 그대로 보존
                       ↓
         → [최종] 644D normalized ⊕ 90D raw = 734D main tensor
```


// =====================================================================
//  부록: 설계 vs 구현 차원 매핑
// =====================================================================
= 부록: 설계 vs 구현 차원 매핑

#warn[참고][이 부록은 풀뱅크 설계(734D)와 현재 Santander 벤치마크 구현(316D) 간의 차원 차이를 정리한 것입니다. 구현 차원은 `outputs/phase0/feature_schema.json`에서 확인할 수 있습니다.]

#styled-table(
  (1.2fr, 1fr, 1fr, 2fr),
  table.header([*피처 그룹*], [*설계 (734D)*], [*구현 (316D)*], [*비고*]),
  [TDA], [70D], [32D], [tda\_global 16D + tda\_local 16D],
  [HMM], [48D + 5D (별도)], [25D], [main tensor only],
  [Base (Profile 등)], [238D], [47D], [Demographics, RFM, Financial Summary 축소],
  [Graph], [미명시], [66D], [구현에서 독립 그룹으로 추가],
  [Merchant / Hierarchy], [21D], [34D], [MCC levels, brand embeddings 확장],
  [GMM], [22D], [53D], [클러스터 수 및 파생 피처 확장],
  [기타 (Economics, SIR 등)], [335D], [59D], [Mamba, Wave, Crime 등 축소],
  [*합계*], [*734D*], [*316D*], [12 feature groups],
)


// =====================================================================
//  부록: 참고 문헌
// =====================================================================
= 참고 문헌

== 경제학
- Friedman, M. (1957). _A Theory of the Consumption Function_. Princeton UP.
- Hall, R. (1978). Stochastic Implications of the Life Cycle-PIH. _JPE_.
- Hodrick, R. & Prescott, E. (1997). Postwar U.S. Business Cycles. _JMCB_.
- Kalman, R. (1960). A New Approach to Linear Filtering. _J. Basic Engineering_.
- Kahneman, D. & Tversky, A. (1979). Prospect Theory. _Econometrica_.

== 다학제
- Arrhenius, S. (1889). Reaction rates of sucrose inversion.
- Kermack, W. & McKendrick, A. (1927). Mathematical Theory of Epidemics. _Proc. Royal Society_.
- Cohen, L. & Felson, M. (1979). Routine Activity Approach. _ASR_.
- Barabasi, A.-L. (2005). The origin of bursts and heavy tails. _Nature_.
- Shannon, C. (1948). A Mathematical Theory of Communication. _Bell System Technical Journal_.
- Kullback, S. & Leibler, R. (1951). On Information and Sufficiency.

== TDA
- Carlsson, G. (2009). Topology and Data. _Bulletin of the AMS_.
- Edelsbrunner, H., Letscher, D. & Zomorodian, A. (2002). Topological Persistence and Simplification.
- Cohen-Steiner, D., Edelsbrunner, H. & Harer, J. (2007). Stability of Persistence Diagrams. _DCG_.
- Carriere, M. et al. (2020). PersLay. _AISTATS_.

== HMM
- Baum, L. & Petrie, T. (1966). Statistical Inference for Probabilistic Functions of Finite State Markov Chains. _AMS_.
- Rabiner, L. (1989). A Tutorial on HMM. _Proc. IEEE_.
- Dempster, A., Laird, N. & Rubin, D. (1977). Maximum Likelihood from Incomplete Data via EM. _JRSS-B_.

== GMM
- Pearson, K. (1894). Contributions to the Mathematical Theory of Evolution. _Phil. Trans. Royal Society A_.
- Schwarz, G. (1978). Estimating the Dimension of a Model. _Annals of Statistics_.
- Bishop, C. (2006). _Pattern Recognition and Machine Learning_, Ch. 9.

== 시계열 / Mamba
- Gu, A. & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces.
- Cooley, J. & Tukey, J. (1965). An Algorithm for the Machine Calculation of Complex Fourier Series. _Math. Comp._

== Graph
- He, X. et al. (2020). LightGCN: Simplifying and Powering GCN for Recommendation. _SIGIR_.
- Chami, I. et al. (2019). Hyperbolic Graph Convolutional Neural Networks. _NeurIPS_.
- Nickel, M. & Kiela, D. (2017). Poincare Embeddings for Learning Hierarchical Representations. _NeurIPS_.
- Rendle, S. et al. (2009). BPR: Bayesian Personalized Ranking from Implicit Feedback. _UAI_.
