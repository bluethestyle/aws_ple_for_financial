// ============================================================
// Expert Details: 7 Heterogeneous Experts + adaTT + Feature Engineering
// AIOps PLE for Financial Recommendation
// Anthropic Design System
// ============================================================

#let anthropic-bg = rgb("#F0EFEA")
#let anthropic-text = rgb("#141413")
#let anthropic-accent = rgb("#CC785C")
#let anthropic-muted = rgb("#6B7280")
#let anthropic-rule = rgb("#D1D5DB")

#set document(
  title: "Heterogeneous Expert Architecture: Selection Rationale, Mathematical Formulation, and Financial Application",
  author: ("Author 1", "Author 2"),
)

#set page(
  paper: "a4",
  margin: (top: 2.5cm, bottom: 2.5cm, left: 2.5cm, right: 2.5cm),
  fill: anthropic-bg,
  header: context {
    if counter(page).get().first() > 1 [
      #set text(size: 7.5pt, font: ("Pretendard", "New Computer Modern"), fill: anthropic-muted, tracking: 0.12em)
      #smallcaps[Expert Details]
      #h(1fr)
      #smallcaps[Heterogeneous Expert Architecture]
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
  )[#upper[Expert Details]]
  #v(0.5cm)

  #text(size: 26pt, fill: anthropic-text, weight: "bold")[
    Heterogeneous Expert Architecture
  ]

  #v(0.3em)

  #text(size: 14pt, fill: anthropic-muted)[
    Selection Rationale, Mathematical Formulation,\
    and Financial Application
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
      #smallcaps[Expert Details]
      #h(1fr)
      #smallcaps[Heterogeneous Expert Architecture]
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
  #text(weight: "bold")[Abstract.]
  본 문서는 금융 상품 추천을 위한 Progressive Layered Extraction (PLE) 아키텍처에서 사용되는
  7개 이질적 전문가(Heterogeneous Expert)의 선정 근거, 수학적 배경, 금융 도메인 적용 방안을
  상세히 기술한다. 각 전문가는 피처 상호작용(DeepFM), 시계열 분해(Temporal Ensemble),
  계층 구조(Hyperbolic GCN), 위상 구조(PersLay/TDA), 협업 필터링(LightGCN),
  인과 추론(Causal/NOTEARS), 분포 매칭(Optimal Transport)이라는 고유한 수학적 관점을 제공하며,
  어떠한 단일 전문가도 다른 전문가의 역할을 대체할 수 없다.
  추가로 adaTT(Adaptive Task-aware Transfer)를 통한 18개 태스크 간 동적 지식 전이 메커니즘과
  11개 학문 분야에서 도출된 316차원 피처 엔지니어링 체계를 기술한다.
  FeatureRouter 활성화로 각 전문가는 전체 316D 중 지정된 서브셋만 입력으로 받으며
  (deepfm=162D, temporal=127D, hgcn=34D, perslay=32D, causal=158D, lightgcn=66D, ot=124D),
  출력은 64D로 균일하게 유지된다.

  #v(0.3em)
  #text(weight: "bold")[Keywords:]
  Heterogeneous Experts, PLE, DeepFM, Mamba, Hyperbolic GCN,
  Persistent Homology, NOTEARS, Optimal Transport, adaTT, Financial AI
]

#v(1em)

#block(
  width: 100%,
  inset: (left: 14pt, right: 14pt, top: 10pt, bottom: 10pt),
  stroke: (left: 2pt + anthropic-accent),
)[
  #text(weight: "bold", fill: anthropic-accent)[설계 vs 구현 참고.]
  본 문서는 풀뱅크 설계(734D)를 기준으로 작성되었습니다.
  현재 Santander 벤치마크 구현은 316D (12 feature groups)입니다.
  *FeatureRouter 활성화*: 각 전문가는 전체 316D를 모두 받는 것이 아니라
  `feature_groups.yaml`의 `target_experts` 선언에 따라 서브셋만 입력받습니다.
  전문가별 실제 입력 차원: deepfm=162D, temporal=127D, hgcn=34D, perslay=32D,
  causal=158D, lightgcn=66D, ot=124D. 모델 파라미터: 4.77M → 3.16M (34% 감소).
]

#v(1em)

// ============================================================
= DeepFM Expert --- Feature Interaction
// ============================================================

== 선정 근거 (Why DeepFM?)

금융 추천은 개별 피처가 아닌 피처 _상호작용_에 의존한다. 예를 들어 "30대 + 디지털 활용도 높음 + 높은 RFM"이
결합될 때 온라인 투자 전환율이 급등하는 패턴은 단일 피처만으로는 포착할 수 없다.
644개 피처에 대해 모든 pairwise interaction을 명시적으로 모델링하면 $O(n^2) = 207,046$개의 파라미터가
필요하며, 대부분은 sparse 데이터에서 안정적으로 추정할 수 없다.

FM의 low-rank factorization은 교차 파라미터를 $O(n k) = 10,304$개로 감소시키면서도
공유 잠재 벡터를 통해 관측되지 않은 피처 쌍에 대한 일반화를 가능하게 한다.
Deep 컴포넌트는 FM의 2차 한계를 넘어 3차 이상의 암시적 고차 상호작용을 추가한다.

== 대안 비교 (Why Not Alternatives?)

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, left, left, left),
    stroke: 0.5pt,
    [*Aspect*], [*DeepFM*], [*DCNv2*], [*AutoInt (Transformer)*],
    [Interaction order], [FM: 2nd + Deep: arbitrary], [Cross: $(l+1)$-th + Deep], [Self-attention: arbitrary],
    [Parameters], [$tilde 169$K], [$tilde 2.5$M (cross alone)], [$O(n^2)$ attention],
    [Inference speed], [Fastest (2--5x vs Attention)], [Medium], [Slowest],
    [Field awareness], [Yes (28 fields)], [No (raw features)], [Yes (per-feature)],
  ),
  caption: [DeepFM vs. 대안 아키텍처 비교. BARS 벤치마크(2022--2024)에서 DeepFM은 중소규모 데이터셋에서 최저 추론 지연 시간과 함께 일관된 상위 성능을 보인다.],
)

Wide&Deep (Google, 2016)과 달리 DeepFM은 FM과 Deep 간 임베딩을 공유하여
수동 cross-feature engineering을 제거한다. "구조적 효율성(FM) + 보편적 표현력(Deep)"의
조합을 제공한다.

== 수학적 배경 (Key Formulations)

*FM 2nd-order interaction:*
$ hat(y) = w_0 + sum_(i=1)^n w_i x_i + sum_(i=1)^n sum_(j=i+1)^n chevron.l bold(v)_i, bold(v)_j chevron.r x_i x_j $

*Low-rank factorization:*
$ W approx V V^top, quad V in bb(R)^(n times k) $

이로써 파라미터가 $O(n^2)$에서 $O(n k)$로 감소한다. 프로젝트에서는 $n=644$, $k=16$을 사용한다.

*FM Trick ($O(n k)$ computation):*
$ sum_(i<j) chevron.l bold(v)_i, bold(v)_j chevron.r x_i x_j = 1/2 sum_(f=1)^k [ (sum_(i=1)^n v_(i,f) x_i)^2 - sum_(i=1)^n (v_(i,f) x_i)^2 ] $

*Cross Network (DCNv2) layer:*
$ bold(x)_(l+1) = bold(x)_0 circle.small (bold(x)_l W_l + bold(b)_l) + bold(x)_l $

각 레이어가 1차수의 interaction을 추가하며, $l$개 레이어는 최대 $(l+1)$차 interaction을 포착한다.
레이어당 $2d$ 파라미터만 필요하다 (MLP의 $O(d^2)$ 대비).

=== 28-Field Architecture

644D 정규화 피처를 28개 시맨틱 필드로 분할한다 (base 238D, multi-source 91D, extended 84D,
domain 159D, multidisciplinary 24D, model-derived 27D, merchant 21D).
각 필드는 `nn.Linear(d_i, 16)`을 통해 균일한 16D 잠재 공간으로 투영된다.
FM은 필드 레벨 임베딩 $[B, 28, 16]$에서 동작하고,
Deep MLP는 flatten된 $[B, 448]$에서 동작한다.

*Output:* FM 16D + Deep 64D = 80D $arrow$ output layer $arrow$ 64D (PLE gate 입력).

== 금융 적용 (Financial Interpretation)

- 교차 필드 상호작용은 은행업에서 핵심이다: "30대 + 서울 + 높은 디지털 활용" $arrow$ 온라인 투자 전환율 급등;
  "높은 RFM + 낮은 예금" $arrow$ 신용 상품 추천.
- 28-field 설계는 inter-category FM interaction을 가능하게 한다: 64D 카테고리를 4$times$16D로
  분할하면 추가 파라미터 비용 없이 27개의 새로운 FM interaction 쌍이 생성된다.

== 입출력 사양 (I/O Specification)

#figure(
  table(
    columns: (auto, auto),
    stroke: 0.5pt,
    [*Input (설계)*], [644D normalized feature tensor $[B, 644]$],
    [*Input (현재 구현)*], [162D — FeatureRouter가 316D에서 DeepFM 지정 서브셋 슬라이싱],
    [*Internal*], [필드 임베딩 $arrow$ FM + Deep MLP (입력 차원에 따라 자동 조정)],
    [*Output*], [64D expert representation for PLE CGC Gate],
    [*Parameters*], [$tilde 169$K (설계 기준; 162D 입력 시 비례 감소)],
  ),
  caption: [DeepFM Expert 입출력 사양. FeatureRouter 활성화 시 입력 차원이 162D로 축소되며, 출력 64D는 동일하게 유지된다.],
)

== 구현 참고사항 (Implementation Notes)

- PLE의 기본 경량 Shared Expert로 사용. DCNv2Expert는 고차 상호작용이 필요한 경우의 대안이다.
- FM과 Deep이 임베딩을 공유하므로 gradient가 양쪽 경로를 통해 동시에 흐른다.
- FocalLoss 적용 시 pre-activation logits를 전달해야 한다 (double-sigmoid 방지).

*주요 참고문헌:*
Guo et al. (IJCAI 2017), Rendle (ICDM 2010), Wang et al. (KDD 2017, WWW 2021).

#pagebreak()

// ============================================================
= Temporal Ensemble Expert --- Mamba + LNN + PatchTST
// ============================================================

== 선정 근거 (Why Temporal Ensemble?)

정적 피처(나이, 평균 지출)는 거래 시퀀스의 주기성, 추세, 불규칙 이벤트 패턴이라는
시간적 차원을 버린다. 180일 지출 시퀀스를 월 평균으로 압축하면 주간 주기, 추세 방향,
이상 폭발 패턴이 소실된다.

모든 시계열은 $y(t) = T(t) + S(t) + R(t)$ (추세 + 계절성 + 잔차)로 분해되며,
단일 아키텍처로는 세 가지를 모두 최적으로 포착할 수 없다:

#figure(
  table(
    columns: (auto, auto, auto, auto),
    stroke: 0.5pt,
    [*Model*], [*Temporal Pattern*], [*Mechanism*], [*Complexity*],
    [Mamba (SSM)], [Long-range trend], [Selective State Space (S6)], [$O(L)$ linear],
    [LNN (ODE)], [Irregular residuals], [Adaptive time-constant ODE], [$O(1)$ single step],
    [PatchTST], [Global periodicity], [Patch-level self-attention], [$O((L slash P)^2)$],
  ),
  caption: [Temporal Ensemble의 세 모델 -- 시계열 분해 $T + S + R$에 대응.],
)

== 대안 비교 (Why Not Single Model?)

#figure(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt,
    [*Generation*], [*Approach*], [*Limitation*],
    [1st], [ARIMA, Exponential Smoothing], [선형 가정, 수동 differencing],
    [2nd], [LSTM, GRU], [$O(L)$ 순차 병목, vanishing gradient],
    [3rd], [Transformer], [$O(L^2)$ complexity, 약한 순서 인코딩],
    [4th (ours)], [SSM + ODE + Patch Transformer], [모델 복잡성 (entropy monitoring으로 완화)],
  ),
  caption: [시계열 모델링 세대별 비교.],
)

== 수학적 배경 (Key Formulations)

=== Mamba (Selective State Space Model)

*Continuous SSM:*
$ (d bold(x))/(d t) = bold(A) bold(x) + bold(B) u, quad y = bold(C) bold(x) + bold(D) u $

*ZOH Discretization:*
$ macron(bold(A)) = exp(Delta dot bold(A)), quad macron(bold(B)) approx Delta dot bold(B) $

*Discrete recurrence:*
$ bold(h)_t = macron(bold(A)) dot bold(h)_(t-1) + macron(bold(B)) dot bold(x)_t, quad bold(y)_t = bold(C)_t dot bold(h)_t $

*S6 Selective Mechanism (핵심 혁신):*
$ Delta = "softplus"(bold(W)_Delta dot bold(x) + bold(b)_Delta) $
$ bold(B) = bold(W)_B dot bold(x), quad bold(C) = bold(W)_C dot bold(x) $

LTI 시스템과 달리 S6는 $Delta$, $bold(B)$, $bold(C)$를 _입력 의존적_으로 만들어 content-aware processing을
가능하게 한다. $Delta$가 크면 현재 입력을 상태에 강하게 인코딩하고, 작으면 이전 상태를 보존한다.

*금융 해석:* 대규모 거래 $arrow$ 큰 $Delta$ (강하게 기억), 소규모 일상 구매 $arrow$ 작은 $Delta$ (배경 소음으로 빠르게 망각).

*사양:* Transaction Mamba: $d_"model" = 128$, $d_"input" = 16$, $d_"state" = 16$, $"seq_len" = 180$.
Session Mamba: $d_"model" = 64$. HiPPO-style 대각선 초기화 $[-1, -2, ..., -N]$.

=== Liquid Neural Network (LNN)

*Core ODE:*
$ (d bold(h))/(d t) = (-bold(h) + f(bold(x), bold(h)))/(tau(bold(x), bold(h))) $

여기서:
- $-bold(h)$: leak/decay term (입력 없이 0을 향해 망각)
- $f(bold(x), bold(h)) = tanh(bold(W)_f [bold(x); bold(h)] + bold(b)_f)$: target state
- $tau(bold(x), bold(h)) = "Softplus"("MLP"([bold(x); bold(h)])) + 0.1$: adaptive time constant

*Euler discretization:*
$ bold(h)_(t+1) = bold(h)_t + Delta t dot (-bold(h)_t + f(bold(x)_t, bold(h)_t))/(tau(bold(x)_t, bold(h)_t)) $

프로젝트에서는 SingleStep 모드를 사용한다 --- Mamba의 최종 hidden state에 대해 1회 ODE step만 적용.
Mamba가 이미 $O(L)$에서 전체 시퀀스 패턴을 포착하므로 LNN은 _time-scale correction_만 추가한다.

*금융 해석:* 금융 거래 간격은 극도로 불규칙하다 (일내 다수 거래: $Delta t tilde 0.01$일,
주말 공백: $Delta t = 2$일, 장기 휴면: $Delta t > 30$일). Adaptive $tau$가 자동으로 조정한다.

=== PatchTST (Patch Time Series Transformer)

*Patch embedding:*
$ bold(p)_i = bold(W)_"proj" dot "flatten"(bold(x)_[[(i-1)P+1 : i P]]) + bold(b)_"proj" $

$P = 16$으로 180-step 시퀀스가 12개 패치(토큰)가 되어 attention 비용이
$O(180^2) arrow O(12^2) = 144$로 감소한다.

*금융 해석:* 패치 크기 16은 $tilde$2주에 해당하며 급여 주기(격주/월)와 자연스럽게 정렬된다.
각 패치 내에서 지역 패턴(2주 내 일일 지출)을, 패치 간 attention으로 전역 주기성(월급 급등, 분기 보너스)을 포착한다.

=== Ensemble Gating

$ bold(g) = "Softmax"(bold(W)_2 dot "ReLU"(bold(W)_1 dot bold(z)_"cat" + bold(b)_1) + bold(b)_2) in bb(R)^3 $
$ bold(y) = sum_(i=1)^3 g_i dot "Proj"_i (bold(z)_i) in bb(R)^(64) $

$bold(z)_"cat"$는 세 모델 출력의 연결 (192+96+96 = 384D).
Gate entropy $H(bold(g)) = -sum_(i=1)^3 g_i log_2(g_i)$를 모니터링하여
$H < 0.3$ bits일 경우 gate collapse를 감지한다.

== 금융 적용 (Financial Interpretation)

금융 거래는 강한 주기성(월급일, 주말 외식), 점진적 추세(라이프스타일 변화, 이탈 접근),
불규칙 잔차(여행, 사기)를 보인다. 각 구성요소는 모델의 강점에 직접 매핑된다:
Mamba $arrow$ 추세, PatchTST $arrow$ 계절성, LNN $arrow$ 잔차.

== 입출력 사양

#figure(
  table(
    columns: (auto, auto),
    stroke: 0.5pt,
    [*Input*], [Transaction seq $[B, 180, 16]$ + Session seq $[B, 90, 8]$],
    [*Mamba output*], [128D + 64D = 192D],
    [*LNN output*], [64D + 32D = 96D (SingleStep on Mamba final state)],
    [*PatchTST output*], [64D + 32D = 96D],
    [*Ensemble output*], [64D (gated combination for PLE gate)],
  ),
  caption: [Temporal Ensemble Expert 입출력 사양.],
)

== 구현 참고사항

- Mamba $arrow$ LNN은 serial (Mamba 학습 후 LNN이 time-scale 보정), PatchTST는 independent.
  이는 input 분리를 통한 앙상블 다양성 보장 설계이다.
- A 행렬은 HiPPO-style 대각선 $[-1, -2, ..., -N]$으로 초기화하여 다중 스케일 메모리 감쇠를 구현한다.
- Gate entropy가 $log_2(3) approx 1.585$ bits (균일 분포)에서 $< 0.3$ bits로 떨어지면 한 모델이 지배하는 것으로 gate collapse 경고를 발생시킨다.

*주요 참고문헌:*
Gu & Dao (NeurIPS 2023), Hasani et al. (AAAI 2021), Nie et al. (ICLR 2023).

#pagebreak()

// ============================================================
= Hyperbolic GCN Expert --- Hierarchical Structure
// ============================================================

== 선정 근거 (Why Hyperbolic GCN?)

추천 데이터는 두 가지 근본적으로 다른 기하학적 구조를 가진다:
(1) 사용자-아이템 상호작용은 _peer-to-peer_ (계층 없음),
(2) 가맹점 분류 코드(MCC)는 _트리 계층구조_ (Root $arrow$ L1 $arrow$ L2 $arrow$ Brand $arrow$ Branch, $tilde$550K 노드).
단일 기하학으로는 둘 다 효율적으로 표현할 수 없다.

깊이 $d$의 완전 이진 트리를 유클리드 공간에 등간격으로 임베딩하려면 $O(2^d)$ 차원이 필요하다.
$tilde$50K 브랜드 수준 노드를 왜곡 없이 임베딩하려면 수만 차원이 필요하다.
쌍곡 공간(음의 곡률)은 원점에서 멀어질수록 _지수적으로_ 체적이 증가하여 트리 분기와 정확히 일치한다.
*8D Poincare Ball로 전체 MCC 계층구조를 표현할 수 있다.*

Nickel & Kiela (2017)는 WordNet 계층구조에서 5D 쌍곡 임베딩이 200D 유클리드 임베딩을
능가함을 보였다.

== 대안 비교

- *LightGCN vs. NGCF:* feature transformation $W$와 비선형 활성화 $sigma$를 제거하면
  ID 기반 협업 필터링에서 오히려 성능이 _향상_된다. 변환할 raw feature가 없으므로 단순할수록 좋다.
- *H-GCN vs. Euclidean tree embedding:* 깊이 $d$ 완전 이진 트리는 유클리드에서 $O(2^d)$
  차원이 필요하지만 쌍곡 공간에서는 $O(d)$ 차원이면 충분하다.
- *2-Stage vs. end-to-end GCN:* Pinterest PinSage 패턴 --- 오프라인 그래프 사전 계산은
  프로덕션 시스템의 표준이다. 그래프 업데이트 빈도와 모델 학습 빈도를 분리한다.

== 수학적 배경

=== Poincare Ball Model (H-GCN)

$ bb(B)_c^d = { bold(x) in bb(R)^d : c ||bold(x)||^2 < 1 } $

*Exponential map (tangent $arrow$ hyperbolic):*
$ exp_(bold(0))(bold(v)) = tanh(sqrt(c) ||bold(v)||) dot bold(v)/(sqrt(c) ||bold(v)||) $

*Logarithmic map (hyperbolic $arrow$ tangent):*
$ log_(bold(0))(bold(y)) = "arctanh"(sqrt(c) ||bold(y)||) dot bold(y)/(sqrt(c) ||bold(y)||) $

*Poincare distance:*
$ d_(bb(B))(bold(x), bold(y)) = 1/sqrt(c) "arccosh"(1 + (2c ||bold(x) - bold(y)||^2)/((1 - c||bold(x)||^2)(1 - c||bold(y)||^2))) $

경계 근처: 분모 $arrow 0$이므로 거리가 폭발한다 (작은 유클리드 거리 = 큰 쌍곡 거리).
이것이 계층적 깊이를 자연스럽게 인코딩한다.

*Riemannian gradient correction:*
$ nabla_"Riem" f(bold(x)) = ((1 - c||bold(x)||^2)^2)/4 nabla_"Euclid" f(bold(x)) $

*Fermi-Dirac decoder (link prediction):*
$ P("edge" | u, v) = 1/(exp((d_(bb(B))(u,v) - r) slash t) + 1) $

통계 물리학에서 차용. $r$ = margin (Fermi energy), $t$ = temperature.

*Frechet mean (Einstein midpoint approximation):*
$ gamma_i = 1/sqrt(1 - c||bold(x)_i||^2), quad macron(bold(x)) = "proj"((sum_i w_i gamma_i bold(x)_i)/(sum_i w_i gamma_i)) $

Lorentz factor $gamma_i$는 경계 포인트(전문화된 소비자)에 높은 가중치를 부여한다.

== 금융 적용

- *LightGCN*: 다중 hop 협업 시그널로 간접 선호를 포착.
  "고객 A가 스타벅스에서 구매, 고객 B도 스타벅스와 이디야에서 구매" $arrow$ A가 이디야를 선호할 수 있음.
- *H-GCN*: MCC 분류 계층(Root $arrow$ L1(8) $arrow$ L2($tilde$100) $arrow$ Brand($tilde$50K) $arrow$ Branch($tilde$500K))은
  본질적으로 트리 구조이다.
- *Co-visitation edges*: 행동 시그널이 정적 MCC 계층을 보완한다.
  "스타벅스 방문자가 7일 내 이디야도 방문" $arrow$ 지수 시간 감쇠로 가중된 에지 생성.

== 입출력 사양

#figure(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt,
    [*Property*], [*LightGCN*], [*H-GCN*],
    [Nodes], [Customers + Merchants (bipartite)], [Merchants only (MCC tree)],
    [Edges], [Customer-Merchant transactions], [Parent-child + Brand co-visitation],
    [Space], [Euclidean $bb(R)^(64)$], [Poincare Ball $bb(B)^8$],
    [Learning], ["Who likes what" (CF)], ["How merchants relate" (hierarchy)],
    [Output], [Customer embedding 64D (direct)], [Merchant emb $arrow$ per-customer agg 47D],
  ),
  caption: [Dual GCN 아키텍처 비교.],
)

== 구현 참고사항

- *2-Stage pipeline:* Stage 1 (offline) --- 그래프 레벨 학습 (LightGCN: BPR, H-GCN: self-supervised Fermi-Dirac).
  임베딩은 Parquet으로 저장. Stage 2 (online) --- lookup + lightweight MLP adaptation.
  추론 시 그래프 전파 없음 --- 단일 GPU VRAM 친화적.
- Co-visitation 에지의 스케일 팩터 0.5를 적용하여 taxonomy 에지의 구조적 우위를 보존한다.

*주요 참고문헌:*
He et al. (SIGIR 2020), Chami et al. (NeurIPS 2019), Nickel & Kiela (NeurIPS 2017).

#pagebreak()

// ============================================================
= PersLay / TDA Expert --- Topological Structure
// ============================================================

== 선정 근거 (Why TDA/PersLay?)

전통적 통계(평균, 분산, 상관)는 분포의 모멘트를 요약하지만 데이터의 _구조적 형태_ ---
클러스터 연결성, 순환적 소비 패턴, 소비 공간의 void --- 를 놓친다.
평균/분산이 동일한 두 고객이 근본적으로 다른 소비 위상을 가질 수 있다
(하나의 연속 클러스터 vs. 주기적 전환이 있는 두 분리된 클러스터).

Persistent Homology는 모든 스케일에서 데이터를 동시에 관찰하며(filtration),
위상적 특징(연결 컴포넌트 $H_0$, 루프 $H_1$, void $H_2$)이 나타나고 사라지는 것을 추적한다.
세 가지 고유한 속성을 제공한다:
(1) 좌표 불변성, (2) 단일 임계값 선택 없는 다중 해상도 분석,
(3) stability theorem에 의한 수학적으로 보장된 노이즈 강건성.

== 대안 비교

#figure(
  table(
    columns: (auto, auto, auto, auto),
    stroke: 0.5pt,
    [*Method*], [*Variable-size PD*], [*End-to-end learnable*], [*Stability guarantee*],
    [Persistence Images], [Fixed grid], [No], [Approximate],
    [Persistence Landscapes], [Function space], [No (Banach space)], [Yes],
    [*PersLay*], [*Yes (phi + rho)*], [*Yes*], [*Yes (inherits)*],
    [Persformer (2024)], [Yes (attention)], [Yes], [Yes],
  ),
  caption: [PD 처리 방법 비교. PersLay는 sum aggregation으로 gradient bottleneck이 없고 persistence weighting으로 패딩 자동 처리가 가능하여 프로덕션 안정성이 높다.],
)

== 수학적 배경

=== PersLay Layer (Carriere et al., 2020)

$ "PersLay"(D) = rho(sum_((b,d) in D) w(b,d) dot phi(b,d)) $

- $phi$: point transformation (RationalHat or Gaussian)
- $w$: weighting function (persistence-based: $w = |d - b|^p$, or learned)
- $rho$: permutation-invariant aggregation (sum, mean, max, or attention)

=== Homology Groups

$ H_k = "Ker"(partial_k) slash "Im"(partial_(k+1)) $

$beta_k = "rank"(H_k)$: 독립적인 $k$-차원 "구멍"의 수.
$beta_0$ = connected components, $beta_1$ = loops, $beta_2$ = voids.

=== Stability Theorem

$ d_B ("Dgm"(f), "Dgm"(g)) <= ||f - g||_infinity $

입력 섭동이 filtration 함수의 최대 변화로 바운드된다. 위상 피처의 노이즈 강건성을 수학적으로 보장한다.

=== Vietoris-Rips Complex

$ sigma = {x_0, ..., x_k} in "VR"_epsilon (X) <==> d(x_i, x_j) <= epsilon, quad forall i,j $

=== Persistence Entropy (Rucco et al., 2016)

$ E = -sum_(i=1)^N p_i log p_i, quad p_i = (d_i - b_i)/(sum_j (d_j - b_j)) $

높은 entropy = 균일하게 분포된 다양한 위상 피처. 낮은 entropy = 소수의 지배적 패턴.

=== Wasserstein-1 Distance for Phase Transition

$ W_1("PD"_1, "PD"_2) = inf_gamma sum_(x in "PD"_1) ||x - gamma(x)||_infinity $

전반부와 후반부 거래의 persistence diagram 간 구조적 변화를 측정한다.

*Phase transition probability:*
$ P_"transition" = 1/(1 + e^(-2(Delta_"total" - tau))), quad tau = 0.5 $

=== 5-Block Multi-Beta Architecture

#figure(
  table(
    columns: (auto, auto, auto, auto),
    stroke: 0.5pt,
    [*Block*], [*Input*], [*Homology*], [*Output*],
    [short\_beta0], [90-day app logs $[B, 200, 3]$], [$H_0$ (clusters)], [64D],
    [short\_beta1], [90-day app logs], [$H_1$ (loops)], [64D],
    [long\_beta0], [12-month txn $[B, 150, 3]$], [$H_0$ (clusters)], [64D],
    [long\_beta1], [12-month transactions], [$H_1$ (loops)], [64D],
    [long\_beta2], [12-month transactions], [$H_2$ (voids)], [64D],
  ),
  caption: [PersLay 5-block 구성.],
)

Short concat 128D + Long concat 192D + Global stats MLP 32D + Phase transition 10D = *362D $arrow$ final\_mlp $arrow$ 64D*.

=== TDA Offline Feature Pipeline (70D)

*70D = 24D (short) + 36D (long) + 10D (phase):*

- *tda\_short (24D):* 90일 앱 로그. 6D point cloud per transaction.
  $H_0 + H_1 times 6$ features (entropy, lifetime mean/std/min/max/median) $times 2$ scopes.
- *tda\_long (36D):* 12개월 카드 거래. $H_0 + H_1 + H_2 times 6$ features $times 2$ scopes.
- *phase\_transition (10D):* PD Distance 4D + Transition Detection 6D.

== 금융 적용

- *$H_0$ (connected components):* 분리된 소비 클러스터가 라이프스타일 세분화를 드러낸다.
  "식료품 클러스터" vs "여행 클러스터"가 다른 스케일에서 합쳐지는 것은 소비 다양화를 나타낸다.
- *$H_1$ (loops):* 순환적 소비 패턴(월간: 식료품 $arrow$ 교통 $arrow$ 오락 $arrow$ 식료품)이
  persistent loop로 포착된다. 강한 loop = 습관적 패턴 = 예측 가능한 행동.
- *$H_2$ (voids):* 금액-카테고리-시간 공간의 3D void는 체계적 회피 패턴을 드러낸다
  (예: 중간 규모 지출 부재 = 소액 + 대액만).
- *Phase transition:* Wasserstein distance로 행동 변화를 정량화한다
  (이직, 생애 이벤트, 재정 위기).

== 입출력 사양

#figure(
  table(
    columns: (auto, auto),
    stroke: 0.5pt,
    [*PersLay Input*], [Raw persistence diagrams (dual mode: raw PD / pre-computed 70D fallback)],
    [*PersLay Output*], [64D expert representation for PLE gate],
    [*TDA Offline Output*], [70D features integrated into main 734D tensor],
    [*Computation*], [Ripser++ (GPU) $arrow$ Ripser (CPU) $arrow$ giotto-tda (fallback)],
  ),
  caption: [PersLay / TDA Expert 입출력 사양.],
)

== 구현 참고사항

- Production config: RationalHatPhi + persistence weighting + sum aggregation.
- Cold-start 4-stage progressive TDA: Day 0 (18D median) $arrow$ 7--30d (9D histogram) $arrow$ \<12m (24D H0,H1) $arrow$ 12m+ (36D full).
- Time-stratified sampling: 고객당 최대 1000 포인트, $k$개 시간 버킷에 걸쳐 층화 추출하여 시간 순서를 보존한다.

*주요 참고문헌:*
Carriere et al. (AISTATS 2020), Cohen-Steiner et al. (DCG 2007), Bauer (2021).

#pagebreak()

// ============================================================
= LightGCN Expert --- Collaborative Filtering
// ============================================================

== 선정 근거 (Why LightGCN?)

추천 시스템에서 사용자-아이템 상호작용 그래프는 _peer-to-peer_ 구조(계층 없음)를 가진다.
이 구조에서의 핵심 질문은 "누가 무엇을 좋아하는가"이며, 다중 hop 협업 시그널을 통해
직접 상호작용하지 않은 아이템에 대한 간접 선호를 추론해야 한다.

LightGCN은 NGCF에서 feature transformation $W$와 nonlinear activation $sigma$를 _제거_하여
ID 기반 협업 필터링에서 오히려 성능을 향상시킨 아키텍처이다.
변환할 raw feature가 없는 상황에서 단순할수록 좋다는 원칙을 따른다.

== 대안 비교

- *NGCF:* feature transformation + nonlinearity 포함. 협업 필터링에서 불필요한 복잡성.
- *GraphSAGE:* 이웃 샘플링 기반으로 inductive 학습에 적합하지만, transductive 설정에서
  LightGCN의 전체 이웃 활용이 우수하다.
- *GAT:* attention 메커니즘이 과적합 위험을 증가시키며, 비용 대비 이득이 미미하다.

== 수학적 배경

=== Message Passing

$ bold(e)_u^((k+1)) = sum_(i in cal(N)_u) 1/(sqrt(|cal(N)_u|) dot sqrt(|cal(N)_i|)) dot bold(e)_i^((k)) $

대칭 정규화 $tilde(A) = D^(-1 slash 2) A D^(-1 slash 2)$는 발신자(인기 아이템)와 수신자 영향을 모두 감쇠시킨다.

*Layer combination:*
$ bold(e)_u^"final" = 1/(L+1) sum_(k=0)^L bold(e)_u^((k)) $

모든 hop(0-hop self + 1,2,3-hop neighbor)에 대한 균일 평균. 학습 가능한 attention weight 없이도 경험적으로 우수하며 과적합을 방지한다.

*BPR Loss:*
$ cal(L)_"BPR" = -sum_((u, i^+, i^-)) log sigma(hat(y)_(u i^+) - hat(y)_(u i^-)) + lambda ||Theta||^2 $

== 금융 적용

- *Cross-selling:* "고객 A가 스타벅스에서 구매, 고객 B도 스타벅스와 이디야에서 구매"
  $arrow$ A가 이디야를 선호할 수 있음. 은행업에서의 교차 판매에 핵심적이다.
- *Cold-start 완화:* sparse한 사용자도 다중 hop을 통해 유사 사용자의 풍부한 시그널을 활용할 수 있다.
- *확장성:* 추론 시 그래프 전파 없이 사전 계산된 임베딩을 lookup하므로 실시간 서빙에 적합하다.

== 입출력 사양

#figure(
  table(
    columns: (auto, auto),
    stroke: 0.5pt,
    [*Input*], [User-item bipartite graph (customer-merchant transactions)],
    [*Embedding dim*], [64D (Euclidean $bb(R)^(64)$)],
    [*Layers*], [3 hops with uniform averaging],
    [*Loss*], [BPR (pairwise ranking)],
    [*Output*], [Customer embedding 64D for PLE gate],
  ),
  caption: [LightGCN Expert 입출력 사양.],
)

== 구현 참고사항

- 2-Stage pipeline의 Stage 1에서 오프라인으로 BPR 학습 후 임베딩을 Parquet으로 저장한다.
- L2 정규화는 초기 임베딩에만 적용하고 GCN 출력에는 적용하지 않는다.
- H-GCN과 _별도_ 전문가로 분리하여 유클리드(CF)와 쌍곡(계층) 기하학의 독립적 gradient flow를 보장한다.

*주요 참고문헌:*
He et al. (SIGIR 2020), Rendle et al. (UAI 2009), Kipf & Welling (ICLR 2017).

#pagebreak()

// ============================================================
= Causal Expert (NOTEARS) --- Causal Inference <sec6-causal>
// ============================================================

== 선정 근거 (Why Causal Expert?)

표준 추천 시스템은 _상관관계_("A를 산 고객이 B도 산다")에 의존하며,
허위 연관(spurious association)과 진정한 인과 효과를 혼동한다.
예: "프리미엄 카드 소지자의 여행 보험 가입률이 높다"는 소득 수준이라는
교란 변수(confounder)에 의한 것일 수 있다 --- 카드가 보험 가입을 _야기_하는 것이 아니다.

A/B 테스트는 gold standard이지만 규모 확장이 불가능하다
(18개 태스크 $times$ $N$개 전략 = 실행 불가), 느리고 (수 주 소요),
집단 수준 ATE만 제공한다.

== 대안 비교

- *A/B Testing:* Gold standard이지만 규모 확장 불가, 개인 수준 ITE 불가.
- *GES/PC Algorithm:* 빈도주의 인과 구조 학습이지만 조건부 독립 검정의 통계적 검정력 이슈.
- *DoWhy/EconML:* 인과 효과 추정에 특화되지만 DAG 학습은 별도.
- *DAGMA (Bello et al., 2022):* NOTEARS 개선판으로 향후 교체 가능한 후보.

== 수학적 배경

=== Feature Compression

$ bold(z) = "Compressor"(bold(x)): bb(R)^(644) arrow bb(R)^(128) arrow bb(R)^(32) $

644D 정규화 피처를 32개 인과 변수로 축소한다.
DAG 인접 행렬이 $644^2 approx 410$K 엔트리로 폭발하는 것을 방지한다.

=== SCM (Structural Causal Model) Intervention

$ hat(bold(z)) = bold(z) + bold(z)(bold(W) circle.small bold(W)) $

- $bold(W) in bb(R)^(32 times 32)$: 학습 가능한 가중 인접 행렬 (`nn.Parameter`)
- $bold(W) circle.small bold(W)$: element-wise 제곱 $arrow$ _비음수_ 인과 강도 보장
- $W_(i,j)^2$: 변수 $j$에서 $i$로의 인과 영향 강도
- Residual connection ($bold(z) +$)이 원본 정보를 보존하면서 인과 조정을 추가

=== NOTEARS Acyclicity Constraint

$ h(bold(W)) = "tr"(e^(bold(W) circle.small bold(W))) - d = 0 $

*수학적 해석:*
$e^(bold(M))$의 $(i,i)$ 대각 원소는 노드 $i$에서 자신으로 돌아오는 모든 가중 경로의 합이다.
그래프가 DAG(비순환)이면 이러한 회귀 경로가 존재하지 않으므로
$e^(bold(M))_(i,i) = 1$ (항등 행렬 기여분만), $"tr"(e^(bold(M))) = d$가 된다.

*Taylor 10-term approximation:*
$ e^(bold(M)) approx sum_(k=0)^9 (bold(M)^k)/(k!) $

길이 10까지의 cycle을 감지한다. 32-node DAG에서 10-hop 이상의 cycle은 실질적으로 불가능하다.

=== DAG Regularization Loss

$ cal(L)_"DAG" = lambda_"acyclic" dot h(bold(W)) + lambda_"sparse" dot ||bold(W) circle.small bold(W)||_1 $

#figure(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt,
    [*Hyperparameter*], [*Default*], [*Role*],
    [`dag_lambda`], [0.01], [Acyclicity constraint strength],
    [`sparsity_lambda`], [0.001], [Edge sparsity (L1 on adjacency)],
    [`n_causal_vars`], [32], [Number of causal nodes],
  ),
  caption: [Causal Expert 하이퍼파라미터. `dag_lambda > 0.1`이면 $W$가 영행렬로 붕괴한다 (DAG 페널티가 태스크 손실을 지배, Expert가 항등 함수로 퇴화).],
)

== 금융 적용

- 교란 변수(income)를 제거하고 _방향성_ 인과 관계를 학습하여
  "이 추천이 행동 변화를 _야기_할 것인가?"에 답한다.
- SCM의 방향성($W_(i,j) eq.not W_(j,i)$)이 설명 가능한 추천 경로를 제공한다.
- 관찰 데이터에서 individual treatment effect (ITE) 추정을 가능하게 한다.

== 입출력 사양

#figure(
  table(
    columns: (auto, auto),
    stroke: 0.5pt,
    [*Input*], [644D normalized features $[B, 644]$],
    [*Compressor*], [$644 arrow 128 arrow 32$ (causal variables)],
    [*SCM*], [$32 times 32$ learnable adjacency $bold(W)$],
    [*Output*], [64D causal representation + DAG (visualization)],
    [*Auxiliary loss*], [$cal(L)_"DAG" = lambda_"acyclic" dot h(bold(W)) + lambda_"sparse" dot ||bold(W)^2||_1$],
  ),
  caption: [Causal Expert 입출력 사양.],
)

== 구현 참고사항

- NOTEARS 논문은 Augmented Lagrangian (엄격한 등식)을 사용하지만 구현에서는 simple penalty method를 사용한다.
  End-to-end MTL joint training 환경에서 더 안정적이다.
- 인접 행렬 $bold(W)$의 Hadamard square ($bold(W) circle.small bold(W)$)는
  sign-agnostic 원본과 달리 비음수 인과 강도를 강제한다.
- OT Expert와 _별도_ 전문가로 유지한다: NOTEARS acyclicity constraint와
  Sinkhorn entropy regularization의 loss surface 기하학이 완전히 다르다.

*주요 참고문헌:*
Zheng et al. (NeurIPS 2018), Bello et al. (ICML 2022).

#pagebreak()

// ============================================================
= Optimal Transport Expert --- Distributional Matching
// ============================================================

== 선정 근거 (Why Optimal Transport?)

고객의 소비 패턴을 프로토타입 프로필과 비교할 때 피처 공간의 기하학적 구조를 존중해야 한다.
KL divergence는 분포의 support가 겹치지 않으면 정의되지 않고,
유클리드 거리는 피처 공간의 기하학적 구조를 무시한다.
Wasserstein distance는 ground metric space의 _기하학_을 반영한다 ---
"서울--인천"이 "서울--부산"보다 가깝다는 것을 확률 질량이 겹치지 않아도 포착한다.

Causal Expert와 상보적: Causal은 "이 추천이 행동 변화를 _야기_할 것인가?" (방향성),
OT는 "이 고객의 소비 분포가 목표 프로필에 얼마나 _가까운가_?" (기하학적 거리).

== 대안 비교

- *KL Divergence:* support가 겹치지 않으면 undefined. 비대칭.
- *Total Variation:* ground metric 무시, 분포 형태만 비교.
- *Euclidean distance:* 피처 공간의 의미적 구조 무시.
- *Sliced Wasserstein:* 계산 효율적 대안, 향후 독립적으로 교체 가능.

== 수학적 배경

=== Distribution Projection

$ bold(mu) = "softmax"("DistProjector"(bold(x))) in Delta^(32) $

644D 피처를 확률 심플렉스로 변환하여 각 고객의 피처 프로필을
32개 잠재 카테고리에 대한 이산 분포로 표현한다.

=== Learnable Reference Distributions

$ bold(nu)_k = "softmax"(bold(ell)_k) in Delta^(32), quad k = 1, ..., 16 $

16개의 학습 가능한 프로토타입 고객 프로필 (`nn.Parameter`), `randn(16, 32) * 0.1`으로 초기화.

=== PSD Cost Matrix

$ bold(C) = bold(M)^top bold(M) in bb(R)^(32 times 32) $

양의 준정부호성(PSD) 보장: $bold(x)^top (bold(M)^top bold(M)) bold(x) = ||bold(M) bold(x)||^2 >= 0$.
비용 행렬이 _학습 가능_하여 태스크에 최적화된 시맨틱 거리를 제공한다.

=== Entropy-Regularized Optimal Transport

*Kantorovich problem with entropic regularization:*
$ min_(bold(P) in cal(U)(bold(mu), bold(nu))) chevron.l bold(P), bold(C) chevron.r + epsilon dot H(bold(P)) $

여기서:
- $cal(U)(bold(mu), bold(nu)) = {bold(P) >= 0 : bold(P) bold(1) = bold(mu), bold(P)^top bold(1) = bold(nu)}$
- $H(bold(P)) = -sum_(i,j) P_(i,j) log P_(i,j)$: entropy regularization
- $epsilon = 0.1$: regularization coefficient

Entropy regularization은 문제를 *strictly convex*로 만든다 (유일 해, 수렴 보장).

=== Log-Domain Sinkhorn Algorithm

$ bold(u)_"new" = log bold(mu) - "logsumexp"(-bold(C)/epsilon + bold(v)) $
$ bold(v)_"new" = log bold(nu) - "logsumexp"(-bold(C)^top/epsilon + bold(u)) $

Log-domain 계산으로 $epsilon$이 작을 때의 floating-point underflow를 방지한다.
10회 반복이면 실용적 수렴에 충분하다.

=== Wasserstein Distance Vector

$ bold(w) = [W(bold(mu), bold(nu)_1), W(bold(mu), bold(nu)_2), ..., W(bold(mu), bold(nu)_(16))] in bb(R)^(16) $

여기서 $W(bold(mu), bold(nu)_k) = chevron.l bold(P), bold(C) chevron.r_F = sum_(i,j) P_(i,j) dot C_(i,j)$.

이것은 *distributional coordinate system*을 생성한다 ---
각 고객이 16개 참조 프로토타입까지의 거리로 위치가 결정된다.

=== Wasserstein Encoder

$ bold(o) = "WassersteinEncoder"(bold(w)): bb(R)^(16) arrow bb(R)^(128) arrow bb(R)^(64) $

== 금융 적용

Wasserstein distance는 "이 고객의 소비 패턴이 전형적인 여행형/저축형/외식형 프로필과
얼마나 다른가, 그리고 _어떤 카테고리가 어느 방향으로 이동해야_ 매칭되는가?"를 정량화한다.

이는 KL divergence나 유클리드 거리로는 불가능한 방향적(directional) 정보를 제공한다.

== 입출력 사양

#figure(
  table(
    columns: (auto, auto),
    stroke: 0.5pt,
    [*Input*], [644D normalized features $[B, 644]$],
    [*Distribution*], [$644 arrow 32$ probability simplex $Delta^(32)$],
    [*References*], [16 learnable prototypes $in Delta^(32)$],
    [*Cost matrix*], [Learnable PSD: $bold(M)^top bold(M) in bb(R)^(32 times 32)$],
    [*Sinkhorn*], [10 iterations, log-domain, $epsilon = 0.1$],
    [*Output*], [64D expert representation for PLE gate],
  ),
  caption: [Optimal Transport Expert 입출력 사양.],
)

== 구현 참고사항

- Cuturi (2013) 원본은 고정 비용 행렬과 단일 target을 사용하지만,
  구현에서는 학습 가능한 PSD 비용 행렬과 16개 학습 가능 프로토타입을 사용한다.
- Sinkhorn 반복 횟수를 10으로 고정하여 학습 루프에서의 계산 비용을 예측 가능하게 한다.
- Causal Expert와의 synergy: DeepFM은 대칭적 피처 상호작용 $chevron.l bold(v)_i, bold(v)_j chevron.r$,
  Causal은 비대칭 방향적 인과 $W_(i,j)^2$,
  OT는 거리 함수(metric) $W(mu, nu_k)$를 추출한다.
  세 전문가는 동일한 644D 입력에서 수학적으로 _완전히 다른_ 구조를 추출한다.

*주요 참고문헌:*
Cuturi (NeurIPS 2013), Kantorovich (1942).

#pagebreak()

// ============================================================
= adaTT (Adaptive Task-aware Transfer) <sec8-adatt>
// ============================================================

== 동기: Multi-Task Learning에서의 Negative Transfer

18개 동시 태스크가 전문가 파라미터를 공유할 때
gradient 충돌이 negative transfer를 야기한다.
고정 타워 MTL의 세 가지 근본적 한계:
(1) 공유 backbone이 모든 태스크에 동일하게 영향 --- 한 태스크의 최적화가 다른 태스크 예측을 악화시켜도 감지/방지 메커니즘 없음,
(2) 18개 태스크 간 어떤 쌍이 서로 돕는지/해치는지 측정 불가,
(3) 고정 가중치로는 학습 단계에 따라 변하는 태스크 관계를 추적 불가.

== Core Mechanism: Gradient Cosine Similarity

$ cos(theta_(i,j)) = (bold(g)_i dot bold(g)_j)/(||bold(g)_i|| dot ||bold(g)_j||) $

여기서 $bold(g)_i = nabla_theta cal(L)_i$는 태스크 $i$의 손실에 대한 공유 전문가 파라미터의 gradient.

*왜 cosine인가 (유클리드 아님):*
(1) 스케일 불변 --- 태스크 손실이 수 자릿수 차이나도 방향만 비교,
(2) 해석 가능한 범위 $[-1, 1]$ $arrow$ positive/negative transfer에 직접 매핑,
(3) 효율적 계산: 정규화 후 단일 행렬 곱 $hat(bold(G)) hat(bold(G))^top$으로 모든 $n^2$ 유사도 계산.

=== EMA Stabilization

$ bold(A)_t = alpha dot bold(A)_(t-1) + (1 - alpha) dot cos(theta_t) $

$alpha = 0.9$ (유효 윈도우 $approx 10$ 관측). IIR 1차 저역통과 필터와 동등하며
고주파 배치 노이즈를 제거하면서 진정한 태스크 관계 추세를 보존한다.

=== Transfer-Enhanced Loss

$ cal(L)_i^"adaTT" = cal(L)_i + lambda dot sum_(j eq.not i) w_(i arrow j) dot cal(L)_j $

$lambda = 0.1$ (다른 태스크로부터 10% 영향), `max_transfer_ratio = 0.5`
(전이 손실이 원래 손실의 50%를 초과할 수 없음).

*Gradient 영향:*
$ nabla_theta cal(L)_i^"adaTT" = nabla_theta cal(L)_i + lambda sum_(j eq.not i) w_(i arrow j) nabla_theta cal(L)_j $

두 번째 항은 다중 태스크에 유익한 방향으로 공유 파라미터를 조정하는 보정 벡터이다.

=== Transfer Weight Computation (4-stage)

$ bold(R) = (bold(W) + bold(A)) dot (1 - r) + bold(P) dot r $
$ bold(R)_(i,j) arrow.l 0 quad "if" bold(A)_(i,j) < tau_"neg" $
$ bold(R)_(i,i) = 0 $
$ w_(i arrow j) = "softmax"(bold(R)_(i,j) / T) $

- $bold(W)$: 학습 가능한 전이 가중치 (`nn.Parameter`, 0으로 초기화)
- $bold(A)$: EMA affinity matrix
- $bold(P)$: Group Prior matrix (도메인 지식)
- $r$: Prior blend ratio ($0.5 arrow 0.1$로 annealing)
- $tau_"neg" = -0.1$: Negative transfer threshold
- $T = 1.0$: Softmax temperature

=== Group Prior

#figure(
  table(
    columns: (auto, auto, auto, auto),
    stroke: 0.5pt,
    [*Group*], [*Tasks*], [*Intra-strength*], [*Business Meaning*],
    [engagement], [has\_nba, engagement\_score, cross\_sell\_count,\ will\_acquire\_deposits, will\_acquire\_investments,\ will\_acquire\_accounts, will\_acquire\_lending,\ will\_acquire\_payments], [0.8], [고객 관여/전환],
    [lifecycle], [churn\_signal, product\_stability,\ tenure\_stage, segment\_prediction], [0.7], [고객 생애주기],
    [value], [income\_tier, spend\_level, nba\_primary], [0.6], [고객 가치/행동],
    [consumption], [next\_mcc, mcc\_diversity\_trend, top\_mcc\_shift], [0.7], [소비 패턴],
  ),
  caption: [adaTT 태스크 그룹 정의. Inter-group strength: 0.3.],
)

*Prior Blend Annealing (Bayesian 해석):*
$ r(e) = r_"start" - (r_"start" - r_"end") dot min((e - e_"warmup")/(e_"freeze" - e_"warmup"), 1.0) $

$r: 0.5 arrow 0.1$은 prior-to-posterior 전환을 구현한다:
초기 학습은 도메인 지식(prior)에 의존, 후기 학습은 관측된 gradient affinity(likelihood)를 신뢰한다.

=== 3-Phase Schedule

#figure(
  table(
    columns: (auto, auto, auto, auto),
    stroke: 0.5pt,
    [*Phase*], [*Period*], [*Behavior*], [*Purpose*],
    [Warmup], [Epoch 0 -- warmup], [Affinity만 계산, 전이 손실 없음], [안정적 affinity 데이터 축적],
    [Dynamic], [warmup -- freeze], [Active transfer + annealing prior], [태스크 관계 학습 및 적용],
    [Frozen], [freeze -- end], [고정된 전이 가중치 (detached)], [fine-tuning 안정화, gradient overhead 제거],
  ),
  caption: [adaTT 3-Phase 학습 스케줄.],
)

=== Negative Transfer Detection

$ bold(R)_(i,j) = cases(bold(R)_(i,j) & "if" bold(A)_(i,j) > tau_"neg", 0 & "otherwise") $

$tau_"neg" = -0.1$ (0이 아님)로 약한 음의 상관(noise 가능성)을 허용하면서
명확한 적대적 gradient는 차단한다.

== Transformer Self-Attention과의 유사성

#figure(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt,
    [*Role*], [*Transformer*], [*adaTT*],
    [Query], [Current token's query], [Current task's gradient direction],
    [Key], [Other tokens' response], [Other tasks' gradient directions],
    [Similarity], [$Q K^top / sqrt(d_k)$], [Gradient cosine similarity],
    [Value], [Other tokens' information], [Other tasks' loss values],
    [Output], [Weighted context], [Transfer loss],
  ),
  caption: [adaTT와 Transformer self-attention의 구조적 유사성.],
)

== 대안 비교

#figure(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt,
    [*Method*], [*Mechanism*], [*adaTT 대비 한계*],
    [Fixed Weighting], [Manual task weights], [동적 affinity 측정 불가],
    [GradNorm], [Gradient magnitude balancing], [방향이 아닌 크기만 사용],
    [PCGrad], [Project conflicting gradients], [positive knowledge 선택적 전이 불가],
    [Nash-MTL], [Nash bargaining for Pareto], [계산 비용 ($O(n^2 d)$ vs optimization)],
    [CAGrad], [Worst-case gradient alignment], [측정과 적용의 분리(modularity) 부족],
  ),
  caption: [MTL 접근법 비교.],
)

== 구현 참고사항

- *Phase 1 (Shared Expert Pretrain):* adaTT active --- 15 epochs 동안 gradient extraction + transfer loss.
- *Phase 2 (Cluster Finetune):* adaTT disabled --- Shared experts frozen, cluster-specific sub-heads만 8 epochs 학습.
- Hypernetworks (Ha et al., 2017)의 경량 변형: 학습된 task embedding 대신 관측된 gradient를
  conditioning signal로 사용하여 변화하는 태스크 관계에 zero-delay 적응.
- `detect_negative_transfer()` API가 각 태스크의 적대적 태스크 목록을 반환
  (예: `{"churn_signal": ["has_nba", "engagement_score"]}`).

*주요 참고문헌:*
Tang et al. (RecSys 2020), Yu et al. (NeurIPS 2020), Fifty et al. (ICML 2021),
Chen et al. (ICML 2018), Navon et al. (ICML 2022).

#pagebreak()

// ============================================================
= Feature Engineering Overview --- 11 Disciplines, 316D <sec9-features>
// ============================================================

== 피처 엔지니어링 철학

전통적 통계 피처는 단일 렌즈로 데이터를 바라본다.
다른 학문 분야들은 수세기에 걸쳐 특정 패턴 유형에 최적화된 수학적 도구를 개발해왔다.
핵심 통찰은 *구조적 동형(structural isomorphism)* ---
수학적 관계 구조가 도메인 객체와 무관하게 동일할 때
수식은 표면적 도메인에 관계없이 같은 패턴을 포착한다.

#note[FeatureRouter와 피처 그룹 라우팅][
  전체 316D 피처 텐서는 모든 전문가에게 동일하게 전달되는 것이 아니다.
  `feature_groups.yaml`의 `target_experts` 선언에 따라 `FeatureRouter`가
  각 전문가에게 관련 서브셋만 슬라이싱하여 전달한다.
  예를 들어 PersLay는 TDA 피처 그룹(32D)만, LightGCN은 그래프 피처 그룹(66D)만 입력받는다.
  전문가별 입력 차원: deepfm=162D, temporal=127D, hgcn=34D, perslay=32D,
  causal=158D, lightgcn=66D, ot=124D.
  피처 그룹 라우팅 구성은 `feature_groups.yaml`에서만 관리하며 코드 수정이 불필요하다.
]

== 11개 학문 분야별 피처 분류

#figure(
  table(
    columns: (auto, auto, auto, auto),
    stroke: 0.5pt,
    [*Discipline*], [*Dim*], [*Pattern*], [*Mathematical Tool*],
    [Economics (PIH)], [8D], [영구/일시 소득 분해], [HP Filter, Kalman Filter],
    [Economics (Micro)], [9D], [탄력성, 소비 평활화], [Arc elasticity, HHI, Shannon entropy],
    [Chemical Kinetics], [6D], [행동 변화 속도/가속도], [Arrhenius equation, finite differences],
    [SIR Epidemiology], [5D], [카테고리 채택 확산], [ODE compartmental model],
    [Routine Activity (Criminology)], [5D], [규칙성, burstiness, 시간 이상치], [Circular statistics, burstiness index],
    [Wave Interference], [8D], [주기 분해, 위상 동기화], [FFT, Hilbert transform, PLV],
    [TDA (Topology)], [70D], [소비 위상 구조, 위상 변이], [Persistent Homology, Wasserstein-1],
    [GMM (Statistics)], [22D], [소프트 세분화, 불확실성], [EM algorithm, BIC],
    [HMM (Probabilistic)], [48D], [잠재 상태 전이, 동적 궤적], [Forward-Backward, Viterbi, ODE dynamics],
    [MAB (Decision Theory)], [4D], [탐색/활용 균형], [HHI trend, recency-weighted entropy],
    [Graph Embedding], [111D], [협업 필터링 + 계층 구조], [LightGCN (64D) + H-GCN (47D)],
  ),
  caption: [11개 학문 분야에서 도출된 피처 체계. 총 316D (TDA 70D + GMM 22D + HMM 48D + Economics 17D + Multidisciplinary 24D + MAB 4D + Graph 111D + LNN statistics 18D + HMM 5D summary = 316D+ 모델 파생 피처).],
)

== Economics Features (17D)

=== Friedman PIH (8D)

관측 소득을 영구(permanent) 성분과 일시(transitory) 성분으로 분해한다:
$ Y_t = Y_t^P + Y_t^T $

소비 함수: $C_t = k(r, w, u) dot Y_t^P$ ---
소비자는 영구 소득에 비례하여 지출하고, 일시 소득은 저축/투자로 향한다.

세 가지 추정 방법을 지원한다:
- Moving Average: $hat(Y)_t^P = 1/L sum_(i=0)^(L-1) Y_(t-i)$, $L=12$
- HP Filter: $min_tau {sum_t (Y_t - tau_t)^2 + lambda sum_t [(tau_(t+1)-tau_t) - (tau_t - tau_(t-1))]^2}$, $lambda=14400$
- Kalman Filter: $K_"ss" approx 0.27$ (73% prior weight, 27% observation weight)

=== Microeconomic Behavior (9D)

*Income Elasticity:*
$ epsilon_Y = (partial Q)/(partial Y) dot Y/Q = (d ln Q)/(d ln Y) $

$epsilon_Y > 1$: 사치재 행동, $0 < epsilon_Y < 1$: 필수재, $epsilon_Y < 0$: 열등재.

*Consumption Smoothing (Hall, 1978):*
$ C_t = C_(t-1) + epsilon_t $

Feature: `consumption_smoothing = mu/sigma` (소비의 Sharpe ratio 유사체).

*Spending Diversification (Shannon Entropy):*
$ H = -sum_(i=1)^N s_i ln(s_i) $

*Category Concentration (HHI):*
$ "HHI" = sum_i s_i^2 $

Shannon entropy와 HHI는 Renyi entropy $H_alpha$의 특수 경우 ($alpha arrow 1$: Shannon, $alpha = 2$: HHI).

== Multidisciplinary Features (24D)

네 개 모듈이 데이터의 거의 직교하는 projection을 포착한다:
- *Chemical Kinetics:* 시간의 _미분적 구조_ (1차, 2차 도함수)
- *Epidemic Diffusion:* _상태 공간 전이 구조_ ($S arrow I arrow R$)
- *Crime Pattern:* 시계열의 _통계적 텍스처_ (주기성, 클러스터링, 분산)
- *Interference:* _주파수 도메인 스펙트럼 구조_ (FFT, coherence, 위상)

교차 모듈 조합으로 개별 모듈에서는 보이지 않는 패턴을 발견한다:
예: 높은 `catalyst_sensitivity` + 높은 `burstiness` = 월급일 폭발 소비자
(월 초 타겟 프로모션 최적 대상).

== GMM Clustering Features (22D)

$ p(bold(x)) = sum_(k=1)^K pi_k cal(N)(bold(x) | bold(mu)_k, bold(Sigma)_k) $

$K = 20$ 클러스터, $D = 40$ 입력 차원, full covariance.
출력: 20D cluster probabilities $gamma_(n k)$ + cluster ID + cluster entropy.

*K-Means 대비 GMM의 핵심 장점:*
`GroupTaskExpertBasket`의 20개 클러스터 sub-head 출력을 $gamma_(n k)$로 가중 앙상블한다.
소프트 할당으로 경계 고객의 추천 품질이 향상된다.

== HMM Triple-Mode Features (48D)

세 개의 병렬 Hidden Markov Model:

#figure(
  table(
    columns: (auto, auto, auto, auto),
    stroke: 0.5pt,
    [*Mode*], [*States*], [*Time Scale*], [*Target Tasks*],
    [Journey (AICRA)], [5], [Days/weeks], [CTR, CVR],
    [Lifecycle], [5], [Months/years], [Churn, Retention],
    [Behavior], [6], [Monthly patterns], [NBA, balance\_util],
  ),
  caption: [HMM Triple-Mode 구성. 각 mode 16D = state probs + meta features + ODE dynamics. 총 48D.],
)

*ODE Dynamics Bridge (6D per mode):*
Viterbi 상태 궤적에서 추출 --- velocity, acceleration, Lyapunov exponent,
cycle period, attractor strength, trajectory length.

== Five-Axis Feature Taxonomy

전체 시스템의 316D main tensor (구현 기준) + 별도 입력은 5개 피처 축에 걸쳐 있다.
설계 기준 734D main tensor + 68D separate input도 동일 분류 체계를 따른다:

+ *Static/Snapshot:* demographics, account status
+ *Time-series:* Mamba/LNN-derived temporal patterns
+ *Hierarchical:* merchant hierarchy, graph embeddings
+ *Item/Product:* product interaction features
+ *Model-derived:* HMM 5D summary, Bandit 4D, LNN statistics 18D

FeatureRouter는 이 5축 분류와 `target_experts` 매핑을 결합하여
각 전문가가 자신의 수학적 관점에 적합한 피처 축을 입력받도록 보장한다.

== Two-Level Ensemble Architecture

전체 아키텍처는 *2단계 앙상블*:
- *Level 1:* Temporal Expert 내에서 Mamba/LNN/PatchTST를 learned gating으로 결합
- *Level 2:* 7개 Shared Expert (PersLay, DeepFM, Temporal, LightGCN, H-GCN, Causal, OT) 전체에서
  CGC Gate Attention이 태스크별 결합 수행

FeatureRouter 활성화 이후 각 Expert는 이종 차원의 입력을 받지만
출력은 64D로 균일하게 정렬되어 CGC Gate의 attention 계산이 변경 없이 동작한다.
이 계층적 앙상블이 intra-expert diversity (시간적 다중 해상도)와
inter-expert complementarity (패턴/위상/시간/관계/인과/분포)를 모두 보장한다.

// ============================================================
// References
// ============================================================

#pagebreak()

#heading(numbering: none)[References]

#set text(size: 9pt)
#set par(hanging-indent: 1.5em)

Bauer, U. (2021). Ripser: efficient computation of Vietoris-Rips persistence barcodes. _Journal of Applied and Computational Topology_.

Bello, K. et al. (2022). DAGMA: Learning DAGs via M-matrices and a Log-Determinant Acyclicity Characterization. _ICML_.

Carriere, M. et al. (2020). PersLay: A Neural Network Layer for Persistence Diagrams. _AISTATS_.

Chami, I. et al. (2019). Hyperbolic Graph Convolutional Neural Networks. _NeurIPS_.

Chen, Z. et al. (2018). GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks. _ICML_.

Cohen-Steiner, D., Edelsbrunner, H., & Harer, J. (2007). Stability of Persistence Diagrams. _Discrete & Computational Geometry_.

Cuturi, M. (2013). Sinkhorn Distances: Lightspeed Computation of Optimal Transport. _NeurIPS_.

Dempster, A., Laird, N., & Rubin, D. (1977). Maximum Likelihood from Incomplete Data via the EM Algorithm. _JRSS-B_.

Fifty, C. et al. (2021). Efficiently Identifying Task Groupings for Multi-Task Learning. _NeurIPS_.

Friedman, M. (1957). _A Theory of the Consumption Function_. Princeton UP.

Gu, A. & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. _NeurIPS_.

Guo, H. et al. (2017). DeepFM: A Factorization-Machine based Neural Network for CTR Prediction. _IJCAI_.

Ha, D. et al. (2017). HyperNetworks. _ICLR_.

Hall, R. (1978). Stochastic Implications of the Life Cycle-Permanent Income Hypothesis. _JPE_.

Hasani, R. et al. (2021). Liquid Time-constant Networks. _AAAI_.

He, X. et al. (2020). LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. _SIGIR_.

Navon, A. et al. (2022). Multi-Task Learning as a Bargaining Game. _ICML_.

Nickel, M. & Kiela, D. (2017). Poincare Embeddings for Learning Hierarchical Representations. _NeurIPS_.

Nie, Y. et al. (2023). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. _ICLR_.

Rendle, S. (2010). Factorization Machines. _ICDM_.

Rendle, S. et al. (2009). BPR: Bayesian Personalized Ranking from Implicit Feedback. _UAI_.

Tang, H. et al. (2020). Progressive Layered Extraction: A Novel Multi-Task Learning Model for Personalized Recommendations. _RecSys_.

Wang, R. et al. (2017). Deep & Cross Network for Ad Click Predictions. _KDD_.

Wang, R. et al. (2021). DCN V2: Improved Deep & Cross Network. _WWW_.

Yu, T. et al. (2020). Gradient Surgery for Multi-Task Learning. _NeurIPS_.

Zheng, X. et al. (2018). DAGs with NO TEARS: Continuous Optimization for Structure Learning. _NeurIPS_.
