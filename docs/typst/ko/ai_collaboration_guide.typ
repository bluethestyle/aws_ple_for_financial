// ─────────────────────────────────────────────
// AI 협업 가이드라인: 지식의 확장과 문제 해결을 위한 사고법
// Typst Web App Compatible — Anthropic Design System
// ─────────────────────────────────────────────

#let anthropic-bg = rgb("#F0EFEA")
#let anthropic-text = rgb("#141413")
#let anthropic-accent = rgb("#CC785C")
#let anthropic-muted = rgb("#6B7280")
#let anthropic-rule = rgb("#D1D5DB")

// Legacy aliases for component compatibility
#let navy = anthropic-text
#let burgundy = anthropic-accent
#let gold = anthropic-accent
#let cream = anthropic-bg
#let dark-cream = rgb("#E8E7E2")
#let ink = anthropic-text
#let muted = anthropic-muted
#let light-rule = anthropic-rule

// ── Page setup ──
#set page(
  paper: "a4",
  margin: (top: 2.5cm, bottom: 2.5cm, left: 2.5cm, right: 2.5cm),
  fill: anthropic-bg,
  header: context {
    if counter(page).get().first() > 1 [
      #set text(size: 7.5pt, font: ("Pretendard", "New Computer Modern"), fill: anthropic-muted, tracking: 0.12em)
      #smallcaps[AI 협업 가이드라인]
      #h(1fr)
      #smallcaps[지식의 확장과 문제 해결을 위한 사고법]
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

// ── Base text ──
#set text(
  font: ("Pretendard", "New Computer Modern"),
  size: 10pt,
  fill: anthropic-text,
  lang: "ko",
)

#set par(
  justify: true,
  leading: 0.8em,
  first-line-indent: 1.2em,
  spacing: 1.5em,
)

// ── Heading styles ──
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

// ── Custom components ──

// Section break (replaces ornamental divider)
#let ornament() = {
  v(0.4cm)
  align(center)[
    #line(length: 30%, stroke: 0.5pt + anthropic-rule)
  ]
  v(0.4cm)
}

// Small section break
#let small-ornament() = {
  v(0.2cm)
  align(center)[
    #line(length: 20%, stroke: 0.3pt + anthropic-rule)
  ]
  v(0.2cm)
}

// Epigraph / pull quote
#let epigraph(body, source: none) = {
  v(0.3cm)
  set par(first-line-indent: 0pt)
  block(
    width: 100%,
    stroke: (left: 2pt + anthropic-accent),
    inset: (left: 14pt, right: 14pt, top: 8pt, bottom: 8pt),
  )[
    #text(size: 10pt, fill: anthropic-muted, style: "italic")[#body]
    #if source != none [
      #v(4pt)
      #text(size: 8pt, fill: anthropic-muted, tracking: 0.1em)[— #source]
    ]
  ]
  v(0.3cm)
}

// Level indicator box
#let level-box(level, title, desc, examples) = {
  set par(first-line-indent: 0pt)
  v(0.15cm)
  block(
    width: 100%,
    stroke: (left: 2pt + anthropic-accent),
    inset: (left: 14pt, right: 14pt, top: 10pt, bottom: 10pt),
  )[
    #grid(
      columns: (auto, 1fr),
      gutter: 10pt,
      align(center + horizon)[
        #box(
          width: 28pt,
          height: 28pt,
          radius: 50%,
          fill: anthropic-accent,
        )[
          #align(center + horizon)[
            #text(size: 13pt, fill: white, weight: "bold")[#level]
          ]
        ]
      ],
      [
        #text(size: 12pt, fill: anthropic-text, weight: "bold")[#title]
        #v(2pt)
        #text(size: 10pt, fill: anthropic-muted, style: "italic")[#desc]
        #v(2pt)
        #for ex in examples [
          #text(size: 10pt, fill: anthropic-text)[• #ex] #linebreak()
        ]
      ],
    )
  ]
  v(0.15cm)
}

// Scenario flow block
#let scenario(title, bad-label, bad-flow, good-label, good-flow) = {
  set par(first-line-indent: 0pt)
  v(0.15cm)
  block(
    width: 100%,
    stroke: (left: 2pt + anthropic-accent),
    inset: (left: 14pt, right: 14pt, top: 10pt, bottom: 10pt),
  )[
    #text(size: 11pt, fill: anthropic-text, weight: "bold")[#title]
    #v(6pt)
    // Bad flow
    #block(
      width: 100%,
      stroke: (left: 2pt + rgb("#D4A0A0")),
      inset: (left: 10pt, right: 10pt, top: 6pt, bottom: 6pt),
    )[
      #text(size: 10pt, fill: anthropic-accent, weight: "bold")[#upper[#bad-label]]
      #v(3pt)
      #text(size: 10pt, fill: anthropic-text)[#bad-flow]
    ]
    #v(4pt)
    // Good flow
    #block(
      width: 100%,
      stroke: (left: 2pt + rgb("#6B9E6B")),
      inset: (left: 10pt, right: 10pt, top: 6pt, bottom: 6pt),
    )[
      #text(size: 10pt, fill: rgb("#2D5A2D"), weight: "bold")[#upper[#good-label]]
      #v(3pt)
      #text(size: 10pt, fill: anthropic-text)[#good-flow]
    ]
  ]
  v(0.15cm)
}

// Principle table row
#let principle-row(name, desc, practice) = {
  (
    [#text(fill: anthropic-text, weight: "bold", size: 9.5pt)[#name]],
    [#text(size: 9pt)[#desc]],
    [#text(size: 8.5pt, fill: anthropic-muted, style: "italic")[#practice]],
  )
}

// Template box
#let template-box(title, body) = {
  set par(first-line-indent: 0pt)
  v(0.15cm)
  block(
    width: 100%,
    stroke: (left: 2pt + anthropic-accent),
    inset: (left: 14pt, right: 14pt, top: 10pt, bottom: 10pt),
  )[
    #text(size: 11pt, fill: anthropic-accent, weight: "bold")[#title]
    #v(2pt)
    #text(size: 10pt, fill: anthropic-text)[#body]
  ]
  v(0.15cm)
}

// Anti-pattern box
#let anti-pattern(number, title, desc, alternative) = {
  set par(first-line-indent: 0pt)
  v(0.2cm)
  block(
    width: 100%,
    inset: (left: 0pt, right: 0pt, y: 0pt),
  )[
    #grid(
      columns: (22pt, 1fr),
      gutter: 8pt,
      align(center)[
        #text(size: 16pt, fill: anthropic-accent, weight: "bold")[#number]
      ],
      [
        #text(size: 11pt, fill: anthropic-text, weight: "bold")[#title]
        #v(2pt)
        #text(size: 10pt, fill: anthropic-text)[#desc]
        #v(3pt)
        #text(size: 10pt, fill: anthropic-muted)[-> ]
        #text(size: 10pt, fill: anthropic-accent, style: "italic")[#alternative]
      ],
    )
  ]
  v(2pt)
  line(length: 100%, stroke: 0.2pt + anthropic-rule)
  v(0.1cm)
}

// Dialogue pattern
#let pattern-block(title, principle, examples) = {
  set par(first-line-indent: 0pt)
  v(0.15cm)
  block(width: 100%)[
    #text(size: 10pt, fill: anthropic-accent, style: "italic")[#principle]
    #v(6pt)
    #block(
      width: 100%,
      stroke: (left: 2pt + anthropic-accent),
      inset: (left: 14pt, right: 14pt, top: 10pt, bottom: 10pt),
    )[
      #text(size: 10pt, fill: anthropic-text, weight: "bold")[실무 적용 예시]
      #v(4pt)
      #for ex in examples [
        #text(size: 10pt, fill: anthropic-text)[• #ex]
        #v(2pt)
      ]
    ]
  ]
  v(0.15cm)
}


// ═══════════════════════════════════════════════
// TITLE PAGE
// ═══════════════════════════════════════════════

#set page(header: none, footer: none)

#v(3cm)

#align(center)[
  #text(
    size: 10pt,
    fill: anthropic-muted,
    tracking: 0.5em,
    weight: "regular",
  )[#upper[A Practical Guide to]]
  #v(0.5cm)

  #text(
    size: 26pt,
    fill: anthropic-text,
    weight: "bold",
  )[AI 협업 가이드라인]
  #v(0.1cm)
  #text(
    size: 26pt,
    fill: anthropic-text,
    weight: "bold",
  )[지식의 확장과 문제 해결을 위한 사고법]

  #v(0.6cm)
  #line(length: 30%, stroke: 0.5pt + anthropic-rule)
  #v(0.6cm)

  #text(size: 10pt, fill: anthropic-muted, tracking: 0.2em)[ON THE ART OF INQUIRY\ IN THE AGE OF ARTIFICIAL INTELLIGENCE]
]

#v(2cm)

#align(center)[
  #block(
    width: 70%,
    stroke: (left: 2pt + anthropic-accent),
    inset: (left: 14pt, right: 14pt, top: 8pt, bottom: 8pt),
  )[
    #set par(first-line-indent: 0pt)
    #text(size: 10pt, fill: anthropic-muted, style: "italic")[
      "좋은 도구는 손의 연장이지만,\ 좋은 질문은 사고의 연장이다."
    ]
    #v(0.5cm)
    #text(size: 9pt, fill: anthropic-muted)[
      이 가이드는 AI를 단순한 정보 검색 도구가 아닌\ 사고의 파트너로 활용하기 위한 원칙을 다룬다.\ 프롬프트 기법 이전에, '무엇을 물어야 하는가'와\ '어떤 태도로 협업해야 하는가'라는 근본적 질문에서 출발한다.
    ]
  ]
]

#v(1fr)

#align(center)[
  #text(size: 10pt, fill: anthropic-muted, tracking: 0.15em)[금융 · 마케팅 · 사업 기획 실무자를 위한 실전 가이드]
  #v(0.3cm)
  #line(length: 30%, stroke: 0.3pt + anthropic-rule)
]

#pagebreak()

// ═══════════════════════════════════════════════
// TABLE OF CONTENTS
// ═══════════════════════════════════════════════

#set page(
  header: context {
    if counter(page).get().first() > 1 [
      #set text(size: 9pt, font: ("Pretendard", "New Computer Modern"), fill: muted, tracking: 0.12em)
      #smallcaps[AI 협업 가이드라인]
      #h(1fr)
      #smallcaps[지식의 확장과 문제 해결을 위한 사고법]
      #v(4pt)
      #line(length: 100%, stroke: 0.4pt + light-rule)
    ]
  },
  footer: context {
    let pg = counter(page).get().first()
    if pg > 1 [
      #line(length: 100%, stroke: 0.3pt + light-rule)
      #v(4pt)
      #set text(size: 11pt, font: ("Pretendard", "New Computer Modern"), fill: muted)
      #h(1fr)
      — #pg —
      #h(1fr)
    ]
  },
)

#v(1.0cm)
#align(center)[
  #text(size: 8pt, fill: anthropic-muted, tracking: 0.4em)[#upper[Contents]]
  #v(0.2cm)
  #text(size: 16pt, fill: anthropic-text, weight: "bold")[목 차]
  #v(0.3cm)
  #line(length: 30%, stroke: 0.5pt + anthropic-accent)
]
#v(0.8cm)

#set par(first-line-indent: 0pt)

#{
  let toc-entry(num, title, target, sub-items) = {
    v(0.1cm)
    block[
      #link(target)[
        #grid(
          columns: (auto, 1fr),
          gutter: 8pt,
          text(size: 10.5pt, fill: navy, weight: "bold")[#num],
          text(size: 10.5pt, fill: navy, weight: "bold")[#title],
        )
      ]
      #v(1pt)
      #for (item-text, item-target) in sub-items [
        #h(24pt)
        #link(item-target)[#text(size: 11pt, fill: muted)[#item-text]]
        #linebreak()
      ]
    ]
    v(0.0cm)
    line(length: 100%, stroke: 0.2pt + light-rule)
  }

  toc-entry([서문], [왜 프롬프트 기법만으로는 부족한가], <preface>, ())
  toc-entry([I], [좋은 질문의 해부학], <part1>, (
    ([1.1  질문의 4계층], <sec1-1>),
    ([1.2  좋은 질문의 세 가지 조건], <sec1-2>),
  ))
  toc-entry([II], [사고 확장을 위한 대화 패턴], <part2>, (
    ([2.1  역방향 질문법: 결론에서 전제로], <sec2-1>),
    ([2.2  경계 탐색법: 극단적 조건에서 사고하기], <sec2-2>),
    ([2.3  유추 전이법: 다른 분야의 렌즈로 보기], <sec2-3>),
    ([2.4  시간축 변환법: 과거와 미래의 시점], <sec2-4>),
  ))
  toc-entry([III], [문제 해결을 위한 협업 전략], <part3>, (
    ([3.1  단계적 분해 대화], <sec3-1>),
    ([3.2  반론 요청과 스틸맨], <sec3-2>),
    ([3.3  점진적 정교화], <sec3-3>),
    ([3.4  맥락의 축적과 관리], <sec3-4>),
  ))
  toc-entry([IV], [피해야 할 안티패턴], <part4>, ())
  toc-entry([V], [실전 시나리오별 질문 설계], <part5>, (
    ([A. 시장 분석과 신사업 기획], <scen-a>),
    ([B. 여신 심사와 리스크 판단], <scen-b>),
    ([C. 마케팅 캠페인 기획], <scen-c>),
    ([D. 경영진 보고서 작성], <scen-d>),
    ([E. 업무 프로세스 개선], <scen-e>),
  ))
  toc-entry([VI], [핵심 원칙 요약], <part6>, ())
  toc-entry([부록], [상황별 질문 템플릿], <appendix>, ())
}

#pagebreak()


// ═══════════════════════════════════════════════
// PREFACE
// ═══════════════════════════════════════════════

#v(1.2cm)
#align(center)[
  #text(size: 10pt, fill: gold, tracking: 0.4em)[#upper[Preface]]
  #v(0.15cm)
  #text(size: 16pt, fill: navy, weight: "bold")[서문] <preface>
  #v(0.1cm)
  #text(size: 10pt, fill: muted, style: "italic")[왜 프롬프트 기법만으로는 부족한가]
  #v(0.3cm)
  #line(length: 30%, stroke: 0.5pt + gold)
]
#v(0.6cm)

대부분의 AI 활용 교육은 "어떻게 물어야 하는가"를 가르친다. 역할을 부여하라, 예시를 넣어라, 단계별로 생각하게 하라. 이 기법들은 유용하지만 수단에 불과하다.

진짜 문제는 *"무엇을 물어야 하는가"*\와 *"왜 그것을 묻는가"*\에 있다. 같은 AI를 쓰면서도 어떤 사람은 검색 엔진 수준의 답을 얻고, 어떤 사람은 사고의 돌파구를 얻는다. 그 차이는 프롬프트 기술이 아니라 #text(fill: navy, weight: "bold")[질문의 구조]와 #text(fill: navy, weight: "bold")[협업의 태도]에서 온다.

이 가이드는 AI를 단순한 정보 검색 도구가 아닌 #text(fill: burgundy, style: "italic")[사고의 파트너]로 활용하기 위한 원칙을 다룬다. 금융, 마케팅, 사업 기획 등 실무 현장에서 바로 적용할 수 있는 예시와 함께 구성했다.

#ornament()


#pagebreak()

// ═══════════════════════════════════════════════
// PART I
// ═══════════════════════════════════════════════

= 좋은 질문의 해부학 <part1>

== 질문의 4계층 <sec1-1>

모든 질문은 다음 네 가지 수준 중 하나에 속한다. 수준이 높아질수록 AI와의 대화에서 얻는 가치가 비약적으로 커진다.

#level-box(
  [1],
  [사실 확인 — What],
  [검색으로 충분한 단순 정보 질의. AI는 빠른 검색엔진에 불과하다.],
  (
    ["코픽스 금리가 뭐야?"],
    ["DSR 규제 내용 정리해줘."],
    ["올해 가계대출 증가율은?"],
  ),
)

#level-box(
  [2],
  [구조 이해 — How],
  [메커니즘과 인과관계를 파악하는 질문. 개념 간의 연결고리를 탐색한다.],
  (
    ["코픽스 금리가 변동하면 우리 수신 상품 마진에 어떤 경로로 영향을 주지?"],
    ["고객 이탈률 계산에서 코호트 분석은 어떤 방식으로 작동해?"],
    ["기업 신용평가에서 현금흐름 지표가 재무비율보다 우선시되는 메커니즘은?"],
  ),
)

#level-box(
  [3],
  [판단과 비교 — Why / Which],
  [선택의 근거와 trade-off를 탐색한다. AI가 사고 파트너로 기능하기 시작하는 수준.],
  (
    ["중소기업 여신 심사에서 정량 지표와 정성 평가 중 어떤 것이 부실 예측에 더 유효해?"],
    ["신규 고객 획득 vs. 기존 고객 교차판매, 각각의 전제 조건과 한계는?"],
    ["비대면 채널 확대와 대면 영업 강화, 두 전략의 trade-off를 비교해줘."],
  ),
)

#level-box(
  [4],
  [전제 검토와 재구성 — What if / What's wrong],
  [사고 프레임 자체를 의심하고 재구성한다. 가장 큰 가치를 만들어내는 수준.],
  (
    ["'젊은 고객 = 디지털 선호'라는 가정 자체가 틀렸을 가능성은?"],
    ["고객 만족도와 이탈률의 인과관계가 허구일 수 있는 시나리오는?"],
    ["지금 추진 중인 신사업의 시장 전제가 3년 뒤에도 유효할까?"],
  ),
)

#small-ornament()

== 좋은 질문의 세 가지 조건 <sec1-2>

=== 첫째, 자기 인식의 경계를 명시한다

#set par(first-line-indent: 0pt)
#block(
  width: 100%,
  fill: rgb("#FFF8F6"),
  stroke: 0.3pt + rgb("#D4A0A0"),
  radius: 3pt,
  inset: (x: 12pt, y: 8pt),
)[
  #text(size: 10pt, fill: burgundy, weight: "bold")[나쁜 예] #h(8pt)
  #text(size: 10pt, fill: ink)["마케팅 전략 짜줘."]
]
#v(4pt)
#block(
  width: 100%,
  fill: rgb("#F6FBF6"),
  stroke: 0.3pt + rgb("#A0B8A0"),
  radius: 3pt,
  inset: (x: 12pt, y: 8pt),
)[
  #text(size: 10pt, fill: rgb("#2D5A2D"), weight: "bold")[좋은 예] #h(8pt)
  #text(size: 10pt, fill: ink)["우리는 30\~40대 급여 이체 고객 대상 카드 교차판매를 하고 있는데, 전환율이 2%에서 정체야. 타이밍 문제인지 타겟팅 문제인지 판단이 안 서는데, 어떤 관점에서 진단해야 할까?"]
]
#set par(first-line-indent: 1.2em)

#v(6pt)
후자가 나은 이유는, 질문자가 "자기가 어디까지 파악하고 있고 어디서 막혔는지"를 명확히 했기 때문이다. AI는 그 경계 너머를 탐색할 수 있다.

=== 둘째, 제약조건이 사고의 방향을 잡아준다

#set par(first-line-indent: 0pt)
#block(
  width: 100%,
  fill: rgb("#FFF8F6"),
  stroke: 0.3pt + rgb("#D4A0A0"),
  radius: 3pt,
  inset: (x: 12pt, y: 8pt),
)[
  #text(size: 10pt, fill: burgundy, weight: "bold")[나쁜 예] #h(8pt)
  #text(size: 10pt, fill: ink)["수익성 개선 방안 알려줘."]
]
#v(4pt)
#block(
  width: 100%,
  fill: rgb("#F6FBF6"),
  stroke: 0.3pt + rgb("#A0B8A0"),
  radius: 3pt,
  inset: (x: 12pt, y: 8pt),
)[
  #text(size: 10pt, fill: rgb("#2D5A2D"), weight: "bold")[좋은 예] #h(8pt)
  #text(size: 10pt, fill: ink)["연간 마케팅 예산 5억 원, 고객 접점은 앱 푸시와 문자 중심, 금융 규제상 상품 비교 문구에 제한이 있는 상황에서, 기존 고객 기반 수신 잔액을 15% 늘리는 현실적 방법은?"]
]
#set par(first-line-indent: 1.2em)

#v(6pt)
제약조건은 답을 좁히는 것이 아니라 문제의 본질을 드러낸다. 역설적으로 제약이 많을수록 더 실행 가능한 답이 나온다.

=== 셋째, 답에 의해 자신이 변할 여지를 남긴다

#set par(first-line-indent: 0pt)
#block(
  width: 100%,
  fill: rgb("#FFF8F6"),
  stroke: 0.3pt + rgb("#D4A0A0"),
  radius: 3pt,
  inset: (x: 12pt, y: 8pt),
)[
  #text(size: 10pt, fill: burgundy, weight: "bold")[확증편향] #h(8pt)
  #text(size: 10pt, fill: ink)["이 신사업 계획이 타당한지 확인해줘."]
]
#v(4pt)
#block(
  width: 100%,
  fill: rgb("#F6FBF6"),
  stroke: 0.3pt + rgb("#A0B8A0"),
  radius: 3pt,
  inset: (x: 12pt, y: 8pt),
)[
  #text(size: 10pt, fill: rgb("#2D5A2D"), weight: "bold")[프레임 재구성] #h(8pt)
  #text(size: 10pt, fill: ink)["이 신사업 계획의 핵심 가정 세 가지를 뽑아주고, 각각이 틀렸을 때 사업 전체에 미치는 영향을 평가해줘."]
]
#set par(first-line-indent: 1.2em)

#v(6pt)
질문이 "맞다/틀리다"의 이분법이 아니라 "어떤 조건에서 타당한가"로 열려 있을 때, 대화는 확인이 아닌 발견이 된다.

#pagebreak()

// ═══════════════════════════════════════════════
// PART II
// ═══════════════════════════════════════════════

= 사고 확장을 위한 대화 패턴 <part2>

== 역방향 질문법 <sec2-1>

=== 결론에서 전제로 거슬러 올라가기

대부분은 "이걸 어떻게 할까?"라고 묻는다. 더 생산적인 접근은 반대 방향이다.

#set par(first-line-indent: 0pt)
#block(
  width: 100%,
  fill: dark-cream,
  stroke: 0.3pt + light-rule,
  radius: 3pt,
  inset: (x: 14pt, y: 10pt),
)[
  #text(size: 10pt, fill: muted)[일반적 흐름] #h(12pt)
  #text(size: 10pt, fill: ink)[전제 → 전략 → 실행 → 결과]
  #v(4pt)
  #text(size: 10pt, fill: burgundy)[역방향 질문] #h(12pt)
  #text(size: 10pt, fill: ink, style: "italic")["이 결과가 나오려면 어떤 전제가 참이어야 하지? 그 전제가 깨지는 상황은?"]
]
#set par(first-line-indent: 1.2em)

#pattern-block(
  [역방향 질문법],
  [많은 실무적 판단은 검증되지 않은 가정 위에 서 있다. 이 패턴은 그 '숨겨진 가정'을 발굴하게 만든다.],
  (
    ["이 여신 포트폴리오가 안전하다는 판단은, 어떤 경제 전제에 기대고 있지?"],
    ["이 분기 실적이 좋았다면, 우리 전략 덕인지 시장 환경 덕인지 어떻게 구분해?"],
    ["이 캠페인이 성공했다는 KPI 자체가 잘못 설정된 건 아닐까?"],
  ),
)

== 경계 탐색법 <sec2-2>

=== 극단적 조건에서 사고하기

시스템의 본질은 정상 상태가 아니라 극단에서 드러난다. 이것은 리스크 관리의 본질이기도 하다.

#pattern-block(
  [경계 탐색법],
  [극단적 조건을 탐색하면 평상시에는 보이지 않던 구조적 취약점과 의존성이 드러난다.],
  (
    ["기준금리가 2%p 급등하면 우리 고객 포트폴리오에서 가장 먼저 무너지는 세그먼트는?"],
    ["이 상품 가입자가 예상의 10배가 되면 운영상 어디가 먼저 터지지?"],
    ["경쟁사가 수수료를 0원으로 내리면 우리 사업 모델의 어떤 부분이 살아남아?"],
    ["마케팅 예산이 절반으로 줄면, 남겨야 할 캠페인 하나는 뭐고 그 이유는?"],
  ),
)

== 유추 전이법 <sec2-3>

=== 다른 분야의 렌즈로 보기

복잡한 문제를 같은 프레임 안에서만 바라보면 맹점이 생긴다. 전혀 다른 산업에서 검증된 구조를 자기 문제에 이식하는 것은 실전적 도구다.

#pattern-block(
  [유추 전이법],
  [유추는 장식이 아니라, 문제의 구조적 동형성(structural isomorphism)을 발견하는 방법이다.],
  (
    ["넷플릭스의 추천 시스템 원리를 고객 금융상품 추천에 적용하면 어떤 구조가 될까?"],
    ["항공사 마일리지의 고객 락인 전략을 수신 상품에 비유하면?"],
    ["병원의 환자 분류 체계(트리아지)를 여신 심사 우선순위에 적용할 수 있을까?"],
    ["구독 경제의 이탈 방지 전략이 정기예금 만기 이탈 방지에 주는 시사점은?"],
  ),
)

== 시간축 변환법 <sec2-4>

=== 과거와 미래의 시점에서 바라보기

현재의 관점에 갇히면 문제의 전체 맥락을 놓친다. 시간축을 이동하면 단기 편향을 제거할 수 있다.

#pattern-block(
  [시간축 변환법],
  [현재의 의사결정이 장기적으로 어떤 결과를 가져올지, 과거의 유사 사례에서 무엇을 배울 수 있는지를 탐색한다.],
  (
    ["3년 뒤에 돌아보면, 지금 이 채널 전략에서 가장 후회할 부분은 뭘까?"],
    ["금리가 이렇게 움직인 건 과거에도 있었어? 그때 업계는 어떻게 대응했지?"],
    ["이 고객 행동 패턴이 5년 뒤에도 유효할까, 아니면 세대교체로 소멸할까?"],
    ["지금 추진 중인 디지털 전환이 완료된 시점에서, 오히려 불필요했던 투자는?"],
  ),
)

#pagebreak()

// ═══════════════════════════════════════════════
// PART III
// ═══════════════════════════════════════════════

= 문제 해결을 위한 협업 전략 <part3>

== 단계적 분해 대화 <sec3-1>

복잡한 문제를 한 번에 던지지 않는다. AI와 함께 문제를 분해하는 과정 자체가 해결의 핵심이다.

#set par(first-line-indent: 0pt)
#block(
  width: 100%,
  fill: dark-cream,
  stroke: 0.4pt + light-rule,
  radius: 4pt,
  inset: (x: 14pt, y: 12pt),
)[
  #text(size: 11pt, fill: navy, weight: "bold", tracking: 0.1em)[5단계 분해 프레임워크]
  #v(6pt)
  #grid(
    columns: (24pt, 1fr),
    gutter: 6pt,
    text(size: 14pt, fill: gold, weight: "bold")[1], text(size: 10pt, fill: ink)[#strong[문제 정의] — "내가 해결하려는 것이 정확히 뭐지?"],
    text(size: 14pt, fill: gold, weight: "bold")[2], text(size: 10pt, fill: ink)[#strong[제약 식별] — "어떤 조건들이 해법을 제한하고 있지?"],
    text(size: 14pt, fill: gold, weight: "bold")[3], text(size: 10pt, fill: ink)[#strong[분해] — "독립적으로 풀 수 있는 하위 문제로 나누면?"],
    text(size: 14pt, fill: gold, weight: "bold")[4], text(size: 10pt, fill: ink)[#strong[각개 탐색] — 하위 문제별로 깊이 있는 대화],
    text(size: 14pt, fill: gold, weight: "bold")[5], text(size: 10pt, fill: ink)[#strong[통합 검증] — "부분 해법들을 합치면 전체가 정말 풀리나?"],
  )
]
#set par(first-line-indent: 1.2em)

#v(0.3cm)

#text(size: 11pt, fill: navy, weight: "bold")[실전 예시 — 법인 고객 이탈 방지:]

#set par(first-line-indent: 0pt)
#block(
  width: 100%,
  stroke: (left: 2pt + burgundy),
  inset: (left: 12pt, y: 4pt),
)[
  #set text(size: 10pt)
  #strong[1단계] "법인 고객 이탈"이 정확히 뭘 의미하지? 거래 중단? 잔액 감소? 타행 주거래 전환? \
  #strong[2단계] 제약은? 영업인력 한정, 의사결정자 접근 어려움, 가격 경쟁 한계 \
  #strong[3단계] 하위 문제: 이탈 신호 감지 / 고위험 고객 우선순위 / 잔류 제안 설계 / 효과 측정 \
  #strong[4단계] 각 하위 문제를 별도 대화 턴으로 깊이 탐색 \
  #strong[5단계] 통합 시 현장 영업팀이 실행 가능한 프로세스인지 점검
]
#set par(first-line-indent: 1.2em)

#v(5em)
== 반론 요청과 스틸맨 <sec3-2>

자기 아이디어에 대해 AI에게 "동의"를 구하는 것은 가장 흔한 실수다.

#epigraph(
  ["이게 안 된다고 주장하는 가장 똑똑한 사람의 논리를 구성해봐."],
)

핵심은 *스틸맨(steelman)*—상대 논증의 최강 버전—을 요청하는 것이다. "왜 이게 안 돼?"보다 위의 질문이 훨씬 날카로운 피드백을 이끌어낸다.

#set par(first-line-indent: 0pt)
#block(
  width: 100%,
  fill: dark-cream,
  radius: 3pt,
  stroke: 0.3pt + light-rule,
  inset: (x: 12pt, y: 10pt),
)[
  #text(size: 11pt, fill: navy, weight: "bold", tracking: 0.1em)[활용 예시]
  #v(4pt)
  #text(size: 10pt, fill: ink)[
    • "이 사업계획서의 가장 강력한 반론을 만들어봐." \
    • "이 예산 배분을 반대하는 CFO의 가장 합리적인 논거는?" \
    • "내가 이 투자를 설득하려 하는데, 반대편에서 나올 가장 날카로운 질문 세 가지는?"
  ]
]
#set par(first-line-indent: 1.2em)

#v(4pt)

이것은 보고서 작성 전 상급자의 예상 질문을 시뮬레이션하는 데도 매우 효과적이다.

#v(2em)
== 점진적 정교화 <sec3-3>

한 번의 질문으로 완벽한 결과를 기대하지 않는다. 대화는 탐색의 과정이지, 자판기가 아니다.

#set par(first-line-indent: 0pt)
#block(
  width: 100%,
  fill: dark-cream,
  stroke: 0.3pt + light-rule,
  radius: 3pt,
  inset: (x: 12pt, y: 10pt),
)[
  #text(size: 11pt, fill: navy, weight: "bold", tracking: 0.1em)[각 턴에서의 후속 질문 유형]
  #v(6pt)
  #grid(
    columns: (1fr, 1fr),
    gutter: 8pt,
    text(size: 10pt, fill: ink)[• "이 부분을 더 구체적으로." #text(fill: muted)[ — 깊이 확장]],
    text(size: 10pt, fill: ink)[• "다른 관점에서 보면?" #text(fill: muted)[ — 시야 확장]],
    text(size: 10pt, fill: ink)[• "이게 현장에서 실제로 통할까?" #text(fill: muted)[ — 현실 검증]],
    text(size: 10pt, fill: ink)[• "이 논리에 비약이 있지 않아?" #text(fill: muted)[ — 논리 검증]],
    text(size: 10pt, fill: ink, tracking: -0.01em)[• "경영진에게 한 문장으로 설명하면?" #text(fill: muted)[ — 본질 추출]],
    [],
  )
]
#set par(first-line-indent: 1.2em)

#v(6pt)

#text(size: 11pt, fill: navy, weight: "bold")[실전 예시 — 신상품 기획 (5턴 대화):]

#set par(first-line-indent: 0pt)
#block(
  width: 100%,
  stroke: (left: 2pt + gold),
  inset: (left: 12pt, y: 4pt),
)[
  #set text(size: 11pt)
  #text(fill: gold, weight: "bold")[Turn 1] "30대 맞벌이 부부 대상 적금+보험 결합 상품을 기획하려는데 방향을 잡아줘" \
  #text(fill: gold, weight: "bold")[Turn 2] "고객이 실제로 느끼는 혜택이 뭐지? 경쟁 상품 대비 차별점이 약한 것 같아" \
  #text(fill: gold, weight: "bold")[Turn 3] "규제 관점에서 결합판매 시 유의할 점과 불가능한 구조는?" \
  #text(fill: gold, weight: "bold")[Turn 4] "규제를 우회하지 않으면서 고객 혜택을 극대화하는 대안적 구조는?" \
  #text(fill: gold, weight: "bold")[Turn 5] "내부 품의서 형태로 정리하면 어떤 목차가 되어야 할까?"
]
#set par(first-line-indent: 1.2em)

#v(2em)
== 맥락의 축적과 관리 <sec3-4>

효과적인 협업자는 AI와의 대화에서 맥락을 축적하는 시스템을 만든다.

#set par(first-line-indent: 0pt)
#grid(
  columns: (1fr, 1fr),
  gutter: 10pt,
  block(
    fill: dark-cream,
    stroke: 0.3pt + light-rule,
    radius: 3pt,
    inset: (x: 10pt, y: 8pt),
    width: 100%,
  )[
    #text(size: 11pt, fill: navy, weight: "bold")[메모리 기능 활용]
    #v(3pt)
    #text(size: 10pt, fill: ink)[역할, 도메인, 기술 스택 등 기본 맥락을 AI에 설정한다. 매번 자기소개를 반복할 필요가 없다.]
  ],
  block(
    fill: dark-cream,
    stroke: 0.3pt + light-rule,
    radius: 3pt,
    inset: (x: 10pt, y: 8pt),
    width: 100%,
  )[
    #text(size: 11pt, fill: navy, weight: "bold")[프로젝트별 분리]
    #v(3pt)
    #text(size: 10pt, fill: ink)[여신, 마케팅, 규제 검토 등 주제별로 대화를 분리한다. 집중도가 높아진다.]
  ],
  block(
    fill: dark-cream,
    stroke: 0.3pt + light-rule,
    radius: 3pt,
    inset: (x: 10pt, y: 8pt),
    width: 100%,
  )[
    #text(size: 11pt, fill: navy, weight: "bold")[중간 결론 명시]
    #v(3pt)
    #text(size: 10pt, fill: ink)[3\~5턴마다 "여기까지 정리하면..."을 선언한다. AI와 자기 자신의 사고를 동시에 정리한다.]
  ],
  block(
    fill: dark-cream,
    stroke: 0.3pt + light-rule,
    radius: 3pt,
    inset: (x: 10pt, y: 8pt),
    width: 100%,
  )[
    #text(size: 11pt, fill: navy, weight: "bold")[전제의 명시적 갱신]
    #v(3pt)
    #text(size: 10pt, fill: ink)["아까는 금리 동결을 가정했는데, 인상 시나리오로 바꾸자"처럼 전제 변경을 분명히 한다.]
  ],
)
#set par(first-line-indent: 1.2em)

#pagebreak()

// ═══════════════════════════════════════════════
// PART IV
// ═══════════════════════════════════════════════

= 피해야 할 안티패턴 <part4>

#anti-pattern(
  [①],
  [확증편향 강화],
  ["이 전략이 맞지?"라는 질문은 AI를 동의 기계로 만든다. AI는 기본적으로 사용자의 방향에 맞추려는 경향이 있기에, 이를 의식적으로 깨야 한다.],
  ["이번 분기 실적 부진의 가능한 원인을 세 가지 이상 나열하고, 각각의 설명력을 비교해줘."],
)

#anti-pattern(
  [②],
  [과도한 위임],
  ["보고서 알아서 써줘"는 AI의 강점을 활용하지 못한다. AI는 명확한 제약조건과 평가 기준이 있을 때 가장 뛰어난 결과를 낸다. 방향 설정은 인간의 역할이다.],
  ["독자는 리스크관리위원회 위원 5명, 핵심은 건설업 여신 부실. 이 맥락에서 구조를 잡아줘."],
)

#anti-pattern(
  [③],
  [한 번에 완벽한 답 기대],
  ["마케팅 전략 전체를 세워줘"를 한 번에 기대하면 교과서적이고 표면적인 답이 나온다. 복잡한 문제일수록 여러 턴에 걸친 대화가 효과적이다.],
  [최소 3\~5턴의 반복 대화를 계획하고 시작한다.],
)

#anti-pattern(
  [④],
  [비판 없는 수용],
  [AI의 응답을 그대로 수용하는 것은 가장 위험한 패턴이다. 특히 금융에서는 '그럴듯한 오류'가 치명적일 수 있다. 수치, 규정 인용, 시장 동향은 반드시 원본 대조.],
  ["왜 이렇게 판단했어?", "이 수치의 출처는?", "이 논리에 빠진 전제가 있지 않아?"],
)

#anti-pattern(
  [⑤],
  [도구와의 관계 혼동],
  [AI는 유능한 사고 도구이지 권위 있는 판단자가 아니다. "AI가 이렇게 분석했습니다"는 보고서에서 근거가 될 수 없다. AI의 출력은 '검토해야 할 가설'이다.],
  [최종 판단과 책임은 언제나 사람에게 있다.],
)

#pagebreak()

// ═══════════════════════════════════════════════
// PART V
// ═══════════════════════════════════════════════

= 실전 시나리오별 질문 설계 <part5>

.<scen-a>
 #scenario(
  [A. 시장 분석과 신사업 기획],
  [표면적 접근],
  ["MZ세대 금융 트렌드 알려줘" → 일반론 수용 → 보고서에 붙여넣기],
  [사고 확장형 접근],
  [
    ① "MZ세대 금융 행동에서 기성세대와 _구조적으로_ 다른 점은?" #text(fill: muted, size: 10pt)[ — 구조 파악] \
    ② "그 차이가 우리 수신 상품 설계에 주는 구체적 시사점은?" #text(fill: muted, size: 10pt)[ — 맥락 적용] \
    ③ "이 트렌드가 일시적 유행인지 구조적 변화인지 구분하는 기준은?" #text(fill: muted, size: 10pt)[ — 전제 검증] \
    ④ "구조적 변화라면, 3년 뒤 우리 상품 라인업은 어떤 모습이어야 해?" #text(fill: muted, size: 10pt)[ — 시간축] \
    ⑤ "이 방향과 정반대로 베팅하는 경쟁사가 있다면 그 논리는?" #text(fill: muted, size: 10pt)[ — 스틸맨]
  ],
)

<scen-b>
#scenario(
  [B. 여신 심사와 리스크 판단],
  [표면적 접근],
  ["이 기업 재무제표 분석해줘" → 비율 분석 수용 → 심사 의견서 작성],
  [사고 확장형 접근],
  [
    ① "이 기업의 재무 숫자 이면에 숨겨진 위험 신호가 있다면?" #text(fill: muted, size: 10pt)[ — 숨은 가정] \
    ② "같은 업종·규모에서 부실 발생 기업들의 공통 패턴은?" #text(fill: muted, size: 10pt)[ — 패턴 비교] \
    ③ "이 기업이 가장 취약해지는 거시경제 시나리오는?" #text(fill: muted, size: 10pt)[ — 경계 탐색] \
    ④ "'이 기업은 안전하다'는 내 판단이 틀렸을 근거 세 가지를 만들어봐" #text(fill: muted, size: 10pt)[ — 확증편향 방지] \
    ⑤ "그 리스크를 반영한 여신 조건 설계는?" #text(fill: muted, size: 10pt)[ — 실행 연결]
  ],
)

#pagebreak()
<scen-c>
#scenario(
  [C. 마케팅 캠페인 기획],
  [표면적 접근],
  ["카드 프로모션 아이디어 내줘" → 아이디어 수용 → 실행],
  [사고 확장형 접근],
  [
    ① "최근 3개월간 전환율이 가장 높았던 캠페인의 성공 요인을 분해하면?" #text(fill: muted, size: 10pt)[ — 과거 기반] \
    ② "그 성공 요인이 이번에도 통할까? 시장 환경이 달라진 점은?" #text(fill: muted, size: 10pt)[ — 전제 유효성] \
    ③ "같은 예산으로 완전히 다른 접근을 한다면 어떤 구조가 가능해?" #text(fill: muted, size: 10pt)[ — 대안 탐색] \
    ④ "이 캠페인이 실패하는 시나리오 세 가지와 각각의 사전 징후는?" #text(fill: muted, size: 10pt)[ — 리스크] \
    ⑤ "성공/실패를 판단하는 KPI를 잘못 잡았을 가능성도 검토해줘" #text(fill: muted, size: 10pt)[ — 측정 프레임]
  ],
)

#text[ ] <scen-d>
#scenario(
  [D. 경영진 보고서 작성],
  [표면적 접근],
  ["실적 보고서 써줘" → 결과물 제출],
  [사고 확장형 접근],
  [
    ① "이 보고서를 읽는 임원이 가장 먼저 던질 질문은?" #text(fill: muted, size: 10pt)[ — 독자 관점] \
    ② "그 질문에 대한 우리의 답이 약하다면, 어떤 데이터를 보완해야 해?" #text(fill: muted, size: 10pt)[ — 근거 강화] \
    ③ "'좋아 보이지만 실은 문제인 지표'가 이 보고서에 있다면?" #text(fill: muted, size: 10pt)[ — 표면 너머] \
    ④ "결론을 반대로 해석할 수 있는 여지는? 그 반론에 어떻게 대응?" #text(fill: muted, size: 10pt)[ — 반론 대비] \
    ⑤ "핵심 메시지를 한 문장으로 압축하면?" #text(fill: muted, size: 10pt)[ — 본질 추출]
  ],
)

#text[ ] <scen-e>
#scenario(
  [E. 업무 프로세스 개선],
  [표면적 접근],
  ["업무 효율화 방안 알려줘" → 일반론 수용],
  [사고 확장형 접근],
  [
    ① "우리 팀의 주간 반복 업무 중 시간 대비 가치가 가장 낮은 건?" #text(fill: muted, size: 10pt)[ — 현황 진단] \
    ② "이 업무가 존재하는 원래 목적이 지금도 유효해?" #text(fill: muted, size: 10pt)[ — 존재 이유 검증] \
    ③ "이걸 없앴을 때 발생할 수 있는 부작용은?" #text(fill: muted, size: 10pt)[ — 제거 비용] \
    ④ "같은 목적을 달성하면서 소요 시간을 1/3로 줄이는 방법은?" #text(fill: muted, size: 10pt)[ — 제약 하 최적화] \
    ⑤ "다른 산업에서 비슷한 프로세스를 어떻게 처리하고 있어?" #text(fill: muted, size: 10pt)[ — 유추 전이]
  ],
)

#pagebreak()

// ═══════════════════════════════════════════════
// PART VI
// ═══════════════════════════════════════════════

= 핵심 원칙 요약 <part6>

#v(0.3cm)

#set par(first-line-indent: 0pt)

#{
  let header-cell(body) = table.cell(
    fill: navy,
    inset: (x: 10pt, y: 8pt),
  )[#align(center)[#text(size: 11pt, fill: cream, weight: "bold", tracking: 0.05em)[#body]]]

  let body-cell(body) = table.cell(
    inset: (x: 10pt, y: 8pt),
  )[#text(size: 10pt, fill: ink)[#body]]

  let alt-cell(body) = table.cell(
    fill: dark-cream,
    inset: (x: 10pt, y: 8pt),
  )[#text(size: 10pt, fill: ink)[#body]]

  table(
    columns: (0.46fr, 1.01fr, 1fr),
    stroke: 0.4pt + light-rule,
    align: center+horizon,
    header-cell[원칙], header-cell[설명], header-cell[실무 적용],

    body-cell[#text(fill: navy, weight: "bold")[무지의 자각]],
    body-cell[모르는 것이 무엇인지 아는 것이 출발점],
    body-cell[#text(style: "italic", fill: muted)["내가 확신이 없는 부분은..."으로 시작]],

    alt-cell[#text(fill: navy, weight: "bold")[전제의 의심]],
    alt-cell[답이 아니라 질문의 전제를 검토하라],
    alt-cell[#text(style: "italic", fill: muted)["이 분석의 전제가 틀렸을 가능성을\ 검토해줘"]],

    body-cell[#text(fill: navy, weight: "bold")[제약의 역설]],
    body-cell[제약이 많을수록\ 실행 가능한 답이 나온다],
    body-cell[#text(style: "italic", fill: muted)[예산, 인력, 규제, 일정 등\ 현실 조건을 명시]],

    alt-cell[#text(fill: navy, weight: "bold")[점진적 정교화]],
    alt-cell[반복적 대화로 수렴하라],
    alt-cell[#text(style: "italic", fill: muted)[최소 3~5턴의 대화를 계획하고 시작]],

    body-cell[#text(fill: navy, weight: "bold")[비판적 수용]],
    body-cell[AI의 출력은 가설이지, 결론이 아니다],
    body-cell[#text(style: "italic", fill: muted)[수치와 규정은 반드시 원본으로\ 대조 검증]],

    alt-cell[#text(fill: navy, weight: "bold")[맥락의 축적]],
    alt-cell[연속 대화의 가치가\ 단발 질문의 합보다 크다],
    alt-cell[#text(style: "italic", fill: muted)[메모리, 프로젝트 분리, 중간 요약 활용]],

    body-cell[#text(fill: navy, weight: "bold")[변화의 여지]],
    body-cell[좋은 질문은 질문자 자신이\ 변할 준비가 된 질문],
    body-cell[#text(style: "italic", fill: muted)["내 생각이 틀렸을 수도 있다"는\ 전제로 시작]],
  )
}

#set par(first-line-indent: 1.2em)

#pagebreak()
#ornament()

// ═══════════════════════════════════════════════
// APPENDIX
// ═══════════════════════════════════════════════

#v(0.0cm)
#text[ ] <appendix>
#align(center)[
  #text(size: 8pt, fill: gold, tracking: 0.4em)[#upper[Appendix]]
  #v(0.15cm)
  #text(size: 16pt, fill: navy, weight: "bold")[부록: 상황별 질문 템플릿]
  #v(0.15cm)
  #text(size: 9pt, fill: muted, style: "italic")[[ ] 안의 내용을 자신의 상황에 맞게 교체하여 사용한다.]
  #v(0.3cm)
  #line(length: 30%, stroke: 0.5pt + gold)
]
#v(0.4cm)

#template-box(
  [숨은 전제 발굴],
  [[현재 진행 중인 프로젝트/전략]이 성공하려면 반드시 참이어야 하는 가정 세 가지는 뭐야? 그리고 각 가정이 깨질 확률은?],
)

#template-box(
  [반론 시뮬레이션],
  [내가 [의사결정/제안]을 하려고 해. 이것에 반대하는 가장 합리적인 사람의 논리를 구성해줘.],
)

#template-box(
  [극단 시나리오],
  [[현재 시스템/전략]이 완벽하게 실패하는 시나리오를 만들어봐. 어떤 조건이 겹쳐야 그렇게 되지?],
)

#template-box(
  [유추 전이],
  [[현재 문제]와 구조적으로 비슷한 다른 산업의 사례가 있을까? 그들은 어떻게 해결했어?],
)

#template-box(
  [시간축 전환],
  [지금으로부터 [N]년 뒤에 돌아보면, 현재 [의사결정]에서 가장 아쉬울 점은 뭘까?],
)

#template-box(
  [본질 추출],
  [지금까지 논의한 내용을 의사결정권자에게 한 문장으로 전달해야 한다면?],
)

#template-box(
  [측정 프레임 검토],
  [우리가 [이 지표]로 성과를 측정하고 있는데, 이 지표가 실제 목표와 괴리될 수 있는 상황은?],
)

#pagebreak()
#ornament()

// ═══════════════════════════════════════════════
// CLOSING
// ═══════════════════════════════════════════════

#v(0.0cm)
#align(center)[
  #text(size: 8pt, fill: gold, tracking: 0.4em)[#upper[Epilogue]]
  #v(0.15cm)
  #text(size: 14pt, fill: navy, weight: "bold")[맺음말]
  #v(0.15cm)
  #text(size: 10pt, fill: burgundy, style: "italic")[질문하는 실무자의 역할]
  #v(0.3cm)
  #line(length: 30%, stroke: 0.5pt + gold)
]
#v(0.5cm)

AI가 아무리 발전해도 대체할 수 없는 것이 있다. #text(fill: navy, weight: "bold")["무엇이 중요한 문제인가"]를 정의하는 것, #text(fill: navy, weight: "bold")["이 답이 우리 현장에서 통하는가"]를 판단하는 것, 그리고 #text(fill: navy, weight: "bold")[최종 의사결정에 대한 책임]을 지는 것이다.

좋은 질문은 기법이 아니라 태도다. 가장 생산적인 AI 협업은 "내가 이미 아는 것을 정리받는 대화"가 아니라 "내가 몰랐던 것을 발견하는 대화"다.\

그리고 그 발견은 "나는 이 문제에 대해 아직 충분히 모른다"는 솔직한 인식에서 시작된다.

도구는 날마다 발전한다. 하지만 좋은 질문을 던지는 능력은 도구가 대신해주지 않는다. 그 능력이야말로, AI 시대에 실무자가 갖출 수 있는 가장 강력한 경쟁력이다.

