// ─────────────────────────────────────────────
// 개발 스토리: AI 에이전트 팀으로 차세대 추천 시스템을 만들기까지
// Typst Web App Compatible
// ─────────────────────────────────────────────

#let navy = rgb("#1B2A4A")
#let burgundy = rgb("#6B2D3E")
#let gold = rgb("#B8860B")
#let cream = rgb("#FAF6F0")
#let dark-cream = rgb("#F0EBE1")
#let ink = rgb("#2C2C2C")
#let muted = rgb("#5A5A5A")
#let light-rule = rgb("#C4B8A8")

// ── Page setup ──
#set page(
  paper: "a4",
  margin: (top: 2.5cm, bottom: 2.5cm, left: 2.5cm, right: 2.5cm),
  fill: cream,
  header: context {
    if counter(page).get().first() > 1 [
      #set text(size: 7.5pt, font: "New Computer Modern", fill: muted, tracking: 0.12em)
      #smallcaps[개발 스토리]
      #h(1fr)
      #smallcaps[AI 에이전트 팀으로 차세대 추천 시스템을 만들기까지]
      #v(4pt)
      #line(length: 100%, stroke: 0.4pt + light-rule)
    ]
  },
  footer: context {
    let pg = counter(page).get().first()
    if pg > 1 [
      #line(length: 100%, stroke: 0.3pt + light-rule)
      #v(4pt)
      #set text(size: 8pt, font: "New Computer Modern", fill: muted)
      #h(1fr)
      — #pg —
      #h(1fr)
    ]
  },
)

// ── Base text ──
#set text(
  font: "New Computer Modern",
  size: 10pt,
  fill: ink,
  lang: "ko",
)

#set par(
  justify: true,
  leading: 0.78em,
  first-line-indent: 1.2em,
)

// ── Heading styles ──
#show heading.where(level: 1): it => {
  v(0.4cm)
  set par(first-line-indent: 0pt)
  align(center)[
    #block(width: 100%)[
      #v(6pt)
      #line(length: 40%, stroke: 0.8pt + gold)
      #v(8pt)
      #text(size: 18pt, fill: navy, weight: "bold")[#it.body]
      #v(8pt)
      #line(length: 40%, stroke: 0.8pt + gold)
      #v(6pt)
    ]
  ]
  v(0.4cm)
}

#show heading.where(level: 2): it => {
  v(0.3cm)
  set par(first-line-indent: 0pt)
  block[
    #text(size: 13pt, fill: navy, weight: "bold")[#it.body]
    #v(0pt)
    #line(length: 100%, stroke: (paint: light-rule, thickness: 0.4pt, dash: "loosely-dotted"))
  ]
  v(0.15cm)
}

#show heading.where(level: 3): it => {
  v(0.15cm)
  set par(first-line-indent: 0pt)
  block[
    #text(size: 11pt, fill: burgundy, weight: "bold", style: "italic")[#it.body]
  ]
  v(0.1cm)
}

// ── Custom components ──
#let ornament() = {
  v(0.3cm)
  align(center)[
    #text(size: 11pt, fill: gold)[✦ #h(6pt) ◆ #h(6pt) ✦]
  ]
  v(0.3cm)
}

#let info-box(title, body) = {
  set par(first-line-indent: 0pt)
  v(0.1cm)
  block(
    width: 100%,
    stroke: (left: 2.5pt + burgundy, rest: 0.3pt + light-rule),
    radius: (right: 3pt),
    fill: dark-cream,
    inset: (left: 14pt, right: 14pt, top: 10pt, bottom: 10pt),
  )[
    #text(size: 11pt, fill: navy, weight: "bold")[#title]
    #v(4pt)
    #text(size: 10pt, fill: ink)[#body]
  ]
  v(0.1cm)
}

#let quote-box(body) = {
  set par(first-line-indent: 0pt)
  v(0.15cm)
  block(
    width: 100%,
    inset: (left: 2cm, right: 2cm, top: 0.3cm, bottom: 0.3cm),
  )[
    #align(center)[
      #text(size: 11pt, fill: navy, style: "italic")[#body]
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
  #line(length: 50%, stroke: 0.6pt + gold)
  #v(0.6cm)

  #text(
    size: 10pt,
    fill: gold,
    tracking: 0.5em,
    weight: "regular",
  )[#upper[Development Story]]
  #v(0.4cm)

  #text(
    size: 26pt,
    fill: navy,
    weight: "bold",
  )[AI 에이전트 팀으로]
  #v(0.1cm)
  #text(
    size: 26pt,
    fill: navy,
    weight: "bold",
  )[차세대 추천 시스템을 만들기까지]
  #v(0.3cm)

  #line(length: 20%, stroke: 0.5pt + light-rule)
  #v(0.2cm)

  #text(
    size: 13pt,
    fill: burgundy,
    style: "italic",
  )[소비자 GPU 1대, 3인 팀, 그리고 AI 협업의 기록]

  #v(1cm)
  #line(length: 50%, stroke: 0.6pt + gold)
]

#v(2cm)

#align(center)[
  #block(
    width: 70%,
    inset: (x: 1.5cm, y: 1cm),
  )[
    #set par(first-line-indent: 0pt)
    #text(size: 10pt, fill: ink, style: "italic")[
      "AI가 코드를 쓰지만,\ 설계 판단은 사람이 한다."
    ]
    #v(0.5cm)
    #text(size: 9pt, fill: muted)[
      인프라 예산 없이, 소비자 GPU 1대와 AI 에이전트들의 조합으로\
      18-task, 7-expert PLE+adaTT 추천 시스템을 구축한 과정을 기록한다.
    ]
  ]
]

#v(1fr)

#align(center)[
  #text(size: 10pt, fill: muted, tracking: 0.15em)[AIOps PLE Platform]
  #v(0.3cm)
  #line(length: 30%, stroke: 0.3pt + light-rule)
]

#pagebreak()


// ═══════════════════════════════════════════════
// CONTENT
// ═══════════════════════════════════════════════

#set page(
  header: context {
    if counter(page).get().first() > 1 [
      #set text(size: 7.5pt, font: "New Computer Modern", fill: muted, tracking: 0.12em)
      #smallcaps[개발 스토리]
      #h(1fr)
      #smallcaps[AI 에이전트 팀으로 차세대 추천 시스템을 만들기까지]
      #v(4pt)
      #line(length: 100%, stroke: 0.4pt + light-rule)
    ]
  },
  footer: context {
    let pg = counter(page).get().first()
    if pg > 1 [
      #line(length: 100%, stroke: 0.3pt + light-rule)
      #v(4pt)
      #set text(size: 8pt, font: "New Computer Modern", fill: muted)
      #h(1fr)
      — #pg —
      #h(1fr)
    ]
  },
)


= 프로젝트 배경

== 팀 구성과 제약조건

프로젝트 팀은 3명이었다. 데이터사이언티스트 겸 PM 1명과 엔지니어 2명. 전용 인프라 예산은 없었고, 개발에 사용할 수 있는 GPU는 로컬 PC에 장착된 소비자용 RTX 4070 (12GB VRAM) 1대가 전부였다.

#info-box(
  [제약조건 요약],
  [
    • *팀 규모*: 3명 (PM/데이터사이언티스트 1 + 엔지니어 2) \
    • *인프라*: 전용 GPU 서버 없음. RTX 4070 12GB 1대 (로컬) \
    • *예산*: 인프라 구매 예산 없음. AWS SageMaker spot 인스턴스 활용 \
    • *일정*: 기존 ALS 기반 추천 시스템을 대체할 차세대 시스템 구축
  ],
)

== 프로젝트 목표

기존 금융 상품 추천 시스템은 ALS (Alternating Least Squares) 기반의 협업 필터링이었다. 이를 PLE (Progressive Layered Extraction) + adaTT (Adaptive Task Transfer) 아키텍처 기반의 멀티태스크 딥러닝 추천 시스템으로 교체하는 것이 목표였다. 18개 태스크를 7개 전문가 네트워크가 처리하며, 태스크 간 관계를 명시적으로 모델링하는 구조다.

#ornament()


= AI 에이전트 조직화

== Phase별 AI 도구 운용

프로젝트의 각 단계에서 서로 다른 AI 도구를 전략적으로 배치했다. 각 도구의 강점을 살려 역할을 분담하는 것이 핵심이었다.

=== Phase A: 아이디에이션 (Gemini)

초기 컨셉 탐색과 브레인스토밍에는 Gemini를 활용했다. 광범위한 지식 베이스를 바탕으로 아키텍처 후보군을 빠르게 스캔하고, 다양한 접근법의 장단점을 비교하는 데 효과적이었다.

=== Phase B: 기술 검증 (Claude Opus)

아이디에이션 결과를 구체적인 아키텍처로 발전시키는 단계에서는 Claude Opus를 사용했다. 수학적 엄밀성이 필요한 loss function 설계, 데이터 리키지 검증, 정규화 파이프라인 설계 등 기술적 깊이가 요구되는 작업에 집중했다.

=== Phase C: 코드 환경 정비 (Cursor)

GitHub 기반의 코드 환경 구성, 프로젝트 구조 설계, 초기 보일러플레이트 생성은 Cursor로 수행했다. IDE 통합 환경에서의 빠른 코드 네비게이션과 리팩토링이 강점이었다.

=== Phase D: 병렬 구현 (Claude Code — Opus/Sonnet)

본격적인 구현 단계에서는 각 팀원이 AI 에이전트의 "팀장" 역할을 맡았다. Claude Code 환경에서 Opus와 Sonnet을 병렬로 운용하여 서로 다른 모듈을 동시에 구현했다.

#v(0.3cm)

#set par(first-line-indent: 0pt)
#{
  let header-cell(body) = table.cell(
    fill: navy,
    inset: (x: 10pt, y: 7pt),
  )[#align(center)[#text(size: 10pt, fill: cream, weight: "bold")[#body]]]

  let body-cell(body) = table.cell(
    inset: (x: 10pt, y: 7pt),
  )[#text(size: 9pt, fill: ink)[#body]]

  let alt-cell(body) = table.cell(
    fill: dark-cream,
    inset: (x: 10pt, y: 7pt),
  )[#text(size: 9pt, fill: ink)[#body]]

  table(
    columns: (0.8fr, 1.2fr, 1.5fr),
    stroke: 0.4pt + light-rule,
    align: left + horizon,
    header-cell[Phase], header-cell[AI 도구], header-cell[역할],
    body-cell[A. 아이디에이션], body-cell[Gemini], body-cell[컨셉 탐색, 아키텍처 후보 스캔, 브레인스토밍],
    alt-cell[B. 기술 검증], alt-cell[Claude Opus], alt-cell[수학적 검증, loss 설계, 리키지 분석, 아키텍처 구체화],
    body-cell[C. 환경 정비], body-cell[Cursor], body-cell[GitHub 구조, 보일러플레이트, IDE 기반 리팩토링],
    alt-cell[D. 병렬 구현], alt-cell[Claude Code (Opus/Sonnet)], alt-cell[모듈별 병렬 코딩, 테스트, 디버깅],
  )
}
#set par(first-line-indent: 1.2em)

#ornament()


= 품질 관리 전략

== CLAUDE.md 가드레일

AI 에이전트가 생성하는 코드의 품질을 보장하기 위해, 프로젝트 루트에 CLAUDE.md 파일을 두어 가드레일을 설정했다. 이 파일은 AI 에이전트가 매 세션 시작 시 읽어들이는 시스템 지침이다.

#info-box(
  [CLAUDE.md 핵심 가드레일],
  [
    • *Config-Driven 원칙*: 모든 파라미터는 YAML에서 읽는다. 하드코딩 금지. \
    • *관심사 분리*: Adapter는 데이터 변환만, Runner는 파이프라인만, train.py는 학습만. \
    • *데이터 리키지 방지*: Scaler는 train split에서만 fit. temporal split에 gap_days 필수. \
    • *코드 검수 4단계*: 컴파일 검증 → 인터페이스 계약 검증 → 하드코딩 스캔 → 관심사 분리 검증.
  ],
)

== 메모리 뱅크

AI 에이전트는 대화 세션이 끝나면 맥락을 잃는다. 이 문제를 해결하기 위해 "메모리 뱅크" 시스템을 도입했다. 세션 진행 상황, 설계 결정 사유, 피드백 이력을 구조화된 마크다운 파일로 관리하여, 새 세션이 시작될 때 AI가 이전 맥락을 빠르게 복원할 수 있게 했다.

== 인터페이스 계약 검증

병렬 에이전트가 서로 다른 모듈을 동시에 수정할 때 가장 큰 위험은 인터페이스 불일치다. 파일 A가 저장하는 키 이름과 파일 B가 읽는 키 이름이 달라지는 문제를 방지하기 위해, 병렬 작업 후에는 반드시 "인터페이스 계약 검증" 단계를 수행했다. cross-file 키/필드 매핑 테이블을 자동 생성하여 불일치를 사전에 탐지했다.

#ornament()


= 기술적 도전과 해결

== Label Leakage 3건 발견 및 수정

모델 학습 초기에 비정상적으로 높은 성능이 관측되었다. 원인 분석 결과, 3건의 label leakage를 발견했다.

#set par(first-line-indent: 0pt)
#block(
  width: 100%,
  stroke: (left: 2pt + burgundy),
  inset: (left: 12pt, y: 6pt),
)[
  #set text(size: 10pt)
  #strong[Leak 1]: 시퀀스 데이터의 마지막 timestep이 레이블 기간과 겹침 \
  #strong[Leak 2]: 미래 시점의 집계 피처가 입력에 포함 \
  #strong[Leak 3]: temporal split에서 gap_days 미설정으로 인접 기간 정보 유출
]
#set par(first-line-indent: 1.2em)

#v(0.15cm)
LeakageValidator를 학습 전 필수 단계로 추가하고, CLAUDE.md 가드레일에 검증 규칙을 명시하여 재발을 방지했다.

== Phase 2 NaN 문제 (FP16 Underflow)

Mixed precision (AMP) 학습 중 Phase 2에서 NaN loss가 발생했다. 원인은 BFloat16 환경에서의 underflow였다. 작은 gradient 값들이 FP16 범위 밖으로 떨어지면서 NaN으로 전파되었다.

#info-box(
  [해결 방법],
  [
    • BFloat16에서 `.float()` 변환 후 `.numpy()` 호출하는 패턴 적용 \
    • subprocess pipe deadlock 문제 동시 수정 \
    • GradScaler 설정 최적화로 underflow 빈도 감소
  ],
)

== GPU 활용률 최적화 (37% → 98%)

초기 학습 시 GPU 활용률이 37%에 불과했다. 병목은 데이터 로딩이었다.

#set par(first-line-indent: 0pt)
#block(
  width: 100%,
  fill: dark-cream,
  stroke: 0.3pt + light-rule,
  radius: 3pt,
  inset: (x: 12pt, y: 10pt),
)[
  #text(size: 10pt, fill: navy, weight: "bold")[최적화 단계]
  #v(4pt)
  #text(size: 10pt, fill: ink)[
    • *batch_size 증가*: 512 → 4096 (941K 데이터에 최적화) \
    • *DataLoader 최적화*: num_workers, pin_memory, prefetch_factor 조정 \
    • *데이터 전처리 분리*: Phase 0에서 텐서를 미리 저장, 학습 시 로드만 수행 \
    • *결과*: GPU 활용률 37% → 98%, 학습 시간 약 3배 단축
  ]
]
#set par(first-line-indent: 1.2em)

== pandas에서 DuckDB/cuDF로 전환

941K 행의 데이터를 pandas로 처리하면 메모리 사용량이 급격히 증가했다. 대규모 데이터 처리 백엔드를 DuckDB (CPU columnar)와 cuDF (GPU)로 전환하여 메모리 효율성과 처리 속도를 동시에 개선했다.

== Docker GPU Passthrough 불안정

Windows 환경에서 Docker를 통한 GPU passthrough가 불안정하게 동작했다. CUDA 버전 불일치, 드라이버 호환성 문제가 반복적으로 발생했다. 결국 Docker를 포기하고 로컬 Python 환경에서 직접 개발하는 방식으로 전환했다.

== torch CPU/CUDA 버전 충돌

conda 환경에서 torch의 CPU 빌드와 CUDA 빌드가 충돌하는 문제가 발생했다. 패키지 의존성 꼬임으로 인해 CUDA가 인식되지 않는 상황이 반복되었다. conda 캐시를 완전히 정리하고 CUDA 버전을 명시적으로 지정하여 환경을 재구성했다.

#ornament()


= 핵심 교훈

== "AI가 코드를 쓰지만, 설계 판단은 사람이 한다"

AI 에이전트는 놀라운 속도로 코드를 생성하지만, 아키텍처 결정, 데이터 리키지 판단, 비용 최적화 전략은 사람의 도메인 지식과 경험에 의존한다. AI는 "어떻게(how)"에 강하지만, "왜(why)"와 "무엇을(what)"은 사람이 정의해야 한다.

#quote-box["가장 위험한 순간은 AI가 '그럴듯한 코드'를 만들어낸 직후다.\ 그때 비판적 검토를 멈추면 기술 부채가 쌓인다."]

== "가드레일 없는 AI 코딩은 기술 부채를 만든다"

CLAUDE.md 없이 AI에게 자유롭게 코딩을 시키면, 하드코딩이 늘어나고 관심사가 뒤섞이며 테스트 불가능한 구조가 만들어진다. 가드레일은 AI의 생산성을 제한하는 것이 아니라, 올바른 방향으로 유도하는 것이다.

== "이종 전문가 철학이 개발 방법론에도 적용된다"

PLE 아키텍처의 핵심 철학인 "이종 전문가(Mixture of Experts)"가 개발 방법론 자체에도 적용되었다. Gemini는 넓은 탐색에, Opus는 깊은 분석에, Cursor는 빠른 환경 구성에, Claude Code는 구현에 각각 전문화되었다. 하나의 AI 도구가 모든 것을 하는 것보다, 각 도구의 강점에 맞는 역할을 배분하는 것이 더 효과적이었다.

#ornament()


= 성과

== 시스템 구축

#set par(first-line-indent: 0pt)
#block(
  width: 100%,
  fill: dark-cream,
  stroke: 0.3pt + light-rule,
  radius: 3pt,
  inset: (x: 14pt, y: 12pt),
)[
  #grid(
    columns: (1fr, 1fr),
    gutter: 12pt,
    [
      #text(size: 11pt, fill: navy, weight: "bold")[추천 시스템]
      #v(4pt)
      #text(size: 10pt, fill: ink)[
        • 18-task 멀티태스크 학습 \
        • 7-expert PLE 네트워크 \
        • adaTT 태스크 간 적응적 전이 \
        • Uncertainty weighting (Kendall et al.) \
        • 로짓 전이 3가지 방식
      ]
    ],
    [
      #text(size: 11pt, fill: navy, weight: "bold")[인프라 및 실험]
      #v(4pt)
      #text(size: 10pt, fill: ink)[
        • 54개 ablation 시나리오 \
        • SageMaker spot 인스턴스 활용 \
        • Phase 0 (CPU) + Phase 1\~2 (GPU) 분리 \
        • Config-driven 파이프라인 아키텍처
      ]
    ],
  )
]
#set par(first-line-indent: 1.2em)

== 문서화

프로젝트를 통해 생산된 기술 문서는 총 9편이다. 아키텍처 개요, 파이프라인 가이드, 전문가 상세, 피처 참조, PLE+adaTT 참조, Causal OT 참조, 증류 참조, 시간 참조, 규제 프레임워크가 모두 Typst 기반으로 작성되었다.

== 논문

연구 결과를 정리하여 arXiv에 제출할 논문 2편을 준비했다. 제한된 자원 환경에서의 대규모 멀티태스크 추천 시스템 구축 경험과, AI 에이전트를 활용한 소규모 팀의 개발 방법론을 다룬다.

#v(1cm)
#ornament()

#align(center)[
  #text(size: 9pt, fill: muted, style: "italic")[
    이 프로젝트는 "자원의 부족"이 아니라 "자원의 재정의"를 통해 완성되었다.\
    소비자 GPU 1대와 AI 에이전트들의 조합이 전용 인프라를 대체할 수 있음을 보여준 사례다.
  ]
]
