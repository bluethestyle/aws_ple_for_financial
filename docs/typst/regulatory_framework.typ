// ─────────────────────────────────────────────────────────
//  금융 AI 추천 시스템 — 규제 준수 프레임워크
//  AWS PLE for Financial · 2026. 04.
// ─────────────────────────────────────────────────────────

// ── 컬러 팔레트 ──
#let navy     = rgb("#003478")
#let red-acc  = rgb("#E8490F")
#let blue     = rgb("#1565C0")
#let teal     = rgb("#00897B")
#let gray-bg  = rgb("#F4F6F8")
#let gray-ln  = rgb("#DEE2E6")
#let txt      = rgb("#212529")
#let txt-sub  = rgb("#6C757D")

// ── 페이지 & 글꼴 ──
#set page(
  paper: "a4",
  margin: (top: 2.5cm, bottom: 2cm, left: 2.2cm, right: 2.2cm),
  header: context {
    if counter(page).get().first() > 1 {
      set text(size: 8pt, fill: txt-sub)
      grid(
        columns: (1fr, auto),
        [금융 AI 추천 시스템 — 규제 준수 프레임워크],
        [#counter(page).display()],
      )
      v(2pt)
      line(length: 100%, stroke: 0.4pt + gray-ln)
    }
  },
  footer: none,
)

#set text(font: ("Noto Sans KR", "Noto Sans"), size: 10pt, fill: txt)
#set par(justify: true, leading: 0.72em, spacing: 0.85em)
#set heading(numbering: none)

// ── heading 스타일 ──
#show heading.where(level: 1): it => {
  v(1.2em)
  block(
    width: 100%,
    below: 0.6em,
  )[
    #text(size: 17pt, weight: "bold", fill: navy)[#it.body]
    #v(-10pt)
    #line(length: 100%, stroke: 2pt + navy)
  ]
}

#show heading.where(level: 2): it => {
  v(0.8em)
  text(size: 12.5pt, weight: "bold", fill: blue)[#it.body]
  v(0.4em)
}

#show heading.where(level: 3): it => {
  v(0.5em)
  text(size: 10.5pt, weight: "bold", fill: txt)[#it.body]
  v(0.3em)
}

// ── 테이블 스타일 ──
#set table(
  inset: 8pt,
  stroke: 0.5pt + gray-ln,
  fill: (_, y) => if y == 0 { navy } else { none },
)
#show table.cell: it => {
  if it.y == 0 {
    set text(fill: white, weight: "bold", size: 9pt)
    it
  } else {
    set text(size: 9pt)
    it
  }
}

// ── 유틸리티 컴포넌트 ──
#let card(title: none, accent: navy, body) = {
  block(
    fill: gray-bg,
    stroke: (left: 3pt + accent, rest: 0.5pt + gray-ln),
    inset: 12pt,
    radius: 4pt,
    width: 100%,
    breakable: true,
  )[
    #if title != none [
      #text(fill: accent, weight: "bold", size: 11pt)[#title]
      #v(4pt)
    ]
    #body
  ]
}

#let tag(label, color: navy) = {
  box(
    fill: color.lighten(85%),
    stroke: 0.5pt + color.lighten(40%),
    inset: (x: 6pt, y: 3pt),
    radius: 3pt,
  )[#text(fill: color, weight: "bold", size: 8pt)[#label]]
}

#let divider() = {
  v(0.6em)
  line(length: 100%, stroke: (paint: gray-ln, thickness: 0.5pt, dash: "dashed"))
  v(0.6em)
}

// ═══════════════════════════════════════════════════════════
//  표지
// ═══════════════════════════════════════════════════════════

#page(header: none, footer: none)[
  #v(3cm)

  #align(center)[
    #block(
      width: 85%,
    )[
      #align(left)[
        #text(size: 11pt, fill: txt-sub)[AWS PLE for Financial]
        #v(0.6em)
        #line(length: 60pt, stroke: 3pt + red-acc)
        #v(0.8em)
        #text(size: 26pt, weight: "bold", fill: navy)[
          금융 AI 규제 준수 프레임워크
        ]
        #v(0.5em)
        #text(size: 14pt, fill: blue)[
          인공지능기본법 · EU AI Act · 금감원 AI RMF 대응#linebreak()컴플라이언스 아키텍처 설계 문서
        ]
        #v(1.5em)
        #line(length: 100%, stroke: 0.5pt + gray-ln)
        #v(1em)
        #grid(
          columns: (auto, 1fr),
          gutter: 8pt,
          text(fill: txt-sub, size: 9.5pt)[문서 분류],
          text(size: 9.5pt, weight: "bold")[기술 문서],
          text(fill: txt-sub, size: 9.5pt)[작성일],
          text(size: 9.5pt, weight: "bold")[2026년 4월],
          text(fill: txt-sub, size: 9.5pt)[버전],
          text(size: 9.5pt, weight: "bold")[v1.0],
          text(fill: txt-sub, size: 9.5pt)[관련 법령],
          text(size: 9.5pt, weight: "bold")[인공지능기본법 (법률 제20676호), EU AI Act, GDPR],
        )
      ]
    ]
  ]

  #v(1fr)

  #align(center)[
    #block(
      fill: navy,
      width: 85%,
      inset: 16pt,
      radius: 4pt,
    )[
      #text(fill: white, size: 9.5pt)[
        *핵심 요약* — 본 문서는 금융 AI 추천 시스템(PLE 기반)이 준수해야 하는 국내외 규제 요건을 체계적으로 매핑하고, 각 요건에 대한 시스템 레벨 대응 아키텍처를 정의합니다. 한국 인공지능기본법(2026.1.22 시행), 금융위 통합 AI 가이드라인 7대 원칙, 금감원 AI RMF, EU AI Act(Art. 13/14/15), GDPR Art. 22를 대상으로 하며, 감사 추적 · 공정성 모니터링 · 드리프트 감시 · 쏠림 탐지 · 킬스위치 · 옵트아웃 · Human-in-the-Loop · 거버넌스 보고서 자동 생성까지 end-to-end 컴플라이언스 아키텍처를 포함합니다.
      ]
    ]
  ]

  #v(2cm)
]

// ═══════════════════════════════════════════════════════════
//  목차
// ═══════════════════════════════════════════════════════════

#outline(
  title: text(fill: navy, size: 14pt, weight: "bold")[목차],
  depth: 2,
  indent: 1.5em,
)

#pagebreak()

// ═══════════════════════════════════════════════════════════
//  1. 금융 AI 규제 환경 개요
// ═══════════════════════════════════════════════════════════

= 금융 AI 규제 환경 개요

== 한국 인공지능기본법

2024년 12월 국회를 통과하고, 2025년 1월 21일 공포된 *「인공지능 발전과 신뢰 기반 조성 등에 관한 기본법」*(법률 제20676호)은 *2026년 1월 22일* 시행되었습니다. 아시아·태평양 지역 최초의 포괄적 AI 입법으로, 위험 기반 접근법(Risk-Based Approach)을 채택합니다.

#card(title: "법률 핵심 구조", accent: navy)[
  #grid(
    columns: (1fr, 1fr),
    gutter: 10pt,
    [
      *투명성 의무* (제31조 / 시행령 제22조)\
      AI 사용 사실 사전 고지, AI 생성물 표시\
      위반 시 과태료 최대 3,000만 원

      *안전성 확보* (제32조 / 시행령 제23조)\
      학습 연산량 10#super[26] FLOPs 이상 초거대 모델 대상\
      (추천 모델은 해당 없음 -- *사용 영역* 기반 고영향 AI 분류 적용)
    ],
    [
      *고영향 AI 확인* (제33조 / 시행령 제24조)\
      사업자의 자체 사전 검토 의무\
      필요시 과기정통부에 확인 요청 (30일 이내 회신)

      *고영향 AI 사업자 책무* (제34조 / 시행령 제26조)\
      위험 관리 방안 수립, 이용자 보호(설명가능성),\
      관련 사항 홈페이지 등에 공개
    ],
  )
]

=== 고영향 AI와 금융

인공지능기본법은 *"사람의 생명·신체의 안전, 기본권에 중대한 영향을 미치거나 위험을 초래할 수 있는 인공지능"*을 고영향 AI로 정의합니다(제2조 제4호). 시행령에서 11개 분야를 열거하며, *금융(신용·대출)* 분야가 명시적으로 포함됩니다.

#table(
  columns: (auto, 1fr, auto),
  align: (center, left, center),
  [분야], [적용 예시], [금융 관련성],
  [의료], [의료기기·디지털 의료기기에 활용되는 AI], [간접],
  [*금융 (신용·대출)*], [*신용평가·대출심사 등 개인 권리관계 평가 AI*], [*직접*],
  [고용], [채용·인사 의사결정 AI], [없음],
  [공공서비스], [자격 확인, 행정 의사결정 AI], [간접],
  [원자력·생체정보], [원자력 시설, 생체인식 분석 AI], [간접],
)

#card(title: "금융분야 시사점", accent: red-acc)[
  *대출심사·신용평가*에 사용되는 AI는 법률상 명시적 고영향 AI에 해당합니다. 금융 AI 추천 시스템이 직접적인 여신 업무를 수행하지 않더라도, *상품 추천·적합성 판단·고객 분류* 등의 AI 활용이 향후 시행령 개정이나 금융위 가이드라인에 의해 고영향 AI로 확대 지정될 가능성이 존재합니다. 따라서 *고영향 AI에 준하는 수준*의 거버넌스 체계를 선제적으로 구축하는 것이 합리적 전략입니다.
]

== 금융위원회 통합 AI 가이드라인 7대 원칙

금융위원회는 인공지능기본법 시행에 맞추어 *「금융분야 AI 가이드라인 개정안」*을 발표했습니다. 업권별 자율규제·모범규준 형태로 *2026년 1분기 시행* 예정이며, 7대 원칙을 중심으로 구성됩니다.

#grid(
  columns: (1fr, 1fr),
  gutter: 8pt,
  card(title: none, accent: navy)[
    #text(fill: navy, weight: "bold")[1. 거버넌스]
    최고경영자 포함 경영진의 역할·책임 분담\
    AI 위험관리 조직을 기획·개발과 *독립 분리*
\
\
    #text(fill: navy, weight: "bold")[2. 합법성]
    AI기본법, 개인정보보호법, 신용정보법 준수
\
\
    #text(fill: navy, weight: "bold")[3. 보조수단성]
    AI는 의사결정 보조 수단,\
    외부 모델 활용 시에도 최종 책임은 금융회사
\
\
    #text(fill: navy, weight: "bold")[4. 신뢰성]
    모델 성능·편향성 관리, 설명가능성(XAI) 확보
  ],
  card(title: none, accent: blue)[
    #text(fill: blue, weight: "bold")[5. 금융안정성]
    제3자 의존성, 시장 동조화 리스크 평가\
    사고 시 금융당국 보고
\
\
    #text(fill: blue, weight: "bold")[6. 신의성실]
    #text(tracking: -0.05em)[*모든 대고객 AI 서비스* 사전고지 (AI기본법보다 넓음)]\
    이해상충 방지, 공정성 기준 설정·평가
\
\
    #text(fill: blue, weight: "bold")[7. 보안성]
    데이터 오염·모델 오염·프롬프트 인젝션 등\
    AI 특화 위협 대응 체계 구축
\
\
    #text(fill: txt-sub, size: 8.5pt)[출처: 금융위원회 보도자료 (2025.12)]
  ],
)

== 금감원 AI 위험관리 프레임워크 (AI RMF)

금융감독원은 2026년 1월 *「금융분야 AI 위험관리 프레임워크(AI RMF)」*를 도입했습니다. 현재 *118개 금융회사가 653개 AI 서비스*를 운영 중이나, 약 *85%가 AI 윤리원칙과 위험관리 기준 미비* 상태입니다.

#card(title: "AI RMF 3대 영역", accent: teal)[
  #grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 8pt,
    align: center,
    [
      *1. 거버넌스*\
      AI 의사결정기구·전담조직 설치\
      위험관리규정 수립\
      기획·개발 조직과 *독립 분리*
    ],
    [
      *2. 위험평가*\
      위험 인식·측정 → 경감 → 잔여위험\
      *고·중·저* 3단계 등급 분류\
      합법성·신뢰성·신의성실·보안성 정량 평가
    ],
    [
      *3. 위험통제*\
      출시 전 위험경감 조치 검증\
      운영단계별 모니터링 기준\
      위험 변경 시 등급 재평가
    ],
  )
]

== EU AI Act와의 비교

#text(size: 9pt)[
#table(
  columns: (0.5fr, 1fr, 1fr),
  align: (center, center, center),
  [구분], [한국 인공지능기본법], [EU AI Act],
  [시행일], [2026. 1. 22.], [고위험 AI: 2027. 12.],
  [분류 체계], [일반/고영향 (2단계)], [금지/고위험/제한/최소 (4단계)],
  [금융 적용], [신용평가·대출심사 명시], [신용평가·보험 산정 포함],
  [금지 AI], [없음], [사회점수 부여, 실시간 원격 생체인식 등],
  [의무 대상], [사업자 중심 단일 기준], [공급자·배포자·이용자 등 공급망 전체],
  [제재 수준], [과태료 최대 3,000만 원], [글로벌 매출 최대 7% 또는 3,500만 유로],
  [접근 방식], [진흥 중심 + 자율규제], [안전 중심 + 강제적 준수],
  [영향평가], [노력 의무], [사전 적합성 평가 법적 의무],
)
]

#card(title: "비교의 시사점", accent: red-acc)[
  한국의 AI 기본법은 EU 대비 제재 수준이 *수천 배 낮고* 자율규제 비중이 크지만, 이는 *시행 초기 연착륙 전략*입니다. 금융위 통합 가이드라인은 AI 기본법보다 적용 범위가 넓어 *모든 대고객 AI 서비스에 사전 고지*를 요구합니다. 향후 시행령 개정과 금융위 세부 규율을 통해 규제 강도가 점진적으로 높아질 것으로 예상되며, *선제적 대비*가 유리합니다.
]

== 글로벌 규제 흐름 요약

금융 AI에 대한 글로벌 규제는 세 가지 방향으로 수렴하고 있습니다.

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 8pt,
  card(title: "투명성·설명가능성", accent: navy)[
    AI 사용 고지 의무 확대\
    의사결정 근거 설명 요구\
    자동화 의사결정 거부권 보장
  ],
  card(title: "공정성·비차별", accent: blue)[
    보호 속성별 편향 정량 측정\
    시장 쏠림·이해상충 방지\
    정기적 공정성 감사 의무화
  ],
  card(title: "안전성·책임성", accent: teal)[
    위험 기반 분류(고위험 AI)\
    긴급 중단 메커니즘 의무\
    감사 추적 및 보존 의무
  ],
)

#pagebreak()

// ═══════════════════════════════════════════════════════════
//  2. 금감원 가이드라인 매핑
// ═══════════════════════════════════════════════════════════

= 금감원 가이드라인 매핑 — 7대 원칙과 시스템 대응

금융위 통합 AI 가이드라인 7대 원칙 및 금감원 AI RMF 요구항목에 대해, PLE 기반 금융 AI 추천 시스템의 대응 현황을 매핑합니다.

== 원칙별 대응 매트릭스

#text(size: 8.5pt)[
#set par(justify: false)
#table(
  columns: (0.8fr, 1.5fr, 0.3fr),
  align: (center, left, center),
  [7대 원칙], [시스템 대응], [수준],

  [1. 거버넌스\
  (AI RMF G-1\~G-6)],
  [3계층 거버넌스 프레임워크 설계 완료\
  (의사결정 위원회 → 운영팀 → 내부 감사)\
  월/분기 거버넌스 보고서 자동 생성\
  36항목 규제 준수 레지스트리 자동 점검],
  [○],

  [2. 합법성\
  (AI RMF R-2)],
  [적격성·적정성 자동 검증 (금소법 §17·18)\
  개인정보 비식별화 (SHA-256 + 도메인별 Salt)\
  신용정보법 보관 의무 준수 (5년)],
  [●],

  [3. 보조수단성\
  (AI RMF C-4)],
  [인간 감독 + Kill Switch 3단계 긴급 차단\
  인적 재처리 라우팅 (P1/P2/P3 SLA)\
  AI 자동화 의사결정 거부권 + 대체 경로],
  [●],

  [4. 신뢰성\
  (AI RMF R-3, C-2\~C-3)],
  [Champion-Challenger 자동 모델 경쟁\
  PSI 기반 드리프트 감지 + 3일 연속 재학습 트리거\
  IG 기반 피처별 기여도 + 자연어 추천 사유 생성],
  [●],

  [5. 금융안정성\
  (AI RMF R-3\~R-4)],
  [HHI·Gini·Entropy 쏠림 탐지\
  DI·SPD·EOD 공정성 3대 지표 자동 측정\
  보호 속성 5개 (연령·성별·지역·소득·생애주기)],
  [●],

  [6. 신의성실\
  (AI RMF R-4)],
  [AI 사용 고지 중앙 관리 + 세그먼트별 안내 분리\
  이해상충 방지 (고수수료 상품 40% 초과 시 자동 페널티)\
  옵트아웃 등록·철회·확인 전 생애주기 관리],
  [●],

  [7. 보안성\
  (AI RMF R-5)],
  [프롬프트 인젝션 방어 8패턴 (한국어 4 + 영어 4)\
  모델 무결성 SHA-256 해시 검증\
  HMAC + 해시 체인 감사 로그 불변성],
  [●],
)
]

#align(right)[
  #text(size: 8pt, fill: txt-sub)[
    ● 충족  ○ 부분 충족 (조직 의사결정 필요)
  ]
]

== AI RMF 영역별 상세 매핑

=== 영역 1: 거버넌스 (G-1 ~ G-6)

#text(size: 8.5pt)[
#set par(justify: false)
#table(
  columns: (0.2fr, 1.2fr, 1.6fr, 0.25fr),
  align: (center, left, left, center),
  [No.], [RMF 요구항목], [현재 대응 현황], [수준],
  [G-1], [AI 최고 의사결정기구 설치], [3계층 거버넌스 프레임워크 설계, 위원회 공식 설치 필요], [△],
  [G-2], [AI 위험관리 전담조직 (독립)], [운영팀 개발·운영 겸임, 독립 기능 분리 필요], [△],
  [G-3], [AI 위험관리규정 수립], [FD-TVS·드리프트 감지 등 로직 존재, 내규 문서화 필요], [○],
  [G-4], [AI 윤리원칙 수립·공표], [미수립 -- 전사 AI 윤리원칙 문서화 필요], [△],
  [G-5], [고위험 AI 사전 승인 절차], [서비스 출시 전 승인 프로세스 공식화 필요], [△],
  [G-6], [AI 활용 현황 정기 보고], [월간 리포트 자동 생성 완비, 보고 라인 공식화 필요], [○],
)
]

=== 영역 2: 위험평가 (R-1 ~ R-6)

#text(size: 8.5pt)[
#set par(justify: false)
#table(
  columns: (0.2fr, 1.2fr, 1.6fr, 0.25fr),
  align: (center, left, left, center),
  [No.], [RMF 요구항목], [현재 대응 현황], [수준],
  [R-1], [AI 서비스별 위험등급 분류], [고영향 AI 해당 가능성 자체 평가 완료, 공식 분류 체계 필요], [○],
  [R-2], [합법성 평가], [적격성 자동 검증 + 비식별화 + 보관 의무 준수], [●],
  [R-3], [신뢰성 평가], [Champion-Challenger + 드리프트 감지 + 공정성 자동 측정], [●],
  [R-4], [신의성실 평가], [보호 속성 5종 DI·SPD·EOD + 이해상충 방지 + Parquet 아카이빙], [●],
  [R-5], [보안성 평가], [로컬 AI + SHA-256 암호화 + 프롬프트 인젝션 방어 + 무결성 검증], [●],
  [R-6], [잔여위험 평가], [FD-TVS 리스크 페널티 자동 차단, Risk Appetite 문서화 필요], [○],
)
]

=== 영역 3: 위험통제 (C-1 ~ C-6)

#text(size: 8.5pt)[
#set par(justify: false)
#table(
  columns: (0.2fr, 1.2fr, 1.6fr, 0.25fr),
  align: (center, left, left, center),
  [No.], [RMF 요구항목], [현재 대응 현황], [수준],
  [C-1], [출시 전 위험경감 조치 검증], [적격성 규칙 + 피로도 필터링 + A/B 테스트 체계], [●],
  [C-2], [운영 모니터링 기준], [드리프트 자동 감지 + 성과 대시보드 + 이상 알림], [●],
  [C-3], [모델 성능 정기 평가·재학습], [주간/월간 자동 재학습 + Champion-Challenger + MLflow 버전 관리], [●],
  [C-4], [인간 감독 체계], [Kill Switch 3단계 + 인적 재처리 라우팅 + 옵트아웃], [●],
  [C-5], [긴급 중단 메커니즘], [GLOBAL/PER_TASK/PER_CLUSTER 3단계 킬스위치], [●],
  [C-6], [감사 추적 확보], [HMAC 해시 체인 + S3 Object Lock + 7개 감사 테이블], [●],
)
]

#pagebreak()

// ═══════════════════════════════════════════════════════════
//  3. EU AI Act 조항별 매핑
// ═══════════════════════════════════════════════════════════

= EU AI Act 조항별 매핑

EU AI Act는 금융 AI 추천 시스템을 *고위험 AI*(Annex III, 5(b): 신용평가·보험 산정)로 분류할 가능성이 높습니다. 주요 조항별 시스템 대응을 매핑합니다.

== Article 13: 투명성 및 정보 제공

#card(title: "Art. 13 요구사항과 시스템 대응", accent: navy)[
  #table(
    columns: (1fr, 1.5fr),
    align: (left, left),
    [Art. 13 요구사항], [시스템 대응],
    [고위험 AI 사용 사실 고지], [AI 사용 고지 중앙 관리 + 세그먼트별 안내 분리\
    모든 추천 출력에 AI 사용 표기 자동 포함],
    [시스템 작동 방식 설명], [3단계 추천 사유 생성 파이프라인\
    (Feature Reverse Mapper → LLM 그라운딩 → Self-Critique)],
    [입력 데이터 사양 공개], [피처 스키마 자동 문서화 (734D 피처 명세)\
    학습 데이터 출처·범위·가명처리 여부 기록],
    [성능 수준 공개], [모델 카드 자동 생성 (아키텍처·성능·피처 중요도)\
    태스크별 독립 AUC 추적 (18개 태스크)],
    [로그 생성 의무], [HMAC + 해시 체인 감사 로그 (S3 Object Lock WORM)\
    추천 이력 전수 Parquet 아카이빙],
  )
]

== Article 14: 인간 감독 (Human Oversight)

#card(title: "Art. 14 요구사항과 시스템 대응", accent: blue)[
  #table(
    columns: (1fr, 1.5fr),
    align: (left, left),
    [Art. 14 요구사항], [시스템 대응],
    [인간이 AI 시스템을 감독할 수 있는\ 인터페이스 제공], [오프라인: 직원 화면에 추천 리스트 + 사유 제공, 최종 권유는 직원 판단\
    온라인: AI 운영팀이 추천 방향·대상·제외 목록 직접 제어],
    [시스템 작동을 개입·중단할 수 있는\ 능력 보장], [Kill Switch 3단계 (GLOBAL/PER_TASK/PER_CLUSTER)\
    드리프트 감지 시 보수적 모드 자동 전환],
    [자동화 결정에 대한 거부·대안 경로], [옵트아웃 등록·철회·확인 + 인적 재처리 3단계 라우팅\
    (P1 긴급 1h / P2 4h / P3 24h SLA)],
  )
]

== Article 15: 정확성, 견고성, 사이버보안

#card(title: "Art. 15 요구사항과 시스템 대응", accent: teal)[
  #table(
    columns: (1fr, 1.5fr),
    align: (left, left),
    [Art. 15 요구사항], [시스템 대응],
    [적절한 정확성 수준 달성 및\ 유지], [Champion-Challenger 자동 모델 경쟁\
    PSI 기반 드리프트 감지 + 3일 연속 임계 초과 시 자동 재학습],
    [오류·결함에 대한 견고성], [Ablation 기반 컴포넌트별 기여도 정량 검증\
    증류 LGBM 폴백 (teacher 성능 이상 시 자동 전환)],
    [사이버보안 위협 대응], [프롬프트 인젝션 방어 8패턴 (심각도 5단계)\
    모델 무결성 SHA-256 검증\
    입력 검증 (포맷 + 크기 제한)],
    [편향 방지 조치], [DI·SPD·EOD 3대 공정성 지표 자동 측정\
    보호 속성 5종 상시 감시\
    임계값 위반 시 자동 인시던트 생성],
  )
]

== GDPR Article 22: 자동화된 의사결정

#card(title: "GDPR Art. 22 및 관련 조항 대응", accent: red-acc)[
  GDPR Art. 22는 정보주체에게 *"프로파일링을 포함한 자동화된 의사결정에만 기반한 결정"*에 대한 거부권을 부여합니다. 한국 개인정보보호법 §37조의2도 동일한 권리를 보장합니다.

  #table(
    columns: (1fr, 1.5fr),
    align: (left, left),
    [GDPR 요구사항], [시스템 대응],
    [자동화 의사결정 거부권\ (Art. 22(1))], [옵트아웃 등록·철회·확인 전 생애주기 관리\
    거부 시 즉시 인적 대체 경로 전환],
    [인적 개입 요청권\ (Art. 22(3))], [인적 재처리 라우팅 (7개 사유별 자동 상담원 전환)\
    P1/P2/P3 SLA 관리],
    [프로파일링 관련 정보 제공\ (Art. 13(2)(f))], [의사결정 기준 자동 문서화 (30개+ 필터 사유 한국어 설명)\
    피처 역매핑 + 태스크별 해석],
    [삭제권 (Art. 17)], [30일 PII 보존 정책 + 암호화 삭제\
    S3 Lifecycle 자동 적용],
    [정보주체 권리 행사 감사\ (Art. 30)], [7개 감사 테이블 중 profiling_audit, opt_out_audit, consent_audit\
    DynamoDB 서버리스 관리 + TTL 자동 정리],
  )
]

#pagebreak()

// ═══════════════════════════════════════════════════════════
//  4. 한국 AI 기본법 — 고위험 AI 분류
// ═══════════════════════════════════════════════════════════

= 한국 AI 기본법 고위험 AI 분류와 금융 적용

== 고영향 AI 해당 가능성 평가

현행법상 금융상품 추천 시스템은 *대출심사에 해당하지 않으므로* 명시적 고영향 AI 분류 대상은 아닙니다. 그러나 다음 시나리오에서 확대 지정 가능성이 존재합니다.

#table(
  columns: (1.5fr, 0.8fr, 1.5fr),
  align: (center, center, center),
  [시나리오], [가능성], [근거],
  [시행령 개정으로 금융상품 추천 전반 포함], [중간], [EU AI Act가 이미 이 범위 포함],
  [고객 분류·세그먼트가 권리관계 평가로 해석], [낮음~중간], [투자 적합성 판단과 유사 구조],
  [보험 상품 추천으로 업무 확대 시], [높음], [보험 산정은 EU·국내 모두 고위험],
  [이탈 예측 기반 차별적 혜택 제공], [중간], [공정성 이슈와 직결],
  [금융위 가이드라인에서 추천 AI 별도 규율], [중간~높음], [통합 가이드라인 시행 후 추이 관찰 필요],
)

#card(title: "대응 전략", accent: navy)[
  현시점에서 직접적 고영향 AI 해당은 아니나, *규제 환경의 방향성*을 고려하면 사실상 *고영향 AI에 준하는 수준*의 거버넌스 체계를 미리 구축하는 것이 합리적 전략입니다. 이는 규제 리스크 최소화뿐 아니라, 향후 업무 확대(보험·펀드 추천 등) 시 즉시 대응 가능한 기반이 됩니다.
]

== AI 기본법 주요 조항별 대응 현황

#text(size: 8.5pt)[
#set par(justify: false)
#table(
  columns: (1.0fr, 0.5fr, 1.7fr, 0.3fr),
  align: (center, center, left, center),
  [요구사항], [근거], [현재 대응], [수준],
  [AI 사용 사실의 사전 고지], [제31조], [AI 사용 고지 중앙 관리, 세그먼트별 안내 분리\
  모든 추천 출력에 자동 포함], [●],
  [AI 생성물의 표시], [제31조], [추천 사유 텍스트에 AI 생성 표기 자동 적용], [●],
  [위험 관리 조치 이행], [제32조], [FD-TVS + Kill Switch(3단계) + 적격성·적정성 검증\
  + 드리프트 감지], [●],
  [고영향 AI 해당 여부 확인], [제33조], [시나리오별 해당 가능성 분석 완료], [●],
  [위험 관리 방안·이용자 보호], [제34조], [안전신뢰문서 + 모델카드 자동 생성\
  추천 사유 3단계 파이프라인], [●],
  [영향평가 실시], [제35조], [시스템 문서화 완비, 정기 실시 프로세스 조직 수립 필요], [○],
  [자동화 의사결정 거부권], [개보법\ §37조의2], [옵트아웃 + 인적 재처리 3단계 라우팅], [●],
  [거버넌스 체계 구축], [7대 원칙], [3계층 프레임워크 설계, 위원회 공식 설치 필요], [○],
)
]

#align(right)[
  #text(size: 8pt, fill: txt-sub)[
    ● 충족  ○ 부분 충족  △ 미비
  ]
]

== 규제 타임라인

#card(title: "주요 규제 일정", accent: navy)[
  #text(size: 9pt)[
  #table(
    columns: (1fr, 2fr),
    align: (center, center),
    [일자], [이벤트],
    [2025. 01. 21], [AI 기본법 공포],
    [2025. 08], [AI 기본법 시행령 입법예고],
    [2025. 12. 22], [금융위 통합 AI 가이드라인(안) 공개],
    [2026. 01. 15], [금감원 AI RMF 도입],
    [*2026. 01. 22*], [*AI 기본법 시행*],
    [2026 Q1], [통합 가이드라인 및 AI RMF 확정·시행],
    [2027. 01 (예상)], [AI 기본법 과태료 실제 부과 시작],
    [2027. 12], [EU AI Act 고위험 AI 규정 완전 적용],
  )
  ]
]

#pagebreak()

// ═══════════════════════════════════════════════════════════
//  5. 컴플라이언스 아키텍처
// ═══════════════════════════════════════════════════════════

= 컴플라이언스 아키텍처

== 전체 구조

규제 준수 인프라를 3개 레이어로 구성합니다.

#card(title: "Layer 1: AWS 네이티브 (자동, 추가 비용 없음)", accent: navy)[
  - *CloudTrail* → 모든 AWS API 호출 자동 기록
  - *S3 Versioning* → 데이터/모델 변경 이력 보존
  - *S3 Object Lock* → 감사 로그 불변성 (WORM)
  - *KMS* → 암호화 키 관리 + 키 사용 감사
  - *IAM Access Log* → 리소스 접근 감사
]

#card(title: "Layer 2: 플랫폼 레벨 (반자동)", accent: blue)[
  - *ComplianceAuditStore* → 7개 감사 테이블 (DynamoDB)
  - *AuditLogger* → HMAC + 해시 체인 (S3 + DynamoDB)
  - *DataLineageTracker* → 피처→원천 추적 (S3 메타데이터)
  - *ExperimentTracker* → 실험 메트릭 (SageMaker Experiments)
  - *DriftDetector* → PSI 모니터링 (SageMaker Model Monitor)
]

#card(title: "Layer 3: 비즈니스/규제 레벨 (명시적)", accent: teal)[
  - *RegulatoryComplianceChecker* → 36항목 자동 점검
  - *FairnessMonitor* → DI/SPD/EOD 보호 속성 감시
  - *HerdingDetector* → 시스템 리스크 쏠림 탐지
  - *IncidentReporter* → 심각도별 자동 보고
  - *GovernanceReportGenerator* → 월/분기 거버넌스 보고서
  - *KillSwitch* → 긴급 모델 비활성화
]

== 감사 추적 (Audit Trail)

=== 불변 로그 — HMAC + 해시 체인

각 감사 로그 엔트리에 HMAC-SHA256 서명을 부여하고, 연속 엔트리 간 SHA256 해시 체인(prev_hash)으로 연결하여 위변조를 방지합니다.

#table(
  columns: (1fr, 1fr),
  align: (center, center),
  [On-Prem], [AWS],
  [로컬 JSONL + HMAC + 해시 체인], [S3 Object Lock (WORM) + HMAC + 해시 체인],
)

*AWS 강화 요소:*
- S3 Object Lock: AWS 레벨에서 물리적 삭제 차단 (관리자도 삭제 불가)
- KMS 암호화: 감사 로그 자체를 암호화
- CloudTrail: 감사 로그에 대한 접근도 감사 (메타 감사)

=== 7개 규제 감사 테이블

DynamoDB 기반 서버리스 관리 (자동 스케일링, 항목별 TTL):

#text(size: 9pt)[
#table(
  columns: (auto, 1fr, auto),
  align: (center, left, center),
  [테이블], [용도], [보존 기간],
  [ks_audit], [킬스위치 활성화/비활성화 이력], [7년],
  [consent_audit], [마케팅 동의 변경 이력 (부여/철회/갱신)], [7년],
  [profiling_audit], [정보주체 권리 행사 (열람/정정/삭제/제한/이동)], [7년],
  [opt_out_audit], [AI 결정 거부 이력], [7년],
  [incident_audit], [규제 인시던트 (심각도별)], [7년],
  [distillation_audit], [교사-학생 모델 성능 갭], [3년],
  [embedding_audit], [임베딩 품질 메트릭], [3년],
)
]

== 공정성 모니터링 (Fairness)

보호 속성 5개에 대해 3대 공정성 지표를 상시 감시합니다.

#grid(
  columns: (1fr, 1fr),
  gutter: 8pt,
  card(title: "보호 속성", accent: navy)[
    - *연령대*: youth / middle / pre_senior / senior
    - *성별*: M / F / unspecified
    - *지역 유형*: metropolitan / urban / rural
    - *소득 분위*: low / middle / high
    - *생애주기*: 6개 클래스
  ],
  card(title: "임계값 및 조치", accent: red-acc)[
    - *DI (Disparate Impact)*: 0.8 ~ 1.25
    - *SPD (Statistical Parity Difference)*: |SPD| ≤ 0.1
    - *EOD (Equal Opportunity Difference)*: |EOD| ≤ 0.1
    \
    DI \< 0.6 → CRITICAL 인시던트\
    DI \< 0.8 → MAJOR 인시던트\
    |SPD| > 0.1 → MINOR 인시던트
  ],
)

== 드리프트 감시 (Drift)

SageMaker Model Monitor (기본) + 커스텀 PSI (확장)를 결합합니다.

#table(
  columns: (1fr, 1fr, 1fr),
  align: (center, center, center),
  [지표], [임계값], [자동 조치],
  [PSI (warning)], [0.1], [경고 로깅],
  [PSI (critical)], [0.25], [담당자 알림],
  [연속 3일 critical], [--], [자동 재학습 트리거 (Step Functions)],
)

*피처별 개별 PSI를 추적*하여, 전체 분포가 안정적이더라도 특정 피처의 급변을 조기에 감지합니다.

== 쏠림 탐지 (Herding)

동일 상품에 추천이 과도하게 집중되는 *시스템 리스크*를 방지합니다.

#table(
  columns: (1fr, 1.5fr, 1fr),
  align: (center, left, center),
  [지표], [설명], [심각도 분류],
  [HHI], [허핀달-허쉬만 지수 -- 시장 집중도], [none → low → medium],
  [Gini 계수], [추천 불평등 측정], [→ high → critical],
  [Entropy], [추천 다양성 측정], [],
  [쏠림률], [최다 추천 상품의 비율], [critical 시 킬스위치 검토],
)

== 킬스위치 (Kill Switch)

3단계 긴급 모델 비활성화 체계입니다.

#card(title: "킬스위치 구조", accent: red-acc)[
  #grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 8pt,
    [
      *GLOBAL*\
      전체 모델 비활성화\
      DynamoDB 상태 저장\
      매 요청마다 확인
    ],
    [
      *PER_TASK*\
      특정 태스크만 비활성화\
      (예: click, purchase)\
      해당 태스크만 폴백
    ],
    [
      *PER_CLUSTER*\
      특정 고객군만 비활성화\
      (예: cluster_5)\
      해당 클러스터만 폴백
    ],
  )
  \
  *폴백 전략*: 규칙 기반 추천 / 이전 모델 롤백 / 추천 비활성화 중 선택
]

== 옵트아웃 (Opt-out) 및 동의 관리

#table(
  columns: (1fr, 1.5fr),
  align: (center, left),
  [기능], [설명],
  [AI 결정 거부 등록], [옵트아웃 등록 → 즉시 인적 대체 경로 전환],
  [거부 철회], [이전 옵트아웃 철회 → AI 추천 재활성화],
  [마케팅 동의 관리], [부여/철회/갱신 이력 추적, 채널별·목적별 옵트인 확인],
  [야간 SMS 차단], [마케팅 동의 여부 + 시간대 자동 필터링],
  [감사 추적], [모든 동의 변경·거부 이력을 consent_audit, opt_out_audit에 기록],
)

== 데이터 보존 정책

S3 Lifecycle Rule으로 자동 적용합니다.

#text(size: 9pt)[
#table(
  columns: (1fr, auto, 1fr, 1fr),
  align: (left, center, center, left),
  [카테고리], [보존 기간], [조치], [규제 근거],
  [Raw Data], [30일], [삭제], [GDPR 최소화 원칙],
  [Processed Features], [90일], [Glacier 아카이브], [재학습 지원],
  [Training Data], [365일], [Glacier 아카이브], [재현성 보장],
  [Model Checkpoints], [365일], [보관], [롤백 지원],
  [Inference Results], [90일], [삭제], [분쟁 대응],
  [Audit Logs], [2,555일 (7년)], [불변 보관 (WORM)], [금융 규제 7년 보존],
  [PII Data], [30일], [암호화 삭제], [GDPR §17 삭제권],
)
]

== 인시던트 관리

심각도별 자동 분류 및 대응 체계입니다.

#table(
  columns: (auto, auto, 1fr, 1fr),
  align: (center, center, left, center),
  [심각도], [대응 시간], [트리거], [보고 대상],
  [CRITICAL], [1시간], [킬스위치 발동, DI\<0.6, 보안 침해], [과기부/금감원/CISO],
  [MAJOR], [4시간], [DI\<0.8, 쏠림 critical, 모델 롤백], [금감원/AI 위원회],
  [MINOR], [24시간], [드리프트 경고, 품질 저하, 쏠림 high], [ML팀],
)

#pagebreak()

// ═══════════════════════════════════════════════════════════
//  6. Human-in-the-Loop 설계
// ═══════════════════════════════════════════════════════════

= Human-in-the-Loop 설계

금감원은 최종적으로 *사람의 개입에 따른 의사결정*을 요구합니다. EU AI Act Art. 14도 동일한 원칙을 명시합니다. 모든 것이 자동화되어 있더라도, 최종 검수는 사람이 수행합니다.

== 설계 원칙

#card(title: "AI는 도구이고, 최종 결정권은 사람에게", accent: navy)[
  PLE 기반 추천 시스템은 "사람을 대체"하는 것이 아니라 "사람의 의사결정을 지원"하는 도구입니다. 이 원칙은 시스템 아키텍처 전반에 내장되어 있습니다.
]

== 채널별 인간 감독 체계

#table(
  columns: (0.8fr, 1.5fr),
  align: (center, left),
  [채널], [인간 감독 방식],
  [오프라인\ (창구)], [AI가 직원 화면에 추천 리스트 + 사유를 *제공*\
  최종 권유 여부는 *직원이 판단*\
  추천 사유를 확인한 뒤 고객 상황에 맞게 권유],
  [온라인\ (앱/웹)], [AI가 추천 리스트를 자동 서빙\
  AI 운영팀이 추천 방향·대상·제외 목록을 *직접 제어*\
  긴급 시 킬스위치로 즉시 차단],
)

== 5대 인간 개입 포인트

=== 1. 추천 사유 샘플링 검수

전수 조사는 불가하더라도, 주기적으로 추천 사유 생성 결과를 샘플링하여 품질·적합성·규제 준수 여부를 검수합니다. 추천 품질 모니터링 시스템(L1 규칙 / L2a 리라이트 / L2b LLM 검증)이 자동 품질 지표를 산출하고, 이상 징후 발견 시 인간 검토를 트리거합니다.

=== 2. 모델 교체 승인

Champion-Challenger 비교 결과를 사람이 확인한 후 승인합니다. 자동 교체 시에도 교체 사유·성능 비교·공정성 지표를 포함한 리포트가 생성되며, 운영팀이 사후 검토합니다.

=== 3. 인시던트 에스컬레이션

자동 탐지된 이상 징후(공정성 위반, 드리프트, 쏠림)를 사람이 판단하여 조치합니다. CRITICAL 인시던트는 킬스위치 자동 발동 후 사람이 원인 분석 및 복구를 수행합니다.

=== 4. 공정성 리뷰

공정성 모니터링 결과를 주기적으로 사람이 리뷰합니다. DI·SPD·EOD 자동 측정 결과에 대해 도메인 전문가가 맥락을 고려한 해석을 제공하고, 필요시 임계값 조정이나 모델 재학습을 결정합니다.

=== 5. 옵트아웃·이의 제기 처리

고객의 AI 결정 거부 및 이의 제기에 대해 인적 재처리 경로가 자동으로 활성화됩니다. 7개 사유별 자동 상담원 전환이 이루어지며, P1(1h)/P2(4h)/P3(24h) SLA로 관리됩니다.

#card(title: "인적 재처리 사유 분류", accent: blue)[
  #grid(
    columns: (1fr, 1fr),
    gutter: 8pt,
    [
      *P1 (긴급 1시간)*\
      - 컴플라이언스 위반 탐지\
      - 적합성 미달 고객\
      - 킬스위치 발동 상황
    ],
    [
      *P2 (4시간) / P3 (24시간)*\
      - AI 결정 거부(옵트아웃)\
      - 저신뢰 추천 결과\
      - 설명 보충 요청\
      - 일반 이의 제기
    ],
  )
]

#pagebreak()

// ═══════════════════════════════════════════════════════════
//  7. 거버넌스 보고서 자동 생성
// ═══════════════════════════════════════════════════════════

= 거버넌스 보고서 자동 생성

== 보고서 체계

월/분기 단위로 자동 생성되어 S3에 저장되고, AI 거버넌스 위원회에 배포됩니다.

=== 9개 섹션 구성

#table(
  columns: (auto, 1fr, 1fr),
  align: (center, left, left),
  [섹션], [내용], [데이터 소스],
  [1], [공정성 요약 (DI/SPD/EOD by 보호 속성)], [FairnessMonitor],
  [2], [드리프트 요약 (기간 내 PSI 통계)], [DriftDetector],
  [3], [인시던트 요약 (CRITICAL/MAJOR/MINOR 건수 + 상세)], [IncidentReporter],
  [4], [모델 변경 이력 (학습/배포/롤백)], [ExperimentTracker],
  [5], [킬스위치 이력 (활성화/비활성화 횟수)], [ComplianceAuditStore],
  [6], [추천 품질 (L1/L2 사유 품질 지표)], [ReasonQualityMonitor],
  [7], [리스크 변동 (쏠림 추이, 리스크 수준)], [HerdingDetector],
  [8], [감사 스토어 요약 (7개 테이블 건수)], [ComplianceAuditStore],
  [9], [경영진 요약 (자동 생성 서술형)], [종합],
)

== 36항목 규제 준수 자동 점검

36항목 규제 준수 레지스트리를 통해 분기별 전체 점검이 자동 실행됩니다.

#grid(
  columns: (1fr, 1fr),
  gutter: 8pt,
  card(title: "A그룹 (구현 완료 18항목)", accent: teal)[
    #set text(size: 9pt)
    모델 카드 · 학습 데이터 문서화 · 안전신뢰문서\
    AI 공시 · 설명 가능성 · 옵트아웃 권리\
    인간 검토 · 킬스위치 · 모델 롤백\
    헬스 체크 · 공정성 모니터링 · 적합성 제약 등
  ],
  card(title: "GAP그룹 (갭 분석 18항목)", accent: red-acc)[
    #set text(size: 9pt)
    드리프트 자동 대응 · 추천 사유 품질 검증\
    SLA 추적 · EU AI Act 고위험 분류\
    PIA (개인정보 영향 평가) 등
  ],
)

#text(size: 9pt)[
*점검 주기:*
- 일간: 킬스위치 상태, 헬스 체크
- 주간: 공정성 지표, 설명 가능성
- 분기: 전체 36항목 종합 점검
]

== 거버넌스 3계층 체계

#block(breakable: false)[
#align(center)[
  #block(
    width: 90%,
    inset: 0pt,
  )[
    #block(
      fill: navy,
      width: 100%,
      inset: 12pt,
      radius: (top: 6pt),
    )[
      #text(fill: white, weight: "bold", size: 10pt)[1계층: 의사결정 -- AI 위험관리위원회]
      #v(0pt)
      #text(fill: white.darken(10%), size: 8.5pt)[
        AI 활용 정책 승인 · 고영향 AI 해당 여부 판단 · 연간 영향평가 승인 · 중대 사고 보고 체계
      ]
    ]
    #v(-2pt)
    #block(
      fill: blue,
      width: 100%,
      inset: 12pt,
    )[
      #text(fill: white, weight: "bold", size: 10pt)[2계층: 실행 -- AI 운영팀]
      #v(0pt)
      #text(fill: white.darken(10%), size: 8.5pt)[
        모델 개발·운영 · 성과 모니터링 · 공정성 점검 · 추천 품질 관리 · 인시던트 대응
      ]
    ]
    #v(-2pt)
    #block(
      fill: teal,
      width: 100%,
      inset: 12pt,
      radius: (bottom: 6pt),
    )[
      #text(fill: white, weight: "bold", size: 10pt)[3계층: 검증 -- 내부 감사·컴플라이언스]
      #v(0pt)
      #text(fill: white.darken(10%), size: 8.5pt)[
        독립적 감사 · 규제 준수 확인 · 영향평가 검토 · 외부 감사 대응
      ]
    ]
  ]
]
]

== 적용 규제 종합 매핑

#text(size: 8.5pt)[
#table(
  columns: (0.8fr, 0.8fr, 1.5fr),
  align: (center, center, left),
  [규제], [관련 조항], [설계 반영],
  [*AI 기본법*], [§31 (AI 생성물 표시)], [AI 공시 + 추천 사유 L1/L2],
  [], [§33 (고위험 AI 거버넌스)], [36항목 레지스트리 + 거버넌스 보고서],
  [], [§34 (위험 관리 기록)], [감사 로그 불변성 + 7개 감사 테이블],
  [*금소법*], [§19 (설명의무)], [피처 역매핑 + 태스크별 해석],
  [*금감원 AI RMF*], [①합법성], [36항목 자동 점검],
  [], [②안전·신뢰], [킬스위치 + 인시던트 보고],
  [], [④신뢰성], [드리프트 감시 + 자동 재학습],
  [], [⑤금융안정성], [공정성 DI/SPD/EOD + 쏠림 탐지],
  [*GDPR*], [§17 (삭제권)], [30일 PII 보존 + 암호화 삭제],
  [], [§22 (자동화 거부)], [opt_out_audit 테이블],
  [], [§35 (DPIA)], [PIA 갭 분석 항목],
  [*개보법*], [§28의2 (가명정보)], [가명 처리 기록 감사],
  [], [§37조의2 (자동화 결정)], [옵트아웃 + 인적 재처리],
  [*EU AI Act*], [Art. 13 (투명성)], [추천 사유 + 모델 카드 + 감사 로그],
  [], [Art. 14 (인간 감독)], [킬스위치 + 인적 재처리 + 채널별 인간 감독],
  [], [Art. 15 (정확성·보안)], [드리프트 감지 + 프롬프트 인젝션 방어],
)
]

#v(1.5em)

#align(center)[
  #block(
    fill: gray-bg,
    stroke: 0.5pt + gray-ln,
    inset: 14pt,
    radius: 4pt,
    width: 90%,
  )[
    #text(fill: txt-sub, size: 8.5pt)[
      본 문서는 2026년 4월 기준으로 작성되었으며, 인공지능기본법 시행령 및 금융위원회 통합 AI 가이드라인의 확정·시행에 따라 내용이 변경될 수 있습니다. 규제 환경 변화 시 문서를 갱신하고, AI 위험관리위원회에 보고하는 절차를 권고합니다.
    ]
  ]
]
