// ─────────────────────────────────────────────────────────
//  AI 추천 시스템 규제 준수 개요서
//  Executive Summary for Financial Supervisory Review
//  AWS PLE for Financial · 2026. 04.
// ─────────────────────────────────────────────────────────

// ── 컬러 팔레트 (Anthropic Design System) ──
#let anthropic-bg = rgb("#F0EFEA")
#let anthropic-text = rgb("#141413")
#let anthropic-accent = rgb("#CC785C")
#let anthropic-muted = rgb("#6B7280")
#let anthropic-rule = rgb("#D1D5DB")

#let navy     = anthropic-text
#let red-acc  = anthropic-accent
#let blue     = anthropic-accent
#let teal     = anthropic-accent
#let gray-bg  = anthropic-bg
#let gray-ln  = anthropic-rule
#let txt      = anthropic-text
#let txt-sub  = anthropic-muted

// ── 페이지 & 글꼴 ──
#set page(
  paper: "a4",
  margin: (top: 2.5cm, bottom: 2.5cm, left: 2.5cm, right: 2.5cm),
  fill: anthropic-bg,
  header: context {
    if counter(page).get().first() > 1 [
      #set text(size: 7.5pt, font: ("Pretendard", "New Computer Modern"), fill: anthropic-muted, tracking: 0.12em)
      #smallcaps[AI 추천 시스템 규제 준수 개요서]
      #h(1fr)
      #smallcaps[AWS PLE for Financial]
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

#set text(font: ("Pretendard", "New Computer Modern"), size: 10pt, fill: anthropic-text, lang: "ko")
#set par(justify: true, leading: 0.8em, spacing: 1.5em)
#set heading(numbering: none)

// ── heading 스타일 ──
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

// ── 테이블 스타일 ──
#set table(
  inset: 8pt,
  stroke: 0.5pt + anthropic-rule,
  fill: (_, y) => if y == 0 { anthropic-accent.lighten(88%) } else { none },
)
#show table.cell: it => {
  if it.y == 0 {
    set text(fill: anthropic-text, weight: "bold", size: 9pt)
    it
  } else {
    set text(size: 9pt)
    it
  }
}

// ── 유틸리티 컴포넌트 ──
#let card(title: none, accent: anthropic-accent, body) = {
  block(
    stroke: (left: 2pt + accent),
    inset: (left: 14pt, right: 14pt, top: 10pt, bottom: 10pt),
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

#let status-tag(label) = {
  let color = if label == "구현완료" { rgb("#16A34A") } else if label == "설계완료" { rgb("#D97706") } else { anthropic-muted }
  box(
    fill: color.lighten(85%),
    stroke: 0.5pt + color.lighten(40%),
    inset: (x: 6pt, y: 3pt),
    radius: 3pt,
  )[#text(fill: color, weight: "bold", size: 8pt)[#label]]
}

// ═══════════════════════════════════════════════════════════
//  표지
// ═══════════════════════════════════════════════════════════

#page(header: none, footer: none)[
  #v(3cm)

  #align(center)[
    #text(
      size: 10pt,
      fill: anthropic-muted,
      tracking: 0.5em,
      weight: "regular",
    )[#upper[Executive Summary]]
    #v(0.5cm)

    #text(size: 26pt, fill: anthropic-text, weight: "bold")[
      AI 추천 시스템 규제 준수 개요서
    ]
    #v(0.3cm)
    #text(size: 14pt, fill: anthropic-muted)[
      금융감독원 검토용 Executive Summary
    ]
    #v(0.6cm)
    #line(length: 30%, stroke: 0.5pt + anthropic-rule)
    #v(1em)
    #grid(
      columns: (auto, 1fr),
      gutter: 8pt,
      text(fill: anthropic-muted, size: 9.5pt)[문서 분류],
      text(size: 9.5pt, fill: anthropic-text, weight: "bold")[경영진 요약],
      text(fill: anthropic-muted, size: 9.5pt)[작성일],
      text(size: 9.5pt, fill: anthropic-text, weight: "bold")[2026년 4월],
      text(fill: anthropic-muted, size: 9.5pt)[버전],
      text(size: 9.5pt, fill: anthropic-text, weight: "bold")[v1.0],
      text(fill: anthropic-muted, size: 9.5pt)[대상 독자],
      text(size: 9.5pt, fill: anthropic-text, weight: "bold")[금감원 경영진 / 팀장급],
      text(fill: anthropic-muted, size: 9.5pt)[상세 문서],
      text(size: 9.5pt, fill: anthropic-text, weight: "bold")[규제 준수 프레임워크 기술참조서 v1.0],
    )
  ]

  #v(1fr)

  #align(center)[
    #block(
      width: 85%,
      stroke: (left: 2pt + anthropic-accent),
      inset: (left: 14pt, right: 14pt, top: 10pt, bottom: 10pt),
    )[
      #text(fill: anthropic-text, size: 9.5pt)[
        *본 문서는 3~5분 내에 읽을 수 있는 경영진 요약*입니다. 각 항목의 구현 상세, 코드 수준 아키텍처, 조항별 상세 매핑은 별도의 '규제 준수 프레임워크 기술참조서'를 참조하십시오.
      ]
    ]
  ]

  #v(2cm)
]

// ═══════════════════════════════════════════════════════════
//  1. 시스템 개요
// ═══════════════════════════════════════════════════════════

= 시스템 개요

우체국금융 체크카드 상품 추천을 위한 AI 시스템으로, *이종 전문가 PLE(Progressive Layered Extraction)* 아키텍처 기반 다중 태스크 학습 모델입니다. 고객의 거래 이력, 인구통계, 상품 보유 현황 등을 분석하여 *18개 예측 태스크*(클릭, 구매, 이탈, CLV 등)를 동시에 수행합니다. 학습된 딥러닝 모델은 LGBM(LightGBM)으로 *증류(distillation)*되어, AWS Lambda 서버리스 환경에서 실시간 서빙됩니다. 이 구조는 GPU 없이도 밀리초 단위 응답이 가능하며, 모델 투명성과 운영 비용 효율성을 동시에 확보합니다.

#pagebreak()

// ═══════════════════════════════════════════════════════════
//  2. 규제 대응 현황 매트릭스
// ═══════════════════════════════════════════════════════════

= 규제 대응 현황 매트릭스

#text(size: 8.5pt)[
#set par(justify: false)
#table(
  columns: (1.0fr, 1.8fr, 1.5fr, 0.55fr),
  align: (center, left, left, center),
  [규제], [핵심 요구사항], [대응 컴포넌트], [상태],

  [*금감원 AI RMF*\
  (2026.01)],
  [거버넌스 체계 구축 (G-1~G-6)\
  위험평가 (R-1~R-6)\
  위험통제 (C-1~C-6)],
  [3계층 거버넌스 프레임워크\
  36항목 자동 점검 레지스트리\
  킬스위치 3단계 + 인시던트 관리],
  [#status-tag("구현완료")],

  [*금융위 통합\ AI 가이드라인*\
  (7대 원칙)],
  [거버넌스, 합법성, 보조수단성\
  신뢰성, 금융안정성, 신의성실, 보안성],
  [7대 원칙 전항목 시스템 매핑 완료\
  거버넌스 위원회 공식 설치는 조직 결정 필요],
  [#status-tag("구현완료")],

  [*인공지능기본법*\
  (2026.01.22 시행)],
  [AI 사용 사전 고지 (제31조)\
  고영향 AI 거버넌스 (제33~34조)\
  영향평가 (제35조)],
  [AI 사용 고지 자동 포함\
  고영향 AI 준하는 수준으로 선제 구축\
  영향평가 프로세스 조직 수립 필요],
  [#status-tag("구현완료")],

  [*EU AI Act*\
  (고위험: 2027.12)],
  [투명성/정보 제공 (Art. 13)\
  인간 감독 (Art. 14)\
  정확성/견고성/보안 (Art. 15)],
  [모델 카드 자동 생성 + 추천 사유\
  킬스위치 + 인적 재처리 라우팅\
  드리프트 감지 + 프롬프트 인젝션 방어],
  [#status-tag("구현완료")],

  [*개인정보보호법*\
  (자동화 결정 거부권)],
  [자동화 의사결정 거부권 (제37조의2)\
  가명정보 처리 (제28조의2)\
  삭제권],
  [옵트아웃 전 생애주기 관리\
  SHA-256 비식별화 + 도메인별 Salt\
  30일 PII 보존 + 암호화 삭제],
  [#status-tag("구현완료")],

  [*금융소비자보호법*],
  [적합성 원칙 (제17조)\
  적정성 원칙 (제18조)\
  설명의무 (제19조)],
  [적격성/적정성 자동 검증\
  피처 역매핑 + 자연어 추천 사유 생성],
  [#status-tag("구현완료")],
)
]

#align(right)[
  #text(size: 8pt, fill: txt-sub)[
    #status-tag("구현완료") 시스템 구현 완료 #h(8pt) #status-tag("설계완료") 설계 완료, 조직 의사결정 필요
  ]
]

#pagebreak()

// ═══════════════════════════════════════════════════════════
//  3. MRM 생애주기
// ═══════════════════════════════════════════════════════════

= 모델리스크관리(MRM) 생애주기

SR 11-7(미 연준/OCC), NIST AI RMF, 금감원 AI RMF를 통합한 5단계 모델 생애주기를 운영합니다.

#v(0.3cm)

// Flow diagram using stacked blocks
#align(center)[
  #block(width: 95%)[
    #grid(
      columns: (1fr, 0.3fr, 1fr, 0.3fr, 1fr, 0.3fr, 1fr, 0.3fr, 1fr),
      align: center + horizon,
      block(
        fill: anthropic-accent.lighten(85%),
        stroke: 1pt + anthropic-accent,
        inset: 10pt,
        radius: 4pt,
        width: 100%,
      )[
        #text(fill: anthropic-accent, weight: "bold", size: 9pt)[1. 개발]
      ],
      text(size: 14pt, fill: anthropic-muted)[#sym.arrow.r],
      block(
        fill: anthropic-accent.lighten(85%),
        stroke: 1pt + anthropic-accent,
        inset: 10pt,
        radius: 4pt,
        width: 100%,
      )[
        #text(fill: anthropic-accent, weight: "bold", size: 9pt)[2. 검증]
      ],
      text(size: 14pt, fill: anthropic-muted)[#sym.arrow.r],
      block(
        fill: anthropic-text,
        stroke: 1pt + anthropic-text,
        inset: 10pt,
        radius: 4pt,
        width: 100%,
      )[
        #text(fill: white, weight: "bold", size: 9pt)[3. 승인]
      ],
      text(size: 14pt, fill: anthropic-muted)[#sym.arrow.r],
      block(
        fill: anthropic-accent.lighten(85%),
        stroke: 1pt + anthropic-accent,
        inset: 10pt,
        radius: 4pt,
        width: 100%,
      )[
        #text(fill: anthropic-accent, weight: "bold", size: 9pt)[4. 모니터링]
      ],
      text(size: 14pt, fill: anthropic-muted)[#sym.arrow.r],
      block(
        fill: anthropic-accent.lighten(85%),
        stroke: 1pt + anthropic-accent,
        inset: 10pt,
        radius: 4pt,
        width: 100%,
      )[
        #text(fill: anthropic-accent, weight: "bold", size: 9pt)[5. 재학습]
      ],
    )
  ]
]

#v(0.3cm)

#text(size: 9pt)[
#set par(justify: false)
#table(
  columns: (0.5fr, 1.2fr, 1.3fr),
  align: (center, left, left),
  [단계], [핵심 활동], [시스템 컴포넌트],
  [1. 개발], [피처 파이프라인 구축, 모델 학습, 하이퍼파라미터 최적화], [PipelineRunner + SageMaker Training],
  [2. 검증], [독립적 성능 평가, 공정성 감사, 편향 테스트], [Champion-Challenger 비교 + FairnessMonitor],
  [3. *승인*], [*AI 위험관리위원회 심의, 수동 배포 승인*], [*auto\_promote=False (자동 배포 차단)*],
  [4. 모니터링], [드리프트 감지, 성능 추적, 공정성 지표 측정], [PSI 기반 드리프트 + DI/SPD/EOD 일간 측정],
  [5. 재학습], [임계값 위반 시 자동 재학습 트리거, 성능 미달 모델 폐기], [3일 연속 PSI 초과 시 자동 재학습],
)
]

#card(title: "핵심 안전장치: 수동 승인 게이트", accent: anthropic-accent)[
  3단계 승인은 반드시 *사람이 수행*합니다. Champion-Challenger 비교 결과(성능, 공정성, 안정성)를 포함한 자동 생성 리포트를 AI 운영팀이 검토하고, AI 위험관리위원회가 최종 승인합니다. 자동 모델 교체는 *명시적으로 차단*되어 있으며, 승인 없이 운영 환경에 배포되는 경로는 존재하지 않습니다.
]

#pagebreak()

// ═══════════════════════════════════════════════════════════
//  4. 핵심 안전장치 요약
// ═══════════════════════════════════════════════════════════

= 핵심 안전장치 요약

== 공정성 모니터링

보호 속성 *5개*(연령대, 성별, 지역 유형, 소득 분위, 생애주기)에 대해 공정성 지표 *3종*을 *일간 자동 측정*합니다.

#text(size: 9pt)[
#table(
  columns: (0.8fr, 1.5fr, 0.8fr),
  align: (center, left, center),
  [지표], [설명], [임계값],
  [DI (Disparate Impact)], [집단 간 긍정 결과 비율], [0.8 ~ 1.25],
  [SPD (Statistical Parity Difference)], [집단 간 선택률 차이], [|SPD| #sym.lt.eq 0.1],
  [EOD (Equal Opportunity Difference)], [집단 간 적중률 차이], [|EOD| #sym.lt.eq 0.1],
)
]

임계값 위반 시 자동으로 인시던트가 생성되며, DI < 0.6은 CRITICAL 등급으로 즉시 에스컬레이션됩니다.

== 드리프트 감시

PSI(Population Stability Index) 기반 피처/예측/라벨 분포 변화를 일간 측정합니다. PSI > 0.25가 *3일 연속* 발생하면 자동 재학습 파이프라인이 트리거됩니다. 전체 분포가 안정적이더라도 개별 피처의 급변을 조기 감지할 수 있도록 피처별 PSI를 개별 추적합니다.

== 감사 추적

모든 의사결정·변경·접근 이력에 *HMAC-SHA256 서명 + 해시 체인*을 적용하여 위변조를 방지합니다. S3 Object Lock(WORM)으로 관리자도 물리적 삭제가 불가하며, 규제 감사 로그는 *7년간 불변 보존*됩니다. DynamoDB 기반 7개 감사 테이블(킬스위치, 동의, 프로파일링, 옵트아웃, 인시던트, 증류, 임베딩)이 운영됩니다.

== 킬스위치 (긴급 차단)

#text(size: 9pt)[
#set par(justify: false)
#table(
  columns: (0.6fr, 1fr, 1.2fr),
  align: (center, left, left),
  [단계], [조치], [폴백 전략],
  [1단계], [문제 모델 즉시 비활성화], [이전 승인 버전으로 자동 롤백],
  [2단계], [이전 버전 롤백 서빙], [롤백 모델도 기준 미달 시 3단계 전환],
  [3단계], [AI 모델 전체 비활성화], [사전 정의된 규칙 기반 추천으로 전환],
)
]

GLOBAL(전체) / PER\_TASK(태스크별) / PER\_CLUSTER(고객군별) 3가지 범위로 세분화 차단이 가능합니다.

== 고객 권리 보장

#text(size: 9pt)[
#set par(justify: false)
#table(
  columns: (0.8fr, 1.8fr),
  align: (center, left),
  [권리], [시스템 대응],
  [옵트아웃 (AI 결정 거부)], [즉시 인적 대체 경로 전환, 등록/철회/확인 전 생애주기 관리],
  [설명 요청], [피처 역매핑 + 자연어 추천 사유 제공, SLA 10일 이내 회신],
  [이의 제기], [7개 사유별 자동 상담원 전환, P1(1h) / P2(4h) / P3(24h) SLA],
  [삭제권], [30일 PII 보존 정책, 암호화 삭제, S3 Lifecycle 자동 적용],
)
]

#pagebreak()

// ═══════════════════════════════════════════════════════════
//  5. 상세 참조 안내
// ═══════════════════════════════════════════════════════════

= 상세 참조 안내

각 항목의 구현 상세는 *'규제 준수 프레임워크 기술참조서'*의 해당 섹션을 참조하십시오.

#text(size: 9pt)[
#set par(justify: false)
#table(
  columns: (1.2fr, 2fr, 0.8fr),
  align: (center, left, center),
  [본 문서 항목], [기술참조서 대응 섹션], [참조서 섹션],
  [시스템 개요], [금융 AI 규제 환경 개요], [제1장],
  [규제 대응 매트릭스 -- 금감원], [금감원 가이드라인 매핑 -- 7대 원칙과 시스템 대응], [제2장],
  [규제 대응 매트릭스 -- EU], [EU AI Act 조항별 매핑 (Art. 13/14/15, GDPR Art. 22)], [제3장],
  [규제 대응 매트릭스 -- AI기본법], [한국 AI 기본법 고영향 AI 분류와 금융 적용], [제4장],
  [핵심 안전장치 (공정성/드리프트/감사/킬스위치)], [컴플라이언스 아키텍처 (3-Layer 구조 상세)], [제5장],
  [MRM 생애주기], [모델리스크관리(MRM) 프레임워크], [제7장],
  [고객 권리 보장], [Human-in-the-Loop 설계 + 옵트아웃 관리], [제6장],
  [거버넌스 보고], [거버넌스 보고서 자동 생성 (9개 섹션 + 36항목 점검)], [제8장],
)
]

#v(1.5em)

#card(title: "거버넌스 보고 체계", accent: anthropic-accent)[
  월/분기 단위 거버넌스 보고서가 자동 생성됩니다. 공정성 요약, 드리프트 요약, 인시던트 현황, 모델 변경 이력, 킬스위치 이력, 추천 품질, 리스크 변동, 감사 스토어 요약, 경영진 요약 등 *9개 섹션*으로 구성됩니다. 36항목 규제 준수 레지스트리를 통해 분기별 전체 자동 점검이 실행됩니다.
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
      본 문서는 2026년 4월 기준으로 작성되었습니다. 인공지능기본법 시행령 및 금융위원회 통합 AI 가이드라인의 확정/시행에 따라 내용이 변경될 수 있습니다. 상세 기술 내용은 '규제 준수 프레임워크 기술참조서 v1.0'을 참조하십시오.
    ]
  ]
]
