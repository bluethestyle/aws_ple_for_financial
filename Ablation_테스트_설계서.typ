// Ablation 테스트 설계서 — PLE-Cluster-adaTT 추천 시스템
// 피처 그룹 & Shared Expert Ablation Study Design

// 1. 디자인 토큰 설정 (VS Code 다크 테마)
#let bg-color = rgb("#1e1e1e")
#let sidebar-color = rgb("#333333")
#let text-color = rgb("#cccccc")
#let accent-color = rgb("#4ec9b0")
#let keyword-color = rgb("#569cd6")
#let function-color = rgb("#dcdcaa")
#let string-color = rgb("#ce9178")
#let comment-color = rgb("#6a9955")
#let box-bg = rgb("#252526")
#let divider-color = rgb("#404040")

// 2. 템플릿 메인 함수
#let dev-whitepaper(
  title: "untitled.typ",
  author: "Anonymous",
  version: "1.0.0",
  doc
) = {
  set text(font: ("Pretendard", "IBM Plex Mono"), fill: text-color, size: 10.5pt)

  set page(
    paper: "a4",
    fill: bg-color,
    margin: (left: 2.8cm, top: 3cm, right: 2cm, bottom: 2cm),

    background: context {
      place(left + top, rect(fill: sidebar-color, width: 1.8cm, height: 100%))
      for i in range(4) {
        place(left + top, dx: 0.65cm, dy: 2cm + (i * 1.5cm),
              circle(radius: 6pt, fill: if i==0 { rgb(255, 255, 255, 51) } else { rgb(128, 128, 128, 128) }))
      }
      place(top + left, dx: 1.8cm, rect(fill: rgb("#252526"), width: 100%, height: 1.2cm))
      place(top + left, dx: 1.8cm,
            rect(fill: bg-color, width: 6cm, height: 1.2cm,
                 stroke: (top: 2pt + accent-color)))
      place(top + left, dx: 2.2cm, dy: 0.45cm,
            text(fill: white, weight: "bold", size: 9pt)[#title])
      place(top + left, dx: 7.2cm, dy: 0.45cm, text(fill: white, size: 8pt)[×])
    },

    footer: context {
      set text(size: 8pt, fill: white)
      rect(fill: rgb("#007acc"), width: 100%, height: 0.6cm, inset: (x: 10pt, y: 4pt),
        grid(
          columns: (1fr, auto, auto, auto),
          gutter: 15pt,
          align(left)[main],
          [Ln #counter(page).display()],
          [UTF-8],
          [Typst v#version]
        )
      )
    }
  )

  set heading(numbering: "1.1.1")
  set par(justify: true, leading: 0.75em, spacing: 0.8em)
  set block(spacing: 1.2em)

  // Level 1
  show heading.where(level: 1): it => {
    pagebreak(weak: true)
    set text(size: 18pt, weight: "bold")
    v(-1.0em)
    line(length: 100%, stroke: 1pt + accent-color)
    v(0.5em)
    text(fill: keyword-color)[class ]
    if it.numbering != none {
      text(fill: accent-color)[Ch#counter(heading).display(it.numbering).]
      h(0.3em)
    }
    text(fill: accent-color)[#it.body]
    text(fill: text-color)[]
    v(0em)
  }

  // Level 2
  show heading.where(level: 2): it => {
    set text(size: 14pt, weight: "semibold")
    v(1.2em)
    h(1em)
    text(fill: keyword-color)[def ]
    if it.numbering != none {
      text(fill: function-color)[#counter(heading).display(it.numbering)]
      h(0.2em)
    }
    text(fill: function-color)[#it.body]
    text(fill: text-color)[(self):]
    v(0.8em)
  }

  // Level 3
  show heading.where(level: 3): it => {
    set text(size: 11.5pt, weight: "semibold")
    v(0.8em)
    h(2em)
    text(fill: comment-color)[]
    if it.numbering != none {
      text(fill: string-color)[#counter(heading).display(it.numbering)]
      h(0.2em)
    }
    text(fill: string-color)[#it.body]
    v(0.5em)
  }

  show raw.where(block: true): block.with(
    fill: rgb("#0d0d0d"),
    inset: 12pt,
    radius: 4pt,
    width: 100%,
    stroke: (left: 2pt + accent-color)
  )

  show raw.where(block: false): box.with(
    fill: rgb("#3c3c3c"),
    outset: (y: 2pt),
    inset: (x: 3pt),
    radius: 2pt
  )

  show quote: block.with(
    fill: box-bg,
    inset: 12pt,
    radius: 4pt,
    stroke: (left: 3pt + keyword-color)
  )

  set list(indent: 1em, body-indent: 0.5em, marker: text(fill: accent-color)[▸])
  set enum(indent: 1em, body-indent: 0.5em, numbering: (..nums) => {
    text(fill: function-color)[#numbering("1.", ..nums)]
  })

  show table: set text(size: 9.5pt)
  show table.cell: it => {
    if it.y == 0 {
      set text(fill: accent-color, weight: "bold")
      it
    } else {
      it
    }
  }

  doc

  v(2em)
  text(size: 18pt, fill: text-color)[]
}

// 커스텀 함수: 정보 박스
#let info-box(title: none, body) = {
  block(
    fill: box-bg,
    stroke: (left: 3pt + keyword-color),
    inset: 12pt,
    radius: 4pt,
    width: 100%,
    breakable: true,
  )[
    #if title != none [
      #text(fill: keyword-color, weight: "bold", size: 10pt)[#title]
      #v(0.3em)
      #line(length: 100%, stroke: 0.5pt + divider-color)
      #v(0.5em)
    ]
    #body
  ]
}

// 커스텀 함수: 수식 박스
#let formula-box(label-text: none, body) = {
  block(
    fill: rgb("#0d0d0d"),
    stroke: (left: 2pt + function-color),
    inset: 10pt,
    radius: 4pt,
    width: 100%,
  )[
    #if label-text != none [
      #text(fill: function-color, size: 9pt, style: "italic")[#label-text]
      #h(1em)
    ]
    #body
  ]
}

// 커스텀 함수: 구분선
#let section-divider() = {
  v(1em)
  line(length: 100%, stroke: (paint: divider-color, thickness: 0.5pt, dash: "dashed"))
  v(1em)
}

// 커스텀 함수: 실험 박스 (ablation 전용)
#let exp-box(id: "EXP-000", title: none, body) = {
  block(
    fill: rgb("#1a2a3a"),
    stroke: (left: 3pt + rgb("#ff9800")),
    inset: 12pt,
    radius: 4pt,
    width: 100%,
    breakable: true,
  )[
    #text(fill: rgb("#ff9800"), weight: "bold", size: 10pt)[#id — #title]
    #v(0.3em)
    #line(length: 100%, stroke: 0.5pt + rgb("#ff9800"))
    #v(0.5em)
    #body
  ]
}

// 커스텀 함수: 가설 박스
#let hypo-box(body) = {
  block(
    fill: rgb("#2a1a3a"),
    stroke: (left: 3pt + rgb("#ce93d8")),
    inset: 10pt,
    radius: 4pt,
    width: 100%,
  )[
    #text(fill: rgb("#ce93d8"), weight: "bold", size: 9pt)[HYPOTHESIS]
    #h(0.5em)
    #body
  ]
}

// ==========================================
// 문서 시작
// ==========================================

#show: doc => dev-whitepaper(
  title: "ablation\_study\_design.typ",
  author: "우체국금융개발원 AI Lab",
  version: "1.2.0",
  doc
)

// 표지
#align(center)[
  #v(6em)
  #text(fill: accent-color, size: 28pt, weight: "bold")[
    Ablation Study Design
  ]
  #v(0.5em)
  #text(fill: text-color, size: 14pt)[
    PLE-GroupTaskExpert-adaTT 추천 시스템
  ]
  #v(0.3em)
  #text(fill: comment-color, size: 14pt)[
    피처 그룹 & Shared Expert 기여도 분석
  ]
  #v(3em)
  #line(length: 60%, stroke: 1pt + divider-color)
  #v(1em)
  #text(fill: function-color, size: 11pt)[우체국금융개발원 AI Lab]
  #v(0.5em)
  #text(fill: comment-color, size: 10pt)[2026-03-10 | Document Version 1.2.0]
  #v(0.5em)
  #text(fill: comment-color, size: 10pt)[
    Model: PLE-GroupTaskExpert-adaTT v3.3 (v2.3 코드 반영) · Features: 734D (644D normalized + 90D raw) · Shared Experts: 6+1 (7종, v3.15) · Causal + OT + SAE + Evidential DL · TaskExpert: GroupEncoder(4)
  ]
]


// ==========================================
= 개요 및 목적
// ==========================================

== Ablation Study 목적

본 문서는 PLE-GroupTaskExpert-adaTT 추천 시스템의 *피처 그룹(Feature Group)* 과 *공유 전문가(Shared Expert)* 각각의 기여도를 정량적으로 측정하기 위한 Ablation 테스트를 설계한다.

**(v3.3 업데이트)** 734D 아키텍처 (644D normalized + 90D raw power-law). GroupTaskExpertBasket이 기본값. ClusterTaskExpertBasket은 v3.1 레거시입니다. **(v3.15)** RawScaleExpert 제거 — 90D raw 피처는 DeepFM/Causal/OT가 734D 전체 입력으로 직접 수신.

#info-box(title: "Ablation Study란?")[
  모델의 특정 구성요소를 체계적으로 제거(ablate) 하거나 교체(substitute) 하면서 전체 성능 변화를 측정하는 실험 방법론. 각 구성요소의 *한계 기여(marginal contribution)* 를 정량화하여 아키텍처 최적화와 리소스 할당의 근거를 제공한다.
]

#v(0.5em)

*핵심 질문:*
+ 734D 피처 (644D normalized + 90D raw) 중 어떤 그룹이 어떤 태스크에 가장 크게 기여하는가?
+ 7개 Shared Expert 중 제거 시 성능 하락이 가장 큰 것은?
+ 피처-Expert 간 최적 조합은 무엇이며, 불필요한 중복이 존재하는가?
+ 계산 비용 대비 성능 효율(Cost-Benefit)이 가장 낮은 구성요소는?

== 실험 대상 시스템 현황

#table(
  columns: (auto, auto),
  fill: (_, y) => if calc.odd(y) { rgb("#2a2a2a") } else { box-bg },
  stroke: 0.5pt + divider-color,
  table.header[항목][현재 값],
  [총 피처 차원], [*734D* (644D normalized Quantile Transform + 90D raw power-law log1p) + 68D (separate\_input) + 6D (coldstart\_signal) = *808D* (EXPECTED\_FEATURE\_DIM\_CAN)],
  [피처 그룹 수], [*7개* 그룹: 7 normalized (base, multi\_source, extended\_source, domain, model\_derived, multidisciplinary, merchant\_hierarchy) = 644D + 90D raw (raw power-law, normalized expert 입력에 통합)],
  [Shared Expert 수], [*7개* (6 공유 + 1 특화: Unified H-GCN 128D, PersLay 64D, DeepFM 64D, Temporal Ensemble 64D, Causal Expert 64D, Optimal Transport Expert 64D, LightGCN 64D) — *v3.15*: RawScaleExpert 제거, 90D raw → DeepFM/Causal/OT 직접 수신 (734D full input)],
  [Expert 출력], [6×64D + 1×128D = 총 *512D* pooling (공유 Expert만 gating)],
  [MTL 태스크], [*18개* (16 enabled) (Binary 4 + Multi-class 7 + Regression 5) \ _온프렘 태스크 수 기준. AWS 벤치마크는 13개로 축소 검증 완료._],
  [Task Expert], [*GroupTaskExpertBasket (v3.3)*: 4 GroupEncoder + 1 ClusterEmbedding + 16 TaskHead (~362K params)],
  [학습 방식], [Phase 1: 전체 30ep → Phase 2: TaskHead fine-tune 20ep \ *v2.3*: freeze\_epoch에 CGC Attention도 adaTT와 동기화하여 동시 freeze],
  [Gate Type], [*Softmax* (AWS ablation 결과: softmax > sigmoid in heterogeneous MTL)],
  [adaTT], [*기본 OFF* (AWS에서 13-task 스케일에서 악화 확인. 온프렘에서 재검증 필요)],
  [Uncertainty Weighting], [loss\_weight 반영 필수 확인 (AWS에서 누락 시 NDCG −0.018 하락 확인)],
)

== 실험 설계 원칙

+ *Single-Factor Ablation*: 한 번에 하나의 구성요소만 제거하여 인과관계 명확화
+ *Controlled Environment*: 동일 seed, 동일 데이터 split, 동일 하이퍼파라미터
+ *Multi-Metric Evaluation*: 태스크 유형별 적합 메트릭 (AUC / Macro-F1 / RMSE / NDCG)
+ *Reproducibility*: 단일 seed (42) 사용. AWS 벤치마크에서 seed 간 차이 ±0.002 AUC 수준으로 구조 간 차이 대비 미미함을 확인.
+ *Cost-Benefit Analysis*: 성능 변화 + 학습 시간 변화 + 파라미터 수 변화 동시 추적


// ==========================================
= AWS 벤치마크 결과 요약 및 온프렘 검증 목표
// ==========================================

== AWS 합성 데이터 핵심 발견 (2026-04-13)

#info-box(title: "AWS 13-task 벤치마크에서 확인된 5가지 발견")[
  + *Uncertainty weighting loss_weight 누락 수정*: 가장 큰 성능 개선 (+0.018 NDCG\@3, +0.031 F1-macro). 아키텍처 변경보다 손실 밸런싱 정확성이 중요.
  + *Softmax > Sigmoid 역전*: 이질적 MTL(binary 7 + multiclass 3 + regression 3)에서 softmax가 sigmoid를 역전. 동질적 task 문헌과 반대 결과.
  + *adaTT 13-task 스케일 실패*: 156개 task 쌍 affinity 추정 불안정으로 전 지표 악화 (−0.019 AUC). SB+adaTT는 중립이나 PLE+adaTT는 악화 — PLE gate와 adaTT loss-level transfer가 충돌.
  + *GradSurgery (gradient-level projection)*: adaTT 대비 악화 없음, F1-macro/MAE 미세 개선. 단 retain_graph VRAM 오버헤드로 12GB GPU에서는 batch 축소 필요.
  + *PLE softmax 단독이 최적 구조*: 구조적 expert 격리(softmax gate)가 수치적 보정(adaTT/GradSurgery)보다 robust.
]

== 온프렘 검증 목표

*Primary*: Expert ablation — 실데이터에서 각 expert의 고유 기여도 측정 (합성 데이터에서는 expert 간 차이 미미)

*Secondary*: AWS 구조 발견의 실데이터 재확인
- softmax vs sigmoid 역전이 유지되는가?
- adaTT가 실데이터에서도 악화하는가? (실데이터의 복잡한 task 간 관계에서 adaTT가 유용할 수 있음)
- shared_bottom vs PLE 차이가 커지는가?

== 온프렘 구조 ablation 최소 세트 (7 시나리오)

#table(
  columns: (auto, auto, 1fr),
  fill: (_, y) => if calc.odd(y) { rgb("#2a2a2a") } else { box-bg },
  stroke: 0.5pt + divider-color,
  table.header[\#][시나리오][목적],
  [1], [shared_bottom], [baseline — PLE 없이 expert 출력 평균],
  [2], [ple_softmax], [최종 구조 후보 — AWS best],
  [3], [ple_sigmoid], [softmax vs sigmoid 역전 검증],
  [4], [ple_softmax + adaTT], [adaTT 실데이터 효과 검증],
  [5], [full − HGCN], [계층구조 expert 필수성 (AWS에서 negative transfer)],
  [6], [full − Temporal], [실 시계열 expert 효과 (AWS 합성에서는 negative transfer, 실데이터에서 reversal 기대)],
  [7], [full − LightGCN], [협업필터링 expert 효과],
  [8], [ple_softmax (Causal confidence gate ON)], [Causal expert spurious DAG 방지 효과 검증],
)

#info-box(title: "시나리오 8: Causal Expert Confidence Gate (신규 구현 필요)")[
  *배경*: AWS 합성 데이터에서 Causal Expert(NOTEARS)가 segment\_prediction F1-macro를 −0.122 악화시킴.
  원인은 합성 데이터에 진짜 인과 구조가 없어서 NOTEARS가 상관관계에서 억지로 DAG를 학습(spurious edge),
  자신있게 틀린 신호를 생성하여 PLE gate가 이를 구분하지 못한 것.

  *해결*: Causal Expert 내부에 confidence gate 추가:
  - DAG의 총 edge weight sum이 threshold 미만이면 출력을 *zero vector*로 대체
  - "인과 관계를 발견하지 못하면 침묵"하는 구조
  - gate가 Causal을 배제하는 것을 학습할 필요 없이 자동으로 해결

  *구현 위치*: `src/models/experts/causal_expert.py`의 forward에서:
  ```python
  dag_strength = self.dag_weights.abs().sum()
  if dag_strength < self.confidence_threshold:
      return torch.zeros_like(output)  # 침묵
  ```

  *비교*:
  - 시나리오 2 (ple\_softmax, gate OFF): 기존 Causal — 항상 출력
  - 시나리오 8 (ple\_softmax, gate ON): confidence gate Causal — 인과 구조 있을 때만 출력
  - 실데이터에서 시나리오 8 > 시나리오 2이면 confidence gate 채택

  *기대*: 실데이터에서 진짜 인과 구조(금리→예금, 상담→가입 등)가 존재하므로
  DAG edge weight가 threshold를 초과하여 Causal이 활성화되고,
  로짓 전이(설계된 경로)와 Causal DAG(발견된 경로)의 이중 인과 검증이 가능해짐.
]


// ==========================================
= Joint Ablation 설계 (피처 + Expert 통합)
// ==========================================

#info-box(title: "설계 원칙: 피처와 Expert를 분리하지 않는다")[
  대부분의 expert가 특정 피처 그룹에 종속되어 있다:
  - H-GCN → hierarchy + merchant 피처
  - PersLay → TDA 피처
  - Temporal Ensemble → 시계열 피처
  - LightGCN → CF 임베딩

  피처를 제거하면 해당 expert가 사실상 비활성화되므로,
  *피처 단독 제거와 expert 단독 제거를 분리하는 것은 무의미*하다.

  AWS 벤치마크와 동일하게 *Joint Ablation*으로 통합한다:
  - *Bottom-up*: DeepFM만 남기고 expert를 하나씩 추가 (해당 피처도 함께 활성화)
  - *Top-down*: 전체 모델에서 expert를 하나씩 제거 (해당 전용 피처도 비활성화)
]

== 현행 Expert 구조 (7 Active: 6 공유 + 1 특화, v3.15)

#table(
  columns: (auto, auto, auto, auto, 1fr),
  fill: (_, y) => if calc.odd(y) { rgb("#2a2a2a") } else { box-bg },
  stroke: 0.5pt + divider-color,
  table.header[Expert ID][유형][이름][입력][출력][역할],
  [SE-1], [공유], [Unified H-GCN], [47D (hierarchy 20D + merchant 27D)], [128D], [MCC/상품/지역 + 가맹점 계층],
  [SE-2], [공유], [PersLay], [70D (TDA)], [64D], [위상적 소비 패턴],
  [SE-3], [공유], [DeepFM], [734D (normalized 644D + raw 90D, `[:, :734]`)], [64D], [28 field FM 상호작용 (v3.15: 734D full input)],
  [SE-4], [공유], [Temporal Ensemble], [24D (seq)], [64D], [시계열 패턴 (Mamba+LNN)],
  [SE-5], [공유], [Causal Expert], [734D (normalized 644D + raw 90D, `[:, :734]`)], [64D], [인과 구조 추론 (NOTEARS DAG, v3.15: 734D full input)],
  [SE-6], [공유], [Optimal Transport Expert], [734D (normalized 644D + raw 90D, `[:, :734]`)], [64D], [분포 정렬 (Sinkhorn Wasserstein, v3.15: 734D full input)],
  [SE-7], [특화], [LightGCN], [64D (CF)], [64D], [User-Item 협업필터링],
)

#v(0.5em)

*Shared Expert 출력 결합*: 6×64D + 1×128D = *512D* → GroupTaskExpertBasket 입력 (+ HMM 32D = 544D)

*GroupTaskExpertBasket (v3.3, 기본값)*:
- 4 GroupEncoder (Engagement, Lifecycle, Value, Consumption)
- 1 ClusterEmbedding (GMM 20 클러스터)
- 16 TaskHead (태스크별 경량 헤드)
- *SAE (Sparse Autoencoder)*: Shared Expert 출력(512D)에서 희소 특징 추출 — 512D→2048D→512D (expansion\_factor=4) → 해석 가능성 향상
- *Evidential DL*: TaskHead 불확실성 추정 (evidential regression / classification) → 신뢰도 calibration
- 파라미터: ~362K (~88% 감소 vs ClusterTaskExpertBasket v3.1 ~3.0M)

== EXP-E: Expert 단일 제거 실험 (Leave-One-Expert-Out)

#info-box(title: "실험 전제 조건 (AWS 결과 반영)")[
  - Gate type: *softmax* (AWS에서 sigmoid 대비 NDCG +0.013 확인)
  - adaTT: *OFF* (AWS에서 PLE+adaTT 충돌 확인. 공정 비교를 위해 기본 OFF)
  - Uncertainty weighting: `loss_weight * (precision * L + log_var)` 공식 확인 필수 (log_var clamp \[-4, 4\], precision clamp \[1e-3, 100\])
  - Dry-run: 학습 시작 전 로그에서 `adaTT: disabled`, `gate_type: softmax` 반드시 확인
]

각 Shared Expert를 하나씩 비활성화하고 나머지 Expert 출력으로 학습한다. GroupTaskExpertBasket 입력 차원이 변경되므로, 제거된 Expert 위치를 zero-padding 하여 512D를 유지한다.

*Note*: Expert ablation용 zero-padding과 HMM default embedding은 별개 메커니즘 — HMM 32D: 미존재 시 learnable default embedding (v2.3)

#exp-box(id: "EXP-E1", title: "Unified H-GCN Expert 제거")[
  *비활성화*: Unified H-GCN Expert (hierarchy 20D + merchant 27D = 47D → 128D) \
  *GroupTaskExpert 입력*: 384D + zero-pad 128D = 512D + HMM 32D \
  *파라미터 감소*: ~300K params \

  #hypo-box[Unified H-GCN은 MCC 4단계 계층 구조 + 가맹점 정보를 통합 인코딩. 제거 시 *brand\_prediction Accuracy −10~20%*, merchant\_affinity R² −15~25%, spending\_category −3~8% 하락 예상. base features의 category(64D)가 부분적으로 보상할 수 있어 하락폭이 완전하지 않을 수 있음.]

  *핵심 관찰*: \
  - Brand\_prediction, Merchant\_affinity (계층 구조 직접 의존) \
  - Spending\_category (MCC 계층 간접 의존) \
  - H-GCN vs base category 64D 간 정보 중복도 측정
]

#exp-box(id: "EXP-E2", title: "PersLay Expert 제거")[
  *비활성화*: PersLay Expert (TDA 70D → 64D) \
  *Task Expert 입력*: 448D + zero-pad 64D = 512D + HMM 32D \
  *파라미터 감소*: ~25K params \

  #hypo-box[PersLay는 위상적 소비 패턴(persistence diagram)을 인코딩하는 *고유한 표현*. domain 피처(TDA 70D)가 main tensor에도 포함되어 있어, DeepFM이 간접적으로 TDA 정보를 활용 가능. 그러나 PersLay의 *전문화된 집약*이 사라지므로, churn AUC −2~5%, consumption\_cycle −3~7% 하락 예상.]

  *핵심 관찰*: \
  - Churn AUC (위상 전이 탐지 의존) \
  - Consumption\_cycle (패턴 안정성 의존) \
  - Retention AUC (장기 패턴 의존)
]

#exp-box(id: "EXP-E3", title: "DeepFM Expert 제거")[
  *비활성화*: DeepFM Expert (734D full `[:, :734]` → 28 fields × 16D → FM + Deep → 64D) \
  *GroupTaskExpert 입력*: 448D + zero-pad 64D = 512D + HMM 32D \
  *파라미터 감소*: ~169K params (가장 큰 Expert) \

  #hypo-box[DeepFM은 *734D 전체를 입력받는* Expert로 (Causal/OT와 동일, v3.15), 피처 간 2차 상호작용을 학습. 제거 시 *전 태스크에 걸친 광범위한 하락* 예상. 특히 CTR AUC −5~10%, CVR AUC −3~8% 하락. GroupEncoder들이 부분 피처만 보므로 cross-feature interaction 학습이 완전히 소실됨.]

  *핵심 관찰*: \
  - 전체 17 태스크 평균 성능 (cross-feature interaction의 전반적 기여) \
  - CTR, CVR AUC (피처 상호작용 의존도 최고) \
  - 피처 상호작용의 전반적 기여도 (FM interaction 독자적 가치 측정)
]

#exp-box(id: "EXP-E4", title: "Temporal Ensemble Expert 제거")[
  *비활성화*: Temporal Ensemble (Mamba + LNN + Transformer → 64D) \
  *GroupTaskExpert 입력*: 448D + zero-pad 64D = 512D + HMM 32D \
  *파라미터 감소*: ~50K params \

  #hypo-box[Temporal Ensemble은 *시계열 전문가*로, 36개월 거래 시퀀스를 인코딩. Mamba(50D)가 domain 피처에도 있지만, Ensemble의 *3-model gating*이 사라짐. Timing Top5-Acc −5~12%, consumption\_cycle −5~10%, engagement R² −3~8% 하락 예상. GroupEncoder 중 Consumption, Lifecycle 그룹이 부분 보상 가능.]

  *핵심 관찰*: \
  - Timing Top5-Accuracy (시간 패턴 직접 의존) \
  - Consumption\_cycle (주기성 탐지 의존) \
  - Engagement R² (행동 트렌드 의존) \
  - LNN(model\_derived 18D)과의 상보성
]

#exp-box(id: "EXP-E5", title: "LightGCN Expert 제거 (특화 Expert)")[
  *비활성화*: LightGCN Expert (collaborative 64D → refine → 64D) \
  *GroupTaskExpert 입력*: 448D + zero-pad 64D = 512D + HMM 32D \
  *파라미터 감소*: ~5K params \

  #hypo-box[LightGCN은 *User-Item 협업 필터링* 전문가(특화 Expert). 현행 그래프 구조(활성 유저 ~5M × 아이템 수백 개)에서는 3-layer 메시지 패싱 시 사실상 모든 유저가 2홉 이내로 연결되어 *oversmoothing 위험*이 높음. 임베딩이 평균으로 수렴하여 고유 시그널이 희석될 가능성 있음. 제거 시 실질적 성능 하락이 *미미할 것으로 예상* (CTR −0~2%, CVR −0~2%). DeepFM과 GroupEncoder가 피처 상호작용으로 잔여 협업 시그널을 보상 가능.]

  *핵심 관찰*: \
  - CTR, CVR AUC (협업 필터링 직접 기여) \
  - NBA Hit\@5 (유사 사용자 패턴 의존) \
  - Cold-start vs Warm-start 고객 그룹 분리 평가 \
  - *Oversmoothing 검증*: LightGCN 출력 임베딩의 유저 간 코사인 유사도 분포 확인 — 평균 유사도 > 0.9이면 oversmoothing 확정
]

#exp-box(id: "EXP-E6", title: "Causal Expert 제거")[
  *비활성화*: Causal Expert (causal graph → do-calculus → 64D) \
  *GroupTaskExpert 입력*: 448D + zero-pad 64D = 512D + HMM 32D \
  *파라미터 감소*: ~40K params \

  #hypo-box[Causal Expert는 피처 간 *인과 구조(do-calculus)* 를 명시적으로 모델링하여 confounding bias를 제거한다. 제거 시 관찰-개입 분포 gap이 증가하여 uplift·CVR 태스크 등 *처치 효과 추정* 정확도 하락 예상 (Uplift AUC −2~6%). SAE와의 연계로 희소 인과 표현이 강화되므로 두 Expert를 동시 제거 시 성능 하락이 가중될 가능성 있음.]

  *핵심 관찰*: \
  - Uplift AUC (처치 효과 추정 직접 의존) \
  - CVR AUC (인과 경로 confounding 제거) \
  - Counterfactual 시나리오 예측 정확도 \
  - SAE와의 상보성 (Gate Weight 분석으로 보완)
]

#exp-box(id: "EXP-E7", title: "Optimal Transport Expert 제거")[
  *비활성화*: Optimal Transport Expert (distribution → Wasserstein → 64D) \
  *GroupTaskExpert 입력*: 448D + zero-pad 64D = 512D + HMM 32D \
  *파라미터 감소*: ~35K params \

  #hypo-box[Optimal Transport Expert는 *Wasserstein 거리* 기반으로 소비 분포 간 정렬을 학습한다. 클러스터 경계 근처 고객의 표현을 부드럽게 이어주며, 분포 이동(distribution shift) 시나리오에 강인성을 제공. 제거 시 GMM 클러스터 경계 고객에서 예측 불안정성 증가, spending\_category 및 consumption\_cycle 정확도 하락 예상 (−2~5%).]

  *핵심 관찰*: \
  - SpendingCategory Accuracy (분포 정렬 직접 의존) \
  - ConsumptionCycle Macro-F1 (분포 이동 강인성) \
  - 클러스터 경계 고객 서브그룹 성능 분리 평가 \
  - OT vs PersLay 상보성 (위상 vs 분포 표현 비교)
]



// ==========================================
= Gate Weight 기반 Expert 기여도 분석
// ==========================================

== EXP-G: CGC Gate Weight 분석

Expert 기여도를 *학습 후 추가 실험 없이* 측정하는 방법.
CGC softmax gate가 각 task에 대해 어떤 expert를 얼마나 선택하는지를
validation set 전체에서 평균하여 *expert × task attribution matrix*를 구성한다.

#info-box(title: "Gate Weight 분석의 의미")[
  gate weight $w_(t,k)$는 task $t$가 expert $k$의 출력에 부여하는 가중치.

  *해석 가능성*: "이 추천은 시계열 패턴(Temporal, 35%) + 상품 계층(H-GCN, 28%)에 기반"
  → 별도 SHAP/LIME 없이 모델 내부 구조가 설명을 제공.

  *Expert 필수성 판단*: 특정 expert의 평균 gate weight가 전 task에서 < 0.05이면
  해당 expert는 사실상 비활성화 → 제거 후보.

  *Task-Expert 친화도*: 어떤 task가 어떤 expert에 의존하는지를 직접 확인
  → ablation 없이도 expert별 전문화 패턴을 시각화.
]

== 분석 메트릭

#table(
  columns: (auto, 1fr),
  fill: (_, y) => if calc.odd(y) { rgb("#2a2a2a") } else { box-bg },
  stroke: 0.5pt + divider-color,
  table.header[메트릭][정의],
  [Expert Utilization], [전 task에서 gate weight > 0.05인 비율. 낮으면 expert가 무시됨],
  [Task Specialization], [특정 expert에 gate weight가 집중된 task 수. 높으면 전문화],
  [Gate Entropy $H_t$], [$-sum_k w_(t,k) log w_(t,k)$. 높으면 여러 expert 균등 활용, 낮으면 소수 집중],
  [Expert Attribution Matrix], [task × expert 히트맵. 행: task, 열: expert, 값: 평균 gate weight],
)

== EXP-E와의 관계

Gate weight 분석과 Joint Ablation(EXP-E)은 *상호 보완*:
- *Gate weight*: 학습된 모델 내부에서 expert 기여를 직접 읽음 (추가 학습 불필요)
- *EXP-E (bottom-up/top-down)*: expert를 실제로 제거하고 성능 변화를 측정 (인과적 증거)

Gate weight에서 높은 기여를 보이는 expert가 EXP-E에서도 제거 시 큰 성능 하락을 보이면
*일관적 증거*. 불일치가 있으면 expert 간 중복/보상 메커니즘이 존재한다는 의미.


// ==========================================
= 실험 프로토콜
// ==========================================

== 데이터 분할 및 환경

#table(
  columns: (auto, auto),
  fill: (_, y) => if calc.odd(y) { rgb("#2a2a2a") } else { box-bg },
  stroke: 0.5pt + divider-color,
  table.header[항목][설정],
  [데이터 분할], [Train 70% / Valid 15% / Test 15% (시간 기반 split)],
  [샘플 수], [전체 ~1M 고객 (GMM 20 클러스터 stratified)],
  [Random Seeds], [42 (단일 seed)],
  [학습 Phase 1], [30 epochs, lr=1e-3, AdamW, cosine decay \ *v3.14*: `expert\_learning\_rates`로 Shared Expert별 개별 lr/wd 지원 (param\_groups)],
  [학습 Phase 2], [20 epochs, lr=5e-4, Shared Experts frozen \ *v2.3*: freeze\_epoch에 CGC Attention도 adaTT와 동기화하여 동시 freeze \ *v3.14*: frozen Expert는 param\_groups에서 자동 제외],
  [배치 크기], [4096 (cluster-balanced sampling)],
  [하드웨어], [RTX 4070 12GB, 128GB RAM],
  [Baseline], [Full model (734D + 7 Experts + SAE + Evidential DL) 성능 — seed 42],
)

#info-box(title: "2단계 실험 전략 (2026-04-14, AWS 결과 기반)")[
  *1차: 10 epoch으로 전 시나리오 실행* (구조 7개 + expert ablation)
  - Early stopping: 미적용 (patience 미지정, 전 epoch 수행)
  - Warmup: 3 epochs (cosine LR 안정화)
  - 목적: 방향성 확인 (softmax vs sigmoid, adaTT 효과, expert 기여도 순위)
  - AWS 결과에서 10ep 시점에 plateau 확인 — 대부분의 결론이 10ep 내에 도출

  *2차: 30 epoch으로 핵심 시나리오만 실행*
  - 대상: 1차에서 teacher 후보(ple_softmax) + 흥미로운 결과 1~2개
  - Warmup: 5 epochs
  - Cosine restart T\_0=10: 3회 cycle로 추가 수렴 여부 확인
  - 10ep에서 plateau 확인되면 2차 생략 가능

  *실행 원칙*:
  - 모든 시나리오에서 `adaTT: OFF`, `gate_type: softmax` 기본 (검증 시나리오 제외)
  - 첫 epoch 로그에서 설정 확인 후 본 실행 진행
  - 시나리오 간 고아 프로세스/VRAM 잔여 확인
]

== 성능 평가 메트릭 매트릭스

#table(
  columns: (auto, auto, auto, auto),
  fill: (_, y) => if calc.odd(y) { rgb("#2a2a2a") } else { box-bg },
  stroke: 0.5pt + divider-color,
  table.header[태스크 유형][태스크명][Primary Metric][Secondary Metric],
  [Binary], [CTR, CVR, Churn, Retention, Uplift], [AUC-ROC], [PR-AUC],
  [Multi-class], [NBA, Life\_stage, Channel], [Macro-F1], [Hit\@5],
  [Multi-class], [Timing, SpendingCategory], [Top5-Accuracy], [Macro-F1],
  [Multi-class], [ConsumptionCycle], [Macro-F1], [Accuracy],
  [Regression], [SpendingBucket], [RMSE], [R²],
  [Regression], [LTV], [MAPE], [R²],
  [Regression], [balance\_util, Engagement, Merchant\_affinity], [RMSE], [R²],
  [Hierarchical], [Brand\_prediction], [MCC-Accuracy], [HR\@10],
)

#v(0.5em)

*통계적 유의성 검정*: \
- 단일 seed (42) 결과 보고 \
- Paired t-test (ablation vs baseline) at $alpha$ = 0.05 \
- 성능 변화 $Delta$ = (ablation − baseline) / baseline × 100%

== 실행 순서 및 우선순위

#table(
  columns: (auto, auto, auto, auto, auto),
  fill: (_, y) => if calc.odd(y) { rgb("#2a2a2a") } else { box-bg },
  stroke: 0.5pt + divider-color,
  table.header[Priority][실험 그룹][실험 수][예상 GPU 시간][근거],
  [P0], [Baseline (Full)], [3 runs], [~15h], [비교 기준 확보],
  [P1], [구조 ablation (7 시나리오)], [7 × 3 = 21 runs], [~105h], [softmax/sigmoid/adaTT/HGCN/Temporal/LightGCN 비교],
  [P2], [EXP-E bottom-up (Expert 하나씩 추가)], [7 × 3 = 21 runs], [~105h], [Joint Ablation — Expert 단계적 기여도 측정],
  [P3], [EXP-E top-down (Expert 하나씩 제거)], [7 × 3 = 21 runs], [~105h], [Joint Ablation — Expert 한계 기여 측정],
  [P4], [Gate Weight 분석], [0 추가 runs], [~0h], [학습된 모델에서 attribution matrix 추출],
  [합계], [], [*66 runs*], [*~330h*], [],
)

#info-box(title: "실행 전략")[
  - *P0 → P1 → P2 → P3* 순서로 실행 (핵심 결과 ~330h, 약 13.75일) \
  - P2·P3 결과로 bottom-up/top-down 불일치 시 expert 간 보상 메커니즘 분석 \
  - Gate Weight 분석(P4)은 추가 학습 없이 P2·P3 체크포인트에서 즉시 추출 \
  - GPU 1장 기준 총 ~14일 소요 → 2장 병렬 시 ~7일
]

#info-box(title: "AWS 교훈: 실험 전 필수 검증")[
  + 모든 시나리오 첫 epoch 로그에서 adaTT/gate_type/GradSurgery 설정 확인
  + Uncertainty weighting의 loss_weight 적용 여부 확인 (누락 시 전 결과 무효)
  + Windows 환경에서는 SetThreadExecutionState로 절전 방지
  + 고아 프로세스(orphan process) 확인 — VRAM/RAM 점유 상태 확인 후 학습 시작
]


// ==========================================
= 결과 분석 프레임워크
// ==========================================

== 피처 기여도 히트맵

피처 그룹(행) × 태스크(열) 매트릭스로 성능 변화율($Delta$%)을 시각화한다.

```
               CTR   CVR  Churn  Ret  NBA  Life  Bal  Eng  Ch  Tim  LTV  ...
base          [■■■] [■■■] [■■ ] [■■] [■■] [■ ] [■■] [■■] [■] [■ ] [■■]
multi_source  [■  ] [■  ] [■■ ] [■ ] [■ ] [■■] [■ ] [■ ] [■] [■ ] [■■]
extended_src  [   ] [   ] [■  ] [  ] [  ] [■ ] [  ] [  ] [■■] [■] [  ]
domain        [■  ] [■  ] [■■■] [■■] [■ ] [■ ] [■ ] [■■] [■] [■■] [■ ]
model_derived [   ] [   ] [   ] [  ] [  ] [  ] [  ] [■ ] [  ] [  ] [  ]
multidisc     [   ] [   ] [   ] [  ] [  ] [  ] [  ] [  ] [  ] [  ] [  ]
merchant_hier [   ] [   ] [   ] [  ] [■ ] [  ] [  ] [  ] [  ] [  ] [  ]

■■■ = Δ > 5%    ■■ = 2~5%    ■ = 0.5~2%    (blank) = < 0.5%
```

== Expert 기여도 레이더 차트

7개 Expert 각각의 제거 시 성능 하락을 태스크 그룹별로 레이더 차트로 표현한다.

```
태스크 그룹 축 (v3.3 — 4그룹):
  - Engagement (CTR, CVR, Engagement, Uplift)
  - Lifecycle (Churn, Retention, Life_stage, LTV)
  - Value (balance_util, Channel, Timing)
  - Consumption (NBA, SpendingCategory, ConsumptionCycle, SpendingBucket,
                  MerchantAffinity, BrandPrediction)
  # v3.3: Personalization 그룹이 Consumption에 통합됨
  # Disabled tasks: Uplift, CategoryUplift (메트릭 수집에서 제외)
```

== Cost-Benefit 분석

각 구성요소의 *성능 기여 대비 비용*을 정량화한다.

#formula-box(label-text: "Efficiency Score")[
  $ "Eff"(C) = Delta_"perf"(C) / ("Params"(C) + alpha dot "Time"(C)) $

  $Delta_"perf"(C)$: 구성요소 C 제거 시 전체 평균 성능 하락 \
  $"Params"(C)$: 구성요소 C의 파라미터 수 (normalized) \
  $"Time"(C)$: 구성요소 C의 학습 시간 증가분 (normalized) \
  $alpha$: 파라미터 vs 시간 가중치 (기본 $alpha$ = 0.5)
]

== 최종 의사결정 매트릭스

#table(
  columns: (auto, 1fr, 1fr, 1fr),
  fill: (_, y) => if calc.odd(y) { rgb("#2a2a2a") } else { box-bg },
  stroke: 0.5pt + divider-color,
  table.header[판정][조건][조치][대상 후보],
  [KEEP], [$Delta$avg > 2% 또는 특정 태스크 $Delta$ > 5%], [유지], [base, domain, DeepFM, Temporal],
  [OPTIMIZE], [0.5% < $Delta$avg < 2%], [경량화 또는 차원 축소 검토], [multi\_source, PersLay, LightGCN],
  [CANDIDATE\ REMOVE], [$Delta$avg < 0.5% 및 RI > 0.7], [제거 후보 (비용 절감)], [multidisciplinary?],
  [MERGE], [RI > 0.8 (두 구성요소 간)], [통합 검토], [H-GCN + Merchant H-GCN?, OT + PersLay?],
)


// ==========================================
= 구현 가이드
// ==========================================

== 코드 수정 지점

=== 피처 마스킹 구현

#info-box(title: "feature_integrator.py 수정 — GroupTaskExpertBasket 활성화")[
  ```python
  # feature_integrator.py — ablation 모드 추가 (GroupEncoder 기반)
  class FeatureIntegrator:
      def integrate(self, ..., ablation_mask: dict = None):
          features = self._concat_all_groups(...)  # 734D (644D norm + 90D raw)
          if ablation_mask:
              for group, (start, end) in GROUP_RANGES.items():
                  if group in ablation_mask:
                      features[:, start:end] = 0.0
          return features

  GROUP_RANGES = {
      "base": (0, 238),
      "multi_source": (238, 329),
      "extended_source": (329, 413),
      "domain": (413, 572),
      "model_derived": (572, 599),
      "multidisciplinary": (599, 623),
      "merchant_hierarchy": (623, 644),
      "raw_power_law": (644, 734),  # v3.3: 90D raw (log1p only)
  }
  ```
]

=== Expert 비활성화 구현

#info-box(title: "ple_group_task_expert.py 수정 — Shared Expert Ablation")[
  ```python
  # ple_group_task_expert.py — Expert ablation 모드 (GroupTaskExpertBasket)
  class PLEGroupTaskExpert(nn.Module):
      def forward(self, inputs, ablate_experts: list = None):
          expert_outputs = []
          # Expert input slicing (v3.3)
          full_features = inputs.features[:, :734]    # 734D full (644D norm + 90D raw)
          # Shared Experts 라우팅 (v3.15: RawScaleExpert 제거됨)
          for name, expert in self.shared_experts.items():
              if ablate_experts and name in ablate_experts:
                  # zero vector 출력 (shape 유지)
                  if name == "unified_hgcn":
                      expert_outputs.append(torch.zeros(batch, 128, device=device))
                  else:
                      expert_outputs.append(torch.zeros(batch, 64, device=device))
              else:
                  if name in ("deepfm", "causal", "optimal_transport"):
                      expert_outputs.append(expert(full_features))  # 734D full
                  else:
                      expert_outputs.append(expert(inputs))
          shared_repr = torch.cat(expert_outputs, dim=-1)  # 512D

          # GroupTaskExpertBasket: 4 GroupEncoder + 1 ClusterEmbedding + 16 TaskHead
          group_outputs = self.group_encoders(shared_repr)  # 4개 그룹 인코딩
          cluster_emb = self.cluster_embedding(cluster_id)  # 32D
          task_outputs = self.task_heads(group_outputs + cluster_emb)  # 16개 헤드
          return task_outputs
  ```
]

=== 실험 실행 스크립트

#info-box(title: "scripts/run_ablation.py 구조")[
  ```python
  ABLATION_CONFIGS = {
      # Joint Expert ablation (v3.15: 7 experts)
      "EXP-E1": {"ablate_experts": ["unified_hgcn"]},
      "EXP-E2": {"ablate_experts": ["perslay"]},
      "EXP-E3": {"ablate_experts": ["deepfm"]},
      "EXP-E4": {"ablate_experts": ["temporal_ensemble"]},
      "EXP-E5": {"ablate_experts": ["lightgcn"]},
      "EXP-E6": {"ablate_experts": ["causal"]},
      "EXP-E7": {"ablate_experts": ["optimal_transport"]},
  }
  SEEDS = [42, 123, 7]

  for exp_id, config in ABLATION_CONFIGS.items():
      seed = 42
          run_training(exp_id, seed, **config)
          evaluate_and_log(exp_id, seed)
  ```
]

== MLflow 실험 추적

#table(
  columns: (auto, 1fr),
  fill: (_, y) => if calc.odd(y) { rgb("#2a2a2a") } else { box-bg },
  stroke: 0.5pt + divider-color,
  table.header[항목][설정],
  [Experiment Name], [`ablation_study_v1`],
  [Run Name Pattern], [`{EXP_ID}_seed{SEED}` (e.g., `EXP-E1_seed42`)],
  [Logged Params], [ablate\_experts, seed, epochs, lr],
  [Logged Metrics], [13 태스크 (활성) × primary + secondary = 26 메트릭],
  [Logged Artifacts], [학습 곡선 CSV, gate weight matrix JSON, 모델 체크포인트],
  [Tags], [`ablation_type` (expert / structure / gate\_weight)],
)


// ==========================================
= 예상 결과 및 활용 방안
// ==========================================

== 예상 기여도 순위 (사전 가설)

=== 피처 그룹 기여도 순위 (예상)

#table(
  columns: (auto, auto, auto, 1fr),
  fill: (_, y) => if calc.odd(y) { rgb("#2a2a2a") } else { box-bg },
  stroke: 0.5pt + divider-color,
  table.header[순위][피처 그룹][예상 $Delta$avg][근거],
  [1], [base (238D)], [−5~10%], [전체 734D의 32.4%, RFM/거래/카테고리 핵심 정보],
  [2], [domain (159D)], [−3~8%], [TDA/Mamba 고유 정보, PersLay/Temporal 직결],
  [3], [multi\_source (91D)], [−2~5%], [금융상품 포트폴리오, lifecycle 태스크 핵심],
  [4], [extended\_source (84D)], [−1~3%], [보험/상담/캠페인, channel 태스크 집중],
  [5], [merchant\_hierarchy (21D)], [−1~2%], [brand/merchant 태스크에 집중적 기여],
  [6], [model\_derived (27D)], [−0.5~1.5%], [요약 정보, HMM triple로 부분 보상],
  [7], [multidisciplinary (24D)], [−0~1%], [실험적 피처, 기여도 불확실],
)

=== Expert 기여도 순위 (예상)

#table(
  columns: (auto, auto, auto, 1fr),
  fill: (_, y) => if calc.odd(y) { rgb("#2a2a2a") } else { box-bg },
  stroke: 0.5pt + divider-color,
  table.header[순위][Expert][예상 $Delta$avg][근거],
  [1], [DeepFM], [−4~8%], [734D full input, cross-interaction 전담 (v3.15: raw 90D 직접 수신)],
  [2], [Temporal Ensemble], [−3~6%], [시계열 전문, 3-model gating 고유 표현],
  [3], [Unified H-GCN], [−3~6%], [계층+가맹점 통합 (128D), brand/merchant 태스크 핵심],
  [4], [PersLay], [−2~5%], [TDA 전문, 위상적 패턴 고유 표현],
  [5], [Causal Expert], [−2~6%], [인과 구조 추론, uplift/CVR 처치 효과 추정 핵심],
  [6], [Optimal Transport Expert], [−2~5%], [분포 정렬, 클러스터 경계 고객 강인성],
  [7], [LightGCN], [−0~2%], [협업필터링 특화, oversmoothing 우려로 실질 기여 미미 예상],
)

== 활용 방안

=== 모델 경량화

- Efficiency Score 하위 구성요소 제거 → 파라미터 수 및 추론 시간 감소
- 서빙 레이턴시 목표 (p99 < 50ms) 달성을 위한 Expert 수 최적화
- Distillation Student (LGBM)에 전달할 피처 선별 근거

=== 아키텍처 개선

- RI > 0.8인 Expert 쌍 → 통합 Expert로 리팩토링
- 기여도 낮은 피처 그룹 → 차원 축소 또는 임베딩 공유
- HMM triple-mode의 독립적 기여 확인 → 라우팅 전략 조정

=== 다음 버전 (v4.0) 로드맵 근거

- Ablation 결과 기반 피처/Expert 선별 → *Lean Architecture* 제안
- 불필요 구성요소 제거 후 VRAM 여유 → 새로운 Expert 추가 여지
- 태스크별 최적 피처 서브셋 → Task-specific Feature Selection 도입 근거
