// ─────────────────────────────────────────────────────────
//  AI Recommendation System Regulatory Compliance Overview
//  Executive Summary for Financial Supervisory Review
//  AWS PLE for Financial · 2026. 04.
// ─────────────────────────────────────────────────────────

// ── Color Palette (Anthropic Design System) ──
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

// ── Page & Font ──
#set page(
  paper: "a4",
  margin: (top: 2.5cm, bottom: 2.5cm, left: 2.5cm, right: 2.5cm),
  fill: anthropic-bg,
  header: context {
    if counter(page).get().first() > 1 [
      #set text(size: 7.5pt, font: "New Computer Modern", fill: anthropic-muted, tracking: 0.12em)
      #smallcaps[AI Recommendation System Regulatory Compliance Overview]
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
      #set text(size: 8pt, font: "New Computer Modern", fill: anthropic-muted)
      #h(1fr)
      — #pg —
      #h(1fr)
    ]
  },
)

#set text(font: "New Computer Modern", size: 10pt, fill: anthropic-text, lang: "en")
#set par(justify: true, leading: 0.5em, spacing: 1.0em)
#set heading(numbering: none)

// ── Heading Styles ──
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

// ── Table Styles ──
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

// ── Utility Components ──
#let card(title: none, accent: anthropic-accent, body) = {
  block(
    stroke: (left: 2pt + accent),
    inset: (left: 8pt, right: 8pt, top: 6pt, bottom: 6pt),
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
  let color = if label == "Implemented" { rgb("#16A34A") } else if label == "Designed" { rgb("#D97706") } else { anthropic-muted }
  box(
    fill: color.lighten(85%),
    stroke: 0.5pt + color.lighten(40%),
    inset: (x: 6pt, y: 3pt),
    radius: 3pt,
  )[#text(fill: color, weight: "bold", size: 8pt)[#label]]
}

// ═══════════════════════════════════════════════════════════
//  Cover Page
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

    #text(size: 24pt, fill: anthropic-text, weight: "bold")[
      AI Recommendation System#linebreak()Regulatory Compliance Overview
    ]
    #v(0.3cm)
    #text(size: 14pt, fill: anthropic-muted)[
      Executive Summary for Financial Supervisory Review
    ]
    #v(0.6cm)
    #line(length: 30%, stroke: 0.5pt + anthropic-rule)
    #v(1em)
    #grid(
      columns: (auto, 1fr),
      gutter: 8pt,
      text(fill: anthropic-muted, size: 9.5pt)[Classification],
      text(size: 9.5pt, fill: anthropic-text, weight: "bold")[Executive Summary],
      text(fill: anthropic-muted, size: 9.5pt)[Date],
      text(size: 9.5pt, fill: anthropic-text, weight: "bold")[April 2026],
      text(fill: anthropic-muted, size: 9.5pt)[Version],
      text(size: 9.5pt, fill: anthropic-text, weight: "bold")[v1.0],
      text(fill: anthropic-muted, size: 9.5pt)[Detailed Reference],
      text(size: 9.5pt, fill: anthropic-text, weight: "bold")[Regulatory Compliance Framework Technical Reference v1.0],
    )
  ]

  #v(1fr)

  #align(center)[
    #block(
      width: 85%,
      stroke: (left: 2pt + anthropic-accent),
      inset: (left: 8pt, right: 8pt, top: 6pt, bottom: 6pt),
    )[
      #text(fill: anthropic-text, size: 9.5pt)[
        *This document is an executive summary designed to be read in 3--5 minutes.* For implementation details, code-level architecture, and clause-by-clause mappings, please refer to the separate 'Regulatory Compliance Framework Technical Reference.'
      ]
    ]
  ]

  #v(2cm)
]

// ═══════════════════════════════════════════════════════════
//  1. System Overview
// ═══════════════════════════════════════════════════════════

= System Overview

This is an AI system for Korea Post Financial check card product recommendations, built on a *heterogeneous-expert PLE (Progressive Layered Extraction)* multi-task learning architecture. It analyzes customer transaction history, demographics, and product holdings to simultaneously perform *13 prediction tasks* (click, purchase, churn, CLV, etc.). The trained deep learning model is *distilled* into LGBM (LightGBM) and served in real-time on AWS Lambda serverless infrastructure via a *3-layer fallback* (distilled LGBM → direct PLE → rule-based engine), ensuring the service never stops even under kill-switch activation. This architecture enables millisecond-level responses without GPU, achieving both model transparency and operational cost efficiency.

// ═══════════════════════════════════════════════════════════
//  2. Regulatory Compliance Matrix
// ═══════════════════════════════════════════════════════════

= Regulatory Compliance Matrix

#text(size: 8.5pt)[
#set par(justify: false)
#table(
  columns: (0.8fr, 1.2fr, 1.5fr, 0.55fr),
  align: (center, left, left, center),
  [Regulation], [Key Requirements], [Response Components], [Status],

  [*FSS AI RMF*\
  (Jan 2026)],
  [Governance framework (G-1~G-6)\
  Risk assessment (R-1~R-6)\
  Risk control (C-1~C-6)],
  [3-tier governance framework\
  36-item auto-check registry\
  3-level kill switch + incident management],
  [#status-tag("Implemented")],

  [*FSC Integrated\ AI Guidelines*\
  (7 Principles)],
  [Governance, legality, subsidiarity\
  reliability, financial stability,\ good faith, security],
  [All 7 principles mapped to system\
  Formal committee establishment\ requires organizational decision],
  [#status-tag("Implemented")],

  [*Korea AI\ Basic Act*\
  (Eff. Jan 22, 2026)],
  [AI usage disclosure (Art. 31)\
  High-impact AI governance (Art. 33--34)\
  Impact assessment (Art. 35)],
  [AI disclosure auto-included in all outputs\
  Proactively built to high-impact AI level\
  Impact assessment org. setup needed],
  [#status-tag("Implemented")],

  [*EU AI Act*\
  (High-risk: Dec 2027)],
  [Transparency/information (Art. 13)\
  Human oversight (Art. 14)\
  Accuracy/robustness/security (Art. 15)],
  [Auto-generated model cards + reason text\
  Kill switch + human re-processing routing\
  Drift detection + prompt injection defense],
  [#status-tag("Implemented")],

  [*Personal Information\ Protection Act*\
  (PIPA)],
  [Right to refuse automated decisions (Art. 37-2)\
  Pseudonymized data (Art. 28-2)\
  Right to erasure],
  [Full opt-out lifecycle management\
  SHA-256 de-identification + domain salt\
  30-day PII retention + encrypted deletion],
  [#status-tag("Implemented")],

  [*Financial Consumer\ Protection Act*],
  [Suitability principle (Art. 17)\
  Appropriateness principle (Art. 18)\
  Duty to explain (Art. 19)],
  [Automated eligibility/appropriateness checks\
  Feature reverse-mapping + NL reason generation],
  [#status-tag("Implemented")],
)
]

#align(right)[
  #text(size: 8pt, fill: txt-sub)[
    #status-tag("Implemented") System implementation complete #h(8pt) #status-tag("Designed") Design complete, organizational decision required
  ]
]

#pagebreak()

// ═══════════════════════════════════════════════════════════
//  3. MRM Lifecycle
// ═══════════════════════════════════════════════════════════

= Model Risk Management (MRM) Lifecycle

A 5-stage model lifecycle integrating SR 11-7 (US Fed/OCC), NIST AI RMF, and FSS AI RMF is in operation.

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
        #text(fill: anthropic-accent, weight: "bold", size: 9pt)[1. Develop]
      ],
      text(size: 14pt, fill: anthropic-muted)[#sym.arrow.r],
      block(
        fill: anthropic-accent.lighten(85%),
        stroke: 1pt + anthropic-accent,
        inset: 10pt,
        radius: 4pt,
        width: 100%,
      )[
        #text(fill: anthropic-accent, weight: "bold", size: 9pt)[2. Validate]
      ],
      text(size: 14pt, fill: anthropic-muted)[#sym.arrow.r],
      block(
        fill: anthropic-text,
        stroke: 1pt + anthropic-text,
        inset: 10pt,
        radius: 4pt,
        width: 100%,
      )[
        #text(fill: white, weight: "bold", size: 9pt)[3. Approve]
      ],
      text(size: 14pt, fill: anthropic-muted)[#sym.arrow.r],
      block(
        fill: anthropic-accent.lighten(85%),
        stroke: 1pt + anthropic-accent,
        inset: 10pt,
        radius: 4pt,
        width: 100%,
      )[
        #text(fill: anthropic-accent, weight: "bold", size: 9pt)[4. Monitor]
      ],
      text(size: 14pt, fill: anthropic-muted)[#sym.arrow.r],
      block(
        fill: anthropic-accent.lighten(85%),
        stroke: 1pt + anthropic-accent,
        inset: 10pt,
        radius: 4pt,
        width: 100%,
      )[
        #text(fill: anthropic-accent, weight: "bold", size: 9pt)[5. Retrain]
      ],
    )
  ]
]

#v(0.3cm)

#text(size: 9pt)[
#set par(justify: false)
#table(
  columns: (0.4fr, 1.2fr, 1.3fr),
  align: (center, left, left),
  [Stage], [Key Activities], [System Components],
  [1. Develop], [Feature pipeline construction, model training, hyperparameter optimization], [PipelineRunner + SageMaker Training],
  [2. Validate], [Independent performance evaluation, fairness audit, bias testing], [Champion-Challenger comparison + FairnessMonitor],
  [3. *Approve*], [*AI Risk Committee review, manual deployment approval*], [*auto\_promote=False (automatic deployment blocked)*],
  [4. Monitor], [Drift detection, performance tracking, fairness metric measurement], [PSI-based drift + DI/SPD/EOD daily measurement],
  [5. Retrain], [Automatic retrain trigger on threshold violation, retire underperforming models], [Auto-retrain on 3 consecutive days of PSI exceedance],
)
]

#card(title: "Key Safeguard: Manual Approval Gate", accent: anthropic-accent)[
  Stage 3 approval is *always performed by humans*. An automatically generated report containing Champion-Challenger comparison results (performance, fairness, stability) is reviewed by the AI operations team, and the AI Risk Management Committee provides final approval. Automatic model replacement is *explicitly blocked* -- there is no pathway to production deployment without approval.
]

// ═══════════════════════════════════════════════════════════
//  4. Key Safeguards Summary
// ═══════════════════════════════════════════════════════════

= Key Safeguards Summary

== Fairness Monitoring

Three fairness metrics are *measured daily and automatically* across *5 protected attributes* (age group, gender, region type, income quintile, lifecycle stage).

#text(size: 9pt)[
#table(
  columns: (1.8fr, 1.0fr, 0.8fr),
  align: (center, center, center),
  [Metric], [Description], [Threshold],
  [DI (Disparate Impact)], [Ratio of positive outcomes between groups], [0.8 -- 1.25],
  [SPD (Statistical Parity Diff.)], [Selection rate difference between groups], [|SPD| #sym.lt.eq 0.1],
  [EOD (Equal Opportunity Diff.)], [True positive rate difference between groups], [|EOD| #sym.lt.eq 0.1],
)
]

Threshold violations automatically generate incidents; DI < 0.6 triggers a CRITICAL-level escalation.

== Drift Surveillance

PSI (Population Stability Index) monitors feature, prediction, and label distribution shifts daily. When PSI > 0.25 persists for *3 consecutive days*, the automatic retraining pipeline is triggered. Per-feature PSI tracking enables early detection of individual feature shifts even when overall distribution appears stable.

== Audit Trail

All decisions, changes, and access records are secured with *HMAC-SHA256 signatures + hash chains* to prevent tampering. S3 Object Lock (WORM) makes physical deletion impossible even for administrators, and regulatory audit logs are *immutably preserved for 7 years*. Seven DynamoDB audit tables (kill switch, consent, profiling, opt-out, incident, distillation, embedding) are in operation.

== Kill Switch (Emergency Shutdown)

#text(size: 9pt)[
#set par(justify: false)
#table(
  columns: (0.6fr, 1fr, 1.2fr),
  align: (center, center, center),
  [Level], [Action], [Fallback Strategy],
  [Level 1], [Immediate deactivation of problem model], [Automatic rollback to previous approved version],
  [Level 2], [Serve rolled-back version], [Escalate to Level 3 if rollback also fails criteria],
  [Level 3], [Deactivate all AI models], [Switch to pre-defined rule-based recommendations],
)
]

Granular shutdown is available at three scopes: GLOBAL (all) / PER\_TASK (per task) / PER\_CLUSTER (per customer segment).

== Compliance Module

Four dedicated compliance components enforce regulatory obligations at every prediction:

#text(size: 9pt)[
#set par(justify: false)
#table(
  columns: (0.9fr, 1.7fr),
  align: (center, left),
  [Component], [Responsibility],
  [ConsentManager], [Marketing consent lifecycle: grant / revoke / renew / per-channel verification],
  [AIOptOut], [AI decision refusal: registration, withdrawal, confirmation, immediate path switch],
  [RegulatoryChecker], [36-item pre-flight compliance check before each prediction is served],
  [ProfilingRights], [Data subject rights: access / correction / deletion / restriction / portability],
)
]

*Security:* PII is masked before entering LLM pipelines. `PromptSanitizer` strips 8 injection patterns from all Bedrock prompts. `ComplianceAuditStore` records every prediction with full compliance context.

== Customer Rights

#text(size: 9pt)[
#set par(justify: false)
#table(
  columns: (0.8fr, 1.8fr),
  align: (center, center),
  [Right], [System Response],
  [Opt-out (refuse AI decisions)], [Immediate switch to human alternative path; full registration/withdrawal/confirmation lifecycle (AIOptOut)],
  [Request explanation], [Feature reverse-mapping + natural language recommendation reasons; SLA within 10 days],
  [File objection], [Automatic routing to agents by 7 reason types; P1 (1h) / P2 (4h) / P3 (24h) SLA],
  [Right to erasure], [30-day PII retention policy; encrypted deletion; S3 Lifecycle auto-applied],
)
]

#pagebreak()
#v(-2.5em)
// ═══════════════════════════════════════════════════════════
//  5. Detailed Reference Guide
// ═══════════════════════════════════════════════════════════

= Detailed Reference Guide

For implementation details on each item, please refer to the corresponding section of the *'Regulatory Compliance Framework Technical Reference.'*

#text(size: 9pt)[
#set par(justify: false)
#table(
  columns: (1.3fr, 2fr, 0.6fr),
  align: (center, center, center),
  [This Document], [Technical Reference Section], [Section],
  [System Overview], [Financial AI Regulatory Environment Overview], [Ch. 1],
  [Compliance Matrix -- FSS], [FSS Guidelines Mapping -- 7 Principles and System Response], [Ch. 2],
  [Compliance Matrix -- EU], [EU AI Act Article Mapping\ (Art. 13/14/15, GDPR Art. 22)], [Ch. 3],
  [Compliance Matrix -- AI Basic Act], [Korea AI Basic Act High-Impact AI Classification], [Ch. 4],
  [Key Safeguards\ (Fairness / Drift / Audit / Kill Switch)], [Compliance Architecture (3-Layer Structure Details)], [Ch. 5],
  [MRM Lifecycle], [Model Risk Management (MRM) Framework], [Ch. 7],
  [Customer Rights], [Human-in-the-Loop Design + Opt-out Management], [Ch. 6],
  [Governance Reporting], [Automated Governance Report Generation (9 Sections + 36-Item Check)], [Ch. 8],
)
]

#v(0.5em)

#card(title: "Governance Reporting Framework", accent: anthropic-accent)[
  Governance reports are automatically generated on a monthly/quarterly basis. They comprise *9 sections*: fairness summary, drift summary, incident status, model change history, kill switch history, recommendation quality, risk trends, audit store summary, and executive summary. A 36-item regulatory compliance registry enables full automated checks on a quarterly cycle.
]

// ============================================================
== Bedrock Data Protection

Customer data is processed exclusively within the AWS Region (ap-northeast-2) during LLM invocation. Bedrock never transmits input/output data to model providers and never uses it for model training. VPC PrivateLink enables invocation without internet traversal, and CloudTrail audits all API calls. This structurally satisfies PIPA cross-border transfer restrictions and FSS data governance requirements.

// ============================================================
== On-Premises Environment

In air-gapped environments, all processing occurs locally with zero external data transfer. Uses Exaone 3.5 (Apache 2.0) + Qwen 14B (Apache 2.0) open-source models, maintaining the same regulatory compliance framework (checklists, audit logs, governance reports) as AWS.

// ============================================================
= Ops/Audit Agents

Practical enforcement of regulatory compliance is handled by two autonomous diagnostic agents. OpsAgent monitors pipeline performance and stability; AuditAgent audits fairness, regulatory compliance, and recommendation reason quality. 3-agent consensus with minority report preservation mitigates hallucination risk, and all diagnostic results accumulate in a case store for regulatory audit evidence.

Detailed design: Design Document 11
