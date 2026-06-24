// ─────────────────────────────────────────────────────────
//  AI Recommendation System — Regulatory Alignment Overview
//  Executive Summary — FSC Financial-Sector AI Guideline (effective 2026-06-22) 7 principles
//  AWS PLE for Financial · 2026. 06. (v2.0)
//  Nature: architecture-alignment summary of an Independent Research reference
//          (not a compliance attestation); claims use 3 states:
//          ● operational / ◐ implemented, not wired / ○ absent
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
  let color = if label == "operational" { rgb("#16A34A") } else if label == "partially-wired" { rgb("#D97706") } else { anthropic-muted }
  box(
    fill: color.lighten(85%),
    stroke: 0.5pt + color.lighten(40%),
    inset: (x: 6pt, y: 3pt),
    radius: 3pt,
  )[#text(fill: color, weight: "bold", size: 8pt)[#label]]
}

// 3-state honest-evaluation badges
#let s-live = text(fill: rgb("#16A34A"), weight: "bold")[●]
#let s-part = text(fill: rgb("#D97706"), weight: "bold")[◐]
#let s-gap = text(fill: anthropic-muted, weight: "bold")[○]

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
      AI Recommendation System#linebreak()Regulatory Alignment Overview
    ]
    #v(0.3cm)
    #text(size: 14pt, fill: anthropic-muted)[
      Independent Research reference --- architecture alignment with the FSC AI Guideline (effective 2026-06-22) 7 principles (3-state honest)
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
      text(size: 9.5pt, fill: anthropic-text, weight: "bold")[June 2026 (reflecting the guideline's entry into force)],
      text(fill: anthropic-muted, size: 9.5pt)[Version],
      text(size: 9.5pt, fill: anthropic-text, weight: "bold")[v2.0],
      text(fill: anthropic-muted, size: 9.5pt)[Detailed Reference],
      text(size: 9.5pt, fill: anthropic-text, weight: "bold")[Regulatory Compliance Framework Technical Reference v2.0],
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
        *This is a 3--5 minute summary.* It summarizes the architecture alignment of the PLE-based recommendation *reference system* (Independent Research) with the FSC Financial-Sector AI Guideline (effective 2026-06-22) 7 principles. *It is not a compliance attestation*; implementation status is distinguished honestly in three states --- #s-live operational / #s-part implemented, not yet wired / #s-gap absent. For implementation details, code-level architecture, and clause-by-clause mappings, please refer to the 'Regulatory Compliance Framework Technical Reference.'
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
  [Regulation], [Key Requirements], [Implementation artifacts / wiring state], [Status],

  [*Financial-Sector\ AI Guideline*\
  (eff. 2026-06-22,\ 7 principles)],
  [Governance, legality, auxiliary nature\
  reliability, financial stability, good faith, security\
  + risk-assessment system (low/medium/high)],
  [Training performance metrics, promotion human gate, PII hash are #s-live\
  Risk assessment, fairness, explanation, kill switch, HITL are #s-part (depend on deployment wiring)\
  Governance bodies, conflict-of-interest mechanisms are #s-gap (organizational/unimplemented)],
  [#status-tag("partially-wired")],

  [*Korea AI\ Basic Act*\
  (eff. 2026-01-22)],
  [Prior notification of AI use (Art. 31)\
  High-impact AI duties (Art. 34)\
  Impact assessment (Art. 35)],
  [AI disclosure module, `KoreanFRIAAssessor` (7-dim)\
  Proactively designed to high-impact AI level\
  Periodic impact assessment is an organizational process],
  [#status-tag("partially-wired")],

  [*EU AI Act*\
  (High-risk: 2027.12)],
  [Transparency/information (Art. 13)\
  Human oversight (Art. 14)\
  Accuracy/robustness/security (Art. 15)],
  [Model cards + reason text, `AnnexIVMapper`\
  Promotion human gate (auto\_promote=false)\
  Drift detection + prompt injection defense],
  [#status-tag("partially-wired")],

  [*Personal Information\ Protection Act*\
  (automated-decision refusal)],
  [Right to refuse automated decisions (Art. 37-2)\
  Pseudonymized data (Art. 28-2)\
  Right to erasure (Art. 17)],
  [Opt-out hook wired into serving (#s-live)\
  SHA-256 one-way hash + per-domain salt\
  30-day PII retention + encrypted-deletion policy],
  [#status-tag("operational")],

  [*Financial Consumer\ Protection Act,*\
  Credit Information Act],
  [Suitability principle (FCPA §17)\
  Duty to explain (FCPA §19)\
  Automated-evaluation explanation (CIA §36-2)],
  [Suitability filter (fail-closed when not assessed)\
  Feature reverse-mapping + reason generation\
  §36-2 explanation elements structured (depends on deployment wiring)],
  [#status-tag("partially-wired")],
)
]

#align(right)[
  #text(size: 8pt, fill: txt-sub)[
    #status-tag("operational") operational #h(6pt) #status-tag("partially-wired") implemented, not wired #h(6pt) #status-tag("absent") absent / organizational decision\
    #h(1fr) (since multiple controls are mixed within one regulation row, the row tag is the representative state; see the middle column for the mixed #s-live #s-part #s-gap detail)
  ]
]

#pagebreak()

// ═══════════════════════════════════════════════════════════
//  3. MRM Lifecycle
// ═══════════════════════════════════════════════════════════

= Model Risk Management (MRM) Lifecycle

A 5-stage model lifecycle is designed, integrating SR 11-7 (US Fed/OCC), NIST AI RMF, and the Financial-Sector AI Risk Management Framework (AI RMF) delegated by the guideline.

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

#card(title: "Key Safeguard: Offline Champion-Challenger Gate + Audit Chain", accent: anthropic-accent)[
  Stage 3 approval runs automatically through the `ModelCompetition.evaluate` offline gate: only challengers that improve the primary metric by at least `min_improvement` (default 0.5%) with no secondary metric degrading beyond `max_degradation` (default 2%) and no outstanding fidelity failures are promoted. Bootstrap and emergency rollback are handled through the `--force-promote` operator override. Every decision (`bootstrap` / `promote` / `reject` / `force_promote`) is written by `AuditLogger.log_model_promotion` to an HMAC-signed, hash-chained S3 WORM audit log, producing a non-repudiable record of who promoted what and why. The AI Risk Management Committee reviews the promotion outcome and the audit chain post-hoc.
]

// ═══════════════════════════════════════════════════════════
//  4. Key Safeguards Summary
// ═══════════════════════════════════════════════════════════

= Key Safeguards Summary

#card(title: "Before reading --- status notation in this section", accent: anthropic-accent)[
  #text(size: 9pt)[The safeguards below are controls whose *modules and logic are implemented*. Because this system has no production deployment and uses no production data, many are in an *implemented-but-not-yet-wired (#s-part)* state, and only some run as *operational (#s-live)* on the training/ingestion path. Phrases like "daily measurement" or "enforced on every prediction" express the *intended behavior once wired*, not an assertion of current operational effect.]
]

== Fairness Monitoring

Three fairness metrics (DI/SPD/EOD, corresponding to the guideline's Parity metrics) are *implemented to measure daily* across *5 protected attributes* (age group, gender, region type, income quintile, lifecycle stage) --- #s-part, active once the prediction log loads protected attributes and outcome labels and is wired into serving.

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

Four dedicated compliance components are designed to enforce regulatory obligations before a prediction is served (#s-part --- applied to every prediction once the deployment serving path is unified and wired to `RecommendationService`/`lambda_handler`):

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
  [Request explanation], [Feature reverse-mapping + natural language recommendation reasons; internal SLA reply within 10 days (statutory 30 days, Enforcement Decree §44-3(5))],
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
  [Compliance Matrix -- guideline], [7-Principle Mapping -- Guideline Check-Items and System Response], [Ch. 2],
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
  Monthly/quarterly governance reports are *implemented to be auto-generated* (#s-part --- operates once real-data supply and schedule wiring are in place). They comprise *9 sections*: fairness summary, drift summary, incident status, model change history, kill switch history, recommendation quality, risk trends, audit store summary, and executive summary. A 36-item regulatory compliance registry is designed to run a full quarterly check.
]

// ============================================================
== Bedrock Data Protection

Customer data is processed exclusively within the AWS Region (ap-northeast-2) during LLM invocation. Bedrock never transmits input/output data to model providers and never uses it for model training. VPC PrivateLink enables invocation without internet traversal, and CloudTrail audits all API calls. This structurally satisfies PIPA cross-border transfer restrictions and FSS data governance requirements.

// ============================================================
== On-Premises Environment

In air-gapped environments, all processing occurs locally with zero external data transfer. Uses Exaone 3.5 (Apache 2.0) + Qwen 14B (Apache 2.0) open-source models, maintaining the same regulatory compliance framework (checklists, audit logs, governance reports) as AWS.

// ============================================================
= Ops/Audit Agents

Practical enforcement of regulatory compliance is handled by two autonomous diagnostic agents. OpsAgent monitors pipeline performance and stability; AuditAgent audits fairness, regulatory compliance, and recommendation reason quality. 3-agent consensus with minority report preservation mitigates hallucination risk: a single FAIL vote escalates (a fail-safe that treats a missed risk as costlier than a false alarm) and every dissent is retained for audit traceability. All diagnostic results accumulate in a case store for regulatory audit evidence.

Detailed design: Design Document 11
