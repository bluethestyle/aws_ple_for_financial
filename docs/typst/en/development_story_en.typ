// ─────────────────────────────────────────────
// Development Story: Building a Next-Generation Recommendation System with AI Agent Teams
// Typst Web App Compatible — Anthropic Design System
// ─────────────────────────────────────────────

#let anthropic-bg = rgb("#F0EFEA")
#let anthropic-text = rgb("#141413")
#let anthropic-accent = rgb("#CC785C")
#let anthropic-muted = rgb("#6B7280")
#let anthropic-rule = rgb("#D1D5DB")

// ── Page setup ──
#set page(
  paper: "a4",
  margin: (top: 2.5cm, bottom: 2.5cm, left: 2.5cm, right: 2.5cm),
  fill: anthropic-bg,
  header: context {
    if counter(page).get().first() > 1 [
      #set text(size: 7.5pt, font: "New Computer Modern", fill: anthropic-muted, tracking: 0.12em)
      #smallcaps[Development Story]
      #h(1fr)
      #smallcaps[Building a Next-Gen Recommendation System with AI Agent Teams]
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

// ── Base text ──
#set text(
  font: "New Computer Modern",
  size: 10pt,
  fill: anthropic-text,
  lang: "en",
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
#let section-break() = {
  v(0.4cm)
  align(center)[
    #line(length: 30%, stroke: 0.5pt + anthropic-rule)
  ]
  v(0.4cm)
}

#let info-box(title, body) = {
  set par(first-line-indent: 0pt)
  v(0.15cm)
  block(
    width: 100%,
    stroke: (left: 2pt + anthropic-accent),
    inset: (left: 14pt, right: 14pt, top: 10pt, bottom: 10pt),
  )[
    #text(size: 11pt, fill: anthropic-text, weight: "bold")[#title]
    #v(4pt)
    #text(size: 10pt, fill: anthropic-text)[#body]
  ]
  v(0.15cm)
}

#let quote-box(body) = {
  set par(first-line-indent: 0pt)
  v(0.15cm)
  block(
    width: 100%,
    stroke: (left: 2pt + anthropic-accent),
    inset: (left: 14pt, right: 14pt, top: 8pt, bottom: 8pt),
  )[
    #text(size: 10pt, fill: anthropic-muted, style: "italic")[#body]
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
  )[#upper[Development Story]]
  #v(0.5cm)

  #text(
    size: 26pt,
    fill: anthropic-text,
    weight: "bold",
  )[Building a Next-Generation]
  #v(0.1cm)
  #text(
    size: 26pt,
    fill: anthropic-text,
    weight: "bold",
  )[Recommendation System with AI Agent Teams]
  #v(0.4cm)

  #line(length: 40%, stroke: 1pt + anthropic-accent)
  #v(0.3cm)

  #text(
    size: 13pt,
    fill: anthropic-muted,
    style: "italic",
  )[One Consumer GPU, a Team of Three, and the Record of AI Collaboration]
]

#v(2cm)

#align(center)[
  #block(
    width: 70%,
    inset: (x: 1.5cm, y: 1cm),
  )[
    #set par(first-line-indent: 0pt)
    #text(size: 10pt, fill: anthropic-text, style: "italic")[
      "AI writes the code,\ but design decisions remain with humans."
    ]
    #v(0.5cm)
    #text(size: 9pt, fill: anthropic-muted)[
      This document records the journey of building an 18-task, 7-expert\
      PLE+adaTT recommendation system — without infrastructure budget,\
      using a single consumer GPU and a team of AI agents.
    ]
  ]
]

#v(1fr)

#align(center)[
  #text(size: 10pt, fill: anthropic-muted, tracking: 0.15em)[AIOps PLE Platform]
  #v(0.3cm)
  #line(length: 30%, stroke: 0.3pt + anthropic-rule)
]

#pagebreak()


// ═══════════════════════════════════════════════
// CONTENT
// ═══════════════════════════════════════════════

#set page(
  header: context {
    if counter(page).get().first() > 1 [
      #set text(size: 7.5pt, font: "New Computer Modern", fill: anthropic-muted, tracking: 0.12em)
      #smallcaps[Development Story]
      #h(1fr)
      #smallcaps[Building a Next-Gen Recommendation System with AI Agent Teams]
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


= Project Background

== Team Composition and Constraints

The project team consisted of three people: one data scientist serving as PM and two engineers. There was no dedicated infrastructure budget, and the only GPU available for development was a single consumer-grade RTX 4070 (12GB VRAM) installed in a local PC.

#info-box(
  [Constraint Summary],
  [
    • *Team size*: 3 members (PM/Data Scientist 1 + Engineers 2) \
    • *Infrastructure*: No dedicated GPU server. One RTX 4070 12GB (local) \
    • *Budget*: No infrastructure procurement budget. AWS SageMaker spot instances used \
    • *Objective*: Build a next-generation system to replace the existing ALS-based recommendation system
  ],
)

== The Reality of Infrastructure Constraints

Support from the organization was virtually nonexistent. Requests for GPU resources were met with "there is nothing we can do," and all project-related expenses --- AI tool subscriptions (Claude Code, Gemini, Cursor), peripherals, and meals --- were covered personally by the PM.

The data collection environment was equally challenging. Requests to migrate to Spark or Impala were denied, forcing the team to work within a HIVE-based bottleneck environment. To overcome this, parallel query logic was designed from scratch to maximize network bandwidth utilization.

The workspace was an unventilated area adjacent to the server room, without adequate cooling.

The two team members were not formally contracted employees but youth advisory supporters --- recent graduates participating in the project while preparing for employment.

All of these constraints ultimately reinforced the design philosophy: "When available resources are extremely limited, architectural efficiency and tool selection become decisively important."

== Project Objective

The existing financial product recommendation system was based on ALS (Alternating Least Squares) collaborative filtering. The goal was to replace it with a multi-task deep learning recommendation system built on PLE (Progressive Layered Extraction) + adaTT (Adaptive Task Transfer) architecture. The system processes 18 tasks through 7 expert networks, explicitly modeling inter-task relationships.

== Architecture Decision Journey

Arriving at the final architecture required exploring and rejecting several candidates.

The legacy system was ALS-based collaborative filtering. The first alternative considered was the *Black-Litterman model*. Originating from portfolio theory, this model combines expert views with market equilibrium via Bayesian updating. However, it was impossible to decompose "how much each model contributed" from the blended posterior distribution. In financial practice, recommendations must be explained to customers, branch staff, and regulators alike --- Bayesian updating rendered individual model contributions opaque, making business explanation impossible.

The second alternative was a *model ensemble* --- training N independent models and combining their outputs. This introduced N management points, N times the serving cost, and explanations like "MLP \#3 contributed 28%" were meaningless in a business context. Both cost and explainability were problematic.

A critical reframing emerged from this process: rather than combining experts *outside* a model, what if we combined them *inside* a single model? This was the turning point toward PLE's heterogeneous expert architecture.

Task group design also involved trial and error. An initial attempt with *GMM cluster subheads* --- K clusters $times$ T tasks = $K times T$ complexity explosion --- was abandoned in favor of 4 Financial DNA task groups (product holding probability, next product, customer value, churn risk), stabilizing the structure.

#section-break()


= Organizing AI Agents

== Phase-by-Phase AI Tool Deployment

Different AI tools were strategically deployed at each stage of the project. The key was leveraging each tool's strengths through deliberate role assignment.

=== Phase A: Ideation (Gemini)

Gemini was used for initial concept exploration and brainstorming. Its broad knowledge base proved effective for rapidly scanning architecture candidates and comparing the pros and cons of various approaches.

=== Phase B: Technical Validation (Claude Opus)

Claude Opus was employed to develop ideation results into concrete architectures. It was focused on tasks requiring technical depth, such as loss function design demanding mathematical rigor, data leakage verification, and normalization pipeline design.

=== Phase C: Code Environment Setup (Cursor)

GitHub-based code environment configuration, project structure design, and initial boilerplate generation were performed with Cursor. Its strength lay in fast code navigation and refactoring within an integrated IDE environment.

=== Phase D: Parallel Implementation (Claude Code — Opus/Sonnet)

In the full implementation phase, each team member served as the "team lead" for their AI agents. Opus and Sonnet were operated in parallel within the Claude Code environment to implement different modules simultaneously.

#v(0.3cm)

#set par(first-line-indent: 0pt)
#{
  let header-cell(body) = table.cell(
    fill: anthropic-text,
    inset: (x: 10pt, y: 7pt),
  )[#align(center)[#text(size: 10pt, fill: anthropic-bg, weight: "bold")[#body]]]

  let body-cell(body) = table.cell(
    inset: (x: 10pt, y: 7pt),
  )[#text(size: 9pt, fill: anthropic-text)[#body]]

  let alt-cell(body) = table.cell(
    fill: white,
    inset: (x: 10pt, y: 7pt),
  )[#text(size: 9pt, fill: anthropic-text)[#body]]

  table(
    columns: (0.8fr, 1.2fr, 1.5fr),
    stroke: 0.4pt + anthropic-rule,
    align: left + horizon,
    header-cell[Phase], header-cell[AI Tool], header-cell[Role],
    body-cell[A. Ideation], body-cell[Gemini], body-cell[Concept exploration, architecture candidate scanning, brainstorming],
    alt-cell[B. Technical Validation], alt-cell[Claude Opus], alt-cell[Mathematical verification, loss design, leakage analysis, architecture refinement],
    body-cell[C. Environment Setup], body-cell[Cursor], body-cell[GitHub structure, boilerplate, IDE-based refactoring],
    alt-cell[D. Parallel Implementation], alt-cell[Claude Code (Opus/Sonnet)], alt-cell[Module-level parallel coding, testing, debugging],
  )
}
#set par(first-line-indent: 1.2em)

#section-break()


= Quality Management Strategy

== CLAUDE.md Guardrails

To ensure the quality of AI-agent-generated code, a CLAUDE.md file was placed at the project root to establish guardrails. This file serves as system instructions that the AI agent reads at the start of every session.

#info-box(
  [Core CLAUDE.md Guardrails],
  [
    • *Config-Driven Principle*: All parameters are read from YAML. No hardcoding. \
    • *Separation of Concerns*: Adapter handles only data transformation, Runner handles only the pipeline, train.py handles only training. \
    • *Data Leakage Prevention*: Scaler is fit only on the train split. gap_days mandatory for temporal splits. \
    • *4-Stage Code Review*: Compilation check → Interface contract verification → Hardcoding scan → Separation of concerns audit.
  ],
)

== Memory Bank

AI agents lose context when a conversation session ends. To address this, a "memory bank" system was introduced. Session progress, design decision rationale, and feedback history were managed as structured markdown files, enabling the AI to quickly restore prior context at the start of each new session.

== Interface Contract Verification

The greatest risk when parallel agents simultaneously modify different modules is interface mismatch — where the key names saved by File A differ from the key names read by File B. To prevent this, an "interface contract verification" step was mandatory after all parallel work. Cross-file key/field mapping tables were auto-generated to detect inconsistencies before they reached production.

#section-break()


= Technical Challenges and Solutions

== Three Label Leakage Cases Discovered and Fixed

Abnormally high performance was observed early in model training. Root cause analysis uncovered three instances of label leakage.

#set par(first-line-indent: 0pt)
#block(
  width: 100%,
  stroke: (left: 2pt + anthropic-accent),
  inset: (left: 12pt, y: 6pt),
)[
  #set text(size: 10pt)
  #strong[Leak 1 --- Duplicate Column]: A `has_nba_1` column existed as a perfect duplicate of the label (correlation = 1.0). It had to be EXCLUDED before label re-derivation. \
  #strong[Leak 2 --- File Loading Order]: `ground_truth.parquet` was loaded instead of `benchmark.parquet`. Glob's alphabetical sorting caused the answer file to be selected first. Resolved by moving the file to a subdirectory. \
  #strong[Leak 3 --- Generator Input Contamination]: Generators such as GMM and model_derived were using label columns as input, producing AUC = 1.0. An auto-exclude mechanism for `label_cols` was added.
]
#set par(first-line-indent: 1.2em)

#v(0.15cm)
A LeakageValidator was added as a mandatory pre-training step, and validation rules were explicitly documented in the CLAUDE.md guardrails to prevent recurrence.

== Phase 2 NaN Issue (FP16 Underflow)

NaN loss appeared during Phase 2 of mixed precision (AMP) training. The cause was underflow in the BFloat16 environment. Small gradient values fell outside the FP16 representable range and propagated as NaN.

#info-box(
  [Resolution],
  [
    • Applied `.float()` conversion before `.numpy()` calls in BFloat16 contexts \
    • Simultaneously fixed a subprocess pipe deadlock issue \
    • Optimized GradScaler settings to reduce underflow frequency
  ],
)

== GPU Utilization Optimization (37% to 98%)

Initial training showed GPU utilization of only 37%. The bottleneck was data loading.

#set par(first-line-indent: 0pt)
#block(
  width: 100%,
  stroke: (left: 2pt + anthropic-accent),
  inset: (left: 12pt, y: 6pt),
)[
  #text(size: 10pt, fill: anthropic-text, weight: "bold")[Optimization Steps]
  #v(4pt)
  #text(size: 10pt, fill: anthropic-text)[
    • *batch_size increase*: 512 → 4096 (optimized for 941K data) \
    • *DataLoader tuning*: Adjusted num_workers, pin_memory, prefetch_factor \
    • *Preprocessing separation*: Tensors pre-saved in Phase 0, loaded directly during training \
    • *Result*: GPU utilization 37% → 98%, training time reduced by approximately 3x
  ]
]
#set par(first-line-indent: 1.2em)

== Transition from pandas to DuckDB/cuDF

Processing 941K rows with pandas caused memory usage to spike. The large-scale data processing backend was migrated to DuckDB (CPU columnar) and cuDF (GPU), simultaneously improving memory efficiency and processing speed.

== Docker GPU Passthrough Instability

GPU passthrough via Docker on Windows proved unstable. CUDA version mismatches and driver compatibility issues occurred repeatedly. Ultimately, Docker was abandoned in favor of direct development in a local Python environment.

== torch CPU/CUDA Version Conflict

A conflict arose between the CPU build and CUDA build of torch in the conda environment. Tangled package dependencies caused CUDA to go unrecognized. The conda cache was fully purged and the environment was rebuilt with explicitly specified CUDA versions.

#section-break()


= Design Philosophy: Where Is the Science?

== "The Audience of Persuasion Is Always a Person"

The end consumers of a financial recommendation system are not algorithms but people. Customers ask "why this product?", branch staff must explain recommendation rationale, and regulators verify the model's decision-making process. A single probability value cannot persuade any of these three groups. Therefore, the criterion for every design decision was: "Can we explain the reason?"

== The 2-Axis Decomposition Framework

The core structure of the architecture is based on a 2-axis decomposition: *Financial DNA* (who is the customer?) $times$ *Data Modality* (what form is the data?). The Financial DNA axis consists of 4 task groups: product holding probability, next product, customer value, and churn risk. The Data Modality axis is composed of heterogeneous experts corresponding to 5 feature types: state, snapshot, time-series, hierarchy, and item. The intersection of these two axes defines the entire model's learning structure.

== The Expert Collapse Problem and the Need for Heterogeneous Experts

When homogeneous MLP experts (e.g., identically structured 3-layer MLPs) are used, *expert collapse* occurs during training --- the gating network selects only one expert while the rest go effectively unused. This problem has been confirmed in large-scale experiments at Pinterest and Kuaishou. Structurally different heterogeneous experts (LightGCN, Causal OT, TDA, GMM, etc.) operate on different input spaces with different computational methods, making it impossible for them to converge to identical representations --- structurally preventing collapse.

== The Dual Role of Features: Prediction Material and Explanation Vocabulary

Features are not merely inputs for prediction performance; they also serve as vocabulary for explaining recommendation rationale. Even a feature with marginal AUC contribution holds irreplaceable value as explanation vocabulary if it enables statements like "this customer's consumption entropy is high, indicating diverse product experience" at the branch level. For this reason, features were never removed solely based on predictive performance criteria.

== From Economics to Data Science

The design philosophy of this project originates from the PM's intellectual journey. Trained in economics and decision science, then moving through financial engineering to data science, a fundamental question emerged: *"Where is the science in data-driven methodology?"*

Economics has accumulated centuries of scientific methodology --- hypothesis formulation, theoretical frameworks, falsifiability. But economics itself borrowed tools from other disciplines: general equilibrium theory drew from thermodynamic equilibrium in physics, game theory from combinatorics in mathematics, econometrics from statistics. Economics is not uniquely scientific; rather, *cross-disciplinary tool transfer is a universal pattern of scientific progress*. Many Nobel laureates in economics came from physics and mathematics --- Samuelson (thermodynamics to economic equilibrium), Black-Scholes (heat equation to option pricing), Nash (fixed-point theorem to game theory), Mandelbrot (fractals to financial volatility). Economics' most powerful tools came from other sciences.

But moving from financial engineering to data science, scientific rigor gradually faded. Machine learning models were at least mathematically transparent --- OLS minimization, SVM margin maximization, decision tree information gain --- one could explain _why_ they work.

Deep learning changed this. "Neural networks" is a metaphorical name rather than a deep study of actual neural architecture, and the answer to "why does this weight have this value?" is merely "the data trained it that way." Science requires explanation. An entire field --- philosophy of science --- emerged to discuss falsifiability, paradigm shifts, and degenerating research programs. Current deep learning approaches felt more like _engineering than science_.

== Structural Isomorphism: Bringing Science Back

The answer to this question was _structural isomorphism_.

Humanity has already discovered powerful scientific methodologies across disciplines over centuries. Chemical kinetics, epidemiology, information theory, topology --- each represents frontier knowledge about what insights and causal relationships can be extracted from given phenomena.

If we properly recognize the structure of our problem (understanding financial customer behavior), we can import solutions from other disciplines that have already solved structurally equivalent problems. Just as Shannon brought Boltzmann's thermodynamic entropy to information theory, and Black-Scholes derived option pricing from the heat diffusion equation.

This is the fundamental motivation behind introducing features from 11 academic disciplines and designing heterogeneous experts specialized in each discipline's mathematical tools. Not "let's create many features," but *"what scientific questions should we ask?"* was the starting point of design.

== The Place of Scientific Methodology in Recommendation

This project demonstrates that recommendation systems can be grounded in _scientific understanding_ rather than mere pattern matching. Not every recommendation needs to end at "similar people bought this" (correlation); it can reach "this customer's consumption dynamics indicate this product is appropriate" (causal explanation).

Pearl's causal inference, Friedman's Permanent Income Hypothesis, Boltzmann's statistical mechanics --- these scientists answered "why?" in their respective fields. Our heterogeneous expert architecture brings their tools to financial recommendation through structural isomorphism, attempting to endow recommendation systems with scientific explainability.

#section-break()


= Key Lessons

== "AI Writes the Code, but Design Decisions Remain with Humans"

AI agents generate code at remarkable speed, but architecture decisions, data leakage judgments, and cost optimization strategies depend on human domain knowledge and experience. AI excels at "how," but "why" and "what" must be defined by humans.

#quote-box["The most dangerous moment is right after AI produces 'plausible-looking code.'\ If you stop critically reviewing at that point, technical debt accumulates."]

== "AI Coding Without Guardrails Creates Technical Debt"

When AI is given free rein to code without a CLAUDE.md, hardcoded values proliferate, concerns become entangled, and untestable structures emerge. Guardrails do not restrict AI's productivity — they channel it in the right direction.

== "The Mixture-of-Experts Philosophy Applies to Development Methodology Too"

The core philosophy of the PLE architecture — Mixture of Experts — was applied to the development methodology itself. Gemini specialized in broad exploration, Opus in deep analysis, Cursor in rapid environment setup, and Claude Code in implementation. Assigning roles aligned to each tool's strengths proved more effective than having a single AI tool do everything.

#section-break()


= Results

== System Built

#set par(first-line-indent: 0pt)
#block(
  width: 100%,
  inset: (x: 14pt, y: 12pt),
)[
  #grid(
    columns: (1fr, 1fr),
    gutter: 12pt,
    [
      #text(size: 11pt, fill: anthropic-text, weight: "bold")[Recommendation System]
      #v(4pt)
      #text(size: 10pt, fill: anthropic-text)[
        • 18-task multi-task learning \
        • 7-expert PLE network \
        • adaTT adaptive inter-task transfer \
        • Uncertainty weighting (Kendall et al.) \
        • Three logit transfer modes
      ]
    ],
    [
      #text(size: 11pt, fill: anthropic-text, weight: "bold")[Infrastructure and Experimentation]
      #v(4pt)
      #text(size: 10pt, fill: anthropic-text)[
        • 24 ablation scenarios \
        • AWS SageMaker spot instances \
        • Phase 0 (CPU) + Phase 1\~2 (GPU) separation \
        • Config-driven pipeline architecture
      ]
    ],
  )
]
#set par(first-line-indent: 1.2em)

== Documentation

A total of nine technical documents were produced throughout the project: architecture overview, pipeline guide, expert details, feature reference, PLE+adaTT reference, Causal OT reference, distillation reference, temporal reference, and regulatory framework — all authored in Typst.

== Papers

Two papers are being prepared for submission to arXiv, covering the experience of building a large-scale multi-task recommendation system under resource constraints, and the development methodology of a small team leveraging AI agents. This may be the first arXiv publication by a practitioner at a Korean financial institution.

== Expert Specialization Revealed by Ablation

Analysis across 24 ablation scenarios clearly demonstrated task-type-specific expert specialization. LightGCN showed the greatest contribution for multiclass tasks (next product prediction), while the Causal expert excelled at regression tasks (customer value estimation). This empirically validates the heterogeneous expert design.

== Evaluation Metric Framework

Gold standard metrics were established by task type: AUC for binary classification, Macro F1 for multiclass classification, and MAE for regression. This prevents the error of comparing all tasks with a single metric and ensures rigorous evaluation aligned with each task's characteristics.

#section-break()


= Future Plans

== Academic and Industry Publications

- *arXiv Paper Submission*: 2 papers (system architecture paper + AI agent development methodology paper)
- *Anthropic Case Study Submission*: A case study on building a financial AI system using Claude Code
- *GARP Submission*: A paper combining the FRM credential with an AI risk management perspective

== Regulatory Engagement

- *FSS AI Basic Act Compliance Review Request*: As enforcement decrees and guidelines for the AI Basic Act are being drafted, the team plans to request a review of this system's explainability framework at the appropriate timing.

== Follow-up Work

- *On-premises Production Data Results*: Performance results from actual production data to be added as paper supplements
- *Public GitHub Repository*: A sanitized version of the codebase with organizational information removed will be published as a public repository

#v(1cm)
#section-break()

#align(center)[
  #text(size: 9pt, fill: anthropic-muted, style: "italic")[
    This project was completed not by overcoming a "lack of resources"\ but by "redefining resources."\
    It demonstrates that one consumer GPU combined with AI agents\ can substitute for dedicated infrastructure.
  ]
]
