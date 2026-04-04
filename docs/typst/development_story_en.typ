// ─────────────────────────────────────────────
// Development Story: Building a Next-Generation Recommendation System with AI Agent Teams
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
      #smallcaps[Development Story]
      #h(1fr)
      #smallcaps[Building a Next-Gen Recommendation System with AI Agent Teams]
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
  lang: "en",
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
  )[Building a Next-Generation]
  #v(0.1cm)
  #text(
    size: 26pt,
    fill: navy,
    weight: "bold",
  )[Recommendation System with AI Agent Teams]
  #v(0.3cm)

  #line(length: 20%, stroke: 0.5pt + light-rule)
  #v(0.2cm)

  #text(
    size: 13pt,
    fill: burgundy,
    style: "italic",
  )[One Consumer GPU, a Team of Three, and the Record of AI Collaboration]

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
      "AI writes the code,\ but design decisions remain with humans."
    ]
    #v(0.5cm)
    #text(size: 9pt, fill: muted)[
      This document records the journey of building an 18-task, 7-expert\
      PLE+adaTT recommendation system — without infrastructure budget,\
      using a single consumer GPU and a team of AI agents.
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
      #smallcaps[Development Story]
      #h(1fr)
      #smallcaps[Building a Next-Gen Recommendation System with AI Agent Teams]
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

== Project Objective

The existing financial product recommendation system was based on ALS (Alternating Least Squares) collaborative filtering. The goal was to replace it with a multi-task deep learning recommendation system built on PLE (Progressive Layered Extraction) + adaTT (Adaptive Task Transfer) architecture. The system processes 18 tasks through 7 expert networks, explicitly modeling inter-task relationships.

#ornament()


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
    header-cell[Phase], header-cell[AI Tool], header-cell[Role],
    body-cell[A. Ideation], body-cell[Gemini], body-cell[Concept exploration, architecture candidate scanning, brainstorming],
    alt-cell[B. Technical Validation], alt-cell[Claude Opus], alt-cell[Mathematical verification, loss design, leakage analysis, architecture refinement],
    body-cell[C. Environment Setup], body-cell[Cursor], body-cell[GitHub structure, boilerplate, IDE-based refactoring],
    alt-cell[D. Parallel Implementation], alt-cell[Claude Code (Opus/Sonnet)], alt-cell[Module-level parallel coding, testing, debugging],
  )
}
#set par(first-line-indent: 1.2em)

#ornament()


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

#ornament()


= Technical Challenges and Solutions

== Three Label Leakage Cases Discovered and Fixed

Abnormally high performance was observed early in model training. Root cause analysis uncovered three instances of label leakage.

#set par(first-line-indent: 0pt)
#block(
  width: 100%,
  stroke: (left: 2pt + burgundy),
  inset: (left: 12pt, y: 6pt),
)[
  #set text(size: 10pt)
  #strong[Leak 1]: The last timestep in sequence data overlapped with the label period \
  #strong[Leak 2]: Aggregate features from future time points were included in the input \
  #strong[Leak 3]: Missing gap_days in the temporal split leaked adjacent-period information
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
  fill: dark-cream,
  stroke: 0.3pt + light-rule,
  radius: 3pt,
  inset: (x: 12pt, y: 10pt),
)[
  #text(size: 10pt, fill: navy, weight: "bold")[Optimization Steps]
  #v(4pt)
  #text(size: 10pt, fill: ink)[
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

#ornament()


= Design Philosophy: Where Is the Science?

== From Economics to Data Science

The design philosophy of this project originates from the PM's intellectual journey. Trained in economics and decision science, then moving through financial engineering to data science, a fundamental question emerged: *"Where is the science in data-driven methodology?"*

Economics has accumulated centuries of scientific methodology --- hypothesis formulation, theoretical frameworks, falsifiability. Friedman's Permanent Income Hypothesis, Arrow-Debreu general equilibrium, Nash equilibrium --- these are scientific theories that _explain_ observable phenomena.

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

#ornament()


= Key Lessons

== "AI Writes the Code, but Design Decisions Remain with Humans"

AI agents generate code at remarkable speed, but architecture decisions, data leakage judgments, and cost optimization strategies depend on human domain knowledge and experience. AI excels at "how," but "why" and "what" must be defined by humans.

#quote-box["The most dangerous moment is right after AI produces 'plausible-looking code.'\ If you stop critically reviewing at that point, technical debt accumulates."]

== "AI Coding Without Guardrails Creates Technical Debt"

When AI is given free rein to code without a CLAUDE.md, hardcoded values proliferate, concerns become entangled, and untestable structures emerge. Guardrails do not restrict AI's productivity — they channel it in the right direction.

== "The Mixture-of-Experts Philosophy Applies to Development Methodology Too"

The core philosophy of the PLE architecture — Mixture of Experts — was applied to the development methodology itself. Gemini specialized in broad exploration, Opus in deep analysis, Cursor in rapid environment setup, and Claude Code in implementation. Assigning roles aligned to each tool's strengths proved more effective than having a single AI tool do everything.

#ornament()


= Results

== System Built

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
      #text(size: 11pt, fill: navy, weight: "bold")[Recommendation System]
      #v(4pt)
      #text(size: 10pt, fill: ink)[
        • 18-task multi-task learning \
        • 7-expert PLE network \
        • adaTT adaptive inter-task transfer \
        • Uncertainty weighting (Kendall et al.) \
        • Three logit transfer modes
      ]
    ],
    [
      #text(size: 11pt, fill: navy, weight: "bold")[Infrastructure and Experimentation]
      #v(4pt)
      #text(size: 10pt, fill: ink)[
        • 54 ablation scenarios \
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

Two papers are being prepared for submission to arXiv, covering the experience of building a large-scale multi-task recommendation system under resource constraints, and the development methodology of a small team leveraging AI agents.

#v(1cm)
#ornament()

#align(center)[
  #text(size: 9pt, fill: muted, style: "italic")[
    This project was completed not by overcoming a "lack of resources"\ but by "redefining resources."\
    It demonstrates that one consumer GPU combined with AI agents\ can substitute for dedicated infrastructure.
  ]
]
