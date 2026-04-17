// =============================================================================
//  PLE + adaTT Technical Reference — AWS PLE for Financial
//  v1.0 · 2026-04-01
// =============================================================================

// ─────────────────────────── Color Palette ───────────────────────────
#let anthropic-bg = rgb("#F0EFEA")
#let anthropic-text = rgb("#141413")
#let anthropic-accent = rgb("#CC785C")
#let anthropic-muted = rgb("#6B7280")
#let anthropic-rule = rgb("#D1D5DB")

// Legacy aliases for component compatibility
#let navy   = anthropic-text
#let teal   = anthropic-accent
#let amber  = anthropic-accent
#let indigo = anthropic-accent
#let rose   = anthropic-accent
#let slate  = anthropic-muted

// ─────────────────────────── Page Settings ───────────────────────────
#set page(
  paper: "a4",
  margin: (top: 2.5cm, bottom: 2.5cm, left: 2.5cm, right: 2.5cm),
  fill: anthropic-bg,
  header: context {
    if counter(page).get().first() > 1 [
      #set text(size: 7.5pt, font: "New Computer Modern", fill: anthropic-muted, tracking: 0.12em)
      #smallcaps[PLE + adaTT Technical Reference]
      #h(1fr)
      #smallcaps[AWS PLE for Financial · v1.0]
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

// ─────────────────────────── Base Text ──────────────────────────
#set text(font: "New Computer Modern", size: 10pt, fill: anthropic-text, lang: "en")
#set par(justify: true, leading: 0.8em, spacing: 1.5em)

// ─────────────────────────── Code Blocks ───────────────────────────
#show raw.where(block: true): it => {
  block(
    width: 100%,
    inset: 12pt,
    radius: 5pt,
    fill: rgb("#f8fafc"),
    stroke: 0.5pt + anthropic-rule,
    text(size: 8.5pt, it),
  )
}
#show raw.where(block: false): box.with(
  fill: rgb("#f1f5f9"),
  inset: (x: 3pt, y: 1pt),
  outset: (y: 2pt),
  radius: 2pt,
)

// ─────────────────────── Equation Block Spacing ──────────────────────────
#show math.equation.where(block: true): it => {
  v(4pt)
  it
  v(4pt)
}

// ─────────────────────────── Heading Styles ──────────────────────────
#show heading.where(level: 1): it => {
  pagebreak(weak: true)
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

// ───────────────────── Custom Components ────────────────────────────
#let note(title, body, accent: anthropic-accent) = {
  block(
    width: 100%,
    inset: (left: 14pt, right: 14pt, top: 10pt, bottom: 10pt),
    stroke: (left: 2pt + accent),
    [
      #text(weight: "bold", size: 9.5pt, fill: accent)[#title]
      #v(2pt)
      #text(size: 9.5pt)[#body]
    ],
  )
}
#let warn(title, body) = note(title, body, accent: anthropic-accent)
#let eq-highlight(body) = {
  block(
    width: 100%,
    inset: 14pt,
    stroke: (left: 2pt + anthropic-accent),
    body,
  )
}
#let chip(label, color: anthropic-accent) = {
  box(
    inset: (x: 6pt, y: 2pt),
    radius: 3pt,
    fill: color.lighten(88%),
    text(8pt, weight: "bold", fill: color.darken(10%))[#label],
  )
}
#let dim-label(body) = text(size: 8.5pt, fill: anthropic-muted)[#body]

// ───────────────── Table Style Function ────────────────────────
#let styled-table(cols, ..args) = {
  set text(size: 9pt)
  table(
    columns: cols,
    inset: 8pt,
    stroke: 0.4pt + anthropic-rule,
    fill: (_, y) => if y == 0 { anthropic-accent.lighten(88%) } else if calc.odd(y) { luma(252) },
    ..args,
  )
}


// =====================================================================
//  Cover Page
// =====================================================================

#set page(header: none, footer: none)

#v(3cm)
#align(center)[
  #text(
    size: 10pt,
    fill: anthropic-muted,
    tracking: 0.5em,
    weight: "regular",
  )[#upper[Tech Reference]]
  #v(0.5cm)

  #text(size: 26pt, fill: anthropic-text, weight: "bold")[
    PLE + adaTT\
    Technical Reference
  ]

  #v(0.3cm)
  #line(length: 30%, stroke: 0.5pt + anthropic-rule)
  #v(0.3cm)

  #text(size: 11pt, fill: anthropic-muted)[
    Progressive Layered Extraction · Adaptive Task Transfer\
    CGC Gate · GroupTaskExpertBasket · Logit Transfer · 2-Phase Training
  ]

  #v(2cm)

  #grid(
    columns: (1fr, 1fr),
    align(left)[
      #text(10pt, fill: anthropic-muted)[Project] \
      #text(12pt, fill: anthropic-text, weight: "bold")[AWS PLE for Financial] \
      #text(9pt, fill: anthropic-muted)[PLE-Cluster-adaTT Architecture]
    ],
    align(right)[
      #text(10pt, fill: anthropic-muted)[Version] \
      #text(12pt, fill: anthropic-text, weight: "bold")[v1.0] \
      #text(9pt, fill: anthropic-muted)[2026-04-01]
    ],
  )

  #v(1cm)
  #line(length: 100%, stroke: 0.5pt + anthropic-rule)
  #v(0.5cm)

  #block(width: 85%, stroke: (left: 2pt + anthropic-accent), inset: (left: 14pt, right: 14pt, top: 8pt, bottom: 8pt))[
    #text(size: 9.5pt, fill: anthropic-muted)[
      This document describes the core structure of the PLE-Cluster-adaTT architecture.
      It covers the progression from Shared-Bottom to PLE, heterogeneous expert Basket design,
      adaTT gradient-based task transfer, Task Groups structure, training strategies,
      and FP16/GradScaler implementation considerations.
    ]
  ]
]

#v(1fr)
#pagebreak()

#set page(
  header: context {
    if counter(page).get().first() > 1 [
      #set text(size: 7.5pt, font: "New Computer Modern", fill: anthropic-muted, tracking: 0.12em)
      #smallcaps[PLE + adaTT Technical Reference]
      #h(1fr)
      #smallcaps[AWS PLE for Financial · v1.0]
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

#outline(title: [Table of Contents], indent: 1.5em, depth: 3)

#v(1em)

#warn[Design vs. Implementation Note][
  This document is written based on the full-bank design (734D).
  The current Santander benchmark implementation is ~349D raw input (13 feature groups), expanding to 403D after Phase 0 log1p expansion.
]

#block(
  width: 100%,
  inset: 10pt,
  radius: 4pt,
  fill: rgb("#fff3cd"),
  stroke: (left: 3pt + rgb("#ffc107")),
)[
  *adaTT Scale Note (2026-04-15).* adaTT degrades at 13-task scale. Loss-level transfer
  (156 task pairs) undoes PLE's representation-level separation, causing gradient interference.
  GradSurgery (PCGrad task-type projection) was tested as a gradient-level alternative to adaTT,
  but showed no meaningful advantage over the PLE-only baseline while incurring significant VRAM
  overhead (retained computation graph); GradSurgery was not adopted for production. The production
  configuration disables both adaTT and GradSurgery. This document is retained as a reference for
  the adaTT design.
]


// =====================================================================
//  1. PLE Architecture
// =====================================================================

= 1. PLE Architecture

== 1.1 Motivation for Multi-Task Learning

The AIOps recommendation system must simultaneously predict 13 tasks.
Leveraging shared patterns across tasks dramatically improves data efficiency.
The total loss is defined as a weighted sum:

$ cal(L)_"MTL" = sum_(k=1)^K w_k dot cal(L)_k (f_k (bold(h)_"shared"(bold(x)))) $

$bold(h)_"shared"$ is difficult to overfit to any single task.
Only representations that are simultaneously useful to all tasks survive,
which constitutes *inter-task regularization*.

== 1.2 Progression from Shared-Bottom to PLE

=== Shared-Bottom (Caruana, 1997)

All tasks share a single trunk and then branch into task-specific heads.

$ bold(h) = f_"shared"(bold(x)) quad arrow.r quad hat(y)_k = f_k^"tower"(bold(h)) $

Implementation is simple and parameter-efficient, but
*Negative Transfer* is severe when task correlation is low.

=== MMoE (Ma et al., KDD 2018)

$N$ experts with identical structure are maintained, and task-specific gates determine their weighted combination.

$ bold(h)_k = sum_(i=1)^N g_(k,i) dot f_i^"expert"(bold(x)), quad bold(g)_k = "Softmax"(bold(W)_k^"gate" dot bold(x)) $

Different expert combinations per task are possible, but
the *Expert Collapse* problem arises where all gates select the same experts.

=== PLE (Tang et al., RecSys 2020)

Experts are explicitly separated into *Shared Experts* $cal(E)^s$ and *Task-specific Experts* $cal(E)^k$,
and the CGC gate combines both types of experts at an optimal ratio.

#eq-highlight[
  $ bold(h)_k = sum_(i=1)^(|cal(E)^s|) g_(k,i)^s dot bold(e)_i^s + sum_(j=1)^(|cal(E)^k|) g_(k,j)^k dot bold(e)_j^k $

  #dim-label[
    $bold(e)_i^s$: output of $i$-th Shared Expert, $bold(e)_j^k$: output of $j$-th Task Expert for task $k$ \
    $g_(k,i)^s, g_(k,j)^k$: CGC gate weights (Softmax-normalized)
  ]
]

Problems PLE solves:
+ *Negative Transfer mitigation*: Task-specific Experts learn specialized patterns without interference
+ *Expert Collapse prevention*: Shared/Task Expert roles are naturally separated
+ *Progressive Extraction*: Stacking multiple layers refines information from low-level to high-level

#styled-table(
  (1.2fr, 2fr, 2fr, 2fr),
  [*Category*], [*Shared-Bottom*], [*MMoE*], [*PLE*],
  [Expert Structure],
    [Single shared trunk],
    [N experts fully shared],
    [Shared + Task-specific separated],
  [Gating],
    [None],
    [Task-wise Softmax gate],
    [CGC: Shared + Task Expert combined],
  [Negative Transfer],
    [High],
    [Medium (Expert Collapse)],
    [Low (explicit separation)],
  [Expert Collapse],
    [N/A],
    [High],
    [Low],
)

== 1.3 CGC (Customized Gate Control)

This implementation has two CGC variants with different roles and output shapes.

=== CGCLayer (weighted sum approach)

The CGC from the original PLE paper, where task-wise gates produce a *weighted sum* of Shared + Task Expert outputs.
Output dimension is fixed at `expert_hidden_dim` (independent of the number of experts).

#eq-highlight[
  $ bold(h)_k = sum_(i=1)^N g_(k,i) dot bold(e)_i, quad bold(g)_k = "Softmax"(bold(W)_k^"gate" dot bold(x)) in RR^N $

  #dim-label[
    $N$: total number of Shared + Task Experts \
    $bold(e)_i in RR^(d_"expert")$: output of $i$-th Expert \
    Result: $bold(h)_k in RR^(d_"expert")$ --- fixed dimension independent of the number of experts
  ]
]

=== CGCAttention (block scaling approach)

Outputs of heterogeneous experts are *concatenated* and then each expert block is scaled by task-specific attention weights. Output dimension is the sum of all expert output dimensions.

#eq-highlight[
  $ bold(w)_k = "Softmax"(bold(W)_k dot bold(h)_"shared" + bold(b)_k) in RR^8 $
  $ tilde(bold(h))_(k,i) = w_(k,i) dot bold(h)_i^"expert" quad "for" i = 1, ..., 8 $
  $ bold(h)_k^"cgc" = [tilde(bold(h))_(k,1) || tilde(bold(h))_(k,2) || ... || tilde(bold(h))_(k,8)] in RR^(576) $

  #dim-label[
    $bold(W)_k in RR^(8 times 576)$: gate weight for task $k$ \
    $w_(k,i)$: attention weight task $k$ assigns to Expert $i$ \
    Result: same 576D but with different expert contribution weights per task
  ]
]

In this PLE-Cluster-adaTT implementation, *CGCAttention* (block scaling) is used for 8 heterogeneous Shared Experts. Analogous to Transformer Attention, the relationship between *Query* (gate weight), *Key* (shared representation), and *Value* (Expert output) enables "selectively combining information in proportion to relevance."

=== Initial Bias Setting (domain_experts)

The per-task `domain_experts` config assigns `bias_high=1.0` to preferred experts
and `bias_low=-1.0` to non-preferred ones, guiding expert selection at the start of training.

=== Entropy Regularization

Entropy of the gate distribution is regularized to prevent Expert Collapse:

$ cal(L)_"entropy" = lambda_"ent" dot (- 1/(|cal(T)|)) sum_(k in cal(T)) H(bold(w)_k), quad H(bold(w)_k) = - sum_(i=1)^8 w_(k,i) dot log(w_(k,i)) $

#dim-label[$lambda_"ent" = 0.01$: minimizing negative entropy increases entropy, encouraging distributed expert utilization]

=== Dimension Normalization (v3.3)

Contribution imbalance due to heterogeneous expert dimension asymmetry (128D vs 64D) is corrected:

$ "scale"_i = sqrt("mean\_dim" / "dim"_i), quad "mean\_dim" = (128 + 64 times 7) / 8 = 72.0 $

#dim-label[unified\_hgcn (128D): scale $approx$ 0.750 (attenuation), others (64D): scale $approx$ 1.061 (amplification)]


// =====================================================================
//  2. Heterogeneous Expert Basket
// =====================================================================

= 2. Heterogeneous Expert Basket

== 2.1 Pool/Basket Pattern

Instead of PLE's identically structured experts, this system combines
*8 heterogeneous domain experts* into a Shared Expert Pool.
Each expert interprets the input data from an entirely different mathematical perspective,
and the CGC Gate learns the optimal per-task combination.

#styled-table(
  (1.3fr, 0.9fr, 0.6fr, 2.5fr),
  [*Expert*], [*Input*], [*Output*], [*Role and Irreplaceability*],
  [DeepFM], [normalized 644D], [64D], [Explicitly captures FM $O(n k)$ second-order feature interactions],
  [LightGCN], [precomputed 64D], [64D], [Bipartite graph-based "similar customers" collaborative signal],
  [Unified HGCN], [27D (merchant\_hierarchy, MCC hierarchy)], [128D], [Encodes MCC hierarchical structure in hyperbolic space via Poincaré embeddings],
  [Temporal], [sequence input], [64D], [Mamba+LNN+Transformer temporal pattern ensemble],
  [PersLay], [Persistence Diagram], [64D], [Topological structure of spending patterns (loops, clusters)],
  [Causal], [161D (demographics, products, txn, derived\_temporal, product\_hierarchy, gmm)], [64D], [Directional causal relation extraction via SCM/NOTEARS],
  [Optimal Transport], [127D (demographics, products, txn, derived\_temporal, gmm)], [64D], [Sinkhorn Wasserstein distributional geometry],
  [RawScale], [raw 90D], [64D], [Preserves power-law distribution information before normalization],
)

Combined dimension: $7 times 64 + 1 times 128 = 576$D.

$ bold(h)_"shared" = ["unified\_hgcn"_(128"D") || "perslay"_(64"D") || "deepfm"_(64"D") || "temporal"_(64"D") || "lightgcn"_(64"D") || "causal"_(64"D") || "OT"_(64"D") || "raw\_scale"_(64"D")] $

== 2.2 Expert Selection Criteria

The 8 experts extract *fundamentally different mathematical structures* from the same customer data:

- *DeepFM/Causal/OT*: receive overlapping but distinct feature subsets (Causal: 161D, OT: 127D) and capture different structures — symmetric interactions (FM), asymmetric causality (DAG), distributional distance (Wasserstein)
- *Temporal*: temporal dynamics of sequential data ($[B, 180, 16]$, $[B, 90, 8]$)
- *PersLay*: topological invariants (Betti numbers, persistence)
- *Unified HGCN*: hierarchical relationships in hyperbolic geometry (merchant_hierarchy, MCC hierarchy; not product_hierarchy)
- *LightGCN*: collaborative filtering signal from the customer-merchant graph; also handles product co-holding collaborative filtering via the bipartite graph
- *RawScale*: raw scale/power-law distribution patterns lost during normalization

== 2.3 Expert Routing

Each expert is activated in the `shared_experts` section of the config,
and different inputs are dispatched per expert name in `_forward_shared_experts()`.
When input data is `None`, a *zero tensor fallback* is applied,
and CGC gating automatically reduces that expert's weight.

HMM Triple-Mode is handled through a separate input path:

#styled-table(
  (1fr, 0.8fr, 0.8fr, 2fr),
  [*HMM Mode*], [*Input*], [*Time Scale*], [*Target Tasks*],
  [Journey], [16D], [daily], [has\_nba, cross\_sell\_count, will\_acquire\_\*],
  [Lifecycle], [16D], [monthly], [churn\_signal, product\_stability, segment\_prediction],
  [Behavior], [16D], [monthly], [nba\_primary, next\_mcc, mcc\_diversity\_trend, top\_mcc\_shift, ...],
)

Each mode consists of 10D base state probabilities + 6D ODE dynamics,
projected from 16D to 32D and concatenated with the CGC output:

$ bold(h)_"hmm"^m = "SiLU"("LayerNorm"("Linear"_(16 arrow 32)(bold(x)_"hmm"^m))) $


// =====================================================================
//  3. adaTT (Adaptive Task-aware Transfer)
// =====================================================================

= 3. adaTT --- Adaptive Task-aware Transfer

== 3.1 Motivation: Negative Transfer

When 13 tasks share Shared Expert parameters, gradient conflicts cause
Negative Transfer. Three limitations of fixed-tower MTL:

+ *Unidirectional sharing*: no mechanism to detect or regulate inter-task interference
+ *Unmeasured interaction*: impossible to quantify which task pairs help or harm each other
+ *Temporal change not reflected*: training-stage-wise changes in task relationships cannot be tracked

adaTT resolves these with *selective transfer*, *gradient cosine similarity*, and a *3-Phase schedule*, respectively.

== 3.2 Gradient Cosine Similarity

For two tasks $i$, $j$ with gradients with respect to Shared Expert parameters $theta$,
denoted $bold(g)_i = nabla_theta cal(L)_i$, $bold(g)_j = nabla_theta cal(L)_j$:

#eq-highlight[
  $ cos(theta_(i,j)) = frac(bold(g)_i dot bold(g)_j, ||bold(g)_i|| dot ||bold(g)_j||) $

  #dim-label[
    $bold(g)_i in RR^d$: flattened gradient vector of task $i$ \
    Result range: $[-1, 1]$ --- positive = positive transfer, negative = negative transfer
  ]
]

*Reasons for using cosine similarity*:
+ *Scale invariance*: compares directions purely without being affected by per-task loss magnitude differences
+ *Interpretability*: normalized to $[-1, 1]$ range for intuitive interpretation
+ *Efficient computation*: L2-normalize gradient matrix $bold(G) in RR^(n times d)$ then compute all $n^2$ similarities via a single GEMM as $hat(bold(G)) hat(bold(G))^top$

```python
norms = grad_matrix.norm(dim=1, keepdim=True).clamp(min=1e-8)
normalized = grad_matrix / norms
affinity = torch.mm(normalized, normalized.t())  # [n_tasks, n_tasks]
```

== 3.3 EMA Stabilization

Gradients from a single batch are noisy, so EMA smoothing is applied:

#eq-highlight[
  $ bold(A)_t = alpha dot bold(A)_(t-1) + (1 - alpha) dot cos(theta_t) $

  #dim-label[
    $alpha = 0.9$: effective window $approx 1/(1 - alpha) = 10$ observations \
    Signal processing perspective: IIR first-order low-pass filter $H(z) = (1-alpha)/(1-alpha z^(-1))$
  ]
]

After EMA update, `.clamp(-1.0, 1.0)` must be applied --- accumulated floating-point errors
can push cosine similarity outside $[-1, 1]$, potentially causing NaN in downstream operations.

== 3.4 Transfer-Enhanced Loss

adaTT-enhanced loss for each task $i$:

#eq-highlight[
  $ cal(L)_i^"adaTT" = cal(L)_i + lambda dot sum_(j != i) w_(i arrow.r j) dot cal(L)_j $

  #dim-label[
    $cal(L)_i$: original loss for task $i$ \
    $lambda = 0.1$: proportion of other tasks' opinions incorporated (10%) \
    $w_(i arrow.r j)$: transfer weight from task $j$ to task $i$ (softmax-normalized)
  ]
]

The system also supports sigmoid gating (NeurIPS 2024, Nguyen et al.). Sigmoid gates allow independent expert contribution, eliminating harmful inter-expert competition in heterogeneous expert environments. Configuration: `gate_type: "sigmoid"` (pipeline.yaml or HP).


Effect on gradients:

$ nabla_theta cal(L)_i^"adaTT" = nabla_theta cal(L)_i + lambda sum_(j != i) w_(i arrow.r j) nabla_theta cal(L)_j $

The second term is a *correction vector* that steers shared parameters toward directions favorable to multiple tasks.

== 3.5 Transfer Weight Computation (4-stage pipeline)

#eq-highlight[
  $ bold(R) &= (bold(W) + bold(A)) dot (1-r) + bold(P) dot r \
    bold(R)_(i,j) &arrow.l 0 quad "if" bold(A)_(i,j) < tau_"neg" \
    bold(R)_(i,i) &= 0 \
    w_(i arrow.r j) &= "softmax"(bold(R)_(i,j) / T) $

  #dim-label[
    $bold(W)$: learnable transfer weights (`nn.Parameter`, initialized to 0) \
    $bold(A)$: EMA affinity matrix \
    $bold(P)$: Group Prior matrix (domain knowledge) \
    $r$: Prior blend ratio (annealed from 0.5 $arrow.r$ 0.1) \
    $tau_"neg" = -0.1$: threshold for blocking negative transfer \
    $T = 1.0$: Softmax temperature
  ]
]

*Meaning of each stage*:
+ Sum learnable weights $bold(W)$ with observed affinity $bold(A)$, blend with domain prior $bold(P)$
+ Zero out paths where affinity falls below the threshold (negative transfer blocking)
+ Exclude self-transfer (diagonal set to 0)
+ Generate probability distribution via Softmax --- scale is constant regardless of task count since values sum to 1

`max_transfer_ratio = 0.5`: Transfer loss cannot exceed 50% of the original loss.

== 3.6 3-Phase Schedule

#styled-table(
  (0.8fr, 1.5fr, 2fr, 2fr),
  [*Phase*], [*Duration*], [*Behavior*], [*Purpose*],
  [1: Warmup], [epoch 0 ~ `warmup_epochs`], [affinity computation only, no transfer loss applied], [Accumulate stable affinity data],
  [2: Dynamic], [`warmup_epochs` ~ `freeze_epoch`], [transfer active, Prior blend annealing], [Learn and apply task relationships],
  [3: Frozen], [`freeze_epoch` ~ end], [transfer weights frozen (`detach`)], [Stabilize fine-tuning, remove gradient overhead],
)

#warn[Validation: freeze_epoch > warmup_epochs][
  If Phase 2 is completely skipped, the learned affinity is never reflected in transfer,
  rendering adaTT usage meaningless. Validated at initialization with an early `ValueError`.
]

#note[adaTT Epoch Budget][
  With 10-epoch runs (warmup=3, freeze=8), adaTT has a limited dynamic window of only 5 epochs (Phase 2: epochs 3–8). This is often insufficient for affinity to converge. *20-epoch runs are recommended for meaningful adaTT evaluation* — they provide a 12-epoch dynamic window (warmup=3, freeze=15) which allows affinity matrices to stabilize before freezing.
]

== 3.7 Group Prior

The Prior blend ratio decreases linearly as training progresses:

$ r(e) = r_"start" - (r_"start" - r_"end") dot min(frac(e - e_"warmup", e_"freeze" - e_"warmup"), 1.0) $

#dim-label[$r_"start" = 0.5 arrow.r r_"end" = 0.1$: rely on domain knowledge early in training, trust observed affinity later]

Bayesian interpretation: transition from Prior ($bold(P)$) to Posterior (observation-based $bold(A)$).
Rely on prior when data is scarce early on, then follow likelihood as data accumulates.

== 3.8 Negative Transfer Detection and Blocking

$ bold(R)_(i,j) = cases(
  bold(R)_(i,j) & "if" bold(A)_(i,j) > tau_"neg",
  0 & "otherwise"
) $

Threshold $tau_"neg" = -0.1$ (not zero, because):
weak negative correlation may be noise and is therefore tolerated;
only clearly opposing gradient directions are blocked.
Setting to 0 would over-prune paths and weaken adaTT effectiveness.

Diagnostic API: `detect_negative_transfer()` returns negative transfer pairs in the form
`{"churn_signal": ["will_acquire_lending", "nba_primary"]}`.

== 3.9 Analogy to Attention Mechanism

adaTT can be interpreted as *Attention in task space*:

#styled-table(
  (1fr, 2fr, 2fr),
  [*Role*], [*Transformer Self-Attention*], [*adaTT Task Transfer*],
  [Query], [current token's query], [current task's gradient direction],
  [Key], [other tokens' response capacity], [other tasks' gradient directions],
  [Similarity], [$bold(Q) bold(K)^top / sqrt(d_k)$], [gradient cosine similarity],
  [Normalization], [softmax], [softmax (temperature $T$)],
  [Value], [other tokens' information], [other tasks' loss values],
  [Output], [weighted context], [transfer loss],
)

Compared to Hypernetworks (Ha et al., 2017), adaTT uses
*observed gradients* instead of learned task embeddings as conditioning signals,
enabling immediate adaptation to changes in task relationships.


// =====================================================================
//  4. Task Groups
// =====================================================================

= 4. Task Groups

== 4.1 Four Financial DNA Groups

Based on domain knowledge, 13 tasks are classified into 4 groups:

#styled-table(
  (1fr, 2.5fr, 0.7fr, 0.7fr, 1.8fr),
  [*Group*], [*Members*], [*Intra*], [*Inter*], [*Business Meaning*],
  [Engagement], [cross\_sell\_count,\ will\_acquire\_deposits, will\_acquire\_investments,\ will\_acquire\_accounts, will\_acquire\_lending,\ will\_acquire\_payments], [0.8], [0.3], [Customer engagement/conversion],
  [Lifecycle], [churn\_signal, product\_stability,\ segment\_prediction], [0.7], [0.3], [Customer lifecycle],
  [Value], [nba\_primary], [0.6], [0.3], [Customer value/behavioral patterns],
  [Consumption], [next\_mcc, mcc\_diversity\_trend, top\_mcc\_shift], [0.7], [0.3], [Spending pattern analysis],
)

*Intra-group strength*: transfer strength between tasks within the same group. The Engagement group (0.8) is highest --- product acquisition tasks share similar structure and are expected to exhibit strong positive transfer.

*Inter-group strength*: transfer strength across groups. Conservative at 0.3 --- cross-group transfer is only activated after verification through gradient observation.

== 4.2 Intra/Inter Strength Design

The Group Prior matrix $bold(P)$ is automatically generated from the group structure:
- Within the same group: intra\_strength (0.6~0.8)
- Across different groups: inter\_group\_strength (0.3)
- Diagonal: 0 (self-transfer excluded)
- Row-normalized so values sum to 1

== 4.3 Three Logit Transfer Approaches

Three transfer methods for *explicit information passing* between tasks:

#styled-table(
  (1.2fr, 1.2fr, 0.8fr, 0.6fr, 2fr),
  [*Source*], [*Target*], [*Type*], [*Strength*], [*Business Meaning*],
  [has\_nba], [nba\_primary], [Sequential], [0.5], [NBA presence $arrow.r$ primary product decision],
  [churn\_signal], [product\_stability], [Inverse], [0.5], [Inverse of churn signal $approx$ product stability],
  [next\_mcc], [mcc\_diversity\_trend], [Feature], [0.5], [Next spend category $arrow.r$ diversity trend],
)

Transfer mechanism (residual form):

$ bold(h)_"tower"^t = bold(h)_"expert"^t + alpha dot "SiLU"("LayerNorm"("Linear"("pred"^s))) $

#dim-label[$alpha = 0.5$: transfer strength. Transfer is naturally ignored if projection weights converge to zero.]

The execution order is automatically derived from dependency relationships in the
`task_relationships` config using *Kahn's algorithm* (topological sort).
A fallback order is applied if a cycle is detected.

== 4.4 GroupTaskExpertBasket (v3.2)

*88% parameter reduction* compared to independent per-cluster×task MLPs ($tilde$3.0M $arrow.r$ $tilde$362K).
Tasks in the same Task Groups share a GroupEncoder, differentiated by ClusterEmbedding:

$ bold(e)_"cluster" = "Embedding"("cluster\_id") in RR^(32) $
$ bold(x)_"input" = ["CGC\_output"_(576"D") || "HMM\_proj"_(32"D") || bold(e)_"cluster"_(32"D")] in RR^(640) $
$ bold(h)_"expert" = "MLP"_(640 arrow 128 arrow 64 arrow 32)(bold(x)_"input") $

=== Soft Routing

For cluster boundary samples, embeddings are mixed using GMM posterior probabilities:

$ bold(e)_"cluster" = sum_(c=0)^(19) p_c dot bold(E)_c in RR^(32) $

#dim-label[$p_c$: GMM posterior probability, implementation: `cluster_probs @ embedding.weight` ($[B, 20] times [20, 32]$)]

Unlike hard assignment, boundary customer predictions are insensitive to fluctuations in cluster assignments.


// =====================================================================
//  5. Training Strategy
// =====================================================================

= 5. Training Strategy

== 5.1 2-Phase Training

=== Phase 1: Shared Expert Pretrain

- *Duration*: `shared_expert_epochs` (default 15)
- *Training target*: entire model --- Shared Experts, CGC, Task Experts, Task Towers
- *adaTT*: active --- gradient extraction and transfer loss applied

=== Phase 2: Cluster Finetune

- *Duration*: `cluster_finetune_epochs` (default 8)
- *Training target*: per-cluster Task Expert sub-heads only
- *adaTT*: *inactive* --- gradient extraction is meaningless since Shared Expert is frozen
- *CGC*: frozen --- learning gating is unnecessary and causes overfitting since inputs (Expert outputs) do not change

Items reset at phase transition:

#styled-table(
  (1.3fr, 3fr),
  [*Reset Item*], [*Reason*],
  [Optimizer], [Shared Expert frozen $arrow.r$ prevent stale momentum],
  [Scheduler], [Phase 2-specific warmup (2 epochs, shorter than Phase 1's 5 epochs)],
  [GradScaler], [AMP scaler state initialization (loss scale changes)],
  [Early stopping], [reset best\_val\_loss, patience\_counter],
  [CGC Attention], [Shared Expert frozen $arrow.r$ freeze CGC together],
)

adaTT is always restored after Phase 2 ends (guaranteed even on exceptions via `finally` block).

== 5.2 Uncertainty Weighting

#chip[Kendall et al., CVPR 2018] Automatic task weighting based on homoscedastic uncertainty:

#eq-highlight[
  $ cal(L)_i^"weighted" = frac(1, 2 sigma_i^2) dot cal(L)_i + frac(1, 2) log sigma_i^2 $

  #dim-label[
    $sigma_i^2 = exp("log\_var"_i)$: learnable uncertainty for task $i$ \
    `log_var` clamp: $[-4.0, 4.0]$, precision clamp: $[0.001, 100.0]$
  ]
]

Tasks with high uncertainty ($sigma_i^2$ large) automatically receive lower weight ($1/(2 sigma_i^2)$),
and the $log sigma_i^2$ regularization term prevents uncertainty from growing unboundedly.
This *automatic balancing* replaces the combinatorial explosion of manually tuning weights for 13 tasks.

Uncertainty Weighting is applied *before* adaTT.
The `task_losses` input to adaTT already reflects uncertainty weighting.

== 5.3 Focal Loss

#chip[Lin et al., ICCV 2017] Loss for class-imbalanced binary classification:

$ "FL"(p_t) = -alpha_t dot (1 - p_t)^gamma dot log(p_t) $

The $(1 - p_t)^gamma$ term is key --- it sharply reduces the loss for easy examples ($p_t approx 1$)
and focuses learning on hard examples ($p_t approx 0$).

#styled-table(
  (1fr, 0.6fr, 1fr, 2.5fr),
  [*Task*], [*weight*], [*Focal $alpha$/$gamma$*], [*Notes*],
  [will\_acquire\_\*], [1.5], [$gamma$=2, $alpha$=0.20], [extremely low positive rate $arrow.r$ weight increased],
  [churn\_signal], [1.2], [$gamma$=2, $alpha$=0.60], [high FN cost $arrow.r$ alpha increased],
  [nba\_primary], [2.0], [CE (multiclass)], [business critical],
  [next\_mcc], [2.0], [CE (multiclass)], [spending category prediction],
)

== 5.4 AMP (Automatic Mixed Precision)

Forward pass runs in fp16 with `torch.amp.autocast`.
Additional 10~15% speed improvement with TF32 + cuDNN benchmark:

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
```

LR scheduler: Linear Warmup (5 epochs, start\_factor=0.1) $arrow.r$ CosineAnnealingWarmRestarts ($T_0$=10, $T_"mult"$=2).
Scheduler is reset in Phase 2 (2-epoch warmup).


// =====================================================================
//  6. Implementation Notes
// =====================================================================

= 6. Implementation Notes

== 6.1 FP16 Considerations

=== Focal Loss float32 Casting

In fp16, intermediate results of `focal_weight * bce` may fall into subnormal range, potentially causing NaN.
The entire focal loss computation is performed in float32:

```python
p_f = pred.squeeze().float().clamp(1e-7, 1 - 1e-7)
```

=== adaTT Gradient Extraction and autocast

adaTT gradient extraction takes place inside autocast,
but the loss computation itself is cast to float32 to ensure numerical stability.

== 6.2 GradScaler Guard

`retain_graph=True` cannot be removed due to architectural constraints ---
`_extract_task_gradients` is called after loss computation and before `backward()`,
and the same computation graph must be reused in the Trainer's `loss.backward()`.

Memory impact:

#styled-table(
  (2fr, 1fr, 2fr),
  [*Component*], [*Memory*], [*Notes*],
  [Forward pass graph], [1x], [baseline],
  [retain\_graph overhead], [$tilde$1x], [additional memory due to graph not being freed],
  [13 task gradients], [$tilde$0.3x], [each gradient is shared\_param\_size],
  [*Total*], [*$tilde$2.3x*], [batch 16384 possible on RTX 4070 12GB],
)

=== Gradient Extraction Frequency Optimization

`adatt_grad_interval=10`: gradient extraction every 10 steps instead of every step.
Sufficiently stable thanks to EMA smoothing, with 1/10 reduction in computational overhead.

=== torch.compiler.disable

`@torch.compiler.disable` decorator applied to `_extract_task_gradients`.
`torch.autograd.grad` has incomplete `requires_grad` tracking inside compiled graphs.
`torch.compile` itself is currently disabled (15-task MTL + retain\_graph + dynamic shapes cause
hundreds of kernel compilations, with first epoch taking 30+ minutes), but applied defensively.

== 6.3 Phase 2 Frozen Layers

Layers frozen in Phase 2:

#styled-table(
  (1.5fr, 1fr, 2.5fr),
  [*Layer*], [*Frozen*], [*Reason*],
  [Shared Experts], [Yes], [Preserve shared representations sufficiently learned in Phase 1],
  [CGC Attention], [Yes], [Expert outputs fixed $arrow.r$ learning gating is unnecessary],
  [adaTT], [Inactive], [Shared Expert frozen $arrow.r$ gradient 0, cosine similarity meaningless],
  [GroupEncoder], [No], [Per-cluster specialization learning (core of Phase 2)],
  [Task Towers], [No], [Fine-tune final prediction layers],
  [ClusterEmbedding], [No], [Refine cluster representations],
)

=== CGC-adaTT Co-freezing

CGC Attention is also frozen together at adaTT `freeze_epoch`.
If CGC continues learning, expert weights change and
the affinity relationships measured by adaTT may be invalidated.

```python
if freeze_epoch is not None and epoch >= freeze_epoch:
    for param in self.task_expert_attention.parameters():
        param.requires_grad = False
    self._cgc_frozen.fill_(True)
```

`_cgc_frozen` is registered as a `register_buffer` so its state is preserved when saving/restoring checkpoints.

== 6.4 Full Loss Computation Pipeline Order

+ *Determine per-task loss type* (focal, huber, MSE, NLL, contrastive)
+ *Apply Focal Loss alpha/gamma* (per-task positive class weights)
+ *Apply loss weights* (Uncertainty Weighting or fixed weights)
+ *Add evidential loss* (auxiliary uncertainty estimation loss)
+ *adaTT transfer loss* (add gradient-based transfer loss)
+ *CGC entropy regularization* (prevent Expert Collapse)

== 6.5 Debugging Guide

#styled-table(
  (1.8fr, 2fr, 2fr),
  [*Symptom*], [*Cause*], [*Resolution*],
  [NaN loss], [fp16 focal loss underflow], [verify float32 casting],
  [Loss divergence at training start], [transfer loss dominating original], [check `max_transfer_ratio` (0.5)],
  [Phase 2 RuntimeError], [missing adaTT deactivation], [verify `model.adatt = None`],
  [`ValueError` at initialization], [`freeze_epoch <= warmup_epochs`], [validate config],
  [Training hang (no response)], [gradient extraction every step], [check `adatt_grad_interval` (default 10)],
  [Checkpoint mismatch], [`fill_()` not used], [verify in-place operation for buffer updates],
)

Characteristics of a healthy affinity matrix:
- Within the same group: positive affinity ($> 0.3$)
- Across different groups: weakly positive or neutral ($-0.1 tilde 0.3$)
- Diagonal: 1.0
- Not saturated at $plus.minus 1$ overall (if saturated, adjust EMA decay rate)


// =====================================================================
//  References
// =====================================================================

= References

+ Caruana, R. (1997). Multitask Learning. _Machine Learning_, 28(1).
+ Ma, J. et al. (2018). Modeling Task Relationships in Multi-Task Learning with Multi-Gate Mixture-of-Experts. _KDD_.
+ Tang, H. et al. (2020). Progressive Layered Extraction (PLE): A Novel MTL Model for Personalized Recommendations. _RecSys_.
+ Kendall, A., Gal, Y. & Cipolla, R. (2018). Multi-Task Learning Using Uncertainty to Weigh Losses. _CVPR_.
+ Lin, T.-Y. et al. (2017). Focal Loss for Dense Object Detection. _ICCV_.
+ Yu, T. et al. (2020). Gradient Surgery for Multi-Task Learning (PCGrad). _NeurIPS_.
+ Chen, Z. et al. (2018). GradNorm: Gradient Normalization for Adaptive Loss Balancing. _ICML_.
+ Fifty, C. et al. (2021). Efficiently Identifying Task Groupings for Multi-Task Learning. _NeurIPS_.
+ Navon, A. et al. (2022). Multi-Task Learning as a Bargaining Game (Nash-MTL). _ICML_.
+ Liu, B. et al. (2021). Conflict-Averse Gradient Descent for Multi-Task Learning (CAGrad). _NeurIPS_.
+ Ha, D. et al. (2017). HyperNetworks. _ICLR_.
+ Vaswani, A. et al. (2017). Attention Is All You Need. _NeurIPS_.
+ Jacobs, R. et al. (1991). Adaptive Mixtures of Local Experts. _Neural Computation_.
+ Fedus, W. et al. (2022). Switch Transformers: Scaling to Trillion Parameter Models. _JMLR_.
