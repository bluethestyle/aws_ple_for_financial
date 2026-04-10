// =============================================================================
//  Temporal Ensemble Expert Technical Reference — AIOps PLE for Financial
//  v1.0 · 2026-04-01
// =============================================================================

#let anthropic-bg = rgb("#F0EFEA")
#let anthropic-text = rgb("#141413")
#let anthropic-accent = rgb("#CC785C")
#let anthropic-muted = rgb("#6B7280")
#let anthropic-rule = rgb("#D1D5DB")

#set page(
  paper: "a4",
  margin: (top: 2.5cm, bottom: 2.5cm, left: 2.5cm, right: 2.5cm),
  fill: anthropic-bg,
  header: context {
    if counter(page).get().first() > 1 [
      #set text(size: 7.5pt, font: "New Computer Modern", fill: anthropic-muted, tracking: 0.12em)
      #smallcaps[Temporal Ensemble Expert]
      #h(1fr)
      #smallcaps[AIOps PLE for Financial]
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
#set par(justify: true, leading: 0.8em, spacing: 1.5em)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")

// ─────────────────── Color Palette ────────────────────
// Legacy aliases for component compatibility
#let navy   = anthropic-text
#let teal   = anthropic-accent
#let amber  = anthropic-accent
#let indigo = anthropic-accent
#let rose   = anthropic-accent
#let slate  = anthropic-muted

// ─────────────────── Heading Styles ────────────────────
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

// ─────────────────── Code Block Style ────────────────
#show raw.where(block: true): it => {
  block(
    width: 100%,
    inset: 10pt,
    radius: 4pt,
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

// ─────────────────── Equation Block Spacing ─────────────────
#show math.equation.where(block: true): it => {
  v(3pt)
  it
  v(3pt)
}

// ─────────────────── Custom Components ────────────────
#let note-box(title, body, accent: anthropic-accent) = {
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

#let warn-box(title, body) = note-box(title, body, accent: anthropic-accent)

#let eq-box(body) = {
  block(
    width: 100%,
    inset: 12pt,
    stroke: (left: 2pt + anthropic-accent),
    body,
  )
}

#let dim(body) = text(size: 8.5pt, fill: anthropic-muted)[#body]

// ─────────────── Table Style Function ─────────────────
#let stbl(cols, ..args) = {
  set text(size: 9pt)
  table(
    columns: cols,
    inset: 7pt,
    stroke: 0.4pt + anthropic-rule,
    fill: (_, y) => if y == 0 { anthropic-accent.lighten(88%) } else if calc.odd(y) { luma(252) },
    ..args,
  )
}

// =================================================================
//  Cover Page
// =================================================================
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
    Temporal Ensemble Expert\
    Technical Reference
  ]

  #v(0.3cm)
  #line(length: 30%, stroke: 0.5pt + anthropic-rule)
  #v(0.3cm)

  #text(size: 12pt, fill: anthropic-muted)[
    Mamba SSM · Liquid Neural Network · PatchTST · Ensemble Gating
  ]

  #v(2cm)

  #text(size: 10pt, fill: anthropic-text)[
    AIOps PLE for Financial Recommendation
  ]

  #v(0.5em)

  #text(size: 9pt, fill: anthropic-muted)[v1.0 · 2026-04-01]

  #v(2cm)

  #block(width: 85%, stroke: (left: 2pt + anthropic-accent), inset: (left: 14pt, right: 14pt, top: 8pt, bottom: 8pt))[
    #text(size: 9.5pt, fill: anthropic-muted)[
      This document describes the mathematical structure, Ensemble Gating Mechanism, and
      Project Implementation Specification of the three models comprising the Temporal Ensemble Expert ---
      Mamba (Selective State Space Model), Liquid Neural Network, and
      PatchTST (Patch Time Series Transformer).
    ]
  ]
]

#v(1fr)
#pagebreak()

#set page(
  header: context {
    if counter(page).get().first() > 1 [
      #set text(size: 7.5pt, font: "New Computer Modern", fill: anthropic-muted, tracking: 0.12em)
      #smallcaps[Temporal Ensemble Expert]
      #h(1fr)
      #smallcaps[AIOps PLE for Financial]
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

#outline(indent: 1.5em, depth: 3)

// =================================================================
= Compound Structure of Financial Time Series
// =================================================================

== Static Features vs. Temporal Features

In traditional recommendation systems, user representations are _static_ vectors.
They consist of time-independent properties such as age, occupation, and preferred categories.
However, real user behavior changes continuously along the time axis,
and reducing it to a single snapshot causes critical behavioral signals such as
_periodicity_, _trends_, and _seasonality_ to disappear.

#stbl(
  (1.5fr, 2.5fr, 2.5fr),
  table.header[*Perspective*][*Static Features*][*Temporal Features*],
  [Representation], [Fixed vector $bold(x) in RR^d$], [Sequence $bold(X) in RR^(T times d)$],
  [Information Loss], [Averaging over time axis $arrow.r$ pattern loss], [Order, interval, and trend preserved],
  [Model Requirements], [MLP, Embedding tables], [SSM, ODE, Transformer and other sequence models],
)

The Temporal Expert retains transaction and session data as sequences,
learning patterns embedded in the temporal dimension and compressing them into a 64D representation.

== Three Components of Time Series

Every time series $y(t)$ fundamentally decomposes into the sum of three components:

#eq-box[
  $ y(t) = T(t) + S(t) + R(t) $

  #dim[
    $T(t)$: Trend --- long-term directional movement \
    $S(t)$: Seasonality --- repetitive periodic patterns (daily, monthly, yearly) \
    $R(t)$: Residual --- irregular variation not explained by trend and seasonality
  ]
]

The three models of the Temporal Expert correspond to this decomposition:
- *Mamba*: Trend (long-range dependencies, capturing directional movement through selective memory)
- *PatchTST*: Seasonality (capturing global periodicity through inter-patch attention)
- *LNN*: Residual (handling irregular variation with adaptive time constants)

== Four Characteristics of Financial Transaction Data

Financial time series have unique characteristics that distinguish them from general time series:

#stbl(
  (1.5fr, 3fr, 2fr),
  table.header[*Characteristic*][*Description*][*Processing Model*],
  [Snapshot discontinuity], [Discrete observations such as month-end balances and quarterly reviews], [Mamba selective memory],
  [Daily transaction bursts], [Dozens of payments and transfers per day], [PatchTST patch aggregation],
  [Irregular intervals], [Business days vs. holidays, active vs. dormant periods], [LNN adaptive $tau$],
  [High-frequency events], [Real-time card authorization streams], [Mamba $O(L)$ linear processing],
)

== Window-Based vs. Event-Based Hybrid

In this project, transaction data is aggregated daily to form _window-based sequences_ (180 timesteps).
These are fed into Mamba and PatchTST.
Simultaneously, _actual time interval_ information between transactions is passed to LNN,
performing event-based temporal awareness correction.

#stbl(
  (1.5fr, 2.5fr, 2.5fr),
  table.header[*Approach*][*Window-Based (Fixed Interval)*][*Event-Based (Irregular Interval)*],
  [Time Model], [$Delta t = "const"$], [$Delta t_i eq.not Delta t_j$],
  [Data Shape], [$bold(X) in RR^(T times d)$], [${(t_i, bold(x)_i)}$ event stream],
  [Suitable Model], [Mamba, PatchTST], [LNN (ODE-based)],
)

// =================================================================
= Mamba: Selective State Space Model
// =================================================================

== Theoretical Background

#note-box(accent: indigo)[Reference Paper][
  Gu & Dao, _"Mamba: Linear-Time Sequence Modeling with Selective State Spaces"_ (NeurIPS 2023). \
  Lineage: HiPPO (Gu et al., NeurIPS 2020) $arrow.r$ S4 (ICLR 2022) $arrow.r$ Mamba (2023).
]

An architecture that extends SSM (State Space Model) with an input-dependent selective mechanism (S6),
combining the linear time complexity of RNNs with the content-aware capabilities of Transformers.

== Continuous State Space Model

Basic form of a linear time-invariant (LTI) system:

#eq-box[
  $ frac(d bold(x), d t) = bold(A) bold(x) + bold(B) u, quad y = bold(C) bold(x) + bold(D) u $

  #dim[
    $bold(x) in RR^N$: hidden state, $u in RR$: input signal, $y in RR$: output signal \
    $bold(A) in RR^(N times N)$: state transition matrix,
    $bold(B) in RR^(N times 1)$: input matrix,
    $bold(C) in RR^(1 times N)$: output matrix
  ]
]

== ZOH Discretization

Converting the continuous system to discrete time steps $Delta$:

$ macron(bold(A)) = exp(Delta dot bold(A)), quad macron(bold(B)) approx Delta dot bold(B) $ <eq-zoh>

$macron(bold(A))$ represents "the state transition that advances exactly $Delta$ time units in continuous time."
The first-order approximation $exp(Delta bold(A)) approx bold(I) + Delta bold(A)$ is the Euler method,
while the exact matrix exponential accounts for infinitely many orders.
When $bold(A)$ is diagonal, $exp(bold(A) t) = "diag"(e^(a_1 t), dots, e^(a_N t))$,
which is why Mamba restricts $bold(A)$ to be diagonal.

== Discrete Recurrence

$ bold(h)_t = macron(bold(A)) dot bold(h)_(t-1) + macron(bold(B)) dot bold(x)_t, quad bold(y)_t = bold(C)_t dot bold(h)_t $ <eq-recurrence>

Intuitively: "today's state = yesterday's memory $times$ decay rate + today's new information."

== S6 Selective Mechanism

In conventional LTI SSMs, $bold(A), bold(B), bold(C)$ are constants independent of input.
Mamba's S6 generates $Delta, bold(B), bold(C)$ _input-dependently_ to achieve content-aware processing.

#eq-box[
  $ Delta = "softplus"(bold(W)_Delta dot bold(x) + bold(b)_Delta) $ <eq-delta>
  $ bold(B) = bold(W)_B dot bold(x), quad bold(C) = bold(W)_C dot bold(x) $ <eq-bc>

  #dim[
    $bold(W)_Delta in RR^(D times r)$: $Delta$ projection ($r = ceil(D \/ 16)$, dt\_rank) \
    softplus guarantees: $Delta > 0$ (time steps are positive)
  ]
]

When $Delta$ is large for a given input, the information at that timestep is strongly memorized (encoded);
when $Delta$ is small, the previous state is retained and the current input is ignored (forgotten).

=== Selective Mechanism in Financial Domain

- Large transaction $arrow.r$ large $Delta$ $arrow.r$ strongly recorded in hidden state (selective memory)
- Small daily transaction $arrow.r$ small $Delta$ $arrow.r$ previous state maintained, processed as background (selective forgetting)
- This mechanism naturally models the _heterogeneous importance_ of financial events

== Selective Sequential Scan

Since $Delta_t, bold(B)_t, bold(C)_t$ differ at each timestep:

$ bold(h)_t = underbrace(exp(Delta_t dot bold(A)), macron(bold(A))_t) dot bold(h)_(t-1) + underbrace(Delta_t dot bold(B)_t, macron(bold(B))_t) dot bold(x)_t, quad bold(y)_t = bold(C)_t^top bold(h)_t $ <eq-s6-scan>

The entire sequence is processed in $O(L)$ linear time complexity.

== Gated MLP + Causal Conv1d Architecture

MambaBlock wraps the SSM with a Gated MLP and 1D causal convolution:

+ *Input Projection*: `d_input` $arrow.r$ `d_model`
+ *LayerNorm* $arrow.r$ *in\_proj*: `d_model` $arrow.r$ `2 * d_inner` (split into two paths)
+ *SSM path*: Causal Conv1d (kernel=4) $arrow.r$ SiLU $arrow.r$ SelectiveSSM (S6)
+ *Gate path*: SiLU activation
+ *Element-wise multiply* (SSM output $circle.small$ Gate output)
+ *out\_proj*: `d_inner` $arrow.r$ `d_model` + Residual connection

The causal Conv1d (kernel=4) references only $t-3, t-2, t-1, t$ at timestep $t$,
mixing local context without leaking future information.

== A Matrix Initialization and Stability

$bold(A)$ is stored in $log$ space and restored as $-exp("A"_"log")$ during the forward pass.
This guarantees that all eigenvalues are _always negative_, ensuring system stability.

The initial value is a HiPPO-style diagonal $[1, 2, dots, N]$.
States with larger indices correspond to higher-frequency polynomial components,
so diagonal elements become $-1, -2, dots, -N$, causing higher-frequency components to decay faster.

== Project Implementation Specification

=== Transaction Mamba

#stbl(
  (1fr, 1fr, 2.5fr),
  table.header[*Parameter*][*Value*][*Description*],
  [`d_model`], [128], [hidden dimension],
  [`d_input`], [16], [card (8D) + deposit (8D) transaction features],
  [`d_inner`], [256], [$128 times 2$ (expand=2)],
  [`d_state`], [16], [SSM state vector dimension],
  [`d_conv`], [4], [1D causal convolution kernel size],
  [`dt_rank`], [8], [$ceil(128 \/ 16) = 8$],
  [`seq_len`], [180], [180-day transaction sequence],
  [Output], [`[B, 128]`], [last timestep `[:, -1, :]`],
)

=== Session Mamba

#stbl(
  (1fr, 1fr, 2.5fr),
  table.header[*Parameter*][*Value*][*Description*],
  [`d_model`], [64], [half dimension compared to transaction],
  [`d_input`], [8], [session feature dimension],
  [`d_inner`], [128], [$64 times 2$ (expand=2)],
  [`seq_len`], [90], [90-day session sequence],
  [Output], [`[B, 64]`], [last timestep],
)

Total Mamba output dimension: $128 + 64 = 192$D (txn + session concat).

== Complexity Analysis

#stbl(
  (1.5fr, 1.5fr, 2.5fr),
  table.header[*Operation*][*Complexity*][*Notes*],
  [Sequential scan], [$O(L dot D dot N)$], [$L$: sequence length, $D$: model dimension, $N$: state dimension],
  [vs. Transformer], [$O(L)$ vs $O(L^2)$], [linear in sequence length],
  [Memory], [$O(D dot N)$], [only hidden state retained (independent of sequence length)],
)

// =================================================================
= Liquid Neural Network (LNN)
// =================================================================

== Theoretical Background

#note-box(accent: indigo)[Reference Paper][
  Hasani et al., _"Liquid Time-constant Networks"_ (AAAI 2021). \
  Lineage: Neural ODE (Chen et al., NeurIPS 2018) + biological neuron decay model + Liquid State Machine (Maass et al., 2002).
]

A continuous-time neural network based on Neural ODE, where the time constant $tau$
adapts dynamically according to the input.
It can naturally handle irregular time intervals (an intrinsic characteristic of financial transactions).

== Core ODE

#eq-box[
  $ frac(d bold(h), d t) = frac(-bold(h) + f(bold(x), bold(h)), tau(bold(x), bold(h))) $ <eq-lnn-ode>

  #dim[
    $bold(h) in RR^H$: hidden state, $bold(x)$: external input \
    $f(dot, dot)$: state update function (target state),
    $tau(dot, dot) > 0$: adaptive time constant
  ]
]

=== Physical Interpretation of Each Term

#stbl(
  (1fr, 3.5fr),
  table.header[*Term*][*Meaning*],
  [$-bold(h)$], [Decay (leak): hidden state converges to 0 without input. Naturally forgets old information.],
  [$f(bold(x), bold(h))$], [Driving force: target state based on new input and current state. $f = tanh(bold(W)_f [bold(x); bold(h)] + bold(b)_f)$],
  [$tau(bold(x), bold(h))$], [Time constant: larger values mean slower change (inertia, state preservation), smaller values mean faster response. $tau = "Softplus"("MLP"([bold(x); bold(h)])) + 0.1$],
)

== Euler Discretization

Converting the continuous ODE to discrete timesteps:

$ bold(h)_(t+1) = bold(h)_t + Delta t dot frac(-bold(h)_t + f(bold(x)_t, bold(h)_t), tau(bold(x)_t, bold(h)_t)) $ <eq-euler>

$Delta t$ is the _actual time interval_ (in days), so the ODE naturally handles irregular transaction intervals.

== Financial Domain Interpretation of the Adaptive Time Constant

Transaction intervals in finance are extremely irregular:

#stbl(
  (2fr, 1.5fr, 2.5fr),
  table.header[*Situation*][*$Delta t$*][*Role of $tau$*],
  [Multiple transactions on the same day], [$tilde 0.01$ days], [Small $tau$: fast response, immediately reflecting each transaction],
  [Weekend gap], [$2$ days], [Medium $tau$: state maintained gradually],
  [Long-term dormancy], [$> 30$ days], [Large $tau$: state preserved, slow decay],
)

RNNs/LSTMs with fixed $tau$ treat all intervals equally.
The adaptive $tau$ responds quickly during active transaction periods
and preserves state during dormant periods, automatically adapting to _customer behavioral rhythms_.

== SingleStep Mode Design

In this project, LNN operates in *SingleStep mode*.
Rather than processing the entire sequence through ODE,
it applies only a _single ODE step_ to Mamba's final hidden state.

=== Design Rationale

- Mamba already captures the full sequence patterns in $O(L)$
- LNN serves only as a _temporal scale correction_ role (avoiding redundant sequence processing)
- Computational cost: $O(1)$ single step vs. $O(L)$ full sequence ODE

#note-box[Mamba $arrow.r$ LNN Serial Structure][
  After Mamba learns sequence patterns,
  LNN applies temporal-aware correction to the final state in a serial structure.
  This corresponds to the decomposition of "trend extraction $arrow.r$ residual correction."
]

== Project Implementation Specification

#stbl(
  (1.5fr, 1.5fr, 1.5fr, 1.5fr),
  table.header[*Parameter*][*LNN txn*][*LNN session*][*Description*],
  [`input_dim`], [128], [64], [Mamba output dimension],
  [`hidden_dim`], [64], [32], [LNN hidden dimension],
  [Output], [`[B, 64]`], [`[B, 32]`], [SingleStep output],
)

Total LNN output dimension: $64 + 32 = 96$D (txn + session concat).

// =================================================================
= PatchTST: Patch Time Series Transformer
// =================================================================

== Theoretical Background

#note-box(accent: indigo)[Reference Paper][
  Nie et al., _"A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"_ (ICLR 2023). \
  Applies the patch concept from ViT (Dosovitskiy et al., ICLR 2021) to time series.
]

The time series is divided into fixed-size patches,
and each patch is treated as a token with Self-Attention applied.
This reduces the attention cost from $O(L^2)$ to $O((L\/P)^2)$
while effectively capturing multi-scale periodic patterns.

== Patch Embedding

$ bold(p)_i = bold(W)_"proj" dot "flatten"(bold(x)_[((i-1)P+1) : (i P)]) + bold(b)_"proj" $ <eq-patch>

With patch size $P = 16$, the 180-timestep transaction sequence is converted to $floor(180 \/ 16) = 11$ patches.
Attention cost: $O(180^2) = 32400$ $arrow.r$ $O(11^2) = 121$.

=== Financial Domain Interpretation of Patch Size

A patch size of 16 corresponds to approximately 2 weeks, naturally aligning with the fundamental unit of payroll cycles (bi-weekly/monthly).
Local patterns within each patch (daily spending) are aggregated,
and inter-patch attention captures global periodicity (monthly salary, quarterly bonuses).

== Multi-Head Self-Attention

$ "Attention"(bold(Q), bold(K), bold(V)) = "softmax"(frac(bold(Q) bold(K)^top, sqrt(d_k))) bold(V) $ <eq-attention>

The $(i,j)$ element of $bold(Q) bold(K)^top$ is the inner product of two patch vectors, proportional to _cosine similarity_.
Softmax converts this to a probability distribution, computing a weighted sum of information from highly relevant patches.
Division by $sqrt(d_k)$ normalizes the inner product variance to maintain healthy softmax gradients.

== Positional Encoding

Sinusoidal positional encoding is used.
For 6--12 patches, it provides sufficient performance compared to learnable positional encoding
and generalizes better to varying sequence lengths.

== Project Implementation Specification

#stbl(
  (1.5fr, 1.5fr, 1.5fr, 2fr),
  table.header[*Parameter*][*PatchTST txn*][*PatchTST session*][*Description*],
  [`d_model`], [64], [32], [Transformer hidden dimension],
  [`nhead`], [4], [2], [number of multi-head attention heads],
  [`num_layers`], [2], [2], [number of Transformer encoder layers],
  [`patch_size`], [16], [16], [patch size (approximately 2 weeks)],
  [Patch count], [11], [5], [$floor(L \/ P)$],
  [Output], [`[B, 64]`], [`[B, 32]`], [AdaptiveAvgPool1d],
)

Total PatchTST output dimension: $64 + 32 = 96$D (txn + session concat).

== Complexity Comparison

#stbl(
  (2fr, 1.5fr, 2fr),
  table.header[*Model*][*Complexity*][*At L=180*],
  [Vanilla Transformer], [$O(L^2)$], [$32,400$],
  [PatchTST ($P=16$)], [$O((L\/P)^2)$], [$121$],
  [Mamba (SSM)], [$O(L)$], [$180$],
)

// =================================================================
= Ensemble Gating
// =================================================================

== Gate Architecture

After concatenating the outputs of three models, a learnable 2-layer gate network
dynamically determines each model's contribution:

#eq-box[
  $ bold(g) = "Softmax"(bold(W)_2 dot "ReLU"(bold(W)_1 dot bold(z)_"cat" + bold(b)_1) + bold(b)_2) in RR^3 $ <eq-gate>
  $ bold(y) = sum_(i=1)^3 g_i dot "Proj"_i (bold(z)_i) in RR^(64) $ <eq-ensemble>

  #dim[
    $bold(z)_"cat"$: concatenation of three model outputs (192 + 96 + 96 = 384D) \
    $"Proj"_i$: projects each model output into a common 64D space \
    $g_i$: weight of model $i$ ($sum g_i = 1$)
  ]
]

== Ensemble Diversity Through Input Separation

#stbl(
  (1.5fr, 2fr, 2.5fr),
  table.header[*Model*][*Input Path*][*Basis for Diversity*],
  [Mamba], [direct input of original sequence], [sequential state space perspective],
  [LNN], [Mamba final state (serial)], [temporal scale correction perspective],
  [PatchTST], [independent input of original sequence], [global attention perspective],
)

The Mamba $arrow.r$ LNN serial + PatchTST independent structure
ensures ensemble diversity through input separation.
Using identical inputs reduces the differentiation effect of gating,
making independent path design essential.

== Gate Entropy Monitoring

The health of the gate distribution is monitored using Shannon entropy:

$ H(bold(g)) = -sum_(i=1)^3 g_i log_2(g_i) $ <eq-entropy>

#stbl(
  (2fr, 1.5fr, 3fr),
  table.header[*State*][*Entropy*][*Interpretation*],
  [Uniform distribution], [$log_2(3) approx 1.585$ bits], [Three models contribute equally (maximum entropy)],
  [Healthy skew], [$0.5 tilde 1.2$ bits], [Adaptive weights according to input (normal operation)],
  [Gate Collapse], [$< 0.3$ bits], [One model dominates, others stop learning (warning)],
)

#warn-box[Gate Collapse][
  When $H < 0.3$ bits, one model monopolizes almost all the weight.
  The remaining two models have their gradients blocked and stop learning,
  rendering the ensemble meaningless.
  Log entropy periodically during training,
  and apply gate temperature scaling or entropy bonus regularization when collapse is detected.
]

== Final Output Flow Summary

$
"txn\_seq" [B, 180, 16] &arrow.r "Mamba"_"txn" arrow.r [B, 128] \
"sess\_seq" [B, 90, 8] &arrow.r "Mamba"_"sess" arrow.r [B, 64] \
"Mamba"_"txn" &arrow.r "LNN"_"txn" arrow.r [B, 64] \
"Mamba"_"sess" &arrow.r "LNN"_"sess" arrow.r [B, 32] \
"txn\_seq" &arrow.r "PatchTST"_"txn" arrow.r [B, 64] \
"sess\_seq" &arrow.r "PatchTST"_"sess" arrow.r [B, 32] \
$

Concat: $[192 + 96 + 96 = 384"D"]$ $arrow.r$ Gate $arrow.r$ Weighted sum $arrow.r$ *64D* final output.

// =================================================================
= I/O Specification and Implementation Notes
// =================================================================

== Full I/O Specification

=== Inputs

#stbl(
  (2fr, 1.5fr, 3fr),
  table.header[*Input*][*Shape*][*Description*],
  [`txn_seq`], [`[B, 180, 16]`], [180-day transaction sequence, card 8D + deposit 8D],
  [`session_seq`], [`[B, 90, 8]`], [90-day session sequence, 8D session features],
)

=== Intermediate Representations

#stbl(
  (2fr, 1fr, 1fr, 1fr),
  table.header[*Model*][*txn output*][*session output*][*Total*],
  [Mamba (SSM)], [128D], [64D], [192D],
  [LNN (ODE)], [64D], [32D], [96D],
  [PatchTST (Attn)], [64D], [32D], [96D],
  [*Concat*], [], [], [*384D*],
)

=== Outputs

#stbl(
  (2fr, 1.5fr, 3fr),
  table.header[*Output*][*Shape*][*Description*],
  [Temporal Expert output], [`[B, 64]`], [final representation after gated weighted sum],
  [Gate weights], [`[B, 3]`], [Mamba, LNN, PatchTST contributions (for monitoring)],
)

The 64D output vector is passed to PLE's CGC Gate Attention,
where it is dynamically combined per task with the outputs of the other 6 experts
(PersLay, DeepFM, LightGCN, Unified H-GCN, Causal, OT).

== Summary: Role Comparison of the Three Models

#stbl(
  (1.2fr, 2fr, 2fr, 1.5fr),
  table.header[*Model*][*Temporal Pattern Captured*][*Mechanism*][*Complexity*],
  [Mamba], [Long-range sequential dependencies (trend)], [Selective State Space (S6)], [$O(L)$ linear],
  [LNN], [Irregular time intervals (residual)], [Adaptive time constant ODE], [$O(1)$ single step],
  [PatchTST], [Global periodicity (seasonality)], [Patch-level Self-Attention], [$O((L\/P)^2)$],
)

== Generational Progression of Time Series Analysis and This Project

#stbl(
  (0.8fr, 2fr, 2fr, 1.5fr),
  table.header[*Generation*][*Approach*][*Limitation*][*Overcoming Model*],
  [1st], [ARIMA, Exponential Smoothing], [linearity assumption, manual differencing], [LSTM, GRU],
  [2nd], [LSTM, GRU (gated RNNs)], [$O(L)$ sequential bottleneck, vanishing gradient], [Transformer],
  [3rd], [Transformer (Self-Attention)], [$O(L^2)$ complexity, weak ordering information], [SSM, PatchTST],
  [4th], [SSM + ODE + Patch Transformer], [model complexity, Gate Collapse risk], [this project],
)

== Implementation Considerations

=== Data Leakage Prevention

- `gap_days` must be set so the last timestep of sequence data does not overlap with the label derivation period (minimum 7 days)
- Mamba's causal Conv1d does not reference future information (left-padding)
- PatchTST attention references the entire sequence without a causal mask, but since the input sequence is already truncated to before the label period, there is no leakage risk

=== Numerical Stability

- Mamba $bold(A)$ matrix: negativity guaranteed by $-exp("A"_"log")$ (prevents divergence)
- Mamba $Delta$: positivity guaranteed by softplus ($Delta > 0$)
- LNN $tau$: lower bound of $"Softplus"(dot) + 0.1$ (prevents zero division)
- Warning logged when gate entropy $< 0.3$ bits

=== Performance Optimization

- Mamba sequential scan: prototype uses Python loop; production recommends `mamba-ssm` CUDA kernel
- AMP (Mixed Precision) enabled: approximately 2x speed improvement on g4dn T4 GPU
- Batch size: adjusted based on VRAM and data scale (4096 recommended for large-scale data)

=== References

#set text(size: 9pt)

#table(
  columns: (2.5fr, 3.5fr, 1.5fr),
  inset: 6pt,
  stroke: 0.4pt + rgb("#e2e8f0"),
  fill: (_, y) => if y == 0 { navy.lighten(88%) } else if calc.odd(y) { luma(252) },
  table.header[*Component*][*Paper*][*Venue*],
  [Mamba SSM], [Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"], [NeurIPS 2023],
  [LNN], [Hasani et al., "Liquid Time-constant Networks"], [AAAI 2021],
  [PatchTST], [Nie et al., "A Time Series is Worth 64 Words"], [ICLR 2023],
  [Neural ODE], [Chen et al., "Neural Ordinary Differential Equations"], [NeurIPS 2018],
  [S4 (SSM)], [Gu et al., "Efficiently Modeling Long Sequences with Structured State Spaces"], [ICLR 2022],
  [HiPPO], [Gu et al., "HiPPO: Recurrent Memory with Optimal Polynomial Projections"], [NeurIPS 2020],
  [ViT], [Dosovitskiy et al., "An Image is Worth 16x16 Words"], [ICLR 2021],
  [CfC], [Hasani et al., "Closed-form Continuous-depth Models"], [Nature MI 2022],
)
