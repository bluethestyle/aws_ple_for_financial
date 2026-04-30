// ============================================================
// Expert Details: 7 Heterogeneous Experts + adaTT + Feature Engineering
// AIOps PLE for Financial Recommendation (English)
// ============================================================

#set document(
  title: "Heterogeneous Expert Architecture: Selection Rationale, Mathematical Formulation, and Financial Application",
  author: ("Author 1", "Author 2"),
)

#set page(
  paper: "a4",
  margin: (x: 2cm, y: 2.5cm),
  numbering: "1",
)

#set text(font: "New Computer Modern", size: 10pt, lang: "en")
#set par(justify: true, leading: 0.6em)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")

#show heading.where(level: 1): it => {
  v(1.2em)
  text(size: 14pt, weight: "bold")[#it]
  v(0.4em)
}

#show heading.where(level: 2): it => {
  v(0.8em)
  text(size: 12pt, weight: "bold")[#it]
  v(0.3em)
}

#show heading.where(level: 3): it => {
  v(0.5em)
  text(size: 10.5pt, weight: "bold")[#it]
  v(0.2em)
}

// ============================================================
// Title
// ============================================================
#align(center)[
  #text(size: 18pt, weight: "bold")[
    Heterogeneous Expert Architecture
  ]

  #v(0.3em)

  #text(size: 14pt)[
    Selection Rationale, Mathematical Formulation,\
    and Financial Application
  ]

  #v(1em)

  #text(size: 11pt)[
    Author 1#super[1], Author 2#super[1]
  ]

  #v(0.3em)

  #text(size: 9pt, style: "italic")[
    #super[1]Organization Name \
    contact\@org.com
  ]

  #v(1.5em)
]

// ============================================================
// Abstract
// ============================================================
#block(
  width: 100%,
  inset: (x: 1em),
  stroke: (left: 2pt + luma(120)),
)[
  #text(weight: "bold")[Scope.]
  - 7 heterogeneous experts: DeepFM, Temporal Ensemble, Hyperbolic GCN, PersLay/TDA, LightGCN, Causal/NOTEARS, Optimal Transport.
  - Per-expert mathematical background, input feature subset, uniform 64D output, financial-domain application.
  - FeatureRouter: each expert receives a subset declared by `target_experts` in `feature_groups.yaml` (deepfm=977D, temporal=116D, hgcn=58D, perslay=32D, causal=129D, lightgcn=955D, ot=95D, mlp=57D).
  - Task basket: 1 MLP expert (total 7 shared + 1 task).
  - adaTT active on 13 tasks; GradSurgery was tested but not adopted (no gain + VRAM overhead).
]

#v(1em)

#block(
  width: 100%,
  inset: 12pt,
  radius: 4pt,
  fill: rgb("#fef3c7"),
  stroke: (left: 3pt + rgb("#d97706")),
)[
  #text(weight: "bold")[Design vs. Implementation Note.]
  This document is written based on the full-bank design (734D).
  The current Santander benchmark implementation uses 1211D input (17 feature groups).
  FeatureRouter is active: each expert receives a per-expert subset
  (deepfm=977D, temporal\_ensemble=116D, hgcn=58D, perslay=32D, causal=129D, lightgcn=955D, ot=95D, mlp=57D),
  model parameters ~2.8M post-FeatureRouter pruning (varies with group toggle). Routing is group-level, auto-built from
  \`target_experts\` in feature_groups.yaml. HGCN receives merchant\_hierarchy (27D) + mcc\_top30\_multihot (31D) = 58D;
  LightGCN receives product\_hierarchy (34D) + graph\_collaborative (66D) + txn\_lag\_tensor (800D) + nba\_label\_multihot (24D) + mcc\_top30\_multihot (31D) = 955D.
  I/O specs below reflect the FeatureRouter-active per-expert input dims.
]

#v(1em)

// ============================================================
= DeepFM Expert --- Feature Interaction
// ============================================================

== Selection Rationale (Why DeepFM?)

Financial recommendation depends not on individual features but on feature _interactions_. For example,
the pattern where "30s + high digital engagement + high RFM" combine to sharply increase online
investment conversion rates cannot be captured by any single feature alone.
Explicitly modeling all pairwise interactions for 644 features requires $O(n^2) = 207,046$ parameters,
most of which cannot be stably estimated from sparse data.

FM's low-rank factorization reduces the interaction parameters to $O(n k) = 10,304$ while enabling
generalization to unobserved feature pairs through shared latent vectors.
The Deep component extends beyond FM's 2nd-order limitation by adding implicit higher-order
interactions of 3rd order and above.

== Alternative Comparison (Why Not Alternatives?)

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, left, left, left),
    stroke: 0.5pt,
    [*Aspect*], [*DeepFM*], [*DCNv2*], [*AutoInt (Transformer)*],
    [Interaction order], [FM: 2nd + Deep: arbitrary], [Cross: $(l+1)$-th + Deep], [Self-attention: arbitrary],
    [Parameters], [$tilde 169$K], [$tilde 2.5$M (cross alone)], [$O(n^2)$ attention],
    [Inference speed], [Fastest (2--5x vs Attention)], [Medium], [Slowest],
    [Field awareness], [Yes (28 fields)], [No (raw features)], [Yes (per-feature)],
  ),
  caption: [DeepFM vs. alternative architectures. In BARS benchmarks (2022--2024), DeepFM consistently achieves top-tier performance with the lowest inference latency on small-to-medium datasets.],
)

Unlike Wide&Deep (Google, 2016), DeepFM shares embeddings between FM and Deep,
eliminating manual cross-feature engineering. It provides the combination of
"structural efficiency (FM) + universal expressiveness (Deep)."

== Key Mathematical Formulations

*FM 2nd-order interaction:*
$ hat(y) = w_0 + sum_(i=1)^n w_i x_i + sum_(i=1)^n sum_(j=i+1)^n chevron.l bold(v)_i, bold(v)_j chevron.r x_i x_j $

*Low-rank factorization:*
$ W approx V V^top, quad V in bb(R)^(n times k) $

This reduces parameters from $O(n^2)$ to $O(n k)$. The project uses $n=644$, $k=16$.

*FM Trick ($O(n k)$ computation):*
$ sum_(i<j) chevron.l bold(v)_i, bold(v)_j chevron.r x_i x_j = 1/2 sum_(f=1)^k [ (sum_(i=1)^n v_(i,f) x_i)^2 - sum_(i=1)^n (v_(i,f) x_i)^2 ] $

*Cross Network (DCNv2) layer:*
$ bold(x)_(l+1) = bold(x)_0 circle.small (bold(x)_l W_l + bold(b)_l) + bold(x)_l $

Each layer adds one order of interaction, and $l$ layers capture up to $(l+1)$-th order interactions.
Only $2d$ parameters per layer (compared to $O(d^2)$ for MLP).

=== 28-Field Architecture

644D normalized features are split into 28 semantic fields (base 238D, multi-source 91D, extended 84D,
domain 159D, multidisciplinary 24D, model-derived 27D, merchant 21D).
Each field is projected into a uniform 16D latent space via `nn.Linear(d_i, 16)`.
FM operates on field-level embeddings $[B, 28, 16]$,
while Deep MLP operates on the flattened $[B, 448]$.

*Output:* FM 16D + Deep 64D = 80D $arrow$ output layer $arrow$ 64D (PLE gate input).

== Financial Application

- Cross-field interactions are essential in banking: "30s + Seoul + high digital engagement" $arrow$ online investment conversion rate surge;
  "high RFM + low deposits" $arrow$ credit product recommendation.
- The 28-field design enables inter-category FM interaction: splitting a 64D category into 4$times$16D
  creates 27 new FM interaction pairs at no additional parameter cost.

== I/O Specification

#figure(
  table(
    columns: (auto, auto),
    stroke: 0.5pt,
    [*Input*], [109D feature subset $[B, 109]$ (FeatureRouter: State-axis groups)],
    [*Internal*], [fields $times$ 16D embeddings $arrow$ FM + Deep],
    [*Output*], [64D expert representation for PLE CGC Gate],
    [*Parameters*], [$tilde 169$K],
  ),
  caption: [DeepFM Expert I/O specification. Input is 109D (FeatureRouter active) vs. 644D in full-bank design.],
)

== Implementation Notes

- Used as PLE's default lightweight Shared Expert. DCNv2Expert is an alternative for cases requiring higher-order interactions.
- Since FM and Deep share embeddings, gradients flow simultaneously through both paths.
- When applying FocalLoss, pre-activation logits must be passed (to prevent double-sigmoid).

#pagebreak()

// ============================================================
= Temporal Ensemble Expert --- Mamba + LNN + PatchTST
// ============================================================

== Selection Rationale (Why Temporal Ensemble?)

Static features (age, average spending) discard the temporal dimension of transaction sequences---periodicity,
trends, and irregular event patterns. Compressing a 180-day spending sequence into a monthly average
loses weekly cycles, trend directions, and anomalous burst patterns.

All time series decompose as $y(t) = T(t) + S(t) + R(t)$ (trend + seasonality + residual),
and no single architecture can optimally capture all three:

#figure(
  table(
    columns: (auto, auto, auto, auto),
    stroke: 0.5pt,
    [*Model*], [*Temporal Pattern*], [*Mechanism*], [*Complexity*],
    [Mamba (SSM)], [Long-range trend], [Selective State Space (S6)], [$O(L)$ linear],
    [LNN (ODE)], [Irregular residuals], [Adaptive time-constant ODE], [$O(1)$ single step],
    [PatchTST], [Global periodicity], [Patch-level self-attention], [$O((L slash P)^2)$],
  ),
  caption: [The three models of the Temporal Ensemble -- corresponding to the time-series decomposition $T + S + R$.],
)

== Alternative Comparison (Why Not Single Model?)

#figure(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt,
    [*Generation*], [*Approach*], [*Limitation*],
    [1st], [ARIMA, Exponential Smoothing], [Linear assumption, manual differencing],
    [2nd], [LSTM, GRU], [$O(L)$ sequential bottleneck, vanishing gradient],
    [3rd], [Transformer], [$O(L^2)$ complexity, weak order encoding],
    [4th (ours)], [SSM + ODE + Patch Transformer], [Model complexity (mitigated by entropy monitoring)],
  ),
  caption: [Comparison by time-series modeling generation.],
)

== Key Mathematical Formulations

=== Mamba (Selective State Space Model)

*Continuous SSM:*
$ (d bold(x))/(d t) = bold(A) bold(x) + bold(B) u, quad y = bold(C) bold(x) + bold(D) u $

*ZOH Discretization:*
$ macron(bold(A)) = exp(Delta dot bold(A)), quad macron(bold(B)) approx Delta dot bold(B) $

*Discrete recurrence:*
$ bold(h)_t = macron(bold(A)) dot bold(h)_(t-1) + macron(bold(B)) dot bold(x)_t, quad bold(y)_t = bold(C)_t dot bold(h)_t $

*S6 Selective Mechanism (key innovation):*
$ Delta = "softplus"(bold(W)_Delta dot bold(x) + bold(b)_Delta) $
$ bold(B) = bold(W)_B dot bold(x), quad bold(C) = bold(W)_C dot bold(x) $

Unlike LTI systems, S6 makes $Delta$, $bold(B)$, $bold(C)$ _input-dependent_, enabling content-aware processing.
A large $Delta$ strongly encodes the current input into state; a small $Delta$ preserves the previous state.

*Financial interpretation:* Large transactions $arrow$ large $Delta$ (strongly remembered), small routine purchases $arrow$ small $Delta$ (quickly forgotten as background noise).

*Specification:* Transaction Mamba: $d_"model" = 128$, $d_"input" = 16$, $d_"state" = 16$, $"seq_len" = 180$.
Session Mamba: $d_"model" = 64$. HiPPO-style diagonal initialization $[-1, -2, ..., -N]$.

=== Liquid Neural Network (LNN)

*Core ODE:*
$ (d bold(h))/(d t) = (-bold(h) + f(bold(x), bold(h)))/(tau(bold(x), bold(h))) $

Where:
- $-bold(h)$: leak/decay term (forgets toward 0 without input)
- $f(bold(x), bold(h)) = tanh(bold(W)_f [bold(x); bold(h)] + bold(b)_f)$: target state
- $tau(bold(x), bold(h)) = "Softplus"("MLP"([bold(x); bold(h)])) + 0.1$: adaptive time constant

*Euler discretization:*
$ bold(h)_(t+1) = bold(h)_t + Delta t dot (-bold(h)_t + f(bold(x)_t, bold(h)_t))/(tau(bold(x)_t, bold(h)_t)) $

The project uses SingleStep mode --- applying only 1 ODE step to Mamba's final hidden state.
Since Mamba already captures the full sequence pattern at $O(L)$, LNN adds only _time-scale correction_.

*Financial interpretation:* Financial transaction intervals are extremely irregular (multiple intraday trades: $Delta t tilde 0.01$ days,
weekend gaps: $Delta t = 2$ days, long dormancy: $Delta t > 30$ days). Adaptive $tau$ adjusts automatically.

=== PatchTST (Patch Time Series Transformer)

*Patch embedding:*
$ bold(p)_i = bold(W)_"proj" dot "flatten"(bold(x)_[[(i-1)P+1 : i P]]) + bold(b)_"proj" $

With $P = 16$, a 180-step sequence becomes 12 patches (tokens), reducing attention cost from
$O(180^2)$ to $O(12^2) = 144$.

*Financial interpretation:* Patch size 16 corresponds to approximately 2 weeks, naturally aligning with salary cycles (biweekly/monthly).
Each patch captures local patterns (daily spending within 2 weeks), while inter-patch attention captures global periodicity (payday surges, quarterly bonuses).

=== Ensemble Gating

$ bold(g) = "Softmax"(bold(W)_2 dot "ReLU"(bold(W)_1 dot bold(z)_"cat" + bold(b)_1) + bold(b)_2) in bb(R)^3 $
$ bold(y) = sum_(i=1)^3 g_i dot "Proj"_i (bold(z)_i) in bb(R)^(64) $

$bold(z)_"cat"$ is the concatenation of the three model outputs (192+96+96 = 384D).
Gate entropy $H(bold(g)) = -sum_(i=1)^3 g_i log_2(g_i)$ is monitored, and
gate collapse is detected when $H < 0.3$ bits.

== Financial Application

Financial transactions exhibit strong periodicity (payday, weekend dining), gradual trends
(lifestyle changes, approaching churn), and irregular residuals (travel, fraud).
Each component maps directly to a model's strength:
Mamba $arrow$ trends, PatchTST $arrow$ seasonality, LNN $arrow$ residuals.

== I/O Specification

#figure(
  table(
    columns: (auto, auto),
    stroke: 0.5pt,
    [*Input*], [129D feature subset (FeatureRouter: Timeseries-axis groups) + Transaction seq $[B, 180, 16]$ + Session seq $[B, 90, 8]$],
    [*Mamba output*], [128D + 64D = 192D],
    [*LNN output*], [64D + 32D = 96D (SingleStep on Mamba final state)],
    [*PatchTST output*], [64D + 32D = 96D],
    [*Ensemble output*], [64D (gated combination for PLE gate)],
  ),
  caption: [Temporal Ensemble Expert I/O specification. FeatureRouter provides 129D static feature subset.],
)

== Implementation Notes

- Mamba $arrow$ LNN is serial (LNN corrects time-scale after Mamba training), PatchTST is independent.
  This is a design for ensuring ensemble diversity through input separation.
- The A matrix is initialized with HiPPO-style diagonal $[-1, -2, ..., -N]$ to implement multi-scale memory decay.
- A gate collapse warning is triggered when gate entropy drops from $log_2(3) approx 1.585$ bits (uniform distribution) to $< 0.3$ bits, indicating one model is dominating.

#pagebreak()

// ============================================================
= Hyperbolic GCN Expert --- Hierarchical Structure
// ============================================================

== Selection Rationale (Why Hyperbolic GCN?)

Recommendation data has two fundamentally different geometric structures:
(1) user-item interactions are _peer-to-peer_ (no hierarchy),
(2) Merchant Category Codes (MCC) form a _tree hierarchy_ (Root $arrow$ L1 $arrow$ L2 $arrow$ Brand $arrow$ Branch, $tilde$550K nodes).
A single geometry cannot efficiently represent both.

Embedding a complete binary tree of depth $d$ with uniform spacing in Euclidean space requires $O(2^d)$ dimensions.
Distortion-free embedding of $tilde$50K brand-level nodes would require tens of thousands of dimensions.
Hyperbolic space (negative curvature) has volume that increases _exponentially_ away from the origin,
exactly matching tree branching.
*An 8D Poincare Ball can represent the entire MCC hierarchy.*

Nickel & Kiela (2017) showed that 5D hyperbolic embeddings outperform 200D Euclidean embeddings
on the WordNet hierarchy.

== Alternative Comparison

- *LightGCN vs. NGCF:* Removing feature transformation $W$ and nonlinear activation $sigma$
  actually _improves_ performance in ID-based collaborative filtering. When there are no raw features to transform, simpler is better.
- *H-GCN vs. Euclidean tree embedding:* A complete binary tree of depth $d$ requires $O(2^d)$
  dimensions in Euclidean space but only $O(d)$ dimensions in hyperbolic space.
- *2-Stage vs. end-to-end GCN:* Pinterest PinSage pattern --- offline graph precomputation is
  the standard for production systems. It decouples graph update frequency from model training frequency.

== Mathematical Background

=== Poincare Ball Model (H-GCN)

$ bb(B)_c^d = { bold(x) in bb(R)^d : c ||bold(x)||^2 < 1 } $

*Exponential map (tangent $arrow$ hyperbolic):*
$ exp_(bold(0))(bold(v)) = tanh(sqrt(c) ||bold(v)||) dot bold(v)/(sqrt(c) ||bold(v)||) $

*Logarithmic map (hyperbolic $arrow$ tangent):*
$ log_(bold(0))(bold(y)) = "arctanh"(sqrt(c) ||bold(y)||) dot bold(y)/(sqrt(c) ||bold(y)||) $

*Poincare distance:*
$ d_(bb(B))(bold(x), bold(y)) = 1/sqrt(c) "arccosh"(1 + (2c ||bold(x) - bold(y)||^2)/((1 - c||bold(x)||^2)(1 - c||bold(y)||^2))) $

Near the boundary: the denominator $arrow 0$, so distance explodes (small Euclidean distance = large hyperbolic distance).
This naturally encodes hierarchical depth.

*Riemannian gradient correction:*
$ nabla_"Riem" f(bold(x)) = ((1 - c||bold(x)||^2)^2)/4 nabla_"Euclid" f(bold(x)) $

*Fermi-Dirac decoder (link prediction):*
$ P("edge" | u, v) = 1/(exp((d_(bb(B))(u,v) - r) slash t) + 1) $

Borrowed from statistical physics. $r$ = margin (Fermi energy), $t$ = temperature.

*Frechet mean (Einstein midpoint approximation):*
$ gamma_i = 1/sqrt(1 - c||bold(x)_i||^2), quad macron(bold(x)) = "proj"((sum_i w_i gamma_i bold(x)_i)/(sum_i w_i gamma_i)) $

The Lorentz factor $gamma_i$ assigns higher weight to boundary points (specialized consumers).

== Financial Application

- *LightGCN*: Captures indirect preferences through multi-hop collaborative signals.
  "Customer A purchases at Starbucks, Customer B purchases at both Starbucks and Ediya" $arrow$ A may prefer Ediya.
- *H-GCN*: The MCC classification hierarchy (Root $arrow$ L1(8) $arrow$ L2($tilde$100) $arrow$ Brand($tilde$50K) $arrow$ Branch($tilde$500K))
  is inherently a tree structure.
- *Co-visitation edges*: Behavioral signals complement the static MCC hierarchy.
  "Starbucks visitors who also visit Ediya within 7 days" $arrow$ edges weighted with exponential time decay.

== I/O Specification

#figure(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt,
    [*Property*], [*LightGCN*], [*H-GCN*],
    [Nodes], [Customers + Merchants (bipartite)], [Merchants only (MCC tree)],
    [Edges], [Customer-Merchant transactions], [Parent-child + Brand co-visitation],
    [Space], [Euclidean $bb(R)^(64)$], [Poincare Ball $bb(B)^8$],
    [Learning], ["Who likes what" (CF)], ["How merchants relate" (hierarchy)],
    [Output], [Customer embedding 64D (direct)], [Merchant emb $arrow$ per-customer agg 27D],
    [FeatureRouter input], [955D subset], [58D subset],
  ),
  caption: [Dual GCN architecture comparison (Phase 0 v3/v4). HGCN receives 58D (merchant\_hierarchy 27D + mcc\_top30\_multihot 31D); LightGCN receives 955D (product\_hierarchy 34D + graph\_collaborative 66D + txn\_lag\_tensor 800D + nba\_label\_multihot 24D + mcc\_top30\_multihot 31D).],
)

=== HGCN Input Detail (27D merchant_hierarchy)

The 27D `merchant_hierarchy` group captures MCC L1→L2 tree structure embedded in Poincaré disk:

#table(
  columns: (auto, auto, 1fr),
  stroke: 0.5pt,
  [*Sub-group*], [*Dim*], [*Content*],
  [L1 Poincaré coords], [4D], [Level-1 MCC category position in Poincaré disk],
  [L2 Poincaré coords], [4D], [Level-2 MCC sub-category position],
  [Brand SVD], [8D], [Brand-level SVD embeddings],
  [Aggregate stats], [4D], [Spend share, frequency, recency by L1 bucket],
  [Depth features], [3D], [Hierarchy depth indicators],
  [Spread features], [4D], [Breadth of MCC usage across tree levels],
)

HGCN *learns MCC L1→L2 tree structure in hyperbolic space* (Poincaré disk). Its role is complementary to LightGCN: HGCN captures "how merchant categories relate hierarchically," while LightGCN captures "which products customers co-hold" (bipartite CF graph).

=== LightGCN Input Detail (955D)

LightGCN receives `product_hierarchy` (34D) + `graph_collaborative` (66D) + `txn_lag_tensor` (800D) + `nba_label_multihot` (24D) + `mcc_top30_multihot` (31D) = 955D total. It operates on the *product-customer bipartite graph* for collaborative filtering; the lag tensor (800D) supplies raw temporal transaction signals to complement graph structure. Complements HGCN's merchant tree focus.

== Implementation Notes

- *2-Stage pipeline:* Stage 1 (offline) --- graph-level training (LightGCN: BPR, H-GCN: self-supervised Fermi-Dirac).
  Embeddings stored as Parquet. Stage 2 (online) --- lookup + lightweight MLP adaptation.
  No graph propagation at inference --- single GPU VRAM friendly.
- Co-visitation edge scale factor of 0.5 is applied to preserve the structural priority of taxonomy edges.

#pagebreak()

// ============================================================
= PersLay / TDA Expert --- Topological Structure
// ============================================================

== Selection Rationale (Why TDA/PersLay?)

Traditional statistics (mean, variance, correlation) summarize distribution moments but miss
the data's _structural shape_ --- cluster connectivity, cyclic consumption patterns,
voids in the spending space.
Two customers with identical mean/variance can have fundamentally different consumption topologies
(one continuous cluster vs. two separated clusters with periodic switching).

Persistent Homology observes data simultaneously at all scales (filtration),
tracking when topological features (connected components $H_0$, loops $H_1$, voids $H_2$) appear and disappear.
It provides three unique properties:
(1) coordinate invariance, (2) multi-resolution analysis without single threshold selection,
(3) mathematically guaranteed noise robustness via the stability theorem.

== Alternative Comparison

#figure(
  table(
    columns: (auto, auto, auto, auto),
    stroke: 0.5pt,
    [*Method*], [*Variable-size PD*], [*End-to-end learnable*], [*Stability guarantee*],
    [Persistence Images], [Fixed grid], [No], [Approximate],
    [Persistence Landscapes], [Function space], [No (Banach space)], [Yes],
    [*PersLay*], [*Yes (phi + rho)*], [*Yes*], [*Yes (inherits)*],
    [Persformer (2024)], [Yes (attention)], [Yes], [Yes],
  ),
  caption: [PD processing method comparison. PersLay has no gradient bottleneck with sum aggregation and handles padding automatically via persistence weighting, ensuring high production stability.],
)

== Mathematical Background

=== PersLay Layer (Carriere et al., 2020)

$ "PersLay"(D) = rho(sum_((b,d) in D) w(b,d) dot phi(b,d)) $

- $phi$: point transformation (RationalHat or Gaussian)
- $w$: weighting function (persistence-based: $w = |d - b|^p$, or learned)
- $rho$: permutation-invariant aggregation (sum, mean, max, or attention)

=== Homology Groups

$ H_k = "Ker"(partial_k) slash "Im"(partial_(k+1)) $

$beta_k = "rank"(H_k)$: the number of independent $k$-dimensional "holes."
$beta_0$ = connected components, $beta_1$ = loops, $beta_2$ = voids.

=== Stability Theorem

$ d_B ("Dgm"(f), "Dgm"(g)) <= ||f - g||_infinity $

Input perturbation is bounded by the maximum change in the filtration function. This mathematically guarantees noise robustness of topological features.

=== Vietoris-Rips Complex

$ sigma = {x_0, ..., x_k} in "VR"_epsilon (X) <==> d(x_i, x_j) <= epsilon, quad forall i,j $

=== Persistence Entropy (Rucco et al., 2016)

$ E = -sum_(i=1)^N p_i log p_i, quad p_i = (d_i - b_i)/(sum_j (d_j - b_j)) $

High entropy = diverse topological features uniformly distributed. Low entropy = a few dominant patterns.

=== Wasserstein-1 Distance for Phase Transition

$ W_1("PD"_1, "PD"_2) = inf_gamma sum_(x in "PD"_1) ||x - gamma(x)||_infinity $

Measures structural change between persistence diagrams of the first-half and second-half transactions.

*Phase transition probability:*
$ P_"transition" = 1/(1 + e^(-2(Delta_"total" - tau))), quad tau = 0.5 $

=== 5-Block Multi-Beta Architecture

#figure(
  table(
    columns: (auto, auto, auto, auto),
    stroke: 0.5pt,
    [*Block*], [*Input*], [*Homology*], [*Output*],
    [short\_beta0], [90-day app logs $[B, 200, 3]$], [$H_0$ (clusters)], [64D],
    [short\_beta1], [90-day app logs], [$H_1$ (loops)], [64D],
    [long\_beta0], [12-month txn $[B, 150, 3]$], [$H_0$ (clusters)], [64D],
    [long\_beta1], [12-month transactions], [$H_1$ (loops)], [64D],
    [long\_beta2], [12-month transactions], [$H_2$ (voids)], [64D],
  ),
  caption: [PersLay 5-block configuration.],
)

Short concat 128D + Long concat 192D + Global stats MLP 32D + Phase transition 10D = *362D $arrow$ final\_mlp $arrow$ 64D*.

=== TDA Offline Feature Pipeline (70D)

*70D = 24D (short) + 36D (long) + 10D (phase):*

- *tda\_short (24D):* 90-day app logs. 6D point cloud per transaction.
  $H_0 + H_1 times 6$ features (entropy, lifetime mean/std/min/max/median) $times 2$ scopes.
- *tda\_long (36D):* 12-month card transactions. $H_0 + H_1 + H_2 times 6$ features $times 2$ scopes.
- *phase\_transition (10D):* PD Distance 4D + Transition Detection 6D.

== Financial Application

- *$H_0$ (connected components):* Separated consumption clusters reveal lifestyle segmentation.
  The merging of "grocery cluster" vs. "travel cluster" at different scales indicates spending diversification.
- *$H_1$ (loops):* Cyclic consumption patterns (monthly: grocery $arrow$ transport $arrow$ entertainment $arrow$ grocery)
  are captured as persistent loops. Strong loops = habitual patterns = predictable behavior.
- *$H_2$ (voids):* 3D voids in the amount-category-time space reveal systematic avoidance patterns
  (e.g., absence of mid-range spending = small + large amounts only).
- *Phase transition:* Wasserstein distance quantifies behavioral change
  (job change, life events, financial crisis).

== I/O Specification

#figure(
  table(
    columns: (auto, auto),
    stroke: 0.5pt,
    [*PersLay Input*], [32D feature subset (FeatureRouter: Snapshot-axis TDA groups) or raw persistence diagrams],
    [*PersLay Output*], [64D expert representation for PLE gate],
    [*TDA Offline Output*], [32D features (tda\_global 16D + tda\_local 16D) integrated into main 1205D tensor],
    [*Computation*], [Ripser++ (GPU) $arrow$ Ripser (CPU) $arrow$ giotto-tda (fallback)],
  ),
  caption: [PersLay / TDA Expert I/O specification. FeatureRouter provides 32D input subset.],
)

== Implementation Notes

- Production config: RationalHatPhi + persistence weighting + sum aggregation.
- Cold-start 4-stage progressive TDA: Day 0 (18D median) $arrow$ 7--30d (9D histogram) $arrow$ \<12m (24D H0,H1) $arrow$ 12m+ (36D full).
- Time-stratified sampling: up to 1000 points per customer, stratified across $k$ time buckets to preserve temporal ordering.

#pagebreak()

// ============================================================
= LightGCN Expert --- Collaborative Filtering
// ============================================================

== Selection Rationale (Why LightGCN?)

In recommendation systems, the user-item interaction graph has a _peer-to-peer_ structure (no hierarchy).
The key question in this structure is "who likes what," and multi-hop collaborative signals
are needed to infer indirect preferences for items without direct interaction.

LightGCN is an architecture that _removes_ feature transformation $W$ and nonlinear activation $sigma$
from NGCF, actually improving performance in ID-based collaborative filtering.
It follows the principle that simpler is better when there are no raw features to transform.

== Alternative Comparison

- *NGCF:* Includes feature transformation + nonlinearity. Unnecessary complexity for collaborative filtering.
- *GraphSAGE:* Neighbor sampling-based, suitable for inductive learning, but LightGCN's full-neighbor
  utilization is superior in transductive settings.
- *GAT:* Attention mechanism increases overfitting risk with marginal gains relative to cost.

== Mathematical Background

=== Message Passing

$ bold(e)_u^((k+1)) = sum_(i in cal(N)_u) 1/(sqrt(|cal(N)_u|) dot sqrt(|cal(N)_i|)) dot bold(e)_i^((k)) $

Symmetric normalization $tilde(A) = D^(-1 slash 2) A D^(-1 slash 2)$ attenuates both sender (popular item) and receiver influence.

*Layer combination:*
$ bold(e)_u^"final" = 1/(L+1) sum_(k=0)^L bold(e)_u^((k)) $

Uniform averaging over all hops (0-hop self + 1,2,3-hop neighbor). Empirically strong without learnable attention weights, preventing overfitting.

*BPR Loss:*
$ cal(L)_"BPR" = -sum_((u, i^+, i^-)) log sigma(hat(y)_(u i^+) - hat(y)_(u i^-)) + lambda ||Theta||^2 $

== Financial Application

- *Cross-selling:* "Customer A purchases at Starbucks, Customer B purchases at both Starbucks and Ediya"
  $arrow$ A may prefer Ediya. Essential for cross-selling in banking.
- *Cold-start mitigation:* Even sparse users can leverage rich signals from similar users through multi-hop propagation.
- *Scalability:* At inference, pre-computed embeddings are looked up without graph propagation, suitable for real-time serving.

== I/O Specification

#figure(
  table(
    columns: (auto, auto),
    stroke: 0.5pt,
    [*Input*], [Product-customer bipartite graph + 955D feature subset (FeatureRouter: product\_hierarchy 34D + graph\_collaborative 66D + txn\_lag\_tensor 800D + nba\_label\_multihot 24D + mcc\_top30\_multihot 31D)],
    [*Embedding dim*], [64D (Euclidean $bb(R)^(64)$)],
    [*Layers*], [3 hops with uniform averaging],
    [*Loss*], [BPR (pairwise ranking)],
    [*Output*], [Customer embedding 64D for PLE gate],
  ),
  caption: [LightGCN Expert I/O specification. Feature input is 955D (FeatureRouter active: product\_hierarchy 34D + graph\_collaborative 66D + txn\_lag\_tensor 800D + nba\_label\_multihot 24D + mcc\_top30\_multihot 31D). Handles product co-holding collaborative filtering; lag tensor provides raw temporal signal complementing graph structure.],
)

== Implementation Notes

- In Stage 1 of the 2-stage pipeline, BPR training is performed offline and embeddings are stored as Parquet.
- L2 regularization is applied only to initial embeddings, not to GCN outputs.
- Separated as a _distinct_ expert from H-GCN to ensure independent gradient flows for Euclidean (CF) and hyperbolic (hierarchy) geometries.
- LightGCN's domain is *product co-holding* (what products customers hold together) --- not MCC tree structure, which is H-GCN's domain.

#pagebreak()

// ============================================================
= Causal Expert (NOTEARS) --- Causal Inference <sec6-causal>
// ============================================================

== Selection Rationale (Why Causal Expert?)

Standard recommendation systems rely on _correlation_ ("customers who bought A also bought B"),
conflating spurious association with true causal effects.
Example: "Premium cardholders have high travel insurance adoption" may be due to income as a
confounder --- the card does not _cause_ insurance adoption.

A/B testing is the gold standard but does not scale
(13 tasks $times$ $N$ strategies = infeasible), is slow (weeks), and provides only
population-level ATE.

== Alternative Comparison

- *A/B Testing:* Gold standard but does not scale, no individual-level ITE.
- *GES/PC Algorithm:* Frequentist causal structure learning but statistical power issues with conditional independence tests.
- *DoWhy/EconML:* Specialized for causal effect estimation but requires separate DAG learning.
- *DAGMA (Bello et al., 2022):* Improved NOTEARS variant, a candidate for future replacement.

== Mathematical Background

=== Feature Compression

$ bold(z) = "Compressor"(bold(x)): bb(R)^(103) arrow bb(R)^(128) arrow bb(R)^(32) $

Reduces 103D feature subset (FeatureRouter active; 644D in full-bank design) to 32 causal variables.
Prevents the DAG adjacency matrix from exploding to large $n^2$ entries.

=== SCM (Structural Causal Model) Intervention

$ hat(bold(z)) = bold(z) + bold(z)(bold(W) circle.small bold(W)) $

- $bold(W) in bb(R)^(32 times 32)$: learnable weighted adjacency matrix (`nn.Parameter`)
- $bold(W) circle.small bold(W)$: element-wise square $arrow$ ensures _non-negative_ causal strength
- $W_(i,j)^2$: causal influence strength from variable $j$ to $i$
- Residual connection ($bold(z) +$) preserves original information while adding causal adjustment

=== NOTEARS Acyclicity Constraint

$ h(bold(W)) = "tr"(e^(bold(W) circle.small bold(W))) - d = 0 $

*Mathematical interpretation:*
The $(i,i)$ diagonal element of $e^(bold(M))$ is the sum of all weighted paths from node $i$ back to itself.
If the graph is a DAG (acyclic), no such return paths exist, so
$e^(bold(M))_(i,i) = 1$ (only the identity matrix contribution), and $"tr"(e^(bold(M))) = d$.

*Taylor 10-term approximation:*
$ e^(bold(M)) approx sum_(k=0)^9 (bold(M)^k)/(k!) $

Detects cycles up to length 10. Cycles longer than 10 hops are practically impossible in a 32-node DAG.

=== DAG Regularization Loss

$ cal(L)_"DAG" = lambda_"acyclic" dot h(bold(W)) + lambda_"sparse" dot ||bold(W) circle.small bold(W)||_1 $

#figure(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt,
    [*Hyperparameter*], [*Default*], [*Role*],
    [`dag_lambda`], [0.01], [Acyclicity constraint strength],
    [`sparsity_lambda`], [0.001], [Edge sparsity (L1 on adjacency)],
    [`n_causal_vars`], [32], [Number of causal nodes],
  ),
  caption: [Causal Expert hyperparameters. When `dag_lambda > 0.1`, $W$ collapses to the zero matrix (DAG penalty dominates task loss, Expert degenerates to the identity function).],
)

== Financial Application

- Removes confounders (income) and learns _directional_ causal relationships to answer
  "Will this recommendation _cause_ behavioral change?"
- The directionality of the SCM ($W_(i,j) eq.not W_(j,i)$) provides explainable recommendation paths.
- Enables individual treatment effect (ITE) estimation from observational data.

== I/O Specification

#figure(
  table(
    columns: (auto, auto),
    stroke: 0.5pt,
    [*Input*], [103D feature subset $[B, 103]$ (FeatureRouter: Snapshot-axis groups)],
    [*Compressor*], [$103 arrow 128 arrow 32$ (causal variables)],
    [*SCM*], [$32 times 32$ learnable adjacency $bold(W)$],
    [*Output*], [64D causal representation + DAG (visualization)],
    [*Auxiliary loss*], [$cal(L)_"DAG" = lambda_"acyclic" dot h(bold(W)) + lambda_"sparse" dot ||bold(W)^2||_1$],
  ),
  caption: [Causal Expert I/O specification. Input is 103D (FeatureRouter active) vs. 644D in full-bank design.],
)

== Implementation Notes

- The NOTEARS paper uses Augmented Lagrangian (strict equality), but our implementation uses the simple penalty method.
  It is more stable in end-to-end MTL joint training environments.
- The Hadamard square ($bold(W) circle.small bold(W)$) of the adjacency matrix $bold(W)$
  enforces non-negative causal strength, unlike the sign-agnostic original.
- Maintained as a _separate_ expert from OT Expert: the NOTEARS acyclicity constraint and
  Sinkhorn entropy regularization have completely different loss surface geometries.

#pagebreak()

// ============================================================
= Optimal Transport Expert --- Distributional Matching
// ============================================================

== Selection Rationale (Why Optimal Transport?)

When comparing a customer's consumption pattern to prototype profiles, the geometric structure
of the feature space must be respected.
KL divergence is undefined when distribution supports do not overlap,
and Euclidean distance ignores the geometric structure of the feature space.
Wasserstein distance reflects the _geometry_ of the ground metric space ---
it captures that "Seoul-Incheon" is closer than "Seoul-Busan" even when probability masses do not overlap.

Complementary to the Causal Expert: Causal answers "Will this recommendation _cause_ behavioral change?" (directionality),
while OT answers "How _close_ is this customer's spending distribution to the target profile?" (geometric distance).

== Alternative Comparison

- *KL Divergence:* Undefined when supports do not overlap. Asymmetric.
- *Total Variation:* Ignores ground metric, compares only distribution shape.
- *Euclidean distance:* Ignores semantic structure of the feature space.
- *Sliced Wasserstein:* Computationally efficient alternative, can be independently swapped in the future.

== Mathematical Background

=== Distribution Projection

$ bold(mu) = "softmax"("DistProjector"(bold(x)_"OT")) in Delta^(32), quad bold(x)_"OT" in bb(R)^(69) $

Transforms 69D feature subset (FeatureRouter active; 644D in full-bank design) into a probability simplex,
representing each customer's feature profile as a discrete distribution over 32 latent categories.

=== Learnable Reference Distributions

$ bold(nu)_k = "softmax"(bold(ell)_k) in Delta^(32), quad k = 1, ..., 16 $

16 learnable prototype customer profiles (`nn.Parameter`), initialized with `randn(16, 32) * 0.1`.

=== PSD Cost Matrix

$ bold(C) = bold(M)^top bold(M) in bb(R)^(32 times 32) $

Positive semi-definiteness (PSD) guarantee: $bold(x)^top (bold(M)^top bold(M)) bold(x) = ||bold(M) bold(x)||^2 >= 0$.
The cost matrix is _learnable_, providing task-optimized semantic distances.

=== Entropy-Regularized Optimal Transport

*Kantorovich problem with entropic regularization:*
$ min_(bold(P) in cal(U)(bold(mu), bold(nu))) chevron.l bold(P), bold(C) chevron.r + epsilon dot H(bold(P)) $

Where:
- $cal(U)(bold(mu), bold(nu)) = {bold(P) >= 0 : bold(P) bold(1) = bold(mu), bold(P)^top bold(1) = bold(nu)}$
- $H(bold(P)) = -sum_(i,j) P_(i,j) log P_(i,j)$: entropy regularization
- $epsilon = 0.1$: regularization coefficient

Entropy regularization makes the problem *strictly convex* (unique solution, guaranteed convergence).

=== Log-Domain Sinkhorn Algorithm

$ bold(u)_"new" = log bold(mu) - "logsumexp"(-bold(C)/epsilon + bold(v)) $
$ bold(v)_"new" = log bold(nu) - "logsumexp"(-bold(C)^top/epsilon + bold(u)) $

Log-domain computation prevents floating-point underflow when $epsilon$ is small.
10 iterations are sufficient for practical convergence.

=== Wasserstein Distance Vector

$ bold(w) = [W(bold(mu), bold(nu)_1), W(bold(mu), bold(nu)_2), ..., W(bold(mu), bold(nu)_(16))] in bb(R)^(16) $

Where $W(bold(mu), bold(nu)_k) = chevron.l bold(P), bold(C) chevron.r_F = sum_(i,j) P_(i,j) dot C_(i,j)$.

This creates a *distributional coordinate system* ---
each customer is positioned by their distances to 16 reference prototypes.

=== Wasserstein Encoder

$ bold(o) = "WassersteinEncoder"(bold(w)): bb(R)^(16) arrow bb(R)^(128) arrow bb(R)^(64) $

== Financial Application

Wasserstein distance quantifies "How different is this customer's spending pattern from the
typical travel/savings/dining profile, and _in which categories and in which direction_ must it shift to match?"

This provides directional information impossible with KL divergence or Euclidean distance.

== I/O Specification

#figure(
  table(
    columns: (auto, auto),
    stroke: 0.5pt,
    [*Input*], [69D feature subset $[B, 69]$ (FeatureRouter: Snapshot-axis groups)],
    [*Distribution*], [$69 arrow 32$ probability simplex $Delta^(32)$],
    [*References*], [16 learnable prototypes $in Delta^(32)$],
    [*Cost matrix*], [Learnable PSD: $bold(M)^top bold(M) in bb(R)^(32 times 32)$],
    [*Sinkhorn*], [10 iterations, log-domain, $epsilon = 0.1$],
    [*Output*], [64D expert representation for PLE gate],
  ),
  caption: [Optimal Transport Expert I/O specification. Input is 69D (FeatureRouter active) vs. 644D in full-bank design.],
)

== Implementation Notes

- Cuturi (2013) uses a fixed cost matrix and single target,
  but our implementation uses a learnable PSD cost matrix and 16 learnable prototypes.
- Sinkhorn iteration count is fixed at 10 to make computation cost predictable during the training loop.
- Synergy with Causal Expert: DeepFM extracts symmetric feature interactions $chevron.l bold(v)_i, bold(v)_j chevron.r$,
  Causal extracts asymmetric directional causality $W_(i,j)^2$,
  OT extracts distance functions (metric) $W(mu, nu_k)$.
  The three experts extract _mathematically completely different_ structures; with FeatureRouter active,
  each operates on its own input subset (causal=129D, ot=95D, deepfm=977D) rather than the same full 1205D input.

#pagebreak()

// ============================================================
= adaTT (Adaptive Task-aware Transfer) <sec8-adatt>
// ============================================================

#block(
  width: 100%,
  inset: 10pt,
  radius: 4pt,
  fill: rgb("#fff3cd"),
  stroke: (left: 3pt + rgb("#ffc107")),
)[
  *Scale Note (2026-04-15).* adaTT degrades at 13-task scale: loss-level transfer undoes PLE's
  representation-level separation (156 task pairs, combinatorial coupling). adaTT is retained here
  for reference. GradSurgery (PCGrad task-type projection) was tested as a gradient-level alternative
  but showed no meaningful advantage over the PLE-only baseline while incurring significant VRAM
  overhead (retained computation graph); GradSurgery was not adopted for production. The production
  configuration disables both adaTT and GradSurgery.
]

== Motivation: Negative Transfer in Multi-Task Learning

When 13 simultaneous tasks share expert parameters,
gradient conflicts cause negative transfer.
Three fundamental limitations of fixed-tower MTL:
(1) The shared backbone equally affects all tasks --- no mechanism to detect/prevent one task's optimization from degrading another's predictions,
(2) Unable to measure which of the 13 task pairs help or harm each other,
(3) Fixed weights cannot track task relationships that change across training stages.

== Core Mechanism: Gradient Cosine Similarity

$ cos(theta_(i,j)) = (bold(g)_i dot bold(g)_j)/(||bold(g)_i|| dot ||bold(g)_j||) $

Where $bold(g)_i = nabla_theta cal(L)_i$ is the gradient of shared expert parameters with respect to task $i$'s loss.

*Why cosine (not Euclidean):*
(1) Scale-invariant --- compares only direction even when task losses differ by orders of magnitude,
(2) Interpretable range $[-1, 1]$ $arrow$ maps directly to positive/negative transfer,
(3) Efficient computation: after normalization, a single matrix product $hat(bold(G)) hat(bold(G))^top$ computes all $n^2$ similarities.

=== EMA Stabilization

$ bold(A)_t = alpha dot bold(A)_(t-1) + (1 - alpha) dot cos(theta_t) $

$alpha = 0.9$ (effective window $approx 10$ observations). Equivalent to a 1st-order IIR low-pass filter,
removing high-frequency batch noise while preserving true task relationship trends.

=== Transfer-Enhanced Loss

$ cal(L)_i^"adaTT" = cal(L)_i + lambda dot sum_(j eq.not i) w_(i arrow j) dot cal(L)_j $

$lambda = 0.1$ (10% influence from other tasks), `max_transfer_ratio = 0.5`
(transfer loss cannot exceed 50% of the original loss).

*Gradient effect:*
$ nabla_theta cal(L)_i^"adaTT" = nabla_theta cal(L)_i + lambda sum_(j eq.not i) w_(i arrow j) nabla_theta cal(L)_j $

The second term is a correction vector that adjusts shared parameters toward directions beneficial to multiple tasks.

=== Transfer Weight Computation (4-stage)

$ bold(R) = (bold(W) + bold(A)) dot (1 - r) + bold(P) dot r $
$ bold(R)_(i,j) arrow.l 0 quad "if" bold(A)_(i,j) < tau_"neg" $
$ bold(R)_(i,i) = 0 $
$ w_(i arrow j) = "softmax"(bold(R)_(i,j) / T) $

- $bold(W)$: Learnable transfer weights (`nn.Parameter`, initialized to 0)
- $bold(A)$: EMA affinity matrix
- $bold(P)$: Group Prior matrix (domain knowledge)
- $r$: Prior blend ratio ($0.5 arrow 0.1$ via annealing)
- $tau_"neg" = -0.1$: Negative transfer threshold
- $T = 1.0$: Softmax temperature

=== Group Prior

#figure(
  table(
    columns: (auto, auto, auto, auto),
    stroke: 0.5pt,
    [*Group*], [*Tasks*], [*Intra-strength*], [*Business Meaning*],
    [engagement], [cross\_sell\_count, will\_acquire\_deposits,\ will\_acquire\_investments, will\_acquire\_accounts,\ will\_acquire\_lending, will\_acquire\_payments], [0.8], [Customer engagement/conversion],
    [lifecycle], [churn\_signal, product\_stability,\ segment\_prediction], [0.7], [Customer lifecycle],
    [value], [nba\_primary], [0.6], [Customer value/behavior],
    [consumption], [next\_mcc, mcc\_diversity\_trend, top\_mcc\_shift], [0.7], [Consumption pattern],
  ),
  caption: [adaTT task group definitions. Inter-group strength: 0.3.],
)

*Prior Blend Annealing (Bayesian interpretation):*
$ r(e) = r_"start" - (r_"start" - r_"end") dot min((e - e_"warmup")/(e_"freeze" - e_"warmup"), 1.0) $

$r: 0.5 arrow 0.1$ implements a prior-to-posterior transition:
early training relies on domain knowledge (prior), later training trusts observed gradient affinity (likelihood).

=== 3-Phase Schedule

#figure(
  table(
    columns: (auto, auto, auto, auto),
    stroke: 0.5pt,
    [*Phase*], [*Period*], [*Behavior*], [*Purpose*],
    [Warmup], [Epoch 0 -- warmup], [Compute affinity only, no transfer loss], [Accumulate stable affinity data],
    [Dynamic], [warmup -- freeze], [Active transfer + annealing prior], [Learn and apply task relationships],
    [Frozen], [freeze -- end], [Fixed transfer weights (detached)], [Fine-tuning stabilization, remove gradient overhead],
  ),
  caption: [adaTT 3-phase training schedule.],
)

=== Negative Transfer Detection

$ bold(R)_(i,j) = cases(bold(R)_(i,j) & "if" bold(A)_(i,j) > tau_"neg", 0 & "otherwise") $

$tau_"neg" = -0.1$ (not 0) allows weakly negative correlations (possible noise) while
blocking clearly adversarial gradients.

== Analogy with Transformer Self-Attention

#figure(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt,
    [*Role*], [*Transformer*], [*adaTT*],
    [Query], [Current token's query], [Current task's gradient direction],
    [Key], [Other tokens' response], [Other tasks' gradient directions],
    [Similarity], [$Q K^top / sqrt(d_k)$], [Gradient cosine similarity],
    [Value], [Other tokens' information], [Other tasks' loss values],
    [Output], [Weighted context], [Transfer loss],
  ),
  caption: [Structural analogy between adaTT and Transformer self-attention.],
)

== Alternative Comparison

#figure(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt,
    [*Method*], [*Mechanism*], [*Limitation vs. adaTT*],
    [Fixed Weighting], [Manual task weights], [Cannot measure dynamic affinity],
    [GradNorm], [Gradient magnitude balancing], [Uses only magnitude, not direction],
    [PCGrad], [Project conflicting gradients], [Cannot selectively transfer positive knowledge],
    [Nash-MTL], [Nash bargaining for Pareto], [Computation cost ($O(n^2 d)$ vs optimization)],
    [CAGrad], [Worst-case gradient alignment], [Lacks separation (modularity) of measurement and application],
  ),
  caption: [MTL approach comparison.],
)

== Implementation Notes

- *Phase 1 (Shared Expert Pretrain):* adaTT active --- gradient extraction + transfer loss for 15 epochs.
- *Phase 2 (Cluster Finetune):* adaTT disabled --- Shared experts frozen, only cluster-specific sub-heads trained for 8 epochs.
- *Warmup/freeze epochs for short ablations:* When running short ablations (3--5 epochs), set `warmup_epochs=1` and `freeze_epochs=3` to ensure the dynamic phase is reached. With default values designed for 15+ epoch runs, short ablations may never exit the warmup phase.
- A lightweight variant of Hypernetworks (Ha et al., 2017): uses observed gradients instead of learned task embeddings
  as the conditioning signal, enabling zero-delay adaptation to changing task relationships.
- `detect_negative_transfer()` API returns the list of adversarial tasks for each task
  (e.g., `{"churn_signal": ["will_acquire_lending", "nba_primary"]}`).

#pagebreak()

// ============================================================
= Feature Engineering Overview --- 11 Disciplines, 1205D Input (17 Groups) + 6 Passthrough = 1211D <sec9-features>
// ============================================================

== Feature Engineering Philosophy

Traditional statistical features view data through a single lens.
Other academic disciplines have developed mathematical tools optimized for specific pattern types
over centuries. The key insight is *structural isomorphism* ---
when mathematical relationship structures are identical regardless of domain objects,
the formulas capture the same patterns irrespective of the surface domain.

== Feature Classification by 11 Academic Disciplines

#figure(
  table(
    columns: (auto, auto, auto, auto),
    stroke: 0.5pt,
    [*Discipline*], [*Dim*], [*Pattern*], [*Mathematical Tool*],
    [Economics (PIH)], [8D], [Permanent/transitory income decomposition], [HP Filter, Kalman Filter],
    [Economics (Micro)], [9D], [Elasticity, consumption smoothing], [Arc elasticity, HHI, Shannon entropy],
    [Chemical Kinetics], [6D], [Behavioral change velocity/acceleration], [Arrhenius equation, finite differences],
    [SIR Epidemiology], [5D], [Category adoption diffusion], [ODE compartmental model],
    [Routine Activity (Criminology)], [5D], [Regularity, burstiness, temporal anomalies], [Circular statistics, burstiness index],
    [Wave Interference], [8D], [Periodic decomposition, phase synchronization], [FFT, Hilbert transform, PLV],
    [TDA (Topology)], [70D], [Consumption topological structure, phase transition], [Persistent Homology, Wasserstein-1],
    [GMM (Statistics)], [22D], [Soft segmentation, uncertainty], [EM algorithm, BIC],
    [HMM (Probabilistic)], [48D], [Latent state transitions, dynamic trajectories], [Forward-Backward, Viterbi, ODE dynamics],
    [MAB (Decision Theory)], [4D], [Explore/exploit balance], [HHI trend, recency-weighted entropy],
    [Graph Embedding], [111D], [Collaborative filtering + hierarchical structure], [LightGCN (64D) + H-GCN (47D)],
  ),
  caption: [Feature framework derived from 11 academic disciplines. Santander implementation: 1211D input (17 feature groups). New groups: txn\_lag\_tensor 800D (lag\_extractor), txn\_rolling\_stats 20D (rolling\_stats\_extractor), nba\_label\_multihot 24D + mcc\_top30\_multihot 31D (topn\_multihot\_extractor). This table reflects the full-bank conceptual design; Santander-specific dims differ (see feature\_groups.yaml).],
)

== Economics Features (17D)

=== Friedman PIH (8D)

Decomposes observed income into permanent and transitory components:
$ Y_t = Y_t^P + Y_t^T $

Consumption function: $C_t = k(r, w, u) dot Y_t^P$ ---
consumers spend proportionally to permanent income, while transitory income goes to savings/investment.

Three estimation methods are supported:
- Moving Average: $hat(Y)_t^P = 1/L sum_(i=0)^(L-1) Y_(t-i)$, $L=12$
- HP Filter: $min_tau {sum_t (Y_t - tau_t)^2 + lambda sum_t [(tau_(t+1)-tau_t) - (tau_t - tau_(t-1))]^2}$, $lambda=14400$
- Kalman Filter: $K_"ss" approx 0.27$ (73% prior weight, 27% observation weight)

=== Microeconomic Behavior (9D)

*Income Elasticity:*
$ epsilon_Y = (partial Q)/(partial Y) dot Y/Q = (d ln Q)/(d ln Y) $

$epsilon_Y > 1$: luxury behavior, $0 < epsilon_Y < 1$: necessity, $epsilon_Y < 0$: inferior good.

*Consumption Smoothing (Hall, 1978):*
$ C_t = C_(t-1) + epsilon_t $

Feature: `consumption_smoothing = mu/sigma` (Sharpe ratio analogue for consumption).

*Spending Diversification (Shannon Entropy):*
$ H = -sum_(i=1)^N s_i ln(s_i) $

*Category Concentration (HHI):*
$ "HHI" = sum_i s_i^2 $

Shannon entropy and HHI are special cases of Renyi entropy $H_alpha$ ($alpha arrow 1$: Shannon, $alpha = 2$: HHI).

== Multidisciplinary Features (24D)

Four modules capture nearly orthogonal projections of the data:
- *Chemical Kinetics:* The _differential structure_ of time (1st, 2nd derivatives)
- *Epidemic Diffusion:* _State space transition structure_ ($S arrow I arrow R$)
- *Crime Pattern:* _Statistical texture_ of time series (periodicity, clustering, dispersion)
- *Interference:* _Frequency-domain spectral structure_ (FFT, coherence, phase)

Cross-module combinations reveal patterns invisible within individual modules:
Example: High `catalyst_sensitivity` + high `burstiness` = payday burst spender
(optimal target for beginning-of-month promotions).

== GMM Clustering Features (22D)

$ p(bold(x)) = sum_(k=1)^K pi_k cal(N)(bold(x) | bold(mu)_k, bold(Sigma)_k) $

$K = 20$ clusters, $D = 40$ input dimensions, full covariance.
Output: 20D cluster probabilities $gamma_(n k)$ + cluster ID + cluster entropy.

*Key advantage of GMM over K-Means:*
`GroupTaskExpertBasket`'s 20 cluster sub-head outputs are weighted-ensembled using $gamma_(n k)$.
Soft assignment improves recommendation quality for boundary customers.

== HMM Triple-Mode Features (48D)

Three parallel Hidden Markov Models:

#figure(
  table(
    columns: (auto, auto, auto, auto),
    stroke: 0.5pt,
    [*Mode*], [*States*], [*Time Scale*], [*Target Tasks*],
    [Journey (AICRA)], [5], [Days/weeks], [CTR, CVR],
    [Lifecycle], [5], [Months/years], [Churn, Retention],
    [Behavior], [6], [Monthly patterns], [NBA, balance\_util],
  ),
  caption: [HMM Triple-Mode configuration. Each mode 16D = state probs + meta features + ODE dynamics. Total 48D.],
)

*ODE Dynamics Bridge (6D per mode):*
Extracted from Viterbi state trajectory --- velocity, acceleration, Lyapunov exponent,
cycle period, attractor strength, trajectory length.

== Five-Axis Feature Taxonomy

The full system's 734D main tensor + 68D separate input spans 5 feature axes:

+ *Static/Snapshot:* demographics, account status
+ *Time-series:* Mamba/LNN-derived temporal patterns
+ *Hierarchical:* merchant hierarchy, graph embeddings
+ *Item/Product:* product interaction features
+ *Model-derived:* HMM 5D summary, Bandit 4D, LNN statistics 18D

== Two-Level Ensemble Architecture

The overall architecture is a *two-level ensemble*:
- *Level 1:* Within the Temporal Expert, Mamba/LNN/PatchTST are combined via learned gating
- *Level 2:* Across all 7 Shared Experts (PersLay, DeepFM, Temporal, LightGCN, H-GCN, Causal, OT),
  CGC Gate Attention performs per-task combination

This hierarchical ensemble ensures both intra-expert diversity (temporal multi-resolution) and
inter-expert complementarity (pattern/topology/temporal/relational/causal/distributional).

// ============================================================
// References
// ============================================================

#pagebreak()

= Further Reading

- *Feature interaction*: Guo et al. IJCAI 2017 (DeepFM); Rendle ICDM 2010 (FM); Wang et al. KDD 2017, WWW 2021 (Deep Cross v1/v2).
- *Time series*: Gu & Dao NeurIPS 2023 (Mamba); Hasani et al. AAAI 2021 (LNN); Nie et al. ICLR 2023 (PatchTST).
- *Graph*: He et al. SIGIR 2020 (LightGCN); Chami et al. NeurIPS 2019 (HGCN); Nickel & Kiela NeurIPS 2017 (Poincare).
- *TDA / PersLay*: Carriere et al. AISTATS 2020; Cohen-Steiner et al. DCG 2007.
- *Causal / NOTEARS / DAGMA*: Pearl 2009; Zheng et al. NeurIPS 2018; Bello et al. ICML 2022.
- *OT / Sinkhorn*: Cuturi NeurIPS 2013; Villani 2009.
- *MTL*: Kendall et al. CVPR 2018 (Uncertainty); Lin et al. ICCV 2017 (Focal); Yu et al. NeurIPS 2020 (PCGrad); Fifty et al. NeurIPS 2021.
