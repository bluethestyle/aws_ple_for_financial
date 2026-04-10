// ============================================================
// Causal Expert & Optimal Transport Expert Technical Reference
// AIOps PLE for Financial Recommendation
// Anthropic Design System
// ============================================================

#let anthropic-bg = rgb("#F0EFEA")
#let anthropic-text = rgb("#141413")
#let anthropic-accent = rgb("#CC785C")
#let anthropic-muted = rgb("#6B7280")
#let anthropic-rule = rgb("#D1D5DB")

#set document(
  title: "Causal Expert & Optimal Transport Expert Technical Reference",
  author: ("Author 1", "Author 2"),
)

#set page(
  paper: "a4",
  margin: (top: 2.5cm, bottom: 2.5cm, left: 2.5cm, right: 2.5cm),
  fill: anthropic-bg,
  header: context {
    if counter(page).get().first() > 1 [
      #set text(size: 7.5pt, font: "New Computer Modern", fill: anthropic-muted, tracking: 0.12em)
      #smallcaps[Tech Reference]
      #h(1fr)
      #smallcaps[Causal Expert & Optimal Transport Expert]
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

#set text(font: "New Computer Modern", size: 10pt, fill: anthropic-text)
#set par(justify: true, leading: 0.8em, spacing: 1.5em)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")

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

// ============================================================
// Title
// ============================================================
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
    Causal Expert & Optimal Transport Expert
  ]

  #v(0.3em)

  #text(size: 14pt, fill: anthropic-muted)[
    Technical Reference: NOTEARS Causal Inference and Sinkhorn Optimal Transport
  ]

  #v(1em)

  #text(size: 11pt, fill: anthropic-text)[
    Author 1#super[1], Author 2#super[1]
  ]

  #v(0.3em)

  #text(size: 9pt, fill: anthropic-muted, style: "italic")[
    #super[1]Organization Name \
    contact\@org.com
  ]

  #v(0.6cm)
  #line(length: 30%, stroke: 0.5pt + anthropic-rule)
]

#v(1fr)
#pagebreak()

#set page(
  header: context {
    if counter(page).get().first() > 1 [
      #set text(size: 7.5pt, font: "New Computer Modern", fill: anthropic-muted, tracking: 0.12em)
      #smallcaps[Tech Reference]
      #h(1fr)
      #smallcaps[Causal Expert & Optimal Transport Expert]
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

// ============================================================
// Abstract
// ============================================================
#block(
  width: 100%,
  inset: (left: 14pt, right: 14pt, top: 10pt, bottom: 10pt),
  stroke: (left: 2pt + anthropic-accent),
)[
  #text(weight: "bold")[Abstract.]
  This document describes the theoretical background, mathematical structure, and implementation
  details of the Causal Expert and the Optimal Transport (OT) Expert — two of the seven Shared
  Experts in the PLE-based financial recommendation system.
  The Causal Expert combines Pearl's Structural Causal Model (SCM) with the NOTEARS continuous
  acyclicity constraint to learn causal relationships among features, while the OT Expert computes
  Wasserstein distances between customer distributions and prototype distributions via Sinkhorn
  entropy-regularized optimal transport.
  Both experts receive the same normalized features as input, yet extract fundamentally different
  mathematical structures: asymmetric causal directionality and distributional geometry, respectively.
  Numerical stability issues encountered under FP16 mixed-precision training and their resolutions
  are also documented.

  #v(0.3em)
  #text(weight: "bold")[Keywords:]
  Causal Inference, NOTEARS, SCM, Optimal Transport, Sinkhorn,
  Wasserstein Distance, DAG, PLE, Financial Recommendation
]

#v(1em)

#block(
  width: 100%,
  inset: (left: 14pt, right: 14pt, top: 10pt, bottom: 10pt),
  stroke: (left: 2pt + anthropic-accent),
)[
  #text(weight: "bold", fill: anthropic-accent)[Design vs. Implementation Note.]
  This document is written against the full-bank design (734D).
  The current Santander benchmark implementation uses 316D (12 feature groups).
]

#v(0.5em)

#outline(indent: 1.5em, depth: 3)

#pagebreak()

// ============================================================
= Causal Expert
// ============================================================

== Pearl Causal Inference Background

=== The Three Stages of Causal Inference

Judea Pearl categorized the epistemological levels of causal inference into three stages.

*Stage 1: Association.* Identifies correlations among variables from observational data.
$P(Y | X)$ --- "How does the probability of $Y$ change when $X$ is observed?" Traditional
recommendation systems operate at this level. The observation that "premium card holders have
higher travel insurance enrollment rates" is an association; it does not identify causal direction.

*Stage 2: Intervention.* Estimates the effect of artificially manipulating a variable.
$P(Y | "do"(X = x))$ --- "If $X$ is set to $x$, how does $Y$ change?"
At this level, the influence of confounding variables can be controlled.
Example: holding income level as a confounder constant, estimate the pure causal effect
of offering a premium card on insurance enrollment.

*Stage 3: Counterfactual.* "Would the outcome have been different if a different action had been taken?"
$P(Y_x | X = x', Y = y')$ --- estimates the individual treatment effect (ITE).
In the recommendation context, this corresponds to "What would have happened if product B,
rather than A, had been recommended to this customer?"

=== The Confounding Trap in Correlation

Relying on correlations in recommendation systems introduces vulnerability to confounding variables.
In the structure "premium card ownership $arrow.l$ high income $arrow$ travel insurance enrollment,"
high income is a confounder that influences both variables. A correlation-based system recommends
travel insurance to premium card holders, but providing the card for free may not change
the insurance enrollment rate at all.

The Causal Expert learns causal directions and strengths among variables via the adjacency matrix
$bold(W)$, embedding an intervention-level (Stage 2) reasoning structure into recommendations.
$W_(i,j)^2$ encodes "how much does variable $i$ change when variable $j$ is intervened upon."


== NOTEARS Continuous DAG Constraint

=== The Challenge of DAG Structure Learning

Learning directed acyclic graph (DAG) structure has traditionally been an NP-hard problem.
The number of possible DAGs over $d$ variables grows super-exponentially
(approximately $4.2 times 10^18$ for $d = 10$).
Existing constraint-based approaches (PC algorithm) and score-based approaches (GES) could not
escape the combinatorial search bottleneck.

=== NOTEARS Core Idea

NOTEARS (Non-combinatorial Optimization via Trace Exponential and Augmented lagRangian for
Structure learning) by Zheng et al. (NeurIPS 2018) completely bypasses combinatorial search
by transforming the acyclicity condition into a differentiable continuous equality constraint.

$ h(bold(W)) = "tr"(e^(bold(W) circle.tiny bold(W))) - d = 0 $ <notears>

Where:
- $e^(bold(M))$: matrix exponential
- $"tr"(dot)$: trace
- $d$: number of causal variables (in this implementation, $d = 32$)
- $bold(W) circle.tiny bold(W)$: Hadamard (element-wise) square, guaranteeing non-negative causal strength

*Mathematical interpretation.* The $(i,i)$ entry of the $k$-th power $bold(A)^k$ is the
weighted count of length-$k$ paths from node $i$ back to itself. Therefore, the diagonal entry
$(e^(bold(A)))_(i,i)$ of the matrix exponential $e^(bold(A)) = sum_(k=0)^(infinity) bold(A)^k / k!$
is the weighted sum of all cyclic paths of any length returning to node $i$.

In a DAG, no cyclic paths exist, so the diagonal entries of all terms with $k >= 1$ are zero.
Only the $k = 0$ (identity matrix) contribution of 1 remains, giving $e^(bold(A))_(i,i) = 1$
and $"tr"(e^(bold(A))) = d$. Therefore $h(bold(W)) = 0$ guarantees the graph is a DAG.

=== Taylor 10-Term Approximation

Direct computation of the matrix exponential requires $O(d^3)$ operations such as eigendecomposition.
The implementation approximates by accumulating terms $k=1$ through $k=10$ of the Taylor series.

$ e^(bold(M)) = bold(I) + sum_(k=1)^(infinity) frac(bold(M)^k, k!) approx bold(I) + sum_(k=1)^(10) frac(bold(M)^k, k!) $ <taylor>

In the actual implementation, the $k=0$ term (identity matrix) is not included in the loop;
the $-d$ subtraction in $h(bold(W)) = "tr"(e^(bold(M))) - d$ eliminates the identity matrix
contribution. The loop accumulates $k=1$ through $k=10$:

```python
M_power = torch.eye(d, device=W.device, dtype=W.dtype)
for i in range(1, 11):       # i = 1, 2, ..., 10  (k=1..10)
    M_power = M_power @ W_sq / i  # k! is constructed automatically
    h = h + torch.trace(M_power)  # accumulate tr(M^k / k!)
# h corresponds to tr(e^M) - d (equivalent to subtracting the trace of the k=0 identity term)
```

Ten terms suffice to detect all cyclic paths of length up to 10; in a 32-node DAG, cycles
of more than 10 hops are practically unlikely. When the elements of $bold(W)$ are small
(initialized at 0.01), higher-order terms decay rapidly, providing sufficient practical precision.


== SCM (Structural Causal Model) Intervention

=== Feature Compressor

A bottleneck module that projects high-dimensional features into the DAG variable space.

$ bold(z) = "Compressor"(bold(x)) : quad RR^(644) -> RR^(128) -> RR^(32) $ <compressor>

Architecture: `Linear(644, 128)` $arrow$ `LayerNorm(128)` $arrow$ `SiLU` $arrow$ `Dropout(0.2)` $arrow$ `Linear(128, 32)`.

Performing causal analysis directly on all 644 raw features would yield
$644^2 approx 414{,}000$ pairwise relationships.
The Compressor summarizes these into 32 core causal variables, alleviating the computational
burden of DAG learning.

=== Causal Intervention Operation

$ hat(bold(z)) = bold(z) + bold(z)(bold(W) circle.tiny bold(W)) $ <scm>

Where:
- $bold(W) in RR^(32 times 32)$: learnable weighted adjacency matrix (`nn.Parameter`)
- $bold(W) circle.tiny bold(W)$: Hadamard square guaranteeing non-negative causal strength
- $W_(i,j)^2$: causal influence strength in the direction variable $j arrow$ variable $i$
- residual connection ($bold(z) +$): adds causally adjusted correction while preserving original information

This generates a corrected representation $hat(bold(z))$ of each customer's latent characteristic
vector $bold(z)$, adjusted by the causal influence that other characteristics exert on it.
The result is a causally adjusted representation, as opposed to a merely correlated one.

The adjacency matrix $bold(W)$ is initialized at small scale via `torch.randn(32, 32) * 0.01`
to ensure stability in the early phase of training.

=== Causal Encoder

Transforms the causally intervened representation into the final output dimension.

$ bold(o) = "CausalEncoder"(hat(bold(z))) : quad RR^(32) -> RR^(128) -> RR^(64) $ <causal-encoder>

Architecture: `Linear(32, 128)` $arrow$ `LayerNorm(128)` $arrow$ `SiLU` $arrow$ `Dropout(0.2)` $arrow$ `Linear(128, 64)` $arrow$ `LayerNorm(64)` $arrow$ `SiLU`.

The 64D output shares the same dimension as other experts (PersLay, DeepFM, etc.),
enabling direct comparison and weighted summation in the CGC gate attention mechanism.


== DAG Regularization Loss

Two regularization terms are used during training to maintain the DAG structure.

$ cal(L)_("DAG") = lambda_("acyclic") dot h(bold(W)) + lambda_("sparse") dot ||bold(W) circle.tiny bold(W)||_1 $ <dag-loss>

#table(
  columns: (1fr, 0.7fr, 2fr),
  inset: 8pt,
  stroke: 0.5pt + luma(200),
  table.header[*Hyperparameter*][*Default*][*Role*],
  [`dag_lambda`], [`0.01`], [Acyclicity constraint strength],
  [`sparsity_lambda`], [`0.001`], [Adjacency matrix sparsity (L1)],
  [`n_causal_vars`], [`32`], [Number of causal variables = number of DAG nodes],
)

#v(0.3em)

The acyclicity term $h(bold(W))$ enforces the no-cycle constraint, while the sparsity term
$||bold(W) circle.tiny bold(W)||_1 = sum_(i,j) W_(i,j)^2$ encourages retaining only a small
number of meaningful causal relationships among the possible $32 times 32 = 1{,}024$ edges.

*Difference from the original paper.* The original paper strictly satisfies the equality
constraint $h(bold(W)) = 0$ via an augmented Lagrangian, whereas this implementation relaxes
it to a simple penalty $lambda dot h(bold(W))$. This is a practical choice to maintain balance
with the other loss terms in MTL while achieving approximate acyclicity.
Convergence to $h(bold(W)) < 0.1$ at the end of training with $lambda = 0.01$ is sufficient.

#block(
  width: 100%,
  inset: 10pt,
  stroke: (left: 3pt + rgb("#d97706")),
  fill: rgb("#fffbeb"),
)[
  #text(weight: "bold")[Caution:] Setting `dag_lambda` $> 0.1$ causes the adjacency matrix
  $bold(W)$ to converge to the zero matrix, destroying the causal structure entirely.
  When the DAG regularization loss dominates the task loss, the expert degenerates into an
  identity function ($hat(bold(z)) approx bold(z)$).
]


== Financial Domain Application

=== Learning Causal Directions Between Behaviors

In the financial domain, the Causal Expert learns causal directions among customer behavioral variables.

- *Confounder control*: From the correlation between "card usage and loyalty," the pure causal
  effect of card usage on loyalty is extracted by controlling for income as a confounding variable.
- *Intervention effect prediction*: Causal grounds for "how will customer behavior change if
  this benefit is provided" are internalized.
- *Interpretability*: The adjacency matrix $bold(W) circle.tiny bold(W)$ can be extracted via
  `get_causal_graph()` and visualized as a heatmap to show which latent causal variables
  strongly influence others.

=== Causal Graph Extraction

```python
def get_causal_graph(self) -> torch.Tensor:
    """graph[i,j] = W[i,j]^2: causal strength of variable j -> variable i"""
    return (self.W * self.W).detach()
```

The returned `[32, 32]` matrix is used for causal path tracing, sparsity monitoring,
and recommendation reason generation (LLM grounding).

=== Original Paper vs. Implementation Comparison

#table(
  columns: (1.2fr, 1.5fr, 1.5fr),
  inset: 8pt,
  stroke: 0.5pt + luma(200),
  table.header[*Item*][*Paper (Zheng et al.)*][*This Implementation*],
  [Purpose], [Learn DAG structure from observational data], [Learn causal relationships among features within an expert],
  [Number of variables], [Tens to hundreds (general-purpose)], [Fixed at 32],
  [Acyclicity], [Augmented Lagrangian (equality constraint)], [Simple penalty: $lambda dot h(bold(W))$],
  [Adjacency matrix], [$bold(W)$ (sign-agnostic)], [$bold(W) circle.tiny bold(W)$ (non-negative)],
  [Training], [Independent optimization], [End-to-end joint MTL training],
  [Output], [DAG adjacency matrix], [64D causal representation + DAG (for visualization)],
)


#pagebreak()

// ============================================================
= Optimal Transport Expert
// ============================================================

== Wasserstein Distance vs. KL Divergence

=== Geometric Perspective on Distribution Comparison

Traditional methods for measuring the difference between two probability distributions include:
- *KL divergence*: $D_("KL")(P || Q) = sum_i P_i log(P_i / Q_i)$
- *JS divergence*: a symmetric version of KL divergence
- *Total variation distance*: $"TV"(P, Q) = 1/2 sum_i |P_i - Q_i|$

These share a common limitation: they ignore the geometric structure of the underlying space.
When distribution $P$ is concentrated in "Seoul," $Q$ in "Incheon," and $R$ in "Busan,"
KL divergence and TV distance yield $"dist"(P, Q) approx "dist"(P, R)$ — when the supports
do not overlap, the distances are identical. Yet intuitively, Seoul–Incheon is closer than Seoul–Busan.

=== Advantages of Wasserstein Distance

The Wasserstein distance (earth mover's distance) reflects the distance structure of the
underlying space.

$ W(bold(mu), bold(nu)) = min_(bold(P) in cal(U)(bold(mu), bold(nu))) chevron.l bold(P), bold(C) chevron.r_F $ <wasserstein>

Where:
- $bold(mu), bold(nu) in Delta^d$: source and target distributions (vectors on the probability simplex)
- $bold(C) in RR^(d times d)$: cost matrix --- $C_(i,j)$ is the cost of transporting one unit of mass from location $i$ to $j$
- $bold(P)$: transport plan --- $P_(i,j)$ is the actual mass transported from $i$ to $j$
- $cal(U)(bold(mu), bold(nu)) = {bold(P) >= 0 : bold(P) bold(1) = bold(mu), bold(P)^top bold(1) = bold(nu)}$: marginal distribution constraints

Key reasons why Wasserstein distance is superior to KL divergence:

#table(
  columns: (1fr, 1.5fr, 1.5fr),
  inset: 8pt,
  stroke: 0.5pt + luma(200),
  table.header[*Property*][*KL Divergence*][*Wasserstein Distance*],
  [Non-overlapping distributions], [Undefined when $Q_i = 0$ ($infinity$)], [Always finite],
  [Geometric awareness], [Ignores distances in the underlying space], [Reflected via the cost matrix],
  [Continuous deformation], [May change discontinuously], [Changes continuously with distribution shift],
  [Transport plan], [Provides only a scalar value], [Provides how mass is transformed ($bold(P)$)],
  [Symmetry], [Asymmetric ($D_("KL")(P||Q) eq.not D_("KL")(Q||P)$)], [Symmetric (metric)],
)


== Sinkhorn Entropy Regularization

=== Computational Limitation of the Kantorovich Problem

The original Kantorovich optimal transport problem is a linear program (LP) with $d^2$ variables
and $2d$ equality constraints, incurring a computational cost of $O(d^3 log d)$.
This is impractical for real-time inference.

=== Entropy Regularization

The key insight of Cuturi (NeurIPS 2013): adding an entropy term fundamentally changes the
structure of the problem.

$ min_(bold(P) in cal(U)(bold(mu), bold(nu))) chevron.l bold(P), bold(C) chevron.r + epsilon dot H(bold(P)) $ <ent-ot>

Where:
- $H(bold(P)) = -sum_(i,j) P_(i,j) log P_(i,j)$: entropy of the transport plan
- $epsilon = 0.1$: regularization coefficient

Adding the entropy term makes the problem strictly convex with a unique solution.
The optimal solution takes the form $bold(P)^* = "diag"(bold(a)) bold(K) "diag"(bold(b))$,
where $bold(K) = exp(-bold(C)/epsilon)$ is the Gibbs kernel and $bold(a), bold(b)$ are
scaling vectors obtained via Sinkhorn alternating normalization.

Computational complexity reduces to $O(d^2 / epsilon^2)$, making it amenable to GPU parallelization.

=== Intuition of Sinkhorn Algorithm

The Sinkhorn algorithm is an alternating projection:

+ Take the Gibbs kernel $bold(K) = exp(-bold(C)/epsilon)$ as the initial transport plan
+ Row normalization: adjust each row sum to match the source distribution $bold(mu)$
+ Column normalization: adjust each column sum to match the target distribution $bold(nu)$
+ Alternating row/column normalization converges to the optimal transport plan satisfying
  both marginal constraints simultaneously


== Log-Domain Sinkhorn Algorithm

=== Numerical Stability Issues

Standard Sinkhorn operates directly on the elements of the Gibbs kernel
$bold(K) = exp(-bold(C)/epsilon)$; when $epsilon$ is small, $exp(-C_(i,j)/epsilon)$
becomes extremely small, causing floating-point underflow.

=== Log-Domain Dual Variable Update

$ bold(u)_("new") = log bold(mu) - "logsumexp"_j (-C_(i,j)/epsilon + v_j) $ <sinkhorn-u>
$ bold(v)_("new") = log bold(nu) - "logsumexp"_i (-C_(i,j)/epsilon + u_i) $ <sinkhorn-v>

Where:
- $bold(u), bold(v)$: log-domain dual variables
- $log bold(K) = -bold(C)/epsilon$: log-domain Gibbs kernel
- `logsumexp`: $log sum_j exp(a_j)$ --- prevents overflow and underflow

*Implementation note:* In the theoretical definition, the $bold(v)$ update uses $bold(C)^top$,
but since the PSD cost matrix in this implementation ($bold(C) = bold(M)^top bold(M)$) is
symmetric, $bold(C)^top = bold(C)$. In code, the same result is obtained by changing the
axis (dim) of `logsumexp` instead of transposing.
- Number of iterations: 10 (default, `sinkhorn_iterations`)

After convergence, the transport plan and Wasserstein distance are computed as:

$ log bold(P) = bold(u) plus.o log bold(K) plus.o bold(v), quad "i.e.," quad log P_(i,j) = u_i + log K_(i,j) + v_j $ <transport-plan>

$ W(bold(mu), bold(nu)_k) = chevron.l bold(P), bold(C) chevron.r_F = sum_(i,j) P_(i,j) dot C_(i,j) $ <wass-dist>


== Learnable Reference Distributions and PSD Cost Matrix

=== Distribution Projection

Transforms customer features into a probability simplex.

$ bold(mu) = "softmax"("DistProjector"(bold(x))) in Delta^(32) $ <dist-proj>

Architecture: `Linear(644, 128)` $arrow$ `LayerNorm(128)` $arrow$ `SiLU` $arrow$ `Dropout(0.2)` $arrow$ `Linear(128, 32)` $arrow$ `softmax`.

Applying softmax ensures the 32 dimensions sum to 1, allowing each dimension to be interpreted
as "the proportion of spending propensity allocated to a category."

=== Learnable Reference Distributions

$ bold(nu)_k = "softmax"(bold(ell)_k) in Delta^(32), quad k = 1, dots, 16 $ <ref-dist>

- $bold(ell)_k$: `nn.Parameter [16, 32]` --- learnable logits
- Initialization: `torch.randn(16, 32) * 0.1`
- Each $bold(nu)_k$ learns the distribution of a typical customer type

The 16 reference distributions are not predefined; rather, they capture naturally emerging
cluster types as the data is trained. For example, $bold(nu)_3$ may learn to represent
a "travel-centric spender" type, while $bold(nu)_7$ may represent a "cost-conscious household" type.

=== PSD Cost Matrix

$ bold(C) = bold(M)^top bold(M) in RR^(32 times 32) $ <psd-cost>

- $bold(M)$: `nn.Parameter [32, 32]` --- learnable basis matrix
- Initialization: $bold(I) + cal(N)(0, 0.01)$

PSD guarantee: $bold(x)^top (bold(M)^top bold(M)) bold(x) = ||bold(M) bold(x)||^2 >= 0$.
Non-negativity of costs is guaranteed automatically, eliminating the physically nonsensical
scenario of "gaining value by transporting." Since the cost is learnable, the model learns
task-specific semantic distances --- e.g., "food $arrow$ dining out is low-cost,
food $arrow$ travel is high-cost."

=== Wasserstein Distance Vector

$ bold(w) = [W(bold(mu), bold(nu)_1), W(bold(mu), bold(nu)_2), dots, W(bold(mu), bold(nu)_(16))] in RR^(16) $ <wass-vec>

Each customer's spending distribution $bold(mu)$ is compared against 16 representative customer
types, yielding a 16-dimensional distance vector. This constitutes a distributional coordinate
system --- the customer is positioned by their distances from 16 reference points.

=== Wasserstein Encoder

$ bold(o) = "WassersteinEncoder"(bold(w)) : quad RR^(16) -> RR^(128) -> RR^(64) $ <wass-encoder>

Architecture: `Linear(16, 128)` $arrow$ `LayerNorm(128)` $arrow$ `SiLU` $arrow$ `Dropout(0.2)` $arrow$ `Linear(128, 64)` $arrow$ `LayerNorm(64)` $arrow$ `SiLU`.


== Financial Domain Application

=== Distributional Shift Between Segments

In the financial domain, the OT Expert performs structural matching between a customer's
spending pattern distribution and typical profile distributions.

The Wasserstein distance quantifies "how different this customer's spending pattern is from
a typical travel / savings / dining-out profile, and in which direction which categories
must shift for the patterns to align." The transport plan $bold(P)$ provides not merely a
scalar distance but a concrete transformation direction.

=== Original Paper vs. Implementation Comparison

#table(
  columns: (1.2fr, 1.5fr, 1.5fr),
  inset: 8pt,
  stroke: 0.5pt + luma(200),
  table.header[*Item*][*Paper (Cuturi 2013)*][*This Implementation*],
  [Distribution input], [Fixed discrete/continuous distributions], [Learned softmax distributions (32D)],
  [Cost matrix], [Fixed (Euclidean, etc.)], [Learnable PSD: $bold(M)^top bold(M)$],
  [Reference distribution], [Single target distribution], [16 learnable prototype distributions],
  [Sinkhorn implementation], [Standard domain (matrix multiplication)], [Log-domain (numerical stability)],
  [Number of iterations], [Until convergence], [Fixed at 10],
  [Output], [Scalar OT distance], [16D distance vector $arrow$ 64D encoding],
)


#pagebreak()

// ============================================================
= FP16 Numerical Stability
// ============================================================

== Challenges of Mixed Precision (AMP) Environment

Enabling AMP (Automatic Mixed Precision) on the g4dn T4 GPU approximately doubles training
speed, but the limited representable range of FP16 ($approx 6 times 10^(-8)$ to $6.5 times 10^4$)
introduces numerical stability issues.

== Causal Expert FP16 Issues

=== Overflow in Taylor Approximation

In the NOTEARS Taylor 10-term approximation, the accumulated matrix powers $bold(M)^k$ can
grow rapidly in magnitude as $k$ increases. When values exceed $6.5 times 10^4$ in FP16,
they become `inf`, which propagates as NaN through the `trace` operation.

*Solution:*
- Keep the initialization scale of $bold(W)$ at `0.01` so that elements of
  $bold(W) circle.tiny bold(W)$ remain on the order of $10^(-4)$
- Prevent rapid growth of $bold(W)$ elements via gradient clipping (`gradient_clip_norm: 5.0`)
- It is recommended to cast the DAG regularization computation to `torch.float32`

=== Adjacency Matrix Element Explosion

When `dag_lambda` is too small, the acyclicity constraint weakens, allowing $bold(W)$ elements
to grow large, which triggers overflow in the higher-order terms of the Taylor approximation.

== OT Expert FP16 Issues

=== Issues in Sinkhorn Log-Domain

The `logsumexp` operation in log-domain Sinkhorn can cause the following issues in FP16:

+ *Softmax probability underflow*: Very small probability values from
  `F.softmax(dist_logits, dim=-1)` fall below the FP16 minimum and become 0.
  Subsequently, `1e-8` in `torch.log(a.clamp(min=1e-8))` is not exactly representable in FP16.

+ *Cost matrix scale*: In computing $-bold(C)/epsilon$ with $epsilon = 0.1$, cost values
  are amplified by a factor of 10. Exceeding the FP16 range causes NaN.

*Solution:*
- Change `clamp(min=1e-7)` to `clamp(min=1e-6)`. Since Sinkhorn internals are cast to FP32,
  `1e-6` is safely representable in FP32.
- Perform Sinkhorn internal operations in `torch.float32` and cast only the output back to FP16
- Use the `@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)` decorator

== Phase 2 NaN Fix Experience

NaN issues encountered during actual training and their resolutions:

#table(
  columns: (0.6fr, 1.2fr, 1.2fr),
  inset: 8pt,
  stroke: 0.5pt + luma(200),
  table.header[*Expert*][*Symptom*][*Cause and Resolution*],
  [Causal],
  [`causal_loss` becomes NaN after epoch 3],
  [$bold(W)$ elements exceed FP16 range. Taylor higher-order term overflow. \
  $arrow$ Cast DAG regularization to FP32, reduce grad clip from 5.0 to 1.0],

  [OT],
  [NaN propagation after Sinkhorn iteration 5],
  [Cost scale amplification in $-bold(C)/epsilon$. \
  $arrow$ Perform Sinkhorn internals in FP32, verify `cost_matrix` initialization $bold(I) + cal(N)(0, 0.01)$],

  [OT],
  [Intermittent NaN in specific batches only],
  [Extreme one-hot distribution in softmax output, `log(0)` propagation. \
  $arrow$ Apply `clamp(min=1e-6)` (safe in FP32 after casting)],
)

#v(0.3em)

#block(
  width: 100%,
  inset: 10pt,
  stroke: (left: 3pt + rgb("#d97706")),
  fill: rgb("#fffbeb"),
)[
  #text(weight: "bold")[FP16 Principle:] For both experts, all core numerical operations
  (Taylor matrix exponential, Sinkhorn log-domain) are performed in FP32; only the input and
  output tensors are cast to FP16. AMP's `GradScaler` automatically skips steps where NaN
  gradients are detected, but frequent skips degrade training quality — the root cause must
  be addressed.
]


#pagebreak()

// ============================================================
= Implementation Notes
// ============================================================

== Common Design Patterns of Both Experts

#table(
  columns: (1.2fr, 1.5fr, 1.5fr),
  inset: 8pt,
  stroke: 0.5pt + luma(200),
  table.header[*Item*][*Causal Expert*][*OT Expert*],
  [Registry Name], [`"causal"`], [`"optimal_transport"`],
  [Input Dimension], [644D (normalized features)], [644D (normalized features)],
  [Latent Space], [32D (causal variables)], [32D (probability simplex)],
  [Output Dimension], [64D], [64D],
  [Core Mechanism], [SCM: $hat(bold(z)) = bold(z) + bold(z)(bold(W) circle.tiny bold(W))$], [Sinkhorn: $W(bold(mu), bold(nu)_k) times 16$],
  [Learnable Parameters], [$bold(W)$ `[32,32]` adjacency matrix], [`reference_logits [16,32]` + `cost_matrix [32,32]`],
  [Regularization Loss], [$cal(L)_("DAG")$ (acyclicity + sparsity)], [None (entropy regularization is implicit)],
  [Interpretable Output], [4D (causal strength, etc.)], [4D (transport cost, etc.)],
)


== PLE Integration

Both experts are integrated into PLE via the same pathway:

```python
# ple_cluster_adatt.py
elif name in ("causal", "optimal_transport"):
    out, _ = expert(inputs.features[:, :644])
```

Only the Causal Expert carries an additional regularization loss:

```python
# ple_cluster_adatt.py -- compute_loss
if self.training and "causal" in self.shared_experts:
    dag_reg = self.shared_experts["causal"].get_dag_regularization()
    total_loss = total_loss + dag_reg
```

The OT Expert has no separate regularization loss; `reference_logits` and `cost_matrix` are
naturally trained via backpropagation through the task loss.


== Mathematical Perspective Comparison of Three Experts

Although DeepFM, Causal, and OT all receive the same normalized 644D features as input,
the mathematical structures they extract are fundamentally different:

#table(
  columns: (0.8fr, 1.3fr, 1fr, 1.3fr),
  inset: 8pt,
  stroke: 0.5pt + luma(200),
  table.header[*Expert*][*Extraction Target*][*Mathematical Property*][*Unique Contribution*],
  [DeepFM], [Symmetric pairwise feature interactions $chevron.l bold(v)_i, bold(v)_j chevron.r$], [Commutative ($i arrow.l.r j$)], [Captures second-order crossings in $O(n k)$],
  [Causal], [Directional causality $W_(i,j)^2$], [Asymmetric, acyclic (DAG)], [Confounder removal, causal direction],
  [OT], [Distributional distance $W(bold(mu), bold(nu)_k)$], [Distance function (metric)], [Geometric position of distributions],
)


== Gradient Flow

=== Causal Expert

```
total_loss
  |-- task_loss -> causal_encoder -> _apply_causal_mechanism
  |                                     -> W (nn.Parameter) <- gradient
  |-- dag_loss -> get_dag_regularization()
                     -> W (nn.Parameter) <- gradient
```

$bold(W)$ receives gradients from both the task loss and the DAG regularization loss.
The balance between the two gradients is controlled by the `dag_lambda` setting.

=== OT Expert

```
total_loss -> wasserstein_encoder -> _sinkhorn_distance -> dist_projector
                                        |-- cost_matrix <- gradient
                                        |-- reference_logits <- gradient
```

Sinkhorn's 10 iterations use unrolled gradients. A larger number of iterations lengthens the
gradient chain, increasing the risk of vanishing/exploding gradients; therefore, gradient
clipping (`gradient_clip_norm: 5.0`) should be used in conjunction.


== Debugging Guide

=== Causal Expert Issues

#table(
  columns: (1fr, 1.2fr, 1.2fr),
  inset: 8pt,
  stroke: 0.5pt + luma(200),
  table.header[*Symptom*][*Cause*][*Action*],
  [DAG collapse ($bold(W) approx 0$)],
  [`dag_lambda` too large ($> 0.1$)],
  [Reduce `dag_lambda` to 0.01 or below],

  [Acyclicity violation ($h(bold(W)) >> 0$)],
  [`dag_lambda` too small or learning rate too large],
  [Gradually increase `dag_lambda`, reduce learning rate],

  [`causal_loss` NaN],
  [Taylor approximation divergence, $bold(W)$ elements too large],
  [Strengthen gradient clip ($5.0 arrow 1.0$), cast to FP32],

  [Expert output is constant],
  [$bold(W)$ training stalled],
  [Set per-expert learning rate, verify initialization],
)

=== OT Expert Issues

#table(
  columns: (1fr, 1.2fr, 1.2fr),
  inset: 8pt,
  stroke: 0.5pt + luma(200),
  table.header[*Symptom*][*Cause*][*Action*],
  [Sinkhorn divergence (NaN/Inf)],
  [$epsilon$ too small ($< 0.01$) or cost scale too large],
  [Set $epsilon >= 0.1$, verify cost initialization],

  [Degenerate distribution (one-hot softmax)],
  [`dist_projector` logits too large],
  [Apply temperature scaling or increase dropout],

  [All distances identical],
  [Reference distribution collapse or zero cost matrix],
  [Monitor diversity of `reference_logits`],

  [Negative OT distance],
  [PSD guarantee on cost matrix violated],
  [Verify `cost_matrix.T @ cost_matrix`],
)


== References

#set text(size: 9pt)

+ Zheng, X., Aragam, B., Ravikumar, P., & Xing, E. P. (2018). DAGs with NO TEARS: Continuous optimization for structure learning. _NeurIPS 2018_.
+ Cuturi, M. (2013). Sinkhorn distances: Lightspeed computation of optimal transport. _NeurIPS 2013_.
+ Pearl, J. (2009). _Causality: Models, reasoning, and inference_ (2nd ed.). Cambridge University Press.
+ Bello, K., Aragam, B., & Ravikumar, P. (2022). DAGMA: Learning DAGs via M-matrices and a log-determinant acyclicity characterization. _ICML 2022_.
+ Kantorovich, L. V. (1942). On the translocation of masses. _Doklady Akademii Nauk_, 37(7-8).
+ Villani, C. (2009). _Optimal transport: Old and new_. Springer.
+ Rubin, D. B. (1974). Estimating causal effects of treatments in randomized and nonrandomized studies. _Journal of Educational Psychology_, 66(5).
+ Sinkhorn, R. (1964). A relationship between arbitrary positive matrices and doubly stochastic matrices. _Annals of Mathematical Statistics_, 35(2).
