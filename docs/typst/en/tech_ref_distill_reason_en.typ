// ============================================================
// Knowledge Distillation · Recommendation Reason Generation Technical Reference
// AIOps PLE for Financial Recommendation
// Anthropic Design System
// ============================================================

#let anthropic-bg = rgb("#F0EFEA")
#let anthropic-text = rgb("#141413")
#let anthropic-accent = rgb("#CC785C")
#let anthropic-muted = rgb("#6B7280")
#let anthropic-rule = rgb("#D1D5DB")

#set document(
  title: "Knowledge Distillation · Recommendation Reason Generation Technical Reference",
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
      #smallcaps[Knowledge Distillation & Recommendation Reasoning]
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
    Knowledge Distillation · Recommendation Reason Generation
  ]

  #v(0.3em)

  #text(size: 14pt, fill: anthropic-muted)[
    Technical Reference: Temperature Scaling, FD-TVS, Grounding, LLM Safety
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
      #smallcaps[Knowledge Distillation & Recommendation Reasoning]
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
  This document describes the theoretical background, mathematical structure, and
  implementation details of the pipeline for distilling knowledge from the PLE-adaTT
  Teacher model into a LightGBM Student, the FD-TVS 4-Stage composite scoring engine,
  feature reverse mapping via 734D grounding, the 2-Layer recommendation reason
  generation architecture, the Safety Gate, and the serving infrastructure.
  Topics covered include dark knowledge transfer via Temperature Scaling ($T = 5$),
  LGBM gain importance-based feature selection (~403D $arrow$ ~140D), LGBM custom objectives,
  the multiplicative combination structure of the FD-TVS master formula,
  2-tier reason generation with L1 Template + L2 LLM, 3-Agent Self-Critique,
  the Safety Gate for compliance with the Financial Consumer Protection Act and AI Basic Act,
  and the end-to-end flow through ONNX + Triton serving.

  #v(0.3em)
  #text(weight: "bold")[Keywords:]
  Knowledge Distillation, Temperature Scaling, Dark Knowledge, LightGBM,
  FD-TVS, Integrated Gradients, Feature Reverse Mapping, Grounding,
  Recommendation Reason Generation, Self-Critique, Safety Gate,
  ONNX, Triton Inference Server, Financial Regulation
]

#v(1em)

#outline(indent: 1.5em, depth: 3)

#pagebreak()

#block(
  width: 100%,
  inset: (x: 1em, y: 0.8em),
  stroke: (left: 3pt + rgb("#d97706")),
  fill: rgb("#fffbeb"),
)[
  #text(weight: "bold", fill: rgb("#92400e"))[Design vs. Implementation Dimensionality Notice] \
  This document is written based on the *full-bank design (734D)*. The current Santander benchmark implementation uses *~349D raw input (13 feature groups), expanding to 403D after Phase 0 log1p expansion*. Dimensional figures in the body (734D, 200D, 140D, etc.) reflect the full-bank design; intermediate dimensions may differ in the actual Santander implementation. For the dimension specifications of the actual implementation, refer to `outputs/phase0/feature_schema.json`.
]

#v(0.5em)

// ============================================================
= Knowledge Distillation
// ============================================================

== Design Philosophy and Motivation

=== Core Problem

The PLE-adaTT Teacher model requires approximately 50M parameters and 20GB of VRAM,
with an inference latency of roughly 50ms per batch of 1,024 samples. For daily batch
inference over millions of customers, GPU costs alone can reach thousands of dollars per
month, and the real-time recommendation SLA of 10ms cannot be met.

=== Solution Strategy

Through Knowledge Distillation (Hinton et al., 2015), the _implicit knowledge (dark knowledge)_
encoded in the Teacher's output distribution is transferred to a LightGBM Student.
The Student achieves approximately 5ms per 1K batch on an 8GB RAM CPU, delivering a
10x speed improvement while keeping performance degradation within 3 percentage points.

=== Teacher--Student Comparison

#table(
  columns: (1fr, 1fr, 1fr),
  inset: 8pt,
  stroke: 0.4pt + luma(200),
  [*Property*], [*PLE Teacher*], [*LGBM Student*],
  [Model Architecture], [PLE-adaTT + Cluster Head], [LightGBM (per-task independent)],
  [Parameters], [$tilde$50M], [$tilde$300--500 trees/task],
  [Memory], [20GB VRAM (GPU)], [$tilde$8GB RAM (CPU)],
  [Inference Speed], [$tilde$50ms / 1K batch], [$tilde$5ms / 1K batch],
  [Feature Dimensionality], [734D (design) / ~349D raw / 403D post-Phase-0 (impl.)], [200D (after IG selection, design basis)],
  [Training Data], [Raw features + labels], [Raw features + Hard Labels + Soft Labels],
)

_Cross-architecture distillation_ (DNN $arrow$ GBDT) is possible because
Knowledge Distillation transfers knowledge through output distributions, not parameters.
"Train with deep learning, serve with GBDT" is the de facto standard in recommendation,
finance, and advertising domains (Borisov et al., NeurIPS 2022).


== Temperature Scaling

=== Mathematical Definition

A Temperature parameter $T$ is introduced into the standard softmax.

$ p_i^T = frac(exp(z_i \/ T), sum_j exp(z_j \/ T)) $ <temp-scaling>

- $T = 1$: Standard softmax. Probability concentrates on the maximum logit.
- $T = 5$ (Default): Smoothed distribution that reveals relative relationships between classes.
- $T arrow infinity$: Uniform distribution. Information is lost.

=== Correspondence with the Boltzmann Distribution

This formula is mathematically isomorphic to the Boltzmann distribution from statistical
mechanics, $P(E_i) = exp(-E_i \/ (k_B T)) \/ Z$. The logit $z_i$ corresponds to
(negative) energy and $T$ to absolute temperature; both fields share the same
exponential family distribution.

=== Temperature Range Configuration

$T$ is constrained to $[3, 7]$.
- $T = 3$: Binary tasks (CTR, CVR, Churn, Retention)
- $T = 5$: Default
- $T = 7$: Multi-class tasks (NBA 12-class, Timing 28-class)
- $T > 10$: Risk of excessive information loss

#block(
  stroke: (left: 2pt + rgb("#6B7280")),
  inset: (left: 8pt, right: 8pt, top: 6pt, bottom: 6pt),
  width: 100%,
)[
  *LGBM Student Temperature* \
  For LGBM students, $T = 1$ (standard softmax) is used when the teacher outputs are passed as soft labels to the GBDT objective. The LGBM custom objective receives probability vectors directly — applying $T > 1$ smoothing on top of LGBM's own internal optimization is redundant and can degrade calibration. The $T = 3$–$5$ range is appropriate for neural-to-neural distillation, not DNN→GBDT.
]


== Dark Knowledge: Information Content of Soft Labels

=== Limitations of Hard Labels

Hard labels (one-hot) encode only $log_2(C)$ bits of information.
For NBA 12-class, this is approximately 3.6 bits, and no relational information
between secondary class choices is captured at all.

=== Rich Structure of Soft Labels

The Teacher's softmax output contains $(C - 1)$ continuous probability values,
providing far more information than $log_2(C)$ bits.

$ p_"teacher" = [0.72, 0.14, 0.08, 0.03, 0.01, ...] $

From this distribution, one can read the primary prediction (72%), the secondary choice
structure (B 14% > C 8%), inter-class similarity, and the level of uncertainty.
This is what Hinton termed _Dark Knowledge_.

=== Label Smoothing Effect

Training with soft labels has a regularization effect analogous to label smoothing.
Whereas hard labels push the model toward extreme probabilities of 0 or 1 (inducing
overfitting), soft labels encourage the model to produce reasonable probability
distributions, thereby promoting generalization.


== Unified Distillation Loss

=== Unified Loss Function

$ cal(L)_"distill" = alpha dot cal(L)_"hard" + (1 - alpha) dot T^2 dot cal(L)_"soft" $ <unified-loss>

- $alpha$: Hard/Soft mixing ratio (Default 0.3, i.e., 30% ground truth + 70% Teacher opinion)
- $T^2$: Gradient magnitude correction scaling

=== Mathematical Derivation of the $T^2$ Correction

From the chain rule, $partial hat(y) \/ partial z = (1\/T) sigma(z\/T)(1 - sigma(z\/T))$.
The $1\/T$ factor accumulates, reducing the total gradient by $1\/T^2$.
Multiplying by $T^2$ restores the original scale. When $T = 5$, the gradient is
reduced to $1\/25$, so multiplying by $T^2 = 25$ compensates.

=== Per-Task Loss Functions

*Binary classification (CTR, CVR, Churn, Retention):*
$ cal(L)_"binary" = alpha dot "BCE"(hat(y), y) + (1 - alpha) dot T^2 dot D_"KL"(p_t || p_s) $
where $p_t = sigma(z_t \/ T)$ and $p_s = sigma(z_s \/ T)$.

*Multi-class (NBA 12-class, Life-stage 6-class, Timing 28-class):*
$ cal(L)_"multi" = alpha dot "CE"(z_s, y) + (1 - alpha) dot T^2 dot D_"KL"("softmax"(z_t \/ T) || "softmax"(z_s \/ T)) $

*Regression (LTV, Engagement):*
$ cal(L)_"reg" = alpha dot "MSE"(hat(y)_s, y) + (1 - alpha) dot "MSE"(hat(y)_s, hat(y)_t) $
Temperature Scaling is not meaningful for regression, so the $T^2$ correction is not applied.

=== KL-Divergence

$ D_"KL"(q || p) = sum_i q_i log frac(q_i, p_i) = underbrace(-H(q), "constant") + underbrace(H(q, p), "cross-entropy") $

Forward KL $D_"KL"("Teacher" || "Student")$ is used. Its mean-seeking characteristic
forces the Student to cover all important modes of the Teacher.
Reverse KL has mode-seeking characteristics and risks ignoring certain modes, making
it unsuitable for distillation.


== LGBM Feature Importance-Based Feature Selection

Feature selection is performed using the *gain importance* of the trained LGBM Student model.
Teacher model IG (Integrated Gradients) attribution is not used — the teacher has already been
distilled, and selecting features from the serving model's (LGBM Student's) perspective aligns
with the design intent.

=== Design Rationale

*Serving model alignment*: The deployed model is the LGBM Student. Gain importance reflects
what features LGBM actually computes. Teacher (PLE) IG attributions can differ substantially
from Student importance due to the architectural gap between deep networks and gradient-boosted trees.

*Operational stability*: LGBM gain computation is a fast post-hoc step on the trained Student.
Teacher IG requires backpropagation through the full PLE graph at production scale
(941K customers, 403 features), causing out-of-memory failures.

*Interpretability alignment*: SHAP/gain explanations derived from the LGBM Student are directly
grounded in the model that generated the recommendation, satisfying EU AI Act Art. 13
(explanations must reflect the actual decision mechanism).

=== 2-Stage Pipeline

_The dimensionality figures below are based on the full-bank design (734D). Intermediate dimensions may differ in the Santander implementation (~349D raw / 403D post-Phase-0)._

*Full-bank design basis:* 734D (design) / ~349D raw / 403D post-Phase-0 (impl.) $arrow$ ~140D (Stage 1) $arrow$ final LGBM input

*Stage 1 -- LGBM Gain Importance Filter:*
Top-$k$ features capturing approximately 95% of cumulative gain importance are retained per task.
The resulting feature set is typically 40--80 features per task (down from 403D).

*Stage 2 -- Mandatory Feature Preservation:*
Seven features always included regardless of LGBM importance:
- TDA: `persistence_entropy`, `landscape_peak`
- Economics: `mpc`, `income_elasticity`, `permanent_income_ratio`
- FinEng: `sharpe_ratio`, `volatility`


== LightGBM Custom Objective

=== Implementation Structure

`DistillationLossNumpy` provides gradient/hessian to LightGBM's `fobj`.

$ "grad" = alpha dot "grad"_"hard" + (1 - alpha) dot T^2 dot "grad"_"soft" $
$ "hess" = alpha dot "hess"_"hard" + (1 - alpha) dot T^2 dot "hess"_"soft" $

=== Soft Label Delivery Technique

Since a LightGBM Dataset supports only `label` and `weight`,
hard labels are passed via `get_label()` and soft labels via `get_weight()`.

=== Distillation Performance Comparison

#table(
  columns: (1fr, 1fr, 1fr, 1fr),
  inset: 8pt,
  stroke: 0.4pt + luma(200),
  [*Method*], [*CTR AUC*], [*NBA Accuracy*], [*LTV RMSE*],
  [LGBM (Hard Label only)], [0.812], [0.634], [1.247],
  [LGBM (Distilled, $T = 5$)], [0.841], [0.698], [1.089],
  [PLE Teacher (original)], [0.856], [0.723], [1.021],
)


== 10-Stage DAG Orchestration

The full distillation pipeline is executed as a 10-stage DAG in `distillation_entrypoint.py`.

#table(
  columns: (auto, 1fr),
  inset: 8pt,
  stroke: 0.4pt + luma(200),
  [*Stage*], [*Content*],
  [1], [Teacher inference (logit generation over the full dataset)],
  [2], [Soft label generation (Temperature Scaling applied)],
  [3], [IG feature selection (734D $arrow$ 200D)],
  [4], [Student training (per-task independent LGBM)],
  [5], [Validation (5 criteria: AUC, RMSE, Accuracy, etc.; metric aggregation is task-type-specific — binary tasks use AUC, regression tasks use RMSE, multi-class tasks use top-k Accuracy)],
  [6], [MLflow model registry registration],
  [7], [ONNX conversion (including ZipMap removal)],
  [8], [Triton packaging (config.pbtxt generation)],
  [9], [Integration validation (ONNX-PyTorch numerical equivalence)],
  [10], [Deployment artifact upload],
)

== Adaptive Distillation Threshold

The distilled LGBM is not accepted unconditionally. Acceptance requires meeting an *adaptive threshold* of at least 2× the random-baseline AUC per task. Below this floor, the student is classified as SKIP --- no worse than random, but the distillation round is discarded and the teacher's direct inference path is activated instead.

#table(
  columns: (auto, 1fr, 1fr),
  inset: 8pt,
  stroke: 0.4pt + luma(200),
  [*Task Type*], [*Floor Threshold*], [*Policy on SKIP*],
  [Binary (CTR/CVR/Churn)], [AUC $>$ 0.5 + $delta_"min"$ (2× random)], [Discard student; activate Layer 2 (teacher direct)],
  [Multi-class (NBA)], [Top-1 Accuracy $>$ 1/C + $delta_"min"$], [Discard student; fall back to popularity rule],
  [Regression (LTV)], [RMSE $<$ baseline mean predictor], [Discard student; use global mean],
)

This prevents deploying a distilled model that is worse than naive baselines, satisfying FSS AI RMF C-1 (pre-launch risk mitigation).

== Calibration: Platt Scaling

FD-TVS Stage 1 requires all task probabilities to be on a common $[0, 1]$ scale. Probability-critical tasks (churn, product_stability, cross_sell_count) apply *Platt scaling* post-distillation.

$ p_"calibrated" = sigma(a dot f(bold(x)) + b) $

where $f(bold(x))$ is the raw LGBM score, and $a, b$ are fitted on a held-out calibration set (never on training data). Calibration reduces Expected Calibration Error (ECE) and ensures that a predicted probability of 0.8 corresponds to an empirical positive rate of approximately 80%.

== Temporal Drift Monitoring

Beyond population-level PSI, three metrics monitor distributional shift over time for the distilled LGBM's input features and prediction outputs:

#table(
  columns: (auto, 1fr, auto),
  inset: 8pt,
  stroke: 0.4pt + luma(200),
  [*Metric*], [*What It Detects*], [*Threshold*],
  [PSI (Population Stability Index)], [Overall input distribution shift], [> 0.25 critical],
  [JSD (Jensen-Shannon Divergence)], [Symmetric distribution distance; more stable for near-zero cells than KL], [> 0.10 alert],
  [Rank Correlation (Spearman)], [Feature importance rank stability across retraining cycles], [< 0.70 alert],
)

All three are computed daily per-feature. The `ConsecutiveDriftTracker` triggers automatic retraining when PSI > 0.25 persists for 3 consecutive days. JSD and rank correlation alerts are forwarded to the OpsAgent for narrative analysis.

#pagebreak()


// ============================================================
= FD-TVS Scoring Engine
// ============================================================

== Design Philosophy

FD-TVS (Financial DNA-based Target Value Score) is a 4-stage composite scoring engine
that combines model predictions with customer-level business context.
The core design principle is *multiplicative combination*:
if any single factor approaches zero, the overall score is vetoed,
enforcing a "risk-first" principle.

A simple summation $sum p_i$ fails to distinguish a window shopper (CTR=0.9, CVR=0.1)
from an actual buyer (CTR=0.5, CVR=0.5) --- both sum to 1.0.
The multiplicative structure grants each dimension veto power.

== Master Formula

$ "FD-TVS" = underbrace(S_"task", "What") times underbrace(W_"DNA", "Who") times underbrace(V_"TDA", "When") times underbrace((1 - R), "Safe?") times underbrace(f dot e, "Appropriate?") $ <fdtvs-master>

#table(
  columns: (auto, 1fr, auto),
  inset: 8pt,
  stroke: 0.4pt + luma(200),
  [*Component*], [*Description*], [*Range*],
  [$S_"task"$], [Task Weighted Sum], [$[0, 1]$],
  [$W_"DNA"$], [Financial DNA Modifier], [$\{0.8, 1.0, 1.2\}$],
  [$V_"TDA"$], [Behavioral Velocity], [$[1.0, 1.15]$],
  [$R$], [Risk Penalty], [$[0, 1]$],
  [$f$], [Fatigue Decay], [$[0, 1]$],
  [$e$], [Engagement Boost], [$[0.85, 1.15]$],
)


== Stage 1: Task-Weighted Sum

$ S_"task" = beta_"CTR" dot p_"CTR" + beta_"CVR" dot p_"CVR" + beta_"NBA" dot p_"NBA" + beta_"LTV" dot p_"LTV" $ <stage1>

Default weights: CVR=0.4 (highest, conversion directly drives revenue), CTR=0.3, NBA=0.2, LTV=0.1.
Since $sum beta_i = 1$ and all $p_i in [0, 1]$, this is a convex combination in the
WSM (Weighted Sum Model, Fishburn 1967) framework, guaranteeing $S_"task" in [0, 1]$.


== Stage 2: Financial DNA Modifier

Based on Friedman's Permanent Income Hypothesis (1957):
consumers make spending decisions based on their permanent (stable) income rather than
transient income fluctuations.

$ W_"DNA" = cases(
  1.2 & "if CV" < 0.2 quad "(Permanent -- stable income)",
  1.0 & "if" 0.2 <= "CV" < 0.5 quad "(Mixed)",
  0.8 & "if CV" >= 0.5 quad "(Transitory -- unstable income)"
) $ <dna-modifier>

Here $"CV" = sigma_"income" \/ mu_"income"$ (coefficient of variation, dimensionless).
Customers with stable income are well-suited for long-term financial products
(annuities, fixed-term deposits), so the recommendation score is boosted by 20%.


== Stage 3: TDA Behavioral Velocity

$ V_"TDA" = 1.0 + gamma_"flare" dot bb(1)["flare"_"detected"] $ <tda-velocity>

$gamma_"flare" = 0.15$. TDA flare detection indicates acceleration of behavioral change,
boosting the score by up to 15%.


== Stage 4: Risk Penalty

$ R = 0.2 dot I_"limit" + 0.3 dot I_"fatigue" + 0.5 dot I_"churn" $ <risk-penalty>

*Rationale for asymmetric weights -- irreversibility:*
- Credit limit exhaustion ($lambda_1 = 0.2$): reversible (limit restored upon repayment)
- Message fatigue ($lambda_2 = 0.3$): partially reversible (recovers over time)
- Customer churn ($lambda_3 = 0.5$): nearly irreversible (re-acquisition cost is 5--7x that of new acquisition)

$(1 - R)$ exercises multiplicative veto power.
As $R arrow 1$, the score converges to 0 regardless of other factors.
In log space, $ln(1 - R) arrow -infinity$ as $R arrow 1$.


== Fatigue Decay

$ f(n) = e^(-lambda n) $ <fatigue>

Exponential decay implements *constant fractional decay*:
$f(n+1) \/ f(n) = e^(-lambda)$ (constant ratio).

Half-life: $n_(1\/2) = ln 2 \/ lambda$. App Push ($lambda = 0.4$): half-life $approx 1.73$ exposures.
Email ($lambda = 0.15$): half-life $approx 4.62$ exposures.


== Confidence Formula

$ "confidence" = |p - 0.5| times 2 $ <confidence>

Normalizes the distance from the decision boundary (0.5) to $[0, 1]$.
Predictions with low confidence are deprioritized in the recommendation quality filter.

#pagebreak()


// ============================================================
= Recommendation Reason Generation
// ============================================================

== 2-Layer Architecture (v3.0.0)

Design philosophy: "Equal explanation for all customers; LLM-enhanced explanation for context-rich customers."

#table(
  columns: (auto, auto, 1fr, auto, auto),
  inset: 8pt,
  stroke: 0.4pt + luma(200),
  [*Layer*], [*Target*], [*Method*], [*LLM Calls*], [*Throughput*],
  [L1], [All 12M customers], [Template via 3-agent pipeline (FactExtractor → InterpretationRegistry → TemplateEngine → SelfChecker)], [0], [$tilde$20 min],
  [L2a], [$tilde$500K/week], [LLM rewrite (Bedrock Claude Haiku, 3-layer safety gate)], [1], [$tilde$1.0 sec/record],
  [L2b], [$tilde$67K sampled], [Quality validation (Bedrock Claude Haiku; factuality, relevance, naturalness)], [1], [--],
)

=== 3-Agent Reason Pipeline (L1)

L1 reason generation runs a deterministic 4-stage agent pipeline --- zero LLM calls:

+ *FactExtractor:* extracts customer-level narrative facts from feature values via YAML-configured Python expressions (e.g., "deposit-focused portfolio", "risk-averse tendency"). Sandboxed, fully deterministic.
+ *InterpretationRegistry:* maps IG attribution top-5 features to financial language via pre-registered 3-tuple enrichments (feature → display_name, direction, explanation). Falls back to `ReverseMapper` (Level RM) if no entry found.
+ *TemplateEngine:* selects template variant via $"hash"("customer"_"id" : "category") mod 5$ and injects FactExtractor output + IG interpretations.
+ *SelfChecker:* rule-based compliance gate --- verifies no prohibited patterns (comparative claims, guaranteed return statements, etc.) before the reason is released.

*Regulatory basis:* Article 19 of the Financial Consumer Protection Act requires equal
duty of explanation for all customers.
L1 fulfills this requirement by providing zero-cost templates to all 12M customers.

*Cost efficiency:* Full LLM processing would consume $tilde$1,000 GPU-hours.
The 2-Layer design achieves the same coverage in $tilde$162 GPU-hours.


== L1 Template Generation

=== Structure

6 categories $times$ 5 variants = 30 templates.

*Deterministic variant selection:*
$ "variant"_"index" = "hash"("customer"_"id" : "category") mod 5 $

The same customer always receives the same variant (consistency + audit reproducibility).

=== Segment Awareness

- *WARMSTART*: Reasons based on IG Top-3 reverse mapping
- *COLDSTART*: Reasons based on popularity + benefits
- *ANONYMOUS*: Reasons based on general popularity

After a rule-based compliance check, an AI-generated disclosure notice is automatically appended.


== L2a LLM Rewrite

Priority queue: rich first, moderate second, sparse excluded.
Rewriting is applied after passing through the 3-Layer Safety Gate.
*AWS deployment:* Bedrock Claude Haiku (\$0.25/1M input tokens; VPC PrivateLink; no data transmitted to Anthropic for training).
*On-premises:* vLLM Qwen3-8B-AWQ on RTX 4070 (12GB VRAM).

=== 4-Layer Prompt Structure

+ *System prompt*: Role definition (financial recommendation reasoning expert) + prohibition rules under the Financial Consumer Protection Act
+ *Few-shot examples*: Tone and format guide per segment
+ *Context injection*: Customer features, IG attributions, consultation history $arrow$ natural language conversion
+ *Output format*: JSON schema (`{"reasons": [...], "summary": "..."}`)

=== Decoding Strategy

- Reason generation: $tau = 0.3$ (fact preservation + slight diversity)
- Critique: $tau = 0.1$ (near-deterministic, consistent quality assessment)
- L2a Rewrite: $tau = 0.3$ (preserve original facts, polish phrasing)


== Self-Critique Verdict

$ "verdict" = cases(
  "pass" & "if" f >= 0.8 "and" c >= 1.0,
  "revise" & "if" f >= 0.5 "and" c >= 1.0,
  "reject" & "otherwise"
) $ <self-critique>

- $f$: factuality score (continuous)
- $c$: compliance score (binary: 1.0 = no violation)

*Compliance takes absolute priority:* any regulatory violation ($c < 1.0$) results in
immediate rejection regardless of factuality.
*At most one revision:* if the verdict is still "revise" after revision, fall back to a
safe template (prevents infinite loops; maximum 3 LLM calls).


== L2b 3-Axis Quality Validation

$ "verdict" = cases(
  "pass" & "if" f >= 0.7 "and" r >= 0.7 "and" n >= 0.7,
  "needs"_"improvement" & "if any score" in [0.5, 0.7),
  "fail" & "if any score" < 0.5
) $ <l2b-validation>

- $f$: factuality, $r$: relevance, $n$: naturalness
- Threshold 0.7 (lower than Self-Critique's 0.8): L2b is post-hoc monitoring, not a real-time gatekeeper

#pagebreak()


// ============================================================
= Grounding + Feature Reverse Mapping
// ============================================================

== Core Problem

PLE-adaTT consumes a 734D feature vector and outputs probability scores, but cannot
explain _why_ a given recommendation was made.
Articles 31 and 34 of the AI Basic Act, and Article 19 of the Financial Consumer
Protection Act, require meaningful explanations.

== Grounding Function

$ f: bb(R)^(644) times cal(I) arrow.r cal(L) $ <grounding-fn>

Here $bb(R)^(644)$ is the normalized feature vector space, $cal(I)$ is the IG attribution
information, and $cal(L)$ is the natural language explanation space.


== 734D Feature Vector Structure

#table(
  columns: (auto, auto, 1fr),
  inset: 8pt,
  stroke: 0.4pt + luma(200),
  [*Range*], [*Dimensions*], [*Content*],
  [profile], [0--238], [Demographics (100D) + RFM (50D) + Financial Summary (88D)],
  [multi\_source], [238--329], [Transaction stats (40D) + Behavioral patterns (51D)],
  [extended\_source], [329--413], [Insurance, consultation, STT, campaign, overseas, open banking],
  [domain], [413--572], [TDA (70D) + GMM (22D) + Mamba (50D) + Economics (17D)],
  [model\_derived], [572--599], [HMM summary, Bandit/MAB, LNN],
  [multidisciplinary], [599--623], [Conversion dynamics, adoption dynamics, cross patterns, routine analysis],
  [merchant\_hierarchy], [623--644], [MCC levels, brand embeddings, statistics, radius],
)

Total: 644D normalized + 90D raw power-law = 734D model input.


== Integrated Gradients Attribution

$ "IG"_i (bold(x)) = (x_i - x'_i) times integral_0^1 frac(partial F(bold(x)' + alpha (bold(x) - bold(x)')), partial x_i) d alpha $

*Why IG is more suitable than SHAP:*
- SHAP: requires evaluating $2^(734)$ subsets (infeasible)
- IG: computed in linear time via a 50-step trapezoidal approximation
- *Completeness axiom:* $sum_i "IG"_i (bold(x)) = F(bold(x)) - F(bold(x)')$ (guaranteed by the gradient theorem in vector calculus)
- *Baseline:* zero vector (corresponds to "average customer" in normalized feature space)

== Reverse Mapping Architecture

$ "ReverseMap": (bold(x) in bb(R)^d, bold(a) in bb(R)^d) arrow.r {(r_k, s_k, t_k)}_(k=1)^K $

- $bold(x)$: feature vector, $bold(a)$: IG attribution vector
- $r_k$: feature range name, $s_k$: summary score, $t_k$: financial language text

*Sub-range slicing:* $t_k = cal(M)_k (g(bold(x)[s_k : e_k]))$
where $g$ is an aggregation function (mean, argmax, threshold comparison),
and $cal(M)_k$ is a domain-expert-designed mapping dictionary (numerical $arrow$ text).

== Module Composition

- *FeatureReverseMapper:* 644D vector $arrow$ financial language text (hierarchical range slicing)
- *MultidisciplinaryInterpreter:* 24D multidisciplinary features $arrow$ business interpretation (4 sub-domains)
- *LanceContextVectorStore:* LanceDB-based customer context storage/retrieval (768D embedding, L2 distance)
- *ContextAssemblyAgent:* IG-based tool selection + multi-source context assembly
- *ConsultationContextExtractor:* STT consultation history extraction + summarization

== Trust Loop

Model prediction $arrow$ IG attribution $arrow$ reverse mapping $arrow$ context assembly $arrow$ LLM reason generation
$arrow$ agent delivery $arrow$ customer persuasion $arrow$ conversion/feedback $arrow$ model improvement.

Without reverse mapping and context assembly, an interpretability gap arises between
model predictions and agent delivery, breaking this Trust Loop.

== Triple Grounding

+ *Feature Grounding:* inject IG Top-5 attributions into the prompt $arrow$ LLM generates reasons grounded in the model's actual decision basis
+ *Customer Grounding:* inject segment, transaction patterns, and consultation history $arrow$ hallucination suppression
+ *Regulatory Grounding:* system prompt prohibition rules + Rule-based Self-Critique $arrow$ compliance enforcement

#pagebreak()


// ============================================================
= Safety Gate
// ============================================================

== Multi-Layer Defense Architecture

A 6-layer safety mechanism for compliance with the Financial Consumer Protection Act
and the AI Basic Act.

#table(
  columns: (auto, 1fr, 1fr),
  inset: 8pt,
  stroke: 0.4pt + luma(200),
  [*Layer*], [*Mechanism*], [*Target Regulation*],
  [1], [System prompt: immutable regulatory prohibition rules], [AI Basic Act Art. 31 & 34],
  [2], [Self-Critique: real-time gatekeeper ($f >= 0.8$, $c = 1.0$)], [FCPA Art. 19 & 21],
  [3], [3-Layer Safety Gate: prompt injection detection + factuality + regulatory compliance], [FCPA Art. 22],
  [4], [L2b Quality Validation: post-hoc monitoring 3-axis assessment], [Internal quality standards],
  [5], [AI Security Checker: injection detection + compliance verification], [AI Basic Act Art. 34],
  [6], [Audit Archiver: immutable Parquet records (DuckDB-based retrieval)], [FSS post-inspection],
)

== Applicable Regulations

- *AI Basic Act Art. 31* (AI usage disclosure): AI-generated disclosure notice automatically appended to all recommendation reasons
- *AI Basic Act Art. 34* (risk management): Safety Gate + audit trail
- *Financial Consumer Protection Act Art. 19* (suitability principle + duty of explanation): L1 guarantees full coverage
- *Financial Consumer Protection Act Art. 21* (advertising regulations): prohibited pattern detection
- *Financial Consumer Protection Act Art. 22* (prohibition of unfair practices): compliance score verification

== Audit Archiving

`RecommendationAuditArchiver` persistently stores all recommendation records as Parquet.
- IG attribution scores, L1 reasons, L2a rewrite results, L2b validation results, processing time
- Efficient retroactive retrieval via DuckDB
- Full decision path tracing for individual recommendation records during FSS post-inspection

#pagebreak()


// ============================================================
= Serving Architecture
// ============================================================

== End-to-End Pipeline

```
PLE-adaTT Teacher (training)
  |-> Knowledge Distillation (T=5, alpha=0.3) + Adaptive Threshold (2x random floor)
    |-> LGBM Student (per-task, 200D features) + Platt Calibration (probability-critical tasks)
      |-> ONNX Export (ZipMap removal)
        |-> Lambda FallbackRouter
            |-> Layer 1: LGBM ONNX (primary)
            |-> Layer 2: PLE SageMaker Endpoint (failover)
            |-> Layer 3: Rule Engine (13 tasks, Financial DNA routing)
          |-> FD-TVS Scoring (4-Stage)
            |-> Feature Grounding (IG -> Reverse Mapping -> Context Assembly)
              |-> Recommendation Reason
                  L1: FactExtractor -> InterpretationRegistry -> TemplateEngine -> SelfChecker
                  L2a: Bedrock Claude Haiku (rewrite)
                  L2b: Bedrock Claude Haiku (validation)
                |-> Audit Archive (Parquet, ComplianceAuditStore)
```

== LGBM $arrow$ ONNX Conversion

=== ZipMap Removal (Required)

LightGBM's ONNX conversion adds a ZipMap operator, producing dictionary output.
Since Triton supports only tensor output, the ZipMap node must be bypassed and
removed from the ONNX graph.

=== Conversion Specifications

- Opset 13: full support for LightGBM operators
- 2-stage validation: (1) `onnx.checker.check_model` specification conformance, (2) dummy inference numerical equivalence test

== Triton Inference Server Configuration

=== Model Deployment

15 ONNX models (per-task) + 1 preprocessor + 1 postprocessor + 15 ensemble schedulers = *32 model configurations*.

=== Dynamic Batching

- Preferred batch sizes: [256, 512, 1024]
- Max queue delay: 100$mu$s
- Preprocessor: CPU $times$ 4 instances (CPU-bound JSON parsing)
- ONNX models: GPU $times$ 2 instances

=== Batch + Real-Time Hybrid

Daily batch processing computes baseline scores for all customers; when a real-time
transaction occurs, FD-TVS scores are immediately recomputed incorporating real-time
features from the Redis cache.
Triton Dynamic Batching queues individual real-time requests into micro-batches,
maintaining GPU utilization.

== Training-Serving Skew Prevention

*Feature Serving Spec* bridges training and serving.
- `feature_selector` outputs `selected_features_{task}.json` during training
- `FeatureServingSpec` loads this at deployment time to guarantee identical feature ordering
- The 7 mandatory features (TDA, Economics, FinEng) are always included

== Calibration Considerations

FD-TVS Stage 1 requires all task probabilities to lie on a common scale $[0, 1]$.
Probability-critical tasks (churn, product_stability, cross_sell_count) apply Platt scaling post-distillation (see §1 Calibration). Other tasks use Temperature Scaling (Guo et al., ICML 2017) as a lightweight alternative.

== 3-Layer Fallback Architecture

The Lambda serving layer implements a `FallbackRouter` that selects among three layers based on availability and confidence:

#table(
  columns: (auto, 1fr, 1fr),
  inset: 8pt,
  stroke: 0.4pt + luma(200),
  [*Layer*], [*Mechanism*], [*Produces*],
  [Layer 1: Distilled LGBM], [ONNX model via Lambda; Platt-calibrated per task], [`scores`, `contributing_features` (IG top-5)],
  [Layer 2: Direct PLE], [SageMaker Endpoint; activated if LGBM SKIP or Lambda cold-start failure], [`scores`, `contributing_features` (IG top-5)],
  [Layer 3: Rule Engine], [Python rule set: 13 task-specific rules + Financial DNA feature routing], [`scores` (heuristic), `contributing_features` (rule-based)],
)

All three layers produce `contributing_features` to maintain explanation compliance under AI Basic Act Art. 34 and Financial Consumer Protection Act Art. 19, even in degraded-mode serving.

=== Rule Engine Design

The rule engine implements 13 task-specific rules organized around Financial DNA feature routing:

- *DNA routing:* permanent income customers ($"CV" < 0.2$) → long-term product rules; transitory customers ($"CV" >= 0.5$) → short-term, low-commitment rules
- *Task rules:* each of the 13 tasks has a heuristic score function based on 3--5 interpretable features (e.g., churn rule uses recency + frequency + support call count)
- *contributing_features:* the rule selects the top-3 features that triggered the rule and returns them as `contributing_features` for the reason generation pipeline

== LLM Distillation: Gemini Teacher $arrow$ Qwen Student (QLoRA)

=== Distinguishing the Two Types of Distillation

#table(
  columns: (auto, 1fr, 1fr),
  inset: 8pt,
  stroke: 0.4pt + luma(200),
  [*Aspect*], [*Predictive Model Distillation*], [*LLM Distillation*],
  [Purpose], [Prediction accuracy], [Text generation quality],
  [Teacher], [PLE-Cluster-adaTT], [Google Gemini],
  [Student], [LightGBM], [Qwen3-8B (on-prem)],
  [Transfer Target], [Soft labels (logits/probs)], [Text output (recommendation reasons)],
  [Loss Function], [KL Divergence + CE], [Cross-Entropy (SFT)],
  [Training Method], [Soft label learning], [QLoRA fine-tuning],
)

*AWS serving note:* In production AWS Lambda, L2a rewrite and L2b validation use *Bedrock Claude Haiku* (not Qwen3-8B). Qwen3-8B QLoRA is the on-premises alternative. Both share the same PromptSanitizer and 3-Layer Safety Gate.

=== QLoRA: Mathematical Foundation of LoRA

$ W' = W_0 + Delta W = W_0 + B A $ <lora>

- $W_0 in bb(R)^(d times k)$: original pre-trained weights (frozen)
- $B in bb(R)^(d times r)$: Down-projection (trainable)
- $A in bb(R)^(r times k)$: Up-projection (trainable)
- $r << min(d, k)$: Rank ($r = 16$ corresponds to 0.78% of the original)

Compression ratio: $r times (d + k) \/ (d times k)$.
Qwen3-8B ($d = k = 4096$, $r = 16$): $131,072 \/ 16,777,216 approx 0.78%$.

=== NF4 Quantization

Distribution-aware quantization that places quantization levels at quantiles of the
standard normal distribution.

$ q_i = Phi^(-1)(i \/ (2^k + 1)) $

Each level covers equal probability mass, satisfying the Lloyd-Max optimality condition.

=== QLoRA Memory Analysis

- Full FT (FP16): weights 16GB + optimizer 32GB + gradients 16GB = 64GB+ (infeasible on RTX 4070)
- QLoRA: base model 4GB (NF4) + LoRA adapter $tilde$40MB = *trainable in 6GB*

=== Self-Consistency Training Data Filtering

$ "consistency"(s_1, s_2, s_3) = min_(i != j) "BERTScore"(s_i, s_j) $ <consistency>

Three outputs are generated from the Gemini Teacher for the same input, and the minimum
pairwise BERTScore is used as a conservative consistency measure.
Only consistent outputs are included in training data to filter out Teacher hallucinations.

#pagebreak()


// ============================================================
= Cost Efficiency and Operational Summary
// ============================================================

== Cost Efficiency Summary

#table(
  columns: (1fr, 1fr, 1fr),
  inset: 8pt,
  stroke: 0.4pt + luma(200),
  [*Component*], [*Design Choice*], [*Effect*],
  [2-Layer Reason Generation], [L1 Template + L2 LLM], [162 vs 1,000 GPU-hours],
  [LGBM Student], [CPU inference], [1/10 cost vs GPU Teacher],
  [QLoRA], [NF4 + LoRA], [6GB vs 64GB+ training],
  [Triton Dynamic Batching], [Micro-batch queuing], [Maximizes GPU utilization],
)

== Cross-Cutting Concerns

=== Financial Domain Specialization

- *Distillation:* TDA, Economics, and FinEng mandatory features are preserved regardless of IG importance
- *Scoring:* DNA modifier based on Friedman's Permanent Income Hypothesis; asymmetric risk weighting based on irreversibility
- *Grounding:* financial language translation via domain-expert-designed mapping dictionaries
- *Reason generation:* regulation-first design (compliance takes precedence over quality)
- *LLM distillation:* financial terminology, compliance-aware tone, and product-specific knowledge transfer

=== End-to-End Pipeline Coherence

Each stage of distillation $arrow$ serving $arrow$ scoring $arrow$ grounding $arrow$ reason generation $arrow$ audit
maintains end-to-end coherence through a shared contract: the Feature Serving Spec,
IG attributions, and reverse mapping dictionaries. If this contract is broken at any
stage, errors propagate to all downstream stages, making per-stage validation essential.

// ============================================================
= Ops/Audit Agent Integration

Recommendation reason quality is monitored under AuditAgent's AV3 viewpoint via a 3-Tier system:
- *Tier 1* (exhaustive): SelfChecker pass/revise/reject rate trends
- *Tier 2* (sampled): stratified sampling across 27 strata → GroundingValidator (reason↔IG alignment)
- *Tier 3* (expert): 50--100 monthly manual reviews → feedback loop

InterpretationRegistry → 3-tuple enrichment → TemplateEngine integration is complete, embedding IG interpretations into L1 reasons. ReverseMapper is integrated as Level RM fallback in InterpretationRegistry, expanding feature interpretation coverage.

== Fact Extraction Layer (New, 2026-04)

`FactExtractor` extracts *customer narrative facts* from feature values.
While `InterpretationRegistry` provides feature-level interpretation,
`FactExtractor` adds customer-level profiles ("deposit-focused portfolio",
"risk-averse tendency").

=== Rule-Based, Zero LLM

Python expressions defined in YAML config (`configs/financial/fact_extraction.yaml`)
are evaluated in a sandboxed environment (`__builtins__` disabled). Zero LLM calls,
fully deterministic. Facts are extracted at Phase 0 batch time and stored in
`ContextVectorStore`; at serving time, they are merely retrieved and injected
into the L2a prompt.

=== L2a Prompt Integration

Both `AsyncReasonOrchestrator.generate_l1()` and `get_best_reason()` retrieve
`context_store.get_context(customer_id).get("customer_facts", [])` and include
them as `context["customer_facts"]` in the SQS message body. `_build_llm_prompt()`
injects them as a "\#\# Customer Facts" section in the final prompt.

Detailed design: Design Document 11 (`docs/design/11_ops_audit_agent.typ`)
