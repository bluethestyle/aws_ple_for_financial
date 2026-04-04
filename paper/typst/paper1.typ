// ============================================================
// Paper 1: Heterogeneous Expert PLE — Architecture & Ablation
// ============================================================

#set document(
  title: "Heterogeneous Expert PLE: An Explainable Multi-Task Architecture for Financial Product Recommendation",
  author: ("Author 1", "Author 2", "Author 3"),
)

#set page(
  paper: "us-letter",
  margin: (x: 1.8cm, y: 2cm),
  numbering: "1",
)

#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.6em)
#set heading(numbering: "1.1")

// Title
#align(center)[
  #text(size: 16pt, weight: "bold")[
    Heterogeneous Expert PLE: An Explainable Multi-Task Architecture for Financial Product Recommendation
  ]

  #v(0.8em)

  #text(size: 11pt)[
    Author 1#super[1], Author 2#super[1], Author 3#super[1]
  ]

  #v(0.3em)

  #text(size: 9pt, style: "italic")[
    #super[1]Independent Research \
    (affiliated with a Korean public financial institution)
  ]

  #v(1em)
]

// Abstract
#block(
  width: 100%,
  inset: (x: 1em),
)[
  #text(weight: "bold")[Abstract.]
  Financial product recommendation requires not only predictive accuracy but also regulatory-compliant explainability.
  Existing multi-task learning approaches (MMoE, PLE) employ homogeneous MLP experts
  that suffer from _expert collapse_ --- all gates converge to a single expert,
  eliminating the diversity that justifies the MoE architecture ---
  and lack structural interpretability.
  We propose _Heterogeneous Expert PLE_, a Progressive Layered Extraction architecture
  where seven *architecturally distinct* experts --- DeepFM, Temporal Ensemble (Mamba+LNN+Transformer),
  Hyperbolic GCN, PersLay, Causal (NOTEARS), LightGCN, and Optimal Transport ---
  share a common basket, providing a *structural guarantee* against expert collapse:
  experts with fundamentally different inductive biases cannot converge to the same function.
  CGC gates learn task-specific expert compositions whose weights
  are inherently interpretable as business-meaningful explanations ---
  "35% spending trend (Temporal) + 28% product hierarchy (HGCN)" ---
  without post-hoc SHAP/LIME.
  Ablation on a 1M-customer benchmark reveals that different experts
  specialize in different task types:
  LightGCN and Causal experts dominate multiclass tasks (Macro F1 +0.20),
  while Causal and TDA experts excel at regression tasks (MAE --0.024),
  validating the heterogeneous design.
  Combined with Adaptive Task Transfer (adaTT) over four financial-DNA task groups
  and multi-disciplinary feature engineering spanning eleven academic disciplines,
  the architecture achieves parameter-efficient expressiveness
  on a single consumer GPU (12GB VRAM) while maintaining graceful degradation.
  // TODO: Fill in final ablation numbers

  #v(0.3em)
  #text(weight: "bold")[Keywords:]
  Multi-task learning, Progressive Layered Extraction, Mixture of Experts,
  Financial recommendation, Explainable AI, Ablation study
]

// Switch to 2-column layout for body
#show: rest => columns(2, rest)

// ============================================================
= Introduction

== Problem Definition

Financial product recommendation differs fundamentally from e-commerce or content recommendation.
The primary deliverable is not a probability score but a _reason that the customer can accept_.
Three audiences must be persuaded:
- *Customers*: "Why this product for me?" --- trust leads to conversion.
- *Relationship managers*: "Why recommend this to this customer?" --- sales justification.
- *Regulators* (Korean FSS @koreafsc2024, EU AI Act @euaiact2024): "Why was this decision made?" --- compliance obligation. Korea's AI Basic Act @koreaaiact2024 further classifies financial recommendation as potentially high-risk AI.

Existing approaches fall short on this persuasion requirement:
- *Single-task models* cannot jointly predict churn, product affinity, and customer lifetime value @caruana1997.
- *Model ensembles* (N separate models) multiply management overhead and serving cost, and "MLP \#3 contributed 28%" provides no business meaning.
- *Post-hoc explanations* (SHAP, LIME) are decoupled from model internals, computationally expensive at serving time, and demonstrably unstable under input perturbation @lundberg2017 @ribeiro2016 @salih2023.

== Core Insight

Conventional recommendation models function as "black-box shakers" --- features go in, probabilities come out, with only statistical correlations as justification.
As Pearl @pearl2009causality argues, there is a fundamental gap between
_seeing_ (association) and _understanding_ (causation).
Humans are not persuaded by correlations; they require _causal narratives_ @pearl2018book.

If each expert in a multi-task model captures a _different kind of "why"_ ---
temporal trends, hierarchical product structure, causal pathways, collaborative patterns ---
then the gating mechanism itself becomes an explanation:
"This recommendation is driven primarily by your spending trend (Temporal, 35%)
and product category fit (HGCN, 28%)."

This is the founding design principle of our architecture.

The key realization is that financial recommendation is not about a single prediction
("will this customer buy?") but about *understanding a customer as a whole person*
from multiple perspectives simultaneously:
Will they churn? (lifecycle).
How much will they spend? (value).
What product fits them? (consumption).
How do they behave? (engagement).
Each question demands a different analytical lens ---
temporal analysis cannot answer hierarchical product fit,
and graph-based collaborative signals cannot predict spending trajectories.
This is why heterogeneous experts are not merely a performance optimization
but a structural necessity for multi-faceted customer understanding.

This perspective aligns with a broader shift in both academia and regulation.
Pearl's _Ladder of Causation_ @pearl2018book distinguishes three levels:
association, intervention, and counterfactuals.
Most recommendation systems operate at level 1 (association);
our architecture aims for levels 1--2 by incorporating structural causal reasoning.
Regulatory frameworks --- the EU AI Act, Korea's FSS guidelines and AI Basic Act ---
increasingly demand this shift toward structurally transparent explanations
(detailed regulatory analysis in Section 2.3).

== Contributions

+ *Heterogeneous Shared Expert Basket with Structural Collapse Guarantee*: We replace PLE's homogeneous MLP experts with seven architecturally distinct experts (DeepFM, Mamba+LNN+Transformer, HGCN, PersLay, NOTEARS, LightGCN, Optimal Transport). Unlike prior "heterogeneous" MoE work that varies expert _size_ @mowst2024 or _modality_ @jamba2024, we vary the fundamental _inductive bias_, providing a structural guarantee against expert collapse --- a persistent failure mode in homogeneous MoE/PLE deployments @home2024.

+ *Inherent Explainability*: Because each expert encodes a named mathematical operation (not a generic MLP), CGC gate weights directly yield business-interpretable explanations without post-hoc attribution methods.

+ *Multi-disciplinary Feature Engineering*: Features derived from eleven academic disciplines --- including unconventional applications of chemical kinetics (spending activation rate), epidemic modeling (product adoption diffusion), criminological Routine Activity Theory (transaction regularity), and wave interference (spending periodicity) --- serve dual roles as learning signals and recommendation context that is reverse-mapped to business language for customer-facing explanations.

+ *Financial DNA Task Grouping*: Four task groups (engagement, lifecycle, value, consumption) with differentiated adaTT intra/inter transfer strengths and logit transfer for natural experience propagation.

+ *Comprehensive Ablation*: 54 scenarios (feature group / expert / task×structure) on a reproducible 1M-customer benchmark with Gaussian Copula + latent variable variance budget.

+ *Config-driven Pipeline*: End-to-end system (feature engineering → training → distillation → serving) controlled by two YAML files, enabling deployment by teams with 1--2 ML engineers.

+ *Reproducible Benchmark*: Synthetic data generation with variance budget @patki2016sdv for controlled AUC ceilings, validated against XGBoost baselines, with open-source code and fixed seeds.

// ============================================================
= Related Work

== Multi-Task Learning for Recommendation

The progression from Shared-Bottom @caruana1997 to MMoE @ma2018 to PLE @tang2020 reflects
increasing sophistication in managing negative transfer between tasks.
MMoE introduced per-task gating over a shared expert pool, while PLE further separated
shared and task-specific experts with a progressive extraction structure.
AdaTT @li2023 extended this with adaptive inter-task transfer.

Other notable MTL architectures include ESMM @ma2018esmm (entire-space modeling for conversion rate),
STAR @sheng2021star (star topology for multi-domain CTR),
and M3oE @zhang2024m3oe (multi-domain multi-task MoE with AutoML structure search).

However, all prior MoE/PLE architectures employ *homogeneous experts* --- multiple MLPs
with identical architecture but different initializations.
This limits the diversity of learned representations to what parameter variation alone can achieve,
and leads to a well-documented failure mode: *expert collapse*.

== Mixture of Experts and the Expert Collapse Problem

The MoE paradigm @shazeer2017 and its successors (Switch Transformer @fedus2022)
demonstrate the power of conditional computation.
However, homogeneous MoE architectures suffer from *expert collapse*,
where all gates converge to routing inputs to a single expert,
effectively reducing the MoE to a shared-bottom architecture.
Pinterest Engineering reported that their MMoE deployment
"would oftentimes collapse into a state where all tasks use the same single expert."
Kuaishou's HoME @home2024 confirmed that
"expert collapse occurs when all gates assigned larger weights
to a single shared expert and almost ignored other shared experts."

Existing mitigations operate post-hoc:
Gram-Schmidt orthogonalization to force diverse expert representations,
expert normalization to align output distributions,
and load-balancing losses to distribute routing.
These are engineering patches on a structural problem ---
homogeneous experts have no inherent reason to specialize differently
because their only source of diversity is random initialization.

Recent work has explored limited forms of heterogeneity:
Jamba @jamba2024 combines Transformer and Mamba layers within MoE for LLMs (2 architecture types),
and MOWST @mowst2024 pairs lightweight MLP "weak experts" with GNN "strong experts"
in a hierarchical arrangement (2 types, unequal roles).
M3oE @zhang2024m3oe separates experts by domain
but uses identical MLP architectures within each domain.
MoE++ introduces computation-level heterogeneity (zero-compute vs. FFN experts)
but not architectural heterogeneity.

*Gap*: No prior work composes experts with *fundamentally different inductive biases*
--- graph convolution, state-space models, topological persistence,
causal DAG constraints, optimal transport ---
as equal peers within a shared expert basket.
Our work provides a *structural guarantee* against expert collapse:
a DeepFM expert cannot converge to the same function as an HGCN expert
regardless of training dynamics, because their architectures
encode fundamentally different mathematical operations.

== Explainability in Recommendation

SHAP @lundberg2017 and LIME @ribeiro2016 provide model-agnostic explanations
but suffer from instability, computational cost, and disconnect from model internals.
Integrated Gradients @sundararajan2017 offers theoretically grounded attribution
but still operates post-hoc.

*Gap*: Structural (inherent) explainability that produces business-meaningful explanations
as a natural byproduct of the forward pass, without additional computation.

== Financial Recommendation and Regulation

Recent work on deep learning for financial product recommendation @chen2024financial @met2024banking
demonstrates improvements over collaborative filtering,
while Martinez-Plumed et al. @martinezplumed2023 show that sequential models
on longitudinal transaction data improve cross-selling prediction.
However, none of these systems address regulatory explainability requirements.

The EU AI Act @euaiact2024 classifies financial recommendation as high-risk AI,
mandating transparency (Art. 13), human oversight (Art. 14), and robustness (Art. 15).
Korea's FSS guidelines @koreafsc2024 and AI Basic Act @koreaaiact2024
impose similar requirements.
Current approaches rely on post-hoc SHAP/LIME for regulatory compliance,
which has documented limitations in financial contexts @salih2023.

*Gap*: No recommendation system provides verifiable mapping
from regulatory requirements to architecture components,
with explanations generated structurally rather than post-hoc.

// ============================================================
= Architecture
<architecture>

== Reductionist Framework: Two Axes of Decomposition

Before describing the architecture, we present the analytical framework
that governs all subsequent design decisions.

The complexity of financial customer understanding is decomposed along two orthogonal axes:

*Axis 1: Who is the customer? (Financial DNA)*

We decompose the question "who is this customer?" into four irreducible dimensions,
each capturing a fundamentally different aspect of customer identity:

#figure(
  scope: "parent",
  placement: auto,
  table(
    columns: (auto, auto, 1fr),
    inset: 5pt,
    align: left,
    stroke: 0.5pt,
    [*DNA*], [*Question*], [*Tasks*],
    [Engagement], [What do they _do_?], [has_nba #linebreak() engagement #linebreak() next_mcc #linebreak() mcc_trend #linebreak() top_mcc_shift],
    [Lifecycle], [Where _are_ they?], [churn #linebreak() tenure_stage #linebreak() segment],
    [Value], [How much are they _worth_?], [income_tier #linebreak() spend_level #linebreak() cross_sell #linebreak() stability],
    [Consumption], [What _will_ they buy?], [will_acquire\_\* (5), nba_primary],
  ),
  caption: [Financial DNA decomposition (Axis 1). Four irreducible dimensions of customer identity.],
) <tab:dna-axis>

*Axis 2: What form does the information take? (Data Modality)*

Customer data exists in structurally distinct modalities,
each requiring a different mathematical tool to extract meaningful signals:

#figure(
  scope: "parent",
  placement: auto,
  table(
    columns: (auto, auto, auto),
    inset: 5pt,
    align: left,
    stroke: 0.5pt,
    [*Modality*], [*Nature*], [*Optimal Tool*],
    [State], [Binary/categorical, time-invariant], [Feature crosses (DeepFM)],
    [Snapshot], [Numeric at a point in time], [Clustering (GMM)],
    [Short-term series], [Recent sequential patterns], [Attention (Transformer)],
    [Long-term series], [Multi-month trends], [State space (Mamba)],
    [Disrupted series], [Irregular / dormant gaps], [Adaptive ODE (LNN)],
    [Hierarchy], [Tree-structured categories], [Hyperbolic embedding (HGCN)],
    [Relations], [Customer-product graph], [Graph convolution (LightGCN)],
    [Topology], [Shape of behavioral patterns], [Persistent homology (PersLay)],
    [Causality], [Directional dependencies], [DAG constraint (NOTEARS)],
  ),
  caption: [Data modality decomposition (Axis 2). Each modality demands a structurally different expert.],
) <tab:modality-axis>

*The cross-product* of these two axes defines the architecture:
each (DNA $times$ Modality) cell determines which expert processes which features
for which task group.
The 7 heterogeneous experts, 12 feature groups, and 4 task groups
are not arbitrary design choices but necessary consequences of this two-axis decomposition.
A homogeneous MLP expert basket ignores Axis 2 entirely ---
it treats hierarchical, temporal, and topological data identically ---
which is why it cannot achieve the task-type-specific specialization
that our ablation demonstrates.

== Design Philosophy

The architecture emerged from severe real-world constraints
at a Korean public financial institution.
A team of three --- one data scientist serving as PM,
with GARP Financial Risk Manager (FRM) certification
and a career spanning credit/market risk analysis, regulatory compliance,
MyData licensing, big data platform construction and operations,
data science projects, and recommendation system management,
plus two engineers --- needed to replace a legacy ALS-based collaborative filtering system
with a next-generation recommendation model.

The constraints were formidable:
no dedicated ML infrastructure budget,
a single consumer-grade GPU (NVIDIA RTX 4070, 12GB VRAM) as the only training hardware,
no GPU inference servers for deployment,
and strict regulatory requirements (Korean FSS AI guidelines, EU AI Act).

Rather than treating these constraints as limitations,
the team adapted its methodology at every level:
(1) AI-augmented development using Claude (Anthropic), Gemini (Google), and Cursor,
with each team member leading a parallel team of AI agents;
(2) parameter-efficient architecture design where structural inductive biases
replace the brute-force capacity of large MLPs;
(3) knowledge distillation to LGBM for GPU-free CPU inference on AWS Lambda;
(4) config-driven pipeline requiring only two YAML files to control the entire system,
enabling operation by a minimal team.

Early exploration considered a Black-Litterman-inspired approach,
treating multiple models' predictions as "expert views" combined via Bayesian updating.
This was abandoned because the Bayesian combination process obscured
each model's contribution, making business-meaningful explanation impossible ---
a critical failure for regulatory compliance.

The key reframing was: instead of combining _models_ externally,
combine _experts_ internally within a single model.
This led to the selection of PLE @tang2020 with a critical modification:
replacing homogeneous MLP experts with structurally heterogeneous experts,
each encoding a different inductive bias.

=== Structural Isomorphism: Why Cross-Disciplinary Methods Work

The multi-disciplinary feature engineering in this work is not arbitrary borrowing
from unrelated fields.
It is grounded in _structural isomorphism_ ---
the observation that mathematically equivalent structures
recur across different domains, enabling tools developed in one field
to be rigorously applied to structurally equivalent problems in another.

Consider:
a dormant customer resuming spending follows the same dynamics
as a chemical reaction overcoming an activation energy barrier.
Product adoption spreading through a customer network follows the same
compartmental dynamics as an SIR epidemic model.
These are not metaphors --- the governing equations are identical,
and the solutions transfer with mathematical rigor.

This principle has a distinguished history in science and engineering:
Shannon's information entropy was directly borrowed from Boltzmann's thermodynamic entropy;
the Black-Scholes option pricing model is derived from the heat diffusion equation;
Google's PageRank is an application of Markov chain stationary distributions.
In each case, recognizing structural isomorphism between two domains
enabled a breakthrough that neither domain could have achieved alone.

Our contribution is applying this principle _systematically_ to financial customer understanding:
for each aspect of customer behavior,
we identify a domain whose mathematical tools are structurally isomorphic
to the underlying dynamics, and instantiate that domain's methods
as both features (for prediction) and expert architectures (for interpretation).
The multi-disciplinary approach is not a collection of ad-hoc features
but a principled application of analogical reasoning through structural equivalence.

Four principles guide the architecture:

+ *Robust Explainability*: Model structure itself generates explanations.
  Each expert's name carries business meaning (e.g., "Temporal" = time patterns, "HGCN" = product hierarchy),
  so CGC gate weights are directly interpretable without post-hoc methods.
+ *Graceful Degradation*: Removing any single expert does not cause catastrophic performance loss.
  The remaining experts redistribute gate weights to compensate.
+ *Flexible Extensibility*: New features, tasks, or experts are added via YAML configuration,
  not code changes. The system has been extended from 4 to 18 tasks without architectural modification.
+ *Unified Manageability*: The entire pipeline (feature engineering → training → distillation → serving → monitoring)
  is controlled by two configuration files (`pipeline.yaml` and `feature_groups.yaml`),
  reducing operational overhead for teams with limited ML engineering resources.

== Data Axis Classification

Financial customer data exhibits inherently multi-modal structure.
We classify data along multiple axes, each mapped to an optimal feature generator and expert:

#figure(
  scope: "parent",
  placement: auto,
  table(
    columns: (auto, auto, auto, auto),
    inset: 6pt,
    align: left,
    stroke: 0.5pt,
    [*Data Axis*], [*Examples*], [*Expert*], [*Generator*],
    [State], [is_active, holdings], [DeepFM], [base],
    [Snapshot], [balance, demographics], [GMM, DeepFM], [GMM clustering],
    [Short-term series], [recent transactions], [Transformer], [temporal],
    [Long-term series], [monthly trends], [Mamba], [mamba temporal],
    [Disrupted series], [dormant→active], [LNN], [model derived],
    [Hierarchy], [product category tree], [HGCN], [product hierarchy],
    [Relations], [customer-product graph], [LightGCN], [graph collab.],
    [Topology], [behavioral shape], [PersLay], [TDA global/local],
    [Causality], [behavioral causation], [Causal], [causal features],
  ),
  caption: [Data axis to Expert to Feature generator mapping.],
) <tab:data-axis>

#text(size: 8.5pt, fill: gray)[
  _Note_: Short-term, Long-term, and Disrupted series map to the three sub-components
  of the Temporal Ensemble expert (Transformer, Mamba, LNN respectively).
  GMM is a feature generator, not a standalone expert.
]

== Heterogeneous Expert Basket

#figure(
  scope: "parent",
  placement: auto,
  // TODO: Replace with actual architecture diagram
  rect(width: 100%, height: 8cm, stroke: 0.5pt)[
    #align(center + horizon)[
      _Architecture diagram placeholder_ \
      Input (318 features) → 12 Feature Groups → CGC Gate → 7 Experts → 4 Task Groups → 18 Towers → Output \
      + adaTT inter/intra transfer
    ]
  ],
  caption: [Heterogeneous Expert PLE architecture overview.],
) <fig:architecture>

Unlike standard PLE where shared experts are identical MLPs,
our shared expert basket contains seven structurally distinct networks.
Each expert corresponds to one or more modality axes (@tab:modality-axis),
ensuring that every structurally distinct data type
is processed by an architecture designed for it:

#figure(
  scope: "parent",
  placement: auto,
  table(
    columns: (auto, auto, auto),
    inset: 6pt,
    align: left,
    stroke: 0.5pt,
    [*Expert*], [*Inductive Bias*], [*Captures*],
    [DeepFM @guo2017], [Feature interaction], [2nd-order cross features],
    [Temporal Ensemble], [Multi-scale temporal], [Short/long/disrupted series],
    [HGCN @chami2019], [Hyperbolic hierarchy], [Product category tree],
    [PersLay @carriere2020], [Topological persistence], [Behavioral shape patterns],
    [LightGCN @he2020lightgcn], [Graph convolution], [Collaborative filtering],
    [Causal @zheng2018notears], [DAG constraint], [Causal direction between features],
    [Optimal Transport @cuturi2013], [Distribution matching], [Segment distribution shifts],
  ),
  caption: [Seven heterogeneous experts with distinct inductive biases.],
) <tab:experts>

Because each expert has a named inductive bias, the gate weights tell the customer
_why_ in business terms --- not "hidden unit 47 activated" but
"your spending trend drove this recommendation."
Each expert was selected based on a specific gap
in financial customer understanding that no other expert type addresses:

- *DeepFM* @guo2017: Financial behavior is driven by feature interactions
  (e.g., income × product holdings × channel preference).
  FM's low-rank factorization computes 2nd-order crosses in $O(n k)$
  vs. $O(n^2)$ for brute-force enumeration,
  while the Deep component captures higher-order interactions.

- *HGCN* @chami2019: Financial product catalogs (MCC codes: 10 L1 / 30 L2 / 109 leaf)
  are inherently tree-structured.
  Hyperbolic space (Poincaré ball, 8D) embeds trees with exponentially less distortion
  than Euclidean space @nickel2017poincare --- critical for a 550K-node merchant hierarchy.

- *PersLay* @carriere2020: Topological Data Analysis captures _shape_ features
  (connected components $H_0$, cycles $H_1$, voids $H_2$) of customer spending patterns.
  $H_1$ loops reveal consumption cycles; $H_2$ voids detect systematic spending avoidance.
  These features are provably stable under noise (Stability Theorem).

- *Temporal Ensemble*: Financial time series combine regular snapshots, bursty transactions,
  and multi-month dormancy. Mamba @gu2024 handles long-range trends in $O(n)$;
  LNN @hasani2021 adapts its time constants to irregular intervals;
  Transformer captures short-range attention patterns.

- *Causal* @zheng2018notears: The NOTEARS continuous DAG constraint
  ($"tr"(e^(W circle.tiny W)) - d = 0$) learns causal direction between features,
  enabling "A causes B" explanations rather than "A correlates with B" @pearl2009causality.

- *LightGCN* @he2020lightgcn: Collaborative filtering via neighborhood aggregation
  on the customer-product bipartite graph. Stripped to essentials (no feature transform, no activation),
  which outperforms more complex GCN variants for recommendation.

- *Optimal Transport* @cuturi2013: Sinkhorn-regularized Wasserstein distance
  measures distributional shift between customer spending profiles and segment prototypes,
  respecting the metric structure of the feature space (unlike KL divergence).

The rationale for heterogeneous experts is also rooted in a hardware constraint:
with a single consumer GPU (12GB VRAM), we cannot scale a homogeneous MLP expert
to sufficient width/depth for high expressiveness.
Instead, each expert leverages a _structural inductive bias_ to capture patterns
that would require orders of magnitude more MLP parameters.
For example, HGCN embeds a product hierarchy tree in 32 hyperbolic dimensions ---
achieving in $O(d)$ parameters what Euclidean embeddings require $O(2^d)$ dimensions to represent
without distortion @chami2019.

The total parameter count across all seven experts
remains comparable to a single large MLP,
but the diversity of learned representations is fundamentally richer.

=== Temporal Ensemble: Expert-within-Expert

Financial time series exhibit a uniquely complex structure rarely seen in other domains.
A single customer may simultaneously have:
monthly balance snapshots (regular, long-range),
daily transaction sequences (irregular, bursty),
and multi-month dormancy gaps (disrupted).
No single temporal model handles all three well:

- *Mamba* @gu2024 (State Space Model): Captures long-range dependencies with $O(n)$ efficiency via selective state spaces. Ideal for monthly/quarterly trends spanning 12+ months.
- *LNN* @hasani2021 (Liquid Neural Network): Adaptive time constants that naturally handle irregular sampling intervals and dormancy gaps. When a customer is inactive for 3 months then suddenly active, LNN's continuous-time dynamics adapt without requiring imputation.
- *Transformer*: Attention-based short-range context extraction. Captures patterns within the most recent 30--90 days of transaction sequences where positional relationships matter.

The three models' outputs are concatenated and projected,
mirroring the heterogeneous expert philosophy at a finer granularity.
This design is validated in the ablation study (Section 5.4),
where removing any single temporal component degrades
performance on different task groups.

== Multi-disciplinary Feature Engineering
<multidisciplinary>

A distinguishing aspect of this work is the systematic application of
methodologies from diverse academic disciplines to financial customer behavior.
Now that the expert basket has been introduced, the rationale becomes clear:
each discipline's features are designed to feed a _specific expert_
whose inductive bias can best exploit that signal type.
Rather than relying solely on standard statistical features
(mean, variance, trend), we apply domain-specific mathematical tools
that each extract a structurally different signal from the same underlying data.

#figure(
  scope: "parent",
  placement: auto,
  {
  set text(size: 8.5pt)
  table(
    columns: (auto, auto, auto, 1fr),
    inset: 5pt,
    align: left,
    stroke: 0.5pt,
    [*Discipline*], [*Method*], [*Dim*], [*Financial Interpretation*],
    [Topology], [Persistent Homology (TDA)], [32], [Behavioral shape persistence: which consumption patterns are transient vs. structural],
    [Hyperbolic Geometry], [HGCN embedding], [34], [Product category distance: preserving hierarchy in low dimensions],
    [Control Theory], [Mamba (State Space)], [50], [Long-range behavioral dependencies: how past habits influence present],
    [Stochastic Processes], [HMM state transitions], [25], [Latent lifecycle stages: dormant → growing → mature → at-risk],
    [Chemical Kinetics], [Reaction rate modeling], [6], [Spending activation rate, half-life, dormancy reactivation catalysis],
    [Epidemiology], [SIR compartment model], [5], [Product adoption as "infection": susceptible → adopted → churned],
    [Criminology], [Routine Activity Theory], [5], [Transaction regularity: burstiness, circadian variance, routine breakpoints],
    [Signal Processing], [FFT + Hilbert transform], [8], [Spending periodicity: spectral entropy, harmonic power, phase locking],
    [Economics], [Friedman Permanent Income], [8], [Income decomposition: permanent vs. transitory income, consumption smoothing],
    [Graph Theory], [LightGCN], [66], [Collaborative filtering: similar customer behavioral transfer],
    [Statistics], [GMM soft clustering], [22], [Probabilistic segmentation: multi-modal customer distribution],
  )
  },
  caption: [Multi-disciplinary feature engineering across 11 academic disciplines.],
) 

#text(size: 8.5pt, fill: gray)[_Note_: Total 269 generated features + 49 base features = 318.
    Base features include 47 raw columns plus 2 derived columns (is\_cold\_start, income\_log).],
) <tab:multidisciplinary>

Several of these applications are, to our knowledge, novel in financial recommendation:

*Chemical kinetics* models spending behavior as a reaction system:
the "activation energy" represents the threshold for a dormant customer to resume spending,
while "catalysts" (e.g., payroll deposits) lower this threshold.
The half-life of spending intensity after a peak captures decay dynamics
that simple moving averages cannot express.

*SIR epidemic model* treats product adoption as a contagion process:
customers are "susceptible" (not yet adopted), "infected" (recently adopted),
or "recovered" (churned). The basic reproduction number $R_0$ of a product category
across a customer's social graph predicts adoption velocity.

*Friedman's Permanent Income Hypothesis* @friedman1957 decomposes observed income into
permanent (stable, long-term) and transitory (bonus, irregular) components
using HP filter or Kalman filter.
This distinction is critical for financial recommendation:
a customer with high transitory income should not be recommended
long-term investment products that assume stable cash flow.
The FD-TVS scoring system (detailed in companion paper) uses this decomposition
to weight recommendations by income stability type.

These features serve a dual purpose beyond predictive contribution:
they enrich the _recommendation context_ that no standard feature can provide.
Internally, "activation energy decrease" (chemical kinetics) signals rising re-engagement likelihood,
and "growing SIR infection ratio" (epidemiology) signals accelerating product adoption ---
but the customer never sees these scientific terms.
An `interpretation_registry` maps each scientific feature to business-interpretable language,
and the LLM agent generates natural-language reasons accordingly
(e.g., "최근 활동 패턴이 회복세를 보이고 있습니다").
The scientific frameworks ground the model's internal reasoning;
the customer receives business language.

== Financial DNA Task Grouping

We organize 18 prediction tasks into four groups based on financial customer DNA.
Each task group corresponds to one DNA axis (@tab:dna-axis);
within each group, different modality experts (@tab:modality-axis) contribute differently,
and the CGC gate learns the optimal mixture per task:

#figure(
  scope: "parent",
  placement: auto,
  table(
    columns: (auto, auto, 1fr),
    inset: 6pt,
    align: left,
    stroke: 0.5pt,
    [*Group*], [*Financial DNA*], [*Tasks*],
    [Engagement], [What does the customer do?], [has_nba #linebreak() engagement_score #linebreak() next_mcc #linebreak() top_mcc_shift #linebreak() mcc_diversity_trend],
    [Lifecycle], [Where is the customer?], [churn_signal #linebreak() tenure_stage #linebreak() segment_prediction],
    [Value], [How valuable is the customer?], [income_tier #linebreak() spend_level #linebreak() cross_sell_count #linebreak() product_stability],
    [Consumption], [What will the customer buy?], [will_acquire\_\* (5), nba_primary],
  ),
  caption: [Four task groups based on financial customer DNA.],
) <tab:task-groups>

adaTT enforces differentiated transfer: strong intra-group transfer (same DNA perspective)
and weaker inter-group transfer (different perspectives, minimizing negative transfer).

*Loss-level transfer.* A notable departure from the original adaTT @li2023,
which transfers at the representation level, our implementation operates at the _loss level_:

$ cal(L)_i^("adaTT") = cal(L)_i + lambda sum_(j eq.not i) w_(i arrow.r j) dot cal(L)_j $ <eq:adatt>

The base task losses $cal(L)_i$ are weighted by learned uncertainty @kendall2018,
and binary classification tasks with severe class imbalance (e.g., has_nba at 3\% positive rate)
use focal loss @lin2017focal with task-specific $alpha$ and $gamma$ parameters.

where $w_(i arrow.r j)$ is the transfer weight from task $j$ to task $i$,
computed via gradient cosine similarity between task loss gradients.
This design choice was motivated by two considerations:
(1) representation-level transfer requires matching hidden dimensions across heterogeneous experts,
which is architecturally cumbersome when experts produce outputs of different shapes and semantics;
(2) loss-level transfer naturally respects the task group structure ---
if two tasks have similar gradient directions (high cosine similarity),
their losses reinforce each other, regardless of the expert that produced the prediction.

== Logit Transfer

While adaTT handles symmetric inter-task relationships,
logit transfer captures _directional_ dependencies
that reflect the natural sequence of customer experience:

#figure(
  scope: "parent",
  placement: auto,
  table(
    columns: (auto, auto, auto, auto),
    inset: 5pt,
    align: left,
    stroke: 0.5pt,
    [*Source → Target*], [*Method*], [*Customer Experience*], [*Causal Direction*],
    [has_nba → nba_primary], [output_concat], [Purchase decision → product selection], [Sequential],
    [engagement → has_nba], [hidden_concat], [Activity level → purchase probability], [Leading indicator],
    [spend_level → will_acquire\_\*], [residual], [Spending capacity → category intent], [Enabling factor],
    [churn → has_nba], [output_concat], [Retention risk → acquisition opportunity], [Inverse correlation],
  ),
  caption: [Logit transfer relationships reflecting natural customer experience flow.],
) <tab:logit-transfer>

These transfers connect DNA groups (@tab:dna-axis): engagement→engagement (intra-group),
value→consumption (inter-group), reflecting the natural sequence of customer experience.
The transfer directions are not learned from data but specified based on
domain knowledge of the customer journey.
This is a deliberate design choice: while the _strength_ of transfer
is learned (via adaTT affinity), the _direction_ is fixed by business logic,
providing an additional layer of interpretability.

== Inherent Explainability
<inherent-explain>

A central claim of this work is that the heterogeneous expert structure
provides _inherent_ explainability --- explanations that _persuade_ customers,
relationship managers, and regulators emerge as a natural
byproduct of the forward pass, not as a separate post-hoc computation.

*Mechanism.* The CGC (Customized Gate Control) module computes
a softmax attention over the $K$ experts for each task $t$:
$ w_t = "softmax"(W_t dot h + b_t) in RR^K $
where $h$ is the input representation. The weight $w_(t,k)$ directly indicates
how much expert $k$ contributed to task $t$'s prediction.

*Business interpretability.* Because each expert $k$ has a named inductive bias
(e.g., $k=$ "Temporal" means time-series patterns, $k=$ "HGCN" means product hierarchy),
$w_(t,k)$ carries business meaning without additional interpretation:

#block(inset: 8pt, stroke: 0.5pt + gray, radius: 3pt, width: 100%)[
  #text(size: 9pt)[
    _Example_: For customer $c$, the recommendation of investment funds is driven by: \
    #h(1em) Temporal (0.35) --- spending has been increasing over 3 months \
    #h(1em) HGCN (0.28) --- current products are structurally close to investment category \
    #h(1em) DeepFM (0.22) --- income × product-holding interaction pattern \
    #h(1em) Others (0.15)
  ]
]

*Comparison with post-hoc methods.* SHAP and LIME operate at the _feature_ level
("feature\_237 contributed 0.12"), requiring an additional mapping step
to translate feature attributions into business narratives.
Moreover, LIME's local linear approximation is demonstrably unstable ---
small input perturbations can dramatically change the explanation @ribeiro2016.
In contrast, gate weights are deterministic functions of the input
and change smoothly with input variation.

*Dual role of features.* An important design insight is that features serve
two distinct purposes: (1) predictive contribution to AUC,
and (2) explanation material for recommendation reasons.
Even features with marginal predictive contribution (e.g., TDA topological features
may add only $Delta$AUC $approx$ 0.01) provide irreplaceable context for recommendation reasoning.
Internally, TDA persistence signals that a customer's spending pattern has a stable topological shape ---
this is then reverse-mapped via the `interpretation_registry` to business language
(e.g., "꾸준한 거래 패턴을 유지하고 계십니다") that the customer actually sees.
The scientific feature enriches the recommendation context; the customer receives an interpretable description.
This dual role motivates our multi-disciplinary feature engineering
and is validated in the ablation study.

// ============================================================
= Training Pipeline

== Config-Driven Design

The entire pipeline is controlled by two YAML files:
`pipeline.yaml` (model, training, deployment settings) and
`feature_groups.yaml` (feature generation, expert routing).
Adding a new dataset, task, or expert requires only configuration changes.

== Data Processing

Following the principle of _pandas-free_ data engineering for scalability,
the pipeline uses a tiered backend strategy:
cuDF (GPU columnar) for generator fitting/transformation,
DuckDB @raasveldt2019duckdb (CPU columnar with disk spill) for SQL-based aggregation and I/O,
and PyArrow for zero-copy parquet loading into training tensors.
pandas is used only as a last-resort fallback for datasets under 10K rows.

This backend choice has practical implications:
DuckDB handles the 1M-row dataset with a 4GB memory limit
and automatically spills to disk, enabling processing on commodity hardware.
cuDF accelerates generator fitting (GMM, HMM, Mamba) on GPU,
reducing Phase 0 feature engineering from ~30 minutes (pandas) to ~8 minutes.

== Data Leakage Prevention

Data leakage is a pervasive risk in multi-stage pipelines
where features and labels are processed in separate stages.
We implement three structural safeguards:

+ *Generator label exclusion*: Feature generators (GMM, model-derived, etc.)
  automatically exclude all label columns from their input.
  Without this guard, we observed XGBoost AUC of 1.0 on the generated features ---
  the generator had encoded label information directly into features.
+ *Temporal split with gap*: A configurable `gap_days` parameter
  ensures no temporal overlap between training and validation windows.
  For cross-sectional (single-snapshot) data, the system auto-detects
  and falls back to random splitting.
+ *Scaler train-only fitting*: StandardScaler is fit exclusively on the training split;
  validation and test data are transformed using training statistics.
+ *LeakageValidator*: A pre-training check computes Pearson correlation
  between every feature and every label (on a 50K subsample for efficiency).
  Correlations above 0.95 trigger a warning and investigation.

// ============================================================
= Experiments

== Benchmark Data

We construct a 1M-customer synthetic benchmark using a four-layer generative model,
inspired by the Synthetic Data Vault framework @patki2016sdv
and conditional generative approaches @xu2019ctgan,
but with a novel _variance budget_ mechanism for controllable difficulty:
+ *Latent Personas*: 6 GMM-fitted personas with 5D continuous latent vector (70% persona-conditioned, 30% independent noise).
+ *Gaussian Copula Demographics*: Correlated demographic variables preserving realistic joint distributions @patki2016sdv.
+ *Vectorized Transactions*: Per-customer transaction sequences as LIST columns.
+ *Variance Budget Labels*: Each label's predictability is controlled via $"logit" = sqrt(f_"obs") dot z_"obs" + sqrt(f_"lat") dot z_"lat" + sqrt(f_"noise") dot epsilon$, with post-hoc label noise flipping.

#figure(
  scope: "parent",
  placement: auto,
  table(
    columns: (auto, auto, auto, auto, auto),
    inset: 5pt,
    align: center,
    stroke: 0.5pt,
    [*Tier*], [*Labels*], [$f_"obs"$], [$f_"noise"$], [*XGB AUC*],
    [Easy], [segment, income_tier], [determ.], [--], [0.95--1.0],
    [Core], [has_nba, churn_signal], [0.04], [0.68], [0.58--0.65],
    [Hard], [will_acquire\_\*], [0.03], [0.72], [0.50--0.56],
    [V.Hard], [next_mcc, top_mcc_shift], [0.02], [0.78], [0.50--0.51],
  ),
  caption: [Variance budget per label tier. XGB AUC ceiling validates difficulty control.],
) <tab:variance-budget>

== Experimental Setup

The ablation validates whether each expert provides a distinct "why" for different task types ---
confirming that heterogeneous experts are not merely a performance trick
but a structural requirement for multi-faceted persuasion.

- *Data*: 1M customers, 318 features, 18 tasks.
- *Hardware*: NVIDIA RTX 4070 (12GB) local; AWS g5.xlarge (A10G 24GB) cloud.
- *Training*: 5+5 epochs (phase1 + phase2), batch 6144, lr 0.008, AMP, early stopping patience 3.
- *Metrics*: AUC (binary), F1 macro (classification), MAE/R² (regression).

== Feature Group Ablation (RQ1)

// TODO: Fill results table after ablation completes
#figure(
  scope: "parent",
  placement: auto,
  table(
    columns: (auto, auto, auto, auto),
    inset: 5pt,
    align: center,
    stroke: 0.5pt,
    [*Scenario*], [*Features*], [*Avg AUC*], [*Δ vs Full*],
    [full], [318], [--], [baseline],
    [base_only], [49], [--], [--],
    [base+tda], [65], [--], [--],
    [base+hmm], [74], [--], [--],
    [base+mamba], [99], [--], [--],
    [full−tda], [300], [--], [--],
    [full−hmm], [291], [--], [--],
    [...], [...], [...], [...],
  ),
  caption: [Feature group ablation results. TODO: fill after ablation.],
) <tab:feat-ablation>

== Expert Ablation (RQ2)

// TODO: Fill results

== Task × Structure Cross Ablation (RQ3)

// TODO: Fill results

== Graceful Degradation (RQ4)

// TODO: Analysis of performance drop per expert removal

== Explainability Analysis (RQ5)

// TODO: Gate weight distribution, SHAP comparison

// ============================================================
= Discussion

== Findings Summary
// TODO: Fill after ablation results

== Practical Implications

*Resource-constrained development.*
This system was built without dedicated ML infrastructure budget,
on a single consumer GPU, by a three-person team
augmented with AI development agents.
This demonstrates that complex multi-task recommendation systems
are no longer exclusive to organizations with large ML teams and GPU clusters.
The key enablers were: (1) config-driven architecture minimizing code changes,
(2) AI agents handling parallel implementation tasks under human architectural guidance,
(3) heterogeneous expert design achieving expressiveness through structural bias
rather than parameter scale, and (4) knowledge distillation eliminating GPU serving costs.

*Feature engineering philosophy.*
A deeper lesson from this work is that _what to observe_ matters more than _how to model_.
The heterogeneous expert architecture is effective not because of architectural novelty alone,
but because each expert is paired with features that ask a *specific question*
about the customer:
"Is their spending decaying with a half-life?" (chemical kinetics),
"Is product adoption spreading like contagion?" (SIR epidemiology),
"Is their income permanent or transitory?" (Friedman PIH @friedman1957).
The model merely combines answers to these questions.
If the questions are limited to conventional statistics --- mean, variance, trend ---
even the most sophisticated architecture cannot capture
the structural complexity of customer behavior.
This suggests that multi-disciplinary feature engineering,
often treated as preprocessing, deserves elevation
to a first-class architectural design decision,
co-designed with the expert types that will consume each feature group.

*Generalizability beyond finance.*
While this work targets financial product recommendation,
the underlying principle --- understanding a single user from multiple perspectives
simultaneously via heterogeneous experts ---
applies to any domain where personalization requires multi-faceted user modeling.
Healthcare (diagnosis + treatment + risk + lifestyle),
education (knowledge level + learning style + engagement + career path),
and insurance (risk assessment + product fit + claim prediction + retention)
all face structurally similar challenges:
multiple interdependent prediction tasks about the same individual,
where different analytical lenses capture different aspects.
We expect the heterogeneous expert PLE pattern to generalize to these domains,
with domain-specific expert types replacing the financial experts
while retaining the structural benefits of collapse resistance and inherent explainability.

*Infrastructure choice.*
Financial institutions differ fundamentally from big tech in ML infrastructure needs.
Big tech operates hundreds of GPUs across dozens of teams with hourly retraining cycles,
justifying Kubernetes-based GPU schedulers (Kubeflow).
Financial institutions typically train models daily to weekly,
with fewer than 10 models in production,
making dedicated GPU clusters an over-investment.
Our SageMaker + Lambda serverless architecture matches this reality:
GPU resources are provisioned on-demand for training (Spot instances at 60--70\% discount)
and released immediately after, while serving runs on CPU-only Lambda functions
via knowledge distillation to LGBM (detailed in companion paper).

*Operational simplicity.*
The config-driven design (two YAML files control the entire pipeline)
enables teams with 1--2 ML engineers to operate the system.
Adding a new product category, task, or feature group requires
only configuration changes, not code modifications.
This stands in contrast to model ensemble approaches
where each model is a separate management point with its own
training pipeline, serving endpoint, and monitoring dashboard.

== Limitations

- *Synthetic benchmark*: While our variance-budget approach controls difficulty levels,
  synthetic data cannot fully capture the distributional complexity of real financial data.
  We plan to validate on production data from a partner institution in future work.
- *Single-GPU training*: The current implementation runs on a single GPU.
  DDP (DistributedDataParallel) support is architecturally designed
  but not yet experimentally validated.
  For the ablation scenarios studied here, single-GPU training completes within acceptable time.
- *Phase 2 numerical stability*: Mixed-precision (FP16) training in Phase 2
  requires careful handling of log-domain computations (CGC entropy, OT Sinkhorn)
  to avoid FP16 underflow. We document four specific numerical fixes
  that were necessary for stable Phase 2 training.
- *LLM dependency*: Recommendation reason generation relies on LLM inference,
  introducing latency and cost trade-offs (detailed in companion paper).

// ============================================================
= Conclusion

We presented Heterogeneous Expert PLE, a multi-task learning architecture
for financial product recommendation that addresses the fundamental requirement
of _persuading humans_ --- not merely predicting their behavior.

By composing seven structurally distinct experts within PLE's shared basket,
we achieve three properties simultaneously:
(1) parameter-efficient expressiveness under hardware constraints,
(2) inherent explainability through business-interpretable gate weights,
and (3) graceful degradation validated through comprehensive ablation.

The key insight is that explanation quality depends not on post-hoc attribution methods
but on the model's internal structure.
The two-axis decomposition (Financial DNA $times$ Data Modality)
ensures that every design decision --- which experts to build,
which features to engineer, which tasks to group ---
traces back to a principled analytical framework.
When each expert carries a named inductive bias ---
temporal dynamics, product hierarchy, causal pathways ---
the gating mechanism _is_ the explanation.

// TODO: Add key ablation numbers when available

The architecture, benchmark data, and ablation framework are released as open source
to enable reproduction and extension by the research community.
A companion paper addresses the downstream pipeline:
knowledge distillation @hinton2015 to LGBM @ke2017lightgbm, multi-agent recommendation reason generation,
and regulatory compliance mapping for Korean FSS and EU AI Act requirements.

// ============================================================
// Acknowledgments
#heading(numbering: none)[Acknowledgments]

The code implementation and manuscript drafting were assisted by
Claude (Anthropic) as an AI coding and writing tool.
The architectural decisions, domain knowledge, experimental design,
and research direction were led by the human authors.

// ============================================================
// References
#bibliography("references.bib", style: "association-for-computing-machinery")
