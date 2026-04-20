// ============================================================
// Paper 3: Loss Dynamics and Gate Selection in Large-Scale MTL
// ============================================================

#set document(
  title: "Scaling Multi-Task Learning Beyond Homogeneous Tasks: Loss Dynamics and Gate Selection in 13-Task Financial Recommendation",
  author: ("Seonkyu Jeong", "Euncheol Sim", "Youngchan Kim"),
)

#set page(
  paper: "us-letter",
  margin: (x: 1.8cm, y: 2cm),
  numbering: "1",
)

#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.6em)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")

// Bibliography setup
#set bibliography(style: "ieee")

// Title
#align(center)[
  #text(size: 14.5pt, weight: "bold")[
    Scaling Multi-Task Learning Beyond Homogeneous Tasks: \
    Loss Dynamics and Gate Selection in \
    13-Task Financial Recommendation
  ]

  #v(0.8em)

  #text(size: 11pt)[
    Seonkyu Jeong#super[1], Euncheol Sim#super[1], Youngchan Kim#super[1]
  ]

  #v(0.3em)

  #text(size: 9pt, style: "italic")[
    #super[1]Independent Research
  ]

  #v(1em)
]

// Abstract
#block(
  width: 100%,
  inset: (x: 1em),
)[
  #text(weight: "bold")[Abstract.]
  Multi-task learning (MTL) architectures such as MMoE and PLE have been validated
  almost exclusively on 2--4 homogeneous tasks (e.g., CTR + CVR).
  We report empirical findings from scaling PLE to *13 heterogeneous tasks*
  --- 7 binary, 3 multiclass, 3 regression ---
  in a financial product recommendation system with 7 structurally distinct experts
  and 1M synthetic customers.
  Six findings challenge conventional MTL wisdom:
  (1) a subtle implementation bug in Kendall et al.'s uncertainty weighting
  --- omitting per-task loss weights --- silently suppresses minority-type tasks,
  producing a +0.018 NDCG\@3 gain when fixed;
  (2) softmax gating outperforms sigmoid for heterogeneous task mixes,
  *reversing* the sigmoid advantage reported in homogeneous settings;
  (3) learned uncertainty weights converge to *identical* values regardless
  of architecture (shared-bottom vs.\ PLE), revealing that uncertainty weighting
  acts as loss-scale normalization rather than structural protection;
  (4) 10-epoch budgets may be insufficient for complex structures to differentiate
  from simpler baselines;
  (5) GroupTaskExpert (GTE) pre-gating *degrades* multiclass performance
  when groups contain mixed task types;
  and (6) gate entropy analysis reveals that CGC extraction-layer gating
  specializes meaningfully (entropy ratios 0.33--0.88 across tasks) while
  attention-level aggregation remains uniformly diffuse (entropy ratio 1.00),
  and composite val-loss is an *unreliable* checkpoint signal because
  regression tasks continue improving after classification/ranking metrics peak.
  We distill these observations into practical guidelines for
  practitioners scaling MTL beyond the homogeneous-task regime.

  #v(0.3em)
  #text(weight: "bold")[Keywords:]
  Multi-task learning, Progressive Layered Extraction, Mixture of Experts,
  gate selection, uncertainty weighting, loss dynamics, heterogeneous tasks
]

// Switch to 2-column layout for body
#show: rest => columns(2, rest)

= Introduction

Multi-task learning (MTL) promises parameter efficiency and positive transfer
by jointly optimizing related tasks @caruana1997.
In recommendation systems, MTL has become standard practice:
MMoE @ma2018 introduced multi-gate mixture-of-experts to handle task conflicts,
PLE @tang2020 added progressive extraction layers with shared and task-specific
expert separation, and AdaTT @li2023 enabled adaptive inter-task transfer.

However, virtually all published MTL architectures for recommendation are
validated on *2--4 homogeneous tasks* --- typically click-through rate (CTR)
and conversion rate (CVR), both binary classification problems with
aligned gradient directions.
When we attempted to scale PLE to *13 heterogeneous tasks*
--- 7 binary, 3 multiclass (4 to 50 classes), and 3 regression ---
several assumptions broke down in ways that existing literature does not address.

The 13-task configuration was not a design preference but a constraint.
Financial regulations mandate distinct prediction targets:
suitability assessment, fairness monitoring across protected attributes,
churn early-warning, and product-level acquisition propensity
each require a separate supervised signal.
Meanwhile, limited infrastructure --- a single desktop GPU (12GB VRAM)
and a 3-person team --- precludes maintaining separate models per task.
The result is a regime that large-scale CTR teams have no reason to enter
(they can afford model-per-task) but that resource-constrained regulated
industries are forced into.

This paper reports six empirical findings from this scaling experience.
We make no claims of state-of-the-art performance;
instead, we document *phenomena and practical guidelines*
that emerge when MTL is pushed beyond the homogeneous-task regime.
Our contributions:

- A diagnosis of how Kendall et al.'s uncertainty weighting
  silently fails when per-task loss weights are omitted (Section 4.1).
- Evidence that *gate type selection depends on task-type homogeneity*,
  not on architectural sophistication (Section 4.2).
- A demonstration that uncertainty weights converge identically
  across architectures, limiting their protective role (Section 4.3).
- Analysis of epoch budget sensitivity in structure comparison (Section 4.4).
- A cautionary finding on pre-gating task grouping (GTE)
  with mixed-type groups (Section 4.5).
- Gate entropy analysis showing that CGC extraction-layer specialization
  is real and task-dependent, while attention-level aggregation collapses
  to uniform averaging; and a demonstration that val-loss is a misleading
  checkpoint criterion when regression and classification tasks coexist
  (Section 4.6).
- A 9-way comparison of fusion augmentations on top of the CGC
  baseline that identifies *two positive recipes on disjoint axes* and
  shows that they are *not additive*. Five representation-additive
  fusions (loss-level adaTT, AdaTT-sp, complementary-gate recovery,
  uncertainty-conditioned expert bank, and MV BRP) all degrade AUC
  monotonically with intervention invasiveness. *Output-space
  boosting with shared-expert gradient isolation* (BRP-detached) ties
  CGC on aggregate AUC ($Delta = -0.0007$; best epoch exceeds baseline
  by $+0.0008$) while lifting F1 macro $+0.007$ and NDCG\@3 $+0.015$
  and retaining $+$256% relative rescue on the hardest multiclass
  task. *Training-time load-balancing regularisation* (NEAS ---
  auxiliary supervision on the inverse-gate aggregation) is the first
  mechanism of the family to actually raise aggregate AUC
  ($Delta = +0.0011$), with a monotone-increasing trajectory and near-
  uniform per-task lifts. Stacking the two positive recipes collapses
  NEAS's AUC gain because the shared experts cannot simultaneously be
  generalists (NEAS) and primary-supporting specialists
  (BRP-detached). The guidance is per-objective: NEAS for aggregate
  AUC and cross-task robustness, BRP-detached for hard-task rescue,
  do not stack (Section 4.7).
- A structural diagnostic showing that the causal expert's learnable
  adjacency matrix $bold(W)$ collapsed to zero across every trained
  checkpoint we examined (four local, two upstream on-prem),
  rendering the expert's forward pass equivalent to a plain MLP in
  spite of its NOTEARS regularisation. The failure is a saddle-point
  problem (both task-loss and reconstruction gradients have a $bold(W)$
  factor that vanishes at the initialisation scale) and resolves
  under a two-part patch: add the original-paper reconstruction term
  ($||bold(z) - bold(z) bold(W)^2||_F^2$) and rescale initialisation
  from $0.01$ to $0.1$. Post-patch the expert learns a valid sparse
  DAG ($bold(W)$ Frobenius $0.34$, $7.3%$ edges active, $h(bold(W)) = 0$),
  but aggregate task metrics are unchanged --- the DAG exists
  structurally but is not routed into prediction by the current
  architecture (Section 4.8).

The system, data generator, and ablation scripts are publicly available.#footnote[
  https://github.com/bluethestyle/aws\_ple\_for\_financial
]

= Related Work

== Multi-Task Learning in Recommendations

Shared-bottom networks @caruana1997 suffer from negative transfer when tasks
conflict. MMoE @ma2018 mitigates this with per-task gating over a shared
expert pool, and PLE @tang2020 further separates shared and task-specific
extraction layers. AdaTT @li2023 adds adaptive task-to-task transfer strength.

A common thread: all evaluations use 2--4 tasks of the *same type*.
MMoE's Census experiment uses 2 binary tasks.
PLE's production evaluation at Tencent uses 2 tasks (CTR + VCR).
AdaTT's Alibaba experiment uses 3 closely related engagement tasks.
No published PLE/MMoE study evaluates on a mix of binary,
multiclass, and regression tasks at the scale we report.

== Loss Balancing in MTL

Kendall et al. @kendall2018 introduced homoscedastic uncertainty weighting,
learning per-task precision parameters that automatically balance loss scales.
GradNorm @chen2018gradnorm dynamically adjusts loss weights based on gradient
norms. MGDA @sener2018multi solves a multi-objective optimization problem
at each step.

These methods are designed for and tested on scenarios where all tasks share
similar loss magnitudes. When binary cross-entropy (scale ~0.5),
multiclass cross-entropy (scale ~3.9 for 50-class), and regression MSE
(scale ~0.01--1.0) coexist, the implicit assumptions of these balancing
methods deserve re-examination.

== Gate Design: Softmax vs.\ Sigmoid

Standard PLE and MMoE use *softmax* gates, enforcing competitive, sum-to-one
expert selection. Nguyen et al. @sigmoid_moe2024 demonstrated that sigmoid gating
--- allowing each expert to contribute independently without competition ---
achieves higher sample efficiency by eliminating inter-expert competition.

This finding, however, was established on homogeneous task sets.
We show that the sigmoid advantage *reverses* when tasks are heterogeneous,
because independent expert activation allows high-gradient binary tasks
to contaminate experts that multiclass tasks depend on.

= Architecture

== PLE with Heterogeneous Expert Basket

Our PLE implementation follows Tang et al. @tang2020 with a key modification:
instead of $K$ identical MLP experts, we employ 7 *architecturally distinct*
experts, each encoding a different inductive bias:

#figure(
  table(
    columns: (auto, auto, auto),
    inset: 5pt,
    align: (left, left, right),
    stroke: 0.5pt,
    table.header([*Expert*], [*Architecture*], [*Input Dim*]),
    [DeepFM], [Factorization Machine + DNN], [168D],
    [Temporal], [Mamba + LNN + Transformer], [139D],
    [HGCN], [Hyperbolic GCN (Poincaré)], [27D],
    [PersLay], [Topological (TDA)], [32D],
    [Causal], [NOTEARS DAG], [161D],
    [LightGCN], [Graph Convolution], [100D],
    [Optimal Transport], [Sinkhorn matching], [127D],
  ),
  caption: [Expert basket. Each expert receives a different feature subset
  via FeatureRouter. Total input space: ~350D (Phase 0 v3/v4).],
) <tab:experts>

A *FeatureRouter* assigns each expert its designated feature groups,
declared in YAML configuration rather than hardcoded.
The sum of per-expert input dimensions (703D) exceeds the total feature
space (350D) because several feature groups are shared across experts
with complementary inductive biases.

== CGC Gating

The Customized Gate Control (CGC) module computes an attention over the $K$
experts for each task $t$:

$ g_t = "softmax"(W_t dot.c h_t) in RR^K $
or
$ g_t = sigma(w_t dot.c h_t) slash sum_j sigma(w_j dot.c h_j) $

where $h_t$ is the shared representation.
*Softmax* gates enforce competitive, sum-to-one allocation.
*Sigmoid* gates allow independent evaluation, normalized post-hoc.
The choice between these two is a central finding of this paper (Section 4.2).

== Uncertainty Weighting

Following Kendall et al. @kendall2018, we learn a log-variance parameter
$log sigma^2_t$ per task. The weighted loss becomes:

$ cal(L)_t^"uw" = w_t dot.c (1 / (2 sigma_t^2) dot.c cal(L)_t + 1/2 log sigma_t^2) $ <eq:uw-correct>

where $w_t$ is the per-task loss weight from configuration.
The precision $1/(2 sigma_t^2)$ and the regularizer $1/2 log sigma_t^2$
are clamped to stable ranges:
$log sigma^2_t in [-4, 4]$, precision $in [10^(-3), 100]$.

The critical detail is that $w_t$ *must multiply the entire expression*,
not just $cal(L)_t$. Omitting $w_t$ is the bug we report in Section 4.1.

== AdaTT Task Groups

Tasks are organized into 4 Financial DNA groups:

#figure(
  table(
    columns: (auto, 1fr),
    inset: 5pt,
    align: (left, left),
    stroke: 0.5pt,
    table.header([*Group*], [*Tasks*]),
    [Engagement], [has\_nba, churn\_signal, next\_mcc, mcc\_trend, top\_mcc\_shift],
    [Lifecycle], [nba\_primary, segment],
    [Value], [cross\_sell\_count, product\_stability],
    [Consumption], [will\_acquire\_\* (5 tasks), mcc\_diversity\_trend],
  ),
  caption: [Financial DNA task grouping for AdaTT transfer.],
) <tab:taskgroups>

AdaTT @li2023 learns intra-group and inter-group transfer strengths.
Note that the Consumption group mixes 5 binary tasks with
1 regression task --- a design that becomes relevant in Section 4.5.

= Results and Analysis

All experiments use 1M synthetic customers, 349 features (Phase 0, benchmark_v12),
10 epochs, batch size 5632, learning rate 0.0005, AMP (FP16), cosine annealing
with warm restarts ($T_0 = 10$). Uncertainty weighting is applied in all runs
unless noted otherwise. Each configuration is run with 3 seeds; we report
medians. Metrics are reported per task type: Avg AUC (binary), Avg F1-macro
(multiclass), Avg MAE (regression), NDCG\@3 and Acc\@3 (ranking).

== Finding 1: The Silent Uncertainty Weighting Bug <find1>

When porting Kendall et al.'s uncertainty weighting from our on-premises codebase
to the AWS implementation, a subtle bug was introduced.
The on-premises code correctly implements @eq:uw-correct:

```python
# On-premises (correct)
loss = loss_weight * (precision * task_loss + log_var)
```

The AWS port omitted the per-task `loss_weight` and the clamping:

```python
# AWS port (buggy)
loss = precision * task_loss + log_var / 2
```

This omission has two consequences:
(1) the `loss_weight` parameter from `pipeline.yaml` --- which compensates for
cross-entropy scale differences between binary (~0.5) and 50-class multiclass
(~3.9) tasks --- is silently ignored;
(2) without clamping, extreme log-variance values can push precision
to numerically unstable regions.

The effect is that multiclass tasks, whose raw loss is ~8× larger than
binary tasks, receive proportionally *less* gradient when precision
parameters converge to similar values.
Binary tasks, being numerically dominant (7 of 13), further suppress
multiclass learning through sheer gradient volume.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    inset: 5pt,
    align: (left, right, right, right),
    stroke: 0.5pt,
    table.header([*Metric*], [*Buggy*], [*Fixed*], [*Δ*]),
    [Avg AUC (binary)], [{{buggy_auc}}], [{{fixed_auc}}], [{{delta_auc}}],
    [Avg F1-macro (multiclass)], [{{buggy_f1}}], [{{fixed_f1}}], [+0.031],
    [NDCG\@3], [{{buggy_ndcg}}], [{{fixed_ndcg}}], [+0.018],
    [Avg MAE (regression)], [{{buggy_mae}}], [{{fixed_mae}}], [{{delta_mae}}],
    [Val Loss], [{{buggy_valloss}}], [{{fixed_valloss}}], [{{delta_valloss}}],
  ),
  caption: [Impact of uncertainty weighting bug fix (shared-bottom baseline).
  F1-macro and NDCG\@3 improve substantially; binary AUC change is minimal
  because binary tasks were not suppressed.],
) <tab:bugfix>

*Lesson*: Uncertainty weighting implementations must be verified against the
*original* mathematical formulation, not just tested for convergence.
The buggy version still converges --- it simply converges to a suboptimal
loss balance that favors the majority task type.

== Finding 2: Gate Selection Depends on Task Homogeneity <find2>

Existing literature suggests that sigmoid gating outperforms softmax
in PLE/MoE architectures @sigmoid_moe2024.
Our results show this holds only when tasks are homogeneous.
With 13 heterogeneous tasks, softmax *consistently* outperforms sigmoid
on ranking metrics:

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    inset: 5pt,
    align: (left, right, right, right, right, right),
    stroke: 0.5pt,
    table.header([*Variant*], [*Val Loss*], [*Avg AUC*], [*Avg F1m*], [*Avg MAE*], [*NDCG\@3*]),
    [Shared Bottom], [{{sb_valloss}}], [{{sb_auc}}], [{{sb_f1}}], [{{sb_mae}}], [{{sb_ndcg}}],
    [PLE Softmax], [{{soft_valloss}}], [{{soft_auc}}], [{{soft_f1}}], [{{soft_mae}}], [*{{soft_ndcg}}*],
    [PLE Sigmoid], [{{sig_valloss}}], [{{sig_auc}}], [{{sig_f1}}], [{{sig_mae}}], [{{sig_ndcg}}],
    [PLE Sigmoid + adaTT], [{{sigadatt_valloss}}], [{{sigadatt_auc}}], [{{sigadatt_f1}}], [{{sigadatt_mae}}], [{{sigadatt_ndcg}}],
  ),
  caption: [Gate type comparison on 13 heterogeneous tasks.
  Softmax achieves the best NDCG\@3. Bold = best per column.],
) <tab:gatetype>

*Why softmax wins here*: In a softmax gate, each task's attention over experts
sums to 1, creating *competitive allocation*. When a binary task claims
Expert A with weight 0.6, only 0.4 remains for other experts.
This competition *isolates* experts per task type,
preventing high-gradient binary tasks from contaminating experts
that multiclass tasks depend on.

Sigmoid gates allow each expert to contribute independently.
This is beneficial when all tasks have similar gradient magnitudes
(homogeneous case), because it enables richer expert combinations.
But with 7 binary tasks producing high-gradient signals and 3 multiclass
tasks producing relatively smaller gradients,
sigmoid allows binary gradients to flow through *all* experts simultaneously,
overwhelming the weaker multiclass signal.

The decision criterion is not architectural sophistication but
*task-type homogeneity*:
- Homogeneous tasks (all binary, or all regression): prefer sigmoid (richer mixing).
- Heterogeneous tasks (mixed types): prefer softmax (gradient isolation).

== Finding 3: Uncertainty Weights Converge Identically Across Architectures <find3>

A surprising observation: the learned uncertainty weights at epoch 10
are *virtually identical* between shared-bottom and PLE-softmax:

#figure(
  table(
    columns: (auto, auto, auto, auto),
    inset: 5pt,
    align: (left, right, right, right),
    stroke: 0.5pt,
    table.header([*Task*], [*Shared-Bottom*], [*PLE Softmax*], [*Δ*]),
    [has\_nba (binary)], [{{sb_hasnba_uw}}], [{{ple_hasnba_uw}}], [{{delta_hasnba_uw}}],
    [nba\_primary (7-class)], [0.3353], [0.3354], [+0.0001],
    [next\_mcc (50-class)], [0.3360], [0.3361], [+0.0001],
    [cross\_sell (regression)], [{{sb_cross_uw}}], [{{ple_cross_uw}}], [{{delta_cross_uw}}],
    [churn\_signal (binary)], [{{sb_churn_uw}}], [{{ple_churn_uw}}], [{{delta_churn_uw}}],
  ),
  caption: [Learned uncertainty weights at epoch 10 (selected tasks).
  Differences are at the 4th decimal place regardless of architecture.],
) <tab:uw-convergence>

This means uncertainty weighting performs *loss-scale normalization* ---
mapping each task's loss to a comparable magnitude --- but does not provide
*structural protection* against gradient interference.
The structural protection comes from gating (Section 4.2):
with identical uncertainty weights, softmax still achieves
NDCG\@3 {{soft_ndcg}} vs.\ shared-bottom {{sb_ndcg}},
a +{{delta_ndcg_gate}} improvement attributable purely to gate structure.

*Implication*: Practitioners should not rely on uncertainty weighting alone
to handle task-type conflicts. It balances scales but does not prevent
gradient contamination. Gate design is the actual mechanism for protecting
minority task types.

== Finding 4: Epoch Budget Sensitivity <find4>

At 10 epochs, the performance gap between structures is small
(AUC varies by ±0.002 across variants). This raises the question:
are complex structures genuinely no better, or simply underfitted?

#figure(
  table(
    columns: (auto, auto, auto, auto),
    inset: 5pt,
    align: (left, right, right, right),
    stroke: 0.5pt,
    table.header([*Epoch*], [*Shared-Bottom*], [*PLE Softmax*], [*PLE Sig + adaTT*]),
    [5], [{{sb_e5_auc}}], [{{soft_e5_auc}}], [{{sigadatt_e5_auc}}],
    [6 (NDCG peak)], [{{sb_e6_auc}}], [{{soft_e6_auc}}], [{{sigadatt_e6_auc}}],
    [10], [{{sb_e10_auc}}], [{{soft_e10_auc}}], [{{sigadatt_e10_auc}}],
    [20], [{{sb_e20_auc}}], [{{soft_e20_auc}}], [{{sigadatt_e20_auc}}],
  ),
  caption: [Avg AUC by epoch. Complex structures may differentiate
  with longer training budgets. 20-epoch results pending.],
) <tab:epoch>

NDCG\@3 peaks at epoch 6, coinciding with the cosine learning rate valley
before restart. This suggests that ranking quality is sensitive to learning
rate scheduling, and the apparent performance plateau at epoch 10 may reflect
a scheduling artifact rather than a convergence plateau.

With cosine warm restarts ($T_"mult" = 2$), the second cycle spans epochs 10--30.
Complex structures (PLE + adaTT) have more parameters to warm up, and may
only differentiate from simpler baselines in the second cosine cycle.

*Guideline*: When comparing MTL structures, ensure the training budget
is at least $2 times T_0$ to observe post-restart behavior.
10-epoch comparisons may prematurely favor simpler architectures.

== Finding 5: GTE Pre-Gating Degrades Mixed-Type Groups <find5>

GroupTaskExpert (GTE) adds a pre-gating layer that partitions expert
representations by task group *before* PLE gating occurs.
The motivation is to strengthen intra-group expert sharing.

However, when a task group contains mixed types
--- the Consumption group has 5 binary tasks and 1 regression task ---
GTE forces shared representation learning across incompatible loss types:

#figure(
  table(
    columns: (auto, auto, auto),
    inset: 5pt,
    align: (left, right, right),
    stroke: 0.5pt,
    table.header([*Variant*], [*NDCG\@3*], [*Avg AUC*]),
    [PLE Softmax], [{{soft_ndcg}}], [{{soft_auc}}],
    [PLE Sigmoid + GTE], [{{siggte_ndcg}}], [{{siggte_auc}}],
    [Δ], [{{delta_gte_ndcg}}], [{{delta_gte_auc}}],
  ),
  caption: [GTE impact. NDCG\@3 drops substantially when GTE forces
  mixed-type groups to share pre-gate representations.],
) <tab:gte>

The mechanism: GTE pools expert outputs at the group level *before*
per-task PLE gating can differentiate them.
Within the Consumption group, the 5 binary tasks' high-gradient signals
dominate the pooled representation, and the single regression task
(`mcc_diversity_trend`) receives a representation already biased
toward binary decision boundaries.
PLE gating, which operates *after* GTE pooling, cannot undo this damage.

*Guideline*: Task groups for GTE (or similar pre-gating mechanisms) must be
homogeneous by task type, not by business semantics.
If a business-meaningful grouping mixes types,
omit GTE and rely on PLE gating alone for inter-expert allocation.

== Finding 6: Gate Entropy and Loss–Metric Decoupling <find6>

=== CGC Gate Entropy Analysis

To understand *how* PLE's CGC gating allocates experts in practice,
we compute the Shannon entropy ratio of each task's gate weight distribution
$g_t in RR^K$ at the end of teacher training (30 epochs).
The entropy ratio is defined as:

$ E_t = H(g_t) / H_"max" = -( sum_k g_{t,k} log g_{t,k} ) / log K $

where $H_"max" = log K$ is the maximum entropy for $K$ experts.
$E_t = 1$ means perfectly uniform expert utilization;
$E_t = 0$ means a single expert captures all weight.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    inset: 5pt,
    align: (left, right, right, left),
    stroke: 0.5pt,
    table.header([*Task*], [*Layer 1*], [*Layer 2*], [*Pattern*]),
    [top\_mcc\_shift], [0.347], [---], [Single-expert dominance],
    [product\_stability], [0.431], [---], [Single-expert dominance],
    [segment\_prediction], [---], [0.332], [Single-expert dominance],
    [cross\_sell\_count], [0.570], [0.614], [Moderate diversity],
    [churn\_signal], [0.691], [0.860], [Moderate → full diversity],
    [nba\_primary], [0.851], [0.839], [Full expert utilization],
    [will\_acquire\_payments], [0.882], [---], [Full expert utilization],
  ),
  caption: [CGC gate entropy ratios by task and PLE layer (teacher model,
  30 epochs). Low entropy ($E_t < 0.45$) indicates 1--2 experts dominate;
  high entropy ($E_t > 0.80$) indicates all 7 experts contribute meaningfully.],
) <tab:gate-entropy>

The entropy ratios reveal three behaviorally distinct task clusters:

*Single-expert dominance* ($E_t$ 0.33--0.43): Tasks such as `top_mcc_shift`
(MCC category shift prediction) and `segment_prediction` (3-class customer segment)
are captured by 1--2 experts. These tasks appear to encode simple patterns
that a single specialized expert --- DeepFM for transactional features,
or HGCN for hierarchical segments --- handles near-optimally.
The low entropy is not a failure mode; it is efficient routing.

*Moderate diversity* ($E_t$ 0.57--0.72): Tasks such as `cross_sell_count`
(count regression) and some binary acquisition tasks draw on 3--4 experts.
These tasks likely require both transactional signals (DeepFM) and
sequence-level patterns (Temporal), explaining partial expert spread.

*Full expert utilization* ($E_t$ 0.85--0.88): `nba_primary` (7-class next
best action) and `will_acquire_payments` (binary) actively use all 7 experts.
These tasks encode complex, multi-faceted customer behavior patterns that no
single architectural inductive bias fully captures.

=== Attention Collapse: A Structural Blind Spot

At the attention aggregation level (shared expert aggregation), all 13 tasks
exhibit an entropy ratio of *exactly 1.000*. The attention mechanism has not
learned to differentiate --- it acts as a simple average over the expert pool.

This is structurally significant. The CGC extraction-layer gates (Table @tab:gate-entropy)
demonstrate that *the model can learn differentiated expert preferences*,
but this capacity is entirely absent at the attention-aggregation level.
Two possible explanations:

+ *Gradient starvation*: The attention parameters receive gradients only
  after passing through per-task tower heads, which already specialize via
  the extraction-layer gates. By the time signal reaches the attention layer,
  per-task distinction may be adequately handled upstream.

+ *Parameterization bottleneck*: If the attention query dimension is small
  relative to the expert embedding space, the attention has insufficient
  capacity to form task-specific preferences across 7 heterogeneous experts.

In either case, the attention component adds parameters without performing
meaningful routing. This is a candidate for architectural simplification ---
replacing attention aggregation with a fixed average or a learned scalar per
task --- in future work.

=== Loss–Metric Decoupling at 30 Epochs

Extending training from 10 to 30 epochs (with $T_0 = 10$, cosine warm
restarts) exposes a fundamental tension in composite loss monitoring.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    inset: 5pt,
    align: (left, right, right, right, right),
    stroke: 0.5pt,
    table.header([*Epoch*], [*Val Loss*], [*Avg AUC*], [*NDCG\@3*], [*Avg MAE*]),
    [1], [32.11], [---], [---], [1.07],
    [10], [26.43], [*0.6726*], [*0.6976*], [1.01],
    [11], [25.89], [0.6702], [0.6853], [0.99],
    [19], [24.01], [0.6718], [0.7004], [0.97],
    [20], [23.52], [0.6691], [0.6657], [0.96],
    [29], [22.95], [0.6704], [0.6540], [0.96],
    [30], [22.68], [0.6687], [---], [0.96],
  ),
  caption: [Loss–metric decoupling over 30 epochs (teacher model, 1M customers,
  $T_0=10$ cosine warm restarts). Val loss decreases monotonically while AUC
  peaks at epoch 10 (−0.4\%p by epoch 30) and NDCG\@3 peaks at epoch 19
  then collapses (−4.4\%p by epoch 29). Avg MAE continues improving throughout.
  Bold = metric peak across all epochs.],
) <tab:loss-metric-decouple>

Val loss decreases monotonically (32.11 → 22.68), which would conventionally
indicate continuous improvement. However:

- *Avg AUC* (binary tasks): peaks at epoch 10 (0.6726), then declines to 0.6687
  by epoch 30 (−0.4%p). The decline is modest but consistent.
- *NDCG\@3* (ranking quality): peaks at epoch 19 (0.7004), then collapses to
  0.6540 at epoch 29 (−4.6%p relative to peak, −3.0%p relative to epoch 10).
- *Avg MAE* (regression tasks): improves steadily, 1.07 → 0.96, throughout
  all 30 epochs.

The root cause is *task-type dominance in composite loss*:
regression tasks contribute continuously shrinking MAE to the composite loss
and pull the aggregate downward even as classification tasks saturate.
With 3 regression tasks whose losses have no natural lower bound (unlike
cross-entropy, which approaches 0 for well-separated data),
the composite loss signal is *not a valid proxy* for classification or ranking
quality after the first cosine cycle.

=== Cosine Restart Oscillation Across Task Types

Cosine warm restarts ($T_0 = 10$) create learning rate spikes at cycle
boundaries. NDCG\@3 exhibits strong oscillation at these boundaries:

#figure(
  table(
    columns: (auto, auto, auto),
    inset: 5pt,
    align: (left, right, left),
    stroke: 0.5pt,
    table.header([*Epoch (event)*], [*NDCG\@3*], [*Change*]),
    [10 (cycle 1 end)], [0.6976], [peak],
    [11 (restart 1)], [0.6853], [−1.2%p (sharp drop)],
    [19 (cycle 2 near-end)], [0.7004], [recovery, new peak],
    [20 (restart 2)], [0.6657], [−3.5%p (sharp drop)],
    [29 (cycle 3 near-end)], [0.6540], [recovery failure],
  ),
  caption: [NDCG\@3 oscillation at cosine restart boundaries.
  Each LR spike pushes the model away from the ranking-optimal parameter region.
  The second restart produces a larger drop than the first, and the third
  cycle fails to recover, indicating progressive divergence.],
) <tab:cosine-oscillation>

The pattern is asymmetric across task types. Avg MAE is *unaffected* by
restarts --- regression loss landscapes are smooth and the optimizer quickly
returns to a low-MAE region after each LR spike. Binary AUC shows a small dip
(−0.4%p per restart) and recovers partially. NDCG\@3 suffers the largest
disruption (up to −3.5%p per restart) because ranking metrics are sensitive to
the relative ordering of scores, and LR restarts temporarily scramble the score
scale before the model re-converges.

*Implication*: For ranking-sensitive applications, cosine restarts with
$T_"mult" = 1$ (constant cycle length) should be replaced with
$T_"mult" = 2$ (doubling cycle length) or with learning rate warmup-then-decay
(no restart), evaluated over a 30-epoch budget.

=== Checkpoint Selection Criterion

These findings jointly establish that *val loss is an invalid checkpoint
criterion* when tasks of different types share a composite loss.
The correct approach is a composite checkpoint metric that weights metric
semantics by task type:

$ "score"_"ckpt" = alpha dot.c "AvgAUC" + beta dot.c "NDCG@3" + gamma dot.c (1 - "NormMAE") $

where $alpha, beta, gamma$ are set to weight task types equally
(e.g., $alpha = beta = gamma = 1/3$) rather than proportional to task count.
In our 13-task configuration, using this composite score selects epoch 10
as the optimal checkpoint, matching the AUC peak --- not epoch 29, which
minimizes val loss.

*Guideline*: When regression tasks are present in a heterogeneous MTL system,
(1) define a composite checkpoint metric across task types before training begins,
(2) checkpoint every epoch and select post-hoc rather than monitoring val loss,
and (3) treat val loss as a diagnostic (indicating regression progress) rather
than as the primary stopping criterion.

== Finding 7: Two Positive Fusion Recipes on Disjoint Axes, Non-Additive Under Composition <find7>

After Finding 2 established that the loss-level `adaTT` variant does not
affect aggregate AUC at the 13-task scale ($Delta = -0.001$, null),
eight further mechanisms were evaluated on the same benchmark to map out
which forms of fusion augmentation (if any) can extract useful signal
beyond CGC's gated selection. The nine-way comparison resolves into
three regions. *Representation-additive fusions* (five variants: loss-
level adaTT, AdaTT-sp, M1 complement, ECEB, and BRP-MV) all inject a
residual into the primary representation or propagate residual-error
gradients into shared experts; all five degrade aggregate AUC with a
magnitude that scales monotonically with the invasiveness of the
intervention. *Output-space boosting with gradient isolation* (BRP-
detached) preserves AUC ($Delta = -0.0007$, tied) while lifting F1 macro
by $+0.007$ and NDCG\@3 by $+0.015$. *Training-time load-balancing
regularisation* (NEAS --- an inverse-gate auxiliary supervision) is the
first mechanism of the family to *raise* aggregate AUC ($Delta = +0.0011$),
with near-uniform small lifts across most tasks. The two positive
recipes act on disjoint axes --- error correction in output space vs.
expert-collapse prevention at the gate --- yet a ninth experiment
stacking them produces a *non-additive* outcome: NEAS's AUC lift
vanishes while BRP-detached's hard-task rescue partially survives,
because the shared experts cannot simultaneously generalise for NEAS
and specialise for the primary-supporting optimum that BRP-detached
relies on.

=== Five Mechanisms, One Aggregate-AUC Conclusion

*Loss-level adaTT* (the variant reported in Paper 1) adds a weighted
cross-task loss term, $L_i + lambda sum_(j != i) w(i arrow.r j) L_j$,
with transfer weights estimated from gradient cosine similarity.

*AdaTT-sp* @li2023 adds a native-expert residual: after the CGC gate
produces the weighted sum, the mean output of the task's own task-specific
experts is re-injected, scaled by a learnable scalar.

*Residual complement (M1)*, introduced in this paper, preserves the
primary gated output and adds a complementary weighting
$(1 - "gate_weights")$ (clamped and renormalised over the expert axis)
applied to the same expert outputs as a residual, scaled by a learnable
scalar. The intent is to recover intra-task signal from experts the gate
down-weighted, without any cross-task mixing.

*ECEB (Error-Conditioned Expert Bank, MV)*, introduced in this paper, is
designed specifically to escape the shared structure of the three above:
the residual is derived from the gate's *entropy* rather than from the
gate's output. Concretely, the recovery path is a task-agnostic consensus
(mean over all expert outputs, no gate weighting), scaled at forward time
by the product of a per-task learnable scalar $sigma(w_t)$ and the
normalised gate entropy $H(g_t)/log N$ (per sample). When the gate is
confident (low entropy), recovery collapses toward zero; when it is
distributed, recovery activates. By construction ECEB eliminates the
"gate-derived residual" factor.

*BRP (Boosting-Residual Path, MV)*, also introduced in this paper, removes
the additive-on-representation structure altogether. A per-task residual
expert bank takes the last CGC layer's `shared_concat` (gate-bypass
feature view) and produces a logit residual matching the primary tower's
output shape. The residual is trained by MSE against the primary's
*detached* prediction error, i.e. $y - "activation"("stop_grad"("primary"))$
(single-stage boosting). The primary pathway trains on ground truth
alone; the combined output $"primary" + sigma(lambda_t) dot "residual"$
is used for inference and evaluation only. The primary representation is
never touched.

*BRP-detached*, a single-line modification of BRP, feeds
`shared_concat.detach()` into the residual bank instead of the raw
`shared_concat`. This cuts any residual-MSE gradient from flowing back
into the shared experts while leaving the primary pathway, parameter
count, and training schedule unchanged. The modification was motivated
by the per-task analysis of BRP below.

*NEAS (Neglected-Expert Auxiliary Supervision)* adds a per-task
auxiliary head consuming the *inverse-gate-weighted aggregation* of the
last CGC layer's expert outputs. The auxiliary target is the primary
task label, and the auxiliary loss (scaled by `aux_weight = 0.05`) is
added to the total loss during training only; inference does not use
the auxiliary head at all. The mechanism explicitly forces neglected
experts --- those the gate de-emphasises for a given task --- to retain
predictive representations, mitigating expert collapse. NEAS is
structurally independent of all residual mechanisms above; it neither
injects a residual nor modifies the primary output.

*NEAS + BRP-detached* stacks the two positive mechanisms, testing
additivity. Both are enabled with their standalone settings; no other
modification.

Results on the 13-task benchmark (10 epochs, seed=42):

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, center, center, center, center),
    stroke: 0.5pt,
    inset: 5pt,
    table.header(
      [*Fusion*], [*Final AUC*], [*F1 macro*], [*NDCG\@3*], [*$Delta$ AUC*]
    ),
    [CGC gate (baseline)],   [0.6728],   [0.2002],   [0.6820],   [---],
    [Loss-level adaTT],      [0.6717],   [0.2013],   [0.6646],   [$-$0.0011],
    [AdaTT-sp (Li 2023)],    [0.6696],   [0.1998],   [0.6570],   [$-$0.0032],
    [M1 complement],         [0.6675],   [0.1998],   [0.6611],   [$-$0.0053],
    [ECEB (MV)],             [0.6665],   [0.1998],   [0.6549],   [$-$0.0063],
    [BRP (MV)],              [0.6650],   [*0.2117*], [*0.7039*], [$-$0.0078],
    [BRP-detached],          [0.6721],   [0.2075],   [0.6965],   [$-$0.0007 (tied)],
    [*NEAS*],                [*0.6739*], [0.2019],   [0.6896],   [*$+$0.0011 (positive)*],
    [NEAS + BRP-detached],   [0.6722],   [0.2062],   [0.6864],   [$-$0.0006 (non-additive)],
  ),
  caption: [9-way comparison of fusion mechanisms on the 13-task
  benchmark. The five representation-additive variants (rows 2--6)
  degrade aggregate AUC below CGC, with magnitude growing monotonically
  in the invasiveness of the intervention. BRP-detached (row 7) ties CGC
  on AUC ($Delta = -0.0007$, best $= 0.6736$ at epoch 8, $+$0.0008 over
  the baseline final) and lifts F1/NDCG\@3. NEAS (row 8) is the first
  mechanism of the family to *raise* aggregate AUC ($Delta = +0.0011$),
  with a monotone-increasing trajectory through all 10 epochs and near-
  uniform small per-task lifts. The combined scenario (row 9) collapses
  NEAS's AUC gain --- the two mechanisms exert opposing pressures on the
  shared experts and do not stack.]
) <tab:fusion9way>

M1's best AUC at epoch 1 (pre-training) with monotone decline thereafter
indicates that training the learnable recovery weight actively degrades
performance --- random initialisation is a less harmful operating point
than the converged weight.

=== Per-Task Breakdown and the Two Outliers

Aggregate deltas are at noise level ($<= 0.005$) across all variants, but
a per-task breakdown reveals three regimes:

- *Gate-saturated tasks* (segment_prediction, top_mcc_shift,
  mcc_diversity_trend) have low gate entropy (ratio $< 0.55$) and are
  insensitive to every recovery mechanism ($abs(Delta) <= 0.003$).
- *Gate-distributed tasks with a strong primary* (churn_signal,
  will_acquire_lending) have high gate entropy (ratio $> 0.82$) and show
  the largest M1 degradation ($-$0.020 and $-$0.009).
- *A single positive outlier* is next_mcc (50 classes, near-random base F1
  $approx$ 0.01): all three recovery variants improve it by
  $+$0.005 to $+$0.008. The gain is large relative to the base but small
  in absolute terms; we attribute it to the near-floor starting point
  rather than to a genuine recovery effect.

The remaining 8 tasks fall within noise ($abs(Delta) <= 0.005$).

=== Gate Entropy Correlation: Weak Signal

To test whether gate entropy structurally predicts recovery benefit, we
extract per-task gate weights from the final CGC layer of the
joint_full checkpoint and correlate task-level entropy with each variant's
$Delta$:

- Loss-level adaTT: $r = -0.31$
- AdaTT-sp: $r = -0.32$
- M1 complement: $r = -0.40$

All three correlations are negative (higher entropy $arrow.r$ more harm)
with consistent sign, but with $n = 13$ and $p approx 0.18$ none meets
conventional significance. The two outliers
--- churn_signal and next_mcc --- are better explained by task-specific
factors (label construction for churn_signal, near-floor base rate for
next_mcc) than by gate entropy. Gate entropy cannot therefore be claimed
as a structural predictor of recovery benefit on this benchmark.

=== BRP, BRP-detached, and NEAS: Diagnosing the Mechanisms

The first four augmented variants (loss-level adaTT, AdaTT-sp, M1, ECEB)
all inject a residual additively into the primary representation and
produce monotone degradation. BRP is the one variant in the family that
places the residual in output space. BRP's aggregate AUC is
nevertheless the *lowest* of the five non-baseline runs
($Delta = -0.0078$), but the drop is accompanied by the *highest* F1
macro and NDCG\@3 of those five (+0.0115 and +0.0219 over CGC). The
BRP result therefore looked at first like a task-balance trade-off
rather than a success.

A per-task breakdown tells a more specific story. next_mcc (50-class,
baseline macro-F1 at near-random 0.0100) improves to 0.0440
(+340% relative) under BRP --- the hard-task rescue effect. In the
opposite direction, churn_signal --- the task with the highest binary
AUC in the baseline (0.6868) --- drops to 0.6512 ($-$0.036) under BRP,
and this single task accounts for most of the aggregate AUC loss.
Excluding churn_signal, BRP's binary AUC sits near baseline; the
remaining five binary tasks each lose between $-$0.001 and $-$0.010.

The mechanism is a shared-expert gradient leak. BRP's residual bank
consumes `shared_concat`, and residual-MSE gradients propagate back
into the shared experts --- only the residual *target* was detached,
not its input. For tasks where the primary is already near its ceiling,
the shared experts had converged to a primary-supporting optimum and
the additional MSE pressure pulls them off it. For tasks where the
primary struggles (next_mcc), the residual supplies signal the primary
could not extract. The aggregate drop is therefore not an algorithmic
limitation of output-space boosting --- it is an implementation
artefact of training the residual bank through the shared
representation.

BRP-detached tests this directly. Swapping `shared_concat` for
`shared_concat.detach()` in the residual bank's input --- no parameter
change, no training-schedule change --- produces the following
per-task pattern:

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, center, center, center, center),
    stroke: 0.5pt,
    inset: 5pt,
    table.header(
      [*Task*], [*CGC*], [*BRP*], [*BRP-detached*], [*Verdict*]
    ),
    [churn_signal (AUC)],             [0.6868], [*0.6512*], [*0.6852*], [restored],
    [will_acquire_lending (AUC)],     [0.6549], [0.6453],   [0.6553],   [restored],
    [will_acquire_deposits (AUC)],    [0.6534], [0.6493],   [0.6536],   [restored],
    [will_acquire_investments (AUC)], [0.6754], [0.6719],   [0.6764],   [restored],
    [next_mcc (F1 macro)],            [0.0100], [*0.0440*], [*0.0356*], [retained (+256%)],
    [remaining 8 tasks],              [---],    [$plus.minus 0.002$], [$plus.minus 0.002$], [unchanged],
  ),
  caption: [Per-task comparison of BRP and BRP-detached versus CGC on the
  subset of tasks where BRP materially changed the metric. Detaching
  `shared_concat` restores every easy-task AUC loss ($-$0.036 on
  churn_signal drops to $-$0.002) while retaining the majority of the
  next_mcc rescue effect (+340% relative reduces to +256%, still
  dominant over CGC's near-random baseline).]
) <tab:brp-pertask>

The diagnosis is therefore confirmed at the per-task level. The easy-
task AUC loss in BRP was caused by residual-MSE gradients reshaping
shared experts; detaching the input cuts that channel, shared experts
remain on the primary-supporting optimum, and the residual bank uses
only its own parameters to learn the hard-task correction.

*NEAS* takes a different path. Rather than a residual, it attaches an
auxiliary supervision signal to the *inverse-gate-weighted
aggregation* of expert outputs: each task's aux head must predict the
primary label using mostly the experts the gate de-emphasised. This
creates gradient pressure on every shared expert to remain predictive
even when the task-level gate concentrates. The effect is a training-
time regulariser against expert collapse. NEAS's trajectory rises
monotonically through all 10 epochs, and its per-task lift is spread
across 11 of 13 tasks (six of seven binaries improve by $+$0.0004 to
$+$0.0029; nba_primary lifts F1 by $+0.0056$). Unlike BRP, NEAS does
not produce a large rescue on any single hard task --- next_mcc's F1
moves only from 0.0100 to 0.0107 under NEAS alone, against $+$0.0256
under BRP-detached --- because its mechanism is prevention-of-loss
rather than targeted-correction.

The two positive recipes therefore act on *disjoint axes*:

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, center, center),
    stroke: 0.5pt,
    inset: 5pt,
    table.header(
      [*Dimension*], [*BRP-detached*], [*NEAS*]
    ),
    [Where it acts],                 [Output-space residual],           [Shared-expert gradients via aux loss],
    [Training signal],               [MSE on primary's detached error], [Task loss on inverse-gate aggregation],
    [Inference overhead],            [Non-zero (residual expert)],       [Zero (training-only regulariser)],
    [Parameter addition],            [0.36M (residual bank)],            [0.17M (aux heads)],
    [Aggregate $Delta$ AUC],         [$-0.0007$ (tied)],                 [$+0.0011$ (positive)],
    [Typical per-task pattern],      [One big rescue ($+$256% next_mcc)], [Many small lifts ($plus.minus 0.003$)],
    [Failure mode if stacked],       [Retains next_mcc rescue],          [AUC lift erased],
  ),
  caption: [Structural comparison of the two positive fusion recipes
  identified on this benchmark.]
) <tab:two-recipes>

=== Conclusion: Three Structural Regions, Non-Additive Composition

Across the nine runs, three structural regions organise the space of
fusion augmentations on top of a heterogeneous-expert PLE with CGC
gating:

+ *Representation-additive fusions fail on aggregate AUC.* Five
  variants --- loss-level adaTT, AdaTT-sp, M1 complement, ECEB, and
  BRP-MV (whose `shared_concat` input propagates residual gradients
  into shared experts) --- all inject or propagate residual-error
  signal into the primary-representation path. All five degrade AUC,
  with a monotone relationship between intervention invasiveness and
  degradation magnitude ($-0.001$ to $-0.008$). The residual's
  definition (gate inverse, own expert, uncertainty-gated consensus,
  boosting error) does not change this; what matters is whether the
  mechanism reaches into the primary-supporting representation.
+ *Output-space boosting with gradient isolation succeeds as a tie-
  plus-ranking-gain recipe.* BRP-detached places the residual in
  output space and trains it as a boosting correction on the primary's
  detached error, while detaching the `shared_concat` input to the
  residual bank so residual-MSE gradients never reach the shared
  experts. The result ties CGC on aggregate AUC ($Delta = -0.0007$;
  best epoch exceeds baseline by $+0.0008$) and lifts F1 macro by
  $+0.007$ and NDCG\@3 by $+0.015$, with $+256$% relative rescue on
  the hardest multiclass task.
+ *Training-time load-balancing regularisation succeeds as an AUC-
  positive recipe.* NEAS attaches a per-task auxiliary head to the
  inverse-gate-weighted aggregation of expert outputs, trained against
  the primary label. It is the first mechanism in this family to
  actually raise aggregate AUC ($Delta = +0.0011$), with a monotone-
  increasing trajectory over 10 epochs and near-uniform per-task
  lifts. The mechanism is zero-overhead at inference (training-only
  regulariser) and adds only 0.17M parameters.

*The two positive recipes act on disjoint axes* (error correction in
output space vs. expert-collapse prevention at the gate) and are
independently reproducible. A ninth experiment stacking them produces
a *non-additive* outcome: NEAS's aggregate AUC lift vanishes
($Delta = -0.0006$), while BRP-detached's hard-task rescue partially
survives (next_mcc F1 $+0.0250$ vs. $+0.0256$ standalone). The
mechanism-level explanation is that NEAS pushes shared experts toward
*generalists* (each must predict under inverse-gate reweighting),
whereas BRP-detached holds shared experts on the *primary-supporting
optimum* via the primary task loss. The shared experts are a finite
resource and cannot simultaneously satisfy both pressures.

The practical reading is therefore per-objective rather than stack-
everything:

- If aggregate AUC and uniform cross-task robustness matter most,
  use *NEAS* (zero inference overhead, $+0.0011$ AUC, monotone
  training trajectory).
- If rescue of a near-random multiclass task matters most, use
  *BRP-detached* (ties AUC, $+$256% relative F1 rescue on the hardest
  task).
- Do *not* stack them as-is; doing so erases NEAS's AUC gain without
  producing the hoped-for additive lift. A mechanism that resolves
  the shared-expert conflict (e.g., a scheduler that alternates NEAS
  and BRP-detached pressure across training phases, or a parameter-
  sharing adapter between the two heads) is a natural follow-up but
  not studied here.

== Finding 8: The Causal Expert's Adjacency Matrix Was a Dead Parameter <find8>

During preparatory diagnostics for a follow-up study reinterpreting
the role of the causal expert, an unexpected failure surfaced in the
baseline architecture itself: the causal expert's learnable adjacency
matrix $bold(W) in RR^(32 times 32)$ --- the DAG structure the expert
is supposed to learn --- had collapsed to essentially zero across every
checkpoint we inspected. The effect is general, not an artefact of any
particular scenario:

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, center, center),
    stroke: 0.5pt,
    inset: 5pt,
    table.header([*Checkpoint*], [*$bold(W)$ Frobenius*], [*Entries* $abs(W) > 0.01$]),
    [struct_13_ple_sigmoid (CGC baseline)], [0.0001], [0%],
    [struct_13_residual_complement (M1)],  [0.0001], [0%],
    [struct_13_eceb (MV)],                 [0.0003], [0%],
    [struct_13_brp_detached],              [0.0001], [0%],
    [upstream on-prem implementation],     [0.0001], [0%],
  ),
  caption: [Causal expert's adjacency matrix across five independently
  trained checkpoints, including two from the upstream on-prem
  implementation that predates the port. In every case the Frobenius
  norm of $bold(W)$ is below its random init scale, and not a single
  off-diagonal entry exceeds 0.01 in magnitude. The DAG the expert is
  supposed to learn does not exist after training.]
) <tab:W-collapse>

=== Root Cause: A Residual That Bypasses the DAG

The expert's forward pass is

$ bold(z)_"hat" = bold(z) + bold(z) bold(W)^2 $

where $bold(z) = "feature_compressor"(bold(x))$ and
$bold(z)_"hat"$ feeds a downstream `causal_encoder` MLP. The residual
term $bold(z)$ carries the full latent content regardless of
$bold(W)$, so the task loss has no structural incentive to push
$bold(W)$ away from zero. The NOTEARS acyclicity and sparsity
regularisers both penalise $bold(W)$ away from being dense, but
neither penalises $bold(W) = 0$ --- in fact the sparsity term is
*minimised* there. The global optimum of the combined objective is
therefore $bold(W) = 0$, and training converges to it reliably.

The learned gradient carries the same problem. Both the task-loss
contribution via $partial bold(z)_"hat" slash partial bold(W)$ and
the NOTEARS reconstruction gradient (when added) are proportional to
$bold(W)$ itself:

$ (partial) / (partial bold(W)) "trace" ((bold(W) dot.circle bold(W))^k)
    thin prop thin bold(W) $

Any near-zero initialisation produces a near-zero gradient, so
$bold(W) = 0$ is a *saddle point* that the optimiser cannot escape on
its own.

=== Patch: Reconstruction Loss + Initialisation Rescale

Two changes, both load-bearing:

+ *Reconstruction regulariser.* The original NOTEARS paper
  minimises $||bold(X) - bold(X) bold(W)||_F^2$ --- an
  explicit reconstruction signal. We adopt the compressed-latent
  analogue as a third term in `get_dag_regularization()`:

  $ cal(L)_"recon" = "mean" ((bold(z) - bold(z) bold(W)^2)^2), quad
    lambda_"recon" = 0.5 $

  This re-introduces the direct pressure on $bold(W)$ that the
  original paper relies on.

+ *Initialisation rescale.* The initial $bold(W) tilde cal(N)(0, 0.01^2)$
  was too small: its $bold(W)^2$ entries sit at $10^(-4)$, a scale at
  which the gradient of either task or reconstruction loss is
  effectively zero. Rescaling the init to $cal(N)(0, 0.1^2)$ keeps
  $bold(W)^2$ on an $O(10^(-2))$ scale (still a small perturbation to
  the residual path) while carrying enough magnitude to propagate
  gradient during early training.

Either change alone is insufficient. Reconstruction without init
rescale was verified on a 10-epoch SageMaker run and left
$bold(W) approx 0$; init rescale without reconstruction would restore
the standard NOTEARS pressure but leave the gradient vanishing
problem.

=== Post-Patch Verification

A 10-epoch SageMaker run with both changes produces a non-trivial DAG
for the first time on this codebase:

- $bold(W)$ Frobenius: $bold(0.338)$ (init scale was 0.1)
- $abs(W)$-threshold sparsity at 0.01: 7.3%
- Acyclicity $h(bold(W)) = 0$ (valid DAG)
- Mean self-loop strength: $0.000$ (diagonal suppressed as intended)
- Top edges by $W_(i j)^2$: `var_23 -> var_13` (0.019),
  `var_9 -> var_13` (0.009), `var_15 -> var_11` (0.007)

The sparsity ratio is in the target range (paper-recommended 5--15%)
and the top edges show a consistent sink (`var_13`), which is the
kind of hub structure a latent DAG ought to exhibit.

=== Aggregate Task Metrics Are Unchanged

Patching the collapse does *not* change downstream task performance.
On the same SageMaker softmax-gate 10-epoch run, aggregate metrics
are within noise of the pre-patch softmax baseline:

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, center, center, center, center),
    stroke: 0.5pt,
    inset: 5pt,
    table.header(
      [*Run*], [*AUC*], [*F1 macro*], [*NDCG\@3*], [*MAE*]
    ),
    [Pre-patch local (softmax, $bold(W) approx 0$)],  [0.6729], [0.2009], [0.6814], [0.9598],
    [Post-patch SageMaker (softmax, $bold(W)$ learned)], [0.6719], [0.2042], [0.6875], [0.9597],
    [$Delta$],                                        [$-0.001$], [$+0.003$], [$+0.006$], [$0$],
  ),
  caption: [Task metric change after the W-collapse patch. Differences
  across AUC, F1 macro, NDCG\@3, and MAE are within the noise band
  observed across the 9-way fusion comparison (Finding 7). The
  structural bug was real, but its resolution does not, by itself,
  translate into task improvement.]
) <tab:W-patch-metrics>

=== Implication

The diagnostic finding is structural but the metric finding is null.
Two readings:

+ The causal expert has been contributing to prediction primarily
  through its `causal_encoder` MLP, which is a regular non-linear
  transform that does not require a meaningful $bold(W)$ to fit.
  Adding a meaningful $bold(W)$ changes the latent pathway
  ($bold(z)_"hat" = bold(z) + bold(z) bold(W)^2$) by a small amount,
  but the downstream encoder adapts and the ensemble's final
  prediction looks the same. The DAG is, in the current architecture,
  *decorative* rather than functionally used by the prediction.
+ For explainability claims that depend on the DAG (attribution,
  counterfactual probes, reason-code generation), this matters.
  Pre-patch the DAG was absent --- so any such claim built on
  `get_causal_graph()` was retrieving essentially noise at init
  scale. Post-patch the DAG exists and is structured, but is not
  itself routed into the prediction path.

This motivates a separate structural study (beyond this paper's
scope) that redefines the causal expert's role so the learned DAG
has a load-bearing route to prediction, not just to the expert's own
internal representation. Candidates include an attribution head that
consumes the DAG directly, a routing signal from the causal expert
to the per-task gate, and a counterfactual probe head. The W-
collapse patch reported here is a pre-condition for any of those
explorations --- without it, the DAG is not there to be routed.

= Discussion

== Practical Guidelines Summary

We distill six findings into five guidelines for practitioners:

+ *Gate selection depends on task-type mix, not on architecture.*
  Use softmax for heterogeneous task mixes (different loss types);
  use sigmoid for homogeneous tasks (all binary, or all regression).
  This is the single highest-impact design decision.

+ *Uncertainty weighting is necessary but not sufficient.*
  It normalizes loss scales --- without it, multiclass tasks are silently
  suppressed. But it does not prevent gradient contamination.
  Gate structure provides the actual isolation. Verify the implementation
  against @eq:uw-correct, especially the per-task $w_t$ and clamping.

+ *Do not mix task types in pre-gating groups.*
  GTE, task-group attention, and similar mechanisms assume intra-group
  homogeneity. Business-meaningful groups that mix binary and regression
  tasks will degrade the minority type.

+ *Use a composite checkpoint metric, not val loss.*
  When regression and classification tasks share a composite loss,
  regression improvement continuously pulls val loss downward while
  classification/ranking metrics saturate or regress.
  Define a type-weighted composite metric (Avg AUC + NDCG\@3 + normalized MAE)
  before training and checkpoint by it (Section 4.6).

+ *Gate entropy reveals architectural waste.*
  If all tasks show uniform attention-level entropy (ratio = 1.000),
  the attention aggregation is not performing routing --- it is averaging.
  Audit entropy ratios at both the extraction and attention levels before
  attributing performance gains to gating mechanisms.

== Limitations

*Synthetic data*: All experiments use a synthetic benchmark with controlled
noise profiles. Real production data may exhibit different gradient dynamics
due to label sparsity, class imbalance, and non-stationary distributions.
We plan to supplement with production results as they become available.

*Single expert basket*: Our findings are specific to PLE with 7 heterogeneous
experts. Homogeneous-expert PLE may exhibit different gate dynamics.

*Epoch budget and task-type interaction*: Finding 4 acknowledges that
10-epoch comparisons may be premature. Finding 6 extends this to 30 epochs
and confirms that additional epochs help regression but harm classification
and ranking. Cross-architecture comparisons (e.g., PLE vs.\ shared-bottom)
at 30 epochs remain pending.

*Single dataset scale*: While 1M customers is representative of
mid-sized financial institutions, the findings may not generalize to
internet-scale datasets (100M+ users) where task gradient dynamics differ.

== Relationship to Companion Papers

This paper complements two companion papers from the same project:
*Paper 1* (architecture and ablation) establishes the heterogeneous expert
PLE design and validates expert specialization via joint feature+expert ablation.
*Paper 2* (serving and ops) covers knowledge distillation, recommendation reason
generation, and regulatory compliance.
The present paper focuses specifically on *loss dynamics and gating behavior*
that emerged during the ablation study but warranted deeper analysis
than Paper 1's scope allowed.

= Conclusion

Scaling multi-task learning from 2--4 homogeneous tasks to 13 heterogeneous tasks
reveals dynamics that existing literature, evaluated primarily on
homogeneous setups, does not address.
The gate type choice --- softmax vs.\ sigmoid --- depends not on
architectural preference but on whether tasks share the same loss type.
Uncertainty weighting normalizes scales but does not isolate gradients.
Pre-gating mechanisms like GTE require type-homogeneous groups.
Training budgets must account for cosine restart cycles before structural
comparisons are meaningful.

Gate entropy analysis adds further precision to this picture:
PLE extraction-layer gating demonstrably specializes (entropy ratios 0.33--0.88),
with simple-pattern tasks concentrating on 1--2 experts and complex tasks
distributing weight across all 7.
Attention-level aggregation, however, collapses to uniform averaging
(entropy ratio 1.00 for all tasks), suggesting that this component
does not perform routing in practice.
Finally, composite val loss is an unreliable checkpoint signal once
regression tasks are present --- their continuous improvement masks
classification and ranking metric degradation,
and cosine learning-rate restarts amplify this divergence by disproportionately
disrupting the ranking-optimal parameter region.

These findings are not novel algorithms but practical diagnostics.
We hope they prevent other practitioners from re-discovering the same pitfalls
when scaling MTL to real-world heterogeneous task portfolios.

Finding 7 maps the search space for fusion augmentation on top of CGC
into three regions. *Representation-additive fusions* --- loss-level
adaTT, AdaTT-sp, M1 complement, ECEB, and MV BRP --- all propagate
residual-error signal into the primary-representation path and
uniformly degrade aggregate AUC, with magnitude scaling monotonically
in the invasiveness of the intervention. *Output-space boosting with
shared-expert gradient isolation* (BRP-detached) ties CGC on
aggregate AUC ($Delta = -0.0007$, best epoch $0.6736 > 0.6728$) while
improving F1 macro by $+0.007$ and NDCG\@3 by $+0.015$ and retaining
a $+$256% relative rescue on the hardest multiclass task.
*Training-time load-balancing regularisation* (NEAS) is the first
mechanism of the family to actually raise aggregate AUC
($Delta = +0.0011$), with a monotone-increasing trajectory and near-
uniform per-task lifts. The two positive recipes act on disjoint
axes (error correction in output space vs. expert-collapse prevention
at the gate) and are *not additive*: stacking them collapses NEAS's
AUC gain because the shared experts cannot simultaneously be
generalists (NEAS) and primary-supporting specialists (BRP-detached).
The practical guidance is per-objective: NEAS for aggregate AUC and
uniform cross-task robustness, BRP-detached for hard-task rescue.
Follow-up work will validate both recipes across seeds and datasets
and evaluate mechanisms that could resolve the NEAS $times$ BRP-
detached conflict (phased-schedule training, parameter-shared heads)
in a SageMaker setting where the compute budget allows systematic
sweeps.

// ============================================================
= Author Contributions

*Seonkyu Jeong* (PM / Lead Architect / Data Scientist):
Conceived the study, designed the ablation framework, identified all five findings,
wrote the manuscript. Led AI-augmented development methodology.

*Euncheol Sim*: Data pipeline, feature engineering, ablation execution.

*Youngchan Kim*: Model training, mathematical verification, loss weighting implementation.

All authors collaborated through Scrum sprints with rapid feedback cycles.

== Funding

This research received no external funding.
All costs --- including AI development tools, AWS SageMaker cloud training,
and operational expenses --- were borne by the first author's personal funds.
Development was conducted on a single desktop-grade GPU
(NVIDIA RTX 4070, 12GB VRAM).

// ============================================================

#bibliography("references.bib")
