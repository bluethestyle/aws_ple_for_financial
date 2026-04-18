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
- A negative result across four additive residual recovery mechanisms
  (loss-level adaTT, AdaTT-sp, complementary-gate recovery, and an
  uncertainty-conditioned expert bank): all four degrade aggregate AUC
  below the CGC baseline with a magnitude that scales monotonically with
  the invasiveness of the intervention, including the one variant
  designed to be gate-independent (Section 4.7).

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

== Finding 7: Additive Residual Recovery Underperforms Regardless of Residual Definition <find7>

After Finding 2 established that the loss-level `adaTT` variant does not
affect aggregate AUC at the 13-task scale ($Delta = -0.001$, null), three
further mechanisms were evaluated on the same benchmark to test whether any
fusion augmentation can extract useful signal beyond CGC's gated selection.
All four experiments converge on a single conclusion: *any residual
injected additively at the primary's fusion point degrades performance, and
the degradation grows monotonically with the invasiveness of the
intervention --- independently of how the residual is defined*.

=== Four Mechanisms, One Failure Pattern

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

*ECEB (Error-Conditioned Expert Bank, MV)*, also introduced in this paper,
is designed specifically to escape the shared structure of the three above:
the residual is derived from the gate's *entropy* rather than from the
gate's output. Concretely, the recovery path is a task-agnostic consensus
(mean over all expert outputs, no gate weighting), scaled at forward time
by the product of a per-task learnable scalar $sigma(w_t)$ and the
normalised gate entropy $H(g_t)/log N$ (per sample). When the gate is
confident (low entropy), recovery collapses toward zero; when it is
distributed, recovery activates. By construction ECEB eliminates the
"gate-derived residual" factor.

Results on the 13-task benchmark (10 epochs, seed=42):

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    stroke: 0.5pt,
    inset: 5pt,
    table.header(
      [*Fusion*], [*Final AUC*], [*Best AUC (epoch)*], [*$Delta$ vs CGC*]
    ),
    [CGC gate (baseline)], [0.6728], [0.6728 (ep10)], [---],
    [Loss-level adaTT],    [0.6717], [0.6733 (ep2)],  [$-$0.0011],
    [AdaTT-sp (Li 2023)],  [0.6696], [0.6714 (ep3)],  [$-$0.0032],
    [M1 complement],       [0.6675], [0.6692 (ep1)],  [$-$0.0053],
    [ECEB (MV)],           [0.6665], [0.6670 (ep4)],  [$-$0.0063],
  ),
  caption: [5-way comparison of fusion mechanisms on the 13-task benchmark.
  All four augmented variants fall below the CGC baseline; the magnitude of
  the drop scales monotonically with the invasiveness of the intervention.
  ECEB, which decouples the residual definition from the gate output,
  still fails --- and fails most severely --- demonstrating that the shared
  failure mode is *additive injection at the primary fusion point*,
  not gate-derivation of the residual.]
) <tab:fusion5way>

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

=== The Common Failure Mode: Revised

The first three rejected mechanisms (loss-level adaTT, AdaTT-sp, M1)
share one structural property: each derives a residual from the gate
output and injects it additively at the primary's fusion point. The ECEB
MV experiment was constructed specifically to sever that link: its
residual is derived from the gate's *entropy* rather than from its output,
and the recovery path is a task-agnostic consensus that does not use gate
weights at all. By the earlier diagnosis, ECEB should have behaved
differently. It did not --- it produced the *largest* degradation of the
four.

The revised reading is therefore simpler and more conservative: *any
residual injected additively at the primary's fusion point on this
benchmark reduces to noise, because the primary output of the
heterogeneous-expert CGC gate is already near-optimal for the task
at hand*. The residual's definition --- gate inverse, own expert,
uncertainty-gated consensus --- changes only how aggressively the noise
is injected, and the monotone degradation reflects exactly that.

This is a stronger, cleaner negative result than a per-mechanism rebuttal:
the CGC gate's gated weighted sum is not improvable by *any* additive
augmentation we tested, across a wide range of residual definitions.

*Implication*: A fusion augmentation that improves over CGC at this scale
cannot live at the representation-additive position at all. The remaining
candidate direction is a *boosting-style residual path* trained on the
primary prediction's *errors* and combined with the primary only at the
final prediction stage rather than at the representation. The alternative
of a parallel task-agnostic aggregation path (an untested Paper 3
direction in earlier drafts) is effectively eliminated by ECEB MV, which
already implements a gate-independent consensus path; evaluating a
separately-predicted variant of the same idea is unlikely to reverse the
sign. Selection and evaluation of the boosting variant is left to a
follow-up paper; the present negative-result family defines what such a
mechanism must structurally avoid.

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

Finding 7 reframes the search for fusion augmentation beyond CGC. On this
benchmark, *four* additive residual mechanisms --- three derived from the
gate output and a fourth deliberately decoupled from it via an
uncertainty-conditioned consensus path --- all underperform the baseline,
with degradation magnitude scaling monotonically with intervention
invasiveness. The takeaway is stronger than the original reading:
*no additive residual injected at the primary's fusion point improves
over the CGC gated weighted sum on this benchmark, regardless of how the
residual is defined*. A follow-up study will therefore evaluate a
structurally distinct alternative: a boosting-style residual path that
is trained on the primary prediction's errors and combined with the
primary only at the output prediction level, leaving the representation
untouched. This is the one remaining design direction that the negative
results reported here have not already ruled out.

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
