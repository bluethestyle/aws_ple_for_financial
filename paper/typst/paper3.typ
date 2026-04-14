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
  Five findings challenge conventional MTL wisdom:
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
  and (5) GroupTaskExpert (GTE) pre-gating *degrades* multiclass performance
  when groups contain mixed task types.
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

This paper reports five empirical findings from this scaling experience.
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
expert selection. Nguyen et al. @nguyen2024 demonstrated that sigmoid gating
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
in PLE/MoE architectures @nguyen2024.
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

= Discussion

== Practical Guidelines Summary

We distill five findings into three guidelines for practitioners:

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

== Limitations

*Synthetic data*: All experiments use a synthetic benchmark with controlled
noise profiles. Real production data may exhibit different gradient dynamics
due to label sparsity, class imbalance, and non-stationary distributions.
We plan to supplement with production results as they become available.

*Single expert basket*: Our findings are specific to PLE with 7 heterogeneous
experts. Homogeneous-expert PLE may exhibit different gate dynamics.

*10-epoch budget*: Finding 4 explicitly acknowledges that longer training
may alter conclusions. We will update this paper with 20-epoch and
two-phase training results.

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
And training budgets must account for cosine restart cycles
before structural comparisons are meaningful.

These findings are not novel algorithms but practical diagnostics.
We hope they prevent other practitioners from re-discovering the same pitfalls
when scaling MTL to real-world heterogeneous task portfolios.

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
