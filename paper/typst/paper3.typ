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
  Eleven findings organize into three themes.
  *Loss dynamics and gating* (Findings 1--6): an uncertainty-weighting implementation bug silently suppresses minority-type tasks (+0.018 NDCG\@3 when fixed); softmax gating outperforms sigmoid for heterogeneous task mixes, *reversing* the homogeneous-setting advantage; learned uncertainty weights converge identically across architectures; 10-epoch budgets may be insufficient for structural comparisons; GroupTaskExpert pre-gating degrades multiclass performance in mixed-type groups; gate entropy analysis shows extraction-layer specialization (entropy ratios 0.33--0.88) while attention-level aggregation collapses to uniform averaging (ratio 1.00), and composite val-loss is an unreliable checkpoint signal once regression tasks are present.
  *Fusion augmentation trade-offs* (Finding 7): a 9-way comparison isolates two positive recipes on disjoint axes --- output-space boosting with gradient isolation (BRP-detached, hard-task rescue) and inverse-gate auxiliary supervision (NEAS, aggregate-AUC gain) --- that do *not* compose additively.
  *Causal expert reinterpretation* (Findings 8--11): the causal expert's adjacency matrix $bold(W)$ collapses to zero in every trained checkpoint examined, rendering its forward equivalent to a plain MLP; a two-part patch (NOTEARS reconstruction loss + initialisation rescale) restores DAG learning at zero task-metric cost; routing the functional DAG into consumable outputs yields a Causal Explainability Head (CEH, per-sample attribution) and a Causal Guardrail (CG, z-space-Mahalanobis OOD detection at 100% TPR / 5% FPR on three synthetic probes); and W-amplification via init $0.1 arrow 0.3$ + $lambda_"recon" 0.5 arrow 2.0$ grows the adjacency matrix $14 times$ in Frobenius norm with zero primary-task cost, establishing that the "decorative DAG" is a training-choice artefact, not an architectural constraint.
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

This paper reports eleven empirical findings from this scaling
experience, organised into three themes: loss dynamics and gating
(Findings 1--6), fusion augmentation trade-offs (Finding 7), and
causal expert reinterpretation (Findings 8--11). We make no claims
of state-of-the-art performance; instead, we document *phenomena
and practical guidelines* that emerge when MTL is pushed beyond the
homogeneous-task regime. Our contributions:

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
- Two Axis-3 candidates that *route* the now-functional DAG into
  consumable outputs:

  Causal Explainability Head (CEH), a small MLP
  mapping the causal expert's output to a per-sample per-feature
  attribution vector trained via MSE against gradient $times$ input.
  MV result preserves the primary AUC within noise ($+0.0015$ over
  the post-patch softmax baseline), marginally strengthens the DAG
  itself ($bold(W)$ Frobenius $0.338 arrow 0.366$, sparse edges
  $7.3% arrow 8.5%$), and produces a live attribution vector for
  downstream audit-log persistence. A post-hoc quality evaluation
  revealed that under the raw grad $times$ input target the head
  collapsed to a near-global importance vector (between/within-sample
  variance ratio $0.055$, top-10 feature overlap $0.791$ across
  samples). A minimum-intervention iteration (v2, "demeaned target")
  subtracts the batch mean from the supervision signal and restores
  per-sample discrimination without disturbing the primary task
  (variance ratio $0.055 arrow 0.719$, top-10 overlap
  $0.791 arrow 0.281$, primary AUC unchanged within noise,
  Section 4.9.4). Audit-log integration and cross-dataset reproduction
  remain deferred.

  Causal Guardrail (CG), a per-prediction reliability flag derived
  from the causal expert's latent $bold(z)$. A W-reconstruction
  formulation (CG v1) fails at chance-level discrimination because
  the learned $bold(W)$ is too small; a z-space Mahalanobis
  formulation (CG v2) detects three types of synthetic OOD at $100%$
  TPR with $5%$ FPR (Section 4.10). A follow-up W-amplification
  experiment (W init $0.1 arrow 0.3$, $lambda_"recon"
  0.5 arrow 2.0$) grows the adjacency matrix $14 times$ in
  Frobenius norm and from $8.5%$ to $59.5%$ active edges with zero
  primary-task cost, partially fixing CG v1 but leaving
  latent-space CG v2 dominant (Section 4.11). Finding 11's settled
  result: the "decorative DAG" from Finding 8 is a training-choice
  artefact, not an architectural constraint.

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

== Finding 9: Causal Explainability Head (CEH) — First Axis-3 Attempt <find9>

As the first of several Axis-3 candidates that re-wire the causal
expert's role (CEH / CG / CTGR / CRCG / CCP), CEH adds a per-sample
per-feature attribution vector on top of the causal expert. The
motivation is explicit: now that Finding 8 gives the expert a real
DAG, the DAG needs to leave the expert's internals and reach a
consumer. CEH is the shortest path from "DAG exists" to "DAG has a
per-prediction output that an audit log can persist."

=== Design

A small MLP head maps the expert's 64-dim output back to an
$"input_dim"$-wide attribution vector, and is trained via MSE to
align with the gradient $times$ input saliency baseline of the
expert's own scalar output w.r.t. the input:

$ cal(L)_"attr" = "mean"(abs("head"("output") - (gradient_x "output".sum() dot.o x))^2) $

Gradient $times$ input is computed with one extra forward pass on a
cloned, `requires_grad=True` copy of the input, so the main forward
graph is untouched. Cost is $approx$ 14% extra compute inside the
causal expert (about $1 slash 7$ of shared-expert time). Primary
prediction path is unchanged; CEH is a training-time regulariser
plus an inference-time side output.

Inference: `expert._last_attribution` exposes a per-sample vector of
length $"input_dim"$. The pipeline-side integration with the audit
log (HMAC-signed persistence, SR 11-7 MRM) is Paper 2 v2 scope.

=== MV Result (SageMaker teacher_full + CEH, 10 epochs, softmax gate)

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    stroke: 0.5pt,
    inset: 5pt,
    table.header(
      [*Metric*], [*Pre-patch (W ≈ 0)*], [*Post-patch (no CEH)*], [*Post-patch + CEH*]
    ),
    [Primary AUC],                [0.6729], [0.6719], [*0.6734*],
    [F1 macro],                   [0.2009], [0.2042], [0.1994],
    [NDCG\@3],                    [0.6814], [0.6875], [0.6842],
    [MAE],                        [0.9598], [0.9597], [0.9609],
    [$bold(W)$ Frobenius],        [0.0001], [0.338],  [*0.366*],
    [$abs(W) > 0.01$ sparse edges], [0%], [7.3%], [*8.5%*],
    [Attribution head trained],   [n/a],   [n/a],    [*yes*],
  ),
  caption: [CEH MV result. Primary AUC is preserved within noise;
  $bold(W)$ Frobenius and sparse-edge share are both slightly higher
  than the post-patch no-CEH run, suggesting the attribution-head
  gradient provides an additional structural signal that reinforces
  the DAG. Attribution head training was verified by post-hoc
  inspection of the head's weights and biases (non-zero biases
  starting from zero-init).]
) <tab:ceh-mv>

The attribution head's layer-3 bias (init = $bold(0)$) moved to
$||bold(b)|| = 0.03 plus.minus 0.04$ per element and layer-0 bias to
$0.08 plus.minus 0.08$, which is straightforward evidence that the
MSE alignment loss was contributing gradient to the head.

=== What the MV Does Not Verify

CEH MV confirms that:

+ The head trains without disrupting the primary pathway.
+ The W-collapse patch (Finding 8) remains intact under the extra
  attribution-training signal --- in fact the DAG's Frobenius norm
  and sparse-edge count both increase marginally.
+ The per-prediction attribution vector exists and can be consumed
  downstream.

CEH MV does *not* yet verify:

- *Attribution quality.* The head fits the gradient $times$ input
  target by construction; whether the learned output carries
  per-sample signal or merely reproduces a global importance pattern
  required a dedicated post-hoc evaluation. That evaluation
  (Section 4.9.3) showed a near-global collapse under the raw target
  and motivated the v2 iteration (Section 4.9.4), which resolved the
  collapse.
- *Audit-log utility.* The per-sample vector is now available but
  has not yet been routed into an HMAC-signed, append-only audit log
  record that regulators can query. That integration is Paper 2 v2
  scope.
- *Downstream metric impact.* CEH's small AUC lift is $+0.0015$
  over post-patch softmax, within the noise band of the 9-way
  comparison. We do not claim it as a significant improvement.

=== Attribution Quality Evaluation (Post-hoc)

Because the MV alone only shows the head *trains*, we ran a
dedicated post-hoc evaluation on 5,000 validation samples to ask
whether the trained head produces meaningful per-sample explanations
or merely a smoothed global pattern. Four measurements:

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, left, left),
    stroke: 0.5pt,
    [*Measurement*], [*Value*], [*Interpretation*],
    [Spearman corr. (CEH vs. grad $times$ input)], [mean $0.259$, median $0.252$], [Partial fit to training target],
    [Between-sample / within-sample variance], [$0.055$], [Attribution varies much more within a sample than across samples],
    [Mean top-10 feature overlap across samples], [$0.791$], [Different samples share $~80%$ of their top features],
    [Stability under input noise ($sigma = 0.05$)], [Spearman $0.985$], [Very stable (trivially consistent with near-global output)],
  ),
  caption: [CEH attribution quality measurements on the post-patch
  CEH checkpoint. Taken together, the low between-over-within ratio
  ($0.055$) and high top-10 overlap ($0.791$) indicate the head has
  largely collapsed to a global importance vector with only small
  per-sample perturbations --- the opposite of what a per-prediction
  attribution head should produce.],
)<tbl:ceh-quality>

Per-group attribution mass corroborates the flatness. CEH allocates
$32.1%$ to txn\_behavior versus $44.7%$ under grad $times$ input on
the same samples, and overweights product\_holdings ($12.6%$ vs.
$5.5%$) and product\_hierarchy ($11.3%$ vs. $3.8%$). The learned
distribution is demonstrably flatter than its own training target.

*Interpretation.* The head fits its target only weakly
($rho approx 0.26$) and discards the per-sample component of that
target almost entirely. The most likely mechanism: the target itself
---  grad $times$ input of the causal encoder's summed output --- has a
large sample-invariant component, and a 64-hidden single-layer MLP
can capture that global component with low loss while ignoring the
sample-specific residual. The head is not broken; the target is too
flat for a thin MLP to be forced into per-sample discrimination.

*What this rules in/out.*

- Rules in: infrastructure path (Finding 9 MV) is correct --- a
  functional DAG feeds an attribution consumer without destabilising
  primary prediction or the DAG itself.
- Rules out: the current target design (grad $times$ input of
  causal-encoder output sum) is *not* sufficient to produce a
  regulator-usable per-sample explanation. Additional Axis-3 candidates
  that depend on per-sample attribution quality (CRCG in particular)
  should not be evaluated against this baseline without target refinement.

*Target refinement candidates.*

+ *Demeaned target:* subtract the batch mean of grad $times$ input
  before using it as supervision. Forces the head to learn per-sample
  deviation from the global pattern rather than re-learning the
  pattern itself. Smallest code change; tests the "target is too flat"
  hypothesis directly. *Run; see Section 4.9.4 below.*
+ *Primary-prediction target:* replace grad of the causal encoder
  output with grad of a specific task logit (e.g., churn\_signal).
  Aligns the attribution with what downstream consumers actually ask
  explanations for, at the cost of per-task heads.
+ *Larger / deeper head:* doubled hidden dim and an extra layer, to
  test whether the 64-hidden bottleneck is the limiting factor rather
  than the target.
+ *DAG-path target:* use the learned $bold(W)$ to construct
  feature-to-output influence paths as supervision, shifting from
  local gradient-based to structural-graph-based attribution.

=== Iteration v2: Demeaned Target Resolves the Collapse

We ran candidate 1 (demeaned target) as a minimal-intervention test
of the "target is too flat" hypothesis. Config flag
`ceh.target_mode: "demeaned"` subtracts the per-feature batch mean
from the grad $times$ input supervision before MSE; everything else
identical to the v1 MV run (same architecture, same hyperparameters,
same data, same 10 epochs on SageMaker g4dn.xlarge). The same
post-hoc evaluation on 5,000 validation samples produces:

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, left, left, left),
    stroke: 0.5pt,
    [*Measurement*], [*Raw (v1)*], [*Demeaned (v2)*], [*Change*],
    [Between-sample / within-sample variance], [$0.055$], [$0.719$], [$13 times$ larger],
    [Top-10 feature overlap across samples], [$0.791$], [$0.281$], [$65%$ smaller],
    [Stability under input noise ($sigma = 0.05$)], [Spearman $0.985$], [Spearman $0.953$], [Still stable],
    [Primary AUC (churn\_signal)], [$0.6866$], [$0.6870$], [Within noise],
  ),
  caption: [CEH attribution quality, v1 (raw grad $times$ input
  target) vs. v2 (demeaned target). The two discriminative-power
  measurements move by more than an order of magnitude while primary
  AUC is unchanged --- the head now learns per-sample deviation
  rather than reproducing a global importance vector, at no cost to
  the downstream task.],
)<tbl:ceh-v1-v2>

The result confirms the hypothesis from Section 4.9.3: the collapse
was a training-target artefact, not a head-capacity or
architectural limitation. Minimum intervention (three lines plus a
config flag) restores per-sample discrimination.

Per-group attribution mass also rebalances substantially under the
demeaned target. txn\_behavior drops from $32.1%$ to $18.3%$ and
gmm\_clustering from $28.5%$ to $21.4%$, while product\_hierarchy
jumps from $11.3%$ to $30.3%$ and product\_holdings from $12.6%$ to
$21.3%$. The demeaned head prefers feature groups that carry per-
sample distinguishing signal (product taxonomy) over the globally
high-variance groups that dominated the raw head.

*Caveats.*

- Spearman correlation against *raw* grad $times$ input falls from
  $0.259$ to $0.096$ between v1 and v2. This is expected and not a
  regression: the v2 head is supervised against demeaned
  grad $times$ input, so the raw version is no longer its training
  target. Quality is measured by the target-independent metrics
  in @tbl:ceh-v1-v2 (variance ratio, top-K overlap, stability).
- Per-sample discrimination is necessary but not sufficient for
  regulator-usable explanations. Human evaluation on sample cases,
  alignment with domain expectations, and audit-log integration
  remain outside this paper's scope.
- We report v2 as a single-seed 10-epoch run on Santander. Cross-seed
  stability and cross-dataset reproduction are not verified here.

*Next iteration directions (not run).* Primary-task-gradient target
remains the natural next test: replacing the causal-encoder output
sum with a specific task logit aligns the attribution with what
downstream consumers actually ask about ("why did this customer
receive a high churn score?"). The demeaned infrastructure stays,
only the scalar the gradient is taken against changes. Larger head
and DAG-path targets remain as lower-priority alternatives.

=== Why CEH First (Not CG / CTGR / CRCG / CCP)

Of the five Axis-3 candidates, CEH has the smallest structural
footprint: a single MLP head with an MSE supervision signal, no
change to task routing (vs. CTGR), no serving-time path branching
(vs. CG), no cross-module wiring (vs. CRCG tying into Paper 2's
reason generator). This lets us confirm the basic premise --- that
downstream consumers can extract something meaningful from the now-
functional DAG --- with the cleanest possible test before committing
to a heavier redesign. Follow-up candidates are evaluated against
CEH's baseline rather than against no-causal-routing-at-all.

== Finding 10: Causal Guardrail (CG) --- Second Axis-3 Attempt <find10>

Where CEH answers "why did the model recommend this?", CG
(Causal Guardrail) answers the adjacent question "can we trust
this recommendation?" --- a per-prediction reliability signal
that routes suspicious inputs to a fallback or to human review
rather than silently passing them through. SR 11-7 "known
limitations" reporting and EU AI Act Art. 9 risk-management both
require a runtime mechanism for this; the natural hook on the
causal expert is its DAG structure.

=== MV Formulation v1 (W-Reconstruction) Fails

The first formulation reused the NOTEARS reconstruction residual
from Finding 8. Training optimises
$||bold(z) - bold(z) bold(W)^2||_F^2$ to keep $bold(W)$ out of the
zero saddle; at serving time the per-sample version
$||bold(z)_i - bold(z)_i bold(W)^2||^2 / ||bold(z)_i||^2$ is a
direct measure of how well the *learned DAG* reconstructs the
input's compressed causal state. A well-fit input reconstructs
cleanly; an OOD input should not.

We evaluated on the teacher_ceh_demeaned checkpoint with 5,000
validation samples plus three synthetic OOD probes (uniform random
in [-3,3], column-permuted, extreme-tail percentiles). All
distributions concentrated within a narrow window near 1.0
(in-dist median 0.9995, p99 1.0055; OOD medians 0.9998--0.9972).
At the p95 threshold calibrated on the in-distribution set, TPR on
OOD probes was 6.8% / 8.1% / 0% --- effectively chance-level versus
the 5% FPR baseline.

The failure mechanism is the same "decorative DAG" observation
from Finding 8: the learned $bold(W)$ is small enough ($||bold(W)||_F
approx 0.36$ on an $8 times 8$ matrix, so $bold(W)^2$ is
$O(0.13)$ in Frobenius) that $bold(z) bold(W)^2$ barely perturbs
$bold(z)$. The residual ratio $approx 1.0$ regardless of whether
the input fits the DAG, so the signal is degenerate.

=== MV Formulation v2 (z-Space Mahalanobis) Works

Since $bold(W)$ itself is too weak to drive discrimination, we
tested whether the *causal latent* $bold(z)$ carries
distributional signal independently. Let $bold(mu), bold(sigma)$ be
the per-dimension mean and standard deviation of $bold(z)$ over the
in-distribution batch; the score is the standardised-sum-of-squares
$d_i = sum_j ((z_(i,j) - mu_j) / sigma_j)^2$.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    stroke: 0.5pt,
    [*Probe*], [*median*], [*p95*], [*p99*], [*max*],
    [In-distribution], [$23.6$], [$62.8$], [$268.1$], [$983.6$],
    [Uniform random], [$749.5$], [$1200.9$], [$1458.5$], [$1936.7$],
    [Column-permuted], [$537.4$], [$1076.0$], [$1336.1$], [$1873.9$],
    [Extreme-tail], [$479.3$], [$479.3$], [$479.3$], [$479.3$],
  ),
  caption: [z-space Mahalanobis CG v2 score distributions on 5,000
  validation samples and three synthetic OOD probes. Every OOD
  median lies above the in-distribution $"p99"$, giving near-perfect
  separation.],
)<tbl:cg-v2-dists>

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, right, right),
    stroke: 0.5pt,
    [*Probe*], [*TPR @ ID p95*], [*FPR @ ID p95*],
    [Uniform random], [$100.0%$], [$5.0%$],
    [Column-permuted], [$100.0%$], [$5.0%$],
    [Extreme-tail], [$100.0%$], [$5.0%$],
  ),
  caption: [OOD detection rates at the recommended CG v2 threshold
  (in-distribution $"p95" = 62.8$). All three probe types flag at the
  expected false-positive rate.],
)<tbl:cg-v2-rates>

CG is therefore operational via the causal latent even though it
fails via the DAG weights. The latent carries structure the
W-reconstruction does not expose. Serving integration: a caller
computes $bold(mu), bold(sigma)$ once from a reference population,
caches them, and at inference time thresholds the per-sample score
to decide pass-through vs. fallback.

=== What CG MV Verifies / Does Not

Verified:
+ The z-space formulation detects three qualitatively different OOD
  probes at effectively $100%$ TPR with $5%$ FPR --- the primitive
  works.
+ v1 vs v2 comparison isolates *where* the signal actually lives:
  the causal latent, not the learned DAG weights. Reinforces the
  Finding-8 observation that $bold(W)$ is structurally present but
  under-utilised by the current architecture.
+ Zero training cost: CG is a post-hoc analysis on the existing
  teacher_ceh_demeaned checkpoint. No new SageMaker job.

Not yet verified:
- *Real-world OOD*: the three probes are synthetic. A production
  guardrail also needs to fire on realistic distribution drift
  (temporal shift, subgroup imbalance, adversarial perturbation).
  These are Paper 2 monitoring responsibilities, not tested here.
- *Downstream metric impact*: CG produces a flag; we have not yet
  measured whether routing flagged predictions to a fallback
  improves end-to-end task metrics or calibration. Demands an
  integration with the 3-layer fallback router.
- *Threshold drift*: $bold(mu), bold(sigma)$ computed on a
  reference batch will need periodic recalibration as the input
  distribution evolves. Not tested here.

=== Implications for Remaining Axis-3 Candidates

The v1/v2 contrast clarifies what to expect for CTGR and CRCG.
Both depend on the *learned causal structure* ($bold(W)$) being
informative in the current architecture, not merely present. The
Finding 10 experiment is direct evidence that $bold(W)$ is not
currently strong enough for structural uses: CTGR's routing and
CRCG's reason paths would inherit the same weak signal that doomed
CG v1. Either W needs to be amplified (larger init, stronger
recon lambda, or DAG-routed residual path) before those candidates
are worth running, or CTGR/CRCG should be redesigned to draw from
the latent rather than the weights --- mirroring the v1$arrow.r$v2
pivot here. Finding 11 (below) runs the amplification experiment
and produces a partially mixed answer.

== Finding 11: W-Amplification Test <find11>

Finding 10 identified two hypotheses for why CG v1 fails: (a) the
learned $bold(W)$ is simply too small to drive meaningful
reconstruction, or (b) the $||bold(z) - bold(z) bold(W)^2||^2$
formulation is structurally limited regardless of $bold(W)$'s
magnitude. We tested (a) by re-training one scenario
(teacher_ceh_w_amp) with two amplification knobs: $bold(W)$ init
scale $0.1 arrow.r 0.3$ and $lambda_"recon"$ $0.5 arrow.r 2.0$. All
other hyperparameters identical to teacher_ceh_demeaned.

=== Training-Side Result: W Amplifies, Primary Unharmed

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right),
    stroke: 0.5pt,
    [*Metric*], [*Baseline*], [*W-amp*], [*Change*],
    [$||bold(W)||_F$], [$0.363$], [$5.028$], [$13.9 times$],
    [Active edges ($|W| > 0.01$)], [$8.5%$], [$59.5%$], [$7.0 times$],
    [Max $|W_(i j)|$], [$0.11$], [$0.77$], [$7.0 times$],
    [Primary AUC (churn\_signal)], [$0.6870$], [$0.6865$], [within noise],
    [Loss], [$25.62$], [$25.61$], [within noise],
  ),
  caption: [W-amplification training-side outcome. The learned
  adjacency matrix grows by an order of magnitude in every
  structural measure while primary task metrics are preserved.
  Finding 8's "decorative DAG" observation is directly and
  cheaply reversed.],
)<tbl:wamp-training>

The amplification is free in task-metric terms: the larger $bold(W)$
pushes $bold(W)^2$ from Frobenius $0.13$ to $2.19$, so the SCM
residual $bold(z) + bold(z) bold(W)^2$ now contributes a
meaningfully non-trivial perturbation, yet primary AUC shifts by
only $0.0005$ (within noise). The decorative-DAG regime was a
training choice (too-small init + too-weak recon term), not an
architectural constraint.

=== CG v1 Improves but Remains Architecturally Limited

With a 14$times$ larger $bold(W)$, CG v1's discriminative power
does increase but does not approach v2's ceiling:

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, right, right),
    stroke: 0.5pt,
    [*Probe*], [*Baseline TPR @ ID p95*], [*W-amp TPR @ ID p95*],
    [Uniform random], [$6.8%$], [$22.7%$],
    [Column-permuted], [$8.1%$], [$18.6%$],
    [Extreme-tail], [$0.0%$], [$0.0%$],
  ),
  caption: [CG v1 OOD detection before and after W-amplification
  (FPR fixed at $5%$). Discrimination improves from chance to
  weak-but-nonzero for two probe types; extreme-tail stays at zero
  regardless, exposing a structural limit.],
)<tbl:wamp-v1>

The extreme-tail result is the key signal. That probe sets every
sample to the same 99th-percentile vector, so every sample has an
identical $bold(z)$ and therefore an identical residual ---
$||bold(z) - bold(z) bold(W)^2||$ is a point estimate, unable to
discriminate by construction. No amount of W amplification fixes
this because the formulation is a *per-sample* measure that offers
no vantage point on the *distribution* of inputs. CG v2's z-space
Mahalanobis, by contrast, is explicitly a distance from a reference
distribution, so it handles the degenerate-OOD case trivially
($100%$ TPR).

=== CG v2 Unaffected

v2's discrimination is unchanged after amplification: $100%$ TPR
across the three probes at the recommended $"p95"$ threshold. The
z-space pathway does not depend on $bold(W)$'s magnitude;
amplification neither helps nor hurts it.

=== What Finding 11 Settles / Leaves Open

Settled:
+ $bold(W)$ is *not* architecturally doomed. With init $0.3$ +
  $lambda_"recon" = 2.0$, it grows to a non-trivial sparse structure
  in $10$ epochs, with zero primary-task cost.
+ CG v1's failure in Finding 10 was *partly* due to small $bold(W)$
  (now contributes ~$15$--$20$ TPR points) and *partly* due to a
  formulation ceiling (extreme-tail probe exposes a structural
  limit).
+ Latent-space CG (v2) remains dominant for OOD detection; W-based
  guardrails are a supplement at best, not a replacement.

Open:
- Does amplified $bold(W)$ unlock CTGR and CRCG, or do those
  candidates hit similar structural ceilings? Finding 11 answers
  only the CG question directly; the training-side amplification
  means CTGR/CRCG can now at least be *tried* without the Finding
  10 preconditions blocking them.
- Does an even more aggressive amplification (init $0.5$,
  $lambda_"recon" = 5$, removed sparsity) push v1 to a usable level
  or crash primary AUC? Not tested; the $0.3 / 2.0$ setting was
  chosen as a single-job minimum-intervention.
- Does latent-space CG's $100%$ TPR survive *realistic* drift
  (temporal, subgroup, adversarial) rather than synthetic probes?
  Paper 2 monitoring scope, not tested here.

=== Updated Recommendation for Remaining Axis-3 Candidates

Finding 10 recommended either amplifying $bold(W)$ or redesigning
the siblings around the latent. Finding 11 chooses the first
option and shows it partially works: $bold(W)$ is now meaningful,
but the latent-based formulation still ceiling-beats a W-based one
when both are available. The practical recommendation for CTGR /
CRCG / CCP is therefore:

+ Run them on a W-amplified teacher, so W is *capable* of carrying
  structural information.
+ Simultaneously evaluate a latent-based alternative for each
  candidate. In every case where both are measurable, treat the
  latent version as the baseline to beat rather than assuming the
  W-based version is preferable.

= Discussion

== Practical Guidelines Summary

We distill the eleven findings into nine guidelines for practitioners,
grouped by the three themes.

=== Loss Dynamics and Gating (Findings 1--6)

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

=== Fusion Augmentation (Finding 7)

+ *Match the fusion recipe to the objective; do not stack them.*
  For aggregate-AUC gains with uniform per-task lift, prefer inverse-gate
  auxiliary supervision (NEAS). For hard-task rescue at the cost of
  aggregate AUC parity, prefer output-space boosting with shared-expert
  gradient isolation (BRP-detached). The two recipes operate on disjoint
  axes --- gate-level load balancing vs. output-level error correction ---
  and stacking them collapses the aggregate-AUC gain because shared
  experts cannot simultaneously be generalists (NEAS) and primary-
  supporting specialists (BRP-detached).

=== Causal Expert Reinterpretation (Findings 8--11)

+ *Verify the causal DAG is actually learning before consuming it.*
  A $bold(W)$ that drifts to zero during training is a silent failure
  mode: NOTEARS acyclicity and sparsity penalties are satisfied
  trivially at $bold(W) = 0$, and an SCM residual forward
  ($bold(z) + bold(z) bold(W)^2$) still propagates the primary signal
  through $bold(z)$. Monitor Frobenius norm, active-edge count, and
  acyclicity value $h(bold(W))$ at training time, not just final
  task metrics.

+ *Decorative DAG is a training choice, not an architectural constraint.*
  With init scale $0.1$ and reconstruction-loss weight $0.5$, $bold(W)$
  lands at Frobenius ~$0.36$ on an $8 times 8$ matrix --- "present but
  too weak to matter". Raising init to $0.3$ and $lambda_"recon"$ to
  $2.0$ amplifies the learned matrix $14 times$ with zero primary-task
  cost. Downstream uses that rely on $bold(W)$ carrying structural
  information should be evaluated against an amplified training run,
  not the default.

+ *Check attribution-head target design for sample-variance collapse.*
  A target with large sample-invariant content (e.g., raw gradient $times$
  input of an aggregate output) lets a thin MLP re-learn the global
  pattern and ignore the per-sample residual. Verify that attribution
  varies sample-to-sample (between/within variance ratio, top-$K$
  overlap across samples) before claiming per-prediction
  explainability. A minimal fix --- subtract the batch mean of the
  target --- forces per-sample deviation learning and (in our case)
  raised the variance ratio from $0.055$ to $0.719$ at no task cost.

+ *Prefer latent-space over weight-space formulations for guardrails.*
  A per-sample W-reconstruction residual has no distribution awareness:
  if every sample produces an identical latent, the residual is a
  point and the guardrail cannot fire. Mahalanobis-style distance
  from a cached in-distribution reference batch is distribution-aware
  by construction and hit 100% OOD TPR at 5% FPR across three probes
  in our setup. Default to the latent formulation; use the
  weight-space version only as a supplement.

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
Findings 8--11 in particular are reported on a single seed and a
single dataset (Santander); cross-seed stability and cross-dataset
reproduction are deferred.

*Synthetic OOD probes for CG (Finding 10--11)*: The Causal Guardrail
evaluation uses three synthetic out-of-distribution probes (uniform
random, column-permuted, extreme-tail). Real-world distribution drift
(temporal shift, subgroup imbalance, adversarial perturbation) is
expected to differ in both structure and difficulty. CG v2's 100%
TPR at 5% FPR on synthetic probes is a sanity-check ceiling, not a
production-ready number.

*Attribution meaningfulness not human-evaluated*: CEH v2 (Finding 9)
produces per-sample attributions that discriminate across samples
(variance ratio 0.719, top-10 overlap 0.281), but we have not
assessed whether the resulting top-$K$ features agree with domain-
expert expectations or with alternative attribution methods
(Integrated Gradients, DAG-path traversal). A human-evaluation pass
is future work.

*Remaining Axis-3 candidates (CTGR, CRCG, CCP)*: Findings 10--11
establish the precondition for the three unexplored candidates ---
$bold(W)$ is trainable --- but the candidates themselves are not yet
evaluated. CG's v1$arrow.r$v2 pivot also introduces a concrete
prediction (latent-based formulations should be evaluated alongside
weight-based ones), which has not been tested on the remaining
candidates.

== Relationship to Companion Papers

This paper complements two companion papers from the same project:
*Paper 1* (architecture and ablation) establishes the heterogeneous
expert PLE design and validates expert specialization via joint
feature+expert ablation. *Paper 2* (serving and ops) covers knowledge
distillation, recommendation reason generation, and regulatory
compliance. The present paper focuses specifically on *loss dynamics
and gating behavior* (Findings 1--6), *fusion augmentation trade-offs*
(Finding 7), and *causal expert reinterpretation* (Findings 8--11)
that emerged during the ablation study but warranted deeper analysis
than Paper 1's scope allowed.

Findings 9--11 are directly consumed by Paper 2's v2 audit
infrastructure: the CEH attribution vector feeds
`AuditLogger.log_attribution`, and the CG z-space coherence score
feeds `AuditLogger.log_guardrail`, producing a complete
HMAC-signed hash-chained per-prediction record (explanation +
reliability). The training-time machinery that makes the DAG
learnable (Finding 8) and the amplification knobs that make $bold(W)$
structurally meaningful (Finding 11) are therefore not standalone
curiosities in Paper 3 but prerequisites for the regulator-usable
audit surface delivered in Paper 2 v2.

= Conclusion

Scaling multi-task learning from 2--4 homogeneous tasks to 13
heterogeneous tasks surfaces three families of dynamics that
existing literature, evaluated primarily on homogeneous setups,
does not address.

*Loss dynamics and gating* (Findings 1--6): gate type choice ---
softmax vs.\ sigmoid --- depends on whether tasks share a loss
type, not on architectural preference; uncertainty weighting
normalizes scales but does not isolate gradients and converges
identically across architectures; pre-gating mechanisms like GTE
require type-homogeneous groups; training budgets must absorb cosine
restart cycles before structural comparisons are meaningful; gate
entropy analysis shows that extraction-layer gating specializes
(entropy ratios $0.33$--$0.88$) while attention-level aggregation
collapses to uniform averaging (ratio $1.00$); and composite val
loss is an unreliable checkpoint signal once regression tasks are
present, because their continuous improvement masks classification
and ranking degradation.

*Fusion augmentation trade-offs* (Finding 7): a 9-way comparison of
representation-level and output-level fusions on top of CGC maps the
design space into three regions. Representation-additive fusions ---
loss-level adaTT, AdaTT-sp, M1 complement, ECEB, and MV BRP ---
propagate residual-error signal into the primary-representation path
and uniformly degrade aggregate AUC, with magnitude scaling in the
invasiveness of the intervention. Output-space boosting with shared-
expert gradient isolation (BRP-detached) ties CGC on aggregate AUC
while lifting F1 macro and NDCG\@3 and retaining a +256% relative
rescue on the hardest multiclass task. Training-time load-balancing
regularisation (NEAS) is the first mechanism of the family to
actually raise aggregate AUC ($Delta = +0.0011$) with near-uniform
per-task lifts. The two positive recipes act on disjoint axes and
are not additive --- stacking them collapses NEAS's AUC gain because
the shared experts cannot simultaneously be generalists (NEAS) and
primary-supporting specialists (BRP-detached). The practical
guidance is per-objective.

*Causal expert reinterpretation* (Findings 8--11): the causal
expert's adjacency matrix $bold(W)$ collapsed to zero in every
trained checkpoint we examined, rendering the expert equivalent to
a plain MLP. A two-part patch (NOTEARS reconstruction loss +
initialisation rescale) restores DAG learning at zero primary-task
cost, but the DAG is initially "decorative" --- structurally valid
and not routed into prediction. We report two Axis-3 candidates that
route the functional DAG into consumable per-prediction outputs:
a Causal Explainability Head (CEH) producing a per-sample attribution
vector, and a Causal Guardrail (CG) producing a reliability flag.
CEH's first formulation collapsed to a global importance pattern;
a minimum-intervention "demeaned target" variant restored per-sample
discrimination. CG's first formulation failed at chance-level
discrimination; a z-space Mahalanobis formulation hit $100%$ TPR
at $5%$ FPR on three synthetic OOD probes. A W-amplification
experiment then established that the decorative DAG is a
training-choice artefact, not an architectural constraint ---
init $0.1 arrow.r 0.3$ and $lambda_"recon" 0.5 arrow.r 2.0$ grows
$||bold(W)||_F$ 14-fold at zero task cost. Combined with the
HMAC-signed hash-chained audit trail described in the companion
paper, CEH + CG produce a regulator-usable per-prediction record
that pairs *what* the model recommended (attribution) with
*whether* that recommendation can be trusted (reliability).

These findings are not novel algorithms but practical diagnostics
and minimum-viable candidates on top of them. We hope they prevent
other practitioners from re-discovering the same pitfalls when
scaling MTL to real-world heterogeneous task portfolios, and that
they motivate later work to validate the positive recipes
(NEAS, BRP-detached, CEH v2, CG v2, W-amplified teachers) across
additional seeds and datasets.

// ============================================================
= Author Contributions

*Seonkyu Jeong* (PM / Lead Architect / Data Scientist):
Conceived the study, designed the ablation framework, identified all
eleven findings, wrote the manuscript. Led AI-augmented development
methodology.

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
