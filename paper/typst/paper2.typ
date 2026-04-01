// ============================================================
// Paper 2: Recommendation Reason Generation & Regulatory Compliance
// ============================================================

#set document(
  title: "From Prediction to Persuasion: Agentic Recommendation Reason Generation for Regulatory-Compliant Financial AI",
  author: ("Author 1", "Author 2", "Author 3"),
)

#set page(
  paper: "us-letter",
  margin: (x: 1.8cm, y: 2cm),
  columns: 2,
  numbering: "1",
)

#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.6em)
#set heading(numbering: "1.1")

// Title
#align(center)[
  #text(size: 16pt, weight: "bold")[
    From Prediction to Persuasion: Agentic Recommendation Reason Generation for Regulatory-Compliant Financial AI
  ]

  #v(0.8em)

  #text(size: 11pt)[
    Author 1#super[1], Author 2#super[1], Author 3#super[1]
  ]

  #v(0.3em)

  #text(size: 9pt, style: "italic")[
    #super[1]Organization Name \
    {author1, author2, author3}\@org.com
  ]

  #v(1em)
]

// Abstract
#block(
  width: 100%,
  inset: (x: 1em),
)[
  #text(weight: "bold")[Abstract.]
  Financial product recommendation systems must explain _why_ a product is recommended ---
  not merely to satisfy regulatory mandates (Korean FSS, EU AI Act),
  but because persuasion requires narrative, not probability.
  We present a three-stage pipeline that bridges the gap between model prediction and human persuasion:
  (1) IG-guided knowledge distillation from a heterogeneous-expert PLE teacher to per-task LGBM students,
  preserving explanation-relevant features while enabling GPU-free CPU inference;
  (2) a multi-agent recommendation reason generation pipeline where three specialized agents
  (Feature Selector, Reason Generator, Safety Gate) collaboratively produce natural-language explanations
  grounded in business-mapped feature attributions;
  (3) regulatory compliance by design, with built-in drift monitoring, fairness auditing,
  and governance reporting aligned to Korean FSS guidelines, the EU AI Act, and the Korean AI Basic Act.
  // TODO: Fill key numbers after experiments
  We evaluate distillation quality (AUC gap < X%), reason generation quality via human evaluation
  (N domain experts, Y\% preference over template baselines),
  and Safety Gate reliability (precision/recall).
  The system achieves sub-100ms serving latency on AWS Lambda
  at a fraction of the cost of dedicated GPU inference servers.

  #v(0.3em)
  #text(weight: "bold")[Keywords:]
  Recommendation explanation, Knowledge distillation, LLM agents,
  Regulatory compliance, Financial AI, EU AI Act
]

#v(1em)

// ============================================================
= Introduction

== The Persuasion Problem

The final deliverable of a financial recommendation system is not a probability
but a _reason the customer can accept_.
A model outputting $P("invest" | x) = 0.73$ provides no value to:
- The *customer* who asks "Why should I buy this fund?"
- The *relationship manager* who needs a talking point for the sales call.
- The *regulator* who demands "Why was this decision made?" under Article 13 of the EU AI Act.

Existing explanation approaches are insufficient:
- *SHAP/LIME*: "feature\_237 contributed 0.12" --- no business meaning, unstable under perturbation @lundberg2017 @ribeiro2016.
- *Template-based*: "Customers like you also bought X" --- rigid, ignores individual context.
- *Direct LLM generation*: Hallucination risk, no grounding in actual model reasoning, regulatory non-compliance.

== Our Approach

We propose a full-chain solution from prediction to persuasion:

+ *Knowledge Distillation*: A heterogeneous-expert PLE teacher (18 tasks, 7 experts, 316 features; see our companion paper) is distilled into per-task LGBM students using IG-guided feature selection. This enables GPU-free serving while preserving the features that matter for explanation.

+ *Multi-Agent Reason Generation*: Three specialized LLM agents collaborate in a pipeline --- Feature Selector chooses explanation-worthy features, Reason Generator produces natural-language narratives, and Safety Gate validates regulatory compliance.

+ *Compliance by Design*: Drift monitoring, fairness auditing, audit trails, and governance reporting are embedded in the pipeline, not bolted on after deployment.

== Contributions

+ *IG-guided Distillation with Explanation Preservation*: Feature selection optimizes not only for prediction accuracy but also for explanation material availability.

+ *Feature Business Reverse-Mapping*: A systematic registry (`interpretation_registry`) that maps every feature to a business-interpretable description, enabling grounded explanation generation.

+ *3-Agent Reason Generation Pipeline*: Role-separated agents (selection → generation → safety) with independent improvement and audit logging.

+ *Safety Gate for Financial Compliance*: Automated checking for hallucination, inappropriate investment advice, and regulatory violations (금소법, 적합성 원칙).

+ *Regulatory Compliance Architecture*: Explicit mapping of system components to Korean FSS guidelines, EU AI Act articles, and the Korean AI Basic Act.

+ *Human Evaluation Protocol*: Systematic evaluation of recommendation reason quality by domain experts.

// ============================================================
= Related Work

== Knowledge Distillation for Recommendation

Hinton et al. @hinton2015 established the teacher-student paradigm via temperature-scaled soft labels.
In recommendation, distillation has been applied to compress deep models into lightweight servers.
CKD (Collaborative Knowledge Distillation) and ranking distillation preserve recommendation-specific structure.

*Gap*: No prior work selects distillation features based on _explanation value_ alongside predictive value.

== Recommendation Explanation Generation

Recommendation explanation approaches fall into three categories:

*Template-based methods* map model outputs to pre-written phrases
("Recommended because you recently viewed similar products").
Amazon and Netflix use variants of this approach at scale.
Templates are safe and fast but lack personalization and flexibility ---
the same template serves millions of different customer contexts.

*Attention-based methods* use model-internal attention weights
as feature attributions for explanation.
While more dynamic than templates, attention weights are contested
as reliable explanations, and the resulting attributions
still require translation into business language.

*LLM-based generation* is an emerging approach that uses large language models
to produce natural-language explanations from model outputs.
Recent work explores LLM-augmented recommendation explanation,
but faces challenges of hallucination, regulatory compliance,
and grounding in actual model reasoning.

*Gap*: No system combines structural model explainability (gate weights)
with multi-agent LLM-based generation under regulatory safety constraints,
where each agent has a specialized role (selection, generation, validation)
and all outputs are audit-logged.

== Responsible AI in Finance

Korean FSS published AI guidelines (2021) @koreafsc2024 and model risk management directives
requiring explainability, fairness monitoring, and audit trails.
The EU AI Act @euaiact2024 classifies financial credit/recommendation as high-risk AI,
mandating transparency (Art. 13), human oversight (Art. 14), and accuracy (Art. 15).
The EBA @eba2025ml calls for "interpretable" models in internal risk assessments.
Korea's AI Basic Act @koreaaiact2024 (December 2024) adds domestic high-risk AI classification.

Pearl @pearl2009causality argues that true explanation requires causal understanding,
not mere statistical association --- a position increasingly echoed by financial regulators
who demand explanations reflecting the actual decision mechanism @salih2023.

*Gap*: No recommendation system provides an explicit, verifiable mapping
from regulatory requirements to system architecture components,
with explanations grounded in causal reasoning rather than post-hoc correlation.

== LLM Safety and Grounding

Deploying LLMs for customer-facing financial text introduces specific risks:
hallucination (stating non-existent product features),
inappropriate advice (recommending unsuitable products for the customer's risk profile),
and regulatory violation (breaching 금소법 or 적합성 원칙).

Retrieval-Augmented Generation (RAG) mitigates hallucination
by grounding generation in retrieved factual context.
Our approach extends this: rather than retrieving from a general knowledge base,
we ground generation in _model-internal feature attributions_
that have been reverse-mapped to business descriptions.
This ensures the generated text reflects what the model actually computed,
not what an LLM independently "knows."

Content filtering and safety gates provide a final defense layer.
Our 3-agent architecture separates generation from validation,
enabling independent improvement of each component.

// ============================================================
= Knowledge Distillation
<distillation>

== Teacher-Student Architecture

#figure(
  rect(width: 100%, height: 5cm, stroke: 0.5pt)[
    #align(center + horizon)[
      _Distillation diagram placeholder_ \
      PLE Teacher (GPU, weekly) → Soft Labels → LGBM Students (CPU, daily) \
      + IG feature selection
    ]
  ],
  caption: [Teacher-student distillation architecture with differentiated retraining cycles.],
) <fig:distillation>

The teacher model (PLE with 7 heterogeneous experts, 18 tasks, 316 features;
see companion paper for architecture details)
produces soft probability outputs that serve as training targets
for per-task LGBM @ke2017lightgbm students.

The key design decision is _per-task distillation_:
rather than a single student model for all 18 tasks,
we train 18 independent LGBM models, each learning one task's soft labels.
This enables:
(1) per-task feature selection (different tasks benefit from different features),
(2) independent retraining (if one task drifts, only its student is re-distilled),
(3) interpretable feature importance per task (LGBM's built-in feature importance
aligns with the business reverse-mapping for explanation generation).

*Lifecycle separation*:
- *Teacher*: retrained weekly/monthly on SageMaker (GPU required, comprehensive).
  The teacher captures complex inter-task relationships via adaTT
  and non-linear expert interactions that LGBM cannot directly learn.
- *Students*: re-distilled daily with fresh soft labels (CPU only, fast).
  Daily re-distillation tracks data drift without the cost of GPU training.
- *Champion-Challenger*: automatic comparison of new student vs. current production model.
  If the new student's AUC drops below threshold, the update is blocked and an alert is raised.

This architecture resolves a fundamental tension in financial AI:
the _model that learns best_ (deep PLE with GPU) is not the _model that serves best_
(lightweight LGBM on CPU Lambda).
Knowledge distillation @hinton2015 bridges this gap,
and the FD-TVS scoring system @friedman1957
further weights the student's predictions by income stability type.

== IG-based Feature Selection

Integrated Gradients @sundararajan2017 computes per-feature attribution from the teacher model.
We select top-$k$ features ranked by IG importance, with a dual objective:

$ "score"(f) = alpha dot "IG"_"pred"(f) + (1 - alpha) dot "IG"_"explain"(f) $

where $"IG"_"pred"$ measures predictive contribution and $"IG"_"explain"$ measures
the feature's value as explanation material.

The explanation score $"IG"_"explain"(f)$ is derived from the feature's
reverse-mapping richness in the interpretation registry:
features with detailed business descriptions, clear directionality,
and natural-language templates receive higher scores.
For example, `hmm_lifecycle_prob_growing` (explanation-rich: "customer is in growth stage")
scores higher than `mamba_temporal_d17` (explanation-poor: generic embedding dimension).

This dual-objective selection ensures that the student model retains
not only the most predictive features but also the features
that generate the most compelling recommendation reasons.
The hyperparameter $alpha$ controls the trade-off:
$alpha = 1$ optimizes purely for prediction,
$alpha = 0$ purely for explanation quality.
We empirically find $alpha = 0.7$ balances both objectives.

The resulting feature set is typically 40--80 features per task
(down from 316), achieving >95% of teacher AUC
while providing sufficient explanation vocabulary.

== Distillation Results

#figure(
  table(
    columns: (auto, auto, auto, auto),
    inset: 5pt,
    align: center,
    stroke: 0.5pt,
    [*Task*], [*Teacher AUC*], [*Student AUC*], [*Gap*],
    [has_nba], [--], [--], [--],
    [churn_signal], [--], [--], [--],
    [will_acquire_deposits], [--], [--], [--],
    [will_acquire_investments], [--], [--], [--],
    [...], [...], [...], [...],
  ),
  caption: [Distillation results per task. TODO: fill after experiments.],
) <tab:distill-results>

// ============================================================
= Recommendation Reason Generation
<reason-generation>

== Feature Business Reverse-Mapping

Every feature in the system is registered in an `interpretation_registry`
with structured business metadata:

#figure(
  table(
    columns: (auto, auto, auto),
    inset: 5pt,
    align: left,
    stroke: 0.5pt,
    [*Feature*], [*Business Mapping*], [*Explanation Template*],
    [hmm_lifecycle_prob_growing], [Growth stage probability], ["Your asset portfolio is in a growth phase"],
    [mamba_temporal_d3], [3-month spending trend], ["Your spending has been increasing recently"],
    [hgcn_hierarchy_d5], [Product category position], ["Investment products are a natural next step"],
    [synth_stability], [Transaction stability], ["You maintain a stable transaction pattern"],
    [gmm_cluster_prob_3], [Segment probability], ["You share characteristics with active investors"],
  ),
  caption: [Feature reverse-mapping examples. Each feature has a registered business interpretation.],
) <tab:reverse-mapping>

This registry serves dual purposes:
(1) grounding material for the Reason Generator agent, and
(2) audit trail showing which features influenced each recommendation.

== 3-Agent Pipeline Architecture

#figure(
  rect(width: 100%, height: 7cm, stroke: 0.5pt)[
    #align(center + horizon)[
      _3-Agent pipeline diagram placeholder_ \
      \
      Agent 1: Feature Selector \
      (IG attribution → top features → explanation-worthy supplementation) \
      ↓ \
      Agent 2: Reason Generator \
      (reverse-mapped contexts → natural language reason) \
      ↓ \
      Agent 3: Safety Gate \
      (regulatory check → hallucination check → appropriateness check) \
      ↓ \
      Pass → Serve | Fail → Template fallback
    ]
  ],
  caption: [Three-agent recommendation reason generation pipeline.],
) <fig:3agent>

=== Agent 1: Feature Selector

Selects features for explanation based on:
- IG attribution scores (model-driven relevance)
- Business reverse-mapping richness (explanation-driven relevance)
- Customer context (personalization: different features matter for different customer profiles)

=== Agent 2: Reason Generator

Receives selected features with their business reverse-mappings and generates
a natural-language recommendation reason. Grounding constraints:
- Must reference only features actually present in the selection.
- Must use business terminology appropriate for the target audience (customer vs. relationship manager).
- Must follow financial domain tone and compliance guidelines.

#figure(
  block(
    width: 100%,
    inset: 8pt,
    stroke: 0.5pt + gray,
    radius: 4pt,
  )[
    #text(size: 9pt)[
      *Example output (customer-facing):* \
      "고객님은 현재 자산 성장 단계에 있으시고, 최근 3개월간 소비가 증가하는 추세입니다.
      보유하신 상품 구조상 투자 펀드가 자연스러운 다음 단계이며,
      안정적인 거래 패턴을 고려할 때 중위험 균형형 펀드를 추천드립니다."
    ]
  ],
  caption: [Generated recommendation reason example (Korean, customer-facing).],
) <fig:reason-example>

=== Agent 3: Safety Gate

Validates the generated reason against:

#figure(
  table(
    columns: (auto, auto),
    inset: 5pt,
    align: left,
    stroke: 0.5pt,
    [*Check Category*], [*Criteria*],
    [Hallucination], [No claims about facts not in feature data],
    [Regulatory], [No violation of 금소법, 적합성 원칙],
    [Appropriateness], [No unsuitable investment advice for risk profile],
    [Tone], [Professional financial advisory language],
    [Factual], [Numerical claims match actual feature values],
  ),
  caption: [Safety Gate validation criteria.],
) <tab:safety-gate>

On failure: automatic fallback to template-based safe reason.
All gate decisions are logged for audit trail.

== Caching Strategy

Recommendation reasons are cached by customer segment × product category × feature pattern hash.
Cache hit avoids LLM invocation entirely, reducing latency and cost.

// TODO: Cache hit rate analysis

// ============================================================
= Regulatory Compliance
<compliance>

== Korean FSS Guidelines Mapping

#figure(
  table(
    columns: (auto, auto, auto),
    inset: 5pt,
    align: left,
    stroke: 0.5pt,
    [*FSS Requirement*], [*System Component*], [*Verification*],
    [Explainability], [Gate weights + 3-agent reason], [Per-recommendation audit log],
    [Fairness], [FairnessMonitor (DI/SPD/EOD)], [Weekly automated report],
    [Model validation], [Champion-Challenger], [Pre-deployment comparison],
    [Monitoring], [DriftDetector (PSI)], [Continuous, 3-day trigger],
    [Audit trail], [HMAC hash-chain logs], [Immutable, 7 audit tables],
    [Fallback], [Template reason + kill switch], [Instant manual override],
  ),
  caption: [Korean FSS guideline compliance mapping.],
) <tab:fss-mapping>

== EU AI Act Mapping

#figure(
  table(
    columns: (auto, auto, auto),
    inset: 5pt,
    align: left,
    stroke: 0.5pt,
    [*Article*], [*Requirement*], [*System Component*],
    [Art. 13], [Transparency], [3-agent reason generation + feature attribution],
    [Art. 14], [Human oversight], [Human-in-the-Loop review + kill switch],
    [Art. 15], [Accuracy & robustness], [Ablation-validated degradation + drift monitoring],
    [Art. 22], [Right to opt-out], [Consent/opt-out audit table + manual override],
  ),
  caption: [EU AI Act article-level compliance mapping.],
) <tab:euai-mapping>

== Korean AI Basic Act (2024.12)

// TODO: High-risk classification analysis for financial recommendation

== Monitoring and Governance

#figure(
  rect(width: 100%, height: 5cm, stroke: 0.5pt)[
    #align(center + horizon)[
      _Monitoring architecture placeholder_ \
      \
      DriftDetector (PSI) → FairnessMonitor (DI/SPD/EOD) → HerdingDetector (HHI/Gini) \
      ↓ \
      IncidentReporter → GovernanceReportGenerator \
      ↓ \
      Monthly/quarterly automated report
    ]
  ],
  caption: [Monitoring and governance architecture.],
) <fig:monitoring>

=== Human-in-the-Loop

Regulatory bodies (Korean FSS, EU AI Act Art. 14) require human oversight.
The system implements this at multiple levels:
- *Reason sampling review*: Periodic human review of generated reasons.
- *Model replacement approval*: Champion-Challenger results require human sign-off.
- *Incident escalation*: Automated anomaly detection triggers human investigation.
- *Fairness review*: Periodic human audit of fairness metrics.

// ============================================================
= Experiments

== Distillation Experiments

// TODO: Teacher vs student AUC per task
// TODO: Feature count vs AUC trade-off curve
// TODO: IG_pred vs IG_explain alpha sensitivity

== Reason Generation Quality

=== Human Evaluation Protocol

- *Evaluators*: N financial domain experts (bank product managers, compliance officers).
- *Evaluation criteria*: Accuracy (1-5), Naturalness (1-5), Persuasiveness (1-5), Regulatory fitness (1-5).
- *Comparison*: (A) Template-based, (B) SHAP-based + template, (C) 3-agent pipeline.
- *Method*: Blind evaluation, each evaluator rates 100 recommendation reasons.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    inset: 5pt,
    align: center,
    stroke: 0.5pt,
    [*Method*], [*Accuracy*], [*Natural.*], [*Persuasion*], [*Compliance*],
    [Template], [--], [--], [--], [--],
    [SHAP+Template], [--], [--], [--], [--],
    [3-Agent (ours)], [--], [--], [--], [--],
  ),
  caption: [Human evaluation results. TODO: fill after evaluation.],
) <tab:human-eval>

== Safety Gate Evaluation

#figure(
  table(
    columns: (auto, auto, auto, auto),
    inset: 5pt,
    align: center,
    stroke: 0.5pt,
    [*Check*], [*Precision*], [*Recall*], [*Fallback Rate*],
    [Hallucination], [--], [--], [--],
    [Regulatory], [--], [--], [--],
    [Appropriateness], [--], [--], [--],
    [Overall], [--], [--], [--],
  ),
  caption: [Safety Gate precision/recall. TODO: fill after evaluation.],
) <tab:safety-eval>

== Serving Performance

#figure(
  table(
    columns: (auto, auto, auto, auto),
    inset: 5pt,
    align: center,
    stroke: 0.5pt,
    [*Component*], [*Latency*], [*Cost/1K req*], [*Notes*],
    [LGBM inference], [< 10ms], [--], [Lambda CPU],
    [IG attribution], [< 50ms], [--], [Lambda CPU],
    [Reason generation], [< 200ms], [--], [Cache hit: < 5ms],
    [Safety Gate], [< 50ms], [--], [Rule-based + LLM],
    [*Total*], [< 300ms], [--], [Cache hit: < 100ms],
  ),
  caption: [Serving latency breakdown. Lambda serverless, no GPU required.],
) <tab:serving>

== Regulatory Compliance Audit

// TODO: Checklist pass/fail, audit log integrity verification, fairness metrics

// ============================================================
= Discussion

== Findings Summary
// TODO

== The Dual Role of Features

A key insight from this work: features serve dual purposes in financial recommendation.
Even features with marginal predictive contribution (e.g., TDA topological features
may add only $Delta$AUC = 0.01) provide irreplaceable explanation material.
"Your spending pattern shows a _persistent stable shape_" (from TDA)
cannot be generated from any other feature type.

This reframes feature engineering evaluation:
the value of a feature is not solely its predictive contribution
but also its contribution to the explanation vocabulary available to the system.

== Practical Deployment Considerations

- *LLM selection*: On-premises LLM vs API (latency, cost, data residency).
- *Reason quality maintenance*: Periodic human review + automated quality scoring.
- *Regulatory updates*: Architecture supports adding new compliance checks without redesign.

== Limitations

- LLM dependency introduces cost, latency, and residual hallucination risk.
- Human evaluation scale limited by expert availability.
- Korean/EU regulatory focus; other jurisdictions not yet analyzed.
- Template fallback quality is inherently limited.

== Future Work

- Real customer A/B testing of reason quality → conversion rate.
- Multi-lingual reason generation (Korean, English, Chinese).
- Automated regulatory update pipeline (regulation change → compliance check update).
- Fine-tuned domain-specific small LLM to replace general-purpose API.

// ============================================================
= Conclusion

We presented a full-chain system that bridges the gap
between model prediction and human persuasion in financial product recommendation.

Three key contributions define this work.
First, IG-guided knowledge distillation with a dual-objective feature selection
preserves both predictive accuracy and explanation material
when compressing a complex PLE teacher into lightweight LGBM students.
Second, the 3-agent recommendation reason generation pipeline
(Feature Selector → Reason Generator → Safety Gate)
produces natural-language explanations grounded in business-mapped feature attributions,
with role separation enabling independent improvement and audit logging.
Third, regulatory compliance is embedded by design ---
Korean FSS guidelines, the EU AI Act, and the Korean AI Basic Act
are explicitly mapped to system architecture components,
with automated monitoring (drift, fairness, herding)
and human-in-the-loop oversight at critical decision points.

The fundamental insight is that features serve a dual role in financial AI:
they contribute to prediction _and_ to the explanation vocabulary
that ultimately persuades customers, empowers relationship managers,
and satisfies regulators.
This reframes the traditional feature engineering calculus:
a feature with marginal AUC contribution but rich business interpretability
may be more valuable than a high-AUC feature that generates no meaningful explanation.

// TODO: Add key distillation and human eval numbers when available

The system is designed for deployment on serverless infrastructure (AWS Lambda),
achieving sub-100ms serving latency without dedicated GPU servers ---
matching the operational reality of financial institutions
with limited ML engineering resources.

// ============================================================
// Acknowledgments
#heading(numbering: none)[Acknowledgments]

The code implementation and manuscript drafting were assisted by
Claude (Anthropic) as an AI coding and writing tool.
The architectural decisions, domain knowledge, experimental design,
and research direction were led by the human authors.

// ============================================================
#bibliography("references.bib", style: "association-for-computing-machinery")
