# Expert Theory Part 4: Distillation, Scoring, Grounding, and Recommendation Reason Generation

> Extracted from technical reference documents (v1.0, 2026-03-05) for Paper 2.
> Covers the post-training pipeline: Knowledge Distillation, FD-TVS Scoring, Feature Grounding, Recommendation Reason Generation, and LLM Distillation (Gemini QLoRA).

---

## 1. Knowledge Distillation: PLE Teacher to LGBM Student

### 1.1 Design Rationale

**Core problem**: PLE-adaTT Teacher (50M params, 20GB VRAM, ~50ms/1K batch) is too expensive for production serving of millions of customers.

**Solution**: Knowledge Distillation transfers "dark knowledge" (Hinton et al., 2015) from the Teacher's soft-label output distribution into a LightGBM Student model that runs on 8GB RAM CPU at ~5ms/1K batch, achieving 10x speedup with performance loss within 3%p.

**Key design decisions**:
- **Cross-architecture distillation** (DNN -> GBDT): Possible because KD transfers knowledge through output distributions, not parameters. The paradigm of "train with deep learning, serve with GBDT" has become a de facto standard in recommendation, finance, and advertising domains (Borisov et al., NeurIPS 2022).
- **Per-task independent LGBM models**: LightGBM does not natively support multi-output. Each of the 14 tasks gets an independent Student model, but indirectly benefits from multi-task knowledge through soft labels.
- **IG-based feature selection**: 734D -> 200D via Integrated Gradients, then further to ~140D via LGBM importance filtering.

### 1.2 Mathematical Formulations

#### Temperature Scaling

$$p_i^T = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

- $T = 1$: Standard softmax (concentrated on argmax)
- $T = 5$ (default): Smoothed distribution revealing inter-class relationships ("dark knowledge")
- $T \to \infty$: Uniform distribution

**Boltzmann connection**: The formula is mathematically isomorphic to the Boltzmann distribution in statistical mechanics, where logits $z_i$ correspond to (negative) energy states and $T$ is absolute temperature. Both belong to the exponential family.

**Temperature range**: $T \in [3, 7]$, with $T=3$ for binary tasks, $T=5$ default, $T=7$ for multiclass tasks (NBA 12-class, Timing 28-class). $T > 10$ risks excessive information loss.

#### Unified Distillation Loss

$$\mathcal{L}_{\text{distill}} = \alpha \cdot \mathcal{L}_{\text{hard}} + (1 - \alpha) \cdot T^2 \cdot \mathcal{L}_{\text{soft}}$$

- $\alpha$: Hard/soft ratio (default: 0.3, meaning 30% ground truth, 70% teacher opinion)
- $T^2$ scaling: Compensates for gradient magnitude reduction. When $T=5$, gradients shrink by $1/25$; multiplying by $T^2=25$ restores the original scale.

**Mathematical derivation of $T^2$**: From the chain rule, $\frac{\partial \hat{y}}{\partial z} = \frac{1}{T} \sigma(z/T)(1 - \sigma(z/T))$. The $1/T$ factor accumulates to shrink the overall gradient by $1/T^2$.

#### Task-Specific Loss Functions

**Binary (CTR, CVR, Churn, Retention)**:
$$\mathcal{L}_{\text{binary}} = \alpha \cdot \text{BCE}(\hat{y}, y) + (1-\alpha) \cdot T^2 \cdot \text{KL}(p_t \| p_s)$$

where $p_t = \sigma(z_t/T)$, $p_s = \sigma(z_s/T)$.

**Multiclass (NBA 12-class, Life-stage 6-class, Timing 28-class)**:
$$\mathcal{L}_{\text{multiclass}} = \alpha \cdot \text{CE}(z_s, y) + (1-\alpha) \cdot T^2 \cdot \text{KL}(\text{softmax}(z_t/T) \| \text{softmax}(z_s/T))$$

**Regression (LTV, Engagement)**:
$$\mathcal{L}_{\text{regression}} = \alpha \cdot \text{MSE}(\hat{y}_s, y) + (1-\alpha) \cdot \text{MSE}(\hat{y}_s, \hat{y}_t)$$

No $T^2$ scaling for regression (temperature scaling is meaningless for continuous values).

#### KL-Divergence

$$D_{\text{KL}}(q \| p) = \sum_i q_i \log \frac{q_i}{p_i} = \underbrace{-H(q)}_{\text{const}} + \underbrace{H(q, p)}_{\text{cross-entropy}}$$

Forward KL $D_{\text{KL}}(\text{Teacher} \| \text{Student})$ is used (not reverse KL) to ensure the Student covers all modes the Teacher considers important (mean-seeking rather than mode-seeking property).

### 1.3 Dark Knowledge: Information-Theoretic Justification

- Hard label encodes $\log_2(C)$ bits per sample (e.g., ~3.6 bits for 12-class NBA)
- Soft label encodes $(C-1)$ continuous probability values >> $\log_2(C)$ bits
- Soft labels provide natural label smoothing regularization
- Cross-task knowledge transfer: PLE's shared representations are implicitly encoded in soft labels, allowing independent LGBM students to indirectly benefit from multi-task learning

**Empirical evidence** (from reference document):

| Method | CTR AUC | NBA Accuracy | LTV RMSE |
|--------|---------|-------------|----------|
| LGBM (Hard Label only) | 0.812 | 0.634 | 1.247 |
| LGBM (Distilled, T=5) | 0.841 | 0.698 | 1.089 |
| PLE Teacher (original) | 0.856 | 0.723 | 1.021 |

### 1.4 Feature Selection: IG-Based 3-Stage Pipeline

**Stage 1 - Integrated Gradients (734D -> 200D)**:
$$\text{IG}(x)_i = (x_i - x'_i) \times \int_0^1 \frac{\partial F(x' + \alpha(x - x'))}{\partial x_i} d\alpha$$

- Baseline: zero vector (suitable for normalized features)
- Steps: 50 (trapezoidal rule)
- Completeness axiom: $\sum_i \text{IG}(x)_i = F(x) - F(x')$ (no attribution leakage)

**Stage 2 - LGBM Importance Filter (200D -> ~140D)**: Remove bottom 30% by gain importance.

**Stage 3 - Mandatory Features**: 7 features always included regardless of IG/LGBM importance:
- TDA: `persistence_entropy`, `landscape_peak`
- Economics: `mpc`, `income_elasticity`, `permanent_income_ratio`
- FinEng: `sharpe_ratio`, `volatility`

### 1.5 LightGBM Custom Objective

The NumPy implementation (`DistillationLossNumpy`) provides gradient/hessian to LightGBM's `fobj`:

```
grad = alpha * grad_hard + (1 - alpha) * T^2 * grad_soft
hess = alpha * hess_hard + (1 - alpha) * T^2 * hess_soft
```

**Soft label passing trick**: Since LightGBM Dataset only supports `label` and `weight`, soft labels are passed via `get_weight()` (hard labels via `get_label()`).

### 1.6 10-Stage DAG Orchestration

The full distillation pipeline is orchestrated in `distillation_entrypoint.py` as a 10-stage DAG: Teacher inference -> Soft label generation -> IG feature selection -> Student training (per-task) -> Validation -> MLflow registry -> ONNX export -> Triton packaging.

---

## 2. FD-TVS Scoring Engine

### 2.1 Design Philosophy

**FD-TVS (Financial DNA-based Target Value Score)** is a 4-stage composite scoring engine that combines model predictions with customer-level business context. The key design principle is **multiplicative combination**: any single factor approaching zero vetoes the entire score, enforcing a "risk-first" principle.

**Why not simple summation?**
- Simple sum $\sum p_i$ cannot distinguish a window shopper (CTR=0.9, CVR=0.1) from a real buyer (CTR=0.5, CVR=0.5) -- both sum to 1.0
- Multiplicative structure gives each dimension **veto power**
- Combines WSM (Stage 1, additive) and WPM (Stages 2-4, multiplicative) in a hybrid structure

### 2.2 Master Formula

$$\text{FD-TVS} = \underbrace{S_{\text{task}}}_{\text{What}} \times \underbrace{W_{\text{DNA}}}_{\text{Who}} \times \underbrace{V_{\text{TDA}}}_{\text{When}} \times \underbrace{(1 - R)}_{\text{Safe?}} \times \underbrace{f \cdot e}_{\text{Appropriate?}}$$

- $S_{\text{task}}$: Task Weighted Sum [0, 1]
- $W_{\text{DNA}}$: DNA Modifier {0.8, 1.0, 1.2}
- $V_{\text{TDA}}$: Behavioral Velocity [1.0, 1.15]
- $R$: Risk Penalty [0, 1]
- $f$: Fatigue Decay [0, 1]
- $e$: Engagement Boost [0.85, 1.15]

### 2.3 Stage 1: Task-Weighted Sum

$$S_{\text{task}} = \beta_{\text{CTR}} \cdot p_{\text{CTR}} + \beta_{\text{CVR}} \cdot p_{\text{CVR}} + \beta_{\text{NBA}} \cdot p_{\text{NBA}} + \beta_{\text{LTV}} \cdot p_{\text{LTV}}$$

Default weights: CVR=0.4 (highest, conversion directly impacts revenue), CTR=0.3, NBA=0.2, LTV=0.1.

**MCDM justification**: This is a Weighted Sum Model (WSM, Fishburn 1967) -- a convex combination guaranteeing $S_{\text{task}} \in [0, 1]$ when $\sum \beta_i = 1$ and all $p_i \in [0, 1]$.

### 2.4 Stage 2: Financial DNA Modifier

Based on **Friedman's Permanent Income Hypothesis** (1957): consumers base spending on permanent (stable) income, not transitory fluctuations.

$$W_{\text{DNA}} = \begin{cases} 1.2 & \text{if CV} < 0.2 \quad \text{(Permanent -- stable income)} \\ 1.0 & \text{if } 0.2 \leq \text{CV} < 0.5 \quad \text{(Mixed)} \\ 0.8 & \text{if CV} \geq 0.5 \quad \text{(Transitory -- unstable)} \end{cases}$$

where $\text{CV} = \sigma_{\text{income}} / \mu_{\text{income}}$ (coefficient of variation, dimensionless).

**Financial domain justification**: Stable-income customers are more suitable for long-term financial products (pension, term deposits), so their recommendation scores are boosted by 20%.

### 2.5 Stage 3: TDA Behavioral Velocity

$$V_{\text{TDA}} = 1.0 + \gamma_{\text{flare}} \cdot \mathbb{1}[\text{flare\_detected}]$$

$\gamma_{\text{flare}} = 0.15$. TDA flare detection indicates accelerating behavioral change, boosting the score by up to 15%.

### 2.6 Stage 4: Risk Penalty

$$R = 0.2 \cdot I_{\text{limit}} + 0.3 \cdot I_{\text{fatigue}} + 0.5 \cdot I_{\text{churn}}$$

**Asymmetric weighting rationale based on irreversibility**:
- Credit limit exhaustion ($\lambda_1 = 0.2$): Reversible (repayment restores limit)
- Message fatigue ($\lambda_2 = 0.3$): Partially reversible (time heals)
- Customer churn ($\lambda_3 = 0.5$): Nearly irreversible (reacquisition costs 5-7x vs new acquisition)

The $(1 - R)$ term acts as a multiplicative veto: if $R \to 1$, the score collapses to zero regardless of other factors. In log space: $\ln(1-R) \to -\infty$ as $R \to 1$.

### 2.7 Fatigue Decay (Exponential)

$$f(n) = e^{-\lambda n}$$

**Constant fractional decay**: $f(n+1)/f(n) = e^{-\lambda}$ (constant ratio).

Half-life: $n_{1/2} = \ln 2 / \lambda$. For App Push ($\lambda = 0.4$): half-life $\approx 1.73$ messages. For Email ($\lambda = 0.15$): half-life $\approx 4.62$ messages.

### 2.8 Confidence Formula

$$\text{confidence} = |p - 0.5| \times 2$$

Measures distance from the decision boundary (0.5), normalized to [0, 1]. Used as a recommendation quality filter -- low-confidence predictions are deprioritized.

---

## 3. Inference Pipeline: ONNX + Triton

### 3.1 Architecture

6-stage journey: Training -> Distillation -> ONNX Conversion -> Triton Deployment -> FD-TVS Scoring -> Customer Serving.

**Batch + Real-time hybrid**: Daily batch for baseline scores; real-time Redis features for FD-TVS recalculation on transactions. Triton Dynamic Batching enables this by queuing individual requests into micro-batches.

### 3.2 LGBM to ONNX Conversion

- **ZipMap removal** (critical): LightGBM's ONNX conversion adds ZipMap operators producing dictionary outputs. Triton only supports tensor outputs, so ZipMap must be removed by bypassing the node in the ONNX graph.
- **Opset 13**: Full LightGBM operator support
- **2-stage validation**: (1) `onnx.checker.check_model` for spec compliance, (2) dummy inference test

### 3.3 Triton Configuration

- 15 ONNX models (task-specific) + 1 preprocessor + 1 postprocessor + 15 ensemble schedulers = **32 model configs**
- Dynamic Batching: preferred sizes [256, 512, 1024], max queue delay 100us
- Preprocessor: CPU x4 instances (CPU-bound JSON parsing)
- ONNX models: GPU x2 instances

### 3.4 Training-Serving Skew Prevention

**Feature Serving Spec** bridges training and serving:
- `feature_selector` outputs `selected_features_{task}.json` during training
- `FeatureServingSpec` loads these at deploy time, ensuring identical feature ordering
- 7 mandatory features always included (TDA, Economics, FinEng)

### 3.5 Calibration Considerations

FD-TVS Stage 1 requires all task probabilities to be on a common scale [0, 1]. If CTR is overconfident and CVR is underconfident, weighted summation is biased. Temperature Scaling (Guo et al., ICML 2017) is identified as a future improvement for post-hoc calibration.

---

## 4. Grounding: Feature Reverse Mapping

### 4.1 Core Problem

PLE-adaTT consumes a 734D feature vector (644D normalized + 90D raw power-law) and outputs probability scores, but cannot explain *why*. Financial regulations (AI Basic Act Art. 31/34, Financial Consumer Protection Act Art. 19) require meaningful explanations.

**Grounding function**: $f: \mathbb{R}^{644} \times \mathcal{I} \to \mathcal{L}$, where $\mathcal{I}$ is IG attribution information and $\mathcal{L}$ is the space of human-readable natural language explanations.

### 4.2 Interpretability vs Explainability

The system adopts an **Explainability-first** design (post-hoc explanation via IG + reverse mapping + LLM), since PLE-adaTT's multi-expert towers with gate networks make inherent interpretability infeasible. This aligns with EU AI Act (August 2024) requirements for high-risk AI systems.

### 4.3 Integrated Gradients for Feature Attribution

$$\text{IG}_i(\mathbf{x}) = (x_i - x'_i) \times \int_0^1 \frac{\partial F(\mathbf{x}' + \alpha(\mathbf{x} - \mathbf{x}'))}{\partial x_i} d\alpha$$

**Why IG over SHAP**: SHAP requires $2^{734}$ subset evaluations (infeasible). IG uses path integral with 50-step trapezoidal approximation (linear time). IG satisfies the **completeness axiom**: $\sum_i \text{IG}_i(\mathbf{x}) = F(\mathbf{x}) - F(\mathbf{x}')$ (guaranteed by the Gradient Theorem from vector calculus -- no attribution leakage).

**Baseline**: Zero vector (appropriate for normalized features where 0 represents "average customer" or "absence of information").

### 4.4 734D Feature Vector Structure

| Range | Dims | Description |
|-------|------|-------------|
| profile | 0-238 | Demographics (100D) + RFM (50D) + Financial Summary (88D) |
| multi_source | 238-329 | Transaction stats (40D) + Behavioral patterns (51D) |
| extended_source | 329-413 | Insurance, consultation, STT, campaign, overseas, open banking |
| domain | 413-572 | TDA (70D) + GMM (22D) + Mamba (50D) + Economics (17D) |
| model_derived | 572-599 | HMM summary, Bandit/MAB, LNN |
| multidisciplinary | 599-623 | Conversion dynamics, adoption dynamics, cross-patterns, routine analysis |
| merchant_hierarchy | 623-644 | MCC levels, brand embeddings, statistics, radius |

Total: 644D normalized + 90D raw power-law = 734D model input.

### 4.5 Reverse Mapping Architecture

$$\text{ReverseMap}: (\mathbf{x} \in \mathbb{R}^d, \mathbf{a} \in \mathbb{R}^d) \to \{(r_k, s_k, t_k)\}_{k=1}^K$$

- $\mathbf{x}$: Feature vector, $\mathbf{a}$: IG attribution vector
- $r_k$: Feature range name, $s_k$: Summary score, $t_k$: Financial language text

**Sub-range slicing pattern**: $t_k = \mathcal{M}_k(g(\mathbf{x}[s_k : e_k]))$ where $g$ is an aggregation function (mean, argmax, threshold comparison) and $\mathcal{M}_k$ is a domain-expert-designed mapping dictionary (numeric -> text).

### 4.6 Modules

- **FeatureReverseMapper**: 644D vector -> financial language text via hierarchical range slicing
- **MultidisciplinaryInterpreter**: 24D multidisciplinary features -> business interpretations (4 sub-domains: conversion dynamics, adoption dynamics, cross-patterns, routine analysis)
- **LanceContextVectorStore**: LanceDB-based customer context storage/retrieval, similar customer search (L2 distance on 768D embeddings)
- **ContextAssemblyAgent**: IG-driven tool selection + multi-source context assembly
- **ConsultationContextExtractor**: STT consultation history extraction + summarization

### 4.7 Trust Loop

Model Prediction -> IG Attribution -> Reverse Mapping (Feature -> Financial Language) -> Context Assembly (Consultation + Similar Customers) -> LLM Reason Generation -> Advisor Delivery -> Customer Persuasion -> Conversion/Feedback -> Model Improvement.

Without reverse mapping and context assembly, there is an interpretability gap between model prediction and advisor delivery, breaking this trust loop.

---

## 5. Recommendation Reason Generation

### 5.1 2-Layer Architecture (v3.0.0)

**Design philosophy**: "Equal reasons for all customers; LLM-enhanced reasons for context-rich customers."

| Layer | Target | Method | LLM Calls | Throughput |
|-------|--------|--------|-----------|-----------|
| L1 | 12M (all) | Template (6 categories x 5 variants, hash selection) | 0 | ~20 min |
| L2a | ~500K/week (rich + moderate) | LLM rewrite (vLLM Qwen3-8B-AWQ, 3-layer safety gate) | 1 | ~1.0 sec/item |
| L2b | ~67K sampling | Quality validation (factuality, relevance, naturalness) | 1 | - |

**Regulatory justification**: Financial Consumer Protection Act S19 requires equal explanation duty for all customers. L1 fulfills this with template-based reasons at zero GPU cost. L2a selectively rewrites for customers with rich context.

**Cost comparison**: Full LLM processing of 12M customers would require ~1,000 GPU-hours; 2-Layer design uses ~162 GPU-hours.

### 5.2 L1 Template Generation

- 6 categories x 5 variants = 30 templates
- **Deterministic variant selection**: $\text{variant\_index} = \text{hash}(\text{customer\_id} : \text{category}) \mod 5$ -- same customer always receives same variant (consistency + audit reproducibility)
- **Segment-aware**: WARMSTART (IG Top-3 reverse mapping), COLDSTART (popularity + benefits), ANONYMOUS (generic popularity)
- Rule-based compliance check with automatic AI disclosure notice attachment

### 5.3 L2a LLM Rewrite

- Priority queue: rich first, moderate second, sparse excluded
- **3-Layer Safety Gate**: prompt injection detection, factuality check, regulatory compliance
- vLLM Qwen3-8B-AWQ on RTX 4070 (12GB VRAM)
- `generation_method = "template_l1_l2a_rewrite"`

### 5.4 Self-Critique Judgment

$$\text{verdict} = \begin{cases} \text{pass} & \text{if } f \geq 0.8 \text{ and } c \geq 1.0 \\ \text{revise} & \text{if } f \geq 0.5 \text{ and } c \geq 1.0 \\ \text{reject} & \text{otherwise} \end{cases}$$

- $f$ = factuality score (continuous), $c$ = compliance score (binary: 1.0 = no violations)
- **Compliance takes absolute priority**: Any regulatory violation ($c < 1.0$) causes immediate rejection regardless of factuality
- **Maximum 1 revision**: If revised output still gets "revise", it falls back to safe template (prevents infinite loops, caps LLM calls at 3)

### 5.5 L2b 3-Axis Quality Validation

$$\text{verdict} = \begin{cases} \text{pass} & \text{if } f \geq 0.7 \text{ and } r \geq 0.7 \text{ and } n \geq 0.7 \\ \text{needs\_improvement} & \text{if any score} \in [0.5, 0.7) \\ \text{fail} & \text{if any score} < 0.5 \end{cases}$$

- $f$ = factuality, $r$ = relevance, $n$ = naturalness (added for L2a rewrite quality)
- Threshold 0.7 (vs Self-Critique's 0.8): L2b is post-hoc monitoring, not real-time gatekeeper

### 5.6 Prompt Engineering Strategy

4-layer prompt structure:
1. **System prompt**: Role definition (financial recommendation specialist) + regulatory violation prohibition rules
2. **Few-shot examples**: Segment-specific examples for tone and format guidance
3. **Context injection**: Customer features, IG attributions, consultation history -> natural language
4. **Output format**: JSON schema (`{"reasons": [...], "summary": "..."}`)

**Decoding strategies**:
- Reason generation: $\tau = 0.3$ (factuality-preserving with slight diversity)
- Critique: $\tau = 0.1$ (near-deterministic for consistent quality assessment)
- L2a rewrite: $\tau = 0.3$ (preserve original facts, polish expression)

### 5.7 Safety and Compliance

- **AI Basic Act Art. 31** (AI usage notification): Automatic AI-generated notice attached to all reasons
- **AI Basic Act Art. 34** (risk management): Safety gate + audit trail
- **Financial Consumer Protection Act Art. 19** (suitability principle + explanation duty): L1 ensures equal coverage
- **Financial Consumer Protection Act Art. 21** (advertising regulation): Prohibited pattern detection
- **Financial Consumer Protection Act Art. 22** (unfair practices prohibition): Compliance scoring

### 5.8 Audit Archiving

All recommendation records are persisted to Parquet via `RecommendationAuditArchiver`:
- IG attribution scores, L1 reasons, L2a rewrite results, L2b validation results, processing time
- Enables retroactive audit queries by financial supervisory authority
- DuckDB-backed for efficient querying

### 5.9 Three-Fold Grounding

The term "Grounding" has three meanings in this system:
1. **Feature Grounding**: IG Top-5 attributions injected into prompt -> LLM generates reasons based on actual model judgment
2. **Customer Grounding**: Segment, transaction patterns, consultation history -> suppresses hallucination
3. **Regulatory Grounding**: System prompt prohibitions + rule-based Self-Critique -> enforces compliance

---

## 6. LLM Distillation: Gemini Teacher to Qwen Student (QLoRA)

### 6.1 Two Distinct Distillations

| Aspect | Prediction Model Distillation | LLM Distillation (this section) |
|--------|------------------------------|--------------------------------|
| Purpose | Prediction accuracy | Text generation quality |
| Teacher | PLE-Cluster-adaTT | Google Gemini (Pro/Flash) |
| Student | LightGBM | Qwen3-8B |
| Transfer target | Soft labels (logits/probs) | Text outputs (recommendation reasons) |
| Loss function | KL Divergence + CE | Cross-Entropy (SFT) |
| Training method | Soft label learning | QLoRA fine-tuning |

### 6.2 Current System Limitations

Qwen3-8B-AWQ is a general-purpose pre-trained model without domain-specific fine-tuning:
- Insufficient financial domain specificity (terminology, product understanding)
- High revise/reject rate in Self-Critique (poor first-pass quality)
- Elevated Safety Gate failure rate (Gate 2 compliance violations)
- Tone mismatch across customer segments (VIP vs general)

### 6.3 QLoRA: Why Not Full Fine-Tuning

**Memory analysis for Qwen3-8B Full FT (FP16)**:
- Model weights: 16 GB
- Optimizer (Adam): 32 GB (momentum $m$ + variance $v$ each FP32)
- Gradients: 16 GB
- Total: 64+ GB (RTX 4070 has only 12 GB)

**QLoRA solution**: Base model 4GB (NF4) + LoRA adapter ~40MB = **6GB for training**.

### 6.4 LoRA Mathematical Foundation

$$W' = W_0 + \Delta W = W_0 + BA$$

- $W_0 \in \mathbb{R}^{d \times k}$: Original pre-trained weights (frozen)
- $B \in \mathbb{R}^{d \times r}$: Down-projection (trainable)
- $A \in \mathbb{R}^{r \times k}$: Up-projection (trainable)
- $r \ll \min(d, k)$: Rank (e.g., $r=16$ for 0.78% of original parameters)

**Compression ratio**: $\frac{r \times (d+k)}{d \times k}$. For Qwen3-8B ($d=k=4096$, $r=16$): $131,072 / 16,777,216 \approx 0.78\%$.

**Forward pass**: $h = W_0 x + BAx$ (original knowledge + domain adaptation residual).

**Intrinsic Dimensionality Hypothesis** (Aghajanyan et al., 2021): RoBERTa-Large (355M params) needs only 896 intrinsic dimensions for MRPC task adaptation -- 0.00025% of total parameters. Hu et al. (2021) showed GPT-3 175B achieves >97% of Full FT performance with $r=4$.

**Eckart-Young Theorem**: Guarantees that rank-$r$ SVD approximation is optimal under Frobenius norm. If singular values of $\Delta W$ decay rapidly, small $r$ captures the essential change.

### 6.5 NF4 Quantization

**Distribution-aware quantization**: NF4 places quantization levels at the quantiles of the standard normal distribution:

$$q_i = \Phi^{-1}(i / (2^k + 1))$$

This ensures each quantization level covers equal probability mass (Lloyd-Max optimal condition). Dense region near zero gets fine-grained levels; sparse tails get coarse levels.

**QLoRA separation principle**:
- Read-only knowledge ($W_0$): 4-bit NF4 (extreme compression OK since unchanged)
- Learnable adaptation ($B$, $A$): BF16/FP16 (gradient precision critical for learning quality)

**Key finding** (Dettmers et al., NeurIPS 2023): QLoRA (NF4 + LoRA) reproduces 99.3% of 16-bit LoRA performance on Guanaco benchmark. The quantization noise $\delta x$ is small enough not to affect LoRA's convergence direction.

### 6.6 LoRA Initialization

$$B = \mathbf{0}, \quad A \sim \mathcal{N}(0, \sigma^2)$$

At initialization, $\Delta W = BA = \mathbf{0}$: model output is exactly identical to the original. $A$ is random to break symmetry so different neurons adapt in different directions.

### 6.7 Self-Consistency Filtering for Training Data

$$\text{consistency}(s_1, s_2, s_3) = \min_{i \neq j} \text{BERTScore}(s_i, s_j)$$

Generate 3 outputs for the same input from Gemini Teacher; take the minimum pairwise BERTScore as a conservative consistency measure. Only consistent outputs are included in training data, filtering out Teacher hallucinations.

### 6.8 QLoRA Forward Pass

$$Y = X \cdot \text{dequant}(W_0^{\text{NF4}}) + X \cdot (BA) \cdot \alpha/r$$

Two signals: (1) pre-trained knowledge signal from 4-bit compressed base model, (2) domain adaptation signal from learnable adapter. $\alpha/r$ is the "volume knob" for the adaptation signal.

### 6.9 Integration Strategy

- **Zero-change to existing pipeline**: `LLMProvider`, `AgenticReasonOrchestrator` unchanged
- **Merged model deployment**: After training, merge LoRA adapter into base weights -> re-quantize (AWQ) -> serve via vLLM
- **A/B testing**: Compare fine-tuned model vs base model on Self-Critique pass rate, Safety Gate failure rate, L2b quality scores

---

## 7. Cross-Cutting Themes for Paper 2

### 7.1 Financial Domain Specificity

Every component is designed with financial domain constraints:
- **Distillation**: Mandatory features (TDA, Economics, FinEng) preserved regardless of IG importance
- **Scoring**: Friedman's Permanent Income Hypothesis for DNA modifier; asymmetric risk weighting based on irreversibility
- **Grounding**: Domain-expert-designed mapping dictionaries for financial language translation
- **Reason generation**: Compliance-first design (regulations override quality)
- **LLM distillation**: Financial terminology, compliance-aware tone, product-specific knowledge transfer

### 7.2 Safety and Compliance Architecture

Multi-layer defense:
1. **System prompt**: Invariant regulatory prohibitions
2. **Self-Critique**: Real-time gatekeeper (factuality >= 0.8, compliance = 1.0)
3. **3-Layer Safety Gate**: Prompt injection + factuality + regulatory compliance
4. **L2b Quality Validation**: Post-hoc monitoring with 3-axis evaluation
5. **AI Security Checker**: Injection detection + compliance verification
6. **Audit Archiver**: Immutable Parquet records for regulatory audit trail

### 7.3 Cost Efficiency

- **2-Layer architecture**: 162 GPU-hours vs 1,000 GPU-hours for full LLM coverage
- **LGBM Student**: CPU inference at 1/10 the cost of GPU Teacher inference
- **QLoRA**: 6GB training on consumer GPU vs 64GB+ for Full FT
- **Triton Dynamic Batching**: Maximizes GPU utilization for hybrid batch/real-time serving

### 7.4 End-to-End Pipeline

```
PLE-adaTT Teacher (Training)
    |
    v
Knowledge Distillation (T=5, alpha=0.3)
    |
    v
LGBM Student (per-task, 200D features)
    |
    v
ONNX Export (ZipMap removed)
    |
    v
Triton Inference Server (15 tasks, Dynamic Batching)
    |
    v
FD-TVS Scoring (4-stage: Task -> DNA -> TDA -> Risk)
    |
    v
Feature Grounding (IG -> Reverse Mapping -> Context Assembly)
    |
    v
Recommendation Reason (L1 Template -> L2a LLM Rewrite -> L2b Validation)
    |
    v
Audit Archive (Parquet, regulatory compliance)
```
