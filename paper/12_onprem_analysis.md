# On-Premises PLE Project Analysis

> Source: `github.com/bluethestyle/gotothemoon` (cloned at `c:/Users/user/Desktop/ttm/gotothemoon`)
> Analysis date: 2026-04-01
> Total commits analyzed: ~450

---

## 1. Design Evolution Timeline

### Phase 1: Infrastructure & Data Pipeline (commits a7581b6 ~ cf055e2)
- **3-Tier Integer Indexing Pipeline** for PII management
- Hive-based data ingestion with metadata governance
- DuckDB integration for columnar processing

### Phase 2: Feature Engineering Foundation (commits 6631cd2 ~ 0552246)
- PLE task abstraction and DVC metadata versioning (ec0d24d)
- Feature Importance via Integrated Gradients documented (c473d2a)
- **11-task dependent variable management system** established (0552246)
  - Commit message: "feat: PLE-adaTT 11개 태스크 종속변수 관리 체계 구축"
- Time series window management system (6e6b857)

### Phase 3: Task Expansion & Business Redesign (commits 23bad2e ~ e5c74f0)
- Uplift/LTV/Channel/Timing tasks implemented (23bad2e)
- **CTR/CVR card funnel rewrite** + task module on/off system (e5c74f0)
  - Commit message: "feat: CTR/CVR 카드퍼널 기반 재작성 + 태스크 모듈 on/off 시스템 구현"
- Brand prediction / merchant affinity generators added (00b09b0)
- Cold-start / anonymous customer pipeline end-to-end (8e36b8f)

### Phase 4: Architecture Consolidation — "Plan C" (commits d7dd9c1 ~ bb3defe)
- **"Plan C" + Legacy Integration Architecture v3.1.0** (d7dd9c1)
  - This was the pivotal moment: merging legacy MLP-based experts with the new PLE-Cluster-adaTT structure
- Merchant HGCN Expert integration with shared_experts (3f370fa)
- Commit: "feat: C안 통합 완료 - 학습/증류/추론 파이프라인 전환" (bb3defe)

### Phase 5: Expert Architecture Refinement (commits 3783799 ~ 1f2b3fb)
- **v2.3 Risk Mitigation**: HMM default embedding, adaTT annealing, CGC entropy regularization (3783799)
- **v3.2**: Causal Expert, OT Expert, SAE, Evidential DL + "customer context architect" design philosophy (9fdbd00)
  - Commit message: "feat: v3.2 — Causal Expert, OT Expert, SAE, Evidential DL + 고객 맥락 아키텍트 설계 철학"
- H-GCN hyperbolic personality strengthening + transaction-weighted time decay (9bf7bff)
- **v3.3**: H-GCN hyperbolic enhancement + Task/Expert structural mismatch fixes (1f2b3fb)

### Phase 6: Regulatory Compliance & Distillation (commits 0f7fb76 ~ f9ec5ba)
- **A-group 18 regulatory compliance implementations** (0f7fb76)
  - Parts 1-4 detailed design specifications
- Knowledge distillation and retraining Critical 16 fixes (18c1123)
- LGBM Student local batch packaging + Triton deployment (fb3bd62)
- **3-Tier to 2-Layer recommendation reason architecture** transition (03bca3c)

### Phase 7: Feature Normalization & Dimension Stabilization (commits c656192 ~ d1d6bdd)
- **Feature Normalization v3.7**: 734D architecture (644D normalized + 90D raw power-law) (c656192)
  - This is the definitive normalization strategy that the AWS version inherited
- Task-specific feature interpretation differentiation, Phases 1-7 (d5d2cd8 ~ f43bbb1)

### Phase 8: Production Hardening (commits 68cc012 ~ 012883c)
- **pandas dependency removal**: 13 violating files converted to DuckDB/PyArrow (68cc012)
  - Commit: "fix: pandas 의존성 제거 — 13개 위반 파일 DuckDB/PyArrow 전환"
- Feature engineering pipeline with DAG, GMM, TDA, HMM parameter tuning
- Ablation test framework v3.3 architecture alignment (5c545bb)

---

## 2. Key Architectural Decisions

### 2.1 "Plan C" — The Consolidation Decision

The project went through at least three architectural plans before settling on "Plan C" (C안):

**Before Plan C** (deprecated `expert_networks.py`, 1099 lines):
- Simple MLP-based experts (Hyperbolic, TDA, Temporal)
- Comment from code: "문제: Expert 분리의 이점 없음 (모두 동일한 MLP)"

**Plan C** (the current `ple_cluster_adatt.py`):
- Domain-specialized models as Shared Experts (H-GCN, PersLay, DeepFM, Temporal Ensemble, LightGCN, Causal, OT)
- Task Experts with 20 GMM cluster subheads (320 total subheads)
- adaTT for loss-level adaptive task transfer
- Design doc quote: "PLE 단층 구조 — Expert 출력 1회 통과 단층 구조. 6개 도메인 Expert가 각자 내부 다층 구조를 보유하므로 외부 반복 층 불필요."

### 2.2 Expert Selection: Hard Routing, Not Soft Gating

A critical design change documented in the architecture doc (2026-02-19 audit):
> "Expert 선택 메커니즘 — ~~GatingNetwork(Softmax/Top-K 기반 확률적 가중 합산)~~ → GMM cluster_id 기반 Hard Routing. ClusterTaskExpert가 cluster_id로 서브헤드를 확정 선택. cluster_id=-1(미할당) 시에만 cluster_probs 가중 합산(Soft Routing)으로 fallback."

This means the PLE gating was replaced by GMM-based deterministic routing, which is a departure from the original PLE paper's soft gating mechanism.

### 2.3 Task Group Evolution

The adaTT task groups evolved from the original implementation:

**Current 4-group structure** (from `adatt.py`):
```
engagement: [CTR, CVR, Engagement, Uplift] — intra strength 0.8
lifecycle:  [Churn, Retention, Life_stage, LTV] — intra strength 0.7
value:      [Balance_util, Channel, Timing] — intra strength 0.6
consumption:[NBA, Spending_category, Consumption_cycle, Spending_bucket, Merchant_affinity, Brand_prediction] — intra strength 0.7
inter-group strength: 0.3
```

**v3.3 change** noted in config: "personalization→consumption 병합" — a personalization group was merged into consumption.

**adaTT 3-Phase mechanism**:
- Phase 1 (epoch 1-10): Affinity measurement via gradient cosine similarity
- Phase 2 (epoch 11+): Dynamic transfer with per-epoch updates
- Phase 3 (late training): Transfer weights frozen + fine-tuning

### 2.4 Normalization Strategy (3-Stage)

From the feature engineering design doc and code:
```
Stage 1: Power-law detection (skew+kurt → log-log R²) + log1p copy creation
Stage 2: StandardScaler (continuous columns only, binary excluded, TRAIN fit only)
Stage 3: Power-law _log copies are NOT scaled (raw magnitude preserved)
```

Final dimension: **734D = 644D normalized + 90D raw power-law**

### 2.5 Design Philosophy — "Model Structure is Already Sufficient"

From `06a_모델_아키텍처_개요.md`:
```
1. 모델 구조는 이미 충분 → 추가 복잡도는 운영 부담만 증가
2. 진짜 차별화는 "피처의 도메인 해석력"
3. 성능 목표: Non-Regression (Hit@5 30%+ realistic, 50% stretch)
4. 벡터DB = 해석의 허브
```

This philosophy — that the real differentiator is feature interpretability, not model complexity — is a key insight for the paper.

---

## 3. Features in On-Prem Not Yet in AWS Version

### 3.1 Comprehensive Expert Portfolio (8 active shared experts)

The on-prem system has a much richer set of shared experts:
1. **Unified H-GCN** (128D) — Hyperbolic GCN for hierarchy + merchant
2. **PersLay** (64D) — Persistence Diagram processing (Carriere et al., 2020)
3. **DeepFM** (64D) — Field-level independent embeddings (28 fields)
4. **Temporal Ensemble** (64D) — Mamba + LNN + Transformer
5. **LightGCN** (64D) — Graph-based collaborative filtering
6. **Causal Expert** (64D) — SCM/NOTEARS (Zheng et al., NeurIPS 2018)
7. **OT Expert** (64D) — Sinkhorn Optimal Transport (Cuturi, NeurIPS 2013)
8. **RawScale Expert** (64D) — Non-normalized power-law features

Additionally, SAE (Sparse Autoencoder) and Evidential DL layers for analysis.

### 3.2 Leakage Guard (v3.13.1)

A sophisticated per-task data leakage prevention mechanism (`LeakageGuard` class):
- Pre-computes per-task binary masks over the 734D feature tensor
- Uses a post-hoc correction approach: subtracts leaked-feature contribution from task-expert output
- The correction is gradient-detached to prevent the model from learning to reconstruct leaked features

### 3.3 Full Distillation Pipeline (PLE Teacher → LGBM Student)

Complete implementation:
- `soft_label_generator.py` — Temperature-scaled soft labels
- `lgbm_student.py` — Per-task LGBM models (binary/multiclass/regression)
- `feature_selector.py` — LGBM gain importance-based feature selection (403D → ~140D + mandatory features)
- `distillation_loss.py` — L_distill = alpha * L_hard + (1-alpha) * T^2 * L_soft
- `distillation_validator.py` — Quality validation

### 3.4 FD-TVS Scoring Engine

A novel scoring system — **Financial DNA-based Target Value Score**:
```
FD-TVS = S_task x W_DNA x V_TDA x (1 - R_penalty) x fatigue_decay x engagement_boost
```
- **Stage 1** (Task Weighted Sum): beta_CTR*P_CTR + beta_CVR*P_CVR + beta_NBA*P_NBA + beta_LTV*P_LTV
- **Stage 2** (DNA Modifier): Based on Friedman's Permanent Income Hypothesis — CV(income) classifies customers as Permanent/Mixed/Transitory
- **Stage 3** (Behavioral Velocity): V_TDA = 1 + gamma * I_Flare (TDA persistence flare detection)
- **Stage 4** (Risk Penalty): Regulatory and risk adjustments

### 3.5 Multi-Disciplinary Feature Engineering (24D)

Four cross-disciplinary feature extractors:
1. **Chemical Kinetics** (6D) — Category activation rate, spending half-life, acceleration, dormancy reactivation, catalyst sensitivity, saturation proximity
2. **Epidemic Diffusion** (5D) — SIR model applied to MCC category adoption: Susceptible/Infected/Recovered ratios
3. **Interference** (8D) — Consumption cross-patterns: KL divergence, HHI, Spearman correlations
4. **Crime Pattern** (5D) — Routine Activity Theory for spatiotemporal spending regularity: burstiness, recurrence period, circular variance

### 3.6 Recommendation Reason Grounding Pipeline

An agentic grounding pipeline for explainability:
- `FeatureReverseMapper` — 644D features back to interpretable text
- `MultidisciplinaryInterpreter` — Business language interpretation
- `ConsultationContextExtractor` — Customer consultation history
- LanceDB vector store for context storage
- `RecommendationReasonGenerator` — vLLM/Qwen3-8B-AWQ for natural language reasons
- **Agentic extension**: Tier-based differentiated quality, IG-based dynamic context, Self-Critique validation

### 3.7 Regulatory Compliance Infrastructure

Extensive compliance modules aligned with Korean financial regulations:
- `ai_decision_opt_out.py` — 개인정보보호법 제37조의2 (automated decision refusal rights)
- `kill_switch.py` — 3-level emergency shutdown (GLOBAL/PER_TASK/PER_CLUSTER)
- `conflict_of_interest_detector.py` — Fee/margin bias detection (금감원 7대 원칙 ⑥)
- `human_fallback_router.py` — Human review escalation
- `marketing_consent.py` — Marketing consent filtering
- `explanation_sla_tracker.py` — Explanation request SLA tracking
- Per-task regulatory risk mapping matrix (14 tasks x regulatory requirements)

### 3.8 HMM Triple-Mode Routing

Three separate HMM modes routed to tasks:
- **JOURNEY** (16D): CTR/CVR/Engagement tasks (AWARENESS→ADVOCACY stages)
- **LIFECYCLE** (16D): Churn/Retention/Life-stage tasks (NEW→CHURNED stages)
- **BEHAVIOR** (16D): NBA/Balance_util tasks (DORMANT→INVESTOR states)
Each mode: base 10D + ODE dynamics bridge 6D = 16D

### 3.9 Cluster Stability Mechanisms

- **Hungarian Algorithm**: Weekly GMM retraining uses `linear_sum_assignment` for optimal previous-to-new center matching, preserving cluster ID continuity
- **Cluster Balanced Sampling**: Inverse-size weighted oversampling (power=0.5) to prevent minority cluster under-training
- **COLDSTART→WARMSTART dynamic promotion**: GMM inference promotes customers with NaN<50% and max_prob>0.3
- **GMM Health Gate**: 3-way judgment (proceed/retrain/rollback) with `.backup` model rollback
- **Cluster x Task performance monitoring**: 14 tasks x 20 clusters, >10% deviation alerts

---

## 4. Design Intent from Commit Messages

### Architecture Evolution
- `"feat: C안 + 레거시 통합 아키텍처 v3.1.0"` — The Plan C consolidation
- `"feat: v3.2 — Causal Expert, OT Expert, SAE, Evidential DL + 고객 맥락 아키텍트 설계 철학"` — "Customer Context Architect" as a design philosophy
- `"feat: 계층 임베딩 시스템 통합 — Generator 2→1, Expert 2→1 (Strategy C)"` — Hierarchy consolidation
- `"fix: 모델 입력 차원 통일 (694D → config-based 644D)"` — Config-driven dimensions

### Regulatory and Quality
- `"feat: A그룹 18건 규제준수 구현 — Part1~Part4 세부설계서 기반 전량 구축"` — Full regulatory implementation
- `"fix: pandas 의존성 제거 — 13개 위반 파일 DuckDB/PyArrow 전환"` — Strict no-pandas policy
- `"fix: IMMEDIATE 4건 + SPRINT1 13건 — 17건 자동 수정 완료"` — Systematic code review remediation

### Feature Engineering
- `"feat: 다학제 24D 피처 전면 재설계 — SQL 교체·그라운딩 소비자 친화 용어·Interference 8D 부활"` — Multi-disciplinary redesign
- `"feat: Feature Normalization v3.7 — 734D 아키텍처 전수 반영 (644D normalized + 90D raw power-law)"` — Definitive normalization strategy

### Operational Maturity
- `"docs: Complete code review deliverables - 146 findings, TOP 8 risks, 315h roadmap"` — Extensive code review
- `"feat: SPRINT1-1 학습-추론 전처리 일관성 확보 - quantile 저장/복원"` — Train-serve consistency

---

## 5. On-Prem vs AWS Version Comparison

| Aspect | On-Prem (gotothemoon) | AWS (aws_ple_for_financial) |
|--------|----------------------|---------------------------|
| Data Pipeline | Hive → DuckDB → Parquet | S3 Parquet (Santander benchmark) |
| Orchestration | Airflow DAGs + DockerOperator | SageMaker Training Jobs |
| Shared Experts | 8 active (H-GCN, PersLay, DeepFM, Temporal, LightGCN, Causal, OT, RawScale) | Feature generators (TDA, GMM, HMM, Economics, etc.) |
| Cluster Routing | GMM Hard Routing with Hungarian stability | GMM-based with config-driven routing |
| Task Count | 16 active / 18 defined | Configurable via pipeline.yaml |
| Scoring | FD-TVS (4-stage with Friedman DNA modifier) | Not yet implemented |
| Distillation | Full PLE→LGBM with IG feature selection | Design only |
| Explainability | Full grounding pipeline with vLLM | Not yet implemented |
| Compliance | 18-regulation mapping, kill switch, opt-out | Not yet implemented |
| Feature Dims | 734D (644 normalized + 90 raw) | Configurable |
| Normalization | 3-stage (power-law detect → StandardScaler → raw preserve) | 3-stage (same strategy) |
| Data Backend | DuckDB mandatory (pandas banned for >10K) | DuckDB/NumPy/PyArrow |

---

## 6. Unique Insights for the Paper

### 6.1 The "Feature Interpretability > Model Complexity" Principle
The on-prem project explicitly documents that model architecture is "already sufficient" and the real differentiator is domain interpretability. This is a mature perspective that evolved through months of iteration.

### 6.2 Multi-Disciplinary Feature Engineering Rationale
Applying concepts from chemistry (reaction kinetics), epidemiology (SIR model), criminology (Routine Activity Theory), and wave interference to financial behavior analysis is novel. Each has a clear mapping:
- Chemical kinetics → category activation/dormancy dynamics
- Epidemic diffusion → product adoption lifecycle
- Crime pattern → spatiotemporal spending regularity
- Interference → cross-category spending competition

### 6.3 Friedman's Permanent Income Hypothesis in Scoring
Using the coefficient of variation of income to classify customers as Permanent/Mixed/Transitory income types, then weighting recommendations accordingly, is a direct application of macro-economic theory to personalization.

### 6.4 TDA Behavioral Velocity (Flare Detection)
Using persistence diagram characteristics to detect behavioral "flares" (sudden increases in topological complexity) and boosting recommendations during these windows is a novel use of TDA in recommendation systems.

### 6.5 Leakage Guard as Post-Hoc Correction
Rather than recomputing shared expert outputs per task (computationally expensive), the LeakageGuard uses gradient-detached correction signals to attenuate leaked feature contributions. This is a practical engineering solution to a theoretical concern.

### 6.6 Evolution from Soft Gating to Hard Routing
The project explicitly abandoned PLE's original soft gating mechanism (GatingNetwork with Softmax/Top-K) in favor of GMM-based hard routing with cluster subheads. The rationale: cluster ID provides a more stable and interpretable routing signal than learned gating weights.

### 6.7 adaTT as Loss-Level Transfer (Not Representation-Level)
The actual adaTT implementation operates at the loss level, not the representation level. This is emphasized in multiple places: "실제 구현: Loss-level Adaptive Transfer (representation-level이 아닌 loss-level에서 태스크 간 전이)".

### 6.8 Scale of the Regulatory Compliance Effort
The on-prem project has dedicated compliance modules for Korean financial AI regulations (금감원 7대 원칙, 금소법, 개인정보보호법, AI 기본법), including per-task regulatory risk mapping, a kill switch with 3 granularity levels, and automated decision opt-out processing. This represents significant engineering effort beyond the model itself.
