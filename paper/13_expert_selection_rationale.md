# Expert Selection Rationale

> **Note**: This document uses on-premises dimensions (734D features, 16 tasks).
> For the AWS v1 paper version (349D, 13 tasks), see the v1 Zenodo preprint
> (DOI: 10.5281/zenodo.19621884) or the feature-router table in
> paper/01_architecture_positioning.md.

Each expert in the PLE architecture was chosen to address a specific gap that no other component covers. This document extracts the "why this expert" justification from the technical reference documents.

---

## 1. DeepFM Expert -- Feature Interaction

**FeatureRouter 라우팅:** demographics, product_holdings, txn_behavior, derived_temporal, gmm_clustering, model_derived → **109D** (전체 316D 중 feature interaction에 유효한 연속형 피처만 수신)

**Problem it solves:** Financial recommendation depends on feature *interactions* (e.g., "30s + digital-savvy + high RFM" jointly predict online investment conversion), not individual features alone. A linear model with explicit cross-terms requires O(n^2) parameters (207K for 644 features), most of which cannot be reliably estimated from sparse data.

**Why DeepFM over alternatives:** FM's low-rank factorization reduces cross-parameters to O(nk) = 10,304 while enabling generalization to unobserved feature pairs via shared latent vectors. The Deep component adds implicit high-order interactions (3+ features) that FM's 2nd-order limitation cannot capture. Unlike Wide&Deep (Google 2016), DeepFM shares embeddings between FM and Deep, eliminating manual cross-feature engineering. The combination provides "structural efficiency (FM) + universal expressiveness (Deep)" -- FM explicitly captures pairwise interactions that MLP would learn inefficiently, while MLP captures arbitrarily complex nonlinear patterns that FM structurally cannot represent.

**Financial domain justification:** Patterns like "high RFM + digital adoption + economic uncertainty -> safe-asset preference surge" are 3rd-order interactions requiring both FM (efficient 2nd-order) and Deep (implicit higher-order). FeatureRouter가 그래프/토폴로지 피처를 제외하고 109D만 전달함으로써 FM의 파라미터 효율성이 더욱 강화된다 (교차항 수 O(nk): 316D 대비 65% 감소).

---

## 2. TDA Features (70D) + PersLay Expert (64D) -- Topological Structure

**FeatureRouter 라우팅:** tda_global, tda_local → **32D** (TDA 피처만 수신. 인구통계·거래 피처는 제외하여 토폴로지 신호 순도를 유지)

**Problem it solves:** Traditional statistics (mean, variance, correlation) summarize distribution moments but miss the *structural shape* of data -- cluster connectivity, cyclic consumption patterns, and voids in spending space. Two customers with identical mean/variance can have fundamentally different consumption topologies (one contiguous cluster vs. two separated clusters with periodic transitions).

**Why TDA/Persistent Homology over alternatives:** Persistent Homology observes data at all scales simultaneously (filtration), tracking when topological features (connected components H0, loops H1, voids H2) appear and disappear. Features that persist across many scales are signal; those that vanish quickly are noise. This provides three unique properties unavailable from any statistical or neural approach: (1) coordinate invariance -- rotation/scaling does not change topology, (2) multi-resolution analysis without choosing a single threshold, and (3) mathematically guaranteed noise robustness via the stability theorem. PersLay (Carriere et al.) makes Persistence Diagrams differentiable for end-to-end learning, enabling the PLE gate to weight topological information per-task.

**Financial domain justification:** Consumption cyclicity (loops), life-stage transitions (topological phase changes), and category concentration (connected components) are structurally encoded. The 70D offline TDA features capture short-term (90-day app) and long-term (12-month transaction) topology plus phase transitions between windows. FeatureRouter가 순수 TDA 피처 32D만 전달함으로써 PersLay가 비토폴로지 신호에 의해 교란되지 않고 위상 구조 학습에 집중할 수 있다.

---

## 3. Hyperbolic GCN (H-GCN) + LightGCN -- Graph Structure

**FeatureRouter 라우팅:**
- **H-GCN**: product_hierarchy → **34D** (계층 구조 피처만 수신. 34D로도 쌍곡 공간에서 MCC 트리 전체를 효율적으로 인코딩)
- **LightGCN**: graph_collaborative → **66D** (협업 필터링 그래프 피처만 수신)

**Problem it solves:** Recommendation data has two fundamentally different geometric structures: (1) user-item interactions are *peer-to-peer* (no hierarchy), and (2) merchant category codes (MCC) form a *tree* hierarchy (Root -> L1 -> L2 -> Brand -> Branch with ~550K nodes). No single geometry can efficiently represent both.

**Why Hyperbolic GCN, not just Euclidean GCN:** A complete binary tree of depth d has 2^d leaf nodes. Embedding these with equal spacing in Euclidean space requires O(2^d) dimensions -- embedding ~50K brand-level nodes distortion-free would need tens of thousands of dimensions. Hyperbolic space (negative curvature) expands *exponentially* with distance from origin, exactly matching tree branching. An 8D Poincare Ball suffices for the full MCC hierarchy. Nickel & Kiela (2017) showed 5D hyperbolic embeddings outperform 200D Euclidean embeddings on WordNet hierarchies. The project uses dual GCN: LightGCN (Euclidean) for collaborative filtering ("who likes what") and H-GCN (Poincare Ball) for hierarchical structure ("how merchants relate structurally"), with 2-stage learning to fit single-GPU constraints.

**Financial domain justification:** MCC hierarchy encodes merchant taxonomy critical for category-level recommendation. Cold-start and sparse users benefit from hierarchical structure signals that collaborative filtering alone cannot provide. FeatureRouter가 H-GCN에 순수 계층 피처 34D만, LightGCN에 협업 피처 66D만 전달하여 두 그래프 expert가 각자의 기하학적 공간(쌍곡·유클리드)에 최적화된 신호만 처리한다.

---

## 4. Causal Expert + OT Expert -- Causal Inference and Distributional Matching

**FeatureRouter 라우팅:**
- **Causal Expert**: demographics, product_holdings, txn_behavior, derived_temporal, product_hierarchy, gmm_clustering → **103D** (인과 추론에 필요한 공변량 집합 전체 수신. 그래프·TDA 피처는 제외하여 DAG 추정 안정성 확보)
- **OT Expert**: demographics, product_holdings, txn_behavior, derived_temporal, gmm_clustering → **69D** (분포 매칭에 필요한 연속형 피처만 수신. 계층·그래프 피처는 거리 계산의 기하학적 의미를 훼손할 수 있어 제외)

**Problem it solves:** Standard recommendation systems rely on *correlation* ("customers who bought A also bought B"), which conflates spurious associations with genuine causal effects. Example: "premium card holders have high travel insurance uptake" may be confounded by income level -- the card does not *cause* insurance purchase. Additionally, comparing customer-to-prototype similarity requires respecting the geometric structure of the underlying feature space, which KL divergence and total variation distance ignore.

**Why Causal + OT over alternatives (e.g., A/B testing):** A/B testing is the gold standard but impractical at scale (16 tasks x N strategies = infeasible), slow (weeks to significance), and provides only group-level ATE. The Causal Expert learns a DAG adjacency matrix W via NOTEARS, extracting *directional* causal relationships (asymmetric, W_ij != W_ji) and enabling individual treatment effect (ITE) estimation from observational data. The OT Expert uses Sinkhorn-regularized optimal transport to compute distributional distances that respect the metric structure of feature space (unlike KL divergence, Wasserstein distance reflects that Seoul-Incheon is closer than Seoul-Busan). The two experts answer complementary questions: Causal asks "will this recommendation *cause* behavioral change?" (directional); OT asks "how well does this customer's spending distribution *match* the target profile?" (geometric distance).

**Financial domain justification:** Confounders are pervasive in financial data (income drives both product holding and spending patterns). Causal DAG structure provides explainable recommendation pathways. OT-based distributional matching enables precise customer-to-archetype positioning. Causal Expert가 103D로 인과 공변량을 충분히 확보하면서도 TDA·그래프 피처를 제외한 것은 NOTEARS DAG 추정의 수치적 안정성을 위한 설계적 선택이다.

---

## 5. Temporal Ensemble Expert (Mamba + LNN + PatchTST) -- Time Series

**FeatureRouter 라우팅:** txn_behavior, hmm_states, mamba_temporal, model_derived → **129D** (시계열·시퀀스 피처만 수신. 인구통계·정적 피처는 제외하여 temporal 신호 순도를 유지)

**Problem it solves:** Static features (age, average spend) discard the temporal dimension -- periodicity, trends, and irregular event patterns in transaction sequences. Compressing a 180-day spending sequence to a single monthly average loses weekly cycles, trend direction, and anomalous bursts.

**Why a three-model ensemble over a single temporal model:** Every time series decomposes into trend T(t) + seasonality S(t) + residual R(t). No single architecture optimally captures all three: (1) Mamba (Selective SSM) excels at long-range sequential dependencies (trend) via selective state transitions but struggles with precise periodicity beyond ~100 steps due to exponential decay. (2) PatchTST (Patch Transformer) excels at global periodic pattern matching (seasonality) via self-attention over 16-day patches but handles ordering only implicitly. (3) LNN (Liquid Neural Network / ODE-based) naturally handles irregular time intervals (residual) via adaptive time constants but has limited capacity per single ODE step. The Ensemble Gating dynamically assigns per-customer weights, determining which model best explains each customer's temporal pattern -- this is the "4th generation" of time-series modeling, combining the strengths of all prior paradigms (ARIMA -> LSTM -> Transformer -> SSM+ODE+Transformer ensemble).

**Financial domain justification:** Financial transactions exhibit strong periodicity (payday cycles, weekend dining), gradual trends (lifestyle changes, approaching churn), and irregular residuals (travel, fraud). Each component maps directly to a model's strength. FeatureRouter가 txn_behavior·hmm_states·mamba_temporal·model_derived 129D만 전달하여 시계열 모델이 정적 피처의 분포 편향 없이 순수 동적 패턴에 집중하도록 한다.

---

## 6. HMM Triple-Mode (48D) -- Latent State Transitions

**Problem it solves:** Observable transaction data (amounts, frequencies, category diversity) cannot distinguish customers in qualitatively different behavioral *states* -- the same 100K monthly spend could come from "actively exploring new services" or "making final purchases before churning." These latent psychological states are not directly measurable.

**Why HMM over alternatives:** HMM is specifically designed for inferring unobservable states from observable emissions via probabilistic reasoning (Forward-Backward algorithm). Unlike rule-based segmentation ("monthly spend > 500K = active"), HMM provides *soft assignment* probability vectors ("80% active, 15% growing, 5% at-risk"), which are richer features. The Triple-Mode design (Journey 5-state / Lifecycle 5-state / Behavior 6-state) captures three temporal scales in parallel, compensating for the Markov assumption's inability to capture long-range dependencies. The transition matrix A encodes behavioral dynamics -- e.g., a_RETENTION,ADVOCACY = 0.18 means 18% of retained customers escalate to advocacy per period.

**Financial domain justification:** Customer lifecycle dynamics (acquisition -> growth -> maturity -> churn) naturally map to HMM state transitions. Regime-switching models (Hamilton 1989) -- the financial analog -- use identical mathematics for bull/bear market detection. HMM provides the dynamic trajectory ("this customer is currently transitioning from active to at-risk") that static segmentation cannot.

---

## 7. GMM Clustering Features (22D) -- Static Soft Segmentation

**Problem it solves:** Customer populations are inherently heterogeneous -- a single Gaussian cannot describe both "students with small weekend transactions" and "high-net-worth daily investors" in the same table. Hard clustering (K-Means) forces binary membership, losing boundary information.

**Why GMM over K-Means:** GMM provides *soft probabilistic membership* (posterior gamma_nk via Bayes' theorem), generating a 20D probability vector per customer instead of a single cluster ID. This probability vector serves as a soft routing signal for the GroupTaskExpertBasket's 20 cluster-specific sub-heads. Each component's full covariance matrix captures ellipsoidal cluster shapes (axis-aligned clusters would miss correlated feature patterns). The theoretical justification rests on the Central Limit Theorem: within each homogeneous subpopulation, aggregate features (monthly spend = sum of hundreds of independent transactions) converge to Gaussian, making GMM's per-component Gaussian assumption well-founded even when the overall distribution is multimodal.

**Financial domain justification:** The 22D output (20D cluster probabilities + cluster ID + entropy) enables the model to softly route customers to behavioral archetype experts. Entropy measures segmentation ambiguity -- high-entropy customers sit between archetypes and may need different recommendation strategies.

---

## 8. Multidisciplinary Features (24D) -- Cross-Domain Pattern Extraction

**Problem it solves:** Traditional statistical features view data through a single lens. Different academic disciplines have spent centuries developing mathematical tools optimized for specific pattern types -- reaction kinetics for transformation speed, epidemiology for diffusion dynamics, criminology for routine regularity, wave physics for periodic decomposition. These tools extract information that is nearly orthogonal to statistical summaries.

**Why chemical kinetics, SIR, criminology, and wave physics:** The key insight is *structural isomorphism* -- the mathematical equations are identical regardless of the domain object. Exponential decay in radioactive half-life and in transaction frequency decline share the same equation; the SIR compartmental model for disease spread and for product category adoption share the same ODE structure. Specifically: (1) Chemical kinetics (6D): activation energy (barrier to category entry), half-life (transaction decay rate), catalytic effects (promotions); (2) SIR epidemiology (5D): R0 (category adoption virality threshold), susceptible/infected/recovered ratios; (3) Routine Activity Theory (5D): burstiness (Barabasi 2005), circular statistics for time-of-day patterns, routine deviation detection; (4) Wave interference (8D): FFT spectral decomposition, phase synchronization (PLV), cross-spectral coherence.

**Financial domain justification:** Each framework extracts a distinct behavioral dimension from the same card transaction data -- transformation dynamics, adoption spread, routine regularity, and spectral periodicity -- that statistical features structurally cannot capture.

---

## 9. Economics Features (17D) -- Theory-Driven Behavioral Decomposition

**Problem it solves:** Pure descriptive statistics (mean, std, skewness) summarize the *shape* of spending data but not *why* that shape occurs. Two customers with identical average monthly spend of 3M KRW may have completely different economic structures -- one has stable 4M salary with regular 3M spending; the other has 2M salary plus quarterly bonuses with concentrated spending.

**Why Friedman PIH and marginal utility:** Friedman's Permanent Income Hypothesis (8D) decomposes observed income into permanent (stable, long-run) and transitory (bonus, windfall) components, revealing that consumption should respond differently to each type. This encodes *causal economic structure* that pure statistics miss. The financial behavior indicators (9D) apply microeconomic consumer theory: income elasticity (luxury vs. necessity goods classification), Herfindahl-Hirschman Index (spending concentration), coefficient of variation (income stability), and marginal propensity to consume. These are dimensionless ratios invariant to income scale -- a customer earning 30M and one earning 300M can be directly compared on elasticity, reducing feature scaling burden.

**Financial domain justification:** The permanent/transitory decomposition directly determines optimal card product recommendations (regular-discount cards for high permanent income vs. cashback cards for bonus-driven spenders). Income elasticity identifies whether a customer treats a spending category as luxury (elastic) or necessity (inelastic), guiding recommendation urgency.

---

## 10. adaTT (Adaptive Task-aware Transfer) -- Dynamic Multi-Task Orchestration

**Problem it solves:** Standard PLE with fixed task towers has three fundamental limitations: (1) shared backbone parameters affect all tasks equally, so optimizing CTR can degrade Churn prediction with no mechanism to detect or prevent this; (2) no measurement of which task pairs help vs. hurt each other among 16 tasks competing for shared parameters; (3) fixed weights cannot track changing task relationships across training phases (early: CTR and CVR align; late: they diverge).

**Why adaTT over standard PLE gating or fixed multi-task architectures:** adaTT performs *task-space attention* -- analogous to Transformer self-attention but operating over tasks instead of tokens. Each task's gradient direction serves as Query, other tasks' gradient directions serve as Keys, and gradient cosine similarity determines transfer weights. Only task pairs with aligned gradients share knowledge; conflicting gradients are blocked. This is a lightweight variant of Hypernetworks (Ha et al. 2017) that uses observed gradient signals instead of a full weight-generation network, achieving parameter efficiency (n^2 transfer matrix + prior vs. full hypernetwork). The 3-Phase training schedule adapts transfer intensity: warm-up (broad sharing) -> main (selective transfer) -> fine-tune (task-specific refinement).

**Financial domain justification:** Financial recommendation's 16 tasks have complex inter-dependencies -- CTR and CVR may positively transfer, but Churn and LTV may conflict. adaTT quantitatively measures these relationships via gradient cosine similarity and dynamically adjusts, preventing negative transfer that fixed architectures silently suffer.
