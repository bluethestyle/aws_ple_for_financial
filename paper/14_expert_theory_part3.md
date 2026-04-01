# Expert Theory Extraction Part 3: Economics, Multidisciplinary, GMM, and adaTT

Extracted from technical reference documents v3.14 (2026-03-05) for arxiv paper use.

---

## 1. Economics Features (17D)

### 1.1 Core Theory: Friedman's Permanent Income Hypothesis (PIH)

**Theoretical Foundation (Friedman, 1957)**

The PIH decomposes observed income into permanent and transitory components:

$$Y_t = Y_t^P + Y_t^T$$

where $Y_t^P$ is permanent income (long-term stable) and $Y_t^T$ is transitory income (temporary fluctuations). The consumption function follows:

$$C_t = k(r, w, u) \cdot Y_t^P$$

Key implication: consumers spend proportionally to permanent income; transitory income is directed to savings/investment.

**Three estimation methods are supported:**

| Method | Formula | Complexity |
|--------|---------|------------|
| Moving Average | $\hat{Y}_t^P = \frac{1}{L}\sum_{i=0}^{L-1} Y_{t-i}$, $L=12$ | Low (SQL-native) |
| HP Filter | $\min_\tau \left\{\sum_t (Y_t - \tau_t)^2 + \lambda \sum_t [(\tau_{t+1}-\tau_t) - (\tau_t - \tau_{t-1})]^2\right\}$, $\lambda=14400$ | Medium |
| Kalman Filter | State: $Y_{t+1}^P = Y_t^P + \eta_t$, Obs: $Y_t = Y_t^P + \epsilon_t$, $Q/R = 0.1$ | High |

**HP Filter key property:** The FOC yields $(I + \lambda D^\top D)\tau = Y$, a positive definite banded system solvable in $O(T)$ via Cholesky decomposition. Ravn-Uhlig (2002) standard: $\lambda = 14400$ for monthly data.

**Kalman Filter key property:** Steady-state Kalman gain $K_{ss} \approx 0.27$ (with $Q/R=0.1$), meaning 73% weight on prior estimate vs 27% on new observation -- a conservative estimator.

### 1.2 Income Decomposition Output (8D)

| Feature | Formula | Financial Interpretation |
|---------|---------|------------------------|
| `permanent_income_avg` | $\text{mean}(\hat{Y}^P)$ | Long-term stable income level |
| `permanent_income_stability` | $\sigma(\hat{Y}^P) / \mu(\hat{Y}^P)$ | CV of permanent income; low = stable career |
| `permanent_income_growth` | $(\hat{Y}_T^P - \hat{Y}_1^P)/\hat{Y}_1^P$ | Income trajectory for card tier upgrades |
| `permanent_income_trend` | REGR_SLOPE / polyfit | Robust long-term growth direction |
| `transitory_income_avg` | $\text{mean}(\hat{Y}^T)$ | Bonus frequency indicator (should be ~0) |
| `transitory_income_volatility` | $\sigma(\hat{Y}^T)$ | Income uncertainty magnitude |
| `transitory_income_max` | $\max(\hat{Y}^T)$ | Largest bonus event proxy |
| `bonus_frequency` | $\text{count}(\hat{Y}^T > 0.5\hat{Y}^P) / N$ | Fraction of months with large bonuses |

### 1.3 Microeconomic Behavior Features (9D)

**Income Elasticity of Demand:**

$$\epsilon_Y = \frac{\partial Q}{\partial Y} \cdot \frac{Y}{Q} = \frac{d \ln Q}{d \ln Y}$$

- $\epsilon_Y > 1$: luxury goods behavior (spending grows faster than income)
- $0 < \epsilon_Y < 1$: necessity goods behavior
- $\epsilon_Y < 0$: inferior goods (spending decreases with income)

Implemented as arc elasticity: $\hat{\epsilon}_Y = \frac{1}{T}\sum_{t=1}^T \frac{\Delta S_t / S_{t-1}}{\Delta Y_t / Y_{t-1}}$

**Consumption Smoothing (Hall, 1978):**

Under rational expectations + PIH, optimal consumption follows a random walk:

$$C_t = C_{t-1} + \epsilon_t, \quad \epsilon_t \sim WN(0, \sigma^2)$$

Feature: `consumption_smoothing = mu/sigma` (inverse CV, analogous to Sharpe ratio for consumption).

**Time Discounting:**

$$V_0 = \sum_{t=0}^T \beta^t u(C_t), \quad 0 < \beta < 1$$

`discount_rate_proxy = first_half_spending / second_half_spending` serves as a behavioral proxy for $1/\beta$.

**Spending Diversification (Shannon Entropy):**

$$H = -\sum_{i=1}^N s_i \ln(s_i)$$

**Category Concentration (HHI):**

$$\text{HHI} = \sum_i s_i^2$$

Both are special cases of Renyi entropy $H_\alpha = \frac{1}{1-\alpha}\ln(\sum s_i^\alpha)$: Shannon at $\alpha \to 1$, HHI at $\alpha = 2$ (since $H_2 = -\ln(\text{HHI})$). HHI is sensitive to dominant categories; entropy captures tail diversity.

### 1.4 Financial Domain Justification

Economics features provide three advantages over pure statistical features:

1. **Causal structure encoding**: PIH decomposition separates signal (permanent) from noise (transitory), enabling the model to learn behavioral *direction* rather than mere correlation.
2. **Interpretability**: $\epsilon_Y = 1.3$ immediately conveys "luxury consumption tendency" -- directly aligned with XAI requirements.
3. **Domain normalization**: Dimensionless ratios (elasticity, CV, HHI) are invariant to income scale, currency, and price level, reducing feature scaling burden.

**Comparison with statistical features:**

| Aspect | Statistical Feature | Economics Feature |
|--------|-------------------|------------------|
| Income level | Monthly deposit mean | `permanent_income_avg` (noise-removed) |
| Income variability | Monthly deposit std | `transitory_income_volatility` (theoretically decomposed) |
| Consumption variability | Monthly spending CV | `consumption_smoothing` (distance from theoretical optimum) |
| Category distribution | Mode category ratio | `category_hhi` + `spending_diversification` (dual measurement) |
| Scale invariance | Requires normalization | Many dimensionless metrics (natural scale) |

### 1.5 Key References

- Friedman, M. (1957). *A Theory of the Consumption Function*. Princeton UP.
- Hall, R. (1978). Stochastic Implications of the Life Cycle-Permanent Income Hypothesis. *JPE*.
- Hodrick, R. & Prescott, E. (1997). Postwar U.S. Business Cycles. *JMCB*. ($\lambda = 14400$ for monthly data)
- Kalman, R. (1960). A New Approach to Linear Filtering and Prediction Problems. *J. Basic Engineering*.
- Kahneman, D. & Tversky, A. (1979). Prospect Theory. *Econometrica*.
- Ravn, M. & Uhlig, H. (2002). On Adjusting the HP Filter for the Frequency of Observations.

---

## 2. Multidisciplinary Features (24D)

### 2.1 Core Idea: Structural Isomorphism

The multidisciplinary approach applies mathematical frameworks from four scientific disciplines to financial transaction data, justified by *structural isomorphism* -- when the mathematical relationship structure between objects is identical across domains, the formulas capture the same patterns regardless of the surface-level domain.

| Discipline | Dimensions | Pattern Captured | Mathematical Tool |
|-----------|-----------|-----------------|-------------------|
| Chemical Kinetics | 6D | Velocity, acceleration of behavioral change | Arrhenius equation, finite differences |
| SIR Epidemiology | 5D | Adoption diffusion, compartmental transitions | ODE compartmental model |
| Crime Pattern / Routine Activity Theory | 5D | Regularity, burstiness, temporal anomaly | Circular statistics, burstiness index |
| Wave Interference / Spectral Analysis | 8D | Frequency decomposition, phase synchronization | FFT, Hilbert transform, KL divergence |

### 2.2 Chemical Kinetics (6D)

**Arrhenius equation applied to category adoption:**

$$k = A e^{-E_a / RT}$$

Financial interpretation: Category switching frequency ($k$) increases exponentially when entry barriers ($E_a$) decrease or consumer activity ($T$) increases.

**Key features:**

| Feature | Definition | Financial Meaning |
|---------|-----------|------------------|
| `new_category_activation_rate` | New MCC count in last 30d / active MCC in last 30d | Inverse activation energy proxy |
| `spending_half_life` | Median inter-transaction interval (days) | Analogous to chemical half-life $T_{1/2} = \ln 2 / k$ |
| `spending_acceleration` | $f''(t) \approx f(t+\Delta t) - 2f(t) + f(t-\Delta t)$ | 2nd-order finite difference: acceleration (+) or deceleration (-) |
| `dormancy_reactivation_rate` | MCC present in W1, absent W2, reappearing W3 | Catalytic reactivation rate |
| `catalyst_sensitivity` | Early-month avg daily spend / late-month avg daily spend | Payday catalyst elasticity |
| `saturation_proximity` | max_amount / (avg_amount + std_amount) | Proximity to consumption ceiling |

**Spending acceleration** is the discrete 2nd derivative, derived from Taylor expansion: $f(t \pm \Delta t) = f(t) \pm f'(t)\Delta t + \frac{1}{2}f''(t)\Delta t^2 + O(\Delta t^3)$. Adding the two eliminates $f'$, yielding the 2nd-order approximation.

### 2.3 SIR Epidemiology (5D)

**Compartmental model (Kermack & McKendrick, 1927):**

$$\frac{dS}{dt} = -\beta SI, \quad \frac{dI}{dt} = \beta SI - \gamma I, \quad \frac{dR}{dt} = \gamma I$$

Basic reproduction number: $R_0 = \beta / \gamma$. When $R_0 > 1$, adoption spreads (supercritical); when $R_0 < 1$, adoption dies out.

**Financial domain mapping:**

| Compartment | Epidemiology | Financial Interpretation |
|-------------|-------------|------------------------|
| S (Susceptible) | Not yet infected | Population Top-15 MCC categories not yet used by customer |
| I (Infected) | Currently spreading | Categories with recent 30d daily avg frequency > prior period |
| R (Recovered) | Immune/recovered | Categories used previously but inactive in recent 30d |

**Infected classification criterion:**

$$\text{infected} = \begin{cases} 1 & \text{if } \text{recent\_count} > \text{older\_count} \times \frac{30}{L-30} \\ 0 & \text{otherwise} \end{cases}$$

The correction factor $30/(L-30)$ normalizes for period length differences.

**Customer profiling via SIR ratios:**

| Type | S | I | R | Interpretation |
|------|---|---|---|---------------|
| Exploration-ready | High | Low | Low | Large recommendation opportunity space |
| Active adoption | Medium | High | Low | Optimal cross-sell timing |
| Stable usage | Low | Low | Low | Loyal customer |
| Contracting | Low | Low | High | Retention campaign target |

### 2.4 Crime Pattern / Routine Activity Theory (5D)

**Core formula (Cohen & Felson, 1979):**

$$\text{Crime Opportunity} = \text{Motivated Offender} \times \text{Suitable Target} \times \text{Absence of Guardian}$$

The multiplicative structure means any single factor at zero prevents the event -- a bottleneck effect.

**Burstiness (Barabasi, 2005):**

$$B = \frac{\sigma_\tau - \mu_\tau}{\sigma_\tau + \mu_\tau} \in [-1, 1]$$

- $B = -1$: perfectly regular intervals (periodic payments)
- $B = 0$: Poisson process (random)
- $B = +1$: extreme clustering (burst shopping)

**Circular variance** for transaction time analysis:

$$\bar{\mathbf{R}} = \left(\frac{1}{n}\sum \cos\theta_i, \frac{1}{n}\sum \sin\theta_i\right), \quad \text{CV} = 1 - |\bar{\mathbf{R}}|$$

where $\theta = 2\pi h / 24$. Essential because Euclidean distance treats 23:00 and 01:00 as 22 hours apart, while circular statistics correctly computes 2 hours.

### 2.5 Wave Interference / Spectral Analysis (8D)

**KL Divergence (weekday vs weekend spending distributions):**

$$D_{KL}(P\|Q) = \sum P(x) \ln\frac{P(x)}{Q(x)}$$

Always non-negative (Gibbs inequality via Jensen). Measures information loss when approximating distribution $P$ with $Q$.

**Phase Locking Value (from neuroscience functional connectivity):**

$$\text{PLV} = \frac{1}{T}\left|\sum_{t=1}^T e^{j(\phi_x(t) - \phi_y(t))}\right|$$

Measures consistency of phase difference between two category spending rhythms via Hilbert transform.

**Cross-spectral Coherence:**

$$C_{xy}(f) = \frac{|S_{xy}(f)|^2}{S_{xx}(f) \cdot S_{yy}(f)}$$

Frequency-resolved correlation identifying at which periodicities categories synchronize.

**Spectral Entropy** (normalized Shannon entropy of power spectrum): measures predictability of spending periodicity.

### 2.6 Information-Theoretic Justification

The four modules capture near-orthogonal projections of the data:
- Chemical Kinetics: *differential structure* of time (1st, 2nd derivatives)
- Epidemic Diffusion: *state space transition structure* (S -> I -> R)
- Crime Pattern: *statistical texture* of time series (periodicity, clustering, variance)
- Interference: *frequency domain spectral structure* (FFT, coherence, phase)

Cross-module combinations reveal patterns invisible to individual modules: e.g., high `catalyst_sensitivity` + high `burstiness` = payday burst spender (optimal for early-month targeted promotions).

### 2.7 Key References

- Arrhenius, S. (1889). Reaction rates of sucrose inversion.
- Kermack, W. & McKendrick, A. (1927). A Contribution to the Mathematical Theory of Epidemics. *Proc. Royal Society*.
- Cohen, L. & Felson, M. (1979). Routine Activity Approach. *ASR*.
- Barabasi, A.-L. (2005). The origin of bursts and heavy tails in human dynamics. *Nature*.
- Shannon, C. (1948). A Mathematical Theory of Communication. *Bell System Technical Journal*.
- Kullback, S. & Leibler, R. (1951). On Information and Sufficiency.

---

## 3. GMM Clustering Features (22D)

### 3.1 Theoretical Foundation

**Gaussian Mixture Model:**

$$p(\mathbf{x}) = \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

where $\pi_k \geq 0$, $\sum_k \pi_k = 1$, and each component is a multivariate Gaussian:

$$\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) = \frac{1}{(2\pi)^{D/2}|\boldsymbol{\Sigma}_k|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_k)^\top \boldsymbol{\Sigma}_k^{-1}(\mathbf{x}-\boldsymbol{\mu}_k)\right)$$

**Configuration:** $K=20$ clusters, $D=40$ input dimensions, `covariance_type="full"`.

### 3.2 EM Algorithm

**E-Step (posterior responsibility):**

$$\gamma_{nk} = \frac{\pi_k \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_j \pi_j \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}$$

This is direct Bayes' theorem: prior $\pi_k$ combined with likelihood $\mathcal{N}_k$ yields posterior $\gamma_{nk}$.

**M-Step:** Update $\boldsymbol{\mu}_k$, $\boldsymbol{\Sigma}_k$, $\pi_k$ using $\gamma_{nk}$ as weights.

**Convergence guarantee:** Via Jensen inequality ($\ln$ is concave), EM constructs a lower bound (ELBO) that is monotonically non-decreasing. Not guaranteed to reach global optimum; mitigated by `n_init=10`.

### 3.3 Model Selection: BIC

$$\text{BIC} = -2\ln\hat{L} + k\ln(n)$$

Chosen over AIC ($-2\ln\hat{L} + 2k$) because with $n$ in hundreds of thousands, BIC's $k\ln(n)$ penalty more effectively prevents overfitting.

### 3.4 Output Features (22D)

| Feature | Dimensions | Description |
|---------|-----------|-------------|
| `cluster_prob_00` -- `cluster_prob_19` | 20D | Soft assignment probabilities $\gamma_{nk}$ (sum = 1.0) |
| `cluster_id` | 1D | Hard assignment $\arg\max_k \gamma_{nk}$ |
| `cluster_entropy` | 1D | $H_n = -\sum_k \gamma_{nk}\ln(\gamma_{nk} + \epsilon)$ |

Entropy interpretation:
- $H \approx 0$: clear behavioral archetype (confident classification)
- $H = \ln(20) \approx 2.996$: uniform distribution (cold-start / unclassifiable)

### 3.5 GMM vs K-Means: Why Soft Assignment

| Aspect | K-Means | GMM |
|--------|---------|-----|
| Assignment | Hard: one-hot (1 bit) | Soft: probability vector (~4.32 bits max) |
| Boundary customers | Arbitrary assignment, unstable | Natural probability spread across adjacent clusters |
| Cluster shape | Spherical (Euclidean distance) | Ellipsoidal (Mahalanobis distance via full covariance) |
| Uncertainty quantification | None | Entropy-based confidence |
| Gradient compatibility | Discontinuous (argmin) | Continuous (differentiable softmax-like) |
| Downstream use | Single sub-head active | Multiple sub-heads ensembled via $\gamma_{nk}$ weighting |

**Core advantage for PLE architecture:** The `GroupTaskExpertBasket` module uses 20 cluster sub-heads whose outputs are weighted by $\gamma_{nk}$. Soft assignment enables probabilistic ensemble across sub-heads, improving recommendation quality for boundary customers.

**Mahalanobis distance** (used internally): $d_M = \sqrt{(\mathbf{x}-\boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})}$ accounts for feature correlations and scale differences, generalizing Euclidean distance ($\boldsymbol{\Sigma} = \mathbf{I}$).

### 3.6 GMM vs HMM Distinction

| Aspect | GMM (this module) | HMM (separate module) |
|--------|-------------------|----------------------|
| Analysis axis | Cross-sectional (single timepoint) | Temporal (past -> present -> future) |
| Core question | "What *type* is this customer?" | "What *stage* is this customer at?" |
| Time dependency | None (static) | Transition matrix $\mathbf{A}$ captures dynamics |
| Output | 22D in main 734D tensor (Domain) | 48D separate input; 5D summary in main tensor |
| PLE role | GroupTaskExpertBasket soft routing | HMM Triple-Mode Projector |

### 3.7 Key References

- Pearson, K. (1894). First mixture model (crab forehead ratio data).
- Dempster, A., Laird, N., & Rubin, D. (1977). Maximum Likelihood from Incomplete Data via the EM Algorithm. *JRSS-B*.
- Schwarz, G. (1978). Estimating the Dimension of a Model. *Annals of Statistics* (BIC).
- Bishop, C. (2006). *Pattern Recognition and Machine Learning*, Ch. 9.

---

## 4. adaTT (Adaptive Task-aware Transfer)

### 4.1 Motivation: Negative Transfer in Multi-Task Learning

With 16 simultaneous tasks (CTR, CVR, Churn, NBA, etc.) sharing expert parameters, gradient conflicts cause negative transfer: updating shared parameters to improve one task degrades another. Fixed-tower MTL has three fundamental limitations:

1. **Unidirectional sharing**: No mechanism to detect or control task interference
2. **No interaction measurement**: Tasks compete implicitly for shared parameters
3. **No temporal adaptation**: Task relationships change during training

### 4.2 Core Mechanism: Gradient Cosine Similarity

$$\cos(\theta_{i,j}) = \frac{\mathbf{g}_i \cdot \mathbf{g}_j}{\|\mathbf{g}_i\| \cdot \|\mathbf{g}_j\|}$$

where $\mathbf{g}_i = \nabla_\theta \mathcal{L}_i$ is the gradient of task $i$'s loss w.r.t. shared expert parameters.

**Why cosine (not Euclidean):**
1. **Scale invariance**: Task losses differ by orders of magnitude; cosine compares direction only
2. **Interpretable range**: $[-1, 1]$ maps directly to positive/negative transfer
3. **Efficient computation**: Single matrix multiply $\hat{G}\hat{G}^\top$ computes all $n^2$ similarities

Implementation: Stack task gradients into $\mathbf{G} \in \mathbb{R}^{n \times d}$, L2-normalize rows, compute $\hat{\mathbf{G}}\hat{\mathbf{G}}^\top$.

### 4.3 EMA Stabilization

$$\mathbf{A}_t = \alpha \cdot \mathbf{A}_{t-1} + (1-\alpha) \cdot \cos(\theta_t)$$

with $\alpha = 0.9$ (effective window $\approx 10$ observations). Equivalent to IIR 1st-order low-pass filter $H(z) = (1-\alpha)/(1-\alpha z^{-1})$ that removes high-frequency batch noise while preserving true task relationship trends.

### 4.4 Transfer-Enhanced Loss

$$\mathcal{L}_i^{\text{adaTT}} = \mathcal{L}_i + \lambda \cdot \sum_{j \neq i} w_{i \to j} \cdot \mathcal{L}_j$$

with $\lambda = 0.1$ (10% influence from other tasks) and `max_transfer_ratio = 0.5` (transfer loss cannot exceed 50% of original loss).

**Gradient impact:**

$$\nabla_\theta \mathcal{L}_i^{\text{adaTT}} = \nabla_\theta \mathcal{L}_i + \lambda \sum_{j \neq i} w_{i \to j} \nabla_\theta \mathcal{L}_j$$

The second term is a correction vector steering shared parameters toward directions beneficial for multiple tasks.

### 4.5 Transfer Weight Computation (4-stage pipeline)

$$\mathbf{R} = (\mathbf{W} + \mathbf{A}) \cdot (1-r) + \mathbf{P} \cdot r$$
$$\mathbf{R}_{i,j} \leftarrow 0 \quad \text{if } \mathbf{A}_{i,j} < \tau_{\text{neg}}$$
$$\mathbf{R}_{i,i} = 0$$
$$w_{i \to j} = \text{softmax}(\mathbf{R}_{i,j} / T)$$

where:
- $\mathbf{W}$: learnable transfer weights (`nn.Parameter`, initialized to 0)
- $\mathbf{A}$: EMA affinity matrix
- $\mathbf{P}$: Group Prior matrix (domain knowledge)
- $r$: Prior blend ratio (annealed from 0.5 to 0.1)
- $\tau_{\text{neg}} = -0.1$: Negative transfer threshold
- $T = 1.0$: Softmax temperature

**Softmax temperature interpretation** (from Boltzmann distribution $p_i \propto e^{-E_i/k_BT}$):
- $T \to 0^+$: concentrates on single best task (one-hot)
- $T \to \infty$: uniform distribution $1/n$
- $T = 1.0$: balanced selection with competitive attention

### 4.6 Group Prior

Four task groups defined by domain knowledge:

| Group | Tasks | Intra-strength | Business Meaning |
|-------|-------|---------------|-----------------|
| engagement | ctr, cvr, engagement, uplift | 0.8 | Customer engagement/conversion |
| lifecycle | churn, retention, life_stage, ltv | 0.7 | Customer lifecycle |
| value | balance_util, channel, timing | 0.6 | Customer value/behavior |
| consumption | nba, spending_category, consumption_cycle, spending_bucket, merchant_affinity, brand_prediction | 0.7 | Consumption pattern |

Inter-group strength: 0.3 (cross-group transfer kept conservative).

**Prior Blend Annealing (Bayesian interpretation):**

$$r(e) = r_{\text{start}} - (r_{\text{start}} - r_{\text{end}}) \cdot \min\left(\frac{e - e_{\text{warmup}}}{e_{\text{freeze}} - e_{\text{warmup}}}, 1.0\right)$$

$r: 0.5 \to 0.1$ implements prior-to-posterior transition: early training relies on domain knowledge (prior), later training trusts observed gradient affinity (likelihood).

### 4.7 3-Phase Schedule

| Phase | Period | Behavior | Purpose |
|-------|--------|----------|---------|
| **Warmup** | Epoch 0 -- `warmup_epochs` | Compute affinity only, no transfer loss | Accumulate stable affinity data |
| **Dynamic** | `warmup_epochs` -- `freeze_epoch` | Active transfer with annealing prior | Learn and apply task relationships |
| **Frozen** | `freeze_epoch` -- end | Fixed transfer weights (detached) | Stabilize fine-tuning, remove gradient overhead |

Validation: `freeze_epoch > warmup_epochs` enforced to prevent skipping Phase 2 entirely.

### 4.8 Negative Transfer Detection and Blocking

$$\mathbf{R}_{i,j} = \begin{cases} \mathbf{R}_{i,j} & \text{if } \mathbf{A}_{i,j} > \tau_{\text{neg}} \\ 0 & \text{otherwise} \end{cases}$$

Threshold $\tau_{\text{neg}} = -0.1$ (not 0) to allow weak negative correlations (likely noise) while blocking clear antagonistic gradients.

Diagnostic API: `detect_negative_transfer()` returns dict mapping each task to its antagonistic counterparts, e.g., `{"churn": ["ctr", "engagement"]}`.

### 4.9 Connection to Attention and Conditional Computation

adaTT performs *task-space attention* analogous to Transformer self-attention:

| Role | Transformer Self-Attention | adaTT Task Transfer |
|------|--------------------------|-------------------|
| Query | Current token's query | Current task's gradient direction |
| Key | Other tokens' response potential | Other tasks' gradient directions |
| Similarity | $QK^\top / \sqrt{d_k}$ | Gradient cosine similarity |
| Normalization | softmax | softmax (temperature $T$) |
| Value | Other tokens' information | Other tasks' loss values |
| Output | Weighted context | Transfer loss |

adaTT is also related to **Hypernetworks** (Ha et al., 2017) but uses observed gradients as conditioning signal rather than learned task embeddings, enabling zero-delay adaptation to changing task relationships.

### 4.10 2-Phase Training Integration

- **Phase 1 (Shared Expert Pretrain)**: adaTT active -- gradient extraction and transfer loss applied over `shared_expert_epochs` (default 15)
- **Phase 2 (Cluster Finetune)**: adaTT disabled -- Shared experts frozen, gradient extraction meaningless; only cluster-specific sub-heads trained for `cluster_finetune_epochs` (default 8)

### 4.11 Comparison with Alternative MTL Approaches

| Method | Mechanism | adaTT Advantage |
|--------|-----------|----------------|
| Fixed Weighting (Kendall et al., 2018) | Manual task weights | adaTT measures affinity dynamically |
| GradNorm (Chen et al., 2018) | Gradient magnitude balancing | adaTT uses direction, not magnitude |
| PCGrad (Yu et al., 2020) | Project conflicting gradients | adaTT selectively transfers positive knowledge |
| Nash-MTL (Navon et al., 2022) | Nash bargaining for Pareto direction | adaTT is computationally lighter ($O(n^2 d)$ vs optimization) |
| CAGrad (Liu et al., 2021) | Maximize worst-case gradient alignment | adaTT separates measurement from application (modular) |

### 4.12 Key References

- Caruana, R. (1997). Multitask Learning. *Machine Learning*.
- Tang, H. et al. (2020). Progressive Layered Extraction (PLE). *RecSys*.
- Fifty, C. et al. (2021). Task Affinity Grouping (TAG). *ICML*.
- Yu, T. et al. (2020). Gradient Surgery for Multi-Task Learning (PCGrad). *NeurIPS*.
- Chen, Z. et al. (2018). GradNorm. *ICML*.
- Navon, A. et al. (2022). Nash-MTL. *ICML*.
- Liu, B. et al. (2021). Conflict-Averse Gradient Descent (CAGrad). *NeurIPS*.
- Ha, D. et al. (2017). HyperNetworks. *ICLR*.
- Bengio, Y. et al. (2013). Conditional Computation.
- Fedus, W. et al. (2022). Switch Transformer.

---

## 5. Cross-Module Integration Summary

### 5.1 Feature Tensor Composition (734D)

| Feature Group | Dimensions | Relevant Modules in This Document |
|--------------|-----------|----------------------------------|
| Base (RFM, Transaction, Temporal, Category) | 238D | -- |
| Multi-source (Deposit, Credit, Investment, Digital) | 91D | -- |
| Extended-source (Insurance, Refund, Consultation) | 84D | -- |
| **Domain (TDA + GMM + Mamba + Economics)** | **159D** | **GMM 22D, Economics 17D** |
| Model-derived (HMM + Bandit + LNN) | 27D | -- |
| **Multidisciplinary** | **24D** | **Chemical Kinetics 6D, Epidemic 5D, Crime 5D, Interference 8D** |
| Merchant Hierarchy | 21D | -- |
| **Total** | **644D base + 90D raw power-law = 734D** | |

### 5.2 Pipeline Dependencies

```
Economics (17D) --> GMM (22D) --> 734D Integration --> PLE-Cluster-adaTT
                                                         |
Multidisciplinary (24D) -------------------------------->|
                                                         |
                                              adaTT manages 16-task transfer
```

GMM's 40D input includes 4 Economics features (`permanent_income_avg`, `transitory_income_volatility`, `income_elasticity`, `spending_risk`), establishing a strict DAG dependency: Economics -> GMM -> Integration.

### 5.3 Expert Routing

- **Economics 17D + GMM 22D**: Part of Domain Features (159D) -> routed to Domain Experts in PLE
- **GMM cluster probabilities**: Drive `GroupTaskExpertBasket` 20 sub-head soft routing
- **Multidisciplinary 24D**: Part of main 734D tensor -> fed to 3 Shared Experts (DeepFM, Causal, OT)
- **adaTT**: Operates at loss level, modulating how 16 task towers influence each other via gradient-based affinity
