# Expert Theory Part 2: Temporal, Causal-OT, and HMM Experts

> **Note**: This document uses on-premises dimensions (734D features, 16 tasks).
> For the AWS v1 paper version (349D, 13 tasks), see the v1 Zenodo preprint
> (DOI: 10.5281/zenodo.19621884) or the feature-router table in
> paper/01_architecture_positioning.md.

Extracted from technical reference documents (v3.14/v3.2, 2026-03-05) for arxiv paper.

---

## 1. Temporal Ensemble Expert

### 1.1 Core Architecture

Three-model ensemble capturing multi-resolution temporal patterns from financial transaction sequences:

| Model | Temporal Pattern | Mechanism | Complexity |
|-------|-----------------|-----------|------------|
| **Mamba (SSM)** | Long-range sequential dependencies | Selective State Space (S6) | O(L) linear |
| **LNN** | Irregular time intervals | Adaptive time-constant ODE | O(1) single step |
| **PatchTST** | Global periodicity | Patch-level self-attention | O((L/P)^2) |

**Design rationale -- time series decomposition:**
The three models correspond to the classical decomposition y(t) = T(t) + S(t) + R(t):
- Mamba captures **trend** (long-term directional shifts via selective memory)
- PatchTST captures **seasonality** (periodic patterns via global attention across patches)
- LNN captures **residual** (irregular dynamics via adaptive time constants)

### 1.2 Mamba (Selective State Space Model)

**Reference:** Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (NeurIPS 2023)

**Continuous SSM:**
$$\frac{d\mathbf{x}}{dt} = \mathbf{A}\mathbf{x} + \mathbf{B}u, \quad y = \mathbf{C}\mathbf{x} + \mathbf{D}u$$

**ZOH Discretization:**
$$\bar{\mathbf{A}} = \exp(\Delta \cdot \mathbf{A}), \quad \bar{\mathbf{B}} \approx \Delta \cdot \mathbf{B}$$

**Discrete recurrence:**
$$\mathbf{h}_t = \bar{\mathbf{A}} \cdot \mathbf{h}_{t-1} + \bar{\mathbf{B}} \cdot \mathbf{x}_t, \quad \mathbf{y}_t = \mathbf{C}_t \cdot \mathbf{h}_t$$

**S6 Selective Mechanism (key innovation):**
$$\Delta = \text{softplus}(\mathbf{W}_\Delta \cdot \mathbf{x} + \mathbf{b}_\Delta)$$
$$\mathbf{B} = \mathbf{W}_B \cdot \mathbf{x}, \quad \mathbf{C} = \mathbf{W}_C \cdot \mathbf{x}$$

Unlike LTI systems where A, B, C are input-independent constants, S6 makes Delta, B, C **input-dependent**, enabling content-aware processing. Large Delta values strongly encode the current input into state; small Delta values preserve previous state (selective memory/forgetting).

**Financial domain justification:**
- Large transactions yield large Delta (strongly remembered)
- Small routine purchases yield small Delta (quickly forgotten as background)
- This selective mechanism naturally models the heterogeneous importance of financial events

**Project specs:**
- Transaction Mamba: d_model=128, d_input=16 (card 8D + deposit 8D), d_state=16, seq_len=180
- Session Mamba: d_model=64, d_input=8, d_state=16, seq_len=90
- A matrix initialized as HiPPO-style diagonal [-1, -2, ..., -N] for multi-scale memory decay
- Total Mamba output: 128 + 64 = 192D

### 1.3 Liquid Neural Network (LNN)

**Reference:** Hasani et al., "Liquid Time-constant Networks" (AAAI 2021)

**Core ODE:**
$$\frac{d\mathbf{h}}{dt} = \frac{-\mathbf{h} + f(\mathbf{x}, \mathbf{h})}{\tau(\mathbf{x}, \mathbf{h})}$$

where:
- $-\mathbf{h}$: leak/decay term (forgetting toward zero without input)
- $f(\mathbf{x}, \mathbf{h}) = \tanh(\mathbf{W}_f[\mathbf{x};\mathbf{h}] + \mathbf{b}_f)$: target state (driving force)
- $\tau(\mathbf{x}, \mathbf{h}) = \text{Softplus}(\text{MLP}([\mathbf{x};\mathbf{h}])) + 0.1$: adaptive time constant

**Euler discretization:**
$$\mathbf{h}_{t+1} = \mathbf{h}_t + \Delta t \cdot \frac{-\mathbf{h}_t + f(\mathbf{x}_t, \mathbf{h}_t)}{\tau(\mathbf{x}_t, \mathbf{h}_t)}$$

**SingleStep mode design:**
The project uses LNN in SingleStep mode -- only processing Mamba's final hidden state with one ODE step, not the full sequence. Rationale: Mamba already captures full sequence patterns at O(L); LNN adds **time-scale correction** without redundant sequence processing.

**Financial domain justification:**
Financial transaction intervals are highly irregular:
- Intraday multiple transactions: Delta_t ~ 0.01 days
- Weekend gaps: Delta_t = 2 days  
- Long dormancy: Delta_t > 30 days

Fixed-tau RNN/LSTM treats all intervals identically. Adaptive tau automatically adjusts: small tau for active trading periods (fast response), large tau for dormancy (state preservation).

**Project specs:**
- LNN txn: input_dim=128 (Mamba output), hidden_dim=64
- LNN session: input_dim=64, hidden_dim=32
- Total LNN output: 64 + 32 = 96D

### 1.4 PatchTST (Patch Time Series Transformer)

**Reference:** Nie et al., "A Time Series is Worth 64 Words" (ICLR 2023)

**Patch embedding:**
$$\mathbf{p}_i = \mathbf{W}_\text{proj} \cdot \text{flatten}(\mathbf{x}_{[(i-1)P+1 : iP]}) + \mathbf{b}_\text{proj}$$

With P=16, a 180-step transaction sequence becomes 12 patches (tokens), reducing attention cost from O(180^2) to O(12^2) = 144.

**Multi-head self-attention:**
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

**Financial domain justification:**
Patch size 16 corresponds to ~2 weeks, naturally aligning with salary cycles (biweekly/monthly). Each patch captures local patterns (daily spending within 2 weeks), while inter-patch attention captures global periodicity (monthly salary spikes, quarterly bonuses).

**Project specs:**
- PatchTST txn: d_model=64, nhead=4, num_layers=2, 12 patches
- PatchTST session: d_model=32, nhead=2, num_layers=2, 6 patches
- Sinusoidal positional encoding (sufficient for 6-12 patches)
- AdaptiveAvgPool1d for fixed-size output
- Total PatchTST output: 64 + 32 = 96D

### 1.5 Ensemble Gating

**Gate architecture:**
$$\mathbf{g} = \text{Softmax}(\mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \cdot \mathbf{z}_\text{cat} + \mathbf{b}_1) + \mathbf{b}_2) \in \mathbb{R}^3$$
$$\mathbf{y} = \sum_{i=1}^{3} g_i \cdot \text{Proj}_i(\mathbf{z}_i) \in \mathbb{R}^{64}$$

where z_cat is the concatenation of all three model outputs (192+96+96 = 384D), and Proj_i projects each model's output to a common 64D space.

**Gate entropy monitoring:**
$$H(\mathbf{g}) = -\sum_{i=1}^{3} g_i \log_2(g_i)$$

- Maximum entropy: log_2(3) ~ 1.585 bits (uniform distribution)
- Gate collapse threshold: H < 0.3 bits (one model dominates, others stop learning)

**Design: Mamba->LNN serial + PatchTST independent:**
- Mamba->LNN serial: Mamba learns sequence patterns, LNN adds time-scale correction on final state
- PatchTST receives raw sequences independently, ensuring ensemble diversity through input separation

### 1.6 Comparison with Alternatives

| Generation | Approach | Limitation | Our Solution |
|-----------|----------|------------|-------------|
| 1st | ARIMA, Exponential Smoothing | Linear assumption, manual differencing | - |
| 2nd | LSTM, GRU | O(L) sequential bottleneck, vanishing gradient | - |
| 3rd | Transformer (Self-Attention) | O(L^2) complexity, weak ordering | Mamba, PatchTST |
| 4th (ours) | SSM + ODE + Patch Transformer Ensemble | Model complexity, gate collapse risk | Entropy monitoring |

---

## 2. Causal Expert (NOTEARS-based)

### 2.1 Core Architecture

**Reference:** Zheng et al., "DAGs with NO TEARS" (NeurIPS 2018)

Three-stage pipeline: Feature Compression -> SCM Causal Intervention -> Causal Encoding

**Feature Compressor:**
$$\mathbf{z} = \text{Compressor}(\mathbf{x}): \mathbb{R}^{644} \to \mathbb{R}^{128} \to \mathbb{R}^{32}$$

Reduces 644D normalized features to 32 causal variables, preventing the DAG adjacency matrix from exploding to 644^2 ~ 410K entries.

### 2.2 SCM (Structural Causal Model) Intervention

$$\hat{\mathbf{z}} = \mathbf{z} + \mathbf{z}(\mathbf{W} \circ \mathbf{W})$$

where:
- $\mathbf{W} \in \mathbb{R}^{32 \times 32}$: learnable weighted adjacency matrix (nn.Parameter)
- $\mathbf{W} \circ \mathbf{W}$: element-wise (Hadamard) square, guaranteeing **non-negative** causal strengths
- $W_{i,j}^2$: causal influence strength from variable j to variable i
- Residual connection ($\mathbf{z} +$) preserves original information while adding causal adjustment

**Financial domain justification:**
Correlation-based recommendations are vulnerable to confounders. Example: "Premium card holders have high travel insurance uptake" is confounded by income level. The SCM learns directed causal structure so that recommendations are based on **interventional** effects, not mere correlations.

### 2.3 NOTEARS Acyclicity Constraint

$$h(\mathbf{W}) = \text{tr}(e^{\mathbf{W} \circ \mathbf{W}}) - d = 0$$

**Mathematical interpretation:**
The (i,i) diagonal element of $e^{\mathbf{M}}$ sums all weighted paths from node i back to itself. If the graph is a DAG (no cycles), no such return paths exist, so $e^{\mathbf{M}}_{i,i} = 1$ (only the identity matrix contribution remains), yielding $\text{tr}(e^{\mathbf{M}}) = d$.

**Taylor 10-term approximation:**
$$e^{\mathbf{M}} \approx \sum_{k=0}^{9} \frac{\mathbf{M}^k}{k!}$$

This detects cycles up to length 10. For a 32-node DAG, cycles longer than 10 hops are practically impossible.

### 2.4 DAG Regularization Loss

$$\mathcal{L}_\text{DAG} = \lambda_\text{acyclic} \cdot h(\mathbf{W}) + \lambda_\text{sparse} \cdot \|\mathbf{W} \circ \mathbf{W}\|_1$$

| Hyperparameter | Default | Role |
|---------------|---------|------|
| dag_lambda | 0.01 | Acyclicity constraint strength |
| sparsity_lambda | 0.001 | Edge sparsity (L1 on adjacency) |
| n_causal_vars | 32 | Number of causal nodes |

**Warning:** dag_lambda > 0.1 causes W to collapse to zero matrix (DAG penalty dominates task loss, Expert degenerates to identity function).

### 2.5 Comparison: NOTEARS Paper vs Implementation

| Aspect | Paper (Zheng et al.) | Our Implementation |
|--------|---------------------|-------------------|
| Purpose | DAG structure learning from observations | Feature causal relationship learning within Expert |
| Acyclicity enforcement | Augmented Lagrangian (strict equality) | Simple penalty method: lambda * h(W) |
| Adjacency matrix | W (sign-agnostic) | W o W (non-negative forced) |
| Learning | Independent optimization | End-to-end MTL joint training |
| Output | DAG adjacency matrix | 64D causal representation + DAG (visualization) |

---

## 3. Optimal Transport Expert (Sinkhorn-based)

### 3.1 Core Architecture

**Reference:** Cuturi, "Sinkhorn Distances: Lightspeed Computation of Optimal Transport" (NeurIPS 2013)

### 3.2 Distribution Projection

$$\boldsymbol{\mu} = \text{softmax}(\text{DistProjector}(\mathbf{x})) \in \Delta^{32}$$

Transforms 644D features into a probability simplex, representing each customer's feature profile as a discrete distribution over 32 latent categories.

### 3.3 Learnable Reference Distributions

$$\boldsymbol{\nu}_k = \text{softmax}(\boldsymbol{\ell}_k) \in \Delta^{32}, \quad k = 1, \ldots, 16$$

16 learnable prototypical customer profiles (nn.Parameter), initialized as randn(16, 32) * 0.1.

### 3.4 PSD Cost Matrix

$$\mathbf{C} = \mathbf{M}^\top \mathbf{M} \in \mathbb{R}^{32 \times 32}$$

Guarantees positive semi-definiteness: $\mathbf{x}^\top(\mathbf{M}^\top\mathbf{M})\mathbf{x} = \|\mathbf{M}\mathbf{x}\|^2 \geq 0$. The cost matrix is **learnable**, allowing task-optimized semantic distances between distribution support points.

### 3.5 Entropy-Regularized Optimal Transport

**Kantorovich problem with entropic regularization:**
$$\min_{\mathbf{P} \in \mathcal{U}(\boldsymbol{\mu}, \boldsymbol{\nu})} \langle \mathbf{P}, \mathbf{C} \rangle + \epsilon \cdot H(\mathbf{P})$$

where:
- $\mathcal{U}(\boldsymbol{\mu}, \boldsymbol{\nu}) = \{\mathbf{P} \geq 0 : \mathbf{P}\mathbf{1} = \boldsymbol{\mu}, \mathbf{P}^\top\mathbf{1} = \boldsymbol{\nu}\}$
- $H(\mathbf{P}) = -\sum_{i,j} P_{i,j} \log P_{i,j}$: entropy regularization
- $\epsilon = 0.1$: regularization coefficient

Entropy regularization makes the problem **strictly convex** (unique solution, guaranteed convergence).

### 3.6 Log-Domain Sinkhorn Algorithm

$$\mathbf{u}_\text{new} = \log \boldsymbol{\mu} - \text{logsumexp}(-\mathbf{C}/\epsilon + \mathbf{v})$$
$$\mathbf{v}_\text{new} = \log \boldsymbol{\nu} - \text{logsumexp}(-\mathbf{C}^\top/\epsilon + \mathbf{u})$$

Log-domain computation prevents floating-point underflow when epsilon is small. 10 iterations suffice for practical convergence.

### 3.7 Wasserstein Distance Vector

$$\mathbf{w} = [W(\boldsymbol{\mu}, \boldsymbol{\nu}_1), W(\boldsymbol{\mu}, \boldsymbol{\nu}_2), \ldots, W(\boldsymbol{\mu}, \boldsymbol{\nu}_{16})] \in \mathbb{R}^{16}$$

where $W(\boldsymbol{\mu}, \boldsymbol{\nu}_k) = \langle \mathbf{P}, \mathbf{C} \rangle_F = \sum_{i,j} P_{i,j} \cdot C_{i,j}$

This creates a **distributional coordinate system** -- each customer is positioned by distances to 16 reference prototypes.

### 3.8 Wasserstein Encoder

$$\mathbf{o} = \text{WassersteinEncoder}(\mathbf{w}): \mathbb{R}^{16} \to \mathbb{R}^{128} \to \mathbb{R}^{64}$$

### 3.9 Financial Domain Justification

**Why Wasserstein over KL divergence or Euclidean distance:**
- KL divergence is undefined when distributions have non-overlapping support
- Euclidean distance ignores the geometric structure of the underlying feature space
- Wasserstein distance reflects the **geometry** of the ground metric space -- "Seoul-Incheon" is closer than "Seoul-Busan" even when probability mass doesn't overlap

**Financial application:** Wasserstein distance quantifies "how different is this customer's consumption pattern from a typical travel-type/saving-type/dining-type profile, and **which categories need to shift in which direction** to match?"

### 3.10 Comparison: Cuturi Paper vs Implementation

| Aspect | Paper (Cuturi 2013) | Our Implementation |
|--------|---------------------|-------------------|
| Distributions | Fixed discrete/continuous | Learned softmax distributions (32D) |
| Cost matrix | Fixed (Euclidean, etc.) | Learnable PSD: M^T M |
| Reference distributions | Single target | 16 learnable prototypes |
| Sinkhorn domain | Standard (matrix multiply) | Log-domain (numerical stability) |
| Iterations | Until convergence | Fixed 10 |
| Output | Scalar OT distance | 16D distance vector -> 64D encoding |

---

## 4. Causal + OT Expert Synergy

### 4.1 Complementary Perspectives on Same Input

Both experts receive the same 644D normalized features but extract fundamentally different mathematical structures:

| Expert | Extracts from 644D | Mathematical Property | Unique Contribution |
|--------|-------------------|----------------------|-------------------|
| **DeepFM** | Symmetric feature interactions $\langle \mathbf{v}_i, \mathbf{v}_j \rangle$ | Exchangeable ($i \leftrightarrow j$) | Explicit 2nd-order cross in O(nk) |
| **Causal** | Directional causation $W_{i,j}^2$ | Asymmetric, acyclic (DAG) | Confounder removal, causal direction |
| **OT** | Customer-prototype distribution distance $W(\mu, \nu_k)$ | Distance function (metric) | Geometric positioning of customer distributions |

### 4.2 Why Separate Experts (Not Merged)

1. **Gradient interference prevention:** NOTEARS acyclicity constraint ($\text{tr}(e^{W \circ W}) = d$) and Sinkhorn entropy regularization ($\epsilon H(P)$) have completely different loss surface geometries
2. **Independent CGC Gate selection:** Task-specific weighting -- e.g., Churn tasks weight Causal higher; Cross-sell tasks weight OT higher
3. **Modular replaceability:** Causal Expert can swap NOTEARS -> GES/PC; OT Expert can swap Sinkhorn -> Sliced Wasserstein independently

---

## 5. HMM Triple-Mode Features (48D)

### 5.1 Core Architecture

Three parallel Hidden Markov Models capturing different temporal scales of customer behavior:

| Mode | States | Time Scale | Target Tasks |
|------|--------|-----------|-------------|
| **Journey** (AICRA) | 5 (Awareness/Consideration/Purchase/Retention/Advocacy) | Days/weeks | CTR, CVR |
| **Lifecycle** | 5 (New/Growing/Mature/At_Risk/Churned) | Months/years | Churn, Retention |
| **Behavior** | 6 (Dormant/Conservative/Routine/Exploratory/Splurge/Investor) | Monthly patterns | NBA, balance_util |

Each mode outputs 16D = state_probs + meta_features + ODE_dynamics. Total: 48D separate input to PLE.

### 5.2 Gaussian HMM Formulation

**Model parameters:** $\lambda = (\pi, A, B)$

- $\pi_i = P(q_1 = S_i)$: initial state probabilities
- $a_{ij} = P(q_{t+1} = S_j | q_t = S_i)$: state transition probabilities
- $b_j(\mathbf{o}) = \mathcal{N}(\mathbf{o}; \boldsymbol{\mu}_j, \Sigma_j)$: Gaussian emission probabilities

**Observation vector** (3D):
$$\mathbf{o}_t = [\ln(\text{txn\_amount} + 1), \ln(\text{txn\_count} + 1), \text{mcc\_diversity}]$$

### 5.3 Three Fundamental Algorithms

**Forward variable:**
$$\alpha_t(i) = P(\mathbf{o}_1, \ldots, \mathbf{o}_t, q_t = S_i | \lambda)$$
$$\alpha_{t+1}(j) = \left[\sum_{i=1}^{N} \alpha_t(i) \cdot a_{ij}\right] \cdot b_j(\mathbf{o}_{t+1})$$

**Backward variable:**
$$\beta_t(i) = P(\mathbf{o}_{t+1}, \ldots, \mathbf{o}_T | q_t = S_i, \lambda)$$
$$\beta_t(i) = \sum_{j=1}^{N} a_{ij} \cdot b_j(\mathbf{o}_{t+1}) \cdot \beta_{t+1}(j)$$

**State posterior (feature output):**
$$\gamma_t(i) = P(q_t = S_i | \mathbf{O}, \lambda) = \frac{\alpha_t(i) \cdot \beta_t(i)}{\sum_{j=1}^{N} \alpha_t(j) \cdot \beta_t(j)}$$

**Transition posterior (for learning):**
$$\xi_t(i,j) = \frac{\alpha_t(i) \cdot a_{ij} \cdot b_j(\mathbf{o}_{t+1}) \cdot \beta_{t+1}(j)}{\sum_i \sum_j \alpha_t(i) \cdot a_{ij} \cdot b_j(\mathbf{o}_{t+1}) \cdot \beta_{t+1}(j)}$$

**M-step parameter updates:**
$$\hat{a}_{ij} = \frac{\sum_{t=1}^{T-1} \xi_t(i,j)}{\sum_{t=1}^{T-1} \gamma_t(i)}, \quad \hat{\mu}_j = \frac{\sum_{t=1}^{T} \gamma_t(j) \cdot \mathbf{o}_t}{\sum_{t=1}^{T} \gamma_t(j)}$$

**Viterbi decoding (MAP state sequence):**
$$\delta_t(j) = \max_{q_1, \ldots, q_{t-1}} P(q_1, \ldots, q_t = S_j, \mathbf{o}_1, \ldots, \mathbf{o}_t | \lambda)$$
$$\delta_t(j) = \max_{1 \leq i \leq N} [\delta_{t-1}(i) \cdot a_{ij}] \cdot b_j(\mathbf{o}_t)$$

Complexity: O(N^2 T) via dynamic programming, avoiding exhaustive N^T enumeration.

### 5.4 Transition Matrices (Domain Expert Initialized)

**Journey mode:**
$$A_\text{journey} = \begin{pmatrix} 0.60 & 0.30 & 0.08 & 0.015 & 0.005 \\ 0.10 & 0.50 & 0.30 & 0.08 & 0.02 \\ 0.05 & 0.10 & 0.50 & 0.30 & 0.05 \\ 0.02 & 0.05 & 0.15 & 0.60 & 0.18 \\ 0.01 & 0.02 & 0.07 & 0.20 & 0.70 \end{pmatrix}$$

**Lifecycle mode:**
$$A_\text{lifecycle} = \begin{pmatrix} 0.40 & 0.45 & 0.10 & 0.04 & 0.01 \\ 0.05 & 0.50 & 0.35 & 0.08 & 0.02 \\ 0.01 & 0.10 & 0.70 & 0.15 & 0.04 \\ 0.02 & 0.05 & 0.15 & 0.50 & 0.28 \\ 0.05 & 0.03 & 0.02 & 0.10 & 0.80 \end{pmatrix}$$

**Behavior mode (6 states):**
$$A_\text{behavior} = \begin{pmatrix} 0.70 & 0.15 & 0.08 & 0.04 & 0.02 & 0.01 \\ 0.10 & 0.60 & 0.20 & 0.05 & 0.03 & 0.02 \\ 0.05 & 0.15 & 0.55 & 0.15 & 0.07 & 0.03 \\ 0.03 & 0.10 & 0.20 & 0.50 & 0.12 & 0.05 \\ 0.05 & 0.15 & 0.25 & 0.15 & 0.35 & 0.05 \\ 0.02 & 0.08 & 0.15 & 0.10 & 0.05 & 0.60 \end{pmatrix}$$

These are domain-expert initial values refined by Baum-Welch EM (n_iter=200, tol=1e-2).

### 5.5 Output Feature Structure (Per Mode, 16D)

| Feature | Dimensions | Description |
|---------|-----------|-------------|
| state_prob_0 .. state_prob_{N-1} | 5D or 6D | Posterior state probabilities gamma_t(i) |
| state_duration | 1D | Consecutive timesteps in current dominant state |
| transition_stability | 1D | 1 - state_change_rate |
| transition_entropy | 1D | Shannon entropy of transition pairs, normalized by log(N^2) |
| dominant_state | 1D | Mode (most frequent) state index |
| state_change_rate | 1D | State changes / (T-1) |
| **ODE Dynamics** (v3.2.0) | 6D | Extracted from Viterbi state trajectory |

**ODE Dynamics Bridge (6D):**

| Feature | Meaning | Computation |
|---------|---------|-------------|
| ode_velocity | Mean trajectory speed | mean(\|Delta q_t\|) |
| ode_acceleration | Rate of change of change | mean(\|Delta^2 q_t\|) |
| ode_lyapunov | Instability ratio (2nd half / 1st half) | Sigma_2nd / (Sigma_1st + eps) |
| ode_cycle_period | Periodicity detection | autocorrelation peak / max_lag |
| ode_attractor | Terminal state concentration | max(f_tail) / max(f_overall) |
| ode_trajectory_len | Total trajectory distance | sum(\|Delta q_t\|) / T |

### 5.6 HMM vs GMM: Dynamic vs Static Segmentation

| Aspect | HMM (this module) | GMM (separate module) |
|--------|-------------------|----------------------|
| Analysis axis | Temporal (past -> present -> future) | Cross-sectional (single time point) |
| Core question | "What *stage* is this customer in?" | "What *type* is this customer?" |
| Time dependency | Yes (transition matrix A) | None (static) |
| Output | 48D separate input + 5D summary | 22D in main 734D tensor |
| PLE routing | HMM Triple-Mode Projector (dedicated path) | GroupTaskExpertBasket soft routing |

### 5.7 Stationary Distribution Analysis

The stationary distribution pi* satisfying pi* A = pi* reveals long-term customer population equilibrium. The second-largest eigenvalue |lambda_2| of A determines convergence speed -- when |lambda_2| is close to 1, customer states have long memory, meaning HMM features retain predictive power over extended periods.

### 5.8 Financial Domain Justification

**Why HMM over rule-based classification:**
- Soft assignment: "80% active, 15% growing, 5% at-risk" vs binary "active/inactive"
- Probabilistic uncertainty encoded in feature vectors
- Continuous, differentiable outputs suitable for gradient-based downstream learning
- Domain-expert interpretability: each dimension has clear meaning

**Why Triple-Mode (not single HMM):**
The Markov property (future depends only on present, not full history) is a simplification. Triple-mode compensates by operating at three different time scales:
- Journey: short-term (days/weeks) purchase funnel position
- Lifecycle: long-term (months/years) customer maturity
- Behavior: consumption pattern type (monthly behavioral clusters)

### 5.9 Comparison: Theory vs Implementation

| Aspect | Standard HMM | Our Implementation |
|--------|-------------|-------------------|
| State count selection | BIC/AIC automatic | Domain knowledge (5/5/6 preset) |
| Initialization | Random or K-means | Domain expert designed (pi, A, mu preset) |
| Covariance type | Full or diagonal | Diagonal only (overfitting prevention) |
| Multi-mode | Single HMM typical | Triple-Mode parallel (Journey/Lifecycle/Behavior) |
| ODE dynamics | Separate research area | 6D dynamics extracted from Viterbi trajectory (v3.2.0) |
| GPU support | Limited | pomegranate DenseHMM (PyTorch backend) |

---

## 6. Multi-Armed Bandit Features (4D)

### 6.1 Exploration-Exploitation Features

MAB theory applied as **feature engineering** (not policy learning):

**exploration_intensity:**
$$\text{exploration\_intensity} = \frac{\text{new unique categories in recent period}}{\text{total unique categories}}$$

**category_concentration_trend (HHI-based):**
$$\text{HHI}_\text{period} = \sum_{c=1}^{C} (s_c^\text{period})^2, \quad \text{trend} = \text{HHI}_\text{recent} - \text{HHI}_\text{earlier}$$

Positive = exploitation (concentration increasing); negative = exploration (diversification).

**recency_weighted_entropy:**
$$w_t = \exp(-\lambda(T-t)), \quad \lambda = 0.03$$
$$\tilde{s}_c = \sum_{t: \text{cat}(t)=c} \text{amount}_t \cdot w_t, \quad p_c = \tilde{s}_c / \sum_c \tilde{s}_c$$
$$\text{entropy} = -\sum_{c=1}^{C} p_c \ln p_c$$

**new_category_spend_ratio:**
$$\text{ratio} = \frac{\text{spend in newly discovered categories (recent)}}{\text{total recent spend}}$$

### 6.2 Time Split Strategy

$$\text{recent\_days} = \max\left(\left\lfloor \text{lookback} \times \frac{1}{3} \right\rfloor, 7\right)$$

With 90-day lookback: recent 30 days vs earlier 60 days.

### 6.3 Implementation Note

Bandit features require **no model training** -- computed entirely via SQL aggregation (DuckDB), using a 5-stage CTE pipeline. This makes them computationally cheap and parallelizable with HMM in the feature DAG.

---

## 7. Key Architectural Insights for Paper

### 7.1 The Five-Axis Feature Taxonomy

The system's 734D main tensor + 68D separate input spans five feature axes:
1. **Static/Snapshot** (demographics, account status)
2. **Time-series** (Mamba/LNN-derived temporal patterns)
3. **Hierarchical** (merchant hierarchy, graph embeddings)
4. **Item/Product** (product interaction features)
5. **Model-derived** (HMM 5D summary, Bandit 4D, LNN statistics 18D)

### 7.2 Separate Input vs Main Tensor Design

HMM 48D uses a **separate input path** with dedicated projectors per mode, rather than being concatenated into the main tensor. This enables:
- Task-specific routing (Journey->CTR, Lifecycle->Churn)
- Independent gradient flow (HMM features don't interfere with main tensor Expert training)
- Interpretable expert-level attribution

### 7.3 Ensemble of Ensembles

The overall architecture is a **two-level ensemble:**
- Level 1: Within Temporal Expert, Mamba/LNN/PatchTST are combined via learned gating
- Level 2: Across all 7 Shared Experts (PersLay, DeepFM, Temporal, LightGCN, Unified H-GCN, Causal, OT), CGC Gate Attention performs task-specific combination

This hierarchical ensemble ensures both intra-expert diversity (temporal multi-resolution) and inter-expert complementarity (pattern/topology/temporal/relational/causal/distributional).

### 7.4 Paper-Ready Citations

| Component | Paper | Venue |
|-----------|-------|-------|
| Mamba SSM | Gu & Dao (2023) | NeurIPS 2023 |
| LNN | Hasani et al. (2021) | AAAI 2021 |
| PatchTST | Nie et al. (2023) | ICLR 2023 |
| NOTEARS | Zheng et al. (2018) | NeurIPS 2018 |
| Sinkhorn Distances | Cuturi (2013) | NeurIPS 2013 |
| HMM Tutorial | Rabiner (1989) | Proc. IEEE |
| Viterbi Algorithm | Viterbi (1967) | IEEE Trans. IT |
| Baum-Welch | Baum et al. (1970) | Ann. Math. Stat. |
| Neural ODE | Chen et al. (2018) | NeurIPS 2018 |
| S4 (SSM predecessor) | Gu et al. (2022) | ICLR 2022 |
| HiPPO | Gu et al. (2020) | NeurIPS 2020 |
| ViT (PatchTST inspiration) | Dosovitskiy et al. (2021) | ICLR 2021 |
| MoE (gating foundation) | Jacobs et al. (1991) | Neural Computation |
| Kantorovich OT | Kantorovich (1942) | Doklady |
| DAGMA (NOTEARS improvement) | Bello et al. (2022) | ICML 2022 |
