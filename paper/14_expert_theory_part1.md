# Expert Theory Extraction Part 1: DeepFM, GCN, PersLay, TDA Features

> **Note**: This document uses on-premises dimensions (734D features, 16 tasks).
> For the AWS v1 paper version (349D, 13 tasks), see the v1 Zenodo preprint
> (DOI: 10.5281/zenodo.19621884) or the feature-router table in
> paper/01_architecture_positioning.md.

Extracted from technical reference documents (v3.14--v3.15) for arxiv paper use.

---

## 1. DeepFM Expert

### Key Mathematical Formulations

**FM 2nd-order interaction (core equation):**
$$\hat{y} = w_0 + \sum_{i=1}^n w_i x_i + \sum_{i=1}^n \sum_{j=i+1}^n \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j$$

**Low-rank factorization of interaction matrix:**
$$W \approx VV^\top, \quad V \in \mathbb{R}^{n \times k}$$

Reduces parameters from $O(n^2)$ to $O(nk)$. Project uses $n=644$ features, $k=16$ latent dims: 207,046 pairwise params reduced to 10,304.

**FM Trick (O(nk) computation):**
$$\sum_{i<j} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j = \frac{1}{2} \sum_{f=1}^k \left[ \left(\sum_{i=1}^n v_{i,f} x_i\right)^2 - \sum_{i=1}^n (v_{i,f} x_i)^2 \right]$$

**Cross Network (DCN) layer:**
$$\mathbf{x}_{l+1} = \mathbf{x}_0 \odot (\mathbf{x}_l W_l + \mathbf{b}_l) + \mathbf{x}_l$$

Each layer adds one order of interaction; $l$ layers capture up to $(l+1)$-order. Only $2d$ parameters per layer (vs. $O(d^2)$ for MLP).

**DCNv2 (full-rank cross):**
$$\mathbf{x}_{l+1} = \mathbf{x}_0 \odot f(\mathbf{x}_l; W_l, \mathbf{b}_l) + \mathbf{x}_l, \quad f = \text{Tanh}(W_2 \cdot \text{Tanh}(W_1 \mathbf{x}_l))$$

### Architecture: 28-Field Design

644D normalized features split into 28 semantic fields (base 238D, multi-source 91D, extended 84D, domain 159D, multidisciplinary 24D, model-derived 27D, merchant 21D). Each field projected via `nn.Linear(d_i, 16)` to uniform 16D latent space. FM operates on field-level embeddings [B, 28, 16], Deep MLP on flattened [B, 448]. Output: FM 16D + Deep 64D = 80D -> output_layer -> 64D for PLE gate.

### Financial Domain Justification

- Cross-field interactions are critical in banking: "30s + Seoul + high digital engagement" -> online investment conversion surge; "high RFM + low deposits" -> credit product recommendation.
- FM efficiently captures 2nd-order interactions (e.g., product_cat x region_cat = regional product preference), while Deep Network captures higher-order nonlinear patterns (e.g., RFM + digital + macro uncertainty -> safe asset preference).
- 28-field design enables inter-category FM interactions: splitting 64D category into 4x16D (customer/product/region/channel) adds 27 new FM interaction pairs at negligible parameter cost.

### Comparison with Alternatives

| Aspect | DeepFM | DCNv2 | Transformer (AutoInt) |
|--------|--------|-------|----------------------|
| Interaction order | FM: 2nd + Deep: arbitrary | Cross: (l+1)th + Deep | Self-attention: arbitrary |
| Parameters | ~169K | ~2.5M (cross alone) | $O(n^2)$ attention |
| Inference speed | Fastest (2-5x vs Attention) | Medium | Slowest |
| Field awareness | Yes (28 fields) | No (raw features) | Yes (per-feature) |

Project uses DeepFM as default lightweight Shared Expert; DCNv2Expert available as alternative for higher-order needs. Per BARS benchmarking (2022-2024): DeepFM consistently top-tier on small/medium datasets with lowest inference latency.

### Key References
- Guo et al., "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction" (IJCAI 2017)
- Wang et al., "Deep & Cross Network for Ad Click Predictions" (KDD 2017)
- Wang et al., "DCN V2: Improved Deep & Cross Network" (WWW 2021)
- Rendle, "Factorization Machines" (ICDM 2010)

---

## 2. GCN Expert (LightGCN + Hyperbolic GCN)

### Key Mathematical Formulations

**LightGCN message passing:**
$$\mathbf{e}_u^{(k+1)} = \sum_{i \in \mathcal{N}_u} \frac{1}{\sqrt{|\mathcal{N}_u|} \cdot \sqrt{|\mathcal{N}_i|}} \cdot \mathbf{e}_i^{(k)}$$

Symmetric normalization $\tilde{A} = D^{-1/2} A D^{-1/2}$ dampens both sender (popular item) and receiver influence.

**Layer combination:**
$$\mathbf{e}_u^{\text{final}} = \frac{1}{L+1} \sum_{k=0}^{L} \mathbf{e}_u^{(k)}$$

Uniform average across all hops (0-hop self + 1,2,3-hop neighbors). No learnable attention weights -- empirically superior and prevents overfitting.

**BPR Loss (Bayesian Personalized Ranking):**
$$\mathcal{L}_{\text{BPR}} = -\sum_{(u, i^+, i^-)} \log \sigma(\hat{y}_{ui^+} - \hat{y}_{ui^-}) + \lambda \|\Theta\|^2$$

Pairwise ranking loss derived from MAP estimation; optimizes relative ordering rather than absolute scores. L2 regularization applied to initial embeddings only (not GCN outputs).

**Poincare Ball model (H-GCN):**
$$\mathbb{B}_c^d = \{ \mathbf{x} \in \mathbb{R}^d : c\|\mathbf{x}\|^2 < 1 \}$$

**Exponential map (tangent -> hyperbolic):**
$$\exp_{\mathbf{0}}(\mathbf{v}) = \tanh(\sqrt{c}\|\mathbf{v}\|) \cdot \frac{\mathbf{v}}{\sqrt{c}\|\mathbf{v}\|}$$

**Logarithmic map (hyperbolic -> tangent):**
$$\log_{\mathbf{0}}(\mathbf{y}) = \text{arctanh}(\sqrt{c}\|\mathbf{y}\|) \cdot \frac{\mathbf{y}}{\sqrt{c}\|\mathbf{y}\|}$$

**Poincare distance:**
$$d_{\mathbb{B}}(\mathbf{x}, \mathbf{y}) = \frac{1}{\sqrt{c}} \text{arccosh}\left(1 + \frac{2c\|\mathbf{x}-\mathbf{y}\|^2}{(1-c\|\mathbf{x}\|^2)(1-c\|\mathbf{y}\|^2)}\right)$$

Near boundary: denominator -> 0, distance explodes (small Euclidean distance = large hyperbolic distance). This encodes hierarchical depth naturally.

**Riemannian gradient correction:**
$$\nabla_{\text{Riem}} f(\mathbf{x}) = \frac{(1-c\|\mathbf{x}\|^2)^2}{4} \nabla_{\text{Euclid}} f(\mathbf{x})$$

Near boundary: correction factor -> 0 (conservative updates). Essential for stable learning on Poincare Ball.

**Fermi-Dirac decoder (link prediction):**
$$P(\text{edge}|u,v) = \frac{1}{\exp((d_{\mathbb{B}}(u,v) - r)/t) + 1}$$

Borrowed from statistical physics; $r$ = margin (Fermi energy), $t$ = temperature. Sharp boundary between connected/disconnected pairs.

**Frechet mean (Einstein midpoint approximation):**
$$\gamma_i = \frac{1}{\sqrt{1-c\|\mathbf{x}_i\|^2}}, \quad \bar{\mathbf{x}} = \text{proj}\left(\frac{\sum_i w_i \gamma_i \mathbf{x}_i}{\sum_i w_i \gamma_i}\right)$$

Lorentz factor $\gamma_i$ gives higher weight to boundary points (specialized consumers).

### Architecture: Dual GCN + 2-Stage Learning

| Property | LightGCN | H-GCN |
|----------|----------|-------|
| Nodes | Customers + Merchants (bipartite) | Merchants only (MCC tree) |
| Edges | Customer-Merchant transactions | Parent-child hierarchy + Brand-Brand co-visitation |
| Space | Euclidean $\mathbb{R}^{64}$ | Poincare Ball $\mathbb{B}^8$ |
| Learning | "Who likes what" (collaborative filtering) | "How merchants relate structurally" |
| Output | Customer embedding 64D (direct) | Merchant embedding -> per-customer aggregation 47D (indirect) |

**2-Stage pipeline:** Stage 1 (offline): graph-level learning (LightGCN: BPR, H-GCN: self-supervised Fermi-Dirac). Embeddings stored as Parquet. Stage 2 (online): lookup + lightweight MLP adaptation. No graph propagation at inference time -- VRAM-friendly for single GPU.

### Financial Domain Justification

- **LightGCN**: Multi-hop collaborative signals capture indirect preferences. "Customer A bought at Starbucks, Customer B also bought at Starbucks and Ediya" -> A may prefer Ediya. Critical for cross-selling in banking.
- **H-GCN**: MCC classification hierarchy (Root -> L1(8) -> L2(~100) -> Brand(~50K) -> Branch(~500K)) is inherently tree-structured. Hyperbolic space embeds trees with exponential volume growth matching tree branching -- 8D Poincare Ball suffices for ~550K nodes (Euclidean would need thousands of dimensions). This is the mathematical basis for the Nickel & Kiela (2017) result: 5D hyperbolic > 200D Euclidean for WordNet hierarchy.
- **Co-visitation edges**: Behavioral signal complements static MCC hierarchy. "Starbucks visitors also visit Ediya within 7 days" creates edges weighted by recency (exponential time decay). Scale factor 0.5 vs taxonomy edges preserves structural dominance.

### Comparison with Alternatives

- LightGCN vs. NGCF: Removing feature transformation $W$ and nonlinear activation $\sigma$ *improves* performance in ID-based collaborative filtering (no raw features to transform). Simpler = better here.
- H-GCN vs. Euclidean tree embedding: Depth-$d$ complete binary tree needs $O(2^d)$ Euclidean dimensions but only $O(d)$ hyperbolic dimensions.
- 2-Stage vs. end-to-end GCN: Pinterest PinSage pattern -- offline graph precomputation is standard for production systems. Decouples graph update frequency from model training frequency.

### Key References
- He et al., "LightGCN: Simplifying and Powering GCN for Recommendation" (SIGIR 2020)
- Chami et al., "Hyperbolic Graph Convolutional Neural Networks" (NeurIPS 2019)
- Nickel & Kiela, "Poincare Embeddings for Learning Hierarchical Representations" (NeurIPS 2017)
- Rendle et al., "BPR: Bayesian Personalized Ranking from Implicit Feedback" (UAI 2009)
- Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional Networks" (ICLR 2017)

---

## 3. PersLay Expert

### Key Mathematical Formulations

**PersLay layer (Carriere et al., NeurIPS 2020):**
$$\text{PersLay}(D) = \rho\left(\sum_{(b,d) \in D} w(b,d) \cdot \phi(b,d)\right)$$

- $\phi$: point transformation (RationalHat or Gaussian)
- $w$: weighting function (persistence-based: $w = |d-b|^p$, or learned)
- $\rho$: permutation-invariant aggregation (sum, mean, max, or attention)

**Stability theorem (Bottleneck distance):**
$$d_B(\text{Dgm}(f), \text{Dgm}(g)) \leq \|f - g\|_\infty$$

Input perturbation bounded by max change in filtration function. Mathematically guarantees noise robustness of topological features.

**Wasserstein-q distance between diagrams:**
$$W_q(D_1, D_2) = \left(\inf_{\gamma: D_1 \to D_2} \sum_{p \in D_1} \|p - \gamma(p)\|_\infty^q\right)^{1/q}$$

**Homology groups:**
$$H_k = \text{Ker}(\partial_k) / \text{Im}(\partial_{k+1})$$

$\beta_k = \text{rank}(H_k)$: number of independent $k$-dimensional "holes". $\beta_0$ = connected components, $\beta_1$ = loops, $\beta_2$ = voids.

### Architecture: 5-Block Multi-Beta Design

| Block | Input | Homology | Output |
|-------|-------|----------|--------|
| short_beta0 | 90-day app logs [B, 200, 3] | $H_0$ (clusters) | 64D |
| short_beta1 | 90-day app logs | $H_1$ (loops) | 64D |
| long_beta0 | 12-month transactions [B, 150, 3] | $H_0$ (clusters) | 64D |
| long_beta1 | 12-month transactions | $H_1$ (loops) | 64D |
| long_beta2 | 12-month transactions | $H_2$ (voids) | 64D |

Short concat 128D + Long concat 192D + Global stats MLP 32D + Phase transition 10D = **362D -> final_mlp -> 64D** for PLE gate.

Production config: RationalHatPhi + persistence weighting + sum aggregation. Dual-mode: Raw Diagram (production) with Pre-computed 70D fallback.

### Financial Domain Justification

- **$H_0$ (connected components)**: Separated consumption clusters reveal lifestyle segmentation. "Grocery cluster" vs "travel cluster" that merge at different scales indicate spending diversification.
- **$H_1$ (loops)**: Cyclical consumption patterns (monthly: groceries -> transport -> entertainment -> groceries) captured as persistent loops. Strong loops = habitual patterns = predictable behavior.
- **$H_2$ (voids, long-range only)**: 3D voids in amount-category-time space reveal systematic avoidance patterns (e.g., no mid-range spending = only small + large transactions). Reflects lifestyle constraints.
- **Short vs Long**: 90-day captures recent behavioral shifts; 12-month captures stable structural patterns. Dual timescale prevents both recency bias and stale signals.
- Topological features are coordinate-invariant and noise-robust (stability theorem). Traditional statistics (mean, variance) cannot distinguish between a single cluster and two separated clusters with identical moments.

### Comparison with Alternatives

| Method | Handles variable-size PD | End-to-end learnable | Stability guarantee |
|--------|-------------------------|---------------------|-------------------|
| Persistence Images | Fixed grid | No | Approximate |
| Persistence Landscapes | Function space | No (Banach space) | Yes |
| **PersLay** | **Yes (phi + rho)** | **Yes** | **Yes (inherits)** |
| Persformer (2024) | Yes (attention) | Yes | Yes |

PersLay chosen for production stability (sum aggregation = no gradient bottleneck, persistence weighting = automatic padding handling). Persformer's self-attention across diagram points is a potential future upgrade.

### Key References
- Carriere et al., "PersLay: A Neural Network Layer for Persistence Diagrams and New Graph Topological Signatures" (AISTATS 2020)
- Edelsbrunner, Letscher, Zomorodian, "Topological Persistence and Simplification" (2002)
- Cohen-Steiner, Edelsbrunner, Harer, "Stability of Persistence Diagrams" (DCG 2007)
- Chazal et al., "Proximity of persistence modules and their diagrams" (2009)

---

## 4. TDA Feature Pipeline (Offline 70D)

### Key Mathematical Formulations

**Vietoris-Rips complex:**
$$\sigma = \{x_0, \ldots, x_k\} \in \text{VR}_\epsilon(X) \iff d(x_i, x_j) \leq \epsilon, \;\forall\, i,j$$

**Persistence Entropy (Rucco et al., 2016):**
$$E = -\sum_{i=1}^N p_i \log p_i, \quad p_i = \frac{d_i - b_i}{\sum_j (d_j - b_j)}$$

High entropy = diverse topological features uniformly distributed. Low entropy = few dominant patterns.

**Wasserstein-1 distance for phase transition detection:**
$$W_1(\text{PD}_1, \text{PD}_2) = \inf_{\gamma} \sum_{x \in \text{PD}_1} \|x - \gamma(x)\|_\infty$$

Measures structural change between first-half and second-half transaction persistence diagrams.

**Phase transition probability (Sigmoid):**
$$P_{\text{transition}} = \frac{1}{1 + e^{-2(\Delta_{\text{total}} - \tau)}}, \quad \tau = 0.5$$

### Feature Architecture: 70D = 24D (short) + 36D (long) + 10D (phase)

**tda_short (24D)**: 90-day app logs. Input: 6D point cloud per transaction (log-amount, MCC percentile rank, sin/cos day-of-week, sin/cos hour). $H_0 + H_1$ x 6 features (entropy, lifetime mean/std/min/max/median) x 2 scopes (global/local).

**tda_long (36D)**: 12-month card transactions. Same 6D point cloud. $H_0 + H_1 + H_2$ x 6 features x 2 scopes. $H_2$ included here because sufficient data density (hundreds-thousands of transactions).

**phase_transition (10D)**:
- PD Distance 4D: $W_1$ distance ($H_0$, $H_1$), total topological change, max structural shift
- Transition Detection 6D: transition probability, imminence, frequency, direction (+1 expand/-1 contract), magnitude, phase confidence

**5-Phase classification**: Stable / Growing ($\Delta\beta_0 > 0$) / Shrinking ($\Delta\beta_0 < 0$) / Chaotic (entropy > 1.5x mean) / Transitioning

### Financial Domain Justification

- **Consumption topology captures lifestyle structure**: Two customers with identical mean spending and variance can have fundamentally different consumption *shapes* -- one homogeneous cluster vs. two separated clusters with cyclic transitions. Only topological features distinguish these.
- **Phase transition = regime change detection**: Wasserstein distance between first-half/second-half PDs quantifies behavioral shifts (job change, life events, financial distress). Sigmoid probability provides smooth binary signal.
- **$H_2$ voids in long-range**: Systematic absence patterns in amount-category-time space (e.g., no mid-range spending, no weekday leisure) reflect hard lifestyle constraints -- more stable predictors than positive signals.
- **Coordinate invariance**: Topological features are invariant to rotation, scaling, and continuous deformation. Robust across different normalization schemes and feature engineering choices.
- **Stability theorem**: Mathematical guarantee that small measurement noise produces small changes in topological features -- rare among feature engineering methods.

### Cold-Start Strategy (4-Stage Progressive TDA)

| Stage | Condition | Output | Method |
|-------|-----------|--------|--------|
| 1 | Day 0, 0 transactions | 18D | Global distribution median (demographic segment) |
| 2 | 7-30 days, 3+ transactions | 9D | Histogram-based TDA approximation |
| 3 | <12 months, 30+ transactions | 24D | Ripser + H0,H1 + Global/Local |
| 4 | 12+ months, 30+ transactions | 36D | Ripser + H0,H1,H2 + Global/Local |

Demographic similarity via k-NN (k=5) with inverse distance weighting (Shepard's method, 1968).

### Computation Engine Priority Chain

1. **Ripser++ (CUDA GPU)**: C++ + CUDA kernel, fastest. `rpp.run("--format point-cloud", ...)`
2. **Ripser (CPU)**: C++ bindings + optional GPU distance matrix via CuPy (10-50x acceleration). Chunk-based distance matrix for n > 5000.
3. **giotto-tda (CPU)**: Most stable, richest API, slowest. Fallback.

Time-stratified sampling: max 1000 points per customer, stratified across $k$ temporal buckets to preserve temporal order.

### TDA Mapper (2D supplementary)

Mapper algorithm (Singh, Memoli, Carlsson, 2007): Filter function (projection) -> Cubical cover (10 intervals, 20% overlap) -> DBSCAN per interval -> graph construction. Output: betweenness centrality + closeness centrality. Stored separately from 70D main features.

### Key References
- Edelsbrunner et al., "Topological Persistence and Simplification" (2000/2002)
- Cohen-Steiner, Edelsbrunner, Harer, "Stability of Persistence Diagrams" (DCG 2007)
- Rucco et al., "Persistence Entropy" (2016)
- Bauer, "Ripser: efficient computation of Vietoris-Rips persistence barcodes" (2021)
- Zhang et al., "Ripser++ GPU-accelerated computation" (2020)
- Bubenik, "Statistical Topological Data Analysis using Persistence Landscapes" (JMLR 2015)
- Kerber, Morozov, Nigmetov, "Geometry Helps to Compare Persistence Diagrams" (2017)
- Singh, Memoli, Carlsson, "Topological Methods for the Analysis of High Dimensional Data Sets" (2007)
- Carlsson, "Topology and Data" (Bulletin of the AMS, 2009)
