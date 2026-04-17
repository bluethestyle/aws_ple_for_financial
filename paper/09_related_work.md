# 9. Related Work

This section surveys the literature most relevant to our system: a heterogeneous-expert PLE + adaTT architecture for multi-task financial product recommendation with inherent explainability, knowledge distillation to LGBM, and regulatory compliance.

Our work is available as two open preprints on Zenodo:
- Paper 1 (Heterogeneous Expert PLE): https://doi.org/10.5281/zenodo.19621884
- Paper 2 (Agentic Reason Generation): https://doi.org/10.5281/zenodo.19622052

---

## 9.1 Multi-Task Learning for Recommendation Systems

Multi-task learning (MTL) has become the dominant paradigm for industrial recommendation systems that must optimize multiple objectives simultaneously (e.g., click, conversion, engagement).

### 9.1.1 Shared-Bottom and Hard Parameter Sharing

**Caruana (1997)** established that MTL improves generalization by leveraging domain-specific information in the training signals of related tasks. The shared-bottom architecture, which feeds a common representation into task-specific towers, remains the simplest MTL baseline but suffers from **negative transfer** when tasks conflict.

- R. Caruana, "Multitask Learning," *Machine Learning*, vol. 28, no. 1, pp. 41--75, 1997.

### 9.1.2 MMoE: Multi-gate Mixture-of-Experts

**Ma et al. (2018)** proposed MMoE, which replaces the shared bottom with multiple expert sub-networks and task-specific gating networks. Each task learns its own soft mixture over all experts, enabling differential knowledge sharing. MMoE was deployed at Google for content recommendation and demonstrated robustness when task correlations are weak.

- J. Ma, Z. Zhao, X. Yi, J. Chen, L. Hong, and E. H. Chi, "Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts," in *Proc. KDD*, 2018, pp. 1930--1939.

**Gap filled by our work:** MMoE uses homogeneous MLP experts. Our system extends the MoE paradigm with *heterogeneous* expert architectures (DeepFM, Mamba, HGCN, TDA, etc.), where each expert type captures a fundamentally different inductive bias rather than relying on random initialization diversity alone.

### 9.1.3 PLE: Progressive Layered Extraction

**Tang et al. (2020)** introduced PLE, which explicitly separates shared and task-specific expert networks and employs a progressive routing mechanism across multiple extraction layers. PLE addresses the "seesaw phenomenon" where improving one task hurts another, and was deployed on Tencent's video recommendation platform (1B+ samples), yielding +2.23% view-count and +1.84% watch time.

- H. Tang, J. Liu, M. Zhao, and X. Gong, "Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations," in *Proc. RecSys*, 2020, pp. 269--278.

**Gap filled by our work:** PLE's experts are identical MLPs differentiated only by which tasks they serve. We retain PLE's progressive extraction structure but populate the expert basket with 7 architecturally distinct expert types, each matched to a specific data modality (temporal, topological, hierarchical, causal, interaction-graph, cross-feature, distributional).

### 9.1.4 AdaTT: Adaptive Task-to-Task Fusion

**Li et al. (2023)** proposed AdaTT, which builds task-to-task fusion units at multiple levels using residual and gating mechanisms for adaptive knowledge transfer between tasks. AdaTT extends PLE with explicit inter-task transfer pathways and achieves state-of-the-art on KDD Cup benchmarks.

- C. Li, Y. Liu, L. Liao, S. Yang, Y. Wang, P. Yao, S. Yuan, Y. Hao, and G. Chen, "AdaTT: Adaptive Task-to-Task Fusion Network for Multitask Learning in Recommendations," in *Proc. KDD*, 2023, pp. 4489--4500.

**Gap filled by our work:** We integrate adaTT's task-group fusion on top of our heterogeneous PLE backbone, enabling not just task-to-task but also expert-type-to-task adaptive transfer. Our task groups are defined by domain semantics (product clusters) rather than arbitrary partitioning.

### 9.1.5 Other Notable MTL Architectures

| Paper | Venue | Contribution | Relation to Our Work |
|-------|-------|-------------|---------------------|
| **ESMM** (Ma et al., 2018) | SIGIR 2018 | Entire-space modeling for CVR via CTR auxiliary task; eliminates sample selection bias | Our multi-task formulation similarly chains dependent tasks (click -> ownership -> cross-sell) but generalizes to 13 tasks |
| **STAR** (Sheng et al., 2021) | CIKM 2021 | Star topology with shared center + domain-specific parameters for multi-domain CTR | Complementary to our approach; STAR addresses multi-domain while we address multi-task with heterogeneous experts |
| **M3oE** (Zhang et al., 2024) | SIGIR 2024 | Multi-domain multi-task MoE with three disentangled expert modules and AutoML structure search | Most recent MoE-based MTL; uses homogeneous experts with structural search, whereas we use pre-defined heterogeneous expert types matched to data modalities |

---

## 9.2 Mixture of Experts and Heterogeneous Architectures

### 9.2.1 Sparsely-Gated MoE

**Shazeer et al. (2017)** introduced the sparsely-gated MoE layer with up to thousands of feed-forward sub-networks and a trainable gating network for sparse expert selection. This foundational work demonstrated >1000x model capacity increase with minor computational overhead, applied to language modeling (137B parameters).

- N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton, and J. Dean, "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer," in *Proc. ICLR*, 2017.

### 9.2.2 Switch Transformer

**Fedus et al. (2022)** simplified MoE routing to top-1 expert selection (Switch Transformer), achieving 7x pre-training speedup with stable training in bfloat16. Published in JMLR, it became the blueprint for efficient sparse models.

- W. Fedus, B. Zoph, and N. Shazeer, "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity," *JMLR*, vol. 23, no. 120, pp. 1--40, 2022.

### 9.2.3 The Heterogeneous Expert Gap

Virtually all MoE literature uses **homogeneous** experts (identical FFN/MLP architectures differentiated only by learned weights). MoE++ (ICLR 2025) introduced heterogeneity at the *computation level* (zero-computation vs. FFN experts) but not at the *architectural level*. MTmixAtt explored heterogeneous feature grouping with attention mechanisms for recommendation.

**Gap filled by our work:** To our knowledge, our system is the first to instantiate a PLE-style MoE where each expert slot is occupied by a fundamentally different neural architecture (DeepFM, Mamba+LNN+Transformer ensemble, HGCN, PersLay/TDA, Causal net, LightGCN, Optimal Transport), creating a true "expert basket" that mirrors the multi-disciplinary nature of financial decision-making.

---

## 9.3 Individual Expert Technologies

### 9.3.1 DeepFM: Cross-Feature Interactions

**Guo et al. (2017)** proposed DeepFM, combining factorization machines (low-order interactions) with deep networks (high-order interactions) in a shared-embedding architecture, eliminating the need for feature engineering. Widely adopted for CTR prediction.

- H. Guo, R. Tang, Y. Ye, Z. Li, and X. He, "DeepFM: A Factorization-Machine Based Neural Network for CTR Prediction," in *Proc. IJCAI*, 2017, pp. 1725--1731.

**Role in our system:** DeepFM serves as the cross-feature interaction expert, capturing product-demographic and product-behavioral feature crosses that traditional MLPs learn inefficiently.

### 9.3.2 Mamba: State Space Models for Sequences

**Gu and Dao (2024)** introduced Mamba, a selective state-space model that achieves linear-time sequence processing with input-dependent state transitions via a selective scan mechanism. Mamba matches or exceeds Transformer performance on language modeling while scaling linearly with sequence length. MambaTS (2024) adapted this for long-term time series forecasting.

- A. Gu and T. Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces," in *Proc. COLM*, 2024.
- L. Zhong, S. Wen, J. Li, and Q. Chen, "MambaTS: Improved Selective State Space Models for Long-term Time Series Forecasting," in *Proc. NeurIPS*, 2024.

**Role in our system:** Mamba forms the backbone of our Temporal Ensemble expert, processing 17-month customer behavioral sequences with O(n) complexity, enabling efficient long-range dependency capture for temporal patterns.

### 9.3.3 Liquid Neural Networks

**Hasani et al. (2021)** proposed Liquid Time-Constant (LTC) networks inspired by C. elegans neural circuits. LTC networks use first-order dynamical systems with input-dependent time constants, achieving superior expressivity for time-series prediction with far fewer neurons than RNNs, and providing inherent interpretability through analyzable dynamics.

- R. Hasani, M. Lechner, A. Amini, D. Rus, and R. Grosu, "Liquid Time-constant Networks," in *Proc. AAAI*, 2021, pp. 7657--7666.

**Role in our system:** LNN complements Mamba in our Temporal Ensemble by providing adaptive time-constant modeling for irregularly sampled financial events (e.g., variable transaction frequencies).

### 9.3.4 Hyperbolic Graph Convolutional Networks (HGCN)

**Chami et al. (2019)** introduced HGCN, the first inductive GCN operating in hyperbolic space (hyperboloid model), learning node representations that naturally encode hierarchical structure. HGCN achieved up to 63.1% error reduction in link prediction over Euclidean GCNs. This builds on **Nickel and Kiela (2017)**'s Poincare embeddings for hierarchical representation.

- I. Chami, R. Ying, C. Re, and J. Leskovec, "Hyperbolic Graph Convolutional Neural Networks," in *Proc. NeurIPS*, 2019, pp. 4869--4880.
- M. Nickel and D. Kiela, "Poincare Embeddings for Learning Hierarchical Representations," in *Proc. NeurIPS*, 2017, pp. 6338--6347.

**Role in our system:** HGCN encodes the product taxonomy hierarchy (savings -> deposits -> time deposits) in hyperbolic space, capturing the tree-like structure of financial product catalogs more faithfully than Euclidean GCNs.

### 9.3.5 Topological Data Analysis and PersLay

**Carriere et al. (2020)** introduced PersLay, a neural network layer that processes persistence diagrams (from persistent homology) through learnable weight functions and permutation-invariant operators, enabling end-to-end integration of TDA with deep learning.

- M. Carriere, F. Chazal, Y. Ike, T. Lacombe, M. Royer, and Y. Umeda, "PersLay: A Neural Network Layer for Persistence Diagrams and New Graph Topological Signatures," *JMLR*, vol. 21, no. 1, pp. 1--33, 2020.

**Role in our system:** The TDA/PersLay expert captures topological features of customer behavior point clouds (e.g., persistent loops in spending patterns), providing shape-based features that are invariant to continuous deformations--a perspective entirely absent from standard tabular ML.

### 9.3.6 LightGCN for Collaborative Filtering

**He et al. (2020)** proposed LightGCN, which strips GCN down to its essential component (neighborhood aggregation) for collaborative filtering, removing feature transformation and nonlinear activation. LightGCN achieved ~16% improvement over NGCF and became the de facto graph-based recommendation baseline.

- X. He, K. Deng, X. Wang, Y. Li, Y. Zhang, and M. Wang, "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation," in *Proc. SIGIR*, 2020, pp. 639--648.

**Role in our system:** LightGCN serves as our interaction-graph expert, propagating collaborative signals across the customer-product bipartite graph to capture "users who own product A also tend to own product B" patterns.

### 9.3.7 Optimal Transport in Machine Learning

**Cuturi (2013)** made optimal transport computationally tractable by introducing Sinkhorn regularization, enabling OT distance computation orders of magnitude faster than classical solvers. Recent surveys (2023) document OT applications across supervised, unsupervised, and transfer learning.

- M. Cuturi, "Sinkhorn Distances: Lightspeed Computation of Optimal Transport," in *Proc. NeurIPS*, 2013, pp. 2292--2300.
- A. Khamis, R. Tsuchida, M. Tarek, V. Campagnolo, and M. Nabi, "Recent Advances in Optimal Transport for Machine Learning," *arXiv:2306.16156*, 2023.

**Role in our system:** The Optimal Transport expert computes distributional distances between customer spending profiles and product-typical profiles, providing a geometrically meaningful similarity measure for recommendation.

### 9.3.8 Causal Inference in Recommendation

**Gao et al. (2024)** provide a comprehensive survey on causal inference for recommendation, covering deconfounding, counterfactual reasoning, and treatment effect estimation. Methods include backdoor/front-door adjustment, instrumental variables, and representation learning for debiasing.

- Y. Gao, K. Cai, S. Chen, Y. Li, S. Jin, D. Jin, and Y. Li, "A Survey on Causal Inference for Recommendation," *The Innovation*, vol. 5, no. 2, 2024.

**Role in our system:** Our Causal expert models treatment effects (recommendation exposure -> product adoption), distinguishing genuine causal uplift from confounded correlations in observational banking data.

---

## 9.4 Explainability in Recommendation Systems

### 9.4.1 Post-Hoc Methods: SHAP and LIME

**Lundberg and Lee (2017)** unified feature attribution methods under SHAP (SHapley Additive exPlanations), while **Ribeiro et al. (2016)** proposed LIME (Local Interpretable Model-agnostic Explanations) using local linear surrogates.

- S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," in *Proc. NeurIPS*, 2017, pp. 4765--4774.
- M. T. Ribeiro, S. Singh, and C. Guestrin, "Why Should I Trust You? Explaining the Predictions of Any Classifier," in *Proc. KDD*, 2016, pp. 1135--1144.

**Limitations documented in the literature:** Both methods treat features as independent (violating financial feature correlations), are computationally expensive for real-time serving, produce unstable explanations across similar inputs, and cannot infer causality (Salih et al., 2023). LIME's local linearity assumption fails for highly nonlinear recommendation models.

### 9.4.2 Integrated Gradients

**Sundararajan et al. (2017)** proposed Integrated Gradients (IG), an axiomatic attribution method satisfying sensitivity and implementation invariance, requiring only standard gradient calls. IG is used in our distillation pipeline for feature selection.

- M. Sundararajan, A. Taly, and Q. Yan, "Axiomatic Attribution for Deep Networks," in *Proc. ICML*, 2017, pp. 3319--3328.

### 9.4.3 Inherent vs. Post-Hoc Explainability

Recent literature (Salih et al., 2025; Frontiers in AI, 2024) highlights that post-hoc methods provide *explanations of model behavior* rather than *explanations of the decision process itself*. The EU AI Act and financial regulators increasingly demand explanations that reflect the actual decision mechanism.

**Gap filled by our work:** Our architecture provides **inherent explainability** through expert gate weights. When the system recommends a product, the gate activations directly reveal *which expert types contributed* (e.g., "temporal pattern 42%, product hierarchy 28%, collaborative 18%"), providing structurally faithful explanations without post-hoc approximation. This is fundamentally different from SHAP/LIME, which explain a black-box after the fact.

---

## 9.5 Financial Recommendation Systems

### 9.5.1 Deep Learning for Financial Products

**Chen et al. (2024)** proposed a Transformer + transfer learning system for bank product recommendation, demonstrating that deep learning outperforms traditional collaborative filtering for financial products. **Met et al. (2024)** evaluated ML algorithms for SME banking product recommendation.

- W. Chen, Y. Liu, and H. Zhang, "Deep Learning-Powered Financial Product Recommendation System in Banks: Integration of Transformer and Transfer Learning," *J. Org. End User Comp.*, vol. 36, no. 1, 2024.
- A. Met, D. Korkmaz, and K. Tuna, "Product Recommendation System With Machine Learning Algorithms for SME Banking," *Int. J. Intell. Syst.*, vol. 2024, art. 5585575, 2024.

### 9.5.2 Cross-Selling with Sequential Models

**Martinez-Plumed et al. (2023)** demonstrated that deep learning models processing longitudinal transaction data as sequential input (without aggregation) significantly improve cross-selling prediction accuracy for consumer loans, analyzing ~800K credit card transactions.

- F. Martinez-Plumed et al., "Improving the Predictive Accuracy of the Cross-selling of Consumer Loans Using Deep Learning Networks," *Ann. Oper. Res.*, 2023.
- R. Loureiro and J. Guerreiro, "From Collaborative Filtering to Deep Learning: Advancing Recommender Systems with Longitudinal Data in the Financial Services Industry," *Eur. J. Oper. Res.*, 2025.

### 9.5.3 The Santander Benchmark

The Kaggle Santander Product Recommendation competition (2016) established a widely-used benchmark: ~950K customers, 24 banking products, 17 months of longitudinal data. Top solutions used gradient boosting with hand-crafted lag features.

**Gap filled by our work:** We go far beyond gradient boosting on lag features by (1) applying multi-task deep learning with heterogeneous experts, (2) incorporating topological/hyperbolic/causal feature engineering, (3) generating a synthetic benchmark with controllable difficulty via Gaussian Copula variance budgets, and (4) providing inherent explainability suitable for regulatory compliance.

---

## 9.6 Knowledge Distillation for Deployment

### 9.6.1 General Knowledge Distillation

**Hinton et al. (2015)** established the teacher-student framework where a large teacher network's soft predictions guide the training of a smaller student model.

- G. Hinton, O. Vinyals, and J. Dean, "Distilling the Knowledge in a Neural Network," *arXiv:1503.02531*, 2015.

### 9.6.2 Distillation to Tree-Based Models

Recent work demonstrates that distilling deep models to LightGBM/XGBoost achieves **87.7% model size reduction** with negligible accuracy loss, with student models yielding ~73% faster inference (18ms per 1000 samples) and macro-F1 of 99.7%, making them suitable for CPU-based edge deployment. LightGBM's histogram-based approach enables efficient integration in regulated environments where model transparency is prioritized.

- G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu, "LightGBM: A Highly Efficient Gradient Boosting Decision Tree," in *Proc. NeurIPS*, 2017, pp. 3146--3154.

### 9.6.3 Distillation for Recommendation

**Curriculum-Scheduled Knowledge Distillation (CKD)** from multiple pre-trained teachers for multi-domain sequential recommendation demonstrates effective cross-domain knowledge transfer via curriculum learning.

**Gap filled by our work:** We distill a complex heterogeneous-expert PLE (teacher) to a single LGBM (student) using LGBM gain importance-based feature selection to identify the most informative features. The LGBM student serves in a serverless (Lambda) environment with <50ms latency, while the teacher's expert gate weights generate human-readable recommendation reasons.

---

## 9.7 Regulatory Compliance for AI in Finance

### 9.7.1 EU AI Act

The EU AI Act (effective 2024, fully applicable by 2026) classifies credit scoring and financial product recommendation as **high-risk AI systems**, mandating: (1) transparency in individual decisions, (2) human oversight, (3) documentation of training data and model architecture, and (4) bias monitoring. Explainability requirements remain technically underspecified, creating implementation challenges.

- European Parliament, "Regulation (EU) 2024/1689 laying down harmonised rules on artificial intelligence (AI Act)," *Official Journal of the EU*, 2024.

### 9.7.2 Korean Financial AI Guidelines

Korea's Financial Services Commission (FSC) issued AI guidelines for the financial sector (2021, updated 2024), emphasizing seven principles including transparency, fairness, and accountability. The Financial Supervisory Service (FSS) established the "Financial AI Counsel" in March 2024 with the Korea Credit Information Service and Financial Security Institute. Korea's AI Basic Act (passed December 2024, effective January 2026) provides a comprehensive national framework.

- Financial Services Commission (Korea), "Financial Sector AI Guidelines," 2024.
- National Assembly of Korea, "Act on the Development of Artificial Intelligence and Establishment of Trust," December 2024.

### 9.7.3 XAI for Financial Compliance

The CFA Institute and regulatory bodies increasingly require that AI-driven financial recommendations provide explanations interpretable by both compliance officers and end customers. Current approaches rely on post-hoc SHAP/LIME, which face documented issues of instability and feature-dependence violations in financial data.

**Gap filled by our work:** Our inherent explainability via expert gate weights satisfies both EU AI Act transparency requirements and Korean FSS guidelines without relying on post-hoc approximations. The recommendation reason generation pipeline transforms gate weights into natural-language explanations (e.g., "Recommended based on your 6-month spending trend [temporal expert] and similar customers' portfolios [collaborative expert]").

---

## 9.8 Multi-Task Loss Balancing

### 9.8.1 Uncertainty Weighting

**Kendall et al. (2018)** proposed weighting multi-task losses by learned homoscedastic uncertainty, providing a principled alternative to manual loss weight tuning. The method derives per-task weights from a Bayesian framework and demonstrates improved performance over grid-searched weights.

- A. Kendall, Y. Gal, and R. Cipolla, "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics," in *Proc. CVPR*, 2018, pp. 7482--7491.

### 9.8.2 Focal Loss for Class Imbalance

**Lin et al. (2017)** introduced Focal Loss, which down-weights well-classified examples via a modulating factor (1-p_t)^gamma, focusing learning on hard negatives. Originally for object detection, it has been widely adopted for imbalanced classification including recommendation tasks.

- T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, "Focal Loss for Dense Object Detection," in *Proc. ICCV*, 2017, pp. 2980--2988.

**Our approach:** We combine Kendall uncertainty weighting (for inter-task balancing across 13 tasks) with Focal Loss (for intra-task class imbalance handling), configured entirely through pipeline.yaml.

---

## 9.9 Synthetic Data for ML Benchmarks

### 9.9.1 Gaussian Copula and CTGAN

**Patki et al. (2016)** introduced the Synthetic Data Vault (SDV) with Gaussian Copula-based tabular data synthesis. **Xu et al. (2019)** proposed CTGAN, a conditional GAN specifically designed for mixed-type tabular data with mode-specific normalization.

- N. Patki, R. Wedge, and K. Veeramachaneni, "The Synthetic Data Vault," in *Proc. IEEE DSAA*, 2016, pp. 399--410.
- L. Xu, M. Skoularidou, A. Cuesta-Infante, and K. Veeramachaneni, "Modeling Tabular Data Using Conditional GAN," in *Proc. NeurIPS*, 2019, pp. 7335--7345.

### 9.9.2 Benchmarking Synthetic Data Quality

Recent comparative studies (2024-2025) evaluate SDV's Gaussian Copula, CTGAN, and TVAE across utility, privacy, and distributional fidelity metrics. CTGAN achieves the best overall performance, while Gaussian Copula excels at preserving conditional dependencies but lags in predictive tasks.

**Gap filled by our work:** We introduce a **variance-budget framework** for synthetic financial benchmark data: (1) Gaussian Copula captures inter-feature dependencies, (2) a latent-variable noise model with explicit variance allocation controls task difficulty, and (3) the budget parameter enables systematic ablation of model performance under varying signal-to-noise ratios. This goes beyond standard synthetic data generation by providing *controllable difficulty* for benchmarking multi-task financial recommendation systems.

---

## 9.10 Summary of Positioning

| Dimension | Prior Art | Our Contribution |
|-----------|-----------|-----------------|
| MTL Architecture | PLE (homogeneous MLP experts) | Heterogeneous expert basket (7 architectures) |
| Expert Types | Identical FFNs | DeepFM, Mamba+LNN+Transformer, HGCN, PersLay, Causal, LightGCN, OT |
| Task Transfer | MMoE gating / AdaTT fusion | AdaTT on heterogeneous PLE backbone |
| Explainability | Post-hoc SHAP/LIME | Inherent via expert gate weights |
| Feature Engineering | Standard tabular features | Multi-disciplinary (topology, hyperbolic geometry, causal, OT) |
| Deployment | GPU endpoint or batch | Knowledge distillation to LGBM + serverless Lambda |
| Regulatory | Ad-hoc compliance | Built-in FSS/EU AI Act alignment with reason generation |
| Benchmarking | Fixed datasets | Gaussian Copula + variance budget for controllable difficulty |
| Financial Domain | Single-task or simple MTL | 13-task multi-product recommendation with temporal modeling |

Our work synthesizes advances across multi-task learning, mixture-of-experts, diverse neural architectures, explainable AI, knowledge distillation, and financial regulation into a unified, production-ready system. The key novelty lies not in any single component but in their principled integration: heterogeneous experts matched to data modalities, inherent explainability through architectural design rather than post-hoc approximation, and a complete pipeline from training to compliant serverless serving.
