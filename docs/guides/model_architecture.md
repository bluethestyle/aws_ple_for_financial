# Model Architecture Guide

This guide covers the PLE (Progressive Layered Extraction) multi-task learning
architecture, its expert networks, task heads, loss functions, and training
configuration.

---

## Table of Contents

1. [PLE Architecture Overview](#ple-architecture-overview)
2. [CGC Gating](#cgc-gating)
3. [Adaptive Task Transfer (adaTT)](#adaptive-task-transfer-adatt)
4. [GradSurgery](#gradsurgery)
5. [Built-in Experts (7 types)](#built-in-experts)
6. [Adding a Custom Expert](#adding-a-custom-expert)
7. [Task Types (5 types)](#task-types)
8. [Adding a Custom Task Head](#adding-a-custom-task-head)
9. [Loss Functions (9 types)](#loss-functions)
10. [Multi-task Learning Configuration](#multi-task-learning-configuration)
11. [2-Phase Training](#2-phase-training)
12. [Knowledge Distillation (PLE to LGBM)](#knowledge-distillation)

---

## PLE Architecture Overview

The platform implements Progressive Layered Extraction (PLE), an advanced
multi-task learning architecture that solves the "seesaw" problem in multi-task
learning -- where improving one task degrades another.

```
                Input Features (~349D raw; 403D after Phase 0 log-transform)
                               |
                    FeatureRouter [ACTIVE]
                    (auto-built from feature_groups.yaml target_experts)
          /         |         |        |         |         |         \
      [109D]     [129D]     [27D]    [32D]    [103D]   [100D]     [69D]   [~349D]
        |           |         |        |         |         |         |       |
   +--------+  +------+  +------+  +------+  +------+  +------+  +------+ +-----+
   | DeepFM |  |Temp. |  | HGCN |  |Pers- |  |Causal|  |Light-|  | OT   | | MLP |
   | Shared |  |Ens.  |  |Shared|  |Lay   |  |Shared|  |GCN   |  |Shared| |Task |
   +--------+  +------+  +------+  +------+  +------+  +------+  +------+ +-----+
          \        |         |        |         |         |         |       /
           +-------+---------+--------+---------+---------+---------+------+
                             |                  |
                     CGC Gating (softmax)   CGC Gating (softmax)
                     (Task A)               (Task B)    <-- softmax outperforms sigmoid in heterogeneous MTL
                             |                  |
                     GradSurgery Projection      <-- tested, not adopted in production (see §GradSurgery)
                     (3 task-type groups)        <-- adaTT is canonical loss-level transfer; GradSurgery showed no improvement
                             |                  |
                     Task Tower A      Task Tower B        <-- task-specific MLPs
                     [128 -> 64]       [128 -> 64]
                             |                  |
                     Output A          Output B            <-- predictions + loss
                     (sigmoid)         (softmax)
```

### Key components

| Component | File | Purpose |
|---|---|---|
| `PLEConfig` | `core/model/ple/config.py` | All model hyperparameters — constructed exclusively by `config_builder.py` (single source of truth) |
| `build_ple_config()` | `core/model/config_builder.py` | Single source of truth for PLEConfig assembly — train.py calls only this function |
| `PLEModel` | `core/model/ple/model.py` | Main model class |
| `CGCLayer` | `core/model/ple/gating.py` | Customized Gate Control (softmax gate) |
| `GradSurgery` | `core/model/ple/grad_surgery.py` | Alternative to adaTT; tested but not adopted (see §GradSurgery or Paper 1 §5.4) |
| `AdaTT` | `core/model/ple/adatt.py` | Adaptive Task Transfer (disabled at 13-task scale; 156-pair instability) |
| `FeatureRouter` | `core/model/ple/feature_router.py` | Expert input routing — **active**, auto-built from `feature_groups.yaml` `target_experts`; routes heterogeneous input dims per expert |
| `ExpertRegistry` | `core/model/experts/registry.py` | Expert plugin system |
| `TaskRegistry` | `core/task/registry.py` | Task head plugin system |
| `PLEPredictor` | `core/inference/predictor.py` | Checkpoint loading + batch inference single interface |
| `PLEEvaluator` | `core/evaluation/evaluator.py` | Per-task metric aggregation separated by task type |

---

## CGC Gating

Customized Gate Control (CGC) is the per-task attention mechanism that lets each
task learn which experts are most relevant to it.

Because **FeatureRouter is now active**, each shared expert receives a
different input dimensionality rather than a uniform total.
Expert outputs are projected to a common `output_dim` (default 64) before the
CGC gate concatenates them, so gating arithmetic remains dimension-agnostic.

**Paper 1 finding**: **Softmax gating outperforms sigmoid in heterogeneous MTL settings.**
With 13 tasks spanning 7 binary, 3 multiclass, and 3 regression types, softmax provides
protective isolation of minority-type tasks from majority-type gradient corruption.
This reverses the conventional preference for sigmoid found in homogeneous-task literature.
Use `gate_type: softmax` (default); `sigmoid` is available for ablation only.

### How it works

For each task, CGC computes a softmax attention weight over all expert outputs:

```
gate_weights = softmax(W_gate @ concat(expert_outputs) + bias)
gated_output = sum(gate_weights[i] * expert_output[i])
```

### Configuration

```yaml
model:
  cgc:
    enabled: true
    bias_high: 1.0          # Initial bias toward task-specific experts
    bias_low: -1.0           # Initial bias away from shared experts
    dim_normalize: false     # Normalize by sqrt(dim)
    entropy_lambda: 0.01     # Entropy regularization (prevents collapse)
```

### Entropy regularization

`entropy_lambda` adds a penalty to the loss that encourages the gate to use
multiple experts rather than collapsing to a single one:

```
L_entropy = -entropy_lambda * sum(gate_weights * log(gate_weights))
```

---

## Adaptive Task Transfer (adaTT)

> **Status (Paper 1 finding, revised 2026-04-17)**: At 13-task scale the
> loss-level adaTT mechanism is **null** after correcting five
> implementation bugs — ΔAUC = −0.001 vs PLE-only baseline, within
> single-seed measurement noise. An earlier draft reported −0.019 and
> attributed it to algorithmic instability; the corrected measurements
> locate the cause in the implementation rather than the mechanism.
> **GradSurgery was evaluated as an alternative and was *not* adopted**
> either — it matched adaTT within noise and added non-trivial VRAM
> overhead from `retain_graph`. Production runs PLE softmax directly
> without adaTT or GradSurgery; both configs are retained for ablation
> comparison only with `enabled: false`.

adaTT measures loss-level task affinity and dynamically transfers knowledge
between related tasks while blocking negative transfer.

### How it works

1. **Affinity measurement**: After each `grad_interval` steps, compute the
   cosine similarity of per-task gradient vectors.
2. **Transfer matrix**: Build a task-to-task transfer matrix where positive
   affinity enables knowledge sharing and negative affinity blocks it.
3. **Warmup**: During `warmup_epochs`, use prior-based transfer (task group
   definitions) blended with measured affinity.
4. **Negative transfer detection**: If affinity drops below
   `negative_transfer_threshold`, transfer is zeroed out.

### Why adaTT fails at 13-task scale

With 13 tasks, there are $13 \times 12 = 156$ directed transfer pairs.
Given only 7 active affinity-measurement epochs (10 total minus 3 warmup),
each pair receives insufficient gradient samples, producing noisy affinity
estimates that corrupt transfer. The root cause is a scaling mismatch: the
original adaTT paper validated on 2–4 tasks, not 13.

### Configuration

```yaml
model:
  adatt:
    enabled: true
    transfer_lambda: 0.1           # Transfer strength
    temperature: 1.0               # Softmax temperature for affinity
    warmup_epochs: 10              # Epochs before measuring affinity
    negative_transfer_threshold: -0.1
    ema_decay: 0.9                 # EMA smoothing of affinity matrix
    prior_blend_start: 0.5         # Prior weight at start of warmup
    prior_blend_end: 0.1           # Prior weight at end of warmup
    grad_interval: 10              # Steps between affinity measurements
    max_transfer_ratio: 0.5        # Cap on transfer amount

    # Canonical 4 task groups (Financial DNA). Used for post-hoc
    # interpretation and rule-based fallback template selection — NOT
    # for CGC router gating (routing is per-task).
    task_groups:
      engagement:
        members: [churn_signal, top_mcc_shift]
        intra_strength: 0.7
      lifecycle:
        members: [segment_prediction, will_acquire_deposits, will_acquire_cards,
                  will_acquire_loans, will_acquire_funds, will_acquire_insurance]
        intra_strength: 0.7
      value:
        members: [nba_primary, cross_sell_count]
        intra_strength: 0.7
      consumption:
        members: [next_mcc, mcc_diversity_trend, product_stability]
        intra_strength: 0.7
    inter_group_strength: 0.3      # Cross-group prior transfer
```

---

## GradSurgery

> **Status: Tested but not adopted in production.** Results shown below
> are from the ablation study; production deployment uses PLE softmax
> without GradSurgery or adaTT.

GradSurgery was evaluated as an alternative to adaTT at 13-task scale. Instead
of estimating all 156 pair-wise affinities, it groups tasks into 3 task-type
buckets (binary / multiclass / regression) and projects conflicting gradients
between these groups using PCGrad-style cosine projection. The experiment showed
no meaningful AUC/F1 improvement over the PLE-only baseline while incurring
significant VRAM overhead due to the retained computation graph.

### Comparison vs adaTT (ablation, both not adopted)

| | adaTT (post-bugfix) | GradSurgery |
|---|---|---|
| Transfer pairs | 156 (13×12), loss-level hybrid | 3 (task-type groups), gradient-level PCGrad |
| Stability at 13 tasks | Acceptable after 5 impl bug fixes | Stable |
| Ablation result (AUC) | −0.001 vs PLE-softmax (within noise) | within noise vs PLE-softmax |
| VRAM overhead | Low | Non-trivial (`retain_graph`) |
| Warmup required | 10 epochs | None |
| Production status | **not adopted** | **not adopted** |

### Configuration

```yaml
model:
  grad_surgery:
    enabled: false         # experiment flag — false in production (not adopted)
    task_type_groups:
      binary:   [churn_signal, will_acquire_deposits, will_acquire_investments,
                 will_acquire_accounts, will_acquire_lending, will_acquire_payments,
                 top_mcc_shift]
      multiclass: [nba_primary, segment_prediction, next_mcc]
      regression: [product_stability, cross_sell_count, mcc_diversity_trend]
    projection_strength: 1.0   # 1.0 = full PCGrad projection (ablation reproduction only)

  # adaTT also disabled in production
  adatt:
    enabled: false
```

**File:** `core/model/ple/grad_surgery.py`

---

## Built-in Experts

Seven expert network architectures ship with the platform. All are registered
in `core/model/experts/` via `@ExpertRegistry.register()`.

The 7 shared experts correspond to the 5-axis feature taxonomy:
- **Snapshot axis**: DeepFM (tabular interactions)
- **Temporal axis**: Temporal Ensemble (Mamba + PatchTST)
- **Hierarchy axis**: HGCN (MCC merchant category hierarchy)
- **Topology axis**: PersLay (TDA behavioral shape)
- **Causal axis**: Causal (DAG-based representation)
- **Relations axis**: LightGCN (customer-product graph)
- **Distribution axis**: Optimal Transport (segment distribution shifts)

The MLP is the per-task expert (not a shared expert) — each task has its own
MLP tower on top of the shared CGC-gated representation.

### 1. DeepFM Expert (`deepfm`)

**File:** `core/model/experts/deepfm.py`

Deep Factorization Machine combining:
- **FM layer**: Efficient 2nd-order feature interactions
- **Deep network**: Higher-order non-linear interactions
- **Cross network (DCN)**: Optional explicit feature crossing

Best for tabular data with discrete feature interactions (e.g., user segment
x product category).

```yaml
experts:
  deepfm:
    type: deepfm
    enabled: true
    output_dim: 64
    embedding_dim: 16
    num_fields: 10
    deep_hidden_dims: [256, 128]
    use_cross_network: true
    cross_layers: 3
    dropout: 0.1
```

### 2. Temporal Ensemble Expert (`temporal`)

**File:** `core/model/experts/temporal.py`

Ensembles Mamba (SSM) and PatchTST (Transformer) with learned per-sample gating.

- **Mamba branch**: Linear-complexity long-range dependency modelling
- **PatchTST branch**: Patch-based attention for periodic patterns
- **Gating network**: Decides per-sample how much to trust each branch

```yaml
experts:
  temporal:
    type: temporal
    enabled: true
    output_dim: 64
    d_model: 128
    mamba_d_state: 16
    patch_size: 16
    n_heads: 4
    n_transformer_layers: 2
    dropout: 0.1
```

### 4. HGCN Expert (`hgcn`)

**File:** `core/model/experts/hgcn.py`

Hyperbolic Graph Convolutional Network that embeds the MCC merchant category
hierarchy (10 L1 categories → 30 L2 subcategories → 109 leaf codes) into
Poincare space. Receives `merchant_hierarchy` features (27D) via FeatureRouter.

```yaml
experts:
  hgcn:
    type: hgcn
    enabled: true
    output_dim: 64
    hidden_dim: 64
    curvature: 1.0
    dropout: 0.1
```

### 5. PersLay Expert (`perslay`)

**File:** `core/model/experts/perslay.py`

Topological Data Analysis via persistence diagram vectorisation. Captures
behavioral shape features (e.g., cyclic spending patterns, lifestyle transitions)
that are invisible to point-cloud methods. Receives `tda_global` and `tda_local`
features (32D total) via FeatureRouter.

```yaml
experts:
  perslay:
    type: perslay
    enabled: true
    output_dim: 64
    hidden_dim: 64
    n_elements: 100
    dropout: 0.1
```

### 6. LightGCN Expert (`lightgcn`)

**File:** `core/model/experts/lightgcn.py`

Graph convolution for collaborative filtering on the customer-product bipartite
co-holding graph (24 products). Distinct from HGCN: LightGCN learns
customer-product affinity, not product taxonomy. Receives `product_hierarchy`
and `graph_collaborative` features (100D) via FeatureRouter.

```yaml
experts:
  lightgcn:
    type: lightgcn
    enabled: true
    output_dim: 64
    hidden_dim: 64
    n_layers: 2
    dropout: 0.1
```

### 7. Causal Expert (`causal`)

**File:** `core/model/experts/causal.py`

Learns a directed acyclic graph (DAG) over a compressed variable space using
the NOTEARS continuous optimization approach. Produces causally-informed
representations.

```yaml
experts:
  causal:
    type: causal
    enabled: true
    output_dim: 64
    hidden_dim: 128
    n_causal_vars: 32
    dag_lambda: 0.01           # Acyclicity loss weight
    sparsity_lambda: 0.001     # L1 sparsity on adjacency matrix
    dropout: 0.2
```

### 7. Optimal Transport Expert (`optimal_transport`)

**File:** `core/model/experts/ot.py`

Projects features onto probability simplices and computes Sinkhorn distances
to learnable reference distributions. Produces geometry-aware representations.

```yaml
experts:
  optimal_transport:
    type: optimal_transport
    enabled: true
    output_dim: 64
    hidden_dim: 128
    n_reference_distributions: 16
    sinkhorn_iterations: 10
    sinkhorn_epsilon: 0.1
    distribution_dim: 32
    dropout: 0.2
```

---

## Adding a Custom Expert

### Step 1: Create the expert class

Create `core/model/experts/my_expert.py`:

```python
"""My custom expert network."""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from .base import AbstractExpert
from .registry import ExpertRegistry


@ExpertRegistry.register("my_expert")
class MyExpert(AbstractExpert):
    """Custom expert that applies attention-based feature aggregation.

    Config keys
    -----------
    output_dim : int
        Expert output dimension (default 64).
    n_heads : int
        Number of attention heads (default 4).
    dropout : float
        Dropout rate (default 0.1).
    """

    def __init__(self, input_dim: int, config: Dict[str, Any]):
        super().__init__(input_dim, config)

        self._output_dim = config.get("output_dim", 64)
        n_heads = config.get("n_heads", 4)
        dropout = config.get("dropout", 0.1)

        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.projection = nn.Sequential(
            nn.Linear(input_dim, self._output_dim),
            nn.LayerNorm(self._output_dim),
            nn.ReLU(),
        )

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # x shape: [batch, input_dim]
        # Add sequence dimension for attention
        x_seq = x.unsqueeze(1)  # [batch, 1, input_dim]
        attn_out, _ = self.attention(x_seq, x_seq, x_seq)
        pooled = attn_out.squeeze(1)  # [batch, input_dim]
        return self.projection(pooled)  # [batch, output_dim]
```

### Step 2: Register for auto-discovery

Add to `core/model/experts/__init__.py`:

```python
from . import my_expert  # <-- add this
```

### Step 3: Use in configuration

```yaml
model:
  experts:
    my_expert:
      type: my_expert
      enabled: true
      output_dim: 64
      n_heads: 4
      dropout: 0.1
```

---

## Task Types

Five task types are built in. Each maps to a concrete `AbstractTask` subclass
registered in `core/task/registry.py`.

### 1. Binary Classification (`binary`)

Single-target binary prediction (click, purchase, churn).

| Property | Value |
|---|---|
| Default loss | BCE with logits |
| Output activation | Sigmoid |
| Output dim | 1 |
| Primary metric | AUC-ROC |

```yaml
tasks:
  - name: click
    type: binary
    loss: focal            # or "bce"
    loss_weight: 1.0
    label_col: clicked
    tower_dims: [128, 64]
```

### 2. Multiclass Classification (`multiclass`)

Multi-class prediction (product category, segment).

| Property | Value |
|---|---|
| Default loss | Cross-Entropy |
| Output activation | Softmax |
| Output dim | `num_classes` |
| Primary metric | Accuracy / Macro-F1 |

```yaml
tasks:
  - name: product_type
    type: multiclass
    loss: focal
    num_classes: 5
    output_dim: 5
    label_col: product_label
```

### 3. Regression (`regression`)

Continuous value prediction (lifetime value, spending amount).

| Property | Value |
|---|---|
| Default loss | Huber |
| Output activation | Linear (identity) |
| Output dim | 1 |
| Primary metric | RMSE / MAE |

```yaml
tasks:
  - name: ltv
    type: regression
    loss: huber            # or "mse", "mae", "quantile"
    loss_weight: 0.5
    label_col: lifetime_value
    normalize_target: true
    huber_delta: 1.0
```

### 4. Ranking (`ranking`)

Listwise ranking of items (next-best-action).

| Property | Value |
|---|---|
| Default loss | ListNet |
| Output activation | Linear |
| Output dim | 1 |
| Primary metric | NDCG |

```yaml
tasks:
  - name: nba
    type: ranking
    loss: listnet
    label_col: relevance_score
```

### 5. Contrastive (`contrastive`)

Embedding-based retrieval via InfoNCE loss. Produces L2-normalised query
embeddings for ANN lookup.

| Property | Value |
|---|---|
| Default loss | InfoNCE |
| Output activation | L2 normalisation |
| Output dim | embedding dimension |
| Primary metric | Recall@K |

```yaml
tasks:
  - name: retrieval
    type: contrastive
    loss: infonce
    temperature: 0.07
    output_dim: 64
    extra:
      n_keys: 50000
      embedding_dim: 64
```

---

## Adding a Custom Task Head

### Step 1: Create the task class

```python
"""Custom uplift modelling task head."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from core.task.base import AbstractTask, TaskConfig, TaskOutput
from core.task.registry import TaskRegistry


@TaskRegistry.register("uplift")
class UpliftTask(AbstractTask):
    """T-Learner uplift task head.

    Trains two sub-towers: one for treatment, one for control.
    The uplift is the difference in predicted outcomes.
    """

    def __init__(self, config: TaskConfig, tower_input_dim: int, **kwargs):
        super().__init__(config, tower_input_dim, **kwargs)
        # Override: build two towers
        self.treatment_tower = self._build_tower(tower_input_dim)
        self.control_tower = self._build_tower(tower_input_dim)

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
        treatment_flags: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if treatment_flags is None:
            return self.loss_fn(logits, labels.float()).mean() * self.config.loss_weight

        treatment_mask = treatment_flags.bool()
        loss = torch.tensor(0.0, device=logits.device)

        if treatment_mask.any():
            t_loss = self.loss_fn(logits[treatment_mask], labels[treatment_mask].float())
            loss = loss + t_loss.mean()
        if (~treatment_mask).any():
            c_loss = self.loss_fn(logits[~treatment_mask], labels[~treatment_mask].float())
            loss = loss + c_loss.mean()

        return self.uncertainty_weighted_loss(loss) * self.config.loss_weight

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits)
```

### Step 2: Use in configuration

```yaml
tasks:
  - name: uplift_purchase
    type: uplift
    loss: bce
    loss_weight: 1.0
    label_col: purchased
    extra:
      uplift_method: t_learner
```

---

## Loss Functions

Nine loss functions are available, implemented in `core/task/losses.py`.

| Loss | Type | Key Parameters | Use Case |
|---|---|---|---|
| `bce` | Classification | `pos_weight` | Binary classification (balanced data) |
| `focal` | Classification | `alpha`, `gamma` | Binary/multi-class with class imbalance |
| `ce` | Classification | `label_smoothing` | Multi-class classification |
| `mse` | Regression | -- | Mean squared error |
| `mae` | Regression | -- | Mean absolute error (robust to outliers) |
| `huber` | Regression | `delta` | Robust regression (MSE near zero, MAE far) |
| `quantile` | Regression | `quantiles` | Distributional regression |
| `listnet` | Ranking | -- | Listwise ranking |
| `infonce` | Retrieval | `temperature` | Contrastive learning |

### Loss selection

Losses are selected per-task via `loss_type` in `TaskConfig`. The special value
`auto` selects the canonical default for the task type:

| Task Type | Auto Loss |
|---|---|
| `binary` | `bce` |
| `multiclass` | `ce` |
| `regression` | `huber` |
| `ranking` | `listnet` |
| `contrastive` | `infonce` |

### Focal Loss details

Focal Loss (Lin et al., 2017) down-weights well-classified examples:

```
FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
```

- `gamma = 0` recovers standard cross-entropy
- `gamma = 2.0` (default) aggressively focuses on hard examples
- `alpha = 0.25` (default) balances positive/negative classes

```yaml
tasks:
  - name: rare_event
    type: binary
    loss: focal
    focal_alpha: 0.25
    focal_gamma: 2.0
```

---

## Multi-task Learning Configuration

### Full PLEConfig example

```yaml
model:
  # Global dimensions
  input_dim: 403               # Total feature dim after Phase 0 (349D raw + 54 log-transform copies).
                               # With FeatureRouter active, each expert receives a
                               # routed subset: deepfm=109D, temporal_ensemble=129D,
                               # hgcn=27D, perslay=32D, causal=103D, lightgcn=100D,
                               # optimal_transport=69D.
                               # input_dim is derived dynamically from feature_schema.json.
  task_expert_output_dim: 32

  # Task definitions
  task_names: [click, convert, ltv, nba]

  # Shared experts
  num_shared_experts: 4
  shared_expert:
    hidden_dims: [256, 256]
    output_dim: 64
    dropout: 0.1
    activation: relu
    use_layer_norm: true

  # Task experts
  num_task_experts_per_task: 1
  task_expert:
    hidden_dims: [128, 64]
    output_dim: 32

  # PLE stacking
  num_extraction_layers: 2     # Deeper = more capacity, but slower

  # CGC
  cgc:
    enabled: true
    entropy_lambda: 0.01

  # adaTT
  adatt:
    enabled: true
    transfer_lambda: 0.1
    warmup_epochs: 10
    task_groups:
      engagement:
        members: [click, convert]
        intra_strength: 0.7
      value:
        members: [ltv, nba]
        intra_strength: 0.7
    inter_group_strength: 0.3

  # Loss weighting strategy
  loss_weighting:
    strategy: uncertainty      # "fixed" | "uncertainty" | "gradnorm" | "dwa"

  # Task tower
  task_tower:
    hidden_dims: [64, 32]
    dropout: 0.2

  # Task-specific overrides
  task_overrides:
    click:
      output_dim: 1
      activation: sigmoid
      task_type: binary
    convert:
      output_dim: 1
      activation: sigmoid
      task_type: binary
    ltv:
      output_dim: 1
      activation: null
      task_type: regression
    nba:
      output_dim: 5
      activation: softmax
      task_type: multiclass
```

### Loss weighting strategies

| Strategy | Description | When to use |
|---|---|---|
| `fixed` | Static weights from `TaskConfig.loss_weight` | Baseline, well-understood task balance |
| `uncertainty` | Learned per-task uncertainty (Kendall 2018) with per-task `loss_weight` correction | Multiple tasks with unknown noise levels |
| `gradnorm` | Gradient-normalisation (Chen 2018) | Tasks with very different loss scales |
| `dwa` | Dynamic Weight Average | Tasks with varying training speeds |

---

## 2-Phase Training

The platform supports a 2-phase training strategy that improves multi-task
generalisation:

### Phase 1: All-parameter training

- **Duration**: First 15 epochs (configurable via `phase1.epochs`)
- **What trains**: All shared experts, CGC gating, task experts, and towers
- **Purpose**: Learn joint representations before task-head specialisation

### Phase 2: Shared-frozen task-head fine-tuning

- **Duration**: 8 additional epochs (configurable via `phase2.epochs`)
- **What is frozen**: Shared expert weights
- **What trains**: Task-specific tower heads only
- **Purpose**: Fine-tune task heads without disturbing shared representations

### Current training hyperparameters (santander, 941K users)

```yaml
training:
  batch_size: 5632           # VRAM-optimised for g4dn.xlarge T4
  learning_rate: 0.0005
  weight_decay: 0.01
  gradient_clip_norm: 5.0
  epochs: 50
  phase1:
    epochs: 15
  phase2:
    epochs: 8
    freeze_shared: true

  scheduler:
    type: cosine
    warmup_epochs: 3          # 3-epoch warmup before cosine annealing

  amp:
    enabled: true             # AMP FP16 — ~2x speedup on T4 GPU
    dtype: float16
```

---

## Knowledge Distillation

After training the PLE model, knowledge is distilled to a LightGBM student
model for low-latency serving.

### Why distill?

| | PLE (Teacher) | LGBM (Student) |
|---|---|---|
| Latency | ~50ms | ~2ms |
| GPU required | Yes | No |
| Feature interactions | Learned | Manual or soft-label based |
| Serving cost | High (GPU instances) | Low (CPU Lambda) |

### How it works

1. **Soft labels**: PLE produces probability predictions on the training data.
2. **Feature selection**: Feature groups with `distill: true` are included.
3. **Weighted training**: Each group's `distill_weight` controls its
   contribution to the distillation loss.
4. **Per-task models**: One LGBM model is trained per task using soft labels.

### Configuration

Feature groups control distillation participation:

```yaml
feature_groups:
  - name: tda_topology
    distill: true              # Include in distillation
    distill_weight: 0.5        # Hard to distill, lower weight

  - name: hmm_states
    distill: false             # Skip (not distillable to LGBM)

  - name: base_profile
    distill: true
    distill_weight: 1.0        # Easy to distill, full weight
```

The distillation pipeline reads these settings automatically:

```python
pipeline = FeatureGroupPipeline(groups)
pipeline.fit(train_df)

# Automatically computed from feature group configs
distill_config = pipeline.distillation_config
# [
#   {"name": "tda_topology", "dim_range": (4, 74), "weight": 0.5, "output_dim": 70},
#   {"name": "base_profile", "dim_range": (0, 4), "weight": 1.0, "output_dim": 4},
# ]
```

---

## PLEConfig Build — config_builder.py (single source of truth)

> **2026-04-14**: `PLEConfig` inline construction delegated to `core/model/config_builder.py`.
> The previous pattern of managing 435+ lines of model-build logic directly in train.py
> is retired. **config_builder.py is the only place** where task_loss_weights, adaTT
> task_groups, logit_transfers, and FeatureRouter injection are assembled.

```python
# train.py — caller side (3 lines to get a complete PLEConfig)
from core.model.config_builder import build_ple_config

ple_config = build_ple_config(pipeline_cfg, feature_schema)
model = PLEModel(ple_config)
```

When adding new parameters, modify only `config_builder.py`. Do not touch train.py.

---

## PLEPredictor Usage

`PLEPredictor` in `core/inference/predictor.py` provides a single interface for
checkpoint loading and batch inference.

```python
from core.inference.predictor import PLEPredictor

# Restore config + weights from checkpoint
predictor = PLEPredictor.from_checkpoint(
    checkpoint_path="/opt/ml/model/best_model.pt",
    device="cuda",
)

# Batch inference — task_name -> numpy array
predictions = predictor.predict(eval_dataloader)
# {
#   "churn_signal": np.ndarray(shape=(N,)),      # binary sigmoid
#   "nba_primary":  np.ndarray(shape=(N, 5)),    # multiclass softmax
#   "cross_sell_count": np.ndarray(shape=(N,)),  # regression
# }
```

## PLEEvaluator Usage

`PLEEvaluator` in `core/evaluation/evaluator.py` aggregates metrics separated
by task type. Cross-task averaging is not used — metric semantics are
incompatible across binary / multiclass / regression tasks.

```python
from core.evaluation.evaluator import PLEEvaluator

evaluator = PLEEvaluator(task_configs)  # list from pipeline.yaml tasks
result = evaluator.evaluate(predictions, labels)

# result.avg_auc        -> mean AUC-ROC across binary tasks
# result.avg_f1_macro   -> mean Macro-F1 across multiclass tasks
# result.avg_mae        -> mean MAE across regression tasks
# result.per_task       -> {task_name: {"metric": value, ...}}
```
