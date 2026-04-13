# Model Architecture Guide

This guide covers the PLE (Progressive Layered Extraction) multi-task learning
architecture, its expert networks, task heads, loss functions, and training
configuration.

---

## Table of Contents

1. [PLE Architecture Overview](#ple-architecture-overview)
2. [CGC Gating](#cgc-gating)
3. [Adaptive Task Transfer (adaTT)](#adaptive-task-transfer-adatt)
4. [Built-in Experts (6 types)](#built-in-experts)
5. [Adding a Custom Expert](#adding-a-custom-expert)
6. [Task Types (5 types)](#task-types)
7. [Adding a Custom Task Head](#adding-a-custom-task-head)
8. [Loss Functions (9 types)](#loss-functions)
9. [Multi-task Learning Configuration](#multi-task-learning-configuration)
10. [2-Phase Training](#2-phase-training)
11. [Knowledge Distillation (PLE to LGBM)](#knowledge-distillation)

---

## PLE Architecture Overview

The platform implements Progressive Layered Extraction (PLE), an advanced
multi-task learning architecture that solves the "seesaw" problem in multi-task
learning -- where improving one task degrades another.

```
                        Input Features (~350D total)
                               |
                    FeatureRouter [ACTIVE]
                    (auto-built from feature_groups.yaml target_experts)
          /         |         |        |         |         |         \
      [109D]     [129D]     [34D]    [32D]    [103D]    [66D]     [69D]   [~350D]
        |           |         |        |         |         |         |       |
   +--------+  +------+  +------+  +------+  +------+  +------+  +------+ +-----+
   | DeepFM |  |Temp. |  | HGCN |  |Pers- |  |Causal|  |Light-|  | OT   | | MLP |
   | Shared |  |Ens.  |  |Shared|  |Lay   |  |Shared|  |GCN   |  |Shared| |Task |
   +--------+  +------+  +------+  +------+  +------+  +------+  +------+ +-----+
          \        |         |        |         |         |         |       /
           +-------+---------+--------+---------+---------+---------+------+
                             |                  |
                     CGC Gating          CGC Gating        <-- per-task attention over experts
                     (Task A)            (Task B)
                             |                  |
                     adaTT Transfer Matrix       <-- gradient-based task affinity
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
| `PLEConfig` | `core/model/ple/config.py` | All model hyperparameters |
| `PLEModel` | `core/model/ple/model.py` | Main model class |
| `CGCLayer` | `core/model/ple/gating.py` | Customized Gate Control |
| `AdaTT` | `core/model/ple/adatt.py` | Adaptive Task Transfer |
| `FeatureRouter` | `core/model/ple/feature_router.py` | Expert input routing — **active**, auto-built from `feature_groups.yaml` `target_experts`; routes heterogeneous input dims per expert (32D–316D) |
| `ExpertRegistry` | `core/model/experts/registry.py` | Expert plugin system |
| `TaskRegistry` | `core/task/registry.py` | Task head plugin system |

---

## CGC Gating

Customized Gate Control (CGC) is the per-task attention mechanism that lets each
task learn which experts are most relevant to it.

Because **FeatureRouter is now active**, each shared expert receives a
different input dimensionality (32D–316D) rather than the uniform 316D total.
Expert outputs are projected to a common `output_dim` (default 64) before the
CGC gate concatenates them, so gating arithmetic remains dimension-agnostic.

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

adaTT measures gradient-based task affinity and dynamically transfers knowledge
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

    # Task groups define prior relationships
    task_groups:
      engagement:
        members: [ctr, cvr]
        intra_strength: 0.7
      value:
        members: [ltv, churn]
        intra_strength: 0.7
    inter_group_strength: 0.3      # Cross-group prior transfer
```

---

## Built-in Experts

Six expert network architectures ship with the platform. All are registered
in `core/model/experts/` via `@ExpertRegistry.register()`.

### 1. MLP Expert (`mlp`)

**File:** `core/model/experts/mlp.py`

A straightforward multi-layer perceptron. Serves as the baseline expert.

```yaml
experts:
  mlp:
    type: mlp
    enabled: true
    output_dim: 64
    hidden_dims: [128, 64]
    dropout: 0.2
    use_layer_norm: true
    activation: relu          # "relu" or "silu"
```

### 2. DeepFM Expert (`deepfm`)

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

### 3. Temporal Ensemble Expert (`temporal`)

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

### 4. Mamba Expert (`mamba`)

**File:** `core/model/experts/mamba.py`

Standalone Selective State Space Model (S6) for efficient long-sequence
modelling. O(L) complexity vs O(L^2) for attention.

```yaml
experts:
  mamba:
    type: mamba
    enabled: true
    output_dim: 64
    d_model: 128
    d_state: 16
    d_conv: 4
    expand_factor: 2
    n_layers: 2
    dropout: 0.1
```

### 5. Causal Expert (`causal`)

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

### 6. Optimal Transport Expert (`optimal_transport`)

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
  input_dim: 316               # Total feature dim (set by FeatureGroupPipeline.total_dim).
                               # With FeatureRouter active, each expert receives a
                               # routed subset: deepfm=109D, temporal_ensemble=129D,
                               # hgcn=34D, perslay=32D, causal=103D, lightgcn=66D,
                               # optimal_transport=69D, mlp_task=51D.
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
| `uncertainty` | Learned per-task uncertainty (Kendall 2018) | Multiple tasks with unknown noise levels |
| `gradnorm` | Gradient-normalisation (Chen 2018) | Tasks with very different loss scales |
| `dwa` | Dynamic Weight Average | Tasks with varying training speeds |

---

## 2-Phase Training

The platform supports a 2-phase training strategy that improves multi-task
generalisation:

### Phase 1: Shared Expert Warmup

- **Duration**: First N epochs (configurable)
- **What trains**: Shared experts and CGC gating
- **What is frozen**: Task-specific expert weights and tower weights
- **Purpose**: Learn good shared representations before tasks diverge

### Phase 2: Task Fine-tuning

- **Duration**: Remaining epochs
- **What trains**: Everything (shared + task experts + towers)
- **What changes**: Lower learning rate, shorter warmup
- **Purpose**: Specialise each task head while preserving shared structure

### Configuration

```yaml
training:
  # Phase 1
  phase1_epochs: 20
  phase1_lr: 0.001
  phase1_freeze_task_experts: true

  # Phase 2
  phase2_epochs: 30
  phase2_lr: 0.0003
  phase2_freeze_task_experts: false

  # Scheduler
  scheduler:
    name: cosine
    warmup_epochs: 5
    cosine_t0: 10
    phase2_warmup_epochs: 2
    phase2_cosine_t0: 6

  # Optimizer
  optimizer:
    name: adamw
    learning_rate: 0.001
    weight_decay: 0.01
    expert_lr_overrides:
      causal:
        lr: 0.0005
        weight_decay: 0.001

  # Mixed precision
  amp:
    enabled: true
    dtype: float16

  # Gradient handling
  gradient:
    clip_norm: 5.0
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
