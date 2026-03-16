# Feature Engineering Guide

This is the most important guide in the platform documentation. The feature
engineering system is designed around **Feature Groups** -- self-contained
units that define what features exist, how they are created, which experts
receive them, how to interpret them, and whether to include them in knowledge
distillation.

**Design principle**: one config object, many consumers. Adding a new feature
group requires editing only the YAML configuration; all downstream systems
(expert routing, interpretation, distillation) discover it automatically.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Built-in Transformers (7 types)](#built-in-transformers)
3. [Built-in Generators (5 types)](#built-in-generators)
4. [Creating a Custom Generator](#creating-a-custom-generator)
5. [Creating a Custom Transformer](#creating-a-custom-transformer)
6. [FeatureGroup Config Reference](#featuregroup-config-reference)
7. [Expert Routing](#expert-routing)
8. [Auto-propagation](#auto-propagation)
9. [Container Isolation](#container-isolation)
10. [DataFrame Backend](#dataframe-backend)

---

## Architecture Overview

```
                        feature_groups.yaml
                               |
                               v
                    FeatureGroupRegistry
                    (list of FeatureGroupConfig)
                               |
                    +----------+----------+
                    |                     |
            group_type="transform"  group_type="generate"
                    |                     |
            TransformerChain        FeatureGenerator
            (FeatureRegistry)       (FeatureGeneratorRegistry)
                    |                     |
                    +----------+----------+
                               |
                    FeatureGroupPipeline
                     .fit_transform(df)
                               |
                    +----------+----------+-----------+
                    |          |          |           |
               expert_routing  |    interpretation   distillation_config
               (PLE model)     |    (ReasonGenerator)  (Distill pipeline)
                          total_dim
                          group_ranges
                          (IG attribution)
```

**Key classes** (all in `core/feature/`):

| Class | File | Purpose |
|---|---|---|
| `AbstractFeatureTransformer` | `base.py` | Base class for column-level transformers |
| `FeatureRegistry` | `registry.py` | Plugin registry for transformers |
| `AbstractFeatureGenerator` | `generator.py` | Base class for feature generators |
| `FeatureGeneratorRegistry` | `generator.py` | Plugin registry for generators |
| `FeatureGroupConfig` | `group.py` | Single feature group definition |
| `FeatureGroupRegistry` | `group.py` | Manages all groups, provides lookups |
| `FeatureGroupPipeline` | `group_pipeline.py` | Orchestrator: fit, transform, routing |

---

## Built-in Transformers

Seven transformers ship out of the box. All are registered in
`core/feature/transformers.py` via the `@FeatureRegistry.register()` decorator.

Transformers **modify existing columns** (in contrast to generators, which
create new columns). They follow the sklearn `fit()` / `transform()` pattern.

### 1. StandardScaler

Z-score normalisation: `(x - mean) / std`.

```yaml
# YAML config
transformers: [standard_scaler]
columns: [age, income, tenure]
```

```python
# Python
from core.feature.transformers import StandardScaler

scaler = StandardScaler(columns=["age", "income"], clip_std=3.0)
scaler.fit(train_df)
scaled_df = scaler.transform(test_df)
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `columns` | `list[str]` | `None` (all numeric) | Columns to scale |
| `clip_std` | `float` | `None` | Clip scaled values to `[-clip_std, clip_std]` |

### 2. QuantileTransformer

Rank-based mapping to a normal or uniform distribution. Wraps
`sklearn.preprocessing.QuantileTransformer` internally.

```yaml
transformers: [quantile_transformer]
transformer_params:
  quantile_transformer:
    n_quantiles: 1000
    output_distribution: normal
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `columns` | `list[str]` | `None` | Columns to transform |
| `n_quantiles` | `int` | `1000` | Number of quantiles |
| `output_distribution` | `str` | `"normal"` | `"normal"` or `"uniform"` |
| `random_state` | `int` | `42` | Random seed |

### 3. LogTransformer

Applies `log1p(max(x, 0))` to reduce heavy-tail skew. Useful for financial
transaction amounts, balances, and other power-law distributed features.

```yaml
transformers: [log_transformer]
transformer_params:
  log_transformer:
    add_raw_copy: true
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `columns` | `list[str]` | `None` | Columns to transform |
| `add_raw_copy` | `bool` | `False` | Keep original as `{col}_raw` |

### 4. MinMaxScaler

Scale each column to the [0, 1] range (or a custom range).

```yaml
transformers: [minmax_scaler]
transformer_params:
  minmax_scaler:
    feature_range: [0.0, 1.0]
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `columns` | `list[str]` | `None` | Columns to scale |
| `feature_range` | `tuple` | `(0.0, 1.0)` | Desired output range |

### 5. LabelEncoder

Map each unique string value to a sequential integer. Unknown values seen at
`transform()` time are mapped to a configurable `unknown_value` index.

```yaml
transformers: [label_encoder]
columns: [user_segment, item_category, platform]
transformer_params:
  label_encoder:
    unknown_value: -1
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `columns` | `list[str]` | `None` (all object/category) | Columns to encode |
| `unknown_value` | `int` | `-1` | Code for unseen categories |

### 6. HashEncoder

Deterministic hash encoding for high-cardinality categoricals. Maps any string
to `[0, n_bins)` using MD5. Handles unseen values gracefully (same hash
function applies to any input).

```yaml
transformers: [hash_encoder]
columns: [merchant_id, product_sku]
transformer_params:
  hash_encoder:
    n_bins: 2048
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `columns` | `list[str]` | `None` | Columns to hash |
| `n_bins` | `int` | `1024` | Number of hash buckets |

### 7. NullFiller

Configurable null imputation with five strategies.

```yaml
transformers: [null_filler, standard_scaler]
transformer_params:
  null_filler:
    strategy: median
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `columns` | `list[str]` | `None` (all) | Columns to fill |
| `strategy` | `str` | `"zero"` | `"mean"`, `"median"`, `"zero"`, `"mode"`, `"constant"` |
| `fill_value` | `Any` | `None` | Value for `strategy="constant"` |

### Chaining Transformers

Transformers can be chained in order. The output of transformer N feeds into
transformer N+1:

```yaml
feature_groups:
  - name: financial_metrics
    type: transform
    transformers: [null_filler, log_transformer, standard_scaler]
    columns: [balance, monthly_spend, transaction_amount]
    transformer_params:
      null_filler:
        strategy: median
      log_transformer:
        add_raw_copy: false
```

---

## Built-in Generators

Five generator families ship with the platform. All are registered in
`core/feature/generators/` via `@FeatureGeneratorRegistry.register()`.

Generators **create entirely new feature columns** from raw data. They produce
a DataFrame containing only the newly generated columns; the
`FeatureGroupPipeline` concatenates this with other group outputs.

### 1. TDA Extractor (`tda_extractor`)

**File:** `core/feature/generators/tda.py`

Topological Data Analysis -- extracts persistent homology features from
customer transaction time series. Captures the "shape" of financial
behaviour that traditional statistics miss.

```yaml
- name: tda_topology
  type: generate
  generator: tda_extractor
  generator_params:
    short_window_days: 90
    long_window_days: 365
  output_dim: 70
  target_experts: [temporal]
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `short_window_days` | `int` | `90` | Short-window lookback (days) |
| `long_window_days` | `int` | `365` | Long-window lookback (days) |

**Output:** 70 features representing persistence diagrams, Betti numbers, and
topological summary statistics across two time windows.

### 2. HMM Triple Mode (`hmm_triple_mode`)

**File:** `core/feature/generators/hmm.py`

Hidden Markov Model state estimation across three behavioural modes. Each mode
captures a different aspect of customer dynamics.

```yaml
- name: hmm_states
  type: generate
  generator: hmm_triple_mode
  generator_params:
    modes: [journey, lifecycle, behavior]
    state_dim: 16
  output_dim: 48
  target_experts: [temporal]
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `modes` | `list[str]` | `["journey", "lifecycle", "behavior"]` | HMM mode names |
| `state_dim` | `int` | `16` | Hidden state dimension per mode |

**Output:** `len(modes) * state_dim` features (default: 3 * 16 = 48).

**Modes explained:**
- **journey**: Customer acquisition/engagement journey stages
- **lifecycle**: Long-term lifecycle phases (new, growing, mature, declining)
- **behavior**: Short-term behavioural patterns (active, dormant, bursty)

### 3. Hyperbolic Graph Embedding (`hyperbolic_embedding`)

**File:** `core/feature/generators/graph.py`

Embeds hierarchical category/product/region structures into hyperbolic
(Poincare) space, where distance from the origin captures generality and
angular position captures semantic similarity.

```yaml
- name: graph_embeddings
  type: generate
  generator: hyperbolic_embedding
  generator_params:
    hierarchy_sources: [category, product, region]
    curvature: 1.0
  output_dim: 20
  target_experts: []     # broadcast to all experts
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `hierarchy_sources` | `list[str]` | `[]` | Column names with hierarchical structure |
| `curvature` | `float` | `1.0` | Poincare ball curvature |

**Output:** 20 features (embedding coordinates in hyperbolic space).

### 4. Multidisciplinary (`multidisciplinary`)

**File:** `core/feature/generators/multidisciplinary.py`

Applies computational models from physics, chemistry, and social science to
financial behaviour data. Each module captures a different metaphorical
dynamic.

```yaml
- name: multidisciplinary
  type: generate
  generator: multidisciplinary
  generator_params:
    modules: [chemical_kinetics, epidemic_diffusion, interference, crime_pattern]
  output_dim: 24
  target_experts: [deepfm]
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `modules` | `list[str]` | `[]` | Which computational modules to enable |

**Available modules:**

| Module | Metaphor | Features |
|---|---|---|
| `chemical_kinetics` | Spending as chemical reactions (activation energy, rate constants) | 6 features |
| `epidemic_diffusion` | Product adoption as epidemic spreading (R0, recovery rate) | 6 features |
| `interference` | Multi-channel interactions as wave interference (constructive/destructive) | 6 features |
| `crime_pattern` | Anomalous behaviour detection using crime pattern analysis | 6 features |

### 5. Temporal Pattern (`temporal_pattern`)

**File:** `core/feature/generators/temporal.py`

Computes rolling-window aggregation features over multiple time horizons.
Produces trend, volatility, and acceleration metrics.

```yaml
- name: temporal_patterns
  type: generate
  generator: temporal_pattern
  generator_params:
    windows: [7, 30, 90]
    features: [amount, count]
  output_dim: 30
  target_experts: [temporal, mamba]
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `windows` | `list[int]` | `[7, 30, 90]` | Rolling window sizes (days) |
| `features` | `list[str]` | `[]` | Base feature columns to aggregate |

**Output:** `len(windows) * len(features) * n_stats` features, where `n_stats`
includes mean, std, trend slope, min, max per window-feature combination.

---

## Creating a Custom Generator

This is a full step-by-step tutorial for adding a new feature generator.

### Step 1: Create the generator class

Create a new file (e.g., `core/feature/generators/my_generator.py`):

```python
"""
My Custom Feature Generator.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from core.feature.generator import AbstractFeatureGenerator, FeatureGeneratorRegistry

logger = logging.getLogger(__name__)


@FeatureGeneratorRegistry.register(
    "my_generator",
    description="Generates custom domain-specific features.",
    tags=["custom", "domain"],
)
class MyGenerator(AbstractFeatureGenerator):
    """Custom generator that computes ratio and interaction features.

    Parameters
    ----------
    numerator_col : str
        Column name for the numerator.
    denominator_col : str
        Column name for the denominator.
    interaction_cols : list[str]
        Columns to compute pairwise interactions.
    """

    def __init__(
        self,
        numerator_col: str = "spend",
        denominator_col: str = "income",
        interaction_cols: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._numerator_col = numerator_col
        self._denominator_col = denominator_col
        self._interaction_cols = interaction_cols or []
        self._n_interactions = len(self._interaction_cols) * (len(self._interaction_cols) - 1) // 2

    def fit(self, df: Any, **context: Any) -> "MyGenerator":
        """Learn any internal parameters (statistics, thresholds, etc.)."""
        # For stateless generators, just mark as fitted
        # For stateful generators, compute running statistics here
        self._fitted = True
        logger.info("MyGenerator fitted on %d rows", len(df))
        return self

    def generate(self, df: Any, **context: Any) -> Any:
        """Generate new feature columns."""
        result = pd.DataFrame(index=df.index)

        # Feature 1: ratio
        denom = df[self._denominator_col].replace(0, 1)
        result["ratio_feature"] = df[self._numerator_col] / denom

        # Feature 2: log ratio
        result["log_ratio"] = np.log1p(np.maximum(result["ratio_feature"].values, 0))

        # Feature 3+: pairwise interactions
        for i, col_a in enumerate(self._interaction_cols):
            for col_b in self._interaction_cols[i + 1:]:
                name = f"interact_{col_a}_{col_b}"
                result[name] = df[col_a] * df[col_b]

        return result

    @property
    def output_dim(self) -> int:
        """Number of new feature columns produced."""
        return 2 + self._n_interactions  # ratio + log_ratio + interactions

    @property
    def output_columns(self) -> List[str]:
        """Explicit list of generated column names."""
        cols = ["ratio_feature", "log_ratio"]
        for i, col_a in enumerate(self._interaction_cols):
            for col_b in self._interaction_cols[i + 1:]:
                cols.append(f"interact_{col_a}_{col_b}")
        return cols
```

### Step 2: Register for auto-discovery

Make sure the module is imported at startup. Add it to
`core/feature/generators/__init__.py`:

```python
from . import tda, hmm, graph, multidisciplinary, temporal
from . import my_generator  # <-- add this line
```

### Step 3: Add to feature_groups.yaml

```yaml
feature_groups:
  # ... existing groups ...

  - name: my_custom_features
    type: generate
    generator: my_generator
    generator_params:
      numerator_col: monthly_spend
      denominator_col: annual_income
      interaction_cols: [tenure, balance, txn_count]
    output_dim: 5           # 2 + C(3,2) = 5
    target_experts: [deepfm, mlp]
    runtime: local
    interpretation:
      category: custom_ratios
      template: "{feature} shows a {direction} relationship with spending efficiency"
      narrative_lens: value
      primary_tasks: [ltv, cvr]
    distill: true
    distill_weight: 1.0
```

### Step 4: Verify it works

```python
from core.feature.generator import FeatureGeneratorRegistry

# Check registration
assert "my_generator" in FeatureGeneratorRegistry.list_registered()

# Build and test
gen = FeatureGeneratorRegistry.build(
    "my_generator",
    numerator_col="monthly_spend",
    denominator_col="annual_income",
    interaction_cols=["tenure", "balance"],
)

print(gen.output_dim)      # 3
print(gen.output_columns)  # ['ratio_feature', 'log_ratio', 'interact_tenure_balance']

# Test with data
import pandas as pd
df = pd.DataFrame({
    "monthly_spend": [100, 200, 300],
    "annual_income": [50000, 60000, 70000],
    "tenure": [12, 24, 36],
    "balance": [1000, 2000, 3000],
})
gen.fit(df)
features = gen.generate(df)
print(features)
```

---

## Creating a Custom Transformer

Transformers are simpler than generators -- they modify existing columns rather
than creating new ones.

### Step 1: Create the transformer class

```python
"""Custom transformer example: Winsorize extreme values."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.feature.base import AbstractFeatureTransformer
from core.feature.registry import FeatureRegistry


@FeatureRegistry.register(
    "winsorizer",
    description="Clip extreme values to [lower_pct, upper_pct] quantiles.",
    tags=["numeric", "outlier"],
)
class Winsorizer(AbstractFeatureTransformer):
    """Winsorize (clip) features to quantile bounds.

    Parameters
    ----------
    columns : list[str], optional
        Columns to winsorize. None = all numeric.
    lower_pct : float
        Lower quantile (default 0.01 = 1st percentile).
    upper_pct : float
        Upper quantile (default 0.99 = 99th percentile).
    """

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        lower_pct: float = 0.01,
        upper_pct: float = 0.99,
        **kwargs: Any,
    ) -> None:
        super().__init__(columns=columns, **kwargs)
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct
        self._bounds: Dict[str, tuple] = {}

    def fit(self, df: pd.DataFrame) -> "Winsorizer":
        cols = self._resolve_columns(df)
        self._fit_columns = cols
        self._bounds = {}
        for col in cols:
            lo = float(df[col].quantile(self.lower_pct))
            hi = float(df[col].quantile(self.upper_pct))
            self._bounds[col] = (lo, hi)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Winsorizer must be fitted before transform().")
        df = df.copy()
        for col in self._fit_columns:
            lo, hi = self._bounds[col]
            df[col] = df[col].clip(lo, hi)
        return df
```

### Step 2: Use in configuration

```yaml
feature_groups:
  - name: cleaned_financials
    type: transform
    transformers: [null_filler, winsorizer, standard_scaler]
    columns: [balance, monthly_spend, credit_limit]
    transformer_params:
      null_filler:
        strategy: median
      winsorizer:
        lower_pct: 0.01
        upper_pct: 0.99
    output_dim: 3
    target_experts: [deepfm, mlp]
```

---

## FeatureGroup Config Reference

Every field of `FeatureGroupConfig` (defined in `core/feature/group.py`):

### Core Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | (required) | Unique group identifier |
| `type` | `str` | `"transform"` | `"transform"` or `"generate"` |

### Generator Fields (type="generate")

| Field | Type | Default | Description |
|---|---|---|---|
| `generator` | `str` | `None` | Registry name of the generator |
| `generator_params` | `dict` | `{}` | Keyword arguments to the generator constructor |

### Transformer Fields (type="transform")

| Field | Type | Default | Description |
|---|---|---|---|
| `transformers` | `list[str]` | `[]` | Ordered list of transformer registry names |
| `transformer_params` | `dict` | `{}` | `{transformer_name: {param: value}}` overrides |
| `columns` | `list[str]` | `[]` | Input columns for transform-type groups |

### Output Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `output_dim` | `int` | `0` | Dimension of this group's output vector (auto-detected if 0) |
| `output_columns` | `list[str]` | `[]` | Explicit output column names (auto-populated from generator if empty) |

### Expert Routing

| Field | Type | Default | Description |
|---|---|---|---|
| `target_experts` | `list[str]` | `[]` | PLE experts that receive this group. Empty = broadcast to all |

### Interpretation

| Field | Type | Default | Description |
|---|---|---|---|
| `interpretation.category` | `str` | `"general"` | Semantic category for grouping in explanations |
| `interpretation.template` | `str` | `"{feature} is {value}, indicating {direction}"` | Format-string for rendering feature contributions |
| `interpretation.narrative_lens` | `str` | `"engagement"` | Perspective for LLM reason generation |
| `interpretation.primary_tasks` | `list[str]` | `[]` | Tasks for which this group is most relevant |

### Runtime Isolation

| Field | Type | Default | Description |
|---|---|---|---|
| `runtime` | `str` | `"local"` | `"local"` or `"container"` |
| `container.image` | `str` | `""` | ECR image URI (required when `runtime="container"`) |
| `container.instance_type` | `str` | `"ml.m5.xlarge"` | SageMaker instance type |
| `container.instance_count` | `int` | `1` | Number of processing instances |
| `container.volume_size_gb` | `int` | `30` | EBS volume size |
| `container.max_runtime_seconds` | `int` | `3600` | Job timeout |
| `container.requirements` | `list[str]` | `[]` | Extra pip packages |
| `container.env` | `dict` | `{}` | Environment variables |
| `container.s3_staging_prefix` | `str` | `"s3://sagemaker-default/..."` | S3 staging path |

### Distillation

| Field | Type | Default | Description |
|---|---|---|---|
| `distill` | `bool` | `True` | Include in knowledge distillation |
| `distill_weight` | `float` | `1.0` | Relative importance weight in distillation loss |

### Toggle

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `True` | Master toggle. Disabled groups are skipped entirely |

### Complete YAML Example

```yaml
feature_groups:
  - name: tda_topology
    type: generate
    generator: tda_extractor
    generator_params:
      short_window_days: 90
      long_window_days: 365
    output_dim: 70
    target_experts: [temporal]
    runtime: local
    interpretation:
      category: domain_topology
      template: "Transaction topology shows {pattern} pattern"
      narrative_lens: engagement
      primary_tasks: [ctr, cvr]
    distill: true
    distill_weight: 0.5
    enabled: true
```

---

## Expert Routing

Expert routing controls which features flow to which expert networks in the PLE
architecture.

### How it works

```
Feature Groups:                Expert Networks:
+-----------------+
| base_profile    |--+-------> [deepfm]
| (dim=4)         |  +-------> [mlp]
+-----------------+
| tda_topology    |----------> [temporal]
| (dim=70)        |
+-----------------+
| graph_embeddings|----------> [broadcast to ALL]
| (dim=20)        |
+-----------------+
| temporal_patterns|-+-------> [temporal]
| (dim=30)        |  +-------> [mamba]
+-----------------+
```

1. Each feature group specifies `target_experts: [expert1, expert2]`.
2. `FeatureGroupPipeline.expert_routing` computes a mapping of expert name
   to the integer feature indices it should receive.
3. The PLE model's `FeatureRouter` uses this mapping to slice the
   concatenated feature tensor before feeding each expert.
4. If `target_experts` is empty, the group is broadcast to all experts.

### Accessing the routing map

```python
pipeline = FeatureGroupPipeline(groups)
pipeline.fit(train_df)

# Expert -> feature indices
routing = pipeline.expert_routing
# e.g., {"deepfm": [0,1,2,3], "temporal": [4,...,73,124,...,153], "mamba": [124,...,153]}
```

### Why expert routing matters

Without routing, every expert sees every feature. This is wasteful:
- DeepFM excels at tabular feature interactions -- it does not need TDA topology
  features.
- Temporal experts (Mamba, PatchTST) are designed for sequential patterns --
  they should focus on temporal and state features.
- Causal experts learn DAG structure -- feeding them irrelevant features adds
  noise.

Expert routing improves both accuracy and training efficiency.

---

## Auto-propagation

When you define a feature group, several downstream systems automatically
discover and configure themselves:

### 1. Interpretation propagation

The `interpretation` block on each group flows to:

- **ReverseMapper**: Uses `category` and `template` to generate human-readable
  feature contribution descriptions.
- **TemplateEngine**: Uses `narrative_lens` and `primary_tasks` to select
  appropriate recommendation reason templates.
- **Task-aware IG**: Uses `primary_tasks` to weight feature importance per task.

```python
# Automatic: no manual wiring needed
pipeline = FeatureGroupPipeline(groups)
interp_map = pipeline.interpretation_map
# {"tda_topology": FeatureInterpretationConfig(category="domain_topology", ...)}
```

### 2. Distillation propagation

Groups with `distill: true` are automatically included in the knowledge
distillation pipeline. The `distill_weight` controls relative importance.

```python
distill_config = pipeline.distillation_config
# [{"name": "tda_topology", "dim_range": (4, 74), "weight": 0.5, "output_dim": 70}]
```

### 3. Dimension tracking

The pipeline automatically computes contiguous dimension ranges for each group:

```python
ranges = pipeline.group_ranges
# {"base_profile": (0, 4), "tda_topology": (4, 74), ...}

total = pipeline.total_dim
# 196 (sum of all enabled group output_dim)
```

These ranges are essential for Integrated Gradients attribution (slicing the
gradient vector per group).

---

## Container Isolation

Some generators have conflicting Python dependencies (e.g., TDA needs `ripser`,
graph needs `torch-geometric`). Container isolation lets you run generators in
isolated Docker containers via SageMaker Processing Jobs.

### When to use container isolation

- Generator has dependencies that conflict with the main environment
- Generator requires a GPU but the orchestrator does not
- Generator is computationally expensive and benefits from a dedicated instance
- Generator is developed by a separate team and needs a separate deployment cycle

### Configuration

```yaml
feature_groups:
  - name: tda_topology
    type: generate
    generator: tda_extractor
    generator_params:
      short_window_days: 90
    output_dim: 70
    target_experts: [temporal]
    runtime: container                    # <-- switch from "local" to "container"
    container:
      image: "123456789.dkr.ecr.ap-northeast-2.amazonaws.com/feature-tda:latest"
      instance_type: ml.m5.xlarge
      instance_count: 1
      volume_size_gb: 30
      max_runtime_seconds: 3600
      requirements: [ripser, persim]      # Extra pip installs
      env:
        FEATURE_GROUP_NAME: tda_topology
      s3_staging_prefix: "s3://my-bucket/feature-pipeline/staging"
```

### How it works

```
FeatureGroupPipeline.transform()
        |
        v
  runtime == "container"?
        |
    Yes: Upload input DataFrame to S3 as Parquet
         --> Launch SageMaker Processing Job
             (with specified container image)
         --> Wait for completion (polls every 15s)
         --> Download output Parquet from S3
         --> Return as DataFrame
        |
    No:  Run generator/transformer in-process
```

### Building a container image

```dockerfile
FROM python:3.11-slim

WORKDIR /opt/ml/processing

# Install generator dependencies
RUN pip install ripser persim numpy pandas pyarrow

# Copy generator code
COPY core/feature/generators/tda.py /opt/ml/processing/

# Entry point reads from /opt/ml/processing/input/
# and writes to /opt/ml/processing/output/
COPY containers/feature/entrypoint.py /opt/ml/processing/
ENTRYPOINT ["python", "entrypoint.py"]
```

---

## DataFrame Backend

The platform supports three DataFrame backends, selected automatically at import
time. All feature engineering code uses the global `df_backend` singleton.

### Backend selection priority

```
1. DuckDB  (default)    -- pip install duckdb
2. cuDF    (GPU)         -- pip install cudf-cu12
3. pandas  (fallback)    -- always available
```

### DuckDB (default)

- SQL-native operations
- Zero-copy Parquet I/O
- Automatic disk spill for out-of-core processing
- Direct S3 reads via httpfs extension
- Best for: most workloads, especially Parquet-heavy pipelines

```python
from core.data.dataframe import df_backend

df = df_backend.read_parquet("s3://bucket/data.parquet")
df = df_backend.query("SELECT * FROM df WHERE amount > 100", df=df)
```

### cuDF (GPU)

- RAPIDS GPU-accelerated DataFrames
- Automatic switch when GPU is available and data exceeds a configurable row
  threshold
- Best for: large-scale feature engineering (>1M rows)

### pandas (fallback)

- Used only when neither DuckDB nor cuDF is installed
- Full API compatibility
- Best for: development and testing

### Overriding the backend

Set environment variables before import:

```bash
export PLE_DATAFRAME_BACKEND=pandas    # Force pandas
export PLE_DATAFRAME_BACKEND=duckdb    # Force DuckDB
export PLE_DATAFRAME_BACKEND=cudf      # Force cuDF
```

Note: All transformers operate on pandas DataFrames internally. The backend
handles conversion to/from pandas automatically, so transformer code does not
need to be backend-aware.
