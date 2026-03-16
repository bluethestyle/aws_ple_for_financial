# Plugin Development Guide

The AWS PLE Platform uses a consistent registry/decorator pattern for all
extension points. This guide covers every plugin type with full code examples
and config snippets.

---

## Table of Contents

1. [Plugin Architecture](#plugin-architecture)
2. [Custom Feature Generators](#custom-feature-generators)
3. [Custom Feature Transformers](#custom-feature-transformers)
4. [Custom Experts](#custom-experts)
5. [Custom Task Heads](#custom-task-heads)
6. [Custom Scorers](#custom-scorers)
7. [Custom Constraint Filters](#custom-constraint-filters)
8. [Custom LLM Providers](#custom-llm-providers)

---

## Plugin Architecture

Every extension point in the platform follows the same three-step pattern:

```
1. DEFINE:    Create a class inheriting from the abstract base
2. REGISTER:  Use the @Registry.register("name") decorator
3. CONFIGURE: Reference by name in YAML config
```

```
           Registry Pattern
           ================

    @Registry.register("my_plugin")
    class MyPlugin(AbstractBase):     <--- Step 1 + 2 (code)
        ...

    config:
      plugin_name: my_plugin          <--- Step 3 (YAML)
      plugin_params:
        param1: value1

         |
         v

    Registry.build("my_plugin", param1="value1")
    --> MyPlugin(param1="value1")
```

### Extension point summary

| Extension | Abstract Base | Registry | File |
|---|---|---|---|
| Feature Generator | `AbstractFeatureGenerator` | `FeatureGeneratorRegistry` | `core/feature/generator.py` |
| Feature Transformer | `AbstractFeatureTransformer` | `FeatureRegistry` | `core/feature/registry.py` |
| Expert Network | `AbstractExpert` | `ExpertRegistry` | `core/model/experts/registry.py` |
| Task Head | `AbstractTask` | `TaskRegistry` | `core/task/registry.py` |
| Scorer | `AbstractScorer` | `ScorerRegistry` | `core/recommendation/scorer.py` |
| Constraint Filter | `AbstractFilter` | `FilterRegistry` | `core/recommendation/constraint_engine.py` |
| LLM Provider | `AbstractLLMProvider` | `LLMProviderFactory` | `core/recommendation/reason/llm_provider.py` |

---

## Custom Feature Generators

Generators create entirely new feature columns from raw data.

### Abstract interface

```python
class AbstractFeatureGenerator(ABC):
    def fit(self, df, **context) -> self:       # Learn internal state
    def generate(self, df, **context) -> DataFrame:  # Produce new columns
    @property
    def output_dim(self) -> int:                # Number of new columns
    @property
    def output_columns(self) -> list[str]:      # Column names
```

### Full example: Fourier features

```python
"""
Fourier Feature Generator -- periodic pattern extraction via FFT.

File: core/feature/generators/fourier.py
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.feature.generator import AbstractFeatureGenerator, FeatureGeneratorRegistry

logger = logging.getLogger(__name__)


@FeatureGeneratorRegistry.register(
    "fourier_features",
    description="Extract dominant Fourier frequencies from time series columns.",
    tags=["temporal", "frequency"],
)
class FourierFeatureGenerator(AbstractFeatureGenerator):
    """Extract top-K Fourier frequencies and their amplitudes.

    Parameters
    ----------
    columns : list[str]
        Columns to analyse (each treated as a time series).
    top_k : int
        Number of dominant frequencies to extract per column.
    include_amplitude : bool
        Whether to include amplitude alongside frequency.
    """

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        top_k: int = 5,
        include_amplitude: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._columns = columns or []
        self._top_k = top_k
        self._include_amplitude = include_amplitude
        self._trained_freq_indices: Dict[str, np.ndarray] = {}

    def fit(self, df: Any, **context: Any) -> "FourierFeatureGenerator":
        """Identify dominant frequency indices from training data."""
        pdf = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)

        for col in self._columns:
            if col not in pdf.columns:
                continue
            values = pdf[col].fillna(0).values.astype(np.float64)
            fft_vals = np.fft.rfft(values)
            magnitudes = np.abs(fft_vals)
            # Exclude DC component (index 0)
            magnitudes[0] = 0
            top_indices = np.argsort(magnitudes)[-self._top_k:][::-1]
            self._trained_freq_indices[col] = top_indices

        self._fitted = True
        logger.info(
            "FourierFeatureGenerator fitted on %d columns, top_k=%d",
            len(self._columns), self._top_k,
        )
        return self

    def generate(self, df: Any, **context: Any) -> Any:
        """Generate Fourier frequency features."""
        pdf = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
        result = pd.DataFrame(index=pdf.index)

        for col in self._columns:
            if col not in pdf.columns or col not in self._trained_freq_indices:
                continue

            indices = self._trained_freq_indices[col]
            for rank, freq_idx in enumerate(indices):
                result[f"{col}_freq_{rank}"] = float(freq_idx)
                if self._include_amplitude:
                    # Per-row amplitude at the dominant frequency
                    # (simplified: use global amplitude from training)
                    result[f"{col}_amp_{rank}"] = 0.0  # placeholder

        return result

    @property
    def output_dim(self) -> int:
        multiplier = 2 if self._include_amplitude else 1
        return len(self._columns) * self._top_k * multiplier

    @property
    def output_columns(self) -> List[str]:
        cols = []
        for col in self._columns:
            for rank in range(self._top_k):
                cols.append(f"{col}_freq_{rank}")
                if self._include_amplitude:
                    cols.append(f"{col}_amp_{rank}")
        return cols
```

### Registration and configuration

Add to `core/feature/generators/__init__.py`:

```python
from . import fourier
```

YAML configuration:

```yaml
feature_groups:
  - name: fourier_spending
    type: generate
    generator: fourier_features
    generator_params:
      columns: [daily_spend, daily_txn_count]
      top_k: 5
      include_amplitude: true
    output_dim: 20
    target_experts: [temporal]
    interpretation:
      category: frequency_analysis
      template: "Spending frequency analysis shows {direction} periodicity"
      narrative_lens: engagement
      primary_tasks: [timing, engagement]
```

### Container-isolated generator

For generators with heavy or conflicting dependencies:

```yaml
  - name: fourier_spending
    type: generate
    generator: fourier_features
    generator_params:
      columns: [daily_spend]
      top_k: 5
    output_dim: 10
    runtime: container
    container:
      image: "123456789.dkr.ecr.ap-northeast-2.amazonaws.com/feature-fourier:latest"
      instance_type: ml.m5.xlarge
      requirements: [scipy, pywt]     # Extra packages for the container
      env:
        SCIPY_BACKEND: numpy
```

---

## Custom Feature Transformers

Transformers modify existing columns (in place or with copies).

### Abstract interface

```python
class AbstractFeatureTransformer(ABC):
    def fit(self, df: pd.DataFrame) -> self:
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
```

### Full example: Power transform

```python
"""
Box-Cox / Yeo-Johnson power transform.

File: core/feature/transformers_custom.py (or add to transformers.py)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.feature.base import AbstractFeatureTransformer
from core.feature.registry import FeatureRegistry


@FeatureRegistry.register(
    "power_transformer",
    description="Box-Cox or Yeo-Johnson power transform for Gaussianisation.",
    tags=["numeric", "scaler"],
)
class PowerTransformer(AbstractFeatureTransformer):
    """Apply sklearn PowerTransformer.

    Parameters
    ----------
    columns : list[str], optional
    method : str
        ``"yeo-johnson"`` (handles negative values) or ``"box-cox"``
        (requires positive values).
    standardize : bool
        Whether to apply zero-mean unit-variance normalisation after
        the power transform.
    """

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        method: str = "yeo-johnson",
        standardize: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(columns=columns, **kwargs)
        self.method = method
        self.standardize = standardize
        self._pt = None

    def fit(self, df: pd.DataFrame) -> "PowerTransformer":
        from sklearn.preprocessing import PowerTransformer as SklearnPT

        cols = self._resolve_columns(df)
        self._fit_columns = cols
        self._pt = SklearnPT(method=self.method, standardize=self.standardize)
        self._pt.fit(df[cols].values.astype(np.float64))
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("PowerTransformer must be fitted first.")
        df = df.copy()
        cols = self._fit_columns
        df[cols] = self._pt.transform(df[cols].values.astype(np.float64))
        return df
```

### Configuration

```yaml
feature_groups:
  - name: gaussianised_metrics
    type: transform
    transformers: [null_filler, power_transformer]
    columns: [balance, monthly_spend, credit_utilisation]
    transformer_params:
      null_filler:
        strategy: zero
      power_transformer:
        method: yeo-johnson
        standardize: true
    output_dim: 3
    target_experts: [deepfm, mlp]
```

---

## Custom Experts

Expert networks are the building blocks of the PLE architecture.

### Abstract interface

```python
class AbstractExpert(nn.Module, ABC):
    def __init__(self, input_dim: int, config: Dict[str, Any]):
    @property
    def output_dim(self) -> int:
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
```

### Full example: Graph Attention Expert

```python
"""
Graph Attention Expert -- applies multi-head attention treating features
as graph nodes.

File: core/model/experts/graph_attention.py
"""
from __future__ import annotations

import math
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AbstractExpert
from .registry import ExpertRegistry


@ExpertRegistry.register("graph_attention")
class GraphAttentionExpert(AbstractExpert):
    """Multi-head attention over features treated as a fully-connected graph.

    Config keys
    -----------
    output_dim : int
        Expert output dimension (default 64).
    n_heads : int
        Number of attention heads (default 4).
    n_layers : int
        Number of stacked attention layers (default 2).
    dropout : float
        Dropout rate (default 0.1).
    """

    def __init__(self, input_dim: int, config: Dict[str, Any]):
        super().__init__(input_dim, config)

        self._output_dim = config.get("output_dim", 64)
        n_heads = config.get("n_heads", 4)
        n_layers = config.get("n_layers", 2)
        dropout = config.get("dropout", 0.1)
        d_model = config.get("d_model", 128)

        # Project features into d_model space
        self.input_proj = nn.Linear(input_dim, d_model)

        # Stacked self-attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, self._output_dim),
            nn.LayerNorm(self._output_dim),
            nn.ReLU(),
        )

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [batch, input_dim] feature tensor

        Returns:
            [batch, output_dim] expert output
        """
        # Treat each feature as a "token" in a 1-length sequence
        h = self.input_proj(x).unsqueeze(1)  # [batch, 1, d_model]
        h = self.encoder(h)                   # [batch, 1, d_model]
        h = h.squeeze(1)                      # [batch, d_model]
        return self.output_proj(h)            # [batch, output_dim]
```

### Registration and configuration

Add to `core/model/experts/__init__.py`:

```python
from . import graph_attention
```

Model configuration:

```yaml
model:
  experts:
    graph_attention:
      type: graph_attention
      enabled: true
      output_dim: 64
      n_heads: 4
      n_layers: 2
      d_model: 128
      dropout: 0.1
```

Route specific features to the expert:

```yaml
feature_groups:
  - name: graph_embeddings
    target_experts: [graph_attention]
```

---

## Custom Task Heads

Task heads convert gated expert output into predictions and loss.

### Abstract interface

```python
class AbstractTask(ABC, nn.Module):
    def __init__(self, config: TaskConfig, tower_input_dim: int):
    def compute_loss(self, logits, labels, sample_weights=None, **kwargs) -> Tensor:
    def predict(self, logits) -> Tensor:
    # forward() is implemented by the base class
```

### Full example: Ordinal regression task

```python
"""
Ordinal Regression Task -- for ordered categorical targets.

File: core/task/ordinal.py
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.task.base import AbstractTask, TaskConfig, TaskOutput
from core.task.registry import TaskRegistry


@TaskRegistry.register("ordinal")
class OrdinalTask(AbstractTask):
    """Ordinal regression via cumulative link model.

    Predicts P(Y > k) for each threshold k, then derives ordinal
    class probabilities as P(Y = k) = P(Y > k-1) - P(Y > k).

    Expects labels as integer ordinal levels (0, 1, ..., K-1).
    """

    def __init__(self, config: TaskConfig, tower_input_dim: int, **kwargs):
        # Override output_dim to be K-1 thresholds for K classes
        n_classes = config.num_classes or config.output_dim
        config.output_dim = n_classes - 1
        super().__init__(config, tower_input_dim, **kwargs)

        self.n_classes = n_classes
        # Learnable thresholds (initialised as evenly spaced)
        initial_thresholds = torch.linspace(-2, 2, n_classes - 1)
        self.thresholds = nn.Parameter(initial_thresholds)

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # logits: [B, 1] (scalar latent value)
        # labels: [B] integer ordinal levels
        latent = logits.squeeze(-1)  # [B]
        targets = labels.long()

        # Cumulative probabilities: P(Y > k) = sigmoid(latent - threshold_k)
        # Loss: sum of binary cross-entropy at each threshold
        loss = torch.tensor(0.0, device=logits.device)
        for k in range(self.n_classes - 1):
            cumulative_prob = torch.sigmoid(latent - self.thresholds[k])
            target_k = (targets > k).float()
            loss = loss + F.binary_cross_entropy(
                cumulative_prob, target_k, reduction="none"
            ).mean()

        loss = loss / (self.n_classes - 1)

        if sample_weights is not None:
            loss = (loss * sample_weights).mean()

        loss = self.uncertainty_weighted_loss(loss)
        return loss * self.config.loss_weight

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        """Predict ordinal class probabilities."""
        latent = logits.squeeze(-1)  # [B]

        # P(Y > k) for each threshold
        cum_probs = torch.sigmoid(
            latent.unsqueeze(-1) - self.thresholds.unsqueeze(0)
        )  # [B, K-1]

        # P(Y = k) = P(Y > k-1) - P(Y > k)
        ones = torch.ones(cum_probs.size(0), 1, device=cum_probs.device)
        zeros = torch.zeros(cum_probs.size(0), 1, device=cum_probs.device)
        extended = torch.cat([ones, cum_probs, zeros], dim=-1)  # [B, K+1]
        probs = extended[:, :-1] - extended[:, 1:]  # [B, K]

        return probs
```

### Configuration

```yaml
tasks:
  - name: satisfaction
    type: ordinal
    num_classes: 5                    # 5 ordinal levels (e.g., 1-5 stars)
    loss_weight: 1.0
    label_col: satisfaction_level
    tower_dims: [128, 64]
    primary_metric: mae
```

---

## Custom Scorers

Scorers combine multi-task predictions into a single priority score.

### Abstract interface

```python
class AbstractScorer(ABC):
    def __init__(self, config: Dict[str, Any]):
    def score(self, customer_id, item_id, predictions, context=None) -> ScoringResult:
    def score_batch(self, records) -> List[ScoringResult]:   # default: loops over score()
```

### Full example: Bayesian scorer

```python
"""
Bayesian Scorer -- Thompson Sampling based scoring.

File: core/recommendation/scorers/bayesian.py
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from core.recommendation.scorer import (
    AbstractScorer,
    ScorerRegistry,
    ScoringResult,
)


@ScorerRegistry.register("bayesian_thompson")
class BayesianThompsonScorer(AbstractScorer):
    """Thompson Sampling scorer for exploration-exploitation balance.

    Treats each task prediction as a Beta distribution parameter and
    samples from it. This naturally trades off exploitation (high
    predicted scores) with exploration (uncertain predictions).

    Config example::

        scorer:
          bayesian_thompson:
            task_weights:
              ctr: 0.3
              cvr: 0.4
              ltv: 0.3
            exploration_factor: 1.0   # Higher = more exploration
            seed: 42
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.task_weights = config.get("task_weights", {})
        self.exploration_factor = config.get("exploration_factor", 1.0)
        seed = config.get("seed", 42)
        self._rng = np.random.default_rng(seed)

    def score(
        self,
        customer_id: str,
        item_id: str,
        predictions: Dict[str, float],
        context: Optional[Dict[str, Any]] = None,
    ) -> ScoringResult:
        components: Dict[str, float] = {}
        sampled_score = 0.0
        total_weight = 0.0

        for task, weight in self.task_weights.items():
            if task not in predictions:
                continue

            p = float(np.clip(predictions[task], 1e-6, 1 - 1e-6))

            # Beta distribution parameters from prediction probability
            alpha = p * self.exploration_factor * 10
            beta_param = (1 - p) * self.exploration_factor * 10

            # Thompson sample
            sample = float(self._rng.beta(max(alpha, 0.1), max(beta_param, 0.1)))
            sampled_score += weight * sample
            total_weight += weight
            components[f"sample_{task}"] = sample
            components[f"pred_{task}"] = p

        if total_weight > 0:
            sampled_score /= total_weight

        return ScoringResult(
            customer_id=customer_id,
            item_id=item_id,
            score=sampled_score,
            components=components,
            metadata={"exploration_factor": self.exploration_factor},
        )
```

### Configuration

```yaml
pipeline:
  scorer_name: bayesian_thompson

scorer:
  bayesian_thompson:
    task_weights:
      ctr: 0.3
      cvr: 0.4
      ltv: 0.3
    exploration_factor: 1.0
    seed: 42
```

---

## Custom Constraint Filters

Filters eliminate ineligible candidate items before top-K selection.

### Abstract interface

```python
class AbstractFilter(ABC):
    def __init__(self, config: Dict[str, Any]):
    def check(self, candidate, customer_context) -> FilterResult:
```

### Full example: Regulatory compliance filter

```python
"""
Regulatory Compliance Filter -- blocks products based on regulatory rules.

File: core/recommendation/filters/regulatory.py
"""
from __future__ import annotations

from typing import Any, Dict

from core.recommendation.constraint_engine import (
    AbstractFilter,
    FilterRegistry,
    FilterResult,
)


@FilterRegistry.register("regulatory")
class RegulatoryComplianceFilter(AbstractFilter):
    """Block products that violate regulatory requirements.

    Checks:
    - Age restrictions (e.g., credit products require age >= 18)
    - Income thresholds (e.g., high-risk products require income verification)
    - Geographic restrictions (some products not available in all regions)

    Config example::

        filters:
          regulatory:
            enabled: true
            min_age: 18
            min_income_for_high_risk: 50000
            restricted_regions: [region_x, region_y]
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.min_age = config.get("min_age", 18)
        self.min_income_high_risk = config.get("min_income_for_high_risk", 50000)
        self.restricted_regions = set(config.get("restricted_regions", []))

    def check(
        self,
        candidate: Dict[str, Any],
        customer_context: Dict[str, Any],
    ) -> FilterResult:
        # Age check
        age = customer_context.get("age", 999)
        if age < self.min_age:
            return FilterResult(
                passed=False,
                filter_name="regulatory",
                reason=f"Customer age ({age}) below minimum ({self.min_age})",
                details={"check": "age", "value": age, "threshold": self.min_age},
            )

        # Income check for high-risk products
        item_info = candidate.get("item_info", {})
        if item_info.get("risk_level") == "high":
            income = customer_context.get("annual_income", 0)
            if income < self.min_income_high_risk:
                return FilterResult(
                    passed=False,
                    filter_name="regulatory",
                    reason=f"Income ({income}) below threshold for high-risk product",
                    details={"check": "income", "value": income},
                )

        # Geographic restriction
        region = customer_context.get("region", "")
        if region in self.restricted_regions:
            return FilterResult(
                passed=False,
                filter_name="regulatory",
                reason=f"Product not available in region: {region}",
                details={"check": "region", "value": region},
            )

        return FilterResult(passed=True, filter_name="regulatory")
```

### Configuration

```yaml
filters:
  regulatory:
    enabled: true
    min_age: 18
    min_income_for_high_risk: 50000
    restricted_regions:
      - restricted_zone_a
      - restricted_zone_b
```

---

## Custom LLM Providers

LLM providers power the optional AI-based self-checking of recommendation
reasons.

### Abstract interface

```python
class AbstractLLMProvider(ABC):
    def __init__(self, config: Dict[str, Any]):
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
```

### Full example: Anthropic Claude via API

```python
"""
Claude LLM Provider via Anthropic API.

File: core/recommendation/reason/providers/claude_api.py
"""
from __future__ import annotations

from typing import Any, Dict

from core.recommendation.reason.llm_provider import (
    AbstractLLMProvider,
    LLMProviderFactory,
)


@LLMProviderFactory.register("claude_api")
class ClaudeAPIProvider(AbstractLLMProvider):
    """Direct Anthropic API provider for Claude models.

    Config::

        llm_provider:
          backend: claude_api
          claude_api:
            api_key_env: ANTHROPIC_API_KEY   # env var name
            model: claude-sonnet-4-20250514
            max_tokens: 512
            temperature: 0.0
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        import os
        api_key_env = config.get("api_key_env", "ANTHROPIC_API_KEY")
        self._api_key = os.environ.get(api_key_env, "")
        self._model = config.get("model", "claude-sonnet-4-20250514")
        self._max_tokens = config.get("max_tokens", 512)
        self._temperature = config.get("temperature", 0.0)

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=self._api_key)
        response = client.messages.create(
            model=self._model,
            max_tokens=max_tokens or self._max_tokens,
            temperature=self._temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
```

### Configuration

```yaml
llm_provider:
  backend: claude_api
  claude_api:
    api_key_env: ANTHROPIC_API_KEY
    model: claude-sonnet-4-20250514
    max_tokens: 512
    temperature: 0.0
```

---

## Plugin Discovery

All plugins must be imported before they can be used. The recommended approach
is to add imports to the relevant `__init__.py` file:

| Plugin Type | Register in |
|---|---|
| Feature Generator | `core/feature/generators/__init__.py` |
| Feature Transformer | `core/feature/transformers.py` or custom module imported in `__init__.py` |
| Expert | `core/model/experts/__init__.py` |
| Task Head | `core/task/__init__.py` |
| Scorer | Import in your pipeline entry point |
| Filter | Import in your pipeline entry point |
| LLM Provider | Import in your pipeline entry point |

### Verifying registration

```python
# Check what's registered
from core.feature.generator import FeatureGeneratorRegistry
print(FeatureGeneratorRegistry.list_registered())

from core.feature.registry import FeatureRegistry
print(FeatureRegistry.list_registered())

from core.model.experts.registry import ExpertRegistry
print(ExpertRegistry.list_available())

from core.task.registry import TaskRegistry
print(TaskRegistry.list_registered())

from core.recommendation.scorer import ScorerRegistry
print(ScorerRegistry.list_registered())
```

### Testing plugins

Every plugin should be testable in isolation:

```python
# Test a generator
gen = FeatureGeneratorRegistry.build("my_generator", param1="value")
gen.fit(test_df)
result = gen.generate(test_df)
assert result.shape[1] == gen.output_dim
assert list(result.columns) == gen.output_columns

# Test an expert
expert = ExpertRegistry.create("my_expert", input_dim=128, config={"output_dim": 64})
x = torch.randn(32, 128)
out = expert(x)
assert out.shape == (32, 64)

# Test a scorer
scorer = ScorerRegistry.create("my_scorer", config)
result = scorer.score("user_1", "item_1", {"ctr": 0.8, "cvr": 0.3})
assert isinstance(result.score, float)
```
