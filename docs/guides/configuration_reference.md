# Configuration Reference

Complete reference for all YAML configuration files in the platform.

---

## Table of Contents

1. [Config Architecture](#config-architecture)
2. [feature_groups.yaml](#feature_groupsyaml)
3. [pipeline.yaml (common)](#pipelineyaml-common)
4. [Dataset Config (datasets/*.yaml)](#dataset-config-datasetsyaml)
5. [recommendation.yaml](#recommendationyaml)
6. [monitoring.yaml](#monitoringyaml)
7. [Compliance Config](#compliance-config)
8. [Serving Config](#serving-config)
9. [Training Config](#training-config)
10. [PLE Model Config](#ple-model-config)
11. [Agent Configuration (agent.yaml)](#agent-configuration-configsfinancialagent-yaml)
12. [Checklist Configuration (checklist.yaml)](#checklist-configuration-configsfinancialchecklist-yaml)

---

## Config Architecture

### Directory layout

```
configs/
├── pipeline.yaml              ← common: model, training, distillation, aws
├── datasets/
│   ├── santander.yaml         ← Santander benchmark (tasks, labels, ablation)
│   └── example.yaml           ← template for new users
├── santander/
│   ├── feature_groups.yaml    ← feature group definitions
│   └── item_universe.yaml     ← NBA item catalog
└── financial/                 ← on-prem operation configs
    ├── agent.yaml
    ├── checklist.yaml
    └── fact_extraction.yaml
```

### What goes where

| Belongs in `pipeline.yaml` | Belongs in `datasets/*.yaml` |
|---|---|
| PLE model architecture (expert basket, CGC, adaTT) | Task definitions (`tasks:`) |
| Training hyperparameters (LR, scheduler, AMP) | Label derivation rules |
| Distillation config | Adapter name |
| AWS infrastructure (instance type, S3 bucket, Spot) | Data source path (`data.source`) |
| Serving and monitoring defaults | Feature column lists |
| | Ablation scenario matrix |
| | Scoring / constraint overrides |
| | Sequence config |

### Merge behaviour (`deep_merge`)

The system uses a **3-layer split-config** (CLAUDE.md §1.1):

1. **`configs/pipeline.yaml`** — common settings shared across datasets (model, training, distillation, AWS, compliance, monitoring).
2. **`configs/datasets/{name}.yaml`** — dataset-specific overrides (tasks, labels, adapter, ablation scenarios).
3. **`configs/{name}/feature_groups.yaml`** — feature-group definitions and expert routing, referenced from the dataset YAML via `feature_groups_file`.

At runtime layers 1 and 2 are deep-merged with `load_merged_config(common, dataset)`:

1. `pipeline.yaml` is loaded as the base.
2. `datasets/<name>.yaml` is recursively merged on top.
3. For dict values, keys are merged recursively (neither file wins wholesale).
4. For list values and scalar values, the dataset file wins.

Layer 3 (`feature_groups.yaml`) is loaded separately by the feature-engineering stage based on the path declared in the merged config (`feature_groups_file`). Adding a new dataset therefore touches all three files: copy `configs/datasets/example.yaml`, copy `configs/{existing}/feature_groups.yaml`, and implement the adapter module.

```python
# core/pipeline/config.py
config = load_merged_config(
    "configs/pipeline.yaml",
    "configs/datasets/santander.yaml"
)
```

### CLI / SageMaker usage

```bash
# Local training — split-config pattern
python containers/training/train.py \
  --config configs/pipeline.yaml \
  --dataset configs/datasets/my_bank.yaml

# SageMaker hyperparameters
hyperparameters = {
    "config":         "configs/pipeline.yaml",
    "dataset_config": "configs/datasets/my_bank.yaml",
}
```

**Backward compatibility:** If `dataset_config` / `--dataset` is omitted,
the single file passed to `--config` is used as-is (legacy pattern). Existing
single-file setups require no changes.

### Creating a new dataset config

```bash
# Step 1 — copy the template
cp configs/datasets/example.yaml configs/datasets/my_bank.yaml

# Step 2 — edit my_bank.yaml: fill tasks, labels, adapter, data.source
# Step 3 — add feature groups
cp -r configs/santander configs/my_bank
# edit configs/my_bank/feature_groups.yaml

# Step 4 — implement adapter
cp src/adapters/santander.py src/adapters/my_bank.py
# edit my_bank.py: raw → standardized DataFrame

# Step 5 — run
python containers/training/train.py \
  --config configs/pipeline.yaml \
  --dataset configs/datasets/my_bank.yaml
```

Minimum viable dataset YAML:

```yaml
dataset:
  name: my_bank
  feature_groups_file: configs/my_bank/feature_groups.yaml

adapter: my_bank

data:
  source: s3://my-bucket/data/train/
  format: parquet
  id_col: customer_id          # read from config, never hardcoded in code
  date_col: snapshot_date

tasks:
  - name: churn
    type: binary
    loss: focal
    loss_weight: 1.0
    label_col: is_churned

features:
  numeric:
    - age
    - tenure_months
  categorical:
    - region_code
    - segment
```

---

## feature_groups.yaml

**File:** `configs/feature_groups.yaml`
**Loaded by:** `core.feature.group_config.load_feature_groups()`

Defines every feature group in the pipeline. This is the single source of truth
for feature engineering -- downstream systems (expert routing, interpretation,
distillation) auto-configure from these definitions.

### Top-level structure

```yaml
feature_groups:
  - name: <string>                    # REQUIRED. Unique group identifier
    type: <string>                    # REQUIRED. "transform" or "generate"
    # ... (see fields below)
```

### Transform-type group fields

```yaml
feature_groups:
  - name: base_profile
    type: transform

    # Transformer chain (applied in order)
    transformers:                      # REQUIRED for type=transform
      - standard_scaler
      - null_filler

    # Per-transformer parameter overrides
    transformer_params:               # Optional
      null_filler:
        strategy: median
      standard_scaler:
        clip_std: 3.0

    # Input columns
    columns:                          # REQUIRED for type=transform
      - age
      - income
      - tenure

    # Output dimension (auto-detected if 0)
    output_dim: 3                     # Optional, default: len(columns)

    # Output column names (auto-detected if empty)
    output_columns: []                # Optional
```

### Generate-type group fields

```yaml
feature_groups:
  - name: tda_topology
    type: generate

    # Generator registry name
    generator: tda_extractor          # REQUIRED for type=generate

    # Generator constructor parameters
    generator_params:                 # Optional
      short_window_days: 90
      long_window_days: 365

    # Output dimension
    output_dim: 70                    # Recommended (auto-detected from generator if 0)
```

### Common fields (both types)

```yaml
    # Expert routing: which PLE experts receive this group's features
    target_experts:                   # Optional, default: [] (broadcast to all)
      - deepfm
      - mlp

    # Interpretation metadata for recommendation explanations
    interpretation:                   # Optional
      category: demographics          # Semantic category for grouping
      template: "{feature} is {value}, indicating a {direction} customer profile"
      narrative_lens: lifecycle        # Perspective for LLM reasons
      primary_tasks:                  # Tasks for which this group matters most
        - churn
        - ltv

    # Runtime isolation
    runtime: local                    # "local" (default) or "container"
    container:                        # Only when runtime=container
      image: "123456789.dkr.ecr.region.amazonaws.com/repo:tag"  # REQUIRED
      instance_type: ml.m5.xlarge     # Default: ml.m5.xlarge
      instance_count: 1               # Default: 1
      volume_size_gb: 30              # Default: 30
      max_runtime_seconds: 3600       # Default: 3600
      requirements:                   # Extra pip packages
        - ripser
        - persim
      env:                            # Environment variables
        MY_VAR: my_value
      s3_staging_prefix: "s3://bucket/staging"

    # Normalization exclusions (CLAUDE.md §1.9)
    exclude_from_scaler: []           # Optional. Skip StandardScaler for the listed sub-kinds.
                                      # Valid items: `categorical_id` (integers that look numeric
                                      # but encode IDs — e.g. customer_id-derived buckets) and
                                      # `probability` (already in [0,1] such as calibrated
                                      # downstream scores). Applying StandardScaler to these
                                      # distorts the distribution, so Stage 2 of the 3-stage
                                      # normalization pipeline skips them.

    # Knowledge distillation inclusion
    distill: true                     # Default: true
    distill_weight: 1.0               # Default: 1.0 (relative weight)

    # Master toggle
    enabled: true                     # Default: true
```

### Available transformers

| Name | Description | Tags |
|---|---|---|
| `standard_scaler` | Z-score normalisation (mean=0, std=1) | numeric, scaler |
| `quantile_transformer` | Quantile-based mapping to normal/uniform | numeric, scaler |
| `log_transformer` | `log1p(max(x, 0))` for heavy tails | numeric, scaler |
| `minmax_scaler` | Scale to [0, 1] range | numeric, scaler |
| `label_encoder` | Categorical string to integer | categorical, encoder |
| `hash_encoder` | Deterministic hash to `[0, n_bins)` | categorical, encoder |
| `null_filler` | Configurable null imputation | numeric, categorical, imputer |

### Available generators

| Name | Description |
|---|---|
| `tda_extractor` | Topological Data Analysis (persistent homology) |
| `hmm_triple_mode` | Hidden Markov Model state estimation |
| `hyperbolic_embedding` | Hyperbolic graph embeddings |
| `multidisciplinary` | Cross-domain computational models |
| `temporal_pattern` | Rolling-window aggregation features |
| `mamba` | Mamba SSM-based temporal sequence features |
| `gmm` | Gaussian Mixture Model cluster assignments |
| `model_features` | Model-derived features (distillation outputs) |
| `merchant_hierarchy` | MCC hierarchy + brand SVD features |
| `graph` | Graph collaborative filtering features |

---

## pipeline.yaml (common)

**File:** `configs/pipeline.yaml`
**Loaded by:** `core.pipeline.config.load_merged_config()`

Dataset-agnostic defaults. Do **not** put task lists, label derivations, or
adapter settings here — those belong in `configs/datasets/<name>.yaml`.

### Distillation block

```yaml
distillation:
  # Threshold below which a teacher is considered too weak for distillation
  # and the LGBM student is trained on hard labels directly (MRM safety
  # floor, CLAUDE.md §1.8). Expressed as a multiple of the random-baseline
  # metric for that task type.
  teacher_threshold: 2.0              # 2x random baseline minimum

  # Calibration configuration (CLAUDE.md §1.8). Dataset-agnostic — the
  # task list is declared per dataset. Only probability-critical binary
  # classification tasks need Platt scaling; ranking-oriented binary
  # tasks and regression tasks routed to Layer 3 do NOT receive
  # calibration.
  calibration:
    enabled: true
    method: platt                     # Only "platt" supported currently
    tasks: []                         # Override per dataset in datasets/{name}.yaml
                                      # Example (santander): [churn_signal]
                                      # Set per dataset, not here.
```

### Compliance block (see `## Compliance Config` below for detail)

```yaml
compliance:
  tracking:
    backend: sagemaker                # Shipped default (2026-04-21). See §Compliance Config.
  promotion_gate:
    enabled: true                     # Production default. See §Compliance Config.
```

### Serving block (see `## Serving Config` below for detail)

```yaml
serving:
  competition:
    auto_promote: false               # Posture enforcement. See §Serving Config.
```

## Dataset Config (datasets/*.yaml)

**Files:** `configs/datasets/*.yaml`
**Loaded by:** `core.pipeline.config.load_merged_config()` (merged on top of pipeline.yaml)

Dataset-specific settings. At minimum, define `tasks`, `adapter`, `data.source`,
and `feature_groups_file`. All other keys fall back to `pipeline.yaml` defaults.

Top-level configuration for a training pipeline run.

### Complete schema

```yaml
# Pipeline identity
task_name: my_pipeline                # Human-readable name

# Task definitions
tasks:
  - name: click                       # REQUIRED. Unique task name
    type: binary                      # REQUIRED. binary|multiclass|regression|ranking|contrastive
    loss: focal                       # Optional. bce|focal|ce|mse|mae|huber|quantile|listnet|infonce|auto
    loss_weight: 1.0                  # Optional. Static weight in multi-task loss
    label_col: clicked                # REQUIRED. Column name of the target label

    # Classification-specific
    num_classes: 5                    # Required for multiclass
    label_smoothing: 0.0             # Optional

    # Regression-specific
    normalize_target: false           # Optional. Z-score target during training
    huber_delta: 1.0                  # Optional. Delta for Huber loss

    # Focal loss parameters
    focal_alpha: 0.25                 # Optional
    focal_gamma: 2.0                  # Optional

    # Contrastive-specific
    temperature: 0.07                 # Optional. InfoNCE temperature

    # Tower architecture
    tower_dims: [128, 64]             # Optional. Hidden layer sizes
    tower_dropout: 0.2                # Optional
    tower_activation: silu            # Optional. relu|silu|gelu|tanh|leaky_relu

    # Metrics
    primary_metric: auc               # Optional. Metric for model selection
    secondary_metrics: [accuracy]     # Optional

# Data source
data:
  source: s3://bucket/data/train/     # REQUIRED. File path or S3 URI
  format: parquet                     # REQUIRED. parquet|csv
  train_split: 0.8                    # Optional. Default: 0.8
  val_split: 0.1                      # Optional. Default: 0.1

# Feature specification
features:
  numeric:                            # List of numeric column names
    - user_age
    - item_price
  categorical:                        # List of categorical column names
    - user_segment
    - item_category
  embedding_dim: 16                   # Optional. Embedding dim for categoricals
  transformers:                       # Optional. Transformer chain
    - name: standard_scaler
      cols: [user_age, item_price]
    - name: label_encoder
      cols: [user_segment, item_category]

# Model architecture
model:
  architecture: ple                   # "ple" or "lgbm"
  num_shared_experts: 2               # PLE-specific
  num_task_experts: 2                 # PLE-specific
  expert_hidden_dim: 256              # PLE-specific
  num_layers: 2                       # PLE-specific
  tower_dims: [128, 64]              # Task tower dimensions
  dropout: 0.1                        # Global dropout

# Training hyperparameters
training:
  batch_size: 2048
  epochs: 20
  learning_rate: 0.001
  early_stopping_patience: 5
  seed: 42

# AWS settings
aws:
  region: ap-northeast-2
  s3_bucket: my-bucket
  instance_type: ml.g4dn.xlarge
  use_spot: true
  max_run_seconds: 7200
  role_arn: arn:aws:iam::123456789:role/SageMakerRole
```

---

## recommendation.yaml

**File:** `configs/recommendation.yaml`
**Loaded by:** `core.recommendation.pipeline.RecommendationPipeline(config)`

Drives the full recommendation pipeline: scoring, filtering, top-K selection,
reason generation, and compliance checking.

### Pipeline orchestration

```yaml
pipeline:
  scorer_name: weighted_sum           # "weighted_sum" or "fd_tvs" or custom
  enable_reasons: true                # Generate recommendation reasons
  enable_self_check: true             # Verify reasons for compliance
  enable_reverse_mapping: true        # IG feature interpretation
```

### Scoring layer

```yaml
scorer:
  # Default: weighted sum
  weighted_sum:
    weights:                          # REQUIRED. Task name -> weight
      ctr: 0.25
      cvr: 0.35
      nba: 0.25
      ltv: 0.15
    min_score: 0.0                    # Floor
    max_score: 1.0                    # Ceiling

  # Advanced: 4-stage composite (FD-TVS)
  fd_tvs:
    task_weights:                     # REQUIRED. Base task weights (same as weighted_sum)
      ctr: 0.25
      cvr: 0.35
      nba: 0.25
      ltv: 0.15

    # Dynamic task-level weights (AWS design improvement over on-prem product-level weights)
    segment_task_weights:             # Per-segment multipliers applied on top of task_weights
      high_value:                     # Customer segment name
        nba: 1.5                      # Task name: multiplier (clipped to 1.0–1.5)
        ltv: 1.3
      transitory_income:
        ltv: 0.8                      # Downweight long-horizon tasks for volatile income
    dynamic_weight_rules:             # Behavior-based boosting rules (feature > threshold → task boost)
      - feature: mcc_diversity_trend  # Feature name (after normalization)
        threshold: 0.7                # Trigger threshold
        task: nba                     # Task to boost
        multiplier: 1.3               # Boost factor (clipped to 1.0–1.5)
      - feature: churn_signal
        threshold: 0.6
        task: ctr
        multiplier: 1.2

    # Stage 2: contextual modifier
    modifier_map:                     # context.modifier_segment -> multiplier
      high: 1.2
      medium: 1.0
      low: 0.8
    modifier_default: 1.0

    # Stage 3: behavioral velocity
    gamma_velocity: 0.15              # Velocity = 1 + gamma * flare

    # Stage 4: risk penalty
    risk_weight_limit_util: 0.3       # Weight for limit utilisation risk
    risk_weight_churn: 0.5            # Weight for churn risk
    risk_weight_message_freq: 0.2     # Weight for message frequency risk
    risk_threshold_limit_util: 0.8    # Trigger threshold
    risk_threshold_churn: 0.3
    risk_threshold_message_count: 3

    # Fatigue decay
    fatigue_base_decay: 0.85          # Exponential decay base
    fatigue_channel_multiplier:       # Per-channel multiplier
      app_push: 1.0
      email: 0.7
      sms: 0.9
      default: 1.0

    # Engagement boost
    engagement_boost_scale: 0.15      # 1 + scale * engagement_score
```

### Constraint filtering

```yaml
constraint_engine:
  fail_fast: true                     # Stop on first filter failure

filters:
  fatigue:
    enabled: true
    max_messages_7d: 5                # Hard cap on 7-day message count
    channel_decay:                    # Per-channel exponential decay
      app_push: 0.85
      email: 0.90
      sms: 0.80
      default: 0.85

  eligibility:
    enabled: true
    min_score: 0.05                   # Minimum composite score
    churn_threshold: 0.6              # Exclude if churn_prob > threshold

  owned_product:
    enabled: true                     # Exclude already-owned products
```

### Top-K selection

```yaml
selector:
  k: 5                               # Number of recommendations to return
  min_score: 0.0                      # Floor score for selection
  diversity_method: mmr               # "none" | "mmr" | "dpp"
  diversity_lambda: 0.5               # 1.0 = all relevance, 0.0 = all diversity
```

### Reason generation

```yaml
reason:
  template_engine:
    top_k_features: 3                 # Number of IG features for reasons

    feature_category_map:             # Feature prefix -> template category
      spend_: spending_pattern
      txn_count_: frequency_pattern
      amt_: spending_pattern
      merchant_: frequency_pattern
      life_stage_: life_stage
      benefit_: benefit_match
      tenure_: loyalty_pattern
      credit_: credit_pattern

    template_pool:                    # Category -> list of templates
      spending_pattern:
        - "{item_name} offers strong benefits aligned with your {category} spending."
        - "Your spending profile in {category} makes {item_name} a natural fit."
      frequency_pattern:
        - "{item_name} is designed for frequent {merchant_type} users like you."
      life_stage:
        - "{item_name} is optimised for customers at your current life stage."
      # ... (see recommendation.yaml for full list)

    task_frames:                      # Per-task narrative framing
      churn:
        frame: retention
        narrative: "We value your relationship..."
      ltv:
        frame: growth
        narrative: "Maximise long-term value..."
```

### Self-checker (compliance)

```yaml
  self_checker:
    enable_llm_check: false           # Set true if LLM provider available

    compliance_rules:                 # Regex patterns that must NOT appear
      guaranteed_return:
        pattern: "guaranteed\\s+return"
        severity: critical
        description: "Must not promise guaranteed returns."
      risk_free:
        pattern: "risk[- ]?free"
        severity: critical
      time_pressure:
        pattern: "limited\\s+time|act\\s+now|hurry"
        severity: major
      absolute_superlative:
        pattern: "\\bbest\\b|\\bbiggest\\b"
        severity: minor

    injection_patterns:               # Prompt injection detection
      - "ignore\\s+(previous|all)\\s+instructions?"
      - "<\\s*/?\\s*system\\s*>"
      - "you\\s+are\\s+now"

    ai_disclosure: >                  # Text appended to recommendations
      This recommendation was generated by an AI system.
```

### LLM provider

```yaml
llm_provider:
  backend: dummy                      # "bedrock" | "openai" | "dummy"
  bedrock:
    model_id: us.anthropic.claude-haiku-4-5-20251001-v1:0   # cross-region inference profile
    region: ap-northeast-2
    max_tokens: 512
    temperature: 0.0
  openai:
    model: gpt-4o-mini
    max_tokens: 512
    temperature: 0.0
```

---

## monitoring.yaml

**File:** `configs/monitoring.yaml`
**Loaded by:** Individual monitoring modules

### Fairness monitoring

```yaml
fairness:
  thresholds:
    di_lower: 0.8                     # Disparate Impact lower bound
    di_upper: 1.25                    # Disparate Impact upper bound
    spd_max: 0.1                      # Statistical Parity Difference max
    eod_max: 0.1                      # Equal Opportunity Difference max

  protected_attributes:               # Attributes to monitor
    - age_group
    - gender
    - region_type
    - income_tier
    - life_stage

  group_pairs:                        # Pairwise comparison groups
    age_group:
      - ["25-34", "65+"]              # [privileged, unprivileged]
    gender:
      - ["M", "F"]

  auto_incident: true                 # Create incident on violation
  schedule:
    frequency: daily
    time: "06:00"
    timezone: UTC

  # Parquet archive (CLAUDE.md §1.12, commit 51149f3 2026-04-21).
  # When set, `FairnessMonitor.archive_metrics()` appends every evaluation
  # to this path in addition to the in-memory history. The PromotionGate
  # MetadataAggregator reads this archive as the `fairness_risk` dimension
  # source, so leaving the path unset collapses the dimension to its
  # heuristic 0.5 fallback and the gate converges to a conservative
  # LIMITED verdict. Lambda `containers/lambda/fairness_evaluation.py`
  # calls `monitor.archive_metrics()` on every scheduled run so the
  # archive actually fills over time.
  archive_parquet_path: s3://aiops-ple-financial/compliance/fairness/metrics.parquet
```

### Drift detection

```yaml
drift:
  psi_threshold_warning: 0.1          # PSI warning threshold
  psi_threshold_critical: 0.25        # PSI critical threshold
  n_bins: 10                          # Histogram bins for PSI

  consecutive_tracker:
    monitoring_dir: monitoring/drift_results
    consecutive_threshold: 3           # Retrain after N consecutive critical days
    critical_feature_threshold: 5      # Min critical features for "critical day"

  baseline:
    source: s3://bucket/baselines/latest/
    format: parquet

  # Parquet archive (CLAUDE.md §1.12). When set, every `detect_drift()`
  # call optionally writes one row per feature to the archive, preserving
  # PSI scores, severity labels, and the thresholds active at evaluation
  # time. Consumed by the weekly governance markdown report.
  archive_parquet_path: s3://aiops-ple-financial/compliance/drift/metrics.parquet
```

### Incident management

```yaml
incidents:
  severity_criteria:
    critical:
      conditions:                     # List of trigger conditions
        - kill_switch_activation
        - di_below_0.6
        - security_breach
      response_time: "1 hour"
      escalate: true
    major:
      conditions:
        - di_below_0.8
        - herding_critical
        - consecutive_drift_retrain
      response_time: "4 hours"
      escalate: true
    minor:
      conditions:
        - drift_warning
        - quality_drop
      response_time: "24 hours"
      escalate: false

  sns_topic_arn: ""                   # AWS SNS topic for notifications
  region: ap-northeast-2
  s3_bucket: ""                       # Incident archive bucket
  s3_prefix: incidents
```

### Herding detection

```yaml
herding:
  hhi_warning: 0.15                   # HHI warning threshold
  hhi_critical: 0.25                  # HHI critical threshold
  top_k_items: 10                     # Track concentration in top-K
  top_k_concentration_warning: 0.5    # Warning if top-K > 50%
  top_k_concentration_critical: 0.7   # Critical if top-K > 70%
  window_hours: 24                    # Rolling evaluation window
```

### General settings

```yaml
general:
  log_level: INFO
  output_dir: monitoring/reports
  governance_report:
    enabled: true
    frequency: weekly
    include_fairness: true
    include_drift: true
    include_incidents: true
    include_herding: true
    output_format: json               # "json" or "html"
```

---

## Compliance Config

**File:** `configs/pipeline.yaml` — top-level `compliance:` block.
**Loaded by:** `core.compliance.*` modules (see CLAUDE.md §1.10–§1.16).

### `compliance.tracking`

Regulatory artifact tracking via SageMaker Experiments (replaces the
on-prem MLflow + DVC path — CLAUDE.md §1.14).

```yaml
compliance:
  tracking:
    # Shipped default (2026-04-21, commit 9426162). IAM reachability
    # verified against production account: list_experiments succeeds,
    # describe_experiment returns ResourceNotFound (not access-denied)
    # for unseeded names, confirming the training-job role carries
    # Experiments permissions. `in_memory` is retained for hermetic
    # unit tests and local development.
    backend: sagemaker                # "sagemaker" | "in_memory"

    experiment_name: aiops-ple-financial
    region: ap-northeast-2

    # Only 4 artifact types are recorded:
    # - fria_assessment          (Korean AI Basic Act §35 FRIA 7-dim)
    # - ai_risk_assessment       (EU AI Act Art. 9 FRIA 5-dim)
    # - compliance_registry_sweep (36-item ComplianceRegistry A+GAP)
    # - promotion_gate_verdict   (Champion-Challenger gate output)
    # Any other artifact is tagged `custom` via `log_custom_artifact`.
    # TrialComponent names are capped at 120 characters (SageMaker hard
    # limit) — format `<artifact_type>-<artifact_id>`[:120].
```

### `compliance.promotion_gate`

Per-dimension gate applied immediately after the ModelCompetition
verdict. Blocks promotion when regulatory dimensions (fairness,
privacy, lineage, risk) are not adequately substantiated, even if
metrics gate passes (CLAUDE.md §1.10, §1.11, §1.16).

```yaml
compliance:
  promotion_gate:
    # Production default (2026-04-21, PR #2/#3). Requires dimension
    # score providers to be wired — see CLAUDE.md §1.16. With providers
    # not wired, every dimension collapses to its heuristic 0.5 fallback
    # and the gate converges to a conservative LIMITED verdict
    # regardless of challenger quality. MetadataAggregator + 6 source
    # live-wiring (see below) closes this gap.
    enabled: true

    # Earliest-wins composition: manual overrides apply first, then
    # metadata aggregator, then per-rule heuristic fallbacks.
    providers:
      manual_overrides: {}            # Optional {model_version: {dimension: score}}
      aggregator:
        sources:
          # Six evidence sources aggregated into dimension scores and
          # embedded in `GateVerdict.details` as a per-dimension
          # derivation trail.
          fairness_archive_parquet_path: s3://aiops-ple-financial/compliance/fairness/metrics.parquet
          drift_archive_parquet_path:    s3://aiops-ple-financial/compliance/drift/metrics.parquet
          review_queue_table:            aiops-ple-review-queue
          model_registry_table:          aiops-ple-model-registry
          lineage_mapping_path:          configs/financial/feature_groups.yaml
          llm_config_path:               configs/pipeline.yaml

    # Heuristic transforms are audit-traceable: each rule declares
    # (metadata_path, transform, fallback). Supported transforms are
    # `identity`, `one_minus`, `log10_ratio` (requires
    # ratio_denominator). Values clipped to [0, 1]. Any new transform
    # must be registered in code, not YAML.
    heuristic_fallback: 0.5

    # Per-verdict action thresholds.
    thresholds:
      accept_min_score: 0.75
      limited_min_score: 0.50
      # Below limited_min_score → reject.
```

### `compliance.audit_sql` (S6)

Offline SQL over S3 Parquet archives via DuckDB httpfs — used by
regulators for batch queries (CLAUDE.md §1.15). Athena is deliberately
**not** the default; DuckDB httpfs extension reads s3:// URIs natively
and adds zero infrastructure cost.

```yaml
compliance:
  audit_sql:
    backend: duckdb                   # "duckdb" (default) | "athena" (future)
    paths:
      opt_out:   s3://aiops-ple-financial/compliance/opt_out/*.parquet
      events:    s3://aiops-ple-financial/compliance/events/*.parquet
      fairness:  s3://aiops-ple-financial/compliance/fairness/metrics.parquet
      drift:     s3://aiops-ple-financial/compliance/drift/metrics.parquet
    # View names restricted to [A-Za-z0-9_]+ (injection guard).
```

### `compliance.ai_security` (C4)

Prompt-injection and output-leak pattern catalog (CLAUDE.md §1.17).
Extend in YAML, not code, so security ops can respond to new attack
patterns without a release.

```yaml
compliance:
  ai_security:
    prompt_injection_patterns:        # Default: 14 patterns
      # - ignore previous instructions
      # - DAN mode
      # - reveal system prompt
      # - base64-encoded payloads
      # - ... (extend as needed)
    output_leak_patterns:             # Default: 8 patterns
      # - system-prompt echo
      # - meta-instruction leak
      # - Korean resident-registration number
      # - card number PII
      # - ... (extend as needed)
```

---

## Serving Config

**File:** Embedded in serving YAML or passed programmatically
**Class:** `core.serving.config.ServingConfig`

```yaml
serving:
  # Compute mode
  mode: auto                          # "auto" | "lambda" | "ecs"
  auto_threshold: 150000000           # Monthly requests threshold for auto mode

  # Feature store
  feature_store: auto                 # "auto" | "memory" | "dynamodb"
  auto_feature_threshold: 5000000     # User count threshold for auto mode
  feature_store_config:
    table_name: ple-feature-store
    region: ap-northeast-2

  # Lambda settings
  lambda:
    memory_mb: 1024
    timeout_seconds: 30
    reserved_concurrency: 100

  # ECS settings
  ecs:
    cpu: 1024                         # millicores
    memory: 2048                      # MB
    min_tasks: 2
    max_tasks: 10
    target_cpu_pct: 70

  # Model
  model:
    s3_uri: s3://bucket/models/lgbm/latest/
    tasks_meta:
      - name: ctr
        type: binary
      - name: cvr
        type: binary
      - name: ltv
        type: regression

  # A/B testing
  ab_test:
    enabled: false
    variants:
      - name: control
        model_path: s3://bucket/models/v1/
        weight: 0.8
      - name: challenger
        model_path: s3://bucket/models/v2/
        weight: 0.2

  # Offline Champion-Challenger competition (CLAUDE.md §1.10, §1.12).
  # The offline gate lives in `scripts/submit_pipeline.py::_decide_promotion`
  # and applies a fixed 4-step short-circuit ladder:
  #   (1) --force-promote  → promote unconditionally (operator override)
  #   (2) no champion      → bootstrap promote
  #   (3) fidelity failed  → reject (safety floor, Competition skipped)
  #   (4) else             → ModelCompetition.evaluate() verdict
  competition:
    # Production posture: FORCE auto_promote=false. Code-level
    # CompetitionConfig.auto_promote default is retained at True for
    # legacy test fixtures, but pipeline.yaml's False overrides at
    # runtime. Under this posture, a challenger that beats all metric
    # thresholds is still NOT promoted: the operator must re-run with
    # `--force-promote`. Satisfies EU AI Act Art. 14 (human oversight)
    # and SR 11-7 (model risk management).
    auto_promote: false

    primary_metric: avg_auc           # Default primary competition metric
    min_improvement: 0.005            # 0.5% absolute gain required
    max_degradation: 0.02             # 2% absolute regression tolerated on secondaries
    significance_level: 0.05          # Paired-bootstrap two-sided p threshold

  # Kill switch
  kill_switch:
    table_name: ple-kill-switch
    fallback_strategy: rule_based     # "rule_based" | "cached" | "default"

  # Recommendation pipeline
  pipeline:
    enabled: false
    scorer_name: weighted_sum
    enable_reasons: false
```

---

## Training Config

**Class:** `core.training.config.TrainingConfig`

```yaml
training:
  # Optimizer
  optimizer:
    name: adamw                       # Only adamw supported currently
    learning_rate: 0.001
    weight_decay: 0.01
    betas: [0.9, 0.999]
    eps: 1.0e-8
    expert_lr_overrides:              # Per-expert LR overrides
      causal:
        lr: 0.0005
        weight_decay: 0.001

  # Scheduler
  scheduler:
    name: cosine                      # "cosine" | "linear" | "none"
    warmup_epochs: 5
    warmup_start_factor: 0.1
    cosine_t0: 10
    cosine_t_mult: 2
    cosine_eta_min: 0.0
    phase2_warmup_epochs: 2
    phase2_cosine_t0: 6

  # Mixed precision
  amp:
    enabled: true
    dtype: float16                    # "float16" | "bfloat16"

  # Gradient handling
  gradient:
    clip_norm: 5.0

  # General
  batch_size: 2048
  epochs: 50
  seed: 42
  early_stopping_patience: 10
```

---

## PLE Model Config

**Class:** `core.model.ple.config.PLEConfig`

```yaml
model:
  # Global dimensions
  input_dim: 349                      # Set by FeatureGroupPipeline.total_dim (full feature tensor)
                                      # NOTE: With FeatureRouter active, each expert receives a
                                      # per-expert subset — NOT the full 349D. The global input_dim
                                      # represents the total feature tensor fed to FeatureRouter,
                                      # which then slices per-expert views via feature_group_ranges.
                                      # Per-expert dims (canonical v1): deepfm=168D, temporal_ensemble=139D,
                                      # hgcn=27D, perslay=32D, causal=161D, lightgcn=100D, ot=127D.
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
    use_batch_norm: false

  # Task experts
  num_task_experts_per_task: 1
  task_expert:
    hidden_dims: [128, 64]
    output_dim: 32

  # PLE stacking
  num_extraction_layers: 2

  # CGC (Customized Gate Control)
  # NOTE: With FeatureRouter active, CGC Layer 0 receives heterogeneous expert inputs
  # (each expert has a different input_dim per its target_experts routing). Layers 1-2
  # use homogeneous MLP experts for abstraction. dim_normalize=true is recommended
  # when per-expert input dims differ significantly (e.g., hgcn=34D vs deepfm=109D).
  cgc:
    enabled: true
    bias_high: 1.0
    bias_low: -1.0
    dim_normalize: false
    entropy_lambda: 0.01

  # Cluster sub-heads
  cluster:
    n_clusters: 0                     # 0 = disabled
    cluster_embed_dim: 32
    subhead_hidden_dim: 64
    subhead_output_dim: 32

  # adaTT (Adaptive Task Transfer, loss-level TAG+GradNorm hybrid).
  # STATUS: loss-level adaTT has a null effect at 13-task scale
  # (ΔAUC ≈ -0.001, within single-seed noise) after five implementation
  # bugs were corrected. Kept behind `enabled` for ablation comparison,
  # NOT enabled in production. Task groups below are the canonical four
  # Financial-DNA groups used for post-hoc interpretation and for the
  # rule-based fallback layer's template selection; they do NOT gate
  # the CGC router directly (routing is per-task via the CGC gate).
  adatt:
    enabled: false                    # Production default for 13-task benchmark
    transfer_lambda: 0.1
    temperature: 1.0
    warmup_epochs: 10
    freeze_epoch: null
    negative_transfer_threshold: -0.1
    ema_decay: 0.9
    prior_blend_start: 0.5
    prior_blend_end: 0.1
    inter_group_strength: 0.3
    grad_interval: 10
    max_transfer_ratio: 0.5
    task_groups:                       # Financial DNA 4-group split (post-hoc)
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

  # Logit transfer
  logit_transfers:
    - source: churn_signal
      target: will_acquire_deposits
      enabled: true
  logit_transfer_strength: 0.5

  # Task tower
  task_tower:
    hidden_dims: [64, 32]
    dropout: 0.2

  # Loss weighting
  loss_weighting:
    strategy: uncertainty             # "fixed"|"uncertainty"|"gradnorm"|"dwa"
    gradnorm_alpha: 1.5
    gradnorm_interval: 1
    dwa_temperature: 2.0
    dwa_window_size: 5

  # Dropout
  dropout: 0.1

  # Expert input routing (from FeatureGroupPipeline)
  expert_input_routing:
    - expert_name: shared_0
      input_groups: [base_profile, tda_topology]
    - expert_name: shared_1
      input_groups: []                # Empty = all features

  # Per-task overrides
  task_overrides:
    click:
      output_dim: 1
      activation: sigmoid
      task_type: binary
    ltv:
      output_dim: 1
      activation: null
      task_type: regression
```

---

## Agent Configuration (`configs/financial/agent.yaml`)

**File:** `configs/financial/agent.yaml`
**Purpose:** AIOps agent orchestration configuration. Composed of four sections: ops checkpoints, audit viewpoints, multi-agent consensus, and model assignments.

### ops

Declares the list of operational pipeline checkpoints (CPs) and their execution schedules.

```yaml
ops:
  checkpoints: [CP1, CP2, CP3, CP4, CP5, CP6, CP7]
  schedule:
    CP1: event       # Event-triggered (on data arrival)
    CP2: 5min        # Every 5 minutes
    CP3: 1h          # Every 1 hour
    CP4: 1h
    CP5: daily       # Once daily
    CP6: daily
    CP7: event
```

- `checkpoints`: List of CP IDs to execute. Registered in order; remove a CP from the list to deactivate it.
- `schedule`: CP ID to execution frequency. Allowed values: `event` | `5min` | `1h` | `daily`.

### audit

Declares the list of audit viewpoints (AVs) and their execution schedules.

```yaml
audit:
  viewpoints: [AV1, AV2, AV3, AV4, AV5]
  schedule:
    AV1: daily
    AV2: daily
    AV3: 1h
    AV4: daily
    AV5: daily
```

- `viewpoints`: List of active AV IDs (AV1-AV5).
- `schedule`: AV ID to execution frequency.

### consensus

Controls multi-agent consensus behavior.

```yaml
consensus:
  model: sonnet          # Base model used for consensus
  agents: 3              # Number of independent agents
  parallel: true         # Parallel execution (false = sequential)
  apply_to:              # List of CPs/AVs to apply consensus to
    - CP4
    - AV3
    - AV5
```

- `model`: Model identifier for consensus agents. `sonnet` = Claude Sonnet.
- `agents`: Number of independent agents that each render a judgment before majority/weighted consensus.
- `parallel`: `true` runs all agents concurrently (cost-efficient); `false` runs sequentially (easier to debug).
- `apply_to`: List of CP/AV IDs to apply consensus to. CPs/AVs not listed here run with a single agent.

### models

Declares model assignments per pipeline stage. Each key is a role; the value is a model identifier.

```yaml
models:
  reason_generation: us.anthropic.claude-sonnet-4-5-20251101-v1:0  # L2a reason rewrite + critique (Bedrock cross-region profile)
  reason_critique:   us.anthropic.claude-sonnet-4-5-20251101-v1:0  # Self-critique (generator <= critic model principle)
  factuality_check:  us.anthropic.claude-haiku-4-5-20251001-v1:0   # Factuality check (fast and low-cost)
  agent_dialog:      us.anthropic.claude-sonnet-4-5-20251101-v1:0  # Ops/Audit inter-agent dialog/negotiation
  agent_consensus:   us.anthropic.claude-sonnet-4-5-20251101-v1:0  # Multi-agent consensus (x agents count)
  deep_audit:        claude-opus    # Deep audit (quarterly, high-cost)
  embeddings:        titan-embed-v2 # AWS Titan Embeddings V2
```

| Role | Model | Notes |
|---|---|---|
| `reason_generation` | Claude Sonnet 4.6 | L2a reason rewrite — Bedrock cross-region profile, Korean-capable |
| `reason_critique` | Claude Sonnet 4.6 | Self-critique; same model as generator (generator ≤ critic principle) |
| `factuality_check` | Claude Haiku 4.5 | Low-latency/low-cost factuality check |
| `agent_dialog` | Claude Sonnet 4.6 | Ops/Audit inter-agent negotiation dialog |
| `agent_consensus` | Claude Sonnet 4.6 | Consensus voting (parallel x 3 by default) |
| `deep_audit` | Claude Opus | High-cost -- only runs when `apply_to` conditions are met |
| `embeddings` | Titan Embeddings V2 | AWS Bedrock native embeddings |

### On-Premises Model Configuration

On-premises deployments use a different models section in `agent.yaml`:

```yaml
models:
  reason_generation: exaone_7b    # Exaone 3.5 7.8B
  reason_critique: exaone_7b
  agent_consensus: qwen_14b_q4    # Qwen 2.5 14B Q4
  embeddings: minilm_v2           # sentence-transformers
model_paths:
  exaone_7b: "/models/exaone-3.5-7.8b-instruct"
  qwen_14b_q4: "/models/qwen2.5-14b-instruct-q4_k_m.gguf"
gpu:
  sequential_loading: true  # 12GB VRAM — cannot load simultaneously
```

---

## Checklist Configuration (`configs/financial/checklist.yaml`)

**File:** `configs/financial/checklist.yaml`
**Purpose:** Defines 48+ checklist items across 6 pipeline parts (P1-P6). Each item declares which tool to run, what threshold to apply, and which verdict logic to use.

### Part Structure (P1-P6)

| Part | Scope | Approx. Items |
|---|---|---|
| P1 | Data quality (input integrity, NaN/outliers, schema) | ~10 |
| P2 | Feature engineering (zero-variance, distributions, leakage) | ~8 |
| P3 | Label derivation (class balance, positive rate) | ~6 |
| P4 | Model training (loss convergence, gradients, AUC) | ~10 |
| P5 | Inference/recommendation (filters, score distributions, fatigue) | ~8 |
| P6 | Audit/compliance (fairness, drift, regulations) | ~8 |

### Item Structure

Each checklist item consists of three fields.

```yaml
checklist:
  P1:
    - id: P1-01
      description: "NaN ratio check"
      tool_name: nan_ratio_checker       # Tool (function/class) to execute
      threshold:                         # Verdict threshold
        max_nan_ratio: 0.05              # FAIL if exceeds 5%
      verdict_logic: threshold_le        # Verdict method (see below)

    - id: P1-02
      description: "Schema column count validation"
      tool_name: schema_column_checker
      threshold:
        expected_min_columns: 30
      verdict_logic: threshold_ge

  P4:
    - id: P4-03
      description: "AUC minimum threshold"
      tool_name: auc_evaluator
      threshold:
        min_auc: 0.70
      verdict_logic: threshold_ge

    - id: P4-07
      description: "NaN/Inf loss detection"
      tool_name: loss_nan_detector
      threshold: null                    # FAIL on any occurrence (binary verdict)
      verdict_logic: no_occurrence
```

### Allowed verdict_logic Values

| Value | Meaning |
|---|---|
| `threshold_le` | Measured value <= threshold -> PASS |
| `threshold_ge` | Measured value >= threshold -> PASS |
| `threshold_between` | lower <= measured value <= upper -> PASS |
| `no_occurrence` | No event occurred -> PASS |
| `all_pass` | All sub-items PASS -> PASS |
| `custom` | Uses the `verdict` field returned by `tool_name` directly |

### Total Item Count and Extension

48 items are provided by default. To add new items, simply add entries to `checklist.yaml`. Tools registered under `tool_name` are auto-dispatched without Python code changes. Tool registration is done via the `@register_tool` decorator in `core/audit/tool_registry.py`.

## Fact Extraction (`configs/financial/fact_extraction.yaml`)

Customer-level descriptive fact extraction rules. Loaded by `FactExtractor` and applied at Phase 0 batch time.

### Structure

```yaml
version: "1.0"
rules:
  - name: "Deposit-focused portfolio"     # Korean-language fact injected into reason prompts
    condition: "deposit_balance_ratio > 0.6"  # Python expression
    required_features: ["deposit_balance_ratio"]  # pre-check
```

### Condition Expressions

- Evaluated as Python expressions; only safe builtins are allowed (`abs`, `min`, `max`, `len`, `any`, `all`, `round`, `int`, `float`)
- Runs in a `__builtins__`-blocked sandbox preventing arbitrary code execution
- Basic operators: `>`, `<`, `>=`, `<=`, `==`, `!=`, `and`, `or`, `not`
- Compound conditions: `"x > 0.3 and x < 0.7"`

### Categories (15 default rules)

| Category | Examples |
|---|---|
| Portfolio composition | Deposit-focused, fund-focused, loan-burdened |
| Recent interests | Increased fund inquiries, active card usage, rising investment inquiries |
| Risk appetite | Risk-averse, aggressive investor |
| Life cycle | New customer, long-term customer |
| Engagement | Digital channel preference, high-frequency transactions |
| Spending patterns | Stable transactions, increasing spending |
| Churn risk | Churn warning signals |

### Fact Quality Validation

Use `FactExtractor.validate_rules(sample_features)` for pre-flight validation:

```python
from core.recommendation.reason.fact_extractor import FactExtractor

extractor = FactExtractor("configs/financial/fact_extraction.yaml")
report = extractor.validate_rules(sample_features=my_sample)
# {"total_rules": 15, "valid": 15, "invalid": 0, "skipped_missing_features": 0}
```
