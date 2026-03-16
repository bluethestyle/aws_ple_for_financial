# Configuration Reference

Complete reference for all YAML configuration files in the platform.

---

## Table of Contents

1. [feature_groups.yaml](#feature_groupsyaml)
2. [Pipeline Config (examples/*.yaml)](#pipeline-config)
3. [recommendation.yaml](#recommendationyaml)
4. [monitoring.yaml](#monitoringyaml)
5. [Serving Config](#serving-config)
6. [Training Config](#training-config)
7. [PLE Model Config](#ple-model-config)

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

---

## Pipeline Config

**Files:** `configs/examples/*.yaml`
**Loaded by:** `core.pipeline.config.load_config()`

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
    task_weights:                     # REQUIRED. Same as weighted_sum
      ctr: 0.25
      cvr: 0.35
      nba: 0.25
      ltv: 0.15

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
    model_id: anthropic.claude-3-haiku-20240307-v1:0
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
  input_dim: 196                      # Set by FeatureGroupPipeline.total_dim
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

  # adaTT (Adaptive Task Transfer)
  adatt:
    enabled: true
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
    task_groups:
      engagement:
        members: [click, convert]
        intra_strength: 0.7
      value:
        members: [ltv, nba]
        intra_strength: 0.7

  # Logit transfer
  logit_transfers:
    - source: click
      target: convert
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
