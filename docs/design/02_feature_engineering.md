# 02. Feature Engineering — 5-Axis 피처 매핑, Generator, 정규화, LabelDeriver

## 10+ Stage 파이프라인에서의 위치

Feature Engineering은 Stage 4 ~ Stage 6을 담당한다:

```
Stage 4:   FeatureGroupPipeline + Normalization (per axis generators + PowerLawAwareScaler)
Stage 5:   LabelDeriver (13 tasks, config-driven derivation)
Stage 5.5: LeakageValidator (sequence/correlation/product/temporal)
Stage 6:   SequenceBuilder (time-based + sliding window → sequences.npy, seq_lengths.npy)
```

---

## FeatureGroupPipeline — 피처 엔지니어링 오케스트레이터

`core/pipeline/features.py`의 `FeatureGroupPipeline`이 Stage 4 피처 엔지니어링의 주 진입점이다.

### Expert Routing — Feature Group 단위 원칙 (2026-04-11)

**expert_routing은 개별 컬럼 이름이 아닌 feature group 이름을 기준으로 해야 한다.** Phase 0의 3-stage 정규화는 power-law 컬럼마다 `_log` 접미사 복사본을 추가하므로, 컬럼 이름 기반 routing은 이 재배열 이후 잘못된 슬라이스를 Expert에 전달한다.

`feature_group_ranges`는 연속된 컬럼 블록으로 저장해야 한다 (contiguous blocks). min~max index 방식은 정규화 후 삽입된 `_log` 컬럼이 블록 경계를 넘어갈 때 인접 group의 컬럼을 포함하는 오류를 일으킨다. 비연속 매칭 발생 시 가장 긴 연속 블록을 사용하거나 컬럼 이름 리스트로 range를 저장한다.

### FeatureGroupPipeline vs Adapter 역할 분리

| 역할 | DataAdapter (Stage 1) | FeatureGroupPipeline (Stage 4) |
|------|----------------------|-------------------------------|
| DuckDB 집계 | O (24M → 2,000 user) | X |
| RFM 기본 피처 | O (base aggregation) | X |
| TDA (global/local) | X | O — Generator 호출 |
| HMM 상태 추정 | X | O — Generator 호출 |
| Mamba SSM | X | O — Generator 호출 |
| Graph 임베딩 | X | O — Generator 호출 |
| GMM 클러스터링 | X | O — Generator 호출 |
| Normalization | X | O — PowerLawAwareScaler |

### TDA Global/Local 구분

TDA 피처는 두 개의 별도 Generator 호출로 분리된다:

- **TDA Global** (Snapshot 축, 16D): 12개월 장기 윈도우 Persistence Diagram (h0+h1, 8 stats each)
- **TDA Local** (Timeseries 축, 16D): 90일 단기 윈도우 Persistence Diagram (h0+h1, 8 stats each)

---

## 5-Axis Feature Group 매핑 (Santander 데이터)

### 전체 구조

```
5-Axis Classification (Santander, 17 groups, 1211D)
    ↓
    ├── State (정적)
    │   ├── demographics      (11D) — 나이/소득/재직기간/상품수/활성여부 + 6 categorical codes
    │   ├── product_holdings  (24D) — 24개 prod_* 이진 보유 플래그
    │   ├── txn_behavior      (14D) — synth_* RFM + 시간대/안정성/사기비율
    │   └── derived_temporal  ( 4D) — 총 취득/해지/관측월/상품다양성
    │
    ├── Snapshot (장기)
    │   ├── tda_global        (16D) — 12개월 Persistence Diagram h0+h1 (8 stats each)
    │   ├── hmm_states        (25D) — HMM Triple-Mode (journey/lifecycle/behavior)
    │   ├── gmm_clustering    (22D) — K=20 soft probs + entropy + dominant  (v14: K=14 → 16D)
    │   └── model_derived     (27D) — KMeans(5) + Bandit(4) + LNN(18)
    │
    ├── Timeseries (단기 시계열)
    │   ├── tda_local         (16D) — 90일 Persistence Diagram h0+h1
    │   ├── mamba_temporal    (50D) — Mamba SSM (cached_embedding_uri 로 GPU 사전계산)
    │   ├── txn_lag_tensor   (800D) — K=200 lag 평탄화 (amount/mcc/day/hour × 200)
    │   └── txn_rolling_stats (20D) — 4 윈도우(7/30/90/180d) × 5 메트릭
    │
    ├── Hierarchy (구조)
    │   ├── product_hierarchy (34D) — 24상품 Poincaré 임베딩 32D + 2 통계
    │   └── merchant_hierarchy(27D) — MCC 계층 Poincaré 임베딩 (HGCN 라우팅)
    │
    └── Item (관계)
        ├── graph_collaborative(66D) — LightGCN 협업 필터링 64D + 2 통계
        ├── nba_label_multihot (24D) — 24상품 고정어휘 NBA multi-hot
        └── mcc_top30_multihot (31D) — Top-30 MCC multi-hot + others 1열
```

### 차원 요약

| Axis | Feature Groups | Total Dim | Target Experts |
|------|---------------|-----------|----------------|
| **State** | 4 groups | 53D | DeepFM, MLP, Causal, OT |
| **Snapshot** | 4 groups | 90D | PersLay, Temporal, DeepFM, Causal, OT |
| **Timeseries** | 4 groups | 882D | Temporal Ensemble, DeepFM, Causal, OT |
| **Hierarchy** | 2 groups | 61D | LightGCN, Causal, HGCN |
| **Item** | 3 groups | 121D | LightGCN, DeepFM, HGCN |
| **합계** | **17 groups** | **1211D** | 7종 Shared Expert Pool (+ MLP per-task tower) |

---

## Axis별 Feature Generator 상세

### State 축 — 정적 속성

```yaml
- name: demographics
  axis: state
  group_type: transform
  transformers: [quantile_transformer]
  output_dim: 11    # 5 numeric + 6 categorical codes (dense-rank, not one-hot)
  target_experts: [deepfm, mlp, causal, optimal_transport]

- name: product_holdings
  axis: state
  group_type: transform
  transformers: []   # binary, no scaling
  output_dim: 24    # 24 prod_* binary flags
  target_experts: [deepfm, mlp, causal, optimal_transport]

- name: txn_behavior
  axis: state
  group_type: transform
  transformers: [standard_scaler]
  output_dim: 14    # synth_* RFM + time-of-day + stability + fraud_ratio
  target_experts: [deepfm, temporal_ensemble, causal, optimal_transport]

- name: derived_temporal
  axis: state
  group_type: transform
  transformers: [standard_scaler]
  output_dim: 4     # total_acquisitions, total_churns, months_observed, product_diversity
  target_experts: [deepfm, causal, optimal_transport]
```

### Snapshot 축 — 장기 패턴

```yaml
- name: tda_global
  axis: snapshot
  generator: tda_global
  generator_params:
    input_filter:
      dtype: continuous
      exclude_binary: true
      min_nunique: 20
    max_homology_dim: 1    # h0 + h1 only (h2 excluded)
  output_dim: 16           # 2 dims × 8 stats each
  target_experts: [perslay]

- name: hmm_states
  axis: snapshot
  generator: hmm
  generator_params:
    input_filter:
      include_prefix: ["synth_"]
    modes: [journey, lifecycle, behavior]   # Triple-Mode
    # v14 Phase 0 (2026-05-04): mode_observation_cols per-mode 입력 분리
    mode_observation_cols:
      journey:   [synth_unique_mcc, synth_unique_merchants, synth_recency_days,
                  synth_frequency, synth_unique_mcc_l1_idx]
      lifecycle: [synth_monthly_txns, synth_avg_amount, synth_monthly_spend,
                  synth_monetary, synth_stability]
      behavior:  [synth_morning_ratio, synth_afternoon_ratio, synth_evening_ratio,
                  synth_night_ratio, synth_fraud_ratio]
  output_dim: 25           # 3 modes × ~8 stats + 3 state-id cols
  target_experts: [temporal_ensemble]
  # HMM Triple-Mode → task group routing:
  #   journey  → value, consumption groups
  #   lifecycle → lifecycle group
  #   behavior  → engagement group

- name: gmm_clustering
  axis: snapshot
  generator: gmm
  generator_params:
    input_filter:
      dtype: continuous
      exclude_binary: true
      min_nunique: 10
      exclude_columns: [customer_id]
    n_clusters: 14         # v14 Phase 0 (2026-05-04): K=20 → K=14 (BIC sweep, dead clusters 제거)
    covariance_type: full
    max_iter: 200
  output_dim: 16           # K soft probs(14) + entropy(1) + dominant(1)
  target_experts: [deepfm, mlp, causal, optimal_transport]
  # NOTE: GMM soft labels — posterior probabilities + Shannon entropy

- name: model_derived
  axis: snapshot
  generator: model_features
  generator_params:
    input_filter:
      dtype: continuous
      exclude_binary: true
      exclude_columns: [customer_id]
    kmeans_dim: 5
    bandit_dim: 4
    lnn_dim: 18
  output_dim: 27           # KMeans(5) + Bandit/MAB(4) + LNN(18)
  target_experts: [temporal_ensemble, deepfm]
```

### Timeseries 축 — 단기 시퀀스

```yaml
- name: tda_local
  axis: timeseries
  generator: tda_local
  generator_params:
    input_filter:
      dtype: continuous
      exclude_binary: true
      include_prefix: ["synth_"]
    max_homology_dim: 1    # h0 + h1 only
  output_dim: 16           # 2 dims × 8 stats each
  target_experts: [perslay]

- name: mamba_temporal
  axis: timeseries
  generator: mamba
  generator_params:
    input_filter:
      include_prefix: ["synth_"]
    d_model: 128
    output_dim: 50
    prefer_gpu: true
    entity_column: customer_id
    base_batch_size: 256
    cached_embedding_uri: s3://aiops-ple-financial/santander_ple/mamba/embedding.parquet
    # cached_embedding_uri: GPU 사전계산 Mamba 임베딩 parquet.
    # Phase 0 CPU job이 DuckDB JOIN으로 가져옴 (60분 SSM fallback 불필요).
    # 입력 피처 변경 또는 모델 가중치 drift 시 GPU job 재실행 필요.
  target_experts: [temporal_ensemble]

- name: txn_lag_tensor
  axis: timeseries
  generator: lag_extractor
  generator_params:
    sequence_columns:
      amount: txn_amount_seq
      mcc: txn_mcc_seq
      day: txn_day_offset_seq
      hour: txn_hour_seq
    k: 200
    truncate_seq_last: 1   # last MCC == next_mcc label
    pad_value: 0.0
  output_dim: 800          # K=200 × 4 features (amount/mcc/day/hour)
  target_experts: [deepfm, lightgcn]
  # 800D is intentionally high; distill=false to keep student load manageable

- name: txn_rolling_stats
  axis: timeseries
  generator: rolling_stats_extractor
  generator_params:
    amount_column: txn_amount_seq
    day_offset_column: txn_day_offset_seq
    windows_days: [7, 30, 90, 180]
    metrics: [sum, mean, std, count, days_active]
    truncate_seq_last: 1
  output_dim: 20           # 4 windows × 5 metrics
  target_experts: [deepfm, causal, optimal_transport]
```

### Hierarchy 축 — 구조적 계층

```yaml
- name: product_hierarchy
  axis: hierarchy
  generator: graph
  generator_params:
    input_filter:
      include_prefix: ["prod_"]
    use_poincare: true
    curvature: 1.0
    embedding_dim: 32
    entity_column: customer_id
  output_dim: 34              # 32 Poincaré dims + 2 geometric stats
  target_experts: [lightgcn, causal]

- name: merchant_hierarchy
  axis: hierarchy
  generator: merchant_hierarchy
  generator_params:
    mcc_hierarchy_path: configs/mcc_hierarchy.yaml   # ISO 18245, L1/L2/L3
    truncate_seq_last: 1      # drop last MCC = next_mcc label, prevents leakage
    l1_radius: 0.8
    l2_radius: 0.5
    time_decay_lambda: 0.01
  output_dim: 27              # Poincaré 27D embedding (쌍곡 공간, MCC 계층 구조 인코딩)
  target_experts: [hgcn]
  # MCC Hierarchy: 10 L1 groups, ~30 L2 subcategories, 109 L3 codes in dataset
  # NOTE: txn_mcc_seq 마지막 1원소 제거 → next_mcc 누수 방지
```

### Item 축 — 관계적 상호작용

```yaml
- name: graph_collaborative
  axis: item
  generator: graph
  generator_params:
    input_filter:
      dtype: all_numeric
      exclude_columns: [customer_id, has_nba, churn_signal, product_stability]
      exclude_prefix: ["label_"]
    embedding_dim: 64
    use_poincare: false
    entity_column: customer_id
  output_dim: 66              # 64 LightGCN dims + 2 graph stats
  target_experts: [lightgcn]

- name: nba_label_multihot
  axis: item
  generator: topn_multihot_extractor
  generator_params:
    source_column: nba_label
    mode: fixed_vocab
    vocab: [0..23]            # 24 Santander product slots
    binary: true
  output_dim: 24
  target_experts: [lightgcn, deepfm]

- name: mcc_top30_multihot
  axis: item
  generator: topn_multihot_extractor
  generator_params:
    source_column: txn_mcc_seq
    mode: top_n
    top_n: 30
    include_others: true
    truncate_seq_last: 1      # last MCC == next_mcc label, drop to avoid leak
    binary: true
  output_dim: 31              # 30 top MCC bins + 1 others col
  target_experts: [lightgcn, deepfm, hgcn]
```

---

## Feature Generator Registry

### 11 구현 완료 Generator

모든 Generator는 `core/feature/generators/` 디렉토리에 구현되어 있으며, `FeatureGeneratorRegistry`에 등록된다. cuDF primary / pandas fallback 패턴을 따른다. Generator output은 cuDF DataFrame 또는 pandas DataFrame이다.

| # | 등록 이름 | 파일 | Axis | Santander Output | GPU 가속 | 설명 |
|---|-----------|------|------|-----------------|---------|------|
| 1 | `tda_global` | `core/feature/generators/tda.py` | Snapshot | 16D | cuPY + ripser | Persistence Diagram h0+h1 (장기 윈도우) |
| 2 | `tda_local` | `core/feature/generators/tda.py` | Timeseries | 16D | cuPY + ripser | Persistence Diagram h0+h1 (단기 90일) |
| 3 | `hmm` | `core/feature/generators/hmm.py` | Snapshot | 25D | hmmlearn | HMM Triple-Mode 상태 (journey/lifecycle/behavior) |
| 4 | `mamba` | `core/feature/generators/mamba.py` | Timeseries | 50D | GPU (mamba-ssm) | Mamba SSM — cached_embedding_uri로 Phase 0 CPU join |
| 5 | `graph` | `core/feature/generators/graph.py` | Hierarchy/Item | 34D / 66D | - | Poincaré 임베딩 (product_hierarchy) / LightGCN (graph_collaborative) |
| 6 | `gmm` | `core/feature/generators/gmm.py` | Snapshot | 22D / 16D (v14) | cuML (optional) | GMM soft labels (K=20 → K=14 in v14) + entropy + dominant |
| 7 | `model_features` | `core/feature/generators/model_features.py` | Snapshot | 27D | - | KMeans(5D) + Bandit(4D) + LNN(18D) |
| 8 | `merchant_hierarchy` | `core/feature/generators/merchant_hierarchy.py` | Hierarchy | 27D | - | MCC 계층 Poincaré 쌍곡 임베딩; truncate_seq_last=1 (누수 방지) |
| 9 | `lag_extractor` | `core/feature/generators/lag_extractor.py` | Timeseries | 800D | - | K=200 lag 평탄화 (amount/mcc/day/hour × 200) |
| 10 | `rolling_stats_extractor` | `core/feature/generators/rolling_stats_extractor.py` | Timeseries | 20D | - | 4 윈도우(7/30/90/180d) × 5 메트릭 (sum/mean/std/count/days_active) |
| 11 | `topn_multihot_extractor` | `core/feature/generators/topn_multihot_extractor.py` | Item | 24D / 31D | - | fixed_vocab(NBA 24상품) / top_n(MCC top-30+others) 모드 |

### GPU Utility Layer

`core/feature/generators/gpu_utils.py`가 모든 Generator에 공통 GPU 유틸리티를 제공:
- Device detection (`get_device(prefer_gpu=True)`)
- Adaptive batch sizing
- OOM-retry decorator (GPU 메모리 부족 시 자동 CPU fallback)
- `has_cudf()`, `has_cuml()`, `has_cupy()` lazy import 체크

---

## Stage 4: Feature Integration + Normalization

### Canonical 3-Stage Normalization (CLAUDE.md §2.1, §1.9)

```
Numeric Features (all axes merged)
    ↓
┌─ Stage 1 ─── 멱법칙 자동 감지 (skew + kurt 스크린 → log-log R²) ──▶ log1p(x) 복사본 생성
│                                                                     (`{col}_log` 컬럼으로 피처 행렬 말미에 append)
│
├─ Stage 2 ─── TRAIN split only fit StandardScaler (z-score: mean=0, std=1)
│   · `exclude_from_scaler: [categorical_id, probability]` 선언 컬럼 제외
│     (ID 버킷 정수, [0,1] 확률 컬럼은 scaling 시 분포 왜곡되므로 skip — CLAUDE.md §1.9)
│   · 연속형 (non-binary, non-categorical-ID, non-probability) 컬럼에만 적용
│   · v14 Phase 0 (2026-05-04): `_probability_prefixes` default 10 패턴 + `_prob_`
│     infix 자동 감지 — GMM/HMM probability 컬럼이 z-score 로 ±1.7 까지 왜곡되던
│     버그 차단 (`core/pipeline/normalizer.py`)
│
└─ Stage 3 ─── Stage 1 에서 생성된 `_log` 컬럼은 **raw magnitude 보존**
                (scaler 재적용 금지 — heavy-tail 신호 유지)
    ↓
    scaler_params.json 저장 (mean, std, log1p_cols, exclude_cols)
    ↓
[Stage 6 완료 후] PipelineRunner._rebuild_group_ranges_post_normalization()
    · Stage 1 이 `_log` 컬럼을 말미에 append 하고 컬럼 순서를 재배열하므로,
      pre-normalization 의 `feature_group_ranges` 는 stale 상태가 된다
    · 각 feature group 의 original + `_log` offspring 을 post-normalization
      컬럼 순서에서 재위치 → longest contiguous block 을 새 range 로 emit
    · 비연속 매칭 발생 시 WARNING, 14 regression test 로 잠금
    · 상세: CLAUDE.md §1.7, commit `ec8587b` (2026-04-21), 테스트
      `tests/test_group_ranges_rebuild.py`
```

### 구현

```python
# core/feature/normalizer.py
class PowerLawAwareScaler:
    """
    3-Stage 정규화의 Stage 1+2+3 을 래핑한 클래스.

    Stage 1: fit 시 각 컬럼의 skew+kurt → log-log R² 검정으로 멱법칙 감지.
             감지된 컬럼에 log1p 변환본(`{col}_log`)을 피처 행렬 말미에 append.
    Stage 2: TRAIN split 에서만 fit. `exclude_from_scaler` 규칙에 따라
             categorical_id / probability sub-kind 를 skip. 연속형에만 z-score.
    Stage 3: Stage 1 에서 생성된 `_log` 컬럼은 raw magnitude 를 보존 — 재스케일 금지.
    get_params() → scaler_params.json 저장 (mean, std, log1p_cols, exclude_cols).
    cuPY 가속: cupy 설치 시 skew/mean/std 계산을 GPU 에서 수행.
    """
```

> **왜 Stage 3 에서 `_log` 를 보존하는가** — FeatureRouter 가 expert 마다 두 뷰 (centered 원본 + tail-preserving log) 를 모두 받을 수 있도록 설계된 구조이기 때문이다. `_log` 를 Stage 2 scaler 에 포함시키면 heavy-tail 신호가 평탄화되어 DeepFM / OT 같은 분포 의존 expert 의 성능이 저하된다.

---

## Schema-Invariant Audit (Phase 0 v3, 2026-04-28)

> 관련 commit: `2c93b1f`, `4b1a40c`, `dfd116c`, `ecfdeb9`, `c6dbe3e`, `9308916`, `88f7a7b` (모두 `main`, 미push). 검증 Job: `santander-ple-phase0-0428-1628`.

### Schema Invariant (스키마 불변식)

Phase 0 산출물의 모든 다운스트림 컴포넌트 (`FeatureRouter`, expert dispatch, group-level routing, IG attribution) 가 의존하는 단일 invariant 는 다음과 같다:

```
feature_groups.yaml registry order
  == Stage 3 (FeatureGroupPipeline) concat order
  == features.parquet column order
  == sequential feature_group_ranges (no gaps, no overlaps)
```

이 등식이 성립하면 `PipelineRunner._rebuild_group_ranges_post_normalization` 은 **no-op 으로 수렴**한다 (입력 range == 출력 range). 위반 시 FeatureRouter 가 out-of-bounds index 로 `features` tensor 를 슬라이스하고, GPU 경로에서 `device-side assert triggered` 만 떠서 `CUDA_LAUNCH_BLOCKING=1` 없이는 디버깅이 불가능하다.

### 발견된 3 종 위반 경로 + 픽스

#### 1) Placeholder `output_columns` 가 실제 generator output 을 가린 경우 (commit `2c93b1f`)

`FeatureGroupConfig.__post_init__` 는 dataclass 생성 시 `output_columns` 가 비어 있으면 placeholder 를 자동으로 채운다:

```python
# core/feature/group_pipeline.py (수정 전)
self.output_columns = [f"{name}_{i}" for i in range(output_dim)]
# → ["mamba_temporal_0", "mamba_temporal_1", ..., "mamba_temporal_49"]
```

`FeatureGroupPipeline.fit()` 안의 3 군데 post-fit assignment 가 `if not group.output_columns:` guard 로 보호되고 있었는데, placeholder 가 이미 채워져 있으니 guard 가 **항상 False** 였다. 그래서 generator 의 실제 output 컬럼명 (`mamba_d0..mamba_d49`, `tda_global_h0_*` 등) 은 끝까지 group 에 등록되지 않았다.

Stage 6 의 `_rebuild_group_ranges_post_normalization` 은 이 placeholder 들을 post-normalization `feature_cols` 에서 lookup 하는데 — 당연히 매치되지 않으니 — **모든 group 을 silently 드롭**하고 stale Stage 3 range 를 그대로 유지했다. 픽스는 guard 를 모두 제거하고 항상 `gen.output_columns` 로 덮어쓰는 것이다.

#### 2) Stage 6 normalizer 의 컬럼 reorder (commit `c6dbe3e`)

`PowerLawAwareScaler.transform_sql` / `transform` 은 projection 을 **고정 bucket 순서**로 emit 했다:

```
[continuous block] → [binary block] → [categorical_int block]
                  → [probability block] → [_log copies]
```

`demographics` 처럼 group 안에 dtype 이 섞여 있으면 (continuous 9 + binary 2 — `gender`, `is_active`) bucket 순서로 재배열되어 binary 컬럼이 원래 위치에서 **약 500 자리 떨어진 곳**으로 흩어진다. Stage 3 는 그룹별 contiguous 였는데 Stage 6 가 깨버린 것.

픽스는 `feature_cols` 를 입력 순서로 순회하면서 각 컬럼의 projection 을 in-place 로 emit 하도록 변경. `_log` 복사본만 진짜 추가분이므로 tail 에 append 하는 동작은 유지.

#### 3) Stage 6 입력 `feature_cols` 가 adapter dictate 순서를 상속 (commit `88f7a7b`)

Normalizer 가 order-preserving 으로 고쳐져도, **입력으로 받는 `feature_cols`** 자체가 DuckDB DESCRIBE 순서 (= adapter 가 컬럼을 만든 순서) 였다. Adapter 는 group registry 와 무관하게 dtype/생성 시점 기준으로 컬럼을 emit 하므로 demographics 의 5 declared cols 가 흩어졌다.

`PipelineRunner._reorder_feature_cols_by_group` helper 를 추가하여, 각 enabled group 의 `output_columns` 를 registry 순서로 먼저 배치하고, 어떤 group 에도 routed 되지 않은 passthrough 컬럼은 tail 에 모은다. 이 시점에 invariant 가 처음으로 성립한다.

### 보조 픽스

- **`_compute_dim_ranges` 는 YAML 의 `output_dim` 대신 `len(group.output_columns)` 사용** (commit `4b1a40c`). YAML 값과 generator 실제 출력이 어긋날 때 range 가 허위 길이로 계산되는 것을 방지.
- **`feature_groups.yaml::output_dim` 을 generator 실측에 맞춰 재정렬** (commit `dfd116c`):
  - demographics 38 → 11
  - tda_global 36 → 16
  - tda_local 24 → 16
  - hmm_states 48 → 25
  - product_hierarchy 32 → 34
  - graph_collaborative 64 → 66
- **`FeatureSpec.meta_cols` / `TaskSpec.derive` 를 dataclass 에 추가** (commit `ecfdeb9`). 둘 다 YAML 에는 선언되어 있었지만 dataclass 가 받지 못해 load 시 silent drop. 결과적으로 `snapshot_date` 와 `has_nba` (= `nba_primary.derive.filter_col`) 가 features.parquet 에 새어들어가고 있었다.

### 방어 가드 (Defensive Guards)

침묵 실패를 fail-fast 로 전환하기 위해 두 군데에 가드를 추가:

```python
# core/model/config_builder.py
# end > n_feat_cols 인 group_range 는 build 시 drop + WARN
ranges = [(g, s, e) for (g, s, e) in ranges if e <= n_feat_cols]

# core/model/feature_router.py (FeatureRouter.route)
# slice index 가 features.shape[-1] 를 넘으면 raise (CUDA assert 대신)
assert idx.max() < features.shape[-1], "router index OOB ..."
```

### 검증 결과 (Phase 0 v3, Job `santander-ple-phase0-0428-1628`)

- `feature_columns` 총 **1211 개**.
- 17 개 group 모두 **연속 (sequential), 갭 없음**:
  ```
  demographics             [   0,    5)
  product_holdings         [   5,   29)
  ...
  mcc_top30_multihot       [1174, 1205)
  ```
- 6 개 trailing passthrough cols: `gender`, `segment`, `country`, `channel`, `age_group`, `income_group` (Stage 2 가 auto-encode 한 categorical 들로, 어떤 group 의 `columns:` 에도 declared 되지 않음).
- `snapshot_date` / `has_nba` 는 **features.parquet 에서 제외**.
- `demographics` 의 declared 5 cols 가 모두 contiguous block 에 포함.

이 시점부터 `_rebuild_group_ranges_post_normalization` 은 input range 와 동일한 output 을 emit 하며 (no-op), expert routing / IG attribution / group-level FRIA 분석 이 모두 동일한 인덱스 공간을 공유한다.

---

## v14 Phase 0 Fixes (2026-05-04)

> 관련 파일: `core/feature/generators/hmm.py`, `core/feature/generators/gmm.py`,
> `configs/santander/feature_groups.yaml`, `core/pipeline/normalizer.py`.

### 1) HMM mode-specific observation columns (`core/feature/generators/hmm.py`)

**Previous**: 3 mode (journey / lifecycle / behavior) 가 모두 동일 input X 를 받아 학습되었다.
세 mode 모두 `n_states=5`, `random_state=42` 로 고정되어 있어 journey 와 lifecycle 이
**결정론적으로 같은 모델로 수렴** — 두 state assignment 간 Cramer V = 1.0 (완전 동일).

**Current**: `mode_observation_cols` 로 mode 별 입력 컬럼을 분리:

| Mode | 의미 | Observation columns |
|------|------|---------------------|
| `journey` | engagement (다양성/활성도) | `synth_unique_mcc`, `synth_unique_merchants`, `synth_recency_days`, `synth_frequency`, `synth_unique_mcc_l1_idx` |
| `lifecycle` | volume / maturity | `synth_monthly_txns`, `synth_avg_amount`, `synth_monthly_spend`, `synth_monetary`, `synth_stability` |
| `behavior` | temporal pattern | `synth_morning_ratio`, `synth_afternoon_ratio`, `synth_evening_ratio`, `synth_night_ratio`, `synth_fraud_ratio` |

추가로 mode 별 `random_state` 를 mode 이름의 hash 로 자동 생성하여, 동일 입력 분포에서도
서로 다른 local optima 로 수렴하도록 보장. 검증 결과 journey vs lifecycle Cramer V 가
**1.0 → 0.5281** 로 떨어져 mode 분리가 성립한다.

### 2) GMM K=14 + flat-kwarg parsing fix (`core/feature/generators/gmm.py`, `feature_groups.yaml`)

**Previous**: `K=20`, 1M-row BIC sweep 에서 7 개 cluster 가 dead (각 1 row 만 점유),
output_dim=22 (= 20 soft probs + entropy + dominant). 추가로 `GMMClusteringGenerator.__init__`
의 flat-kwarg parsing 버그로 yaml 의 `n_clusters: 14` 가 `GMMConfig` 까지 전파되지
않아 K override 자체가 무시되고 있었다.

**Current**: `K=14`, output_dim=16. 1M rows BIC sweep 결과:

| K | BIC | Dead clusters |
|---|-----|---------------|
| 14 | -58.8M | 0 (모든 cluster ≥ 0.24% 점유) |
| 20 | -63.1M | 6 |

Flat-kwarg parsing 도 수정하여 `feature_groups.yaml::generator_params.n_clusters` 가
실제 GMM 학습에 반영되도록 함.

### 3) Normalizer probability exclusion (`core/pipeline/normalizer.py`)

**Previous**: GMM soft probs / HMM posterior probs 등 이미 [0, 1] 범위인 컬럼이
`StandardScaler` 에 포함되어 z-score 변환 후 약 ±1.7 범위로 분포가 왜곡되었다.

**Current**: `_probability_prefixes` 의 default 를 10 개 패턴으로 확장하고, 컬럼 이름
중간에 `_prob_` infix 가 있으면 자동 감지하도록 normalizer 보강. CLAUDE.md §1.9 의
`exclude_from_scaler: [categorical_id, probability]` 정책을 코드 레벨에서 구현. 검증 결과
probability 컬럼이 정확히 `[0.0, 1.0]` 범위를 유지한다.

이 3 가지 픽스는 §1.7 (group-level routing) / §1.9 (probability exclusion) 의 정책
조항이 코드와 어긋나 있던 부분을 동기화한 것이며, Phase 0 v3 schema-invariant audit
이후 발견된 잔존 누수 (mode redundancy, dead cluster, prob distortion) 를 차단한다.

---

## Santander 데이터 피처 상세 (configs/santander/)

### Categorical Embeddings

| 컬럼 | Cardinality | Embedding Dim | 비고 |
|------|-------------|---------------|------|
| `gender` | 2 | 4 | F, M |
| `segment` | 4 | 4 | 01-TOP, 02-PARTICULARES, 03-UNIVERSITARIO, UNKNOWN |
| `country` | 118 | 16 | 118개국 (ES 96%) |
| `channel` | 163 | 16 | 163개 채널 코드 |
| `age_group` | 5 | 4 | young/adult/middle/senior/elderly |
| `income_group` | 5 | 4 | low/mid/high/very_high/unknown |

### Numeric Features

| 컬럼 | 범위 | 특이사항 |
|------|------|---------|
| `age` | 18-100 | 정상 범위 |
| `income` | 0-28.8M EUR | 25.4% missing (0=결측) |
| `tenure_months` | -999999~256 | sentinel 처리 필요 |
| `is_active` | 0/1 | 42% active |
| `num_products` | 0-15 | avg 1.3 |

---

## 13-Task Label Architecture

### 4-Tier 태스크 구조

| Tier | 설명 | 태스크 수 | 예시 |
|------|------|----------|------|
| **Tier 1** | Core targets (직접 레이블) | 3 | churn_signal, product_stability, nba_primary |
| **Tier 2** | Derived targets (규칙 유도) | 1 | cross_sell_count |
| **Tier 3** | Product group + segmentation | 6 | will_acquire_{deposits,investments,accounts,lending,payments}, segment_prediction |
| **Tier 5** | Transaction-based NBA | 3 | next_mcc, mcc_diversity_trend, top_mcc_shift |

> **참고**: has_nba (binary)는 2026-04-12에 nba_primary (multiclass)로 통합됨 — nba_primary의 class 0이 "NBA 없음"을 나타내므로 has_nba는 중복 태스크로 제거됨. 18→14→13 태스크로 축소.

### Per-task Focal Alpha Calibration

Binary 태스크의 `focal_alpha`는 positive rate에 따라 calibrated:

| 태스크 | Positive Rate | focal_alpha | 근거 |
|--------|-------------|-------------|------|
| `churn_signal` | 5.1% | 0.85 | 매우 불균형 |
| `will_acquire_deposits` | ~1% | 0.95 | 극단적 불균형 |
| `will_acquire_investments` | ~2% | 0.90 | 매우 불균형 |
| `will_acquire_accounts` | ~5% | 0.85 | 불균형 |
| `will_acquire_lending` | ~1% | 0.90 | 극단적 |
| `will_acquire_payments` | ~3% | 0.85 | 매우 불균형 |
| `top_mcc_shift` | ~10% | 0.70 | 경도 불균형 |

### 4 Semantic Groups (adaTT Transfer Units)

```yaml
task_groups:
  engagement:   [next_mcc, top_mcc_shift]
  lifecycle:    [churn_signal, product_stability, segment_prediction]
  value:        [mcc_diversity_trend, cross_sell_count]
  consumption:  [nba_primary, will_acquire_deposits,
                 will_acquire_investments, will_acquire_accounts,
                 will_acquire_lending, will_acquire_payments]
```

---

## Item Universe + Product Hierarchy

### Item Universe (24 금융 상품)

```
┌──────────────────────────────────────────────────┐
│ 예금: saving, guarantee, checking, derivados     │
│ 계좌: payroll_acct, junior_acct, particular_acct │
│       particular_plus, e_account, home_acct      │
│ 예탁: short_deposit, medium_deposit, long_deposit│
│ 투자: funds, securities, pension_plan,           │
│       pension_deposit                             │
│ 대출: mortgage, loans                            │
│ 결제: credit_card, direct_debit, auto_debit      │
│ 세금: taxes                                      │
│ 보수: payroll                                    │
└──────────────────────────────────────────────────┘
```

### Product Hierarchy Config

```json
{
  "hierarchy": {
    "level_1": {
      "deposits": ["saving", "short_deposit", "medium_deposit", "long_deposit"],
      "accounts": ["checking", "payroll_acct", "junior_acct", "particular_acct", "..."],
      "investments": ["funds", "securities", "pension_plan", "pension_deposit"],
      "lending": ["mortgage", "loans"],
      "payments": ["credit_card", "direct_debit", "auto_debit"]
    }
  }
}
```

---

## 피처 차원 관리 — 동적 계산

```python
# core/feature/schema.py
class FeatureSchema:
    @property
    def input_dim(self) -> int:
        return sum(g.output_dim for g in self.groups if g.enabled)

    @property
    def axis_dims(self) -> dict[str, int]:
        dims = defaultdict(int)
        for g in self.groups:
            if g.enabled:
                dims[g.axis] += g.output_dim
        return dict(dims)
        # → {"state": 314, "snapshot": 139, "timeseries": 214, "hierarchy": 41, "item": 64}
```

---

## SageMaker Processing에서 실행

```
Step Functions
    ↓
SageMaker Processing Job
    ├── 입력: s3://bucket/data/encrypted/
    ├── 코드: core/feature/pipeline_builder.py
    ├── config: configs/santander/feature_groups.yaml
    ├── 엔진: DuckDB (인프로세스)
    ├── GPU: cuPY/cuDF optional (ml.g4dn.xlarge)
    └── 출력: s3://bucket/features/v{version}/
         ├── features.parquet
         ├── labels.parquet
         ├── sequences.npy
         ├── seq_lengths.npy
         ├── feature_schema.json
         ├── label_schema.json
         ├── split_indices.json
         ├── scaler_params.json
         ├── label_transforms.json
         ├── item_universe/
         │   ├── product_hierarchy.json
         │   └── customer_product_graph.parquet
         └── transformers/
             └── transformer_00_PowerLawAwareScaler.pkl
```

---

## 현재 vs AWS — 피처 엔지니어링 비교

| 항목 | 현재 (On-Prem) | AWS (설계) | 변경 이유 |
|------|---------------|-----------|----------|
| 피처 분류 | 없음 (flat 목록) | **5-Axis** (State/Snapshot/Timeseries/Hierarchy/Item) | Expert 라우팅 명시적 기반 |
| 피처 정의 | 코드 내 하드코딩 | YAML 선언형 (feature_groups.yaml) | 코드 변경 없이 피처 추가/삭제 |
| 차원 관리 | 644D, 734D 하드코딩 | `FeatureSchema.input_dim` 동적 계산 + axis_dims | 피처 변경 시 자동 반영 |
| 정규화 | QuantileTransform + Raw Power-Law | **StandardScaler + power-law 자동 감지** (skew>2.0 → log1p+raw) | 단일 경로, 분포 보존 |
| TDA | 단일 윈도우 | **Global(365일)/Local(90일) 분리** → Snapshot/Timeseries 축 | 장기/단기 위상 분리 |
| HMM | 코드에 직접 구현 | Generator Registry (hmm_triple_mode) | 선택적 활성화 |
| Graph | HGCN만 | **Poincare + LightGCN** (Hierarchy + Item 축) | 계층+협업 분리 |
| Item Universe | 없음 | **고객x상품 bipartite graph** (24 금융 상품) | 상품 추천 핵심 |
| 레이블 생성 | 코드 하드코딩 | **LabelDeriver** (config-driven 13 tasks) | 선언적, 재현 가능 |
| 누수 방지 | 없음 | **LeakageValidator** (4-check) + temporal split | 자동 누수 감지 |
| 피처 버전 | 없음 | features/v{version}/ | 재현성, 롤백 |
| Cold Start | 없음 | **is_cold_start flag + sequence-derived feature zeroing** | cold start 고객 대응 |
| GPU 가속 | 없음 | cuDF primary (generators), cuPY (TDA, scaler), cuML (GMM) | 5-10x 가속 |
