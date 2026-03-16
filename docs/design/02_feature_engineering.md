# 02. Feature Engineering — 피처 파이프라인, 트랜스포머, 정규화

## 현재 (On-Prem) 분석

### 구조
- **총 피처 차원**: 734D main + 74D separate + 3600D sequence (flat) = 4,408D
- **카테고리**: Base(238D), Multi-Source(91D), Extended(84D), Domain(159D), Model-Derived(27D), Multidisciplinary(24D), Merchant(21D), Power-Law(90D)
- **Separate**: Hyperbolic(20D), HMM Triple-Mode(48D), Coldstart(6D)

### 문제점
1. **피처 추출기가 도메인에 종속**: `rfm_extractor.py`, `card_ingestion.py` 등 금융 전용
2. **차원이 코드에 하드코딩**: 644D, 734D, 74D 등이 여러 파일에 흩어져 있음
3. **정규화 전략 복잡**: Quantile Transform + Raw Power-Law 이중 파이프라인
4. **피처 통합이 모놀리식**: feature_integrator.py 하나가 600M × 700+ 컬럼 처리

### 유지할 패턴
- **DuckDB 기반 피처 처리**: 메모리 효율적 대규모 처리
- **Per-feature scaler pickle**: 서빙 시 동일 변환 적용 가능
- **Power-law feature 보존**: Quantile Transform의 절대값 정보 손실 보상

---

## AWS 설계

### 피처 파이프라인 구조

```
Schema (YAML)
    ↓
    ├── Numeric Features ──▶ [StandardScaler / QuantileTransformer / LogTransformer]
    ├── Categorical Features ──▶ [LabelEncoder / HashEncoder / EmbeddingLookup]
    ├── Sequence Features ──▶ [SequencePadder / WindowAggregator]
    └── Custom Features ──▶ [Plugin Registry에서 동적 로드]
    ↓
FeaturePipeline (체이닝)
    ↓
S3 features/v{version}/
```

### YAML 기반 피처 정의

```yaml
# configs/features.yaml
features:
  # ── 기본 피처 (스키마에서 직접 추출) ──
  numeric:
    - name: total_spend
      source: transactions
      agg: "SUM(amount)"
      transformer: quantile    # 정규화 방법
    - name: tx_count
      source: transactions
      agg: "COUNT(*)"
      transformer: log1p       # 로그 변환 (power-law 분포)
    - name: avg_amount
      source: transactions
      agg: "AVG(amount)"
      transformer: standard    # z-score

  categorical:
    - name: top_category
      source: transactions
      agg: "MODE(category)"
      encoder: label           # label encoding
      embedding_dim: 16        # 임베딩 시 차원
    - name: user_segment
      source: user_profiles
      encoder: onehot

  # ── 시퀀스 피처 ──
  sequences:
    - name: tx_sequence
      source: transactions
      columns: [amount, category_id, hour, dow]
      max_length: 180
      sort_by: timestamp

  # ── 도메인 전용 피처 (플러그인) ──
  plugins:
    - name: tda_features
      type: tda_extractor       # @FeatureRegistry.register("tda_extractor")
      enabled: true
      params:
        short_window_days: 90
        long_window_days: 365
        output_dim: 70
    - name: graph_embeddings
      type: graph_embedding     # @FeatureRegistry.register("graph_embedding")
      enabled: false            # 그래프 데이터 없으면 비활성화
      params:
        embedding_dim: 64

  # ── Power-law 보존 설정 ──
  power_law:
    enabled: true
    detection: auto             # 자동 감지 (skewness > 2.0)
    transform: log1p            # 원본에 log1p 적용한 복사본 생성
```

### 피처 트랜스포머 플러그인

```python
# core/feature/transformers/standard.py
@FeatureRegistry.register("standard")
class StandardScaler(AbstractFeatureTransformer):
    def fit(self, df):
        self.mean = df[self.cols].mean()
        self.std = df[self.cols].std()
        return self

    def transform(self, df):
        df = df.copy()
        df[self.cols] = (df[self.cols] - self.mean) / (self.std + 1e-8)
        return df


# core/feature/transformers/quantile.py
@FeatureRegistry.register("quantile")
class QuantileTransformer(AbstractFeatureTransformer):
    """On-Prem의 normalizer.py 로직을 재현. 이상치에 강건."""
    ...


# plugins/features/tda_extractor.py  (도메인 전용 플러그인)
@FeatureRegistry.register("tda_extractor")
class TDAFeatureExtractor(AbstractFeatureTransformer):
    """위상 데이터 분석 피처. 선택적 활성화."""
    ...
```

### 피처 파이프라인 자동 구성

```python
# core/feature/pipeline_builder.py
class FeaturePipelineBuilder:
    """
    YAML config를 읽어 FeaturePipeline을 자동 구성합니다.

    config의 features: 블록 → 적절한 Transformer 체인 생성
    config의 plugins: 블록 → FeatureRegistry에서 동적 로드
    """

    def build(self, config: FeatureSpec) -> FeaturePipeline:
        transformers = []

        # 1. Numeric transformers
        for group in self._group_by_transformer(config.numeric):
            transformer = FeatureRegistry.build(
                group.transformer_name, cols=group.col_names
            )
            transformers.append(transformer)

        # 2. Categorical encoders
        for cat in config.categorical:
            encoder = FeatureRegistry.build(cat.encoder, cols=[cat.name])
            transformers.append(encoder)

        # 3. Plugin features (선택적)
        for plugin in config.plugins:
            if plugin.enabled:
                transformer = FeatureRegistry.build(plugin.type, **plugin.params)
                transformers.append(transformer)

        # 4. Power-law preservation
        if config.power_law.enabled:
            power_law_cols = self._detect_power_law(config)
            transformers.append(
                FeatureRegistry.build("log1p", cols=power_law_cols)
            )

        schema = self._build_schema(config)
        return FeaturePipeline(schema, transformers)
```

### SageMaker Processing에서 실행

```
Step Functions
    ↓
SageMaker Processing Job
    ├── 입력: s3://bucket/data/processed/
    ├── 코드: core/feature/pipeline_builder.py
    ├── config: configs/features.yaml
    ├── 엔진: DuckDB (인프로세스, 메모리 효율적)
    └── 출력: s3://bucket/features/v{version}/
         ├── train.parquet
         ├── val.parquet
         ├── test.parquet
         ├── schema.json          ← 실제 생성된 피처 메타데이터
         └── transformers/        ← fit된 scaler/encoder pickle
             ├── transformer_00_StandardScaler.pkl
             └── transformer_01_LabelEncoder.pkl
```

### 피처 차원 관리 — 동적 계산

```python
# core/feature/schema.py (확장)
class FeatureSchema:
    """
    피처 차원을 하드코딩하지 않고 config에서 동적으로 계산합니다.

    On-Prem에서 644D, 734D 등이 코드 곳곳에 하드코딩된 문제를 해결합니다.
    """

    @property
    def input_dim(self) -> int:
        """모델 입력 차원을 config 기반으로 계산."""
        dim = 0
        dim += len(self.numeric)                                    # 숫자형 피처 수
        dim += sum(e.embedding_dim for e in self.categorical)       # 카테고리 임베딩 합
        dim += sum(p.output_dim for p in self.plugins if p.enabled) # 플러그인 출력 합
        if self.power_law.enabled:
            dim += len(self._detect_power_law_cols())               # power-law 복사본
        return dim

    @property
    def sequence_dim(self) -> int:
        """시퀀스 피처 총 차원."""
        return sum(len(s.columns) * s.max_length for s in self.sequences)
```

---

## 현재 vs AWS — 피처 엔지니어링 비교

| 항목 | 현재 (On-Prem) | AWS (설계) | 변경 이유 |
|------|---------------|-----------|----------|
| 피처 정의 | 코드 내 하드코딩 | YAML 선언형 | 코드 변경 없이 피처 추가/삭제 |
| 차원 관리 | 644D, 734D 하드코딩 | input_dim 동적 계산 | 피처 변경 시 자동 반영 |
| 정규화 | normalizer.py 모놀리식 | Transformer 플러그인 체이닝 | 재사용, 테스트 용이 |
| 도메인 피처 | 코드에 직접 구현 | Plugin Registry (선택적) | 도메인 무관하게 on/off |
| 피처 버전 | 없음 | features/v{version}/ | 재현성, 롤백 |
| 서빙 일관성 | scaler pickle 수동 관리 | transformers/ 자동 저장 | 학습-추론 불일치 방지 |
| 실행 환경 | 로컬 Python | SageMaker Processing | 확장성, 비용 최적화 |
