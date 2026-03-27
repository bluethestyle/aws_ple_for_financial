# Claude Code Agent Guidelines — AIOps PLE Platform

## 1. 절대 준수사항 (MUST)

### 1.1 Config-Driven 원칙
- **모든 파라미터는 YAML config에서 읽는다.** Python 코드에 컬럼명, 경계값, 시나리오 목록, AWS 상수를 하드코딩하지 않는다.
- 새 데이터셋 지원 시 YAML config 파일(pipeline.yaml, feature_groups.yaml, item_universe.yaml)만 추가한다. train.py, adapter, ablation script에 dataset-specific 코드를 넣지 않는다.
- `feature_groups.yaml`의 `input_filter`로 generator 입력을 선언한다. adapter에서 `product_cols`, `synth_cols` 같은 하드코딩 라우팅을 하지 않는다.

### 1.2 관심사 분리 (Separation of Concerns)
- **Adapter**: raw data → standardized DataFrame. 전처리/피처생성/레이블파생을 하지 않는다.
- **PipelineRunner**: 전처리 → 피처생성 → 레이블파생 → 정규화 → 텐서 저장 (Phase 0).
- **train.py**: training-ready 데이터 로드 → 모델 빌드 → 학습. 전처리 코드를 넣지 않는다.
- 코드가 500줄을 넘으면 관심사 분리가 되지 않은 것이다.

### 1.3 데이터 리키지 방지
- **Scaler는 TRAIN split에서만 fit**한다. val/test는 train에서 fit된 scaler로 transform만 한다.
- **Temporal split 시 gap_days를 반드시 설정**한다 (최소 7일).
- **시퀀스 데이터의 마지막 timestep이 레이블과 겹치지 않는지 반드시 검증**한다.
- **LeakageValidator를 학습 전에 반드시 호출**한다.

### 1.4 실험 전 검증 (Pre-flight Check)
- SageMaker Job 제출 전에 반드시 다음을 확인한다:
  1. **Phase 0 출력 검증**: feature_stats.json에서 zero-variance 컬럼, NaN 비율, 생성된 피처 컬럼 수 확인
  2. **Generator 입력 검증**: 각 generator가 받는 입력 컬럼의 dtype과 분포가 generator 설계 의도와 일치하는지 확인 (예: GMM은 continuous만, TDA는 시계열 또는 고차원 점구름)
  3. **레이블 분포 검증**: label_stats.json에서 class balance, positive rate 확인
  4. **Dry run**: `--dry-run` 플래그로 Job 구성을 먼저 확인한 후 제출
  5. **소규모 테스트**: 50K subsample로 end-to-end 성공을 확인한 후 전체 데이터 실행

### 1.5 비용 관리
- **ProfilerReport는 반드시 비활성화**한다 (`disable_profiler=True`).
- **max_run은 넉넉하게 설정**하되, 비용은 실제 사용 시간만 청구되므로 문제없다.
- **batch_size는 VRAM과 데이터 크기에 맞게 최적화**한다 (941K 데이터 → 4096 권장).
- **spot 인스턴스 8대 이상 동시 사용 시 중단 빈도가 높아진다** — checkpoint를 반드시 활성화한다.
- 실험 전 `aws ce get-cost-and-usage`로 현재 비용을 확인한다.

## 2. 아키텍처 규칙

### 2.1 정규화 파이프라인 (3-stage)
```
Stage 1: 멱법칙 감지 (skew+kurt → log-log R²) + log1p 복사본 생성
Stage 2: StandardScaler (continuous 컬럼만, binary 제외, TRAIN fit only)
Stage 3: 멱법칙 _log 복사본은 스케일링하지 않음 (raw magnitude 보존)
```

### 2.2 모델 설정
- `task_loss_weights`: pipeline.yaml에서 읽어 PLEConfig에 전달
- `adaTT task_groups`: pipeline.yaml에서 읽어 `AdaTTConfig.from_pipeline_groups()` 호출
- `task_group_map`: adaTT groups에서 자동 빌드 (GroupEncoder + HMM routing 활성화)
- `logit_transfers`: pipeline.yaml task_relationships에서 읽어 전달
- `FocalLoss`: pre-activation logits를 전달 (double-sigmoid 방지)

### 2.3 Generator 입력 라우팅
```yaml
# feature_groups.yaml에서 선언 (하드코딩 금지)
generator_params:
  input_filter:
    dtype: continuous       # continuous | all_numeric
    exclude_binary: true    # GMM, TDA 등
    min_nunique: 20         # 이산 변수 제외
    include_prefix: [synth_] # 특정 컬럼 그룹만
```

## 3. 온프렘 프로젝트와의 일관성

### 3.1 반드시 유지해야 하는 온프렘 설계 요소
- 5축 피처 분류 (상태/스냅샷/시계열/계층/아이템)
- 피처 그룹별 전문가 라우팅 (target_experts)
- 태스크 그룹 기반 adaTT (intra/inter strength)
- 3단계 정규화 (log transform → scaler → raw copy)
- 로짓 전이 3가지 방식 (output_concat / hidden_concat / residual)
- Uncertainty weighting (Kendall et al.)
- 증류 시 IG 기반 피처 선택

### 3.2 AWS에서 달라도 되는 요소
- Airflow DAG → SageMaker 순차 실행 (오케스트레이션 방식)
- DuckDB 파일 기반 → S3 Parquet (저장소)
- Docker GPU 컨테이너 → SageMaker Training Job (실행 환경)

## 4. 코드 검수 기준 (커밋 전 필수)

모든 코드 작업 후, 커밋 전에 반드시 아래 **4단계 검수**를 통과해야 한다. 하나라도 빠지면 "완료"로 보고하지 않는다.

### 4.1 컴파일 검증
- 수정된 모든 `.py` 파일에 `py_compile.compile(f, doraise=True)` 실행
- 수정된 모든 `.yaml` 파일에 `yaml.safe_load()` 실행

### 4.2 인터페이스 계약 검증
- **파일 A가 저장하는 키 이름**과 **파일 B가 읽는 키 이름**이 일치하는지 확인
- 예: runner.py가 `feature_schema["group_ranges"]`로 저장하면, train.py도 `schema["group_ranges"]`로 읽어야 한다
- 병렬 에이전트 작업 후에는 반드시 **cross-file 키/필드 매핑 테이블**을 만들어 검증

### 4.3 하드코딩 스캔
- 실행 경로의 모든 파일에서 다음을 grep 검색:
  - **컬럼명**: `"customer_id"`, `"snapshot_date"`, `"income"`, `"tenure"`, `"prod_"`, `"synth_"`, `"seq_"`, `"nba_"` 등 dataset-specific 문자열
  - **AWS 상수**: `"ap-northeast"`, `"aiops-ple"`, `"ml.g4dn"`, `"795833"` 등
  - **매직넘버**: config에서 읽어야 하는 임계값, 경계값, 서브샘플 크기
- config fallback (`config.get("key", "하드코딩값")`)도 dataset-specific이면 안 된다
- `id_cols`, `date_cols`, `label_cols`는 반드시 config에서 읽는다

### 4.4 관심사 분리 검증
- train.py에 전처리 코드(fillna, LabelEncoder, scaler, label derivation)가 없는지 확인
- adapter에 generator 호출이 없는지 확인
- ablation script에 시나리오/전문가/태스크 목록이 하드코딩되지 않았는지 확인

### 4.5 "완료" 보고 기준
- 위 4단계를 **모두 통과한 후에만** "0건 잔여" 또는 "완료"로 보고한다
- 일부만 검증한 경우 "컴파일 OK, 인터페이스 미검증, 하드코딩 미스캔"으로 명시한다

## 5. 서브에이전트 사용 규칙
- 서브에이전트는 **반드시 opus 모델**만 사용한다.
- 병렬 서브에이전트 실행 시 각각의 역할을 명확히 구분한다.
- 서브에이전트 결과를 반드시 검수한 후 커밋한다.
- **병렬 에이전트 작업 후에는 반드시 인터페이스 계약 검증 에이전트를 추가로 실행한다.**

## 6. 금지사항
- 로컬에서 pip install이나 패키지 실행 금지 (SageMaker에서만 테스트)
- train.py에 전처리 코드를 넣지 않는다
- adapter에 generator 호출을 하드코딩하지 않는다
- FEATURE_GROUP_COLUMN_PREFIXES 같은 하드코딩 매핑을 만들지 않는다
- `_derive_santander_labels()` 같은 dataset-specific 함수를 train.py에 넣지 않는다
- 실험 결과 검증 없이 다음 Phase로 넘어가지 않는다
