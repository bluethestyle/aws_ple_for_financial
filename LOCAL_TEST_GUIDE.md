# 로컬 GPU PC 테스트 가이드

이 문서는 Claude Code에게 전달할 프롬프트입니다.
64GB RAM + RTX 4070 PC에서 실행합니다.

---

## 프롬프트 (아래를 복사해서 Claude Code에 붙여넣기)

```
이 프로젝트는 AWS SageMaker 기반 AIOps PLE Platform의 ablation 테스트 파이프라인이야.
지금까지 SageMaker에서 코드 디버깅하다가 비용만 $74 날렸어.
이제 로컬 PC(64GB RAM + RTX 4070)에서 코드를 완전히 검증한 후에 SageMaker에는 한 번만 제출할 거야.

## 환경 설정

1. 레포 클론 및 의존성 설치:
```bash
git clone https://github.com/bluethestyle/aws_ple_for_financial.git
cd aws_ple_for_financial
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install duckdb pyarrow pyyaml omegaconf scikit-learn hmmlearn scipy lightgbm
```

2. 데이터 다운로드 (S3 → 로컬):
```bash
aws s3 cp s3://aiops-ple-financial/santander-ablation/20260327-150559/phase0/data/santander_final.parquet data/synthetic/santander_final.parquet
```

## 테스트 순서

### Step 1: Phase 0 (전처리 + 피처 생성)

SageMaker 환경 시뮬레이션 위해 메모리 제한:
```bash
python adapters/santander_adapter.py \
  --input-dir data/synthetic/ \
  --output-dir outputs/phase0/ \
  --pipeline configs/santander/pipeline.yaml \
  --feature-groups-config configs/santander/feature_groups.yaml
```

확인할 것:
- outputs/phase0/santander_final.parquet 생성됐는지
- feature_stats.json에서 zero-variance 컬럼 수
- 각 generator가 정상 실행됐는지 (tda, graph, hmm, mamba, gmm, model_derived)
- list 컬럼(31개)이 parquet에 보존됐는지
- 소요 시간 (SageMaker ml.m5.2xlarge 기준 ~2시간, 로컬은 더 빠를 수 있음)

### Step 2: train.py 단일 시나리오 테스트

Phase 0 출력으로 학습 테스트:
```bash
python containers/training/train.py \
  --config configs/santander/pipeline.yaml \
  --epochs 3 \
  --batch_size 4096 \
  --learning_rate 0.008 \
  --seed 42
```

환경변수로 데이터 경로 지정:
```bash
export SM_CHANNEL_TRAIN=outputs/phase0/
export SM_OUTPUT_DATA_DIR=outputs/training/
export SM_MODEL_DIR=outputs/model/
```

확인할 것:
- DuckDB로 parquet 로드 성공하는지 (pd.read_parquet가 아닌 duckdb.execute)
- list 컬럼이 자동 스킵되는지
- LabelDeriver가 18개 레이블 전부 파생하는지
- 3-stage 정규화 (power-law 감지 → StandardScaler → raw log copy)
- PLEModel 빌드 성공 (18 tasks, 7 experts, adaTT, logit transfer)
- Forward pass + loss 계산 성공 (NaN 없이)
- 3 epoch 학습 완료
- eval_metrics.json 생성
- CUDA GPU 사용 확인 (torch.cuda.is_available() == True)

### Step 3: 피처 ablation 시나리오 테스트

removed_feature_groups 적용 테스트:
```bash
python containers/training/train.py \
  --config configs/santander/pipeline.yaml \
  --epochs 1 \
  --batch_size 4096 \
  --removed_feature_groups '["tda_global", "tda_local"]' \
  --ablation_scenario no_tda
```

확인할 것:
- schema["group_ranges"]에서 tda 컬럼이 정확히 제거되는지
- input_dim이 동적으로 줄어드는지
- 모델이 줄어든 input_dim으로 정상 빌드되는지

### Step 4: Task x Structure ablation 테스트

active_tasks + use_ple/use_adatt:
```bash
# 4 tasks + PLE only
python containers/training/train.py \
  --config configs/santander/pipeline.yaml \
  --epochs 1 \
  --batch_size 4096 \
  --active_tasks '["has_nba", "churn_signal", "product_stability", "nba_primary"]' \
  --use_ple true \
  --use_adatt false

# 4 tasks + shared bottom (no PLE, no adaTT)
python containers/training/train.py \
  --config configs/santander/pipeline.yaml \
  --epochs 1 \
  --batch_size 4096 \
  --active_tasks '["has_nba", "churn_signal", "product_stability", "nba_primary"]' \
  --use_ple false \
  --use_adatt false
```

확인할 것:
- active_tasks 필터링이 정확히 적용되는지 (4/18 태스크만)
- task_groups가 active tasks에 맞게 필터링되는지
- use_ple=false일 때 단일 레이어로 축소되는지
- use_adatt=false일 때 adaTT가 비활성화되는지

### Step 5: 메모리 제한 테스트 (SageMaker 시뮬레이션)

g4dn.xlarge의 16GB RAM 제한을 시뮬레이션:
```python
# test_memory_limit.py
import duckdb
con = duckdb.connect()
con.execute("SET memory_limit='12GB'")  # 16GB 중 4GB는 OS/PyTorch용

# Phase 0 parquet 로드
df = con.execute("SELECT * FROM 'outputs/phase0/santander_final.parquet'").df()
print(f"Shape: {df.shape}, Memory: {df.memory_usage(deep=True).sum()/1e6:.0f} MB")
con.close()
```

12GB 제한에서도 OOM 없이 로드되면 SageMaker에서도 안전합니다.

## 성공 기준

모든 Step이 에러 없이 완료되면:
1. eval_metrics.json에 18개 태스크 메트릭이 전부 있어야 함
2. binary 태스크: AUC > 0 (0이면 예측 실패)
3. regression 태스크: MAE가 NaN이 아니어야 함
4. multiclass 태스크: F1 macro > 0
5. loss가 NaN이 아니어야 함

## 발견되는 에러는 전부 로컬에서 수정

SageMaker에 제출하기 전에 위 5개 Step을 전부 통과해야 합니다.
에러가 나면 코드를 수정하고 다시 테스트합니다.
SageMaker 비용은 $0입니다.

## 주요 참고 파일

- CLAUDE.md: 코드 작업 가드레일 (필수 읽기)
- configs/santander/pipeline.yaml: 전체 파이프라인 설정
- configs/santander/feature_groups.yaml: 피처 그룹 + generator 설정
- containers/training/train.py: 학습 스크립트 (~500줄)
- adapters/santander_adapter.py: 데이터 어댑터
- core/pipeline/runner.py: Phase 0 파이프라인 러너
- core/pipeline/normalizer.py: 3-stage 정규화
- core/pipeline/label_deriver.py: 레이블 파생 엔진
- core/model/ple/model.py: PLE 모델
```
