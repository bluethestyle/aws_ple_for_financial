# 04. Training Pipeline — SageMaker Training, 2-Phase, 실험 관리

## 현재 (On-Prem) 분석

### 학습 구조
- **2-Phase**: Phase1 (Shared Expert 15 epoch) → Phase2 (Cluster Head 8 epoch)
- **옵티마이저**: AdamW (lr=0.0005, weight_decay=0.01)
- **스케줄러**: CosineAnnealingWarmRestarts (T0=10, Tmult=2)
- **Mixed Precision**: fp16 (AMP)
- **Gradient**: Clip=5.0, Accumulation=4 (effective batch: 16384)
- **실험 관리**: MLflow (Docker 자체 호스팅)
- **체크포인트**: 로컬 디스크, max 5개

### 문제점
1. **학습 환경 고정**: 특정 GPU 머신에 종속
2. **MLflow 서버 유지비용**: Docker로 상시 가동 필요
3. **체크포인트 유실 위험**: 로컬 디스크만 사용
4. **재현성 미흡**: 환경 + 코드 + 데이터 버전이 느슨하게 연결

### 유지할 패턴
- **2-Phase Training**: 검증된 전략
- **Dynamic Loss Weighting**: 불확실성 기반 태스크 가중치 조절
- **Expert별 Learning Rate**: 안정적 학습에 효과적

---

## AWS 설계

### 학습 파이프라인 흐름

```
configs/training.yaml
         ↓
Step Functions (오케스트레이션)
         ↓
   ┌─────────────┐
   │ Phase 1     │ SageMaker Training Job #1
   │ Shared      │ - Spot Instance (g4dn.xlarge)
   │ Experts     │ - 체크포인트 → S3 (Spot 중단 대비)
   │ (15 epoch)  │ - 메트릭 → SageMaker Experiments
   └──────┬──────┘
          │ S3에 Phase1 체크포인트 저장
          ▼
   ┌─────────────┐
   │ Phase 2     │ SageMaker Training Job #2
   │ Task Heads  │ - Phase1 체크포인트 로드
   │ Fine-tune   │ - Shared Expert freeze (선택적)
   │ (8 epoch)   │ - 최종 모델 → S3
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │ 평가         │ SageMaker Processing Job
   │ + 등록       │ - 테스트셋 평가
   │              │ - Champion/Challenger 비교
   │              │ - 합격 시 Model Registry 등록
   └─────────────┘
```

### Training Config (YAML)

```yaml
# configs/training.yaml
training:
  seed: 42

  # ── Phase 1: Shared Expert 학습 ──
  phase1:
    epochs: 15
    batch_size: 4096
    optimizer:
      type: adamw
      lr: 0.0005
      weight_decay: 0.01
    scheduler:
      type: cosine_warmup
      warmup_epochs: 5
      T_0: 10
      T_mult: 2
    gradient:
      clip_norm: 5.0
      accumulation_steps: 4
    mixed_precision: true     # fp16

    # Expert별 학습률 (선택적)
    expert_lr_overrides:
      temporal: 0.0003        # 시퀀스 Expert는 보수적으로
      hgcn: 0.0005
      causal: 0.0001          # DAG 제약 → 매우 보수적

  # ── Phase 2: Task Head 미세 조정 ──
  phase2:
    epochs: 8
    freeze_shared: true       # Shared Expert 동결
    batch_size: 4096
    optimizer:
      type: adamw
      lr: 0.0002              # Phase1보다 낮은 lr
      weight_decay: 0.01
    scheduler:
      type: cosine_warmup
      warmup_epochs: 2

  # ── 조기 종료 ──
  early_stopping:
    patience: 5
    metric: val_loss
    mode: min

  # ── 체크포인트 ──
  checkpoint:
    save_every_n_epochs: 5
    max_keep: 3
    save_best: true
    s3_path: s3://aiops-ple-financial/models/checkpoints/

  # ── Loss 가중치 전략 ──
  loss_weighting:
    strategy: uncertainty      # uncertainty | fixed | gradnorm
    # uncertainty: 각 태스크의 불확실성(σ²)으로 자동 가중치 조절

# ── SageMaker 설정 ──
aws:
  instance_type: ml.g4dn.xlarge
  use_spot: true
  max_run_seconds: 14400      # 4시간
  volume_size_gb: 50
```

### SageMaker Training 래퍼

```python
# aws/sagemaker/trainer.py (확장)
class SageMakerTrainer:
    """
    2-Phase Training을 SageMaker Job 2개로 분리 실행합니다.

    Phase1 완료 → S3에 체크포인트 → Phase2에서 로드
    Spot 인스턴스 중단 시 → S3 체크포인트에서 자동 재개
    """

    def launch(self) -> dict:
        # Phase 1
        phase1_result = self._launch_phase(
            phase="phase1",
            hyperparameters=self.config.training.phase1,
            checkpoint_s3=self.config.training.checkpoint.s3_path,
        )

        # Phase 2 (Phase1 체크포인트를 입력으로)
        phase2_result = self._launch_phase(
            phase="phase2",
            hyperparameters=self.config.training.phase2,
            model_uri=phase1_result["s3_model_uri"],
            checkpoint_s3=self.config.training.checkpoint.s3_path,
        )

        return phase2_result
```

### 실험 관리 — MLflow vs SageMaker Experiments

| 항목 | MLflow (현재) | SageMaker Experiments (AWS) |
|------|-------------|---------------------------|
| 서버 | Docker 자체 호스팅 (상시 가동) | AWS 관리형 (비용 0) |
| 메트릭 로깅 | `mlflow.log_metric()` | `sagemaker.experiments` |
| 모델 저장 | MLflow Model Registry | S3 + 메타데이터 |
| 하이퍼파라미터 | 수동 로깅 | YAML에서 자동 추출 |
| 실험 비교 | MLflow UI | SageMaker Studio |

```python
# core/experiment/tracker.py
class ExperimentTracker:
    """
    MLflow / SageMaker Experiments 추상화.

    로컬 개발: MLflow (docker 불필요, 파일 백엔드)
    AWS 실행: SageMaker Experiments (자동)
    """

    def __init__(self, backend="auto"):
        if backend == "auto":
            self.backend = "sagemaker" if self._in_sagemaker() else "mlflow"

    def log_metric(self, name: str, value: float, step: int = None):
        ...

    def log_params(self, params: dict):
        ...

    def log_model(self, model_path: str, metadata: dict):
        ...
```

### Champion/Challenger 평가

```
현재 모델 (Champion)
    vs.
새 모델 (Challenger)
    ↓
┌─────────────────────────────────────────┐
│ 평가 기준                                │
│                                         │
│ 1. 주요 메트릭 (태스크별)                 │
│    - Binary: AUC-ROC > champion - 0.01  │
│    - Regression: MAE < champion + 5%    │
│    - Multiclass: F1-macro > champion    │
│                                         │
│ 2. 추론 성능                             │
│    - Latency p99 < 100ms               │
│    - Throughput > 1000 req/s            │
│                                         │
│ 3. 안정성                                │
│    - 테스트셋 분포 PSI < 0.1             │
│    - 예측 분포 변화 < 10%                │
│                                         │
│ → 모두 통과: 자동 등록                    │
│ → 하나라도 실패: 알림 → 수동 검토         │
└─────────────────────────────────────────┘
```

---

## Spot 인스턴스 전략

```
비용 비교:
  On-Demand g4dn.xlarge: $0.526/hr
  Spot g4dn.xlarge:      ~$0.16/hr (70% 절감)

Spot 중단 대비:
  1. 매 epoch마다 S3에 체크포인트 저장
  2. SageMaker CheckpointConfig 설정 → 자동 재개
  3. max_wait = max_run + 1hr (Spot 재할당 대기)

학습 4시간 기준:
  On-Demand: $2.10
  Spot:      ~$0.64  ← 이 비용으로 학습 1회
```

---

## 현재 vs AWS — 학습 파이프라인 비교

| 항목 | 현재 (On-Prem) | AWS (설계) | 변경 이유 |
|------|---------------|-----------|----------|
| 실행 환경 | 고정 GPU 서버 | SageMaker Spot (필요 시만) | 비용 70% 절감 |
| 2-Phase | trainer.py 내부 로직 | SageMaker Job 2개 분리 | 각 Phase 독립 재실행 가능 |
| 체크포인트 | 로컬 디스크 | S3 (Spot 중단 자동 대비) | 내구성, 재개 |
| 실험 관리 | MLflow Docker | SageMaker Experiments | 서버 유지비 0 |
| 모델 비교 | 수동 | Champion/Challenger 자동화 | 배포 안정성 |
| 재현성 | 느슨 (코드+데이터 별도) | YAML config + S3 데이터 버전 | 완전 재현 |
