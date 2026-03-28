# SageMaker 자원 추정 리포트 생성기 설계

## 목적

로컬 모드에서 학습/처리 작업을 실행한 후, 실제 SageMaker 제출 시 필요한 인스턴스 사양과 예상 비용을 자동으로 리포트하여 **실패 없는 1회 제출**을 보장한다.

## 배경

| 문제 | 원인 | 비용 |
|------|------|------|
| OOM 실패 | 데이터 크기 대비 인스턴스 RAM 부족 | 실패당 $2~3 |
| VRAM 초과 | 모델+배치 크기 대비 GPU 메모리 부족 | 실패당 $2~3 |
| 과잉 인스턴스 | 필요량 모르고 큰 인스턴스 선택 | 시간당 $1~4 낭비 |
| 시간 초과 | max_run 추정 불가 → 과대 설정 | 대기 비용 발생 |

3월 기준 실패 비용 $14+ 발생. 사전 추정으로 전액 절약 가능했음.

## 실행 흐름

```
[로컬 학습 실행]
    │
    ├── psutil: peak RSS (CPU RAM)
    ├── torch.cuda: max_memory_allocated (GPU VRAM)
    ├── time: wall clock / GPU time
    ├── data: input parquet 크기, 행수, 컬럼수, dtype별 메모리
    └── model: 파라미터 수, optimizer state 크기
    │
    ▼
[ResourceEstimator 분석]
    │
    ├── 1. 메모리 요구량 산출
    │     RAM: peak_rss × 1.3 (안전 마진 30%)
    │     VRAM: peak_vram × 1.2 (안전 마진 20%)
    │
    ├── 2. 인스턴스 매칭
    │     요구 RAM/VRAM에 맞는 최소 인스턴스 자동 선택
    │     CPU/GPU 각각 추천
    │
    ├── 3. 비용 추정
    │     On-Demand 가격 × 예상 시간
    │     Spot 가격 × 예상 시간 (70% 할인 적용)
    │
    ├── 4. max_run / max_wait 추천
    │     로컬 실행 시간 × 1.5 (네트워크/다운로드 오버헤드)
    │
    └── 5. 위험 요소 경고
          list/struct 컬럼 존재 여부
          zero-variance 컬럼 비율
          데이터 크기 대비 인스턴스 여유율
    │
    ▼
[리포트 생성]
    resource_estimate_report.json + 콘솔 요약 출력
```

## 측정 항목

### 데이터 프로파일링

| 항목 | 측정 방법 | 용도 |
|------|----------|------|
| parquet 파일 크기 | os.path.getsize() | S3 전송 시간 추정 |
| 행 수 | len(df) | 배치 수 계산 |
| 컬럼 수 (scalar) | dtype 필터 | 유효 피처 수 |
| 컬럼 수 (list/struct) | pyarrow schema | OOM 위험 경고 |
| zero-variance 컬럼 수 | std == 0 카운트 | 불필요 컬럼 제거 권고 |
| DataFrame 메모리 | df.memory_usage(deep=True) | RAM 요구량 기준 |

### 런타임 프로파일링

| 항목 | 측정 방법 | 용도 |
|------|----------|------|
| Peak CPU RAM | psutil.Process().memory_info().rss | 인스턴스 RAM 선택 |
| Peak GPU VRAM | torch.cuda.max_memory_allocated() | GPU 인스턴스 선택 |
| 학습 시간 (wall clock) | time.time() | 비용 추정 |
| GPU 연산 시간 | torch.cuda.Event + elapsed_time | GPU 활용률 |
| 에폭당 시간 | epoch_times 리스트 | max_run 추정 |
| 배치당 시간 | batch_times 리스트 | 병목 분석 |

### 모델 프로파일링

| 항목 | 측정 방법 | 용도 |
|------|----------|------|
| 총 파라미터 수 | sum(p.numel()) | 모델 크기 추정 |
| 모델 메모리 | 파라미터 × dtype 크기 | VRAM 요구량 |
| Optimizer 상태 | Adam = 파라미터 × 3 | 추가 VRAM |
| Gradient 메모리 | 파라미터 × dtype | 추가 VRAM |

## 인스턴스 매칭 로직

```python
INSTANCE_SPECS = {
    # CPU 인스턴스
    "ml.m5.large":    {"vcpu": 2,  "ram_gb": 8,   "gpu": None, "on_demand": 0.115, "spot": 0.035},
    "ml.m5.xlarge":   {"vcpu": 4,  "ram_gb": 16,  "gpu": None, "on_demand": 0.230, "spot": 0.069},
    "ml.m5.2xlarge":  {"vcpu": 8,  "ram_gb": 32,  "gpu": None, "on_demand": 0.461, "spot": 0.138},
    "ml.m5.4xlarge":  {"vcpu": 16, "ram_gb": 64,  "gpu": None, "on_demand": 0.922, "spot": 0.277},
    # GPU 인스턴스
    "ml.g4dn.xlarge": {"vcpu": 4,  "ram_gb": 16,  "gpu": "T4",   "vram_gb": 16, "on_demand": 0.736, "spot": 0.221},
    "ml.g5.xlarge":   {"vcpu": 4,  "ram_gb": 16,  "gpu": "A10G", "vram_gb": 24, "on_demand": 1.408, "spot": 0.422},
    "ml.p3.2xlarge":  {"vcpu": 8,  "ram_gb": 61,  "gpu": "V100", "vram_gb": 16, "on_demand": 3.825, "spot": 1.148},
}

def recommend_instance(peak_ram_gb, peak_vram_gb=None):
    """
    안전 마진 적용 후 최소 비용 인스턴스 추천
    RAM: peak × 1.3 (30% 마진)
    VRAM: peak × 1.2 (20% 마진)
    """
    required_ram = peak_ram_gb * 1.3
    required_vram = peak_vram_gb * 1.2 if peak_vram_gb else None

    candidates = []
    for name, spec in INSTANCE_SPECS.items():
        if spec["ram_gb"] < required_ram:
            continue
        if required_vram and (not spec.get("vram_gb") or spec["vram_gb"] < required_vram):
            continue
        candidates.append((name, spec))

    # Spot 가격 기준 최소 비용 선택
    return sorted(candidates, key=lambda x: x[1]["spot"])[0]
```

## 리포트 출력 형식

### 콘솔 요약

```
═══════════════════════════════════════════════════
  SageMaker Resource Estimate Report
═══════════════════════════════════════════════════

  Data Profile
  ─────────────────────────────────────
  Input:        santander_final.parquet (49.4 MB)
  Rows:         941,132
  Columns:      303 (scalar 190 / list 30 / zero-var 113)
  DataFrame:    2.3 GB (scalar only)

  Runtime Profile (Local)
  ─────────────────────────────────────
  Peak RAM:     7.8 GB
  Peak VRAM:    5.2 GB
  Wall time:    23 min 41 sec
  Per epoch:    2 min 22 sec (10 epochs)

  Model Profile
  ─────────────────────────────────────
  Parameters:   2.4M
  Model mem:    9.2 MB
  Optimizer:    27.6 MB (Adam)
  Total GPU:    ~5.2 GB (data + model + grad + optim)

  ✅ Recommended Instance
  ─────────────────────────────────────
  GPU:          ml.g4dn.xlarge (T4 16GB, RAM 16GB)
  Fit:          RAM 7.8/16 (49%) | VRAM 5.2/16 (33%)

  Cost Estimate (10 epochs):
    On-Demand:  24 min × $0.736/hr = $0.29
    Spot:       24 min × $0.221/hr = $0.09

  max_run:      2100 sec (35 min, 로컬 × 1.5)
  max_wait:     3600 sec (max_run + 25 min)

  ⚠️ Warnings
  ─────────────────────────────────────
  - 113 zero-variance columns detected (37%)
    → 제거 시 RAM 1.1GB 절약 가능
  - 30 list/struct columns in parquet
    → scalar만 로드하는 필터 필수 (OOM 방지)
═══════════════════════════════════════════════════
```

### JSON 리포트 (resource_estimate_report.json)

```json
{
  "timestamp": "2026-03-28T14:30:00",
  "data_profile": {
    "file": "santander_final.parquet",
    "file_size_mb": 49.4,
    "rows": 941132,
    "columns_total": 303,
    "columns_scalar": 190,
    "columns_list": 30,
    "columns_zero_var": 113,
    "dataframe_memory_gb": 2.3
  },
  "runtime_profile": {
    "peak_ram_gb": 7.8,
    "peak_vram_gb": 5.2,
    "wall_time_sec": 1421,
    "epoch_time_sec": 142,
    "epochs": 10,
    "gpu_utilization_pct": 78
  },
  "model_profile": {
    "total_params": 2400000,
    "model_memory_mb": 9.2,
    "optimizer_memory_mb": 27.6,
    "total_gpu_memory_gb": 5.2
  },
  "recommendation": {
    "instance_type": "ml.g4dn.xlarge",
    "ram_gb": 16,
    "vram_gb": 16,
    "ram_usage_pct": 49,
    "vram_usage_pct": 33,
    "cost_on_demand": 0.29,
    "cost_spot": 0.09,
    "max_run_sec": 2100,
    "max_wait_sec": 3600
  },
  "warnings": [
    "113 zero-variance columns (37%) — 제거 시 RAM 1.1GB 절약",
    "30 list/struct columns — scalar 필터 필수"
  ]
}
```

## 구현 위치

```
scripts/
  resource_estimator.py      ← 메인 모듈

사용 방법:
  1. 로컬 학습 시 자동 측정
     python containers/training/train.py --local --estimate-resources

  2. 독립 실행 (데이터 프로파일링만)
     python scripts/resource_estimator.py --data path/to/data.parquet

  3. 학습 후 리포트 생성
     python scripts/resource_estimator.py --from-log training_log.json
```

## CLAUDE.md 추가 규칙

```
### 1.4 실험 전 검증 (Pre-flight Check)
  6. **로컬 자원 추정**: SageMaker Job 제출 전에 반드시 로컬에서
     resource_estimator.py를 실행하여 RAM/VRAM 사용량을 확인하고,
     추천 인스턴스와 예상 비용을 검증한다.
     로컬 검증 없이 SageMaker에 직접 제출하지 않는다.
```
