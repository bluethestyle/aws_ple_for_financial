# 운영/감사 에이전트 온프렘 구현 핸드오프 문서

> **작성일**: 2026-04-10
> **대상**: 온프렘 프로젝트 개발팀
> **참조**: `docs/design/11_ops_audit_agent.typ` (전체 설계), `core/agent/` (AWS 구현)

---

## 1. 개요

AWS 프로젝트에서 운영/감사 에이전트를 구현 완료했습니다. 온프렘에서 동일한 아키텍처를 구현할 때 참고할 내용을 정리합니다.

**핵심 원칙: 동일한 룰 엔진 + 다른 LLM 레이어**

```
┌─────────────────────────────────────────┐
│         공통 (AWS = 온프렘)               │
│  룰 엔진, 체크리스트, 도구 카탈로그,      │
│  케이스 스토어, 이벤트 브릿지             │
└──────────────────┬──────────────────────┘
                   │
       ┌───────────┴───────────┐
       │                       │
  ┌────▼─────┐          ┌─────▼──────┐
  │   AWS    │          │   온프렘    │
  │          │          │            │
  │ Bedrock  │          │ 로컬 GPU   │
  │ Sonnet×3 │          │ Qwen 14B   │
  │ Sonnet   │          │ Exaone 7.8B│
  │ 독립투표  │          │ 2-Round    │
  └──────────┘          └────────────┘
```

---

## 2. 그대로 가져갈 수 있는 것 (변경 없음)

AWS 구현의 다음 파일/모듈은 **온프렘에서 그대로 사용** 가능합니다:

| 모듈 | 파일 | 이유 |
|---|---|---|
| ToolRegistry | `core/agent/tool_registry.py` | Python 직접 호출 모드가 이미 구현됨 (Bedrock 내보내기 무시하면 됨) |
| BaseAgent | `core/agent/base.py` | 체크리스트 로딩, 판정 로직 동일 |
| ChangeDetector | `core/agent/change_detector.py` | git hook, pipeline state 콜백 동일 |
| OpsCollector | `core/agent/ops/collector.py` | 7개 체크포인트 수집 로직 동일 |
| OpsDiagnoser | `core/agent/ops/diagnoser.py` | 연쇄 영향 룰 동일 |
| OpsReporter | `core/agent/ops/reporter.py` | 템플릿 리포트 동일 |
| StratifiedReasonSampler | `core/agent/audit/reason_sampler.py` | 층화추출 로직 동일 |
| GroundingValidator | `core/agent/audit/grounding_validator.py` | 사유 품질 검증 동일 |
| Tier1Aggregator | `core/agent/audit/tier1_aggregator.py` | SelfChecker 집계 동일 |
| IntersectionalFairnessAnalyzer | `core/agent/audit/intersectional_fairness.py` | 교차 보호속성 분석 동일 |
| BiasStageAttributor | `core/agent/audit/bias_stage_attributor.py` | 편향 단계 분리 동일 |
| DiagnosticCaseStore | `core/agent/case_store.py` | LanceDB/numpy 듀얼 백엔드 이미 구현 |
| AgentEventBridge | `core/agent/event_bridge.py` | 에이전트 간 트리거 동일 |
| checklist.yaml | `configs/financial/checklist.yaml` | 48개 체크리스트 항목 동일 |
| _PipelineState 콜백 | `core/pipeline/runner.py` | 콜백 메커니즘 동일 |

**총 15개 모듈 중 15개가 재사용 가능** — 룰 엔진 기반이라 환경 무관.

---

## 3. 온프렘에서 다르게 구현해야 하는 것

### 3.1 합의 메커니즘: 독립 투표 → 2-Round 하이브리드

AWS의 `core/agent/consensus.py`는 Sonnet×3 독립 병렬 투표입니다.
온프렘에서는 **2-Round 하이브리드**로 변경해야 합니다.

#### 왜 다른가

- 14B 로컬 모델은 Sonnet보다 약하므로 에이전트 수를 늘려야 함 (3→5~7)
- 순수 독립 투표만으로는 14B의 추론력이 부족
- 그렇다고 순수 델파이(순차 심의)는 수렴 편향으로 마이너리티가 사라짐
- 따라서 **Round 1 독립 투표 + Round 2 순차 심의** 하이브리드

#### 구현 스펙

```
Round 1: 독립 투표 (마이너리티 보존)
  - Qwen 2.5 14B Q4 × 5개 (기본) 또는 7개 (고위험)
  - 각 에이전트가 서로의 출력을 보지 않음
  - temperature 변동 (0.3~0.7)으로 다양성 확보
  - GPU 1장이라 물리적으로 순차 실행이지만, 논리적으로는 독립
  - 이 시점에서 마이너리티 확정 — 이후 삭제 불가

Round 2: 순차 심의 (논거 보강)
  - 2개 에이전트가 Round 1 전체 결과를 보고 심의
  - 다수의견 논거 정리 + 소수의견 타당성 평가
  - 마이너리티를 없애는 것이 아니라 근거를 구체화하는 역할
```

#### 핵심 원칙

**Round 1에서 확정된 마이너리티는 절대 삭제되지 않는다.**

Round 2에서 "타당성 낮다"고 평가되더라도 원 의견은 보존됩니다.

#### Round 1 프롬프트 템플릿

```
## 진단 대상
체크항목 {item_id}: {description}
현재 값: {measurements}
룰 엔진 판정: {rule_engine_verdict}

## 참고 데이터
{similar_cases}

## 지시
이 판정에 대해 독립적으로 판단하세요.
PASS/WARN/FAIL 중 하나를 선택하고 상세한 근거를 제시하세요.
JSON 형식: {"verdict": "...", "confidence": 0.0-1.0, "reasoning": "...", "recommendation": "..."}
```

#### Round 2 프롬프트 템플릿

```
## 진단 대상
{동일}

## Round 1 투표 결과
에이전트 ①: {verdict} — "{reasoning}"
에이전트 ②: {verdict} — "{reasoning}"
...

집계: {verdict_counts}
마이너리티: {minority_agents}

## 지시
Round 1 전체 의견을 검토하고:
1. 다수의견의 논거를 종합 정리하세요.
2. 마이너리티의 근거가 타당한지 평가하세요.
   단, 마이너리티를 기각하더라도 원 의견은 보존됩니다.
3. 최종 종합 판정을 제시하세요.
```

#### 수정 포인트 (consensus.py 기준)

```python
# AWS 버전의 _collect_parallel() 대신:
def _collect_round1(self, agent_ids, base_prompt) -> List[AgentVote]:
    """Round 1: 독립 투표 (순차 실행, 논리적 독립)"""
    votes = []
    for i, aid in enumerate(agent_ids):
        # temperature 변동으로 다양성 확보
        temp = 0.3 + (0.4 * i / max(len(agent_ids) - 1, 1))
        vote = self._single_vote(aid, base_prompt, temperature=temp)
        votes.append(vote)
    return votes

def _collect_round2(self, round1_votes, base_prompt) -> List[AgentVote]:
    """Round 2: Round 1 결과를 보고 순차 심의"""
    round1_summary = self._format_round1_summary(round1_votes)
    round2_prompt = f"{base_prompt}\n\n## Round 1 결과\n{round1_summary}"
    
    votes = []
    for aid in ["synthesizer_1", "synthesizer_2"]:
        if votes:
            round2_prompt += f"\n\n## 이전 심의\n{votes[-1].reasoning}"
        vote = self._single_vote(aid, round2_prompt)
        votes.append(vote)
    return votes
```

### 3.2 LLM 프로바이더: Bedrock → 로컬 GPU

AWS의 `llm_provider.py`에 `LocalLLMProvider` 스텁이 있습니다. 온프렘에서 실제 구현이 필요합니다.

#### 모델 배정

| 용도 | 모델 | VRAM | 이유 |
|---|---|---|---|
| 추천사유 L2a 생성/critique | Exaone 3.5 7.8B | ~8GB | 한국어 자연스러움 필요 |
| 에이전트 합의 (Round 1+2) | Qwen 2.5 14B Q4 | ~9GB | 논리력/추론력 필요 |
| 임베딩 | sentence-transformers (all-MiniLM-L6-v2) | ~0.5GB | 케이스 스토어용 |

#### VRAM 관리: 순차 로딩

Exaone 8GB + Qwen 9GB = 17GB > RTX 4070 12GB — 동시 로딩 불가.

```
실행 순서:
1. 룰 엔진 체크리스트 실행 (GPU 불필요)
2. Qwen 14B Q4 로드 → 에이전트 합의 실행 → 언로드 (~25분)
3. Exaone 7.8B 로드 → 추천사유 L2a 배치 → 언로드 (별도 스케줄)
```

#### LocalLLMProvider 구현 가이드

```python
class LocalLLMProvider(AbstractLLMProvider):
    """On-prem GPU LLM provider using vLLM or llama.cpp."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        self._model_path = config.get("model_path", "")
        self._backend = config.get("backend", "vllm")  # "vllm" or "llamacpp"
        self._model = None
    
    def load(self) -> None:
        """Load model to GPU."""
        if self._backend == "vllm":
            from vllm import LLM
            self._model = LLM(model=self._model_path, dtype="float16")
        elif self._backend == "llamacpp":
            from llama_cpp import Llama
            self._model = Llama(model_path=self._model_path, n_gpu_layers=-1)
    
    def unload(self) -> None:
        """Unload model from GPU."""
        del self._model
        self._model = None
        import torch
        torch.cuda.empty_cache()
    
    def generate(self, prompt: str, **kwargs) -> str:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        # vLLM or llama.cpp 호출
        ...
    
    def is_available(self) -> bool:
        return self._model is not None
```

#### 모델 매니저

모델 로딩/언로딩을 관리하는 상위 컴포넌트가 필요합니다:

```python
class ModelManager:
    """Manages sequential model loading/unloading on single GPU."""
    
    def __init__(self, providers: Dict[str, LocalLLMProvider]) -> None:
        self._providers = providers
        self._active: Optional[str] = None
    
    def activate(self, name: str) -> LocalLLMProvider:
        """Unload current model, load requested model."""
        if self._active and self._active != name:
            self._providers[self._active].unload()
        self._providers[name].load()
        self._active = name
        return self._providers[name]
    
    def deactivate_all(self) -> None:
        if self._active:
            self._providers[self._active].unload()
            self._active = None
```

### 3.3 Bedrock Dialog → 제외

`core/agent/bedrock_dialog.py`는 온프렘에서 사용하지 않습니다.

온프렘에서의 담당자 인터페이스:
- 정형 리포트 (JSON/YAML) → 대시보드 또는 이메일로 전달
- 추가 맥락이 필요하면 담당자가 직접 수치를 보고 판단
- 또는 필요 시 수동으로 Claude 등 외부 LLM에 질의

### 3.4 NotificationService: SNS → 대체

`core/agent/notification.py`의 SNS 채널은 온프렘에서 사용 불가.

대체 옵션:
- **Slack webhook**: 그대로 사용 가능 (인터넷 연결 시)
- **이메일 (SMTP)**: 사내 메일 서버 직접 연동
- **사내 메신저 API**: 조직별 상이

### 3.5 CloudWatch 메트릭 → 대체

`query_cloudwatch_metrics` 도구는 온프렘에서 사용 불가.

대체 옵션:
- **Prometheus + Grafana**: 서빙 latency, A/B 메트릭 수집
- **커스텀 메트릭 파일**: JSON/CSV로 주기 기록 → 도구가 파일 읽기
- **DuckDB 집계**: audit_archive Parquet를 DuckDB로 직접 쿼리

---

## 4. 온프렘 전용 설정 (agent.yaml)

```yaml
agent:
  # 공통 설정은 AWS와 동일
  ops:
    checkpoints: [CP1, CP2, CP3, CP4, CP5, CP6, CP7]
    schedule:
      CP1: event
      CP2: event
      CP3: event
      CP4: event
      CP5: 5min
      CP6: 1h
      CP7: daily

  audit:
    viewpoints: [AV1, AV2, AV3, AV4, AV5]
    schedule:
      AV1: daily
      AV2: daily
      AV3: mixed
      AV4: weekly
      AV5: on_change

  # 온프렘 전용 합의 설정
  consensus:
    mode: two_round_hybrid  # AWS는 "independent_voting"
    round1:
      model: qwen_14b_q4
      agents: 5              # 기본 5, 고위험 7
      temperature_range: [0.3, 0.7]
    round2:
      model: qwen_14b_q4
      agents: 2              # 항상 2
    apply_to:
      - checklist_warn_fail
      - impact_review
      - regulatory_judgment

  # 온프렘 모델 설정
  models:
    reason_generation: exaone_7b    # Exaone 3.5 7.8B
    reason_critique: exaone_7b
    agent_consensus: qwen_14b_q4    # Qwen 2.5 14B Q4
    embeddings: minilm_v2           # sentence-transformers
    # dialog, deep_audit 없음 (온프렘에서 미제공)

  # 모델 경로
  model_paths:
    exaone_7b: "/models/exaone-3.5-7.8b-instruct"
    qwen_14b_q4: "/models/qwen2.5-14b-instruct-q4_k_m.gguf"
    minilm_v2: "/models/all-MiniLM-L6-v2"

  # GPU 관리
  gpu:
    device: "cuda:0"
    vram_gb: 12
    sequential_loading: true  # 모델 순차 로딩 (동시 불가)
```

---

## 5. 구현 우선순위 (온프렘)

온프렘은 AWS보다 범위가 좁습니다 (Bedrock dialog, Sonnet 없음).

```
1. LocalLLMProvider 구현 (vLLM/llama.cpp) + ModelManager
   → 이것이 온프렘의 핵심 신규 개발

2. ConsensusArbiter 2-Round 하이브리드 모드 추가
   → AWS consensus.py를 확장하거나 별도 클래스

3. CloudWatch 대체 도구 구현
   → query_cloudwatch_metrics → query_local_metrics

4. NotificationService SMTP 채널 추가
   → 기존 Slack은 그대로, SNS 대신 SMTP

5. 나머지는 AWS 코드 그대로 복사
```

---

## 6. 소요시간 추정

| 태스크 | 예상 LOC | 예상 기간 |
|---|---|---|
| LocalLLMProvider + ModelManager | ~200 | 1일 |
| 2-Round 하이브리드 ConsensusArbiter | ~150 | 1일 |
| CloudWatch 대체 도구 | ~80 | 0.5일 |
| NotificationService SMTP | ~50 | 0.5일 |
| AWS 코드 이식 + 테스트 | -- | 1일 |
| **합계** | **~480** | **~4일** |

AWS 구현 3,700 LOC 중 **3,200 LOC은 그대로 재사용**, 추가 개발 ~480 LOC.

---

## 7. 주의사항

### 7.1 마이너리티 리포트 보존 원칙

2-Round 하이브리드에서 가장 중요한 원칙:
**Round 1에서 확정된 마이너리티는 Round 2에서 삭제할 수 없다.**

이 원칙이 깨지면 에이전트의 감사 적합성이 훼손됩니다.
금감원이 "왜 이 소수 의견이 사라졌나?"라고 물었을 때 답이 없어집니다.

### 7.2 VRAM 관리 실패 시

Qwen 14B Q4가 OOM을 일으키면:
- `n_gpu_layers`를 줄여서 CPU offloading
- 또는 Qwen 7B Q4 (~5GB)로 대체 (추론력 저하 감수)
- 또는 Exaone 7.8B를 합의에도 사용 (한국어는 좋지만 추론력은 Qwen 14B보다 낮음)

### 7.3 Exaone 3.5 라이선스

Apache 2.0 — 상업적 사용 가능, 수정/재배포 자유.
단, LG AI Research 귀속 표시 필요.

### 7.4 K-Exaone 업그레이드 경로

K-Exaone (236B MoE, 23B active)이 오픈소스화되면:
- 추론 시 23B만 활성 → VRAM ~14GB, RTX 4070으로 빠듯하지만 가능 가능성
- 한국어 품질이 글로벌 7위 수준으로 대폭 상승
- Exaone 3.5 7.8B → K-Exaone으로 교체하면 사유 품질 + 합의 품질 모두 향상
- `model_paths` config만 변경하면 되도록 설계할 것

### 7.5 데이터 처리 백엔드

온프렘에서 audit_archive 등의 Parquet 쿼리는 DuckDB 사용 (CLAUDE.md 정책):
- `pd.read_parquet()` 대신 `duckdb.execute("SELECT ... FROM 'file.parquet'")`
- 집계/통계는 SQL로 처리

---

## 8. 테스트 체크리스트

온프렘 구현 완료 후 확인할 항목:

- [ ] Qwen 14B Q4 로드 → 추론 → 언로드가 OOM 없이 동작하는가
- [ ] Exaone 7.8B 로드 → 추론 → 언로드가 OOM 없이 동작하는가
- [ ] 모델 전환 (Qwen → Exaone) 시 VRAM이 완전히 해제되는가
- [ ] Round 1 독립 투표에서 5개 에이전트가 실제로 다른 출력을 내는가
- [ ] Round 1에서 확정된 마이너리티가 Round 2 후에도 보존되는가
- [ ] 체크리스트 48개 항목이 전부 도구 호출 + 판정이 되는가
- [ ] 케이스 스토어에 저장/검색/통계가 정상 동작하는가
- [ ] 정형 리포트가 ops_report / audit_report YAML 형식에 맞는가
- [ ] WARN/FAIL 10개 항목 합의 처리가 30분 이내에 완료되는가
