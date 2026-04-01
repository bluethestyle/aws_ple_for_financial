# Anthropic 우수사례 제출 관점

## 1. Anthropic이 원하는 스토리

### 핵심 메시지
"3명의 금융 도메인 전문가가 Claude를 중심으로 한 AI 에이전트 팀을 운용하여,
10명 규모의 ML 엔지니어링 팀이 필요한 복잡한 금융 AI 시스템을
단기간에 설계부터 프로덕션 수준까지 구축한 사례"

### Anthropic 관점에서의 매력 포인트

**1. 생산성 혁신 (Productivity)**
- PM 1 + 팀원 2 → 18-task, 7-expert, 48-scenario ablation 시스템 구축
- Claude Opus: 아키텍처 설계, 기술 검증, 복잡한 버그 진단
- Claude Sonnet: 코드 구현, 리팩토링, 병렬 작업
- Cursor + Claude Code Extension + Teams: 도구별 특화 활용
- 기존 대비 3-5배 생산성 향상 추정

**2. AI가 AI를 만드는 구조 (AI-Augmented AI Development)**
- Claude가 PLE+adaTT 모델 코드를 구현
- 그 모델의 출력을 LLM Agent(Claude/Gemini)가 추천사유로 변환
- Safety Gate Agent가 생성된 사유를 검증
- 개발 도구 → 제품 내 AI → 제품의 안전장치까지 AI가 관통

**3. 책임있는 AI (Responsible AI — Anthropic 철학 부합)**
- Safety Gate: 추천사유의 할루시네이션/규제위반 자동 검증
- Human-in-the-Loop: 완전 자동화가 아닌 사람 개입 원칙
- Evidential Uncertainty: 모델이 "모른다"고 말할 수 있는 능력
- 금감원 + EU AI Act 규제 대응이 아키텍처에 내장
- Constitutional AI 철학과 자연스럽게 연결

**4. Enterprise 레퍼런스 (비즈니스 가치)**
- 학술 데모가 아닌 실제 금융사 운영 가능 수준
- SageMaker + Lambda 서버리스 → 인프라 비용 최소화
- config-driven → 새 금융사에 빠른 도입 가능
- Anthropic의 enterprise 고객(금융사) 유치에 활용 가능한 사례

**5. Claude Code 활용 모범 사례 (Developer Tools)**
- CLAUDE.md: 프로젝트 가드레일로 코드 품질 유지
  - Config-driven 원칙, 관심사 분리, 검수 기준 등을 명시
  - 여러 AI 에이전트가 병렬로 작업해도 일관성 유지
- 메모리 뱅크: 세션 간 맥락 유지 → 컨텍스트 윈도우 제약 극복
- 서브에이전트 규칙: Opus/Sonnet 역할 분담, 병렬 운용
- 다른 기업 고객에게 "이렇게 쓰면 대규모 프로젝트도 가능합니다" 사례

## 2. 제출 프레임워크

### Before → After
```
Before:
  - ALS 기반 단순 추천, 설명 불가, 규제 대응 수동
  - ML 운영 인력 부족, 새 모델 도입 엄두 못 냄

After (Claude 활용):
  - 18-task 멀티태스크 추천, 구조적 설명 가능
  - 금감원/EU AI Act 대응 자동화
  - 3명이 전체 시스템 구축 및 운영
```

### 정량적 성과 (ablation 결과 확정 후 업데이트)
- 모델 성능: AUC 기존 대비 향상 폭 (ablation 결과)
- 개발 생산성: 코드 라인 수 대비 투입 인력/기간
- 비용 효율: SageMaker Spot + Lambda vs 기존 인프라 비교
- 운영 효율: 관리 포인트 N개 → 1개 (config-driven)

### Claude 활용 상세
| 활용 단계 | Claude 역할 | 구체적 기여 |
|-----------|------------|-----------|
| 설계 | Opus | 아키텍처 트레이드오프 분석, 기술 실현 가능성 검증 |
| 구현 | Opus + Sonnet | 레이어별 병렬 코드 생성, 10개 generator GPU 전환 |
| 디버깅 | Opus (Claude Code) | label leakage 3건 발견 및 수정, GPU 최적화 |
| 검수 | Opus | 인터페이스 계약 검증, 하드코딩 스캔, 컴파일 체크 |
| 문서 | Opus | 설계 문서 6종, 가이드 5종, 논문 재료 8종 |
| 운영 | Claude Code Extension | 실시간 모니터링, ablation 진행 관리 |

## 3. 제출 시 강조할 차별점

### 다른 AI 코딩 사례와의 차이
대부분의 AI 코딩 사례: "코파일럿으로 코드 자동완성"
이 프로젝트: "AI 에이전트 팀을 조직하여 복잡한 시스템을 설계부터 구축"

- 단순 코드 생성이 아니라 **아키텍처 의사결정**에 AI 참여
- 단일 에이전트가 아니라 **다중 에이전트 병렬 운용**
- 코딩 보조가 아니라 **AI가 AI 제품을 만드는** 메타 구조
- 가드레일(CLAUDE.md)로 **품질 관리까지 체계화**

### 금융 도메인 특수성
- "빠르게 만들었다"만으로는 부족 → "규제를 준수하면서 빠르게 만들었다"
- Safety Gate, Human-in-the-Loop, 감사 추적이 Anthropic의 안전한 AI 철학과 부합
- 금융사라는 보수적 산업에서 AI 에이전트를 적극 활용한 선례
