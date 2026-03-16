from dataclasses import dataclass, field


@dataclass
class PLEConfig:
    """
    Progressive Layered Extraction 모델 설정.

    YAML config의 `model:` 블록과 1:1 대응합니다.
    """
    input_dim: int                          # 피처 벡터 차원
    num_tasks: int                          # 동시 학습할 태스크 수
    num_shared_experts: int = 2             # 공유 Expert 수 (모든 태스크가 공유)
    num_task_experts: int = 2               # 태스크별 전용 Expert 수
    expert_hidden_dim: int = 256            # Expert 내부 hidden 차원
    num_layers: int = 2                     # PLE layer 수 (CGC 반복 횟수)
    tower_dims: list[int] = field(default_factory=lambda: [128, 64])
    dropout: float = 0.1
