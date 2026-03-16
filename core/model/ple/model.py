"""
Progressive Layered Extraction (PLE) — Tang et al., RecSys 2020

멀티태스크 학습을 위한 범용 아키텍처.
각 태스크는 전용 Expert + 공유 Expert를 Gating Network로 선택적으로 결합합니다.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import PLEConfig
from ...task.base import AbstractTask, TaskOutput


class Expert(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatingNetwork(nn.Module):
    """각 태스크가 어떤 Expert 출력을 얼마나 사용할지 결정합니다."""

    def __init__(self, input_dim: int, num_experts: int):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x: torch.Tensor, expert_outputs: torch.Tensor) -> torch.Tensor:
        # expert_outputs: (batch, num_experts, hidden_dim)
        weights = F.softmax(self.gate(x), dim=-1)          # (batch, num_experts)
        return (weights.unsqueeze(-1) * expert_outputs).sum(dim=1)  # (batch, hidden_dim)


class CGCLayer(nn.Module):
    """Customized Gate Control (CGC) — PLE의 핵심 레이어."""

    def __init__(self, input_dim: int, num_tasks: int, config: PLEConfig):
        super().__init__()
        hidden = config.expert_hidden_dim
        dropout = config.dropout

        self.shared_experts = nn.ModuleList([
            Expert(input_dim, hidden, dropout)
            for _ in range(config.num_shared_experts)
        ])
        self.task_experts = nn.ModuleList([
            nn.ModuleList([Expert(input_dim, hidden, dropout) for _ in range(config.num_task_experts)])
            for _ in range(num_tasks)
        ])
        num_total = config.num_shared_experts + config.num_task_experts
        self.gating = nn.ModuleList([
            GatingNetwork(input_dim, num_total) for _ in range(num_tasks)
        ])

    def forward(self, x: torch.Tensor, task_inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        shared_outs = torch.stack([e(x) for e in self.shared_experts], dim=1)

        task_outputs = []
        for i, gate in enumerate(self.gating):
            task_outs = torch.stack([e(task_inputs[i]) for e in self.task_experts[i]], dim=1)
            all_outs = torch.cat([task_outs, shared_outs], dim=1)
            task_outputs.append(gate(task_inputs[i], all_outs))

        return task_outputs


class PLEModel(nn.Module):
    """
    PLE 기반 멀티태스크 학습 모델.

    태스크 정의는 외부에서 TaskRegistry를 통해 주입됩니다.
    이 클래스는 PLE 구조만 담당하고 비즈니스 로직은 갖지 않습니다.

    Example:
        tasks = [
            TaskRegistry.build(TaskConfig("ctr", TaskType.BINARY, ...), tower_input_dim=64),
            TaskRegistry.build(TaskConfig("ltv", TaskType.REGRESSION, ...), tower_input_dim=64),
        ]
        model = PLEModel(config, tasks)
        outputs = model(x, labels={"ctr": y_ctr, "ltv": y_ltv})
    """

    def __init__(self, config: PLEConfig, tasks: list[AbstractTask]):
        super().__init__()
        self.config = config
        self.tasks = nn.ModuleList(tasks)
        self.task_names = [t.name for t in tasks]

        self.layers = nn.ModuleList()
        in_dim = config.input_dim
        for _ in range(config.num_layers):
            self.layers.append(CGCLayer(in_dim, len(tasks), config))
            in_dim = config.expert_hidden_dim

    def forward(
        self,
        x: torch.Tensor,
        labels: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, TaskOutput]:
        task_inputs = [x] * len(self.tasks)

        for layer in self.layers:
            task_inputs = layer(x, task_inputs)

        outputs = {}
        for i, task in enumerate(self.tasks):
            lbl = labels.get(task.name) if labels else None
            outputs[task.name] = task(task_inputs[i], lbl)

        return outputs

    def compute_total_loss(self, outputs: Dict[str, TaskOutput]) -> torch.Tensor:
        total = torch.tensor(0.0)
        for i, task in enumerate(self.tasks):
            if outputs[task.name].loss is not None:
                total = total + task.config.loss_weight * outputs[task.name].loss
        return total
