from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LGBMConfig:
    num_leaves: int = 63
    max_depth: int = -1
    learning_rate: float = 0.05
    n_estimators: int = 500
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    class_weight: Optional[str] = "balanced"
    extra_params: dict = field(default_factory=dict)
