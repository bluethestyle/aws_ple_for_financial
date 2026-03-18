"""
LightGBM multi-task wrapper.

Trains and predicts with independent LGBM models per task.
Shares the same interface as PLE, so it can be swapped in pipelines.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import LGBMConfig
from ...task.types import TaskType


class LGBMModel:
    """
    Collection of independent LGBM models, one per task.

    Example:
        model = LGBMModel(config, tasks_meta)
        model.fit(X_train, y_dict)
        preds = model.predict(X_test)   # {"task_name": np.ndarray, ...}
    """

    def __init__(self, config: LGBMConfig, tasks_meta: list[dict]):
        """
        Args:
            tasks_meta: [{"name": "ctr", "type": "binary"}, ...]
        """
        self.config = config
        self.tasks_meta = tasks_meta
        self.models: Dict[str, object] = {}

    def _build_booster(self, task_type: str):
        import lightgbm as lgb
        params = {
            "num_leaves": self.config.num_leaves,
            "max_depth": self.config.max_depth,
            "learning_rate": self.config.learning_rate,
            "n_estimators": self.config.n_estimators,
            "min_child_samples": self.config.min_child_samples,
            "subsample": self.config.subsample,
            "colsample_bytree": self.config.colsample_bytree,
            "reg_alpha": self.config.reg_alpha,
            "reg_lambda": self.config.reg_lambda,
            **self.config.extra_params,
        }
        if task_type == TaskType.BINARY:
            return lgb.LGBMClassifier(objective="binary", **params)
        elif task_type == TaskType.MULTICLASS:
            return lgb.LGBMClassifier(objective="multiclass", **params)
        else:
            return lgb.LGBMRegressor(objective="regression_l1", **params)

    def fit(
        self,
        X: "pd.DataFrame | np.ndarray",
        y_dict: Dict[str, np.ndarray],
        eval_set: Optional[tuple] = None,
    ) -> "LGBMModel":
        for task in self.tasks_meta:
            name = task["name"]
            booster = self._build_booster(task["type"])
            fit_kwargs = {}
            if eval_set:
                fit_kwargs["eval_set"] = [(eval_set[0], eval_set[1][name])]
                fit_kwargs["callbacks"] = [__import__("lightgbm").early_stopping(50, verbose=False)]
            booster.fit(X, y_dict[name], **fit_kwargs)
            self.models[name] = booster
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> Dict[str, np.ndarray]:
        results = {}
        for task in self.tasks_meta:
            name = task["name"]
            model = self.models[name]
            if task["type"] in (TaskType.BINARY, TaskType.MULTICLASS):
                results[name] = model.predict_proba(X)
            else:
                results[name] = model.predict(X)
        return results

    def save(self, dir_path: str) -> None:
        import pickle
        from pathlib import Path
        p = Path(dir_path)
        p.mkdir(parents=True, exist_ok=True)
        for name, model in self.models.items():
            with open(p / f"{name}.pkl", "wb") as f:
                pickle.dump(model, f)

    @classmethod
    def load(cls, dir_path: str, config: LGBMConfig, tasks_meta: list[dict]) -> "LGBMModel":
        import pickle
        from pathlib import Path
        instance = cls(config, tasks_meta)
        p = Path(dir_path)
        for task in tasks_meta:
            with open(p / f"{task['name']}.pkl", "rb") as f:
                instance.models[task["name"]] = pickle.load(f)
        return instance
