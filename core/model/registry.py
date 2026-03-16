from typing import Type


class ModelRegistry:
    """
    모델 아키텍처 플러그인 레지스트리.

    Example:
        @ModelRegistry.register("my_model")
        class MyModel:
            ...

        model = ModelRegistry.build("my_model", config=..., tasks=...)
    """

    _registry: dict[str, Type] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(model_cls: Type):
            cls._registry[name] = model_cls
            return model_cls
        return decorator

    @classmethod
    def build(cls, name: str, **kwargs):
        if name not in cls._registry:
            raise KeyError(f"Unknown model '{name}'. Registered: {list(cls._registry.keys())}")
        return cls._registry[name](**kwargs)

    @classmethod
    def list_registered(cls) -> list[str]:
        return list(cls._registry.keys())


# 기본 모델 등록
from .ple import PLEModel
from .lgbm import LGBMModel

ModelRegistry.register("ple")(PLEModel)
ModelRegistry.register("lgbm")(LGBMModel)
