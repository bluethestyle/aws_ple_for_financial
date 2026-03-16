"""Model architecture plugin registry."""
from typing import Type


class ModelRegistry:
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


# PLE is always registered (torch-only)
from .ple import PLEModel
ModelRegistry.register("ple")(PLEModel)

# LGBM is registered lazily to avoid pandas import at module load time
try:
    from .lgbm import LGBMModel
    ModelRegistry.register("lgbm")(LGBMModel)
except ImportError:
    pass
