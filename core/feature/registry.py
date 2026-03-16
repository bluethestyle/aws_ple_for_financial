from typing import Type
from .base import AbstractFeatureTransformer


class FeatureRegistry:
    """
    피처 트랜스포머 플러그인 레지스트리.

    Example:
        @FeatureRegistry.register("log_scale")
        class LogScaler(AbstractFeatureTransformer):
            ...

        transformer = FeatureRegistry.build("log_scale", cols=["price"])
    """

    _registry: dict[str, Type[AbstractFeatureTransformer]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(transformer_cls: Type[AbstractFeatureTransformer]):
            cls._registry[name] = transformer_cls
            return transformer_cls
        return decorator

    @classmethod
    def build(cls, name: str, **kwargs) -> AbstractFeatureTransformer:
        if name not in cls._registry:
            raise KeyError(f"Unknown transformer '{name}'. Registered: {list(cls._registry.keys())}")
        return cls._registry[name](**kwargs)

    @classmethod
    def list_registered(cls) -> list[str]:
        return list(cls._registry.keys())
