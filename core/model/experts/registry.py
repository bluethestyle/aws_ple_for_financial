"""
Expert Registry -- factory pattern for dynamic expert creation.

Usage::

    # Registration (typically at module import time via decorator)
    @ExpertRegistry.register("deepfm")
    class DeepFMExpert(AbstractExpert):
        ...

    # Creation
    expert = ExpertRegistry.create("deepfm", input_dim=128, config={...})

    # Bulk creation from a config dict
    experts = ExpertRegistry.create_from_config(shared_experts_cfg)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Type

import torch.nn as nn

from .base import AbstractExpert

logger = logging.getLogger(__name__)


class ExpertRegistry:
    """
    Singleton registry that maps string names to ``AbstractExpert`` sub-classes.

    Thread-safety note: registration happens at import time (module-level
    decorators) before any worker threads start, so no locking is needed.
    """

    _registry: Dict[str, Type[AbstractExpert]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, name: str):
        """
        Class decorator that registers an expert class under *name*.

        Example::

            @ExpertRegistry.register("mlp")
            class MLPExpert(AbstractExpert):
                ...
        """
        def decorator(expert_cls: Type[AbstractExpert]):
            if name in cls._registry:
                logger.warning(
                    "Overwriting existing expert registration: %s (%s -> %s)",
                    name, cls._registry[name].__name__, expert_cls.__name__,
                )
            cls._registry[name] = expert_cls
            logger.debug("Expert registered: %s -> %s", name, expert_cls.__name__)
            return expert_cls
        return decorator

    # ------------------------------------------------------------------
    # Instantiation
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        name: str,
        input_dim: int,
        config: Dict[str, Any],
    ) -> AbstractExpert:
        """
        Build a single expert by *name*.

        Parameters
        ----------
        name : str
            Registered expert name (e.g. ``"deepfm"``).
        input_dim : int
            Dimensionality of the input features.
        config : dict
            Expert-specific configuration forwarded to ``__init__``.

        Returns
        -------
        AbstractExpert
            A fully initialised expert module.
        """
        if name not in cls._registry:
            raise ValueError(
                f"Unknown expert: {name!r}. "
                f"Available: {cls.list_available()}"
            )
        expert_cls = cls._registry[name]

        # Strip meta-keys that are not constructor arguments
        clean_cfg = {
            k: v for k, v in config.items()
            if k not in ("enabled", "type", "name")
        }

        try:
            expert = expert_cls(input_dim=input_dim, config=clean_cfg)
            logger.info(
                "Expert created: %s (class=%s, params=%s)",
                name, expert_cls.__name__, f"{expert.count_parameters():,}",
            )
            return expert
        except Exception:
            logger.exception("Failed to create expert: %s", name)
            raise

    @classmethod
    def create_from_config(
        cls,
        experts_config: Dict[str, Dict[str, Any]],
        default_input_dim: int,
    ) -> nn.ModuleDict:
        """
        Create multiple experts from a config dict.

        Parameters
        ----------
        experts_config : dict
            ``{expert_name: {type, enabled, input_dim, ...}, ...}``
        default_input_dim : int
            Fallback ``input_dim`` if not specified per-expert.

        Returns
        -------
        nn.ModuleDict
            ``{name: AbstractExpert}`` for all enabled experts.
        """
        experts = nn.ModuleDict()

        for name, ecfg in experts_config.items():
            if not isinstance(ecfg, dict):
                continue
            if not ecfg.get("enabled", True):
                logger.info("Expert disabled, skipping: %s", name)
                continue

            expert_type = ecfg.get("type", name)
            if not cls.is_registered(expert_type):
                logger.warning(
                    "Unregistered expert type %r for %r, skipping.",
                    expert_type, name,
                )
                continue

            dim = ecfg.get("input_dim", default_input_dim)
            experts[name] = cls.create(expert_type, input_dim=dim, config=ecfg)

        logger.info("Created experts: %s", list(experts.keys()))
        return experts

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @classmethod
    def list_available(cls) -> List[str]:
        """Return names of all registered experts."""
        return sorted(cls._registry.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        return name in cls._registry

    @classmethod
    def get_class(cls, name: str) -> Type[AbstractExpert]:
        if name not in cls._registry:
            raise ValueError(
                f"Unknown expert: {name!r}. "
                f"Available: {cls.list_available()}"
            )
        return cls._registry[name]


def get_total_expert_output_dim(experts: nn.ModuleDict) -> int:
    """Sum of ``output_dim`` across all experts in a ``ModuleDict``."""
    total = 0
    for name, expert in experts.items():
        if hasattr(expert, "output_dim"):
            total += expert.output_dim
        else:
            logger.warning(
                "Expert %r has no output_dim attribute; assuming 64.", name,
            )
            total += 64
    return total
