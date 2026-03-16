"""
Feature Registry — plugin discovery and transformer catalogue.

Transformers are registered via the ``@FeatureRegistry.register("name")``
decorator.  The registry provides factory methods to instantiate
transformers by name and list all available plugins.

Example::

    @FeatureRegistry.register("standard_scaler")
    class StandardScaler(AbstractFeatureTransformer):
        ...

    scaler = FeatureRegistry.build("standard_scaler", columns=["amount"])
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Type

from .base import AbstractFeatureTransformer

logger = logging.getLogger(__name__)


class FeatureRegistry:
    """
    Central catalogue for feature transformer plugins.

    All registration state is stored at the **class** level so a single
    ``@FeatureRegistry.register(...)`` decorator is sufficient regardless
    of how many ``FeatureRegistry`` instances exist.

    Methods
    -------
    register(name)
        Class-method decorator that adds a transformer class.
    build(name, **kwargs)
        Instantiate a registered transformer by name.
    list_registered()
        Return all registered names.
    get_class(name)
        Return the class without instantiation.
    """

    _registry: Dict[str, Type[AbstractFeatureTransformer]] = {}
    _metadata: Dict[str, Dict[str, Any]] = {}

    # ── Registration ───────────────────────────────────────────────────

    @classmethod
    def register(
        cls,
        name: str,
        *,
        description: str = "",
        tags: Optional[List[str]] = None,
    ):
        """Decorator that registers a transformer class.

        Parameters
        ----------
        name : str
            Unique name for the transformer (used in configs and
            ``build()``).
        description : str, optional
            Short human-readable description.
        tags : list[str], optional
            Searchable tags (e.g. ``["numeric", "scaler"]``).

        Returns
        -------
        Callable
            The original class, unmodified.
        """

        def decorator(
            transformer_cls: Type[AbstractFeatureTransformer],
        ) -> Type[AbstractFeatureTransformer]:
            if name in cls._registry:
                logger.warning(
                    "Overwriting existing transformer '%s' (%s) with %s",
                    name,
                    cls._registry[name].__name__,
                    transformer_cls.__name__,
                )
            cls._registry[name] = transformer_cls
            cls._metadata[name] = {
                "class": transformer_cls.__name__,
                "description": description or transformer_cls.__doc__ or "",
                "tags": tags or [],
            }
            # Also attach the registry name onto the class for introspection
            transformer_cls.name = name  # type: ignore[attr-defined]
            return transformer_cls

        return decorator

    # ── Lookup ─────────────────────────────────────────────────────────

    @classmethod
    def build(cls, name: str, **kwargs: Any) -> AbstractFeatureTransformer:
        """Instantiate a transformer by its registered *name*.

        Parameters
        ----------
        name : str
            Registered transformer name.
        **kwargs
            Forwarded to the transformer constructor.

        Returns
        -------
        AbstractFeatureTransformer

        Raises
        ------
        KeyError
            If *name* has not been registered.
        """
        if name not in cls._registry:
            raise KeyError(
                f"Unknown transformer '{name}'. "
                f"Registered: {cls.list_registered()}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def get_class(cls, name: str) -> Type[AbstractFeatureTransformer]:
        """Return the transformer class (not an instance)."""
        if name not in cls._registry:
            raise KeyError(
                f"Unknown transformer '{name}'. "
                f"Registered: {cls.list_registered()}"
            )
        return cls._registry[name]

    @classmethod
    def list_registered(cls) -> List[str]:
        """Return a sorted list of all registered transformer names."""
        return sorted(cls._registry.keys())

    @classmethod
    def find_by_tag(cls, tag: str) -> List[str]:
        """Return transformer names that carry *tag*."""
        return sorted(
            name
            for name, meta in cls._metadata.items()
            if tag in meta.get("tags", [])
        )

    @classmethod
    def info(cls, name: str) -> Dict[str, Any]:
        """Return metadata for a registered transformer."""
        if name not in cls._metadata:
            raise KeyError(f"No metadata for '{name}'")
        return dict(cls._metadata[name])

    @classmethod
    def clear(cls) -> None:
        """Remove all registrations (useful in tests)."""
        cls._registry.clear()
        cls._metadata.clear()

    # ── Repr ───────────────────────────────────────────────────────────

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"FeatureRegistry(registered={FeatureRegistry.list_registered()})"
        )
