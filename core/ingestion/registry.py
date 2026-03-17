"""
Domain Ingestor Registry -- plugin registry for domain data ingestors.

Mirrors the design of :class:`FeatureGeneratorRegistry` in
``core.feature.generator`` but is specialised for
:class:`AbstractDomainIngestor` subclasses.

The registry acts as the **Pool** tier: all available domain ingestors
are registered here; downstream config selects a subset for a specific
pipeline run.

Usage::

    @DomainRegistry.register("customer_master")
    class CustomerMasterIngestor(AbstractDomainIngestor):
        ...

    ingestor = DomainRegistry.create("customer_master", config={...})

Thread-safety note: registration happens at import time (module-level
decorators) before any worker threads start, so no locking is needed.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Type

from .base import AbstractDomainIngestor

logger = logging.getLogger(__name__)


class DomainRegistry:
    """Plugin registry for domain ingestors.

    All concrete :class:`AbstractDomainIngestor` subclasses register
    themselves via ``@DomainRegistry.register("name")`` so they can
    be looked up and instantiated by name at runtime.
    """

    _registry: Dict[str, Type[AbstractDomainIngestor]] = {}
    _metadata: Dict[str, Dict[str, Any]] = {}

    # -- Registration ------------------------------------------------------

    @classmethod
    def register(
        cls,
        name: str,
        *,
        description: str = "",
        tags: Optional[List[str]] = None,
    ):
        """Decorator that registers a domain ingestor class.

        Parameters
        ----------
        name : str
            Unique name (used in configs and ``create()``).
        description : str, optional
            Short human-readable description.
        tags : list[str], optional
            Searchable tags.

        Returns
        -------
        Callable
            The original class, unmodified.
        """

        def decorator(
            ingestor_cls: Type[AbstractDomainIngestor],
        ) -> Type[AbstractDomainIngestor]:
            if name in cls._registry:
                logger.warning(
                    "Overwriting existing ingestor '%s' (%s) with %s",
                    name,
                    cls._registry[name].__name__,
                    ingestor_cls.__name__,
                )
            cls._registry[name] = ingestor_cls
            cls._metadata[name] = {
                "class": ingestor_cls.__name__,
                "description": description or ingestor_cls.__doc__ or "",
                "tags": tags or [],
            }
            ingestor_cls.name = name  # type: ignore[attr-defined]
            logger.debug(
                "Domain ingestor registered: %s -> %s",
                name, ingestor_cls.__name__,
            )
            return ingestor_cls

        return decorator

    # -- Instantiation -----------------------------------------------------

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> AbstractDomainIngestor:
        """Instantiate an ingestor by its registered *name*.

        Parameters
        ----------
        name : str
            Registered ingestor name.
        **kwargs
            Forwarded to the ingestor constructor.

        Returns
        -------
        AbstractDomainIngestor

        Raises
        ------
        KeyError
            If *name* has not been registered.
        """
        if name not in cls._registry:
            raise KeyError(
                f"Unknown domain ingestor '{name}'. "
                f"Available: {cls.list_available()}"
            )
        ingestor_cls = cls._registry[name]
        try:
            instance = ingestor_cls(**kwargs)
            logger.debug(
                "Domain ingestor created: %s (class=%s)",
                name, ingestor_cls.__name__,
            )
            return instance
        except Exception:
            logger.exception("Failed to create domain ingestor: %s", name)
            raise

    # -- Lookup / Introspection --------------------------------------------

    @classmethod
    def get_class(cls, name: str) -> Type[AbstractDomainIngestor]:
        """Return the ingestor class (not an instance)."""
        if name not in cls._registry:
            raise KeyError(
                f"Unknown domain ingestor '{name}'. "
                f"Available: {cls.list_available()}"
            )
        return cls._registry[name]

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check whether *name* is registered."""
        return name in cls._registry

    @classmethod
    def list_available(cls) -> List[str]:
        """Return a sorted list of all registered ingestor names."""
        return sorted(cls._registry.keys())

    @classmethod
    def get_info(cls, name: str) -> Dict[str, Any]:
        """Return metadata for a registered ingestor.

        Parameters
        ----------
        name : str
            Registered ingestor name.

        Returns
        -------
        dict
            Keys: ``name``, ``class``, ``description``, ``tags``.
        """
        if name not in cls._registry:
            raise KeyError(
                f"Unknown domain ingestor '{name}'. "
                f"Available: {cls.list_available()}"
            )
        meta = dict(cls._metadata.get(name, {}))
        meta["name"] = name
        return meta

    @classmethod
    def list_all_info(cls) -> List[Dict[str, Any]]:
        """Return metadata dicts for all registered ingestors."""
        return [cls.get_info(name) for name in cls.list_available()]

    @classmethod
    def find_by_tag(cls, tag: str) -> List[str]:
        """Return ingestor names that carry *tag*."""
        return sorted(
            name
            for name, meta in cls._metadata.items()
            if tag in meta.get("tags", [])
        )

    @classmethod
    def clear(cls) -> None:
        """Remove all registrations (useful in tests)."""
        cls._registry.clear()
        cls._metadata.clear()
