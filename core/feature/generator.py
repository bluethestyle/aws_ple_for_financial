"""
Feature Generator abstraction -- creates NEW features from raw data.

Unlike :class:`AbstractFeatureTransformer` (which modifies existing columns),
generators produce entirely new columns that did not exist in the input
DataFrame.  Examples include topological data analysis (TDA), hidden Markov
model state estimation, graph embeddings, and multidisciplinary features
drawn from fields outside machine learning.

Generators follow the same ``fit`` / ``generate`` lifecycle as transformers
but return a DataFrame containing **only** the newly created columns.  The
:class:`FeatureGroupPipeline` is responsible for concatenating generator
output with the rest of the feature matrix.

Registration
------------
All concrete generators are registered via the
``@FeatureGeneratorRegistry.register("name")`` decorator so they can be
referenced by name in :class:`FeatureGroupConfig` definitions.

Example::

    @FeatureGeneratorRegistry.register("tda_extractor")
    class TDAFeatureGenerator(AbstractFeatureGenerator):
        ...

    gen = FeatureGeneratorRegistry.build("tda_extractor", dim=8)
"""

from __future__ import annotations

import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


# ======================================================================
# Abstract base
# ======================================================================


class AbstractFeatureGenerator(ABC):
    """Base class for feature GENERATORS.

    A generator creates entirely new feature columns from raw data.
    This is fundamentally different from a transformer which modifies
    existing columns in place.

    Subclasses must implement:
        * ``fit``      -- learn any internal state from training data.
        * ``generate`` -- produce a DataFrame of **only** new columns.
        * ``output_dim``     -- number of new columns produced.
        * ``output_columns`` -- explicit list of new column names.

    Attributes
    ----------
    name : str
        Registry name (set automatically by the decorator).
    fitted : bool
        Whether ``fit()`` has been called.
    """

    name: str = "base_generator"

    def __init__(self, **kwargs: Any) -> None:
        self._fitted = False
        self._extra_params = kwargs

    @property
    def fitted(self) -> bool:
        return self._fitted

    # -- Core API ------------------------------------------------------

    @abstractmethod
    def fit(self, df: Any, **context: Any) -> "AbstractFeatureGenerator":
        """Learn internal parameters from *df*.

        Parameters
        ----------
        df : DataFrame
            Training data (pandas, cuDF, or any backend-native type).
            Concrete generators should use ``df_backend.to_pandas(df)``
            if they require pandas-specific APIs.
        **context
            Arbitrary keyword arguments that generators may use
            (e.g. ``target_col``, ``time_col``).

        Returns
        -------
        AbstractFeatureGenerator
            ``self``, for chaining.
        """
        ...

    @abstractmethod
    def generate(self, df: Any, **context: Any) -> Any:
        """Generate new feature columns from *df*.

        The returned DataFrame must have the **same row count** as *df*
        and contain **only** the newly generated columns (not the
        original columns).

        Parameters
        ----------
        df : DataFrame
            Input data (pandas, cuDF, or any backend-native type).
        **context
            Same keyword arguments accepted by ``fit``.

        Returns
        -------
        DataFrame
            DataFrame with only the new feature columns, indexed
            identically to *df*.
        """
        ...

    def fit_generate(self, df: Any, **context: Any) -> Any:
        """Convenience: ``fit(df).generate(df)``."""
        return self.fit(df, **context).generate(df, **context)

    # -- Output description --------------------------------------------

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Number of new feature columns this generator produces."""
        ...

    @property
    @abstractmethod
    def output_columns(self) -> List[str]:
        """Explicit list of generated column names."""
        ...

    # -- Serialisation -------------------------------------------------

    def save(self, path: str) -> None:
        """Persist the generator to *path* via pickle."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.debug("Saved %s to %s", type(self).__name__, path)

    @classmethod
    def load(cls, path: str) -> "AbstractFeatureGenerator":
        """Load a generator from *path*."""
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, AbstractFeatureGenerator):
            raise TypeError(
                f"Expected AbstractFeatureGenerator, got {type(obj).__name__}"
            )
        return obj

    def get_params(self) -> Dict[str, Any]:
        """Return generator parameters as a dictionary."""
        return {
            "name": self.name,
            "fitted": self._fitted,
            "output_dim": self.output_dim,
            "output_columns": self.output_columns,
        }

    # -- Repr ----------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{type(self).__name__}("
            f"output_dim={self.output_dim}, "
            f"fitted={self._fitted})"
        )


# ======================================================================
# Generator Registry
# ======================================================================


class FeatureGeneratorRegistry:
    """Plugin registry for feature generators.

    Mirrors the design of :class:`FeatureRegistry` but is specialised
    for :class:`AbstractFeatureGenerator` subclasses.

    Usage::

        @FeatureGeneratorRegistry.register("tda_extractor")
        class TDAFeatureGenerator(AbstractFeatureGenerator):
            ...

        gen = FeatureGeneratorRegistry.build("tda_extractor", dim=8)
    """

    _registry: Dict[str, Type[AbstractFeatureGenerator]] = {}
    _metadata: Dict[str, Dict[str, Any]] = {}

    # -- Registration --------------------------------------------------

    @classmethod
    def register(
        cls,
        name: str,
        *,
        description: str = "",
        tags: Optional[List[str]] = None,
    ):
        """Decorator that registers a generator class.

        Parameters
        ----------
        name : str
            Unique name (used in configs and ``build()``).
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
            gen_cls: Type[AbstractFeatureGenerator],
        ) -> Type[AbstractFeatureGenerator]:
            if name in cls._registry:
                logger.warning(
                    "Overwriting existing generator '%s' (%s) with %s",
                    name,
                    cls._registry[name].__name__,
                    gen_cls.__name__,
                )
            cls._registry[name] = gen_cls
            cls._metadata[name] = {
                "class": gen_cls.__name__,
                "description": description or gen_cls.__doc__ or "",
                "tags": tags or [],
            }
            gen_cls.name = name  # type: ignore[attr-defined]
            return gen_cls

        return decorator

    # -- Lookup --------------------------------------------------------

    @classmethod
    def build(cls, name: str, **kwargs: Any) -> AbstractFeatureGenerator:
        """Instantiate a generator by its registered *name*.

        Parameters
        ----------
        name : str
            Registered generator name.
        **kwargs
            Forwarded to the generator constructor.

        Returns
        -------
        AbstractFeatureGenerator

        Raises
        ------
        KeyError
            If *name* has not been registered.
        """
        if name not in cls._registry:
            raise KeyError(
                f"Unknown generator '{name}'. "
                f"Registered: {cls.list_registered()}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def get_class(cls, name: str) -> Type[AbstractFeatureGenerator]:
        """Return the generator class (not an instance)."""
        if name not in cls._registry:
            raise KeyError(
                f"Unknown generator '{name}'. "
                f"Registered: {cls.list_registered()}"
            )
        return cls._registry[name]

    @classmethod
    def list_registered(cls) -> List[str]:
        """Return a sorted list of all registered generator names."""
        return sorted(cls._registry.keys())

    @classmethod
    def find_by_tag(cls, tag: str) -> List[str]:
        """Return generator names that carry *tag*."""
        return sorted(
            name
            for name, meta in cls._metadata.items()
            if tag in meta.get("tags", [])
        )

    @classmethod
    def info(cls, name: str) -> Dict[str, Any]:
        """Return metadata for a registered generator."""
        if name not in cls._metadata:
            raise KeyError(f"No metadata for '{name}'")
        return dict(cls._metadata[name])

    @classmethod
    def clear(cls) -> None:
        """Remove all registrations (useful in tests)."""
        cls._registry.clear()
        cls._metadata.clear()
