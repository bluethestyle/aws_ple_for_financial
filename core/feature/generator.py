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

Pool / Basket pattern
---------------------
Mirrors the 3-tier expert selection architecture from
``core.model.ple.experts``:

1. **Generator Pool** -- all generators registered via
   ``@FeatureGeneratorRegistry.register``.
2. **Generator Basket** -- config-defined subset selected from the pool
   for a specific pipeline run (driven by ``FeatureGroupConfig``).
3. **Pipeline Execution** -- runtime invocation of generators in the
   basket via :class:`FeatureGroupPipeline`.

Example::

    @FeatureGeneratorRegistry.register("tda_extractor")
    class TDAFeatureGenerator(AbstractFeatureGenerator):
        ...

    gen = FeatureGeneratorRegistry.create("tda_extractor", dim=8)
"""

from __future__ import annotations

import importlib
import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import numpy as np

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
        * ``fit``            -- learn any internal state from training data.
        * ``generate``       -- produce a DataFrame of **only** new columns.
        * ``output_dim``     -- number of new columns produced.
        * ``output_columns`` -- explicit list of new column names.

    Subclasses may override:
        * ``supports_gpu``       -- whether this generator can use GPU (default ``False``).
        * ``required_libraries`` -- heavy Python packages needed at runtime.
        * ``container_image``    -- ECR image URI for container execution.
        * ``estimated_output_dim`` -- classmethod for pre-instantiation dim estimation.

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

    # -- GPU / device support ------------------------------------------

    @property
    def supports_gpu(self) -> bool:
        """Whether this generator can utilise GPU acceleration.

        Override in subclasses that leverage CUDA (e.g. via torch,
        cuML, mamba_ssm).  Defaults to ``False``.
        """
        return False

    @property
    def required_libraries(self) -> List[str]:
        """Python packages required at runtime.

        Return a list of importable module names (e.g.
        ``["ripser", "persim"]``).  These are checked by
        :meth:`check_dependencies` and used by the registry for
        introspection.  Defaults to an empty list.
        """
        return []

    @property
    def container_image(self) -> str:
        """ECR image URI this generator should run inside.

        Return an empty string if the generator can run locally without
        a specialised container.  Override in subclasses that need
        specific system-level dependencies.
        """
        return ""

    @property
    def device(self) -> str:
        """Auto-detect the best device for this generator.

        Returns ``"cuda"`` if the generator supports GPU and CUDA is
        available; otherwise ``"cpu"``.  Uses lazy import of ``torch``
        to avoid hard dependency.
        """
        if not self.supports_gpu:
            return "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    def check_dependencies(self) -> bool:
        """Validate that all ``required_libraries`` are importable.

        Returns
        -------
        bool
            ``True`` if all dependencies are available, ``False``
            otherwise.  Logs warnings for missing packages.
        """
        all_ok = True
        for lib in self.required_libraries:
            try:
                importlib.import_module(lib)
            except ImportError:
                logger.warning(
                    "Generator '%s' requires '%s' which is not installed.",
                    self.name, lib,
                )
                all_ok = False
        return all_ok

    @classmethod
    def estimated_output_dim(cls, config: Dict[str, Any]) -> int:
        """Estimate the number of output columns from a config dict.

        This classmethod allows callers to estimate output dimensionality
        *before* instantiating the generator (useful for PLE model
        construction and dimension planning).

        Parameters
        ----------
        config : dict
            Generator constructor parameters.

        Returns
        -------
        int
            Estimated number of output columns.  The default
            implementation returns ``0`` (unknown); subclasses should
            override with a meaningful estimate.
        """
        return 0

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

    # -- Input conversion utility --------------------------------------

    def _input_to_numpy(
        self,
        df: Any,
        columns: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """Convert any input DataFrame to ``{col: np.ndarray}`` dict.

        Generators often need raw numpy arrays for numerical computation
        (e.g. GMM, TDA, HMM).  This helper abstracts over the input type
        so generators don't need to know whether they received a cuDF,
        DuckDB, or pandas DataFrame.

        Dispatch order (fastest first):

        * **cuDF** -- ``df[col].values.get()`` (GPU -> CPU).
        * **DuckDB relation** -- ``.fetchnumpy()``.
        * **pandas** -- ``df[col].values``.

        Parameters
        ----------
        df : DataFrame
            Input data (cuDF, pandas, DuckDB relation, or similar).
        columns : list[str], optional
            Subset of columns to extract.  ``None`` extracts all.

        Returns
        -------
        dict[str, numpy.ndarray]
            Column name -> 1-D numpy array mapping.
        """
        from core.data.dataframe import df_backend
        return df_backend.to_numpy_dict(df, columns)

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
            "supports_gpu": self.supports_gpu,
            "device": self.device,
            "required_libraries": self.required_libraries,
            "container_image": self.container_image,
        }

    # -- Repr ----------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{type(self).__name__}("
            f"output_dim={self.output_dim}, "
            f"fitted={self._fitted}, "
            f"device={self.device!r})"
        )


# ======================================================================
# Generator Registry (Pool)
# ======================================================================


class FeatureGeneratorRegistry:
    """Plugin registry for feature generators (the Generator Pool).

    Mirrors the design of :class:`ExpertRegistry` in
    ``core.model.experts.registry`` but is specialised for
    :class:`AbstractFeatureGenerator` subclasses.

    The registry acts as the **Pool** tier in the Pool/Basket pattern:
    all available generators are registered here; downstream config
    selects a subset (the "basket") for a specific pipeline run.

    Usage::

        @FeatureGeneratorRegistry.register("tda_extractor")
        class TDAFeatureGenerator(AbstractFeatureGenerator):
            ...

        gen = FeatureGeneratorRegistry.create("tda_extractor", dim=8)

    Thread-safety note: registration happens at import time (module-level
    decorators) before any worker threads start, so no locking is needed.
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
            logger.debug(
                "Generator registered: %s -> %s", name, gen_cls.__name__,
            )
            return gen_cls

        return decorator

    # -- Instantiation -------------------------------------------------

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> AbstractFeatureGenerator:
        """Instantiate a generator by its registered *name*.

        This is the primary factory method (equivalent to
        ``ExpertRegistry.create``).

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
                f"Available: {cls.list_available()}"
            )
        gen_cls = cls._registry[name]
        try:
            instance = gen_cls(**kwargs)
            logger.debug(
                "Generator created: %s (class=%s)",
                name, gen_cls.__name__,
            )
            return instance
        except Exception:
            logger.exception("Failed to create generator: %s", name)
            raise

    @classmethod
    def build(cls, name: str, **kwargs: Any) -> AbstractFeatureGenerator:
        """Alias for :meth:`create` (backward compatibility).

        Parameters
        ----------
        name : str
            Registered generator name.
        **kwargs
            Forwarded to the generator constructor.

        Returns
        -------
        AbstractFeatureGenerator
        """
        return cls.create(name, **kwargs)

    # -- Lookup / Introspection ----------------------------------------

    @classmethod
    def get_class(cls, name: str) -> Type[AbstractFeatureGenerator]:
        """Return the generator class (not an instance)."""
        if name not in cls._registry:
            raise KeyError(
                f"Unknown generator '{name}'. "
                f"Available: {cls.list_available()}"
            )
        return cls._registry[name]

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check whether *name* is registered."""
        return name in cls._registry

    @classmethod
    def list_available(cls) -> List[str]:
        """Return a sorted list of all registered generator names."""
        return sorted(cls._registry.keys())

    @classmethod
    def list_registered(cls) -> List[str]:
        """Alias for :meth:`list_available` (backward compatibility)."""
        return cls.list_available()

    @classmethod
    def list_gpu_capable(cls) -> List[str]:
        """Return names of generators that support GPU acceleration.

        This method instantiates each generator class with no arguments
        to query the ``supports_gpu`` property.  Generators that fail
        to instantiate without arguments are skipped.

        Returns
        -------
        list[str]
            Sorted list of generator names with ``supports_gpu=True``.
        """
        gpu_names: List[str] = []
        for name, gen_cls in cls._registry.items():
            try:
                # Read supports_gpu directly from the class dict to avoid
                # instantiation.  Class-level bool attributes are preferred
                # over instance-level assignments for GPU capability.
                val = gen_cls.__dict__.get("supports_gpu")
                if val is True:
                    gpu_names.append(name)
            except Exception:
                # If we cannot determine GPU support, skip silently.
                pass
        return sorted(gpu_names)

    @classmethod
    def get_info(cls, name: str) -> Dict[str, Any]:
        """Return metadata for a registered generator.

        Parameters
        ----------
        name : str
            Registered generator name.

        Returns
        -------
        dict
            Keys: ``name``, ``class``, ``description``, ``tags``,
            ``supports_gpu``, ``required_libraries``,
            ``container_image``.
        """
        if name not in cls._registry:
            raise KeyError(
                f"Unknown generator '{name}'. "
                f"Available: {cls.list_available()}"
            )
        meta = dict(cls._metadata.get(name, {}))
        meta["name"] = name

        # Extract runtime info from class-level attributes where possible
        gen_cls = cls._registry[name]
        try:
            # Prefer class-level attributes to avoid instantiation
            val = gen_cls.__dict__.get("supports_gpu")
            meta["supports_gpu"] = val if isinstance(val, bool) else False

            libs = gen_cls.__dict__.get("required_libraries")
            meta["required_libraries"] = libs if isinstance(libs, list) else []

            img = gen_cls.__dict__.get("container_image")
            meta["container_image"] = img if isinstance(img, str) else ""
        except Exception:
            meta["supports_gpu"] = False
            meta["required_libraries"] = []
            meta["container_image"] = ""

        return meta

    @classmethod
    def list_all_info(cls) -> List[Dict[str, Any]]:
        """Return metadata dicts for all registered generators.

        Useful for summary display or dashboard integration.

        Returns
        -------
        list[dict]
            One info dict per registered generator, sorted by name.
        """
        return [cls.get_info(name) for name in cls.list_available()]

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
        """Alias for :meth:`get_info` (backward compatibility)."""
        return cls.get_info(name)

    @classmethod
    def clear(cls) -> None:
        """Remove all registrations (useful in tests)."""
        cls._registry.clear()
        cls._metadata.clear()
