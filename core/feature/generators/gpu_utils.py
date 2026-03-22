"""
GPU utility functions for feature generators.

Provides device detection, adaptive batch sizing, tensor/numpy conversion,
and an OOM-retry decorator that gracefully degrades when GPU memory is
insufficient.

All torch imports are lazy so that modules importing this file still work
in environments where PyTorch is not installed (numpy-only fallback).

Usage::

    from core.feature.generators.gpu_utils import get_device, ensure_numpy

    device = get_device()
    arr = ensure_numpy(some_tensor)
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy torch import
# ---------------------------------------------------------------------------
try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def get_device(prefer_gpu: bool = True) -> Any:
    """Return the best available device.

    Returns a ``torch.device`` when torch is installed, or the string
    ``"cpu"`` otherwise.

    Parameters
    ----------
    prefer_gpu : bool
        If *True* (default), return ``torch.device("cuda")`` when CUDA
        is available.

    Returns
    -------
    torch.device or str
        ``torch.device("cuda")`` / ``torch.device("cpu")`` when torch
        is installed, ``"cpu"`` otherwise.
    """
    if torch is None:
        return "cpu"
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_device_description(device: Any) -> str:
    """Return a human-readable description of the device.

    Example: ``"cuda:0 (NVIDIA T4, 15.8GB)"`` or ``"cpu"``.
    """
    device_str = str(device)
    if torch is None or not device_str.startswith("cuda"):
        return device_str
    try:
        idx = int(device_str.split(":")[-1]) if ":" in device_str else 0
        name = torch.cuda.get_device_name(idx)
        mem_gb = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
        return f"{device_str} ({name}, {mem_gb:.1f}GB)"
    except Exception:
        return device_str


def get_gpu_memory() -> int:
    """Return total GPU memory in bytes, or ``0`` if unavailable.

    Returns
    -------
    int
        Total GPU memory in bytes for the current CUDA device, or ``0``
        if CUDA is unavailable or ``torch`` is not installed.
    """
    if torch is None or not torch.cuda.is_available():
        return 0
    try:
        return torch.cuda.get_device_properties(0).total_memory
    except Exception:
        return 0


def get_gpu_memory_mb() -> float:
    """Return total GPU memory in megabytes, or 0 if unavailable."""
    return get_gpu_memory() / (1024 ** 2) if get_gpu_memory() > 0 else 0.0


def get_gpu_free_memory_mb() -> float:
    """Return *free* GPU memory in megabytes, or 0 if unavailable."""
    if torch is None or not torch.cuda.is_available():
        return 0.0
    try:
        free, _ = torch.cuda.mem_get_info(0)
        return free / (1024 ** 2)
    except Exception:
        return 0.0


def log_device_info() -> None:
    """Log information about the available compute device.

    Logs GPU name, memory, and CUDA version when a GPU is available.
    Logs a CPU-only message otherwise.  Safe to call when ``torch`` is
    not installed.
    """
    if torch is None:
        logger.info("torch is not installed -- running on CPU only")
        return

    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        logger.info(
            "GPU detected: %s (%.1f GB, CUDA %s, device=%d)",
            props.name,
            props.total_memory / (1024 ** 3),
            torch.version.cuda or "unknown",
            idx,
        )
    else:
        logger.info(
            "No CUDA GPU detected -- running on CPU (torch %s)",
            torch.__version__,
        )


# ---------------------------------------------------------------------------
# Adaptive batch sizing
# ---------------------------------------------------------------------------

def adaptive_batch_size(
    total_samples: int,
    base_batch: int = 2048,
    gpu_memory: Optional[int] = None,
    bytes_per_sample: int = 4096,
    memory_fraction: float = 0.5,
) -> int:
    """Compute a batch size that fits comfortably in GPU memory.

    When GPU memory is available, scales the batch size up to use
    ``memory_fraction`` of total GPU memory.  When no GPU is available
    (or ``torch`` is not installed), returns ``base_batch``.

    Parameters
    ----------
    total_samples : int
        Total number of items to process.
    base_batch : int
        Preferred batch size when memory is plentiful.
    gpu_memory : int or None
        Override GPU memory in bytes; auto-detected when *None*.
    bytes_per_sample : int
        Estimated bytes per sample (including intermediate tensors).
    memory_fraction : float
        Fraction of GPU memory to target (0.0 -- 1.0).

    Returns
    -------
    int
        Batch size, at least 1 and at most *total_samples*.
    """
    if gpu_memory is None:
        gpu_memory = get_gpu_memory()

    if gpu_memory <= 0 or bytes_per_sample <= 0:
        return min(base_batch, max(total_samples, 1))

    usable_memory = int(gpu_memory * memory_fraction)
    computed_batch = max(usable_memory // bytes_per_sample, 1)

    # Clamp between base_batch and total_samples
    batch = max(computed_batch, base_batch)
    batch = min(batch, max(total_samples, 1))

    logger.debug(
        "Adaptive batch size: %d (gpu_mem=%.1f GB, usable=%.1f GB, "
        "bytes_per_sample=%d, total_samples=%d)",
        batch,
        gpu_memory / (1024 ** 3),
        usable_memory / (1024 ** 3),
        bytes_per_sample,
        total_samples,
    )
    return batch


# ---------------------------------------------------------------------------
# Tensor / numpy conversion
# ---------------------------------------------------------------------------

def ensure_numpy(x: Any) -> np.ndarray:
    """Convert *x* to a numpy array regardless of input type.

    Handles:
    - ``numpy.ndarray`` -- returned as-is.
    - ``torch.Tensor`` -- detached, moved to CPU, converted to numpy.
    - Any other type -- passed through ``np.asarray()``.

    Parameters
    ----------
    x : array-like or torch.Tensor
        The value to convert.

    Returns
    -------
    numpy.ndarray
    """
    if isinstance(x, np.ndarray):
        return x
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def ensure_tensor(
    x: Any,
    device: Optional[Any] = None,
    dtype: Optional[Any] = None,
) -> Any:
    """Convert *x* to a :class:`torch.Tensor` on *device*.

    Handles:
    - ``torch.Tensor`` -- moved to *device* if specified.
    - ``numpy.ndarray`` -- converted via ``torch.as_tensor()``.
    - Other types -- passed through ``torch.tensor()``.

    Returns the input unchanged if torch is unavailable.

    Parameters
    ----------
    x : array-like, numpy.ndarray, or torch.Tensor
        The value to convert.
    device : torch.device or str, optional
        Target device.  Defaults to the result of :func:`get_device`.
    dtype : torch.dtype, optional
        Target dtype (e.g. ``torch.float32``).

    Returns
    -------
    torch.Tensor

    Raises
    ------
    ImportError
        If ``torch`` is not installed (returns input unchanged).
    """
    if torch is None:
        return x
    if device is None:
        device = get_device()
    if dtype is None:
        dtype = torch.float32
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# OOM-retry decorator
# ---------------------------------------------------------------------------

def oom_retry(
    max_retries: int = 3,
    batch_arg: str = "batch_size",
    min_batch: int = 64,
) -> Callable:
    """Decorator that retries a function on CUDA OOM, halving the batch size.

    The decorated function **must** accept a keyword argument named
    *batch_arg* (default ``"batch_size"``).  On each OOM the batch is
    halved and the CUDA cache is cleared.

    Parameters
    ----------
    max_retries : int
        Maximum number of retries before raising.
    batch_arg : str
        Name of the batch-size keyword argument.
    min_batch : int
        Stop retrying when the batch drops below this value.
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_batch = kwargs.get(batch_arg)
            if current_batch is None:
                # No batch arg -- just call directly
                return fn(*args, **kwargs)

            for attempt in range(max_retries + 1):
                try:
                    kwargs[batch_arg] = current_batch
                    return fn(*args, **kwargs)
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower() and torch is not None:
                        torch.cuda.empty_cache()
                        current_batch = max(min_batch, current_batch // 2)
                        logger.warning(
                            "OOM on attempt %d/%d, reducing %s to %d",
                            attempt + 1,
                            max_retries + 1,
                            batch_arg,
                            current_batch,
                        )
                        if attempt == max_retries:
                            raise
                    else:
                        raise

        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

class StepTimer:
    """Context manager that logs elapsed wall time for a named step.

    Usage::

        with StepTimer("build_graph", logger):
            ...
    """

    def __init__(self, step_name: str, log: logging.Logger = logger):
        self.step_name = step_name
        self.log = log
        self.elapsed: float = 0.0

    def __enter__(self) -> "StepTimer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.elapsed = time.perf_counter() - self.start
        self.log.info("[%s] completed in %.2fs", self.step_name, self.elapsed)


# ---------------------------------------------------------------------------
# CuPy / cuDF / cuML availability checks
# ---------------------------------------------------------------------------

_cupy_available: Optional[bool] = None
_cudf_available: Optional[bool] = None
_cuml_available: Optional[bool] = None
_duckdb_available: Optional[bool] = None


def has_cupy() -> bool:
    """Check whether CuPy is importable and a CUDA GPU is accessible.

    Result is cached after the first call.  Returns False if CuPy is
    installed but no CUDA device is found.
    """
    global _cupy_available
    if _cupy_available is not None:
        return _cupy_available
    try:
        import cupy as cp  # noqa: F401
        # Verify a device is actually usable
        cp.cuda.Device(0).compute_capability
        _cupy_available = True
        logger.info("CuPy detected: GPU-accelerated array operations available")
    except Exception:
        _cupy_available = False
    return _cupy_available


def has_cudf() -> bool:
    """Check whether cuDF (RAPIDS) is importable.

    Result is cached after the first call.
    """
    global _cudf_available
    if _cudf_available is not None:
        return _cudf_available
    try:
        import cudf  # noqa: F401
        _cudf_available = True
        logger.info("cuDF detected: GPU-accelerated DataFrame operations available")
    except Exception:
        _cudf_available = False
    return _cudf_available


def has_cuml() -> bool:
    """Check whether cuML (RAPIDS) is importable.

    Result is cached after the first call.
    """
    global _cuml_available
    if _cuml_available is not None:
        return _cuml_available
    try:
        import cuml  # noqa: F401
        _cuml_available = True
        logger.info("cuML detected: GPU-accelerated ML estimators available")
    except Exception:
        _cuml_available = False
    return _cuml_available


def has_duckdb() -> bool:
    """Check whether DuckDB is importable.

    Result is cached after the first call.
    """
    global _duckdb_available
    if _duckdb_available is not None:
        return _duckdb_available
    try:
        import duckdb  # noqa: F401
        _duckdb_available = True
        logger.info("DuckDB detected: analytical query engine available")
    except Exception:
        _duckdb_available = False
    return _duckdb_available


def get_dataframe_backend() -> str:
    """Determine the best available DataFrame backend.

    Checks availability in order of preference and returns one of:
    - ``"cudf"`` -- GPU-accelerated RAPIDS DataFrames
    - ``"duckdb"`` -- analytical SQL engine with vectorized execution
    - ``"pandas"`` -- standard CPU DataFrames (always available)

    Returns
    -------
    str
        One of ``"cudf"``, ``"duckdb"``, or ``"pandas"``.
    """
    if has_cudf():
        return "cudf"
    if has_duckdb():
        return "duckdb"
    return "pandas"


def cupy_pairwise_distances(X: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distance matrix on GPU using CuPy.

    This is a reusable utility for any generator that needs O(n^2) pairwise
    distances.  GPU parallelisation gives ~50x speedup for n > 5000.

    Parameters
    ----------
    X : np.ndarray
        Input matrix of shape ``(n_samples, n_features)``.

    Returns
    -------
    np.ndarray
        Distance matrix of shape ``(n_samples, n_samples)`` on CPU.

    Raises
    ------
    ImportError
        If CuPy is not available.
    RuntimeError
        If the GPU computation fails (caller should fall back to CPU).
    """
    import cupy as cp

    X_gpu = cp.asarray(X)
    # Efficient pairwise distance using the identity:
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a . b
    sq_norms = cp.sum(X_gpu ** 2, axis=1)
    dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2.0 * X_gpu @ X_gpu.T
    cp.clip(dist_sq, 0.0, None, out=dist_sq)
    dists = cp.sqrt(dist_sq)
    return cp.asnumpy(dists)
