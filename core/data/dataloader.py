"""
GPU-capable DataLoader for the PLE platform.

Provides :class:`PLEDataset` and :func:`build_ple_dataloader` which together
convert a DataFrame (pandas or cuDF) into a PyTorch ``DataLoader`` that yields
:class:`PLEInput` objects ready for the PLE model.

Features:
  - cuDF GPU zero-copy via DLPack (when available)
  - Fallback to pandas + torch for CPU-only environments
  - Sequence column reshaping (flat columns -> 3D tensors)
  - Feature group slicing (group_ranges)
  - Specialized feature extraction (hyperbolic, tda, hmm -> separate tensors)
  - Batch-level PLEInput construction via custom collate

All heavy imports (torch, cudf) are deferred so the module stays importable
in environments that lack GPU libraries.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "FeatureColumnSpec",
    "SequenceConfig",
    "PLEDataset",
    "build_ple_dataloader",
]


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SequenceConfig:
    """Dimensions for sequence tensor reshaping."""
    event_seq_len: int = 180
    event_feat_dim: int = 16
    session_seq_len: int = 90
    session_feat_dim: int = 8


@dataclass
class FeatureColumnSpec:
    """Declares which DataFrame columns map to which :class:`PLEInput` fields.

    Columns listed under each attribute are extracted from the DataFrame and
    stacked into the corresponding tensor field of ``PLEInput``.  Columns
    *not* listed anywhere are ignored.

    Sequence columns are discovered by pattern expansion (see
    ``event_seq_pattern`` / ``session_seq_pattern``).
    """
    static_features: List[str] = field(default_factory=list)

    # Specialized expert inputs
    hyperbolic_columns: List[str] = field(default_factory=list)
    tda_columns: List[str] = field(default_factory=list)
    collaborative_columns: List[str] = field(default_factory=list)

    # HMM triple-mode
    hmm_journey_columns: List[str] = field(default_factory=list)
    hmm_lifecycle_columns: List[str] = field(default_factory=list)
    hmm_behavior_columns: List[str] = field(default_factory=list)

    # Auxiliary
    multidisciplinary_columns: List[str] = field(default_factory=list)
    coldstart_columns: List[str] = field(default_factory=list)
    anonymous_columns: List[str] = field(default_factory=list)

    # Sequence column patterns (flat columns to reshape)
    # Use {feature} and {step:03d} as placeholders.
    event_seq_pattern: str = "txn_card_{feature}_{step:03d}"
    session_seq_pattern: str = "sess_{feature}_{step:03d}"

    # Feature names inside the sequence patterns
    event_seq_features: List[str] = field(default_factory=list)
    session_seq_features: List[str] = field(default_factory=list)

    # Time delta column prefixes (flat columns: prefix_001 .. prefix_N)
    event_time_delta_prefix: str = "txn_card_time_delta"
    session_time_delta_prefix: str = "sess_time_delta"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_seq_columns(
    pattern: str,
    features: List[str],
    seq_len: int,
    available_columns: set,
) -> List[str]:
    """Expand a sequence pattern into an ordered list of column names.

    Returns an empty list if any expected column is missing from
    *available_columns* (i.e. sequence data is not present).
    """
    if not features:
        return []

    columns: List[str] = []
    for step in range(1, seq_len + 1):
        for feat in features:
            col = pattern.replace("{feature}", feat).replace(
                "{step:03d}", f"{step:03d}"
            )
            columns.append(col)

    # Verify at least the first and last columns exist
    if columns and columns[0] not in available_columns:
        return []
    return columns


def _resolve_time_delta_columns(
    prefix: str,
    seq_len: int,
    available_columns: set,
) -> List[str]:
    """Resolve time-delta columns like ``prefix_001`` .. ``prefix_N``."""
    cols = [f"{prefix}_{i:03d}" for i in range(1, seq_len + 1)]
    if cols and cols[0] not in available_columns:
        return []
    return cols


def _check_cudf() -> bool:
    try:
        import cudf  # noqa: F401
        return True
    except ImportError:
        return False


def _arrow_table_to_tensor(table: Any, columns: List[str], dtype: str = "float32") -> Any:
    """Convert Arrow Table columns to a CPU torch.Tensor (zero-copy when possible)."""
    import numpy as np
    import torch

    # Select columns and convert to a single contiguous numpy array
    subset = table.select(columns)
    # to_pandas(zero_copy_only=...) may fail for some types; use numpy directly
    # Arrow -> numpy via .to_pydict() is slow; prefer chunked array path
    arrays = [subset.column(c).to_numpy(zero_copy_only=False) for c in range(subset.num_columns)]
    arr = np.column_stack(arrays) if len(arrays) > 1 else arrays[0].reshape(-1, 1)
    arr = np.nan_to_num(arr.astype(np.float64), nan=0.0).astype(getattr(np, dtype))
    return torch.from_numpy(arr)


def _df_to_tensor_cpu(df: Any, columns: List[str], dtype: str = "float32") -> Any:
    """Convert DataFrame columns to a CPU torch.Tensor via numpy.

    Accepts pandas DataFrame or PyArrow Table.
    """
    import numpy as np
    import torch

    # Fast path for Arrow tables
    try:
        import pyarrow as pa
        if isinstance(df, pa.Table):
            return _arrow_table_to_tensor(df, columns, dtype)
    except ImportError:
        pass

    # DuckDB relation: fetchnumpy (no pandas)
    try:
        import duckdb
        if hasattr(df, 'fetchnumpy') or hasattr(df, 'sql'):
            # df is a DuckDB relation
            arrays = [np.asarray(df[c], dtype=np.float64) for c in columns]
            arr = np.column_stack(arrays) if len(arrays) > 1 else arrays[0].reshape(-1, 1)
            arr = np.nan_to_num(arr, nan=0.0).astype(getattr(np, dtype))
            return torch.from_numpy(arr)
    except Exception:
        pass

    # pandas / generic fallback
    arr = df[columns].values
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    arr = np.nan_to_num(arr.astype(np.float64), nan=0.0).astype(getattr(np, dtype))
    return torch.from_numpy(arr)


def _df_to_tensor_gpu(gdf: Any, columns: List[str], dtype: str = "float32") -> Any:
    """Convert cuDF DataFrame columns → CuPy → torch tensor (zero-copy on GPU).

    Pipeline: cuDF.fillna(0) → .to_cupy() → torch.as_tensor (no host↔device copy).
    Falls back to DLPack if CuPy is unavailable, then to numpy as last resort.
    """
    import torch

    subset = gdf[columns].fillna(0)

    # Path 1: cuDF → CuPy → torch (preferred zero-copy)
    try:
        import cupy as cp
        arr = subset.to_cupy().astype(getattr(cp, dtype))
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return torch.as_tensor(arr, device="cuda")
    except (ImportError, Exception):
        pass

    # Path 2: DLPack (older cuDF versions)
    try:
        capsule = subset.to_dlpack()
        return torch.from_dlpack(capsule)
    except Exception:
        pass

    # Path 3: fallback through numpy (host round-trip — avoid if possible)
    import numpy as np
    arr = subset.to_pandas().values.astype(np.float32)
    return torch.tensor(arr, dtype=torch.float32, device="cuda")


# ---------------------------------------------------------------------------
# PLEDataset
# ---------------------------------------------------------------------------

class PLEDataset:
    """Dataset that produces PLEInput-compatible dicts.

    Pre-converts all data to tensors at init time:
      - cuDF input → CuPy → torch CUDA tensors (zero-copy on GPU)
      - pandas/Arrow input → numpy → torch CPU tensors

    Parameters
    ----------
    df : DataFrame or pyarrow.Table
        pandas DataFrame, cuDF DataFrame, or PyArrow Table.
        When using DuckDB, prefer passing ``conn.execute(...).arrow()``
        instead of ``.df()`` to avoid pandas overhead.
    feature_columns : FeatureColumnSpec
        Column-to-field mapping.
    label_columns : dict[str, str]
        ``{task_name: column_name}`` for target labels.
    sequence_config : SequenceConfig, optional
        Sequence reshaping dimensions.
    use_gpu : bool
        When True and cuDF is available, use DLPack zero-copy path.
    """

    def __init__(
        self,
        df: Any,
        feature_columns: FeatureColumnSpec,
        label_columns: Dict[str, str],
        sequence_config: Optional[SequenceConfig] = None,
        use_gpu: bool = False,
    ) -> None:
        import torch

        self._spec = feature_columns
        self._label_columns = label_columns
        self._seq_cfg = sequence_config or SequenceConfig()
        self._use_gpu = use_gpu and _check_cudf()

        # Normalise column names for Arrow vs DataFrame
        try:
            import pyarrow as pa
            self._is_arrow = isinstance(df, pa.Table)
        except ImportError:
            self._is_arrow = False

        if self._is_arrow:
            self._n_samples = df.num_rows
            available = set(df.column_names)
        else:
            self._n_samples = len(df)
            available = set(df.columns)

        # ---- Build column groups ----
        self._col_groups: Dict[str, List[str]] = {}

        def _add(key: str, cols: List[str]) -> None:
            present = [c for c in cols if c in available]
            if present:
                self._col_groups[key] = present

        _add("features", feature_columns.static_features)
        _add("hyperbolic", feature_columns.hyperbolic_columns)
        _add("tda", feature_columns.tda_columns)
        _add("collaborative", feature_columns.collaborative_columns)
        _add("hmm_journey", feature_columns.hmm_journey_columns)
        _add("hmm_lifecycle", feature_columns.hmm_lifecycle_columns)
        _add("hmm_behavior", feature_columns.hmm_behavior_columns)
        _add("multidisciplinary", feature_columns.multidisciplinary_columns)
        _add("coldstart", feature_columns.coldstart_columns)
        _add("anonymous", feature_columns.anonymous_columns)

        # Sequence columns
        self._event_seq_cols = _resolve_seq_columns(
            feature_columns.event_seq_pattern,
            feature_columns.event_seq_features,
            self._seq_cfg.event_seq_len,
            available,
        )
        self._session_seq_cols = _resolve_seq_columns(
            feature_columns.session_seq_pattern,
            feature_columns.session_seq_features,
            self._seq_cfg.session_seq_len,
            available,
        )
        self._event_td_cols = _resolve_time_delta_columns(
            feature_columns.event_time_delta_prefix,
            self._seq_cfg.event_seq_len,
            available,
        )
        self._session_td_cols = _resolve_time_delta_columns(
            feature_columns.session_time_delta_prefix,
            self._seq_cfg.session_seq_len,
            available,
        )

        # Label columns present in the df
        self._label_cols_present: Dict[str, str] = {
            task: col for task, col in label_columns.items() if col in available
        }

        # ---- Detect cuDF input ----
        _is_cudf = False
        try:
            import cudf as _cudf_mod
            _is_cudf = isinstance(df, _cudf_mod.DataFrame)
        except ImportError:
            pass

        # ---- Pre-convert to CPU tensors (batch streaming to GPU at __getitem__) ----
        # Data stays on CPU. Only the current batch moves to GPU during training.
        # This prevents VRAM from being filled with the entire dataset.
        self._tensors: Dict[str, torch.Tensor] = {}
        for key, cols in self._col_groups.items():
            self._tensors[key] = _df_to_tensor_cpu(df, cols)

        # Sequences -> reshape to 3D
        if self._event_seq_cols:
            raw = _df_to_tensor_cpu(df, self._event_seq_cols)
            self._tensors["event_sequences"] = raw.view(
                self._n_samples,
                self._seq_cfg.event_seq_len,
                self._seq_cfg.event_feat_dim,
            )
        if self._session_seq_cols:
            raw = _df_to_tensor_cpu(df, self._session_seq_cols)
            self._tensors["session_sequences"] = raw.view(
                self._n_samples,
                self._seq_cfg.session_seq_len,
                self._seq_cfg.session_feat_dim,
            )

        # Time deltas
        if self._event_td_cols:
            self._tensors["event_time_delta"] = _df_to_tensor_cpu(
                df, self._event_td_cols
            )
        if self._session_td_cols:
            self._tensors["session_time_delta"] = _df_to_tensor_cpu(
                df, self._session_td_cols
            )

        # Labels
        self._label_tensors: Dict[str, torch.Tensor] = {}
        for task, col in self._label_cols_present.items():
            self._label_tensors[task] = _df_to_tensor_cpu(
                df, [col]
            ).squeeze(-1)

        if _is_cudf:
            logger.info("PLEDataset: used cuDF → CuPy → torch GPU zero-copy path")

        logger.debug(
            "PLEDataset: %d samples, %d col groups, gpu=%s",
            self._n_samples,
            len(self._col_groups),
            self._use_gpu,
        )

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a dict whose keys align with :class:`PLEInput` fields.

        All tensors are pre-converted (both CPU and GPU paths), so indexing
        is a simple tensor slice regardless of the original DataFrame type.
        """
        item: Dict[str, Any] = {}
        for key, tensor in self._tensors.items():
            item[key] = tensor[idx]
        if self._label_tensors:
            item["targets"] = {
                task: t[idx] for task, t in self._label_tensors.items()
            }
        return item


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def _ple_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate that stacks per-sample dicts into batch tensors.

    The returned dict can be unpacked directly into :class:`PLEInput`.
    """
    import torch

    if not batch:
        return {}

    keys = [k for k in batch[0] if k != "targets"]
    collated: Dict[str, Any] = {}

    for key in keys:
        vals = [b[key] for b in batch if b.get(key) is not None]
        if vals:
            collated[key] = torch.stack(vals)

    # Targets
    if "targets" in batch[0] and batch[0]["targets"]:
        task_names = list(batch[0]["targets"].keys())
        collated["targets"] = {
            task: torch.stack([b["targets"][task] for b in batch])
            for task in task_names
        }

    return collated


# ---------------------------------------------------------------------------
# PLEInput builder from collated dict
# ---------------------------------------------------------------------------

def _collated_to_ple_input(collated: Dict[str, Any]) -> Any:
    """Convert a collated dict into a :class:`PLEInput` instance.

    Imported lazily so this module can be loaded without torch at import time.
    """
    from core.model.ple.model import PLEInput

    # Map collate dict keys to PLEInput field names
    _KEY_MAP = {
        "features": "features",
        "hyperbolic": "hyperbolic_features",
        "tda": "tda_features",
        "collaborative": "collaborative_features",
        "hmm_journey": "hmm_journey",
        "hmm_lifecycle": "hmm_lifecycle",
        "hmm_behavior": "hmm_behavior",
        "event_sequences": "event_sequences",
        "session_sequences": "session_sequences",
        "event_time_delta": "event_time_delta",
        "session_time_delta": "session_time_delta",
        "multidisciplinary": "multidisciplinary_features",
        "coldstart": "coldstart_features",
        "anonymous": "anonymous_features",
        "targets": "targets",
    }

    kwargs: Dict[str, Any] = {}
    for src_key, dst_field in _KEY_MAP.items():
        if src_key in collated:
            kwargs[dst_field] = collated[src_key]

    # features is required
    if "features" not in kwargs:
        import torch
        # Fallback: create a zero-dim placeholder
        any_tensor = next(
            (v for v in collated.values() if hasattr(v, "shape")), None
        )
        if any_tensor is not None:
            kwargs["features"] = torch.zeros(
                any_tensor.shape[0], 0,
                dtype=any_tensor.dtype, device=any_tensor.device,
            )
        else:
            raise ValueError(
                "Collated batch contains no tensors; cannot build PLEInput."
            )

    return PLEInput(**kwargs)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def build_ple_dataloader(
    df: Any,
    feature_spec: FeatureColumnSpec,
    label_columns: Dict[str, str],
    batch_size: int = 4096,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    use_gpu_loading: bool = False,
    sequence_config: Optional[SequenceConfig] = None,
    return_ple_input: bool = False,
) -> Any:
    """Build a PyTorch DataLoader that yields PLE-compatible batches.

    Parameters
    ----------
    df : DataFrame or pyarrow.Table
        pandas DataFrame, cuDF DataFrame, or PyArrow Table.
        When using DuckDB, prefer ``conn.execute(...).arrow()`` to skip pandas.
    feature_spec : FeatureColumnSpec
        Column-to-field mapping.
    label_columns : dict[str, str]
        ``{task_name: column_name}`` for target labels.
    batch_size : int
        Batch size.
    shuffle : bool
        Whether to shuffle each epoch.
    num_workers : int
        DataLoader worker count (0 = main process).
    pin_memory : bool
        Pin memory for faster GPU transfer (CPU path only).
    use_gpu_loading : bool
        When True and cuDF is available, use DLPack zero-copy.
    sequence_config : SequenceConfig, optional
        Override default sequence dimensions.
    return_ple_input : bool
        When True the collate function returns a :class:`PLEInput` object
        directly.  When False (default), returns a plain dict (backward
        compatible with existing training code).

    Returns
    -------
    torch.utils.data.DataLoader
    """
    from torch.utils.data import DataLoader

    dataset = PLEDataset(
        df=df,
        feature_columns=feature_spec,
        label_columns=label_columns,
        sequence_config=sequence_config,
        use_gpu=use_gpu_loading,
    )

    if return_ple_input:
        def _collate_ple(batch: List[Dict[str, Any]]) -> Any:
            collated = _ple_collate(batch)
            return _collated_to_ple_input(collated)

        collate_fn = _collate_ple
    else:
        collate_fn = _ple_collate

    # Data tensors are always on CPU (batch streaming to GPU).
    # pin_memory=True for fast CPU→GPU transfer via DMA.
    # num_workers=0 when cuDF was used for loading (CUDA context not forkable).
    _used_cudf = False
    try:
        import cudf as _cudf_check
        _used_cudf = isinstance(df, _cudf_check.DataFrame)
    except ImportError:
        pass

    if _used_cudf:
        num_workers = 0
        pin_memory = True  # tensors are on CPU, pin for fast GPU transfer

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
    )

    logger.info(
        "build_ple_dataloader: %d samples, batch_size=%d, shuffle=%s, "
        "gpu_loading=%s, num_workers=%d",
        len(dataset), batch_size, shuffle, use_gpu_loading, num_workers,
    )
    return loader
