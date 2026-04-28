"""
Mamba Feature Generator -- Selective State Space Model temporal embeddings.

Extracts temporal representations from sequential data (e.g. transaction
histories, session logs) using the Mamba architecture's selective scan
mechanism.

Hardware acceleration strategy:
  * **Primary**: ``mamba_ssm`` library with CUDA-optimised selective scan
    kernel (O(L) complexity).
  * **Fallback**: pure ``torch`` CPU implementation with sequential scan.
  * **Last resort**: numpy-only simplified state space model using matrix
    exponential discretisation.

Output: ``output_dim`` (default 50) dimensions after PCA compression from
the Mamba hidden state (default 256D).

References
----------
Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*,
arXiv 2312.00752, 2023.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.data.dataframe import df_backend
from ..generator import AbstractFeatureGenerator, FeatureGeneratorRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy torch / mamba_ssm imports
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

try:
    import mamba_ssm  # CUDA-optimised selective scan
except ImportError:
    mamba_ssm = None  # type: ignore[assignment]

try:
    import cudf  # GPU-accelerated DataFrame
    _HAS_CUDF = True
except ImportError:
    _HAS_CUDF = False

from .gpu_utils import (
    StepTimer,
    adaptive_batch_size,
    ensure_numpy,
    ensure_tensor,
    get_device,
    get_device_description,
)


# ======================================================================
# Config
# ======================================================================

@dataclass
class MambaConfig:
    """Configuration for :class:`MambaFeatureGenerator`."""

    d_model: int = 256
    d_state: int = 16
    d_conv: int = 4
    expand_factor: int = 2
    seq_len: int = 90
    output_dim: int = 50
    use_pca: bool = True
    num_epochs: int = 20
    learning_rate: float = 0.001
    dropout: float = 0.1
    max_retries: int = 3
    base_batch_size: int = 512

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "MambaConfig":
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in cfg.items() if k in valid})


# ======================================================================
# Torch sub-modules (only instantiated when torch is available)
# ======================================================================

class _SelectiveSSMBlock(nn.Module):
    """Pure-PyTorch Mamba block: projection -> conv1d -> SSM scan -> output.

    This is used as a fallback when ``mamba_ssm`` is not installed.
    """

    def __init__(self, d_input: int, cfg: MambaConfig):
        super().__init__()
        self.d_model = cfg.d_model
        self.d_inner = cfg.d_model * cfg.expand_factor
        self.d_state = cfg.d_state
        self.dt_rank = max(1, math.ceil(self.d_inner / 16))

        # Input projection
        self.input_proj = nn.Linear(d_input, cfg.d_model)
        self.norm = nn.LayerNorm(cfg.d_model)

        # Two parallel paths: SSM path + gate path
        self.in_proj = nn.Linear(cfg.d_model, self.d_inner * 2, bias=False)

        # Depthwise conv for local patterns
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=cfg.d_conv,
            padding=cfg.d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        # SSM parameters
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + cfg.d_state * 2, bias=False
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A matrix: diagonal, negative for stability
        A = torch.arange(
            1, cfg.d_state + 1, dtype=torch.float32
        ).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # Skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

        # Delta initialisation
        dt_min, dt_max = 0.001, 0.1
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input sequence.

        Parameters
        ----------
        x : torch.Tensor
            ``[batch, seq_len, d_input]``

        Returns
        -------
        torch.Tensor
            ``[batch, seq_len, d_model]``
        """
        x = self.input_proj(x)
        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)

        # Conv1d path
        x_conv = x_ssm.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :x_ssm.shape[1]]  # causal trim
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)

        # SSM path
        y = self._ssm_scan(x_conv)

        # Gated output
        z = F.silu(z)
        output = y * z
        output = self.out_proj(output)
        output = self.dropout(output) + residual
        return output

    def _ssm_scan(self, x: torch.Tensor) -> torch.Tensor:
        """Sequential SSM scan.

        Discretise: A_bar = exp(delta * A), B_bar = delta * B
        Scan: h[t] = A_bar * h[t-1] + B_bar * x[t]
               y[t] = C * h[t]
        """
        batch, seq_len, _ = x.shape

        x_proj = self.x_proj(x)
        dt, B, C = torch.split(
            x_proj, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        dt = F.softplus(self.dt_proj(dt))  # [B, L, D]
        A = -torch.exp(self.A_log.float())  # [D, N]

        # Zero-Order Hold discretisation
        dA = torch.exp(torch.einsum("bld,dn->bldn", dt, A))
        dB = torch.einsum("bld,bln->bldn", dt, B)
        dBx = dB * x.unsqueeze(-1)

        h = torch.zeros(
            batch, self.d_inner, self.d_state, device=x.device, dtype=x.dtype
        )
        outputs = []

        for t in range(seq_len):
            h = dA[:, t] * h + dBx[:, t]
            y_t = torch.einsum("bdn,bn->bd", h, C[:, t])
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        return y


class _MambaEncoder(nn.Module):
    """Encoder that wraps one or more Mamba blocks and pools the output."""

    def __init__(self, d_input: int, cfg: MambaConfig, use_mamba_ssm: bool = False):
        super().__init__()
        self.cfg = cfg
        self.use_mamba_ssm = use_mamba_ssm

        if use_mamba_ssm and mamba_ssm is not None:
            # Use CUDA-optimised Mamba block
            self.input_proj = nn.Linear(d_input, cfg.d_model)
            self.mamba_block = mamba_ssm.Mamba(
                d_model=cfg.d_model,
                d_state=cfg.d_state,
                d_conv=cfg.d_conv,
                expand=cfg.expand_factor,
            )
            self.norm = nn.LayerNorm(cfg.d_model)
        else:
            # Pure-PyTorch fallback
            self.mamba_block = _SelectiveSSMBlock(d_input, cfg)
            self.input_proj = None
            self.norm = nn.LayerNorm(cfg.d_model)

        self.output_head = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode sequences and return pooled representation.

        Parameters
        ----------
        x : torch.Tensor
            ``[batch, seq_len, d_input]``

        Returns
        -------
        torch.Tensor
            ``[batch, d_model]``
        """
        if self.use_mamba_ssm and self.input_proj is not None:
            x = self.input_proj(x)
            h = self.mamba_block(x)
            h = self.norm(h)
        else:
            h = self.mamba_block(x)
            h = self.norm(h)

        # Mean pooling over sequence
        pooled = h.mean(dim=1)
        return self.output_head(pooled)


# ======================================================================
# Registered generator
# ======================================================================

@FeatureGeneratorRegistry.register(
    "mamba",
    description="Mamba SSM temporal embedding features for sequential data.",
    tags=["temporal", "sequence", "ssm", "mamba", "gpu"],
)
class MambaFeatureGenerator(AbstractFeatureGenerator):
    """Generate temporal embeddings using the Mamba selective state space model.

    Processes sequential data (e.g. transaction histories grouped by
    customer) through a Mamba block, then optionally compresses the
    hidden representation via PCA to ``output_dim`` dimensions.

    Parameters
    ----------
    d_model : int
        Hidden dimension of the Mamba block (default 256).
    d_state : int
        SSM latent state dimension (default 16).
    d_conv : int
        Depthwise convolution kernel size (default 4).
    expand_factor : int
        Expansion factor for inner dimension (default 2).
    seq_len : int
        Target sequence length; shorter sequences are padded, longer
        ones are truncated (default 90).
    output_dim : int
        Final output dimension after PCA compression (default 50).
    use_pca : bool
        Apply PCA to compress d_model -> output_dim (default True).
    entity_column : str
        Column identifying the entity whose sequences to process
        (default ``"user_id"``).
    time_column : str
        Column for temporal ordering within each entity (default
        ``"timestamp"``).
    feature_columns : list[str] or None
        Columns used as sequence features; ``None`` = all numeric.
    prefix : str
        Column name prefix (default ``"mamba"``).
    random_state : int
        Random seed (default 42).
    prefer_gpu : bool
        Try to use CUDA when available (default True).
    """

    supports_gpu: bool = True
    required_libraries: List[str] = ["torch"]

    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
        seq_len: int = 90,
        output_dim: int = 50,
        use_pca: bool = True,
        num_epochs: int = 20,
        learning_rate: float = 0.001,
        entity_column: str = "user_id",
        time_column: str = "timestamp",
        feature_columns: Optional[List[str]] = None,
        prefix: str = "mamba",
        random_state: int = 42,
        prefer_gpu: bool = True,
        base_batch_size: int = 512,
        cached_embedding_uri: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.cfg = MambaConfig(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor,
            seq_len=seq_len,
            output_dim=output_dim,
            use_pca=use_pca,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            base_batch_size=base_batch_size,
        )
        self.entity_column = entity_column
        self.time_column = time_column
        self.feature_columns = feature_columns
        self.prefix = prefix
        self.random_state = random_state
        self.prefer_gpu = prefer_gpu
        # When set, ``fit()`` skips the (CUDA-only) Mamba training and
        # ``generate()`` reads the precomputed embedding parquet from
        # this URI (local path or s3://...). The parquet must contain
        # one row per entity, with ``entity_column`` plus 50 columns
        # named per ``output_columns``.
        self.cached_embedding_uri = cached_embedding_uri

        # Fitted state
        self._pca_components: Optional[np.ndarray] = None
        self._pca_mean: Optional[np.ndarray] = None
        self._entity_embeddings: Optional[Dict[Any, np.ndarray]] = None
        self._default_embedding: Optional[np.ndarray] = None
        self._d_input: Optional[int] = None

        # Determine device
        self._device = get_device(prefer_gpu) if torch is not None else "cpu"
        device_desc = (
            get_device_description(self._device)
            if torch is not None
            else "cpu (torch unavailable)"
        )
        mamba_status = "mamba_ssm available" if mamba_ssm is not None else "pure-torch"
        logger.info(
            "MambaFeatureGenerator: using %s (%s)", device_desc, mamba_status
        )

    # -- Output description ------------------------------------------------

    @property
    def output_dim(self) -> int:
        return self.cfg.output_dim

    @property
    def output_columns(self) -> List[str]:
        return [f"{self.prefix}_d{i}" for i in range(self.cfg.output_dim)]

    @property
    def input_cols(self) -> List[str]:
        """Source columns Mamba reads at fit/generate.

        On the cached path only ``entity_column`` is needed (the
        embeddings come from disk). On the training path we need the
        full sequence-feature set plus the time ordering column.
        """
        cols: List[str] = [self.entity_column]
        if self.cached_embedding_uri:
            return cols
        if self.time_column:
            cols.append(self.time_column)
        if self.feature_columns:
            cols.extend(c for c in self.feature_columns if c not in cols)
        return cols

    # -- Core API ----------------------------------------------------------

    def fit(self, df: Any, **context: Any) -> "MambaFeatureGenerator":
        """Learn temporal embeddings from sequential data.

        Groups data by ``entity_column``, orders by ``time_column``,
        and passes each entity's sequence through a Mamba encoder.
        Falls back to numpy state space approximation if torch is
        unavailable.

        When ``cached_embedding_uri`` is set, training is skipped
        entirely: we just load the per-entity embedding parquet that
        a separate GPU pre-compute job (scripts/precompute_mamba.py)
        already produced, and ``generate()`` becomes a lookup. This
        is the production path on CPU Phase-0 instances that lack
        the CUDA wheel for ``mamba_ssm``.
        """
        if self.cached_embedding_uri:
            self._load_cached_embeddings(self.cached_embedding_uri)
            self._fitted = True
            logger.info(
                "MambaFeatureGenerator: skipped fit() — loaded %d cached "
                "embeddings from %s (output_dim=%d)",
                len(self._entity_embeddings or {}),
                self.cached_embedding_uri,
                self.cfg.output_dim,
            )
            return self

        pdf = df_backend.to_pandas(df) if not isinstance(df, pd.DataFrame) else df
        rng = np.random.RandomState(self.random_state)

        entities, sequences, d_input = self._build_sequences(pdf)
        self._d_input = d_input
        n_entities = len(entities)

        logger.info(
            "MambaFeatureGenerator.fit: %d entities, seq_len=%d, d_input=%d",
            n_entities, self.cfg.seq_len, d_input,
        )

        if torch is not None:
            raw_embeddings = self._fit_torch(sequences, d_input, rng)
        else:
            raw_embeddings = self._fit_numpy(sequences, d_input, rng)

        # PCA compression: d_model -> output_dim
        if self.cfg.use_pca and raw_embeddings.shape[1] > self.cfg.output_dim:
            with StepTimer("pca_compression", logger):
                compressed = self._fit_pca(raw_embeddings)
        else:
            compressed = raw_embeddings[:, : self.cfg.output_dim]

        # Store per-entity embeddings
        self._entity_embeddings = {
            entities[i]: compressed[i] for i in range(n_entities)
        }
        self._default_embedding = np.zeros(self.cfg.output_dim, dtype=np.float32)

        self._fitted = True
        logger.info(
            "MambaFeatureGenerator fitted: %d entities, output_dim=%d",
            n_entities, self.cfg.output_dim,
        )
        return self

    def _load_cached_embeddings(self, uri: str) -> None:
        """Load per-entity Mamba embeddings produced by the GPU pre-compute job.

        ``uri`` may be a local path or an ``s3://`` URI. For S3 we
        download to a temp file via boto3 first because DuckDB's
        httpfs extension does not pick up the SageMaker instance's
        IAM role from the AWS credential chain by default — the read
        fails with HTTP 403 against the private bucket. boto3 uses
        the standard credential chain (env / instance profile / role)
        so the download just works inside the training container.

        The parquet must contain ``self.entity_column`` plus the
        columns named in ``self.output_columns`` (one row per entity).
        """
        import duckdb as _ddb_mamba
        emb_cols = self.output_columns
        proj = ", ".join(f'"{c}"' for c in [self.entity_column] + emb_cols)

        local_path = uri
        if uri.startswith("s3://"):
            import os
            import tempfile

            import boto3

            no_proto = uri[len("s3://"):]
            bucket, _, key = no_proto.partition("/")
            tmp_dir = tempfile.mkdtemp(prefix="mamba_emb_")
            local_path = os.path.join(tmp_dir, "embedding.parquet")
            boto3.client("s3").download_file(bucket, key, local_path)

        _con = _ddb_mamba.connect()
        try:
            arr = _con.execute(
                f"SELECT {proj} FROM read_parquet('{local_path}')"
            ).fetch_arrow_table()
        finally:
            _con.close()
        # Build the lookup dict. Arrow → numpy avoids per-row Python
        # object overhead.
        entity_arr = arr.column(self.entity_column).to_numpy(
            zero_copy_only=False
        )
        emb_matrix = np.stack(
            [arr.column(c).to_numpy(zero_copy_only=False) for c in emb_cols],
            axis=1,
        ).astype(np.float32, copy=False)
        self._entity_embeddings = {
            entity_arr[i]: emb_matrix[i] for i in range(len(entity_arr))
        }
        self._default_embedding = np.zeros(
            self.cfg.output_dim, dtype=np.float32
        )

    def generate(self, df: Any, **context: Any) -> Any:
        """Look up pre-computed Mamba embeddings for each row."""
        if not self._fitted:
            raise RuntimeError(
                "MambaFeatureGenerator must be fitted before generate()."
            )

        pdf = df_backend.to_pandas(df) if not isinstance(df, pd.DataFrame) else df
        n_rows = len(pdf)

        result = np.zeros((n_rows, self.cfg.output_dim), dtype=np.float32)

        if self.entity_column in pdf.columns:
            entity_keys = pdf[self.entity_column].values
        else:
            entity_keys = pdf.index.values

        for i, key in enumerate(entity_keys):
            result[i] = self._entity_embeddings.get(
                key, self._default_embedding
            )

        data = {
            f"{self.prefix}_d{j}": result[:, j]
            for j in range(self.cfg.output_dim)
        }
        return df_backend.from_dict(data, index=pdf.index)

    # ==================================================================
    # Private -- sequence construction
    # ==================================================================

    def _build_sequences(
        self, pdf: pd.DataFrame
    ) -> Tuple[List[Any], np.ndarray, int]:
        """Build padded sequences per entity.

        Uses cuDF for GPU-accelerated sort and groupby when available,
        falling back to pandas otherwise.

        Returns
        -------
        entities : list
            Unique entity identifiers.
        sequences : np.ndarray
            ``[n_entities, seq_len, d_input]`` padded sequences.
        d_input : int
            Number of feature dimensions per time step.
        """
        # Resolve feature columns (always on pandas for schema inspection)
        if self.feature_columns:
            fcols = [c for c in self.feature_columns if c in pdf.columns]
        else:
            fcols = pdf.select_dtypes(include=["number"]).columns.tolist()
            for exclude in [self.entity_column, self.time_column]:
                if exclude in fcols:
                    fcols.remove(exclude)

        d_input = len(fcols) if fcols else 1

        # ------------------------------------------------------------------
        # cuDF GPU path: sort + groupby on GPU, extract numpy values
        # ------------------------------------------------------------------
        if _HAS_CUDF:
            try:
                return self._build_sequences_cudf(pdf, fcols, d_input)
            except Exception:
                logger.debug(
                    "cuDF sequence build failed, falling back to pandas",
                    exc_info=True,
                )

        # ------------------------------------------------------------------
        # pandas CPU fallback
        # ------------------------------------------------------------------
        return self._build_sequences_pandas(pdf, fcols, d_input)

    def _build_sequences_cudf(
        self,
        pdf: pd.DataFrame,
        fcols: List[str],
        d_input: int,
    ) -> Tuple[List[Any], np.ndarray, int]:
        """cuDF-accelerated sort + groupby path for sequence construction."""
        gdf = cudf.DataFrame(pdf) if not isinstance(pdf, cudf.DataFrame) else pdf

        # Sort by entity + time on GPU
        if self.time_column in gdf.columns:
            gdf = gdf.sort_values([self.entity_column, self.time_column])

        # Group by entity on GPU
        if self.entity_column in gdf.columns:
            groups = gdf.groupby(self.entity_column)
            entities = list(groups.groups.keys())
        else:
            entities = gdf.index.unique().to_pandas().tolist()
            groups = gdf.groupby(gdf.index)

        n_entities = len(entities)
        seq_len = self.cfg.seq_len
        sequences = np.zeros(
            (n_entities, seq_len, d_input), dtype=np.float32
        )

        for idx, entity in enumerate(entities):
            try:
                group = groups.get_group(entity)
            except KeyError:
                continue

            if fcols:
                # .values.get on cuDF returns cupy array; convert to numpy
                vals = group[fcols].to_pandas().values.astype(np.float32)
            else:
                vals = (
                    group.select_dtypes(include=["number"])
                    .to_pandas()
                    .values.astype(np.float32)
                )
                if vals.shape[1] == 0:
                    vals = np.ones((len(group), 1), dtype=np.float32)

            actual_len = min(vals.shape[0], seq_len)
            sequences[idx, seq_len - actual_len:, :vals.shape[1]] = vals[
                -actual_len:
            ]

        logger.debug(
            "Built %d sequences via cuDF (seq_len=%d, d_input=%d)",
            n_entities, seq_len, d_input,
        )
        return entities, sequences, d_input

    def _build_sequences_pandas(
        self,
        pdf: pd.DataFrame,
        fcols: List[str],
        d_input: int,
    ) -> Tuple[List[Any], np.ndarray, int]:
        """Pandas CPU fallback for sequence construction."""
        # Sort by time if available
        if self.time_column in pdf.columns:
            pdf = pdf.sort_values([self.entity_column, self.time_column])

        # Group by entity
        if self.entity_column in pdf.columns:
            groups = pdf.groupby(self.entity_column)
            entities = list(groups.groups.keys())
        else:
            entities = list(pdf.index.unique())
            groups = pdf.groupby(pdf.index)

        n_entities = len(entities)
        seq_len = self.cfg.seq_len
        sequences = np.zeros(
            (n_entities, seq_len, d_input), dtype=np.float32
        )

        for idx, entity in enumerate(entities):
            try:
                group = groups.get_group(entity)
            except KeyError:
                continue

            if fcols:
                vals = group[fcols].values.astype(np.float32)
            else:
                vals = group.select_dtypes(include=["number"]).values.astype(
                    np.float32
                )
                if vals.shape[1] == 0:
                    vals = np.ones((len(group), 1), dtype=np.float32)

            # Truncate or pad
            actual_len = min(vals.shape[0], seq_len)
            # Right-align: most recent data at the end
            sequences[idx, seq_len - actual_len:, :vals.shape[1]] = vals[
                -actual_len:
            ]

        return entities, sequences, d_input

    # ==================================================================
    # Private -- torch training
    # ==================================================================

    def _fit_torch(
        self,
        sequences: np.ndarray,
        d_input: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """Encode sequences using torch Mamba model.

        Returns ``[n_entities, d_model]`` raw embeddings.
        """
        device = self._device
        device_str = str(device)
        n_entities = sequences.shape[0]

        use_mamba_ssm_cuda = (
            mamba_ssm is not None and device_str.startswith("cuda")
        )

        with StepTimer("build_mamba_encoder", logger):
            model = _MambaEncoder(
                d_input=d_input,
                cfg=self.cfg,
                use_mamba_ssm=use_mamba_ssm_cuda,
            ).to(device)

        # Self-supervised: reconstruction loss on input features
        optimiser = torch.optim.Adam(
            model.parameters(), lr=self.cfg.learning_rate
        )

        batch_size = adaptive_batch_size(
            total_samples=n_entities,
            base_batch=self.cfg.base_batch_size,
            bytes_per_sample=self.cfg.seq_len * d_input * 4 * 3,
        )

        use_amp = device_str.startswith("cuda")
        if use_amp:
            # torch.amp.GradScaler is the 2.4+ API; PyTorch 2.1 (the
            # SageMaker GPU DLC ships 2.1) raises AttributeError when
            # accessing it, so fall back to the legacy
            # torch.cuda.amp.GradScaler path on that error class too.
            try:
                scaler = torch.amp.GradScaler()
            except (TypeError, AttributeError):
                scaler = torch.cuda.amp.GradScaler()
        else:
            scaler = None

        with StepTimer("mamba_training", logger):
            self._train_mamba(
                model=model,
                sequences=sequences,
                batch_size=batch_size,
                device=device,
                rng=rng,
                optimiser=optimiser,
                scaler=scaler,
                use_amp=use_amp,
            )

        # Extract embeddings
        with StepTimer("mamba_inference", logger):
            embeddings = self._encode_all(model, sequences, batch_size, device)

        return embeddings

    def _train_mamba(
        self,
        model: nn.Module,
        sequences: np.ndarray,
        batch_size: int,
        device: Any,
        rng: np.random.RandomState,
        optimiser: Any,
        scaler: Any,
        use_amp: bool,
    ) -> None:
        """Train the Mamba encoder with self-supervised reconstruction loss."""
        model.train()
        n = sequences.shape[0]
        max_retries = self.cfg.max_retries

        for epoch in range(self.cfg.num_epochs):
            # Shuffle
            perm = rng.permutation(n)
            total_loss = 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                idx = perm[start:end]
                batch = torch.tensor(
                    sequences[idx], dtype=torch.float32, device=device
                )

                for attempt in range(max_retries):
                    try:
                        optimiser.zero_grad()

                        if use_amp:
                            # PyTorch 2.1 doesn't have torch.amp.autocast,
                            # only torch.cuda.amp.autocast. Use the legacy
                            # path to stay compatible with the SageMaker
                            # GPU DLC.
                            try:
                                _amp_ctx = torch.amp.autocast("cuda")
                            except (TypeError, AttributeError):
                                _amp_ctx = torch.cuda.amp.autocast()
                            with _amp_ctx:
                                encoded = model(batch)
                                # Self-supervised: predict mean of input
                                target = batch.mean(dim=1)
                                # Project encoded to input dim for loss
                                if encoded.shape[-1] != target.shape[-1]:
                                    target = F.adaptive_avg_pool1d(
                                        target.unsqueeze(1),
                                        encoded.shape[-1],
                                    ).squeeze(1)
                                loss = F.mse_loss(encoded, target)
                            scaler.scale(loss).backward()
                            scaler.step(optimiser)
                            scaler.update()
                        else:
                            encoded = model(batch)
                            target = batch.mean(dim=1)
                            if encoded.shape[-1] != target.shape[-1]:
                                target = F.adaptive_avg_pool1d(
                                    target.unsqueeze(1),
                                    encoded.shape[-1],
                                ).squeeze(1)
                            loss = F.mse_loss(encoded, target)
                            loss.backward()
                            optimiser.step()

                        total_loss += loss.item()
                        n_batches += 1
                        break

                    except RuntimeError as e:
                        if "out of memory" in str(e).lower() and torch is not None:
                            torch.cuda.empty_cache()
                            batch_size = max(32, batch_size // 2)
                            logger.warning(
                                "OOM on epoch %d, reducing batch to %d",
                                epoch, batch_size,
                            )
                            if attempt == max_retries - 1:
                                raise
                        else:
                            raise

            if epoch % 5 == 0 or epoch == self.cfg.num_epochs - 1:
                avg_loss = total_loss / max(n_batches, 1)
                logger.info(
                    "  epoch %d/%d  loss=%.4f",
                    epoch + 1, self.cfg.num_epochs, avg_loss,
                )

    def _encode_all(
        self,
        model: nn.Module,
        sequences: np.ndarray,
        batch_size: int,
        device: Any,
    ) -> np.ndarray:
        """Run inference on all sequences and return numpy embeddings."""
        model.eval()
        n = sequences.shape[0]
        all_embs = []

        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch = torch.tensor(
                    sequences[start:end], dtype=torch.float32, device=device
                )
                emb = model(batch).detach().cpu().numpy()
                all_embs.append(emb)

        return np.concatenate(all_embs, axis=0).astype(np.float32)

    # ==================================================================
    # Private -- numpy fallback
    # ==================================================================

    def _fit_numpy(
        self,
        sequences: np.ndarray,
        d_input: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """Simplified state space model using numpy matrix exponential.

        Returns ``[n_entities, d_model]`` approximate embeddings.
        """
        logger.info("MambaFeatureGenerator: using numpy-only fallback (SSM)")

        n_entities, seq_len, _ = sequences.shape
        d_model = min(self.cfg.d_model, 64)  # keep reasonable for numpy
        d_state = min(self.cfg.d_state, 8)

        with StepTimer("numpy_ssm_embedding", logger):
            # Random projection: d_input -> d_model
            W_in = rng.randn(d_input, d_model).astype(np.float32) * 0.1

            # SSM parameters (fixed, not learned)
            A = -np.arange(1, d_state + 1, dtype=np.float32)  # [N]
            B_proj = rng.randn(d_model, d_state).astype(np.float32) * 0.01
            C_proj = rng.randn(d_state, d_model).astype(np.float32) * 0.01
            delta = 0.01  # fixed discretisation step

            # Discretise: A_bar = exp(delta * A)
            A_bar = np.exp(delta * A)  # [N]
            B_bar_scale = delta

            embeddings = np.zeros((n_entities, d_model), dtype=np.float32)

            for i in range(n_entities):
                seq = sequences[i]  # [seq_len, d_input]
                x = seq @ W_in  # [seq_len, d_model]

                # Sequential scan
                h = np.zeros((d_model, d_state), dtype=np.float32)

                for t in range(seq_len):
                    x_t = x[t]  # [d_model]
                    B_t = x_t[:, None] * B_proj * B_bar_scale  # [d_model, N]
                    h = A_bar[None, :] * h + B_t  # [d_model, N]

                # Output: C * h, averaged over d_model
                y = h @ C_proj  # [d_model, d_model]
                embeddings[i] = np.mean(y, axis=0)

        # Pad to full d_model if we reduced it
        if d_model < self.cfg.d_model:
            pad = np.zeros(
                (n_entities, self.cfg.d_model - d_model), dtype=np.float32
            )
            embeddings = np.concatenate([embeddings, pad], axis=1)

        return embeddings

    # ==================================================================
    # Private -- PCA compression
    # ==================================================================

    def _fit_pca(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit PCA and compress embeddings from d_model to output_dim.

        Stores PCA components for use in transform/generate if needed.
        """
        n, d = embeddings.shape
        k = min(self.cfg.output_dim, d, n)

        # Centre
        self._pca_mean = embeddings.mean(axis=0)
        centred = embeddings - self._pca_mean

        # SVD-based PCA
        try:
            U, S, Vt = np.linalg.svd(centred, full_matrices=False)
            self._pca_components = Vt[:k]  # [k, d]
            compressed = centred @ self._pca_components.T  # [n, k]
        except np.linalg.LinAlgError:
            logger.warning("PCA SVD failed, using truncation instead")
            self._pca_components = None
            compressed = embeddings[:, :k]

        # Pad if k < output_dim
        if compressed.shape[1] < self.cfg.output_dim:
            pad = np.zeros(
                (n, self.cfg.output_dim - compressed.shape[1]),
                dtype=np.float32,
            )
            compressed = np.concatenate([compressed, pad], axis=1)

        return compressed.astype(np.float32)
