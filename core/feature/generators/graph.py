"""
Graph Embedding Feature Generator -- LightGCN-based graph embeddings with
optional Poincare ball projection.

Generates dense vector representations of entities by constructing a kNN
similarity graph from feature vectors and learning embeddings via LightGCN
message passing with BPR self-supervised loss.

Hardware acceleration strategy:
  * **Primary**: ``torch.cuda`` (GPU) with adaptive batch sizing and mixed
    precision training.
  * **Fallback**: ``torch`` on CPU.
  * **Last resort**: numpy-only approximate embeddings via truncated SVD of
    the adjacency matrix.

Output: ``embedding_dim`` (default 20) + ``norm`` + ``depth`` = 22 dimensions.

References
----------
He et al., *LightGCN: Simplifying and Powering Graph Convolution Network
for Recommendation*, SIGIR 2020.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.data.dataframe import df_backend
from ..generator import AbstractFeatureGenerator, FeatureGeneratorRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy torch imports
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

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
class GraphConfig:
    """Configuration for :class:`GraphEmbeddingGenerator`."""

    embedding_dim: int = 20
    num_layers: int = 3
    k_neighbors: int = 10
    num_epochs: int = 30
    learning_rate: float = 0.005
    use_bpr: bool = True
    use_poincare: bool = True
    curvature: float = 1.0
    dropout: float = 0.1
    max_retries: int = 3
    base_batch_size: int = 2048

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "GraphConfig":
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in cfg.items() if k in valid})


# ======================================================================
# Torch sub-modules (only instantiated when torch is available)
# ======================================================================

def _build_lightgcn_layers(hidden_dim: int, num_layers: int, dropout: float):
    """Build parameter-free LightGCN aggregation layers."""

    class _LightGCNConvLayer(nn.Module):
        """Single LightGCN layer: neighbourhood mean aggregation."""

        def __init__(self, dim: int, drop: float):
            super().__init__()
            self.dropout = nn.Dropout(drop)

        def forward(
            self, x: torch.Tensor, adj: torch.Tensor
        ) -> torch.Tensor:
            """Propagate via sparse/dense adjacency *adj*.

            Parameters
            ----------
            x : torch.Tensor
                ``[N, dim]`` node embeddings.
            adj : torch.Tensor
                ``[N, N]`` normalised adjacency (dense or sparse).
            """
            out = torch.mm(adj, x) if not adj.is_sparse else torch.sparse.mm(adj, x)
            return self.dropout(out)

    layers = nn.ModuleList(
        [_LightGCNConvLayer(hidden_dim, dropout) for _ in range(num_layers)]
    )
    return layers


class _GraphEmbeddingModel(nn.Module):
    """LightGCN embedding model with BPR loss and layer combination."""

    def __init__(self, n_nodes: int, cfg: GraphConfig):
        super().__init__()
        self.cfg = cfg
        self.embeddings = nn.Embedding(n_nodes, cfg.embedding_dim)
        nn.init.xavier_uniform_(self.embeddings.weight)
        self.conv_layers = _build_lightgcn_layers(
            cfg.embedding_dim, cfg.num_layers, cfg.dropout
        )
        # Learnable layer combination weights
        self.layer_weights = nn.Parameter(
            torch.ones(cfg.num_layers + 1) / (cfg.num_layers + 1)
        )

    def forward(self, adj: torch.Tensor) -> torch.Tensor:
        """Return combined embeddings for all nodes.

        Parameters
        ----------
        adj : torch.Tensor
            Normalised adjacency matrix ``[N, N]``.

        Returns
        -------
        torch.Tensor
            ``[N, embedding_dim]``
        """
        h = self.embeddings.weight
        all_layers = [h]
        for conv in self.conv_layers:
            h = conv(h, adj)
            all_layers.append(h)
        weights = F.softmax(self.layer_weights, dim=0)
        combined = sum(w * emb for w, emb in zip(weights, all_layers))
        return combined

    def bpr_loss(
        self,
        adj: torch.Tensor,
        pos_edges: torch.Tensor,
        neg_edges: torch.Tensor,
    ) -> torch.Tensor:
        """BPR pairwise ranking loss.

        Parameters
        ----------
        adj : torch.Tensor
            Normalised adjacency.
        pos_edges : torch.Tensor
            ``[E, 2]`` positive edge indices.
        neg_edges : torch.Tensor
            ``[E, 2]`` negative edge indices.
        """
        embs = self.forward(adj)
        pos_src = embs[pos_edges[:, 0]]
        pos_dst = embs[pos_edges[:, 1]]
        neg_dst = embs[neg_edges[:, 1]]

        pos_score = (pos_src * pos_dst).sum(dim=-1)
        neg_score = (pos_src * neg_dst).sum(dim=-1)
        return -F.logsigmoid(pos_score - neg_score).mean()

    def reconstruction_loss(self, adj: torch.Tensor) -> torch.Tensor:
        """Simple adjacency reconstruction loss (when BPR is disabled)."""
        embs = self.forward(adj)
        pred = torch.mm(embs, embs.t())
        target = adj.to_dense() if adj.is_sparse else adj
        return F.mse_loss(pred, target)


# ======================================================================
# Registered generator
# ======================================================================

@FeatureGeneratorRegistry.register(
    "graph",
    description="LightGCN-based graph embedding features with optional Poincare projection.",
    tags=["graph", "embedding", "lightgcn", "gpu"],
)
class GraphEmbeddingGenerator(AbstractFeatureGenerator):
    """Generate graph embedding features via LightGCN message passing.

    Constructs a kNN similarity graph from numeric features, learns
    node embeddings through parameter-free neighbourhood aggregation,
    and optionally projects them onto the Poincare ball for hierarchical
    structure.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of learned embeddings (default 20).
    num_layers : int
        LightGCN propagation layers (default 3).
    k_neighbors : int
        kNN graph construction parameter (default 10).
    num_epochs : int
        Training epochs (default 30).
    learning_rate : float
        Optimiser learning rate (default 0.005).
    use_bpr : bool
        Use BPR self-supervised loss (default True).
    use_poincare : bool
        Apply Poincare ball projection (default True).
    curvature : float
        Poincare ball curvature (default 1.0).
    entity_column : str
        Column identifying entities (default ``"user_id"``).
    feature_columns : list[str] or None
        Columns used for kNN graph; ``None`` = all numeric.
    prefix : str
        Column name prefix (default ``"graph"``).
    random_state : int
        Random seed (default 42).
    prefer_gpu : bool
        Try to use CUDA when available (default True).
    """

    supports_gpu: bool = True
    required_libraries: List[str] = ["torch"]

    def __init__(
        self,
        embedding_dim: int = 20,
        num_layers: int = 3,
        k_neighbors: int = 10,
        num_epochs: int = 30,
        learning_rate: float = 0.005,
        use_bpr: bool = True,
        use_poincare: bool = True,
        curvature: float = 1.0,
        entity_column: str = "user_id",
        feature_columns: Optional[List[str]] = None,
        prefix: str = "graph",
        random_state: int = 42,
        prefer_gpu: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.cfg = GraphConfig(
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            k_neighbors=k_neighbors,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            use_bpr=use_bpr,
            use_poincare=use_poincare,
            curvature=curvature,
        )
        self.entity_column = entity_column
        self.feature_columns = feature_columns
        self.prefix = prefix
        self.random_state = random_state
        self.prefer_gpu = prefer_gpu

        # Fitted state
        self._embedding_table: Optional[Dict[Any, np.ndarray]] = None
        self._default_embedding: Optional[np.ndarray] = None

        # Determine device
        self._device = get_device(prefer_gpu) if torch is not None else "cpu"
        device_desc = get_device_description(self._device) if torch is not None else "cpu (torch unavailable)"
        logger.info("GraphEmbeddingGenerator: using %s", device_desc)

    # -- Output description ------------------------------------------------

    @property
    def output_dim(self) -> int:
        """embedding_dim + norm + depth = embedding_dim + 2."""
        return self.cfg.embedding_dim + 2

    @property
    def output_columns(self) -> List[str]:
        cols = [f"{self.prefix}_d{i}" for i in range(self.cfg.embedding_dim)]
        cols.append(f"{self.prefix}_norm")
        cols.append(f"{self.prefix}_depth")
        return cols

    # -- Core API ----------------------------------------------------------

    def fit(self, df: Any, **context: Any) -> "GraphEmbeddingGenerator":
        """Learn graph embeddings from training data.

        Builds a kNN similarity graph, trains LightGCN embeddings
        (GPU/CPU), and stores per-entity vectors.  Falls back to
        numpy SVD if torch is unavailable.
        """
        pdf = df_backend.to_pandas(df) if not isinstance(df, pd.DataFrame) else df
        rng = np.random.RandomState(self.random_state)

        # Resolve entities and features
        entities, features = self._extract_entities_features(pdf)
        n_nodes = len(entities)
        entity_list = list(entities)

        logger.info(
            "GraphEmbeddingGenerator.fit: %d entities, %d feature dims",
            n_nodes, features.shape[1],
        )

        if torch is not None:
            embeddings = self._fit_torch(features, n_nodes, rng)
        else:
            embeddings = self._fit_numpy(features, n_nodes, rng)

        # Optional Poincare ball projection
        if self.cfg.use_poincare:
            with StepTimer("poincare_projection", logger):
                embeddings = self._project_poincare(embeddings)

        # Store embedding table
        self._embedding_table = {
            entity_list[i]: embeddings[i] for i in range(n_nodes)
        }
        self._default_embedding = np.zeros(self.cfg.embedding_dim, dtype=np.float32)

        self._fitted = True
        logger.info(
            "GraphEmbeddingGenerator fitted: %d entities, dim=%d, "
            "layers=%d, poincare=%s",
            n_nodes, self.cfg.embedding_dim, self.cfg.num_layers,
            self.cfg.use_poincare,
        )
        return self

    def generate(self, df: Any, **context: Any) -> Any:
        """Look up learned embeddings and compute norm + depth features."""
        if not self._fitted:
            raise RuntimeError(
                "GraphEmbeddingGenerator must be fitted before generate()."
            )

        pdf = df_backend.to_pandas(df) if not isinstance(df, pd.DataFrame) else df
        n_rows = len(pdf)

        embeddings = np.zeros(
            (n_rows, self.cfg.embedding_dim), dtype=np.float32
        )
        norms = np.zeros(n_rows, dtype=np.float32)
        depths = np.zeros(n_rows, dtype=np.float32)

        # Resolve entity keys
        if self.entity_column in pdf.columns:
            entity_keys = pdf[self.entity_column].values
        else:
            entity_keys = pdf.index.values

        for i, key in enumerate(entity_keys):
            emb = self._embedding_table.get(key, self._default_embedding)
            embeddings[i] = emb
            norm = float(np.linalg.norm(emb))
            norms[i] = norm
            # Hyperbolic distance from origin ≈ atanh(norm) / sqrt(curvature)
            depths[i] = float(np.arctanh(min(norm, 0.999))) / max(
                math.sqrt(self.cfg.curvature), 1e-6
            )

        data = {}
        for j in range(self.cfg.embedding_dim):
            data[f"{self.prefix}_d{j}"] = embeddings[:, j]
        data[f"{self.prefix}_norm"] = norms
        data[f"{self.prefix}_depth"] = depths

        return df_backend.from_dict(data, index=pdf.index)

    # ==================================================================
    # Private -- graph construction
    # ==================================================================

    def _extract_entities_features(
        self, pdf: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (unique_entities, feature_matrix)."""
        if self.entity_column in pdf.columns:
            entities = pdf[self.entity_column].unique()
        else:
            entities = pdf.index.unique().values

        # Feature columns for kNN
        if self.feature_columns:
            fcols = [c for c in self.feature_columns if c in pdf.columns]
        else:
            fcols = pdf.select_dtypes(include=["number"]).columns.tolist()
            # Exclude entity column if numeric
            if self.entity_column in fcols:
                fcols.remove(self.entity_column)

        # Build per-entity feature vectors (mean aggregation for duplicates)
        if self.entity_column in pdf.columns:
            grouped = pdf.groupby(self.entity_column)[fcols].mean()
            # Reorder to match entities
            features = grouped.reindex(entities).fillna(0.0).values.astype(np.float32)
        else:
            features = pdf[fcols].values.astype(np.float32)

        return entities, features

    def _build_knn_adjacency_numpy(
        self, features: np.ndarray, k: int
    ) -> np.ndarray:
        """Build a symmetric kNN adjacency matrix using cosine similarity.

        Returns a dense ``[N, N]`` float32 normalised adjacency with self-loops.
        """
        n = features.shape[0]
        k = min(k, n - 1)

        # Normalise for cosine similarity
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normed = features / norms

        # Cosine similarity matrix
        sim = normed @ normed.T  # [N, N]

        # Zero out self-similarity
        np.fill_diagonal(sim, -np.inf)

        # kNN: keep top-k per row
        adj = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            topk_idx = np.argpartition(sim[i], -k)[-k:]
            adj[i, topk_idx] = 1.0

        # Symmetrise
        adj = np.maximum(adj, adj.T)

        # Add self-loops
        adj += np.eye(n, dtype=np.float32)

        # Normalise: D^{-1/2} A D^{-1/2}
        deg = adj.sum(axis=1)
        deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
        adj = adj * deg_inv_sqrt[:, None] * deg_inv_sqrt[None, :]

        return adj

    def _sample_bpr_edges(
        self,
        adj_dense: np.ndarray,
        n_edges: int,
        rng: np.random.RandomState,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample positive and negative edge pairs for BPR loss.

        Returns (pos_edges, neg_edges) each ``[E, 2]``.
        """
        n = adj_dense.shape[0]
        rows, cols = np.where(adj_dense > 0)
        # Remove self-loops from positives
        mask = rows != cols
        rows, cols = rows[mask], cols[mask]

        if len(rows) == 0:
            # Fallback: random edges
            rows = rng.randint(0, n, size=n_edges)
            cols = rng.randint(0, n, size=n_edges)
            neg_cols = rng.randint(0, n, size=n_edges)
            return (
                np.stack([rows, cols], axis=1).astype(np.int64),
                np.stack([rows, neg_cols], axis=1).astype(np.int64),
            )

        # Sample positive edges
        idx = rng.choice(len(rows), size=min(n_edges, len(rows)), replace=True)
        pos = np.stack([rows[idx], cols[idx]], axis=1).astype(np.int64)

        # Negative edges: replace destination with random node
        neg_dst = rng.randint(0, n, size=len(pos))
        neg = np.stack([pos[:, 0], neg_dst], axis=1).astype(np.int64)

        return pos, neg

    # ==================================================================
    # Private -- torch training
    # ==================================================================

    def _fit_torch(
        self,
        features: np.ndarray,
        n_nodes: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """Train LightGCN embeddings using torch (GPU or CPU)."""
        device = self._device

        with StepTimer("build_knn_graph", logger):
            adj_np = self._build_knn_adjacency_numpy(features, self.cfg.k_neighbors)

        with StepTimer("prepare_torch_tensors", logger):
            adj_t = torch.tensor(adj_np, dtype=torch.float32, device=device)
            model = _GraphEmbeddingModel(n_nodes, self.cfg).to(device)
            optimiser = torch.optim.Adam(
                model.parameters(), lr=self.cfg.learning_rate
            )

        # Determine batch size for edge sampling
        n_edges_per_epoch = min(n_nodes * self.cfg.k_neighbors, 50000)
        batch_size = adaptive_batch_size(
            total_samples=n_edges_per_epoch,
            base_batch=self.cfg.base_batch_size,
            bytes_per_sample=self.cfg.embedding_dim * 4 * 3,  # 3 vecs per edge
        )

        device_str = str(device)
        use_amp = device_str.startswith("cuda")
        if use_amp:
            try:
                scaler = torch.amp.GradScaler()
            except TypeError:
                scaler = torch.cuda.amp.GradScaler()
        else:
            scaler = None

        with StepTimer("lightgcn_training", logger):
            self._train_embeddings(
                model=model,
                adj_t=adj_t,
                adj_np=adj_np,
                n_nodes=n_nodes,
                n_edges=n_edges_per_epoch,
                batch_size=batch_size,
                device=device,
                rng=rng,
                optimiser=optimiser,
                scaler=scaler,
                use_amp=use_amp,
            )

        # Extract final embeddings
        model.eval()
        with torch.no_grad():
            embs = model(adj_t).detach().cpu().numpy()

        return embs.astype(np.float32)

    def _train_embeddings(
        self,
        model,
        adj_t,
        adj_np: np.ndarray,
        n_nodes: int,
        n_edges: int,
        batch_size: int,
        device: str,
        rng: np.random.RandomState,
        optimiser,
        scaler,
        use_amp: bool,
    ) -> None:
        """Run the training loop with OOM retry."""
        model.train()
        max_retries = self.cfg.max_retries

        for epoch in range(self.cfg.num_epochs):
            pos_edges_np, neg_edges_np = self._sample_bpr_edges(
                adj_np, n_edges, rng
            )

            for attempt in range(max_retries):
                try:
                    pos_edges = torch.tensor(
                        pos_edges_np, dtype=torch.long, device=device
                    )
                    neg_edges = torch.tensor(
                        neg_edges_np, dtype=torch.long, device=device
                    )

                    optimiser.zero_grad()

                    if use_amp:
                        with torch.amp.autocast("cuda"):
                            if self.cfg.use_bpr:
                                loss = model.bpr_loss(adj_t, pos_edges, neg_edges)
                            else:
                                loss = model.reconstruction_loss(adj_t)
                        scaler.scale(loss).backward()
                        scaler.step(optimiser)
                        scaler.update()
                    else:
                        if self.cfg.use_bpr:
                            loss = model.bpr_loss(adj_t, pos_edges, neg_edges)
                        else:
                            loss = model.reconstruction_loss(adj_t)
                        loss.backward()
                        optimiser.step()

                    break  # success

                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and torch is not None:
                        torch.cuda.empty_cache()
                        batch_size = max(64, batch_size // 2)
                        n_edges = max(1000, n_edges // 2)
                        logger.warning(
                            "OOM on epoch %d attempt %d, reducing batch to %d, edges to %d",
                            epoch, attempt + 1, batch_size, n_edges,
                        )
                        if attempt == max_retries - 1:
                            raise
                    else:
                        raise

            if epoch % 10 == 0 or epoch == self.cfg.num_epochs - 1:
                logger.info(
                    "  epoch %d/%d  loss=%.4f",
                    epoch + 1, self.cfg.num_epochs, loss.item(),
                )

    # ==================================================================
    # Private -- numpy fallback
    # ==================================================================

    def _fit_numpy(
        self,
        features: np.ndarray,
        n_nodes: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """Approximate graph embeddings via truncated SVD of adjacency matrix.

        This is the last-resort fallback when torch is not installed.
        """
        logger.info("GraphEmbeddingGenerator: using numpy-only fallback (SVD)")

        with StepTimer("build_knn_graph", logger):
            adj = self._build_knn_adjacency_numpy(features, self.cfg.k_neighbors)

        with StepTimer("svd_embedding", logger):
            # Truncated SVD of the adjacency matrix
            try:
                from numpy.linalg import svd

                U, S, Vt = svd(adj, full_matrices=False)
                k = min(self.cfg.embedding_dim, U.shape[1])
                embeddings = U[:, :k] * np.sqrt(S[:k])[None, :]
            except np.linalg.LinAlgError:
                logger.warning("SVD failed, using random initialisation")
                embeddings = rng.randn(n_nodes, self.cfg.embedding_dim).astype(
                    np.float32
                )
                embeddings *= 0.1

            # Pad or truncate to embedding_dim
            if embeddings.shape[1] < self.cfg.embedding_dim:
                pad = np.zeros(
                    (n_nodes, self.cfg.embedding_dim - embeddings.shape[1]),
                    dtype=np.float32,
                )
                embeddings = np.concatenate([embeddings, pad], axis=1)
            elif embeddings.shape[1] > self.cfg.embedding_dim:
                embeddings = embeddings[:, : self.cfg.embedding_dim]

        return embeddings.astype(np.float32)

    # ==================================================================
    # Private -- Poincare ball projection
    # ==================================================================

    def _project_poincare(self, embeddings: np.ndarray) -> np.ndarray:
        """Project embeddings onto the Poincare ball (norm < 1).

        Uses the exponential map at the origin to ensure all points
        lie strictly inside the unit ball.
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)

        # tanh squashing ensures norm < 1
        target_norms = np.tanh(norms * 0.5)
        projected = embeddings / norms * target_norms

        return projected.astype(np.float32)
