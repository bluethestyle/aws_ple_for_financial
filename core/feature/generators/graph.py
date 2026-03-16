"""
Graph Embedding Feature Generator -- Hyperbolic / Poincare ball embeddings.

Generates dense vector representations of entities (users, products) by
embedding a relational graph into hyperbolic space.  Hyperbolic embeddings
are particularly well-suited for financial product hierarchies because:

* **Hierarchical structure**: Financial products have natural tree-like
  taxonomies (asset class -> sub-class -> instrument type -> specific product).
  Hyperbolic space can embed trees with arbitrarily low distortion, unlike
  Euclidean space.
* **Power-law degree distributions**: User-product interaction graphs
  follow power-law patterns that hyperbolic geometry naturally captures.
* **Poincare ball model**: Provides a differentiable manifold amenable
  to gradient-based optimisation.

This is a **placeholder implementation** that generates synthetic embeddings.
A production implementation would use ``geoopt`` or a custom Riemannian SGD
optimiser on actual interaction graphs.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.data.dataframe import df_backend
from ..generator import AbstractFeatureGenerator, FeatureGeneratorRegistry

logger = logging.getLogger(__name__)


@FeatureGeneratorRegistry.register(
    "hyperbolic_embedding",
    description="Hyperbolic (Poincare ball) graph embedding features.",
    tags=["graph", "embedding", "hyperbolic", "relational"],
)
class GraphEmbeddingGenerator(AbstractFeatureGenerator):
    """Generate hyperbolic graph embedding features.

    Embeds entities from a relational graph into the Poincare ball model
    of hyperbolic space.  Each entity receives a dense vector whose
    components capture hierarchical and relational structure.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of the Poincare ball embedding.
    entity_column : str
        Column identifying the entity to embed (e.g. ``"user_id"``).
    interaction_columns : list[str]
        Columns representing interaction signals used to construct
        the graph (e.g. product holdings, transaction types).
    curvature : float
        Negative curvature parameter of the Poincare ball (``-c``).
        Larger values (more negative curvature) allocate more
        representational capacity to hierarchical depth.
    n_negative_samples : int
        Number of negative samples per positive edge in contrastive
        learning.
    learning_rate : float
        Riemannian SGD learning rate.
    n_epochs : int
        Training epochs for the embedding model.
    prefix : str
        Column name prefix for generated features.
    random_state : int
        Random seed.
    """

    def __init__(
        self,
        embedding_dim: int = 16,
        entity_column: str = "user_id",
        interaction_columns: Optional[List[str]] = None,
        curvature: float = 1.0,
        n_negative_samples: int = 5,
        learning_rate: float = 0.01,
        n_epochs: int = 50,
        prefix: str = "graph_hyp",
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.entity_column = entity_column
        self.interaction_columns = interaction_columns or []
        self.curvature = curvature
        self.n_negative_samples = n_negative_samples
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.prefix = prefix
        self.random_state = random_state

        # Fitted state
        self._embedding_table: Optional[Dict[Any, np.ndarray]] = None
        self._default_embedding: Optional[np.ndarray] = None

    # -- Output description --------------------------------------------

    @property
    def output_dim(self) -> int:
        """Embedding dimensionality plus 2 derived features.

        Output = embedding_dim + hyperbolic_norm + hierarchy_depth.
        """
        return self.embedding_dim + 2

    @property
    def output_columns(self) -> List[str]:
        """Generated column names."""
        cols = [f"{self.prefix}_d{i}" for i in range(self.embedding_dim)]
        cols.append(f"{self.prefix}_norm")
        cols.append(f"{self.prefix}_depth")
        return cols

    # -- Core API ------------------------------------------------------

    def fit(self, df: Any, **context: Any) -> "GraphEmbeddingGenerator":
        """Learn entity embeddings from the training graph.

        .. note::
           Placeholder: assigns random embeddings inside the Poincare
           ball to each unique entity.  Replace with actual Riemannian
           SGD optimisation (e.g. ``geoopt``) for production use.
        """
        pdf = df_backend.to_pandas(df) if not isinstance(df, pd.DataFrame) else df
        rng = np.random.RandomState(self.random_state)

        # Collect unique entities
        if self.entity_column in pdf.columns:
            entities = pdf[self.entity_column].unique()
        else:
            entities = pdf.index.unique()

        # Generate random embeddings inside the Poincare ball (norm < 1)
        self._embedding_table = {}
        for entity in entities:
            # Sample from a Gaussian, then project into the ball
            raw = rng.randn(self.embedding_dim).astype(np.float32)
            # Ensure norm < 1 (Poincare ball constraint)
            norm = np.linalg.norm(raw)
            if norm > 0:
                # Scale to a random radius < 1
                target_norm = rng.uniform(0.1, 0.95)
                raw = raw / norm * target_norm
            self._embedding_table[entity] = raw

        # Default embedding for unseen entities (near origin = generic)
        self._default_embedding = np.zeros(self.embedding_dim, dtype=np.float32)

        self._fitted = True
        logger.info(
            "GraphEmbeddingGenerator fitted: %d entities, "
            "embedding_dim=%d, curvature=%.2f",
            len(entities), self.embedding_dim, self.curvature,
        )
        return self

    def generate(self, df: Any, **context: Any) -> Any:
        """Look up or compute hyperbolic embeddings for each row.

        For each row, retrieves the pre-computed embedding for the
        entity and computes two derived features:
        * **norm**: Poincare ball norm (distance from origin).  Entities
          closer to the boundary are more "specific" in the hierarchy.
        * **depth**: estimated hierarchical depth (log-transform of norm).
        """
        if not self._fitted:
            raise RuntimeError(
                "GraphEmbeddingGenerator must be fitted before generate()."
            )

        pdf = df_backend.to_pandas(df) if not isinstance(df, pd.DataFrame) else df
        n_rows = len(pdf)
        embeddings = np.zeros(
            (n_rows, self.embedding_dim), dtype=np.float32
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
            # Hyperbolic distance from origin = atanh(norm) / sqrt(curvature)
            # Approximated as hierarchy depth
            depths[i] = float(np.arctanh(min(norm, 0.999))) / max(
                np.sqrt(self.curvature), 1e-6
            )

        # Assemble result DataFrame
        data = {}
        for i in range(self.embedding_dim):
            data[f"{self.prefix}_d{i}"] = embeddings[:, i]
        data[f"{self.prefix}_norm"] = norms
        data[f"{self.prefix}_depth"] = depths

        return df_backend.from_dict(data, index=pdf.index)
