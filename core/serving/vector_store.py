"""
Serving-Layer Vector Store (Stage C Production)
=================================================

Production vector store for real-time customer context retrieval
during recommendation serving. Provides ANN (Approximate Nearest
Neighbor) search over customer embeddings with sub-10ms latency.

Extends the offline :class:`core.recommendation.reason.context_store.ContextVectorStore`
with production features: sharding, replication, and incremental updates.

This is a STUB -- production implementation is not needed for ablation.
Use :class:`core.recommendation.reason.context_store.ContextVectorStore`
for offline evaluation instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


__all__ = ["ServingVectorStore", "SearchResult"]


@dataclass
class SearchResult:
    """Single ANN search result."""

    customer_id: str
    score: float
    metadata: Dict[str, Any]


class ServingVectorStore:
    """Production vector store for real-time context retrieval.

    This is a Stage C production module. Not required for ablation.

    Supported backends (production):
        - LanceDB with IVF-PQ index
        - FAISS with HNSW index
        - OpenSearch k-NN plugin

    Args:
        backend: Vector DB backend identifier.
        store_path: Path to persistent store directory.
        config: Backend-specific configuration.
    """

    def __init__(
        self,
        backend: str = "lancedb",
        store_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        raise NotImplementedError(
            "ServingVectorStore is a Stage C production module. "
            "Use core.recommendation.reason.context_store.ContextVectorStore "
            "for offline ablation evaluation."
        )

    def build_index(
        self,
        customer_ids: List[str],
        embeddings: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Build ANN index from customer embeddings."""
        raise NotImplementedError

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Find k nearest customers to query vector."""
        raise NotImplementedError

    def upsert(
        self,
        customer_ids: List[str],
        embeddings: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """Incrementally update/insert customer embeddings."""
        raise NotImplementedError

    def delete(self, customer_ids: List[str]) -> int:
        """Remove customers from the index."""
        raise NotImplementedError

    def stats(self) -> Dict[str, Any]:
        """Return index statistics (size, dimension, backend info)."""
        raise NotImplementedError
