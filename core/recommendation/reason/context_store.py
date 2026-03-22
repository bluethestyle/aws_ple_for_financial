"""
Customer Context Vector Store for Retrieval-Augmented Grounding
================================================================

Lightweight vector store abstraction for storing and retrieving customer
context vectors used by the LLM grounding layer in reason generation.

Supports two backends:
    - **LanceDB** (preferred): columnar vector DB with ANN indexing.
    - **numpy** (fallback): brute-force cosine similarity on in-memory arrays.

The ``"auto"`` backend tries LanceDB first, then falls back to numpy.

Reference:
    Adapted from ``gotothemoon/workspace/code/src/grounding/context_vector_store.py``
    (LanceContextVectorStore) for the AWS pipeline.

Usage::

    store = ContextVectorStore(store_path="context_store/", backend="auto")
    store.build(customer_ids, feature_vectors, metadata)
    results = store.search(query_vector, k=10)
    store.save()
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["ContextVectorStore"]


class ContextVectorStore:
    """Store and retrieve customer context vectors for LLM grounding.

    Supports LanceDB (if available) or numpy fallback for similarity search.

    Args:
        store_path: Directory where the vector store persists its data.
        backend: ``"lancedb"`` | ``"numpy"`` | ``"auto"``.
                 ``"auto"`` tries LanceDB first, then falls back to numpy.
    """

    def __init__(
        self,
        store_path: str = "context_store/",
        backend: str = "auto",
    ) -> None:
        self._backend = self._select_backend(backend)
        self._store_path = Path(store_path)
        self._store_path.mkdir(parents=True, exist_ok=True)

        # numpy backend state
        self._ids: Optional[np.ndarray] = None
        self._vectors: Optional[np.ndarray] = None
        self._metadata: Dict[str, Dict[str, Any]] = {}

        # lancedb backend state
        self._db = None
        self._table = None

        logger.info(
            "ContextVectorStore initialised: backend=%s, path=%s",
            self._backend,
            self._store_path,
        )

    # ------------------------------------------------------------------
    # Backend selection
    # ------------------------------------------------------------------

    @staticmethod
    def _select_backend(backend: str) -> str:
        """Choose backend, falling back to numpy when LanceDB is unavailable."""
        if backend == "auto":
            try:
                import lancedb  # noqa: F401

                return "lancedb"
            except ImportError:
                logger.info(
                    "LanceDB not installed, falling back to numpy backend."
                )
                return "numpy"
        return backend

    @property
    def backend(self) -> str:
        """The active backend name."""
        return self._backend

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        customer_ids: Any,
        feature_vectors: np.ndarray,
        metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """Build the vector store from customer feature vectors.

        Args:
            customer_ids: Array-like of customer identifiers.
            feature_vectors: ``(n_customers, dim)`` numpy array of embeddings.
            metadata: Optional ``{customer_id: {key: value, ...}}`` context
                      info associated with each customer.
        """
        if self._backend == "lancedb":
            self._build_lancedb(customer_ids, feature_vectors, metadata)
        else:
            self._build_numpy(customer_ids, feature_vectors, metadata)

        logger.info(
            "ContextVectorStore.build: %d customers, dim=%d, backend=%s",
            len(customer_ids),
            feature_vectors.shape[1] if feature_vectors.ndim == 2 else 0,
            self._backend,
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Find the *k* most similar customers to *query_vector*.

        Args:
            query_vector: 1-D numpy array of the same dimensionality as the
                          stored vectors.
            k: Number of neighbours to return.

        Returns:
            List of ``(customer_id, similarity_score, metadata_dict)`` tuples
            ordered by descending similarity.
        """
        if self._backend == "lancedb":
            return self._search_lancedb(query_vector, k)
        return self._search_numpy(query_vector, k)

    # ------------------------------------------------------------------
    # Get context for a specific customer
    # ------------------------------------------------------------------

    def get_context(self, customer_id: str) -> Dict[str, Any]:
        """Retrieve full context for a specific customer.

        Args:
            customer_id: The customer identifier.

        Returns:
            Metadata dict for the customer, or empty dict if not found.
        """
        if self._backend == "lancedb":
            return self._get_context_lancedb(customer_id)
        return self._metadata.get(str(customer_id), {})

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> None:
        """Persist the store to disk.

        For the numpy backend this saves ``.npz`` + ``.json`` files.
        LanceDB persists automatically to its ``store_path``.
        """
        save_dir = Path(path) if path else self._store_path
        save_dir.mkdir(parents=True, exist_ok=True)

        if self._backend == "numpy":
            self._save_numpy(save_dir)
        else:
            # LanceDB persists on write; save metadata separately.
            meta_path = save_dir / "metadata.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(self._metadata, f, default=str, ensure_ascii=False)

        logger.info("ContextVectorStore saved to %s", save_dir)

    def load(self, path: Optional[str] = None) -> None:
        """Load a previously saved store.

        Args:
            path: Directory to load from (defaults to ``store_path``).
        """
        load_dir = Path(path) if path else self._store_path

        if self._backend == "numpy":
            self._load_numpy(load_dir)
        else:
            self._load_lancedb(load_dir)

        logger.info("ContextVectorStore loaded from %s", load_dir)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return basic statistics about the store."""
        if self._backend == "numpy" and self._ids is not None:
            return {
                "backend": self._backend,
                "num_customers": len(self._ids),
                "vector_dim": self._vectors.shape[1] if self._vectors is not None else 0,
                "metadata_keys": len(self._metadata),
            }
        if self._backend == "lancedb" and self._table is not None:
            try:
                count = self._table.count_rows()
            except Exception:
                count = -1
            return {
                "backend": self._backend,
                "num_customers": count,
                "metadata_keys": len(self._metadata),
            }
        return {"backend": self._backend, "num_customers": 0}

    # ==================================================================
    # numpy backend implementation
    # ==================================================================

    def _build_numpy(
        self,
        ids: Any,
        vectors: np.ndarray,
        metadata: Optional[Dict[str, Dict[str, Any]]],
    ) -> None:
        """Fallback: store as numpy arrays with brute-force cosine similarity."""
        self._ids = np.array([str(i) for i in ids])
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
        self._vectors = vectors / norms
        self._metadata = metadata or {}

    def _search_numpy(
        self, query: np.ndarray, k: int,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        if self._ids is None or self._vectors is None:
            return []
        query_norm = query / (np.linalg.norm(query) + 1e-8)
        similarities = self._vectors @ query_norm
        actual_k = min(k, len(self._ids))
        top_k_idx = np.argsort(similarities)[-actual_k:][::-1]
        return [
            (
                str(self._ids[i]),
                float(similarities[i]),
                self._metadata.get(str(self._ids[i]), {}),
            )
            for i in top_k_idx
        ]

    def _save_numpy(self, save_dir: Path) -> None:
        if self._ids is not None and self._vectors is not None:
            np.savez(
                save_dir / "vectors.npz",
                ids=self._ids,
                vectors=self._vectors,
            )
        meta_path = save_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, default=str, ensure_ascii=False)

    def _load_numpy(self, load_dir: Path) -> None:
        npz_path = load_dir / "vectors.npz"
        if npz_path.exists():
            data = np.load(npz_path, allow_pickle=True)
            self._ids = data["ids"]
            self._vectors = data["vectors"]
        meta_path = load_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)

    # ==================================================================
    # LanceDB backend implementation
    # ==================================================================

    def _build_lancedb(
        self,
        ids: Any,
        vectors: np.ndarray,
        metadata: Optional[Dict[str, Dict[str, Any]]],
    ) -> None:
        """Build a LanceDB table from customer vectors."""
        import lancedb
        import pyarrow as pa

        self._metadata = metadata or {}
        db = lancedb.connect(str(self._store_path))
        self._db = db

        # Construct records
        records: List[Dict[str, Any]] = []
        for idx, cid in enumerate(ids):
            cid_str = str(cid)
            record = {
                "customer_id": cid_str,
                "vector": vectors[idx].tolist(),
            }
            # Flatten metadata into a JSON string column
            meta = self._metadata.get(cid_str, {})
            record["metadata_json"] = json.dumps(meta, default=str, ensure_ascii=False)
            records.append(record)

        # Create or overwrite table
        try:
            db.drop_table("customer_context", ignore_missing=True)
        except Exception:
            pass
        self._table = db.create_table("customer_context", records)

    def _search_lancedb(
        self, query: np.ndarray, k: int,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        if self._table is None:
            return []
        try:
            results = (
                self._table.search(query.tolist())
                .limit(k)
                .to_list()
            )
            output = []
            for row in results:
                cid = row.get("customer_id", "")
                # LanceDB returns _distance (L2) by default; convert to similarity
                distance = row.get("_distance", 0.0)
                similarity = 1.0 / (1.0 + distance)
                meta = {}
                if "metadata_json" in row:
                    try:
                        meta = json.loads(row["metadata_json"])
                    except (json.JSONDecodeError, TypeError):
                        pass
                output.append((cid, similarity, meta))
            return output
        except Exception as exc:
            logger.warning("LanceDB search failed: %s", exc)
            return []

    def _get_context_lancedb(self, customer_id: str) -> Dict[str, Any]:
        if self._table is None:
            return {}
        try:
            cid_escaped = str(customer_id).replace("'", "''")
            rows = (
                self._table.search()
                .where(f"customer_id = '{cid_escaped}'")
                .limit(1)
                .to_list()
            )
            if rows:
                meta_json = rows[0].get("metadata_json", "{}")
                return json.loads(meta_json)
        except Exception as exc:
            logger.warning(
                "LanceDB get_context failed for %s: %s", customer_id, exc,
            )
        return self._metadata.get(str(customer_id), {})

    def _load_lancedb(self, load_dir: Path) -> None:
        import lancedb

        db = lancedb.connect(str(load_dir))
        self._db = db
        try:
            self._table = db.open_table("customer_context")
        except Exception as exc:
            logger.warning("Could not open LanceDB table: %s", exc)
            self._table = None

        meta_path = load_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)
