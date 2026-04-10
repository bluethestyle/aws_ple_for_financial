"""
Diagnostic Case Store — LanceDB Knowledge Base
=================================================

Stores diagnostic cases (ops/audit findings) for:
    1. Similar case search (vector similarity)
    2. Statistical analysis (structured metadata queries)
    3. Resolution tracking (problem → action → outcome)

Same backend pattern as ContextVectorStore:
    - LanceDB preferred (ANN search)
    - numpy fallback (brute-force cosine)

Case schema includes consensus_type for minority report tracking.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["DiagnosticCaseStore"]


class DiagnosticCaseStore:
    """Store and retrieve diagnostic cases for agent knowledge base.

    Supports LanceDB (if available) or numpy fallback for similarity search.

    Args:
        store_path: Directory for persistence.
        backend: ``"lancedb"`` | ``"numpy"`` | ``"auto"``.
                 ``"auto"`` tries LanceDB first, then falls back to numpy.
        embedding_dim: Dimension of text embeddings (default 384 for MiniLM).
    """

    def __init__(
        self,
        store_path: str = "diagnostic_cases/",
        backend: str = "auto",
        embedding_dim: int = 384,
    ) -> None:
        self._backend = self._select_backend(backend)
        self._store_path = Path(store_path)
        self._store_path.mkdir(parents=True, exist_ok=True)
        self._embedding_dim = embedding_dim

        # numpy backend state
        self._cases: List[Dict[str, Any]] = []
        self._vectors: Optional[np.ndarray] = None

        # lancedb backend state
        self._db = None
        self._table = None

        logger.info(
            "DiagnosticCaseStore: backend=%s, path=%s",
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
                logger.info("LanceDB not installed, falling back to numpy backend.")
                return "numpy"
        return backend

    @property
    def backend(self) -> str:
        """The active backend name."""
        return self._backend

    # ------------------------------------------------------------------
    # Save case
    # ------------------------------------------------------------------

    def save_case(
        self,
        case: Dict[str, Any],
        vector: Optional[np.ndarray] = None,
    ) -> str:
        """Save a diagnostic case.

        Args:
            case: Case dict with keys: case_id, timestamp, agent, pipeline_part,
                  check_item, verdict, severity, finding, likely_cause,
                  suggested_action, metrics, consensus_type, consensus_detail,
                  resolution, resolved_at, post_resolution_verdict.
            vector: Pre-computed embedding vector. If None, case is stored
                    without vector (metadata-only; excluded from similarity search).

        Returns:
            case_id of the saved case.
        """
        # Auto-generate case_id if not provided
        if "case_id" not in case:
            agent = case.get("agent", "UNK").upper()
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M%S")
            case["case_id"] = f"{agent}-{ts}-{len(self._cases):03d}"

        if "timestamp" not in case:
            case["timestamp"] = datetime.now(timezone.utc).isoformat()

        if self._backend == "lancedb" and vector is not None:
            self._save_lancedb(case, vector)
        else:
            self._save_numpy(case, vector)

        logger.debug("Saved case %s", case["case_id"])
        return case["case_id"]

    def _save_numpy(self, case: Dict[str, Any], vector: Optional[np.ndarray]) -> None:
        self._cases.append(case)
        if vector is not None:
            vec = vector.reshape(1, -1)
            if self._vectors is None:
                self._vectors = vec
            else:
                self._vectors = np.vstack([self._vectors, vec])

    def _save_lancedb(self, case: Dict[str, Any], vector: np.ndarray) -> None:
        try:
            import lancedb  # noqa: F401
        except ImportError:
            self._save_numpy(case, vector)
            return

        if self._db is None:
            import lancedb as _lancedb

            self._db = _lancedb.connect(str(self._store_path / "lancedb"))

        record = {
            "case_id": case["case_id"],
            "vector": vector.tolist(),
            "metadata_json": json.dumps(case, default=str, ensure_ascii=False),
        }

        if self._table is None:
            try:
                self._table = self._db.open_table("diagnostic_cases")
                self._table.add([record])
            except Exception:
                self._table = self._db.create_table("diagnostic_cases", [record])
        else:
            self._table.add([record])

        # Also keep in memory for statistics / resolution updates
        self._cases.append(case)

    # ------------------------------------------------------------------
    # Search similar cases
    # ------------------------------------------------------------------

    def search_similar(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        pipeline_part: Optional[str] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Find similar diagnostic cases by vector similarity.

        Args:
            query_vector: Embedding of the current finding text.
            k: Number of similar cases to return.
            pipeline_part: Optional filter to same pipeline part (e.g. ``"P1"``).

        Returns:
            List of ``(case_dict, similarity_score)`` tuples, descending similarity.
        """
        if self._backend == "lancedb" and self._table is not None:
            return self._search_lancedb(query_vector, k, pipeline_part)
        return self._search_numpy(query_vector, k, pipeline_part)

    def _search_numpy(
        self,
        query: np.ndarray,
        k: int,
        part_filter: Optional[str],
    ) -> List[Tuple[Dict[str, Any], float]]:
        if self._vectors is None or len(self._cases) == 0:
            return []

        # Normalize query
        query_norm = query / (np.linalg.norm(query) + 1e-8)

        # Compute cosine similarity
        norms = np.linalg.norm(self._vectors, axis=1, keepdims=True) + 1e-8
        normalized = self._vectors / norms
        similarities = (normalized @ query_norm.reshape(-1, 1)).flatten()

        # Apply pipeline_part filter
        indices = list(range(len(self._cases)))
        if part_filter:
            indices = [
                i for i in indices
                if self._cases[i].get("pipeline_part") == part_filter
            ]

        # Sort by similarity descending
        scored = [
            (i, similarities[i])
            for i in indices
            if i < len(similarities)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            (self._cases[idx], round(float(score), 4))
            for idx, score in scored[:k]
        ]

    def _search_lancedb(
        self,
        query: np.ndarray,
        k: int,
        part_filter: Optional[str],
    ) -> List[Tuple[Dict[str, Any], float]]:
        try:
            # Over-fetch when filtering so we can trim after
            limit = k * 3 if part_filter else k
            raw_results = (
                self._table.search(query.tolist())
                .limit(limit)
                .to_list()
            )

            results: List[Tuple[Dict[str, Any], float]] = []
            for row in raw_results:
                case = json.loads(row.get("metadata_json", "{}"))
                if part_filter and case.get("pipeline_part") != part_filter:
                    continue
                # LanceDB returns L2 distance by default; convert to [0, 1] similarity
                dist = row.get("_distance", 1.0)
                similarity = max(0.0, 1.0 - dist)
                results.append((case, round(similarity, 4)))
                if len(results) >= k:
                    break
            return results
        except Exception as exc:
            logger.warning("LanceDB search failed, falling back to numpy: %s", exc)
            return self._search_numpy(query, k, part_filter)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(
        self,
        pipeline_part: Optional[str] = None,
        check_item: Optional[str] = None,
        period_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Get aggregated case statistics.

        Args:
            pipeline_part: Filter by pipeline part label (e.g. ``"P1"``–``"P6"``).
            check_item: Filter by specific checklist item name.
            period_days: Only include cases from the last *N* days.
                         Requires ``timestamp`` field in ISO-8601 format.

        Returns:
            Dict with keys: total_cases, by_verdict, by_consensus_type,
            resolved_count, resolution_rate.
        """
        filtered = self._cases

        if pipeline_part:
            filtered = [c for c in filtered if c.get("pipeline_part") == pipeline_part]
        if check_item:
            filtered = [c for c in filtered if c.get("check_item") == check_item]
        if period_days is not None:
            cutoff = datetime.now(timezone.utc).timestamp() - period_days * 86400
            kept = []
            for c in filtered:
                ts_str = c.get("timestamp")
                if ts_str:
                    try:
                        ts = datetime.fromisoformat(ts_str).timestamp()
                        if ts >= cutoff:
                            kept.append(c)
                    except ValueError:
                        kept.append(c)  # malformed timestamp — keep conservatively
                else:
                    kept.append(c)
            filtered = kept

        verdict_counts: Dict[str, int] = defaultdict(int)
        consensus_counts: Dict[str, int] = defaultdict(int)
        resolution_count = 0

        for c in filtered:
            verdict_counts[c.get("verdict", "unknown")] += 1
            consensus_counts[c.get("consensus_type", "none")] += 1
            if c.get("resolution"):
                resolution_count += 1

        total = len(filtered)
        return {
            "total_cases": total,
            "by_verdict": dict(verdict_counts),
            "by_consensus_type": dict(consensus_counts),
            "resolved_count": resolution_count,
            "resolution_rate": round(resolution_count / total, 4) if total else 0.0,
        }

    # ------------------------------------------------------------------
    # Resolution tracking
    # ------------------------------------------------------------------

    def update_resolution(
        self,
        case_id: str,
        resolution: str,
        resolved_at: Optional[str] = None,
        post_resolution_verdict: Optional[str] = None,
    ) -> bool:
        """Update a case with resolution information.

        Args:
            case_id: The case identifier to update.
            resolution: Free-text description of action taken / outcome.
            resolved_at: ISO-8601 timestamp; defaults to current UTC time.
            post_resolution_verdict: Optional re-evaluated verdict after fix.

        Returns:
            ``True`` if the case was found and updated, ``False`` otherwise.
        """
        if not resolved_at:
            resolved_at = datetime.now(timezone.utc).isoformat()

        for case in self._cases:
            if case.get("case_id") == case_id:
                case["resolution"] = resolution
                case["resolved_at"] = resolved_at
                if post_resolution_verdict is not None:
                    case["post_resolution_verdict"] = post_resolution_verdict
                logger.info("Updated resolution for case %s", case_id)
                return True

        logger.warning("Case %s not found for resolution update", case_id)
        return False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> None:
        """Persist the store to disk.

        Cases are saved as JSONL; vectors as ``.npy``.
        LanceDB data is already persistent on write; JSONL is saved alongside
        for metadata-only cases and resolution updates.
        """
        save_dir = Path(path) if path else self._store_path
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save cases as JSONL (both backends)
        cases_path = save_dir / "cases.jsonl"
        with open(cases_path, "w", encoding="utf-8") as f:
            for case in self._cases:
                f.write(json.dumps(case, default=str, ensure_ascii=False) + "\n")

        # Save numpy vectors when using numpy backend
        if self._backend == "numpy" and self._vectors is not None:
            np.save(str(save_dir / "vectors.npy"), self._vectors)

        logger.info(
            "DiagnosticCaseStore saved: %d cases to %s",
            len(self._cases),
            save_dir,
        )

    def load(self, path: Optional[str] = None) -> None:
        """Load a previously saved store.

        Args:
            path: Directory to load from (defaults to ``store_path``).
        """
        load_dir = Path(path) if path else self._store_path

        # Load cases from JSONL
        cases_path = load_dir / "cases.jsonl"
        if cases_path.exists():
            self._cases = []
            with open(cases_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._cases.append(json.loads(line))

        # Load numpy vectors
        vectors_path = load_dir / "vectors.npy"
        if vectors_path.exists():
            self._vectors = np.load(str(vectors_path))

        # Re-connect LanceDB table
        if self._backend == "lancedb":
            try:
                import lancedb

                self._db = lancedb.connect(str(load_dir / "lancedb"))
                self._table = self._db.open_table("diagnostic_cases")
            except Exception as exc:
                logger.warning("Could not open LanceDB table: %s", exc)
                self._table = None

        logger.info(
            "DiagnosticCaseStore loaded: %d cases from %s",
            len(self._cases),
            load_dir,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def case_count(self) -> int:
        """Number of cases currently held in memory."""
        return len(self._cases)

    def get_case(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Return the full case dict for *case_id*, or ``None`` if not found."""
        for case in self._cases:
            if case.get("case_id") == case_id:
                return case
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Return basic store-level statistics (mirrors ContextVectorStore API)."""
        return {
            "backend": self._backend,
            "case_count": self.case_count,
            "has_vectors": self._vectors is not None,
            "vector_dim": (
                self._vectors.shape[1]
                if self._vectors is not None and self._vectors.ndim == 2
                else self._embedding_dim
            ),
        }
