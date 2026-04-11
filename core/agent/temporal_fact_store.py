"""
Temporal Fact Store — Time-Valid Facts with Snapshot Queries
===============================================================

LanceDB-backed store for facts with temporal validity ranges.
Enables point-in-time snapshot queries for audit evidence:
    "What facts were valid for customer A at 2026-03-15?"

Same backend pattern as DiagnosticCaseStore (LanceDB + numpy fallback).
Facts are never deleted — expire_fact() sets valid_to instead.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["TemporalFactStore"]


class TemporalFactStore:
    """Store and query facts with temporal validity ranges.

    Schema:
        fact_id: str (uuid)
        entity_type: str (e.g., "customer", "model", "recommendation")
        entity_id: str
        attribute: str (e.g., "segment", "version", "verdict")
        value: str (JSON-serialized)
        valid_from: str (ISO UTC)
        valid_to: Optional[str] (ISO UTC; None = currently valid)
        source: str (e.g., "pipeline", "agent", "operator")
        vector: Optional[List[float]] (embedding of natural-language description)

    Args:
        store_path: Directory for persistence.
        backend: "lancedb" | "numpy" | "auto".
        embedding_dim: Dimension of text embeddings (default 384).
    """

    def __init__(
        self,
        store_path: str = "temporal_facts/",
        backend: str = "auto",
        embedding_dim: int = 384,
    ) -> None:
        self._backend = self._select_backend(backend)
        self._store_path = Path(store_path)
        self._store_path.mkdir(parents=True, exist_ok=True)
        self._embedding_dim = embedding_dim

        # In-memory cache (populated for both backends)
        self._facts: List[Dict[str, Any]] = []

        # LanceDB state
        self._db = None
        self._table = None

        logger.info("TemporalFactStore: backend=%s, path=%s", self._backend, self._store_path)

    @staticmethod
    def _select_backend(backend: str) -> str:
        if backend == "auto":
            try:
                import lancedb  # noqa
                return "lancedb"
            except ImportError:
                return "numpy"
        return backend

    def save_fact(self, fact: Dict[str, Any], vector: Optional[np.ndarray] = None) -> str:
        """Save a fact with auto-generated fact_id and timestamps.

        Args:
            fact: Dict with entity_type, entity_id, attribute, value, source.
            vector: Optional embedding of natural-language description.

        Returns:
            fact_id of the saved fact.
        """
        if "fact_id" not in fact:
            fact["fact_id"] = str(uuid.uuid4())
        if "valid_from" not in fact:
            fact["valid_from"] = datetime.now(timezone.utc).isoformat()
        if "valid_to" not in fact:
            fact["valid_to"] = None

        self._facts.append(fact)

        if self._backend == "lancedb" and vector is not None:
            self._save_lancedb(fact, vector)

        logger.debug("Saved fact %s: %s=%s", fact["fact_id"], fact.get("attribute"), fact.get("value"))
        return fact["fact_id"]

    def _save_lancedb(self, fact: Dict[str, Any], vector: np.ndarray) -> None:
        try:
            import lancedb
        except ImportError:
            return

        if self._db is None:
            self._db = lancedb.connect(str(self._store_path / "lancedb"))

        record = {
            "fact_id": fact["fact_id"],
            "vector": vector.tolist(),
            "metadata_json": json.dumps(fact, default=str, ensure_ascii=False),
        }

        if self._table is None:
            try:
                self._table = self._db.open_table("temporal_facts")
                self._table.add([record])
            except Exception:
                self._table = self._db.create_table("temporal_facts", [record])
        else:
            self._table.add([record])

    def snapshot_at(
        self,
        entity_id: Optional[str] = None,
        entity_type: Optional[str] = None,
        attribute: Optional[str] = None,
        at_time: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get all facts valid at a specific point in time.

        Args:
            entity_id: Optional filter by entity_id.
            entity_type: Optional filter by entity_type.
            attribute: Optional filter by attribute name.
            at_time: ISO UTC timestamp. Defaults to now.

        Returns:
            List of fact dicts that were valid at the given time.
        """
        if at_time is None:
            at_time = datetime.now(timezone.utc).isoformat()

        results = []
        for fact in self._facts:
            # Entity filter
            if entity_id and fact.get("entity_id") != entity_id:
                continue
            if entity_type and fact.get("entity_type") != entity_type:
                continue
            if attribute and fact.get("attribute") != attribute:
                continue

            # Temporal validity check
            valid_from = fact.get("valid_from", "")
            valid_to = fact.get("valid_to")

            if valid_from > at_time:
                continue  # not yet valid
            if valid_to is not None and valid_to <= at_time:
                continue  # expired before at_time

            results.append(fact)

        return results

    def get_timeline(self, entity_id: str, attribute: str) -> List[Dict[str, Any]]:
        """Get all values of an attribute for an entity over time.

        Returns facts sorted by valid_from ascending.
        """
        facts = [
            f for f in self._facts
            if f.get("entity_id") == entity_id and f.get("attribute") == attribute
        ]
        facts.sort(key=lambda f: f.get("valid_from", ""))
        return facts

    def expire_fact(self, fact_id: str, valid_to: Optional[str] = None) -> bool:
        """Mark a fact as expired (sets valid_to).

        Never deletes — preserves audit trail.

        Args:
            fact_id: ID of the fact to expire.
            valid_to: Expiration timestamp. Defaults to now.

        Returns:
            True if fact was found and expired.
        """
        if valid_to is None:
            valid_to = datetime.now(timezone.utc).isoformat()

        for fact in self._facts:
            if fact.get("fact_id") == fact_id:
                fact["valid_to"] = valid_to
                logger.info("Expired fact %s at %s", fact_id, valid_to)
                return True

        logger.warning("Fact %s not found for expiration", fact_id)
        return False

    def save(self, path: Optional[str] = None) -> None:
        """Persist store to disk as JSONL."""
        save_dir = Path(path) if path else self._store_path
        save_dir.mkdir(parents=True, exist_ok=True)

        facts_path = save_dir / "facts.jsonl"
        with open(facts_path, "w", encoding="utf-8") as f:
            for fact in self._facts:
                f.write(json.dumps(fact, default=str, ensure_ascii=False) + "\n")

        logger.info("TemporalFactStore saved: %d facts to %s", len(self._facts), save_dir)

    def load(self, path: Optional[str] = None) -> None:
        """Load store from disk."""
        load_dir = Path(path) if path else self._store_path

        facts_path = load_dir / "facts.jsonl"
        if facts_path.exists():
            self._facts = []
            with open(facts_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._facts.append(json.loads(line))

        if self._backend == "lancedb":
            try:
                import lancedb
                self._db = lancedb.connect(str(load_dir / "lancedb"))
                self._table = self._db.open_table("temporal_facts")
            except Exception:
                pass

        logger.info("TemporalFactStore loaded: %d facts from %s", len(self._facts), load_dir)

    @property
    def fact_count(self) -> int:
        return len(self._facts)
