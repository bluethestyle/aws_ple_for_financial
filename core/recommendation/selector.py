"""
Top-K Selection with Diversity
===============================

Selects the final recommendation set from scored, filtered candidates.

Supports three diversity strategies, selectable from config:

    * ``none``  -- Pure score-based ranking.
    * ``mmr``   -- Maximal Marginal Relevance.
    * ``dpp``   -- Determinantal Point Process (greedy approximation).

Config example::

    selector:
      k: 5
      min_score: 0.0
      diversity_method: mmr    # none | mmr | dpp
      diversity_lambda: 0.5    # balance: 1.0 = all relevance, 0.0 = all diversity
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "TopKSelector",
    "DiversityMethod",
]


class DiversityMethod(str, Enum):
    """Supported diversity strategies."""
    NONE = "none"
    MMR = "mmr"
    DPP = "dpp"


class TopKSelector:
    """Score-based top-K selection with optional diversity re-ranking.

    Args:
        config: Selector configuration dict.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        sel_cfg = config.get("selector", {})
        self.k: int = sel_cfg.get("k", 5)
        self.min_score: float = sel_cfg.get("min_score", 0.0)

        method_str = sel_cfg.get("diversity_method", "none").lower()
        try:
            self.diversity_method = DiversityMethod(method_str)
        except ValueError:
            logger.warning(
                "Unknown diversity method '%s', falling back to 'none'",
                method_str,
            )
            self.diversity_method = DiversityMethod.NONE

        self.diversity_lambda: float = sel_cfg.get("diversity_lambda", 0.5)

        logger.info(
            "TopKSelector: k=%d, min_score=%.3f, diversity=%s, lambda=%.2f",
            self.k, self.min_score, self.diversity_method.value,
            self.diversity_lambda,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(
        self,
        candidates: List[Dict[str, Any]],
        item_features: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[Dict[str, Any]]:
        """Select top-K items from scored candidates.

        Each candidate dict must contain at least ``item_id`` and ``score``.
        ``item_features`` is required for MMR / DPP diversity.

        Args:
            candidates: List of candidate dicts.
            item_features: Mapping from item_id to feature vector (for
                           diversity computation).

        Returns:
            Top-K candidates with ``rank`` injected, sorted by final order.
        """
        # Pre-filter by min_score
        filtered = [c for c in candidates if c.get("score", 0.0) >= self.min_score]

        if not filtered:
            logger.warning("TopKSelector: no candidates above min_score=%.3f", self.min_score)
            return []

        # Sort by score descending
        filtered.sort(key=lambda c: c.get("score", 0.0), reverse=True)

        # Apply diversity strategy
        if self.diversity_method == DiversityMethod.NONE or item_features is None:
            selected = filtered[: self.k]
        elif self.diversity_method == DiversityMethod.MMR:
            selected = self._mmr_select(filtered, item_features)
        elif self.diversity_method == DiversityMethod.DPP:
            selected = self._dpp_select(filtered, item_features)
        else:
            selected = filtered[: self.k]

        # Assign ranks
        for rank, cand in enumerate(selected, start=1):
            cand["rank"] = rank

        return selected

    # ------------------------------------------------------------------
    # MMR: Maximal Marginal Relevance
    # ------------------------------------------------------------------

    def _mmr_select(
        self,
        candidates: List[Dict[str, Any]],
        item_features: Dict[str, np.ndarray],
    ) -> List[Dict[str, Any]]:
        """Greedy MMR selection.

        MMR(d_i) = lambda * score(d_i)
                   - (1 - lambda) * max_{d_j in S} sim(d_i, d_j)

        where S is the already-selected set.
        """
        lam = self.diversity_lambda
        remaining = list(candidates)
        selected: List[Dict[str, Any]] = []

        # Normalise scores to [0, 1] for fair combination
        scores = np.array([c.get("score", 0.0) for c in remaining])
        score_min, score_max = scores.min(), scores.max()
        score_range = score_max - score_min if score_max > score_min else 1.0

        while len(selected) < self.k and remaining:
            best_idx = -1
            best_mmr = -float("inf")

            for i, cand in enumerate(remaining):
                rel = (cand.get("score", 0.0) - score_min) / score_range
                max_sim = 0.0

                feat_i = item_features.get(cand.get("item_id", ""))
                if feat_i is not None and selected:
                    for sel in selected:
                        feat_j = item_features.get(sel.get("item_id", ""))
                        if feat_j is not None:
                            sim = self._cosine_similarity(feat_i, feat_j)
                            max_sim = max(max_sim, sim)

                mmr_val = lam * rel - (1.0 - lam) * max_sim
                if mmr_val > best_mmr:
                    best_mmr = mmr_val
                    best_idx = i

            if best_idx >= 0:
                selected.append(remaining.pop(best_idx))
            else:
                break

        return selected

    # ------------------------------------------------------------------
    # DPP: Determinantal Point Process (greedy)
    # ------------------------------------------------------------------

    def _dpp_select(
        self,
        candidates: List[Dict[str, Any]],
        item_features: Dict[str, np.ndarray],
    ) -> List[Dict[str, Any]]:
        """Greedy DPP approximation.

        Constructs L = diag(q) * S * diag(q) where q_i = score_i and
        S_ij = cosine_similarity(feat_i, feat_j), then greedily picks
        the item that maximises det(L_Y + epsilon * I).
        """
        lam = self.diversity_lambda
        pool = candidates[: min(len(candidates), self.k * 5)]  # limit pool

        n = len(pool)
        if n == 0:
            return []

        # Collect feature vectors
        feat_dim = None
        vecs = []
        for c in pool:
            fv = item_features.get(c.get("item_id", ""))
            if fv is not None:
                vecs.append(fv)
                if feat_dim is None:
                    feat_dim = len(fv)
            else:
                vecs.append(None)

        # Fallback: if too few features, just return score-ranked
        available_vecs = [v for v in vecs if v is not None]
        if len(available_vecs) < 2 or feat_dim is None:
            return pool[: self.k]

        # Build quality-diversity decomposition
        scores = np.array([c.get("score", 0.0) for c in pool])
        scores_norm = scores / (scores.max() + 1e-12)

        # Similarity matrix
        S = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                if vecs[i] is not None and vecs[j] is not None:
                    sim = self._cosine_similarity(vecs[i], vecs[j])
                    S[i, j] = sim
                    S[j, i] = sim

        # L kernel
        q = lam * scores_norm + (1.0 - lam) * np.ones(n)
        L = np.outer(q, q) * S

        # Greedy selection
        selected_idx: List[int] = []
        remaining_idx = list(range(n))

        for _ in range(min(self.k, n)):
            best_i = -1
            best_gain = -float("inf")

            for idx in remaining_idx:
                trial = selected_idx + [idx]
                L_sub = L[np.ix_(trial, trial)]
                gain = np.linalg.det(L_sub + 1e-8 * np.eye(len(trial)))
                if gain > best_gain:
                    best_gain = gain
                    best_i = idx

            if best_i >= 0:
                selected_idx.append(best_i)
                remaining_idx.remove(best_i)
            else:
                break

        return [pool[i] for i in selected_idx]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity clamped to [0, 1]."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-12 or norm_b < 1e-12:
            return 0.0
        sim = float(np.dot(a, b) / (norm_a * norm_b))
        return max(0.0, min(1.0, sim))
