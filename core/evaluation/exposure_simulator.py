"""
Exposure simulation for position-bias correction in recommendation evaluation.

Models how recommendation *position* affects user attention (and thus observed
rewards) using a cascade attention model.  This is critical for unbiased
offline evaluation because items shown at higher positions receive more
exposure regardless of relevance.

Provides:
- Position-bias weight estimation
- Cascade attention model for user browsing behaviour
- Exposure-adjusted reward computation

Usage::

    sim = ExposureSimulator(max_positions=20, decay_type="cascade")
    adjusted = sim.adjust_rewards(rewards, positions)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ExposureConfig:
    """Configuration for exposure / position-bias simulation.

    Attributes
    ----------
    max_positions : int
        Maximum number of recommendation slots modelled.
    decay_type : str
        Attention decay model: ``"cascade"``, ``"geometric"``, or
        ``"reciprocal"``.
    cascade_continue_prob : float
        Probability that a user continues scanning to the next position
        (cascade model only).
    geometric_decay : float
        Decay factor for the geometric model ``gamma^k``.
    eta : float
        Exponent for the reciprocal (power-law) model ``1 / k^eta``.
    normalize : bool
        If True, examination probabilities are normalised so that the
        maximum is 1.0.
    """

    max_positions: int = 20
    decay_type: str = "cascade"
    cascade_continue_prob: float = 0.85
    geometric_decay: float = 0.9
    eta: float = 1.0
    normalize: bool = True


# ---------------------------------------------------------------------------
# ExposureSimulator
# ---------------------------------------------------------------------------

class ExposureSimulator:
    """Simulate and correct for position-dependent exposure bias.

    Users do not examine every recommended item with equal probability.
    This simulator computes *examination probabilities* for each position
    under a configurable attention model and uses them to de-bias rewards
    or to generate synthetic exposure data.

    Parameters
    ----------
    max_positions : int
        Number of recommendation slots.
    decay_type : str
        ``"cascade"`` | ``"geometric"`` | ``"reciprocal"``.
    cascade_continue_prob : float
        Cascade model parameter ``c`` -- probability of scanning position
        ``k+1`` given that position ``k`` was examined.
    geometric_decay : float
        ``gamma`` for geometric model ``P(exam @ k) = gamma^(k-1)``.
    eta : float
        Exponent for reciprocal model ``P(exam @ k) = 1 / k^eta``.
    normalize : bool
        Normalise so that position-1 examination probability is 1.0.

    Examples
    --------
    >>> sim = ExposureSimulator(max_positions=10, decay_type="cascade")
    >>> probs = sim.examination_probabilities()
    >>> adjusted = sim.adjust_rewards(rewards, positions)
    """

    def __init__(
        self,
        max_positions: int = 20,
        decay_type: str = "cascade",
        cascade_continue_prob: float = 0.85,
        geometric_decay: float = 0.9,
        eta: float = 1.0,
        normalize: bool = True,
    ) -> None:
        self.config = ExposureConfig(
            max_positions=max_positions,
            decay_type=decay_type,
            cascade_continue_prob=cascade_continue_prob,
            geometric_decay=geometric_decay,
            eta=eta,
            normalize=normalize,
        )
        self._exam_probs: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Examination probabilities
    # ------------------------------------------------------------------

    def examination_probabilities(self) -> np.ndarray:
        """Compute examination probability for each position (1-indexed).

        Returns an array of length ``max_positions`` where ``probs[k-1]``
        is the probability that position *k* is examined.

        Returns
        -------
        np.ndarray
            Examination probabilities, shape ``(max_positions,)``.
        """
        if self._exam_probs is not None:
            return self._exam_probs

        K = self.config.max_positions
        probs = np.zeros(K)

        if self.config.decay_type == "cascade":
            probs = self._cascade_model(K)
        elif self.config.decay_type == "geometric":
            gamma = self.config.geometric_decay
            probs = np.array([gamma ** k for k in range(K)])
        elif self.config.decay_type == "reciprocal":
            eta = self.config.eta
            probs = np.array([1.0 / ((k + 1) ** eta) for k in range(K)])
        else:
            raise ValueError(
                f"Unknown decay_type '{self.config.decay_type}'. "
                f"Choose from: cascade, geometric, reciprocal."
            )

        if self.config.normalize and probs[0] > 0:
            probs = probs / probs[0]

        self._exam_probs = probs
        logger.debug(
            "Examination probs (first 5): %s", probs[:5].tolist(),
        )
        return probs

    def _cascade_model(self, K: int) -> np.ndarray:
        """Cascade attention model.

        Position 1 is always examined.  For position ``k > 1``, the user
        examines it if they examined position ``k-1`` **and** decided to
        continue browsing (with probability ``c``).

        .. math::
            P(\\text{exam} @ k) = c^{k-1}

        This is equivalent to a geometric model with ``gamma = c``, but the
        cascade framing is the standard in the position-bias literature
        (Craswell et al., 2008).
        """
        c = self.config.cascade_continue_prob
        return np.array([c ** k for k in range(K)])

    # ------------------------------------------------------------------
    # Reward adjustment
    # ------------------------------------------------------------------

    def adjust_rewards(
        self,
        rewards: np.ndarray,
        positions: np.ndarray,
    ) -> np.ndarray:
        """Adjust observed rewards for position bias via IPS-style weighting.

        Items at lower (numerically higher) positions receive less attention,
        so their observed rewards are *up-weighted* to compensate.

        .. math::
            r_{adj, i} = \\frac{r_i}{P(\\text{exam} @ k_i)}

        Parameters
        ----------
        rewards : np.ndarray
            Observed rewards, shape ``(n,)``.
        positions : np.ndarray
            1-indexed position of each item, shape ``(n,)``.

        Returns
        -------
        np.ndarray
            Exposure-adjusted rewards, shape ``(n,)``.
        """
        rewards = np.asarray(rewards, dtype=np.float64)
        positions = np.asarray(positions, dtype=np.int64)

        exam_probs = self.examination_probabilities()

        # Clip positions to valid range
        pos_clamped = np.clip(positions - 1, 0, len(exam_probs) - 1)
        exam_at_pos = exam_probs[pos_clamped]

        # Avoid division by zero
        safe_exam = np.maximum(exam_at_pos, 1e-10)
        adjusted = rewards / safe_exam

        logger.debug(
            "Reward adjustment: mean_raw=%.4f, mean_adjusted=%.4f",
            float(np.mean(rewards)), float(np.mean(adjusted)),
        )
        return adjusted

    def exposure_weights(
        self,
        positions: np.ndarray,
    ) -> np.ndarray:
        """Return inverse examination-probability weights for given positions.

        These weights can be multiplied into importance weights for
        position-debiased OPE.

        Parameters
        ----------
        positions : np.ndarray
            1-indexed positions, shape ``(n,)``.

        Returns
        -------
        np.ndarray
            Inverse examination-probability weights, shape ``(n,)``.
        """
        positions = np.asarray(positions, dtype=np.int64)
        exam_probs = self.examination_probabilities()
        pos_clamped = np.clip(positions - 1, 0, len(exam_probs) - 1)
        return 1.0 / np.maximum(exam_probs[pos_clamped], 1e-10)

    # ------------------------------------------------------------------
    # Synthetic exposure simulation
    # ------------------------------------------------------------------

    def simulate_exposures(
        self,
        n_users: int,
        n_items_per_user: int,
        relevance_scores: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, np.ndarray]:
        """Simulate user browsing sessions under the cascade model.

        For each user, items are ranked and the user examines positions
        sequentially, stopping with probability ``1 - c`` at each step.
        Clicks occur when an examined item is relevant (drawn via
        Bernoulli with probability ``relevance_score``).

        Parameters
        ----------
        n_users : int
            Number of simulated user sessions.
        n_items_per_user : int
            Slate length (clamped to ``max_positions``).
        relevance_scores : np.ndarray, optional
            Per-item relevance probabilities of shape
            ``(n_users, n_items_per_user)``.  Defaults to uniform 0.3.
        rng : np.random.Generator, optional
            Random generator for reproducibility.

        Returns
        -------
        dict[str, np.ndarray]
            ``{"examined", "clicked", "positions", "relevance"}``.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        K = min(n_items_per_user, self.config.max_positions)
        exam_probs = self.examination_probabilities()[:K]

        if relevance_scores is None:
            relevance_scores = np.full((n_users, K), 0.3)
        else:
            relevance_scores = np.asarray(relevance_scores)[:, :K]

        examined = np.zeros((n_users, K), dtype=bool)
        clicked = np.zeros((n_users, K), dtype=bool)

        continue_prob = self.config.cascade_continue_prob
        for u in range(n_users):
            for k in range(K):
                if k == 0 or rng.random() < continue_prob:
                    examined[u, k] = True
                    if rng.random() < relevance_scores[u, k]:
                        clicked[u, k] = True
                else:
                    # Cascade: stop examining once the user leaves
                    break

        positions = np.tile(np.arange(1, K + 1), (n_users, 1))

        logger.info(
            "Simulated %d sessions x %d positions: "
            "exam_rate=%.3f, click_rate=%.3f",
            n_users, K,
            float(examined.mean()), float(clicked.mean()),
        )

        return {
            "examined": examined,
            "clicked": clicked,
            "positions": positions,
            "relevance": relevance_scores,
        }

    # ------------------------------------------------------------------
    # Position bias estimation from logged data
    # ------------------------------------------------------------------

    def estimate_position_bias(
        self,
        clicks: np.ndarray,
        positions: np.ndarray,
    ) -> np.ndarray:
        """Estimate empirical examination probabilities from logged clicks.

        Uses the simple ratio estimator:
        ``P(exam @ k) = avg_click_rate(k) / avg_click_rate(1)``.

        Parameters
        ----------
        clicks : np.ndarray
            Binary click indicator, shape ``(n,)``.
        positions : np.ndarray
            1-indexed positions, shape ``(n,)``.

        Returns
        -------
        np.ndarray
            Estimated examination probabilities, one per position up to
            ``max_positions``.
        """
        clicks = np.asarray(clicks, dtype=np.float64)
        positions = np.asarray(positions, dtype=np.int64)

        K = self.config.max_positions
        click_rates = np.zeros(K)
        counts = np.zeros(K)

        for k in range(1, K + 1):
            mask = positions == k
            if mask.any():
                click_rates[k - 1] = float(clicks[mask].mean())
                counts[k - 1] = float(mask.sum())

        # Normalise by position-1 click rate
        if click_rates[0] > 0:
            click_rates = click_rates / click_rates[0]

        # Store for later use
        self._exam_probs = click_rates

        logger.info(
            "Position bias estimated from %d interactions (first 5: %s).",
            int(counts.sum()), click_rates[:5].tolist(),
        )
        return click_rates


__all__ = ["ExposureSimulator", "ExposureConfig"]
