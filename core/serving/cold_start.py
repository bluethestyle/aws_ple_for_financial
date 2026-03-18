"""
Cold-Start Handler
==================

Manages the full cold-start recommendation path for users who have no
entry in the Feature Store.

Three user states are handled:

* **ANONYMOUS** (segment=0): No user_id context at all.
  → Generic popularity-only recommendations.

* **COLDSTART** (segment=1): Known user_id but no historical features yet
  (new signup, first login, or feature pipeline hasn't processed them yet).
  → Popularity-based candidates + model score with a default feature vector
    that has cold-start signals set.

* **WARMSTART** (segment=2): Normal returning user with features.
  → Standard path (not handled here).

Design decisions:
- The default feature vector is all-zeros with ``is_coldstart=1``,
  ``coldstart_confidence=1.0``, and ``customer_segment=1``.  LGBM student
  models trained with cold-start signals in the feature set will produce
  meaningful probability estimates even for this sparse input.
- Popularity candidates come from a configurable catalog (S3 JSON or static
  config list).  The catalog is loaded once at cold start and cached.
- The caller (RecommendationService / Lambda handler) checks if features are
  None and delegates to this handler.

Usage::

    handler = ColdStartHandler.from_config(config)

    # Decide segment from available signals
    segment = handler.classify(user_id=uid, context=ctx)

    # Get a feature vector for LGBM inference (COLDSTART only)
    features = handler.default_features(feature_names)

    # Get popularity-based candidate items
    candidates = handler.popularity_candidates(k=20)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["ColdStartHandler", "UserSegment"]


# ---------------------------------------------------------------------------
# Segment constants (match schema.yaml)
# ---------------------------------------------------------------------------

class UserSegment:
    ANONYMOUS = "ANONYMOUS"    # segment=0: no user_id context
    COLDSTART = "COLDSTART"    # segment=1: user_id known, no history
    WARMSTART = "WARMSTART"    # segment=2: full feature history available


# ---------------------------------------------------------------------------
# ColdStartHandler
# ---------------------------------------------------------------------------

class ColdStartHandler:
    """Handles recommendations for users with no feature store entry.

    Args:
        popularity_catalog: Static list of popular item dicts
            ``[{"item_id": "...", "score": 0.9, "benefit_type": "cashback"}, ...]``.
            Used as fallback when no model inference is possible.
        coldstart_features: Feature names the model expects.  When provided,
            :meth:`default_features` returns a dict with these names set to 0
            plus cold-start signal overrides.
        anonymous_threshold_days: Users with ``account_age_days`` below this
            are routed to ANONYMOUS instead of COLDSTART.  Set to 0 to disable.
        popularity_catalog_s3_uri: S3 URI for a JSON popularity catalog.
            Loaded at initialisation time when provided.
        region: AWS region for S3 access.
    """

    def __init__(
        self,
        popularity_catalog: Optional[List[Dict[str, Any]]] = None,
        coldstart_features: Optional[List[str]] = None,
        anonymous_threshold_days: int = 0,
        popularity_catalog_s3_uri: str = "",
        region: str = "ap-northeast-2",
    ) -> None:
        self._coldstart_features = coldstart_features or []
        self._anonymous_threshold_days = anonymous_threshold_days
        self._region = region

        # Load catalog: prefer static list, then S3
        if popularity_catalog:
            self._catalog = popularity_catalog
        elif popularity_catalog_s3_uri:
            self._catalog = self._load_catalog_from_s3(popularity_catalog_s3_uri)
        else:
            self._catalog = []

        logger.info(
            "ColdStartHandler: %d popularity items loaded, "
            "anonymous_threshold_days=%d",
            len(self._catalog), anonymous_threshold_days,
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ColdStartHandler":
        """Build a ColdStartHandler from the recommendation config dict.

        Expects a ``cold_start:`` section::

            cold_start:
              anonymous_threshold_days: 0
              popularity_catalog_s3_uri: ""
              popularity_catalog:
                - item_id: "PROD_001"
                  score: 0.95
                  benefit_type: cashback
                  name: "Cashback Premium Card"
                - item_id: "PROD_002"
                  score: 0.90
                  benefit_type: mileage
                  name: "Travel Rewards Card"
        """
        cs_cfg = config.get("cold_start", {})

        return cls(
            popularity_catalog=cs_cfg.get("popularity_catalog", []),
            coldstart_features=cs_cfg.get("coldstart_features", []),
            anonymous_threshold_days=cs_cfg.get("anonymous_threshold_days", 0),
            popularity_catalog_s3_uri=cs_cfg.get("popularity_catalog_s3_uri", ""),
            region=config.get("region", "ap-northeast-2"),
        )

    # ------------------------------------------------------------------
    # Segment classification
    # ------------------------------------------------------------------

    def classify(
        self,
        user_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Classify a user as ANONYMOUS or COLDSTART.

        Args:
            user_id: User identifier.  Empty string → ANONYMOUS.
            context: Optional context dict.  Checked fields:
                - ``segment``: If already set (0/1/2 or string), trust it.
                - ``account_age_days``: If below threshold → ANONYMOUS.
                - ``is_anonymous``: Explicit override.

        Returns:
            :attr:`UserSegment.ANONYMOUS` or :attr:`UserSegment.COLDSTART`.
        """
        ctx = context or {}

        # Explicit override
        if ctx.get("is_anonymous") or not user_id:
            return UserSegment.ANONYMOUS

        # Already classified by upstream (e.g. API gateway auth layer)
        seg = ctx.get("segment")
        if seg is not None:
            seg_str = str(seg).upper()
            if seg_str in ("0", "ANONYMOUS"):
                return UserSegment.ANONYMOUS
            if seg_str in ("1", "COLDSTART"):
                return UserSegment.COLDSTART

        # Account age heuristic
        account_age = ctx.get("account_age_days")
        if (
            account_age is not None
            and self._anonymous_threshold_days > 0
            and int(account_age) < self._anonymous_threshold_days
        ):
            return UserSegment.ANONYMOUS

        return UserSegment.COLDSTART

    # ------------------------------------------------------------------
    # Default feature vector
    # ------------------------------------------------------------------

    def default_features(
        self,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Return a zero-filled feature vector with cold-start overrides.

        The returned dict can be passed directly to LGBM for inference.
        Models trained with cold-start signals will produce meaningful
        (though lower-confidence) outputs.

        Cold-start signal overrides (6D from schema.yaml):
            - ``is_coldstart = 1``
            - ``customer_segment = 1``  (COLDSTART)
            - ``coldstart_confidence = 1.0``
            - ``coldstart_seq_confidence = 0.0``
            - ``coldstart_feature_confidence = 0.0``
            - ``coldstart_confidence_decay = 1.0``

        Args:
            feature_names: Optional list of feature names to include.
                Falls back to :attr:`_coldstart_features` if not provided.

        Returns:
            Dict mapping feature name → default value.
        """
        names = feature_names or self._coldstart_features
        features: Dict[str, Any] = {name: 0.0 for name in names}

        # Cold-start signal overrides
        coldstart_overrides = {
            "is_coldstart": 1,
            "customer_segment": 1,
            "coldstart_confidence": 1.0,
            "coldstart_seq_confidence": 0.0,
            "coldstart_feature_confidence": 0.0,
            "coldstart_confidence_decay": 1.0,
        }
        for k, v in coldstart_overrides.items():
            if k in features or not names:
                features[k] = v

        return features

    # ------------------------------------------------------------------
    # Popularity candidates
    # ------------------------------------------------------------------

    def popularity_candidates(
        self,
        k: int = 20,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Return top-k popularity candidates.

        Each candidate is a dict compatible with the recommendation pipeline's
        candidate format::

            {
                "item_id": "PROD_001",
                "predictions": {"ctr": 0.0, "cvr": 0.0, ...},  # zero-filled
                "item_info": {"name": "...", "benefit_type": "..."},
                "ig_top_features": [],   # empty for cold-start
                "popularity_score": 0.95,
                "is_coldstart_candidate": True,
            }

        Args:
            k: Maximum number of candidates to return.
            context: Optional context for future personalised filtering.

        Returns:
            List of candidate dicts (at most *k* items).
        """
        if not self._catalog:
            logger.warning(
                "ColdStartHandler: popularity catalog is empty. "
                "Configure cold_start.popularity_catalog in recommendation.yaml",
            )
            return []

        sorted_catalog = sorted(
            self._catalog,
            key=lambda x: x.get("score", 0.0),
            reverse=True,
        )

        candidates = []
        for item in sorted_catalog[:k]:
            candidates.append({
                "item_id": item.get("item_id", ""),
                "predictions": {},          # pipeline scorer will use popularity_score
                "item_info": {
                    "name": item.get("name", item.get("item_id", "")),
                    "benefit_type": item.get("benefit_type", ""),
                    "category": item.get("category", ""),
                },
                "ig_top_features": [],
                "popularity_score": float(item.get("score", 0.0)),
                "is_coldstart_candidate": True,
            })

        return candidates

    # ------------------------------------------------------------------
    # Build PredictionResponse for cold-start
    # ------------------------------------------------------------------

    def build_response(
        self,
        user_id: str,
        segment: str,
        candidates: List[Dict[str, Any]],
        elapsed_ms: float = 0.0,
        pipeline: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build a structured cold-start recommendation response.

        If a pipeline is provided, routes candidates through it with
        the appropriate segment label so the template engine generates
        correct COLDSTART/ANONYMOUS reasons.

        Args:
            user_id: User identifier.
            segment: ``"COLDSTART"`` or ``"ANONYMOUS"``.
            candidates: Popularity candidates from :meth:`popularity_candidates`.
            elapsed_ms: Time already spent before this call.
            pipeline: Optional :class:`~core.recommendation.pipeline.RecommendationPipeline`.
            context: Request context.

        Returns:
            Dict suitable for JSON serialisation.
        """
        ctx = dict(context or {})
        ctx["segment"] = segment

        recommendations = []
        if pipeline is not None and candidates:
            try:
                result = pipeline.recommend(
                    customer_id=user_id,
                    candidate_items=candidates,
                    customer_context=ctx,
                )
                recommendations = [
                    {
                        "item_id": item.item_id,
                        "rank": item.rank,
                        "score": item.score,
                        "score_components": item.score_components,
                        "reasons": item.reasons,
                        "metadata": item.metadata,
                    }
                    for item in result.items
                ]
            except Exception:
                logger.exception(
                    "ColdStartHandler: pipeline failed for user_id=%s", user_id,
                )
                # Fall through to raw candidates
        else:
            # No pipeline: return raw ranked candidates
            for i, cand in enumerate(candidates, start=1):
                recommendations.append({
                    "item_id": cand["item_id"],
                    "rank": i,
                    "score": cand.get("popularity_score", 0.0),
                    "score_components": {"popularity": cand.get("popularity_score", 0.0)},
                    "reasons": [],
                    "metadata": {"is_coldstart_candidate": True},
                })

        return {
            "user_id": user_id,
            "segment": segment,
            "is_coldstart": True,
            "predictions": {},
            "recommendations": recommendations,
            "elapsed_ms": elapsed_ms,
            "metadata": {
                "coldstart_path": True,
                "candidates_provided": len(candidates),
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_catalog_from_s3(
        self, s3_uri: str,
    ) -> List[Dict[str, Any]]:
        """Download popularity catalog JSON from S3."""
        try:
            import boto3
            import json

            s3 = boto3.client("s3", region_name=self._region)
            path = s3_uri.replace("s3://", "")
            parts = path.split("/", 1)
            bucket, key = parts[0], (parts[1] if len(parts) > 1 else "")
            obj = s3.get_object(Bucket=bucket, Key=key)
            data = json.loads(obj["Body"].read())
            catalog = data if isinstance(data, list) else data.get("items", [])
            logger.info(
                "Loaded %d items from popularity catalog: %s",
                len(catalog), s3_uri,
            )
            return catalog
        except Exception as e:
            logger.warning("Failed to load popularity catalog from %s: %s", s3_uri, e)
            return []
