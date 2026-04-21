"""
core.serving.review
===================

Human-in-the-loop review queue (Sprint 3 M1). Required by AI기본법 §34
(인간 감독) and SR 11-7 (effective challenge) for tier-2 / tier-3
recommendations that must not auto-execute.

Tiering policy (driven by pipeline.yaml::serving.review):
- Tier 1 : post-hoc 5% sample (random)    - low-risk automations
- Tier 2 : 100% mandatory agent review    - medium-risk
- Tier 3 : mandatory HumanFallback path   - high-risk / opt-out
"""

from core.serving.review.human_review_queue import (
    HumanReviewQueue,
    ReviewConfig,
    ReviewItem,
    ReviewState,
    build_human_review_queue,
)

__all__ = [
    "HumanReviewQueue",
    "ReviewConfig",
    "ReviewItem",
    "ReviewState",
    "build_human_review_queue",
]
