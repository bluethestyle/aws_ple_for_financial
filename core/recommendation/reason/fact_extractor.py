"""
Fact Extractor — Rule-Based Customer Narrative Facts
========================================================

Mem0-inspired fact compression layer without LLM calls.
Extracts narrative customer facts from raw feature dicts
using rule-based Python expressions defined in YAML config.

Example output:
    ["예적금 중심 포트폴리오", "최근 3개월 펀드 관심 증가", "리스크 회피 성향"]

Usage::

    extractor = FactExtractor("configs/financial/fact_extraction.yaml")
    facts = extractor.extract({
        "deposit_balance_ratio": 0.75,
        "fund_view_count_3m": 8,
        "risk_tolerance_score": 0.2,
    })
    # → ["예적금 중심 포트폴리오", "최근 3개월 펀드 관심 증가", "리스크 회피 성향"]

Integration:
    - Phase 0 batch: extract_batch() runs for all customers
    - Results stored in LanceDB ContextVectorStore metadata as "customer_facts"
    - AsyncReasonOrchestrator reads facts for L2a prompt enrichment
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["FactExtractor"]


# Safe builtins available inside rule conditions
_SAFE_BUILTINS = {
    "abs": abs,
    "min": min,
    "max": max,
    "len": len,
    "any": any,
    "all": all,
    "round": round,
    "int": int,
    "float": float,
}


class FactExtractor:
    """Rule-based customer fact extractor.

    Loads rules from YAML and evaluates them against customer feature dicts.
    Each rule has a name (the fact string) and a condition (Python expression).

    Args:
        config_path: Path to fact_extraction.yaml.
    """

    def __init__(self, config_path: str) -> None:
        self._config_path = Path(config_path)
        self._rules: List[Dict[str, Any]] = []
        self._load_rules()

    def _load_rules(self) -> None:
        """Load and validate rules from YAML config."""
        if not self._config_path.exists():
            logger.warning("Fact extraction config not found: %s", self._config_path)
            return

        with open(self._config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        raw_rules = data.get("rules", [])
        for rule in raw_rules:
            if "name" not in rule or "condition" not in rule:
                logger.warning("Skipping rule without name/condition: %s", rule)
                continue
            self._rules.append({
                "name": rule["name"],
                "condition": rule["condition"],
                "required_features": rule.get("required_features", []),
            })

        logger.info(
            "FactExtractor loaded %d rules from %s",
            len(self._rules),
            self._config_path,
        )

    def extract(self, customer_features: Dict[str, Any]) -> List[str]:
        """Extract facts from a single customer feature dict.

        Args:
            customer_features: Dict mapping feature_name → value.

        Returns:
            List of fact strings whose conditions evaluated to True.
        """
        facts = []
        for rule in self._rules:
            # Pre-check: all required features must exist
            required = rule.get("required_features", [])
            if required and not all(f in customer_features for f in required):
                continue

            if self._evaluate_condition(rule["condition"], customer_features):
                facts.append(rule["name"])

        return facts

    def extract_batch(self, features_df: "pd.DataFrame") -> List[List[str]]:
        """Extract facts for all rows in a DataFrame.

        Args:
            features_df: DataFrame with one row per customer.

        Returns:
            List of fact lists, one per row.
        """
        try:
            import pandas as pd  # noqa
        except ImportError:
            logger.error("pandas required for extract_batch")
            return []

        results = []
        for row in features_df.itertuples(index=False):
            features = row._asdict() if hasattr(row, "_asdict") else dict(row._fields)
            results.append(self.extract(features))
        return results

    def _evaluate_condition(self, condition: str, features: Dict[str, Any]) -> bool:
        """Safely evaluate a condition expression against features.

        Uses eval() with restricted builtins. Returns False on any error.
        """
        try:
            # Build namespace: safe builtins + features
            namespace = dict(_SAFE_BUILTINS)
            namespace.update(features)
            result = eval(condition, {"__builtins__": {}}, namespace)
            return bool(result)
        except Exception as e:
            logger.debug("Condition evaluation failed: '%s' — %s", condition, e)
            return False

    def validate_rules(
        self, sample_features: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Dry-run all rules against a sample features dict and report issues.

        Args:
            sample_features: Optional sample dict. If None, uses empty dict.

        Returns:
            Dict with 'valid', 'invalid', 'skipped' rule counts and error details.
        """
        sample = sample_features or {}
        valid = 0
        invalid = []
        skipped = 0

        for rule in self._rules:
            required = rule.get("required_features", [])
            if required and not all(f in sample for f in required):
                skipped += 1
                continue

            try:
                namespace = dict(_SAFE_BUILTINS)
                namespace.update(sample)
                eval(rule["condition"], {"__builtins__": {}}, namespace)
                valid += 1
            except Exception as e:
                invalid.append({
                    "name": rule["name"],
                    "condition": rule["condition"],
                    "error": str(e),
                })

        return {
            "total_rules": len(self._rules),
            "valid": valid,
            "invalid": len(invalid),
            "skipped_missing_features": skipped,
            "invalid_details": invalid,
        }

    @property
    def rule_count(self) -> int:
        return len(self._rules)
