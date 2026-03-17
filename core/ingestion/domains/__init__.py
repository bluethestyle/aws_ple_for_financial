"""
Domain ingestor registry auto-loader.

Importing this package triggers all @DomainRegistry.register() decorators,
making every domain ingestor discoverable at runtime.
"""

from . import customer_master, account, card, transaction
from . import g6_management, g7_membership, g9_fund_forex
from . import insurance, consultation, campaign, marketing
from . import open_banking, e_finance, merchant

__all__ = [
    "customer_master",
    "account",
    "card",
    "transaction",
    "g6_management",
    "g7_membership",
    "g9_fund_forex",
    "insurance",
    "consultation",
    "campaign",
    "marketing",
    "open_banking",
    "e_finance",
    "merchant",
]
