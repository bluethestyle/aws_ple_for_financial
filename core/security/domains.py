"""
PII Domain definitions and column-to-domain mapping.

Each PII domain (CUSTOMER, ACCOUNT, CARD, etc.) uses a unique salt
for SHA256 hashing. This prevents cross-domain rainbow table attacks.
"""
from enum import Enum
from typing import Dict

class PIIDomain(Enum):
    CUSTOMER = "customer"
    ACCOUNT = "account"
    CARD = "card"
    MERCHANT = "merchant"
    TRANSACTION = "transaction"
    INSURANCE = "insurance"
    CONSULTATION = "consultation"
    CAMPAIGN = "campaign"
    MARKETING = "marketing"
    OPEN_BANKING = "open_banking"
    E_FINANCE = "e_finance"
    MEMBERSHIP = "membership"
    FUND_FOREX = "fund_forex"
    CONTACT = "contact"        # phone, email
    PERSONAL_ID = "personal_id"  # SSN, passport
    DEFAULT = "default"

# Default column name -> domain mapping
# Keys are lowercase column name patterns (substring match)
COLUMN_DOMAIN_MAP: Dict[str, PIIDomain] = {
    "customer_id": PIIDomain.CUSTOMER,
    "cust_no": PIIDomain.CUSTOMER,
    "cust_id": PIIDomain.CUSTOMER,
    "csno": PIIDomain.CUSTOMER,
    "account_no": PIIDomain.ACCOUNT,
    "actno": PIIDomain.ACCOUNT,
    "acct_no": PIIDomain.ACCOUNT,
    "deps_actno": PIIDomain.ACCOUNT,
    "card_no": PIIDomain.CARD,
    "card_number": PIIDomain.CARD,
    "chk_cdno": PIIDomain.CARD,
    "merchant_id": PIIDomain.MERCHANT,
    "frcs_no": PIIDomain.MERCHANT,
    "insurance_policy": PIIDomain.INSURANCE,
    "insr_plcy_no": PIIDomain.INSURANCE,
    "phone": PIIDomain.CONTACT,
    "telno": PIIDomain.CONTACT,
    "hp_no": PIIDomain.CONTACT,
    "email": PIIDomain.CONTACT,
    "rrno": PIIDomain.PERSONAL_ID,
    "brno": PIIDomain.PERSONAL_ID,
    "name": PIIDomain.CUSTOMER,
    "cust_nm": PIIDomain.CUSTOMER,
}

def resolve_domain(column_name: str) -> PIIDomain:
    """Resolve PII domain for a column name.
    Uses substring matching against COLUMN_DOMAIN_MAP.
    Returns PIIDomain.DEFAULT if no match found.
    """
    lower = column_name.lower()
    # Exact match first
    if lower in COLUMN_DOMAIN_MAP:
        return COLUMN_DOMAIN_MAP[lower]
    # Substring match
    for pattern, domain in COLUMN_DOMAIN_MAP.items():
        if pattern in lower:
            return domain
    return PIIDomain.DEFAULT
