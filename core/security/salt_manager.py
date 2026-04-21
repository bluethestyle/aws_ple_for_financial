"""
Salt Manager -- retrieve domain-specific salts from AWS Secrets Manager.

Production: Secrets Manager secret containing JSON:
  {"customer": "salt_value_1", "account": "salt_value_2", ...}

Local development: LocalSaltManager reads from environment variables:
  SALT_CUSTOMER=xxx, SALT_ACCOUNT=yyy, ...
  Or from a local JSON file.
"""
import json, logging, os, time
from typing import Dict, Optional
from .domains import PIIDomain

logger = logging.getLogger(__name__)

class SaltManager:
    """Retrieve domain salts from AWS Secrets Manager with in-memory caching.

    ``secret_name`` and ``region`` have no hardcoded AWS-specific defaults per
    CLAUDE.md §1.1. Callers should build the secret name from
    ``pipeline.yaml::aws.s3_bucket`` (e.g. ``{bucket}/pii-salts``) and pass
    ``aws.region``. ``region=None`` lets boto3 resolve from env / credentials.
    The fetch path raises if ``secret_name`` is missing.
    """

    def __init__(self, secret_name: Optional[str] = None,
                 region: Optional[str] = None, cache_ttl: int = 3600):
        self._secret_name = secret_name
        self._region = region
        self._cache_ttl = cache_ttl
        self._cache: Dict[str, str] = {}
        self._cache_time: float = 0

    def get_salt(self, domain: PIIDomain) -> bytes:
        """Get salt bytes for a domain. Cached."""
        self._ensure_cache()
        key = domain.value
        if key not in self._cache:
            raise KeyError(f"No salt configured for domain '{key}'. "
                          f"Available: {list(self._cache.keys())}")
        return self._cache[key].encode("utf-8")

    def get_all_salts(self) -> Dict[PIIDomain, bytes]:
        self._ensure_cache()
        return {PIIDomain(k): v.encode("utf-8") for k, v in self._cache.items()
                if k in [d.value for d in PIIDomain]}

    def _ensure_cache(self):
        if self._cache and (time.time() - self._cache_time) < self._cache_ttl:
            return
        self._cache = self._fetch_from_secrets_manager()
        self._cache_time = time.time()

    def _fetch_from_secrets_manager(self) -> Dict[str, str]:
        if not self._secret_name:
            logger.warning(
                "SaltManager: secret_name is not configured; returning empty "
                "salts. Pass secret_name derived from pipeline.yaml::aws.s3_bucket."
            )
            return {}
        try:
            import boto3
            client = boto3.client("secretsmanager", region_name=self._region)
            resp = client.get_secret_value(SecretId=self._secret_name)
            secret = json.loads(resp["SecretString"])
            logger.info("Loaded %d domain salts from Secrets Manager '%s'",
                       len(secret), self._secret_name)
            return secret
        except Exception as e:
            logger.warning("Secrets Manager unavailable (%s), using empty salts", e)
            return {}


class LocalSaltManager(SaltManager):
    """Local development salt manager. Reads from env vars or JSON file."""

    def __init__(self, salt_file: str = "", env_prefix: str = "SALT_"):
        super().__init__()
        self._salt_file = salt_file
        self._env_prefix = env_prefix

    def _fetch_from_secrets_manager(self) -> Dict[str, str]:
        # Try JSON file first
        if self._salt_file and os.path.exists(self._salt_file):
            with open(self._salt_file, "r") as f:
                salts = json.loads(f.read())
                logger.info("Loaded %d salts from %s", len(salts), self._salt_file)
                return salts

        # Fall back to environment variables
        salts = {}
        for domain in PIIDomain:
            env_key = f"{self._env_prefix}{domain.value.upper()}"
            val = os.environ.get(env_key)
            if val:
                salts[domain.value] = val

        if not salts:
            # Generate deterministic defaults for development only
            logger.warning("No salts configured. Using development defaults (NOT FOR PRODUCTION).")
            import hashlib
            for domain in PIIDomain:
                salts[domain.value] = hashlib.sha256(f"dev_salt_{domain.value}".encode()).hexdigest()[:32]

        logger.info("LocalSaltManager: %d domain salts loaded", len(salts))
        return salts
