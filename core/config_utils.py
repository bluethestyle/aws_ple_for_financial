"""Config utility re-exports for convenient top-level import.

Thin wrapper so callers can do::

    from core.config_utils import load_merged_config, deep_merge

instead of reaching into ``core.pipeline.config`` directly.
"""

from core.pipeline.config import deep_merge, load_merged_config, load_config

__all__ = ["deep_merge", "load_merged_config", "load_config"]
