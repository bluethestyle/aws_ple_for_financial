# Adapters -- raw dataset → pipeline-ready parquet converters.
#
# Importing this package triggers @AdapterRegistry.register decorators on
# every shipped adapter, making them resolvable by name from
# AdapterRegistry.build("<name>", ...). Entry points (containers/phase0/
# entrypoint.py, scripts that invoke PipelineRunner programmatically)
# only need ``import adapters`` -- they no longer have to know which
# adapter modules exist.
from __future__ import annotations

from . import santander_adapter  # noqa: F401  # registers "santander"
