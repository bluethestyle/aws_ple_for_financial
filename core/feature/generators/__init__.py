"""
Built-in feature generators.

Importing this package triggers the ``@FeatureGeneratorRegistry.register``
decorators on all built-in generators, making them available by name.

Available generators
--------------------
* ``tda``                  -- Topological Data Analysis features (persistence diagrams).
* ``phase_transition``     -- Topological phase transition detection via persistence diagram distances.
* ``hmm``                  -- Triple-mode HMM state estimation (journey / lifecycle / behavior).
* ``gmm``                  -- Gaussian Mixture Model soft clustering with BIC validation.
* ``graph``                -- LightGCN graph embeddings with optional Poincare projection.
* ``mamba``                -- Mamba SSM temporal embedding features.
* ``multidisciplinary``    -- Chemical kinetics, epidemic diffusion, interference, crime patterns.
* ``temporal``             -- Temporal rolling aggregation, cyclical encoding, and velocity features.
* ``economics``            -- Economic / financial behavior features (income decomposition + financial behavior).
* ``merchant_hierarchy``   -- Merchant hierarchy features (MCC levels, brand SVD, aggregate stats, radius).
* ``model_features``       -- Model-derived features: HMM summary + Bandit/MAB + LNN temporal dynamics (27D).
* ``lag_extractor``        -- K-step right-aligned lag flattening of LIST sequence columns (axis-1).
* ``rolling_stats_extractor`` -- Rolling-window stats (sum/mean/std/count/days_active) over LIST sequences (axis-2).
* ``topn_multihot_extractor`` -- Top-N or fixed-vocab multi-hot encoding of LIST<int> columns (axis-3).

Pool / Basket pattern
---------------------
All generators register themselves into the **Generator Pool** (the
``FeatureGeneratorRegistry``).  Downstream config selects a subset
(the "basket") for a specific pipeline run via ``FeatureGroupConfig``.

Usage::

    from core.feature.generators import FeatureGeneratorRegistry

    # List all available generators in the pool
    print(FeatureGeneratorRegistry.list_available())

    # Check GPU-capable generators
    print(FeatureGeneratorRegistry.list_gpu_capable())

    # Create a generator by name
    gen = FeatureGeneratorRegistry.create("tda", max_homology_dim=2)

    # Get summary info for all generators
    for info in FeatureGeneratorRegistry.list_all_info():
        print(f"  {info['name']}: gpu={info['supports_gpu']}, libs={info['required_libraries']}")
"""

from __future__ import annotations

# Re-export the registry for convenient access
from ..generator import AbstractFeatureGenerator, FeatureGeneratorRegistry

# Import all generator modules to trigger @register decorators.
# Each module registers its generator(s) at import time.
from . import tda                # noqa: F401
from . import hmm                # noqa: F401
from . import graph              # noqa: F401
from . import mamba              # noqa: F401
from . import multidisciplinary  # noqa: F401
from . import temporal           # noqa: F401
from . import gmm                # noqa: F401
from . import phase_transition   # noqa: F401
from . import economics             # noqa: F401
from . import merchant_hierarchy    # noqa: F401
from . import model_features        # noqa: F401
from . import lag_extractor         # noqa: F401
from . import rolling_stats_extractor  # noqa: F401
from . import topn_multihot_extractor  # noqa: F401

# GPU utilities
from . import gpu_utils       # noqa: F401

__all__ = [
    "AbstractFeatureGenerator",
    "FeatureGeneratorRegistry",
    "gpu_utils",
]
