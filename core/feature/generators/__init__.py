"""
Built-in feature generators.

Importing this package triggers the ``@FeatureGeneratorRegistry.register``
decorators on all built-in generators, making them available by name.

Available generators
--------------------
* ``tda_extractor``       -- Topological Data Analysis features (persistence diagrams).
* ``hmm_triple_mode``     -- Hidden Markov Model state estimation (journey / lifecycle / behavior).
* ``hyperbolic_embedding`` -- Hyperbolic (Poincare ball) graph embeddings.
* ``multidisciplinary``   -- Chemical kinetics, epidemic diffusion, interference, crime patterns.
* ``temporal_pattern``    -- Temporal sequence aggregation and cyclical encoding.
"""

from . import tda          # noqa: F401
from . import hmm          # noqa: F401
from . import graph        # noqa: F401
from . import multidisciplinary  # noqa: F401
from . import temporal     # noqa: F401
