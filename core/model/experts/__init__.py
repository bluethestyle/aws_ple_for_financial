"""
Expert network implementations for the PLE multi-task learning platform.

All experts inherit from :class:`AbstractExpert` and self-register with
:class:`ExpertRegistry` at import time via the ``@ExpertRegistry.register``
decorator.

Registered experts
------------------
* ``mlp``               -- :class:`MLPExpert` (default baseline)
* ``deepfm``            -- :class:`DeepFMExpert` (FM + Deep feature interaction)
* ``mamba``             -- :class:`MambaExpert` (Selective State Space Model)
* ``temporal_ensemble`` -- :class:`TemporalEnsembleExpert` (Mamba + Transformer)
* ``causal``            -- :class:`CausalExpert` (NOTEARS DAG)
* ``optimal_transport`` -- :class:`OptimalTransportExpert` (Sinkhorn)
* ``hgcn``              -- :class:`UnifiedHGCNExpert` (Hyperbolic GCN)
* ``perslay``           -- :class:`PersLayExpert` (Topological PersLay)
* ``lightgcn``          -- :class:`LightGCNExpert` (Light Graph Conv)
* ``autoint``           -- :class:`AutoIntExpert` (Self-Attention Interaction)
* ``xdeepfm``           -- :class:`XDeepFMExpert` (CIN + Deep)

Usage::

    from core.model.experts import ExpertRegistry

    # List all registered experts
    ExpertRegistry.list_available()

    # Create a single expert
    expert = ExpertRegistry.create(
        "deepfm",
        input_dim=128,
        config={"output_dim": 64, "field_dims": [32, 32, 64]},
    )

    # Bulk-create from a config dict
    experts = ExpertRegistry.create_from_config(
        experts_config={
            "interaction": {"type": "deepfm", "output_dim": 64, ...},
            "causal":      {"type": "causal",  "output_dim": 64, ...},
        },
        default_input_dim=128,
    )
"""

# Base class and utilities
from .base import AbstractExpert, init_expert_weights

# Registry (import before concrete experts so the decorator target exists)
from .registry import ExpertRegistry, get_total_expert_output_dim

# Concrete expert implementations -- importing them triggers registration
from .mlp import MLPExpert
from .deepfm import DeepFMExpert
from .mamba import MambaBlock, MambaExpert, StackedMambaBlocks
from .temporal import TemporalEnsembleExpert
from .causal import CausalExpert
from .ot import OptimalTransportExpert
from .hgcn import UnifiedHGCNExpert
from .perslay import PersLayExpert
from .lightgcn import LightGCNExpert
from .autoint import AutoIntExpert
from .xdeepfm import XDeepFMExpert

__all__ = [
    # Base
    "AbstractExpert",
    "init_expert_weights",
    # Registry
    "ExpertRegistry",
    "get_total_expert_output_dim",
    # Experts
    "MLPExpert",
    "DeepFMExpert",
    "MambaBlock",
    "MambaExpert",
    "StackedMambaBlocks",
    "TemporalEnsembleExpert",
    "CausalExpert",
    "OptimalTransportExpert",
    "UnifiedHGCNExpert",
    "PersLayExpert",
    "LightGCNExpert",
    "AutoIntExpert",
    "XDeepFMExpert",
]
