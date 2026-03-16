"""
Special-purpose neural network layers for the PLE platform.

* :class:`SparseAutoencoder` -- interpretability via sparse dictionary
  learning (Anthropic-style SAE).
* :class:`EvidentialLayer` -- calibrated epistemic uncertainty via
  Evidential Deep Learning.
"""

from .evidential import EvidentialLayer
from .sae import SparseAutoencoder

__all__ = ["SparseAutoencoder", "EvidentialLayer"]
