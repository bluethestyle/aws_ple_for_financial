"""
DeepFM Expert -- Deep Factorization Machine for feature interaction learning.

Combines:

* **FM (Factorization Machine)** -- efficient 2nd-order feature interactions
  via the identity  ``0.5 * [(sum V_i)^2 - sum(V_i^2)]``.
* **Deep Network** -- higher-order non-linear interactions through an MLP.
* **Cross Network (DCN)** -- optional explicit feature crossing.

The input feature vector is split into *fields* (logical sub-groups) and each
field is independently embedded before being fed to FM / Deep / Cross.

Field splitting modes:

* **Explicit list** -- ``field_dims: [34, 16, 80, ...]`` in config.
* **Auto from feature groups** -- ``field_dims: "auto"`` reads
  ``feature_group_ranges`` from the PLE config / feature_schema.json.
  Each feature group becomes one FM field.
* **Fixed fallback** -- when ``field_dims: "auto"`` but no group_ranges
  are available, splits every ``field_split_size`` (default 20) features
  into one field.

References
----------
* Guo et al., *DeepFM: A Factorization-Machine based Neural Network for CTR
  Prediction*, IJCAI 2017.
* Wang et al., *Deep & Cross Network for Ad Click Predictions*, ADKDD 2017.
* Wang et al., *DCN V2: Improved Deep & Cross Network*, WWW 2021.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .base import AbstractExpert
from .registry import ExpertRegistry

logger = logging.getLogger(__name__)


def _derive_field_dims_from_group_ranges(
    group_ranges: Dict[str, tuple],
    input_dim: int,
) -> List[int]:
    """Derive field_dims from feature_group_ranges.

    Each feature group becomes one FM field.  Groups are sorted by their
    start index.  Any gaps between groups (or a trailing remainder) are
    collected into additional fields so the dims always sum to *input_dim*.

    Parameters
    ----------
    group_ranges : dict
        ``{group_name: (start, end)}`` where ``end`` is exclusive.
    input_dim : int
        Total feature dimension that the field_dims must sum to.

    Returns
    -------
    list[int]
        Per-field dimensions that sum to *input_dim*.
    """
    if not group_ranges:
        return [input_dim]

    sorted_groups = sorted(group_ranges.items(), key=lambda kv: kv[1][0])
    dims: List[int] = []
    prev_end = 0

    for name, (start, end) in sorted_groups:
        start, end = int(start), int(end)
        # Fill any gap before this group
        if start > prev_end:
            dims.append(start - prev_end)
        dims.append(end - start)
        prev_end = end

    # Trailing remainder
    if prev_end < input_dim:
        dims.append(input_dim - prev_end)

    total = sum(dims)
    if total != input_dim:
        logger.warning(
            "field_dims from group_ranges sum to %d but input_dim=%d; "
            "falling back to fixed split",
            total, input_dim,
        )
        return _fixed_split(input_dim, 20)

    return dims


def _fixed_split(input_dim: int, split_size: int) -> List[int]:
    """Split *input_dim* into fields of *split_size*, last field gets the remainder."""
    if split_size <= 0 or split_size >= input_dim:
        return [input_dim]
    n_full = input_dim // split_size
    remainder = input_dim % split_size
    dims = [split_size] * n_full
    if remainder > 0:
        dims.append(remainder)
    return dims


# =============================================================================
# Sub-modules
# =============================================================================

class FMLayer(nn.Module):
    """
    Factorization Machine interaction layer.

    Computes pairwise 2nd-order interactions in O(n*k) where
    n = number of fields and k = embedding dimension::

        y = 0.5 * [ (sum embeddings)^2  -  sum(embeddings^2) ]
    """

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        embeddings : torch.Tensor
            ``[batch, num_fields, embedding_dim]``

        Returns
        -------
        torch.Tensor
            ``[batch, embedding_dim]``
        """
        sum_sq = embeddings.sum(dim=1) ** 2
        sq_sum = (embeddings ** 2).sum(dim=1)
        return 0.5 * (sum_sq - sq_sum)


class CrossNetwork(nn.Module):
    """
    Cross Network from DCN.

    Each layer computes::

        x_{l+1} = x_0 * (W_l @ x_l + b_l)  +  x_l

    Learns explicit feature crosses up to ``(num_layers + 1)``-th order.
    """

    def __init__(self, input_dim: int, num_layers: int = 3):
        super().__init__()
        self.num_layers = num_layers
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, 1) * 0.01)
            for _ in range(num_layers)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim))
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        xl = x
        for i in range(self.num_layers):
            cross = x0 * torch.matmul(xl, self.weights[i]).squeeze(-1)
            xl = cross + self.biases[i] + xl
        return xl


class _DeepNet(nn.Module):
    """Simple MLP used as the Deep branch."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.2,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h))
            elif use_batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# DeepFM Expert
# =============================================================================

@ExpertRegistry.register("deepfm")
class DeepFMExpert(AbstractExpert):
    """
    DeepFM Expert -- FM + Deep for joint low/high-order feature interactions.

    The input feature vector is split into *fields* (logical sub-groups) and
    each field is independently embedded into ``embedding_dim`` before being
    fed to the FM / Deep / Cross branches.  This allows FM to learn meaningful
    2nd-order interactions between semantic feature groups (e.g. demographics
    x product holdings), matching the on-prem 28-field design.

    Config keys
    -----------
    output_dim : int
        Expert output dimension (default 64).
    embedding_dim : int
        Per-field embedding size (default 16).
    field_dims : ``"auto"`` | list[int] | None
        * ``"auto"`` -- derive field boundaries from
          ``feature_group_ranges`` (each feature group = one FM field).
          Falls back to fixed split of ``field_split_size`` if group_ranges
          are unavailable.
        * ``list[int]`` -- explicit per-field sizes.  Must sum to
          ``input_dim``.
        * ``None`` -- treat entire input as 1 field (degenerates to MLP;
          FM interaction is meaningless).
    feature_group_ranges : dict, optional
        ``{group_name: (start, end)}`` injected by train.py from
        ``feature_schema.json``'s ``group_ranges``.  Used when
        ``field_dims == "auto"``.
    field_split_size : int
        Fallback fixed field width when ``field_dims == "auto"`` but no
        group_ranges are available (default 20).
    hidden_dims : list[int]
        Deep branch hidden sizes (default ``[256, 128, 64]``).
    use_fm : bool
        Enable FM branch (default ``True``).
    use_deep : bool
        Enable Deep branch (default ``True``).
    use_cross : bool
        Enable Cross Network branch (default ``False``).
    cross_layers : int
        Number of cross layers (default 3).
    dropout : float
        Dropout rate (default 0.2).
    """

    INTERPRET_DIM = 4

    def __init__(self, input_dim: int, config: Dict[str, Any]):
        super().__init__(input_dim, config)

        self._output_dim: int = config.get("output_dim", 64)
        embedding_dim: int = config.get("embedding_dim", 16)
        hidden_dims: List[int] = config.get("hidden_dims", [256, 128, 64])
        dropout: float = config.get("dropout", 0.2)

        self.use_fm: bool = config.get("use_fm", True)
        self.use_deep: bool = config.get("use_deep", True)
        self.use_cross: bool = config.get("use_cross", False)
        cross_layers: int = config.get("cross_layers", 3)

        # -- Field definitions ------------------------------------------------
        raw_field_dims = config.get("field_dims")

        if raw_field_dims == "auto":
            # Auto-detect from feature_group_ranges (injected by train.py)
            group_ranges: Optional[Dict[str, tuple]] = config.get(
                "feature_group_ranges"
            )
            if group_ranges:
                field_dims_resolved = _derive_field_dims_from_group_ranges(
                    group_ranges, input_dim
                )
                logger.info(
                    "DeepFM field_dims='auto': derived %d fields from "
                    "feature_group_ranges",
                    len(field_dims_resolved),
                )
            else:
                split_size = config.get("field_split_size", 20)
                field_dims_resolved = _fixed_split(input_dim, split_size)
                logger.warning(
                    "DeepFM field_dims='auto' but no feature_group_ranges "
                    "available; falling back to fixed split of %d -> %d fields",
                    split_size, len(field_dims_resolved),
                )
        elif isinstance(raw_field_dims, list):
            field_dims_resolved = list(raw_field_dims)
        elif raw_field_dims is None:
            # Legacy fallback: single field (FM interaction is meaningless)
            field_dims_resolved = [input_dim]
            logger.warning(
                "DeepFM: field_dims not set -- treating entire input as 1 "
                "field.  FM interaction is effectively disabled.  Set "
                "field_dims='auto' to derive fields from feature groups."
            )
        else:
            raise ValueError(
                f"DeepFM: invalid field_dims={raw_field_dims!r}. "
                f"Expected 'auto', list[int], or None."
            )

        self.field_dims = field_dims_resolved
        self.num_fields = len(self.field_dims)

        field_total = sum(self.field_dims)
        if field_total != input_dim:
            raise ValueError(
                f"Sum of field_dims ({field_total}) != input_dim ({input_dim}). "
                f"Provide field_dims that partition the input exactly."
            )

        # -- Per-field embedding layers ----------------------------------------
        self.field_embeddings = nn.ModuleList([
            nn.Linear(d, embedding_dim) for d in self.field_dims
        ])

        flat_emb_dim = self.num_fields * embedding_dim

        # -- FM ----------------------------------------------------------------
        if self.use_fm:
            self.fm = FMLayer()

        # -- Cross Network -----------------------------------------------------
        if self.use_cross:
            self.cross_network = CrossNetwork(flat_emb_dim, cross_layers)

        # -- Deep Network ------------------------------------------------------
        if self.use_deep:
            self.deep_network = _DeepNet(
                input_dim=flat_emb_dim,
                hidden_dims=hidden_dims,
                output_dim=hidden_dims[-1],
                dropout=dropout,
            )

        # -- Output projection -------------------------------------------------
        final_dim = 0
        if self.use_fm:
            final_dim += embedding_dim
        if self.use_cross:
            final_dim += flat_emb_dim
        if self.use_deep:
            final_dim += hidden_dims[-1]

        self.output_layer = nn.Sequential(
            nn.Linear(final_dim, self._output_dim),
            nn.LayerNorm(self._output_dim),
            nn.SiLU(),
        )

        # Interpretable projection: 4D output
        # [low-order interaction, high-order interaction, sparse pattern, dense pattern]
        self.interpret_proj = nn.Linear(self._output_dim, self.INTERPRET_DIM)

        # Stores detached interpretable scores from the last forward pass
        self._interpretable_scores: Optional[torch.Tensor] = None

        logger.info(
            "DeepFMExpert: input=%d, fields=%d, emb=%d, "
            "FM=%s, Deep=%s, Cross=%s -> output=%d  (params=%s)",
            input_dim, self.num_fields, embedding_dim,
            self.use_fm, self.use_deep, self.use_cross,
            self._output_dim, f"{self.count_parameters():,}",
        )

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            ``[batch, input_dim]``

        Returns
        -------
        torch.Tensor
            ``[batch, output_dim]``

        Side Effects
        ------------
        Stores ``self._interpretable_scores`` (detached ``[batch, 4]``)
        with semantic dimensions:
        ``[low_order_interaction, high_order_interaction, sparse_pattern, dense_pattern]``.
        """
        outputs: list[torch.Tensor] = []

        # Per-field embedding
        field_embs: list[torch.Tensor] = []
        offset = 0
        for dim_i, emb in zip(self.field_dims, self.field_embeddings):
            field_embs.append(emb(x[:, offset : offset + dim_i]))
            offset += dim_i

        # [batch, num_fields, embedding_dim]
        embeddings = torch.stack(field_embs, dim=1)

        # FM: field-level 2nd-order interactions
        if self.use_fm:
            outputs.append(self.fm(embeddings))

        # Flatten for Cross / Deep
        flat = embeddings.view(x.size(0), -1)

        # Cross Network: explicit feature crosses
        if self.use_cross:
            outputs.append(self.cross_network(flat))

        # Deep: higher-order non-linear interactions
        if self.use_deep:
            outputs.append(self.deep_network(flat))

        combined = torch.cat(outputs, dim=-1)
        expert_output = self.output_layer(combined)

        # Interpretable projection (detached -- not part of main gradient path)
        self._interpretable_scores = self.interpret_proj(expert_output).detach()

        return expert_output
