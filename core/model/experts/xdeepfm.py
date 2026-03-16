"""
xDeepFM Expert -- eXtreme Deep Factorization Machine with Compressed
Interaction Network (CIN).

Extends DeepFM with explicit high-order feature interactions via CIN,
which generates feature interactions at each layer in a vector-wise
manner (rather than bit-wise as in standard DNNs).

Architecture::

    input (input_dim)
        -> field_embeddings  [num_fields, embedding_dim]
        -> CIN (Compressed Interaction Network)   -> cin_output
        -> DNN (standard Deep branch)              -> dnn_output
        -> concat [cin_output, dnn_output]
        -> output_proj (Linear -> LN -> SiLU) -> [output_dim]

References
----------
Lian et al., *xDeepFM: Combining Explicit and Implicit Feature
Interactions for Recommender Systems*, KDD 2018.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AbstractExpert
from .registry import ExpertRegistry

logger = logging.getLogger(__name__)


class CINLayer(nn.Module):
    """Single CIN (Compressed Interaction Network) layer.

    Computes vector-wise interactions between the current hidden state
    and the original field embeddings.
    """

    def __init__(self, num_fields: int, prev_layer_size: int, cin_layer_size: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=num_fields * prev_layer_size,
            out_channels=cin_layer_size,
            kernel_size=1,
        )

    def forward(
        self,
        x0: torch.Tensor,
        xk: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x0 : torch.Tensor
            ``[batch, num_fields, embedding_dim]`` original embeddings.
        xk : torch.Tensor
            ``[batch, prev_layer_size, embedding_dim]`` previous CIN layer.

        Returns
        -------
        torch.Tensor
            ``[batch, cin_layer_size, embedding_dim]``
        """
        B, H, D = xk.shape
        M = x0.shape[1]

        # Outer product: [B, H*M, D]
        x0_expand = x0.unsqueeze(1).expand(-1, H, -1, -1)   # [B, H, M, D]
        xk_expand = xk.unsqueeze(2).expand(-1, -1, M, -1)   # [B, H, M, D]
        outer = (x0_expand * xk_expand).reshape(B, H * M, D)  # [B, H*M, D]

        # 1x1 convolution to compress
        out = self.conv(outer)  # [B, cin_layer_size, D]
        return out


class CIN(nn.Module):
    """Compressed Interaction Network."""

    def __init__(
        self,
        num_fields: int,
        cin_layer_sizes: List[int],
        embedding_dim: int,
    ):
        super().__init__()
        self.cin_layers = nn.ModuleList()
        prev_size = num_fields

        for size in cin_layer_sizes:
            self.cin_layers.append(CINLayer(num_fields, prev_size, size))
            prev_size = size

        self.output_dim = sum(cin_layer_sizes)

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x0 : torch.Tensor
            ``[batch, num_fields, embedding_dim]``

        Returns
        -------
        torch.Tensor
            ``[batch, sum(cin_layer_sizes)]``
        """
        pooled_outputs: list[torch.Tensor] = []
        xk = x0

        for cin_layer in self.cin_layers:
            xk = F.relu(cin_layer(x0, xk))
            # Sum pooling over embedding dimension
            pooled_outputs.append(xk.sum(dim=-1))  # [B, cin_layer_size]

        return torch.cat(pooled_outputs, dim=-1)


@ExpertRegistry.register("xdeepfm")
class XDeepFMExpert(AbstractExpert):
    """
    xDeepFM expert with Compressed Interaction Network.

    Config keys
    -----------
    output_dim : int
        Expert output dimension (default 64).
    embedding_dim : int
        Per-field embedding dimension (default 16).
    cin_layer_sizes : list[int]
        CIN hidden layer sizes (default ``[64, 64]``).
    dnn_hidden_dims : list[int]
        DNN hidden layer sizes (default ``[128, 64]``).
    field_dims : list[int] or None
        Sizes of each logical field.  Must sum to ``input_dim``.
        If None, auto-splits.
    dropout : float
        Dropout rate (default 0.2).
    """

    def __init__(self, input_dim: int, config: Dict[str, Any]):
        super().__init__(input_dim, config)

        self._output_dim: int = config.get("output_dim", 64)
        embedding_dim: int = config.get("embedding_dim", 16)
        cin_layer_sizes: List[int] = config.get("cin_layer_sizes", [64, 64])
        dnn_hidden_dims: List[int] = config.get("dnn_hidden_dims", [128, 64])
        dropout: float = config.get("dropout", 0.2)

        # Field definitions
        field_dims: Optional[List[int]] = config.get("field_dims")
        if field_dims is None:
            num_fields = max(input_dim // embedding_dim, 1)
            field_dims = [input_dim // num_fields] * num_fields
            remainder = input_dim - sum(field_dims)
            if remainder > 0:
                field_dims[-1] += remainder
        self.field_dims = field_dims
        self.num_fields = len(field_dims)

        # Per-field embedding
        self.field_embeddings = nn.ModuleList([
            nn.Linear(d, embedding_dim) for d in self.field_dims
        ])

        # CIN branch
        self.cin = CIN(self.num_fields, cin_layer_sizes, embedding_dim)

        # DNN branch
        flat_dim = self.num_fields * embedding_dim
        dnn_layers: list[nn.Module] = []
        prev_dim = flat_dim
        for hdim in dnn_hidden_dims:
            dnn_layers.extend([
                nn.Linear(prev_dim, hdim),
                nn.LayerNorm(hdim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hdim
        self.dnn = nn.Sequential(*dnn_layers)

        # Output projection
        combined_dim = self.cin.output_dim + dnn_hidden_dims[-1]
        self.output_proj = nn.Sequential(
            nn.Linear(combined_dim, self._output_dim),
            nn.LayerNorm(self._output_dim),
            nn.SiLU(),
        )

        logger.info(
            "XDeepFMExpert: input=%d, fields=%d, emb=%d, "
            "CIN=%s, DNN=%s -> output=%d  (params=%s)",
            input_dim, self.num_fields, embedding_dim,
            cin_layer_sizes, dnn_hidden_dims,
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
        """
        # Per-field embedding
        field_embs: list[torch.Tensor] = []
        offset = 0
        for dim_i, emb in zip(self.field_dims, self.field_embeddings):
            field_embs.append(emb(x[:, offset:offset + dim_i]))
            offset += dim_i

        embeddings = torch.stack(field_embs, dim=1)  # [B, F, E]

        # CIN branch
        cin_out = self.cin(embeddings)  # [B, sum(cin_sizes)]

        # DNN branch
        flat = embeddings.view(x.size(0), -1)
        dnn_out = self.dnn(flat)  # [B, dnn_hidden_dims[-1]]

        # Combine and project
        combined = torch.cat([cin_out, dnn_out], dim=-1)
        return self.output_proj(combined)
