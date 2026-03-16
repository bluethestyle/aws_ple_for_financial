"""
AutoInt Expert -- Automatic Feature Interaction Learning via
multi-head self-attention.

Splits input features into fields, embeds each field, and applies
multi-head self-attention to learn arbitrary-order feature interactions.
The attention mechanism automatically identifies relevant feature
combinations without manual feature engineering.

Architecture::

    input (input_dim)
        -> field_embeddings  [num_fields, embedding_dim]
        -> MultiHeadSelfAttention x num_layers
        -> flatten
        -> output_proj (Linear -> LN -> SiLU) -> [output_dim]

References
----------
Song et al., *AutoInt: Automatic Feature Interaction Learning via
Self-Attentive Neural Networks*, CIKM 2019.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AbstractExpert
from .registry import ExpertRegistry

logger = logging.getLogger(__name__)


class InteractionAttention(nn.Module):
    """Multi-head self-attention for feature interaction learning."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            ``[batch, num_fields, embed_dim]``

        Returns
        -------
        torch.Tensor
            ``[batch, num_fields, embed_dim]``
        """
        B, N, D = x.shape
        residual = x

        Q = self.W_q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.W_out(out)

        return self.norm(out + residual)


@ExpertRegistry.register("autoint")
class AutoIntExpert(AbstractExpert):
    """
    AutoInt expert for automatic feature interaction learning.

    Config keys
    -----------
    output_dim : int
        Expert output dimension (default 64).
    embedding_dim : int
        Per-field embedding dimension (default 16).
    num_heads : int
        Number of attention heads (default 4).
    num_layers : int
        Number of self-attention layers (default 3).
    field_dims : list[int] or None
        Sizes of each logical field.  Must sum to ``input_dim``.
        If None, splits input into chunks of size ``embedding_dim``.
    dropout : float
        Dropout rate (default 0.1).
    """

    def __init__(self, input_dim: int, config: Dict[str, Any]):
        super().__init__(input_dim, config)

        self._output_dim: int = config.get("output_dim", 64)
        embedding_dim: int = config.get("embedding_dim", 16)
        num_heads: int = config.get("num_heads", 4)
        num_layers: int = config.get("num_layers", 3)
        dropout: float = config.get("dropout", 0.1)

        # Field definitions
        field_dims: Optional[List[int]] = config.get("field_dims")
        if field_dims is None:
            # Auto-split into equal chunks
            num_fields = max(input_dim // embedding_dim, 1)
            field_dims = [input_dim // num_fields] * num_fields
            remainder = input_dim - sum(field_dims)
            if remainder > 0:
                field_dims[-1] += remainder
        self.field_dims = field_dims
        self.num_fields = len(field_dims)

        # Per-field embedding layers
        self.field_embeddings = nn.ModuleList([
            nn.Linear(d, embedding_dim) for d in self.field_dims
        ])

        # Self-attention layers
        self.attention_layers = nn.ModuleList([
            InteractionAttention(embedding_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Output projection
        flat_dim = self.num_fields * embedding_dim
        self.output_proj = nn.Sequential(
            nn.Linear(flat_dim, self._output_dim),
            nn.LayerNorm(self._output_dim),
            nn.SiLU(),
        )

        logger.info(
            "AutoIntExpert: input=%d, fields=%d, emb=%d, heads=%d, "
            "layers=%d -> output=%d  (params=%s)",
            input_dim, self.num_fields, embedding_dim, num_heads,
            num_layers, self._output_dim, f"{self.count_parameters():,}",
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

        # [batch, num_fields, embedding_dim]
        h = torch.stack(field_embs, dim=1)

        # Self-attention layers
        for attn_layer in self.attention_layers:
            h = attn_layer(h)

        # Flatten and project
        flat = h.view(x.size(0), -1)
        return self.output_proj(flat)
