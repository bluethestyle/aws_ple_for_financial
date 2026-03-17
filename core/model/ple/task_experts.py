"""
Task Expert Architecture -- GroupEncoder + ClusterEmbedding + TaskHead.

Matches the original gotothemoon v3.2 GroupTaskExpertBasket design:
- GroupEncoder: shared MLP per task group (4 groups, not 16 individual)
- ClusterEmbedding: GMM cluster -> learned embedding (condition on user segment)
- TaskHead: lightweight per-task MLP projection

Flow:
    CGC output (extraction_dim)
    -> GroupEncoder[task_group] -> (group_output_dim)
    + ClusterEmbedding(cluster_id) -> (cluster_embed_dim)
    -> concat (96D) -> TaskTower[task_name] -> prediction

No TaskHead bottleneck -- GroupEncoder output goes directly to TaskTower.
Parameter efficiency: ~88% reduction vs independent experts per task.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ============================================================================
# GroupEncoder
# ============================================================================

class GroupEncoder(nn.Module):
    """Per-group shared MLP encoder.

    One instance is shared across all tasks in the same task group,
    providing parameter efficiency (4 encoders instead of 16).

    Args:
        input_dim: Input feature width (CGC extraction output).
        hidden_dim: Hidden layer width.
        output_dim: Output representation width.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, input_dim) -> (batch, output_dim)."""
        return self.network(x)


# ============================================================================
# ClusterEmbedding
# ============================================================================

class ClusterEmbedding(nn.Module):
    """GMM cluster -> learned embedding for cluster-conditional routing.

    Supports both hard (cluster_id index) and soft (cluster_probs weighted)
    lookup modes.

    Args:
        n_clusters: Number of GMM clusters.
        embed_dim: Embedding dimension per cluster.
    """

    def __init__(self, n_clusters: int = 20, embed_dim: int = 32):
        super().__init__()
        self.n_clusters = n_clusters
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(n_clusters, embed_dim)

    def forward(
        self,
        cluster_ids: Optional[torch.Tensor] = None,
        cluster_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Lookup cluster embeddings.

        Args:
            cluster_ids: ``(batch,)`` hard cluster assignments (long).
            cluster_probs: ``(batch, n_clusters)`` soft cluster probabilities.

        Returns:
            ``(batch, embed_dim)`` cluster embedding.  If both inputs are
            ``None``, returns a zero tensor.
        """
        if cluster_probs is not None:
            # Soft lookup: weighted combination of all cluster embeddings
            return cluster_probs @ self.embedding.weight  # (batch, embed_dim)
        elif cluster_ids is not None:
            # Hard lookup: index into embedding table
            return self.embedding(cluster_ids.long())  # (batch, embed_dim)
        else:
            # No cluster info -- return zeros (will be handled by caller)
            batch_size = 1  # fallback; caller should avoid this path
            device = self.embedding.weight.device
            return torch.zeros(batch_size, self.embed_dim, device=device)


# ============================================================================
# GroupTaskExpertBasket (orchestrator)
# ============================================================================

class GroupTaskExpertBasket(nn.Module):
    """Orchestrator for the GroupEncoder + ClusterEmbedding + TaskHead architecture.

    Owns one :class:`GroupEncoder` per task group, a single
    :class:`ClusterEmbedding`, and one :class:`TaskHead` per task.

    Args:
        input_dim: CGC extraction output dimension.
        task_names: List of all task name strings.
        task_group_map: ``{task_name: group_name}`` mapping.
        group_hidden_dim: Hidden dimension for GroupEncoder.
        group_output_dim: Output dimension for GroupEncoder.
        cluster_embed_dim: Embedding dimension for ClusterEmbedding
            (set to 0 to disable cluster conditioning).
        n_clusters: Number of GMM clusters (0 disables clustering).
        task_output_dim: Output dimension of each TaskHead.
        task_head_hidden_dim: Hidden dimension for TaskHead.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        task_names: List[str],
        task_group_map: Dict[str, str],
        group_hidden_dim: int = 128,
        group_output_dim: int = 64,
        cluster_embed_dim: int = 32,
        n_clusters: int = 0,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.task_names = list(task_names)
        self.task_group_map = dict(task_group_map)

        # Identify unique groups
        unique_groups = sorted(set(task_group_map.values()))

        # Build one GroupEncoder per unique group
        self.group_encoders = nn.ModuleDict()
        for group_name in unique_groups:
            self.group_encoders[group_name] = GroupEncoder(
                input_dim=input_dim,
                hidden_dim=group_hidden_dim,
                output_dim=group_output_dim,
                dropout=dropout,
            )

        # Default encoder for tasks without a group mapping
        self.default_encoder = GroupEncoder(
            input_dim=input_dim,
            hidden_dim=group_hidden_dim,
            output_dim=group_output_dim,
            dropout=dropout,
        )

        # Cluster embedding (disabled when n_clusters <= 0)
        if n_clusters > 0 and cluster_embed_dim > 0:
            self.cluster_embedding = ClusterEmbedding(
                n_clusters=n_clusters,
                embed_dim=cluster_embed_dim,
            )
            self._output_dim = group_output_dim + cluster_embed_dim
        else:
            self.cluster_embedding = None
            self._output_dim = group_output_dim

        # Log architecture summary
        logger.info(
            "GroupTaskExpertBasket: %d groups (%s), "
            "cluster_embed=%s, output_dim=%d → TaskTower directly",
            len(unique_groups),
            unique_groups,
            f"{cluster_embed_dim}D (n={n_clusters})" if n_clusters > 0 else "disabled",
            self._output_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        task_name: str,
        cluster_ids: Optional[torch.Tensor] = None,
        cluster_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Process a single task through GroupEncoder + ClusterEmbedding + TaskHead.

        Args:
            x: ``(batch, input_dim)`` CGC extraction output for this task.
            task_name: Which task to process.
            cluster_ids: ``(batch,)`` hard cluster assignments (optional).
            cluster_probs: ``(batch, n_clusters)`` soft cluster probs (optional).

        Returns:
            ``(batch, task_output_dim)`` task expert output.
        """
        # 1. Route to group encoder
        group = self.task_group_map.get(task_name)
        if group and group in self.group_encoders:
            group_out = self.group_encoders[group](x)
        else:
            group_out = self.default_encoder(x)

        # 2. Cluster embedding (if enabled and cluster info available)
        if (self.cluster_embedding is not None
                and (cluster_ids is not None or cluster_probs is not None)):
            cluster_out = self.cluster_embedding(cluster_ids, cluster_probs)
            return torch.cat([group_out, cluster_out], dim=-1)

        return group_out

    @property
    def output_dim(self) -> int:
        """Output dimension: group_output_dim (+ cluster_embed_dim if enabled)."""
        return self._output_dim
