"""
Loss function implementations for the PLE multi-task learning platform.

Each loss is a standalone ``nn.Module`` that can be instantiated from config.
The factory helper :func:`build_loss` maps a :class:`LossType` enum to the
correct module with the right hyper-parameters.

Included losses
---------------
* **FocalLoss** -- class-imbalance-aware BCE / CE (Lin et al., 2017)
* **QuantileLoss** -- pinball loss for distributional regression
* **InfoNCELoss** -- contrastive / retrieval loss (Oord et al., 2018)
* **ListNetLoss** -- listwise ranking loss
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .types import LossType

__all__ = [
    "FocalLoss",
    "QuantileLoss",
    "InfoNCELoss",
    "ListNetLoss",
    "build_loss",
]


# ════════════════════════════════════════════════════════════════
# Focal Loss
# ════════════════════════════════════════════════════════════════


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.

    Supports both binary (1-D logits) and multi-class (2-D logits) inputs.

    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

    Args:
        alpha: Weighting factor for the positive class (binary) or uniform
            weight (multi-class).  Set to ``-1`` to disable alpha weighting.
        gamma: Focusing parameter.  ``gamma = 0`` recovers standard CE.
        reduction: ``"none"`` | ``"mean"`` | ``"sum"``.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "none",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            inputs: Raw logits -- shape ``(N,)`` for binary or ``(N, C)`` for
                multi-class classification.
            targets: Ground-truth labels -- shape ``(N,)`` with values in
                ``{0, 1}`` (binary) or ``{0, ..., C-1}`` (multi-class).

        Returns:
            Loss tensor whose shape depends on *reduction*.
        """
        if inputs.dim() == 1 or (inputs.dim() == 2 and inputs.size(1) == 1):
            return self._binary_focal(inputs.view(-1), targets.view(-1))
        return self._multiclass_focal(inputs, targets)

    # ── private helpers ─────────────────────────────────────────

    def _binary_focal(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(inputs)
        ce = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction="none")
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_weight = alpha_t * (1.0 - p_t) ** self.gamma
        loss = focal_weight * ce
        return self._reduce(loss)

    def _multiclass_focal(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(inputs, targets.long(), reduction="none")
        p = F.softmax(inputs, dim=-1)
        p_t = p.gather(1, targets.long().unsqueeze(1)).squeeze(1)
        focal_weight = self.alpha * (1.0 - p_t) ** self.gamma
        loss = focal_weight * ce
        return self._reduce(loss)

    def _reduce(self, loss: torch.Tensor) -> torch.Tensor:
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# ════════════════════════════════════════════════════════════════
# Quantile Loss
# ════════════════════════════════════════════════════════════════


class QuantileLoss(nn.Module):
    """Pinball (quantile) loss for distributional regression.

    Args:
        quantiles: Target quantile levels (e.g. ``[0.1, 0.5, 0.9]``).
    """

    def __init__(self, quantiles: Optional[List[float]] = None) -> None:
        super().__init__()
        self.quantiles = quantiles or [0.1, 0.5, 0.9]

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute quantile loss.

        Args:
            predictions: Shape ``(N,)`` or ``(N, Q)`` where *Q* matches
                ``len(self.quantiles)``.
            targets: Shape ``(N,)``.

        Returns:
            Scalar loss (mean over samples and quantiles).
        """
        losses = []
        for i, q in enumerate(self.quantiles):
            pred = predictions[:, i] if predictions.dim() > 1 else predictions
            errors = targets - pred
            losses.append(torch.max((q - 1.0) * errors, q * errors))
        return torch.stack(losses, dim=-1).mean()


# ════════════════════════════════════════════════════════════════
# InfoNCE / NT-Xent (Contrastive)
# ════════════════════════════════════════════════════════════════


class InfoNCELoss(nn.Module):
    """InfoNCE / NT-Xent contrastive loss.

    When explicit negatives are provided the loss is computed over the
    ``[positive ; negatives]`` set per sample.  Otherwise, *in-batch*
    negatives are used (all other positives in the mini-batch become
    negatives).

    Reference:
        Oord et al., "Representation Learning with Contrastive Predictive
        Coding", 2018.

    Args:
        temperature: Scaling factor applied to cosine similarities.
        reduction: ``"mean"`` | ``"sum"`` | ``"none"``.
    """

    def __init__(self, temperature: float = 0.07, reduction: str = "mean") -> None:
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        query: torch.Tensor,
        positive: torch.Tensor,
        negatives: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute InfoNCE loss.

        Args:
            query: ``(B, D)`` query embeddings.
            positive: ``(B, D)`` positive-key embeddings.
            negatives: Optional ``(B, N, D)`` explicit negative embeddings.
                If ``None``, in-batch negatives are used.

        Returns:
            Scalar (or per-sample) loss depending on *reduction*.
        """
        query = F.normalize(query, p=2, dim=-1)
        positive = F.normalize(positive, p=2, dim=-1)

        pos_score = (query * positive).sum(dim=-1) / self.temperature  # (B,)

        if negatives is not None:
            negatives = F.normalize(negatives, p=2, dim=-1)
            neg_scores = torch.bmm(
                negatives, query.unsqueeze(-1)
            ).squeeze(-1) / self.temperature  # (B, N)
            logits = torch.cat([pos_score.unsqueeze(1), neg_scores], dim=1)  # (B, 1+N)
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
            return F.cross_entropy(logits, labels, reduction=self.reduction)

        # In-batch negatives: similarity matrix (B, B)
        sim = torch.mm(query, positive.t()) / self.temperature
        labels = torch.arange(sim.size(0), device=sim.device)
        return F.cross_entropy(sim, labels, reduction=self.reduction)


# ════════════════════════════════════════════════════════════════
# ListNet (Ranking)
# ════════════════════════════════════════════════════════════════


class ListNetLoss(nn.Module):
    """ListNet listwise ranking loss.

    Computes the KL divergence between the softmax of predicted scores and
    the softmax of ground-truth relevance labels.

    Reference:
        Cao et al., "Learning to Rank: From Pairwise Approach to Listwise
        Approach", ICML 2007.
    """

    def __init__(self, eps: float = 1e-10) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute ListNet loss.

        Args:
            scores: Predicted relevance scores ``(B, L)`` or ``(L,)`` for a
                single list.
            labels: Ground-truth relevance ``(B, L)`` or ``(L,)``.

        Returns:
            Scalar loss.
        """
        pred_probs = torch.softmax(scores.float(), dim=-1)
        true_probs = torch.softmax(labels.float(), dim=-1)
        return -(true_probs * torch.log(pred_probs + self.eps)).sum(dim=-1).mean()


# ════════════════════════════════════════════════════════════════
# Factory
# ════════════════════════════════════════════════════════════════


def build_loss(
    loss_type: LossType,
    *,
    # Focal
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    # Huber
    huber_delta: float = 1.0,
    # Quantile
    quantiles: Optional[List[float]] = None,
    # InfoNCE
    temperature: float = 0.07,
    # Pos-weight (BCE)
    pos_weight: Optional[float] = None,
    # Label smoothing
    label_smoothing: float = 0.0,
    # Common
    reduction: str = "none",
) -> nn.Module:
    """Instantiate a loss module from a :class:`LossType` enum value.

    This is the recommended entry point -- callers never need to import
    individual loss classes directly.

    Args:
        loss_type: Which loss to build.
        **kwargs: Forwarded to the concrete loss constructor.

    Returns:
        An ``nn.Module`` whose ``forward`` signature matches the conventions
        used by the task heads.

    Raises:
        ValueError: If *loss_type* is not recognised.
    """
    if loss_type == LossType.BCE:
        pw = torch.tensor([pos_weight]) if pos_weight is not None else None
        return nn.BCEWithLogitsLoss(pos_weight=pw, reduction=reduction)

    if loss_type == LossType.FOCAL:
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction=reduction)

    if loss_type == LossType.CROSS_ENTROPY:
        return nn.CrossEntropyLoss(
            label_smoothing=label_smoothing, reduction=reduction,
            ignore_index=-1,  # skip samples with no valid label
        )

    if loss_type == LossType.MSE:
        return nn.MSELoss(reduction=reduction)

    if loss_type == LossType.MAE:
        return nn.L1Loss(reduction=reduction)

    if loss_type == LossType.HUBER:
        return nn.HuberLoss(delta=huber_delta, reduction=reduction)

    if loss_type == LossType.QUANTILE:
        return QuantileLoss(quantiles=quantiles)

    if loss_type == LossType.LISTNET:
        return ListNetLoss()

    if loss_type == LossType.INFONCE:
        return InfoNCELoss(temperature=temperature, reduction="mean")

    if loss_type == LossType.AUTO:
        raise ValueError(
            "LossType.AUTO cannot be used directly with build_loss(); "
            "resolve it to a concrete type first."
        )

    raise ValueError(f"Unknown loss type: {loss_type!r}")
