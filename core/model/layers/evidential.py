"""
Evidential Deep Learning layer for uncertainty quantification.

This module outputs **epistemic uncertainty** alongside point predictions by
parameterising a *higher-order* distribution over the model's predictive
distribution:

* **Binary classification** -- Beta(alpha, beta)
    * prediction = alpha / (alpha + beta)
    * uncertainty = 1 / (alpha + beta)

* **Multi-class classification** -- Dirichlet(alpha_1, ..., alpha_K)
    * prediction = alpha_k / sum(alpha)
    * uncertainty = K / sum(alpha)

* **Regression** -- Normal-Inverse-Gamma (mu, v, alpha, beta)
    * prediction = mu
    * uncertainty = beta / (v * (alpha - 1))

The evidential loss consists of a Type-II Maximum Likelihood term plus a KL
regularisation term that is linearly annealed over the first
*annealing_epochs* training epochs.

References:
    Sensoy et al., "Evidential Deep Learning to Quantify Classification
    Uncertainty", NeurIPS 2018.

    Amini et al., "Deep Evidential Regression", NeurIPS 2020.

Example::

    layer = EvidentialLayer(input_dim=64, task_type="binary")
    prediction, evidence_info = layer(features)
    loss = layer.compute_evidential_loss(evidence_info, targets, epoch=5)
    uncertainty = evidence_info["uncertainty"]  # (B,)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["EvidentialLayer"]


# ════════════════════════════════════════════════════════════════
# KL divergence helpers
# ════════════════════════════════════════════════════════════════


def _kl_beta(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """KL divergence from Beta(alpha, beta) to Beta(1, 1) (uniform).

    Args:
        alpha: Shape ``(...)``, alpha > 0.
        beta: Shape ``(...)``, beta > 0.

    Returns:
        Per-element KL divergence, same shape as inputs.
    """
    return (
        torch.lgamma(alpha + beta)
        - torch.lgamma(alpha)
        - torch.lgamma(beta)
        + (alpha - 1) * torch.digamma(alpha)
        + (beta - 1) * torch.digamma(beta)
        - (alpha + beta - 2) * torch.digamma(alpha + beta)
    )


def _kl_dirichlet(alpha: torch.Tensor) -> torch.Tensor:
    """KL divergence from Dir(alpha) to Dir(1, ..., 1) (uniform simplex).

    Args:
        alpha: Shape ``(B, K)``, concentration parameters, all > 0.

    Returns:
        Per-sample KL divergence, shape ``(B,)``.
    """
    K = alpha.size(-1)
    S = alpha.sum(dim=-1, keepdim=True)  # (B, 1)
    return (
        torch.lgamma(S.squeeze(-1))
        - torch.lgamma(torch.tensor(float(K), device=alpha.device, dtype=alpha.dtype))
        - torch.lgamma(alpha).sum(dim=-1)
        + ((alpha - 1) * (torch.digamma(alpha) - torch.digamma(S))).sum(dim=-1)
    )


# ════════════════════════════════════════════════════════════════
# Evidential Layer
# ════════════════════════════════════════════════════════════════


class EvidentialLayer(nn.Module):
    """Evidential Deep Learning layer providing calibrated uncertainty.

    This layer sits after a task-specific expert/tower and transforms the
    hidden representation into distribution parameters that encode both
    the prediction and the model's *epistemic uncertainty*.

    Args:
        input_dim: Dimensionality of the incoming feature vector.
        task_type: One of ``"binary"``, ``"multiclass"``, or ``"regression"``.
        output_dim: Number of classes for multiclass; ignored for binary
            (always 2 evidence neurons) and regression (always 4).
        kl_lambda: Weight of the KL regularisation term in the loss.
        annealing_epochs: Number of epochs over which the KL weight is
            linearly annealed from 0 to ``kl_lambda``.
    """

    SUPPORTED_TYPES = {"binary", "multiclass", "regression"}

    def __init__(
        self,
        input_dim: int = 32,
        task_type: str = "binary",
        output_dim: int = 1,
        kl_lambda: float = 0.01,
        annealing_epochs: int = 10,
    ) -> None:
        super().__init__()

        if task_type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"task_type must be one of {self.SUPPORTED_TYPES}, got '{task_type}'."
            )

        self.input_dim = input_dim
        self.task_type = task_type
        self.output_dim = output_dim
        self.kl_lambda = kl_lambda
        self.annealing_epochs = annealing_epochs

        # Evidence projection
        if task_type == "binary":
            self.evidence_layer = nn.Linear(input_dim, 2)
        elif task_type == "multiclass":
            self.evidence_layer = nn.Linear(input_dim, output_dim)
        else:  # regression
            self.evidence_layer = nn.Linear(input_dim, 4)

    # ── Forward ─────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute prediction and evidential uncertainty.

        Args:
            x: Expert feature tensor ``(B, input_dim)``.
            valid_mask: Optional ``(B,)`` boolean / 0-1 float tensor. When
                supplied, rows where ``valid_mask == 0`` are guaranteed to
                have a finite prediction and max-uncertainty output even if
                the input row contains NaN / Inf values. When ``None``, the
                layer auto-detects non-finite rows and marks them invalid.

        Returns:
            ``(prediction, evidence_info)`` where ``evidence_info`` also
            includes ``"valid_mask"`` (effective mask) so that the loss
            function can skip invalid rows.
        """
        # Sprint 2 S3: auto-detect invalid rows if no explicit mask given.
        if valid_mask is None:
            finite = torch.isfinite(x).all(dim=-1)
            valid_mask = finite.to(x.dtype)
        else:
            valid_mask = valid_mask.to(x.dtype)
            if valid_mask.dim() != 1 or valid_mask.shape[0] != x.shape[0]:
                raise ValueError(
                    f"valid_mask shape {tuple(valid_mask.shape)} must be "
                    f"(B,) matching x's batch dim {x.shape[0]}"
                )

        # Replace any non-finite inputs with zeros before the linear layer
        # so the forward pass never produces NaN gradients.
        safe_x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if self.task_type == "binary":
            pred, info = self._forward_binary(safe_x)
        elif self.task_type == "multiclass":
            pred, info = self._forward_multiclass(safe_x)
        else:
            pred, info = self._forward_regression(safe_x)

        # Apply the mask: invalid rows keep a neutral prediction (0.5 for
        # binary, 1/K for multiclass, 0.0 for regression) and max uncertainty.
        if (valid_mask == 0).any():
            pred, info = self._apply_valid_mask(pred, info, valid_mask)
        info["valid_mask"] = valid_mask
        return pred, info

    def _apply_valid_mask(
        self,
        pred: torch.Tensor,
        info: Dict[str, torch.Tensor],
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Overwrite invalid-row predictions with a neutral + max-uncertainty
        placeholder. Keeps the evidence parameters in the info dict
        untouched so downstream loss functions can still read them for
        valid rows via the returned mask."""
        if self.task_type == "binary":
            neutral = torch.full_like(pred, 0.5)
        elif self.task_type == "multiclass":
            neutral = torch.full_like(pred, 1.0 / float(self.output_dim))
        else:
            neutral = torch.zeros_like(pred)

        mask_3d = valid_mask
        if pred.dim() == 2 and valid_mask.dim() == 1:
            mask_3d = valid_mask.unsqueeze(-1)
        pred = torch.where(mask_3d.bool(), pred, neutral)

        # Uncertainty → very large value where invalid (model says "I don't know").
        unc = info.get("uncertainty")
        if unc is not None:
            if unc.dim() != valid_mask.dim():
                inv_mask = (valid_mask == 0).unsqueeze(-1) if unc.dim() > 1 else (valid_mask == 0)
            else:
                inv_mask = valid_mask == 0
            # Use 1.0 as the max-uncertainty canonical value; loss functions
            # already clamp this per task.
            unc = torch.where(inv_mask, torch.ones_like(unc), unc)
            info["uncertainty"] = unc
        info["prediction"] = pred
        return pred, info

    def _forward_binary(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raw = self.evidence_layer(x)  # (B, 2)
        alpha = F.softplus(raw[:, 0]) + 1.0
        beta = F.softplus(raw[:, 1]) + 1.0

        pred = alpha / (alpha + beta)
        uncertainty = 1.0 / (alpha + beta)

        return pred, {
            "prediction": pred,
            "uncertainty": uncertainty,
            "alpha": alpha,
            "beta": beta,
        }

    def _forward_multiclass(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raw = self.evidence_layer(x)  # (B, K)
        alpha = F.softplus(raw) + 1.0

        S = alpha.sum(dim=-1, keepdim=True)  # (B, 1)
        pred = alpha / S
        uncertainty = float(self.output_dim) / S.squeeze(-1)

        return pred, {
            "prediction": pred,
            "uncertainty": uncertainty,
            "alpha": alpha,
        }

    def _forward_regression(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raw = self.evidence_layer(x)  # (B, 4)

        mu = raw[:, 0]
        v = F.softplus(raw[:, 1]) + 1e-6
        alpha = F.softplus(raw[:, 2]) + 1.0 + 1e-6  # must be > 1
        beta = F.softplus(raw[:, 3]) + 1e-6

        pred = mu
        uncertainty = beta / (v * (alpha - 1)).clamp(min=1e-6)

        return pred, {
            "prediction": pred,
            "uncertainty": uncertainty,
            "mu": mu,
            "v": v,
            "alpha": alpha,
            "beta": beta,
        }

    # ── Loss ────────────────────────────────────────────────────

    def compute_evidential_loss(
        self,
        evidence_info: Dict[str, torch.Tensor],
        target: torch.Tensor,
        epoch: int = 0,
    ) -> torch.Tensor:
        """Compute the evidential loss for the configured task type.

        The loss is composed of a Type-II Maximum Likelihood (or NIG NLL)
        term plus a KL regularisation term that is linearly annealed over
        the first ``annealing_epochs`` epochs.

        Args:
            evidence_info: Dict returned by :meth:`forward`.
            target: Ground-truth labels.  Shape ``(B,)`` for
                binary/regression, ``(B,)`` integer indices for multiclass.
            epoch: Current training epoch (0-indexed) for KL annealing.

        Returns:
            Scalar loss.
        """
        if self.task_type == "binary":
            return self._loss_binary(evidence_info, target, epoch)
        if self.task_type == "multiclass":
            return self._loss_multiclass(evidence_info, target, epoch)
        return self._loss_regression(evidence_info, target, epoch)

    def _loss_binary(
        self,
        info: Dict[str, torch.Tensor],
        target: torch.Tensor,
        epoch: int,
    ) -> torch.Tensor:
        """Type-II ML + KL loss for Beta distribution."""
        alpha = info["alpha"]
        beta_ = info["beta"]
        S = alpha + beta_
        t = target.float()

        # Expected cross-entropy under Beta
        loss = (
            t * (torch.digamma(S) - torch.digamma(alpha))
            + (1 - t) * (torch.digamma(S) - torch.digamma(beta_))
        )

        # KL with annealing
        anneal = min(1.0, epoch / max(self.annealing_epochs, 1))
        alpha_t = t * (alpha - 1) * (1 - anneal) + 1
        beta_t = (1 - t) * (beta_ - 1) * (1 - anneal) + 1
        kl = _kl_beta(alpha_t, beta_t)

        return (loss + self.kl_lambda * anneal * kl).mean()

    def _loss_multiclass(
        self,
        info: Dict[str, torch.Tensor],
        target: torch.Tensor,
        epoch: int,
    ) -> torch.Tensor:
        """Type-II ML + KL loss for Dirichlet distribution."""
        alpha = info["alpha"]  # (B, K)
        S = alpha.sum(dim=-1, keepdim=True)
        K = alpha.size(-1)

        target_oh = F.one_hot(target.long(), K).float()

        loss = (target_oh * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=-1)

        anneal = min(1.0, epoch / max(self.annealing_epochs, 1))
        alpha_t = target_oh * (alpha - 1) * (1 - anneal) + 1
        kl = _kl_dirichlet(alpha_t)

        return (loss + self.kl_lambda * anneal * kl).mean()

    def _loss_regression(
        self,
        info: Dict[str, torch.Tensor],
        target: torch.Tensor,
        epoch: int,
    ) -> torch.Tensor:
        """NIG negative log-likelihood + regularisation loss."""
        mu = info["mu"]
        v = info["v"]
        alpha = info["alpha"]
        beta_ = info["beta"]
        t = target.float()

        omega = 2 * beta_ * (1 + v)
        nll = (
            0.5 * torch.log(torch.tensor(torch.pi, device=mu.device, dtype=mu.dtype))
            - 0.5 * torch.log(v.clamp(min=1e-8))
            + alpha * torch.log(omega.clamp(min=1e-8))
            - (alpha + 0.5) * torch.log(((t - mu) ** 2) * v + omega)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )

        anneal = min(1.0, epoch / max(self.annealing_epochs, 1))
        reg = (t - mu).abs() * (2 * v + alpha)

        return (nll + self.kl_lambda * anneal * reg).mean()

    # ── Utilities ───────────────────────────────────────────────

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, task_type='{self.task_type}', "
            f"output_dim={self.output_dim}, kl_lambda={self.kl_lambda}, "
            f"annealing_epochs={self.annealing_epochs}"
        )
