"""
Progressive Layered Extraction (PLE) Model.

Full PLE model with:
  - Stacked CGC layers (shared + task-specific experts with gating)
  - CGC Attention (per-task expert weighting over shared outputs)
  - Adaptive Task Transfer (adaTT) for inter-task knowledge transfer
  - Configurable loss weighting (GradNorm / DWA / Uncertainty)
  - Logit transfer (source task output feeds into target task tower)
  - Task towers with configurable activation and output dimensions

This implementation is **domain-agnostic**: all task names, dimensions,
expert types, and transfer relationships come from ``PLEConfig``.

Architecture overview::

    features ─┬─ SharedExperts ─── CGCAttention(task_k) ──┐
              │                                            ├─ TaskExpert_k ─ Tower_k ─ prediction_k
              └─ TaskExpert_k(per task) ──────────────────┘
                                                    ↑
                                              logit_transfer (source_task → target_task)

References:
  - Tang et al., "Progressive Layered Extraction (PLE): A Novel Multi-Task
    Learning (MTL) Model for Personalized Recommendations" (RecSys 2020)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import PLEConfig, ExpertInputConfig
from .experts import CGCLayer, CGCAttention, MLPExpert, ExpertRegistry
from .feature_router import FeatureRouter
from .adatt import AdaptiveTaskTransfer
from .loss_weighting import (
    BaseLossWeighting,
    UncertaintyWeighting,
    create_loss_weighting,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class PLEInput:
    """Model input container.

    Args:
        features: ``(batch, input_dim)`` feature tensor.
        feature_group_ranges: Optional mapping from feature group name to
            ``(start_col, end_col)`` index range within ``features``.
            When provided together with a configured ``FeatureRouter``,
            enables expert-specific feature routing.  When ``None``, all
            experts receive the full feature tensor (backward compatible).
        expert_routing: Optional runtime override for expert-to-group
            mapping.  ``{expert_name: [group_names]}``.  Typically set by
            the ``FeatureGroupPipeline`` when dynamic routing is needed.
            When ``None``, the static config from ``PLEConfig`` is used.
        cluster_ids: ``(batch,)`` cluster assignment (or zeros if no clustering).
        cluster_probs: ``(batch, n_clusters)`` soft cluster probabilities.
        targets: ``{task_name: label_tensor}`` for training.
    """
    features: torch.Tensor
    feature_group_ranges: Optional[Dict[str, Tuple[int, int]]] = None
    expert_routing: Optional[Dict[str, List[str]]] = None
    cluster_ids: Optional[torch.Tensor] = None
    cluster_probs: Optional[torch.Tensor] = None
    targets: Optional[Dict[str, torch.Tensor]] = None

    @property
    def batch_size(self) -> int:
        return self.features.size(0)

    @property
    def device(self) -> torch.device:
        return self.features.device

    def to(self, device: torch.device) -> "PLEInput":
        def _to(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return t.to(device) if t is not None else None

        return PLEInput(
            features=self.features.to(device),
            feature_group_ranges=self.feature_group_ranges,
            expert_routing=self.expert_routing,
            cluster_ids=_to(self.cluster_ids),
            cluster_probs=_to(self.cluster_probs),
            targets={k: v.to(device) for k, v in self.targets.items()}
                if self.targets else None,
        )


@dataclass
class PLEOutput:
    """Model output container.

    Args:
        predictions: ``{task_name: prediction_tensor}``.
        total_loss: Scalar total loss (training only).
        task_losses: ``{task_name: scalar_loss}`` before transfer enhancement.
        transfer_weights: ``(n_tasks, n_tasks)`` adaTT transfer matrix.
        cgc_attention_weights: ``{task_name: (batch, n_experts)}``.
        aux_losses: Additional regularization losses (e.g. CGC entropy).
    """
    predictions: Dict[str, torch.Tensor]
    total_loss: Optional[torch.Tensor] = None
    task_losses: Optional[Dict[str, torch.Tensor]] = None
    transfer_weights: Optional[torch.Tensor] = None
    cgc_attention_weights: Optional[Dict[str, torch.Tensor]] = None
    aux_losses: Optional[Dict[str, float]] = None


# ============================================================================
# Task Tower
# ============================================================================

class TaskTower(nn.Module):
    """Per-task output tower.

    Maps the task expert output to the final prediction.

    Args:
        input_dim: Width of the task expert output.
        output_dim: Number of output units (1 for binary/regression,
            N for multiclass).
        hidden_dims: Hidden layer widths.
        activation: Output activation (``"sigmoid"``, ``"softmax"``, or ``None``).
        dropout: Dropout probability.
        task_type: ``"binary"``, ``"multiclass"``, or ``"regression"``.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dims: Optional[List[int]] = None,
        activation: Optional[str] = "sigmoid",
        dropout: float = 0.2,
        task_type: str = "binary",
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        if task_type == "regression" and activation not in (None, "none"):
            raise ValueError(
                f"Regression tasks require activation=None, got '{activation}'"
            )

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hd),
                nn.LayerNorm(hd),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hd

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.network(x)
        if self.activation == "sigmoid":
            return torch.sigmoid(out)
        elif self.activation == "softmax":
            return F.softmax(out, dim=-1)
        return out


# ============================================================================
# Main PLE Model
# ============================================================================

class PLEModel(nn.Module):
    """Progressive Layered Extraction model.

    This is the top-level model class that assembles all PLE components.
    Task definitions are fully config-driven -- no hardcoded task names,
    dimensions, or domain knowledge.

    Usage::

        config = PLEConfig(
            input_dim=128,
            task_names=["task_a", "task_b", "task_c"],
            ...
        )
        model = PLEModel(config)

        inputs = PLEInput(features=x, targets={"task_a": y_a, ...})
        output = model(inputs)
        loss = output.total_loss

    Args:
        config: ``PLEConfig`` instance with all hyperparameters.
    """

    def __init__(self, config: PLEConfig):
        super().__init__()
        self.config = config
        self.task_names = list(config.task_names)

        if not self.task_names:
            raise ValueError("PLEConfig.task_names must not be empty")

        # -- Build components ------------------------------------------------
        self._build_extraction_layers()
        self._build_cgc_attention()
        self._build_task_experts()
        self._build_adatt()
        self._build_logit_transfer()
        self._build_task_towers()
        self._build_loss_weighting()

        # -- Training state --------------------------------------------------
        self.current_epoch = 0
        self.global_step = 0
        self._adatt_grad_interval = config.adatt.grad_interval

        logger.info(f"PLEModel init: {len(self.task_names)} tasks, "
                     f"input_dim={config.input_dim}")
        logger.info(self.summary())

    # ------------------------------------------------------------------
    # Build methods
    # ------------------------------------------------------------------

    def _build_extraction_layers(self) -> None:
        """Build stacked CGC extraction layers.

        If ``config.expert_input_routing`` is configured, creates a
        ``FeatureRouter`` and passes it to the first CGC layer so that
        each shared expert receives only its designated feature groups.

        Subsequent stacked layers always receive the full expert output
        dimension (routing only applies to the raw feature input in layer 0).
        """
        cfg = self.config
        self.extraction_layers = nn.ModuleList()

        # Build FeatureRouter for the first layer if routing is configured.
        # Routing only applies to layer 0 (raw features). Deeper layers
        # operate on expert output vectors where group structure no longer
        # exists.
        self.feature_router: Optional[FeatureRouter] = None
        shared_expert_names = [
            f"shared_{i}" for i in range(cfg.num_shared_experts)
        ]

        if cfg.expert_input_routing:
            # We need group_ranges at build time.  If the config provides
            # them statically (common for fixed pipelines), we use them.
            # Otherwise we defer routing to forward() via PLEInput.
            #
            # For build-time routing, we need group_ranges from config or
            # a default that covers the full input_dim.
            # The FeatureRouter will be initialised in forward() if
            # group_ranges are only available at runtime.
            #
            # However, to set expert input_dims at build time we must have
            # group_ranges.  We accept a ``feature_group_ranges`` dict on
            # PLEConfig (set by the pipeline before model construction).
            group_ranges = getattr(cfg, "feature_group_ranges", None)
            if group_ranges:
                self.feature_router = FeatureRouter(
                    expert_names=shared_expert_names,
                    routing_config=cfg.expert_input_routing,
                    group_ranges=group_ranges,
                )
                logger.info(
                    f"FeatureRouter created for {len(cfg.expert_input_routing)} "
                    f"routing rules"
                )

        in_dim = cfg.input_dim
        for layer_idx in range(cfg.num_extraction_layers):
            # Only the first layer gets the feature router
            router = self.feature_router if layer_idx == 0 else None
            expert_names = shared_expert_names if layer_idx == 0 else None

            layer = CGCLayer(
                input_dim=in_dim,
                num_tasks=len(self.task_names),
                num_shared_experts=cfg.num_shared_experts,
                num_task_experts=cfg.num_task_experts_per_task,
                expert_hidden_dim=cfg.shared_expert.output_dim,
                dropout=cfg.dropout,
                expert_hidden_dims=cfg.shared_expert.hidden_dims,
                feature_router=router,
                shared_expert_names=expert_names,
            )
            self.extraction_layers.append(layer)
            in_dim = cfg.shared_expert.output_dim  # next layer input = previous output

        self._extraction_output_dim = in_dim

    def _build_cgc_attention(self) -> None:
        """Build per-task CGC attention over shared expert outputs."""
        cfg = self.config
        if not cfg.cgc.enabled:
            self.cgc_attention = None
            return

        # In stacked PLE, attention operates on the final extraction output
        # Each expert output has the same dimension
        expert_dims = [cfg.shared_expert.output_dim] * cfg.num_shared_experts
        expert_names = [f"shared_{i}" for i in range(cfg.num_shared_experts)]

        domain_map = {
            task_name: cfg.get_domain_experts(task_name)
            for task_name in self.task_names
        }

        self.cgc_attention = CGCAttention(
            task_names=self.task_names,
            expert_dims=expert_dims,
            expert_names=expert_names,
            bias_high=cfg.cgc.bias_high,
            bias_low=cfg.cgc.bias_low,
            dim_normalize=cfg.cgc.dim_normalize,
            domain_experts_map=domain_map,
        )

    def _build_task_experts(self) -> None:
        """Build per-task MLP experts."""
        cfg = self.config
        self.task_expert_networks = nn.ModuleDict()

        for task_name in self.task_names:
            self.task_expert_networks[task_name] = MLPExpert(
                input_dim=self._extraction_output_dim,
                output_dim=cfg.task_expert_output_dim,
                hidden_dims=cfg.task_expert.hidden_dims,
                dropout=cfg.task_expert.dropout,
                activation=cfg.task_expert.activation,
                use_layer_norm=cfg.task_expert.use_layer_norm,
            )

    def _build_adatt(self) -> None:
        """Build the Adaptive Task Transfer module."""
        cfg = self.config.adatt
        if not cfg.enabled:
            self.adatt = None
            return

        # Convert TaskGroupDef objects to plain dicts for AdaTT
        task_groups = None
        if cfg.task_groups:
            task_groups = {
                gname: {
                    "members": gdef.members if hasattr(gdef, "members") else gdef.get("members", []),
                    "intra_strength": gdef.intra_strength if hasattr(gdef, "intra_strength") else gdef.get("intra_strength", 0.7),
                }
                for gname, gdef in cfg.task_groups.items()
            }

        self.adatt = AdaptiveTaskTransfer(
            task_names=self.task_names,
            transfer_lambda=cfg.transfer_lambda,
            temperature=cfg.temperature,
            use_group_prior=bool(task_groups),
            warmup_epochs=cfg.warmup_epochs,
            freeze_epoch=cfg.freeze_epoch,
            negative_transfer_threshold=cfg.negative_transfer_threshold,
            task_groups=task_groups,
            inter_group_strength=cfg.inter_group_strength,
            ema_decay=cfg.ema_decay,
            prior_blend_start=cfg.prior_blend_start,
            prior_blend_end=cfg.prior_blend_end,
            max_transfer_ratio=cfg.max_transfer_ratio,
        )

    def _build_logit_transfer(self) -> None:
        """Build logit transfer projections (source task -> target task)."""
        cfg = self.config
        self.logit_transfer_sources: Dict[str, str] = {}
        self.logit_transfer_proj = nn.ModuleDict()
        self.logit_transfer_strength = cfg.logit_transfer_strength

        for lt in cfg.logit_transfers:
            if not lt.enabled:
                continue
            src, tgt = lt.source, lt.target
            if src in self.task_names and tgt in self.task_names:
                self.logit_transfer_sources[tgt] = src
                src_output_dim = cfg.get_task_output_dim(src)
                self.logit_transfer_proj[tgt] = nn.Sequential(
                    nn.Linear(src_output_dim, cfg.task_expert_output_dim),
                    nn.LayerNorm(cfg.task_expert_output_dim),
                    nn.SiLU(),
                )

        if self.logit_transfer_sources:
            logger.info(f"Logit transfer: {self.logit_transfer_sources} "
                        f"(strength={self.logit_transfer_strength})")

    def _build_task_towers(self) -> None:
        """Build per-task output towers."""
        cfg = self.config
        self.task_towers = nn.ModuleDict()

        for task_name in self.task_names:
            self.task_towers[task_name] = TaskTower(
                input_dim=cfg.task_expert_output_dim,
                output_dim=cfg.get_task_output_dim(task_name),
                hidden_dims=cfg.task_tower.hidden_dims,
                activation=cfg.get_task_activation(task_name),
                dropout=cfg.task_tower.dropout,
                task_type=cfg.get_task_type(task_name),
            )

    def _build_loss_weighting(self) -> None:
        """Build loss weighting strategy."""
        cfg = self.config.loss_weighting
        self.loss_weighting = create_loss_weighting(
            strategy=cfg.strategy,
            num_tasks=len(self.task_names),
            task_names=self.task_names,
            config={
                "alpha": cfg.gradnorm_alpha,
                "grad_interval": cfg.gradnorm_interval,
                "temperature": cfg.dwa_temperature,
                "window_size": cfg.dwa_window_size,
            },
        )

    # ------------------------------------------------------------------
    # Task execution order (topological sort)
    # ------------------------------------------------------------------

    def _get_task_execution_order(self) -> List[str]:
        """Derive task execution order from logit transfer dependencies.

        Uses Kahn's algorithm for topological sort.  Tasks without
        dependencies are sorted alphabetically.
        """
        if not self.logit_transfer_sources:
            return sorted(self.task_names)

        # Build dependency graph: target -> {sources}
        deps: Dict[str, set] = defaultdict(set)
        for target, source in self.logit_transfer_sources.items():
            deps[target].add(source)

        # In-degree
        in_degree = {t: 0 for t in self.task_names}
        for target, sources in deps.items():
            in_degree[target] = len([s for s in sources if s in self.task_names])

        # Kahn's algorithm
        queue = sorted([t for t in self.task_names if in_degree[t] == 0])
        result: List[str] = []

        while queue:
            queue.sort()
            task = queue.pop(0)
            result.append(task)
            for target, sources in deps.items():
                if task in sources and target in in_degree:
                    in_degree[target] -= 1
                    if in_degree[target] == 0 and target not in result:
                        queue.append(target)

        # Add remaining tasks (handles cycles gracefully)
        remaining = sorted(t for t in self.task_names if t not in result)
        result.extend(remaining)
        return result

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        inputs: PLEInput,
        compute_loss: bool = True,
    ) -> PLEOutput:
        """Full forward pass.

        Args:
            inputs: ``PLEInput`` container.
            compute_loss: Whether to compute and return losses.

        Returns:
            ``PLEOutput`` with predictions and optional losses.
        """
        features = inputs.features

        # 1. Stacked CGC extraction layers
        task_representations = [features] * len(self.task_names)
        shared_concat = None
        for layer in self.extraction_layers:
            task_representations, shared_concat = layer(features, task_representations)

        # 2. Per-task expert processing
        task_expert_outputs: Dict[str, torch.Tensor] = {}
        for i, task_name in enumerate(self.task_names):
            task_repr = task_representations[i]
            task_expert_outputs[task_name] = self.task_expert_networks[task_name](task_repr)

        # 3. Task towers with logit transfer (in dependency order)
        execution_order = self._get_task_execution_order()
        predictions: Dict[str, torch.Tensor] = {}

        for task_name in execution_order:
            tower_input = task_expert_outputs[task_name]

            # Logit transfer from source task
            if task_name in self.logit_transfer_sources:
                source = self.logit_transfer_sources[task_name]
                if source in predictions:
                    src_out = predictions[source]
                    if src_out.dim() == 1:
                        src_out = src_out.unsqueeze(-1)
                    proj = self.logit_transfer_proj[task_name](src_out)
                    tower_input = tower_input + self.logit_transfer_strength * proj

            predictions[task_name] = self.task_towers[task_name](tower_input)

        # 4. Loss computation
        total_loss = None
        task_losses = None
        transfer_weights = None
        aux_losses: Dict[str, float] = {}

        if compute_loss and inputs.targets is not None:
            task_losses = self._compute_task_losses(predictions, inputs.targets)

            # Gradient extraction for adaTT (at configured interval)
            task_gradients = None
            if self.training and self.adatt is not None:
                if self.global_step % self._adatt_grad_interval == 0:
                    task_gradients = self._extract_task_gradients(task_losses)

            # Apply adaTT transfer enhancement
            if self.adatt is not None:
                enhanced_losses = self.adatt.compute_transfer_loss(
                    task_losses,
                    task_gradients=task_gradients,
                )
                total_loss = sum(enhanced_losses.values())
                transfer_weights = self.adatt.get_transfer_matrix()
            elif self.loss_weighting is not None:
                # Apply loss weighting strategy
                if isinstance(self.loss_weighting, UncertaintyWeighting):
                    total_loss = self.loss_weighting.weighted_loss(task_losses)
                else:
                    weights = self.loss_weighting.compute_weights()
                    total_loss = sum(
                        weights.get(name, 1.0) * loss
                        for name, loss in task_losses.items()
                    )
            else:
                total_loss = sum(task_losses.values())

            # CGC entropy regularization
            if (self.training
                    and self.cgc_attention is not None
                    and shared_concat is not None
                    and self.config.cgc.entropy_lambda > 0):
                ent_loss = self.cgc_attention.entropy_regularization(shared_concat)
                ent_lambda = self.config.cgc.entropy_lambda
                total_loss = total_loss + ent_lambda * ent_loss
                aux_losses["cgc_entropy"] = (ent_lambda * ent_loss).item()

        # CGC attention weights for monitoring (inference only)
        cgc_weights = None
        if not self.training and self.cgc_attention is not None and shared_concat is not None:
            cgc_weights = self.cgc_attention.get_attention_weights(shared_concat)

        return PLEOutput(
            predictions=predictions,
            total_loss=total_loss,
            task_losses=task_losses,
            transfer_weights=transfer_weights,
            cgc_attention_weights=cgc_weights,
            aux_losses=aux_losses if aux_losses else None,
        )

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------

    def _compute_task_losses(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute per-task losses using task-type-appropriate loss functions.

        Override this method to plug in custom loss functions.
        """
        losses: Dict[str, torch.Tensor] = {}

        for task_name in self.task_names:
            if task_name not in targets:
                continue

            pred = predictions[task_name]
            target = targets[task_name]
            task_type = self.config.get_task_type(task_name)

            if task_type == "binary":
                loss = F.binary_cross_entropy(
                    pred.squeeze(-1), target.float(),
                )
            elif task_type == "multiclass":
                loss = F.cross_entropy(pred, target.long())
            elif task_type == "regression":
                loss = F.huber_loss(pred.squeeze(-1), target.float())
            else:
                # Fallback: MSE
                loss = F.mse_loss(pred.squeeze(-1), target.float())

            losses[task_name] = loss

        return losses

    def _extract_task_gradients(
        self,
        task_losses: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Extract flattened gradients for adaTT affinity computation.

        Uses the shared extraction layer parameters as the reference.
        Gradients are zero-padded for unused parameters so that all
        tasks produce the same-length gradient vector.
        """
        shared_params = list(self.extraction_layers.parameters())
        if not shared_params:
            return {}

        gradients: Dict[str, torch.Tensor] = {}
        for task_name, loss in task_losses.items():
            grads = torch.autograd.grad(
                loss,
                shared_params,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
            # Zero-pad for parameters that received no gradient
            flat_parts = []
            for param, g in zip(shared_params, grads):
                if g is not None:
                    flat_parts.append(g.flatten())
                else:
                    flat_parts.append(torch.zeros(param.numel(), device=param.device))
            gradients[task_name] = torch.cat(flat_parts)

        return gradients

    # ------------------------------------------------------------------
    # Epoch / step lifecycle
    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        """Update epoch state for adaTT and loss weighting."""
        self.current_epoch = epoch
        if self.adatt is not None:
            self.adatt.on_epoch_end(epoch)

    def set_global_step(self, step: int) -> None:
        """Update global step counter (called by Trainer)."""
        self.global_step = step

    def update_loss_weights(
        self,
        losses: Dict[str, torch.Tensor],
        epoch: int = 0,
    ) -> None:
        """Update loss weighting strategy (call after each batch/epoch)."""
        if self.loss_weighting is not None:
            shared_params = list(self.extraction_layers.parameters())
            self.loss_weighting.update(
                losses,
                shared_params=shared_params,
                epoch=epoch,
            )

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Human-readable model summary."""
        lines = [
            "PLEModel:",
            f"  Tasks: {self.task_names}",
            f"  Input dim: {self.config.input_dim}",
            f"  Extraction layers: {self.config.num_extraction_layers}",
            f"  Shared experts: {self.config.num_shared_experts}",
            f"  Task experts per task: {self.config.num_task_experts_per_task}",
            f"  Expert hidden dim: {self.config.shared_expert.output_dim}",
            f"  Task expert output dim: {self.config.task_expert_output_dim}",
            f"  adaTT: {'enabled' if self.adatt is not None else 'disabled'}",
            f"  CGC attention: {'enabled' if self.cgc_attention is not None else 'disabled'}",
            f"  Logit transfers: {self.logit_transfer_sources or 'none'}",
            f"  Loss weighting: {self.config.loss_weighting.strategy}",
        ]
        # Feature routing summary
        if self.feature_router is not None:
            lines.append(f"  Feature routing: enabled ({len(self.config.expert_input_routing)} rules)")
            for name in self.feature_router.expert_names:
                dim = self.feature_router.get_expert_input_dim(name)
                routed = "routed" if self.feature_router.has_routing(name) else "full"
                lines.append(f"    {name}: dim={dim} ({routed})")
        else:
            lines.append("  Feature routing: disabled (all experts receive full input)")

        n_params = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lines.append(f"  Parameters: {n_params:,} total, {n_trainable:,} trainable")
        return "\n".join(lines)

    def get_loss_weights(self) -> Dict[str, float]:
        """Return current loss weights (for logging)."""
        if self.loss_weighting is not None:
            return self.loss_weighting.compute_weights()
        return {name: 1.0 for name in self.task_names}
