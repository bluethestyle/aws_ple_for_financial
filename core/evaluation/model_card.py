"""Auto-generate standardized model documentation in Markdown.

Produces a Model Card following the ML model documentation best practices,
including model details, training configuration, per-task performance metrics,
feature importance (IG attribution), expert analysis (CCA + gate weights),
uncertainty analysis, and known limitations.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ModelCardGenerator:
    """Generate a Markdown model card from training and analysis results."""

    def generate(
        self,
        model_info: dict,
        training_results: dict,
        analysis_results: dict,
        output_path: str,
    ) -> str:
        """Generate Markdown model card and save to disk.

        Args:
            model_info: Model architecture metadata (name, params, etc.).
            training_results: Training metrics from Stage 8.
            analysis_results: Analysis artifacts from Stage 8.5.
            output_path: File path to write the model card.

        Returns:
            The generated Markdown content string.
        """
        sections = [
            self._header(model_info),
            self._model_details(model_info),
            self._training_details(training_results),
            self._performance_metrics(training_results),
            self._feature_importance(analysis_results),
            self._expert_analysis(analysis_results),
            self._uncertainty_analysis(analysis_results),
            self._xai_quality_summary(analysis_results),
            self._multidisciplinary_summary(analysis_results),
            self._limitations(),
            self._version_history(model_info),
        ]

        content = "\n\n".join(s for s in sections if s)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info("Model card saved to %s", output_path)
        return content

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------

    def _header(self, info: dict) -> str:
        name = info.get("name", "PLE Multi-Task Model")
        return (
            f"# Model Card: {name}\n\n"
            f"Generated: {datetime.now().isoformat()}"
        )

    def _model_details(self, info: dict) -> str:
        lines = ["## Model Details", ""]
        lines.append(f"- **Architecture**: {info.get('architecture', 'PLE + adaTT')}")
        lines.append(f"- **Tasks**: {info.get('num_tasks', 'N/A')}")
        lines.append(f"- **Shared Experts**: {info.get('num_experts', 'N/A')}")

        total_params = info.get("total_params")
        if total_params is not None:
            lines.append(f"- **Parameters**: {total_params:,}")
        else:
            lines.append("- **Parameters**: N/A")

        lines.append(f"- **Input Dim**: {info.get('input_dim', 'N/A')}")
        lines.append(f"- **PLE Layers**: {info.get('num_layers', 'N/A')}")

        expert_basket = info.get("expert_basket")
        if expert_basket:
            lines.append(f"- **Expert Basket**: {', '.join(expert_basket)}")

        task_groups = info.get("task_groups")
        if task_groups:
            lines.append(f"- **Task Groups**: {', '.join(task_groups)}")

        return "\n".join(lines)

    def _training_details(self, results: dict) -> str:
        lines = ["## Training Details", ""]
        training = results.get("training", results)

        lines.append(f"- **Epochs**: {training.get('epochs_completed', training.get('epochs', 'N/A'))}")
        lines.append(f"- **Batch Size**: {training.get('batch_size', 'N/A')}")
        lines.append(f"- **Learning Rate**: {training.get('learning_rate', 'N/A')}")
        lines.append(f"- **Loss Weighting**: {training.get('loss_weighting', 'uncertainty')}")

        if "best_val_loss" in training:
            lines.append(f"- **Best Val Loss**: {training['best_val_loss']:.4f}")
        if "training_time_seconds" in training:
            minutes = training["training_time_seconds"] / 60
            lines.append(f"- **Training Time**: {minutes:.1f} minutes")

        early_stop = training.get("early_stop_epoch")
        if early_stop:
            lines.append(f"- **Early Stop Epoch**: {early_stop}")

        return "\n".join(lines)

    def _performance_metrics(self, results: dict) -> str:
        """Create a table of per-task metrics."""
        task_metrics = results.get("task_metrics", results.get("per_task_metrics", {}))
        if not task_metrics:
            return ""

        lines = ["## Performance Metrics", ""]

        # Determine columns from first task
        sample_metrics = next(iter(task_metrics.values()), {})
        metric_cols = sorted(
            k for k in sample_metrics.keys()
            if isinstance(sample_metrics[k], (int, float))
        )

        if not metric_cols:
            return ""

        # Table header
        header = "| Task | " + " | ".join(metric_cols) + " |"
        separator = "|------|" + "|".join("------" for _ in metric_cols) + "|"
        lines.append(header)
        lines.append(separator)

        # Table rows
        for task_name, metrics in sorted(task_metrics.items()):
            values = []
            for col in metric_cols:
                v = metrics.get(col)
                if isinstance(v, float):
                    values.append(f"{v:.4f}")
                elif v is not None:
                    values.append(str(v))
                else:
                    values.append("-")
            lines.append(f"| {task_name} | " + " | ".join(values) + " |")

        return "\n".join(lines)

    def _feature_importance(self, analysis: dict) -> str:
        """List IG Top-10 per task."""
        ig = analysis.get("ig_attributions", {})
        if not ig or "error" in ig:
            return ""

        lines = ["## Feature Importance (Integrated Gradients)", ""]

        for task_name, importance in sorted(ig.items()):
            if isinstance(importance, dict) and "error" in importance:
                continue

            lines.append(f"### {task_name}")
            lines.append("")

            # importance is Dict[feature_name, score] or similar
            if isinstance(importance, dict):
                sorted_feats = sorted(
                    importance.items(),
                    key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0,
                    reverse=True,
                )[:10]
                lines.append("| Rank | Feature | Attribution |")
                lines.append("|------|---------|-------------|")
                for rank, (feat, score) in enumerate(sorted_feats, 1):
                    if isinstance(score, (int, float)):
                        lines.append(f"| {rank} | {feat} | {score:.6f} |")
                    else:
                        lines.append(f"| {rank} | {feat} | {score} |")
            elif isinstance(importance, list):
                lines.append("| Rank | Feature | Attribution |")
                lines.append("|------|---------|-------------|")
                for rank, item in enumerate(importance[:10], 1):
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        lines.append(f"| {rank} | {item[0]} | {item[1]:.6f} |")
                    else:
                        lines.append(f"| {rank} | {item} | - |")

            lines.append("")

        return "\n".join(lines)

    def _expert_analysis(self, analysis: dict) -> str:
        """Show CCA matrix and gate weights."""
        sections: List[str] = []

        # CCA / Expert Redundancy
        cca = analysis.get("expert_redundancy", {})
        if cca and "error" not in cca and cca.get("status") != "skipped":
            lines = ["## Expert Redundancy Analysis (CCA)", ""]

            cca_matrix = cca.get("cca_matrix", cca.get("similarity_matrix", {}))
            if cca_matrix and isinstance(cca_matrix, dict):
                expert_names = sorted(cca_matrix.keys())
                # Build table
                header = "| | " + " | ".join(expert_names) + " |"
                separator = "|---|" + "|".join("---" for _ in expert_names) + "|"
                lines.append(header)
                lines.append(separator)
                for name in expert_names:
                    row_data = cca_matrix.get(name, {})
                    values = []
                    for other in expert_names:
                        v = row_data.get(other, "-")
                        if isinstance(v, float):
                            values.append(f"{v:.3f}")
                        else:
                            values.append(str(v))
                    lines.append(f"| {name} | " + " | ".join(values) + " |")
                lines.append("")

            redundant_pairs = cca.get("redundant_pairs", [])
            if redundant_pairs:
                lines.append("**Redundant pairs** (CCA > 0.9):")
                for pair in redundant_pairs:
                    lines.append(f"- {pair}")

            sections.append("\n".join(lines))

        # Gate Weights
        gate = analysis.get("gate_weights", {})
        if gate and "error" not in gate:
            lines = ["## Gate Weight Analysis", ""]

            per_task = gate.get("per_task", gate)
            if isinstance(per_task, dict):
                for task_name, weights in sorted(per_task.items()):
                    if task_name.startswith("_"):
                        continue
                    lines.append(f"### {task_name}")
                    if isinstance(weights, dict):
                        sorted_experts = sorted(
                            weights.items(),
                            key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0,
                            reverse=True,
                        )
                        for expert, weight in sorted_experts:
                            if isinstance(weight, (int, float)):
                                bar = "#" * int(weight * 20)
                                lines.append(f"- {expert}: {weight:.4f} {bar}")
                            else:
                                lines.append(f"- {expert}: {weight}")
                    elif isinstance(weights, list):
                        for i, w in enumerate(weights):
                            if isinstance(w, (int, float)):
                                lines.append(f"- Expert {i}: {w:.4f}")
                    lines.append("")

            sections.append("\n".join(lines))

        return "\n\n".join(sections) if sections else ""

    def _uncertainty_analysis(self, analysis: dict) -> str:
        """Show evidential uncertainty summary if available."""
        uncertainty = analysis.get("uncertainty", analysis.get("evidential", {}))
        if not uncertainty or "error" in uncertainty:
            return ""

        lines = ["## Uncertainty Analysis (Evidential Deep Learning)", ""]

        for task_name, stats in sorted(uncertainty.items()):
            if not isinstance(stats, dict):
                continue
            lines.append(f"### {task_name}")
            for metric, value in sorted(stats.items()):
                if isinstance(value, float):
                    lines.append(f"- {metric}: {value:.4f}")
                else:
                    lines.append(f"- {metric}: {value}")
            lines.append("")

        return "\n".join(lines) if len(lines) > 2 else ""

    def _xai_quality_summary(self, analysis: dict) -> str:
        """Include XAI quality report if available."""
        xai = analysis.get("xai_quality", {})
        if not xai:
            return ""

        lines = ["## XAI Quality Assessment", ""]
        lines.append(f"- **Fidelity**: {xai.get('fidelity', 'N/A')}")
        lines.append(f"- **Completeness**: {xai.get('completeness', 'N/A')}")
        lines.append(f"- **Consistency**: {xai.get('consistency', 'N/A')}")
        lines.append(f"- **Passed**: {xai.get('passed', 'N/A')}")

        missing = xai.get("missing_items", [])
        if missing:
            lines.append(f"- **Missing Items**: {', '.join(missing)}")

        return "\n".join(lines)

    def _multidisciplinary_summary(self, analysis: dict) -> str:
        """Include multidisciplinary feature interpretation summary."""
        multi = analysis.get("multidisciplinary_interpretation", {})
        if not multi:
            return ""

        lines = ["## Multidisciplinary Feature Interpretation", ""]

        # Aggregate statistics
        agg = multi.get("aggregate_statistics", {})
        if agg:
            lines.append("| Domain | Mean Score | Std |")
            lines.append("|--------|-----------|-----|")
            for domain, stats in agg.items():
                if isinstance(stats, dict):
                    lines.append(
                        f"| {domain} | "
                        f"{stats.get('mean_score', 0):.4f} | "
                        f"{stats.get('std_score', 0):.4f} |",
                    )
            lines.append("")

        # Domain descriptions
        domains = multi.get("domains", {})
        if domains:
            for domain, info in domains.items():
                if isinstance(info, dict):
                    label = info.get("label", "")
                    score = info.get("score", 0)
                    impact = info.get("business_impact", "")
                    lines.append(f"- **{domain}**: {label} ({score:.2f}) — {impact}")

        return "\n".join(lines) if len(lines) > 2 else ""

    def _version_history(self, model_info: dict) -> str:
        """Generate version history section."""
        lines = ["## Version History", ""]

        versions = model_info.get("version_history", [])
        if versions:
            lines.append("| Version | Date | Changes |")
            lines.append("|---------|------|---------|")
            for v in versions:
                if isinstance(v, dict):
                    lines.append(
                        f"| {v.get('version', '')} | {v.get('date', '')} | "
                        f"{v.get('changes', '')} |",
                    )
        else:
            version = model_info.get("version", "1.0.0")
            date = datetime.now().strftime("%Y-%m-%d")
            lines.append(f"| {version} | {date} | Initial model card generation |")

        return "\n".join(lines)

    def _limitations(self) -> str:
        return "\n".join([
            "## Limitations and Ethical Considerations",
            "",
            "- **Data Scope**: Trained on synthetic/sampled financial transaction data; "
            "real-world distributions may differ.",
            "- **Temporal Validity**: Model performance may degrade over time as customer "
            "behavior patterns shift (concept drift).",
            "- **Fairness**: No explicit fairness constraints were applied during training. "
            "Demographic parity and equalized odds should be evaluated before deployment.",
            "- **Regulatory**: Explanations are generated post-hoc via Integrated Gradients "
            "and template matching. They approximate but do not guarantee compliance with "
            "Korean Financial Consumer Protection Act Article 19.",
            "- **Cold Start**: Users with fewer than 3 months of transaction history may "
            "receive lower-quality recommendations.",
            "- **Multi-Task Trade-offs**: Shared expert architecture may compromise "
            "individual task performance in favor of overall multi-task balance.",
        ])


__all__ = ["ModelCardGenerator"]
