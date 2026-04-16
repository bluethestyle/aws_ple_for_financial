"""Post-distillation steps: summary building, audit log, governance report, ChangeDetector.

Steps 4.7, 6, and the ChangeDetector emission at the end of run_distillation().
Also provides ``build_summary()`` to keep run_distillation() concise.
All monitoring steps are non-fatal — failures are logged as warnings.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("distill-entry")


def build_summary(
    distillation_cfg: dict,
    distill_tasks: List[str],
    hardlabel_tasks: List[str],
    student_config: Any,
    model_out_dir: Path,
    feature_cols: List[str],
    features_len: int,
    fidelity_results: List[Any],
    failed_count: int,
    passed_count: int,
    feature_selections: Dict[str, Any],
    ig_raw_scores: Dict[str, Any],
    calibrated_models: Dict[str, Any],
    saved_calibrators: Dict[str, str],
) -> Dict[str, Any]:
    """Build the distillation_summary dict from all pipeline outputs.

    Args:
        distillation_cfg:  ``distillation`` sub-dict from pipeline YAML.
        distill_tasks:     Tasks routed to soft-label distillation.
        hardlabel_tasks:   Tasks routed to direct hard-label training.
        student_config:    Trained StudentConfig.
        model_out_dir:     Directory where LGBM models were saved.
        feature_cols:      Ordered list of feature column names.
        features_len:      Row count of the feature matrix.
        fidelity_results:  List of FidelityResult objects.
        failed_count:      Number of fidelity failures.
        passed_count:      Number of fidelity passes.
        feature_selections: Dict of {task_name: FeatureSelectionResult}.
        ig_raw_scores:     Dict of {task_name: raw IG importance array}.
        calibrated_models: Calibrated model wrappers (may be empty).
        saved_calibrators: Dict of {task_name: calibrator_path}.

    Returns:
        Summary dict ready to be JSON-serialised.
    """
    _calib_cfg = distillation_cfg.get("calibration", {})
    _calib_method = _calib_cfg.get("method", "platt")
    _fs_cfg_raw = distillation_cfg.get("feature_selection", {})

    return {
        "tasks_distilled": distill_tasks,
        "tasks_direct_hardlabel": hardlabel_tasks,
        "adaptive_strategy": {
            "distill_count": len(distill_tasks),
            "direct_count": len(hardlabel_tasks),
            "threshold_rule": "binary: AUC>0.60, multiclass: F1>2/K, regression: R2>0.05",
        },
        "temperature": student_config.temperature,
        "alpha": student_config.alpha,
        "lgbm_params": student_config.lgbm_params,
        "task_lgbm_overrides": student_config.task_lgbm_overrides,
        "output_dir": str(model_out_dir),
        "feature_count": len(feature_cols),
        "sample_count": int(features_len),
        "fidelity": {
            "all_passed": failed_count == 0,
            "passed": passed_count,
            "failed": failed_count,
            "per_task": {
                r.task_name: {
                    "passed": r.passed,
                    "metrics": {k: round(v, 4) for k, v in r.metrics.items()},
                }
                for r in fidelity_results
            },
        },
        "feature_selection": {
            task_name: {
                "selected": sel.selected_count,
                "original": sel.original_count,
                "reduction_pct": sel.reduction_pct,
                "method": sel.selection_method,
            }
            for task_name, sel in feature_selections.items()
        },
        "ig_dual_scores": ig_raw_scores,
        "feature_selection_config": {
            "method": _fs_cfg_raw.get("method", "lgbm_gain"),
            "ig_alpha": float(_fs_cfg_raw.get("ig_alpha", 0.7)),
            "cumulative_threshold": float(_fs_cfg_raw.get("cumulative_threshold", 0.95)),
            "ig_sample_size": int(_fs_cfg_raw.get("ig_sample_size", 10000)),
        },
        "calibration": {
            "enabled": bool(calibrated_models),
            "method": _calib_method if calibrated_models else None,
            "calibrated_tasks": list(saved_calibrators.keys()),
            "calibrator_paths": saved_calibrators,
        },
    }


def run_post_distillation(
    raw_yaml: dict,
    pipeline_config: Any,
    distill_tasks: List[str],
    hardlabel_tasks: List[str],
    failed_count: int,
    feature_cols: List[str],
    features_len: int,
    config_path: str,
    summary_path: Path,
    model_out_dir: Path,
    out_dir: Path,
    students: Dict[str, Any],
    summary: Dict[str, Any],
) -> None:
    """Emit audit log, governance report, and ChangeDetector stage event.

    All three sub-steps are wrapped in try/except so a monitoring failure never
    aborts the distillation pipeline.

    Args:
        raw_yaml:          Full merged YAML dict (contains ``monitoring`` section).
        pipeline_config:   Parsed pipeline config.
        distill_tasks:     Task names routed to soft-label distillation.
        hardlabel_tasks:   Task names routed to direct hard-label training.
        failed_count:      Number of fidelity failures (0 = all passed).
        feature_cols:      Ordered list of feature column names.
        features_len:      Row count of the feature matrix.
        config_path:       Path to pipeline.yaml.
        summary_path:      Path where distillation_summary.json was written.
        model_out_dir:     Directory containing saved LGBM models.
        out_dir:           Directory containing fidelity/drift reports.
        students:          Dict of {task_name: lgbm_model}.
        summary:           The distillation summary dict (already serialised to disk).
    """
    # --- Step 4.7: Audit log ---
    _audit_cfg = raw_yaml.get("monitoring", {}).get("audit", {})
    if _audit_cfg.get("enabled", False):
        try:
            from core.monitoring.audit_logger import AuditLogger

            _audit_logger = AuditLogger(
                s3_bucket=_audit_cfg.get("s3_bucket", ""),
                s3_prefix=_audit_cfg.get("s3_prefix", "audit_logs"),
            )
            _audit_logger.log_operation(
                operation="distillation:completed",
                user="system",
                status="SUCCESS",
                metadata={
                    "num_tasks": len(pipeline_config.tasks),
                    "distill_tasks": distill_tasks,
                    "direct_tasks": hardlabel_tasks,
                    "fidelity_passed": failed_count == 0,
                    "feature_count": len(feature_cols),
                    "sample_count": int(features_len),
                    "config_path": config_path,
                },
            )
            logger.info("Audit log entry recorded for distillation:completed")
        except Exception as _audit_exc:
            logger.warning("Audit logging failed (non-fatal): %s", _audit_exc)

    # --- Step 6: Governance report ---
    _gov_cfg = raw_yaml.get("monitoring", {}).get("governance_report", {})
    if _gov_cfg.get("enabled", False):
        try:
            from core.monitoring.governance_report import GovernanceReportGenerator

            _gov_gen = GovernanceReportGenerator(
                s3_bucket=_gov_cfg.get("s3_bucket", ""),
                s3_prefix=_gov_cfg.get("s3_prefix", "governance_reports"),
                system_name=_gov_cfg.get("system_name", "PLE-Cluster-adaTT"),
            )

            _drift_for_gov = {
                "summary": {
                    "drift_detected": bool(summary.get("fidelity", {}).get("failed", 0) > 0),
                    "total_features": summary.get("feature_count", 0),
                    "warning_count": 0,
                    "critical_count": summary.get("fidelity", {}).get("failed", 0),
                    "max_psi": 0.0,
                    "avg_psi": 0.0,
                }
            }

            _gov_period = _gov_cfg.get("period", "monthly")
            _gov_report = _gov_gen.generate_report(
                period=_gov_period,
                drift_data=_drift_for_gov,
                model_changes=[
                    {
                        "event": "distillation_cycle",
                        "tasks_distilled": distill_tasks,
                        "tasks_direct": hardlabel_tasks,
                        "fidelity_all_passed": failed_count == 0,
                    }
                ],
            )
            _gov_report_dict = _gov_gen.to_dict(_gov_report)

            _gov_path = out_dir / "governance_report.json"
            with open(_gov_path, "w", encoding="utf-8") as _gf:
                json.dump(_gov_report_dict, _gf, indent=2, default=str)
            logger.info("Governance report saved to %s", _gov_path)

            _gov_s3_uri = _gov_gen.archive_report(_gov_report)
            if _gov_s3_uri:
                logger.info("Governance report archived: %s", _gov_s3_uri)

        except Exception as _gov_exc:
            logger.warning(
                "Governance report generation failed (non-fatal): %s", _gov_exc, exc_info=True
            )

    # --- ChangeDetector: stage_distill event ---
    try:
        from core.agent.change_detector import ChangeDetector as _ChangeDetector

        _cd = _ChangeDetector()
        _cd.on_pipeline_stage_complete(
            stage="stage_distill",
            artifacts={
                "summary_path": str(summary_path),
                "model_output_dir": str(model_out_dir),
                "output_dir": str(out_dir),
                "tasks_distilled": list(students.keys()),
            },
        )
        logger.info("ChangeDetector: stage_distill event emitted")
    except Exception as _e:
        logger.debug("ChangeDetector stage_distill event failed (non-fatal): %s", _e)
