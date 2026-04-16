"""
Ops Diagnoser — Cross-Checkpoint Correlation Analysis
=======================================================

Analyzes correlations between checkpoint measurements to identify
root causes rather than just symptoms.

Rule patterns:
    - CP2 zero-variance + CP3 AUC drop → feature quality degradation
    - CP2 drift critical 3 days + CP7 CTR drop → retraining needed
    - CP1 row count drop + CP2 NaN increase → upstream data source failure
    - CP4 fidelity gap > 5% + CP6 metric drop → distillation quality issue
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.agent.ops.collector import CheckpointResult

logger = logging.getLogger(__name__)

__all__ = ["OpsDiagnoser", "Diagnosis"]


@dataclass
class Diagnosis:
    """A single cross-checkpoint diagnosis."""
    rule_id: str
    severity: str          # WARNING / CRITICAL
    finding: str           # what was observed
    likely_cause: str      # probable root cause
    suggested_action: str  # recommended next step
    checkpoints: List[str] = field(default_factory=list)  # involved CPs
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity,
            "finding": self.finding,
            "likely_cause": self.likely_cause,
            "suggested_action": self.suggested_action,
            "checkpoints": self.checkpoints,
            "evidence": self.evidence,
        }


class OpsDiagnoser:
    """Cross-checkpoint correlation diagnosis engine.

    Args:
        config: Diagnosis rule thresholds.
        history: Previous checkpoint results for trend analysis.
        case_store: Optional DiagnosticCaseStore for similar past case lookup.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        case_store: Optional[Any] = None,
    ) -> None:
        self._config = config or {}
        self._history = history or []
        self._case_store = case_store  # DiagnosticCaseStore | None

    def _get_case_store(self) -> Optional[Any]:
        """Lazy-init DiagnosticCaseStore if not provided at construction time."""
        if self._case_store is None:
            store_path = self._config.get("case_store_path", "")
            if store_path:
                try:
                    from core.agent.case_store import DiagnosticCaseStore
                    self._case_store = DiagnosticCaseStore(store_path=store_path)
                except Exception as e:
                    logger.warning("DiagnosticCaseStore init failed: %s", e)
                    self._case_store = False  # sentinel: skip future attempts
        obj = self._case_store
        return obj if obj is not False else None

    def diagnose(self, checkpoints: List["CheckpointResult"]) -> List[Diagnosis]:
        """Run all correlation rules against checkpoint results.

        Before applying correlation rules, searches DiagnosticCaseStore for
        similar past cases so rule thresholds can be informed by history.

        Args:
            checkpoints: List of CheckpointResult from OpsCollector.

        Returns:
            List of Diagnosis objects, sorted by severity.
        """
        cp_map = {cp.checkpoint: cp for cp in checkpoints}
        diagnoses = []

        # Search similar past cases from DiagnosticCaseStore (non-fatal)
        try:
            cs = self._get_case_store()
            if cs is not None and cs.case_count > 0:
                # Build a simple numeric query vector from checkpoint statuses
                import numpy as _np
                status_map = {"GREEN": 0.0, "YELLOW": 0.5, "RED": 1.0}
                q_vec = _np.array(
                    [status_map.get(cp.status, 0.0) for cp in checkpoints],
                    dtype=_np.float32,
                )
                # Pad/truncate to embedding_dim
                dim = cs._embedding_dim
                if len(q_vec) < dim:
                    q_vec = _np.pad(q_vec, (0, dim - len(q_vec)))
                else:
                    q_vec = q_vec[:dim]

                similar = cs.search_similar(q_vec, k=3)
                if similar:
                    logger.info(
                        "OpsDiagnoser: found %d similar past cases (top score=%.3f)",
                        len(similar), similar[0][1],
                    )
                    for past_case, score in similar:
                        logger.debug(
                            "  similar case=%s (%.3f): %s",
                            past_case.get("case_id", "?"), score,
                            past_case.get("finding", "")[:80],
                        )
        except Exception:
            logger.debug("DiagnosticCaseStore search failed (non-fatal)", exc_info=True)

        # Run each correlation rule
        rules = [
            self._rule_feature_quality_degradation,
            self._rule_upstream_data_failure,
            self._rule_distillation_quality,
            self._rule_serving_latency_trend,
            self._rule_training_instability,
        ]

        for rule in rules:
            try:
                result = rule(cp_map)
                if result:
                    diagnoses.append(result)
            except Exception as e:
                logger.warning("Diagnosis rule %s failed: %s", rule.__name__, e)

        # Sort: CRITICAL first, then WARNING
        diagnoses.sort(key=lambda d: 0 if d.severity == "CRITICAL" else 1)
        return diagnoses

    def _rule_feature_quality_degradation(self, cp_map: Dict) -> Optional[Diagnosis]:
        """CP2 zero-variance cols + CP3 AUC drop → feature quality issue."""
        cp2 = cp_map.get("CP2")
        cp3 = cp_map.get("CP3")
        if not cp2 or not cp3:
            return None

        zero_var = cp2.measurements.get("zero_variance_cols", 0)
        val_auc = cp3.measurements.get("best_val_auc")

        if zero_var > 0 and val_auc is not None and val_auc < self._config.get("min_val_auc", 0.55):
            return Diagnosis(
                rule_id="CORR-001",
                severity="WARNING",
                finding=f"zero-variance 컬럼 {zero_var}개 + val_auc {val_auc:.4f}",
                likely_cause="피처 품질 저하가 모델 성능에 영향",
                suggested_action="zero-variance 피처 원인 추적 (인제스천 스키마 변경 또는 전처리 오류)",
                checkpoints=["CP2", "CP3"],
                evidence={"zero_variance_cols": zero_var, "val_auc": val_auc},
            )
        return None

    def _rule_upstream_data_failure(self, cp_map: Dict) -> Optional[Diagnosis]:
        """CP1 domain failure + CP2 NaN increase → upstream data source issue."""
        cp1 = cp_map.get("CP1")
        cp2 = cp_map.get("CP2")
        if not cp1 or not cp2:
            return None

        failed_domains = cp1.measurements.get("domains_failed", 0)
        nan_max = cp2.measurements.get("nan_ratio_max", 0.0)

        if failed_domains > 0 and nan_max > 0.2:
            return Diagnosis(
                rule_id="CORR-002",
                severity="CRITICAL",
                finding=f"인제스천 {failed_domains}개 도메인 실패 + NaN 비율 {nan_max:.2%}",
                likely_cause="업스트림 데이터 소스 장애",
                suggested_action="실패 도메인의 원천 데이터 소스 상태 즉시 확인",
                checkpoints=["CP1", "CP2"],
                evidence={"domains_failed": failed_domains, "nan_ratio_max": nan_max},
            )
        return None

    def _rule_distillation_quality(self, cp_map: Dict) -> Optional[Diagnosis]:
        """CP4 fidelity gap > threshold → distillation quality concern."""
        cp4 = cp_map.get("CP4")
        if not cp4:
            return None

        max_gap = cp4.measurements.get("max_gap", 0.0)
        threshold = self._config.get("fidelity_gap_threshold", 0.05)

        if max_gap > threshold:
            return Diagnosis(
                rule_id="CORR-003",
                severity="WARNING",
                finding=f"증류 fidelity gap {max_gap:.4f} > 임계값 {threshold}",
                likely_cause="증류 품질 저하 — 교사 모델 변경 또는 피처 분포 변동",
                suggested_action="해당 태스크의 교사-학생 예측 분포 비교 분석",
                checkpoints=["CP4"],
                evidence={"max_gap": max_gap, "tasks": cp4.measurements.get("tasks_above_threshold", [])},
            )
        return None

    def _rule_serving_latency_trend(self, cp_map: Dict) -> Optional[Diagnosis]:
        """CP6 p95 latency approaching SLA → latency trend warning."""
        cp6 = cp_map.get("CP6")
        if not cp6:
            return None

        p95 = cp6.measurements.get("p95_latency_ms")
        sla = self._config.get("latency_sla_ms", 300)

        if p95 is not None and p95 > sla * 0.8:
            severity = "CRITICAL" if p95 > sla else "WARNING"
            return Diagnosis(
                rule_id="CORR-004",
                severity=severity,
                finding=f"추천 응답 p95 latency {p95:.0f}ms (SLA {sla}ms의 {p95/sla:.0%})",
                likely_cause="feature store 응답시간 증가 또는 사유 생성 병목",
                suggested_action="feature store health_check + context_store 인덱스 상태 확인",
                checkpoints=["CP6"],
                evidence={"p95_latency_ms": p95, "sla_ms": sla},
            )
        return None

    def _rule_training_instability(self, cp_map: Dict) -> Optional[Diagnosis]:
        """CP3 NaN loss or grad explosion → training instability."""
        cp3 = cp_map.get("CP3")
        if not cp3:
            return None

        nan_count = cp3.measurements.get("nan_loss_count", 0)
        grad_max = cp3.measurements.get("grad_norm_max")

        if nan_count > 0:
            return Diagnosis(
                rule_id="CORR-005",
                severity="CRITICAL",
                finding=f"학습 중 NaN loss {nan_count}회 발생",
                likely_cause="학습률 과다 또는 입력 데이터 이상",
                suggested_action="learning rate 확인 + 입력 데이터 NaN/Inf 스캔",
                checkpoints=["CP3"],
                evidence={"nan_loss_count": nan_count},
            )

        if grad_max is not None and grad_max > self._config.get("grad_norm_warning", 100):
            return Diagnosis(
                rule_id="CORR-006",
                severity="WARNING",
                finding=f"gradient norm 최대 {grad_max:.1f}",
                likely_cause="학습 불안정 — gradient clipping 임계값 초과",
                suggested_action="gradient clipping 값 조정 또는 학습률 감소 검토",
                checkpoints=["CP3"],
                evidence={"grad_norm_max": grad_max},
            )
        return None
