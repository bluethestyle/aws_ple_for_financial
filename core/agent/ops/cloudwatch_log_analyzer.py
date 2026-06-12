"""CloudWatchLogAnalyzer — 파이프라인 로그를 증분 윈도우로 분석·트리아지.

온프렘 ops/log_analyzer.py (b67db29a~939960a0) 의 AWS 등가물:
- 파일 glob → CloudWatch Logs ``filter_log_events`` (SageMaker, Lambda 로그그룹)
- 로컬 마커 파일 → S3 객체 또는 로컬 파일 (marker_uri 로 선택)
- GPU 가드 → 제외 (AWS 에는 로컬 GPU 경합 없음)

흐름: [지난 스캔 ~ 지금] 윈도우의 WARNING/ERROR/Traceback 추출(dedupe, cap)
→ ERROR 는 자동 FIX_NOW + investigate(근본원인 조사), WARNING 만 LLM triage
→ 분석 완료 시에만 마커 갱신 (다음 실행이 이어받는 증분).

agent_builder 미주입 시 룰 폴백 (ERROR→FIX_NOW, WARNING→MONITOR) — LLM 비용 0.
"""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = ["CloudWatchLogAnalyzer"]

# WARNING/ERROR/CRITICAL/Traceback 라인 분류용
_LEVEL_RE = re.compile(
    r"\b(WARNING|WARN|ERROR|CRITICAL|FATAL|Traceback)\b", re.IGNORECASE,
)
# CloudWatch Logs filter pattern — 어느 한 단어라도 매치 (?term OR 문법)
_FILTER_PATTERN = "?WARNING ?WARN ?ERROR ?CRITICAL ?FATAL ?Traceback"

_SEVERITY_FACTORS = (
    "규제/컴플라이언스(EU AI Act, SR 11-7, 금소법 §17), 고객 직접노출(추천, 추천사유), "
    "데이터 무결성, PII, 피처 누수, 모델 성능저하, 드리프트, 증가/누적 추세"
)


class CloudWatchLogAnalyzer:
    """CloudWatch Logs 증분 수집 → 트리아지 → 운영자 보고 블록.

    Args:
        log_groups: 스캔할 CloudWatch 로그그룹 이름 목록 (config-driven).
        marker_uri: 증분 마커 저장 위치. ``s3://bucket/key`` 또는 로컬 경로.
        region: AWS region (None 이면 boto3 가 env 에서 해석).
        agent_builder: 호출 시 BedrockDialogSession 류 객체를 반환하는 callable.
            triage(finding, verify=)/investigate(finding, verify=) 를 제공해야
            한다. None 이면 룰 폴백 (LLM 호출 0).
        default_window_hours: 마커가 없을 때의 기본 윈도우.
        max_per_group: 로그그룹당 추출 항목 cap.
        max_total: 전체 추출 항목 cap.
        logs_client / s3_client: 테스트 주입용 (None 이면 boto3 lazy init).
    """

    def __init__(
        self,
        log_groups: List[str],
        marker_uri: str,
        region: Optional[str] = None,
        agent_builder: Optional[Callable[[], Any]] = None,
        default_window_hours: int = 24,
        max_per_group: int = 100,
        max_total: int = 500,
        logs_client: Any = None,
        s3_client: Any = None,
    ) -> None:
        self._log_groups = [g for g in log_groups if g]
        self._marker_uri = marker_uri
        self._region = region
        self._agent_builder = agent_builder
        self._default_window = default_window_hours
        self._max_per_group = int(
            os.getenv("OPS_LOG_MAX_PER_GROUP", str(max_per_group))
        )
        self._max_total = int(os.getenv("OPS_LOG_MAX_TOTAL", str(max_total)))
        self._logs = logs_client
        self._s3 = s3_client

    # ------------------------------------------------------------------
    # AWS clients (lazy)
    # ------------------------------------------------------------------

    def _logs_client(self):
        if self._logs is None:
            import boto3
            self._logs = boto3.client("logs", region_name=self._region)
        return self._logs

    def _s3_client(self):
        if self._s3 is None:
            import boto3
            self._s3 = boto3.client("s3", region_name=self._region)
        return self._s3

    # ------------------------------------------------------------------
    # 증분 마커 (S3 또는 로컬 파일)
    # ------------------------------------------------------------------

    def _window_start(self) -> datetime:
        """마지막 스캔 시각 (마커). 없으면 default_window_hours 전."""
        try:
            ts = self._read_marker()
            if ts:
                return datetime.fromisoformat(ts)
        except Exception as e:
            logger.debug("마커 읽기 실패(무시): %s", e)
        return datetime.now(timezone.utc) - timedelta(hours=self._default_window)

    def _read_marker(self) -> str:
        if not self._marker_uri:
            return ""
        if self._marker_uri.startswith("s3://"):
            bucket, key = self._marker_uri[5:].split("/", 1)
            try:
                obj = self._s3_client().get_object(Bucket=bucket, Key=key)
                return obj["Body"].read().decode("utf-8").strip()
            except Exception:
                return ""  # 첫 실행 (객체 없음)
        path = Path(self._marker_uri)
        return path.read_text(encoding="utf-8").strip() if path.exists() else ""

    def _update_marker(self, until: datetime) -> None:
        try:
            if not self._marker_uri:
                return
            if self._marker_uri.startswith("s3://"):
                bucket, key = self._marker_uri[5:].split("/", 1)
                self._s3_client().put_object(
                    Bucket=bucket, Key=key,
                    Body=until.isoformat().encode("utf-8"),
                )
                return
            path = Path(self._marker_uri)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(until.isoformat(), encoding="utf-8")
        except Exception as e:
            logger.warning("마커 갱신 실패: %s", e)

    # ------------------------------------------------------------------
    # 수집
    # ------------------------------------------------------------------

    def collect(
        self, since: datetime, until: datetime,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """윈도우 내 로그그룹에서 WARN/ERROR 이벤트 추출.

        Returns:
            (항목 목록, 스캔한 로그그룹 수). 항목: {source, level, text}.
        """
        items: List[Dict[str, Any]] = []
        n_groups = 0
        seen: set = set()
        start_ms = int(since.timestamp() * 1000)
        end_ms = int(until.timestamp() * 1000)
        client = self._logs_client()

        for group in self._log_groups:
            if len(items) >= self._max_total:
                break
            per = 0
            kwargs: Dict[str, Any] = {
                "logGroupName": group,
                "startTime": start_ms,
                "endTime": end_ms,
                "filterPattern": _FILTER_PATTERN,
            }
            try:
                while True:
                    resp = client.filter_log_events(**kwargs)
                    for ev in resp.get("events", []):
                        text = (ev.get("message") or "").strip()
                        m = _LEVEL_RE.search(text)
                        if not m:
                            continue
                        key = text[:160]
                        if key in seen:  # 동일 라인 dedupe
                            continue
                        seen.add(key)
                        items.append({
                            "source": group,
                            "level": m.group(1).upper(),
                            "text": text[:300],
                        })
                        per += 1
                        if per >= self._max_per_group or len(items) >= self._max_total:
                            break
                    token = resp.get("nextToken")
                    if (not token or per >= self._max_per_group
                            or len(items) >= self._max_total):
                        break
                    kwargs["nextToken"] = token
                n_groups += 1
            except Exception as e:
                # 로그그룹 미존재(ResourceNotFound) 등 — 스캔 자체는 계속
                logger.debug("로그그룹 스캔 실패 %s: %s", group, e)
        return items, n_groups

    # ------------------------------------------------------------------
    # 분석
    # ------------------------------------------------------------------

    def analyze(self) -> Dict[str, Any]:
        """증분 윈도우 분석 → {window, n_*, summary, items, operator_instructions}.

        분석 완료 시에만 마커 갱신 (실패 시 다음 실행이 같은 윈도우 재분석).
        """
        since = self._window_start()
        until = datetime.now(timezone.utc)
        items, n_groups = self.collect(since, until)

        warnings = [i for i in items if i["level"] in ("WARNING", "WARN")]
        errors = [i for i in items if i["level"] not in ("WARNING", "WARN")]
        result: Dict[str, Any] = {
            "window": {"since": since.isoformat(), "until": until.isoformat()},
            "n_log_groups": n_groups,
            "n_warnings": len(warnings),
            "n_errors": len(errors),
        }
        if not items:
            result["summary"] = "윈도우 내 WARNING/ERROR 로그 없음 (정상)"
            result["items"] = []
            result["operator_instructions"] = []
            self._update_marker(until)
            return result

        triaged = self._triage_items(errors, warnings)
        n_fix = sum(1 for t in triaged if t.get("level") == "FIX_NOW")
        result["items"] = triaged
        result["n_fix_now"] = n_fix
        result["summary"] = (
            f"에러 {len(errors)}건(자동 FIX_NOW), 경고 {len(warnings)}건 분류 "
            f"— FIX_NOW 총 {n_fix}건"
        )
        result["operator_instructions"] = (
            ["FIX_NOW 항목 우선 확인"] if n_fix else ["특이사항 없음 — 관찰 유지"]
        )
        self._update_marker(until)
        return result

    def _triage_items(
        self,
        errors: List[Dict[str, Any]],
        warnings: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """ERROR → 자동 FIX_NOW (+investigate), WARNING → LLM triage (룰 폴백)."""
        triaged: List[Dict[str, Any]] = []
        hcheck = os.getenv(
            "REPORTS_LLM_VERIFY_GROUNDING", "",
        ).strip().lower() not in ("0", "false", "no", "off")

        # ERROR: 본질적으로 실패 → 자동 FIX_NOW. agent 가 있으면 근본원인 조사.
        verify_cap = int(os.getenv("OPS_LOG_VERIFY_MAX", "5"))
        for idx, it in enumerate(errors):
            entry: Dict[str, Any] = {
                "level": "FIX_NOW", "source": it["source"],
                "issue": it["text"][:200], "auto": "error",
            }
            if self._agent_builder is not None and idx < verify_cap:
                entry["verification"] = self._investigate(it, hcheck)
            triaged.append(entry)

        # WARNING: 심각도가 안 드러나므로 agent triage. cap 초과분은 룰(무음 잘림 방지).
        triage_cap = int(os.getenv("OPS_LOG_TRIAGE_MAX", "10"))
        for idx, it in enumerate(warnings):
            if self._agent_builder is None:
                triaged.append({
                    "level": "MONITOR", "source": it["source"],
                    "issue": it["text"][:200],
                })
                continue
            if idx >= triage_cap:
                triaged.append({
                    "level": "MONITOR", "source": it["source"],
                    "issue": it["text"][:200], "overflow": True,
                })
                continue
            triaged.append(self._triage_warning(it, hcheck))
        n_overflow = sum(1 for t in triaged if t.get("overflow"))
        if n_overflow:
            logger.warning(
                "로그 triage: %d건 cap(%d) 초과 → 룰 분류(MONITOR)",
                n_overflow, triage_cap,
            )
        return triaged

    def _investigate(self, item: Dict[str, Any], hcheck: bool) -> Dict[str, Any]:
        try:
            agent = self._agent_builder()
            finding = (
                f"ERROR 로그: {item['source']} — {item['text']}\n"
                "도구로 근본원인과 실제 영향을 직접 확인하세요. "
                "(환경성/일시적이면 명시)"
            )
            r = agent.investigate(finding, verify=hcheck)
            return {
                "reasoning": str(r.get("reasoning", ""))[:600],
                "n_tool_calls": r.get("n_tool_calls", 0),
                "tools": [t.get("name") for t in r.get("tool_calls", [])],
                "grounding_check": r.get("grounding_check"),
            }
        except Exception as e:
            logger.warning("ERROR 조사 실패(무시): %s", e)
            return {"error": str(e)}

    def _triage_warning(self, item: Dict[str, Any], hcheck: bool) -> Dict[str, Any]:
        try:
            agent = self._agent_builder()
            finding = (
                f"WARNING 로그: {item['source']} — {item['text']}\n"
                f"심각도 상향 요소: {_SEVERITY_FACTORS}.\n"
                "도구로 실제 영향을 직접 확인한 뒤 IGNORE/MONITOR/FIX_NOW 분류하세요."
            )
            r = agent.triage(finding, verify=hcheck)
            return {
                "level": self._normalize_level(r.get("triage")),
                "source": item["source"],
                "issue": item["text"][:200],
                "verification": {
                    "reasoning": str(r.get("reasoning", ""))[:600],
                    "n_tool_calls": r.get("n_tool_calls", 0),
                    "tools": [t.get("name") for t in r.get("tool_calls", [])],
                    "grounding_check": r.get("grounding_check"),
                },
            }
        except Exception as e:
            logger.warning("WARNING triage 실패(룰 폴백): %s", e)
            return {
                "level": "MONITOR", "source": item["source"],
                "issue": item["text"][:200],
            }

    @staticmethod
    def _normalize_level(level: Optional[str]) -> str:
        """LLM 이 준 level 을 유효 enum 으로 정규화."""
        s = str(level or "").upper()
        if "FIX_NOW" in s:
            return "FIX_NOW"
        if "IGNORE" in s:
            return "IGNORE"
        return "MONITOR"  # 불명확시 보수적으로 관찰
