"""PORT-06 tests — CloudWatchLogAnalyzer (증분 마커, 수집, 트리아지) +
query_cloudwatch_logs 도구 래퍼 + 옵트인 게이트.

CloudWatch / S3 호출은 전부 fake 클라이언트로 대체 (AWS 호출 없음).

Run: pytest tests/test_cloudwatch_log_analyzer.py -v
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from core.agent.ops.cloudwatch_log_analyzer import CloudWatchLogAnalyzer
from core.agent.pipeline_reports import PipelineJobContext, _log_analysis_enabled


class _FakeLogs:
    """group → 순서대로 소비되는 filter_log_events 응답 목록."""

    def __init__(self, pages_by_group):
        self._pages = {g: list(p) for g, p in pages_by_group.items()}
        self.calls = []

    def filter_log_events(self, **kwargs):
        group = kwargs["logGroupName"]
        self.calls.append(kwargs)
        if group not in self._pages:
            raise RuntimeError("ResourceNotFoundException")
        pages = self._pages[group]
        return pages.pop(0) if pages else {"events": []}


def _ev(msg):
    return {"message": msg, "timestamp": 1, "logStreamName": "s1"}


def _analyzer(tmp_path, pages_by_group, **kw):
    return CloudWatchLogAnalyzer(
        log_groups=list(pages_by_group.keys()),
        marker_uri=str(tmp_path / "marker.txt"),
        logs_client=_FakeLogs(pages_by_group),
        **kw,
    )


class TestMarker:
    def test_default_window_without_marker(self, tmp_path):
        a = _analyzer(tmp_path, {}, default_window_hours=48)
        start = a._window_start()
        expect = datetime.now(timezone.utc) - timedelta(hours=48)
        assert abs((start - expect).total_seconds()) < 60

    def test_marker_roundtrip_local(self, tmp_path):
        a = _analyzer(tmp_path, {})
        ts = datetime(2026, 6, 12, 9, 0, tzinfo=timezone.utc)
        a._update_marker(ts)
        assert a._window_start() == ts

    def test_marker_s3(self, tmp_path):
        store = {}

        class _FakeS3:
            def put_object(self, Bucket, Key, Body):
                store[(Bucket, Key)] = Body

            def get_object(self, Bucket, Key):
                import io
                if (Bucket, Key) not in store:
                    raise RuntimeError("NoSuchKey")
                return {"Body": io.BytesIO(store[(Bucket, Key)])}

        a = CloudWatchLogAnalyzer(
            log_groups=[], marker_uri="s3://bkt/markers/log_scan.txt",
            logs_client=_FakeLogs({}), s3_client=_FakeS3(),
        )
        ts = datetime(2026, 6, 12, 9, 0, tzinfo=timezone.utc)
        a._update_marker(ts)
        assert a._window_start() == ts


class TestCollect:
    def _window(self):
        until = datetime.now(timezone.utc)
        return until - timedelta(hours=1), until

    def test_extract_dedupe_and_classify(self, tmp_path):
        a = _analyzer(tmp_path, {"/aws/lambda/p": [{"events": [
            _ev("2026-06-12 ERROR boom"),
            _ev("2026-06-12 ERROR boom"),          # 동일 라인 dedupe
            _ev("2026-06-12 WARNING psi rising"),
            _ev("plain info line"),                  # 레벨 없음 → 제외
        ]}]})
        items, n = a.collect(*self._window())
        assert n == 1
        assert [(i["level"], i["source"]) for i in items] == [
            ("ERROR", "/aws/lambda/p"), ("WARNING", "/aws/lambda/p"),
        ]

    def test_pagination_and_per_group_cap(self, tmp_path):
        pages = [
            {"events": [_ev(f"ERROR e{i}") for i in range(2)], "nextToken": "t1"},
            {"events": [_ev(f"ERROR e{i}") for i in range(2, 4)]},
        ]
        a = _analyzer(tmp_path, {"g": pages}, max_per_group=3)
        items, _ = a.collect(*self._window())
        assert len(items) == 3  # cap 에서 중단

    def test_missing_group_skipped(self, tmp_path):
        a = CloudWatchLogAnalyzer(
            log_groups=["missing", "ok"],
            marker_uri=str(tmp_path / "m.txt"),
            logs_client=_FakeLogs({"ok": [{"events": [_ev("ERROR x")]}]}),
        )
        items, n_groups = a.collect(*self._window())
        assert len(items) == 1
        assert n_groups == 1  # missing 은 스캔 수에 미포함, 전체는 계속


class _FakeAgent:
    def __init__(self, triage_level="FIX_NOW"):
        self._level = triage_level

    def triage(self, finding, verify=False):
        out = {"triage": self._level, "reasoning": "근거", "n_tool_calls": 1,
               "tool_calls": [{"name": "detect_drift"}]}
        if verify:
            out["grounding_check"] = {"checked": True, "grounded": True}
        return out

    def investigate(self, finding, verify=False):
        out = {"reasoning": "근본원인", "n_tool_calls": 2,
               "tool_calls": [{"name": "query_cloudwatch_logs"}]}
        if verify:
            out["grounding_check"] = {"checked": True, "grounded": True}
        return out


class TestAnalyze:
    def test_rule_fallback_without_agent(self, tmp_path):
        a = _analyzer(tmp_path, {"g": [{"events": [
            _ev("ERROR boom"), _ev("WARNING psi"),
        ]}]})
        out = a.analyze()
        assert out["n_errors"] == 1 and out["n_warnings"] == 1
        levels = {i["issue"][:5]: i["level"] for i in out["items"]}
        assert levels["ERROR"] == "FIX_NOW"
        assert levels["WARNI"] == "MONITOR"
        # ERROR 항목은 auto 표시, agent 없으므로 verification 없음
        err = next(i for i in out["items"] if i["level"] == "FIX_NOW")
        assert err["auto"] == "error"
        assert "verification" not in err
        # 분석 완료 → 마커 갱신
        assert (tmp_path / "marker.txt").exists()

    def test_empty_window_is_normal(self, tmp_path):
        a = _analyzer(tmp_path, {"g": [{"events": []}]})
        out = a.analyze()
        assert "없음" in out["summary"]
        assert out["items"] == []
        assert (tmp_path / "marker.txt").exists()

    def test_agent_triage_and_investigate(self, tmp_path):
        a = _analyzer(
            tmp_path,
            {"g": [{"events": [_ev("ERROR boom"), _ev("WARNING psi up")]}]},
            agent_builder=lambda: _FakeAgent(triage_level="FIX_NOW"),
        )
        out = a.analyze()
        err = next(i for i in out["items"] if i.get("auto") == "error")
        assert err["verification"]["n_tool_calls"] == 2
        warn = next(i for i in out["items"] if i.get("auto") is None)
        assert warn["level"] == "FIX_NOW"  # agent 분류 반영
        assert warn["verification"]["tools"] == ["detect_drift"]
        assert out["n_fix_now"] == 2

    def test_triage_cap_overflow_marked(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPS_LOG_TRIAGE_MAX", "1")
        a = _analyzer(
            tmp_path,
            {"g": [{"events": [_ev("WARNING w1"), _ev("WARNING w2")]}]},
            agent_builder=lambda: _FakeAgent(triage_level="IGNORE"),
        )
        out = a.analyze()
        assert [i["level"] for i in out["items"]] == ["IGNORE", "MONITOR"]
        assert out["items"][1]["overflow"] is True

    def test_agent_failure_falls_back_to_rule(self, tmp_path):
        def _boom():
            raise RuntimeError("bedrock down")

        a = _analyzer(
            tmp_path,
            {"g": [{"events": [_ev("WARNING w1")]}]},
            agent_builder=_boom,
        )
        out = a.analyze()
        assert out["items"][0]["level"] == "MONITOR"  # 실패 swallow → 룰

    def test_normalize_level(self):
        norm = CloudWatchLogAnalyzer._normalize_level
        assert norm("[분류] FIX_NOW") == "FIX_NOW"
        assert norm("ignore") == "IGNORE"
        assert norm("monitor") == "MONITOR"
        assert norm(None) == "MONITOR"
        assert norm("뭔지모름") == "MONITOR"


class TestContextLogGroups:
    def test_derived_from_lambda_names(self):
        ctx = PipelineJobContext(predict_lambda="my-predict", l2a_lambda="my-l2a")
        groups = ctx.resolved_log_groups()
        assert "/aws/lambda/my-predict" in groups
        assert "/aws/lambda/my-l2a" in groups
        assert "/aws/sagemaker/TrainingJobs" in groups

    def test_explicit_override_wins(self):
        ctx = PipelineJobContext(log_groups=["/custom/group"])
        assert ctx.resolved_log_groups() == ["/custom/group"]


class TestQueryCloudWatchLogsTool:
    def test_requires_log_group(self):
        from core.agent.tool_wrappers import query_cloudwatch_logs
        assert "error" in query_cloudwatch_logs()

    def test_returns_events_with_cap(self, monkeypatch):
        from core.agent import tool_wrappers

        class _FakeClient:
            def filter_log_events(self, **kwargs):
                return {"events": [
                    {"timestamp": i, "logStreamName": "s",
                     "message": f"ERROR e{i}"}
                    for i in range(10)
                ]}

        import boto3
        monkeypatch.setattr(boto3, "client", lambda *a, **k: _FakeClient())
        out = tool_wrappers.query_cloudwatch_logs(
            log_group="/aws/lambda/x", max_events=3,
        )
        assert out["n_events"] == 3
        assert out["events"][0]["message"] == "ERROR e0"


class TestOptInGate:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv("REPORTS_LOG_ANALYSIS_ENABLED", raising=False)
        assert _log_analysis_enabled() is False

    def test_enabled(self, monkeypatch):
        monkeypatch.setenv("REPORTS_LOG_ANALYSIS_ENABLED", "1")
        assert _log_analysis_enabled() is True
