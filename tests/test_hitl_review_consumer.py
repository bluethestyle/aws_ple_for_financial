"""PORT-08 tests — HumanReviewQueue 영속화 (load_pending/flush) + consumer 진입점.

predict.py 가 enqueue 하는 생산자 배선은 있었지만 소비 진입점이 0건이던 gap 을
메우는 변경의 회귀 테스트. DynamoDB 는 fake table 로 대체 (AWS 호출 없음).

Run: pytest tests/test_hitl_review_consumer.py -v
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from core.serving.review.human_review_queue import (
    HumanReviewQueue,
    ReviewConfig,
    ReviewState,
)

_CONSUMER = Path(__file__).resolve().parents[1] / "scripts" / "hitl_review_consumer.py"


def _local_cfg(tmp_path, **kw) -> ReviewConfig:
    return ReviewConfig(
        queue_backend="in_memory",
        local_store_path=str(tmp_path / "queue.json"),
        **kw,
    )


class TestLocalStoreRoundTrip:
    def test_flush_then_load_pending(self, tmp_path):
        cfg = _local_cfg(tmp_path)
        q1 = HumanReviewQueue(config=cfg)
        item = q1.enqueue("u1", "rec1", tier=3, payload={"score": 0.9})
        assert item is not None
        assert q1.flush() == 1

        q2 = HumanReviewQueue(config=cfg)
        assert q2.load_pending() == 1
        pending = q2.list_pending(tier=3)
        assert len(pending) == 1
        assert pending[0].review_id == item.review_id
        assert pending[0].payload == {"score": 0.9}

    def test_disposed_items_not_reloaded_as_pending(self, tmp_path):
        cfg = _local_cfg(tmp_path)
        q1 = HumanReviewQueue(config=cfg)
        item = q1.enqueue("u1", "rec1", tier=3)
        q1.flush()

        q2 = HumanReviewQueue(config=cfg)
        q2.load_pending()
        q2.approve(item.review_id, "reviewer1", "확인 완료")
        q2.flush()

        q3 = HumanReviewQueue(config=cfg)
        assert q3.load_pending() == 0
        assert q3.list_pending() == []
        # 영속 파일에는 approved 상태로 남아 있어야 한다 (감사 추적)
        records = json.loads(
            (tmp_path / "queue.json").read_text(encoding="utf-8")
        )
        assert records[0]["state"] == ReviewState.APPROVED
        assert records[0]["reviewer_id"] == "reviewer1"

    def test_load_does_not_clobber_in_process_state(self, tmp_path):
        cfg = _local_cfg(tmp_path)
        q1 = HumanReviewQueue(config=cfg)
        item = q1.enqueue("u1", "rec1", tier=2)
        q1.flush()
        # 같은 프로세스에서 결정 후 load — 디스크의 pending 이 덮어쓰면 안 됨
        q1.approve(item.review_id, "r1", "ok")
        assert q1.load_pending() == 0
        assert q1.get(item.review_id).state == ReviewState.APPROVED

    def test_noop_without_local_store_path(self):
        q = HumanReviewQueue(config=ReviewConfig())  # 순수 in-memory
        q.enqueue("u1", "rec1", tier=3)
        assert q.flush() == 0
        assert q.load_pending() == 0


class _FakeTable:
    def __init__(self, scan_pages=None):
        self.put_calls = []
        self._scan_pages = list(scan_pages or [])

    def put_item(self, Item):
        self.put_calls.append(Item)

    def scan(self, **kwargs):
        if self._scan_pages:
            return self._scan_pages.pop(0)
        return {"Items": []}


class TestDynamoBackend:
    def _queue(self, table) -> HumanReviewQueue:
        q = HumanReviewQueue(
            config=ReviewConfig(queue_backend="dynamodb"),
        )
        q._dynamo = table
        return q

    def test_state_transitions_persist_immediately(self):
        table = _FakeTable()
        q = self._queue(table)
        item = q.enqueue("u1", "rec1", tier=3)
        assert table.put_calls[-1]["review_id"] == item.review_id
        assert table.put_calls[-1]["state"] == ReviewState.PENDING

        q.approve(item.review_id, "r1", "ok")
        assert table.put_calls[-1]["state"] == ReviewState.APPROVED
        # doc 컬럼은 전체 항목 JSON
        doc = json.loads(table.put_calls[-1]["doc"])
        assert doc["disposition_reason"] == "ok"

    def test_load_pending_scans_with_pagination(self):
        def _rec(rid, state):
            return {
                "review_id": rid, "state": state, "tier": 3,
                "doc": json.dumps({
                    "review_id": rid, "user_id": "u", "recommendation_id": "r",
                    "tier": 3, "state": state,
                    "created_at": "2026-06-12T00:00:00+00:00",
                    "payload": {}, "reviewer_id": None,
                    "disposition_at": None, "disposition_reason": None,
                    "modifications": None,
                }),
            }

        pages = [
            {"Items": [_rec("rev_a", "pending")], "LastEvaluatedKey": {"k": 1}},
            {"Items": [_rec("rev_b", "in_review")]},
        ]
        q = self._queue(_FakeTable(scan_pages=pages))
        assert q.load_pending() == 2
        assert len(q.list_pending(tier=3)) == 1  # in_review 는 pending 목록 제외
        assert q.get("rev_b").state == ReviewState.IN_REVIEW

    def test_persist_failure_does_not_block_queue(self):
        class _Boom(_FakeTable):
            def put_item(self, Item):
                raise RuntimeError("dynamodb down")

        q = self._queue(_Boom())
        item = q.enqueue("u1", "rec1", tier=3)  # 예외 swallow
        assert item is not None
        assert q.get(item.review_id).state == ReviewState.PENDING


class TestConsumerScript:
    def _load(self):
        spec = importlib.util.spec_from_file_location(
            "hitl_review_consumer", _CONSUMER,
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_build_queue_from_pipeline_yaml(self):
        mod = self._load()
        q = mod.build_queue()
        assert isinstance(q, HumanReviewQueue)

    def test_audit_callback_chains_to_audit_logger(self, monkeypatch):
        mod = self._load()
        calls = []

        class _FakeAudit:
            def log_operation(self, **kwargs):
                calls.append(kwargs)

        import core.monitoring.audit_logger as al
        monkeypatch.setattr(al, "AuditLogger", lambda: _FakeAudit())
        cb = mod._build_audit_callback()
        assert cb is not None
        cb({"action": "human_review:approved", "review_id": "rev_x",
            "state": "approved", "reviewer_id": "r1"})
        assert calls[0]["operation"] == "human_review:approved"
        assert calls[0]["user"] == "r1"
        assert calls[0]["metadata"]["review_id"] == "rev_x"
