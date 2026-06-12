"""HITL 인적 검토 consumer (운영자 CLI) — HumanReviewQueue 미결 건 검토·결정.

predict.py 의 _triage_for_review (tier 2/3 생산자 배선) 이후의 **consumer
진입점 v1** — 온프렘 scripts/hitl_review_consumer.py (8f38dece) 의 AWS 등가.

흐름:
  1) queue_backend 에서 미결(pending/in_review) 건 로드 (load_pending)
     - dynamodb: ple-review-queue scan
     - in_memory + serving.review.local_store_path: 로컬 JSON
  2) approve/reject/modify 결정 기록 (검토자 ID, 사유)
  3) flush 로 결정 상태 영속화 (dynamodb 는 결정 시점에 자동 영속화)

모든 상태 전이는 AuditLogger.log_operation 으로 HMAC 서명 + hash chain 에
기록된다 — 인적 감독(EU AI Act Art. 14, 금소법 §17)의 추적 근거.

결정이 추천/거부권 레지스트리에 자동 되먹임되는 연계(예: 거부 승인 →
opt-out 반영)는 사유별 의미가 달라 후속 설계 (온프렘 v1 과 동일 범위).

사용:
  python scripts/hitl_review_consumer.py --list [--tier 3]
  python scripts/hitl_review_consumer.py --decide <review_id> \
         --action approve|reject|modify --reviewer <id> [--notes "..."] \
         [--modifications '{"product_id": "P123"}']
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

PIPELINE_YAML = REPO_ROOT / "configs" / "pipeline.yaml"


def _load_pipeline_config() -> dict:
    import yaml
    if PIPELINE_YAML.exists():
        return yaml.safe_load(PIPELINE_YAML.read_text(encoding="utf-8")) or {}
    return {}


def _build_audit_callback():
    """HumanReviewQueue 상태 전이 → AuditLogger hash chain (best-effort)."""
    try:
        from core.monitoring.audit_logger import AuditLogger
        audit = AuditLogger()
    except Exception as exc:  # 감사 로거 불가 시에도 consumer 는 동작
        print(f"[warn] AuditLogger 초기화 실패 — 감사 기록 생략: {exc}")
        return None

    def _callback(event: dict) -> None:
        audit.log_operation(
            operation=event.get("action", "human_review:unknown"),
            input_data={"review_id": event.get("review_id")},
            output_data={"state": event.get("state")},
            user=event.get("reviewer_id") or "hitl_consumer",
            metadata=event,
        )

    return _callback


def build_queue():
    from core.serving.review.human_review_queue import build_human_review_queue
    return build_human_review_queue(
        pipeline_config=_load_pipeline_config(),
        audit_callback=_build_audit_callback(),
    )


def main():
    ap = argparse.ArgumentParser(
        description="HITL 인적 검토 consumer (큐 미결 건 검토·결정)",
    )
    ap.add_argument("--list", action="store_true", help="미결 건 목록 출력")
    ap.add_argument("--tier", type=int, choices=[1, 2, 3], help="tier 필터")
    ap.add_argument("--decide", metavar="REVIEW_ID", help="결정할 review_id")
    ap.add_argument("--action", choices=["approve", "reject", "modify"],
                    help="결정 종류")
    ap.add_argument("--reviewer", default="operator", help="검토자 ID")
    ap.add_argument("--notes", default=None, help="검토 의견 (reject 는 필수)")
    ap.add_argument("--modifications", default=None,
                    help='modify 용 수정 내용 JSON (예: \'{"product_id": "P1"}\')')
    args = ap.parse_args()

    q = build_queue()
    loaded = q.load_pending()

    # 결정 모드
    if args.decide and args.action:
        notes = args.notes or (
            "승인" if args.action == "approve" else f"{args.action} via consumer"
        )
        try:
            if args.action == "approve":
                item = q.approve(args.decide, args.reviewer, notes)
            elif args.action == "reject":
                if not args.notes:
                    print("결정 실패 — reject 는 --notes (사유) 가 필수입니다")
                    sys.exit(1)
                item = q.reject(args.decide, args.reviewer, args.notes)
            else:
                mods = json.loads(args.modifications or "{}")
                if not mods:
                    print("결정 실패 — modify 는 --modifications JSON 이 필수입니다")
                    sys.exit(1)
                item = q.modify(args.decide, args.reviewer, mods, notes)
        except (KeyError, ValueError) as exc:
            print(f"결정 실패: {exc}")
            sys.exit(1)
        print(f"decision: {item.review_id} → {item.state} "
              f"(reviewer={item.reviewer_id}, at={item.disposition_at})")
        flushed = q.flush()
        print(f"flushed → {flushed}건 영속화")
        return

    # 기본: 미결 목록
    stats = q.summary()
    print(f"[load_pending] {loaded}건 로드 | total={stats['total']} "
          f"by_state={stats['by_state']} by_tier={stats['by_tier']}")
    pending = q.list_pending(tier=args.tier)
    for it in pending[:50]:
        print(f"  {it.review_id} | tier={it.tier} | user={it.user_id} | "
              f"rec={it.recommendation_id} | created={it.created_at}")
    if not pending:
        print("  (미결 없음)")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
