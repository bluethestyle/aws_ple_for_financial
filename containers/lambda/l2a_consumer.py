"""
SQS Consumer Lambda for L2a Bedrock reason rewrite.

Triggered by SQS messages from ple-reason-l2a-queue.
Each message contains:
    {
        "user_id": "...",
        "task_name": "churn_signal",
        "l1_text": "고객님의 소중한 거래 관계를...",
        "facts": "신규 고객 (가입 6개월 이내)",
        "similar_context": "유사고객(123): 02-PARTICULARES; ..."
    }

Writes the L2a rewritten reason to DynamoDB reason_cache:
    pk = "{user_id}:{task_name}"
    l2a_reason = "다듬어진 한국어 문장"
    ttl = now + 24h
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict

logger = logging.getLogger()
logger.setLevel(logging.INFO)

REGION = os.environ.get("AWS_REGION", "ap-northeast-2")
CACHE_TABLE = os.environ.get("REASON_CACHE_TABLE", "reason_cache")
CACHE_TTL_HOURS = int(os.environ.get("REASON_CACHE_TTL_HOURS", "24"))

# Model config from env (set in Lambda configuration)
MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID", "global.anthropic.claude-sonnet-4-6"
)
MAX_TOKENS = int(os.environ.get("BEDROCK_MAX_TOKENS", "256"))
TEMPERATURE = float(os.environ.get("BEDROCK_TEMPERATURE", "0.3"))

_BEDROCK_CLIENT = None
_DDB_CLIENT = None


def _get_bedrock():
    global _BEDROCK_CLIENT
    if _BEDROCK_CLIENT is None:
        import boto3
        _BEDROCK_CLIENT = boto3.client("bedrock-runtime", region_name=REGION)
    return _BEDROCK_CLIENT


def _get_ddb():
    global _DDB_CLIENT
    if _DDB_CLIENT is None:
        import boto3
        _DDB_CLIENT = boto3.client("dynamodb", region_name=REGION)
    return _DDB_CLIENT


def _rewrite(l1_text: str, facts: str, similar_context: str) -> str:
    """Call Bedrock to rewrite L1 reason into natural Korean."""
    client = _get_bedrock()

    context_parts = [l1_text]
    if facts:
        context_parts.append(f"고객 특성: {facts}")
    if similar_context:
        context_parts.append(f"유사 고객 참고: {similar_context}")
    user_prompt = "\n".join(context_parts)

    system_prompt = (
        "당신은 금융 상품 추천사유 작성자입니다. "
        "입력된 추천사유를 고객에게 전달할 자연스러운 한국어 1~2문장으로 "
        "다듬어 출력하세요. 분석, 검토, 마크다운, 제목, 목록 없이 "
        "오직 다듬어진 문장만 출력하세요."
    )

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    })

    response = client.invoke_model(
        modelId=MODEL_ID,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    result = json.loads(response["body"].read())
    content = result.get("content", [])
    if content and isinstance(content, list):
        return content[0].get("text", "").strip()
    return ""


def _write_cache(user_id: str, task_name: str, l2a_text: str) -> None:
    """Write L2a reason to DynamoDB reason_cache with TTL."""
    ddb = _get_ddb()
    ttl = int(time.time()) + CACHE_TTL_HOURS * 3600

    ddb.put_item(
        TableName=CACHE_TABLE,
        Item={
            "pk": {"S": f"{user_id}:{task_name}"},
            "user_id": {"S": user_id},
            "task_name": {"S": task_name},
            "l2a_reason": {"S": l2a_text},
            "model_id": {"S": MODEL_ID},
            "created_at": {"S": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
            "ttl": {"N": str(ttl)},
        },
    )


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """SQS trigger handler — process L2a rewrite requests."""
    records = event.get("Records", [])
    success = 0
    failed = 0

    for record in records:
        t0 = time.time()
        try:
            body = json.loads(record.get("body", "{}"))
            user_id = body.get("user_id", "")
            task_name = body.get("task_name", "")
            l1_text = body.get("l1_text", "")
            facts = body.get("facts", "")
            similar_context = body.get("similar_context", "")

            if not l1_text or not user_id:
                logger.warning("Skipping empty message: %s", body)
                failed += 1
                continue

            # Bedrock rewrite
            l2a_text = _rewrite(l1_text, facts, similar_context)

            if l2a_text:
                _write_cache(user_id, task_name, l2a_text)
                success += 1
                logger.info(
                    "L2a OK: user=%s task=%s model=%s %.1fs | %s",
                    user_id, task_name, MODEL_ID,
                    time.time() - t0, l2a_text[:60],
                )
            else:
                # Bedrock returned empty — cache L1 as fallback
                _write_cache(user_id, task_name, l1_text)
                success += 1
                logger.warning(
                    "L2a empty, cached L1: user=%s task=%s", user_id, task_name,
                )

        except Exception:
            failed += 1
            logger.exception("L2a processing failed for record")

    return {
        "batchItemFailures": [],  # SQS batch failure reporting
        "processed": success,
        "failed": failed,
    }
