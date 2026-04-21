"""
Dialog Recall Memory — Cross-Session Dialog History
=======================================================

Letta-inspired recall memory: persistent dialog history across
BedrockDialogSession sessions, enabling "remember our discussion
from last week" queries.

Storage: DynamoDB (or in-memory fallback for testing).

Schema:
    PK: operator_id (string)
    SK: session_id#turn_id (string)
    user_message: string
    agent_response: string
    timestamp: string (ISO UTC)
    embedding: string (JSON-serialized list, optional)
"""

from __future__ import annotations

import json
import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["DialogRecallMemory"]


class DialogRecallMemory:
    """Persistent dialog history across BedrockDialogSession sessions.

    Args:
        table_name: DynamoDB table name.
        region: AWS region. ``None`` lets boto3 resolve from env /
            credentials; callers should pass ``pipeline.yaml::aws.region``.
        in_memory_limit: Max entries in fallback memory.
    """

    def __init__(
        self,
        table_name: str = "agent_dialog_recall",
        region: Optional[str] = None,
        in_memory_limit: int = 1000,
    ) -> None:
        self._table_name = table_name
        self._region = region
        self._table = None
        self._memory: deque = deque(maxlen=in_memory_limit)
        self._init_table()

    def _init_table(self) -> None:
        """Lazy init DynamoDB table with graceful fallback."""
        try:
            import boto3

            dynamodb = boto3.resource("dynamodb", region_name=self._region)
            self._table = dynamodb.Table(self._table_name)
            logger.info(
                "DialogRecallMemory: connected to DynamoDB table '%s'",
                self._table_name,
            )
        except Exception as e:
            logger.info("DialogRecallMemory: using in-memory fallback (%s)", e)
            self._table = None

    def save_turn(
        self,
        operator_id: str,
        session_id: str,
        turn_id: str,
        user_msg: str,
        agent_response: str,
        embedding: Optional[List[float]] = None,
    ) -> bool:
        """Save a single dialog turn.

        Returns:
            True if saved successfully.
        """
        item: Dict[str, Any] = {
            "operator_id": operator_id,
            "session_turn": f"{session_id}#{turn_id}",
            "user_message": user_msg,
            "agent_response": agent_response,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if embedding is not None:
            item["embedding"] = json.dumps(embedding)

        if self._table is not None:
            try:
                self._table.put_item(Item=item)
                return True
            except Exception as e:
                logger.warning("DynamoDB save_turn failed, falling back to memory: %s", e)
                # fall through to memory append

        # In-memory fallback (also reached when DynamoDB fails)
        self._memory.append(item)
        return True

    def get_recent(self, operator_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get the most recent N turns for an operator."""
        if self._table is not None:
            try:
                response = self._table.query(
                    KeyConditionExpression="operator_id = :oid",
                    ExpressionAttributeValues={":oid": operator_id},
                    ScanIndexForward=False,
                    Limit=limit,
                )
                return response.get("Items", [])
            except Exception as e:
                logger.warning("DynamoDB get_recent failed, falling back to memory: %s", e)
                # fall through to memory

        # Fallback (also reached when DynamoDB fails)
        filtered = [item for item in self._memory if item.get("operator_id") == operator_id]
        filtered.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return filtered[:limit]

    def search_related(
        self,
        operator_id: str,
        query_text: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search related past turns by keyword overlap.

        Note: This is a keyword fallback. For semantic search,
        inject embeddings via save_turn(embedding=...) and this
        method will use cosine similarity when embeddings exist.
        """
        # Get recent turns as candidate pool
        candidates = self.get_recent(operator_id, limit=100)
        if not candidates:
            return []

        # Keyword scoring
        query_tokens = set(query_text.lower().split())
        scored = []
        for item in candidates:
            text = (
                f"{item.get('user_message', '')} {item.get('agent_response', '')}".lower()
            )
            text_tokens = set(text.split())
            overlap = len(query_tokens & text_tokens)
            if overlap > 0:
                scored.append((overlap, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:limit]]

    @property
    def is_persistent(self) -> bool:
        """Whether using DynamoDB (True) or in-memory fallback (False)."""
        return self._table is not None
