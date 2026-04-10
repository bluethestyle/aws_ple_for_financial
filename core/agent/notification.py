"""
Notification Service for Agent Reports
=========================================

Delivers agent reports and alerts via multiple channels:
    - Slack (webhook)
    - SNS (AWS)
    - Email (SES — stub for future)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["NotificationService"]


class NotificationService:
    """Multi-channel notification delivery.

    Args:
        config: Notification config with channel-specific settings.
            slack: {webhook_url: str, channel: str}
            sns: {topic_arn: str, region: str}
            email: {recipients: list, sender: str}  # future
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._config = config or {}
        self._slack_config = self._config.get("slack", {})
        self._sns_config = self._config.get("sns", {})

    def send(
        self,
        subject: str,
        body: Dict[str, Any],
        channels: Optional[List[str]] = None,
        severity: str = "INFO",
    ) -> Dict[str, bool]:
        """Send notification to specified channels.

        Args:
            subject: Notification subject/title.
            body: Structured report data.
            channels: List of channels ("slack", "sns"). Default: all configured.
            severity: INFO / WARNING / CRITICAL.

        Returns:
            Dict of {channel: success_bool}.
        """
        if channels is None:
            channels = []
            if self._slack_config.get("webhook_url"):
                channels.append("slack")
            if self._sns_config.get("topic_arn"):
                channels.append("sns")

        results = {}
        for channel in channels:
            try:
                if channel == "slack":
                    results["slack"] = self._send_slack(subject, body, severity)
                elif channel == "sns":
                    results["sns"] = self._send_sns(subject, body, severity)
                else:
                    logger.warning("Unknown notification channel: %s", channel)
                    results[channel] = False
            except Exception as e:
                logger.error("Notification to %s failed: %s", channel, e)
                results[channel] = False

        return results

    def _send_slack(self, subject: str, body: Dict, severity: str) -> bool:
        """Send to Slack via webhook."""
        webhook_url = self._slack_config.get("webhook_url")
        if not webhook_url:
            return False

        emoji = {"CRITICAL": ":red_circle:", "WARNING": ":yellow_circle:", "INFO": ":green_circle:"}.get(severity, ":white_circle:")

        # Build Slack blocks
        text = f"{emoji} *{subject}*\n"

        # Add attention items
        attention = body.get("attention_required", body.get("focus_areas", []))
        if attention:
            for item in attention[:5]:
                finding = item.get("finding", item.get("detail", ""))
                text += f"- {finding}\n"

        payload = {"text": text}

        try:
            import urllib.request
            req = urllib.request.Request(
                webhook_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status == 200
        except Exception as e:
            logger.error("Slack webhook failed: %s", e)
            return False

    def _send_sns(self, subject: str, body: Dict, severity: str) -> bool:
        """Send to AWS SNS topic."""
        topic_arn = self._sns_config.get("topic_arn")
        if not topic_arn:
            return False

        try:
            import boto3
            client = boto3.client(
                "sns",
                region_name=self._sns_config.get("region", "ap-northeast-2"),
            )
            message = json.dumps(body, ensure_ascii=False, indent=2, default=str)
            client.publish(
                TopicArn=topic_arn,
                Subject=f"[{severity}] {subject}"[:100],  # SNS subject limit
                Message=message[:262144],  # SNS message limit
            )
            return True
        except Exception as e:
            logger.error("SNS publish failed: %s", e)
            return False
