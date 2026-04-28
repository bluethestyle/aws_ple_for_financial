"""Dump a SageMaker TrainingJob CloudWatch log via boto3 (Git Bash MSYS-safe)."""
from __future__ import annotations

import sys

import boto3


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: dump_sm_log.py <log-stream-name>", file=sys.stderr)
        sys.exit(2)
    stream = sys.argv[1]
    client = boto3.client("logs", region_name="ap-northeast-2")
    next_token = None
    while True:
        kw = dict(
            logGroupName="/aws/sagemaker/TrainingJobs",
            logStreamName=stream,
            startFromHead=True,
        )
        if next_token:
            kw["nextToken"] = next_token
        resp = client.get_log_events(**kw)
        for ev in resp["events"]:
            line = ev["message"]
            sys.stdout.write(line)
            if not line.endswith("\n"):
                sys.stdout.write("\n")
        new_token = resp.get("nextForwardToken")
        if new_token == next_token or not resp["events"]:
            break
        next_token = new_token


if __name__ == "__main__":
    main()
