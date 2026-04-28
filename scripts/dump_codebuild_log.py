"""One-shot helper: dump a CodeBuild CloudWatch log to stdout.

Git Bash's MSYS path translation mangles the leading slash on the
``/aws/codebuild/...`` log group name when calling the AWS CLI. The
boto3 client takes an unmodified string, so we use it instead.

Usage:
    python scripts/dump_codebuild_log.py <build-id-uuid>

Where build-id-uuid is the part after ":" in the build id, e.g. for
``ple-mamba-image-build:9d3981a3-38a8-4b66-888a-e316319f72d0`` pass
``9d3981a3-38a8-4b66-888a-e316319f72d0``.
"""
from __future__ import annotations

import sys

import boto3


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: dump_codebuild_log.py <build-uuid>", file=sys.stderr)
        sys.exit(2)
    stream = sys.argv[1]
    client = boto3.client("logs", region_name="ap-northeast-2")
    next_token = None
    while True:
        kw = dict(
            logGroupName="/aws/codebuild/ple-mamba-image-build",
            logStreamName=stream,
            startFromHead=True,
        )
        if next_token:
            kw["nextToken"] = next_token
        resp = client.get_log_events(**kw)
        for ev in resp["events"]:
            sys.stdout.write(ev["message"])
            if not ev["message"].endswith("\n"):
                sys.stdout.write("\n")
        if resp.get("nextForwardToken") == next_token:
            break
        next_token = resp["nextForwardToken"]
        if not resp["events"]:
            break


if __name__ == "__main__":
    main()
