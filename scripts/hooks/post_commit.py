#!/usr/bin/env python3
"""
Git post-commit hook for ChangeDetector.

Install:
    cp scripts/hooks/post_commit.py .git/hooks/post-commit
    chmod +x .git/hooks/post-commit

Or symlink:
    ln -s ../../scripts/hooks/post_commit.py .git/hooks/post-commit
"""

import subprocess
import sys


def main():
    """Extract changed files from latest commit and notify ChangeDetector."""
    try:
        # Get commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        commit_hash = result.stdout.strip()

        # Get changed files
        result = subprocess.run(
            ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        changed_files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]

        if not changed_files:
            return

        # Try to notify ChangeDetector
        try:
            from core.agent.change_detector import ChangeDetector
            detector = ChangeDetector()
            event = detector.on_git_commit(commit_hash, changed_files)
            print(f"[post-commit] Change detected: {len(changed_files)} files, parts: {event.affected_parts}")
        except ImportError:
            # ChangeDetector not available — just print
            print(f"[post-commit] {commit_hash[:8]}: {len(changed_files)} files changed")

    except Exception as e:
        # Never block the commit
        print(f"[post-commit] Warning: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
