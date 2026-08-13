#!/usr/bin/env python3
"""Fail CI for high-confidence credential forms in selected tracked source paths."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


PATTERNS = {
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "GitHub personal token": re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
    "Slack token": re.compile(r"\bxox(?:b|p|a|r)-[A-Za-z0-9-]{20,}\b"),
    "Stripe secret key": re.compile(r"\bsk_(?:live|test)_[A-Za-z0-9]{20,}\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def tracked_files(paths: list[str]) -> list[Path]:
    output = subprocess.check_output(["git", "ls-files", "-z", "--", *paths])
    return [Path(item.decode("utf-8")) for item in output.split(b"\0") if item]


def main() -> None:
    paths = sys.argv[1:] or ["."]
    findings: list[str] = []
    for path in tracked_files(paths):
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for number, line in enumerate(content.splitlines(), start=1):
            if "ci-secret-scan: allow-test-fixture" in line:
                continue
            for kind, pattern in PATTERNS.items():
                if pattern.search(line):
                    findings.append(f"{path}:{number}: {kind}")
    if findings:
        raise SystemExit("Potential credentials found:\n" + "\n".join(findings))
    print("Tracked-file credential scan passed.")


if __name__ == "__main__":
    main()
