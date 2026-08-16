#!/usr/bin/env python3
"""Fail when CodeQL reports a Medium-or-higher security finding."""

from __future__ import annotations

import json
import sys
from pathlib import Path


THRESHOLD = 4.0


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: check-codeql-sarif.py <sarif-directory>")
    sarif_files = sorted(Path(sys.argv[1]).rglob("*.sarif"))
    if not sarif_files:
        raise SystemExit("CodeQL gate failed: no SARIF report was produced")
    blocking: list[str] = []
    for sarif_file in sarif_files:
        report = json.loads(sarif_file.read_text(encoding="utf-8"))
        for run in report.get("runs", []):
            rules = {rule.get("id"): rule for rule in run.get("tool", {}).get("driver", {}).get("rules", []) if rule.get("id")}
            for result in run.get("results", []):
                rule_id = result.get("ruleId", "unknown-rule")
                raw_score = rules.get(rule_id, {}).get("properties", {}).get("security-severity")
                try:
                    score = float(raw_score)
                except (TypeError, ValueError):
                    continue
                if score >= THRESHOLD:
                    locations = result.get("locations", [])
                    uri = "unknown-location"
                    if locations:
                        uri = locations[0].get("physicalLocation", {}).get("artifactLocation", {}).get("uri", uri)
                    blocking.append(f"{rule_id} (CVSS {score:g}) at {uri}")
    if blocking:
        print("CodeQL gate failed: Medium-or-higher findings detected:", file=sys.stderr)
        for finding in blocking:
            print(f"- {finding}", file=sys.stderr)
        return 1
    print("CodeQL gate passed: no Medium-or-higher security findings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
