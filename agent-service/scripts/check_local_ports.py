"""Assert that each canonical local port is owned by the intended service."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

SERVICES = {
    8001: ("Chatbot", "/internal/v1/chat/generate"),
    8002: ("Inference Service", "/internal/v1/recommendations/score"),
    8003: ("Cooking Agent", "/internal/v1/agents/cooking-plan/generate"),
    8004: ("Recommendation Agent", "/internal/v1/recommendations/generate"),
}


def main() -> int:
    failed = False
    expected_paths = {path for _, path in SERVICES.values()}
    for port, (name, expected_path) in SERVICES.items():
        url = f"http://127.0.0.1:{port}/openapi.json"
        try:
            with urllib.request.urlopen(url, timeout=3) as response:  # noqa: S310 - fixed loopback URL
                document = json.load(response)
            paths = set(document.get("paths", {}))
            wrong_paths = (paths & expected_paths) - {expected_path}
            if expected_path not in paths or wrong_paths:
                raise ValueError(f"expected {expected_path}; conflicting routes={sorted(wrong_paths)}")
            print(f"PASS {port}: {name} owns {expected_path}")
        except (OSError, ValueError, TypeError, json.JSONDecodeError, urllib.error.URLError) as exc:
            failed = True
            print(f"FAIL {port}: {name}: {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
