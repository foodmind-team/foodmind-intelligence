"""Export or validate the Cooking Plan Agent OpenAPI document."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("COOKING_PLAN_INTERNAL_SERVICE_TOKEN", "openapi-export-local-token")

from cooking_plan_agent.main import create_app  # noqa: E402

REQUIRED_PATHS = {
    "/health/live",
    "/health/ready",
    "/internal/v1/agents/cooking-plan/generate",
    "/internal/v1/cooking-plans/generate",
    "/internal/v2/cooking-plan/tasks",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="validate without writing openapi.json")
    args = parser.parse_args()

    schema = create_app().openapi()
    missing_paths = sorted(REQUIRED_PATHS.difference(schema.get("paths", {})))
    if not str(schema.get("openapi", "")).startswith("3."):
        raise SystemExit("OpenAPI document does not declare a supported 3.x version")
    if missing_paths:
        raise SystemExit(f"OpenAPI document is missing required paths: {', '.join(missing_paths)}")

    rendered = json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    json.loads(rendered)
    if args.check:
        print(f"OpenAPI check passed: {len(schema['paths'])} paths")
        return 0

    output = Path("openapi.json")
    output.write_text(rendered, encoding="utf-8")
    print(f"Exported {output} with {len(schema['paths'])} paths")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
