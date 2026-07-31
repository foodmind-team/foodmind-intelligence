"""OpenAPI spec export and contract-validation script.

Export the FastAPI-generated OpenAPI 3.1 schema to a JSON file and run
basic sanity checks on it.  This provides a contract artifact for Spring
Boot integration.

Usage:
    python scripts/export_openapi.py              # export to openapi.json
    python scripts/export_openapi.py --check      # export + validate
    python scripts/export_openapi.py -o api.json  # custom output path

Handbook 9.1: the public boundary stays in Spring Boot. This script
produces the internal contract artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def export_openapi_spec(output_path: str) -> dict:
    """Export the FastAPI OpenAPI spec from the application.

    Uses FastAPI's built-in .openapi() method rather than TestClient
    to avoid starting a server.  This is deterministic and offline.

    Returns:
        The OpenAPI spec as a Python dict.
    """
    from cooking_plan_agent.main import app

    spec = app.openapi()
    return spec


def validate_openapi_spec(spec: dict) -> list[str]:
    """Run basic sanity checks on an OpenAPI spec.

    Checks:
      - OpenAPI version is 3.x
      - info block has title and version
      - At least one path is defined
      - /health/live and /health/ready exist
      - POST /internal/v1/agents/cooking-plan/generate exists
      - Internal endpoint requires X-Internal-Token header

    Returns:
        List of issue strings. Empty = valid.
    """
    issues: list[str] = []

    # Version check
    version = spec.get("openapi", "")
    if not version.startswith("3."):
        issues.append(f"OpenAPI version is '{version}', expected 3.x")

    # Info block
    info = spec.get("info", {})
    if not info.get("title"):
        issues.append("Missing info.title")
    if not info.get("version"):
        issues.append("Missing info.version")

    # Paths
    paths = spec.get("paths", {})
    if not paths:
        issues.append("No paths defined in spec")
        return issues

    # Health endpoints
    if "/health/live" not in paths:
        issues.append("Missing /health/live endpoint")
    if "/health/ready" not in paths:
        issues.append("Missing /health/ready endpoint")

    # Generate endpoint
    generate_path = "/internal/v1/agents/cooking-plan/generate"
    if generate_path not in paths:
        issues.append(f"Missing {generate_path} endpoint")
    else:
        post_op = paths[generate_path].get("post", {})
        if not post_op:
            issues.append(f"{generate_path} must be POST")

        # Check for X-Internal-Token header parameter
        params = post_op.get("parameters", [])
        has_auth_header = any(
            p.get("name") == "x-internal-token" for p in params
            if isinstance(p, dict)
        )
        if not has_auth_header:
            issues.append(f"{generate_path} missing x-internal-token header parameter")

        # Check response schemas
        responses = post_op.get("responses", {})
        if "200" not in responses:
            issues.append(f"{generate_path} missing 200 response definition")

    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and validate OpenAPI spec.")
    parser.add_argument(
        "-o", "--output", default="openapi.json",
        help="Output file path (default: openapi.json)",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Run validation checks on the exported spec",
    )
    parser.add_argument(
        "--pretty", action="store_true", default=True,
        help="Pretty-print JSON (default: true)",
    )
    args = parser.parse_args()

    print("Exporting OpenAPI spec from cooking_plan_agent.main:app ...")
    spec = export_openapi_spec(args.output)

    indent = 2 if args.pretty else None
    output_path = Path(args.output)
    output_path.write_text(json.dumps(spec, indent=indent, default=str), encoding="utf-8")
    print(f"  Exported to: {output_path.absolute()}")

    if args.check:
        print("\nValidating spec ...")
        issues = validate_openapi_spec(spec)
        if issues:
            print(f"  FAILED ({len(issues)} issue(s)):")
            for issue in issues:
                print(f"    - {issue}")
            sys.exit(1)
        else:
            print("  PASSED — all checks OK")

    print("\nDone.")


if __name__ == "__main__":
    main()
