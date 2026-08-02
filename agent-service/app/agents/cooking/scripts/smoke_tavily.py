"""Manual smoke test for the Tavily search provider (P1-05).

Requires a REAL API key — never run in CI. The key is read from
COOKING_PLAN_TAVILY_API_KEY (or passed via --key) and is never printed.

Usage:
    COOKING_PLAN_TAVILY_API_KEY=tvly-... uv run python scripts/smoke_tavily.py \
        --query "chicken stir fry heat level" --allow-domains seriouseats.com
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Allow running directly from a checkout.
_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from pydantic import SecretStr  # noqa: E402

from cooking_plan_agent.research.providers.tavily import TavilySearchProvider  # noqa: E402


async def smoke(query: str, allow_domains: list[str], key: str) -> None:
    provider = TavilySearchProvider(api_key=SecretStr(key), search_depth="basic", max_results=3)
    try:
        docs = await provider.search(query, 3, include_domains=tuple(allow_domains))
    finally:
        await provider.aclose()

    print(f"Query: {query}")
    print(f"Results: {len(docs)}")
    for doc in docs:
        print(f"  - {doc.title[:60]} | {doc.url} | domain={doc.domain}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tavily provider smoke test (P1-05).")
    parser.add_argument("--query", default="chicken stir fry heat level", help="Search query.")
    parser.add_argument("--allow-domains", nargs="*", default=["seriouseats.com", "bbcgoodfood.com"])
    parser.add_argument("--key", default=None, help="Tavily API key (prefer COOKING_PLAN_TAVILY_API_KEY).")
    args = parser.parse_args()

    key = args.key or os.environ.get("COOKING_PLAN_TAVILY_API_KEY")
    if not key:
        print("No API key found. Set COOKING_PLAN_TAVILY_API_KEY or pass --key.", file=sys.stderr)
        sys.exit(1)

    asyncio.run(smoke(args.query, args.allow_domains, key))


if __name__ == "__main__":
    main()
