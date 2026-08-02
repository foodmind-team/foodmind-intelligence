"""Tavily search provider — real, controlled web search (P1-05).

Implements the SearchProvider Protocol from researcher.py using plain httpx
against the Tavily ``POST /search`` endpoint (no vendor SDK). Key design
points (P1-05):

  - API key is a ``SecretStr`` and must never appear in repr, logs, or error
    responses — it is only placed in the request body.
  - Fixed, controlled parameters: ``topic="general"``, explicit
    ``search_depth``, result cap, and ``include_answer=False`` so the answer
    feature never leaks untracked content into the pipeline.
  - Allow-list enforcement happens in BOTH layers: the request carries
    ``include_domains`` (request-side), and every returned URL is re-checked
    here (defensive: non-http(s), localhost, and private-range hosts are
    dropped) and again by ``filter_by_domain`` in the Researcher (response
    side).
  - 400 / 401 / 429 / 5xx / timeout / malformed JSON are all surfaced as
    provider-level failures that the Researcher turns into empty results →
    NEEDS_CONFIRMATION. Never a raw exception to the workflow.

The provider owns one lifecycle-level ``httpx.AsyncClient`` (closed via
``aclose()``); the app lifespan registers it for shutdown. CI tests inject a
``httpx.MockTransport`` so no real network is ever hit.
"""

from __future__ import annotations

import ipaddress
import logging
from typing import Any
from urllib.parse import urlparse

import httpx
from pydantic import SecretStr

from cooking_plan_agent.domain.models import SearchDocument

logger = logging.getLogger(__name__)

# Blocked local/private hosts — never allowed even if a domain allow-list
# entry accidentally matched.
_LOCALHOST_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})


def _is_safe_url(url: str) -> bool:
    """Return True only for public http(s) URLs (blocks private/local hosts)."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    host = parsed.hostname or ""
    if host.lower() in _LOCALHOST_HOSTS:
        return False
    # Private IPv4 ranges must never be searched or consumed.
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return True  # hostname, not an IP — allow-list check applies elsewhere
    return not ip.is_private and not ip.is_loopback and not ip.is_link_local


class TavilySearchProvider:
    """Tavily-backed SearchProvider with bounded, deterministic parameters."""

    def __init__(
        self,
        api_key: SecretStr | str,
        *,
        base_url: str = "https://api.tavily.com",
        search_depth: str = "basic",
        max_results: int = 3,
        connection_pool_size: int = 10,
        timeout_seconds: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
        if not self._api_key:
            raise ValueError("Tavily API key must not be empty")
        self._search_depth = search_depth
        self._max_results = max(1, int(max_results))
        # Lifecycle-level client: one pool, closed once via aclose().
        self._client = httpx.AsyncClient(
            transport=transport,  # None → default transport (real network)
            timeout=httpx.Timeout(timeout_seconds),
            limits=httpx.Limits(
                max_connections=connection_pool_size,
                max_keepalive_connections=connection_pool_size,
            ),
        )

    async def aclose(self) -> None:
        """Close the shared httpx client (idempotent-safe)."""
        try:
            await self._client.aclose()
        except httpx.TransportError:
            return

    def __repr__(self) -> str:
        """Redacted repr — the API key must never appear (P1-05)."""
        return f"TavilySearchProvider(base_url={self._base_url!r}, search_depth={self._search_depth!r})"

    async def search(
        self,
        query: str,
        max_results: int,
        *,
        include_domains: tuple[str, ...] = (),
    ) -> tuple[SearchDocument, ...]:
        """Execute a Tavily search and normalise results into SearchDocuments.

        Args:
            query: Free-text search query.
            max_results: Requested result cap (bounded by self._max_results).
            include_domains: Request-side allow-list passed to Tavily.

        Returns:
            Normalised SearchDocument tuple. Empty on any failure (the
            Researcher maps that to needs_confirmation).

        Raises:
            httpx.HTTPError / ValueError: transport or protocol failures —
                the Researcher's search() wrapper catches them and degrades.
        """
        result_cap = min(max(1, int(max_results)), self._max_results)
        payload: dict[str, Any] = {
            "api_key": self._api_key,
            "query": query,
            "search_depth": self._search_depth,
            "topic": "general",
            "max_results": result_cap,
            "include_answer": False,  # never surface the LLM answer blob
            "include_raw_content": True,  # evidence extraction reads raw content
            "include_domains": list(include_domains),
        }

        response = await self._client.post(f"{self._base_url}/search", json=payload)
        response.raise_for_status()

        try:
            data = response.json()
        except ValueError as exc:
            raise ValueError("Tavily returned malformed JSON") from exc
        if not isinstance(data, dict):
            raise ValueError("Tavily response is not a JSON object")

        raw_results = data.get("results") or []
        documents: list[SearchDocument] = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not _is_safe_url(url):
                # Defensive layer: block localhost/private/non-http results.
                logger.warning("Dropping unsafe search result URL: %s", url[:120])
                continue
            documents.append(
                SearchDocument(
                    title=str(item.get("title") or "").strip()[:200],
                    url=url,
                    snippet=str(item.get("content") or "").strip()[:1000],
                    raw_content=str(item.get("raw_content") or "").strip()[:10_000],
                    domain=(urlparse(url).hostname or "").lower(),
                )
            )

        return tuple(documents[:result_cap])
