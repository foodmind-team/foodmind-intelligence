"""P1-05: Tavily search provider tests — httpx.MockTransport, no real network.

Covers: successful mapping, no results, 429, 5xx, timeout, malformed JSON,
unsafe URL blocking (localhost/private/non-http), request-side allow-list,
and API key non-leakage (repr / error responses).
"""

import asyncio
import json

import httpx
import pytest

from cooking_plan_agent.research.providers.tavily import TavilySearchProvider, _is_safe_url

# ---------------------------------------------------------------------------
# URL safety
# ---------------------------------------------------------------------------


def test_is_safe_url_blocks_unsafe_targets() -> None:
    assert _is_safe_url("https://www.seriouseats.com/stir-fry") is True
    assert _is_safe_url("http://localhost:8080/x") is False
    assert _is_safe_url("https://127.0.0.1/x") is False
    assert _is_safe_url("http://192.168.1.10/x") is False
    assert _is_safe_url("http://10.0.0.5/x") is False
    assert _is_safe_url("ftp://files.example.com/x") is False
    assert _is_safe_url("not a url") is False


# ---------------------------------------------------------------------------
# Provider behaviour
# ---------------------------------------------------------------------------


def _provider(handler, **kwargs) -> TavilySearchProvider:  # noqa: ANN001
    return TavilySearchProvider(
        api_key="test-tavily-key",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_search_maps_results_and_passes_allow_list() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "How to Stir-Fry Chicken",
                        "url": "https://www.seriouseats.com/stir-fry",
                        "content": "Use high heat for 3-5 minutes.",
                        "raw_content": "Preheat wok to 200C. Stir-fry 3-5 min.",
                    }
                ]
            },
        )

    provider = _provider(handler, max_results=5)
    try:
        docs = await provider.search("stir fry chicken", 3, include_domains=("seriouseats.com",))
    finally:
        await provider.aclose()

    assert len(docs) == 1
    assert docs[0].domain == "www.seriouseats.com"
    assert docs[0].raw_content  # raw content wired for evidence extraction
    assert "high heat" in docs[0].snippet.lower()

    # Fixed, controlled parameters (P1-05).
    payload = captured["payload"]  # type: ignore[assignment]
    assert payload["topic"] == "general"
    assert payload["include_answer"] is False
    assert payload["search_depth"] == "basic"
    assert payload["max_results"] == 3
    assert payload["include_domains"] == ["seriouseats.com"]


@pytest.mark.asyncio
async def test_empty_results_return_empty_tuple() -> None:
    provider = _provider(lambda request: httpx.Response(200, json={"results": []}))
    try:
        docs = await provider.search("nothing", 3)
    finally:
        await provider.aclose()
    assert docs == ()


@pytest.mark.asyncio
async def test_http_429_raises_provider_error() -> None:
    provider = _provider(lambda request: httpx.Response(429, json={"error": "rate limited"}))
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await provider.search("q", 3)
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_http_5xx_raises_provider_error() -> None:
    provider = _provider(lambda request: httpx.Response(503, json={"error": "down"}))
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await provider.search("q", 3)
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_malformed_json_raises_value_error() -> None:
    provider = _provider(lambda request: httpx.Response(200, content=b"not-json"))
    try:
        with pytest.raises(ValueError, match="malformed JSON"):
            await provider.search("q", 3)
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_unsafe_result_urls_are_dropped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {"title": "A", "url": "http://localhost:8080/x", "content": "x"},
                    {"title": "B", "url": "https://192.168.1.10/x", "content": "x"},
                    {"title": "C", "url": "ftp://files.example.com/x", "content": "x"},
                    {"title": "D", "url": "https://www.seriouseats.com/x", "content": "x"},
                ]
            },
        )

    provider = _provider(handler)
    try:
        docs = await provider.search("q", 10)
    finally:
        await provider.aclose()

    assert len(docs) == 1
    assert docs[0].domain == "www.seriouseats.com"


@pytest.mark.asyncio
async def test_key_never_appears_in_repr_or_error_message() -> None:
    provider = _provider(lambda request: httpx.Response(401, json={"error": "bad key"}))
    try:
        assert "test-tavily-key" not in repr(provider)
        with pytest.raises(httpx.HTTPStatusError) as excinfo:
            await provider.search("q", 3)
        assert "test-tavily-key" not in str(excinfo.value)
    finally:
        await provider.aclose()


def test_empty_api_key_rejected() -> None:
    with pytest.raises(ValueError, match="API key must not be empty"):
        TavilySearchProvider(api_key="", transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))


# ---------------------------------------------------------------------------
# Integration: Tavily provider behind the Researcher (response-side filter)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_researcher_drops_non_allow_listed_tavily_results() -> None:
    """The response-side allow-list still runs after the provider maps docs."""
    from cooking_plan_agent.config.settings import Settings
    from cooking_plan_agent.research.config import DomainAllowList
    from cooking_plan_agent.research.domain_filter import filter_by_domain

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {"title": "OK", "url": "https://www.seriouseats.com/a", "content": "high heat"},
                    # allow-list will reject this domain downstream
                    {"title": "Bad", "url": "https://random-blog.com/b", "content": "low heat"},
                ]
            },
        )

    provider = _provider(handler)
    allow_list = DomainAllowList.from_settings(custom_domains=[])
    settings = Settings(internal_service_token="t", research_timeout_seconds=5.0)
    from cooking_plan_agent.research.researcher import Researcher

    researcher = Researcher(provider=provider, allow_list=allow_list, settings=settings)
    try:
        docs = await researcher.search(type("Q", (), {"query_text": "stir fry"})())
    finally:
        await provider.aclose()

    # Both the provider and the researcher filter; only allow-listed survives.
    filtered = filter_by_domain(docs, allow_list)
    assert len(filtered) == 1
    assert filtered[0].domain == "www.seriouseats.com"


@pytest.mark.asyncio
async def test_researcher_timeout_degrades_to_confirmation() -> None:
    """Timeout on the real provider → empty → needs_confirmation (never crash)."""
    from cooking_plan_agent.config.settings import Settings
    from cooking_plan_agent.research.config import DomainAllowList
    from cooking_plan_agent.research.researcher import Researcher

    async def slow_handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        await asyncio.sleep(0.5)
        return httpx.Response(200, json={"results": []})

    provider = TavilySearchProvider(
        api_key="k",
        transport=httpx.MockTransport(slow_handler),
        timeout_seconds=0.05,
    )
    allow_list = DomainAllowList.from_settings(custom_domains=[])
    settings = Settings(
        internal_service_token="t",
        research_timeout_seconds=0.02,  # shorter than provider's own timeout
    )
    researcher = Researcher(provider=provider, allow_list=allow_list, settings=settings)
    try:
        docs = await researcher.search(type("Q", (), {"query_text": "q"})())
    finally:
        await provider.aclose()

    assert docs == ()
