from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from pydantic_core import ValidationError

from chat_agent.clients.backend import BackendToolClient, BackendToolError
from chat_agent.config.settings import Settings
from chat_agent.domain.models import GroundedSource, SourceType
from chat_agent.main import create_app


class FakeBackendTools:
    def __init__(self) -> None:
        self.search_calls: list[dict[str, Any]] = []
        self.explore_calls: list[dict[str, Any]] = []
        self.resolve_calls: list[dict[str, Any]] = []
        self.fail = False
        self.empty_search = False
        self.search_source_type: SourceType = "FOOD_PRODUCT"

    async def search(self, **kwargs: Any) -> tuple[GroundedSource, ...]:
        self.search_calls.append(kwargs)
        if self.fail:
            raise BackendToolError("unavailable")
        if self.empty_search:
            return ()
        return (
            GroundedSource(
                source_type=self.search_source_type,
                source_id=uuid4(),
                title="Oat drink",
                snippet="Unsweetened oat drink",
                grounding_metadata={"origin": "backend_search"},
            ),
        )

    async def explore(self, **kwargs: Any) -> tuple[GroundedSource, ...]:
        self.explore_calls.append(kwargs)
        if self.fail:
            raise BackendToolError("unavailable")
        return (
            GroundedSource(
                source_type="FOOD_RECORD",
                source_id=uuid4(),
                title="Chicken rice",
                subtitle="Orchard Garden Kitchen",
                snippet="at Orchard Garden Kitchen",
                grounding_metadata={"origin": "backend_explore", "hasNext": False},
            ),
        )

    async def resolve(self, **kwargs: Any) -> tuple[GroundedSource, ...]:
        self.resolve_calls.append(kwargs)
        if self.fail:
            raise BackendToolError("unavailable")
        reference_id = kwargs["reference_ids"][0]
        return (
            GroundedSource(
                source_type="FOOD_PRODUCT",
                source_id=uuid4(),
                title="Oat drink",
                snippet="Unsweetened oat drink",
                grounding_metadata={"referenceId": str(reference_id), "origin": "backend_reference_resolve"},
            ),
        )


class FakeLLM:
    def __init__(self, answer: str = "Provider generated answer") -> None:
        self.answer = answer
        self.calls: list[dict[str, Any]] = []

    async def chat(self, messages: list[dict[str, str]], *, timeout_seconds: float | None = None) -> str:
        self.calls.append({"messages": messages, "timeout_seconds": timeout_seconds})
        return self.answer


def request_payload(*, references: bool = False, requested_route: str | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "contractVersion": "chat-agent-v1",
        "requestId": str(uuid4()),
        "sessionId": str(uuid4()),
        "userMessageId": str(uuid4()),
        "traceId": str(uuid4()),
        "expiresAt": (datetime.now(UTC) + timedelta(minutes=1)).isoformat(),
        "message": "Summarise this item" if references else "Hello",
        "requestedRoute": requested_route,
        "delegationToken": "delegation-token",
        "sharedReferences": [],
    }
    if references:
        payload["sharedReferences"] = [
            {
                "referenceId": str(uuid4()),
                "sourceType": "FOOD_PRODUCT",
                "sourceId": str(uuid4()),
                "title": "Oat drink",
                "snippet": "Unsweetened oat drink",
            }
        ]
    return payload


def settings() -> Settings:
    return Settings(environment="test", internal_service_token=SecretStr("test-chat-token"), llm_enabled=False)


def test_navigation_fallback_matches_backend_contract() -> None:
    with TestClient(create_app(settings=settings(), backend_tool_client=FakeBackendTools())) as client:  # type: ignore[arg-type]
        response = client.post(
            "/internal/v1/chat/generate",
            headers={"Authorization": "Bearer test-chat-token"},
            json=request_payload(),
        )
    assert response.status_code == 200
    assert response.json()["status"] == "SUCCEEDED"
    assert response.json()["route"] == "NAVIGATION"
    assert response.json()["responseStatus"] == "FALLBACK_SUCCEEDED"
    assert "Inventory" in response.json()["answer"]
    assert "Shopping Lists" in response.json()["answer"]


def test_grounded_summary_resolves_and_cites_authorised_reference() -> None:
    tools = FakeBackendTools()
    payload = request_payload(references=True)
    with TestClient(create_app(settings=settings(), backend_tool_client=tools)) as client:  # type: ignore[arg-type]
        response = client.post(
            "/internal/v1/chat/generate",
            headers={"Authorization": "Bearer test-chat-token"},
            json=payload,
        )
    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "SUMMARY"
    assert body["sources"][0]["groundingMetadata"]["origin"] == "backend_reference_resolve"
    assert len(tools.resolve_calls) == 1


def test_explicit_search_route_wins_and_searches_without_shared_references() -> None:
    tools = FakeBackendTools()
    payload = request_payload(requested_route="SEARCH")
    payload["message"] = "show platform items"
    with TestClient(create_app(settings=settings(), backend_tool_client=tools)) as client:  # type: ignore[arg-type]
        response = client.post(
            "/internal/v1/chat/generate",
            headers={"Authorization": "Bearer test-chat-token"},
            json=payload,
        )
    assert response.status_code == 200
    assert response.json()["route"] == "SEARCH"
    assert response.json()["sources"][0]["groundingMetadata"]["origin"] == "backend_search"
    assert tools.search_calls[0]["delegation_token"] == "delegation-token"


@pytest.mark.parametrize("message", ["Find the place I recorded recently", "我最近记录的地点是什么？"])
def test_recent_record_intent_explores_food_records_instead_of_full_text_search(message: str) -> None:
    tools = FakeBackendTools()
    payload = request_payload(requested_route="SEARCH")
    payload["message"] = message
    with TestClient(create_app(settings=settings(), backend_tool_client=tools)) as client:  # type: ignore[arg-type]
        response = client.post(
            "/internal/v1/chat/generate",
            headers={"Authorization": "Bearer test-chat-token"},
            json=payload,
        )

    assert response.status_code == 200
    assert response.json()["route"] == "SEARCH"
    assert response.json()["sources"][0]["sourceType"] == "FOOD_RECORD"
    assert "Orchard Garden Kitchen" in response.json()["answer"]
    assert tools.search_calls == []
    assert len(tools.explore_calls) == 1
    assert tools.explore_calls[0]["source_types"] == ["FOOD_RECORD"]
    assert tools.explore_calls[0]["delegation_token"] == "delegation-token"


def test_count_question_routes_to_authorised_search_instead_of_navigation() -> None:
    tools = FakeBackendTools()
    tools.search_source_type = "PLACE"
    llm = FakeLLM("DeepSeek counted the authorised places.")
    payload = request_payload()
    payload["message"] = "Can you see how many restaurants are there?"
    with TestClient(
        create_app(settings=settings(), llm_client=llm, backend_tool_client=tools)  # type: ignore[arg-type]
    ) as client:
        response = client.post(
            "/internal/v1/chat/generate",
            headers={"Authorization": "Bearer test-chat-token"},
            json=payload,
        )
    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "SEARCH"
    assert body["responseStatus"] == "SUCCEEDED"
    assert body["answer"] == "DeepSeek counted the authorised places."
    assert body["agentTraceId"].startswith("chat-llm-")
    assert '"verifiedCount":1' in llm.calls[0]["messages"][1]["content"]
    assert "avoid canned templates" in llm.calls[0]["messages"][0]["content"]
    assert tools.search_calls[0]["query"] == "Can you see how many restaurants are there?"


def test_recommendation_question_is_answered_readonly_instead_of_out_of_scope() -> None:
    payload = request_payload(requested_route="SEARCH")
    payload["message"] = "recommend what I should cook"
    with TestClient(create_app(settings=settings(), backend_tool_client=FakeBackendTools())) as client:  # type: ignore[arg-type]
        response = client.post(
            "/internal/v1/chat/generate",
            headers={"Authorization": "Bearer test-chat-token"},
            json=payload,
        )
    assert response.status_code == 200
    assert response.json()["route"] == "SEARCH"
    assert response.json()["responseStatus"] == "FALLBACK_SUCCEEDED"


def test_search_tool_failure_returns_source_free_navigation() -> None:
    tools = FakeBackendTools()
    tools.fail = True
    payload = request_payload(requested_route="SEARCH")
    with TestClient(create_app(settings=settings(), backend_tool_client=tools)) as client:  # type: ignore[arg-type]
        response = client.post(
            "/internal/v1/chat/generate",
            headers={"Authorization": "Bearer test-chat-token"},
            json=payload,
        )
    assert response.status_code == 200
    assert response.json()["route"] == "NAVIGATION"
    assert response.json()["sources"] == []


def test_authentication_is_required() -> None:
    with TestClient(create_app(settings=settings(), backend_tool_client=FakeBackendTools())) as client:  # type: ignore[arg-type]
        response = client.post("/internal/v1/chat/generate", json=request_payload())
    assert response.status_code == 401
    assert response.json()["error_code"] == "MISSING_AUTHORIZATION_HEADER"


@pytest.mark.asyncio
async def test_backend_tools_send_service_and_delegation_tokens() -> None:
    source_id = uuid4()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer backend-tool-token"
        assert request.headers["X-FoodMind-Delegation"] == "Bearer delegated-user-token"
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "sourceType": "FOOD_PRODUCT",
                        "sourceId": str(source_id),
                        "title": "Oat drink",
                        "snippet": "Unsweetened",
                        "visibility": "PRIVATE",
                        "groupId": None,
                    }
                ],
                "nextCursor": None,
                "hasNext": False,
            },
        )

    raw = httpx.AsyncClient(base_url="http://backend.test", transport=httpx.MockTransport(handler))
    client = BackendToolClient(
        client=raw,
        settings=Settings(
            environment="test",
            backend_base_url="http://backend.test",
            backend_service_token=SecretStr("backend-tool-token"),
        ),
    )
    sources = await client.search(query="oat", delegation_token="delegated-user-token", timeout_seconds=1)
    await client.aclose()

    assert sources[0].source_id == UUID(str(source_id))


@pytest.mark.asyncio
async def test_backend_tools_broaden_empty_restaurant_search_to_authorised_places() -> None:
    source_id = uuid4()
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/internal/v1/search":
            return httpx.Response(200, json={"items": [], "nextCursor": None, "hasNext": False})
        assert request.url.path == "/internal/v1/explore"
        assert b'"PLACE"' in request.content
        return httpx.Response(
            200,
            json={
                "items": [
                    {"sourceType": "PLACE", "sourceId": str(source_id), "title": "Kitchen", "snippet": "Orchard"}
                ],
                "nextCursor": None,
                "hasNext": False,
            },
        )

    raw = httpx.AsyncClient(base_url="http://backend.test", transport=httpx.MockTransport(handler))
    client = BackendToolClient(
        client=raw, settings=Settings(environment="test", backend_base_url="http://backend.test")
    )
    sources = await client.search(query="restaurants", delegation_token="delegated-user-token", timeout_seconds=1)
    await client.aclose()

    assert len(requests) == 2
    assert sources[0].source_type == "PLACE"
    assert sources[0].grounding_metadata["hasNext"] is False


def test_shared_deepseek_key_configures_chat_provider(monkeypatch) -> None:
    monkeypatch.delenv("CHAT_AGENT_LLM_API_KEY", raising=False)
    monkeypatch.setenv("CHAT_AGENT_LLM_ENABLED", "true")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")

    resolved = Settings(environment="test")

    assert resolved.llm_api_key is not None
    assert resolved.llm_api_key.get_secret_value() == "test-deepseek-key"

    with TestClient(create_app(settings=resolved, backend_tool_client=FakeBackendTools())) as client:  # type: ignore[arg-type]
        assert client.app.state.llm_client is not None
        assert client.get("/health/ready").json() == {
            "status": "ready",
            "llmEnabled": True,
            "llmConfigured": True,
            "llmProviderHost": "api.deepseek.com",
            "llmModel": "deepseek-v4-pro",
            "llmThinkingEnabled": False,
            "llmTemperature": 1.0,
            "llmMaxOutputTokens": 800,
        }


def test_enabled_llm_without_api_key_fails_fast(monkeypatch) -> None:
    monkeypatch.delenv("CHAT_AGENT_LLM_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(ValidationError, match="an LLM API key is required"):
        Settings(environment="local", llm_enabled=True, _env_file=None)
