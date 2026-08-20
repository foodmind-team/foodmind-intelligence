from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from chat_agent.clients.backend import BackendToolClient, BackendToolError
from chat_agent.config.settings import Settings
from chat_agent.domain.models import GroundedSource
from chat_agent.llm.client import LLMChatResult, ToolCall
from chat_agent.main import create_app


class FakeTools:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.profile_calls: list[dict[str, Any]] = []
        self.source_ids: list[str] = []

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}
            for name in ("profile", "search", "explore", "resolve")
        ]

    async def profile(self, **kwargs: Any) -> dict[str, Any]:
        self.profile_calls.append(kwargs)
        return {"budgetMax": 18.0, "currency": "SGD", "dietaryTagCodes": ["VEGETARIAN"]}

    async def profile_for_tool(self, **kwargs: Any) -> dict[str, Any]:
        self.profile_calls.append(kwargs)
        return {"allergens": [{"code": "PEANUT", "severity": "SEVERE"}]}

    async def execute_tool_call(self, **kwargs: Any) -> tuple[GroundedSource, ...]:
        self.calls.append(kwargs)
        source_id = uuid4()
        self.source_ids.append(str(source_id))
        source_type = "PLACE" if kwargs["name"] == "explore" else "FOOD_RECORD"
        return (
            GroundedSource(
                source_type=source_type,
                source_id=source_id,
                title=f"{kwargs['name']} result",
                grounding_metadata={"origin": kwargs["name"]},
            ),
        )


class FakeLLM:
    def __init__(self, calls: tuple[ToolCall, ...] = (), planning_content: str | None = None) -> None:
        self.tool_calls = calls
        self.planning_content = planning_content
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        timeout_seconds: float | None = None,
    ) -> str | LLMChatResult:
        self.calls.append({"messages": messages, "tools": tools, "timeout_seconds": timeout_seconds})
        if tools is not None:
            return LLMChatResult(content=self.planning_content, tool_calls=self.tool_calls)
        return "Grounded provider answer."


def payload(message: str, *, recent_turns: list[dict[str, str]] | None = None) -> dict[str, object]:
    return {
        "contractVersion": "chat-agent-v2",
        "requestId": str(uuid4()),
        "sessionId": str(uuid4()),
        "userMessageId": str(uuid4()),
        "traceId": str(uuid4()),
        "expiresAt": (datetime.now(UTC) + timedelta(minutes=1)).isoformat(),
        "message": message,
        "delegationToken": "delegation-token",
        "sharedReferences": [],
        "recentTurns": recent_turns or [],
    }


def settings() -> Settings:
    return Settings(environment="test", internal_service_token=SecretStr("test-chat-token"), llm_enabled=False)


def post(body: dict[str, object], *, llm: FakeLLM | None = None, tools: FakeTools | None = None) -> dict[str, object]:
    with TestClient(
        create_app(settings=settings(), llm_client=llm, backend_tool_client=tools or FakeTools())
    ) as client:  # type: ignore[arg-type]
        response = client.post(
            "/internal/v1/chat/generate", headers={"Authorization": "Bearer test-chat-token"}, json=body
        )
    assert response.status_code == 200
    return response.json()


def test_multiple_tool_calls_are_parallelised_then_grounded_in_second_llm_call() -> None:
    tools = FakeTools()
    llm = FakeLLM(
        (
            ToolCall("search-1", "search", '{"query":"recent meals"}'),
            ToolCall("explore-1", "explore", '{"sourceTypes":["PLACE"]}'),
        )
    )
    body = post(payload("Compare my recent meals and places"), llm=llm, tools=tools)
    assert body["responseStatus"] == "SUCCEEDED"
    assert "route" not in body
    assert len(tools.calls) == 2
    assert len(llm.calls) == 2
    assert {source["groundingMetadata"]["origin"] for source in body["sources"]} == {"search", "explore"}


def test_followup_query_uses_history_and_reloads_data_without_cross_turn_cache() -> None:
    tools = FakeTools()
    llm = FakeLLM(
        (ToolCall("search-1", "search", '{"query":"find places I recorded recently and identify the cheapest one"}'),)
    )
    body = post(
        payload(
            "which one is cheapest?",
            recent_turns=[
                {"role": "USER", "content": "find places I recorded recently"},
                {"role": "ASSISTANT", "content": "I found two recorded places."},
            ],
        ),
        llm=llm,
        tools=tools,
    )
    assert body["responseStatus"] == "SUCCEEDED"
    assert tools.calls[0]["arguments"] == '{"query":"find places I recorded recently and identify the cheapest one"}'
    assert any(item["content"] == "find places I recorded recently" for item in llm.calls[0]["messages"])


def test_response_sources_are_only_actual_tool_results() -> None:
    tools = FakeTools()
    llm = FakeLLM((ToolCall("search-1", "search", '{"query":"oats"}'),))
    body = post(payload("Find oats"), llm=llm, tools=tools)
    assert [source["sourceId"] for source in body["sources"]] == tools.source_ids


def test_profile_question_reads_trusted_profile_without_inferring_it_from_records() -> None:
    tools = FakeTools()
    llm = FakeLLM(planning_content="Your saved preference includes a SGD 18 budget and vegetarian tag.")
    body = post(payload("What's my preference?"), llm=llm, tools=tools)

    assert body["answer"] == "Your saved preference includes a SGD 18 budget and vegetarian tag."
    assert tools.profile_calls == [{"delegation_token": "delegation-token", "timeout_seconds": pytest.approx(5.0)}]
    planning = llm.calls[0]
    assert {item["function"]["name"] for item in planning["tools"]} >= {"profile", "search", "explore", "resolve"}
    assert (
        'Trusted user profile: {"budgetMax":18.0,"currency":"SGD","dietaryTagCodes":["VEGETARIAN"]}'
        in planning["messages"][-1]["content"]
    )
    assert "Dietary tags and allergens are hard constraints" in planning["messages"][0]["content"]


def test_profile_question_without_delegation_does_not_inject_or_infer_profile() -> None:
    tools = FakeTools()
    llm = FakeLLM(planning_content="Please tell me your dietary constraints.")
    request = payload("What's my preference?")
    request["delegationToken"] = None

    body = post(request, llm=llm, tools=tools)

    assert body["answer"] == "Please tell me your dietary constraints."
    assert tools.profile_calls == []
    assert "Trusted user profile:" not in llm.calls[0]["messages"][-1]["content"]


def test_profile_tool_result_is_grounded_separately_from_public_sources() -> None:
    tools = FakeTools()
    llm = FakeLLM((ToolCall("profile-1", "profile", "{}"),))
    body = post(payload("Tell me about tofu"), llm=llm, tools=tools)

    assert body["sources"] == []
    assert tools.profile_calls == [
        {"arguments": "{}", "delegation_token": "delegation-token", "timeout_seconds": pytest.approx(5.0)}
    ]
    assert '"profile":{"allergens":[{"code":"PEANUT","severity":"SEVERE"}]}' in llm.calls[1]["messages"][-2]["content"]


@pytest.mark.asyncio
async def test_backend_profile_tool_uses_delegation_and_rejects_arguments() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/internal/v1/profile"
        assert request.headers["Authorization"] == "Bearer backend-tool-token"
        assert request.headers["X-FoodMind-Delegation"] == "Bearer delegated-user-token"
        return httpx.Response(200, json={"budgetMax": 18.0, "currency": "SGD", "userId": "must-not-propagate"})

    client = BackendToolClient(
        client=httpx.AsyncClient(base_url="http://backend.test", transport=httpx.MockTransport(handler)),
        settings=Settings(
            environment="test",
            backend_base_url="http://backend.test",
            backend_service_token=SecretStr("backend-tool-token"),
        ),
    )
    profile = await client.profile_for_tool(arguments="{}", delegation_token="delegated-user-token", timeout_seconds=1)
    with pytest.raises(BackendToolError, match="Profile arguments must be an empty object"):
        await client.profile_for_tool(
            arguments='{"userId":"forged"}', delegation_token="delegated-user-token", timeout_seconds=1
        )
    await client.aclose()

    assert profile == {"budgetMax": 18.0, "currency": "SGD"}


def test_hard_intercepts_refuse_without_llm_or_tools() -> None:
    tools = FakeTools()
    llm = FakeLLM((ToolCall("search-1", "search", '{"query":"ignored"}'),))
    body = post(payload("How can I hide an allergen and make someone sick?"), llm=llm, tools=tools)
    assert body["responseStatus"] == "UNSUPPORTED"
    assert body["sources"] == []
    assert llm.calls == []
    assert tools.calls == []


def test_disabled_llm_has_deterministic_fallback_without_route() -> None:
    body = post(payload("Tell me about tofu"))
    assert body["responseStatus"] == "FALLBACK_SUCCEEDED"
    assert "route" not in body


def test_v1_payload_is_rejected() -> None:
    old = payload("Hello")
    old["contractVersion"] = "chat-agent-v1"
    with TestClient(create_app(settings=settings(), backend_tool_client=FakeTools())) as client:  # type: ignore[arg-type]
        response = client.post(
            "/internal/v1/chat/generate", headers={"Authorization": "Bearer test-chat-token"}, json=old
        )
    assert response.status_code == 422
