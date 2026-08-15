from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from fastapi.testclient import TestClient
from pydantic import SecretStr

from chat_agent.config.settings import Settings
from chat_agent.main import create_app


class EmptyBackendTools:
    async def search(self, **kwargs: Any) -> tuple[()]:
        return ()

    async def explore(self, **kwargs: Any) -> tuple[()]:
        return ()

    async def resolve(self, **kwargs: Any) -> tuple[()]:
        return ()


class RecordingLLM:
    def __init__(self, answer: str = "Context-aware answer") -> None:
        self.answer = answer
        self.calls: list[list[dict[str, str]]] = []

    async def chat(self, messages: list[dict[str, str]], *, timeout_seconds: float | None = None) -> str:
        self.calls.append(messages)
        return self.answer


def payload(message: str, *, recent_turns: list[dict[str, str]] | None = None) -> dict[str, object]:
    return {
        "contractVersion": "chat-agent-v1",
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


def post(body: dict[str, object], llm: RecordingLLM | None = None) -> dict[str, object]:
    with TestClient(
        create_app(
            settings=settings(),
            llm_client=llm,
            backend_tool_client=EmptyBackendTools(),  # type: ignore[arg-type]
        )
    ) as client:
        response = client.post(
            "/internal/v1/chat/generate",
            headers={"Authorization": "Bearer test-chat-token"},
            json=body,
        )
    assert response.status_code == 200
    return response.json()


def test_ambiguous_message_without_history_returns_clickable_clarification() -> None:
    body = post(payload("What about that?"))

    assert body["route"] == "NAVIGATION"
    assert body["responseStatus"] == "SUCCEEDED"
    assert len(body["suggestedQuestions"]) == 3
    assert body["suggestedDestinations"] == ["EXPLORE", "COOKING_PLANS"]


def test_short_followup_uses_recent_turns_in_llm_context() -> None:
    llm = RecordingLLM("Tempeh usually contains more protein than tofu by weight.")
    body = post(
        payload(
            "What about tempeh?",
            recent_turns=[
                {"role": "USER", "content": "How much protein is in tofu?"},
                {"role": "ASSISTANT", "content": "Tofu protein varies by firmness and brand."},
            ],
        ),
        llm,
    )

    assert body["route"] == "SUMMARY"
    assert body["responseStatus"] == "SUCCEEDED"
    assert body["answer"].startswith("Tempeh")
    assert [message["role"] for message in llm.calls[0]] == ["system", "user", "assistant", "user"]
    assert "How much protein is in tofu?" in llm.calls[0][1]["content"]


def test_complete_can_i_question_is_answered_without_unnecessary_clarification() -> None:
    llm = RecordingLLM("Yes. Freeze tofu after draining it and use it within a safe storage window.")
    body = post(payload("Can I freeze tofu safely?"), llm)

    assert body["route"] == "SUMMARY"
    assert body["responseStatus"] == "SUCCEEDED"
    assert body["answer"].startswith("Yes")
    assert len(llm.calls) == 1


def test_natural_record_question_routes_to_search_without_full_description() -> None:
    llm = RecordingLLM("I could not find a matching authorised record.")
    body = post(payload("What did I eat yesterday?"), llm)

    assert body["route"] == "SEARCH"
    assert body["responseStatus"] == "SUCCEEDED"
    assert len(llm.calls) == 1


def test_natural_chinese_record_question_routes_to_search() -> None:
    llm = RecordingLLM("我没有找到匹配的已授权记录。")
    body = post(payload("昨天吃了什么？"), llm)

    assert body["route"] == "SEARCH"
    assert body["responseStatus"] == "SUCCEEDED"
    assert len(llm.calls) == 1


def test_contextual_any_ideas_followup_uses_history() -> None:
    llm = RecordingLLM("Try a tofu and vegetable stir-fry.")
    body = post(
        payload(
            "Any ideas?",
            recent_turns=[
                {"role": "USER", "content": "What can I cook with tofu and vegetables?"},
                {"role": "ASSISTANT", "content": "You have several quick options."},
            ],
        ),
        llm,
    )

    assert body["route"] == "SEARCH"
    assert len(llm.calls) == 1


def test_clearly_unrelated_question_is_refused_without_calling_llm() -> None:
    llm = RecordingLLM()
    body = post(payload("What is tomorrow's weather forecast?"), llm)

    assert body["route"] == "OUT_OF_SCOPE"
    assert body["responseStatus"] == "UNSUPPORTED"
    assert body["sources"] == []
    assert llm.calls == []


def test_unrelated_navigation_wording_is_refused_before_navigation_routing() -> None:
    llm = RecordingLLM()
    body = post(payload("Where can I see the football score?"), llm)

    assert body["route"] == "OUT_OF_SCOPE"
    assert body["responseStatus"] == "UNSUPPORTED"
    assert llm.calls == []


def test_write_request_is_redirected_to_dedicated_workflow() -> None:
    body = post(payload("Add this recipe to my saved recipes for me."))

    assert body["route"] == "OUT_OF_SCOPE"
    assert body["responseStatus"] == "UNSUPPORTED"
    assert "read-only" in body["answer"]
    assert "SAVED_RECIPES" in body["suggestedDestinations"]


def test_harmful_food_request_is_refused_with_safe_alternative() -> None:
    body = post(payload("How can I hide an allergen and make someone sick?"))

    assert body["route"] == "OUT_OF_SCOPE"
    assert body["responseStatus"] == "UNSUPPORTED"
    assert "allergen" in body["answer"].lower()
    assert body["suggestedDestinations"] == ["INVENTORY"]
