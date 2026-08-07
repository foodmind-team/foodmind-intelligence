from uuid import uuid4

from fastapi.testclient import TestClient
from pydantic import SecretStr

from chat_agent.config.settings import Settings
from chat_agent.main import create_app


def request_payload(*, references: bool = False) -> dict[str, object]:
    payload: dict[str, object] = {
        "contractVersion": "chat-agent-v1",
        "requestId": str(uuid4()),
        "sessionId": str(uuid4()),
        "userMessageId": str(uuid4()),
        "traceId": str(uuid4()),
        "message": "Summarise this item" if references else "Hello",
        "delegationToken": None,
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


def test_navigation_fallback_matches_backend_contract() -> None:
    settings = Settings(environment="test", internal_service_token=SecretStr("test-chat-token"), llm_enabled=False)
    with TestClient(create_app(settings=settings)) as client:
        response = client.post(
            "/internal/v1/chat/generate",
            headers={"Authorization": "Bearer test-chat-token"},
            json=request_payload(),
        )
    assert response.status_code == 200
    assert response.json()["status"] == "SUCCEEDED"
    assert response.json()["route"] == "NAVIGATION"
    assert response.json()["responseStatus"] == "FALLBACK_SUCCEEDED"


def test_grounded_fallback_cites_exact_reference() -> None:
    settings = Settings(environment="test", internal_service_token=SecretStr("test-chat-token"), llm_enabled=False)
    payload = request_payload(references=True)
    with TestClient(create_app(settings=settings)) as client:
        response = client.post(
            "/internal/v1/chat/generate",
            headers={"Authorization": "Bearer test-chat-token"},
            json=payload,
        )
    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "SUMMARY"
    assert body["sources"][0]["sourceId"] == payload["sharedReferences"][0]["sourceId"]


def test_authentication_is_required() -> None:
    settings = Settings(environment="test", internal_service_token=SecretStr("test-chat-token"), llm_enabled=False)
    with TestClient(create_app(settings=settings)) as client:
        response = client.post("/internal/v1/chat/generate", json=request_payload())
    assert response.status_code == 401
    assert response.json()["error_code"] == "MISSING_AUTHORIZATION_HEADER"

