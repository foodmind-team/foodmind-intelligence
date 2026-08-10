from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from cooking_plan_agent.config.settings import get_settings
from cooking_plan_agent.main import create_app

TOKEN = "test-internal-token-abc123"
URL = "/internal/v1/agents/cooking-plan/recipe-imports/parse"


@pytest.fixture(autouse=True)
def _settings(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("COOKING_PLAN_INTERNAL_SERVICE_TOKEN", TOKEN)
    monkeypatch.setenv("COOKING_PLAN_LLM_ENABLED", "false")


@pytest.fixture
def client():
    with TestClient(create_app()) as test_client:
        yield test_client


def _headers() -> dict[str, str]:
    return {"X-Internal-Token": TOKEN, "X-Request-ID": "recipe-import-contract"}


def _text() -> str:
    return """Recipe: Mushroom Toast
2 servings
Ingredients:
4 slices bread
200 g mushrooms
Steps:
1. Toast the bread.
2. Cook the mushrooms for 8 minutes and serve.
---
Recipe: Berry Bowl
Ingredients:
200 g berries
100 g yogurt
Steps:
1. Combine the berries and yogurt.
"""


def test_returns_multiple_drafts_and_structured_question(client: TestClient) -> None:
    response = client.post(URL, headers=_headers(), json={"request_id": "req-1", "text": _text()})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "NEEDS_CLARIFICATION"
    assert [draft["name"] for draft in body["drafts"]] == ["Mushroom Toast", "Berry Bowl"]
    assert body["questions"] == [
        {
            "question_id": "dish-2:servings",
            "draft_id": "dish-2",
            "field_path": "servings",
            "prompt": "How many servings does Berry Bowl make? Enter a whole number from 1 to 50.",
            "response_type": "TEXT",
            "required": True,
            "suggested_value": None,
        }
    ]
    assert response.headers["X-Request-ID"] == "recipe-import-contract"


def test_answers_continue_to_ready(client: TestClient) -> None:
    response = client.post(
        URL,
        headers=_headers(),
        json={
            "request_id": "req-2",
            "text": _text(),
            "answers": [{"question_id": "dish-2:servings", "value": "3"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "READY"
    assert [draft["servings"] for draft in response.json()["drafts"]] == [2, 3]


def test_persisted_snapshot_continues_without_reparsing(client: TestClient) -> None:
    first = client.post(URL, headers=_headers(), json={"request_id": "req-snapshot", "text": _text()})
    assert first.status_code == 200
    first_body = first.json()

    response = client.post(
        URL,
        headers=_headers(),
        json={
            "request_id": "req-snapshot",
            "text": "This text is intentionally not parseable as the original recipes.",
            "answers": [{"question_id": "dish-2:servings", "value": "3"}],
            "drafts": first_body["drafts"],
            "questions": first_body["questions"],
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "READY"
    assert [draft["name"] for draft in response.json()["drafts"]] == ["Mushroom Toast", "Berry Bowl"]
    assert [draft["servings"] for draft in response.json()["drafts"]] == [2, 3]


def test_mixed_language_input_is_accepted(client: TestClient) -> None:
    response = client.post(URL, headers=_headers(), json={"request_id": "req-3", "text": "Make 番茄 pasta"})

    assert response.status_code == 200
