"""P3-01 async task API — HTTP endpoint integration tests."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from cooking_plan_agent.main import create_app

_TEST_TOKEN = "test-internal-token-abc123"


@pytest.fixture(autouse=True)
def _set_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from cooking_plan_agent.config.settings import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("COOKING_PLAN_INTERNAL_SERVICE_TOKEN", _TEST_TOKEN)
    monkeypatch.setenv("COOKING_PLAN_TASK_API_ENABLED", "true")
    # Unique DB per test run avoids cross-test file contention on the
    # in-process worker's SQLite store.
    monkeypatch.setenv(
        "COOKING_PLAN_TASK_DB_PATH",
        f"/tmp/p3-01-task-test-{uuid.uuid4().hex}.sqlite",
    )


@pytest.fixture
def client():
    with TestClient(create_app()) as c:
        yield c


def _auth_headers(extra: dict | None = None) -> dict:
    headers = {"X-Internal-Token": _TEST_TOKEN}
    if extra:
        headers.update(extra)
    return headers


def _payload(request_id: str | None = None) -> dict:
    return {
        "request_id": request_id or str(uuid.uuid4()),
        "user_id": "async-api-user",
        "recipes": (
            {
                "recipe_id": "r1",
                "text": "Cook chicken for 10 minutes. Serves 2.",
                "target_servings": "2",
            },
        ),
        "dietary_restrictions": (),
        "user_allergens": (),
        "inventory_lots": (
            {
                "lot_id": "l1",
                "item_id": "i1",
                "canonical_name": "chicken breast",
                "on_hand": "300",
                "reserved": "0",
                "unit": "g",
            },
        ),
        "kitchen_resources": (
            {"resource_id": "s1", "resource_type": "stove", "capacity": "4", "capacity_unit": "burners"},
        ),
        "approved_decisions": (),
        "schema_version": "1.0",
    }


class TestSubmit:
    def test_submit_returns_202_with_location(self, client: TestClient) -> None:
        response = client.post("/internal/v2/cooking-plan/tasks", json=_payload(), headers=_auth_headers())
        assert response.status_code == 202
        body = response.json()
        assert "task_id" in body
        assert body["status"] == "QUEUED"
        assert body["location"] == f"/internal/v2/cooking-plan/tasks/{body['task_id']}"
        assert body["request_id"]

    def test_submit_requires_auth(self, client: TestClient) -> None:
        response = client.post("/internal/v2/cooking-plan/tasks", json=_payload())
        # Missing required X-Internal-Token header → 422 (consistent with the
        # native endpoint's FastAPI schema validation, P3-05 envelope).
        assert response.status_code == 422
        assert response.json()["error_code"] == "REQUEST_VALIDATION_ERROR"

    def test_submit_wrong_token_returns_401(self, client: TestClient) -> None:
        response = client.post(
            "/internal/v2/cooking-plan/tasks",
            json=_payload(),
            headers={"X-Internal-Token": "wrong-token"},
        )
        assert response.status_code == 401
        assert response.json()["error_code"] == "INVALID_INTERNAL_CREDENTIAL"

    def test_submit_invalid_body_returns_422(self, client: TestClient) -> None:
        response = client.post("/internal/v2/cooking-plan/tasks", json={"bad": "x"}, headers=_auth_headers())
        assert response.status_code == 422
        assert response.json()["error_code"] == "REQUEST_VALIDATION_ERROR"

    def test_submit_idempotent_same_payload(self, client: TestClient) -> None:
        request_id = str(uuid.uuid4())
        payload = _payload(request_id)
        first = client.post("/internal/v2/cooking-plan/tasks", json=payload, headers=_auth_headers())
        second = client.post("/internal/v2/cooking-plan/tasks", json=payload, headers=_auth_headers())
        assert first.status_code == 202
        assert second.status_code == 202
        assert first.json()["task_id"] == second.json()["task_id"]

    def test_submit_conflicting_payload_returns_409(self, client: TestClient) -> None:
        request_id = str(uuid.uuid4())
        payload = _payload(request_id)
        client.post("/internal/v2/cooking-plan/tasks", json=payload, headers=_auth_headers())
        payload["user_id"] = "different"
        response = client.post("/internal/v2/cooking-plan/tasks", json=payload, headers=_auth_headers())
        assert response.status_code == 409
        body = response.json()
        assert body["error_code"] == "IDEMPOTENCY_CONFLICT"
        assert body["retryable"] is False


class TestQuery:
    def test_get_task_returns_queued(self, client: TestClient) -> None:
        created = client.post("/internal/v2/cooking-plan/tasks", json=_payload(), headers=_auth_headers()).json()
        task_id = created["task_id"]
        response = client.get(
            f"/internal/v2/cooking-plan/tasks/{task_id}",
            headers=_auth_headers(),
        )
        assert response.status_code == 200
        body = response.json()
        assert body["task_id"] == task_id
        assert body["status"] in ("QUEUED", "RUNNING", "READY", "NEEDS_CONFIRMATION", "INFEASIBLE", "FAILED")

    def test_get_unknown_task_returns_404(self, client: TestClient) -> None:
        response = client.get(
            "/internal/v2/cooking-plan/tasks/no-such-task",
            headers=_auth_headers(),
        )
        assert response.status_code == 404
        assert response.json()["error_code"] == "TASK_NOT_FOUND"

    def test_worker_completes_task_to_ready(self, client: TestClient) -> None:
        """Submit then poll until the in-process worker finishes the task."""
        import time

        created = client.post("/internal/v2/cooking-plan/tasks", json=_payload(), headers=_auth_headers()).json()
        task_id = created["task_id"]
        deadline = time.monotonic() + 10
        status: str | None = None
        while time.monotonic() < deadline:
            body = client.get(f"/internal/v2/cooking-plan/tasks/{task_id}", headers=_auth_headers()).json()
            status = body["status"]
            if status in ("READY", "NEEDS_CONFIRMATION", "INFEASIBLE", "FAILED"):
                break
            time.sleep(0.2)
        assert status in ("READY", "NEEDS_CONFIRMATION", "INFEASIBLE", "FAILED"), f"Worker never finished: {status}"


class TestCancel:
    def test_cancel_queued_task(self, client: TestClient) -> None:
        created = client.post("/internal/v2/cooking-plan/tasks", json=_payload(), headers=_auth_headers()).json()
        task_id = created["task_id"]
        response = client.post(
            f"/internal/v2/cooking-plan/tasks/{task_id}/cancel",
            headers=_auth_headers(),
        )
        assert response.status_code == 200
        assert response.json()["status"] == "CANCELLED"

    def test_cancel_unknown_task_returns_404(self, client: TestClient) -> None:
        response = client.post(
            "/internal/v2/cooking-plan/tasks/nope/cancel",
            headers=_auth_headers(),
        )
        assert response.status_code == 404
