"""P4-04 task SSE progress endpoint — HTTP contract tests.

Covers the full flow (submit -> SSE progress sequence -> terminal `done` ->
stream close), Last-Event-ID reconnection, polling fallback, auth isolation,
unknown-task 404, and the feature-flag 404 when SSE is disabled.
"""

from __future__ import annotations

import json
import os
import time
import uuid

import pytest
from fastapi.testclient import TestClient

# main 的模块级 import 会触发 get_settings()（internal_service_token 必填）。
# 与 tests/security/test_log_redaction.py 一致：在导入 main 前先注入测试 token。
os.environ.setdefault("COOKING_PLAN_INTERNAL_SERVICE_TOKEN", "test-internal-token-abc123")

from cooking_plan_agent.main import create_app  # noqa: E402

_TEST_TOKEN = "test-internal-token-abc123"
_EVENTS_URL = "/internal/v2/cooking-plan/tasks/{task_id}/events"
_TERMINAL_STATUSES = ("READY", "NEEDS_CONFIRMATION", "INFEASIBLE", "FAILED")


@pytest.fixture(autouse=True)
def _set_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from cooking_plan_agent.config.settings import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("COOKING_PLAN_INTERNAL_SERVICE_TOKEN", _TEST_TOKEN)
    monkeypatch.setenv("COOKING_PLAN_TASK_API_ENABLED", "true")
    monkeypatch.setenv(
        "COOKING_PLAN_TASK_DB_PATH",
        f"/tmp/p4-04-sse-test-{uuid.uuid4().hex}.sqlite",
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
        "user_id": "sse-api-user",
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


def _parse_sse(body: str) -> list[dict]:
    """Parse SSE frames into [{"id", "event", "data"}, ...]; comments skipped."""
    frames = []
    for raw in body.split("\n\n"):
        raw = raw.strip()
        if not raw or raw.startswith(":"):
            continue
        frame: dict = {"data": None}
        for line in raw.splitlines():
            if line.startswith("id:"):
                frame["id"] = int(line[3:].strip())
            elif line.startswith("event:"):
                frame["event"] = line[6:].strip()
            elif line.startswith("data:"):
                frame["data"] = json.loads(line[5:].strip())
        frames.append(frame)
    return frames


def _stream_until_done(client: TestClient, task_id: str, headers: dict | None = None) -> str:
    """Open the SSE stream and collect frames until the terminal `done`."""
    body = ""
    deadline = time.monotonic() + 15
    with client.stream("GET", _EVENTS_URL.format(task_id=task_id), headers=headers or _auth_headers()) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        for chunk in response.iter_text():
            body += chunk
            if "event: done" in body:
                break
            assert time.monotonic() < deadline, "timed out waiting for the done event"
    return body


def _wait_terminal(client: TestClient, task_id: str) -> dict:
    """Poll GET until the task reaches a terminal status (fallback path)."""
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        body = client.get(f"/internal/v2/cooking-plan/tasks/{task_id}", headers=_auth_headers()).json()
        if body["status"] in _TERMINAL_STATUSES:
            return body
        time.sleep(0.2)
    raise AssertionError(f"task {task_id} never reached a terminal status")


class TestSseStream:
    def test_submit_then_stream_progress_to_done(self, client: TestClient) -> None:
        created = client.post("/internal/v2/cooking-plan/tasks", json=_payload(), headers=_auth_headers()).json()
        task_id = created["task_id"]

        body = _stream_until_done(client, task_id)
        frames = _parse_sse(body)
        assert frames, "expected at least one SSE frame"

        # event ids are monotonic and the terminal frame closes the stream.
        ids = [f["id"] for f in frames]
        assert ids == sorted(ids) and len(ids) == len(set(ids)), f"ids not monotonic: {ids}"
        assert frames[-1]["event"] == "done"
        assert frames[-1]["data"]["status"] in _TERMINAL_STATUSES
        assert frames[-1]["data"]["event_id"] == frames[-1]["id"]

        # every frame carries the task identity and the id/data event_id agree.
        for frame in frames:
            assert frame["data"]["task_id"] == task_id
            assert frame["data"]["event_id"] == frame["id"]

        # non-terminal frames are progress events.
        for frame in frames[:-1]:
            assert frame["event"] == "progress"

    def test_reconnect_with_last_event_id_resumes_after_missed_events(self, client: TestClient) -> None:
        created = client.post("/internal/v2/cooking-plan/tasks", json=_payload(), headers=_auth_headers()).json()
        task_id = created["task_id"]

        first_body = _stream_until_done(client, task_id)
        first_frames = _parse_sse(first_body)
        terminal = first_frames[-1]
        assert terminal["event"] == "done"

        # Reconnect with an older Last-Event-ID: only newer events replay.
        stale_id = terminal["id"] - 1
        replayed = _parse_sse(
            _stream_until_done(client, task_id, headers=_auth_headers({"Last-Event-ID": str(stale_id)}))
        )
        assert [f["id"] for f in replayed] == [terminal["id"]]
        assert replayed[-1]["event"] == "done"

        # Reconnecting at (or past) the terminal id still reports done.
        replayed = _parse_sse(
            _stream_until_done(client, task_id, headers=_auth_headers({"Last-Event-ID": str(terminal["id"])}))
        )
        assert replayed[-1]["event"] == "done"
        assert replayed[-1]["data"]["status"] in _TERMINAL_STATUSES

    def test_polling_fallback_still_returns_terminal_state(self, client: TestClient) -> None:
        created = client.post("/internal/v2/cooking-plan/tasks", json=_payload(), headers=_auth_headers()).json()
        task_id = created["task_id"]

        _stream_until_done(client, task_id)
        # After the stream closes, GET still answers with the final state.
        terminal = _wait_terminal(client, task_id)
        assert terminal["status"] in _TERMINAL_STATUSES


class TestSseNegative:
    def test_sse_requires_auth(self, client: TestClient) -> None:
        created = client.post("/internal/v2/cooking-plan/tasks", json=_payload(), headers=_auth_headers()).json()
        with client.stream("GET", _EVENTS_URL.format(task_id=created["task_id"])) as response:
            # Missing required X-Internal-Token header -> 422 (schema validation,
            # consistent with the other task endpoints).
            assert response.status_code == 422
            assert json.loads(response.read())["error_code"] == "REQUEST_VALIDATION_ERROR"

    def test_sse_wrong_token_returns_401(self, client: TestClient) -> None:
        created = client.post("/internal/v2/cooking-plan/tasks", json=_payload(), headers=_auth_headers()).json()
        with client.stream(
            "GET",
            _EVENTS_URL.format(task_id=created["task_id"]),
            headers={"X-Internal-Token": "wrong-token"},
        ) as response:
            assert response.status_code == 401
            assert json.loads(response.read())["error_code"] == "INVALID_INTERNAL_CREDENTIAL"

    def test_sse_unknown_task_returns_404(self, client: TestClient) -> None:
        with client.stream(
            "GET",
            _EVENTS_URL.format(task_id="no-such-task"),
            headers=_auth_headers(),
        ) as response:
            assert response.status_code == 404
            assert json.loads(response.read())["error_code"] == "TASK_NOT_FOUND"


@pytest.fixture
def client_sse_disabled(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from cooking_plan_agent.config.settings import get_settings

    monkeypatch.setenv("COOKING_PLAN_TASK_SSE_ENABLED", "false")
    get_settings.cache_clear()
    with TestClient(create_app()) as c:
        yield c


class TestSseFeatureFlag:
    def test_disabled_returns_404_but_polling_unaffected(self, client_sse_disabled: TestClient) -> None:
        created = client_sse_disabled.post(
            "/internal/v2/cooking-plan/tasks", json=_payload(), headers=_auth_headers()
        ).json()
        task_id = created["task_id"]

        with client_sse_disabled.stream(
            "GET", _EVENTS_URL.format(task_id=task_id), headers=_auth_headers()
        ) as response:
            assert response.status_code == 404
            assert json.loads(response.read())["error_code"] == "SSE_DISABLED"

        # Polling endpoints remain available with the flag off.
        terminal = _wait_terminal(client_sse_disabled, task_id)
        assert terminal["status"] in _TERMINAL_STATUSES
