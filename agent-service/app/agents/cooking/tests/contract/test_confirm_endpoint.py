"""P5-4: 确认续答 API 契约测试。"""

import pytest
from fastapi.testclient import TestClient

from cooking_plan_agent.main import create_app

_TEST_TOKEN = "test-internal-token-abc123"


@pytest.fixture(autouse=True)
def _set_token(monkeypatch: pytest.MonkeyPatch) -> None:
    from cooking_plan_agent.config.settings import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("COOKING_PLAN_INTERNAL_SERVICE_TOKEN", _TEST_TOKEN)
    monkeypatch.setenv("COOKING_PLAN_LLM_ENABLED", "false")


@pytest.fixture
def client():
    with TestClient(create_app()) as c:
        yield c


def _auth_headers() -> dict[str, str]:
    return {"X-Internal-Token": _TEST_TOKEN}


def test_confirm_missing_token_is_rejected(client: TestClient) -> None:
    """缺 X-Internal-Token → 422（Header 必填，与既有 generate 端点一致）。"""
    response = client.post(
        "/internal/v1/agents/cooking-plan/plans/p1/confirm",
        json={"plan_id": "p1", "answers": []},
    )
    assert response.status_code == 422


def test_confirm_wrong_token_returns_401(client: TestClient) -> None:
    response = client.post(
        "/internal/v1/agents/cooking-plan/plans/p1/confirm",
        json={"plan_id": "p1", "answers": []},
        headers={"X-Internal-Token": "wrong-token"},
    )
    assert response.status_code == 401


def test_confirm_answers_required(client: TestClient) -> None:
    """answers 是必填字段 —— 缺失时 422（新端点契约）。"""
    response = client.post(
        "/internal/v1/agents/cooking-plan/plans/p1/confirm",
        json={"plan_id": "p1"},
        headers=_auth_headers(),
    )
    assert response.status_code == 422


def test_confirm_with_valid_body_returns_plan_response(client: TestClient) -> None:
    """无挂起对话（dialog 默认关闭）→ 服务返回 FAILED，而非崩溃/静默重跑。"""
    response = client.post(
        "/internal/v1/agents/cooking-plan/plans/p1/confirm",
        json={"plan_id": "p1", "answers": [{"question_id": "q1", "value": "opt1"}], "user_id": "u1"},
        headers=_auth_headers(),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "FAILED"


def test_confirm_openapi_path_registered(client: TestClient) -> None:
    """新端点出现在 OpenAPI 文档中（导出 check 的 REQUIRED_PATHS 不受影响）。"""
    schema = client.get("/openapi.json").json()
    paths = schema.get("paths", {})
    assert "/internal/v1/agents/cooking-plan/plans/{plan_id}/confirm" in paths
