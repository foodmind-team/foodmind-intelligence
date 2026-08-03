from conftest import AGENT_FIXTURES, load_json
from fastapi.testclient import TestClient
from pydantic import SecretStr

from recommendation_agent.config.settings import Settings
from recommendation_agent.main import create_app


def test_oversized_body_is_rejected_before_json_parsing() -> None:
    settings = Settings(app_env="test", internal_service_token=SecretStr("test-token"), max_request_bytes=1024)
    with TestClient(create_app(settings=settings)) as client:
        response = client.post(
            "/internal/v1/recommendations/generate",
            content=b"x" * 1025,
            headers={"Authorization": "Bearer test-token", "Content-Type": "application/json"},
        )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "REQUEST_TOO_LARGE"


def test_deployment_candidate_cap_is_enforced() -> None:
    settings = Settings(app_env="test", internal_service_token=SecretStr("test-token"), max_candidates=2)
    with TestClient(create_app(settings=settings)) as client:
        response = client.post(
            "/internal/v1/recommendations/generate",
            json=load_json(AGENT_FIXTURES / "valid-normal.json"),
            headers={"Authorization": "Bearer test-token"},
        )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "REQUEST_TOO_LARGE"


def test_unsafe_correlation_is_replaced() -> None:
    settings = Settings(app_env="test", internal_service_token=SecretStr("test-token"))
    with TestClient(create_app(settings=settings)) as client:
        response = client.get("/health/live", headers={"X-Request-ID": "bad\nlog"})
    assert response.status_code == 200
    assert response.headers["x-request-id"] != "bad\nlog"
    assert len(response.headers["x-request-id"]) == 32
