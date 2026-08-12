from fastapi.testclient import TestClient
from pydantic import SecretStr

from recommendation_agent.config.settings import Settings
from recommendation_agent.main import create_app


def test_production_disables_interactive_docs_and_keeps_health_safe() -> None:
    settings = Settings(
        app_env="production",
        internal_service_token=SecretStr("production-agent-token-value"),
        inference_service_token=SecretStr("production-inference-token-value"),
        inference_base_url="https://inference.internal",
        llm_enabled=False,
    )
    with TestClient(create_app(settings=settings)) as client:
        assert client.get("/docs").status_code == 404
        assert client.get("/redoc").status_code == 404
        assert client.get("/openapi.json").status_code == 404
        live = client.get("/health/live")
        ready = client.get("/health/ready")
    rendered = live.text + ready.text
    assert live.status_code == 200
    assert ready.status_code == 503
    assert ready.json()["checks"]["inference"] is False
    assert "production-agent-token-value" not in rendered
    assert "inference.internal" not in rendered
