from pydantic import SecretStr

from recommendation_agent.config.settings import Settings
from recommendation_agent.main import create_app


def test_openapi_exposes_only_private_agent_and_health_routes() -> None:
    app = create_app(settings=Settings(app_env="test", internal_service_token=SecretStr("test-token")))
    assert set(app.openapi()["paths"]) == {
        "/health/live",
        "/health/ready",
        "/internal/v1/recommendations/generate",
    }
