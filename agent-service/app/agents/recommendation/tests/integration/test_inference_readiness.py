import httpx
from fastapi.testclient import TestClient

from recommendation_agent.config.settings import Settings
from recommendation_agent.main import create_app


def test_one_lifespan_client_is_closed_and_readiness_is_safe() -> None:
    raw = httpx.AsyncClient(
        base_url="http://inference.test", transport=httpx.MockTransport(lambda _: httpx.Response(200))
    )
    with TestClient(
        create_app(settings=Settings(app_env="test"), inference_http_client=raw, install_default_workflow=False)
    ) as client:
        body = client.get("/health/ready").json()
        assert body["checks"]["inference"] is True
        assert "inference.test" not in repr(body)
        assert "token" not in repr(body).casefold()
        assert raw.is_closed is False
    assert raw.is_closed is True
