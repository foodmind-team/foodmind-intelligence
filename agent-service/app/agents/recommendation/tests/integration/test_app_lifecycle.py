import httpx
from fastapi.testclient import TestClient

from recommendation_agent.config.settings import Settings
from recommendation_agent.main import create_app


def test_liveness_and_foundation_readiness() -> None:
    inference = httpx.AsyncClient(
        base_url="http://inference.test",
        transport=httpx.MockTransport(lambda _: httpx.Response(503)),
    )
    with TestClient(
        create_app(
            settings=Settings(app_env="test"),
            inference_http_client=inference,
            install_default_workflow=False,
        )
    ) as client:
        assert client.get("/health/live").json() == {"status": "alive"}
        ready = client.get("/health/ready")
        assert ready.status_code == 503
        assert ready.json() == {
            "status": "not_ready",
            "checks": {
                "configuration": True,
                "inference": False,
                "policies": False,
                "workflow": False,
                "shutdown": True,
            },
        }
