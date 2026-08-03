import copy

import httpx
from conftest import AGENT_FIXTURES, GOLDEN_FIXTURES, REPOSITORY_ROOT, load_json
from fastapi.testclient import TestClient
from pydantic import SecretStr

from recommendation_agent.config.settings import Settings
from recommendation_agent.main import create_app

INFERENCE_FIXTURE = (
    REPOSITORY_ROOT / "contracts/internal/inference/recommendation/v1/consumer-fixtures/valid-hybrid.json"
)


def test_production_workflow_matches_golden_response() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=INFERENCE_FIXTURE.read_bytes())

    raw = httpx.AsyncClient(base_url="http://inference.test", transport=httpx.MockTransport(handler))
    settings = Settings(
        app_env="test",
        internal_service_token=SecretStr("golden-agent-token"),
        inference_service_token=SecretStr("golden-inference-token"),
    )
    with TestClient(create_app(settings=settings, inference_http_client=raw)) as client:
        assert client.get("/health/ready").status_code == 200
        response = client.post(
            "/internal/v1/recommendations/generate",
            json=load_json(AGENT_FIXTURES / "valid-normal.json"),
            headers={"Authorization": "Bearer golden-agent-token"},
        )
    assert response.status_code == 200
    actual = response.json()
    expected = load_json(GOLDEN_FIXTURES / "expected-agent-response.json")
    actual["agentTraceId"] = expected["agentTraceId"]
    assert actual == expected


def test_same_fixture_has_deterministic_semantics_except_agent_trace() -> None:
    response_body = INFERENCE_FIXTURE.read_bytes()
    raw = httpx.AsyncClient(
        base_url="http://inference.test",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=response_body)),
    )
    settings = Settings(app_env="test", internal_service_token=SecretStr("repeat-token"))
    with TestClient(create_app(settings=settings, inference_http_client=raw)) as client:
        values = [
            client.post(
                "/internal/v1/recommendations/generate",
                json=copy.deepcopy(load_json(AGENT_FIXTURES / "valid-normal.json")),
                headers={"Authorization": "Bearer repeat-token"},
            ).json()
            for _ in range(2)
        ]
    values[1]["agentTraceId"] = values[0]["agentTraceId"]
    assert values[0] == values[1]
