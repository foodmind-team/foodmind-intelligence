"""Run a redacted, fixture-only private Recommendation Agent smoke test."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import httpx
from fastapi.testclient import TestClient
from pydantic import SecretStr

from recommendation_agent.config.settings import Settings
from recommendation_agent.main import create_app
from recommendation_agent.schemas.agent_v2 import AgentResponse

REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
GOLDEN = REPOSITORY_ROOT / "artifacts/test-fixtures/recommendation/agent-golden-v2"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("fixture must be an object")
    return value


def fixture_smoke() -> None:
    logging.getLogger("httpx").setLevel(logging.WARNING)
    request_fixture = _load(GOLDEN / "request.json")
    inference_bytes = (GOLDEN / "inference-response.json").read_bytes()
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if request.headers.get("authorization") != "Bearer fixture-inference-token":
            return httpx.Response(401)
        if request.headers.get("x-inference-contract-version") != "recommendation-inference-v1":
            return httpx.Response(400)
        return httpx.Response(200, content=inference_bytes, headers={"Content-Type": "application/json"})

    inference_client = httpx.AsyncClient(
        base_url="http://fixture-inference.invalid",
        transport=httpx.MockTransport(handler),
    )
    settings = Settings(
        app_env="test",
        internal_service_token=SecretStr("fixture-agent-token"),
        inference_service_token=SecretStr("fixture-inference-token"),
    )
    with TestClient(create_app(settings=settings, inference_http_client=inference_client)) as client:
        ready = client.get("/health/ready")
        response = client.post(
            "/internal/v1/recommendations/generate",
            json=request_fixture,
            headers={
                "Authorization": "Bearer fixture-agent-token",
                "X-Request-ID": request_fixture["requestId"],
                "X-Trace-ID": request_fixture["traceId"],
            },
        )
    if ready.status_code != 200 or response.status_code != 200:
        raise RuntimeError("fixture smoke failed")
    parsed = AgentResponse.model_validate_json(response.content)
    if calls != 1 or not parsed.recommendations or len(parsed.recommendations) > 3:
        raise RuntimeError("fixture smoke invariant failed")
    print(
        "PASS fixture private smoke: ready=200 response=200 inferenceCalls=1 "
        f"results={len(parsed.recommendations)} contract={parsed.contract_version}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fixture",), required=True)
    args = parser.parse_args()
    if args.mode == "fixture":
        fixture_smoke()


if __name__ == "__main__":
    main()
