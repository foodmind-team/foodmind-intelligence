import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from conftest import AGENT_FIXTURES, REPOSITORY_ROOT
from pydantic import SecretStr

from recommendation_agent.clients.inference import RecommendationInferenceHttpClient, command_from_agent_request
from recommendation_agent.config.settings import Settings
from recommendation_agent.domain.errors import AgentError, ErrorCode
from recommendation_agent.schemas.agent_v2 import AgentRequest
from recommendation_agent.time.budget import DeadlineBudget

INFERENCE_FIXTURES = REPOSITORY_ROOT / "contracts/internal/inference/recommendation/v1/consumer-fixtures"


@dataclass
class FakeClock:
    now: datetime
    tick: float = 10.0

    def utc_now(self) -> datetime:
        return self.now

    def monotonic(self) -> float:
        return self.tick


def _request() -> AgentRequest:
    return AgentRequest.model_validate_json((AGENT_FIXTURES / "valid-normal.json").read_text(encoding="utf-8"))


def _budget(clock: FakeClock) -> DeadlineBudget:
    return DeadlineBudget.from_absolute(clock.now + timedelta(seconds=2), clock=clock, minimum_seconds=0.1)


def _settings(**changes: Any) -> Settings:
    return Settings(
        app_env="test",
        internal_service_token=SecretStr("agent-test-token"),
        inference_service_token=SecretStr("inference-secret-canary"),
        inference_base_url="https://inference.internal",
        **changes,
    )


@pytest.mark.asyncio
async def test_valid_request_calls_once_with_frozen_headers_and_body() -> None:
    attempts: list[httpx.Request] = []
    response_body = (INFERENCE_FIXTURES / "valid-hybrid.json").read_bytes()

    async def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(200, content=response_body)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(base_url="https://inference.internal", transport=transport) as http_client:
        client = RecommendationInferenceHttpClient(client=http_client, settings=_settings())
        clock = FakeClock(datetime(2030, 1, 1, tzinfo=UTC))
        result = await client.score(command_from_agent_request(_request()), budget=_budget(clock))

    assert len(attempts) == 1
    sent = attempts[0]
    assert sent.headers["authorization"] == "Bearer inference-secret-canary"
    assert sent.headers["x-inference-contract-version"] == "recommendation-inference-v1"
    body = json.loads(sent.content)
    assert body["deadlineAt"] == "2030-01-01T00:00:02Z"
    assert "novelty" not in body["candidates"][0]["evidence"]
    assert len(result.candidates) == 5
    assert client.metrics.calls_total == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        ("connection", ErrorCode.INFERENCE_CONNECTION_FAILED),
        ("timeout", ErrorCode.INFERENCE_TIMEOUT),
        ("http", ErrorCode.INFERENCE_HTTP_ERROR),
        ("malformed", ErrorCode.INFERENCE_MALFORMED_RESPONSE),
    ],
)
async def test_transport_failures_never_retry(failure: str, expected: ErrorCode) -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if failure == "connection":
            raise httpx.ConnectError("sensitive-origin-canary", request=request)
        if failure == "timeout":
            raise httpx.ReadTimeout("sensitive-timeout-canary", request=request)
        if failure == "http":
            return httpx.Response(500, content=b"sensitive-upstream-body-canary")
        return httpx.Response(200, content=b"not-json")

    async with httpx.AsyncClient(base_url="https://inference.internal", transport=httpx.MockTransport(handler)) as raw:
        client = RecommendationInferenceHttpClient(client=raw, settings=_settings())
        clock = FakeClock(datetime(2030, 1, 1, tzinfo=UTC))
        with pytest.raises(AgentError) as captured:
            await client.score(command_from_agent_request(_request()), budget=_budget(clock))
    assert captured.value.code is expected
    assert attempts == 1


@pytest.mark.asyncio
async def test_oversized_response_is_rejected_before_parse() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 1025)

    async with httpx.AsyncClient(base_url="https://inference.internal", transport=httpx.MockTransport(handler)) as raw:
        client = RecommendationInferenceHttpClient(client=raw, settings=_settings(inference_max_response_bytes=1024))
        clock = FakeClock(datetime(2030, 1, 1, tzinfo=UTC))
        with pytest.raises(AgentError) as captured:
            await client.score(command_from_agent_request(_request()), budget=_budget(clock))
    assert captured.value.code is ErrorCode.INFERENCE_RESPONSE_TOO_LARGE
