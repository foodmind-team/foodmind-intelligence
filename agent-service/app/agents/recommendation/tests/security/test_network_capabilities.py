from datetime import UTC, datetime, timedelta

import httpx
import pytest
from pydantic import SecretStr, ValidationError
from workflow_helpers import FakeClock, canonical_request

from recommendation_agent.clients.inference import RecommendationInferenceHttpClient, command_from_agent_request
from recommendation_agent.config.settings import Settings
from recommendation_agent.domain.errors import AgentError, ErrorCode
from recommendation_agent.time.budget import DeadlineBudget


def test_production_rejects_public_or_credentialed_inference_origins() -> None:
    for origin in ("https://example.com", "https://user:password@inference.internal", "file:///tmp/socket"):
        with pytest.raises(ValidationError):
            Settings(
                app_env="production",
                internal_service_token=SecretStr("production-agent-token-value"),
                inference_service_token=SecretStr("production-inference-token-value"),
                inference_base_url=origin,
            )


@pytest.mark.asyncio
async def test_redirect_is_not_followed_and_attempt_count_stays_one() -> None:
    attempts = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(302, headers={"Location": "https://unapproved.example/steal"})

    settings = Settings(
        app_env="test",
        inference_base_url="https://inference.internal",
        inference_service_token=SecretStr("network-test-token"),
    )
    clock = FakeClock(datetime(2030, 1, 1, tzinfo=UTC))
    budget = DeadlineBudget.from_absolute(clock.now + timedelta(seconds=2), clock=clock, minimum_seconds=0.1)
    async with httpx.AsyncClient(
        base_url="https://inference.internal",
        transport=httpx.MockTransport(handler),
        follow_redirects=False,
        trust_env=False,
    ) as raw:
        client = RecommendationInferenceHttpClient(client=raw, settings=settings)
        with pytest.raises(AgentError) as captured:
            await client.score(command_from_agent_request(canonical_request()), budget=budget)
    assert captured.value.code is ErrorCode.INFERENCE_HTTP_ERROR
    assert attempts == 1
