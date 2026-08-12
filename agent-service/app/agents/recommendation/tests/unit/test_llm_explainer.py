from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest

from recommendation_agent.config.settings import Settings
from recommendation_agent.domain.models import (
    ReasonCode,
    ReasonedCandidate,
    RecommendationType,
    SelectedCandidate,
)
from recommendation_agent.llm.client import LLMClient, LLMError
from recommendation_agent.llm.explainer import LLMExplanationRenderer
from recommendation_agent.time.budget import DeadlineBudget, SystemClock


class FakeLLMClient:
    def __init__(self, response: dict[str, Any] | Exception) -> None:
        self.response = response
        self.calls = 0

    async def chat_json(self, _messages: list[dict[str, str]], *, timeout_seconds: float) -> dict[str, Any]:
        assert timeout_seconds > 0
        self.calls += 1
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def candidates() -> tuple[ReasonedCandidate, ...]:
    return (
        ReasonedCandidate(
            SelectedCandidate("candidate-1", RecommendationType.PERSONAL, 0.9, 0.8),
            (ReasonCode.PREFERENCE_MATCH,),
        ),
        ReasonedCandidate(
            SelectedCandidate("candidate-2", RecommendationType.EXPLORATORY, 0.8, 0.7),
            (ReasonCode.WANT_TO_TRY,),
        ),
    )


def budget() -> DeadlineBudget:
    return DeadlineBudget.from_absolute(
        datetime.now(UTC) + timedelta(seconds=5),
        clock=SystemClock(),
        minimum_seconds=0.1,
    )


@pytest.mark.asyncio
async def test_llm_explainer_preserves_selected_order_and_approved_facts() -> None:
    client = FakeLLMClient(
        {
            "explanations": [
                {"candidateId": "candidate-2", "explanation": "You marked this as Want to Try."},
                {"candidateId": "candidate-1", "explanation": "It matches your saved preferences."},
            ]
        }
    )
    renderer = LLMExplanationRenderer(client=client, settings=Settings(app_env="test"))  # type: ignore[arg-type]

    rendered = await renderer.render(candidates(), budget=budget())

    assert client.calls == 1
    assert [item.reasoned.selection.candidate_id for item in rendered] == ["candidate-1", "candidate-2"]
    assert [item.explanation for item in rendered] == [
        "It matches your saved preferences.",
        "You marked this as Want to Try.",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        {"explanations": [{"candidateId": "candidate-1", "explanation": "It is cheap and nearby."}]},
        {"explanations": [{"candidateId": "invented", "explanation": "It matches your saved preferences."}]},
        LLMError("provider unavailable"),
    ],
)
async def test_llm_explainer_falls_back_without_changing_ml_results(response: dict[str, Any] | Exception) -> None:
    renderer = LLMExplanationRenderer(
        client=FakeLLMClient(response),  # type: ignore[arg-type]
        settings=Settings(app_env="test"),
    )

    rendered = await renderer.render(candidates(), budget=budget())

    assert [item.reasoned.selection.candidate_id for item in rendered] == ["candidate-1", "candidate-2"]
    assert [item.explanation for item in rendered] == [
        "It matches your saved preferences.",
        "You marked this as Want to Try.",
    ]


@pytest.mark.asyncio
async def test_llm_client_does_not_retry_provider_failures() -> None:
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(502, json={"error": "unavailable"})

    client = LLMClient(
        base_url="https://llm.test",
        model="test-model",
        api_key="temporary-test-key",
        timeout_seconds=1.0,
        temperature=0.0,
        max_output_tokens=128,
        connection_pool_size=1,
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(LLMError, match="provider request failed"):
            await client.chat_json([], timeout_seconds=1.0)
    finally:
        await client.aclose()

    assert attempts == 1
