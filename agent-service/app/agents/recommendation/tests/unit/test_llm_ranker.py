from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from recommendation_agent.config.settings import Settings
from recommendation_agent.domain.errors import AgentError, ErrorCode
from recommendation_agent.domain.models import InferenceCandidate, InferenceCommand, InferenceEvidence
from recommendation_agent.llm.ranker import LLMRanker
from recommendation_agent.time.budget import DeadlineBudget, SystemClock


class FakeLLMClient:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response

    async def chat_json(self, _messages: list[dict[str, str]], *, timeout_seconds: float) -> dict[str, Any]:
        assert timeout_seconds > 0
        return self.response


def command() -> InferenceCommand:
    evidence = InferenceEvidence(
        preference_match=0.8,
        want_to_try=False,
        group_preference_rate=None,
        group_eligible_member_count=0,
        context_match=1.0,
        cleanliness_observed=True,
    )
    return InferenceCommand(
        request_id="request-1",
        trace_id="trace-1",
        deadline_at=datetime.now(UTC) + timedelta(seconds=5),
        feature_schema_version="recommendation-features-v2",
        model_user_key="a" * 43,
        model_key_version="hmac-sha256-v1",
        candidates=(
            InferenceCandidate("candidate-1", "b" * 43, "c" * 43, evidence),
            InferenceCandidate("candidate-2", "d" * 43, "e" * 43, evidence),
        ),
    )


def budget() -> DeadlineBudget:
    return DeadlineBudget.from_absolute(
        datetime.now(UTC) + timedelta(seconds=5),
        clock=SystemClock(),
        minimum_seconds=0.1,
    )


@pytest.mark.asyncio
async def test_llm_ranker_returns_scores_for_every_opaque_candidate() -> None:
    client = FakeLLMClient(
        {
            "predictions": [
                {"candidateId": "candidate-2", "probability": 0.9, "modelScore": 0.8},
                {"candidateId": "candidate-1", "probability": 0.7, "modelScore": 0.6},
            ]
        }
    )
    ranker = LLMRanker(client=client, settings=Settings(app_env="test"))  # type: ignore[arg-type]

    result = await ranker.score(command(), budget=budget())

    assert [item.candidate_id for item in result.candidates] == ["candidate-1", "candidate-2"]
    assert [item.probability for item in result.candidates] == [0.7, 0.9]
    assert all(item.evidence.preference_match == 0.8 for item in result.candidates)


@pytest.mark.asyncio
async def test_llm_ranker_rejects_missing_or_invented_candidates() -> None:
    client = FakeLLMClient({"predictions": [{"candidateId": "invented", "probability": 0.9, "modelScore": 0.8}]})
    ranker = LLMRanker(client=client, settings=Settings(app_env="test"))  # type: ignore[arg-type]

    with pytest.raises(AgentError) as raised:
        await ranker.score(command(), budget=budget())

    assert raised.value.code == ErrorCode.INFERENCE_UNAVAILABLE
