import copy
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from conftest import AGENT_FIXTURES, REPOSITORY_ROOT, load_json
from pydantic import SecretStr

from recommendation_agent.clients.inference import RecommendationInferenceHttpClient, command_from_agent_request
from recommendation_agent.config.settings import Settings
from recommendation_agent.domain.errors import AgentError, ErrorCode
from recommendation_agent.schemas.agent_v2 import AgentRequest
from recommendation_agent.time.budget import DeadlineBudget

FIXTURES = REPOSITORY_ROOT / "contracts/internal/inference/recommendation/v1/consumer-fixtures"


@dataclass
class FakeClock:
    now: datetime
    tick: float = 10.0

    def utc_now(self) -> datetime:
        return self.now

    def monotonic(self) -> float:
        return self.tick


def _settings() -> Settings:
    return Settings(
        app_env="test",
        internal_service_token=SecretStr("agent-test-token"),
        inference_service_token=SecretStr("inference-secret-canary"),
        inference_base_url="https://inference.internal",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("unknown", ErrorCode.UNKNOWN_CANDIDATE),
        ("duplicate", ErrorCode.DUPLICATE_CANDIDATE),
        ("missing", ErrorCode.MISSING_CANDIDATE),
        ("probability", ErrorCode.INVALID_PROBABILITY),
        ("cf", ErrorCode.INVALID_EVIDENCE),
        ("evidence", ErrorCode.INVALID_EVIDENCE),
        ("model", ErrorCode.MODEL_VERSION_MISMATCH),
        ("package", ErrorCode.MODEL_PACKAGE_MISMATCH),
        ("feature", ErrorCode.FEATURE_VERSION_MISMATCH),
        ("key", ErrorCode.MODEL_KEY_VERSION_MISMATCH),
        ("echo", ErrorCode.INFERENCE_CONTRACT_MISMATCH),
    ],
)
async def test_invalid_downstream_results_map_to_typed_failures(mutation: str, expected: ErrorCode) -> None:
    payload: dict[str, Any] = copy.deepcopy(load_json(FIXTURES / "valid-hybrid.json"))
    predictions = payload["predictions"]
    if mutation == "unknown":
        predictions[0]["candidateId"] = "outside-request"
    elif mutation == "duplicate":
        predictions[1]["candidateId"] = predictions[0]["candidateId"]
    elif mutation == "missing":
        predictions.pop()
    elif mutation == "probability":
        predictions[0]["probability"] = 1.1
    elif mutation == "cf":
        predictions[0]["userCf"]["available"] = False
    elif mutation == "evidence":
        predictions[0]["signals"]["wantToTry"] = True
    elif mutation == "model":
        payload["modelVersion"] = "other-model"
    elif mutation == "package":
        payload["modelPackageVersion"] = "other-package"
    elif mutation == "feature":
        payload["featureSchemaVersion"] = "other-feature"
    elif mutation == "key":
        payload["modelKeyVersion"] = "other-key"
    elif mutation == "echo":
        payload["requestId"] = "other-request"

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    clock = FakeClock(datetime(2030, 1, 1, tzinfo=UTC))
    request = AgentRequest.model_validate_json((AGENT_FIXTURES / "valid-normal.json").read_text(encoding="utf-8"))
    budget = DeadlineBudget.from_absolute(clock.now + timedelta(seconds=2), clock=clock, minimum_seconds=0.1)
    async with httpx.AsyncClient(base_url="https://inference.internal", transport=httpx.MockTransport(handler)) as raw:
        client = RecommendationInferenceHttpClient(client=raw, settings=_settings())
        with pytest.raises(AgentError) as captured:
            await client.score(command_from_agent_request(request), budget=budget)
    assert captured.value.code is expected
