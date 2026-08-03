import asyncio
import statistics
from dataclasses import replace
from time import perf_counter
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from workflow_helpers import FakeClock, SpyInference, canonical_request, canonical_result, workflow_context

from recommendation_agent.config.settings import Settings
from recommendation_agent.domain.models import InferenceResult
from recommendation_agent.main import create_app
from recommendation_agent.reasons.deriver import DeterministicReasonDeriver
from recommendation_agent.reasons.renderer import DeterministicExplanationRenderer
from recommendation_agent.schemas.agent_v2 import AgentRequest
from recommendation_agent.selection.selector import DeterministicResultSelector
from recommendation_agent.workflow.context import WorkflowContext
from recommendation_agent.workflow.graph import BoundedRecommendationWorkflow


def _scenario(count: int) -> tuple[AgentRequest, InferenceResult]:
    request = canonical_request()
    result = canonical_result()
    request_candidate = request.candidates[0]
    scored_candidate = result.candidates[0]
    request_candidates = tuple(
        request_candidate.model_copy(
            update={
                "candidate_id": f"candidate-perf-{index:03d}",
                "model_meal_key": f"meal_key_perf_{index:04d}",
                "model_offering_key": f"offering_perf_{index:04d}",
            }
        )
        for index in range(count)
    )
    scored_candidates = tuple(
        replace(
            scored_candidate,
            candidate_id=f"candidate-perf-{index:03d}",
            probability=1.0 - (index / max(1, count)) * 0.5,
        )
        for index in range(count)
    )
    return request.model_copy(update={"candidates": request_candidates}), replace(result, candidates=scored_candidates)


@pytest.mark.asyncio
@pytest.mark.parametrize("candidate_count", [1, 10, 100])
async def test_agent_only_p95_stays_within_accepted_100ms_budget(candidate_count: int) -> None:
    request, result = _scenario(candidate_count)
    base = workflow_context(inference=SpyInference(result=result), clock=FakeClock())
    context = WorkflowContext(
        inference=base.inference,
        selector=DeterministicResultSelector(),
        reason_deriver=DeterministicReasonDeriver(),
        renderer=DeterministicExplanationRenderer(),
        settings=base.settings,
        clock=base.clock,
        metrics=base.metrics,
    )
    workflow = BoundedRecommendationWorkflow(context)
    durations: list[float] = []
    for iteration in range(20):
        started = perf_counter()
        await workflow.run(request, agent_trace_id=f"agent-perf-{iteration}")
        durations.append(perf_counter() - started)
    p95 = statistics.quantiles(durations, n=20)[18]
    assert p95 < 0.1, f"agent-only p95={p95:.6f}s for {candidate_count} candidates"


def _wire_result(request: AgentRequest, result: InferenceResult) -> dict[str, Any]:
    return {
        "contractVersion": result.inference_contract_version,
        "requestId": request.request_id,
        "traceId": request.trace_id,
        "status": "success",
        "modelVersion": result.model_version,
        "modelPackageVersion": result.model_package_version,
        "featureSchemaVersion": result.feature_schema_version,
        "modelKeyVersion": result.model_key_version,
        "predictions": [
            {
                "candidateId": candidate.candidate_id,
                "probability": candidate.probability,
                "modelScore": candidate.model_score,
                "userCf": {
                    "available": candidate.user_cf.available,
                    "score": candidate.user_cf.score,
                    "neighborSupport": candidate.user_cf.support,
                },
                "itemCf": {
                    "available": candidate.item_cf.available,
                    "score": candidate.item_cf.score,
                    "supportingItemCount": candidate.item_cf.support,
                },
                "signals": {
                    "preferenceMatch": candidate.evidence.preference_match,
                    "wantToTry": candidate.evidence.want_to_try,
                    "groupPreferenceRate": candidate.evidence.group_preference_rate,
                    "groupEligibleMemberCount": candidate.evidence.group_eligible_member_count,
                    "contextMatch": candidate.evidence.context_match,
                    "cleanlinessObserved": candidate.evidence.cleanliness_observed,
                },
            }
            for candidate in result.candidates
        ],
    }


@pytest.mark.parametrize("candidate_count", [1, 10, 100])
def test_private_http_fixture_total_p95_stays_within_accepted_800ms_budget(candidate_count: int) -> None:
    request, result = _scenario(candidate_count)
    payload = _wire_result(request, result)
    calls = 0

    async def handler(http_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert len(http_request.content) <= 262_144
        await asyncio.sleep(0.005)
        return httpx.Response(200, json=payload)

    raw = httpx.AsyncClient(base_url="http://fixture-inference.test", transport=httpx.MockTransport(handler))
    settings = Settings(
        app_env="test",
        internal_service_token=SecretStr("performance-agent-token"),
        inference_service_token=SecretStr("performance-inference-token"),
    )
    durations: list[float] = []
    with TestClient(create_app(settings=settings, inference_http_client=raw)) as client:
        for _iteration in range(20):
            started = perf_counter()
            response = client.post(
                "/internal/v1/recommendations/generate",
                content=request.model_dump_json(by_alias=True),
                headers={
                    "Authorization": "Bearer performance-agent-token",
                    "Content-Type": "application/json",
                },
            )
            durations.append(perf_counter() - started)
            assert response.status_code == 200
    p95 = statistics.quantiles(durations, n=20)[18]
    assert calls == 20
    assert p95 < 0.8, f"fixture-transport total p95={p95:.6f}s for {candidate_count} candidates"
