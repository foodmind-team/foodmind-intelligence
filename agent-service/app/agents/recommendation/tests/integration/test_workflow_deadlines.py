import asyncio
from datetime import timedelta

import pytest
from workflow_helpers import FakeClock, SpyInference, canonical_request, workflow_context

from recommendation_agent.application.ports import ResultSelector
from recommendation_agent.domain.errors import AgentError, ErrorCode
from recommendation_agent.domain.models import InferenceResult, SelectedCandidate
from recommendation_agent.schemas.agent_v2 import AgentRequest
from recommendation_agent.workflow.graph import BoundedRecommendationWorkflow


class SlowSelector(ResultSelector):
    async def select(self, _request: AgentRequest, _result: InferenceResult) -> tuple[SelectedCandidate, ...]:
        await asyncio.sleep(10)
        return ()


@pytest.mark.asyncio
async def test_expired_request_invokes_inference_zero_times() -> None:
    clock = FakeClock()
    inference = SpyInference()
    payload = canonical_request().model_dump(mode="json", by_alias=True)
    payload["decisionAt"] = (clock.now - timedelta(seconds=2)).isoformat().replace("+00:00", "Z")
    payload["deadlineAt"] = (clock.now - timedelta(seconds=1)).isoformat().replace("+00:00", "Z")
    request = AgentRequest.model_validate(payload)
    workflow = BoundedRecommendationWorkflow(workflow_context(inference=inference, clock=clock))
    with pytest.raises(AgentError) as captured:
        await workflow.run(request, agent_trace_id="agent-expired")
    assert captured.value.code is ErrorCode.DEADLINE_EXPIRED
    assert inference.calls == 0


@pytest.mark.asyncio
async def test_slow_policy_is_cancelled_by_outer_deadline() -> None:
    clock = FakeClock()
    payload = canonical_request().model_dump(mode="json", by_alias=True)
    payload["deadlineAt"] = (clock.now + timedelta(milliseconds=120)).isoformat().replace("+00:00", "Z")
    request = AgentRequest.model_validate(payload)
    workflow = BoundedRecommendationWorkflow(workflow_context(selector=SlowSelector(), clock=clock))
    with pytest.raises(AgentError) as captured:
        await workflow.run(request, agent_trace_id="agent-slow")
    assert captured.value.code is ErrorCode.DEADLINE_EXHAUSTED
