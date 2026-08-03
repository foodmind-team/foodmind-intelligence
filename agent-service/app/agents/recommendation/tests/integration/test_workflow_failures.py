import pytest
from workflow_helpers import FixtureSelector, SpyInference, canonical_request, workflow_context

from recommendation_agent.domain.errors import AgentError, ErrorCode
from recommendation_agent.workflow.graph import BoundedRecommendationWorkflow


@pytest.mark.asyncio
async def test_inference_failure_short_circuits_policy_ports_without_retry() -> None:
    inference = SpyInference(error=AgentError(ErrorCode.INFERENCE_TIMEOUT, http_status=504, retryable=True))
    selector = FixtureSelector()
    workflow = BoundedRecommendationWorkflow(workflow_context(inference=inference, selector=selector))
    with pytest.raises(AgentError) as captured:
        await workflow.run(canonical_request(), agent_trace_id="agent-failure")
    assert captured.value.code is ErrorCode.INFERENCE_TIMEOUT
    assert captured.value.failure_content is not None
    assert captured.value.failure_content["status"] == "failure"
    assert inference.calls == 1
    assert selector.calls == 0


@pytest.mark.asyncio
async def test_malicious_selector_cannot_inject_unknown_candidate() -> None:
    selector = FixtureSelector(malicious_candidate_id="candidate-outside-request")
    workflow = BoundedRecommendationWorkflow(workflow_context(selector=selector))
    with pytest.raises(AgentError) as captured:
        await workflow.run(canonical_request(), agent_trace_id="agent-malicious")
    assert captured.value.code is ErrorCode.UNKNOWN_CANDIDATE
