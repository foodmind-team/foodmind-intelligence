import pytest
from workflow_helpers import FakeClock, canonical_request, workflow_context

from recommendation_agent.workflow.nodes import WorkflowNodes
from recommendation_agent.workflow.state import RecommendationState


@pytest.mark.asyncio
async def test_validate_node_returns_only_its_documented_update() -> None:
    clock = FakeClock()
    nodes = WorkflowNodes(workflow_context(clock=clock))
    state: RecommendationState = {
        "request": canonical_request(),
        "agent_trace_id": "agent-test",
        "deadline_expiry": 102.0,
        "inference_calls": 0,
        "node_trace": (),
    }
    update = await nodes.validate_envelope(state)
    assert update == {"node_trace": ("validate_envelope",)}
