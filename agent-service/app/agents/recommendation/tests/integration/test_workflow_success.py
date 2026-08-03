from typing import cast

import pytest
from workflow_helpers import (
    FixtureReasons,
    FixtureRenderer,
    FixtureSelector,
    SpyInference,
    canonical_request,
    workflow_context,
)

from recommendation_agent.workflow.graph import BoundedRecommendationWorkflow


@pytest.mark.asyncio
async def test_success_visits_ports_once_and_is_deterministic() -> None:
    inference = SpyInference()
    selector = FixtureSelector()
    reasons = FixtureReasons()
    renderer = FixtureRenderer()
    workflow = BoundedRecommendationWorkflow(
        workflow_context(inference=inference, selector=selector, reasons=reasons, renderer=renderer)
    )
    first = await workflow.run(canonical_request(), agent_trace_id="agent-deterministic")
    second = await workflow.run(canonical_request(), agent_trace_id="agent-deterministic")
    assert first.model_dump(mode="json", by_alias=True) == second.model_dump(mode="json", by_alias=True)
    assert (inference.calls, selector.calls, reasons.calls, renderer.calls) == (2, 2, 2, 2)
    assert [item.rank for item in first.recommendations] == [1, 2, 3]
    snapshot = workflow.context.metrics.snapshot()
    assert cast(dict[str, int], snapshot["requests"])["success"] == 2
    assert cast(dict[str, int], snapshot["stageCounts"])["score_once"] == 2
