import pytest
from workflow_helpers import canonical_request, canonical_result

from recommendation_agent.reasons.deriver import DeterministicReasonDeriver
from recommendation_agent.reasons.renderer import DeterministicExplanationRenderer
from recommendation_agent.selection.selector import DeterministicResultSelector


@pytest.mark.asyncio
async def test_renderer_composes_only_fixed_observational_templates() -> None:
    request = canonical_request()
    result = canonical_result()
    selections = await DeterministicResultSelector().select(request, result)
    reasoned = await DeterministicReasonDeriver().derive(request, result, selections)
    rendered = await DeterministicExplanationRenderer().render(reasoned)
    assert rendered[0].explanation == "People with similar preferences also liked this. It resembles meals you liked."
    assert all(len(item.explanation) <= 160 for item in rendered)
