import pytest
from workflow_helpers import canonical_request, canonical_result

from recommendation_agent.domain.models import ReasonCode
from recommendation_agent.reasons.deriver import DeterministicReasonDeriver
from recommendation_agent.selection.selector import DeterministicResultSelector


@pytest.mark.asyncio
async def test_reason_priority_and_max_count_are_frozen() -> None:
    request = canonical_request()
    result = canonical_result()
    selections = await DeterministicResultSelector().select(request, result)
    reasoned = await DeterministicReasonDeriver().derive(request, result, selections)
    assert [item.reasons for item in reasoned] == [
        (ReasonCode.USER_CF, ReasonCode.WANT_TO_TRY),
        (ReasonCode.USER_CF, ReasonCode.ITEM_CF),
        (ReasonCode.ITEM_CF, ReasonCode.GROUP_POPULAR),
    ]
