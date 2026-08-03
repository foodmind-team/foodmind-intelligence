from dataclasses import replace

import pytest
from workflow_helpers import canonical_request, canonical_result

from recommendation_agent.domain.models import CollaborativeSignal, RecommendationType
from recommendation_agent.selection.selector import DeterministicResultSelector


@pytest.mark.asyncio
async def test_normal_policy_selects_fixed_type_order_and_unique_candidates() -> None:
    selections = await DeterministicResultSelector().select(canonical_request(), canonical_result())
    assert [item.candidate_id for item in selections] == ["candidate-a", "candidate-b", "candidate-c"]
    assert [item.recommendation_type for item in selections] == [
        RecommendationType.PERSONAL,
        RecommendationType.EXPLORATORY,
        RecommendationType.GROUP_INSPIRED,
    ]
    assert [item.probability for item in selections] == [0.91, 0.82, 0.78]


@pytest.mark.asyncio
async def test_personal_preference_applies_at_equal_tie_band_but_not_outside() -> None:
    result = canonical_result()
    first = replace(
        result.candidates[0],
        probability=0.9,
        user_cf=CollaborativeSignal(False, None, 0),
        evidence=replace(result.candidates[0].evidence, preference_match=0.5),
    )
    at_band = replace(result.candidates[1], probability=0.87)
    equal_result = replace(result, candidates=(first, at_band))
    equal = await DeterministicResultSelector().select(canonical_request(), equal_result)
    assert equal[0].candidate_id == "candidate-b"
    assert equal[0].recommendation_type is RecommendationType.PERSONAL

    outside_result = replace(equal_result, candidates=(first, replace(at_band, probability=0.869)))
    outside = await DeterministicResultSelector().select(canonical_request(), outside_result)
    assert all(item.recommendation_type is not RecommendationType.PERSONAL for item in outside)


@pytest.mark.asyncio
async def test_exploratory_bonus_cannot_displace_materially_higher_lead() -> None:
    result = canonical_result()
    first = replace(
        result.candidates[0],
        probability=0.9,
        user_cf=CollaborativeSignal(False, None, 0),
        evidence=replace(result.candidates[0].evidence, preference_match=0.5),
    )
    second = replace(
        result.candidates[1],
        probability=0.84,
        user_cf=CollaborativeSignal(False, None, 0),
        evidence=replace(result.candidates[1].evidence, preference_match=0.5),
    )
    selections = await DeterministicResultSelector().select(
        canonical_request(), replace(result, candidates=(first, second))
    )
    assert selections[0].candidate_id == "candidate-a"
    assert selections[0].recommendation_type is RecommendationType.EXPLORATORY


@pytest.mark.asyncio
async def test_group_support_never_uses_user_cf_proxy() -> None:
    result = canonical_result()
    candidate = replace(
        result.candidates[1],
        user_cf=CollaborativeSignal(True, 1.0, 100),
        evidence=replace(result.candidates[1].evidence, group_preference_rate=0.59, group_eligible_member_count=100),
    )
    selections = await DeterministicResultSelector().select(
        canonical_request(), replace(result, candidates=(candidate,))
    )
    assert all(item.recommendation_type is not RecommendationType.GROUP_INSPIRED for item in selections)
