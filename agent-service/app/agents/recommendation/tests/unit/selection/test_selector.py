from dataclasses import replace

import pytest
from workflow_helpers import canonical_request, canonical_result

from recommendation_agent.domain.models import CollaborativeSignal, RecommendationType
from recommendation_agent.selection.selector import DeterministicResultSelector


@pytest.mark.asyncio
async def test_normal_policy_selects_fixed_type_order_and_unique_candidates() -> None:
    selections = await DeterministicResultSelector().select(canonical_request(), canonical_result())
    assert [item.candidate_id for item in selections] == ["candidate-b", "candidate-a", "candidate-c"]
    assert [item.recommendation_type for item in selections] == [
        RecommendationType.PERSONAL,
        RecommendationType.PERSONAL,
        RecommendationType.GROUP_INSPIRED,
    ]
    assert [item.probability for item in selections] == [0.82, 0.91, 0.78]


@pytest.mark.asyncio
async def test_highest_probability_is_always_the_lead() -> None:
    result = canonical_result()
    first = replace(
        result.candidates[0],
        probability=0.9,
        user_cf=CollaborativeSignal(False, None, 0),
        evidence=replace(result.candidates[0].evidence, preference_match=0.5),
    )
    second = replace(
        result.candidates[1], probability=0.87, evidence=replace(result.candidates[1].evidence, want_to_try=False)
    )
    selections = await DeterministicResultSelector().select(
        canonical_request(), replace(result, candidates=(first, second))
    )
    assert [item.candidate_id for item in selections] == ["candidate-a", "candidate-b"]
    assert selections[0].recommendation_type is RecommendationType.EXPLORATORY


@pytest.mark.asyncio
async def test_want_to_try_candidate_precedes_a_higher_probability_and_is_personal() -> None:
    result = canonical_result()
    higher_probability = replace(
        result.candidates[0],
        probability=0.95,
        user_cf=CollaborativeSignal(False, None, 0),
        evidence=replace(result.candidates[0].evidence, preference_match=0.5, want_to_try=False),
    )
    want_to_try = replace(
        result.candidates[1],
        probability=0.10,
        user_cf=CollaborativeSignal(False, None, 0),
        evidence=replace(result.candidates[1].evidence, preference_match=0.5, want_to_try=True),
    )

    selections = await DeterministicResultSelector().select(
        canonical_request(), replace(result, candidates=(higher_probability, want_to_try))
    )

    assert [item.candidate_id for item in selections] == ["candidate-b", "candidate-a"]
    assert selections[0].recommendation_type is RecommendationType.PERSONAL


@pytest.mark.asyncio
async def test_novelty_cannot_displace_a_higher_ml_score() -> None:
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
        evidence=replace(result.candidates[1].evidence, preference_match=0.5, want_to_try=False),
    )
    selections = await DeterministicResultSelector().select(
        canonical_request(), replace(result, candidates=(first, second))
    )
    assert [item.candidate_id for item in selections] == ["candidate-a", "candidate-b"]
    assert [item.probability for item in selections] == [0.9, 0.84]


@pytest.mark.asyncio
async def test_diverse_alternative_can_displace_a_repeated_offering() -> None:
    request = canonical_request()
    result = canonical_result()
    repeated = replace(
        result.candidates[1], probability=0.86, evidence=replace(result.candidates[1].evidence, want_to_try=False)
    )
    diverse = replace(result.candidates[2], probability=0.83)
    repeated_request = request.candidates[1].model_copy(
        update={"model_offering_key": request.candidates[0].model_offering_key}
    )

    selections = await DeterministicResultSelector().select(
        request.model_copy(update={"candidates": (request.candidates[0], repeated_request, request.candidates[2])}),
        replace(result, candidates=(result.candidates[0], repeated, diverse)),
    )

    assert [item.candidate_id for item in selections] == ["candidate-a", "candidate-c", "candidate-b"]


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
