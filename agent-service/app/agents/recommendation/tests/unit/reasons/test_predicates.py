from dataclasses import replace

import pytest
from workflow_helpers import canonical_result

from recommendation_agent.domain.models import CollaborativeSignal, ReasonCode
from recommendation_agent.policy.reason_predicates import PREDICATES


@pytest.mark.parametrize(
    "reason",
    list(ReasonCode),
)
def test_each_reason_has_positive_evidence(reason: ReasonCode) -> None:
    candidates = canonical_result().candidates
    positive = {
        ReasonCode.USER_CF: candidates[0],
        ReasonCode.ITEM_CF: candidates[0],
        ReasonCode.PREFERENCE_MATCH: candidates[0],
        ReasonCode.WANT_TO_TRY: candidates[1],
        ReasonCode.GROUP_POPULAR: candidates[2],
        ReasonCode.CONTEXT_MATCH: candidates[0],
        ReasonCode.CLEANLINESS_OBSERVED: candidates[0],
    }
    assert PREDICATES[reason](positive[reason])


def test_cf_thresholds_require_availability_score_and_correct_support_counter() -> None:
    candidate = canonical_result().candidates[0]
    assert not PREDICATES[ReasonCode.USER_CF](replace(candidate, user_cf=CollaborativeSignal(True, 0.6, 2)))
    assert not PREDICATES[ReasonCode.USER_CF](replace(candidate, user_cf=CollaborativeSignal(False, None, 100)))
    assert not PREDICATES[ReasonCode.ITEM_CF](replace(candidate, item_cf=CollaborativeSignal(True, 0.599, 100)))
    group_only = canonical_result().candidates[2]
    assert PREDICATES[ReasonCode.GROUP_POPULAR](group_only)
    assert not PREDICATES[ReasonCode.USER_CF](group_only)


def test_backend_fact_thresholds_use_positive_evidence_not_non_conflict() -> None:
    candidate = canonical_result().candidates[0]
    evidence = replace(
        candidate.evidence,
        preference_match=0.699,
        want_to_try=False,
        group_preference_rate=1.0,
        group_eligible_member_count=1,
        context_match=0.699,
        cleanliness_observed=False,
    )
    near_miss = replace(candidate, evidence=evidence)
    for reason in (
        ReasonCode.PREFERENCE_MATCH,
        ReasonCode.WANT_TO_TRY,
        ReasonCode.GROUP_POPULAR,
        ReasonCode.CONTEXT_MATCH,
        ReasonCode.CLEANLINESS_OBSERVED,
    ):
        assert not PREDICATES[reason](near_miss)
