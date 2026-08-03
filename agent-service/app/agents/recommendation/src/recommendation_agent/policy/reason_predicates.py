"""Central evidence predicates mirrored by Backend consumer validation."""

import math
from dataclasses import dataclass

from recommendation_agent.domain.models import ReasonCode, ScoredCandidate
from recommendation_agent.policy.versions import REASON_POLICY_VERSION


@dataclass(frozen=True, slots=True)
class ReasonPolicy:
    version: str = REASON_POLICY_VERSION
    user_cf_score_threshold: float = 0.60
    user_cf_support_threshold: int = 3
    item_cf_score_threshold: float = 0.60
    item_cf_support_threshold: int = 2
    preference_match_threshold: float = 0.70
    group_rate_threshold: float = 0.60
    group_member_threshold: int = 2
    context_match_threshold: float = 0.70
    max_reasons: int = 2
    priority: tuple[ReasonCode, ...] = (
        ReasonCode.USER_CF,
        ReasonCode.ITEM_CF,
        ReasonCode.PREFERENCE_MATCH,
        ReasonCode.WANT_TO_TRY,
        ReasonCode.GROUP_POPULAR,
        ReasonCode.CONTEXT_MATCH,
        ReasonCode.CLEANLINESS_OBSERVED,
    )

    def validate(self) -> None:
        if len(self.priority) != len(set(self.priority)) or set(self.priority) != set(ReasonCode):
            raise RuntimeError("reason priority must contain every reason exactly once")
        if self.max_reasons != 2:
            raise RuntimeError("reason count changed without a policy version")


REASON_POLICY = ReasonPolicy()
REASON_POLICY.validate()


def user_cf_supported(candidate: ScoredCandidate, policy: ReasonPolicy = REASON_POLICY) -> bool:
    signal = candidate.user_cf
    return (
        signal.available
        and signal.score is not None
        and math.isfinite(signal.score)
        and signal.score >= policy.user_cf_score_threshold
        and signal.support >= policy.user_cf_support_threshold
    )


def item_cf_supported(candidate: ScoredCandidate, policy: ReasonPolicy = REASON_POLICY) -> bool:
    signal = candidate.item_cf
    return (
        signal.available
        and signal.score is not None
        and math.isfinite(signal.score)
        and signal.score >= policy.item_cf_score_threshold
        and signal.support >= policy.item_cf_support_threshold
    )


def preference_supported(candidate: ScoredCandidate, policy: ReasonPolicy = REASON_POLICY) -> bool:
    return candidate.evidence.preference_match >= policy.preference_match_threshold


def want_to_try_supported(candidate: ScoredCandidate, _policy: ReasonPolicy = REASON_POLICY) -> bool:
    return candidate.evidence.want_to_try


def group_supported(candidate: ScoredCandidate, policy: ReasonPolicy = REASON_POLICY) -> bool:
    rate = candidate.evidence.group_preference_rate
    return (
        rate is not None
        and rate >= policy.group_rate_threshold
        and candidate.evidence.group_eligible_member_count >= policy.group_member_threshold
    )


def context_supported(candidate: ScoredCandidate, policy: ReasonPolicy = REASON_POLICY) -> bool:
    match = candidate.evidence.context_match
    return match is not None and match >= policy.context_match_threshold


def cleanliness_supported(candidate: ScoredCandidate, _policy: ReasonPolicy = REASON_POLICY) -> bool:
    return candidate.evidence.cleanliness_observed


PREDICATES = {
    ReasonCode.USER_CF: user_cf_supported,
    ReasonCode.ITEM_CF: item_cf_supported,
    ReasonCode.PREFERENCE_MATCH: preference_supported,
    ReasonCode.WANT_TO_TRY: want_to_try_supported,
    ReasonCode.GROUP_POPULAR: group_supported,
    ReasonCode.CONTEXT_MATCH: context_supported,
    ReasonCode.CLEANLINESS_OBSERVED: cleanliness_supported,
}
