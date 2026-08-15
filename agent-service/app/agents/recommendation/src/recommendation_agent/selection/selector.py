"""Stable model-led selection with greedy diversity reranking."""

from recommendation_agent.domain.models import (
    InferenceResult,
    RecommendationType,
    ScoredCandidate,
    SelectedCandidate,
)
from recommendation_agent.policy.diversity import DIVERSITY_POLICY, DiversityPolicy
from recommendation_agent.policy.reason_predicates import group_supported, preference_supported, user_cf_supported
from recommendation_agent.schemas.agent_v2 import AgentRequest, Candidate
from recommendation_agent.selection.similarity import similarity_penalty


class DeterministicResultSelector:
    def __init__(self, policy: DiversityPolicy = DIVERSITY_POLICY) -> None:
        policy.validate()
        self._policy = policy

    async def select(self, request: AgentRequest, result: InferenceResult) -> tuple[SelectedCandidate, ...]:
        request_index = {candidate.candidate_id: index for index, candidate in enumerate(request.candidates)}
        request_candidates = {candidate.candidate_id: candidate for candidate in request.candidates}
        ordered = sorted(result.candidates, key=lambda candidate: _stable_key(candidate, request_index))
        if not ordered:
            return ()

        selected_candidates = [ordered.pop(0)]
        while ordered and len(selected_candidates) < self._policy.max_results:
            used_types = {_recommendation_type(candidate) for candidate in selected_candidates}
            candidate = min(
                ordered,
                key=lambda item: _diversity_key(
                    item,
                    selected_candidates,
                    request_candidates,
                    request_index,
                    used_types,
                    self._policy,
                ),
            )
            selected_candidates.append(candidate)
            ordered.remove(candidate)

        selected = [_selection(candidate, _recommendation_type(candidate)) for candidate in selected_candidates]
        return tuple(selected)


def _stable_key(candidate: ScoredCandidate, request_index: dict[str, int]) -> tuple[float, float, int, str]:
    return (
        -candidate.probability,
        -candidate.model_score,
        request_index[candidate.candidate_id],
        candidate.candidate_id,
    )


def _diversity_key(
    candidate: ScoredCandidate,
    selected: list[ScoredCandidate],
    request_candidates: dict[str, Candidate],
    request_index: dict[str, int],
    used_types: set[RecommendationType],
    policy: DiversityPolicy,
) -> tuple[float, int, float, float, int, str]:
    request_candidate = request_candidates[candidate.candidate_id]
    penalty = max(
        similarity_penalty(request_candidate, request_candidates[item.candidate_id], policy) for item in selected
    )
    novelty_bonus = min(request_candidate.evidence.novelty * policy.novelty_bonus_multiplier, policy.novelty_bonus_cap)
    adjusted = candidate.probability + novelty_bonus - penalty
    recommendation_type = _recommendation_type(candidate)
    return (
        -adjusted,
        0 if recommendation_type not in used_types else 1,
        -candidate.probability,
        -candidate.model_score,
        request_index[candidate.candidate_id],
        candidate.candidate_id,
    )


def _recommendation_type(candidate: ScoredCandidate) -> RecommendationType:
    if group_supported(candidate):
        return RecommendationType.GROUP_INSPIRED
    if user_cf_supported(candidate) or preference_supported(candidate):
        return RecommendationType.PERSONAL
    return RecommendationType.EXPLORATORY


def _selection(candidate: ScoredCandidate, recommendation_type: RecommendationType) -> SelectedCandidate:
    return SelectedCandidate(candidate.candidate_id, recommendation_type, candidate.probability, candidate.model_score)
