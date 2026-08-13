"""Stable score-ordered recommendation selection."""

from recommendation_agent.domain.models import (
    InferenceResult,
    RecommendationType,
    ScoredCandidate,
    SelectedCandidate,
)
from recommendation_agent.policy.diversity import DIVERSITY_POLICY, DiversityPolicy
from recommendation_agent.policy.reason_predicates import group_supported
from recommendation_agent.schemas.agent_v2 import AgentRequest


class DeterministicResultSelector:
    def __init__(self, policy: DiversityPolicy = DIVERSITY_POLICY) -> None:
        policy.validate()
        self._policy = policy

    async def select(self, request: AgentRequest, result: InferenceResult) -> tuple[SelectedCandidate, ...]:
        request_index = {candidate.candidate_id: index for index, candidate in enumerate(request.candidates)}
        ordered = sorted(result.candidates, key=lambda candidate: _stable_key(candidate, request_index))
        selected: list[SelectedCandidate] = []
        for index, candidate in enumerate(ordered[: self._policy.max_results]):
            recommendation_type = (
                RecommendationType.PERSONAL
                if index == 0
                else (
                    RecommendationType.GROUP_INSPIRED if group_supported(candidate) else RecommendationType.EXPLORATORY
                )
            )
            selected.append(_selection(candidate, recommendation_type))
        return tuple(selected)


def _stable_key(candidate: ScoredCandidate, request_index: dict[str, int]) -> tuple[float, float, int, str]:
    return (
        -candidate.probability,
        -candidate.model_score,
        request_index[candidate.candidate_id],
        candidate.candidate_id,
    )


def _selection(candidate: ScoredCandidate, recommendation_type: RecommendationType) -> SelectedCandidate:
    return SelectedCandidate(candidate.candidate_id, recommendation_type, candidate.probability, candidate.model_score)
