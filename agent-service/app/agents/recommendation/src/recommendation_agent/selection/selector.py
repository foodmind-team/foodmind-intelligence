"""Stable Personal, Exploratory, and Group-inspired selection."""

from recommendation_agent.domain.models import (
    InferenceResult,
    RecommendationType,
    ScoredCandidate,
    SelectedCandidate,
)
from recommendation_agent.policy.diversity import DIVERSITY_POLICY, DiversityPolicy
from recommendation_agent.policy.reason_predicates import preference_supported, user_cf_supported
from recommendation_agent.schemas.agent_v2 import AgentRequest, Candidate
from recommendation_agent.selection.similarity import similarity_penalty


class DeterministicResultSelector:
    def __init__(self, policy: DiversityPolicy = DIVERSITY_POLICY) -> None:
        policy.validate()
        self._policy = policy

    async def select(self, request: AgentRequest, result: InferenceResult) -> tuple[SelectedCandidate, ...]:
        request_index = {candidate.candidate_id: index for index, candidate in enumerate(request.candidates)}
        request_by_id = {candidate.candidate_id: candidate for candidate in request.candidates}
        scored = tuple(result.candidates)
        if not scored:
            return ()
        selected: list[SelectedCandidate] = []
        selected_ids: set[str] = set()

        lead = self._personal_lead(scored, request_index)
        if lead is not None:
            selected.append(_selection(lead, RecommendationType.PERSONAL))
            selected_ids.add(lead.candidate_id)

        exploratory = self._exploratory(
            scored,
            selected_ids=selected_ids,
            request_by_id=request_by_id,
            request_index=request_index,
        )
        if exploratory is not None:
            selected.append(_selection(exploratory, RecommendationType.EXPLORATORY))
            selected_ids.add(exploratory.candidate_id)

        group = self._group(scored, selected_ids=selected_ids, request_index=request_index)
        if group is not None:
            selected.append(_selection(group, RecommendationType.GROUP_INSPIRED))

        return tuple(selected[: self._policy.max_results])

    def _personal_lead(
        self,
        scored: tuple[ScoredCandidate, ...],
        request_index: dict[str, int],
    ) -> ScoredCandidate | None:
        maximum = max(candidate.probability for candidate in scored)
        eligible = [
            candidate
            for candidate in scored
            if maximum - candidate.probability <= self._policy.lead_tie_band + 1e-12
            and (user_cf_supported(candidate) or preference_supported(candidate))
        ]
        if not eligible:
            return None
        return min(eligible, key=lambda candidate: _stable_key(candidate, request_index))

    def _exploratory(
        self,
        scored: tuple[ScoredCandidate, ...],
        *,
        selected_ids: set[str],
        request_by_id: dict[str, Candidate],
        request_index: dict[str, int],
    ) -> ScoredCandidate | None:
        eligible = [
            candidate
            for candidate in scored
            if candidate.candidate_id not in selected_ids and request_by_id[candidate.candidate_id].evidence.novelty > 0
        ]
        if not selected_ids and eligible:
            maximum = max(candidate.probability for candidate in eligible)
            eligible = [
                candidate
                for candidate in eligible
                if maximum - candidate.probability <= self._policy.lead_tie_band + 1e-12
            ]
        if not eligible:
            return None

        def utility_key(candidate: ScoredCandidate) -> tuple[float, float, float, int, str]:
            request_candidate = request_by_id[candidate.candidate_id]
            bonus = min(
                self._policy.novelty_bonus_cap,
                self._policy.novelty_bonus_multiplier * request_candidate.evidence.novelty,
            )
            penalty = sum(
                similarity_penalty(request_candidate, request_by_id[selected_id], self._policy)
                for selected_id in selected_ids
            )
            utility = candidate.probability + bonus - penalty
            return (
                -utility,
                -candidate.probability,
                -candidate.model_score,
                request_index[candidate.candidate_id],
                candidate.candidate_id,
            )

        return min(eligible, key=utility_key)

    def _group(
        self,
        scored: tuple[ScoredCandidate, ...],
        *,
        selected_ids: set[str],
        request_index: dict[str, int],
    ) -> ScoredCandidate | None:
        eligible = [
            candidate
            for candidate in scored
            if candidate.candidate_id not in selected_ids
            and candidate.evidence.group_preference_rate is not None
            and candidate.evidence.group_preference_rate >= self._policy.group_rate_threshold
            and candidate.evidence.group_eligible_member_count >= self._policy.group_member_threshold
        ]
        if not selected_ids and eligible:
            maximum = max(candidate.probability for candidate in eligible)
            eligible = [
                candidate
                for candidate in eligible
                if maximum - candidate.probability <= self._policy.lead_tie_band + 1e-12
            ]
        return min(eligible, key=lambda candidate: _stable_key(candidate, request_index)) if eligible else None


def _stable_key(candidate: ScoredCandidate, request_index: dict[str, int]) -> tuple[float, float, int, str]:
    return (
        -candidate.probability,
        -candidate.model_score,
        request_index[candidate.candidate_id],
        candidate.candidate_id,
    )


def _selection(candidate: ScoredCandidate, recommendation_type: RecommendationType) -> SelectedCandidate:
    return SelectedCandidate(candidate.candidate_id, recommendation_type, candidate.probability, candidate.model_score)
