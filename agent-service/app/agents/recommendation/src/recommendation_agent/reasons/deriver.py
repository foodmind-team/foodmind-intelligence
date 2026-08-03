"""Deterministic reason derivation from the frozen predicate table."""

from recommendation_agent.domain.errors import AgentError, ErrorCode
from recommendation_agent.domain.models import InferenceResult, ReasonedCandidate, SelectedCandidate
from recommendation_agent.policy.reason_predicates import PREDICATES, REASON_POLICY, ReasonPolicy
from recommendation_agent.schemas.agent_v2 import AgentRequest


class DeterministicReasonDeriver:
    def __init__(self, policy: ReasonPolicy = REASON_POLICY) -> None:
        policy.validate()
        self._policy = policy

    async def derive(
        self,
        request: AgentRequest,
        result: InferenceResult,
        selections: tuple[SelectedCandidate, ...],
    ) -> tuple[ReasonedCandidate, ...]:
        del request
        scored_by_id = {candidate.candidate_id: candidate for candidate in result.candidates}
        reasoned: list[ReasonedCandidate] = []
        for selection in selections:
            candidate = scored_by_id.get(selection.candidate_id)
            if candidate is None:
                raise AgentError(ErrorCode.UNKNOWN_CANDIDATE, http_status=500)
            reasons = tuple(reason for reason in self._policy.priority if PREDICATES[reason](candidate, self._policy))[
                : self._policy.max_reasons
            ]
            if not reasons:
                raise AgentError(ErrorCode.UNSUPPORTED_REASON, http_status=500)
            reasoned.append(ReasonedCandidate(selection, reasons))
        return tuple(reasoned)
