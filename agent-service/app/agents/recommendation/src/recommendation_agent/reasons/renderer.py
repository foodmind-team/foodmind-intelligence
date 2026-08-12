"""Template-only explanation rendering with defense-in-depth validation."""

import re

from recommendation_agent.domain.errors import AgentError, ErrorCode
from recommendation_agent.domain.models import ReasonedCandidate, RenderedCandidate
from recommendation_agent.policy.reason_predicates import REASON_POLICY, ReasonPolicy
from recommendation_agent.reasons.templates import TEMPLATES
from recommendation_agent.time.budget import DeadlineBudget

_FORBIDDEN = re.compile(
    r"\b(?:safe|safest|allergen-free|allergy-safe|healthy|healthiest|medical|guaranteed|best|perfect)\b",
    re.IGNORECASE,
)
_MARKUP_OR_CONTROL = re.compile(r"[<>\[\]{}\x00-\x1f\x7f]")


def validate_explanation(value: str) -> str:
    if _MARKUP_OR_CONTROL.search(value):
        raise AgentError(ErrorCode.UNSAFE_TEMPLATE, http_status=500)
    normalized = " ".join(value.split())
    if (
        not normalized
        or not normalized.isascii()
        or len(normalized) > 160
        or len(normalized.encode("utf-8")) > 240
        or _FORBIDDEN.search(normalized)
    ):
        raise AgentError(ErrorCode.UNSAFE_TEMPLATE, http_status=500)
    return normalized


class DeterministicExplanationRenderer:
    def __init__(self, policy: ReasonPolicy = REASON_POLICY) -> None:
        policy.validate()
        self._policy = policy

    async def render(
        self,
        candidates: tuple[ReasonedCandidate, ...],
        *,
        budget: DeadlineBudget | None = None,
    ) -> tuple[RenderedCandidate, ...]:
        del budget
        rendered: list[RenderedCandidate] = []
        for candidate in candidates:
            reasons = candidate.reasons
            if not reasons or len(reasons) > self._policy.max_reasons or len(reasons) != len(set(reasons)):
                raise AgentError(ErrorCode.UNSUPPORTED_REASON, http_status=500)
            try:
                explanation = " ".join(TEMPLATES[reason] for reason in reasons)
            except KeyError as exc:
                raise AgentError(ErrorCode.UNSUPPORTED_REASON, http_status=500) from exc
            rendered.append(RenderedCandidate(candidate, validate_explanation(explanation)))
        return tuple(rendered)
