"""DeepSeek explanation rendering over already-selected, policy-approved facts."""

from __future__ import annotations

import json
import re
from typing import Any

from recommendation_agent.config.settings import Settings
from recommendation_agent.domain.models import ReasonedCandidate, RenderedCandidate
from recommendation_agent.llm.client import LLMClient, LLMError
from recommendation_agent.reasons.renderer import DeterministicExplanationRenderer, validate_explanation
from recommendation_agent.reasons.templates import TEMPLATES
from recommendation_agent.time.budget import DeadlineBudget

_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_NEUTRAL_WORDS = {
    "a",
    "also",
    "and",
    "available",
    "because",
    "based",
    "candidate",
    "choice",
    "current",
    "evidence",
    "for",
    "is",
    "on",
    "recommendation",
    "signals",
    "the",
    "this",
    "you",
}


class LLMExplanationRenderer:
    """Use one bounded LLM call without allowing it to alter ranking or facts."""

    def __init__(self, *, client: LLMClient, settings: Settings) -> None:
        self._client = client
        self._settings = settings
        self._fallback = DeterministicExplanationRenderer()

    async def render(
        self,
        candidates: tuple[ReasonedCandidate, ...],
        *,
        budget: DeadlineBudget | None = None,
    ) -> tuple[RenderedCandidate, ...]:
        fallback = await self._fallback.render(candidates, budget=budget)
        if not candidates or budget is None:
            return fallback
        try:
            timeout = budget.downstream_timeout(
                configured_seconds=self._settings.llm_timeout_seconds,
                guard_seconds=self._settings.deadline_guard_ms / 1000.0,
            )
            raw = await self._client.chat_json(_messages(candidates), timeout_seconds=timeout)
            rendered = _validated_explanations(raw, candidates)
            budget.ensure_remaining(guard_seconds=self._settings.deadline_guard_ms / 1000.0)
            return rendered
        except (LLMError, ValueError, TypeError):
            return fallback


def _messages(candidates: tuple[ReasonedCandidate, ...]) -> list[dict[str, str]]:
    items = [
        {
            "candidateId": item.selection.candidate_id,
            "allowedFacts": [TEMPLATES[reason] for reason in item.reasons],
        }
        for item in candidates
    ]
    system = """You write short FoodMind recommendation explanations.
The candidate order, scores, reason codes, and allowed facts are already final.
Return exactly one explanation for every supplied candidateId and no other IDs.
Use only facts and ordinary connecting words present in the supplied allowedFacts.
Do not add prices, distance, nutrition, health, safety, quality, availability, or personal facts.
Return JSON only: {\"explanations\":[{\"candidateId\":\"...\",\"explanation\":\"...\"}]}.
Each explanation must be plain ASCII text, at most 160 characters, with no markup."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps({"candidates": items}, separators=(",", ":"))},
    ]


def _validated_explanations(
    raw: dict[str, Any],
    candidates: tuple[ReasonedCandidate, ...],
) -> tuple[RenderedCandidate, ...]:
    items = raw.get("explanations")
    if not isinstance(items, list):
        raise ValueError("explanations must be an array")
    expected = {item.selection.candidate_id: item for item in candidates}
    explanations: dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("explanation must be an object")
        candidate_id = item.get("candidateId")
        explanation = item.get("explanation")
        if (
            not isinstance(candidate_id, str)
            or candidate_id not in expected
            or candidate_id in explanations
            or not isinstance(explanation, str)
        ):
            raise ValueError("explanation candidate IDs do not match the selected candidates")
        normalized = validate_explanation(explanation)
        allowed_words = set(_NEUTRAL_WORDS)
        for reason in expected[candidate_id].reasons:
            allowed_words.update(word.casefold() for word in _WORD.findall(TEMPLATES[reason]))
        supplied_words = {word.casefold() for word in _WORD.findall(normalized)}
        if not supplied_words or not supplied_words.issubset(allowed_words):
            raise ValueError("explanation contains a fact outside the approved reason vocabulary")
        explanations[candidate_id] = normalized
    if set(explanations) != set(expected):
        raise ValueError("every selected candidate must be explained exactly once")
    return tuple(
        RenderedCandidate(candidate, explanations[candidate.selection.candidate_id]) for candidate in candidates
    )
