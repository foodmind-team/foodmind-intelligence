"""DeepSeek-backed implementation of the recommendation inference port."""

from __future__ import annotations

import json
import math
from typing import Any

from recommendation_agent.config.settings import Settings
from recommendation_agent.domain.errors import AgentError, ErrorCode
from recommendation_agent.domain.models import (
    CollaborativeSignal,
    InferenceCommand,
    InferenceResult,
    ScoredCandidate,
)
from recommendation_agent.llm.client import LLMClient, LLMError
from recommendation_agent.time.budget import DeadlineBudget


class LLMRanker:
    """Ask DeepSeek for scores while retaining server-owned evidence and policies."""

    def __init__(self, *, client: LLMClient, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    async def score(self, command: InferenceCommand, *, budget: DeadlineBudget) -> InferenceResult:
        timeout = budget.downstream_timeout(
            configured_seconds=self._settings.llm_timeout_seconds,
            guard_seconds=self._settings.deadline_guard_ms / 1000.0,
        )
        try:
            raw = await self._client.chat_json(_messages(command), timeout_seconds=timeout)
            predictions = _validated_predictions(raw, command)
        except (LLMError, ValueError, TypeError) as exc:
            raise AgentError(ErrorCode.INFERENCE_UNAVAILABLE, http_status=503, retryable=True) from exc
        budget.ensure_remaining(guard_seconds=self._settings.deadline_guard_ms / 1000.0)
        return InferenceResult(
            model_version=self._settings.accepted_model_version,
            model_package_version=self._settings.accepted_model_package_version,
            feature_schema_version=self._settings.accepted_feature_schema_version,
            inference_contract_version=self._settings.accepted_inference_contract_version,
            model_key_version=self._settings.accepted_model_key_version,
            candidates=tuple(
                ScoredCandidate(
                    candidate_id=candidate.candidate_id,
                    probability=predictions[candidate.candidate_id][0],
                    model_score=predictions[candidate.candidate_id][1],
                    user_cf=CollaborativeSignal(False, None, 0),
                    item_cf=CollaborativeSignal(False, None, 0),
                    evidence=candidate.evidence,
                )
                for candidate in command.candidates
            ),
        )

    async def aclose(self) -> None:
        await self._client.aclose()


def _messages(command: InferenceCommand) -> list[dict[str, str]]:
    candidates = [
        {
            "candidateId": item.candidate_id,
            "evidence": {
                "preferenceMatch": item.evidence.preference_match,
                "wantToTry": item.evidence.want_to_try,
                "groupPreferenceRate": item.evidence.group_preference_rate,
                "groupEligibleMemberCount": item.evidence.group_eligible_member_count,
                "contextMatch": item.evidence.context_match,
                "cleanlinessObserved": item.evidence.cleanliness_observed,
            },
        }
        for item in command.candidates
    ]
    system = """You are a bounded food recommendation ranking model.
Rank only the supplied opaque candidate IDs from their evidence. Do not add or omit candidates.
Return one JSON object with a predictions array. Every prediction must contain candidateId, probability, and modelScore.
probability and modelScore must each be JSON numbers from 0.0 to 1.0. Higher means a better contextual match.
Do not include prose, markdown, personal data, health claims, or candidate facts not present in the evidence."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps({"candidates": candidates}, separators=(",", ":"))},
    ]


def _validated_predictions(raw: dict[str, Any], command: InferenceCommand) -> dict[str, tuple[float, float]]:
    items = raw.get("predictions")
    if not isinstance(items, list):
        raise ValueError("predictions must be an array")
    expected = {item.candidate_id for item in command.candidates}
    result: dict[str, tuple[float, float]] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("prediction must be an object")
        candidate_id = item.get("candidateId")
        probability = item.get("probability")
        model_score = item.get("modelScore")
        if not isinstance(candidate_id, str) or candidate_id not in expected or candidate_id in result:
            raise ValueError("prediction candidate IDs do not match the request")
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or isinstance(model_score, bool)
            or not isinstance(model_score, (int, float))
        ):
            raise ValueError("scores must be numbers")
        probability = float(probability)
        model_score = float(model_score)
        if not math.isfinite(probability) or not math.isfinite(model_score):
            raise ValueError("scores must be finite")
        if not 0.0 <= probability <= 1.0 or not 0.0 <= model_score <= 1.0:
            raise ValueError("scores must be between zero and one")
        result[candidate_id] = (probability, model_score)
    if set(result) != expected:
        raise ValueError("every requested candidate must be scored exactly once")
    return result
