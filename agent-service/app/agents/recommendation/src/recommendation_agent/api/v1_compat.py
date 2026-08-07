"""Explicit local bridge from Backend recommendation-agent-v1 to Agent v2."""

import base64
import hashlib
import hmac
import re
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from recommendation_agent.api.backpressure import RequestLimiter
from recommendation_agent.api.dependencies import get_agent_service, require_internal_service
from recommendation_agent.application.service import RecommendationAgentService
from recommendation_agent.schemas.agent_v1_compat import (
    V1Candidate,
    V1CandidateResponse,
    V1Request,
    V1Response,
)
from recommendation_agent.schemas.agent_v2 import AgentRequest, AgentResponse, Candidate

router = APIRouter(prefix="/internal/compat/v1/recommendations", tags=["local-v1-compatibility"])


@router.post(
    "/generate",
    response_model=V1Response,
    dependencies=[Depends(require_internal_service)],
)
async def generate_v1_compatible(
    request: Request,
    body: V1Request,
    service: Annotated[RecommendationAgentService, Depends(get_agent_service)],
) -> V1Response:
    settings = request.app.state.settings
    if not settings.enable_v1_compatibility:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="compatibility route is disabled")
    return await execute_v1_request(request, body, service)


async def execute_v1_request(
    request: Request,
    body: V1Request,
    service: RecommendationAgentService,
) -> V1Response:
    """Translate the Backend v1 wire envelope into the canonical v2 workflow."""

    settings = request.app.state.settings
    secret = settings.compatibility_model_key_secret.get_secret_value().encode("utf-8")
    decision_at = datetime.now(UTC)
    deadline_at = body.deadline_at
    minimum_deadline = decision_at + timedelta(milliseconds=settings.min_deadline_budget_ms + 50)
    if deadline_at <= minimum_deadline:
        deadline_at = decision_at + timedelta(seconds=2)
    agent_request = AgentRequest.model_validate(
        {
            "contractVersion": "recommendation-agent-v2",
            "featureSchemaVersion": "recommendation-features-v2",
            "requestId": body.request_id,
            "sessionId": body.session_id,
            "traceId": body.trace_id,
            "deadlineAt": deadline_at,
            "decisionAt": decision_at,
            "modelUserKey": _model_key(secret, f"v1-user-session:{body.session_id}"),
            "modelKeyVersion": "hmac-sha256-v1",
            "candidates": [
                _candidate_payload(candidate, body.preference_context, secret) for candidate in body.candidates
            ],
        }
    )
    limiter: RequestLimiter = request.app.state.request_limiter
    async with limiter.lease():
        response = await service.execute(agent_request)
    return _v1_response(body, response, agent_request)


def _candidate_payload(candidate: V1Candidate, preferences: dict[str, Any], secret: bytes) -> dict[str, Any]:
    features = candidate.features
    group_count = max(0, min(100, _integer(features.get("groupRecordCount"))))
    group_rate = _ratio_from_rating(features.get("groupAverageRating")) if group_count > 0 else None
    personal_count = max(0, _integer(features.get("personalRecordCount")))
    return {
        "candidateId": candidate.candidate_id,
        "modelMealKey": _model_key(secret, f"meal:{candidate.place_meal_id}"),
        "modelOfferingKey": _model_key(secret, f"offering:{candidate.place_meal_id}"),
        "evidence": {
            "preferenceMatch": _preference_match(features, preferences),
            "wantToTry": bool(features.get("wantToTry", False)),
            "groupPreferenceRate": group_rate,
            "groupEligibleMemberCount": group_count,
            # Every v1 candidate has already passed Backend hard context filters.
            "contextMatch": 1.0,
            "cleanlinessObserved": features.get("cleanlinessScore") is not None,
            "novelty": max(0.0, min(1.0, 1.0 / (1.0 + personal_count))),
            "cuisineCode": _code(features.get("cuisineCode"), "UNKNOWN"),
            "categoryCode": _code(features.get("mealType"), "MEAL"),
        },
    }


def _preference_match(features: dict[str, Any], preferences: dict[str, Any]) -> float:
    observations: list[bool] = []
    cuisine = _code(features.get("cuisineCode"), "UNKNOWN")
    liked_cuisines = {_code(value, "UNKNOWN") for value in _strings(preferences.get("likedCuisineCodes"))}
    if liked_cuisines:
        observations.append(cuisine in liked_cuisines)
    meal_type = _code(features.get("mealType"), "MEAL")
    preferred_meals = {_code(value, "MEAL") for value in _strings(preferences.get("preferredMealTypes"))}
    if preferred_meals:
        observations.append(meal_type in preferred_meals)
    if not observations:
        return 0.75
    return sum(observations) / len(observations)


def _v1_response(body: V1Request, response: AgentResponse, request: AgentRequest) -> V1Response:
    source_by_id = {candidate.candidate_id: candidate for candidate in body.candidates}
    v2_by_id: dict[str, Candidate] = {candidate.candidate_id: candidate for candidate in request.candidates}
    candidates = tuple(
        V1CandidateResponse.model_validate(
            {
                "candidateId": item.candidate_id,
                "rank": item.rank,
                "recommendationType": item.recommendation_type.value,
                "modelScore": item.probability,
                "reasonCodes": _v1_reason_codes(item.reasons, source_by_id[item.candidate_id].features),
                "explanation": item.explanation,
                "featureSnapshot": source_by_id[item.candidate_id].features,
            }
        )
        for item in response.recommendations
        if item.candidate_id in source_by_id and item.candidate_id in v2_by_id
    )
    return V1Response.model_validate(
        {
            "requestId": body.request_id,
            "sessionId": body.session_id,
            "traceId": body.trace_id,
            "agentTraceId": response.agent_trace_id,
            "modelVersion": response.model_version,
            "candidates": candidates,
        }
    )


def _v1_reason_codes(reasons: tuple[Any, ...], features: dict[str, Any]) -> tuple[str, ...]:
    mapped: list[str] = []
    for reason in reasons:
        code = getattr(reason, "value", str(reason))
        target = {
            "USER_CF": "SIMILAR_USERS_LIKED",
            "ITEM_CF": "SIMILAR_TO_LIKED_MEALS",
            "PREFERENCE_MATCH": "CUISINE_MATCH",
            "WANT_TO_TRY": "WANT_TO_TRY",
            "GROUP_POPULAR": "TRUSTED_GROUP_RATING",
            "CONTEXT_MATCH": "NEARBY" if features.get("distanceKm") is not None else "NOT_RECENTLY_REPEATED",
            "CLEANLINESS_OBSERVED": "NOT_RECENTLY_REPEATED",
        }.get(code, "NOT_RECENTLY_REPEATED")
        if target == "SIMILAR_USERS_LIKED" and _integer(features.get("groupRecordCount")) <= 0:
            target = "NOT_RECENTLY_REPEATED"
        if target == "SIMILAR_TO_LIKED_MEALS" and _integer(features.get("personalRecordCount")) <= 0:
            target = "NOT_RECENTLY_REPEATED"
        if target == "WANT_TO_TRY" and not bool(features.get("wantToTry")):
            target = "NOT_RECENTLY_REPEATED"
        if target not in mapped:
            mapped.append(target)
    return tuple(mapped or ["NOT_RECENTLY_REPEATED"])


def _model_key(secret: bytes, value: str) -> str:
    digest = hmac.new(secret, value.encode("utf-8"), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _code(value: object, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Z0-9_]", "_", str(value or fallback).upper()).strip("_")
    return (cleaned or fallback)[:32]


def _strings(value: object) -> tuple[str, ...]:
    return tuple(item for item in value if isinstance(item, str)) if isinstance(value, list) else ()


def _integer(value: object) -> int:
    if not isinstance(value, (str, int, float)):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _ratio_from_rating(value: object) -> float:
    if not isinstance(value, (str, int, float)):
        return 0.0
    try:
        return max(0.0, min(1.0, float(value) / 5.0))
    except (TypeError, ValueError):
        return 0.0
