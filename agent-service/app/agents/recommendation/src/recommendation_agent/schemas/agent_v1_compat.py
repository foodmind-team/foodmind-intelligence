"""Local-only compatibility models for the Backend's frozen v1 contract."""

from typing import Annotated, Any, Literal

from pydantic import BeforeValidator, Field, StrictStr

from recommendation_agent.schemas.agent_v2 import SafeId, StrictWireModel, UtcDateTime, _tuple_from_json


class V1Candidate(StrictWireModel):
    candidate_id: SafeId
    candidate_key: SafeId
    features: dict[str, Any]


class V1Request(StrictWireModel):
    contract_version: Literal["recommendation-agent-v1"]
    request_id: SafeId
    session_id: SafeId
    trace_id: SafeId
    deadline_at: UtcDateTime
    request_context: dict[str, Any]
    preference_context: dict[str, Any]
    candidates: Annotated[
        tuple[V1Candidate, ...],
        BeforeValidator(_tuple_from_json),
        Field(min_length=1, max_length=100),
    ]


class V1CandidateResponse(StrictWireModel):
    candidate_id: SafeId
    rank: int
    recommendation_type: Literal["PERSONAL", "EXPLORATORY", "GROUP_INSPIRED"]
    model_score: float
    reason_codes: tuple[StrictStr, ...]
    explanation: StrictStr
    feature_snapshot: dict[str, Any]


class V1Response(StrictWireModel):
    contract_version: Literal["recommendation-agent-v1"] = "recommendation-agent-v1"
    request_id: SafeId
    session_id: SafeId
    trace_id: SafeId
    agent_trace_id: SafeId
    status: Literal["SUCCEEDED"] = "SUCCEEDED"
    model_version: StrictStr
    feature_schema_version: Literal["recommendation-features-v1"] = "recommendation-features-v1"
    candidates: tuple[V1CandidateResponse, ...]
