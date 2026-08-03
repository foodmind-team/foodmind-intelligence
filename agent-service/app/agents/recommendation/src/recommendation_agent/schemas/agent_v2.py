"""Strict Pydantic mirrors of the Recommendation Agent v2 wire contracts."""

from datetime import UTC, datetime
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    StringConstraints,
    field_validator,
    model_validator,
)

from recommendation_agent.domain.errors import ErrorCode
from recommendation_agent.domain.models import ReasonCode, RecommendationType


def _to_camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part.capitalize() for part in tail)


def _parse_utc_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        if len(value) > 35:
            raise ValueError("timestamp is too long")
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("timestamp must be ISO 8601") from exc
    else:
        raise ValueError("timestamp must be a string")
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError("timestamp must be UTC")
    return parsed.astimezone(UTC)


def _tuple_from_json(value: object) -> object:
    if isinstance(value, list):
        return tuple(value)
    return value


SafeId = Annotated[
    StrictStr,
    StringConstraints(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$"),
]
ModelKey = Annotated[
    StrictStr,
    StringConstraints(min_length=16, max_length=64, pattern=r"^[A-Za-z0-9_-]+$"),
]
ModelUserKey = Annotated[
    StrictStr,
    StringConstraints(min_length=43, max_length=64, pattern=r"^[A-Za-z0-9_-]+$"),
]
Ratio = Annotated[StrictFloat, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
ModelScore = Annotated[StrictFloat, Field(ge=-100.0, le=100.0, allow_inf_nan=False)]
UtcDateTime = Annotated[datetime, BeforeValidator(_parse_utc_datetime)]


class StrictWireModel(BaseModel):
    """Immutable camelCase model with no coercion or unknown fields."""

    model_config = ConfigDict(
        alias_generator=_to_camel,
        extra="forbid",
        frozen=True,
        populate_by_name=False,
        strict=True,
    )


class CandidateEvidence(StrictWireModel):
    preference_match: Ratio
    want_to_try: StrictBool
    group_preference_rate: Ratio | None
    group_eligible_member_count: Annotated[StrictInt, Field(ge=0, le=100)]
    context_match: Ratio | None
    cleanliness_observed: StrictBool
    novelty: Ratio
    cuisine_code: Annotated[StrictStr, StringConstraints(min_length=1, max_length=32, pattern=r"^[A-Z0-9_]+$")]
    category_code: Annotated[StrictStr, StringConstraints(min_length=1, max_length=32, pattern=r"^[A-Z0-9_]+$")]

    @model_validator(mode="after")
    def validate_group_optional_state(self) -> Self:
        if self.group_eligible_member_count == 0 and self.group_preference_rate is not None:
            raise ValueError("group rate must be null when no eligible members exist")
        if self.group_eligible_member_count > 0 and self.group_preference_rate is None:
            raise ValueError("group rate is required when eligible members exist")
        return self


class Candidate(StrictWireModel):
    candidate_id: SafeId
    model_meal_key: ModelKey = Field(repr=False)
    model_offering_key: ModelKey = Field(repr=False)
    evidence: CandidateEvidence = Field(repr=False)


class AgentRequest(StrictWireModel):
    contract_version: Literal["recommendation-agent-v2"]
    feature_schema_version: Literal["recommendation-features-v2"]
    request_id: SafeId
    session_id: SafeId
    trace_id: SafeId
    deadline_at: UtcDateTime
    decision_at: UtcDateTime
    model_user_key: ModelUserKey = Field(repr=False)
    model_key_version: Literal["hmac-sha256-v1"]
    candidates: Annotated[
        tuple[Candidate, ...],
        BeforeValidator(_tuple_from_json),
        Field(min_length=1, max_length=100),
    ]

    @model_validator(mode="after")
    def validate_request_integrity(self) -> Self:
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate IDs must be unique")
        if self.decision_at > self.deadline_at:
            raise ValueError("decision timestamp must not follow deadline")
        return self


_FORBIDDEN_CLAIMS = (
    "guaranteed",
    "best",
    "healthy",
    "healthiest",
    "safe",
    "safest",
    "allergen-free",
    "allergy-safe",
    "medical",
    "perfect",
)


class Recommendation(StrictWireModel):
    candidate_id: SafeId
    rank: Annotated[StrictInt, Field(ge=1, le=3)]
    recommendation_type: RecommendationType
    probability: Ratio
    model_score: ModelScore
    reasons: Annotated[
        tuple[ReasonCode, ...],
        BeforeValidator(_tuple_from_json),
        Field(min_length=1, max_length=2),
    ]
    explanation: Annotated[
        StrictStr,
        StringConstraints(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9 ,.'()-]+$"),
    ]

    @field_validator("reasons")
    @classmethod
    def reasons_are_unique(cls, value: tuple[ReasonCode, ...]) -> tuple[ReasonCode, ...]:
        if len(value) != len(set(value)):
            raise ValueError("reason codes must be unique")
        return value

    @field_validator("explanation")
    @classmethod
    def explanation_is_safe(cls, value: str) -> str:
        lowered = value.casefold()
        if any(claim in lowered for claim in _FORBIDDEN_CLAIMS):
            raise ValueError("explanation contains a forbidden claim")
        return value


class AgentResponse(StrictWireModel):
    contract_version: Literal["recommendation-agent-v2"] = "recommendation-agent-v2"
    request_id: SafeId
    session_id: SafeId
    trace_id: SafeId
    agent_trace_id: SafeId
    status: Literal["success"] = "success"
    model_version: Literal["hybrid-ranking-v1"] = "hybrid-ranking-v1"
    model_package_version: Literal["recommendation-package-v1"] = "recommendation-package-v1"
    feature_schema_version: Literal["recommendation-features-v2"] = "recommendation-features-v2"
    inference_contract_version: Literal["recommendation-inference-v1"] = "recommendation-inference-v1"
    model_key_version: Literal["hmac-sha256-v1"] = "hmac-sha256-v1"
    diversity_policy_version: Literal["recommendation-diversity-v1"] = "recommendation-diversity-v1"
    reason_policy_version: Literal["recommendation-reasons-v1"] = "recommendation-reasons-v1"
    template_version: Literal["recommendation-template-v1"] = "recommendation-template-v1"
    recommendations: Annotated[tuple[Recommendation, ...], BeforeValidator(_tuple_from_json), Field(max_length=3)]

    @model_validator(mode="after")
    def validate_result_integrity(self) -> Self:
        ids = [item.candidate_id for item in self.recommendations]
        if len(ids) != len(set(ids)):
            raise ValueError("result candidate IDs must be unique")
        ranks = [item.rank for item in self.recommendations]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("result ranks must be contiguous and ordered")
        return self


class FailureDetail(StrictWireModel):
    code: ErrorCode
    retryable: StrictBool


class AgentFailure(StrictWireModel):
    contract_version: Literal["recommendation-agent-v2"] = "recommendation-agent-v2"
    request_id: SafeId
    session_id: SafeId
    trace_id: SafeId
    agent_trace_id: SafeId
    status: Literal["failure"] = "failure"
    error: FailureDetail
