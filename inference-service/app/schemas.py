from __future__ import annotations

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
    model_validator,
)


def _to_camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part.capitalize() for part in tail)


def _utc_datetime(value: object) -> datetime:
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise ValueError("timestamp must be ISO 8601")
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError("timestamp must use UTC")
    return parsed.astimezone(UTC)


def _tuple_from_json(value: object) -> object:
    return tuple(value) if isinstance(value, list) else value


class WireModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=_to_camel,
        extra="forbid",
        frozen=True,
        populate_by_name=False,
        strict=True,
    )


SafeId = Annotated[StrictStr, StringConstraints(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")]
ModelKey = Annotated[StrictStr, StringConstraints(min_length=16, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")]
ModelUserKey = Annotated[StrictStr, StringConstraints(min_length=43, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")]
Ratio = Annotated[StrictFloat, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
UtcDateTime = Annotated[datetime, BeforeValidator(_utc_datetime)]


class Evidence(WireModel):
    preference_match: Ratio
    want_to_try: StrictBool
    group_preference_rate: Ratio | None
    group_eligible_member_count: Annotated[StrictInt, Field(ge=0, le=100)]
    context_match: Ratio | None
    cleanliness_observed: StrictBool

    @model_validator(mode="after")
    def validate_group_state(self) -> Self:
        if (self.group_eligible_member_count == 0) != (self.group_preference_rate is None):
            raise ValueError("group evidence availability is inconsistent")
        return self


class Candidate(WireModel):
    candidate_id: SafeId
    model_meal_key: ModelKey = Field(repr=False)
    model_offering_key: ModelKey = Field(repr=False)
    evidence: Evidence = Field(repr=False)


class InferenceRequest(WireModel):
    contract_version: Literal["recommendation-inference-v1"]
    request_id: SafeId
    trace_id: SafeId
    deadline_at: UtcDateTime
    feature_schema_version: Literal["recommendation-features-v2"]
    model_user_key: ModelUserKey = Field(repr=False)
    model_key_version: Literal["hmac-sha256-v1"]
    candidates: Annotated[
        tuple[Candidate, ...],
        BeforeValidator(_tuple_from_json),
        Field(min_length=1, max_length=100),
    ]


class UserCf(WireModel):
    available: Literal[False] = False
    score: None = None
    neighbor_support: Literal[0] = 0


class ItemCf(WireModel):
    available: Literal[False] = False
    score: None = None
    supporting_item_count: Literal[0] = 0


class Prediction(WireModel):
    candidate_id: SafeId
    probability: Ratio
    model_score: Annotated[StrictFloat, Field(ge=-100.0, le=100.0, allow_inf_nan=False)]
    user_cf: UserCf = UserCf()
    item_cf: ItemCf = ItemCf()
    signals: Evidence


class InferenceSuccess(WireModel):
    contract_version: Literal["recommendation-inference-v1"] = "recommendation-inference-v1"
    request_id: SafeId
    trace_id: SafeId
    status: Literal["success"] = "success"
    model_version: Literal["hybrid-ranking-v1"] = "hybrid-ranking-v1"
    model_package_version: Literal["recommendation-package-v1"] = "recommendation-package-v1"
    feature_schema_version: Literal["recommendation-features-v2"] = "recommendation-features-v2"
    model_key_version: Literal["hmac-sha256-v1"] = "hmac-sha256-v1"
    predictions: tuple[Prediction, ...]


class FailureDetail(WireModel):
    code: Literal["MODEL_PACKAGE_INCOMPATIBLE", "INFERENCE_UNAVAILABLE"]


class InferenceFailure(WireModel):
    contract_version: Literal["recommendation-inference-v1"] = "recommendation-inference-v1"
    request_id: SafeId
    trace_id: SafeId
    status: Literal["failure"] = "failure"
    model_version: StrictStr = "hybrid-ranking-v1"
    model_package_version: StrictStr = "recommendation-package-v1"
    feature_schema_version: StrictStr = "recommendation-features-v2"
    model_key_version: StrictStr = "hmac-sha256-v1"
    error: FailureDetail
