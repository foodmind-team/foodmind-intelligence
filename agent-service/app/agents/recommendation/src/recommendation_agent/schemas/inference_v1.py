"""Strict consumer models for recommendation-inference-v1."""

from typing import Annotated, Literal, Self

from pydantic import BeforeValidator, Field, StrictBool, StrictFloat, StrictInt, StrictStr, TypeAdapter, model_validator

from recommendation_agent.schemas.agent_v2 import (
    ModelKey,
    ModelScore,
    ModelUserKey,
    Ratio,
    SafeId,
    StrictWireModel,
    UtcDateTime,
    _tuple_from_json,
)


class InferenceEvidenceWire(StrictWireModel):
    preference_match: Ratio
    want_to_try: StrictBool
    group_preference_rate: Ratio | None
    group_eligible_member_count: Annotated[StrictInt, Field(ge=0, le=100)]
    context_match: Ratio | None
    cleanliness_observed: StrictBool

    @model_validator(mode="after")
    def validate_group_state(self) -> Self:
        if (self.group_eligible_member_count == 0) != (self.group_preference_rate is None):
            raise ValueError("group evidence availability is ambiguous")
        return self


class InferenceCandidateWire(StrictWireModel):
    candidate_id: SafeId
    model_meal_key: ModelKey = Field(repr=False)
    model_offering_key: ModelKey = Field(repr=False)
    evidence: InferenceEvidenceWire = Field(repr=False)


class InferenceRequest(StrictWireModel):
    contract_version: Literal["recommendation-inference-v1"] = "recommendation-inference-v1"
    request_id: SafeId
    trace_id: SafeId
    deadline_at: UtcDateTime
    feature_schema_version: Literal["recommendation-features-v2"]
    model_user_key: ModelUserKey = Field(repr=False)
    model_key_version: Literal["hmac-sha256-v1"]
    candidates: Annotated[
        tuple[InferenceCandidateWire, ...], BeforeValidator(_tuple_from_json), Field(min_length=1, max_length=100)
    ]


class CfSignal(StrictWireModel):
    available: StrictBool
    score: Ratio | None

    @model_validator(mode="after")
    def validate_availability(self) -> Self:
        if self.available != (self.score is not None):
            raise ValueError("CF availability and score are inconsistent")
        return self


class UserCfSignal(CfSignal):
    neighbor_support: Annotated[StrictInt, Field(ge=0, le=1_000_000)]

    @model_validator(mode="after")
    def validate_support(self) -> Self:
        if self.available != (self.neighbor_support > 0):
            raise ValueError("UserCF availability and support are inconsistent")
        return self


class ItemCfSignal(CfSignal):
    supporting_item_count: Annotated[StrictInt, Field(ge=0, le=1_000_000)]

    @model_validator(mode="after")
    def validate_support(self) -> Self:
        if self.available != (self.supporting_item_count > 0):
            raise ValueError("ItemCF availability and support are inconsistent")
        return self


class InferencePrediction(StrictWireModel):
    candidate_id: SafeId
    probability: Ratio
    model_score: ModelScore
    user_cf: UserCfSignal
    item_cf: ItemCfSignal
    signals: InferenceEvidenceWire = Field(repr=False)


class InferenceSuccess(StrictWireModel):
    contract_version: Literal["recommendation-inference-v1"]
    request_id: SafeId
    trace_id: SafeId
    status: Literal["success"]
    model_version: Literal["hybrid-ranking-v1"]
    model_package_version: Literal["recommendation-package-v1"]
    feature_schema_version: Literal["recommendation-features-v2"]
    model_key_version: Literal["hmac-sha256-v1"]
    predictions: Annotated[tuple[InferencePrediction, ...], BeforeValidator(_tuple_from_json), Field(max_length=100)]


class InferenceFailureDetail(StrictWireModel):
    code: Literal["MODEL_PACKAGE_INCOMPATIBLE", "INFERENCE_UNAVAILABLE"]


class InferenceFailure(StrictWireModel):
    contract_version: Literal["recommendation-inference-v1"]
    request_id: SafeId
    trace_id: SafeId
    status: Literal["failure"]
    model_version: StrictStr
    model_package_version: StrictStr
    feature_schema_version: StrictStr
    model_key_version: StrictStr
    error: InferenceFailureDetail


InferenceResponse = Annotated[InferenceSuccess | InferenceFailure, Field(discriminator="status")]
INFERENCE_RESPONSE_ADAPTER: TypeAdapter[InferenceResponse] = TypeAdapter(InferenceResponse)


FiniteFloat = Annotated[StrictFloat, Field(allow_inf_nan=False)]
