"""Framework-free domain values crossing the inference port."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class RecommendationType(StrEnum):
    PERSONAL = "PERSONAL"
    EXPLORATORY = "EXPLORATORY"
    GROUP_INSPIRED = "GROUP_INSPIRED"


class ReasonCode(StrEnum):
    USER_CF = "USER_CF"
    ITEM_CF = "ITEM_CF"
    PREFERENCE_MATCH = "PREFERENCE_MATCH"
    WANT_TO_TRY = "WANT_TO_TRY"
    GROUP_POPULAR = "GROUP_POPULAR"
    CONTEXT_MATCH = "CONTEXT_MATCH"
    CLEANLINESS_OBSERVED = "CLEANLINESS_OBSERVED"


@dataclass(frozen=True, slots=True)
class InferenceEvidence:
    preference_match: float
    want_to_try: bool
    group_preference_rate: float | None
    group_eligible_member_count: int
    context_match: float | None
    cleanliness_observed: bool


@dataclass(frozen=True, slots=True)
class InferenceCandidate:
    candidate_id: str
    model_meal_key: str = field(repr=False)
    model_offering_key: str = field(repr=False)
    evidence: InferenceEvidence = field(repr=False)


@dataclass(frozen=True, slots=True)
class InferenceCommand:
    request_id: str
    trace_id: str
    deadline_at: datetime
    feature_schema_version: str
    model_user_key: str = field(repr=False)
    model_key_version: str = field(repr=False)
    candidates: tuple[InferenceCandidate, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class CollaborativeSignal:
    available: bool
    score: float | None
    support: int


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    candidate_id: str
    probability: float
    model_score: float
    user_cf: CollaborativeSignal
    item_cf: CollaborativeSignal
    evidence: InferenceEvidence = field(repr=False)


@dataclass(frozen=True, slots=True)
class InferenceResult:
    model_version: str
    model_package_version: str
    feature_schema_version: str
    inference_contract_version: str
    model_key_version: str
    candidates: tuple[ScoredCandidate, ...]


@dataclass(frozen=True, slots=True)
class SelectedCandidate:
    candidate_id: str
    recommendation_type: RecommendationType
    probability: float
    model_score: float


@dataclass(frozen=True, slots=True)
class ReasonedCandidate:
    selection: SelectedCandidate
    reasons: tuple[ReasonCode, ...]


@dataclass(frozen=True, slots=True)
class RenderedCandidate:
    reasoned: ReasonedCandidate
    explanation: str
