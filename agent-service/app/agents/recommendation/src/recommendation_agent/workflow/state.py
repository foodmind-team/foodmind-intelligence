"""Typed workflow state and its secret-minimized diagnostic serialization."""

from dataclasses import asdict, dataclass
from typing import Any, TypedDict

from recommendation_agent.domain.errors import ErrorCode
from recommendation_agent.domain.models import InferenceResult, ReasonedCandidate, RenderedCandidate, SelectedCandidate
from recommendation_agent.schemas.agent_v2 import AgentFailure, AgentRequest, AgentResponse


@dataclass(frozen=True, slots=True)
class FailureRecord:
    code: ErrorCode
    http_status: int
    retryable: bool = False


class RecommendationState(TypedDict, total=False):
    request: AgentRequest
    agent_trace_id: str
    deadline_expiry: float
    inference_calls: int
    inference_result: InferenceResult
    compatibility_validated: bool
    selections: tuple[SelectedCandidate, ...]
    reasoned_candidates: tuple[ReasonedCandidate, ...]
    rendered_candidates: tuple[RenderedCandidate, ...]
    response: AgentResponse
    failure_response: AgentFailure
    failure: FailureRecord
    node_trace: tuple[str, ...]


def serialize_state(state: RecommendationState) -> dict[str, Any]:
    """Create JSON-safe diagnostic state without model keys, features, clients, or errors."""

    request = state.get("request")
    result = state.get("inference_result")
    serialized: dict[str, Any] = {
        "agentTraceId": state.get("agent_trace_id"),
        "deadlineExpiry": state.get("deadline_expiry"),
        "inferenceCalls": state.get("inference_calls", 0),
        "compatibilityValidated": state.get("compatibility_validated", False),
        "nodeTrace": list(state.get("node_trace", ())),
    }
    if request is not None:
        serialized["request"] = {
            "contractVersion": request.contract_version,
            "featureSchemaVersion": request.feature_schema_version,
            "requestId": request.request_id,
            "sessionId": request.session_id,
            "traceId": request.trace_id,
            "deadlineAt": request.deadline_at.isoformat(),
            "candidateIds": [candidate.candidate_id for candidate in request.candidates],
        }
    if result is not None:
        serialized["inference"] = {
            "modelVersion": result.model_version,
            "modelPackageVersion": result.model_package_version,
            "featureSchemaVersion": result.feature_schema_version,
            "inferenceContractVersion": result.inference_contract_version,
            "modelKeyVersion": result.model_key_version,
            "candidateIds": [candidate.candidate_id for candidate in result.candidates],
        }
    if "selections" in state:
        serialized["selections"] = [asdict(selection) for selection in state["selections"]]
    if "failure" in state:
        failure = state["failure"]
        serialized["failure"] = {
            "code": failure.code.value,
            "httpStatus": failure.http_status,
            "retryable": failure.retryable,
        }
    if "response" in state:
        serialized["response"] = state["response"].model_dump(mode="json", by_alias=True)
    if "failure_response" in state:
        serialized["failureResponse"] = state["failure_response"].model_dump(mode="json", by_alias=True)
    return serialized
