"""Purely bounded workflow nodes; only score_once invokes inference."""

import math
from collections.abc import Awaitable
from typing import Any

from pydantic import ValidationError

from recommendation_agent.clients.inference import command_from_agent_request
from recommendation_agent.domain.errors import AgentError, ErrorCode
from recommendation_agent.domain.models import InferenceResult, ReasonedCandidate, RenderedCandidate, SelectedCandidate
from recommendation_agent.schemas.agent_v2 import AgentFailure, AgentResponse
from recommendation_agent.time.budget import DeadlineBudget
from recommendation_agent.workflow.context import WorkflowContext
from recommendation_agent.workflow.state import FailureRecord, RecommendationState


class WorkflowNodes:
    def __init__(self, context: WorkflowContext) -> None:
        self._context = context

    def _trace(self, state: RecommendationState, node: str) -> tuple[str, ...]:
        return (*state.get("node_trace", ()), node)

    def _failure(self, error: AgentError, *, trace: tuple[str, ...]) -> dict[str, Any]:
        return {
            "failure": FailureRecord(error.code, error.http_status, error.retryable),
            "node_trace": trace,
        }

    def _budget(self, state: RecommendationState) -> DeadlineBudget:
        expiry = state.get("deadline_expiry")
        if expiry is None:
            raise AgentError(ErrorCode.DEADLINE_EXPIRED, http_status=408)
        return DeadlineBudget.from_monotonic_expiry(expiry, clock=self._context.clock)

    async def validate_envelope(self, state: RecommendationState) -> dict[str, Any]:
        trace = self._trace(state, "validate_envelope")
        if "failure" in state:
            return {"node_trace": trace}
        request = state["request"]
        try:
            if request.contract_version not in self._context.settings.supported_contract_versions:
                raise AgentError(ErrorCode.UNSUPPORTED_AGENT_VERSION, http_status=400)
            if request.feature_schema_version != self._context.settings.accepted_feature_schema_version:
                raise AgentError(ErrorCode.UNSUPPORTED_FEATURE_VERSION, http_status=400)
            if request.model_key_version != self._context.settings.accepted_model_key_version:
                raise AgentError(ErrorCode.MODEL_KEY_VERSION_MISMATCH, http_status=400)
            if len(request.candidates) > self._context.settings.max_candidates:
                raise AgentError(ErrorCode.REQUEST_TOO_LARGE, http_status=413)
            candidate_ids = [candidate.candidate_id for candidate in request.candidates]
            if len(candidate_ids) != len(set(candidate_ids)):
                raise AgentError(ErrorCode.DUPLICATE_CANDIDATE, http_status=400)
            self._budget(state).ensure_remaining()
        except AgentError as error:
            return self._failure(error, trace=trace)
        return {"node_trace": trace}

    async def score_once(self, state: RecommendationState) -> dict[str, Any]:
        trace = self._trace(state, "score_once")
        calls = state.get("inference_calls", 0)
        if calls != 0:
            return {
                **self._failure(AgentError(ErrorCode.INTERNAL_ERROR, http_status=500), trace=trace),
                "inference_calls": calls,
            }
        try:
            result = await self._context.inference.score(
                command_from_agent_request(state["request"]),
                budget=self._budget(state),
            )
        except AgentError as error:
            return {**self._failure(error, trace=trace), "inference_calls": 1}
        except Exception:  # noqa: BLE001 - port boundary maps unknown failures without retaining details
            return {
                **self._failure(
                    AgentError(ErrorCode.INFERENCE_CONNECTION_FAILED, http_status=502, retryable=True), trace=trace
                ),
                "inference_calls": 1,
            }
        return {"inference_result": result, "inference_calls": 1, "node_trace": trace}

    async def validate_compatibility(self, state: RecommendationState) -> dict[str, Any]:
        trace = self._trace(state, "validate_compatibility")
        try:
            result = state["inference_result"]
            request = state["request"]
            self._validate_versions(result)
            request_ids = {candidate.candidate_id for candidate in request.candidates}
            result_ids = [candidate.candidate_id for candidate in result.candidates]
            if len(result_ids) != len(set(result_ids)):
                raise AgentError(ErrorCode.DUPLICATE_CANDIDATE, http_status=502)
            if any(candidate_id not in request_ids for candidate_id in result_ids):
                raise AgentError(ErrorCode.UNKNOWN_CANDIDATE, http_status=502)
            if set(result_ids) != request_ids:
                raise AgentError(ErrorCode.MISSING_CANDIDATE, http_status=502)
            for candidate in result.candidates:
                if not math.isfinite(candidate.probability) or not 0 <= candidate.probability <= 1:
                    raise AgentError(ErrorCode.INVALID_PROBABILITY, http_status=502)
                for signal in (candidate.user_cf, candidate.item_cf):
                    if signal.available != (signal.score is not None and signal.support > 0):
                        raise AgentError(ErrorCode.INVALID_EVIDENCE, http_status=502)
            self._budget(state).ensure_remaining(guard_seconds=self._context.settings.deadline_guard_ms / 1000.0)
        except AgentError as error:
            return self._failure(error, trace=trace)
        return {"compatibility_validated": True, "node_trace": trace}

    def _validate_versions(self, result: InferenceResult) -> None:
        checks = (
            (
                result.inference_contract_version,
                self._context.settings.accepted_inference_contract_version,
                ErrorCode.INFERENCE_CONTRACT_MISMATCH,
            ),
            (result.model_version, self._context.settings.accepted_model_version, ErrorCode.MODEL_VERSION_MISMATCH),
            (
                result.model_package_version,
                self._context.settings.accepted_model_package_version,
                ErrorCode.MODEL_PACKAGE_MISMATCH,
            ),
            (
                result.feature_schema_version,
                self._context.settings.accepted_feature_schema_version,
                ErrorCode.FEATURE_VERSION_MISMATCH,
            ),
            (
                result.model_key_version,
                self._context.settings.accepted_model_key_version,
                ErrorCode.MODEL_KEY_VERSION_MISMATCH,
            ),
        )
        for actual, expected, error_code in checks:
            if actual != expected:
                raise AgentError(error_code, http_status=502)

    async def select_results(self, state: RecommendationState) -> dict[str, Any]:
        return await self._policy_node(
            state,
            "select_results",
            ErrorCode.RESULT_SELECTION_FAILED,
            "selections",
            self._context.selector.select(state["request"], state["inference_result"]),
        )

    async def derive_reasons(self, state: RecommendationState) -> dict[str, Any]:
        return await self._policy_node(
            state,
            "derive_reasons",
            ErrorCode.UNSUPPORTED_REASON,
            "reasoned_candidates",
            self._context.reason_deriver.derive(state["request"], state["inference_result"], state["selections"]),
        )

    async def render_explanations(self, state: RecommendationState) -> dict[str, Any]:
        return await self._policy_node(
            state,
            "render_explanations",
            ErrorCode.UNSAFE_TEMPLATE,
            "rendered_candidates",
            self._context.renderer.render(state["reasoned_candidates"], budget=self._budget(state)),
        )

    async def _policy_node(
        self,
        state: RecommendationState,
        name: str,
        error_code: ErrorCode,
        output_key: str,
        call: Awaitable[tuple[SelectedCandidate, ...] | tuple[ReasonedCandidate, ...] | tuple[RenderedCandidate, ...]],
    ) -> dict[str, Any]:
        trace = self._trace(state, name)
        try:
            result = await call
            self._budget(state).ensure_remaining()
        except AgentError as error:
            return self._failure(error, trace=trace)
        except Exception:  # noqa: BLE001 - policy boundary maps unknown failures without retaining details
            return self._failure(AgentError(error_code, http_status=500), trace=trace)
        return {output_key: result, "node_trace": trace}

    async def build_success(self, state: RecommendationState) -> dict[str, Any]:
        trace = self._trace(state, "build_success")
        try:
            self._budget(state).ensure_remaining()
            if not state.get("compatibility_validated", False):
                raise AgentError(ErrorCode.RESULT_SELECTION_FAILED, http_status=500)
            selections = state["selections"]
            reasoned = state["reasoned_candidates"]
            rendered = state["rendered_candidates"]
            if tuple(item.selection for item in reasoned) != selections:
                raise AgentError(ErrorCode.RESULT_SELECTION_FAILED, http_status=500)
            if tuple(item.reasoned for item in rendered) != reasoned:
                raise AgentError(ErrorCode.RESULT_SELECTION_FAILED, http_status=500)
            request_ids = {candidate.candidate_id for candidate in state["request"].candidates}
            selected_ids = [item.reasoned.selection.candidate_id for item in rendered]
            if len(rendered) > 3:
                raise AgentError(ErrorCode.RESULT_SELECTION_FAILED, http_status=500)
            if not set(selected_ids).issubset(request_ids):
                raise AgentError(ErrorCode.UNKNOWN_CANDIDATE, http_status=500)
            if len(selected_ids) != len(set(selected_ids)):
                raise AgentError(ErrorCode.RESULT_SELECTION_FAILED, http_status=500)
            scored_by_id = {item.candidate_id: item for item in state["inference_result"].candidates}
            for selection in selections:
                scored = scored_by_id.get(selection.candidate_id)
                if (
                    scored is None
                    or selection.probability != scored.probability
                    or selection.model_score != scored.model_score
                ):
                    raise AgentError(ErrorCode.RESULT_SELECTION_FAILED, http_status=500)
            response = AgentResponse.model_validate(
                {
                    "contractVersion": "recommendation-agent-v2",
                    "requestId": state["request"].request_id,
                    "sessionId": state["request"].session_id,
                    "traceId": state["request"].trace_id,
                    "agentTraceId": state["agent_trace_id"],
                    "status": "success",
                    "modelVersion": state["inference_result"].model_version,
                    "modelPackageVersion": state["inference_result"].model_package_version,
                    "featureSchemaVersion": state["inference_result"].feature_schema_version,
                    "inferenceContractVersion": state["inference_result"].inference_contract_version,
                    "modelKeyVersion": state["inference_result"].model_key_version,
                    "diversityPolicyVersion": "recommendation-diversity-v2",
                    "reasonPolicyVersion": "recommendation-reasons-v1",
                    "templateVersion": "recommendation-template-v1",
                    "recommendations": [
                        {
                            "candidateId": item.reasoned.selection.candidate_id,
                            "rank": rank,
                            "recommendationType": item.reasoned.selection.recommendation_type,
                            "probability": item.reasoned.selection.probability,
                            "modelScore": item.reasoned.selection.model_score,
                            "reasons": list(item.reasoned.reasons),
                            "explanation": item.explanation,
                        }
                        for rank, item in enumerate(rendered, start=1)
                    ],
                }
            )
        except (AgentError, ValidationError) as error:
            agent_error = (
                error
                if isinstance(error, AgentError)
                else AgentError(ErrorCode.RESULT_SELECTION_FAILED, http_status=500)
            )
            return self._failure(agent_error, trace=trace)
        return {"response": response, "node_trace": trace}

    async def build_failure(self, state: RecommendationState) -> dict[str, Any]:
        trace = self._trace(state, "build_failure")
        failure = state.get("failure") or FailureRecord(ErrorCode.INTERNAL_ERROR, 500)
        request = state["request"]
        response = AgentFailure.model_validate(
            {
                "contractVersion": "recommendation-agent-v2",
                "requestId": request.request_id,
                "sessionId": request.session_id,
                "traceId": request.trace_id,
                "agentTraceId": state["agent_trace_id"],
                "status": "failure",
                "error": {"code": failure.code, "retryable": failure.retryable},
            }
        )
        return {"failure_response": response, "failure": failure, "node_trace": trace}
