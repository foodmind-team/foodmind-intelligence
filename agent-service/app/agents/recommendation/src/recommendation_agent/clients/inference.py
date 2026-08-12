"""One-attempt authenticated HTTP adapter for recommendation-inference-v1."""

import asyncio
import json
from dataclasses import dataclass
from time import monotonic
from typing import Any, NoReturn, cast

import httpx
from pydantic import ValidationError

from recommendation_agent.config.settings import Settings
from recommendation_agent.domain.errors import AgentError, ErrorCode
from recommendation_agent.domain.models import (
    CollaborativeSignal,
    InferenceCandidate,
    InferenceCommand,
    InferenceEvidence,
    InferenceResult,
    ScoredCandidate,
)
from recommendation_agent.observability.metrics import MetricsRegistry
from recommendation_agent.schemas.agent_v2 import AgentRequest
from recommendation_agent.schemas.inference_v1 import (
    INFERENCE_RESPONSE_ADAPTER,
    InferenceFailure,
    InferenceRequest,
    InferenceSuccess,
)
from recommendation_agent.time.budget import DeadlineBudget


@dataclass(slots=True)
class InferenceMetrics:
    calls_total: int = 0
    failures_total: int = 0
    last_duration_ms: float = 0.0
    last_candidate_count: int = 0
    last_result_category: str = "never_called"


def command_from_agent_request(request: AgentRequest) -> InferenceCommand:
    return InferenceCommand(
        request_id=request.request_id,
        trace_id=request.trace_id,
        deadline_at=request.deadline_at,
        feature_schema_version=request.feature_schema_version,
        model_user_key=request.model_user_key,
        model_key_version=request.model_key_version,
        candidates=tuple(
            InferenceCandidate(
                candidate_id=candidate.candidate_id,
                model_meal_key=candidate.model_meal_key,
                model_offering_key=candidate.model_offering_key,
                evidence=InferenceEvidence(
                    preference_match=candidate.evidence.preference_match,
                    want_to_try=candidate.evidence.want_to_try,
                    group_preference_rate=candidate.evidence.group_preference_rate,
                    group_eligible_member_count=candidate.evidence.group_eligible_member_count,
                    context_match=candidate.evidence.context_match,
                    cleanliness_observed=candidate.evidence.cleanliness_observed,
                ),
            )
            for candidate in request.candidates
        ),
    )


class RecommendationInferenceHttpClient:
    """Strict HTTP implementation of the inference port with no retry path."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        settings: Settings,
        metrics_registry: MetricsRegistry | None = None,
    ) -> None:
        self._client = client
        self._settings = settings
        self.metrics = InferenceMetrics()
        self._metrics_registry = metrics_registry or MetricsRegistry()

    async def score(self, command: InferenceCommand, *, budget: DeadlineBudget) -> InferenceResult:
        timeout_seconds = budget.downstream_timeout(
            configured_seconds=self._settings.inference_total_timeout_ms / 1000.0,
            guard_seconds=self._settings.deadline_guard_ms / 1000.0,
        )
        wire_request = _wire_request(command)
        request = self._client.build_request(
            "POST",
            self._settings.inference_endpoint_path,
            headers={
                "Authorization": f"Bearer {self._settings.inference_service_token.get_secret_value()}",
                "Content-Type": "application/json",
                "X-Request-ID": command.request_id,
                "X-Trace-ID": command.trace_id,
                "X-Inference-Contract-Version": self._settings.accepted_inference_contract_version,
                "X-Feature-Schema-Version": self._settings.accepted_feature_schema_version,
                "X-Model-Key-Version": self._settings.accepted_model_key_version,
            },
            content=wire_request.model_dump_json(by_alias=True),
            timeout=httpx.Timeout(
                timeout_seconds,
                connect=min(timeout_seconds, self._settings.inference_connect_timeout_ms / 1000.0),
                pool=min(timeout_seconds, self._settings.inference_pool_timeout_ms / 1000.0),
            ),
        )
        self.metrics.calls_total += 1
        self.metrics.last_candidate_count = len(command.candidates)
        started = monotonic()
        try:
            response = await self._send_once(request, timeout_seconds=timeout_seconds)
            result = _parse_response(response, command=command, settings=self._settings)
            budget.ensure_remaining(guard_seconds=self._settings.deadline_guard_ms / 1000.0)
        except AgentError as exc:
            self.metrics.failures_total += 1
            self.metrics.last_result_category = exc.code.value
            self._metrics_registry.record_inference(
                result=exc.code.value,
                duration_seconds=max(0.0, monotonic() - started),
            )
            raise
        finally:
            self.metrics.last_duration_ms = max(0.0, (monotonic() - started) * 1000.0)
        self.metrics.last_result_category = "success"
        self._metrics_registry.record_inference(
            result="success",
            duration_seconds=max(0.0, monotonic() - started),
        )
        return result

    async def is_ready(self) -> bool:
        try:
            response = await self._client.get(
                self._settings.inference_readiness_path,
                timeout=min(0.5, self._settings.inference_connect_timeout_ms / 1000.0),
            )
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def _send_once(self, request: httpx.Request, *, timeout_seconds: float) -> bytes:
        response: httpx.Response | None = None
        try:
            async with asyncio.timeout(timeout_seconds):
                response = await self._client.send(request, stream=True)
                if not 200 <= response.status_code < 300:
                    if response.status_code == 503:
                        raise AgentError(ErrorCode.INFERENCE_UNAVAILABLE, http_status=503, retryable=True)
                    raise AgentError(
                        ErrorCode.INFERENCE_HTTP_ERROR,
                        http_status=502,
                        retryable=response.status_code >= 500,
                    )
                chunks: list[bytes] = []
                size = 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > self._settings.inference_max_response_bytes:
                        raise AgentError(ErrorCode.INFERENCE_RESPONSE_TOO_LARGE, http_status=502, retryable=True)
                    chunks.append(chunk)
                return b"".join(chunks)
        except AgentError:
            raise
        except httpx.TimeoutException as exc:
            raise AgentError(ErrorCode.INFERENCE_TIMEOUT, http_status=504, retryable=True) from exc
        except TimeoutError as exc:
            raise AgentError(ErrorCode.INFERENCE_TIMEOUT, http_status=504, retryable=True) from exc
        except httpx.RequestError as exc:
            raise AgentError(ErrorCode.INFERENCE_CONNECTION_FAILED, http_status=502, retryable=True) from exc
        finally:
            if response is not None:
                await response.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()


def _wire_request(command: InferenceCommand) -> InferenceRequest:
    return InferenceRequest.model_validate(
        {
            "contractVersion": "recommendation-inference-v1",
            "requestId": command.request_id,
            "traceId": command.trace_id,
            "deadlineAt": command.deadline_at,
            "featureSchemaVersion": command.feature_schema_version,
            "modelUserKey": command.model_user_key,
            "modelKeyVersion": command.model_key_version,
            "candidates": [
                {
                    "candidateId": candidate.candidate_id,
                    "modelMealKey": candidate.model_meal_key,
                    "modelOfferingKey": candidate.model_offering_key,
                    "evidence": {
                        "preferenceMatch": candidate.evidence.preference_match,
                        "wantToTry": candidate.evidence.want_to_try,
                        "groupPreferenceRate": candidate.evidence.group_preference_rate,
                        "groupEligibleMemberCount": candidate.evidence.group_eligible_member_count,
                        "contextMatch": candidate.evidence.context_match,
                        "cleanlinessObserved": candidate.evidence.cleanliness_observed,
                    },
                }
                for candidate in command.candidates
            ],
        }
    )


def _reject_constant(value: str) -> NoReturn:
    raise ValueError(f"invalid JSON constant: {value[:0]}")


def _parse_response(content: bytes, *, command: InferenceCommand, settings: Settings) -> InferenceResult:
    try:
        raw = json.loads(content.decode("utf-8"), parse_constant=_reject_constant)
    except ValueError as exc:
        if b"NaN" in content or b"Infinity" in content:
            raise AgentError(ErrorCode.INVALID_PROBABILITY, http_status=502) from exc
        raise AgentError(ErrorCode.INFERENCE_MALFORMED_RESPONSE, http_status=502) from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AgentError(ErrorCode.INFERENCE_MALFORMED_RESPONSE, http_status=502) from exc
    if not isinstance(raw, dict):
        raise AgentError(ErrorCode.INFERENCE_MALFORMED_RESPONSE, http_status=502)
    typed_raw = cast(dict[str, Any], raw)
    _validate_top_level_compatibility(typed_raw, command=command, settings=settings)
    try:
        response = INFERENCE_RESPONSE_ADAPTER.validate_python(typed_raw)
    except ValidationError as exc:
        locations = {str(part) for error in exc.errors(include_input=False) for part in error["loc"]}
        if "probability" in locations:
            raise AgentError(ErrorCode.INVALID_PROBABILITY, http_status=502) from exc
        if locations.intersection({"userCf", "itemCf", "signals"}):
            raise AgentError(ErrorCode.INVALID_EVIDENCE, http_status=502) from exc
        raise AgentError(ErrorCode.INFERENCE_MALFORMED_RESPONSE, http_status=502) from exc
    if isinstance(response, InferenceFailure):
        if response.error.code == "MODEL_PACKAGE_INCOMPATIBLE":
            raise AgentError(ErrorCode.MODEL_PACKAGE_MISMATCH, http_status=502)
        raise AgentError(ErrorCode.INFERENCE_UNAVAILABLE, http_status=503, retryable=True)
    return _validate_success(response, command=command)


def _validate_top_level_compatibility(raw: dict[str, Any], *, command: InferenceCommand, settings: Settings) -> None:
    checks = (
        ("contractVersion", settings.accepted_inference_contract_version, ErrorCode.INFERENCE_CONTRACT_MISMATCH),
        ("requestId", command.request_id, ErrorCode.INFERENCE_CONTRACT_MISMATCH),
        ("traceId", command.trace_id, ErrorCode.INFERENCE_CONTRACT_MISMATCH),
        ("modelVersion", settings.accepted_model_version, ErrorCode.MODEL_VERSION_MISMATCH),
        ("featureSchemaVersion", settings.accepted_feature_schema_version, ErrorCode.FEATURE_VERSION_MISMATCH),
        ("modelKeyVersion", settings.accepted_model_key_version, ErrorCode.MODEL_KEY_VERSION_MISMATCH),
    )
    for field, expected, error_code in checks:
        if field in raw and raw[field] != expected:
            raise AgentError(error_code, http_status=502)
    if raw.get("status") == "success" and raw.get("modelPackageVersion") != settings.accepted_model_package_version:
        raise AgentError(ErrorCode.MODEL_PACKAGE_MISMATCH, http_status=502)


def _validate_success(response: InferenceSuccess, *, command: InferenceCommand) -> InferenceResult:
    request_by_id = {candidate.candidate_id: candidate for candidate in command.candidates}
    response_ids = [prediction.candidate_id for prediction in response.predictions]
    if len(response_ids) != len(set(response_ids)):
        raise AgentError(ErrorCode.DUPLICATE_CANDIDATE, http_status=502)
    if any(candidate_id not in request_by_id for candidate_id in response_ids):
        raise AgentError(ErrorCode.UNKNOWN_CANDIDATE, http_status=502)
    if set(response_ids) != set(request_by_id):
        raise AgentError(ErrorCode.MISSING_CANDIDATE, http_status=502)
    scored: list[ScoredCandidate] = []
    for prediction in response.predictions:
        expected = request_by_id[prediction.candidate_id]
        evidence = InferenceEvidence(
            preference_match=prediction.signals.preference_match,
            want_to_try=prediction.signals.want_to_try,
            group_preference_rate=prediction.signals.group_preference_rate,
            group_eligible_member_count=prediction.signals.group_eligible_member_count,
            context_match=prediction.signals.context_match,
            cleanliness_observed=prediction.signals.cleanliness_observed,
        )
        if evidence != expected.evidence:
            raise AgentError(ErrorCode.INVALID_EVIDENCE, http_status=502)
        scored.append(
            ScoredCandidate(
                candidate_id=prediction.candidate_id,
                probability=prediction.probability,
                model_score=prediction.model_score,
                user_cf=CollaborativeSignal(
                    prediction.user_cf.available,
                    prediction.user_cf.score,
                    prediction.user_cf.neighbor_support,
                ),
                item_cf=CollaborativeSignal(
                    prediction.item_cf.available,
                    prediction.item_cf.score,
                    prediction.item_cf.supporting_item_count,
                ),
                evidence=evidence,
            )
        )
    return InferenceResult(
        model_version=response.model_version,
        model_package_version=response.model_package_version,
        feature_schema_version=response.feature_schema_version,
        inference_contract_version=response.contract_version,
        model_key_version=response.model_key_version,
        candidates=tuple(scored),
    )
