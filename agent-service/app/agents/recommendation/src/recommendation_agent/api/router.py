"""Backend-facing private Recommendation Agent route."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from recommendation_agent.api.backpressure import RequestLimiter
from recommendation_agent.api.dependencies import (
    Correlation,
    extract_correlation,
    get_agent_service,
    require_internal_service,
)
from recommendation_agent.api.v1_compat import execute_v1_request
from recommendation_agent.application.service import RecommendationAgentService
from recommendation_agent.domain.errors import AgentError, ErrorCode
from recommendation_agent.schemas.agent_v1_compat import V1Request, V1Response
from recommendation_agent.schemas.agent_v2 import AgentFailure, AgentRequest, AgentResponse

router = APIRouter(prefix="/internal/v1/recommendations", tags=["recommendation-agent"])


@router.post(
    "/generate",
    response_model=AgentResponse | V1Response,
    responses={400: {"model": AgentFailure}, 503: {"model": AgentFailure}},
    dependencies=[Depends(require_internal_service)],
)
async def generate_recommendations(
    http_request: Request,
    agent_request: AgentRequest | V1Request,
    correlation: Annotated[Correlation, Depends(extract_correlation)],
    service: Annotated[RecommendationAgentService, Depends(get_agent_service)],
) -> AgentResponse | V1Response:
    del correlation
    http_request.state.request_identifiers = (
        agent_request.request_id,
        agent_request.session_id,
        agent_request.trace_id,
    )
    settings = http_request.app.state.settings
    if isinstance(agent_request, V1Request):
        return await execute_v1_request(http_request, agent_request, service)
    if agent_request.contract_version not in settings.supported_contract_versions:
        raise AgentError(ErrorCode.UNSUPPORTED_AGENT_VERSION, http_status=400)
    if len(agent_request.candidates) > settings.max_candidates:
        raise AgentError(ErrorCode.REQUEST_TOO_LARGE, http_status=413)
    limiter: RequestLimiter = http_request.app.state.request_limiter
    async with limiter.lease():
        return await service.execute(agent_request)
