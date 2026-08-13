"""Application service guarding workflow execution."""

import logging
import uuid

from recommendation_agent.application.ports import AgentWorkflow
from recommendation_agent.domain.errors import AgentError, ErrorCode
from recommendation_agent.schemas.agent_v2 import AgentRequest, AgentResponse

LOGGER = logging.getLogger("recommendation.result")


class RecommendationAgentService:
    """Delegate only to an explicitly installed workflow."""

    def __init__(self, workflow: AgentWorkflow | None) -> None:
        self._workflow = workflow

    @property
    def ready(self) -> bool:
        return self._workflow is not None

    async def execute(self, request: AgentRequest) -> AgentResponse:
        if self._workflow is None:
            raise AgentError(ErrorCode.SERVICE_NOT_READY, http_status=503, retryable=True)
        agent_trace_id = f"agent-{uuid.uuid4().hex}"
        response = await self._workflow.run(request, agent_trace_id=agent_trace_id)
        request_candidates = {candidate.candidate_id for candidate in request.candidates}
        if not {item.candidate_id for item in response.recommendations}.issubset(request_candidates):
            raise AgentError(ErrorCode.UNKNOWN_CANDIDATE, http_status=500)
        LOGGER.info(
            "ml_recommendation_completed",
            extra={
                "fields": {
                    "requestId": request.request_id,
                    "sessionId": request.session_id,
                    "traceId": request.trace_id,
                    "agentTraceId": response.agent_trace_id,
                    "modelVersion": response.model_version,
                    "candidateCount": len(request.candidates),
                    "resultCount": len(response.recommendations),
                    "ranking": [
                        {
                            "candidateId": item.candidate_id,
                            "rank": item.rank,
                            "probability": item.probability,
                            "modelScore": item.model_score,
                            "reasons": [reason.value for reason in item.reasons],
                        }
                        for item in response.recommendations
                    ],
                }
            },
        )
        return response

    async def aclose(self) -> None:
        if self._workflow is not None:
            await self._workflow.aclose()
