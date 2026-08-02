"""Internal API router for the cooking plan agent.

Per handbook 9.2: one endpoint for generation.
POST /internal/v1/agents/cooking-plan/generate

The Spring Boot caller converts pasted text or .txt upload into recipe
text before calling this endpoint. This keeps the internal contract
JSON-only and prevents duplicated multipart/file rules.

Handbook 9.1: the public boundary stays in Spring Boot — this router
validates service authentication and the internal request schema, not
end-user JWTs.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Request

from cooking_plan_agent.api.dependencies import (
    extract_correlation_id,
    require_internal_service,
)
from cooking_plan_agent.application import GenerateCookingPlanService
from cooking_plan_agent.domain.models import GeneratePlanRequest, PlanResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Router — all endpoints require internal service authentication
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/internal/v1/agents/cooking-plan",
    tags=["cooking-plan-agent"],
    dependencies=[Depends(require_internal_service)],
)


# ---------------------------------------------------------------------------
# Service dependency — extracted from request.app.state
# ---------------------------------------------------------------------------


def get_generate_service(request: Request) -> GenerateCookingPlanService:
    """Retrieve the application service from the app's lifespan-injected state.

    Raises AttributeError if the service was not initialised during startup.
    """
    service = request.app.state.generate_plan_service
    if not isinstance(service, GenerateCookingPlanService):
        raise AttributeError("generate_plan_service was not initialised during startup")
    return service


# ---------------------------------------------------------------------------
# Generate endpoint
# ---------------------------------------------------------------------------


@router.post("/generate", response_model=PlanResponse)
async def generate_plan(
    body: GeneratePlanRequest,
    service: Annotated[GenerateCookingPlanService, Depends(get_generate_service)],
    _correlation_id: Annotated[str, Depends(extract_correlation_id)],
) -> PlanResponse:
    """Generate a cooking plan from the supplied recipes and constraints.

    Accepts a JSON request body validated against GeneratePlanRequest.
    Returns a PlanResponse — one of READY, NEEDS_CONFIRMATION, INFEASIBLE,
    or FAILED. All business outcomes return HTTP 200 per handbook 9.8.

    The correlation ID is injected via the X-Request-ID dependency and
    propagated to response headers by the CORSMiddleware (configured in
    create_app).
    """
    logger.info(
        "Generating plan | request_id=%s | recipes=%d | time_limit=%s",
        body.request_id,
        len(body.recipes),
        body.time_limit_minutes,
    )
    return await service.execute(body)
