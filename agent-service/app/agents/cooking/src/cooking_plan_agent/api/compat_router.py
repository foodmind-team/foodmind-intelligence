"""Spring Boot v1 compat router — the endpoint the Java caller actually hits.

POST /internal/v1/cooking-plans/generate

This router mirrors the native agent endpoint but speaks the Java
``cooking-agent-v1`` contract (Bearer auth, camelCase DTOs, deadline
budget).  It is the P0-02 integration baseline: the Spring Boot caller
is unchanged; this endpoint simply translates.

Native endpoint (unchanged): POST /internal/v1/agents/cooking-plan/generate
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Request

from cooking_plan_agent.api.backpressure import request_lease
from cooking_plan_agent.api.compat_models import (
    CompatCookingRequest,
    CompatCookingResponse,
)
from cooking_plan_agent.api.dependencies import (
    extract_correlation_id,
    require_bearer_service,
)
from cooking_plan_agent.application import GenerateCookingPlanService
from cooking_plan_agent.application.contract_adapter import (
    build_internal_request,
    deadline_budget_seconds,
    is_contract_supported,
    selected_recipe_id,
    to_compat_response,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/internal/v1/cooking-plans",
    tags=["cooking-plan-agent-v1-compat"],
    dependencies=[Depends(require_bearer_service)],
)


def get_generate_service(request: Request) -> GenerateCookingPlanService:
    """Retrieve the application service from the app's lifespan-injected state."""
    service = request.app.state.generate_plan_service
    if not isinstance(service, GenerateCookingPlanService):
        raise AttributeError("generate_plan_service was not initialised during startup")
    return service


@router.post("/generate", response_model=CompatCookingResponse)
async def generate_plan_compat(
    body: CompatCookingRequest,
    service: Annotated[GenerateCookingPlanService, Depends(get_generate_service)],
    _correlation_id: Annotated[str, Depends(extract_correlation_id)],
    _lease: Annotated[None, Depends(request_lease)] = None,
) -> CompatCookingResponse:
    """Generate a cooking plan for the Spring Boot v1 contract.

    - Validates the contract version (fast fail on unsupported version).
    - Enforces the request's deadlineAt as an execution budget (fast fail
      when the deadline has already passed).
    - Maps candidate snapshots to internal structured candidates so no LLM
      call is made (P0-02 rule 4).
    - READY → SUCCEEDED; any other terminal state → FAILED.
    """
    if not is_contract_supported(body.contractVersion):
        logger.warning(
            "Unsupported contract version | contract_version=%s",
            body.contractVersion,
        )
        return _failure_response(body, "UNSUPPORTED_CONTRACT_VERSION")

    # Deadline budget: fail fast when the caller's deadline is already past.
    now = datetime.now(UTC)
    budget = deadline_budget_seconds(body.deadlineAt, now)
    if budget is not None and budget <= 0:
        logger.info("Request deadline already passed | request_id=%s", body.requestId)
        return _failure_response(body, "DEADLINE_PASSED")

    internal_request = build_internal_request(body)
    if not internal_request.preparsed_candidates:
        # No usable candidates → the Java side will map to NO_RECIPE_MATCH.
        return _failure_response(body, "NO_USABLE_CANDIDATES")

    response = await service.execute(internal_request)
    source_recipe_id = selected_recipe_id(body)
    return to_compat_response(body, response, source_recipe_id)


def _failure_response(
    body: CompatCookingRequest,
    code: str,
) -> CompatCookingResponse:
    """Build a FAILED compat response with the envelope echoed back."""
    from uuid import uuid4

    return CompatCookingResponse(
        contractVersion=body.contractVersion,
        requestId=body.requestId,
        planId=body.planId,
        traceId=body.traceId,
        agentTraceId=uuid4().hex,
        status="FAILED",
        servings=body.request.servings,
        estimatedCost=None,
        currency=body.request.currency,
    )
