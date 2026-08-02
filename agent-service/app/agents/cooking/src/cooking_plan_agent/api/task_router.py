"""Async task API router (P3-01).

Endpoints (all require the internal service token):
  POST /internal/v2/cooking-plan/tasks          -> 202 + task + status location
  GET  /internal/v2/cooking-plan/tasks/{id}     -> status / progress / result
  POST /internal/v2/cooking-plan/tasks/{id}/cancel -> cooperative cancel

The router keeps the request/response shapes in the unified ErrorEnvelope
contract (P3-05): idempotency conflicts are 409, missing tasks are 404, and
validation/auth/overload errors follow the managed-endpoint envelope.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from cooking_plan_agent.api.backpressure import request_lease
from cooking_plan_agent.api.dependencies import (
    extract_correlation_id,
    require_internal_service,
)
from cooking_plan_agent.domain.models import (
    ErrorEnvelope,
    GeneratePlanRequest,
)
from cooking_plan_agent.tasks.models import TaskProgress, TaskRecord, TaskStatus
from cooking_plan_agent.tasks.service import (
    AsyncTaskService,
    TaskIdempotencyConflict,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/internal/v2/cooking-plan/tasks",
    tags=["cooking-plan-agent-tasks"],
    dependencies=[Depends(require_internal_service)],
)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class TaskSummary(BaseModel):
    """Lightweight task view returned by submit/query endpoints."""

    task_id: str
    status: TaskStatus
    request_id: str
    location: str
    progress: TaskProgress = Field(default_factory=TaskProgress)
    result: dict[str, object] | None = None
    error: dict[str, object] | None = None
    created_at: str
    updated_at: str


class TaskSubmitResponse(BaseModel):
    """202 response: task accepted with polling/SSE location."""

    task_id: str
    status: TaskStatus
    location: str
    request_id: str


def _to_summary(record: TaskRecord) -> TaskSummary:
    """Convert a persisted record to the wire model."""
    return TaskSummary(
        task_id=record.task_id,
        status=record.status,
        request_id=record.request_id,
        location=record.location(),
        progress=record.progress,
        result=record.result,
        error=record.error,
        created_at=record.created_at.isoformat(),
        updated_at=record.updated_at.isoformat(),
    )


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def get_task_service(request: Request) -> AsyncTaskService:
    """Retrieve the async task service from app lifespan state."""
    service = request.app.state.task_service
    if not isinstance(service, AsyncTaskService):
        raise AttributeError("task_service was not initialised during startup")
    return service


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=TaskSubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        202: {"model": TaskSubmitResponse, "description": "Task accepted."},
        409: {"model": ErrorEnvelope, "description": "Idempotency key conflict."},
        422: {"model": ErrorEnvelope, "description": "Request validation failed."},
        503: {"model": ErrorEnvelope, "description": "Overloaded or shutting down."},
    },
)
async def submit_task(
    body: GeneratePlanRequest,
    service: Annotated[AsyncTaskService, Depends(get_task_service)],
    _correlation_id: Annotated[str, Depends(extract_correlation_id)],
    _lease: Annotated[None, Depends(request_lease)] = None,
) -> TaskSubmitResponse:
    """Submit a cooking-plan generation task.

    Idempotent on request_id (D1): re-submitting the same payload returns
    the same task; a different payload for the same request_id is a 409.
    The response advertises the polling location (GET /tasks/{id}).
    """
    try:
        outcome = await service.submit(body)
    except TaskIdempotencyConflict as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "IDEMPOTENCY_CONFLICT",
                "message": f"request_id {exc.request_id} already submitted with a different payload.",
            },
        ) from exc

    logger.info(
        "Task submitted | task_id=%s | created=%s | request_id=%s",
        outcome.task.task_id,
        outcome.created,
        outcome.task.request_id,
    )
    return TaskSubmitResponse(
        task_id=outcome.task.task_id,
        status=outcome.task.status,
        location=outcome.task.location(),
        request_id=outcome.task.request_id,
    )


@router.get(
    "/{task_id}",
    response_model=TaskSummary,
    responses={
        200: {"model": TaskSummary, "description": "Task state."},
        404: {"model": ErrorEnvelope, "description": "Task not found."},
    },
)
async def get_task(
    task_id: str,
    service: Annotated[AsyncTaskService, Depends(get_task_service)],
    _correlation_id: Annotated[str, Depends(extract_correlation_id)],
) -> TaskSummary:
    """Query a task's current status, progress, and terminal result/error."""
    record = await service.get(task_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "TASK_NOT_FOUND", "message": f"Task {task_id} not found."},
        )
    return _to_summary(record)


@router.post(
    "/{task_id}/cancel",
    response_model=TaskSummary,
    responses={
        200: {"model": TaskSummary, "description": "Cancellation accepted."},
        404: {"model": ErrorEnvelope, "description": "Task not found."},
    },
)
async def cancel_task(
    task_id: str,
    service: Annotated[AsyncTaskService, Depends(get_task_service)],
    _correlation_id: Annotated[str, Depends(extract_correlation_id)],
) -> TaskSummary:
    """Cooperatively cancel a QUEUED/RUNNING task (P3-01)."""
    record = await service.cancel(task_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "TASK_NOT_FOUND", "message": f"Task {task_id} not found."},
        )
    return _to_summary(record)
