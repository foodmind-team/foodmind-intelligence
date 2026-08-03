"""Async task API router (P3-01, P4-04).

Endpoints (all require the internal service token):
  POST /internal/v2/cooking-plan/tasks          -> 202 + task + status location
  GET  /internal/v2/cooking-plan/tasks/{id}     -> status / progress / result
  GET  /internal/v2/cooking-plan/tasks/{id}/events -> SSE progress stream (P4-04)
  POST /internal/v2/cooking-plan/tasks/{id}/cancel -> cooperative cancel

The router keeps the request/response shapes in the unified ErrorEnvelope
contract (P3-05): idempotency conflicts are 409, missing tasks are 404, and
validation/auth/overload errors follow the managed-endpoint envelope.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from cooking_plan_agent.api.backpressure import request_lease
from cooking_plan_agent.api.dependencies import (
    extract_correlation_id,
    require_internal_service,
)
from cooking_plan_agent.config.settings import Settings, get_settings
from cooking_plan_agent.domain.models import (
    ErrorEnvelope,
    GeneratePlanRequest,
)
from cooking_plan_agent.tasks.models import TaskProgress, TaskRecord, TaskStatus, is_terminal
from cooking_plan_agent.tasks.service import (
    AsyncTaskService,
    TaskIdempotencyConflict,
)

logger = logging.getLogger(__name__)

# P4-04: idle streams emit an SSE comment frame every this many seconds so
# proxies do not drop the connection while a task is still executing.
SSE_KEEPALIVE_SECONDS = 15.0

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


class TaskEvent(BaseModel):
    """SSE ``data`` payload (P4-04) — a safe task snapshot.

    Mirrors TaskSummary plus the monotonic ``event_id`` used for
    ``Last-Event-ID`` reconnection. Deliberately excludes the raw
    ``request_payload`` (full recipe text / inventory) so the stream only
    exposes the same fields the polling API already returns.
    """

    event_id: int
    task_id: str
    status: TaskStatus
    request_id: str
    location: str
    progress: TaskProgress = Field(default_factory=TaskProgress)
    result: dict[str, object] | None = None
    error: dict[str, object] | None = None
    created_at: str
    updated_at: str


class ExecutionUpdateRequest(BaseModel):
    """One user-confirmed cooking-task transition."""

    cooking_task_id: str = Field(min_length=1, max_length=200)
    status: str = Field(pattern="^(IN_PROGRESS|COMPLETED)$")
    expected_event_id: int = Field(ge=0)


class ExecutionSnapshot(BaseModel):
    """Current ready/blocked task groups for Android and Web clients."""

    event_id: int
    available_tasks: tuple[dict[str, object], ...]
    in_progress_task_ids: tuple[str, ...]
    completed_task_ids: tuple[str, ...]
    blocked_tasks: tuple[dict[str, object], ...]
    is_complete: bool


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


def _to_event(record: TaskRecord) -> TaskEvent:
    """Convert a persisted record to the SSE data payload (P4-04)."""
    return TaskEvent(event_id=record.event_id, **_to_summary(record).model_dump())


def _parse_last_event_id(raw: str | None) -> int:
    """Parse the ``Last-Event-ID`` header; invalid/missing values start over."""
    if raw is None:
        return -1
    try:
        return int(raw)
    except ValueError:
        return -1


def _format_sse_event(record: TaskRecord) -> str:
    """Render a TaskRecord as one SSE event frame (P4-04).

    Non-terminal snapshots are ``progress`` events; the terminal snapshot is
    the ``done`` event that closes the stream. The ``id`` line carries the
    monotonic event_id so clients resume via ``Last-Event-ID``.
    """
    event_type = "done" if is_terminal(record.status) else "progress"
    data = json.dumps(_to_event(record).model_dump(), default=str)
    return f"id: {record.event_id}\nevent: {event_type}\ndata: {data}\n\n"


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


def _execution_response(record: TaskRecord, snapshot: dict[str, object]) -> ExecutionSnapshot:
    """Convert a trusted service snapshot to the stable HTTP response."""
    return ExecutionSnapshot.model_validate({"event_id": record.event_id, **snapshot})


@router.get(
    "/{task_id}/execution",
    response_model=ExecutionSnapshot,
    responses={
        404: {"model": ErrorEnvelope, "description": "Task not found."},
        409: {"model": ErrorEnvelope, "description": "Plan is not READY for execution."},
    },
)
async def get_execution_state(
    task_id: str,
    service: Annotated[AsyncTaskService, Depends(get_task_service)],
    _correlation_id: Annotated[str, Depends(extract_correlation_id)],
) -> ExecutionSnapshot:
    """Return tasks that are executable now, based on actual completion state."""
    record, snapshot = await service.execution_snapshot(task_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail={"code": "TASK_NOT_FOUND", "message": "Task not found."})
    if snapshot is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail={"code": "PLAN_NOT_READY_FOR_EXECUTION", "message": "Plan is not READY for execution."})
    return _execution_response(record, snapshot)


@router.post(
    "/{task_id}/execution",
    response_model=ExecutionSnapshot,
    responses={
        404: {"model": ErrorEnvelope, "description": "Task not found."},
        409: {"model": ErrorEnvelope, "description": "Execution transition rejected."},
    },
)
async def update_execution_state(
    task_id: str,
    body: ExecutionUpdateRequest,
    service: Annotated[AsyncTaskService, Depends(get_task_service)],
    _correlation_id: Annotated[str, Depends(extract_correlation_id)],
) -> ExecutionSnapshot:
    """Mark a cooking task in progress or complete, then return the next group."""
    record, snapshot, error_code = await service.update_execution(
        task_id,
        body.cooking_task_id,
        body.status,
        body.expected_event_id,
    )
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail={"code": "TASK_NOT_FOUND", "message": "Task not found."})
    if error_code is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail={"code": error_code, "message": "Cooking task transition is not currently allowed."})
    assert snapshot is not None
    return _execution_response(record, snapshot)


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


@router.get(
    "/{task_id}/events",
    responses={
        200: {
            "description": (
                "Server-Sent-Events progress stream (P4-04). Frames: "
                "`id: <event_id>` / `event: progress|done` / `data: {TaskEvent}`. "
                "The stream closes after the terminal `done` event; reconnect with "
                "the `Last-Event-ID` header to resume from the first missed event. "
                "Idle streams receive a `: keepalive` comment every 15s."
            ),
            "content": {
                "text/event-stream": {
                    "schema": {
                        "type": "string",
                        "example": (
                            'id: 1\nevent: progress\ndata: {"task_id": "...", "status": "RUNNING"}\n\n'
                            'id: 2\nevent: done\ndata: {"task_id": "...", "status": "READY"}\n\n'
                        ),
                    }
                }
            },
        },
        404: {"model": ErrorEnvelope, "description": "Task not found, or SSE disabled."},
        401: {"model": ErrorEnvelope, "description": "Invalid service credential."},
    },
)
async def stream_task_events(
    task_id: str,
    request: Request,
    service: Annotated[AsyncTaskService, Depends(get_task_service)],
    settings: Annotated[Settings, Depends(get_settings)],
    _correlation_id: Annotated[str, Depends(extract_correlation_id)],
) -> StreamingResponse:
    """Subscribe to a task's SSE progress stream (P4-04).

    Emits a ``progress`` event per persisted state change, then a terminal
    ``done`` event before closing the stream. A client that disconnects may
    reconnect with the ``Last-Event-ID`` header to resume from the first
    missed event; the polling endpoints are unaffected and remain the
    fallback. Streaming is feature-flagged by ``task_sse_enabled``.
    """
    if not settings.task_sse_enabled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "SSE_DISABLED", "message": "Task SSE progress streaming is disabled."},
        )
    record = await service.get(task_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "TASK_NOT_FOUND", "message": f"Task {task_id} not found."},
        )
    last_event_id = _parse_last_event_id(request.headers.get("last-event-id"))

    async def _event_source() -> AsyncIterator[str]:
        async for frame in service.subscribe(
            task_id,
            last_event_id,
            keepalive_seconds=SSE_KEEPALIVE_SECONDS,
        ):
            if frame is None:
                yield ": keepalive\n\n"
            else:
                yield _format_sse_event(frame)

    return StreamingResponse(
        _event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
