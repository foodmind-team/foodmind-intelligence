"""FastAPI global exception handlers — layer 3 of the error boundary.

Per handbook 9.9: typed application errors map to stable response bodies.
Unexpected exceptions are:
  1. Logged once with correlation ID and stack trace on the server.
  2. Returned as a generic error code without provider secrets or recipe text.
  3. Incremented in an error metric.

Handbook 8.10 three-level error boundary:
  Layer 1: domain/application services raise WorkflowException.
  Layer 2: LangGraph workflow nodes catch them, produce NodeExecutionError in state.
  Layer 3 (this file): FastAPI exception handler catches unhandled exceptions
    and returns a stable INTERNAL_ERROR body.

Do NOT place one broad try/except Exception around every route.
"""

import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from cooking_plan_agent.domain.errors import DomainErrorCode, WorkflowException

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response builder — constructs a FailedPlanResponse body
# ---------------------------------------------------------------------------


def _build_error_body(
    error_code: DomainErrorCode,
    message: str,
    correlation_id: str,
) -> dict:
    """Build a stable error response dict matching the FailedPlanResponse shape.

    Never includes provider secrets, stack traces, or user recipe text.
    """
    return {
        "status": "FAILED",
        "error_code": error_code.value,
        "correlation_id": correlation_id,
        "message": message,
    }


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


async def workflow_exception_handler(
    request: Request,
    exc: WorkflowException,
) -> JSONResponse:
    """Handle known domain exceptions with their mapped error code.

    Domain errors are business failures (infeasible schedule, safety violation,
    etc.) — return HTTP 200 with FAILED status per handbook 9.8 policy.

    HTTP 200 is correct here: the business outcome is "this plan cannot be
    generated with the given inputs", which is a valid result, not a protocol
    failure.
    """
    correlation_id = getattr(request.state, "correlation_id", "unknown")
    logger.warning(
        "WorkflowException caught | code=%s | correlation_id=%s | message=%s",
        exc.code.value,
        correlation_id,
        exc.message,
    )
    return JSONResponse(
        status_code=200,
        content=_build_error_body(exc.code, exc.message, correlation_id),
    )


async def generic_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Last-resort handler for unexpected exceptions.

    Returns HTTP 500 with INTERNAL_ERROR code. Logs the full stack trace
    server-side but returns a sanitised body to the client.

    The inner message is intentionally vague — provider secrets, recipe text,
    and stack traces must never leak to the response.
    """
    correlation_id = getattr(request.state, "correlation_id", "unknown")
    logger.exception(
        "Unexpected exception | correlation_id=%s | type=%s",
        correlation_id,
        type(exc).__name__,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_build_error_body(
            DomainErrorCode.INTERNAL_ERROR,
            "An unexpected internal error occurred.",
            correlation_id,
        ),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register all global exception handlers on the FastAPI application.

    Called from create_app() after the app instance is built.

    Handler precedence:
      1. WorkflowException -> 200 + FAILED body (business failure, not error)
      2. Exception -> 500 + INTERNAL_ERROR body (last resort)
    """
    app.add_exception_handler(WorkflowException, workflow_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)
