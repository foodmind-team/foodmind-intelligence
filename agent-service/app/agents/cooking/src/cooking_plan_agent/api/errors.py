"""FastAPI global exception handlers — layer 3 of the error boundary (P3-05).

Per handbook 9.9: typed application errors map to stable response bodies.
Unexpected exceptions are:
  1. Logged once with correlation ID and stack trace on the server.
  2. Returned as a generic error code without provider secrets or recipe text.
  3. Incremented in an error metric.

P3-05 unified error contract: every managed endpoint returns the same
``ErrorEnvelope(status, error_code, message, correlation_id, details,
retryable)`` for protocol/HTTP-level failures. Legal business outcomes
(READY / NEEDS_CONFIRMATION / INFEASIBLE / FAILED) keep their own response
models and are never disguised as protocol errors.

Handbook 8.10 three-level error boundary:
  Layer 1: domain/application services raise WorkflowException.
  Layer 2: LangGraph workflow nodes catch them, produce NodeExecutionError in state.
  Layer 3 (this file): FastAPI exception handler catches unhandled exceptions
    and returns a stable ErrorEnvelope.

Do NOT place one broad try/except Exception around every route.
"""

import logging
import uuid

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from cooking_plan_agent.domain.errors import DomainErrorCode, WorkflowException, is_retryable
from cooking_plan_agent.domain.models import ErrorEnvelope

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Envelope builder — constructs an ErrorEnvelope body
# ---------------------------------------------------------------------------
# details only carries field-level safe diagnostics: validation loc/type,
# or a machine-readable retry hint. Raw input, stack traces, and provider
# payloads are never included (P3-05 details safety rule).


def _correlation_id(request: Request) -> str:
    """Read the correlation ID stored on the request by the DI dependency.

    Auth failures can be raised by the router-level dependency BEFORE the
    ``extract_correlation_id`` endpoint dependency runs, leaving
    ``request.state.correlation_id`` unset. In that case we derive a stable
    ID from the caller's X-Request-ID header (when safe) or generate one,
    and store it on the request so the response middleware echoes the same
    value in the header (P3-05: header and body correlation ID must match).
    """
    existing = getattr(request.state, "correlation_id", None)
    if existing:
        return str(existing)
    supplied = request.headers.get("X-Request-ID")
    cid = supplied if supplied and len(supplied) <= 128 else uuid.uuid4().hex
    request.state.correlation_id = cid
    return cid


# Compat v1 prefix (Spring Boot contract, P0-02). The compat router keeps its
# limited error mapping ({"detail": {"code": ...}}) so the Java caller is
# unaffected by the P3-05 envelope; native/v2 endpoints get the full envelope.
_COMPAT_PATH_PREFIX = "/internal/v1/cooking-plans"


def _is_compat_path(request: Request) -> bool:
    """Return True when the request targets the Spring Boot v1 compat router."""
    return request.url.path.startswith(_COMPAT_PATH_PREFIX)


def _build_envelope(
    *,
    http_status: int,
    error_code: str,
    message: str,
    correlation_id: str,
    details: dict[str, object] | list[dict[str, object]] | None = None,
) -> ErrorEnvelope:
    """Build a stable ErrorEnvelope with catalog-driven retryable."""
    return ErrorEnvelope(
        status=http_status,
        error_code=error_code,
        message=message,
        correlation_id=correlation_id,
        details=details,
        retryable=is_retryable(error_code),
    )


def _http_code_from_exception(exc: StarletteHTTPException) -> str | None:
    """Extract a stable error code from an HTTPException's detail.

    HTTPException.detail is either a dict (e.g. {"code": "..."} from the
    auth/backpressure dependencies) or a plain string (e.g. Starlette's
    default "Not Found"). Only the explicit code form is trusted; string
    details map to a default code derived from the status below.
    """
    if isinstance(exc.detail, dict):
        code = exc.detail.get("code")
        if isinstance(code, str) and code:
            return code
    return None


_DEFAULT_CODES: dict[int, str] = {
    status.HTTP_401_UNAUTHORIZED: "UNAUTHORIZED",
    status.HTTP_403_FORBIDDEN: "FORBIDDEN",
    status.HTTP_404_NOT_FOUND: "NOT_FOUND",
    status.HTTP_409_CONFLICT: "CONFLICT",
    status.HTTP_422_UNPROCESSABLE_ENTITY: "REQUEST_VALIDATION_ERROR",
    status.HTTP_429_TOO_MANY_REQUESTS: "RATE_LIMITED",
    status.HTTP_503_SERVICE_UNAVAILABLE: "SERVICE_UNAVAILABLE",
}


def _default_code_for_status(http_status: int) -> str:
    """Map an HTTP status to a stable default error code."""
    return _DEFAULT_CODES.get(http_status, DomainErrorCode.INTERNAL_ERROR.value)


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


async def workflow_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle known domain exceptions with their mapped error code.

    Registered only for WorkflowException — the assert narrows the type for
    static analysis and always holds at runtime.

    Domain errors are business failures (infeasible schedule, safety violation,
    etc.) — return HTTP 200 with FAILED status per handbook 9.8 policy.
    This is a legal business outcome, NOT a protocol error, so it keeps the
    FailedPlanResponse shape (P3-05: business results are never disguised as
    protocol exceptions).
    """
    assert isinstance(exc, WorkflowException)
    correlation_id = _correlation_id(request)
    logger.warning(
        "WorkflowException caught | code=%s | correlation_id=%s | message=%s",
        exc.code.value,
        correlation_id,
        exc.message,
    )
    return JSONResponse(
        status_code=200,
        content={
            "status": "FAILED",
            "error_code": exc.code.value,
            "correlation_id": correlation_id,
            "message": exc.message,
        },
    )


async def request_validation_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle Pydantic validation failures with a unified envelope (422).

    details carries only field-level loc/type diagnostics — never the raw
    input values that caused the error (P3-05 details safety rule).
    """
    assert isinstance(exc, RequestValidationError)
    correlation_id = _correlation_id(request)

    # Spring Boot v1 compat: keep the limited mapping so the Java caller is
    # unaffected by the P3-05 envelope (P3-05 step 6).
    if _is_compat_path(request):
        return JSONResponse(
            status_code=422,
            content={"detail": {"code": DomainErrorCode.REQUEST_VALIDATION_ERROR.value}},
        )

    logger.warning(
        "Request validation failed | correlation_id=%s | errors=%d",
        correlation_id,
        len(exc.errors()),
    )

    details: list[dict[str, object]] = []
    for err in exc.errors():
        details.append(
            {
                "loc": list(err.get("loc", ())),
                "type": err.get("type", ""),
                "msg": err.get("msg", ""),
            }
        )

    envelope = _build_envelope(
        http_status=status.HTTP_422_UNPROCESSABLE_ENTITY,
        error_code=DomainErrorCode.REQUEST_VALIDATION_ERROR.value,
        message="Request validation failed.",
        correlation_id=correlation_id,
        details=details,
    )
    return JSONResponse(status_code=422, content=envelope.model_dump())


async def http_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle HTTPException (auth, backpressure, not-found, etc.) uniformly.

    The error_code comes from the exception's explicit ``code`` detail when
    present (auth/backpressure dependencies set it); otherwise a stable code
    is derived from the HTTP status. ``retryable`` always comes from the
    error catalog — never from the message text (P3-05 D9).

    Accepts both FastAPI's HTTPException and Starlette's (404/405 raised by
    the router layer); FastAPI's HTTPException subclasses Starlette's.
    """
    assert isinstance(exc, StarletteHTTPException)
    correlation_id = _correlation_id(request)
    error_code = _http_code_from_exception(exc) or _default_code_for_status(exc.status_code)

    # Spring Boot v1 compat: keep the limited {"detail": {"code": ...}} mapping
    # so the Java caller is unaffected by the P3-05 envelope (P3-05 step 6).
    if _is_compat_path(request):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": {"code": error_code}},
            headers=exc.headers,
        )

    envelope = _build_envelope(
        http_status=exc.status_code,
        error_code=error_code,
        message=_http_message(exc),
        correlation_id=correlation_id,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=envelope.model_dump(),
        headers=exc.headers,
    )


def _http_message(exc: StarletteHTTPException) -> str:
    """Stable, non-sensitive message for an HTTPException.

    A dict detail (auth/backpressure) may carry a ``message`` key; otherwise
    use a generic message keyed by status. Never echo raw dict values that
    could contain secrets.
    """
    if isinstance(exc.detail, dict):
        message = exc.detail.get("message")
        if isinstance(message, str) and message:
            return message
    if exc.status_code == status.HTTP_404_NOT_FOUND:
        return "Resource not found."
    if exc.status_code == status.HTTP_401_UNAUTHORIZED:
        return "Authentication failed."
    if exc.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
        return "Service temporarily unavailable. Retry after backoff."
    return "Request could not be processed."


async def generic_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Last-resort handler for unexpected exceptions.

    Returns HTTP 500 with INTERNAL_ERROR code. Logs the full stack trace
    server-side but returns a sanitised envelope to the client. The inner
    message is intentionally vague — provider secrets, recipe text, and
    stack traces must never leak to the response.
    """
    correlation_id = _correlation_id(request)
    logger.exception(
        "Unexpected exception | correlation_id=%s | type=%s",
        correlation_id,
        type(exc).__name__,
    )
    envelope = _build_envelope(
        http_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code=DomainErrorCode.INTERNAL_ERROR.value,
        message="An unexpected internal error occurred.",
        correlation_id=correlation_id,
    )
    return JSONResponse(status_code=500, content=envelope.model_dump())


def register_exception_handlers(app: FastAPI) -> None:
    """Register all global exception handlers on the FastAPI application.

    Called from create_app() after the app instance is built.

    Handler precedence (P3-05):
      1. WorkflowException -> 200 + FailedPlanResponse (business result)
      2. RequestValidationError -> 422 + ErrorEnvelope
      3. HTTPException -> status + ErrorEnvelope (401/403/404/409/503/...)
      4. Exception -> 500 + ErrorEnvelope (last resort)
    """
    app.add_exception_handler(WorkflowException, workflow_exception_handler)
    app.add_exception_handler(RequestValidationError, request_validation_handler)
    # StarletteHTTPException covers FastAPI's HTTPException (a subclass) and
    # router-level 404/405 raised by Starlette.
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)
