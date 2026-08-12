"""Canonical safe API error mapping."""

import logging
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from recommendation_agent.domain.errors import AgentError, ErrorCode

logger = logging.getLogger(__name__)


def _identifiers(request: Request) -> tuple[str, str, str, str]:
    identifiers = getattr(request.state, "request_identifiers", None)
    if isinstance(identifiers, tuple) and len(identifiers) == 3:
        request_id, session_id, trace_id = identifiers
    else:
        correlation = getattr(request.state, "correlation", None)
        request_id = str(getattr(correlation, "request_id", "unavailable"))
        session_id = "unavailable"
        trace_id = str(getattr(correlation, "trace_id", request_id))
    return request_id, session_id, trace_id, f"agent-{uuid.uuid4().hex}"


def failure_content(request: Request, code: ErrorCode, retryable: bool) -> dict[str, Any]:
    request_id, session_id, trace_id, agent_trace_id = _identifiers(request)
    return {
        "contractVersion": "recommendation-agent-v2",
        "requestId": request_id,
        "sessionId": session_id,
        "traceId": trace_id,
        "agentTraceId": agent_trace_id,
        "status": "failure",
        "error": {"code": code.value, "retryable": retryable},
    }


def _safe_http_code(exc: HTTPException) -> str:
    if isinstance(exc.detail, dict):
        code = exc.detail.get("code")
        if isinstance(code, str) and code.isascii() and code.replace("_", "").isalnum() and len(code) <= 64:
            return code
    return "HTTP_ERROR"


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AgentError)
    async def handle_agent_error(request: Request, exc: AgentError) -> JSONResponse:
        request_id, _session_id, trace_id, _agent_trace_id = _identifiers(request)
        logger.warning(
            "recommendation request failed code=%s status=%s retryable=%s request_id=%s trace_id=%s",
            exc.code.value,
            exc.http_status,
            exc.retryable,
            request_id,
            trace_id,
        )
        headers = {"Retry-After": "1"} if exc.code is ErrorCode.SERVICE_OVERLOADED else None
        return JSONResponse(
            status_code=exc.http_status,
            content=exc.failure_content or failure_content(request, exc.code, exc.retryable),
            headers=headers,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, _exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=400, content=failure_content(request, ErrorCode.INVALID_REQUEST, False))

    @app.exception_handler(HTTPException)
    async def handle_http_error(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"status": "failure", "error": {"code": _safe_http_code(exc), "retryable": False}},
            headers=exc.headers,
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, _exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=500, content=failure_content(request, ErrorCode.INTERNAL_ERROR, False))
