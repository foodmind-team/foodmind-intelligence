"""Stable non-sensitive HTTP error envelopes."""

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, _error: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "status": 422,
                "error_code": "SCHEMA_MISMATCH",
                "message": "The request does not match chat-agent-v2.",
                "retryable": False,
            },
        )

    @app.exception_handler(HTTPException)
    async def http_error(_request: Request, error: HTTPException) -> JSONResponse:
        detail = error.detail if isinstance(error.detail, dict) else {}
        return JSONResponse(
            status_code=error.status_code,
            content={
                "status": error.status_code,
                "error_code": str(detail.get("code", "REQUEST_REJECTED")),
                "message": "The Chat Agent rejected the request.",
                "retryable": error.status_code >= 500,
            },
        )
