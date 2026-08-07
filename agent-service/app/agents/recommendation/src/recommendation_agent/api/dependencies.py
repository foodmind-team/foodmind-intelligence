"""Authentication, correlation, and application-state dependencies."""

import uuid
from dataclasses import dataclass
from hmac import compare_digest
from typing import Annotated, cast

from fastapi import Header, HTTPException, Request, status

from recommendation_agent.application.service import RecommendationAgentService
from recommendation_agent.config.settings import Settings

_SAFE_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-")
_MAX_CORRELATION_LENGTH = 64


@dataclass(frozen=True, slots=True)
class Correlation:
    request_id: str
    trace_id: str


def _safe_correlation(raw: str | None) -> str | None:
    if raw is None or not raw or len(raw) > _MAX_CORRELATION_LENGTH:
        return None
    return raw if raw[0].isalnum() and all(char in _SAFE_CHARS for char in raw) else None


def get_request_settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def get_agent_service(request: Request) -> RecommendationAgentService:
    return cast(RecommendationAgentService, request.app.state.agent_service)


async def require_internal_service(
    request: Request,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> None:
    """Require an exact constant-time Bearer service credential."""

    if authorization is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail={"code": "MISSING_AUTHORIZATION_HEADER"})
    scheme, separator, credential = authorization.partition(" ")
    if scheme.lower() != "bearer" or not separator or not credential:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail={"code": "INVALID_AUTHORIZATION_SCHEME"})
    credential = credential.strip()
    settings = get_request_settings(request)
    expected = settings.internal_service_token.get_secret_value()
    if not compare_digest(credential, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail={"code": "INVALID_INTERNAL_CREDENTIAL"})
    if settings.app_env not in {"local", "test", "ci"} and len(expected) < settings.min_service_token_length:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail={"code": "INSUFFICIENT_CREDENTIAL_STRENGTH"})


async def extract_correlation(
    request: Request,
    x_request_id: Annotated[str | None, Header(alias="X-Request-ID")] = None,
    x_trace_id: Annotated[str | None, Header(alias="X-Trace-ID")] = None,
) -> Correlation:
    """Accept only bounded log-safe correlation, otherwise generate it."""

    request_id = _safe_correlation(x_request_id) or uuid.uuid4().hex
    trace_id = _safe_correlation(x_trace_id) or request_id
    correlation = Correlation(request_id=request_id, trace_id=trace_id)
    request.state.correlation = correlation
    return correlation
