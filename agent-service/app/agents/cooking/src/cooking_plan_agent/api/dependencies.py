"""FastAPI dependencies for internal service authentication and correlation ID.

Per handbook 9.3: the FastAPI module validates service authentication and
the internal request schema, NOT end-user JWTs. Spring Boot handles user
auth, ownership, and file validation before calling this internal endpoint.

Per handbook 9.10: accept Spring's X-Request-ID or generate one; return it
in response headers and include it in structured logs and provider metadata.
"""

import uuid
from hmac import compare_digest
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status

from cooking_plan_agent.config.settings import Settings, get_settings

# ---------------------------------------------------------------------------
# Safe characters for correlation IDs: alphanumeric, hyphens, underscores.
# Reject anything outside this set to prevent log injection.
# ---------------------------------------------------------------------------
_CORRELATION_ID_SAFE_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")

# Maximum length of a trusted correlation ID from a caller.
_MAX_CORRELATION_ID_LENGTH = 128


# ---------------------------------------------------------------------------
# Internal service authentication
# ---------------------------------------------------------------------------

# Stable error codes — never echo the supplied credential back to the client.
_AUTH_ERROR_CODES = {
    "missing": "MISSING_AUTHORIZATION_HEADER",
    "scheme": "INVALID_AUTHORIZATION_SCHEME",
    "credential": "INVALID_INTERNAL_CREDENTIAL",
    "weak": "INSUFFICIENT_CREDENTIAL_STRENGTH",
}


def _check_credential(
    credential: str,
    settings: Settings,
) -> str | None:
    """Return an error code when the credential is rejected, else None.

    Shared core for both native (X-Internal-Token) and compat (Bearer)
    authenticators (P0-08 rule 3). All comparisons are constant-time;
    the credential is never included in logs or error responses.
    """
    expected = settings.internal_service_token
    if not credential:
        return _AUTH_ERROR_CODES["missing"]
    if not compare_digest(credential, expected):
        return _AUTH_ERROR_CODES["credential"]
    # Token strength (P0-08 rule 4): non-local environments require a
    # token at least min_service_token_length chars. local/CI may use
    # short test tokens.
    if settings.environment != "local" and len(expected) < settings.min_service_token_length:
        return _AUTH_ERROR_CODES["weak"]
    return None


async def require_internal_service(
    x_internal_token: Annotated[str, Header()],
    settings: Annotated[Settings, Depends(get_settings)],
) -> None:
    """Validate the X-Internal-Token header against the configured secret.

    Uses hmac.compare_digest to prevent timing attacks. Returns nothing on
    success; raises 401 on mismatch.

    Handbook 9.3: for deployed environments, prefer network isolation plus
    a stronger service-authentication mechanism agreed by the team.
    Do not rely only on a hidden URL.
    """
    error_code = _check_credential(x_internal_token, settings)
    if error_code is not None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": error_code},
        )


async def require_bearer_service(
    settings: Annotated[Settings, Depends(get_settings)],
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    """Validate an ``Authorization: Bearer`` service credential.

    The Spring Boot caller sends the service token as a Bearer credential
    (CookingAgentHttpAdapter sets ``Authorization: Bearer <token>``).

    Shares the constant-time credential check with the native authenticator
    (P0-08 rule 3). The header is declared as an explicit optional FastAPI
    parameter so the OpenAPI schema advertises it, but a missing header is
    handled here as 401 (not FastAPI's 422 schema error) to match the compat
    contract.
    """
    if authorization is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": _AUTH_ERROR_CODES["missing"]},
        )
    scheme, _, credential = authorization.partition(" ")
    if scheme.lower() != "bearer" or not credential:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": _AUTH_ERROR_CODES["scheme"]},
        )
    error_code = _check_credential(credential, settings)
    if error_code is not None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": error_code},
        )


# ---------------------------------------------------------------------------
# Correlation ID extraction
# ---------------------------------------------------------------------------


def _validate_correlation_id(raw: str) -> str | None:
    """Return the raw value if it passes safety checks; None otherwise.

    Handbook 9.10: validate length and allowed characters before trusting
    a supplied X-Request-ID. This prevents log-injection attacks and
    ensures the ID is safe to embed in structured logs without escaping.
    """
    if not raw or len(raw) > _MAX_CORRELATION_ID_LENGTH:
        return None
    if not all(c in _CORRELATION_ID_SAFE_CHARS for c in raw):
        return None
    return raw


async def extract_correlation_id(
    request: Request,
    x_request_id: Annotated[str | None, Header(alias="X-Request-ID")] = None,
) -> str:
    """Extract or generate a correlation ID for the current request.

    Accepts Spring's X-Request-ID if present and valid; otherwise generates
    a UUID4. The ID is stored in request.state for access by exception
    handlers and logging middleware.

    Handbook 9.10: return it in response headers via a middleware or router
    response override.
    """
    supplied = _validate_correlation_id(x_request_id) if x_request_id else None
    correlation_id = supplied or uuid.uuid4().hex
    # Attach to request state for downstream consumers (error handlers, logs).
    request.state.correlation_id = correlation_id
    return correlation_id
