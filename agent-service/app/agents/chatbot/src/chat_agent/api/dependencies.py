"""Authentication and application-state dependencies."""

from hmac import compare_digest
from typing import Annotated, cast

from fastapi import Header, HTTPException, Request, status

from chat_agent.clients.backend import BackendToolClient
from chat_agent.config.settings import Settings
from chat_agent.llm.client import LLMClient


def get_settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def get_llm_client(request: Request) -> LLMClient | None:
    return cast(LLMClient | None, request.app.state.llm_client)


def get_backend_tool_client(request: Request) -> BackendToolClient:
    return cast(BackendToolClient, request.app.state.backend_tool_client)


async def require_internal_service(
    request: Request,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> None:
    if authorization is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail={"code": "MISSING_AUTHORIZATION_HEADER"})
    scheme, separator, credential = authorization.partition(" ")
    if scheme.lower() != "bearer" or not separator or not credential:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail={"code": "INVALID_AUTHORIZATION_SCHEME"})
    settings = get_settings(request)
    if not compare_digest(credential, settings.internal_service_token.get_secret_value()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail={"code": "INVALID_INTERNAL_CREDENTIAL"})
