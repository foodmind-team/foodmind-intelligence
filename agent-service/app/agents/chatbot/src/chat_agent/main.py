"""Chat Agent ASGI composition root."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

import httpx
import uvicorn
from fastapi import FastAPI

from chat_agent.api.backpressure import RequestLimiter
from chat_agent.api.errors import register_exception_handlers
from chat_agent.api.router import router
from chat_agent.clients.backend import BackendToolClient
from chat_agent.config.settings import Settings, get_settings
from chat_agent.llm.client import LLMClient


def create_app(
    *,
    settings: Settings | None = None,
    llm_client: LLMClient | None = None,
    backend_tool_client: BackendToolClient | None = None,
) -> FastAPI:
    resolved = settings or get_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        logging.basicConfig(level=resolved.log_level)
        owned_client = llm_client
        owned_tool_client = backend_tool_client
        api_key = resolved.llm_api_key.get_secret_value() if resolved.llm_api_key is not None else None
        if owned_client is None and resolved.llm_enabled and api_key:
            owned_client = LLMClient(
                base_url=resolved.llm_base_url,
                model=resolved.llm_model,
                api_key=api_key,
                timeout_seconds=resolved.llm_timeout_seconds,
                max_retries=resolved.llm_max_retries,
                temperature=resolved.llm_temperature,
                thinking_enabled=resolved.llm_thinking_enabled,
                max_output_tokens=resolved.llm_max_output_tokens,
                connection_pool_size=resolved.llm_connection_pool_size,
            )
        application.state.settings = resolved
        application.state.llm_client = owned_client
        if owned_tool_client is None:
            owned_tool_client = BackendToolClient(
                client=httpx.AsyncClient(
                    base_url=resolved.backend_base_url,
                    follow_redirects=False,
                    trust_env=False,
                ),
                settings=resolved,
            )
        application.state.backend_tool_client = owned_tool_client
        application.state.request_limiter = RequestLimiter(
            max_active=resolved.max_active_requests,
            max_queued=resolved.max_queued_requests,
            timeout_seconds=resolved.queue_timeout_seconds,
        )
        try:
            yield
        finally:
            if owned_client is not None and owned_client is not llm_client:
                await owned_client.aclose()
            if owned_tool_client is not backend_tool_client:
                await owned_tool_client.aclose()

    application = FastAPI(title="FoodMind Chat Agent", version="1.0.0", lifespan=lifespan)
    application.include_router(router)

    @application.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "alive"}

    @application.get("/health/ready")
    async def ready() -> dict[str, str | bool]:
        api_key_configured = resolved.llm_api_key is not None and bool(resolved.llm_api_key.get_secret_value())
        return {
            "status": "ready",
            "llmEnabled": resolved.llm_enabled,
            "llmConfigured": api_key_configured,
            "llmProviderHost": urlsplit(resolved.llm_base_url).hostname or "",
            "llmModel": resolved.llm_model,
            "llmThinkingEnabled": resolved.llm_thinking_enabled,
        }

    register_exception_handlers(application)
    return application


app = create_app()


def run() -> None:
    uvicorn.run("chat_agent.main:app", host="0.0.0.0", port=8001, log_config=None)  # noqa: S104


if __name__ == "__main__":
    run()
