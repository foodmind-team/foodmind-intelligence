"""Recommendation Agent ASGI composition root."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from recommendation_agent.api.backpressure import RequestLimiter
from recommendation_agent.api.errors import register_exception_handlers
from recommendation_agent.api.health import router as health_router
from recommendation_agent.api.middleware import BoundaryMiddleware
from recommendation_agent.api.router import router as agent_router
from recommendation_agent.application.ports import AgentWorkflow
from recommendation_agent.application.service import RecommendationAgentService
from recommendation_agent.clients.inference import RecommendationInferenceHttpClient
from recommendation_agent.config.settings import Settings, get_settings
from recommendation_agent.observability.logging import configure_logging
from recommendation_agent.observability.metrics import MetricsRegistry
from recommendation_agent.reasons.deriver import DeterministicReasonDeriver
from recommendation_agent.reasons.renderer import DeterministicExplanationRenderer
from recommendation_agent.selection.selector import DeterministicResultSelector
from recommendation_agent.time.budget import SystemClock
from recommendation_agent.workflow.context import WorkflowContext
from recommendation_agent.workflow.graph import BoundedRecommendationWorkflow


def create_app(
    *,
    settings: Settings | None = None,
    workflow: AgentWorkflow | None = None,
    workflow_complete: bool = False,
    inference_http_client: httpx.AsyncClient | None = None,
    install_default_workflow: bool = True,
) -> FastAPI:
    """Build the app without network, filesystem, model, or database access."""

    resolved_settings = settings or get_settings()
    metrics_registry = MetricsRegistry()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        configure_logging(resolved_settings.log_level)
        shared_http_client = inference_http_client or httpx.AsyncClient(
            base_url=resolved_settings.inference_base_url,
            follow_redirects=False,
            trust_env=False,
            limits=httpx.Limits(
                max_connections=resolved_settings.inference_max_connections,
                max_keepalive_connections=resolved_settings.inference_max_connections,
            ),
        )
        inference_client = RecommendationInferenceHttpClient(
            client=shared_http_client,
            settings=resolved_settings,
            metrics_registry=metrics_registry,
        )
        active_workflow = workflow
        active_workflow_complete = workflow is not None and workflow_complete
        policies_loaded = False
        if active_workflow is None and install_default_workflow:
            active_workflow = BoundedRecommendationWorkflow(
                WorkflowContext(
                    inference=inference_client,
                    selector=DeterministicResultSelector(),
                    reason_deriver=DeterministicReasonDeriver(),
                    renderer=DeterministicExplanationRenderer(),
                    settings=resolved_settings,
                    clock=SystemClock(),
                    metrics=metrics_registry,
                )
            )
            active_workflow_complete = True
            policies_loaded = True
        application.state.settings = resolved_settings
        application.state.inference_client = inference_client
        application.state.inference_configured = True
        application.state.metrics = metrics_registry
        application.state.policies_loaded = policies_loaded
        application.state.agent_service = RecommendationAgentService(active_workflow)
        application.state.request_limiter = RequestLimiter(
            max_active=resolved_settings.max_active_requests,
            max_queued=resolved_settings.max_queued_requests,
            timeout_seconds=resolved_settings.queue_timeout_seconds,
        )
        application.state.workflow_complete = active_workflow_complete
        application.state.shutting_down = False
        try:
            yield
        finally:
            application.state.shutting_down = True
            await application.state.request_limiter.close()
            await application.state.request_limiter.wait_for_drain(resolved_settings.shutdown_drain_seconds)
            await application.state.agent_service.aclose()
            await inference_client.aclose()

    application = FastAPI(
        title="FoodMind Recommendation Agent",
        version="2.0.0",
        lifespan=lifespan,
        docs_url=None if resolved_settings.app_env in {"staging", "production"} else "/docs",
        redoc_url=None if resolved_settings.app_env in {"staging", "production"} else "/redoc",
        openapi_url=None if resolved_settings.app_env in {"staging", "production"} else "/openapi.json",
    )
    application.add_middleware(BoundaryMiddleware, settings=resolved_settings)
    if resolved_settings.cors_allow_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=list(resolved_settings.cors_allow_origins),
            allow_credentials=True,
            allow_methods=["POST"],
            allow_headers=["Authorization", "Content-Type", "X-Request-ID", "X-Trace-ID"],
            expose_headers=["X-Request-ID", "X-Trace-ID"],
        )
    application.include_router(agent_router)
    application.include_router(health_router)
    register_exception_handlers(application)
    return application


app = create_app()


def run() -> None:
    uvicorn.run("recommendation_agent.main:app", host="0.0.0.0", port=8000, log_config=None)  # noqa: S104


if __name__ == "__main__":
    run()
