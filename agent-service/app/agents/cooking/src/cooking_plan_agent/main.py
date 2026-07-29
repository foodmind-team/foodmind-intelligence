# FastAPI application entry point — owns app creation and route registration.
# The app-factory pattern keeps the ASGI instance lazily constructed and testable.
#
# Per handbook 9.5: construct long-lived HTTP/LLM clients during application
# lifespan and close them cleanly. Tests may override the dependency and
# supply fakes. Avoid creating clients at module import time.
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from cooking_plan_agent.api import register_exception_handlers
from cooking_plan_agent.api import router as agent_router
from cooking_plan_agent.application import GenerateCookingPlanService
from cooking_plan_agent.workflow.context import WorkflowContext
from cooking_plan_agent.workflow.graph import build_cooking_plan_graph

# ---------------------------------------------------------------------------
# Lifespan — constructs long-lived services (handbook 9.5)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build and inject the cooking plan service on startup; clean up on shutdown.

    Handbook 9.5: construct provider clients and workflow context during
    lifespan. Tests may override app.state to inject fakes.
    """
    # Build the immutable workflow context. In MVP, the RecipeExtractor
    # and RecipeResearcher are not yet wired — the graph's stub nodes
    # handle this gracefully. When providers are integrated, instantiate
    # real implementations here and pass them into WorkflowContext.
    workflow_context = WorkflowContext(
        recipe_extractor=None,  # type: ignore[arg-type] — wired when LLM integration lands
        recipe_researcher=None,
    )

    # Build and compile the LangGraph workflow graph once at startup.
    graph = build_cooking_plan_graph()

    # Store the application service on app.state for route-level DI.
    app.state.generate_plan_service = GenerateCookingPlanService(
        graph=graph,
        context=workflow_context,
    )

    yield

    # Cleanup: close any provider clients when wired.
    # await close_provider_clients(...)


# ---------------------------------------------------------------------------
# Correlation ID response middleware (handbook 9.10)
# ---------------------------------------------------------------------------


async def _add_correlation_id_header(request, call_next):
    """Propagate the correlation ID from request.state into response headers.

    This middleware reads the correlation ID set by extract_correlation_id()
    and echoes it as X-Request-ID in the response, so Spring Boot can
    correlate requests end-to-end.
    """
    response = await call_next(request)
    correlation_id = getattr(request.state, "correlation_id", None)
    if correlation_id:
        response.headers["X-Request-ID"] = correlation_id
    return response


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Build and configure the FastAPI application instance.

    Uses the app-factory pattern so tests and alternative entry points can
    construct fresh, isolated instances without relying on module-level state.

    Returns:
        FastAPI: A fully configured but not-yet-running ASGI application.

    Raises:
        None — this function performs no I/O and always succeeds.
    """
    # Create the core FastAPI instance with metadata visible in /docs.
    application = FastAPI(
        title="FoodMind Cooking Plan Agent",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Register the internal agent router (handbook 9.6).
    application.include_router(agent_router)

    # Register global exception handlers (handbook 9.9, layer 3).
    register_exception_handlers(application)

    # Correlation ID middleware: echoes X-Request-ID in every response
    # so Spring Boot can trace requests end-to-end (handbook 9.10).
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["POST"],
        allow_headers=["X-Request-ID", "X-Internal-Token", "Content-Type"],
        expose_headers=["X-Request-ID"],
    )
    # Pure ASGI middleware for correlation ID injection.
    application.middleware("http")(_add_correlation_id_header)

    # Health endpoint: used by orchestrators (e.g. Spring Boot) to verify
    # service liveness before forwarding traffic.
    @application.get("/health/live", tags=["health"])
    async def liveness() -> dict[str, str]:
        return {"status": "alive"}

    return application


# Module-level singleton: the ASGI callable that uvicorn (or any server) imports.
app = create_app()
