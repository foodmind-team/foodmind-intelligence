# FastAPI application entry point — owns app creation and route registration.
# The app-factory pattern keeps the ASGI instance lazily constructed and testable.
#
# Per handbook 9.5: construct long-lived HTTP/LLM clients during application
# lifespan and close them cleanly. Tests may override the dependency and
# supply fakes. Avoid creating clients at module import time.
import json
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from cooking_plan_agent.api import register_exception_handlers
from cooking_plan_agent.api import router as agent_router
from cooking_plan_agent.application import GenerateCookingPlanService
from cooking_plan_agent.workflow.context import WorkflowContext
from cooking_plan_agent.workflow.graph import build_cooking_plan_graph

# ============================================================================
# Structured JSON logging (Handbook 12.7)
# ============================================================================
# Uses python-json-logger if available, falls back to manual JSON formatting.
# Sensitive fields (recipe_text, prompts, tokens) are redacted at the log level.


class _RedactingJsonFormatter(logging.Formatter):
    """JSON log formatter that redacts sensitive fields per Handbook 12.7.

    Fields redacted: recipe text, inventory data, dietary rules,
    provider prompts/responses, credentials, internal service tokens.
    """

    _REDACT_FIELDS = frozenset({
        "recipe_text", "recipes", "inventory", "dietary_rules",
        "dietary_restrictions", "prompt", "response", "provider_payload",
        "internal_service_token", "credential", "api_key",
    })

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id  # type: ignore[attr-defined]
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = str(record.exc_info[1])

        # Redact any sensitive keys that slipped into extra
        for key, value in record.__dict__.items():
            if key in self._REDACT_FIELDS:
                log_entry[f"_{key}_redacted"] = True
            elif key not in {"name", "msg", "args", "levelname", "levelno",
                              "pathname", "filename", "module", "exc_info",
                              "exc_text", "stack_info", "lineno", "funcName",
                              "created", "msecs", "relativeCreated", "thread",
                              "threadName", "process", "message"}:
                log_entry[key] = value

        return json.dumps(log_entry, default=str)


def _configure_structured_logging() -> None:
    """Configure the root logger with JSON formatting for production.

    Handbook 12.7: structured logs with request_id, node, duration_ms, etc.
    In production, set COOKING_PLAN_LOG_FORMAT=json to enable.
    """
    log_format = os.environ.get("COOKING_PLAN_LOG_FORMAT", "text")
    if log_format == "json":
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_RedactingJsonFormatter())
        root = logging.getLogger()
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(os.environ.get("COOKING_PLAN_LOG_LEVEL", "INFO").upper())


_configure_structured_logging()
logger = logging.getLogger(__name__)

# ============================================================================
# Graceful shutdown state (Handbook 12.5)
# ============================================================================
# Track whether the server is shutting down so we can:
#   - Reject new requests with 503
#   - Allow in-flight work to complete (bounded timeout)
#   - Close provider clients cleanly

_shutting_down = False
_active_request_count = 0
_provider_clients: list = []  # populated when LLM/search providers are wired


# ---------------------------------------------------------------------------
# Lifespan — constructs long-lived services (handbook 9.5 + 12.5)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build and inject services on startup; graceful cleanup on shutdown.

    Handbook 9.5: construct provider clients and workflow context during lifespan.
    Handbook 12.5: on shutdown, stop accepting requests, wait for in-flight
    work to finish, close provider clients, flush logs.
    """
    global _shutting_down

    from cooking_plan_agent.config.settings import get_settings
    from cooking_plan_agent.research.config import DomainAllowList
    from cooking_plan_agent.research.providers.fake import FakeSearchProvider
    from cooking_plan_agent.research.researcher import Researcher

    settings = get_settings()

    # Reset shutdown flag on startup (critical for tests: module-level
    # global persists across TestClient instances and must be reset).
    _shutting_down = False

    logger.info(
        "Starting Cooking Plan Agent",
        extra={"environment": settings.environment, "log_level": settings.log_level},
    )

    # Mark service as ready after settings validation passes.
    # Used by /health/ready to report readiness.
    app.state.settings_validated = True

    # Wire RecipeResearcher when web research is enabled (handbook 10.1).
    recipe_researcher: Researcher | None = None
    if settings.web_research_enabled:
        allow_list = DomainAllowList.from_settings(
            custom_domains=settings.allowed_research_domains,
        )
        provider = FakeSearchProvider()
        recipe_researcher = Researcher(
            provider=provider,
            allow_list=allow_list,
            settings=settings,
        )

    workflow_context = WorkflowContext(
        recipe_extractor=None,  # type: ignore[arg-type] — wired when LLM integration lands
        recipe_researcher=recipe_researcher,
    )

    # Build and compile the LangGraph workflow graph once at startup.
    try:
        graph = build_cooking_plan_graph()
        app.state.graph_compiled = True
    except Exception as exc:  # noqa: BLE001 — lifespan must catch all to prevent hung startup
        logger.error("Failed to compile workflow graph", extra={"error": str(exc)})
        app.state.graph_compiled = False

    # Store the application service on app.state for route-level DI.
    app.state.generate_plan_service = GenerateCookingPlanService(
        graph=graph,
        context=workflow_context,
    )

    logger.info("Cooking Plan Agent ready")

    yield  # ---- Server is running ----

    # ---- Graceful shutdown (Handbook 12.5) ----
    logger.info("Shutting down Cooking Plan Agent")
    _shutting_down = True

    # Allow in-flight work to drain (bounded timeout).
    import asyncio
    drain_seconds = 10
    for _ in range(drain_seconds * 10):
        if _active_request_count <= 0:
            break
        await asyncio.sleep(0.1)
    if _active_request_count > 0:
        logger.warning(
            "Forcing shutdown with active requests",
            extra={"active_count": _active_request_count},
        )

    # Close provider clients.
    for client in _provider_clients:
        try:
            if hasattr(client, "aclose"):
                await client.aclose()
            elif hasattr(client, "close"):
                client.close()
        except Exception as exc:  # noqa: BLE001 — client.close() may raise from any provider
            logger.warning("Error closing provider client", extra={"error": str(exc)})

    # Flush structured logs/metrics.
    for handler in logging.getLogger().handlers:
        handler.flush()

    logger.info("Cooking Plan Agent stopped")


# ============================================================================
# SIGTERM handler — triggers graceful shutdown (Handbook 12.5)
# ============================================================================


def _handle_sigterm(signum, frame) -> None:
    """Set the shutdown flag so lifespan cleanup runs on ASGI server stop.

    The ASGI server (uvicorn) receives SIGTERM and calls the lifespan's
    teardown. This handler ensures we also set our shutdown flag early so
    new requests are rejected with 503 before the event loop stops.
    """
    global _shutting_down
    _shutting_down = True
    logger.info("Received SIGTERM — draining in-flight requests")


signal.signal(signal.SIGTERM, _handle_sigterm)


# ---------------------------------------------------------------------------
# Correlation ID response middleware (handbook 9.10)
# ---------------------------------------------------------------------------


async def _add_correlation_id_header(request, call_next):
    """Propagate the correlation ID from request.state into response headers."""
    response = await call_next(request)
    correlation_id = getattr(request.state, "correlation_id", None)
    if correlation_id:
        response.headers["X-Request-ID"] = correlation_id
    return response


# ---------------------------------------------------------------------------
# Shutdown middleware: reject new requests during graceful shutdown (12.5)
# ---------------------------------------------------------------------------


async def _shutdown_middleware(request, call_next):
    """Reject new requests with 503 when the server is shutting down.

    Handbook 12.5: stop accepting new requests; allow bounded in-flight
    work to finish or cancel cleanly.
    """
    global _active_request_count
    if _shutting_down:
        from starlette.responses import JSONResponse
        return JSONResponse(
            status_code=503,
            content={
                "status": "FAILED",
                "error_code": "SHUTTING_DOWN",
                "message": "Service is shutting down. Please retry.",
            },
        )
    _active_request_count += 1
    try:
        response = await call_next(request)
        return response
    finally:
        _active_request_count -= 1


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Build and configure the FastAPI application instance."""
    application = FastAPI(
        title="FoodMind Cooking Plan Agent",
        version="1.0.0",
        lifespan=lifespan,
    )

    application.include_router(agent_router)
    register_exception_handlers(application)

    # Shutdown middleware: must be outermost so it runs first on every request.
    application.middleware("http")(_shutdown_middleware)

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["POST"],
        allow_headers=["X-Request-ID", "X-Internal-Token", "Content-Type"],
        expose_headers=["X-Request-ID"],
    )
    application.middleware("http")(_add_correlation_id_header)

    # ---- Health endpoints (Handbook 12.4) ----

    @application.get("/health/live", tags=["health"])
    async def liveness() -> dict[str, str]:
        """Liveness: process/event loop is alive. No external calls.

        Handbook 12.4: used by orchestrators to verify basic process health.
        """
        return {"status": "alive"}

    @application.get("/health/ready", tags=["health"])
    async def readiness() -> dict:
        """Readiness: application is ready to serve traffic.

        Handbook 12.4: checks that the application graph/services were constructed
        and local configuration is valid. Returns 503 if not ready.
        """
        from starlette.responses import JSONResponse

        settings_ok = getattr(application.state, "settings_validated", False)
        graph_ok = getattr(application.state, "graph_compiled", False)
        shutting_down = _shutting_down

        ready = settings_ok and graph_ok and not shutting_down
        status_code = 200 if ready else 503

        return JSONResponse(
            status_code=status_code,
            content={
                "status": "ready" if ready else "not_ready",
                "checks": {
                    "settings_validated": settings_ok,
                    "graph_compiled": graph_ok,
                    "shutting_down": shutting_down,
                },
            },
        )

    return application


# Module-level singleton: the ASGI callable that uvicorn (or any server) imports.
app = create_app()
