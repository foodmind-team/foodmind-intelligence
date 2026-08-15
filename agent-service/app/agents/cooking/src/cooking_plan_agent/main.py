# FastAPI application entry point — owns app creation and route registration.
# The app-factory pattern keeps the ASGI instance lazily constructed and testable.
#
# Per handbook 9.5: construct long-lived HTTP/LLM clients during application
# lifespan and close them cleanly. Tests may override the dependency and
# supply fakes. Avoid creating clients at module import time.
import logging
import signal
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from cooking_plan_agent.api import register_exception_handlers
from cooking_plan_agent.api import router as agent_router
from cooking_plan_agent.api.compat_router import router as compat_router
from cooking_plan_agent.api.health import router as health_router
from cooking_plan_agent.api.middleware import (
    add_correlation_id_header,
    get_active_request_count,
    set_shutting_down,
    shutdown_middleware,
)
from cooking_plan_agent.application import GenerateCookingPlanService, ParseRecipeImportService
from cooking_plan_agent.observability.logging import RedactingJsonFormatter, configure_structured_logging
from cooking_plan_agent.safety.engine import SafetyEngine
from cooking_plan_agent.workflow.context import WorkflowContext
from cooking_plan_agent.workflow.graph import build_cooking_plan_graph

configure_structured_logging()
# Compatibility for existing integrations that imported the formatter from the
# historical application entry point.
_RedactingJsonFormatter = RedactingJsonFormatter
logger = logging.getLogger(__name__)

# ============================================================================
# Provider clients (Handbook 12.5)
# ============================================================================
# Long-lived clients are closed cleanly on shutdown. Shutdown gating state
# (_shutting_down / _active_request_count) lives in api.middleware.
_provider_clients: list[object] = []  # populated when LLM/search providers are wired


# ---------------------------------------------------------------------------
# Lifespan — constructs long-lived services (handbook 9.5 + 12.5)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build and inject services on startup; graceful cleanup on shutdown.

    Handbook 9.5: construct provider clients and workflow context during lifespan.
    Handbook 12.5: on shutdown, stop accepting requests, wait for in-flight
    work to finish, close provider clients, flush logs.
    """
    from cooking_plan_agent.config.settings import get_settings

    settings = get_settings()

    # P1-03: process-level request limiter (active + queued layers). Created
    # here so it can be injected via FastAPI dependencies; health endpoints
    # bypass it and read the snapshot instead.
    from cooking_plan_agent.api.backpressure import RequestLimiter

    app.state.request_limiter = RequestLimiter(
        max_active=settings.max_active_requests,
        max_queued=settings.max_queued_requests,
        queue_timeout_seconds=settings.queue_timeout_seconds,
    )

    # P1-06: intermediate-artifact cache (parse/research results). In-memory,
    # instance-level; disabled by default. Only affects performance.
    from cooking_plan_agent.infrastructure.cache import InMemoryTTLCache

    cache: InMemoryTTLCache[str, object] | None = None
    if settings.cache_enabled:
        cache = InMemoryTTLCache(
            max_entries=settings.cache_max_entries,
            max_item_size_bytes=settings.cache_max_item_bytes,
            default_ttl_seconds=settings.cache_ttl_seconds,
        )
        _provider_clients.append(cache)  # closed (cleared) on shutdown

    # P2-06: workflow checkpointer (node-boundary persistence). Created in
    # the lifespan so no connection is opened at import time; closed on
    # shutdown. None keeps the graph stateless (pre-P2-06 behaviour).
    from cooking_plan_agent.infrastructure.checkpointer import (
        CheckpointProvider,
        create_checkpoint_provider,
    )

    checkpoint_provider: CheckpointProvider | None = create_checkpoint_provider(settings)
    if checkpoint_provider is not None:
        try:
            await checkpoint_provider.astart()
            app.state.checkpoint_enabled = True
            logger.info(
                "Checkpoint persistence enabled | backend=%s",
                settings.checkpoint_backend,
            )
        except Exception as exc:  # noqa: BLE001 — startup must not hang on storage failure
            logger.error(
                "Failed to start checkpointer — running stateless",
                extra={"error": str(exc)},
            )
            checkpoint_provider = None
    else:
        app.state.checkpoint_enabled = False

    # Reset shutdown flag on startup (critical for tests: module-level
    # global persists across TestClient instances and must be reset).
    set_shutting_down(False)

    logger.info(
        "Starting Cooking Plan Agent",
        extra={"environment": settings.environment, "log_level": settings.log_level},
    )

    # Mark service as ready after settings validation passes.
    # Used by /health/ready to report readiness.
    app.state.settings_validated = True

    # ---- LLM wiring (local Ollama via OpenAI-compatible API) ----
    # Provider-neutral: any OpenAI-compatible endpoint can be swapped via
    # COOKING_PLAN_LLM_* settings. LLM is enabled by default; set
    # llm_enabled=False to keep the rule-based pipeline (e.g. CI stays
    # offline-deterministic).
    from cooking_plan_agent.llm import (
        LLMClient,
        LLMKnowledgeResearcher,
        LLMPlanExplainer,
        LLMRecipeExtractor,
        LLMRecipeImportExtractor,
    )

    # LLM master switch: a None client means "LLM disabled" and the whole
    # pipeline falls back to rule-based parsing.
    llm_client: LLMClient | None = None
    if settings.llm_enabled:
        # Create the process-level LLM client. Provider-neutral: base_url/model
        # can be swapped to any OpenAI-compatible endpoint (Ollama / cloud) via
        # COOKING_PLAN_LLM_* env vars.
        # api_key is unwrapped from SecretStr; None when unset (local Ollama needs none).
        llm_client = LLMClient(
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            api_key=settings.llm_api_key.get_secret_value() if settings.llm_api_key else None,
            # Timeout / retries / temperature / output cap / pool — all from COOKING_PLAN_LLM_*.
            timeout_seconds=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
            temperature=settings.llm_temperature,
            max_output_tokens=settings.llm_max_output_tokens,
            connection_pool_size=settings.llm_connection_pool_size,
        )
        # P1-02: one lifecycle-level client, created once and closed exactly
        # once on shutdown (no per-call connection-pool rebuild).
        _provider_clients.append(llm_client)
        # Schedule explainer (P4-01): LLM-generated "why this schedule" prose —
        # an additive capability with a deterministic fallback on failure.
        app.state.llm_explainer = LLMPlanExplainer(llm_client)
        logger.info(
            "LLM integration enabled",
            extra={"llm_model": settings.llm_model, "llm_base_url": settings.llm_base_url},
        )
    else:
        # LLM disabled: leave the explainer unset so downstream graph nodes
        # take the deterministic / disabled branch.
        app.state.llm_explainer = None

    # RecipeExtractor: single-recipe structuring. LLM-backed when the client is
    # enabled; None otherwise — downstream nodes fall back to the rule-based
    # extractor (parsing.extractor.RecipeExtractor) on a None value.
    # `recipe_extraction_llm_timeout_seconds` is intentionally shorter than the
    # shared LLM timeout so interactive plan generation still leaves time for
    # gap completion and scheduling before the request deadline.
    recipe_extractor = (
        LLMRecipeExtractor(
            llm_client,
            timeout_seconds=settings.recipe_extraction_llm_timeout_seconds,
        )
        if llm_client is not None
        else None
    )

    # Recipe import is a separate application boundary: it extracts multiple
    # partial recipes and asks only for facts required before persistence.
    from cooking_plan_agent.parsing.recipe_imports import DeterministicRecipeImportExtractor

    # The rule importer is always built: it splits pasted multi-dish text into
    # blocks and extracts each with the rule extractor. When LLM is enabled it
    # doubles as the LLM importer's fallback (LLM failure -> rule importer);
    # when LLM is disabled it is used directly.
    deterministic_importer = DeterministicRecipeImportExtractor()
    recipe_import_extractor = (
        LLMRecipeImportExtractor(
            llm_client,
            deterministic_importer,
            # Shorter timeout: import is served through a Backend client with a
            # 30s read deadline, so fall back before exhausting that boundary.
            timeout_seconds=settings.recipe_import_llm_timeout_seconds,
            # Larger output cap: one JSON with a ~6-dish array exceeds the shared
            # 2048-token budget; truncation used to force the weak rule fallback.
            max_output_tokens=settings.recipe_import_max_output_tokens,
        )
        if llm_client is not None
        else deterministic_importer
    )
    # answer_normaliser translates free-text clarification answers (e.g. Chinese
    # free text) into English. Only the LLM importer has normalise_answers(); the
    # rule importer cannot translate, so the normaliser stays unset.
    app.state.recipe_import_service = ParseRecipeImportService(
        recipe_import_extractor,
        answer_normaliser=(
            recipe_import_extractor if isinstance(recipe_import_extractor, LLMRecipeImportExtractor) else None
        ),
    )

    # RecipeResearcher: gap completion is LLM-only now (web search removed).
    # When the LLM client is present it answers heat/duration/temperature gaps
    # from model knowledge; when absent, no researcher is wired and downstream
    # falls back to confirmation / deterministic defaults.
    recipe_researcher: LLMKnowledgeResearcher | None = (
        LLMKnowledgeResearcher(llm_client) if llm_client is not None else None
    )

    workflow_context = WorkflowContext(
        recipe_extractor=recipe_extractor,  # type: ignore[arg-type]
        recipe_researcher=recipe_researcher,
        safety_engine=SafetyEngine(),
        cache=cache,  # type: ignore[arg-type]
        # P4-01: optional schedule explainer (LLM). None keeps the graph
        # deterministic — the explain node then emits "disabled"/deterministic.
        explainer=app.state.llm_explainer,
    )

    # P5-2: wire the LLM ReAct controller when the agentic mode is enabled.
    # The ToolRegistry is built from the context (its tools mirror the
    # injectable services), so the controller is constructed after the base
    # context and re-assembled via dataclasses.replace — no circular DI.
    if settings.agent_controller_enabled and llm_client is not None:
        import dataclasses

        from cooking_plan_agent.llm import LLMReActController
        from cooking_plan_agent.tooling.registry import ToolRegistry

        registry = ToolRegistry(workflow_context)
        controller = LLMReActController(llm_client, tools=registry.specs())
        workflow_context = dataclasses.replace(workflow_context, agent_controller=controller)
        logger.info(
            "ReAct agent controller enabled | tools=%d | max_steps=%d",
            len(registry.specs()),
            settings.agent_max_steps,
        )

    # P5-4: wire the long-term preference store when persistence is enabled.
    # Missing user_id on a request keeps behaviour identical (zero regression).
    if settings.confirmation_dialog_enabled:
        import dataclasses

        from cooking_plan_agent.infrastructure.preferences import PreferenceStore

        store = PreferenceStore(f"{settings.checkpoint_sqlite_path}.prefs")
        workflow_context = dataclasses.replace(workflow_context, preference_store=store)

    # Build and compile the LangGraph workflow graph once at startup.
    try:
        # P2-06: inject the checkpointer when persistence is enabled so
        # node-boundary state survives process restart.
        graph = build_cooking_plan_graph(checkpointer=checkpoint_provider.checkpointer if checkpoint_provider else None)
        app.state.graph_compiled = True
    except Exception as exc:  # noqa: BLE001 — lifespan must catch all to prevent hung startup
        logger.error("Failed to compile workflow graph", extra={"error": str(exc)})
        app.state.graph_compiled = False

    # Store the application service on app.state for route-level DI.
    app.state.generate_plan_service = GenerateCookingPlanService(
        graph=graph,
        context=workflow_context,
    )

    # P3-01: async task API — in-process worker + SQLite task store. The
    # synchronous endpoints remain; this service only runs when enabled.
    app.state.task_service = None
    task_service: object | None = None
    if settings.task_api_enabled:
        from cooking_plan_agent.tasks.queue import create_task_queue
        from cooking_plan_agent.tasks.repository import SQLiteTaskRepository
        from cooking_plan_agent.tasks.service import AsyncTaskService

        task_repo = SQLiteTaskRepository(settings.task_db_path)
        try:
            await task_repo.astart()
            # P4-05: worker consumes a TaskQueue port selected by settings.
            # Only "inprocess" is supported until Stage B infrastructure
            # (Redis queue + shared quota storage) is approved.
            task_queue = create_task_queue(settings.task_queue_backend, task_repo)
            task_service = AsyncTaskService(
                repository=task_repo,
                generation_service=app.state.generate_plan_service,
                default_ttl_seconds=settings.task_default_ttl_seconds,
                worker_concurrency=settings.task_worker_concurrency,
                queue=task_queue,
            )
            await task_service.astart()
            app.state.task_service = task_service
            logger.info(
                "Async task API enabled | db=%s | worker_concurrency=%d | queue_backend=%s",
                settings.task_db_path,
                settings.task_worker_concurrency,
                settings.task_queue_backend,
            )
        except Exception as exc:  # noqa: BLE001 — startup must not hang on storage failure
            logger.error(
                "Failed to start async task service — task API disabled",
                extra={"error": str(exc)},
            )
            await task_repo.close()

    logger.info("Cooking Plan Agent ready")

    yield  # ---- Server is running ----

    # ---- Graceful shutdown (Handbook 12.5) ----
    logger.info("Shutting down Cooking Plan Agent")
    set_shutting_down(True)

    # Allow in-flight work to drain (bounded timeout).
    import asyncio

    drain_seconds = 10
    for _ in range(drain_seconds * 10):
        if get_active_request_count() <= 0:
            break
        await asyncio.sleep(0.1)
    if get_active_request_count() > 0:
        logger.warning(
            "Forcing shutdown with active requests",
            extra={"active_count": get_active_request_count()},
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

    # P2-06: close the checkpointer connection so the SQLite file is
    # flushed and no file handle leaks across restarts.
    if checkpoint_provider is not None:
        try:
            await checkpoint_provider.aclose()
        except Exception as exc:  # noqa: BLE001 — shutdown must not raise
            logger.warning("Error closing checkpointer", extra={"error": str(exc)})

    # P3-01: stop the async task worker (bounded drain) and close the store.
    if task_service is not None:
        try:
            await task_service.aclose()  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 — shutdown must not raise
            logger.warning("Error closing task service", extra={"error": str(exc)})

    # Flush structured logs/metrics.
    for handler in logging.getLogger().handlers:
        handler.flush()

    logger.info("Cooking Plan Agent stopped")


# ============================================================================
# SIGTERM handler — triggers graceful shutdown (Handbook 12.5)
# ============================================================================


def _handle_sigterm(signum: int, frame: object | None) -> None:
    """Set the shutdown flag so lifespan cleanup runs on ASGI server stop.

    The ASGI server (uvicorn) receives SIGTERM and calls the lifespan's
    teardown. This handler ensures we also set our shutdown flag early so
    new requests are rejected with 503 before the event loop stops.
    """
    set_shutting_down(True)
    logger.info("Received SIGTERM — draining in-flight requests")


signal.signal(signal.SIGTERM, _handle_sigterm)


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
    application.include_router(compat_router)
    # P3-01: async task API (registered only when enabled; the router is
    # harmless to include — auth guards every route).
    from cooking_plan_agent.api.task_router import router as task_router

    application.include_router(task_router)
    register_exception_handlers(application)

    # Shutdown middleware: must be outermost so it runs first on every request.
    application.middleware("http")(shutdown_middleware)

    # ---- CORS (P0-08) ----
    # Internal APIs do NOT enable CORS by default. Only when an explicit
    # allow-list is configured do we register the middleware — and a
    # wildcard origin is rejected outright (credentials + "*" is unsafe).
    from cooking_plan_agent.config.settings import get_settings

    settings = get_settings()
    if settings.cors_allow_origins:
        if "*" in settings.cors_allow_origins:
            raise RuntimeError(
                "cors_allow_origins must not contain '*' — internal API CORS "
                "requires an explicit origin allow-list (P0-08)."
            )
        application.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_allow_origins),
            allow_credentials=True,
            allow_methods=["POST"],
            allow_headers=["X-Request-ID", "X-Internal-Token", "Content-Type"],
            expose_headers=["X-Request-ID"],
        )

    application.middleware("http")(add_correlation_id_header)

    # ---- Health endpoints (Handbook 12.4) ----
    application.include_router(health_router)

    return application


# Module-level singleton: the ASGI callable that uvicorn (or any server) imports.
app = create_app()
