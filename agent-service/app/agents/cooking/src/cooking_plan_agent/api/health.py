"""Health endpoints (Handbook 12.4).

Readiness probes the lifespan-constructed services on ``app.state``; load
probes the request limiter. All three bypass the business limiter so
orchestrators can always probe the process.
"""

from fastapi import APIRouter, Request
from starlette.responses import JSONResponse

from cooking_plan_agent.api.middleware import is_shutting_down

router = APIRouter(tags=["health"])


@router.get("/health/live")
async def liveness() -> dict[str, str]:
    """Liveness: process/event loop is alive. No external calls."""
    return {"status": "alive"}


@router.get("/health/ready", response_model=dict[str, object])
async def readiness(request: Request) -> JSONResponse:
    """Readiness: application is ready to serve traffic.

    Checks that the graph/services were constructed and local configuration is
    valid. Returns 503 if not ready.
    """
    from cooking_plan_agent.config.settings import get_settings

    settings = get_settings()
    state = request.app.state
    settings_ok = getattr(state, "settings_validated", False)
    graph_ok = getattr(state, "graph_compiled", False)
    task_api_ok = not settings.task_api_enabled or getattr(state, "task_service", None) is not None
    shutting_down = is_shutting_down()

    ready = settings_ok and graph_ok and task_api_ok and not shutting_down
    status_code = 200 if ready else 503

    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ready" if ready else "not_ready",
            "checks": {
                "settings_validated": settings_ok,
                "graph_compiled": graph_ok,
                "task_api_ready": task_api_ok,
                "shutting_down": shutting_down,
            },
        },
    )


@router.get("/health/load")
async def load_snapshot(request: Request) -> dict[str, object]:
    """Load snapshot from the request limiter (P1-03)."""
    from cooking_plan_agent.api.backpressure import RequestLimiter

    limiter = getattr(request.app.state, "request_limiter", None)
    if not isinstance(limiter, RequestLimiter):
        return {"limiter": "not_initialised"}
    snapshot = limiter.snapshot()
    return {
        "active": snapshot.active,
        "queued": snapshot.queued,
        "rejected_total": snapshot.rejected_total,
        "queue_wait_ms": snapshot.queue_wait_ms,
    }
