"""Liveness and readiness endpoints."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(tags=["health"])


@router.get("/health/live")
async def liveness() -> dict[str, str]:
    return {"status": "alive"}


@router.get("/health/ready", response_model=None)
async def readiness(request: Request) -> JSONResponse:
    workflow_ready = bool(getattr(request.app.state, "workflow_complete", False))
    inference_configured = bool(getattr(request.app.state, "inference_configured", False))
    policies_loaded = bool(getattr(request.app.state, "policies_loaded", False))
    shutting_down = bool(getattr(request.app.state, "shutting_down", False))
    ready = workflow_ready and inference_configured and policies_loaded and not shutting_down
    request.app.state.metrics.set_readiness(ready)
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "ready" if ready else "not_ready",
            "checks": {
                "configuration": True,
                "inference": inference_configured,
                "policies": policies_loaded,
                "workflow": workflow_ready,
                "shutdown": not shutting_down,
            },
        },
    )
