"""Health endpoints (Handbook 12.4).

Readiness probes the lifespan-constructed services on ``app.state``; load
probes the request limiter. All three bypass the business limiter so
orchestrators can always probe the process.
"""

# 模块概览（中文）：健康检查端点（Handbook 12.4）。
# - readiness：探活 lifespan 构建的服务（挂在 app.state 上）
# - load：探测请求限流器负载
# 三个端点都绕过业务限流器，保证编排器（orchestrator）始终能探测进程。

from fastapi import APIRouter, Request
from starlette.responses import JSONResponse

from cooking_plan_agent.api.middleware import is_shutting_down

router = APIRouter(tags=["health"])


@router.get("/health/live")
async def liveness() -> dict[str, str]:
    """Liveness: process/event loop is alive. No external calls."""
    # 存活探针：进程/事件循环是否存活，不做任何外部调用
    return {"status": "alive"}


@router.get("/health/ready", response_model=dict[str, object])
async def readiness(request: Request) -> JSONResponse:
    """Readiness: application is ready to serve traffic.

    Checks that the graph/services were constructed and local configuration is
    valid. Returns 503 if not ready.
    """
    # 就绪探针：应用是否可对外服务。检查图/服务是否已构建、本地配置是否合法；未就绪返回 503。
    from cooking_plan_agent.config.settings import get_settings

    settings = get_settings()
    state = request.app.state
    settings_ok = getattr(state, "settings_validated", False)  # 配置是否已校验
    graph_ok = getattr(state, "graph_compiled", False)  # 图是否已编译
    # task API：若启用则要求 task_service 已就绪；未启用则视为 OK
    task_api_ok = not settings.task_api_enabled or getattr(state, "task_service", None) is not None
    shutting_down = is_shutting_down()  # 是否正在优雅停机

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
    # 负载探针：从请求限流器取负载快照（P1-03）
    from cooking_plan_agent.api.backpressure import RequestLimiter

    limiter = getattr(request.app.state, "request_limiter", None)
    if not isinstance(limiter, RequestLimiter):
        return {"limiter": "not_initialised"}  # 限流器未初始化
    snapshot = limiter.snapshot()
    return {
        "active": snapshot.active,  # 当前活动请求数
        "queued": snapshot.queued,  # 当前排队请求数
        "rejected_total": snapshot.rejected_total,  # 累计被拒绝数
        "queue_wait_ms": snapshot.queue_wait_ms,  # 最近一次排队等待时长
    }
