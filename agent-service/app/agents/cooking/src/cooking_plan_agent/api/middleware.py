"""HTTP middlewares and their shared graceful-shutdown state.

Registered by ``main.create_app``. The module-level shutdown flags are shared
with ``main.lifespan`` and the SIGTERM handler through the small accessor
functions below, keeping the middlewares self-contained.
"""

# 模块概览（中文）：HTTP 中间件及其共享的优雅停机状态。
# 由 main.create_app 注册。模块级停机标志通过下方小型访问器函数，
# 与 main.lifespan 及 SIGTERM 处理器共享，保持中间件自包含。

from collections.abc import Awaitable, Callable

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

# ---------------------------------------------------------------------------
# Shared shutdown state (Handbook 12.5)
# 共享停机状态（Handbook 12.5）
# ---------------------------------------------------------------------------

_shutting_down = False  # 是否正在停机
_active_request_count = 0  # 当前在途请求数


def set_shutting_down(value: bool) -> None:
    """Set the shutdown flag (True once graceful shutdown begins)."""
    # 设置停机标志（优雅停机开始时置 True）
    global _shutting_down
    _shutting_down = value


def is_shutting_down() -> bool:
    """Return whether the server is shutting down."""
    # 返回服务器是否正在停机
    return _shutting_down


def get_active_request_count() -> int:
    """Return the number of in-flight requests."""
    # 返回在途请求数
    return _active_request_count


# ---------------------------------------------------------------------------
# Correlation ID response middleware (handbook 9.10)
# 关联 ID 响应中间件（handbook 9.10）
# ---------------------------------------------------------------------------


async def add_correlation_id_header(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Propagate the correlation ID from request.state into response headers."""
    # 把 request.state 中的关联 ID 回写到响应头
    response = await call_next(request)
    correlation_id = getattr(request.state, "correlation_id", None)
    if correlation_id:
        response.headers["X-Request-ID"] = correlation_id
    return response


# ---------------------------------------------------------------------------
# Shutdown middleware: reject new requests during graceful shutdown (12.5)
# 停机中间件：优雅停机期间拒绝新请求（12.5）
# ---------------------------------------------------------------------------


async def shutdown_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Reject new requests with 503 when the server is shutting down.

    Handbook 12.5: stop accepting new requests; allow bounded in-flight
    work to finish or cancel cleanly.

    P3-05: the 503 body uses the unified ErrorEnvelope so shutdown is
    indistinguishable (structurally) from backpressure overload.
    """
    # 停机时拒绝新请求并返回 503；让在途请求有界地完成或取消。
    global _active_request_count
    if _shutting_down:
        from cooking_plan_agent.domain.models import ErrorEnvelope

        correlation_id = str(getattr(request.state, "correlation_id", "unknown"))
        envelope = ErrorEnvelope(
            status=503,
            error_code="SHUTTING_DOWN",
            message="Service is shutting down. Please retry.",
            correlation_id=correlation_id,
            retryable=True,
        )
        return JSONResponse(status_code=503, content=envelope.model_dump())
    _active_request_count += 1  # 在途请求数 +1
    try:
        response = await call_next(request)
        return response
    finally:
        _active_request_count -= 1  # 在途请求数 -1
