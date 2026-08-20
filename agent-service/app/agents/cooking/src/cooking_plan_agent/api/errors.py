"""FastAPI global exception handlers — layer 3 of the error boundary (P3-05).

Per handbook 9.9: typed application errors map to stable response bodies.
Unexpected exceptions are:
  1. Logged once with correlation ID and stack trace on the server.
  2. Returned as a generic error code without provider secrets or recipe text.
  3. Incremented in an error metric.

P3-05 unified error contract: every managed endpoint returns the same
``ErrorEnvelope(status, error_code, message, correlation_id, details,
retryable)`` for protocol/HTTP-level failures. Legal business outcomes
(READY / NEEDS_CONFIRMATION / INFEASIBLE / FAILED) keep their own response
models and are never disguised as protocol errors.

Handbook 8.10 three-level error boundary:
  Layer 1: domain/application services raise WorkflowException.
  Layer 2: LangGraph workflow nodes catch them and write a typed
    WorkflowError into the state, routing to an error terminal node.
  Layer 3 (this file): FastAPI exception handler catches unhandled exceptions
    and returns a stable ErrorEnvelope.

Do NOT place one broad try/except Exception around every route.
"""

# 模块概览（中文）：FastAPI 全局异常处理器——错误边界的第 3 层（P3-05）。
# 目标：把已知应用错误映射为稳定响应体；未预期异常则：
#   1. 服务端记录一次（含关联 ID 与堆栈）；
#   2. 返回通用错误码（不泄漏供应商密钥或菜谱文本）；
#   3. 错误指标计数 +1。
# 三层错误边界（Handbook 8.10）：
#   第 1 层：领域/应用服务抛 WorkflowException；
#   第 2 层：LangGraph 工作流节点捕获并写入类型化 WorkflowError，路由到错误终态节点；
#   第 3 层（本文件）：FastAPI 异常处理器兜底，返回稳定 ErrorEnvelope。

import logging
import uuid

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from cooking_plan_agent.domain.errors import (
    DomainErrorCode,
    WorkflowException,
    is_retryable,
    public_message_for,
)
from cooking_plan_agent.domain.models import ErrorEnvelope

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Envelope builder — constructs an ErrorEnvelope body
# ---------------------------------------------------------------------------
# details only carries field-level safe diagnostics: validation loc/type,
# or a machine-readable retry hint. Raw input, stack traces, and provider
# payloads are never included (P3-05 details safety rule).
# details 只承载字段级安全诊断（校验位置/类型或机器可读的重试提示），
# 绝不包含原始输入、堆栈或供应商载荷（P3-05 details 安全规则）。


def _correlation_id(request: Request) -> str:
    """Read the correlation ID stored on the request by the DI dependency.

    Auth failures can be raised by the router-level dependency BEFORE the
    ``extract_correlation_id`` endpoint dependency runs, leaving
    ``request.state.correlation_id`` unset. In that case we derive a stable
    ID from the caller's X-Request-ID header (when safe) or generate one,
    and store it on the request so the response middleware echoes the same
    value in the header (P3-05: header and body correlation ID must match).
    """
    # 读取 DI 依赖存入 request 的关联 ID。
    # 鉴权失败可能发生在 extract_correlation_id 之前，导致 state 里没有值；
    # 此时从 X-Request-ID 头派生稳定 ID（安全时）或生成一个，并写回 state，
    # 保证响应头与响应体的关联 ID 一致（P3-05）。
    existing = getattr(request.state, "correlation_id", None)
    if existing:
        return str(existing)
    supplied = request.headers.get("X-Request-ID")
    cid = supplied if supplied and len(supplied) <= 128 else uuid.uuid4().hex  # 超长则弃用并生成
    request.state.correlation_id = cid
    return cid


# Compat v1 prefix (Spring Boot contract, P0-02). The compat router keeps its
# limited error mapping ({"detail": {"code": ...}}) so the Java caller is
# unaffected by the P3-05 envelope; native/v2 endpoints get the full envelope.
# compat v1 前缀（Spring Boot 契约，P0-02）：compat 路由保留受限错误映射，
# 使 Java 调用方不受 P3-05 envelope 影响；native/v2 端点使用完整 envelope。
_COMPAT_PATH_PREFIX = "/internal/v1/cooking-plans"


def _is_compat_path(request: Request) -> bool:
    """Return True when the request targets the Spring Boot v1 compat router."""
    # 判断请求是否命中 Spring Boot v1 compat 路由
    return request.url.path.startswith(_COMPAT_PATH_PREFIX)


def _build_envelope(
    *,
    http_status: int,
    error_code: str,
    message: str,
    correlation_id: str,
    details: dict[str, object] | list[dict[str, object]] | None = None,
) -> ErrorEnvelope:
    """Build a stable ErrorEnvelope with catalog-driven retryable."""
    # 构建稳定 ErrorEnvelope，retryable 由错误目录（is_retryable）驱动
    return ErrorEnvelope(
        status=http_status,
        error_code=error_code,
        message=message,
        correlation_id=correlation_id,
        details=details,
        retryable=is_retryable(error_code),
    )


def _http_code_from_exception(exc: StarletteHTTPException) -> str | None:
    """Extract a stable error code from an HTTPException's detail.

    HTTPException.detail is either a dict (e.g. {"code": "..."} from the
    auth/backpressure dependencies) or a plain string (e.g. Starlette's
    default "Not Found"). Only the explicit code form is trusted; string
    details map to a default code derived from the status below.
    """
    # 从 HTTPException.detail 提取稳定错误码：
    #   - dict 形式（鉴权/背压依赖设置 {"code": ...}）→ 取 code
    #   - 纯字符串（如 Starlette 默认 "Not Found"）→ 忽略，走下方按状态映射默认码
    if isinstance(exc.detail, dict):
        code = exc.detail.get("code")
        if isinstance(code, str) and code:
            return code
    return None


_DEFAULT_CODES: dict[int, str] = {
    status.HTTP_401_UNAUTHORIZED: "UNAUTHORIZED",
    status.HTTP_403_FORBIDDEN: "FORBIDDEN",
    status.HTTP_404_NOT_FOUND: "NOT_FOUND",
    status.HTTP_409_CONFLICT: "CONFLICT",
    status.HTTP_422_UNPROCESSABLE_CONTENT: "REQUEST_VALIDATION_ERROR",
    status.HTTP_429_TOO_MANY_REQUESTS: "RATE_LIMITED",
    status.HTTP_503_SERVICE_UNAVAILABLE: "SERVICE_UNAVAILABLE",
}


def _default_code_for_status(http_status: int) -> str:
    """Map an HTTP status to a stable default error code."""
    # 按 HTTP 状态映射稳定默认错误码；未知状态回落到 INTERNAL_ERROR
    return _DEFAULT_CODES.get(http_status, DomainErrorCode.INTERNAL_ERROR.value)


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


async def workflow_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle known domain exceptions with their mapped error code.

    Registered only for WorkflowException — the assert narrows the type for
    static analysis and always holds at runtime.

    Domain errors are business failures (infeasible schedule, safety violation,
    etc.) — return HTTP 200 with FAILED status per handbook 9.8 policy.
    This is a legal business outcome, NOT a protocol error, so it keeps the
    FailedPlanResponse shape (P3-05: business results are never disguised as
    protocol exceptions).
    """
    # 已知领域异常：业务失败（不可行排程、安全违规等）→ 按 handbook 9.8 返回 HTTP 200 + FAILED。
    # 这是合法业务结果而非协议错误，故保留 FailedPlanResponse 形状。
    assert isinstance(exc, WorkflowException)  # 断言收窄类型（运行时必成立，仅供静态分析）
    correlation_id = _correlation_id(request)
    logger.warning(
        "WorkflowException caught | code=%s | correlation_id=%s | exception_type=%s",
        exc.code.value,
        correlation_id,
        type(exc).__name__,
    )
    return JSONResponse(
        status_code=200,
        content={
            "status": "FAILED",
            "error_code": exc.code.value,
            "correlation_id": correlation_id,
            # P2-03: client-facing text always resolves through the public
            # message catalog — never echoes the exception message.
            # 面向客户端的文本始终通过公共消息目录解析——绝不回显异常消息本身。
            "message": public_message_for(exc.code.value),
        },
    )


async def request_validation_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle Pydantic validation failures with a unified envelope (422).

    details carries only field-level loc/type diagnostics — never the raw
    input values that caused the error (P3-05 details safety rule).
    """
    # Pydantic 校验失败 → 统一 envelope（422）；details 只含 loc/type，不含原始输入值。
    assert isinstance(exc, RequestValidationError)
    correlation_id = _correlation_id(request)

    # Spring Boot v1 compat: keep the limited mapping so the Java caller is
    # unaffected by the P3-05 envelope (P3-05 step 6).
    # compat 路径保留受限映射，避免影响 Java 调用方
    if _is_compat_path(request):
        return JSONResponse(
            status_code=422,
            content={"detail": {"code": DomainErrorCode.REQUEST_VALIDATION_ERROR.value}},
        )

    logger.warning(
        "Request validation failed | correlation_id=%s | errors=%d",
        correlation_id,
        len(exc.errors()),
    )

    details: list[dict[str, object]] = []
    for err in exc.errors():
        details.append(
            {
                "loc": list(err.get("loc", ())),  # 出错字段位置
                "type": err.get("type", ""),  # 错误类型
                "msg": err.get("msg", ""),  # 错误消息
            }
        )

    envelope = _build_envelope(
        http_status=status.HTTP_422_UNPROCESSABLE_CONTENT,
        error_code=DomainErrorCode.REQUEST_VALIDATION_ERROR.value,
        message="Request validation failed.",
        correlation_id=correlation_id,
        details=details,
    )
    return JSONResponse(status_code=422, content=envelope.model_dump())


async def http_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle HTTPException (auth, backpressure, not-found, etc.) uniformly.

    The error_code comes from the exception's explicit ``code`` detail when
    present (auth/backpressure dependencies set it); otherwise a stable code
    is derived from the HTTP status. ``retryable`` always comes from the
    error catalog — never from the message text (P3-05 D9).

    Accepts both FastAPI's HTTPException and Starlette's (404/405 raised by
    the router layer); FastAPI's HTTPException subclasses Starlette's.
    """
    # 统一处理 HTTPException（鉴权/背压/404 等）。
    # error_code 优先取异常显式 code，否则按状态映射；retryable 始终来自错误目录。
    assert isinstance(exc, StarletteHTTPException)
    correlation_id = _correlation_id(request)
    error_code = _http_code_from_exception(exc) or _default_code_for_status(exc.status_code)

    # Spring Boot v1 compat: keep the limited {"detail": {"code": ...}} mapping
    # so the Java caller is unaffected by the P3-05 envelope (P3-05 step 6).
    # compat 路径保留受限 {"detail": {"code": ...}} 映射
    if _is_compat_path(request):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": {"code": error_code}},
            headers=exc.headers,
        )

    envelope = _build_envelope(
        http_status=exc.status_code,
        error_code=error_code,
        message=_http_message(exc),
        correlation_id=correlation_id,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=envelope.model_dump(),
        headers=exc.headers,
    )


def _http_message(exc: StarletteHTTPException) -> str:
    """Stable, non-sensitive message for an HTTPException.

    A dict detail (auth/backpressure) may carry a ``message`` key; otherwise
    use a generic message keyed by status. Never echo raw dict values that
    could contain secrets.
    """
    # 生成稳定且不含敏感信息的消息；dict detail 可带 message 键，否则按状态给通用文案。
    if isinstance(exc.detail, dict):
        message = exc.detail.get("message")
        if isinstance(message, str) and message:
            return message
    if exc.status_code == status.HTTP_404_NOT_FOUND:
        return "Resource not found."
    if exc.status_code == status.HTTP_401_UNAUTHORIZED:
        return "Authentication failed."
    if exc.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
        return "Service temporarily unavailable. Retry after backoff."
    return "Request could not be processed."


async def generic_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Last-resort handler for unexpected exceptions.

    Returns HTTP 500 with INTERNAL_ERROR code. Logs the full stack trace
    server-side but returns a sanitised envelope to the client. The inner
    message is intentionally vague — provider secrets, recipe text, and
    stack traces must never leak to the response.
    """
    # 兜底处理未预期异常：返回 500 + INTERNAL_ERROR。
    # 服务端记录完整堆栈，但客户端只收到脱敏 envelope（不泄漏密钥/菜谱文本/堆栈）。
    correlation_id = _correlation_id(request)
    logger.exception(
        "Unexpected exception | correlation_id=%s | type=%s",
        correlation_id,
        type(exc).__name__,
    )
    envelope = _build_envelope(
        http_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code=DomainErrorCode.INTERNAL_ERROR.value,
        message="An unexpected internal error occurred.",
        correlation_id=correlation_id,
    )
    return JSONResponse(status_code=500, content=envelope.model_dump())


def register_exception_handlers(app: FastAPI) -> None:
    """Register all global exception handlers on the FastAPI application.

    Called from create_app() after the app instance is built.

    Handler precedence (P3-05):
      1. WorkflowException -> 200 + FailedPlanResponse (business result)
      2. RequestValidationError -> 422 + ErrorEnvelope
      3. HTTPException -> status + ErrorEnvelope (401/403/404/409/503/...)
      4. Exception -> 500 + ErrorEnvelope (last resort)
    """
    # 注册全局异常处理器（由 create_app() 调用）。
    # 优先级（P3-05）：
    #   1. WorkflowException → 200 + FailedPlanResponse（业务结果）
    #   2. RequestValidationError → 422 + ErrorEnvelope
    #   3. HTTPException → 状态码 + ErrorEnvelope
    #   4. Exception → 500 + ErrorEnvelope（兜底）
    app.add_exception_handler(WorkflowException, workflow_exception_handler)
    app.add_exception_handler(RequestValidationError, request_validation_handler)
    # StarletteHTTPException 覆盖 FastAPI 的 HTTPException（其子类）以及
    # 路由层抛出的 404/405。
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)
