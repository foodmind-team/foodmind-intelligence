"""Request-level backpressure — active/queued two-layer limiter (P1-03).

Protects the LLM calls and the CP-SAT solver from burst traffic by bounding
concurrency at the API edge:
  - at most ``max_active_requests`` requests execute at once
  - excess requests QUEUE for at most ``queue_timeout_seconds``
  - when the queue is full or a waiter times out, the request is rejected
    with HTTP 503 + a stable ``OVERLOADED`` code and ``Retry-After``

Design notes:
  - Never a bare Semaphore with unbounded waiting: queued waiters are
    bounded by capacity and by timeout.
  - Cancellation-safe: a cancelled queued waiter releases its slot.
  - Health endpoints bypass the limiter (they are separate routes), so the
    process stays probeable while overloaded.
  - The limiter is process-level. Horizontal scaling limits are a separate
    concern (P3-02).
"""

# 模块概览（中文）：请求级背压 =「活动并发 + 排队」两层限流。
# 目标：限制同时执行的请求数（max_active），多余的排队等待（最多 queue_timeout_seconds），
#       队列满或等待超时则拒绝并返回 503 + OVERLOADED + Retry-After，保护下游 LLM/求解器。

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status


@dataclass(frozen=True)  # frozen=True：快照一旦创建即不可变，可作为纯数据载体传递
class LimiterSnapshot:
    """Point-in-time load view for /health/load and logs.

    Attributes:
        active: Requests currently holding a lease.
        queued: Requests currently waiting for a lease.
        rejected_total: Cumulative requests rejected since startup.
        queue_wait_ms: Most recent successful queue wait in milliseconds.
    """

    # 某一时刻的负载快照，供 /health/load 与日志使用（不是实时可变状态）
    active: int  # 当前持有租约（正在执行）的请求数
    queued: int  # 当前正在排队等待租约的请求数
    rejected_total: int  # 自启动以来累计被拒绝的请求数（单调递增）
    queue_wait_ms: float  # 最近一次成功排队所等待的毫秒数


class RequestLimiter:
    """Active/queued two-layer concurrency limiter.

    Usage (FastAPI dependency)::

        async def request_lease(limiter=Depends(get_limiter)):
            acquired = await limiter.acquire()
            if not acquired:
                raise HTTPException(503, ...)
            try:
                yield
            finally:
                await limiter.release()
    """

    def __init__(
        self,
        max_active: int,
        max_queued: int,
        queue_timeout_seconds: float,
    ) -> None:
        if max_active < 1:
            raise ValueError("max_active must be >= 1")  # 至少允许 1 个并发
        if max_queued < 0:
            raise ValueError("max_queued must be >= 0")  # 队列可为 0（即不排队，满则直接拒绝）

        self._max_active = max_active  # 最大活动并发数（硬上限）
        self._max_queued = max_queued  # 最大排队长度
        # Public so callers can derive Retry-After without reaching into privates.
        # 公开此字段，便于调用方（request_lease）无需访问私有成员即可推导 Retry-After
        self.queue_timeout_seconds = queue_timeout_seconds

        self._active = 0  # 当前持有租约的请求数
        self._queued = 0  # 当前正在排队的请求数
        self._rejected_total = 0  # 累计被拒绝次数（启动后单调递增）
        self._last_queue_wait_ms = 0.0  # 最近一次成功排队的等待时长（毫秒）
        self._condition = asyncio.Condition()  # 并发原语：用锁保护计数器 + 用通知唤醒排队者

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def acquire(self) -> bool:
        """Try to acquire a lease.

        Returns True when a lease was granted (the caller MUST release it).
        Returns False when rejected (queue full or wait timed out).
        """
        start = time.monotonic()  # 记录开始时刻，用于计算排队等待时长
        async with self._condition:  # 拿锁，保护 _active / _queued 计数器的读写
            # ① 快路径：有活动名额，直接占用并返回
            if self._active < self._max_active:
                self._active += 1
                return True

            # ② 队列满：连排队位置都没有了，直接拒绝（快速失败，避免无限堆积）
            if self._queued >= self._max_queued:
                self._rejected_total += 1
                return False

            # ③ 排队等待：先登记占一个排队名额，再循环等待通知
            self._queued += 1
            try:
                # 循环必须重新检查：wait 被唤醒不代表一定有空位（伪唤醒/竞争），须在锁内确认
                while self._active >= self._max_active:
                    try:
                        # Bounded wait — never an infinite bare semaphore.
                        # 有界等待：最多等 queue_timeout_seconds，绝不做无界裸 Semaphore
                        await asyncio.wait_for(self._condition.wait(), self.queue_timeout_seconds)
                    except TimeoutError:
                        # 等待超时：拒绝并返回（finally 会归还排队名额）
                        self._rejected_total += 1
                        return False
                # 成功拿到空位：占用活动名额，并记录本次排队耗时
                self._active += 1
                self._last_queue_wait_ms = (time.monotonic() - start) * 1000
                return True
            finally:
                # 无论成功、超时还是异常退出，都归还排队名额，避免名额泄漏
                self._queued -= 1

    async def release(self) -> None:
        """Release a lease and wake one queued waiter, if any."""
        async with self._condition:
            self._active = max(0, self._active - 1)  # 归还活动名额（max 防御下溢）
            if self._queued > 0:
                # 每次 release 只释放一个名额，故只唤醒 1 个排队者，避免惊群
                self._condition.notify(1)

    def snapshot(self) -> LimiterSnapshot:
        """Return a point-in-time view of load and rejection counters."""
        # asyncio 单线程事件循环下读计数器无需加锁，读到的是一致状态
        return LimiterSnapshot(
            active=self._active,
            queued=self._queued,
            rejected_total=self._rejected_total,
            queue_wait_ms=self._last_queue_wait_ms,
        )


# ---------------------------------------------------------------------------
# FastAPI dependency wiring
# ---------------------------------------------------------------------------

# Stable overload error code and Retry-After units (seconds).
# 稳定的过载错误码；Retry-After 单位为秒
OVERLOADED_ERROR_CODE = "OVERLOADED"


def get_request_limiter(request: Request) -> RequestLimiter:
    """Retrieve the process-level limiter from the app lifespan state.

    The limiter is created during startup from Settings; health routes never
    depend on it, so they stay reachable while the process is overloaded.
    """
    # 从 app.state 取进程级单例；该 limiter 在应用启动（lifespan）时由 Settings 创建
    limiter = getattr(request.app.state, "request_limiter", None)
    if not isinstance(limiter, RequestLimiter):
        # 未初始化则快速失败，而非静默返回一个错误的 limiter
        raise RuntimeError("request_limiter was not initialised during startup")
    return limiter


async def request_lease(
    limiter: Annotated[RequestLimiter, Depends(get_request_limiter)],
) -> AsyncIterator[None]:
    """Acquire a request lease for the duration of the handler (P1-03).

    When the queue is full or the wait times out, the request is rejected
    with HTTP 503, a stable ``OVERLOADED`` code and a ``Retry-After`` header.
    The lease is always released — even when the handler raises or the
    client disconnects (cancellation).
    """
    # 前置阶段：尝试获取租约
    acquired = await limiter.acquire()
    if not acquired:
        # 拒绝：队列满或等待超时，返回 503 + 稳定错误码 + Retry-After（至少 1 秒）
        retry_after = max(1, int(limiter.queue_timeout_seconds))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": OVERLOADED_ERROR_CODE,
                "message": "Service is overloaded. Retry after backoff.",
                "retry_after_seconds": retry_after,
            },
            headers={"Retry-After": str(retry_after)},
        )
    try:
        yield  # 主体阶段：请求处理函数在此执行
    finally:
        # 后置阶段：无论 handler 成功、抛异常还是客户端断开（取消），都释放租约
        await limiter.release()
