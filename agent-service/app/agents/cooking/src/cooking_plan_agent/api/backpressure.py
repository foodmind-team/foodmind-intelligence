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

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status


@dataclass(frozen=True)
class LimiterSnapshot:
    """Point-in-time load view for /health/load and logs.

    Attributes:
        active: Requests currently holding a lease.
        queued: Requests currently waiting for a lease.
        rejected_total: Cumulative requests rejected since startup.
        queue_wait_ms: Most recent successful queue wait in milliseconds.
    """

    active: int
    queued: int
    rejected_total: int
    queue_wait_ms: float


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
            raise ValueError("max_active must be >= 1")
        if max_queued < 0:
            raise ValueError("max_queued must be >= 0")

        self._max_active = max_active
        self._max_queued = max_queued
        # Public so callers can derive Retry-After without reaching into privates.
        self.queue_timeout_seconds = queue_timeout_seconds

        self._active = 0
        self._queued = 0
        self._rejected_total = 0
        self._last_queue_wait_ms = 0.0
        self._condition = asyncio.Condition()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def acquire(self) -> bool:
        """Try to acquire a lease.

        Returns True when a lease was granted (the caller MUST release it).
        Returns False when rejected (queue full or wait timed out).
        """
        start = time.monotonic()
        async with self._condition:
            if self._active < self._max_active:
                self._active += 1
                return True

            if self._queued >= self._max_queued:
                self._rejected_total += 1
                return False

            self._queued += 1
            try:
                while self._active >= self._max_active:
                    try:
                        # Bounded wait — never an infinite bare semaphore.
                        await asyncio.wait_for(self._condition.wait(), self.queue_timeout_seconds)
                    except TimeoutError:
                        self._rejected_total += 1
                        return False
                self._active += 1
                self._last_queue_wait_ms = (time.monotonic() - start) * 1000
                return True
            finally:
                self._queued -= 1

    async def release(self) -> None:
        """Release a lease and wake one queued waiter, if any."""
        async with self._condition:
            self._active = max(0, self._active - 1)
            if self._queued > 0:
                self._condition.notify(1)

    def snapshot(self) -> LimiterSnapshot:
        """Return a point-in-time view of load and rejection counters."""
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
OVERLOADED_ERROR_CODE = "OVERLOADED"


def get_request_limiter(request: Request) -> RequestLimiter:
    """Retrieve the process-level limiter from the app lifespan state.

    The limiter is created during startup from Settings; health routes never
    depend on it, so they stay reachable while the process is overloaded.
    """
    limiter = getattr(request.app.state, "request_limiter", None)
    if not isinstance(limiter, RequestLimiter):
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
    acquired = await limiter.acquire()
    if not acquired:
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
        yield
    finally:
        await limiter.release()
