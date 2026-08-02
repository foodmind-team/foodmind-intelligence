"""P1-03: request-level backpressure tests.

Covers the active/queued two-layer limiter:
  - active never exceeds the cap; queued never exceeds the queue capacity
  - rejections are deterministic (queue-full path, not race-dependent)
  - cancelled waiters and handler failures release their quota
  - HTTP 503 + Retry-After + OVERLOADED code when saturated
  - health/load exposes limiter metrics and bypasses the limiter
"""

import asyncio
from typing import Annotated

import httpx
import pytest
from fastapi import Depends, FastAPI

from cooking_plan_agent.api.backpressure import OVERLOADED_ERROR_CODE, RequestLimiter, request_lease

# ---------------------------------------------------------------------------
# Limiter unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_active_bounded_by_max_active() -> None:
    limiter = RequestLimiter(max_active=2, max_queued=5, queue_timeout_seconds=1.0)

    assert await limiter.acquire() is True
    assert await limiter.acquire() is True
    assert limiter.snapshot().active == 2

    # Third request queues instead of running.
    waiter = asyncio.create_task(limiter.acquire())
    await asyncio.sleep(0.02)
    assert limiter.snapshot().queued == 1
    assert limiter.snapshot().active == 2

    await limiter.release()
    assert await waiter is True
    assert limiter.snapshot().active == 2  # waiter promoted into the active slot

    await limiter.release()
    await limiter.release()


@pytest.mark.asyncio
async def test_rejections_match_configured_capacity() -> None:
    """5 active + 10 queued slots for 30 requests → exactly 15 rejected."""
    limiter = RequestLimiter(max_active=5, max_queued=10, queue_timeout_seconds=5.0)
    results: list[bool] = []

    async def worker() -> None:
        ok = await limiter.acquire()
        try:
            if ok:
                await asyncio.sleep(0.01)  # hold briefly to force queuing
            results.append(ok)
        finally:
            if ok:
                await limiter.release()

    await asyncio.gather(*(worker() for _ in range(30)))

    assert sum(results) == 15, f"Expected 15 granted leases, got {sum(results)}"
    assert limiter.snapshot().rejected_total == 15
    assert limiter.snapshot().active == 0  # all leases released


@pytest.mark.asyncio
async def test_queued_wait_timeout_rejects() -> None:
    """A waiter that exceeds queue_timeout_seconds is rejected, not held forever."""
    limiter = RequestLimiter(max_active=1, max_queued=5, queue_timeout_seconds=0.05)

    assert await limiter.acquire() is True  # saturate the only active slot

    assert await limiter.acquire() is False  # queued → times out → rejected
    assert limiter.snapshot().rejected_total == 1
    assert limiter.snapshot().queued == 0

    await limiter.release()


@pytest.mark.asyncio
async def test_cancelled_queued_waiter_releases_slot() -> None:
    """A cancelled queued waiter must release its queue slot (P1-03)."""
    limiter = RequestLimiter(max_active=1, max_queued=3, queue_timeout_seconds=5.0)

    assert await limiter.acquire() is True
    waiter = asyncio.create_task(limiter.acquire())
    await asyncio.sleep(0.02)
    assert limiter.snapshot().queued == 1

    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    assert limiter.snapshot().queued == 0, "Cancelled waiter leaked a queue slot"

    # The slot is reusable: a fresh waiter can now queue.
    waiter2 = asyncio.create_task(limiter.acquire())
    await asyncio.sleep(0.02)
    assert limiter.snapshot().queued == 1
    waiter2.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter2

    await limiter.release()


# ---------------------------------------------------------------------------
# HTTP wiring tests (single event loop via httpx ASGITransport)
# ---------------------------------------------------------------------------


def _overload_app(max_active: int = 1, max_queued: int = 0) -> FastAPI:
    """Tiny app whose /x endpoint is guarded by request_lease."""
    app = FastAPI()
    app.state.request_limiter = RequestLimiter(
        max_active=max_active,
        max_queued=max_queued,
        queue_timeout_seconds=0.2,
    )

    @app.post("/x")
    async def guarded(_lease: Annotated[None, Depends(request_lease)] = None) -> dict[str, bool]:
        return {"ok": True}

    @app.get("/load")
    async def load() -> dict[str, object]:
        return app.state.request_limiter.snapshot().__dict__

    return app


@pytest.mark.asyncio
async def test_saturated_limiter_returns_503_with_retry_after() -> None:
    app = _overload_app(max_active=1, max_queued=0)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        # Saturate the only lease from the same event loop.
        assert await app.state.request_limiter.acquire() is True

        response = await client.post("/x")
        assert response.status_code == 503
        assert response.headers.get("Retry-After") is not None
        detail = response.json()["detail"]
        assert detail["code"] == OVERLOADED_ERROR_CODE

        await app.state.request_limiter.release()


@pytest.mark.asyncio
async def test_health_load_bypasses_limiter() -> None:
    """The metrics route is NOT guarded by request_lease — it stays probeable."""
    app = _overload_app(max_active=1, max_queued=0)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        # Saturate: the guarded /x would 503, but /load must still answer.
        assert await app.state.request_limiter.acquire() is True

        load_response = await client.get("/load")
        assert load_response.status_code == 200
        body = load_response.json()
        assert body["active"] == 1
        assert body["queued"] == 0

        await app.state.request_limiter.release()


@pytest.mark.asyncio
async def test_handler_exception_releases_lease() -> None:
    """A handler that raises must still release its lease in the dependency finally."""
    app = FastAPI()
    app.state.request_limiter = RequestLimiter(max_active=1, max_queued=1, queue_timeout_seconds=0.2)

    @app.post("/boom")
    async def boom(_lease: Annotated[None, Depends(request_lease)] = None) -> dict[str, bool]:
        raise RuntimeError("handler exploded")

    @app.get("/load")
    async def load() -> dict[str, object]:
        return app.state.request_limiter.snapshot().__dict__

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        response = await client.post("/boom")
        assert response.status_code == 500  # handler error surfaced, not swallowed

        # The lease was released despite the handler failure.
        body = (await client.get("/load")).json()
        assert body["active"] == 0
        assert body["queued"] == 0
