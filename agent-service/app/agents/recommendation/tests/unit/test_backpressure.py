import asyncio

import pytest

from recommendation_agent.api.backpressure import RequestLimiter
from recommendation_agent.domain.errors import AgentError, ErrorCode


@pytest.mark.asyncio
async def test_queue_saturation_releases_all_capacity() -> None:
    limiter = RequestLimiter(max_active=1, max_queued=0, timeout_seconds=0.05)
    async with limiter.lease():
        with pytest.raises(AgentError) as captured:
            async with limiter.lease():
                raise AssertionError("unreachable")
        assert captured.value.code is ErrorCode.SERVICE_OVERLOADED
    assert limiter.snapshot().active == 0
    assert limiter.snapshot().queued == 0


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_leak_queue_slot() -> None:
    limiter = RequestLimiter(max_active=1, max_queued=1, timeout_seconds=1.0)
    async with limiter.lease():
        task = asyncio.create_task(_take_lease(limiter))
        await asyncio.sleep(0)
        assert limiter.snapshot().queued == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert limiter.snapshot().active == 0
    assert limiter.snapshot().queued == 0


async def _take_lease(limiter: RequestLimiter) -> None:
    async with limiter.lease():
        return


@pytest.mark.asyncio
async def test_shutdown_stops_admission_and_drain_is_bounded() -> None:
    limiter = RequestLimiter(max_active=1, max_queued=1, timeout_seconds=0.1)
    await limiter.close()
    with pytest.raises(AgentError) as captured:
        async with limiter.lease():
            raise AssertionError("unreachable")
    assert captured.value.code is ErrorCode.SERVICE_OVERLOADED
    assert limiter.snapshot().accepting is False
    assert await limiter.wait_for_drain(0.01) is True
