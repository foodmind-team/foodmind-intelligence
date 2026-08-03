"""Cancellation-safe bounded request admission."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from recommendation_agent.domain.errors import AgentError, ErrorCode


@dataclass(frozen=True, slots=True)
class LimiterSnapshot:
    active: int
    queued: int
    rejected_total: int
    accepting: bool


class RequestLimiter:
    def __init__(self, *, max_active: int, max_queued: int, timeout_seconds: float) -> None:
        self._semaphore = asyncio.Semaphore(max_active)
        self._max_queued = max_queued
        self._timeout_seconds = timeout_seconds
        self._lock = asyncio.Lock()
        self._active = 0
        self._queued = 0
        self._rejected_total = 0
        self._accepting = True
        self._drained = asyncio.Event()
        self._drained.set()

    def snapshot(self) -> LimiterSnapshot:
        return LimiterSnapshot(self._active, self._queued, self._rejected_total, self._accepting)

    @asynccontextmanager
    async def lease(self) -> AsyncIterator[None]:
        counted_as_queued = False
        acquired = False
        async with self._lock:
            if not self._accepting:
                self._rejected_total += 1
                raise AgentError(ErrorCode.SERVICE_OVERLOADED, http_status=503, retryable=True)
            if self._semaphore.locked():
                if self._queued >= self._max_queued:
                    self._rejected_total += 1
                    raise AgentError(ErrorCode.SERVICE_OVERLOADED, http_status=503, retryable=True)
                self._queued += 1
                counted_as_queued = True
        try:
            try:
                await asyncio.wait_for(self._semaphore.acquire(), timeout=self._timeout_seconds)
                acquired = True
            except TimeoutError as exc:
                async with self._lock:
                    self._rejected_total += 1
                raise AgentError(ErrorCode.SERVICE_OVERLOADED, http_status=503, retryable=True) from exc
            async with self._lock:
                if counted_as_queued:
                    self._queued -= 1
                    counted_as_queued = False
                self._active += 1
                self._drained.clear()
            yield
        finally:
            async with self._lock:
                if counted_as_queued:
                    self._queued -= 1
                if acquired:
                    self._active -= 1
                    if self._active == 0:
                        self._drained.set()
            if acquired:
                self._semaphore.release()

    async def close(self) -> None:
        async with self._lock:
            self._accepting = False

    async def wait_for_drain(self, timeout_seconds: float) -> bool:
        try:
            await asyncio.wait_for(self._drained.wait(), timeout=timeout_seconds)
        except TimeoutError:
            return False
        return True
