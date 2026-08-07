"""Small bounded request limiter for LLM work."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import HTTPException, status


class RequestLimiter:
    def __init__(self, *, max_active: int, max_queued: int, timeout_seconds: float) -> None:
        self._semaphore = asyncio.Semaphore(max_active)
        self._max_queued = max_queued
        self._timeout_seconds = timeout_seconds
        self._waiting = 0
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def lease(self) -> AsyncIterator[None]:
        async with self._lock:
            if self._semaphore.locked() and self._waiting >= self._max_queued:
                raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail={"code": "SERVICE_OVERLOADED"})
            self._waiting += 1
        try:
            try:
                await asyncio.wait_for(self._semaphore.acquire(), timeout=self._timeout_seconds)
            except TimeoutError as exc:
                raise HTTPException(
                    status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail={"code": "SERVICE_OVERLOADED"},
                ) from exc
        finally:
            async with self._lock:
                self._waiting -= 1
        try:
            yield
        finally:
            self._semaphore.release()
