"""Absolute UTC to process-local monotonic deadline conversion."""

import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from recommendation_agent.domain.errors import AgentError, ErrorCode


class Clock(Protocol):
    def utc_now(self) -> datetime: ...

    def monotonic(self) -> float: ...


class SystemClock:
    def utc_now(self) -> datetime:
        return datetime.now(UTC)

    def monotonic(self) -> float:
        return time.monotonic()


@dataclass(frozen=True, slots=True)
class DeadlineBudget:
    _clock: Clock
    _monotonic_expiry: float

    @classmethod
    def from_absolute(
        cls,
        deadline_at: datetime,
        *,
        clock: Clock,
        minimum_seconds: float,
    ) -> "DeadlineBudget":
        utc_now = clock.utc_now()
        monotonic_now = clock.monotonic()
        if deadline_at.tzinfo is None or deadline_at.utcoffset() != UTC.utcoffset(deadline_at):
            raise AgentError(ErrorCode.INVALID_REQUEST, http_status=400)
        remaining = (deadline_at - utc_now).total_seconds()
        if not math.isfinite(remaining) or remaining <= 0:
            raise AgentError(ErrorCode.DEADLINE_EXPIRED, http_status=408)
        if remaining < minimum_seconds:
            raise AgentError(ErrorCode.DEADLINE_INSUFFICIENT, http_status=408)
        return cls(clock, monotonic_now + remaining)

    def remaining(self) -> float:
        return max(0.0, self._monotonic_expiry - self._clock.monotonic())

    @classmethod
    def from_monotonic_expiry(cls, expiry: float, *, clock: Clock) -> "DeadlineBudget":
        if not math.isfinite(expiry):
            raise AgentError(ErrorCode.INVALID_REQUEST, http_status=400)
        return cls(clock, expiry)

    @property
    def monotonic_expiry(self) -> float:
        return self._monotonic_expiry

    def downstream_timeout(self, *, configured_seconds: float, guard_seconds: float) -> float:
        available = self.remaining() - guard_seconds
        if available <= 0:
            raise AgentError(ErrorCode.DEADLINE_EXHAUSTED, http_status=504)
        return min(configured_seconds, available)

    def ensure_remaining(self, *, guard_seconds: float = 0.0) -> None:
        if self.remaining() <= guard_seconds:
            raise AgentError(ErrorCode.DEADLINE_EXHAUSTED, http_status=504)
