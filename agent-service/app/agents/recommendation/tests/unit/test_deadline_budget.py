from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from recommendation_agent.domain.errors import AgentError, ErrorCode
from recommendation_agent.time.budget import DeadlineBudget


@dataclass
class FakeClock:
    now: datetime
    tick: float
    utc_samples: int = 0
    monotonic_samples: int = 0

    def utc_now(self) -> datetime:
        self.utc_samples += 1
        return self.now

    def monotonic(self) -> float:
        self.monotonic_samples += 1
        return self.tick


def test_absolute_deadline_samples_both_clocks_once_and_never_extends() -> None:
    clock = FakeClock(datetime(2030, 1, 1, tzinfo=UTC), 10.0)
    budget = DeadlineBudget.from_absolute(clock.now + timedelta(seconds=2), clock=clock, minimum_seconds=0.1)
    assert (clock.utc_samples, clock.monotonic_samples) == (1, 1)
    assert budget.remaining() == 2.0
    clock.tick = 11.4
    assert budget.remaining() == pytest.approx(0.6)
    assert budget.downstream_timeout(configured_seconds=0.7, guard_seconds=0.05) == pytest.approx(0.55)
    clock.tick = 20.0
    assert budget.remaining() == 0.0


@pytest.mark.parametrize(
    ("offset_ms", "expected"),
    [(-1, ErrorCode.DEADLINE_EXPIRED), (0, ErrorCode.DEADLINE_EXPIRED), (99, ErrorCode.DEADLINE_INSUFFICIENT)],
)
def test_expired_and_near_deadlines_fail_before_work(offset_ms: int, expected: ErrorCode) -> None:
    clock = FakeClock(datetime(2030, 1, 1, tzinfo=UTC), 10.0)
    with pytest.raises(AgentError) as captured:
        DeadlineBudget.from_absolute(
            clock.now + timedelta(milliseconds=offset_ms),
            clock=clock,
            minimum_seconds=0.1,
        )
    assert captured.value.code is expected


def test_guard_is_retained() -> None:
    clock = FakeClock(datetime(2030, 1, 1, tzinfo=UTC), 10.0)
    budget = DeadlineBudget.from_absolute(clock.now + timedelta(milliseconds=100), clock=clock, minimum_seconds=0.1)
    clock.tick += 0.06
    with pytest.raises(AgentError) as captured:
        budget.downstream_timeout(configured_seconds=1.0, guard_seconds=0.05)
    assert captured.value.code is ErrorCode.DEADLINE_EXHAUSTED
