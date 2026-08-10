"""P4-04 task SSE — service-level subscription registry and event_id semantics.

Covers the repository's atomic event_id increments, the subscribe generator
(snapshot replay / Last-Event-ID filtering / terminal-close / keepalive
ticks), and notification fan-out isolation between tasks. HTTP-level
behaviour lives in tests/contract/test_task_sse.py.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

import aiosqlite
import pytest
import pytest_asyncio

from cooking_plan_agent.domain.models import (
    GeneratePlanRequest,
    InventoryLotSnapshot,
    KitchenResourceSnapshot,
    ReadyPlanResponse,
)
from cooking_plan_agent.tasks.models import TaskRecord, TaskStatus, new_task_id
from cooking_plan_agent.tasks.repository import SQLiteTaskRepository
from cooking_plan_agent.tasks.service import AsyncTaskService


def _valid_request(request_id: str = "req-sse-001") -> GeneratePlanRequest:
    return GeneratePlanRequest(
        request_id=request_id,
        user_id="sse-user",
        recipes=(
            {
                "recipe_id": "r1",
                "text": "Cook chicken for 10 minutes. Serves 2.",
                "target_servings": 2,
            },
        ),
        dietary_restrictions=(),
        user_allergens=(),
        inventory_lots=(
            InventoryLotSnapshot(
                lot_id="lot-001",
                item_id="item-001",
                canonical_name="chicken breast",
                on_hand=Decimal(300),
                reserved=Decimal(0),
                unit="g",
            ),
        ),
        kitchen_resources=(
            KitchenResourceSnapshot(
                resource_id="stove-1",
                resource_type="stove",
                capacity=Decimal(4),
                capacity_unit="burners",
            ),
        ),
    )


def _record(request_id: str = "req-r", status: TaskStatus = TaskStatus.QUEUED) -> TaskRecord:
    return TaskRecord(
        task_id=new_task_id(),
        request_id=request_id,
        user_id="u",
        request_payload={"request_id": request_id},
        thread_id=f"{request_id}:0",
        status=status,
    )


class _FakeGeneration:
    """Stands in for GenerateCookingPlanService — instant READY plan."""

    async def execute(
        self,
        request: GeneratePlanRequest,
        thread_id: str | None = None,
        progress_callback=None,
    ):
        return ReadyPlanResponse(
            plan_id="plan-1",
            solver_status="OPTIMAL",
            makespan_minutes=30,
            timeline=(),
            completion_checklist=(),
            mise_en_place=(),
            dish_completions=(),
        )


@pytest_asyncio.fixture
async def repo(tmp_path):
    r = SQLiteTaskRepository(str(tmp_path / "sse.sqlite"))
    await r.astart()
    yield r
    await r.close()


@pytest_asyncio.fixture
async def service(tmp_path):
    repo = SQLiteTaskRepository(str(tmp_path / "svc-sse.sqlite"))
    await repo.astart()
    svc = AsyncTaskService(repository=repo, generation_service=_FakeGeneration(), worker_concurrency=1)
    yield svc, repo
    if svc._worker_task:  # type: ignore[attr-defined]
        svc._worker_task.cancel()  # type: ignore[attr-defined]
    await repo.close()


async def _collect(
    svc: AsyncTaskService,
    task_id: str,
    last_event_id: int = -1,
    *,
    keepalive_seconds: float | None = None,
    timeout: float = 5.0,
) -> list[TaskRecord | None]:
    """Drain a subscription until it closes; return every yielded frame."""

    async def consume() -> list[TaskRecord | None]:
        frames: list[TaskRecord | None] = []
        async for frame in svc.subscribe(task_id, last_event_id, keepalive_seconds=keepalive_seconds):
            frames.append(frame)
        return frames

    return await asyncio.wait_for(consume(), timeout=timeout)


# ---------------------------------------------------------------------------
# Repository — atomic event_id increments
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_event_id_increments_atomically_with_status_changes(repo) -> None:
    record = _record("req-eid")
    await repo.create(record)
    assert (await repo.get(record.task_id)).event_id == 0

    running = record.transition(TaskStatus.RUNNING)
    claimed = await repo.update(running, expected_status=TaskStatus.QUEUED)
    assert claimed is not None and claimed.event_id == 1

    ready = claimed.transition(TaskStatus.READY)
    done = await repo.update(ready, expected_status=TaskStatus.RUNNING)
    assert done is not None and done.event_id == 2


@pytest.mark.asyncio
async def test_failed_conditional_write_does_not_bump_event_id(repo) -> None:
    record = _record("req-stale", status=TaskStatus.RUNNING)
    await repo.create(record)

    # A stale writer expecting QUEUED must fail AND consume no event id.
    stale = record.transition(TaskStatus.READY)
    assert await repo.update(stale, expected_status=TaskStatus.QUEUED) is None
    fresh = await repo.get(record.task_id)
    assert fresh is not None and fresh.event_id == 0


@pytest.mark.asyncio
async def test_claim_bumps_event_id(repo) -> None:
    record = _record("req-claim")
    await repo.create(record)
    claimed = await repo.claim_available(lease_seconds=60.0)
    assert claimed is not None and claimed.event_id == 1
    assert claimed.status == TaskStatus.RUNNING


@pytest.mark.asyncio
async def test_renew_lease_does_not_bump_event_id(repo) -> None:
    record = _record("req-renew")
    await repo.create(record)
    claimed = await repo.claim_available(lease_seconds=60.0)
    assert claimed is not None

    renewed = await repo.renew_lease(record.task_id, 60.0)
    # Lease heartbeats are not state changes — no SSE noise.
    assert renewed is not None and renewed.event_id == claimed.event_id


@pytest.mark.asyncio
async def test_migration_adds_missing_columns_to_old_schema(tmp_path) -> None:
    """A database created by an earlier release is migrated on astart()."""
    db_path = str(tmp_path / "old.sqlite")
    conn = await aiosqlite.connect(db_path)
    await conn.executescript(
        """
        CREATE TABLE cooking_tasks (
            task_id         TEXT PRIMARY KEY,
            request_id      TEXT NOT NULL UNIQUE,
            user_id         TEXT NOT NULL,
            status          TEXT NOT NULL,
            request_payload TEXT NOT NULL,
            thread_id       TEXT NOT NULL,
            revision        INTEGER NOT NULL DEFAULT 0,
            progress        TEXT NOT NULL DEFAULT '{}',
            result          TEXT,
            error           TEXT,
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL,
            expires_at      TEXT
        );
        """
    )
    await conn.commit()
    await conn.close()

    repo = SQLiteTaskRepository(db_path)
    await repo.astart()
    record = _record("req-mig")
    await repo.create(record)
    loaded = await repo.get(record.task_id)
    assert loaded is not None and loaded.event_id == 0 and loaded.attempts == 0

    running = record.transition(TaskStatus.RUNNING)
    claimed = await repo.update(running, expected_status=TaskStatus.QUEUED)
    assert claimed is not None and claimed.event_id == 1
    await repo.close()


# ---------------------------------------------------------------------------
# Service — subscribe generator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subscribe_unknown_task_yields_nothing(service) -> None:
    svc, _ = service
    assert await _collect(svc, "no-such-task") == []


@pytest.mark.asyncio
async def test_subscribe_terminal_task_replays_snapshot_and_closes(service) -> None:
    svc, repo = service
    outcome = await svc.submit(_valid_request("req-term-sse"))
    await svc._execute_claimed()  # type: ignore[attr-defined]  # QUEUED -> RUNNING -> READY
    done = await repo.get(outcome.task.task_id)
    assert done is not None and done.status == TaskStatus.READY

    # A fresh subscription replays the current (terminal) snapshot, then closes.
    frames = await _collect(svc, outcome.task.task_id)
    assert frames == [done]


@pytest.mark.asyncio
async def test_subscribe_last_event_id_filters_replay(service) -> None:
    svc, repo = service
    outcome = await svc.submit(_valid_request("req-resume"))
    await svc._execute_claimed()  # type: ignore[attr-defined]
    done = await repo.get(outcome.task.task_id)
    assert done is not None and done.status == TaskStatus.READY
    assert done.event_id >= 2

    # Reconnecting with a stale id replays only events after it.
    frames = await _collect(svc, outcome.task.task_id, last_event_id=done.event_id - 1)
    assert [f.event_id for f in frames if f is not None] == [done.event_id]

    # Reconnecting at the terminal id still surfaces the final done snapshot.
    frames = await _collect(svc, outcome.task.task_id, last_event_id=done.event_id)
    assert [f.event_id for f in frames if f is not None] == [done.event_id]


@pytest.mark.asyncio
async def test_subscribe_live_updates_stream_until_done(service) -> None:
    svc, repo = service
    outcome = await svc.submit(_valid_request("req-live"))

    async def consume() -> list[TaskRecord]:
        frames: list[TaskRecord] = []
        async for frame in svc.subscribe(outcome.task.task_id, -1, keepalive_seconds=5.0):
            if frame is not None:
                frames.append(frame)
            if frame is not None and frame.status == TaskStatus.READY:
                break
        return frames

    consumer = asyncio.create_task(consume())
    await asyncio.sleep(0.05)  # let the subscription register
    await svc._execute_claimed()  # type: ignore[attr-defined]  # claim + terminal notifications
    frames = await asyncio.wait_for(consumer, timeout=5.0)

    assert frames, "expected at least the terminal snapshot"
    ids = [f.event_id for f in frames]
    assert ids == sorted(ids) and len(ids) == len(set(ids)), f"event ids not monotonic: {ids}"
    assert frames[-1].status == TaskStatus.READY
    assert all(f.task_id == outcome.task.task_id for f in frames)


@pytest.mark.asyncio
async def test_subscribe_emits_keepalive_ticks_while_idle(service) -> None:
    svc, _ = service
    outcome = await svc.submit(_valid_request("req-keep"))  # stays QUEUED, no worker run

    frames: list[TaskRecord | None] = []
    async for frame in svc.subscribe(outcome.task.task_id, -1, keepalive_seconds=0.05):
        frames.append(frame)
        if len(frames) >= 3 and any(f is None for f in frames):
            break

    assert any(f is None for f in frames), "expected a keepalive tick on an idle task"
    assert frames[0] is not None and frames[0].task_id == outcome.task.task_id


# ---------------------------------------------------------------------------
# Service — notification fan-out
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subscribers_are_isolated_per_task(service) -> None:
    svc, repo = service
    task_a = (await svc.submit(_valid_request("req-iso-a"))).task
    task_b = (await svc.submit(_valid_request("req-iso-b"))).task

    received: list[str] = []

    async def consume_a() -> None:
        async for frame in svc.subscribe(task_a.task_id, -1, keepalive_seconds=5.0):
            if frame is not None:
                received.append(frame.task_id)
            if frame is not None and frame.status == TaskStatus.READY:
                return

    consumer = asyncio.create_task(consume_a())
    await asyncio.sleep(0.05)  # let the subscription register

    # A task-B write must never leak into A's subscriber queue.
    await svc.cancel(task_b.task_id)
    await svc._execute_claimed()  # type: ignore[attr-defined]  # completes task A
    await asyncio.wait_for(consumer, timeout=5.0)

    assert received, "expected task A progress events"
    assert all(t == task_a.task_id for t in received)


@pytest.mark.asyncio
async def test_notify_fanout_reaches_all_subscribers(service) -> None:
    svc, repo = service
    outcome = await svc.submit(_valid_request("req-fanout"))

    async def consume() -> list[TaskRecord]:
        frames: list[TaskRecord] = []
        async for frame in svc.subscribe(outcome.task.task_id, -1, keepalive_seconds=5.0):
            if frame is not None:
                frames.append(frame)
            if frame is not None and frame.status == TaskStatus.READY:
                return frames
        return frames

    consumer_a = asyncio.create_task(consume())
    consumer_b = asyncio.create_task(consume())
    await asyncio.sleep(0.05)
    await svc._execute_claimed()  # type: ignore[attr-defined]

    frames_a = await asyncio.wait_for(consumer_a, timeout=5.0)
    frames_b = await asyncio.wait_for(consumer_b, timeout=5.0)
    assert frames_a and frames_b
    assert frames_a[-1].status == TaskStatus.READY and frames_b[-1].status == TaskStatus.READY
