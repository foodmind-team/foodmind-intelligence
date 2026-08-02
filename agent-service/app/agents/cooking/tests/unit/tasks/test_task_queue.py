"""P4-05 task queue port — TaskQueue protocol and InProcessTaskQueue adapter.

Stage A extracts the worker's claim/lease/conditional-write operations
behind a queue port. These tests prove:

- the in-process adapter preserves the repository semantics exactly
  (atomic claim + lease, conditional result write);
- the service consumes the port by default AND when a queue is injected;
- backend selection fails loudly for unapproved backends (Stage B gate).

Fault-injection coverage lives in tests/unit/tasks/test_distributed_workers.py.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
import pytest_asyncio

from cooking_plan_agent.domain.models import (
    GeneratePlanRequest,
    InventoryLotSnapshot,
    KitchenResourceSnapshot,
    ReadyPlanResponse,
)
from cooking_plan_agent.tasks.models import TaskRecord, TaskStatus
from cooking_plan_agent.tasks.queue import (
    InProcessTaskQueue,
    create_task_queue,
)
from cooking_plan_agent.tasks.repository import SQLiteTaskRepository, TaskRepository
from cooking_plan_agent.tasks.service import AsyncTaskService


def _valid_request(request_id: str = "req-queue-001") -> GeneratePlanRequest:
    return GeneratePlanRequest(
        request_id=request_id,
        user_id="queue-user",
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


class _FakeGeneration:
    """Stands in for GenerateCookingPlanService — instant READY plan."""

    async def execute(self, request: GeneratePlanRequest, thread_id: str | None = None):
        return ReadyPlanResponse(
            plan_id="plan-queue-1",
            solver_status="OPTIMAL",
            makespan_minutes=30,
            timeline=(),
            completion_checklist=(),
            mise_en_place=(),
            dish_completions=(),
        )


@pytest_asyncio.fixture
async def repo(tmp_path):
    r = SQLiteTaskRepository(str(tmp_path / "queue.sqlite"))
    await r.astart()
    yield r
    await r.close()


# ---------------------------------------------------------------------------
# InProcessTaskQueue — port semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inprocess_queue_mirrors_repository_semantics(repo) -> None:
    """claim / renew / complete map onto the repository's atomic writes."""
    svc = AsyncTaskService(repository=repo, generation_service=_FakeGeneration())
    outcome = await svc.submit(_valid_request("req-port"))
    queue = InProcessTaskQueue(repo)

    # Claim: QUEUED -> RUNNING with a lease and attempts+1 (exactly one worker).
    claimed = await queue.claim_available(lease_seconds=60.0)
    assert claimed is not None and claimed.task_id == outcome.task.task_id
    assert claimed.status == TaskStatus.RUNNING
    assert claimed.attempts == 1
    # A concurrent claim finds nothing (lease held).
    assert await queue.claim_available(lease_seconds=60.0) is None

    # Renewal extends the held lease.
    renewed = await queue.renew_lease(outcome.task.task_id, 60.0)
    assert renewed is not None and renewed.status == TaskStatus.RUNNING

    # Complete: conditional result write (D2) — succeeds from RUNNING.
    terminal = renewed.transition(TaskStatus.READY)
    updated = await queue.complete(terminal, expected_status=TaskStatus.RUNNING)
    assert updated is not None and updated.status == TaskStatus.READY
    assert updated.result is None  # fake generation path not used here

    # A stale writer expecting QUEUED must fail (no duplicate/clobber).
    stale = await queue.complete(terminal, expected_status=TaskStatus.QUEUED)
    assert stale is None


@pytest.mark.asyncio
async def test_queue_close_does_not_close_shared_repository(repo) -> None:
    """close() releases queue-side resources only — the store stays usable."""
    queue = InProcessTaskQueue(repo)
    await queue.close()
    # The repository connection is owned by the service/lifespan.
    record = TaskRecord(
        task_id="t-close",
        request_id="req-close",
        user_id="u",
        request_payload={"request_id": "req-close"},
        thread_id="req-close:0",
    )
    await repo.create(record)
    assert await repo.get("t-close") is not None


# ---------------------------------------------------------------------------
# Service consumes the port
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_service_defaults_to_inprocess_queue(repo) -> None:
    """Without an explicit queue, the worker uses the in-process adapter."""
    svc = AsyncTaskService(
        repository=repo,
        generation_service=_FakeGeneration(),
        worker_concurrency=1,
    )
    assert isinstance(svc._queue, InProcessTaskQueue)  # type: ignore[attr-defined]

    outcome = await svc.submit(_valid_request("req-default"))
    await svc._execute_claimed()  # type: ignore[attr-defined]
    done = await svc.get(outcome.task.task_id)
    assert done is not None and done.status == TaskStatus.READY


class _RecordingQueue:
    """Records every port call while delegating to the in-process queue.

    Used to prove the worker routes claim / complete through the injected
    queue rather than the repository directly.
    """

    def __init__(self, repository: TaskRepository) -> None:
        self._inner = InProcessTaskQueue(repository)
        self.claims: list[float] = []
        self.completes: list[tuple[TaskStatus, TaskStatus]] = []

    async def claim_available(self, lease_seconds: float) -> TaskRecord | None:
        self.claims.append(lease_seconds)
        return await self._inner.claim_available(lease_seconds)

    async def renew_lease(self, task_id: str, lease_seconds: float) -> TaskRecord | None:
        return await self._inner.renew_lease(task_id, lease_seconds)

    async def complete(self, record: TaskRecord, expected_status: TaskStatus) -> TaskRecord | None:
        self.completes.append((record.status, expected_status))
        return await self._inner.complete(record, expected_status)

    async def close(self) -> None:
        await self._inner.close()


@pytest.mark.asyncio
async def test_injected_queue_is_consumed_by_worker(repo) -> None:
    """The worker claims through the port and commits via queue.complete."""
    queue = _RecordingQueue(repo)
    svc = AsyncTaskService(
        repository=repo,
        generation_service=_FakeGeneration(),
        worker_concurrency=1,
        queue=queue,  # type: ignore[arg-type]
    )
    outcome = await svc.submit(_valid_request("req-inj"))
    await svc._execute_claimed()  # type: ignore[attr-defined]

    # Claim went through the port with the configured lease.
    assert queue.claims == [svc._lease_seconds]  # type: ignore[attr-defined]
    # The terminal write went through the port: READY committed from RUNNING.
    assert (TaskStatus.READY, TaskStatus.RUNNING) in queue.completes

    done = await svc.get(outcome.task.task_id)
    assert done is not None and done.status == TaskStatus.READY


# ---------------------------------------------------------------------------
# Backend selection (Stage A gate)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_task_queue_selects_inprocess(repo) -> None:
    queue = create_task_queue("inprocess", repo)
    assert isinstance(queue, InProcessTaskQueue)


@pytest.mark.asyncio
async def test_create_task_queue_rejects_unapproved_backend(repo) -> None:
    """A distributed backend is Stage B — it must fail loudly, not degrade."""
    with pytest.raises(RuntimeError, match="infrastructure approval"):
        create_task_queue("redis", repo)
    with pytest.raises(RuntimeError, match="infrastructure approval"):
        create_task_queue("celery", repo)
