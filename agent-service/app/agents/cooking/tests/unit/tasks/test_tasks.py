"""P3-01 async task API tests.

Covers the task state machine, SQLite repository idempotency/concurrency,
service submit/cancel/worker execution, and the HTTP endpoints.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest
import pytest_asyncio

from cooking_plan_agent.domain.models import (
    GeneratePlanRequest,
    InventoryLotSnapshot,
    KitchenResourceSnapshot,
    ReadyPlanResponse,
)
from cooking_plan_agent.tasks.models import (
    TaskRecord,
    TaskStatus,
    is_terminal,
    new_task_id,
    validate_transition,
)
from cooking_plan_agent.tasks.repository import (
    DuplicateRequestError,
    SQLiteTaskRepository,
)
from cooking_plan_agent.tasks.service import (
    AsyncTaskService,
    TaskIdempotencyConflict,
)


def _valid_request(request_id: str = "req-async-001") -> GeneratePlanRequest:
    return GeneratePlanRequest(
        request_id=request_id,
        user_id="async-user",
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


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class TestStateMachine:
    def test_valid_transitions(self) -> None:
        assert validate_transition(TaskStatus.QUEUED, TaskStatus.RUNNING)
        assert validate_transition(TaskStatus.QUEUED, TaskStatus.CANCELLED)
        assert validate_transition(TaskStatus.QUEUED, TaskStatus.EXPIRED)
        assert validate_transition(TaskStatus.RUNNING, TaskStatus.READY)
        assert validate_transition(TaskStatus.RUNNING, TaskStatus.NEEDS_CONFIRMATION)
        assert validate_transition(TaskStatus.RUNNING, TaskStatus.INFEASIBLE)
        assert validate_transition(TaskStatus.RUNNING, TaskStatus.FAILED)
        assert validate_transition(TaskStatus.RUNNING, TaskStatus.CANCELLED)
        assert validate_transition(TaskStatus.NEEDS_CONFIRMATION, TaskStatus.RUNNING)

    def test_terminal_is_absorbing(self) -> None:
        for terminal in (
            TaskStatus.READY,
            TaskStatus.INFEASIBLE,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.EXPIRED,
        ):
            assert is_terminal(terminal)
            assert not validate_transition(terminal, TaskStatus.RUNNING)
            assert not validate_transition(terminal, TaskStatus.READY)

    def test_illegal_transition_rejected_by_record(self) -> None:
        record = _record(status=TaskStatus.READY)
        with pytest.raises(ValueError, match="Illegal task transition"):
            record.transition(TaskStatus.RUNNING)


# ---------------------------------------------------------------------------
# SQLite repository
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def repo(tmp_path):
    r = SQLiteTaskRepository(str(tmp_path / "tasks.sqlite"))
    await r.astart()
    yield r
    await r.close()


@pytest.mark.asyncio
async def test_create_and_get_roundtrip(repo) -> None:
    record = _record("req-rt")
    await repo.create(record)
    loaded = await repo.get(record.task_id)
    assert loaded is not None
    assert loaded.request_id == "req-rt"
    assert loaded.request_payload == {"request_id": "req-rt"}


@pytest.mark.asyncio
async def test_repository_creates_missing_parent_directory(tmp_path) -> None:
    db_path = tmp_path / "runtime-data" / "tasks.sqlite"
    repository = SQLiteTaskRepository(str(db_path))

    await repository.astart()

    assert db_path.is_file()
    await repository.close()


@pytest.mark.asyncio
async def test_duplicate_request_id_raises(repo) -> None:
    record = _record("req-dup")
    await repo.create(record)
    with pytest.raises(DuplicateRequestError):
        await repo.create(_record("req-dup"))


@pytest.mark.asyncio
async def test_get_by_request_id(repo) -> None:
    record = _record("req-byid")
    await repo.create(record)
    loaded = await repo.get_by_request_id("req-byid")
    assert loaded is not None and loaded.task_id == record.task_id


# ---------------------------------------------------------------------------
# Runtime cooking execution flow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execution_state_persists_and_unlocks_parallel_work(repo) -> None:
    """Completing crab prep unlocks blanching without blocking shrimp prep."""
    flow = (
        {"task_id": "crab_prep", "depends_on": (), "work_mode": "ACTIVE", "resources": ()},
        {
            "task_id": "crab_heat",
            "depends_on": ("crab_prep",),
            "work_mode": "PASSIVE",
            "resource_needs": ({"resource_type": "stove", "quantity": 1},),
        },
        {"task_id": "shrimp_prep", "depends_on": (), "work_mode": "ACTIVE", "resources": ()},
    )
    record = _record("req-execution", status=TaskStatus.READY).model_copy(
        update={
            "result": {"execution_flow": flow},
            "request_payload": {"kitchen_resources": ({"resource_type": "stove", "capacity": "1"},)},
        }
    )
    await repo.create(record)
    service = AsyncTaskService(repo, generation_service=None)  # type: ignore[arg-type]

    updated, snapshot, error = await service.update_execution(
        record.task_id, "crab_prep", "COMPLETED", expected_event_id=0
    )
    assert error is None
    assert updated is not None and updated.event_id == 1
    assert {item["task_id"] for item in snapshot["available_tasks"]} == {"crab_heat", "shrimp_prep"}  # type: ignore[index]

    updated, snapshot, error = await service.update_execution(
        record.task_id, "crab_heat", "IN_PROGRESS", expected_event_id=1
    )
    assert error is None
    assert updated is not None and updated.execution_state["crab_heat"] == "IN_PROGRESS"
    assert "shrimp_prep" in {item["task_id"] for item in snapshot["available_tasks"]}  # type: ignore[index]

    _latest, _snapshot, error = await service.update_execution(
        record.task_id, "shrimp_prep", "IN_PROGRESS", expected_event_id=1
    )
    assert error == "EXECUTION_STATE_CONFLICT"


@pytest.mark.asyncio
async def test_conditional_update_respects_status(repo) -> None:
    record = _record("req-cond", status=TaskStatus.QUEUED)
    await repo.create(record)

    running = record.transition(TaskStatus.RUNNING)
    updated = await repo.update(running, expected_status=TaskStatus.QUEUED)
    assert updated is not None and updated.status == TaskStatus.RUNNING

    # A stale writer expecting QUEUED must fail (D2).
    stale = record.transition(TaskStatus.RUNNING)
    again = await repo.update(stale, expected_status=TaskStatus.QUEUED)
    assert again is None


@pytest.mark.asyncio
async def test_list_running_only_non_terminal(repo) -> None:
    await repo.create(_record("req-a", status=TaskStatus.QUEUED))
    await repo.create(_record("req-b", status=TaskStatus.RUNNING))
    await repo.create(_record("req-c", status=TaskStatus.READY))
    rows = await repo.list_running()
    assert {r.request_id for r in rows} == {"req-a", "req-b"}


# ---------------------------------------------------------------------------
# Service — submit idempotency
# ---------------------------------------------------------------------------


class _FakeGeneration:
    """Stands in for GenerateCookingPlanService in service-level tests."""

    def __init__(self) -> None:
        self.executed: list[tuple[str, str]] = []

    async def execute(self, request: GeneratePlanRequest, thread_id: str | None = None):
        self.executed.append((request.request_id, thread_id or "none"))

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
async def service(tmp_path):
    repo = SQLiteTaskRepository(str(tmp_path / "svc.sqlite"))
    await repo.astart()
    svc = AsyncTaskService(repository=repo, generation_service=_FakeGeneration(), worker_concurrency=1)
    svc._generation = _FakeGeneration()  # type: ignore[attr-defined]  # explicit fake
    yield svc, repo
    if svc._worker_task:  # type: ignore[attr-defined]
        svc._worker_task.cancel()  # type: ignore[attr-defined]
    await repo.close()


@pytest.mark.asyncio
async def test_submit_creates_queued_task(service) -> None:
    svc, repo = service
    outcome = await svc.submit(_valid_request("req-sub"))
    assert outcome.created is True
    assert outcome.task.status == TaskStatus.QUEUED
    assert outcome.task.request_id == "req-sub"


@pytest.mark.asyncio
async def test_submit_same_payload_returns_same_task(service) -> None:
    svc, _ = service
    first = await svc.submit(_valid_request("req-idem"))
    second = await svc.submit(_valid_request("req-idem"))
    assert second.created is False
    assert second.task.task_id == first.task.task_id


@pytest.mark.asyncio
async def test_submit_conflicting_payload_raises(service) -> None:
    svc, _ = service
    await svc.submit(_valid_request("req-conflict"))
    # Same request_id, different payload (change servings).
    request = _valid_request("req-conflict")
    request = request.model_copy(update={"user_id": "different-user"})
    with pytest.raises(TaskIdempotencyConflict):
        await svc.submit(request)


@pytest.mark.asyncio
async def test_new_submission_cancels_same_users_old_task(service) -> None:
    svc, _ = service
    first = await svc.submit(_valid_request("req-old"))

    second = await svc.submit(_valid_request("req-new"))

    old = await svc.get(first.task.task_id)
    assert old is not None and old.status == TaskStatus.CANCELLED
    assert second.task.status == TaskStatus.QUEUED


@pytest.mark.asyncio
async def test_task_deadline_scales_with_dish_count_and_caps_at_five_minutes(service) -> None:
    svc, _ = service
    one = _valid_request("req-one")
    many = one.model_copy(
        update={
            "request_id": "req-many",
            "recipes": tuple(one.recipes[0].model_copy(update={"recipe_id": f"r{i}"}) for i in range(8)),
        }
    )

    assert svc._task_ttl(one, None).total_seconds() == 180  # type: ignore[attr-defined]
    assert svc._task_ttl(many, None).total_seconds() == 300  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_cancel_queued_task(service) -> None:
    svc, _ = service
    outcome = await svc.submit(_valid_request("req-cancel"))
    cancelled = await svc.cancel(outcome.task.task_id)
    assert cancelled is not None and cancelled.status == TaskStatus.CANCELLED


@pytest.mark.asyncio
async def test_cancel_unknown_task_returns_none(service) -> None:
    svc, _ = service
    assert await svc.cancel("no-such-task") is None


@pytest.mark.asyncio
async def test_cancel_terminal_task_is_noop(service) -> None:
    svc, repo = service
    outcome = await svc.submit(_valid_request("req-term"))
    # Move QUEUED -> RUNNING -> READY (legal path through the state machine).
    running = outcome.task.transition(TaskStatus.RUNNING)
    await repo.update(running, expected_status=TaskStatus.QUEUED)
    ready = running.transition(TaskStatus.READY)
    await repo.update(ready, expected_status=TaskStatus.RUNNING)
    result = await svc.cancel(outcome.task.task_id)
    assert result is not None and result.status == TaskStatus.READY


# ---------------------------------------------------------------------------
# Service — worker execution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_executes_task_to_readiness(tmp_path) -> None:
    """The in-process worker claims a QUEUED task and finishes it as READY."""
    repo = SQLiteTaskRepository(str(tmp_path / "w.sqlite"))
    await repo.astart()
    gen = _FakeGeneration()
    svc = AsyncTaskService(repository=repo, generation_service=gen, worker_concurrency=1)

    outcome = await svc.submit(_valid_request("req-run"))
    assert outcome.created is True

    # Drive one worker claim manually (avoids loop timing flakiness).
    await svc._execute_claimed()  # type: ignore[attr-defined]

    done = await svc.get(outcome.task.task_id)
    assert done is not None
    assert done.status == TaskStatus.READY
    assert done.result is not None and done.result["status"] == "READY"
    assert gen.executed == [("req-run", "req-run:0")]
    await repo.close()


class _StreamingGeneration:
    def __init__(self) -> None:
        self.progress_written = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def execute_with_progress(self, request, thread_id=None, on_progress=None):
        try:
            await on_progress("validate_input", 1)
            await on_progress("parse_recipes", 2)
            self.progress_written.set()
            await self.release.wait()
            return ReadyPlanResponse(
                plan_id="plan-stream",
                solver_status="OPTIMAL",
                makespan_minutes=30,
                timeline=(),
                completion_checklist=(),
                mise_en_place=(),
                dish_completions=(),
            )
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


@pytest.mark.asyncio
async def test_worker_persists_real_node_progress(tmp_path) -> None:
    repo = SQLiteTaskRepository(str(tmp_path / "progress.sqlite"))
    await repo.astart()
    generation = _StreamingGeneration()
    svc = AsyncTaskService(repo, generation, worker_concurrency=1)
    outcome = await svc.submit(_valid_request("req-progress"))

    worker = asyncio.create_task(svc._execute_claimed())  # type: ignore[attr-defined]
    await asyncio.wait_for(generation.progress_written.wait(), timeout=1)
    running = await svc.get(outcome.task.task_id)
    assert running is not None
    assert running.progress.node == "parse_recipes"
    assert running.progress.completed_steps == 2

    generation.release.set()
    await worker
    await repo.close()


@pytest.mark.asyncio
async def test_new_submission_interrupts_old_running_coroutine(tmp_path) -> None:
    repo = SQLiteTaskRepository(str(tmp_path / "cancel-running.sqlite"))
    await repo.astart()
    generation = _StreamingGeneration()
    svc = AsyncTaskService(repo, generation, worker_concurrency=1)
    old = await svc.submit(_valid_request("req-running-old"))

    worker = asyncio.create_task(svc._execute_claimed())  # type: ignore[attr-defined]
    await asyncio.wait_for(generation.progress_written.wait(), timeout=1)
    await svc.submit(_valid_request("req-running-new"))
    await asyncio.wait_for(generation.cancelled.wait(), timeout=1)
    await worker

    cancelled = await svc.get(old.task.task_id)
    assert cancelled is not None and cancelled.status == TaskStatus.CANCELLED
    await repo.close()


@pytest.mark.asyncio
async def test_worker_claim_is_idempotent(tmp_path) -> None:
    """A completed task is not re-claimed (D2)."""
    repo = SQLiteTaskRepository(str(tmp_path / "c.sqlite"))
    await repo.astart()
    svc = AsyncTaskService(repository=repo, generation_service=_FakeGeneration(), worker_concurrency=1)

    outcome = await svc.submit(_valid_request("req-claim"))
    # First claim wins and completes the task.
    await svc._execute_claimed()  # type: ignore[attr-defined]
    first = await svc.get(outcome.task.task_id)
    assert first is not None and first.status == TaskStatus.READY

    # A second claim finds nothing (terminal is absorbing — no double-run).
    await svc._execute_claimed()  # type: ignore[attr-defined]
    second = await svc.get(outcome.task.task_id)
    assert second is not None and second.status == TaskStatus.READY
    await repo.close()


@pytest.mark.asyncio
async def test_recovery_requeues_running_tasks(tmp_path) -> None:
    """After a restart (no lease), RUNNING tasks are re-claimable."""
    repo = SQLiteTaskRepository(str(tmp_path / "r.sqlite"))
    await repo.astart()
    svc = AsyncTaskService(repository=repo, generation_service=_FakeGeneration(), worker_concurrency=1)

    outcome = await svc.submit(_valid_request("req-recover"))
    # Simulate a crash mid-flight: move to RUNNING but never finish.
    running = outcome.task.transition(TaskStatus.RUNNING)
    await repo.update(running, expected_status=TaskStatus.QUEUED)

    # With no lease, the task is still claimable (expired lease = reclaimable).
    claimed = await repo.claim_available(lease_seconds=60.0)
    assert claimed is not None and claimed.task_id == outcome.task.task_id
    assert claimed.status == TaskStatus.RUNNING
    assert claimed.attempts == 1
    await repo.close()


@pytest.mark.asyncio
async def test_expired_task_moves_to_expired(tmp_path) -> None:
    """A QUEUED task past its TTL is moved to EXPIRED by the worker loop."""
    repo = SQLiteTaskRepository(str(tmp_path / "e.sqlite"))
    await repo.astart()
    svc = AsyncTaskService(repository=repo, generation_service=_FakeGeneration(), worker_concurrency=1)

    # Submit with a TTL already in the past.
    outcome = await svc.submit(_valid_request("req-expire"), ttl_seconds=-10)
    assert outcome.created is True

    await svc._expire_stale_tasks()  # type: ignore[attr-defined]
    expired = await svc.get(outcome.task.task_id)
    assert expired is not None and expired.status == TaskStatus.EXPIRED

    # Expired is terminal: the worker must not execute it.
    await svc._execute_claimed()  # type: ignore[attr-defined]
    after = await svc.get(outcome.task.task_id)
    assert after is not None and after.status == TaskStatus.EXPIRED
    await repo.close()
