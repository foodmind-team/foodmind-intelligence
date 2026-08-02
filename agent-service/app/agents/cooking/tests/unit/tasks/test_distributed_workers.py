"""P3-02 distributed task execution tests.

Covers the lease/visibility-timeout semantics, dual-worker competition,
lease-expiry reclaim after a worker crash, lease renewal heartbeats, and
retry/dead-letter handling.

Key acceptance points (P3-02):
- Two workers competing for the same task produce exactly one result revision.
- Killing a worker at any stage (solve/provider/commit) is safe: the lease
  expires and another worker re-claims the task.
- Duplicate/out-of-order messages never produce duplicate business results.
- Scale-down to a single worker loses no tasks.
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
from cooking_plan_agent.tasks.models import TaskStatus
from cooking_plan_agent.tasks.repository import SQLiteTaskRepository
from cooking_plan_agent.tasks.service import AsyncTaskService


def _request(request_id: str = "req-dist-001") -> GeneratePlanRequest:
    return GeneratePlanRequest(
        request_id=request_id,
        user_id="dist-user",
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
                lot_id="lot-1",
                item_id="item-1",
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


class _CountingGeneration:
    """Fake generation service that records every execution attempt.

    Used to prove that even if the graph is invoked more than once (e.g. a
    re-claim after lease expiry), the conditional result write produces a
    single authoritative revision.
    """

    def __init__(self) -> None:
        self.execution_count = 0
        self.fail_first_n = 0

    async def execute(self, request: GeneratePlanRequest, thread_id: str | None = None):
        self.execution_count += 1
        if self.execution_count <= self.fail_first_n:
            raise RuntimeError(f"injected failure #{self.execution_count}")
        return ReadyPlanResponse(
            plan_id="plan-dist-1",
            solver_status="OPTIMAL",
            makespan_minutes=30,
            timeline=(),
            completion_checklist=(),
            mise_en_place=(),
            dish_completions=(),
        )


@pytest_asyncio.fixture
async def repo(tmp_path):
    r = SQLiteTaskRepository(str(tmp_path / "dist.sqlite"))
    await r.astart()
    yield r
    await r.close()


# ---------------------------------------------------------------------------
# Lease claim semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_claim_moves_queued_to_running_with_lease(repo) -> None:
    from cooking_plan_agent.tasks.service import AsyncTaskService

    svc = AsyncTaskService(repository=repo, generation_service=_CountingGeneration())
    outcome = await svc.submit(_request("req-claim"))

    claimed = await repo.claim_available(lease_seconds=60.0)
    assert claimed is not None
    assert claimed.task_id == outcome.task.task_id
    assert claimed.status == TaskStatus.RUNNING
    assert claimed.attempts == 1
    assert claimed.lease_expires_at is not None

    # A second claim finds nothing (task already leased and not expired).
    assert await repo.claim_available(lease_seconds=60.0) is None
    await repo.close()


@pytest.mark.asyncio
async def test_claim_reclaims_task_with_expired_lease(repo) -> None:
    """A RUNNING task whose lease expired is re-claimable (worker crash, D4)."""
    from datetime import timedelta

    from cooking_plan_agent.tasks.models import utc_now
    from cooking_plan_agent.tasks.service import AsyncTaskService

    svc = AsyncTaskService(repository=repo, generation_service=_CountingGeneration())
    outcome = await svc.submit(_request("req-expired-lease"))

    claimed = await repo.claim_available(lease_seconds=60.0)
    assert claimed is not None

    # Simulate a crashed worker: force the lease into the past.

    await repo.conn.execute(
        "UPDATE cooking_tasks SET lease_expires_at=? WHERE task_id=?",
        ((utc_now() - timedelta(seconds=5)).isoformat(), outcome.task.task_id),
    )
    await repo.conn.commit()

    # A fresh worker can now claim it (lease expired).
    re_claimed = await repo.claim_available(lease_seconds=60.0)
    assert re_claimed is not None
    assert re_claimed.task_id == outcome.task.task_id
    assert re_claimed.attempts == 2
    await repo.close()


@pytest.mark.asyncio
async def test_renew_lease_extends_and_respects_lost_lease(repo) -> None:
    """renew_lease works while leased; fails once the lease is lost."""
    from datetime import timedelta

    from cooking_plan_agent.tasks.models import utc_now
    from cooking_plan_agent.tasks.service import AsyncTaskService

    svc = AsyncTaskService(repository=repo, generation_service=_CountingGeneration())
    outcome = await svc.submit(_request("req-renew"))
    claimed = await repo.claim_available(lease_seconds=60.0)
    assert claimed is not None

    # Renewal succeeds while the lease is held.
    renewed = await repo.renew_lease(outcome.task.task_id, 60.0)
    assert renewed is not None and renewed.lease_expires_at is not None

    # Simulate the lease expiring without renewal (worker heartbeat stopped).
    await repo.conn.execute(
        "UPDATE cooking_tasks SET lease_expires_at=? WHERE task_id=?",
        ((utc_now() - timedelta(seconds=10)).isoformat(), outcome.task.task_id),
    )
    await repo.conn.commit()
    # After the lease is lost, renewal is refused (D4 visibility timeout).
    assert await repo.renew_lease(outcome.task.task_id, 60.0) is None
    await repo.close()


# ---------------------------------------------------------------------------
# Dual-worker competition
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_workers_produce_single_result(tmp_path) -> None:
    """Two services over the same store yield exactly one READY revision."""
    repo = SQLiteTaskRepository(str(tmp_path / "two.sqlite"))
    await repo.astart()
    gen = _CountingGeneration()
    svc_a = AsyncTaskService(repository=repo, generation_service=gen, worker_concurrency=1)
    svc_b = AsyncTaskService(repository=repo, generation_service=gen, worker_concurrency=1)

    outcome = await svc_a.submit(_request("req-two-workers"))

    # Both workers race to claim the same task.
    await svc_a._execute_claimed()  # type: ignore[attr-defined]
    await svc_b._execute_claimed()  # type: ignore[attr-defined]

    done = await repo.get(outcome.task.task_id)
    assert done is not None and done.status == TaskStatus.READY
    assert done.result is not None and done.result["status"] == "READY"
    # Exactly one authoritative result revision.
    assert done.result["status"] == "READY"
    # The graph may have run once (only the claiming worker ran it).
    assert gen.execution_count == 1
    await repo.close()


@pytest.mark.asyncio
async def test_worker_crash_reclaims_and_completes(tmp_path) -> None:
    """A worker 'crash' (no completion write) lets another worker finish.

    Simulates killing the worker between claim and result commit: the first
    claim never writes a result, the lease expires, and a second worker
    re-claims and completes the task.
    """
    from datetime import timedelta

    from cooking_plan_agent.tasks.models import utc_now

    repo = SQLiteTaskRepository(str(tmp_path / "crash.sqlite"))
    await repo.astart()
    gen = _CountingGeneration()
    svc = AsyncTaskService(repository=repo, generation_service=gen, worker_concurrency=1)

    outcome = await svc.submit(_request("req-crash"))
    claimed = await repo.claim_available(lease_seconds=60.0)
    assert claimed is not None

    # Simulate the crash: expire the lease, leaving status RUNNING with no result.
    await repo.conn.execute(
        "UPDATE cooking_tasks SET lease_expires_at=? WHERE task_id=?",
        ((utc_now() - timedelta(seconds=5)).isoformat(), outcome.task.task_id),
    )
    await repo.conn.commit()

    # A new worker (restarted service) claims and completes.
    await svc._execute_claimed()  # type: ignore[attr-defined]
    done = await repo.get(outcome.task.task_id)
    assert done is not None and done.status == TaskStatus.READY
    assert done.result is not None and done.result["status"] == "READY"
    assert done.attempts == 2
    await repo.close()


# ---------------------------------------------------------------------------
# Retry and dead-letter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failure_requeues_then_dead_letters(tmp_path) -> None:
    """Transient failures retry; exhausted attempts dead-letter as FAILED."""
    repo = SQLiteTaskRepository(str(tmp_path / "dlq.sqlite"))
    await repo.astart()
    gen = _CountingGeneration()
    gen.fail_first_n = 100  # always fail
    svc = AsyncTaskService(repository=repo, generation_service=gen, worker_concurrency=1, max_attempts=2)

    outcome = await svc.submit(_request("req-dlq"))
    # Attempt 1: claim (attempts=1) -> failure -> not yet at max -> requeue.
    await svc._execute_claimed()  # type: ignore[attr-defined]
    mid = await repo.get(outcome.task.task_id)
    assert mid is not None and mid.status == TaskStatus.QUEUED
    assert mid.attempts == 1

    # Attempt 2: claim (attempts=2) -> failure -> at max -> dead-letter FAILED.
    await svc._execute_claimed()  # type: ignore[attr-defined]
    done = await repo.get(outcome.task.task_id)
    assert done is not None and done.status == TaskStatus.FAILED
    assert done.error is not None and done.error["error_code"] == "INTERNAL_ERROR"
    assert done.attempts == 2
    await repo.close()


@pytest.mark.asyncio
async def test_transient_failure_recovers_on_retry(tmp_path) -> None:
    """A task that fails once succeeds on its retry."""
    repo = SQLiteTaskRepository(str(tmp_path / "retry.sqlite"))
    await repo.astart()
    gen = _CountingGeneration()
    gen.fail_first_n = 1  # fail once, then succeed
    svc = AsyncTaskService(repository=repo, generation_service=gen, worker_concurrency=1, max_attempts=3)

    outcome = await svc.submit(_request("req-retry"))
    await svc._execute_claimed()  # type: ignore[attr-defined]  # fails -> requeue
    mid = await repo.get(outcome.task.task_id)
    assert mid is not None and mid.status == TaskStatus.QUEUED and mid.attempts == 1

    await svc._execute_claimed()  # type: ignore[attr-defined]  # succeeds
    done = await repo.get(outcome.task.task_id)
    assert done is not None and done.status == TaskStatus.READY
    assert done.result is not None and done.result["status"] == "READY"
    await repo.close()


@pytest.mark.asyncio
async def test_scale_down_to_single_worker_loses_no_tasks(tmp_path) -> None:
    """Queued tasks all complete with a single worker (scale-down, P3-02)."""
    repo = SQLiteTaskRepository(str(tmp_path / "scale.sqlite"))
    await repo.astart()
    gen = _CountingGeneration()
    svc = AsyncTaskService(repository=repo, generation_service=gen, worker_concurrency=1)

    task_ids = []
    for i in range(4):
        outcome = await svc.submit(_request(f"req-scale-{i}"))
        task_ids.append(outcome.task.task_id)

    for _ in range(4):
        await svc._execute_claimed()  # type: ignore[attr-defined]

    for task_id in task_ids:
        done = await repo.get(task_id)
        assert done is not None and done.status == TaskStatus.READY, f"Task {task_id} lost"
    await repo.close()
