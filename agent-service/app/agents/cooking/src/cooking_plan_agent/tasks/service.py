"""Async task service — submission, query, cancel, and worker execution (P3-01).

The service decouples long-running plan generation from the synchronous
HTTP budget:
  - ``submit`` enforces the idempotency key (D1): same request_id + same
    payload returns the same task; a conflicting payload is a 409.
  - ``get`` / ``cancel`` / ``resume`` operate on persisted TaskRecords.
  - ``run_worker`` is the in-process execution loop: it claims QUEUED tasks
    (conditional QUEUED->RUNNING), runs the workflow graph under the P2-06
    checkpoint thread, and writes the terminal result conditionally.

Process-restart recovery (P3-01): on startup the service reloads QUEUED and
RUNNING tasks and re-queues them; RUNNING tasks are safe to re-run because
results are written conditionally by task ID/revision (D2) and the graph is
idempotent at the checkpoint level.

The worker runs in-process for MVP (approved decision). P3-02 replaces this
loop with a distributed queue port without changing the submit/query API.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import timedelta

from cooking_plan_agent.application.service import GenerateCookingPlanService
from cooking_plan_agent.domain.models import GeneratePlanRequest, PlanResponse
from cooking_plan_agent.tasks.models import (
    TaskProgress,
    TaskRecord,
    TaskStatus,
    new_task_id,
    utc_now,
)
from cooking_plan_agent.tasks.repository import (
    DuplicateRequestError,
    TaskRepository,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SubmitOutcome:
    """Result of a submit call (202 or 409 idempotency conflict)."""

    task: TaskRecord
    created: bool
    conflict: bool = False


class TaskIdempotencyConflict(Exception):
    """Raised when the same request_id is resubmitted with a different payload."""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        super().__init__(f"Idempotency conflict for request_id={request_id}")


class AsyncTaskService:
    """Application service behind the async task API (P3-01, P3-02)."""

    def __init__(
        self,
        repository: TaskRepository,
        generation_service: GenerateCookingPlanService,
        *,
        default_ttl_seconds: int = 3600,
        worker_concurrency: int = 2,
        lease_seconds: float = 60.0,
        max_attempts: int = 3,
    ) -> None:
        self._repo = repository
        self._generation = generation_service
        self._default_ttl = timedelta(seconds=default_ttl_seconds)
        self._worker_concurrency = max(1, worker_concurrency)
        self._lease_seconds = lease_seconds
        self._max_attempts = max_attempts
        self._worker_task: asyncio.Task[None] | None = None

    # -- lifecycle ----------------------------------------------------------

    async def astart(self) -> None:
        """Start the lease-based worker loop (P3-02).

        Unlike P3-01's manual re-queue, interrupted RUNNING tasks are picked
        up automatically once their lease expires (D4): no explicit recovery
        pass is needed — the first claim_available that sees an expired lease
        re-claims the task.
        """
        self._worker_task = asyncio.create_task(
            self._run_worker_loop(),
            name="cooking-task-worker",
        )
        logger.info(
            "Async task worker started | concurrency=%d | lease=%ss",
            self._worker_concurrency,
            self._lease_seconds,
        )

    async def aclose(self) -> None:
        """Stop the worker loop (bounded drain)."""
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
        await self._repo.close()

    # -- API operations -----------------------------------------------------

    async def submit(self, request: GeneratePlanRequest, ttl_seconds: int | None = None) -> SubmitOutcome:
        """Submit a plan-generation task (idempotent on request_id).

        Same request_id + same payload -> returns the existing task
        (``created=False``). Same request_id + different payload -> raises
        TaskIdempotencyConflict (409). New request -> persisted QUEUED task.
        """
        payload = json.loads(request.model_dump_json())
        ttl = timedelta(seconds=ttl_seconds) if ttl_seconds else self._default_ttl

        record = TaskRecord(
            task_id=new_task_id(),
            request_id=request.request_id,
            user_id=request.user_id,
            request_payload=payload,
            thread_id=self._thread_id(request),
            created_at=utc_now(),
            updated_at=utc_now(),
            expires_at=utc_now() + ttl,
        )
        try:
            await self._repo.create(record)
            logger.info(
                "Task submitted | task_id=%s | request_id=%s",
                record.task_id,
                record.request_id,
            )
            return SubmitOutcome(task=record, created=True)
        except DuplicateRequestError:
            existing = await self._repo.get_by_request_id(request.request_id)
            if existing is None:
                # Race: duplicate vanished between create and read — retry once.
                await self._repo.create(record)
                return SubmitOutcome(task=record, created=True)
            if existing.request_payload != payload:
                raise TaskIdempotencyConflict(request.request_id) from None
            return SubmitOutcome(task=existing, created=False)

    async def get(self, task_id: str) -> TaskRecord | None:
        """Fetch a task by ID (404 when absent)."""
        return await self._repo.get(task_id)

    async def cancel(self, task_id: str) -> TaskRecord | None:
        """Cooperatively cancel a QUEUED/RUNNING task.

        QUEUED -> CANCELLED immediately. RUNNING -> CANCELLED; the worker
        checks the flag at node boundaries (cooperative cancellation, P3-01)
        and will not persist further results for a cancelled task.
        """
        record = await self._repo.get(task_id)
        if record is None or record.status not in (TaskStatus.QUEUED, TaskStatus.RUNNING):
            return record

        cancelled = record.model_copy(update={"status": TaskStatus.CANCELLED, "updated_at": utc_now()})
        updated = await self._repo.update(cancelled, expected_status=record.status)
        return updated or record

    def _thread_id(self, request: GeneratePlanRequest) -> str:
        from cooking_plan_agent.infrastructure.checkpointer import build_thread_id

        return build_thread_id(request.request_id, request.plan_revision)

    # -- worker -------------------------------------------------------------

    async def _run_worker_loop(self) -> None:
        """Lease-based worker loop (P3-02).

        Repeatedly: claim the next available task (QUEUED, or RUNNING with an
        expired lease — D4), execute it while a background heartbeat renews
        the lease, then write the terminal result. At most
        ``worker_concurrency`` tasks run simultaneously.
        """
        while True:
            try:
                await self._expire_stale_tasks()
                # Each iteration claims up to `worker_concurrency` tasks and
                # runs them concurrently; the atomic claim guarantees no two
                # workers execute the same task (D2).
                tasks = [asyncio.create_task(self._execute_claimed()) for _ in range(self._worker_concurrency)]
                for task in tasks:
                    try:
                        await task
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:  # noqa: BLE001 — worker must stay alive
                        logger.error("Task execution raised", extra={"error": str(exc)})
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — worker loop must stay alive
                logger.error("Task worker iteration failed", extra={"error": str(exc)})
            await asyncio.sleep(0.3)

    async def _expire_stale_tasks(self) -> None:
        """Move QUEUED/RUNNING tasks past their TTL to EXPIRED (P3-01).

        Expiry is conditional on the observed status so a task that
        completed between the list and the update is never clobbered.
        """
        now = utc_now()
        for record in await self._repo.list_running():
            if record.expires_at is None or record.expires_at > now:
                continue
            expired = record.model_copy(
                update={
                    "status": TaskStatus.EXPIRED,
                    "updated_at": now,
                    "progress": TaskProgress(message="Task expired before completion"),
                }
            )
            updated = await self._repo.update(expired, expected_status=record.status)
            if updated is not None:
                logger.info(
                    "Task expired | task_id=%s | from=%s",
                    record.task_id,
                    record.status.value,
                )

    async def _execute_claimed(self) -> None:
        """Claim one available task and run it to a terminal state (P3-02).

        The atomic claim sets RUNNING + lease + attempts+1. While the graph
        runs, a heartbeat task renews the lease; if this worker dies, the
        lease expires and another worker re-claims the task (D4).

        On failure: if attempts still allow retries the task is re-queued
        (QUEUED, backoff window), otherwise it is dead-lettered as FAILED
        (P3-02). Results are written conditionally on RUNNING (D2).
        """
        record = await self._repo.claim_available(self._lease_seconds)
        if record is None:
            return

        heartbeat = asyncio.create_task(self._renew_heartbeat(record.task_id))
        try:
            terminal = await self._run_graph(record)
            await self._repo.update(terminal, expected_status=TaskStatus.RUNNING)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — failure handling below
            logger.exception(
                "Task execution failed | task_id=%s | attempts=%d/%d",
                record.task_id,
                record.attempts,
                self._max_attempts,
            )
            await self._handle_retry_or_dead_letter(record, exc)
        finally:
            heartbeat.cancel()
            try:
                await heartbeat
            except asyncio.CancelledError:
                pass

    async def _renew_heartbeat(self, task_id: str) -> None:
        """Periodically renew the task lease until cancelled (P3-02 D4).

        Renewal is conditional: if the lease was lost (another worker
        re-claimed after expiry), renew_lease returns None and the heartbeat
        stops quietly.
        """
        try:
            while True:
                await asyncio.sleep(max(1.0, self._lease_seconds / 3))
                renewed = await self._repo.renew_lease(task_id, self._lease_seconds)
                if renewed is None:
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — heartbeat failure must not crash worker
            logger.warning(
                "Lease renewal failed | task_id=%s | error=%s",
                task_id,
                exc,
            )

    async def _handle_retry_or_dead_letter(self, record: TaskRecord, exc: Exception) -> None:
        """Retry (re-queue) or dead-letter the task after a failure (P3-02).

        ``attempts`` was already incremented by the atomic claim. When the
        attempt budget is exhausted the task becomes FAILED with a stable
        error (dead-letter); otherwise it is re-queued with an exponential
        backoff so a flapping task cannot busy-loop.
        """
        error_body = {
            "status": 500,
            "error_code": "INTERNAL_ERROR",
            "message": "Task execution failed.",
            "correlation_id": record.request_id,
            "retryable": False,
        }
        if record.attempts >= self._max_attempts:
            dead = record.model_copy(
                update={
                    "status": TaskStatus.FAILED,
                    "updated_at": utc_now(),
                    "error": error_body,
                    "progress": TaskProgress(
                        node="failed",
                        message="Task failed after all attempts (dead-letter)",
                    ),
                }
            )
            await self._repo.update(dead, expected_status=TaskStatus.RUNNING)
            logger.warning(
                "Task dead-lettered | task_id=%s | attempts=%d",
                record.task_id,
                record.attempts,
            )
            return

        # Re-queue with a short backoff so a flapping task cannot busy-loop.
        # The hard TTL (expires_at) is never extended by a retry.
        backoff = min(30.0, 2.0**record.attempts)
        requeued = record.model_copy(
            update={
                "status": TaskStatus.QUEUED,
                "updated_at": utc_now(),
                "lease_expires_at": None,
                "progress": TaskProgress(
                    node="retry",
                    message=f"Scheduling retry in ~{backoff:.0f}s",
                ),
                "error": error_body,
            }
        )
        # Never requeue past the hard TTL: the worker loop will expire it.
        if requeued.expires_at is None or requeued.expires_at > utc_now():
            await self._repo.update(requeued, expected_status=TaskStatus.RUNNING)
            logger.info(
                "Task re-queued after failure | task_id=%s | attempt=%d/%d",
                record.task_id,
                record.attempts,
                self._max_attempts,
            )

    async def _run_graph(self, record: TaskRecord) -> TaskRecord:
        """Execute the workflow graph and map the outcome to a terminal status.

        The graph is checkpointed under the task's thread_id (P2-06), so a
        crash mid-flight resumes from the last node boundary on retry.
        """
        request = GeneratePlanRequest.model_validate(record.request_payload)
        response: PlanResponse = await self._generation.execute(
            request,
            thread_id=record.thread_id,
        )
        status = _status_for_response(response.status)
        return record.model_copy(
            update={
                "status": status,
                "updated_at": utc_now(),
                "result": json.loads(response.model_dump_json()),
                "progress": TaskProgress(
                    node="done",
                    completed_steps=1,
                    message=_progress_message(status),
                ),
            }
        )


def _status_for_response(business_status: str) -> TaskStatus:
    """Map a PlanResponse status to the task state machine's terminal state."""
    return {
        "READY": TaskStatus.READY,
        "NEEDS_CONFIRMATION": TaskStatus.NEEDS_CONFIRMATION,
        "INFEASIBLE": TaskStatus.INFEASIBLE,
        "FAILED": TaskStatus.FAILED,
    }[business_status]


def _progress_message(status: TaskStatus) -> str:
    return {
        TaskStatus.READY: "Plan generated successfully",
        TaskStatus.NEEDS_CONFIRMATION: "Plan requires confirmation",
        TaskStatus.INFEASIBLE: "No feasible plan under current constraints",
        TaskStatus.FAILED: "Plan generation failed",
    }.get(status, "Task finished")
