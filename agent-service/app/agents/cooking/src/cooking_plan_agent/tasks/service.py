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

The worker runs in-process for MVP (approved decision). P4-05 Stage A
extracted the worker's claim/lease/conditional-write operations behind a
``TaskQueue`` port (tasks/queue.py); the in-process adapter preserves the
MVP behaviour exactly, and a distributed backend (Stage B, pending
infrastructure approval) can replace it without changing the submit/query
API or the worker-loop structure.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import timedelta

from cooking_plan_agent.application.service import GenerateCookingPlanService
from cooking_plan_agent.domain.models import GeneratePlanRequest, PlanResponse
from cooking_plan_agent.tasks.models import (
    TaskProgress,
    TaskRecord,
    TaskStatus,
    is_terminal,
    new_task_id,
    utc_now,
)
from cooking_plan_agent.tasks.queue import InProcessTaskQueue, TaskQueue
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
        queue: TaskQueue | None = None,
        default_ttl_seconds: int = 3600,
        worker_concurrency: int = 2,
        lease_seconds: float = 60.0,
        max_attempts: int = 3,
    ) -> None:
        self._repo = repository
        self._generation = generation_service
        # P4-05 Stage A: the worker consumes a TaskQueue port instead of
        # touching the repository's claim/lease/conditional-write primitives
        # directly. The default in-process queue preserves the MVP behaviour
        # exactly; a distributed backend (Stage B, pending infrastructure
        # approval) is injected here without changing the service layer.
        self._queue = queue if queue is not None else InProcessTaskQueue(repository)
        self._default_ttl = timedelta(seconds=default_ttl_seconds)
        self._worker_concurrency = max(1, worker_concurrency)
        self._lease_seconds = lease_seconds
        self._max_attempts = max_attempts
        self._worker_task: asyncio.Task[None] | None = None
        # P4-04: SSE subscription registry — task_id -> subscriber queues.
        # Subscribers are only ever notified after a successful persisted
        # state change (see _notify call sites), never from in-memory state.
        self._subscribers: dict[str, set[asyncio.Queue[TaskRecord]]] = {}

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
        # P4-05: release queue-side resources (a no-op for the in-process
        # queue, which shares the repository); the repository is closed last.
        await self._queue.close()
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
        if updated is not None:
            self._notify(updated)
        return updated or record

    async def execution_snapshot(self, task_id: str) -> tuple[TaskRecord | None, dict[str, object] | None]:
        """Load the dependency-driven cooking state for a READY task."""
        from cooking_plan_agent.execution import build_execution_snapshot

        record = await self._repo.get(task_id)
        if record is None or record.status != TaskStatus.READY or not isinstance(record.result, dict):
            return record, None
        return (
            record,
            build_execution_snapshot(
                record.result.get("execution_flow"),
                record.execution_state,
                record.request_payload.get("kitchen_resources", ()),
            ),
        )

    async def update_execution(
        self,
        task_id: str,
        cooking_task_id: str,
        target_status: str,
        expected_event_id: int,
    ) -> tuple[TaskRecord | None, dict[str, object] | None, str | None]:
        """Apply one validated cooking-task transition to a READY plan.

        The event version makes concurrent Android/Web updates safe: a stale
        client receives ``EXECUTION_STATE_CONFLICT`` and reloads the snapshot.
        """
        from cooking_plan_agent.execution import ExecutionStateError, transition_execution_state

        record = await self._repo.get(task_id)
        if record is None or record.status != TaskStatus.READY or not isinstance(record.result, dict):
            return record, None, "PLAN_NOT_READY_FOR_EXECUTION"
        if record.event_id != expected_event_id:
            return record, None, "EXECUTION_STATE_CONFLICT"
        try:
            next_state, snapshot = transition_execution_state(
                record.result.get("execution_flow"),
                record.execution_state,
                record.request_payload.get("kitchen_resources", ()),
                cooking_task_id,
                target_status,
            )
        except ExecutionStateError as exc:
            return record, None, exc.code
        updated = await self._repo.update_execution_state(
            record.model_copy(update={"execution_state": next_state, "updated_at": utc_now()}),
            expected_event_id=expected_event_id,
        )
        if updated is None:
            latest, latest_snapshot = await self.execution_snapshot(task_id)
            return latest, latest_snapshot, "EXECUTION_STATE_CONFLICT"
        self._notify(updated)
        return updated, snapshot, None

    # -- SSE progress subscription (P4-04) -----------------------------------

    async def subscribe(
        self,
        task_id: str,
        last_event_id: int,
        *,
        keepalive_seconds: float | None = None,
    ) -> AsyncIterator[TaskRecord | None]:
        """Yield live task snapshots for an SSE progress stream (P4-04).

        Behaviour:
          - A task that does not exist yields nothing (the router maps the
            absence to 404 before opening the stream).
          - The current snapshot is replayed when ``last_event_id`` is
            older (``Last-Event-ID`` recovery); a terminal task always
            surfaces its final snapshot so a reconnecting client
            immediately learns the task is done.
          - Live worker updates are streamed via the subscription registry.
            ``None`` is yielded every ``keepalive_seconds`` while idle so
            the transport can emit an SSE comment frame; the stream closes
            right after the terminal snapshot.
        """
        snapshot = await self._repo.get(task_id)
        if snapshot is None:
            return

        queue: asyncio.Queue[TaskRecord] = asyncio.Queue()
        self._register(task_id, queue)
        try:
            # Re-read AFTER registering so an update that lands between the
            # snapshot read and registration is either delivered to the queue
            # or superseded by this authoritative re-read — never lost.
            current = await self._repo.get(task_id)
            if current is None:
                return
            terminal = is_terminal(current.status)
            if terminal or current.event_id > last_event_id:
                yield current
                last_event_id = current.event_id
            if terminal:
                return

            while True:
                record = await self._next_event(queue, keepalive_seconds)
                if record is None:
                    yield None  # idle keepalive tick — no state change
                    continue
                if record.event_id <= last_event_id:
                    continue  # stale duplicate — never replayed
                yield record
                last_event_id = record.event_id
                if is_terminal(record.status):
                    return
        finally:
            self._unregister(task_id, queue)

    @staticmethod
    async def _next_event(
        queue: asyncio.Queue[TaskRecord],
        keepalive_seconds: float | None,
    ) -> TaskRecord | None:
        """Await the next snapshot, or None when the keepalive interval elapses.

        Uses two explicit tasks so a timeout (or a generator close while the
        subscription is idle) never leaves a pending ``queue.get`` future
        behind. ``put_nowait`` on an unbounded queue cannot race this: an
        item stays in the queue until a fresh ``get`` consumes it.
        """
        if keepalive_seconds is None:
            return await queue.get()
        get_task = asyncio.create_task(queue.get())
        sleep_task = asyncio.create_task(asyncio.sleep(keepalive_seconds))
        try:
            await asyncio.wait({get_task, sleep_task}, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            get_task.cancel()
            sleep_task.cancel()
            raise
        if get_task.done():
            sleep_task.cancel()
            try:
                await sleep_task
            except asyncio.CancelledError:
                pass
            return get_task.result()
        # keepalive interval elapsed with no state change
        get_task.cancel()
        try:
            await get_task
        except asyncio.CancelledError:
            pass
        return None

    def _register(self, task_id: str, queue: asyncio.Queue[TaskRecord]) -> None:
        """Attach a subscriber queue to a task's fan-out set."""
        self._subscribers.setdefault(task_id, set()).add(queue)

    def _unregister(self, task_id: str, queue: asyncio.Queue[TaskRecord]) -> None:
        """Detach a subscriber queue; drop empty task entries."""
        subscribers = self._subscribers.get(task_id)
        if subscribers is None:
            return
        subscribers.discard(queue)
        if not subscribers:
            self._subscribers.pop(task_id, None)

    def _notify(self, record: TaskRecord) -> None:
        """Fan out a persisted snapshot to every subscriber of the task (P4-04).

        Called only after a successful conditional write (``update`` /
        ``claim_available``), so subscribers observe the same atomic,
        monotonic event_id sequence that a fresh reader would.
        """
        subscribers = self._subscribers.get(record.task_id)
        if not subscribers:
            return
        for queue in tuple(subscribers):
            queue.put_nowait(record)

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
                # SQLite uses one connection for the MVP repository. Claim a
                # batch sequentially before running records concurrently so a
                # second BEGIN IMMEDIATE cannot race the first task's initial
                # progress write on that shared connection.
                claimed: list[TaskRecord] = []
                for _ in range(self._worker_concurrency):
                    record = await self._queue.claim_available(self._lease_seconds)
                    if record is None:
                        break
                    self._notify(record)
                    claimed.append(record)
                tasks = [asyncio.create_task(self._execute_record(record)) for record in claimed]
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
            updated = await self._queue.complete(expired, expected_status=record.status)
            if updated is not None:
                self._notify(updated)
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
        record = await self._queue.claim_available(self._lease_seconds)
        if record is None:
            return
        self._notify(record)  # P4-04: QUEUED -> RUNNING is a progress event

        await self._execute_record(record)

    async def _execute_record(self, record: TaskRecord) -> None:
        """Execute one already-claimed record and conditionally finish it."""

        heartbeat = asyncio.create_task(self._renew_heartbeat(record.task_id))
        try:
            terminal = await self._run_graph(record)
            updated = await self._queue.complete(terminal, expected_status=TaskStatus.RUNNING)
            if updated is not None:
                self._notify(updated)  # P4-04: terminal snapshot -> done event
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
                renewed = await self._queue.renew_lease(task_id, self._lease_seconds)
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
            updated = await self._queue.complete(dead, expected_status=TaskStatus.RUNNING)
            if updated is not None:
                self._notify(updated)  # P4-04: dead-letter terminal -> done event
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
            updated = await self._queue.complete(requeued, expected_status=TaskStatus.RUNNING)
            if updated is not None:
                self._notify(updated)  # P4-04: re-queue progress event
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

        async def _report_progress(node: str, completed_steps: int) -> None:
            # Re-read before every write so the heartbeat's latest lease and
            # any concurrent cancellation are preserved. A terminal/cancelled
            # task simply stops accepting progress updates.
            current = await self._repo.get(record.task_id)
            if current is None or current.status != TaskStatus.RUNNING:
                logger.debug(
                    "Progress update skipped | task_id=%s | node=%s | status=%s",
                    record.task_id,
                    node,
                    current.status.value if current is not None else "missing",
                )
                return
            progressed = current.model_copy(
                update={
                    "updated_at": utc_now(),
                    "progress": TaskProgress(
                        node=node,
                        completed_steps=completed_steps,
                        message=_public_node_message(node),
                    ),
                }
            )
            updated = await self._repo.update(progressed, expected_status=TaskStatus.RUNNING)
            if updated is not None:
                self._notify(updated)
                logger.info(
                    "Task progress | task_id=%s | node=%s | completed_steps=%d",
                    record.task_id,
                    node,
                    completed_steps,
                )

        # Publish an immediate stage before entering LangGraph so even a
        # provider that delays its first stream event never looks idle.
        await _report_progress("validate_input", 0)
        response: PlanResponse = await self._generation.execute(
            request,
            thread_id=record.thread_id,
            progress_callback=_report_progress,
        )
        status = _status_for_response(response.status)
        latest = await self._repo.get(record.task_id)
        base = latest if latest is not None and latest.status == TaskStatus.RUNNING else record
        return base.model_copy(
            update={
                "status": status,
                "updated_at": utc_now(),
                "result": json.loads(response.model_dump_json()),
                "progress": TaskProgress(
                    node="done",
                    completed_steps=base.progress.completed_steps + 1,
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


def _public_node_message(node: str) -> str:
    """Return safe workflow status text without exposing model reasoning."""
    messages = {
        "validate_input": "Checking your recipes and preferences…",
        "parse_recipes": "Structuring ingredients and instructions…",
        "detect_gaps": "Checking recipe details…",
        "infer_local": "Completing recipe details…",
        "research_missing": "Checking missing cooking details…",
        "apply_research_evidence": "Applying verified cooking details…",
        "validate_recipe_ir": "Validating recipe structure…",
        "validate_safety": "Checking food-safety requirements…",
        "check_feasibility": "Checking inventory and kitchen resources…",
        "build_confirmation_response": "Preparing a question for you…",
        "merge_preparation": "Organising ingredient preparation…",
        "build_task_graph": "Building the cooking workflow…",
        "solve_schedule": "Optimising the cooking order…",
        "verify_schedule": "Verifying timing and kitchen constraints…",
        "repair_schedule": "Adjusting the cooking schedule…",
        "agent_controller": "Reviewing the plan…",
        "run_tool": "Checking a planning detail…",
        "apply_confirmation": "Applying your answers…",
        "explain_schedule": "Writing a concise plan summary…",
        "render_ready_response": "Finishing your cooking plan…",
        "render_infeasible_response": "Preparing planning feedback…",
        "render_failed_response": "Preparing an error summary…",
    }
    return messages.get(node, "Building your cooking plan…")
