# ============================================================================
# Tasks 服务 — 提交、查询、取消与工作线程执行（P3-01）
# ============================================================================

"""Async task service — submission, query, cancel, and worker execution (P3-01).

异步任务服务 — 提交、查询、取消与工作线程执行（P3-01）。

The service decouples long-running plan generation from the synchronous
HTTP budget:
该服务将长时间运行的计划生成与同步 HTTP 预算解耦：
  - ``submit`` enforces the idempotency key (D1): same request_id + same
    payload returns the same task; a conflicting payload is a 409.
  - ``submit`` 强制幂等键（D1）：相同 request_id + 相同载荷返回相同任务；冲突载荷为 409。
  - ``get`` / ``cancel`` / ``resume`` operate on persisted TaskRecords.
  - ``get`` / ``cancel`` / ``resume`` 操作持久化的 TaskRecord。
  - ``run_worker`` is the in-process execution loop: it claims QUEUED tasks
    (conditional QUEUED->RUNNING), runs the workflow graph under the P2-06
    checkpoint thread, and writes the terminal result conditionally.
  - ``run_worker`` 是进程内执行循环：它认领 QUEUED 任务（条件 QUEUED->RUNNING），在 P2-06 检查点线程下运行工作流图，并条件性地写入终态结果。

Process-restart recovery (P3-01): on startup the service reloads QUEUED and
RUNNING tasks and re-queues them; RUNNING tasks are safe to re-run because
results are written conditionally by task ID/revision (D2) and the graph is
idempotent at the checkpoint level.
进程重启恢复（P3-01）：启动时服务重新加载 QUEUED 与 RUNNING 任务并重新入队；RUNNING 任务可安全重跑，因为结果按任务 ID/版本号条件写入（D2），且图在检查点层面是幂等的。

The worker runs in-process for MVP (approved decision). P4-05 Stage A
extracted the worker's claim/lease/conditional-write operations behind a
``TaskQueue`` port (tasks/queue.py); the in-process adapter preserves the
MVP behaviour exactly, and a distributed backend (Stage B, pending
infrastructure approval) can replace it without changing the submit/query
API or the worker-loop structure.
MVP 中工作线程在进程内运行（已批准的决策）。P4-05 阶段 A 将工作线程的认领/租约/条件写入操作提取到 ``TaskQueue`` 端口（tasks/queue.py）之后；进程内适配器精确保留 MVP 行为，分布式后端（阶段 B，待基础设施批准）可在不改动提交/查询 API 或工作循环结构的情况下替换它。
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

_MAX_TASK_RUNTIME_SECONDS = 300
_ONE_DISH_RUNTIME_SECONDS = 180
_PER_EXTRA_DISH_SECONDS = 30


class _TaskRunCancelled(Exception):
    """Internal control flow used when a persisted task is cancelled.

    当持久化任务被取消时使用的内部控制流。
    """


@dataclass(frozen=True)
class SubmitOutcome:
    """Result of a submit call (202 or 409 idempotency conflict).

    提交调用的结果（202 或 409 幂等冲突）。
    """

    task: TaskRecord
    created: bool
    conflict: bool = False


class TaskIdempotencyConflict(Exception):
    """Raised when the same request_id is resubmitted with a different payload.

    当同一 request_id 以不同载荷被重新提交时抛出。
    """

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        super().__init__(f"Idempotency conflict for request_id={request_id}")


class AsyncTaskService:
    """Application service behind the async task API (P3-01, P3-02).

    异步任务 API 背后的应用服务（P3-01、P3-02）。
    """

    def __init__(
        self,
        repository: TaskRepository,
        generation_service: GenerateCookingPlanService,
        *,
        queue: TaskQueue | None = None,
        default_ttl_seconds: int = _MAX_TASK_RUNTIME_SECONDS,
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
        # P4-05 阶段 A：工作线程消费 TaskQueue 端口，而不是直接触及仓库的认领/租约/条件写入原语。
        # 默认进程内队列精确保留 MVP 行为；分布式后端（阶段 B，待基础设施批准）在此注入而不改动服务层。
        self._queue = queue if queue is not None else InProcessTaskQueue(repository)
        self._default_ttl = timedelta(seconds=default_ttl_seconds)
        self._worker_concurrency = max(1, worker_concurrency)
        self._lease_seconds = lease_seconds
        self._max_attempts = max_attempts
        self._worker_task: asyncio.Task[None] | None = None
        self._submission_lock = asyncio.Lock()
        self._active_runs: dict[str, asyncio.Task[PlanResponse]] = {}
        # P4-04: SSE subscription registry — task_id -> subscriber queues.
        # Subscribers are only ever notified after a successful persisted
        # state change (see _notify call sites), never from in-memory state.
        # P4-04：SSE 订阅注册表——task_id -> 订阅者队列。
        # 订阅者仅在成功持久化的状态变更之后才被通知（见 _notify 调用点），绝不基于内存状态。
        self._subscribers: dict[str, set[asyncio.Queue[TaskRecord]]] = {}

    # -- lifecycle ----------------------------------------------------------
    # -- 生命周期 ----------------------------------------------------------

    async def astart(self) -> None:
        """Start the lease-based worker loop (P3-02).

        启动基于租约的工作循环（P3-02）。

        Unlike P3-01's manual re-queue, interrupted RUNNING tasks are picked
        up automatically once their lease expires (D4): no explicit recovery
        pass is needed — the first claim_available that sees an expired lease
        re-claims the task.
        与 P3-01 的手动重新入队不同，中断的 RUNNING 任务在其租约过期后会被自动接续（D4）：无需显式恢复遍历——第一个看到过期租约的 claim_available 会重新认领该任务。
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
        """Stop the worker loop (bounded drain).

        停止工作循环（有界排空）。
        """
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
        # P4-05: release queue-side resources (a no-op for the in-process
        # queue, which shares the repository); the repository is closed last.
        # P4-05：释放队列侧资源（对共享仓库的进程内队列而言是空操作）；仓库最后关闭。
        await self._queue.close()
        await self._repo.close()

    # -- API operations -----------------------------------------------------
    # -- API 操作 -----------------------------------------------------

    async def submit(self, request: GeneratePlanRequest, ttl_seconds: int | None = None) -> SubmitOutcome:
        """Submit a plan-generation task (idempotent on request_id).

        提交计划生成任务（以 request_id 幂等）。

        Same request_id + same payload -> returns the existing task
        (``created=False``). Same request_id + different payload -> raises
        TaskIdempotencyConflict (409). New request -> persisted QUEUED task.
        相同 request_id + 相同载荷 -> 返回已存在的任务（``created=False``）。相同 request_id + 不同载荷 -> 抛出 TaskIdempotencyConflict（409）。新请求 -> 持久化的 QUEUED 任务。
        """
        payload = json.loads(request.model_dump_json())
        async with self._submission_lock:
            existing = await self._repo.get_by_request_id(request.request_id)
            if existing is not None:
                if existing.request_payload != payload:
                    raise TaskIdempotencyConflict(request.request_id)
                return SubmitOutcome(task=existing, created=False)

            # A user can own only one active generation. Cancelling before
            # insertion also frees the in-process LLM coroutine immediately.
            # 一个用户只能拥有一个活跃生成。在插入前取消还可立即释放进程内 LLM 协程。
            for active in await self._repo.list_active_by_user(request.user_id):
                await self.cancel(active.task_id)

            ttl = self._task_ttl(request, ttl_seconds)
            now = utc_now()
            record = TaskRecord(
                task_id=new_task_id(),
                request_id=request.request_id,
                user_id=request.user_id,
                request_payload=payload,
                thread_id=self._thread_id(request),
                created_at=now,
                updated_at=now,
                expires_at=now + ttl,
            )
            try:
                await self._repo.create(record)
            except DuplicateRequestError:
                existing = await self._repo.get_by_request_id(request.request_id)
                if existing is None:
                    raise
                if existing.request_payload != payload:
                    raise TaskIdempotencyConflict(request.request_id) from None
                return SubmitOutcome(task=existing, created=False)

            logger.info(
                "Task submitted | task_id=%s | request_id=%s | deadline_seconds=%d",
                record.task_id,
                record.request_id,
                int(ttl.total_seconds()),
            )
            return SubmitOutcome(task=record, created=True)

    def _task_ttl(self, request: GeneratePlanRequest, requested_seconds: int | None) -> timedelta:
        if requested_seconds is not None:
            return timedelta(seconds=min(requested_seconds, _MAX_TASK_RUNTIME_SECONDS))
        dish_count = max(1, len(request.recipes))
        target = _ONE_DISH_RUNTIME_SECONDS + _PER_EXTRA_DISH_SECONDS * (dish_count - 1)
        configured_max = min(int(self._default_ttl.total_seconds()), _MAX_TASK_RUNTIME_SECONDS)
        return timedelta(seconds=min(target, configured_max))

    async def get(self, task_id: str) -> TaskRecord | None:
        """Fetch a task by ID (404 when absent).

        按 ID 获取任务（不存在时为 404）。
        """
        return await self._repo.get(task_id)

    async def cancel(self, task_id: str) -> TaskRecord | None:
        """Cooperatively cancel a QUEUED/RUNNING task.

        协作式取消一个 QUEUED/RUNNING 任务。

        QUEUED -> CANCELLED immediately. RUNNING -> CANCELLED; the worker
        checks the flag at node boundaries (cooperative cancellation, P3-01)
        and will not persist further results for a cancelled task.
        QUEUED -> 立即 CANCELLED。RUNNING -> CANCELLED；工作线程在节点边界检查该标志（协作式取消，P3-01），并且不会为已取消任务持久化进一步结果。
        """
        record = await self._repo.get(task_id)
        if record is None or record.status not in (TaskStatus.QUEUED, TaskStatus.RUNNING):
            return record

        cancelled = record.model_copy(update={"status": TaskStatus.CANCELLED, "updated_at": utc_now()})
        updated = await self._repo.update(cancelled, expected_status=record.status)
        if updated is not None:
            active_run = self._active_runs.get(task_id)
            if active_run is not None:
                active_run.cancel()
            self._notify(updated)
        return updated or record

    async def execution_snapshot(self, task_id: str) -> tuple[TaskRecord | None, dict[str, object] | None]:
        """Load the dependency-driven cooking state for a READY task.

        加载 READY 任务的依赖驱动烹饪状态。
        """
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

        对 READY 计划应用一次经校验的烹饪任务迁移。

        The event version makes concurrent Android/Web updates safe: a stale
        client receives ``EXECUTION_STATE_CONFLICT`` and reloads the snapshot.
        事件版本使并发的 Android/Web 更新安全：陈旧客户端会收到 ``EXECUTION_STATE_CONFLICT`` 并重新加载快照。
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
    # -- SSE 进度订阅（P4-04） -----------------------------------

    async def subscribe(
        self,
        task_id: str,
        last_event_id: int,
        *,
        keepalive_seconds: float | None = None,
    ) -> AsyncIterator[TaskRecord | None]:
        """Yield live task snapshots for an SSE progress stream (P4-04).

        为 SSE 进度流产出实时任务快照（P4-04）。

        Behaviour:
        行为：
          - A task that does not exist yields nothing (the router maps the
            absence to 404 before opening the stream).
          - 不存在的任务不产出任何内容（路由在打开流之前将不存在映射为 404）。
          - The current snapshot is replayed when ``last_event_id`` is
            older (``Last-Event-ID`` recovery); a terminal task always
            surfaces its final snapshot so a reconnecting client
            immediately learns the task is done.
          - 当 ``last_event_id`` 较旧时重放当前快照（``Last-Event-ID`` 恢复）；终态任务总是呈现其最终快照，使重连客户端立即得知任务已完成。
          - Live worker updates are streamed via the subscription registry.
            ``None`` is yielded every ``keepalive_seconds`` while idle so
            the transport can emit an SSE comment frame; the stream closes
            right after the terminal snapshot.
          - 实时工作线程更新通过订阅注册表流式传输。空闲时每隔 ``keepalive_seconds`` 产出一个 ``None``，使传输层可发出 SSE 注释帧；流在终态快照之后立即关闭。
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
            # 在注册之后再读取，以便在快照读取与注册之间到达的更新，要么被投递到队列，要么被这次权威重读取代——绝不会丢失。
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
                    # 空闲 keepalive 心跳——无状态变更
                    continue
                if record.event_id <= last_event_id:
                    continue  # stale duplicate — never replayed
                    # 陈旧重复——绝不重放
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

        等待下一个快照；keepalive 间隔耗尽时返回 None。

        Uses two explicit tasks so a timeout (or a generator close while the
        subscription is idle) never leaves a pending ``queue.get`` future
        behind. ``put_nowait`` on an unbounded queue cannot race this: an
        item stays in the queue until a fresh ``get`` consumes it.
        使用两个显式任务，使超时（或订阅空闲时生成器关闭）绝不会留下挂起的 ``queue.get`` future。无界队列上的 ``put_nowait`` 无法与此竞争：条目会留在队列中，直到新的 ``get`` 消费它。
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
        # keepalive 间隔已耗尽且无状态变更
        get_task.cancel()
        try:
            await get_task
        except asyncio.CancelledError:
            pass
        return None

    def _register(self, task_id: str, queue: asyncio.Queue[TaskRecord]) -> None:
        """Attach a subscriber queue to a task's fan-out set.

        将订阅者队列附加到任务的扇出集合。
        """
        self._subscribers.setdefault(task_id, set()).add(queue)

    def _unregister(self, task_id: str, queue: asyncio.Queue[TaskRecord]) -> None:
        """Detach a subscriber queue; drop empty task entries.

        分离订阅者队列；丢弃空的任务条目。
        """
        subscribers = self._subscribers.get(task_id)
        if subscribers is None:
            return
        subscribers.discard(queue)
        if not subscribers:
            self._subscribers.pop(task_id, None)

    def _notify(self, record: TaskRecord) -> None:
        """Fan out a persisted snapshot to every subscriber of the task (P4-04).

        将持久化的快照扇出给任务的每个订阅者（P4-04）。

        Called only after a successful conditional write (``update`` /
        ``claim_available``), so subscribers observe the same atomic,
        monotonic event_id sequence that a fresh reader would.
        仅在成功的条件写入（``update`` / ``claim_available``）之后调用，因此订阅者观察到与新读者相同的原子、单调 event_id 序列。
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
    # -- 工作线程 -------------------------------------------------------------

    async def _run_worker_loop(self) -> None:
        """Lease-based worker loop (P3-02).

        基于租约的工作循环（P3-02）。

        Repeatedly: claim the next available task (QUEUED, or RUNNING with an
        expired lease — D4), execute it while a background heartbeat renews
        the lease, then write the terminal result. At most
        ``worker_concurrency`` tasks run simultaneously.
        重复地：认领下一个可用任务（QUEUED，或租约已过期的 RUNNING——D4），在后台心跳续期租约的同时执行它，然后写入终态结果。最多同时运行 ``worker_concurrency`` 个任务。
        """
        while True:
            try:
                await self._expire_stale_tasks()
                # Each iteration claims up to `worker_concurrency` tasks and
                # runs them concurrently; the atomic claim guarantees no two
                # workers execute the same task (D2).
                # 每次迭代认领最多 `worker_concurrency` 个任务并并发运行；原子认领保证没有两个工作线程执行同一任务（D2）。
                tasks = [asyncio.create_task(self._execute_claimed()) for _ in range(self._worker_concurrency)]
                for task in tasks:
                    try:
                        await task
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:  # noqa: BLE001 — worker must stay alive
                        # 工作线程必须保持存活
                        logger.error("Task execution raised", extra={"error": str(exc)})
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — worker loop must stay alive
                # 工作循环必须保持存活
                logger.error("Task worker iteration failed", extra={"error": str(exc)})
            await asyncio.sleep(0.3)

    async def _expire_stale_tasks(self) -> None:
        """Move QUEUED/RUNNING tasks past their TTL to EXPIRED (P3-01).

        将超过 TTL 的 QUEUED/RUNNING 任务迁移到 EXPIRED（P3-01）。

        Expiry is conditional on the observed status so a task that
        completed between the list and the update is never clobbered.
        过期以观察到的状态为条件，因此在列表与更新之间完成的任务绝不会被覆盖。
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
                active_run = self._active_runs.get(record.task_id)
                if active_run is not None:
                    active_run.cancel()
                self._notify(updated)
                logger.info(
                    "Task expired | task_id=%s | from=%s",
                    record.task_id,
                    record.status.value,
                )

    async def _execute_claimed(self) -> None:
        """Claim one available task and run it to a terminal state (P3-02).

        认领一个可用任务并将其运行到终态（P3-02）。

        The atomic claim sets RUNNING + lease + attempts+1. While the graph
        runs, a heartbeat task renews the lease; if this worker dies, the
        lease expires and another worker re-claims the task (D4).
        原子认领设置 RUNNING + 租约 + attempts+1。图运行期间，心跳任务续期租约；若此工作线程死亡，租约过期后另一工作线程重新认领该任务（D4）。

        On failure: if attempts still allow retries the task is re-queued
        (QUEUED, backoff window), otherwise it is dead-lettered as FAILED
        (P3-02). Results are written conditionally on RUNNING (D2).
        失败时：若 attempts 仍允许重试，任务被重新入队（QUEUED，退避窗口），否则死信为 FAILED（P3-02）。结果以 RUNNING 为条件写入（D2）。
        """
        record = await self._queue.claim_available(self._lease_seconds)
        if record is None:
            return
        self._notify(record)  # P4-04: QUEUED -> RUNNING is a progress event
        # P4-04：QUEUED -> RUNNING 是进度事件

        heartbeat = asyncio.create_task(self._renew_heartbeat(record.task_id))
        try:
            terminal = await self._run_graph(record)
            updated = await self._queue.complete(terminal, expected_status=TaskStatus.RUNNING)
            if updated is not None:
                self._notify(updated)  # P4-04: terminal snapshot -> done event
                # P4-04：终态快照 -> 完成事件
        except _TaskRunCancelled:
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — failure handling below
            # 失败处理见下方
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

        周期性地续期任务租约，直到被取消（P3-02 D4）。

        Renewal is conditional: if the lease was lost (another worker
        re-claimed after expiry), renew_lease returns None and the heartbeat
        stops quietly.
        续期是条件性的：若租约已丢失（过期后另一工作线程重新认领），renew_lease 返回 None，心跳静默停止。
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
            # 心跳失败绝不能使工作线程崩溃
            logger.warning(
                "Lease renewal failed | task_id=%s | error=%s",
                task_id,
                exc,
            )

    async def _handle_retry_or_dead_letter(self, record: TaskRecord, exc: Exception) -> None:
        """Retry (re-queue) or dead-letter the task after a failure (P3-02).

        失败后重试（重新入队）或死信该任务（P3-02）。

        ``attempts`` was already incremented by the atomic claim. When the
        attempt budget is exhausted the task becomes FAILED with a stable
        error (dead-letter); otherwise it is re-queued with an exponential
        backoff so a flapping task cannot busy-loop.
        ``attempts`` 已由原子认领递增。当尝试预算耗尽时，任务以稳定错误变为 FAILED（死信）；否则以指数退避重新入队，使抖动任务不能忙循环。
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
                # P4-04：死信终态 -> 完成事件
            logger.warning(
                "Task dead-lettered | task_id=%s | attempts=%d",
                record.task_id,
                record.attempts,
            )
            return
        # Re-queue with a short backoff so a flapping task cannot busy-loop.
        # The hard TTL (expires_at) is never extended by a retry.
        # 以短退避重新入队，使抖动任务不能忙循环。硬 TTL（expires_at）绝不被重试延长。
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
        # 绝不在硬 TTL 之后重新入队：工作循环会将其过期。
        if requeued.expires_at is None or requeued.expires_at > utc_now():
            updated = await self._queue.complete(requeued, expected_status=TaskStatus.RUNNING)
            if updated is not None:
                self._notify(updated)  # P4-04: re-queue progress event
                # P4-04：重新入队进度事件
            logger.info(
                "Task re-queued after failure | task_id=%s | attempt=%d/%d",
                record.task_id,
                record.attempts,
                self._max_attempts,
            )

    async def _run_graph(self, record: TaskRecord) -> TaskRecord:
        """Execute the workflow graph and map the outcome to a terminal status.

        执行工作流图并将结果映射为终态。

        The graph is checkpointed under the task's thread_id (P2-06), so a
        crash mid-flight resumes from the last node boundary on retry.
        图在任务的 thread_id 下被检查点化（P2-06），因此中途崩溃会在重试时从最后一个节点边界恢复。
        """
        request = GeneratePlanRequest.model_validate(record.request_payload)

        async def _persist_progress(node: str, completed_steps: int) -> None:
            current = await self._repo.get(record.task_id)
            if current is None or current.status != TaskStatus.RUNNING:
                raise _TaskRunCancelled
            updated = await self._repo.update_progress(
                record.task_id,
                TaskProgress(node=node, completed_steps=completed_steps),
            )
            if updated is None:
                raise _TaskRunCancelled
            self._notify(updated)

        streaming_execute = getattr(self._generation, "execute_with_progress", None)
        if callable(streaming_execute):
            coroutine = streaming_execute(
                request,
                thread_id=record.thread_id,
                on_progress=_persist_progress,
            )
        else:
            coroutine = self._generation.execute(request, thread_id=record.thread_id)

        run_task = asyncio.create_task(coroutine)
        self._active_runs[record.task_id] = run_task
        remaining_seconds = (
            max(0.0, (record.expires_at - utc_now()).total_seconds())
            if record.expires_at is not None
            else float(_MAX_TASK_RUNTIME_SECONDS)
        )
        try:
            response: PlanResponse = await asyncio.wait_for(run_task, timeout=remaining_seconds)
        except TimeoutError:
            return record.model_copy(
                update={
                    "status": TaskStatus.EXPIRED,
                    "updated_at": utc_now(),
                    "progress": TaskProgress(
                        node="expired",
                        message="Plan generation reached its time limit",
                    ),
                }
            )
        except asyncio.CancelledError:
            current = await self._repo.get(record.task_id)
            if current is not None and current.status in (TaskStatus.CANCELLED, TaskStatus.EXPIRED):
                raise _TaskRunCancelled from None
            raise
        finally:
            self._active_runs.pop(record.task_id, None)

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
    """Map a PlanResponse status to the task state machine's terminal state.

    将 PlanResponse 状态映射为任务状态机的终态。
    """
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
