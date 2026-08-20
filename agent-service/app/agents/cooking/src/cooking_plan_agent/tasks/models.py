# ============================================================================
# Tasks 领域模型 — 异步任务状态机与持久化任务记录（P3-01）
# ============================================================================

"""Async task domain models (P3-01).

异步任务领域模型（P3-01）。

Defines the task state machine and the persisted TaskRecord that backs the
async task API. Tasks decouple long-running plan generation from the
synchronous HTTP budget: a client submits, polls, optionally subscribes to
SSE progress, cancels, and — after a process restart — resumes from the
P2-06 checkpointer via the task's thread ID.
定义任务状态机与支撑异步任务 API 的持久化 TaskRecord。任务将长时间运行的计划生成与同步 HTTP 预算解耦：客户端提交、轮询、可选地订阅 SSE 进度、取消，并且——在进程重启后——通过任务的线程 ID 从 P2-06 检查点恢复。

State machine (P3-01):
状态机（P3-01）：
    QUEUED -> RUNNING -> {READY, NEEDS_CONFIRMATION, INFEASIBLE, FAILED}
    QUEUED -> CANCELLED | EXPIRED
    RUNNING -> CANCELLED (cooperative) | EXPIRED
    NEEDS_CONFIRMATION -> RUNNING (resumed via ApprovedDecision, new revision)
    NEEDS_CONFIRMATION -> RUNNING（通过 ApprovedDecision 恢复，新版本号）

The request ID is the idempotency key: re-submitting the same payload
returns the same task; a conflicting payload for the same request ID is a
409 error (D1).
请求 ID 是幂等键：重新提交相同载荷返回相同任务；同一请求 ID 的冲突载荷为 409 错误（D1）。
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from cooking_plan_agent.domain.models import StrictModel


class TaskStatus(StrEnum):
    """Lifecycle status of an async cooking-plan task.

    异步烹饪计划任务的生命周期状态。
    """

    QUEUED = "QUEUED"  # accepted, awaiting a worker
    # 已接受，等待工作线程
    RUNNING = "RUNNING"  # a worker holds the lease and is executing the graph
    # 某个工作线程持有租约并正在执行图
    NEEDS_CONFIRMATION = "NEEDS_CONFIRMATION"  # terminal: user input required
    # 终态：需要用户输入
    READY = "READY"  # terminal: verified plan produced
    # 终态：已产出经校验的计划
    INFEASIBLE = "INFEASIBLE"  # terminal: no feasible plan under constraints
    # 终态：在约束下无可行计划
    FAILED = "FAILED"  # terminal: workflow failed with a stable error
    # 终态：工作流以稳定错误失败
    CANCELLED = "CANCELLED"  # terminal: cancelled by the client
    # 终态：被客户端取消
    EXPIRED = "EXPIRED"  # terminal: lease/queue wait exceeded the TTL
    # 终态：租约/排队等待超过 TTL


# ---------------------------------------------------------------------------
# State machine — legal transitions
# 状态机 — 合法迁移
# ---------------------------------------------------------------------------
# _TRANSITIONS maps a source status to the set of target statuses a task may
# move into. Every status update goes through validate_transition so a
# concurrent worker cannot clobber a terminal result (P3-01 / D2).
# _TRANSITIONS 将源状态映射到任务可迁移的目标状态集合。每次状态更新都经过 validate_transition，以便并发工作线程不能覆盖终态结果（P3-01 / D2）。

_TERMINAL_STATUSES = frozenset(
    {
        TaskStatus.NEEDS_CONFIRMATION,
        TaskStatus.READY,
        TaskStatus.INFEASIBLE,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
        TaskStatus.EXPIRED,
    }
)

_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.QUEUED: frozenset({TaskStatus.RUNNING, TaskStatus.CANCELLED, TaskStatus.EXPIRED}),
    TaskStatus.RUNNING: frozenset(
        {
            TaskStatus.READY,
            TaskStatus.NEEDS_CONFIRMATION,
            TaskStatus.INFEASIBLE,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.EXPIRED,
        }
    ),
    TaskStatus.NEEDS_CONFIRMATION: frozenset({TaskStatus.RUNNING}),
    TaskStatus.READY: frozenset(),
    TaskStatus.INFEASIBLE: frozenset(),
    TaskStatus.FAILED: frozenset(),
    TaskStatus.CANCELLED: frozenset(),
    TaskStatus.EXPIRED: frozenset(),
}


def validate_transition(current: TaskStatus, target: TaskStatus) -> bool:
    """Return True when ``current -> target`` is a legal state transition.

    当 ``current -> target`` 是合法状态迁移时返回 True。

    Terminal states are absorbing: once READY/FAILED/etc., the task can
    never move again (protects against double-execution and stale writes).
    终态是吸收态：一旦进入 READY/FAILED 等状态，任务就再也不能移动（防止重复执行与陈旧写入）。
    """
    if current not in _TRANSITIONS:
        return False
    return target in _TRANSITIONS[current]


def is_terminal(status: TaskStatus) -> bool:
    """Return True when the status is terminal (no further transitions).

    当状态为终态时返回 True（无进一步迁移）。
    """
    return status in _TERMINAL_STATUSES


def utc_now() -> datetime:
    """Timezone-aware UTC now, safe for persistence comparison.

    带时区的 UTC 当前时间，可安全用于持久化比较。
    """
    return datetime.now(UTC)


def new_task_id() -> str:
    """Generate a task ID (UUID4 hex — unique, unguessable, sortable).

    生成任务 ID（UUID4 十六进制——唯一、不可猜测、可排序）。
    """
    return uuid4().hex


class TaskProgress(StrictModel):
    """Progress snapshot reported by the worker at node boundaries (P3-01).

    工作线程在节点边界上报的进度快照（P3-01）。

    ``completed_steps`` is a best-effort node counter; the exact graph has
    branchy routing, so clients must treat it as a monotonic hint, not a
    percentage of a fixed total.
    ``completed_steps`` 是尽力而为的节点计数器；实际图存在分支路由，因此客户端必须将其视为单调提示，而不是固定总数的百分比。
    """

    node: str | None = None
    """Name of the node currently executing (None while queued).

    当前正在执行的节点名称（排队时为 None）。
    """

    completed_steps: int = 0
    """Monotonic count of completed workflow nodes for this attempt.

    本次尝试已完成的工作流节点的单调计数。
    """

    message: str | None = None
    """Optional human-readable progress detail (safe, non-sensitive).

    可选的人类可读进度详情（安全、非敏感）。
    """


class TaskRecord(StrictModel):
    """Persisted async task record (P3-01).

    持久化的异步任务记录（P3-01）。

    Stored via the task repository (SQLite in MVP). ``request_payload`` is
    the serialised GeneratePlanRequest so a restarted worker can rebuild the
    exact input without round-tripping through the API. ``result`` carries
    the serialised PlanResponse for terminal success states; ``error``
    carries a stable error for FAILED.
    通过任务仓库存储（MVP 中为 SQLite）。``request_payload`` 是序列化的 GeneratePlanRequest，以便重启后的工作线程无需经过 API 往返即可重建精确输入。``result`` 携带终态成功状态的序列化 PlanResponse；``error`` 携带 FAILED 的稳定错误。
    """

    task_id: str
    """Unique task identifier returned to the client.

    返回给客户端的唯一任务标识。
    """

    request_id: str
    """Idempotency key (D1): same request_id + same payload => same task.

    幂等键（D1）：相同 request_id + 相同载荷 => 相同任务。
    """

    user_id: str
    """Task owner for access isolation (D3).

    任务所有者，用于访问隔离（D3）。
    """

    status: TaskStatus = TaskStatus.QUEUED
    """Current lifecycle status.

    当前生命周期状态。
    """

    request_payload: dict[str, Any]
    """Serialised GeneratePlanRequest for rebuild/audit.

    序列化的 GeneratePlanRequest，用于重建/审计。
    """

    thread_id: str
    """LangGraph checkpoint thread (P2-06) = request_id:plan_revision.

    LangGraph 检查点线程（P2-06）= request_id:plan_revision。
    """

    revision: int = 0
    """Confirmation revision; new revision after NEEDS_CONFIRMATION resume.

    确认版本号；在 NEEDS_CONFIRMATION 恢复后产生新版本号。
    """

    event_id: int = 0
    """Monotonic progress-event counter (P4-04).

    Incremented atomically on every persisted status/progress change so SSE
    subscribers can resume with ``Last-Event-ID`` without replaying old
    events. 0 = task creation; each successful conditional write (claim or
    update) bumps it by exactly one, in the same SQL statement as the
    status change (no duplicate event IDs under concurrent writers).

    单调进度事件计数器（P4-04）。在每次持久化的状态/进度变更时原子地递增，使 SSE 订阅者能够通过 ``Last-Event-ID`` 恢复而无需重放旧事件。0 = 任务创建；每次成功的条件写入（claim 或 update）都恰好将其加一，且与状态变更位于同一条 SQL 语句中（并发写入下不会出现重复事件 ID）。
    """

    progress: TaskProgress = TaskProgress()
    """Latest progress snapshot.

    最新进度快照。
    """

    result: dict[str, Any] | None = None
    """Serialised PlanResponse for READY/NEEDS_CONFIRMATION/INFEASIBLE.

    READY/NEEDS_CONFIRMATION/INFEASIBLE 的序列化 PlanResponse。
    """

    error: dict[str, Any] | None = None
    """Serialised ErrorEnvelope for FAILED.

    FAILED 的序列化 ErrorEnvelope。
    """

    execution_state: dict[str, str] = {}
    """Runtime cooking-task states, persisted for READY plans and recovery.

    运行时烹饪任务状态，为 READY 计划与恢复而持久化。
    """

    created_at: datetime = utc_now()
    """Submission timestamp.

    提交时间戳。
    """

    updated_at: datetime = utc_now()
    """Last state-change timestamp (drives lease/expiry).

    最后状态变更时间戳（驱动租约/过期）。
    """

    expires_at: datetime | None = None
    """Optional hard deadline; EXPIRED when exceeded.

    可选的硬截止时间；超过后置为 EXPIRED。
    """

    # --- Distributed execution (P3-02) ---
    # --- 分布式执行（P3-02） ---
    attempts: int = 0
    """Number of lease claims so far; capped at max_attempts.

    迄今为止的租约认领次数；上限为 max_attempts。
    """

    max_attempts: int = 3
    """Maximum execution attempts before dead-lettering (P3-02).

    死信（dead-letter）之前的最大执行尝试次数（P3-02）。
    """

    lease_expires_at: datetime | None = None
    """Visibility timeout: when a worker's lease expires, another worker may
    re-claim the task (D4). None when not leased.

    可见性超时：当工作线程的租约过期时，另一工作线程可重新认领该任务（D4）。未租出时为 None。
    """

    def location(self, prefix: str = "/internal/v2/cooking-plan/tasks") -> str:
        """Status query address advertised on submission (202 response).

        提交时通告的状态查询地址（202 响应）。
        """
        return f"{prefix}/{self.task_id}"

    def transition(self, target: TaskStatus) -> TaskRecord:
        """Return a new record with the status moved to ``target``.

        返回状态迁移到 ``target`` 的新记录。

        Raises ValueError for an illegal transition — the caller must
        persist the updated record conditionally.
        对非法迁移抛出 ValueError——调用方必须条件性地持久化更新后的记录。
        """
        if not validate_transition(self.status, target):
            raise ValueError(f"Illegal task transition {self.status.value} -> {target.value} (task_id={self.task_id})")
        return self.model_copy(update={"status": target, "updated_at": utc_now()})
