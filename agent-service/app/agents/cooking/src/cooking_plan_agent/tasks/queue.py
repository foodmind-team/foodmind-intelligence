# ============================================================================
# Tasks 队列端口 — 面向工作线程的任务队列（P4-05 阶段 A）
# ============================================================================

"""Worker-facing task queue port (P4-05 Stage A).

面向工作线程的任务队列端口（P4-05 阶段 A）。

P3-02 (horizontal scaling) requires a worker loop that can consume from a
distributed queue without being tied to the in-process repository. This
module defines the ``TaskQueue`` port the worker consumes, plus the
``InProcessTaskQueue`` adapter that preserves the approved single-process
MVP behaviour unchanged.
P3-02（水平扩展）要求工作循环能够从分布式队列消费，而不受限于进程内仓库。本模块定义了工作线程所消费的 ``TaskQueue`` 端口，以及保持已批准的单进程 MVP 行为不变的 ``InProcessTaskQueue`` 适配器。

The port deliberately mirrors the repository's distributed-execution
primitives — atomic claim + lease (visibility timeout), lease renewal
(heartbeat), and the conditional result write — so a Redis/Postgres-backed
backend (Stage B, pending infrastructure approval) can swap in without
touching the service layer.
该端口有意镜像仓库的分布式执行原语——原子认领 + 租约（可见性超时）、租约续期（心跳）与条件结果写入——以便 Redis/Postgres 后端（阶段 B，待基础设施批准）可以在不改动服务层的情况下替换进来。
"""

from __future__ import annotations

from typing import Protocol

from cooking_plan_agent.tasks.models import TaskRecord, TaskStatus
from cooking_plan_agent.tasks.repository import TaskRepository


class TaskQueue(Protocol):
    """Worker-facing queue port (P4-05 Stage A).

    面向工作线程的队列端口（P4-05 阶段 A）。

    Backends implement three distributed primitives:
    后端实现三个分布式原语：

    - ``claim_available``: atomically claim the next runnable task with a
      lease so exactly one worker executes it (D2).
    - ``claim_available``：原子地认领下一个可运行任务并附带租约，使得恰好一个工作线程执行它（D2）。
    - ``renew_lease``: extend a held lease (heartbeat); when a worker
      crashes, its lease expires and the task becomes claimable again (D4).
    - ``renew_lease``：延长已持有的租约（心跳）；当工作线程崩溃时，其租约过期，任务可被再次认领（D4）。
    - ``complete``: conditional result write — persist the record only while
      the stored status still matches ``expected_status``, so duplicate or
      out-of-order messages never produce duplicate business results.
    - ``complete``：条件结果写入——仅在存储状态仍匹配 ``expected_status`` 时持久化记录，使重复或乱序消息绝不会产生重复的业务结果。

    ``close`` releases queue-side resources; it must NOT close the shared
    task store, which the service owns.
    ``close`` 释放队列侧资源；它绝不能关闭共享的任务存储，后者由服务层拥有。
    """

    async def claim_available(self, lease_seconds: float) -> TaskRecord | None:
        """Atomically claim the next runnable task, or None when idle.

        原子地认领下一个可运行任务；空闲时返回 None。
        """
        ...

    async def renew_lease(self, task_id: str, lease_seconds: float) -> TaskRecord | None:
        """Extend a held lease; None once the lease has been lost.

        延长已持有的租约；租约丢失后返回 None。
        """
        ...

    async def complete(self, record: TaskRecord, expected_status: TaskStatus) -> TaskRecord | None:
        """Conditional result write (D2); None when the status moved on.

        条件结果写入（D2）；状态已变更时返回 None。
        """
        ...

    async def close(self) -> None:
        """Release queue-side resources (not the shared task store).

        释放队列侧资源（而非共享任务存储）。
        """
        ...


class InProcessTaskQueue:
    """In-process queue over the shared task repository (P4-05 Stage A).

    基于共享任务仓库的进程内队列（P4-05 阶段 A）。

    The approved MVP path: claim / renew / complete all delegate to the
    repository's atomic conditional writes, so behaviour is byte-for-byte
    identical to the pre-P4-05 worker. A distributed backend (Stage B,
    pending infrastructure approval) replaces this adapter without changing
    the service layer.
    已批准的 MVP 路径：claim / renew / complete 都委托给仓库的原子条件写入，因此行为与 P4-05 之前的工作线程逐字节一致。分布式后端（阶段 B，待基础设施批准）可在不改动服务层的情况下替换此适配器。
    """

    def __init__(self, repository: TaskRepository) -> None:
        self._repository = repository

    async def claim_available(self, lease_seconds: float) -> TaskRecord | None:
        return await self._repository.claim_available(lease_seconds)

    async def renew_lease(self, task_id: str, lease_seconds: float) -> TaskRecord | None:
        return await self._repository.renew_lease(task_id, lease_seconds)

    async def complete(self, record: TaskRecord, expected_status: TaskStatus) -> TaskRecord | None:
        return await self._repository.update(record, expected_status=expected_status)

    async def close(self) -> None:
        # The queue shares the repository connection; the repository owner
        # (the service / lifespan) closes the store, not the queue.
        # 队列共享仓库连接；仓库的拥有者（服务 / lifespan）负责关闭存储，而非队列。
        return None


# ---------------------------------------------------------------------------
# Backend selection
# 后端选择
# ---------------------------------------------------------------------------

#: Queue backends available in Stage A. Stage B adds ``redis`` after the
#: queue infrastructure (ADR + approval) lands — until then the factory
#: rejects any other value loudly instead of silently degrading.
#: 阶段 A 中可用的队列后端。阶段 B 将在队列基础设施（ADR + 批准）落地后添加 ``redis``——
#: 在此之前，工厂会响亮地拒绝任何其他值，而不是静默降级。
SUPPORTED_TASK_QUEUE_BACKENDS = frozenset({"inprocess"})


def create_task_queue(backend: str, repository: TaskRepository) -> TaskQueue:
    """Build the worker queue backend selected by settings (P4-05).

    构建由设置选定的工作队列后端（P4-05）。

    Stage A ships only the in-process queue. Distributed backends (e.g.
    ``redis``) are Stage B and stay gated on infrastructure approval: an
    unapproved backend value fails startup loudly so a deployment can never
    silently fall back to single-instance semantics.
    阶段 A 仅提供进程内队列。分布式后端（例如 ``redis``）属于阶段 B，且受基础设施批准门控：未获批准的后端值会使启动响亮失败，使部署绝不会静默回退到单实例语义。
    """
    if backend not in SUPPORTED_TASK_QUEUE_BACKENDS:
        raise RuntimeError(
            f"Unsupported task_queue_backend={backend!r}: distributed queue backends "
            "require P4-05 Stage B infrastructure approval and are not enabled. "
            f"Supported backends: {sorted(SUPPORTED_TASK_QUEUE_BACKENDS)}"
        )
    return InProcessTaskQueue(repository)
