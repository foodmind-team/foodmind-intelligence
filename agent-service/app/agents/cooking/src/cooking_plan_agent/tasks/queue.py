"""Worker-facing task queue port (P4-05 Stage A).

P3-02 (horizontal scaling) requires a worker loop that can consume from a
distributed queue without being tied to the in-process repository. This
module defines the ``TaskQueue`` port the worker consumes, plus the
``InProcessTaskQueue`` adapter that preserves the approved single-process
MVP behaviour unchanged.

The port deliberately mirrors the repository's distributed-execution
primitives — atomic claim + lease (visibility timeout), lease renewal
(heartbeat), and the conditional result write — so a Redis/Postgres-backed
backend (Stage B, pending infrastructure approval) can swap in without
touching the service layer.
"""

from __future__ import annotations

from typing import Protocol

from cooking_plan_agent.tasks.models import TaskRecord, TaskStatus
from cooking_plan_agent.tasks.repository import TaskRepository


class TaskQueue(Protocol):
    """Worker-facing queue port (P4-05 Stage A).

    Backends implement three distributed primitives:

    - ``claim_available``: atomically claim the next runnable task with a
      lease so exactly one worker executes it (D2).
    - ``renew_lease``: extend a held lease (heartbeat); when a worker
      crashes, its lease expires and the task becomes claimable again (D4).
    - ``complete``: conditional result write — persist the record only while
      the stored status still matches ``expected_status``, so duplicate or
      out-of-order messages never produce duplicate business results.

    ``close`` releases queue-side resources; it must NOT close the shared
    task store, which the service owns.
    """

    async def claim_available(self, lease_seconds: float) -> TaskRecord | None:
        """Atomically claim the next runnable task, or None when idle."""
        ...

    async def renew_lease(self, task_id: str, lease_seconds: float) -> TaskRecord | None:
        """Extend a held lease; None once the lease has been lost."""
        ...

    async def complete(self, record: TaskRecord, expected_status: TaskStatus) -> TaskRecord | None:
        """Conditional result write (D2); None when the status moved on."""
        ...

    async def close(self) -> None:
        """Release queue-side resources (not the shared task store)."""
        ...


class InProcessTaskQueue:
    """In-process queue over the shared task repository (P4-05 Stage A).

    The approved MVP path: claim / renew / complete all delegate to the
    repository's atomic conditional writes, so behaviour is byte-for-byte
    identical to the pre-P4-05 worker. A distributed backend (Stage B,
    pending infrastructure approval) replaces this adapter without changing
    the service layer.
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
        return None


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

#: Queue backends available in Stage A. Stage B adds ``redis`` after the
#: queue infrastructure (ADR + approval) lands — until then the factory
#: rejects any other value loudly instead of silently degrading.
SUPPORTED_TASK_QUEUE_BACKENDS = frozenset({"inprocess"})


def create_task_queue(backend: str, repository: TaskRepository) -> TaskQueue:
    """Build the worker queue backend selected by settings (P4-05).

    Stage A ships only the in-process queue. Distributed backends (e.g.
    ``redis``) are Stage B and stay gated on infrastructure approval: an
    unapproved backend value fails startup loudly so a deployment can never
    silently fall back to single-instance semantics.
    """
    if backend not in SUPPORTED_TASK_QUEUE_BACKENDS:
        raise RuntimeError(
            f"Unsupported task_queue_backend={backend!r}: distributed queue backends "
            "require P4-05 Stage B infrastructure approval and are not enabled. "
            f"Supported backends: {sorted(SUPPORTED_TASK_QUEUE_BACKENDS)}"
        )
    return InProcessTaskQueue(repository)
