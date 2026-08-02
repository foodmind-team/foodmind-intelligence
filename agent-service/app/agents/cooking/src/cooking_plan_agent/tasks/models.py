"""Async task domain models (P3-01).

Defines the task state machine and the persisted TaskRecord that backs the
async task API. Tasks decouple long-running plan generation from the
synchronous HTTP budget: a client submits, polls, optionally subscribes to
SSE progress, cancels, and — after a process restart — resumes from the
P2-06 checkpointer via the task's thread ID.

State machine (P3-01):
    QUEUED -> RUNNING -> {READY, NEEDS_CONFIRMATION, INFEASIBLE, FAILED}
    QUEUED -> CANCELLED | EXPIRED
    RUNNING -> CANCELLED (cooperative) | EXPIRED
    NEEDS_CONFIRMATION -> RUNNING (resumed via ApprovedDecision, new revision)

The request ID is the idempotency key: re-submitting the same payload
returns the same task; a conflicting payload for the same request ID is a
409 error (D1).
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from cooking_plan_agent.domain.models import StrictModel


class TaskStatus(StrEnum):
    """Lifecycle status of an async cooking-plan task."""

    QUEUED = "QUEUED"  # accepted, awaiting a worker
    RUNNING = "RUNNING"  # a worker holds the lease and is executing the graph
    NEEDS_CONFIRMATION = "NEEDS_CONFIRMATION"  # terminal: user input required
    READY = "READY"  # terminal: verified plan produced
    INFEASIBLE = "INFEASIBLE"  # terminal: no feasible plan under constraints
    FAILED = "FAILED"  # terminal: workflow failed with a stable error
    CANCELLED = "CANCELLED"  # terminal: cancelled by the client
    EXPIRED = "EXPIRED"  # terminal: lease/queue wait exceeded the TTL


# ---------------------------------------------------------------------------
# State machine — legal transitions
# ---------------------------------------------------------------------------
# _TRANSITIONS maps a source status to the set of target statuses a task may
# move into. Every status update goes through validate_transition so a
# concurrent worker cannot clobber a terminal result (P3-01 / D2).

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

    Terminal states are absorbing: once READY/FAILED/etc., the task can
    never move again (protects against double-execution and stale writes).
    """
    if current not in _TRANSITIONS:
        return False
    return target in _TRANSITIONS[current]


def is_terminal(status: TaskStatus) -> bool:
    """Return True when the status is terminal (no further transitions)."""
    return status in _TERMINAL_STATUSES


def utc_now() -> datetime:
    """Timezone-aware UTC now, safe for persistence comparison."""
    return datetime.now(UTC)


def new_task_id() -> str:
    """Generate a task ID (UUID4 hex — unique, unguessable, sortable)."""
    return uuid4().hex


class TaskProgress(StrictModel):
    """Progress snapshot reported by the worker at node boundaries (P3-01).

    ``completed_steps`` is a best-effort node counter; the exact graph has
    branchy routing, so clients must treat it as a monotonic hint, not a
    percentage of a fixed total.
    """

    node: str | None = None
    """Name of the node currently executing (None while queued)."""

    completed_steps: int = 0
    """Monotonic count of completed workflow nodes for this attempt."""

    message: str | None = None
    """Optional human-readable progress detail (safe, non-sensitive)."""


class TaskRecord(StrictModel):
    """Persisted async task record (P3-01).

    Stored via the task repository (SQLite in MVP). ``request_payload`` is
    the serialised GeneratePlanRequest so a restarted worker can rebuild the
    exact input without round-tripping through the API. ``result`` carries
    the serialised PlanResponse for terminal success states; ``error``
    carries a stable error for FAILED.
    """

    task_id: str
    """Unique task identifier returned to the client."""

    request_id: str
    """Idempotency key (D1): same request_id + same payload => same task."""

    user_id: str
    """Task owner for access isolation (D3)."""

    status: TaskStatus = TaskStatus.QUEUED
    """Current lifecycle status."""

    request_payload: dict[str, Any]
    """Serialised GeneratePlanRequest for rebuild/audit."""

    thread_id: str
    """LangGraph checkpoint thread (P2-06) = request_id:plan_revision."""

    revision: int = 0
    """Confirmation revision; new revision after NEEDS_CONFIRMATION resume."""

    progress: TaskProgress = TaskProgress()
    """Latest progress snapshot."""

    result: dict[str, Any] | None = None
    """Serialised PlanResponse for READY/NEEDS_CONFIRMATION/INFEASIBLE."""

    error: dict[str, Any] | None = None
    """Serialised ErrorEnvelope for FAILED."""

    created_at: datetime = utc_now()
    """Submission timestamp."""

    updated_at: datetime = utc_now()
    """Last state-change timestamp (drives lease/expiry)."""

    expires_at: datetime | None = None
    """Optional hard deadline; EXPIRED when exceeded."""

    # --- Distributed execution (P3-02) ---
    attempts: int = 0
    """Number of lease claims so far; capped at max_attempts."""

    max_attempts: int = 3
    """Maximum execution attempts before dead-lettering (P3-02)."""

    lease_expires_at: datetime | None = None
    """Visibility timeout: when a worker's lease expires, another worker may
    re-claim the task (D4). None when not leased."""

    def location(self, prefix: str = "/internal/v2/cooking-plan/tasks") -> str:
        """Status query address advertised on submission (202 response)."""
        return f"{prefix}/{self.task_id}"

    def transition(self, target: TaskStatus) -> TaskRecord:
        """Return a new record with the status moved to ``target``.

        Raises ValueError for an illegal transition — the caller must
        persist the updated record conditionally.
        """
        if not validate_transition(self.status, target):
            raise ValueError(f"Illegal task transition {self.status.value} -> {target.value} (task_id={self.task_id})")
        return self.model_copy(update={"status": target, "updated_at": utc_now()})
