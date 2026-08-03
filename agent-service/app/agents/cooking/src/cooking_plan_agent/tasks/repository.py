"""Task repository — persistence for async tasks (P3-01, P3-02).

MVP uses SQLite (consistent with the approved local-deployment decision).
The repository interface is kept minimal so a Postgres/Redis-backed
implementation can replace it in P3-02 without touching the service layer.

Idempotency (D1): ``create`` is the atomic insert point. A duplicate
request_id returns the existing record; the caller compares payloads to
detect conflicts. Result writes are conditional on the expected status so a
concurrent worker cannot overwrite a newer revision (D2).

Distributed execution (P3-02): ``claim_available`` atomically claims a
QUEUED task — or a RUNNING task whose lease has expired — and moves it to
RUNNING with a new lease. ``renew_lease`` extends a live lease. Workers
renew periodically (D4); a crashed worker's lease expires and another
worker re-claims the task.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from cooking_plan_agent.tasks.models import TaskRecord, TaskStatus, utc_now


def _row_to_record(row: tuple[Any, ...]) -> TaskRecord:
    """Rehydrate a TaskRecord from a SQLite row.

    Column order matches the CREATE TABLE below. JSON columns are decoded;
    None (missing result/error) is preserved.
    """
    (
        _task_id,
        request_id,
        user_id,
        status,
        request_payload_json,
        thread_id,
        revision,
        event_id,
        progress_json,
        result_json,
        error_json,
        execution_state_json,
        created_at_iso,
        updated_at_iso,
        expires_at_iso,
        attempts,
        max_attempts,
        lease_expires_at_iso,
    ) = row
    return TaskRecord(
        task_id=_task_id,
        request_id=request_id,
        user_id=user_id,
        status=TaskStatus(status),
        request_payload=json.loads(request_payload_json),
        thread_id=thread_id,
        revision=revision,
        event_id=event_id,
        progress=_progress_from_json(progress_json),
        result=json.loads(result_json) if result_json else None,
        error=json.loads(error_json) if error_json else None,
        execution_state=json.loads(execution_state_json) if execution_state_json else {},
        created_at=utc_now().fromisoformat(created_at_iso),
        updated_at=utc_now().fromisoformat(updated_at_iso),
        expires_at=utc_now().fromisoformat(expires_at_iso) if expires_at_iso else None,
        attempts=attempts,
        max_attempts=max_attempts,
        lease_expires_at=utc_now().fromisoformat(lease_expires_at_iso) if lease_expires_at_iso else None,
    )


def _progress_from_json(raw: str) -> Any:
    """Deserialise TaskProgress; fall back to an empty snapshot on bad data."""
    from cooking_plan_agent.tasks.models import TaskProgress

    try:
        return TaskProgress.model_validate(json.loads(raw))
    except Exception:  # noqa: BLE001 — defensive rehydration must not crash
        return TaskProgress()


class TaskRepository(ABC):
    """Persistence port for async tasks (P3-01)."""

    @abstractmethod
    async def create(self, record: TaskRecord) -> TaskRecord:
        """Insert a task. Returns the record on first insert.

        Raises DuplicateRequestError when the request_id already exists —
        the caller should load the existing record and compare payloads.
        """

    @abstractmethod
    async def get(self, task_id: str) -> TaskRecord | None:
        """Load a task by ID, or None."""

    @abstractmethod
    async def get_by_request_id(self, request_id: str) -> TaskRecord | None:
        """Load a task by idempotency key, or None."""

    @abstractmethod
    async def update(self, record: TaskRecord, expected_status: TaskStatus) -> TaskRecord | None:
        """Persist the record only when the stored status is ``expected_status``.

        Returns the stored row's rehydrated record when the conditional
        write succeeded, or None when the status had already moved on — the
        caller must reload and decide (D2 conditional result write).
        """

    @abstractmethod
    async def update_execution_state(
        self,
        record: TaskRecord,
        expected_event_id: int,
    ) -> TaskRecord | None:
        """Persist execution state with optimistic event-version control."""

    @abstractmethod
    async def list_running(self) -> list[TaskRecord]:
        """Return QUEUED/RUNNING tasks (recovery after restart, P3-01)."""

    @abstractmethod
    async def claim_available(self, lease_seconds: float) -> TaskRecord | None:
        """Atomically claim the next runnable task (P3-02).

        Claims a QUEUED task, or a RUNNING task whose lease has expired
        (worker crashed — D4). Moves it to RUNNING with a fresh lease and
        increments attempts. Returns None when nothing is claimable.
        """

    @abstractmethod
    async def renew_lease(self, task_id: str, lease_seconds: float) -> TaskRecord | None:
        """Extend the lease of a RUNNING task; None if no longer leased (P3-02)."""

    @abstractmethod
    async def close(self) -> None:
        """Close the underlying connection."""


class DuplicateRequestError(Exception):
    """Raised when a create collides on the idempotency key (request_id)."""


class SQLiteTaskRepository(TaskRepository):
    """SQLite-backed task repository (approved MVP storage).

    Single connection with a write lock via ``BEGIN IMMEDIATE`` is enough
    for the MVP process-internal worker; the schema and queries are designed
    to port to Postgres unchanged in P3-02.
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS cooking_tasks (
        task_id           TEXT PRIMARY KEY,
        request_id        TEXT NOT NULL UNIQUE,
        user_id           TEXT NOT NULL,
        status            TEXT NOT NULL,
        request_payload   TEXT NOT NULL,
        thread_id         TEXT NOT NULL,
        revision          INTEGER NOT NULL DEFAULT 0,
        event_id          INTEGER NOT NULL DEFAULT 0,
        progress          TEXT NOT NULL DEFAULT '{}',
        result            TEXT,
        error             TEXT,
        execution_state   TEXT NOT NULL DEFAULT '{}',
        created_at        TEXT NOT NULL,
        updated_at        TEXT NOT NULL,
        expires_at        TEXT,
        attempts          INTEGER NOT NULL DEFAULT 0,
        max_attempts      INTEGER NOT NULL DEFAULT 3,
        lease_expires_at  TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_cooking_tasks_status ON cooking_tasks (status);
    CREATE INDEX IF NOT EXISTS idx_cooking_tasks_user ON cooking_tasks (user_id);
    """

    # Columns introduced after the initial table definition. ``CREATE TABLE
    # IF NOT EXISTS`` never alters an existing table, so a database created
    # by an earlier release gets the missing columns via a lightweight ALTER
    # TABLE in astart(); fresh databases already include them in _SCHEMA.
    _ADDED_COLUMNS: dict[str, str] = {
        "attempts": "attempts INTEGER NOT NULL DEFAULT 0",
        "max_attempts": "max_attempts INTEGER NOT NULL DEFAULT 3",
        "lease_expires_at": "lease_expires_at TEXT",
        "event_id": "event_id INTEGER NOT NULL DEFAULT 0",
        "execution_state": "execution_state TEXT NOT NULL DEFAULT '{}'",
    }

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn: Any | None = None

    @property
    def conn(self) -> Any:
        """Lazily-open aiosqlite connection (started in astart())."""
        if self._conn is None:
            raise RuntimeError("Task repository not started — call astart() first")
        return self._conn

    async def astart(self) -> None:
        """Open the SQLite connection, create the schema, and migrate columns."""
        import aiosqlite

        if self._conn is None:
            self._conn = await aiosqlite.connect(self._db_path)
            await self._conn.executescript(self._SCHEMA)
            await self._migrate_columns()
            await self._conn.commit()

    async def _migrate_columns(self) -> None:
        """Add columns introduced after the initial schema to existing tables."""
        cursor = await self.conn.execute("PRAGMA table_info(cooking_tasks)")
        rows = await cursor.fetchall()
        await cursor.close()
        existing = {row[1] for row in rows}
        for column, ddl in self._ADDED_COLUMNS.items():
            if column not in existing:
                await self.conn.execute(f"ALTER TABLE cooking_tasks ADD COLUMN {ddl}")

    async def close(self) -> None:
        conn = self._conn
        self._conn = None
        if conn is not None:
            await conn.close()

    # -- helpers -----------------------------------------------------------

    def _columns(self, record: TaskRecord) -> tuple[Any, ...]:
        return (
            record.task_id,
            record.request_id,
            record.user_id,
            record.status.value,
            json.dumps(record.request_payload, default=str),
            record.thread_id,
            record.revision,
            record.event_id,
            record.progress.model_dump_json(),
            json.dumps(record.result, default=str) if record.result else None,
            json.dumps(record.error, default=str) if record.error else None,
            json.dumps(record.execution_state, default=str),
            record.created_at.isoformat(),
            record.updated_at.isoformat(),
            record.expires_at.isoformat() if record.expires_at else None,
            record.attempts,
            record.max_attempts,
            record.lease_expires_at.isoformat() if record.lease_expires_at else None,
        )

    _SELECT = """
        SELECT task_id, request_id, user_id, status, request_payload,
               thread_id, revision, event_id, progress, result, error, execution_state,
               created_at, updated_at, expires_at,
               attempts, max_attempts, lease_expires_at
        FROM cooking_tasks
    """

    async def _fetch_one(self, where: str, params: tuple[Any, ...]) -> TaskRecord | None:
        cursor = await self.conn.execute(f"{self._SELECT} WHERE {where}", params)
        row = await cursor.fetchone()
        await cursor.close()
        return _row_to_record(row) if row else None

    # -- port implementation -----------------------------------------------

    async def create(self, record: TaskRecord) -> TaskRecord:
        existing = await self._fetch_one("request_id = ?", (record.request_id,))
        if existing is not None:
            raise DuplicateRequestError(record.request_id)
        cols = (
            "task_id, request_id, user_id, status, request_payload, thread_id, revision, event_id, progress, "
            "result, error, execution_state, created_at, updated_at, expires_at, attempts, max_attempts, lease_expires_at"
        )
        placeholders = ", ".join("?" for _ in range(18))
        await self.conn.execute(
            f"INSERT INTO cooking_tasks ({cols}) VALUES ({placeholders})",
            self._columns(record),
        )
        await self.conn.commit()
        return record

    async def get(self, task_id: str) -> TaskRecord | None:
        return await self._fetch_one("task_id = ?", (task_id,))

    async def get_by_request_id(self, request_id: str) -> TaskRecord | None:
        return await self._fetch_one("request_id = ?", (request_id,))

    async def update(self, record: TaskRecord, expected_status: TaskStatus) -> TaskRecord | None:
        # Conditional update keyed on the expected status (D2): the stored
        # row only moves when it is still in the state the caller observed,
        # so concurrent workers cannot clobber a newer revision.
        # event_id is bumped in the same statement (P4-04): the increment
        # happens atomically with the conditional write, so a stale writer
        # that loses the race never produces a duplicate event ID.
        cursor = await self.conn.execute(
            """
            UPDATE cooking_tasks
            SET status=?, request_payload=?, thread_id=?, revision=?,
                event_id = event_id + 1,
                progress=?, result=?, error=?, execution_state=?, updated_at=?, expires_at=?,
                attempts=?, max_attempts=?, lease_expires_at=?
            WHERE task_id=? AND status=?
            """,
            (
                record.status.value,
                json.dumps(record.request_payload, default=str),
                record.thread_id,
                record.revision,
                record.progress.model_dump_json(),
                json.dumps(record.result, default=str) if record.result else None,
                json.dumps(record.error, default=str) if record.error else None,
                json.dumps(record.execution_state, default=str),
                record.updated_at.isoformat(),
                record.expires_at.isoformat() if record.expires_at else None,
                record.attempts,
                record.max_attempts,
                record.lease_expires_at.isoformat() if record.lease_expires_at else None,
                record.task_id,
                expected_status.value,
            ),
        )
        await self.conn.commit()
        changed = cursor.rowcount
        await cursor.close()
        if not changed:
            return None
        # Re-read to return the authoritative stored row.
        return await self.get(record.task_id)

    async def update_execution_state(self, record: TaskRecord, expected_event_id: int) -> TaskRecord | None:
        """Atomically persist a READY plan's cooking progress.

        ``event_id`` guards concurrent mobile/web updates; clients that lose
        the race reload the latest snapshot rather than overwriting progress.
        """
        cursor = await self.conn.execute(
            """
            UPDATE cooking_tasks
            SET execution_state=?, updated_at=?, event_id = event_id + 1
            WHERE task_id=? AND status='READY' AND event_id=?
            """,
            (
                json.dumps(record.execution_state, default=str),
                record.updated_at.isoformat(),
                record.task_id,
                expected_event_id,
            ),
        )
        await self.conn.commit()
        changed = cursor.rowcount
        await cursor.close()
        return await self.get(record.task_id) if changed else None

    async def list_running(self) -> list[TaskRecord]:
        cursor = await self.conn.execute(f"{self._SELECT} WHERE status IN ('QUEUED', 'RUNNING') ORDER BY created_at")
        rows = await cursor.fetchall()
        await cursor.close()
        return [_row_to_record(r) for r in rows]

    async def claim_available(self, lease_seconds: float) -> TaskRecord | None:
        """Atomically claim the next runnable task (P3-02).

        Two sources are claimable:
          - QUEUED tasks (fresh submission or re-queued after restart).
          - RUNNING tasks whose lease_expires_at is in the past (the
            leasing worker crashed — D4 visibility timeout).
        The claim is a single conditional UPDATE so concurrent workers can
        never both claim the same task.
        """
        now = utc_now()
        lease_expiry = now + __import__("datetime").timedelta(seconds=lease_seconds)
        now_iso = now.isoformat()
        lease_iso = lease_expiry.isoformat()

        cursor = await self.conn.execute(
            """
            UPDATE cooking_tasks
            SET status='RUNNING', updated_at=?, lease_expires_at=?,
                attempts = attempts + 1,
                event_id = event_id + 1
            WHERE task_id = (
                SELECT task_id FROM cooking_tasks
                WHERE status='QUEUED'
                   OR (
                        status='RUNNING'
                        AND (lease_expires_at IS NULL OR lease_expires_at < ?)
                      )
                AND (expires_at IS NULL OR expires_at > ?)
                ORDER BY created_at
                LIMIT 1
            )
            RETURNING task_id
            """,
            (now_iso, lease_iso, now_iso, now_iso),
        )
        row = await cursor.fetchone()
        await cursor.close()
        await self.conn.commit()
        if row is None:
            return None
        return await self.get(row[0])

    async def renew_lease(self, task_id: str, lease_seconds: float) -> TaskRecord | None:
        """Extend the lease of a RUNNING task (D4 heartbeat).

        Only succeeds while the task is still RUNNING and its lease is
        still held (lease_expires_at >= now) — a task re-claimed by another
        worker after our lease expired must not be renewed.
        """
        now = utc_now()
        lease_expiry = now + __import__("datetime").timedelta(seconds=lease_seconds)
        cursor = await self.conn.execute(
            """
            UPDATE cooking_tasks
            SET lease_expires_at=?, updated_at=?
            WHERE task_id=? AND status='RUNNING'
              AND lease_expires_at IS NOT NULL AND lease_expires_at >= ?
            """,
            (lease_expiry.isoformat(), now.isoformat(), task_id, now.isoformat()),
        )
        await self.conn.commit()
        changed = cursor.rowcount
        await cursor.close()
        if not changed:
            return None
        return await self.get(task_id)
