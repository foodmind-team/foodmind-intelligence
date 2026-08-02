"""Workflow checkpoint persistence (P2-06).

Provides a thin, lifecycle-managed wrapper around LangGraph checkpointers
so the application can inject persistence without importing LangGraph
backends in the API layer.

Design decisions (P2-06):
- The checkpointer is created in the FastAPI lifespan (never at module
  import time) and closed on shutdown, so connections are not leaked.
- ``MemoryCheckpointProvider`` is used in tests/CI: deterministic and
  dependency-free. ``AsyncSqliteProvider`` is the local/MVP default:
  durable across process restarts, which is the hard requirement for
  P3-01 async tasks and human-confirmation resume.
- No pickle fallback: LangGraph serialises with Msgpack by default. We
  never wrap the state in ``pickle`` and the allow-list rejects arbitrary
  classes, so untrusted checkpoint data cannot execute code on load.

Thread identity: ``thread_id`` = f"{request_id}:{plan_revision or 0}".
Revision disambiguates retries with the same request ID after a
confirmation, so state from different attempts never collides.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

if TYPE_CHECKING:
    from cooking_plan_agent.config.settings import Settings

logger = logging.getLogger(__name__)


class CheckpointProvider(Protocol):
    """Lifecycle contract every checkpointer backend must satisfy."""

    @property
    def checkpointer(self) -> BaseCheckpointSaver[str]:
        """The LangGraph-compatible saver injected into the compiled graph."""
        ...

    async def astart(self) -> None:
        """Open connections and create schema if needed (idempotent)."""
        ...

    async def aclose(self) -> None:
        """Close connections; safe to call more than once."""
        ...


class MemoryCheckpointProvider:
    """In-memory checkpointer for tests and stateless runs.

    State does not survive a process restart; used when
    ``checkpoint_backend == "memory"`` or as an explicit test override.
    """

    def __init__(self) -> None:
        self._saver = InMemorySaver()

    @property
    def checkpointer(self) -> BaseCheckpointSaver[str]:
        return self._saver

    async def astart(self) -> None:
        # InMemorySaver has no persistent resources to initialise.
        return None

    async def aclose(self) -> None:
        # Nothing to release.
        return None


class AsyncSqliteProvider:
    """AsyncSqliteSaver-backed provider (local dev / MVP persistence).

    Uses ``aiosqlite`` so the saver never blocks the event loop, and
    survives process restarts, enabling checkpoint resume for P3-01.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._started = False
        self._saver: BaseCheckpointSaver[str] | None = None
        self._conn: Any | None = None

    @property
    def checkpointer(self) -> BaseCheckpointSaver[str]:
        if self._saver is None:
            raise RuntimeError("Checkpointer not started — call astart() first")
        return self._saver

    async def astart(self) -> None:
        """Open the SQLite connection and initialise schema exactly once."""
        if self._started:
            return
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        conn = await aiosqlite.connect(self._db_path)
        saver = AsyncSqliteSaver(conn)
        # schema DDL runs once per database file (CREATE IF NOT EXISTS is idempotent)
        await saver.setup()
        self._saver = saver
        self._conn = conn
        self._started = True
        logger.info("SQLite checkpointer started | path=%s", self._db_path)

    async def aclose(self) -> None:
        """Close the underlying SQLite connection (safe to call repeatedly)."""
        self._saver = None
        self._started = False
        conn = self._conn
        self._conn = None
        if conn is not None:
            await conn.close()


def create_checkpoint_provider(settings: Settings) -> CheckpointProvider | None:
    """Build a CheckpointProvider from settings, or None when disabled.

    Args:
        settings: App settings. ``checkpoint_enabled`` is the master switch;
            when False the workflow runs stateless (pre-P2-06 behaviour).

    Returns:
        A provider matching ``checkpoint_backend``, or None if persistence
        is disabled or the backend name is unknown (unknown backends log a
        warning and degrade to no persistence — safety over silent breakage).
    """
    if not settings.checkpoint_enabled:
        return None

    backend = settings.checkpoint_backend
    if backend == "memory":
        return MemoryCheckpointProvider()
    if backend == "sqlite":
        return AsyncSqliteProvider(settings.checkpoint_sqlite_path)

    logger.warning("Unknown checkpoint_backend=%r — persistence disabled", backend)
    return None


def build_thread_id(request_id: str, plan_revision: str | None = None) -> str:
    """Compose a unique LangGraph thread ID for a request attempt.

    Args:
        request_id: The caller-supplied idempotency key.
        plan_revision: Optional confirmation revision; retries after a
            confirmation carry a new revision so they get a fresh thread.

    Returns:
        f"{request_id}:{plan_revision or 0}" — namespaced per attempt.
    """
    return f"{request_id}:{plan_revision or 0}"
