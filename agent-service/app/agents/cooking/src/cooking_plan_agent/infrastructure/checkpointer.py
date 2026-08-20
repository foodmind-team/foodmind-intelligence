# =============================================================================
# 工作流检查点持久化模块（infrastructure/checkpointer）
# -----------------------------------------------------------------------------
# 提供一个围绕 LangGraph checkpointer 的“薄封装 + 生命周期管理”层，
# 使应用能在不把 LangGraph 后端引入 API 层的前提下注入持久化能力。
# 设计决策（P2-06）：
#   - 检查点器在 FastAPI lifespan 中创建（绝不在模块导入时），关闭时释放，
#     避免连接泄漏。
#   - 测试 / CI 用 MemoryCheckpointProvider：确定性、零依赖；
#     AsyncSqliteProvider 作为本地 / MVP 默认：跨进程重启持久，这是
#     P3-01 异步任务与“人工确认续答”的硬性要求。
#   - 无 pickle 兜底：LangGraph 默认用 Msgpack 序列化；绝不把状态包进 pickle，
#     且 allow-list 拒绝任意类，因此不可信的检查点数据无法在加载时执行代码。
# 线程标识：thread_id = f"{request_id}:{plan_revision or 0}"。
#   用 revision 区分“同一 request_id 在确认后重试”的场景，避免不同尝试的状态互相碰撞。
# =============================================================================

"""Workflow checkpoint persistence (P2-06).

工作流检查点持久化（P2-06）。

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
    """检查点提供者协议：每个后端必须满足的生命周期契约。

    Lifecycle contract every checkpointer backend must satisfy.
    """

    @property
    def checkpointer(self) -> BaseCheckpointSaver[str]:
        """注入到已编译图（compiled graph）的 LangGraph 兼容保存器。"""
        ...

    async def astart(self) -> None:
        """打开连接并按需创建 schema（幂等）。"""
        ...

    async def aclose(self) -> None:
        """关闭连接；可安全地多次调用。"""
        ...


class MemoryCheckpointProvider:
    """内存检查点提供者：用于测试与无状态运行。

    In-memory checkpointer for tests and stateless runs.

    State does not survive a process restart; used when
    ``checkpoint_backend == "memory"`` or as an explicit test override.

    状态不跨进程重启保留；当 ``checkpoint_backend == "memory"`` 或作为显式测试覆盖时使用。
    """

    def __init__(self) -> None:
        self._saver = InMemorySaver()

    @property
    def checkpointer(self) -> BaseCheckpointSaver[str]:
        return self._saver

    async def astart(self) -> None:
        # InMemorySaver has no persistent resources to initialise.
        # InMemorySaver 无可初始化的持久化资源。
        return None

    async def aclose(self) -> None:
        # Nothing to release.
        # 无可释放的资源。
        return None


class AsyncSqliteProvider:
    """基于 AsyncSqliteSaver 的提供者（本地开发 / MVP 持久化）。

    AsyncSqliteSaver-backed provider (local dev / MVP persistence).

    Uses ``aiosqlite`` so the saver never blocks the event loop, and
    survives process restarts, enabling checkpoint resume for P3-01.

    使用 ``aiosqlite`` 使保存器永不阻塞事件循环，且跨进程重启存活，
    从而支撑 P3-01 的检查点续答。
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
        """打开 SQLite 连接并精确初始化一次 schema。"""
        if self._started:
            return
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        conn = await aiosqlite.connect(self._db_path)
        saver = AsyncSqliteSaver(conn)
        # schema DDL runs once per database file (CREATE IF NOT EXISTS is idempotent)
        # schema DDL 每个数据库文件只执行一次（CREATE IF NOT EXISTS 是幂等的）。
        await saver.setup()
        self._saver = saver
        self._conn = conn
        self._started = True
        logger.info("SQLite checkpointer started | path=%s", self._db_path)

    async def aclose(self) -> None:
        """关闭底层 SQLite 连接（可重复调用）。"""
        self._saver = None
        self._started = False
        conn = self._conn
        self._conn = None
        if conn is not None:
            await conn.close()


def create_checkpoint_provider(settings: Settings) -> CheckpointProvider | None:
    """根据 settings 构建 CheckpointProvider；禁用时返回 None。

    Build a CheckpointProvider from settings, or None when disabled.

    Args:
        settings: App settings. ``checkpoint_enabled`` is the master switch;
            when False the workflow runs stateless (pre-P2-06 behaviour).

        settings：应用配置。``checkpoint_enabled`` 是总开关；
            为 False 时工作流以无状态方式运行（P2-06 之前的行为）。

    Returns:
        A provider matching ``checkpoint_backend``, or None if persistence
        is disabled or the backend name is unknown (unknown backends log a
        warning and degrade to no persistence — safety over silent breakage).

        与 ``checkpoint_backend`` 匹配的提供者；若持久化被禁用或后端名未知则返回 None
        （未知后端记录警告并降级为无持久化 —— 宁肯安全降级也不静默损坏）。
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
    """为一次请求尝试组合出唯一的 LangGraph thread ID。

    Compose a unique LangGraph thread ID for a request attempt.

    Args:
        request_id: The caller-supplied idempotency key.
            request_id：调用方提供的幂等键。
        plan_revision: Optional confirmation revision; retries after a
            confirmation carry a new revision so they get a fresh thread.
            plan_revision：可选的确认修订版本；确认后的重试携带新版本，从而获得新线程。

    Returns:
        f"{request_id}:{plan_revision or 0}" — namespaced per attempt.
        f"{request_id}:{plan_revision or 0}" —— 按尝试命名空间隔离。
    """
    return f"{request_id}:{plan_revision or 0}"
