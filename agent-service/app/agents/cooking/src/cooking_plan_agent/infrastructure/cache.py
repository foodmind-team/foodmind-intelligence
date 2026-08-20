# =============================================================================
# 中间产物缓存模块（infrastructure/cache）
# -----------------------------------------------------------------------------
# 实现 P1-06「中间产物缓存」，缓存稳定、可复用的“解析 / 联网研究”结果，
# 避免重复的 LLM / 搜索调用，同时绝不缓存动态的按请求状态。核心要点：
#   - 解析缓存键包含：规范化文本哈希、解析器类型、模型、提示词版本、语言、schema 版本；
#     模型或提示词一旦变更，旧条目自动失效（无需显式清空）。
#   - 研究缓存键包含：查询文本、provider 标签、允许列表、安全策略版本。
#   - 最终的 READY 响应绝不缓存（它们携带 request ID、库存、厨房资源与用户偏好）。
#   - Single-flight 防止同一键的“惊群”重复计算。
#   - 命中 / 未命中 / 淘汰 / 计算耗时指标均被统计。
# 设计：进程级（instance-level）内存 TTL 缓存；分布式缓存属 P3-02 范畴。
#       关闭缓存只影响性能，绝不影响正确性。
# =============================================================================

"""Intermediate-artifact cache (P1-06).

中间产物缓存（P1-06）。

Caches stable, reusable parse and research artifacts so repeated LLM / Search
calls are avoided, while never leaking dynamic per-request state:

  - Parse keys include: normalised-text hash, parser type, model, prompt
    version, language, and schema version. A model or prompt change makes the
    old entries unreachable automatically.
  - Research keys include: query text, provider tag, allow-list, and safety
    policy version.
  - Final READY responses are NEVER cached (they carry request IDs,
    inventory, kitchen resources and user preferences).
  - Single-flight prevents thundering-herd recomputation for the same key.
  - Hit / miss / eviction / compute_ms metrics are tracked.

Design: in-memory TTL cache, instance-level (per process). Distributed cache
is a P3-02 concern. Disabling the cache only affects performance, never
correctness.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol


class Cache(Protocol):
    """缓存协议：工作流节点使用的异步缓存契约。

    Async cache contract used by workflow nodes.

    Deliberately non-generic: concrete implementations use their own type
    parameters; the workflow only needs the object-level contract.

    刻意不做泛型：具体实现使用自己的类型参数；工作流只需对象级契约即可。
    """

    async def get(self, key: object) -> object | None: ...

    async def set(self, key: object, value: object, ttl_seconds: float | None = None) -> None: ...

    async def get_or_compute(
        self,
        key: object,
        ttl_seconds: float | None,
        compute: Callable[[], Awaitable[object]],
    ) -> object: ...


@dataclass(frozen=True)
class CacheStats:
    """累积缓存指标（P1-06）。

    Cumulative cache metrics (P1-06).
    """

    hits: int
    # ↑ 命中次数
    misses: int
    # ↑ 未命中次数
    evictions: int
    # ↑ 淘汰次数
    compute_ms_total: float
    # ↑ 累计计算耗时（毫秒）


@dataclass
class _Entry[V]:
    """缓存条目：值 + 过期时间戳。"""

    value: V
    expires_at: float
    # ↑ 过期时间（基于 time.monotonic() 的单调时钟）


class InMemoryTTLCache[K, V]:
    """有界内存 TTL 缓存：支持 single-flight 与容量上限。

    Bounded in-memory TTL cache with single-flight and size caps.

    Thread-safety: all mutation happens inside the event loop (single-threaded
    per loop); no locks are needed.

    线程安全说明：所有变更都在事件循环内（每个循环单线程）发生，无需加锁。
    """

    def __init__(
        self,
        *,
        max_entries: int = 1000,
        max_item_size_bytes: int | None = 100_000,
        default_ttl_seconds: float = 3600.0,
    ) -> None:
        self._max_entries = max(1, max_entries)
        # ↑ 最大条目数（至少为 1）
        self._max_item_size_bytes = max_item_size_bytes
        # ↑ 单条目最大字节数（None 表示不限制）
        self._default_ttl = default_ttl_seconds
        # ↑ 默认 TTL（秒）
        # Insertion-ordered dict → oldest entry is first for eviction.
        # 插入有序字典 → 最老的条目在最前，便于淘汰。
        self._store: dict[K, _Entry[V]] = {}
        self._inflight: dict[K, asyncio.Future[V]] = {}
        # ↑ 进行中的计算 Future，用于 single-flight
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._compute_ms_total = 0.0

    async def aclose(self) -> None:
        """清空缓存并取消进行中的计算。"""
        self._store.clear()
        for future in self._inflight.values():
            future.cancel()
        self._inflight.clear()

    # ------------------------------------------------------------------
    # Cache interface
    # 缓存接口
    # ------------------------------------------------------------------

    async def get(self, key: K) -> V | None:
        """按键读取，命中则返回；过期则删除并计未命中。"""
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return None
        if entry.expires_at < time.monotonic():
            del self._store[key]
            self._evictions += 1
            self._misses += 1
            return None
        self._hits += 1
        return entry.value

    async def set(self, key: K, value: V, ttl_seconds: float | None = None) -> None:
        """写入条目；超大项不缓存，超容量先淘汰一个。"""
        if not self._under_size_limit(value):
            return  # oversized item is not cached
            # ↑ 超过大小限制的条目直接不缓存
        if key not in self._store and len(self._store) >= self._max_entries:
            self._evict_one()
        ttl = self._default_ttl if ttl_seconds is None else ttl_seconds
        self._store[key] = _Entry(value=value, expires_at=time.monotonic() + ttl)

    async def get_or_compute(
        self,
        key: K,
        ttl_seconds: float | None,
        compute: Callable[[], Awaitable[V]],
    ) -> V:
        """读缓存，未命中则计算并回填；同一键并发调用共享一次计算（single-flight）。"""
        cached = await self.get(key)
        if cached is not None:
            return cached

        # Single-flight: concurrent callers share one in-flight computation.
        # Single-flight：并发调用者共享同一次进行中的计算。
        inflight = self._inflight.get(key)
        if inflight is not None:
            return await asyncio.shield(inflight)
            # ↑ shield 防止外部取消影响共享计算

        loop = asyncio.get_running_loop()
        future: asyncio.Future[V] = loop.create_future()
        self._inflight[key] = future
        try:
            start = time.monotonic()
            value = await compute()
            self._compute_ms_total += (time.monotonic() - start) * 1000
            if not future.done():
                future.set_result(value)
            await self.set(key, value, ttl_seconds)
            return value
        except BaseException as exc:
            if not future.done():
                future.set_exception(exc)
            raise
        finally:
            self._inflight.pop(key, None)
            # Mark a set exception as retrieved so asyncio does not warn about
            # an un-retrieved future exception when no followers awaited it.
            # 将已设置的异常标记为“已检索”，避免没有追随者 await 时 asyncio 警告“未检索的 Future 异常”。
            if future.done() and not future.cancelled():
                future.exception()

    def stats(self) -> CacheStats:
        """返回累积缓存指标快照。"""
        return CacheStats(
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
            compute_ms_total=self._compute_ms_total,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # 内部辅助方法
    # ------------------------------------------------------------------

    def _under_size_limit(self, value: V) -> bool:
        """判断条目是否未超过大小限制。"""
        if self._max_item_size_bytes is None:
            return True
        return self._size_bytes(value) <= self._max_item_size_bytes

    @staticmethod
    def _size_bytes(value: object) -> int:
        """估算对象字节大小：Pydantic 模型用 JSON 长度，否则用 repr 长度。"""
        dumper = getattr(value, "model_dump_json", None)
        if dumper is not None:
            return len(dumper())
        return len(repr(value))

    def _evict_one(self) -> None:
        """淘汰一个条目：优先淘汰已过期的，否则淘汰最老（最早插入）的。"""
        for key in list(self._store):
            if self._store[key].expires_at < time.monotonic():
                del self._store[key]
                self._evictions += 1
                return
        # Nothing expired — evict the oldest inserted entry (dict order).
        # 没有过期项 —— 淘汰最早插入的条目（字典顺序）。
        oldest = next(iter(self._store))
        del self._store[oldest]
        self._evictions += 1


# ===========================================================================
# Cache-key builders (P1-06 rules 2 & 3)
# 缓存键构建器（P1-06 规则 2 与 3）
# ===========================================================================

# Safety-policy version tag for research cache keys (P1-06 rule 3). Bump when
# reconciliation thresholds or safety policy rules change so old evidence is
# never reused under a new policy.
# 研究缓存键的安全策略版本标签（P1-06 规则 3）。当调和阈值或安全策略规则变化时递增，
# 确保新策略下绝不复用旧证据。
RESEARCH_SAFETY_POLICY_VERSION = "1"


def _stable_digest(*parts: str) -> str:
    """确定性、跨进程一致的哈希（不同于内置 hash() 的进程相关性）。"""
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:24]


def build_parse_cache_key(
    text: str,
    *,
    parser_type: str,
    model: str,
    prompt_version: str,
    schema_version: str,
    language: str = "und",
) -> str:
    """构建菜谱解析产物的缓存键。

    Key for a recipe-parse artifact.

    Normalised text hash + parser type + model + prompt version + language +
    schema version: a model or prompt upgrade invalidates old entries without
    explicit eviction (P1-06 rule 2).

    规范化文本哈希 + 解析器类型 + 模型 + 提示词版本 + 语言 + schema 版本：
    模型或提示词升级后旧条目自动失效，无需显式淘汰（P1-06 规则 2）。
    """
    text_digest = _stable_digest(text.strip())
    return _stable_digest(text_digest, parser_type, model, prompt_version, language, schema_version)


def build_research_cache_key(
    query_text: str,
    *,
    provider_tag: str,
    safety_policy_version: str,
    model: str = "",
) -> str:
    """构建研究产物的缓存键（P1-06 规则 3）。

    Key for a research artifact (P1-06 rule 3).

    Includes the query, the provider identity, and the safety policy version
    so evidence is never shared across policy regimes.

    包含查询、provider 身份与安全策略版本，确保证据绝不在不同策略体系间共享。
    """
    return _stable_digest(
        query_text,
        provider_tag,
        safety_policy_version,
        model,
    )
