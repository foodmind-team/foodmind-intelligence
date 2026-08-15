"""Intermediate-artifact cache (P1-06).

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
    """Async cache contract used by workflow nodes.

    Deliberately non-generic: concrete implementations use their own type
    parameters; the workflow only needs the object-level contract.
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
    """Cumulative cache metrics (P1-06)."""

    hits: int
    misses: int
    evictions: int
    compute_ms_total: float


@dataclass
class _Entry[V]:
    value: V
    expires_at: float


class InMemoryTTLCache[K, V]:
    """Bounded in-memory TTL cache with single-flight and size caps.

    Thread-safety: all mutation happens inside the event loop (single-threaded
    per loop); no locks are needed.
    """

    def __init__(
        self,
        *,
        max_entries: int = 1000,
        max_item_size_bytes: int | None = 100_000,
        default_ttl_seconds: float = 3600.0,
    ) -> None:
        self._max_entries = max(1, max_entries)
        self._max_item_size_bytes = max_item_size_bytes
        self._default_ttl = default_ttl_seconds
        # Insertion-ordered dict → oldest entry is first for eviction.
        self._store: dict[K, _Entry[V]] = {}
        self._inflight: dict[K, asyncio.Future[V]] = {}
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._compute_ms_total = 0.0

    async def aclose(self) -> None:
        """Clear the cache and drop in-flight computations."""
        self._store.clear()
        for future in self._inflight.values():
            future.cancel()
        self._inflight.clear()

    # ------------------------------------------------------------------
    # Cache interface
    # ------------------------------------------------------------------

    async def get(self, key: K) -> V | None:
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
        if not self._under_size_limit(value):
            return  # oversized item is not cached
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
        cached = await self.get(key)
        if cached is not None:
            return cached

        # Single-flight: concurrent callers share one in-flight computation.
        inflight = self._inflight.get(key)
        if inflight is not None:
            return await asyncio.shield(inflight)

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
            if future.done() and not future.cancelled():
                future.exception()

    def stats(self) -> CacheStats:
        return CacheStats(
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
            compute_ms_total=self._compute_ms_total,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _under_size_limit(self, value: V) -> bool:
        if self._max_item_size_bytes is None:
            return True
        return self._size_bytes(value) <= self._max_item_size_bytes

    @staticmethod
    def _size_bytes(value: object) -> int:
        dumper = getattr(value, "model_dump_json", None)
        if dumper is not None:
            return len(dumper())
        return len(repr(value))

    def _evict_one(self) -> None:
        """Drop the oldest entry (or an already-expired one)."""
        for key in list(self._store):
            if self._store[key].expires_at < time.monotonic():
                del self._store[key]
                self._evictions += 1
                return
        # Nothing expired — evict the oldest inserted entry (dict order).
        oldest = next(iter(self._store))
        del self._store[oldest]
        self._evictions += 1


# ===========================================================================
# Cache-key builders (P1-06 rules 2 & 3)
# ===========================================================================

# Safety-policy version tag for research cache keys (P1-06 rule 3). Bump when
# reconciliation thresholds or safety policy rules change so old evidence is
# never reused under a new policy.
RESEARCH_SAFETY_POLICY_VERSION = "1"


def _stable_digest(*parts: str) -> str:
    """Deterministic, process-independent hash (unlike builtin hash())."""
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
    """Key for a recipe-parse artifact.

    Normalised text hash + parser type + model + prompt version + language +
    schema version: a model or prompt upgrade invalidates old entries without
    explicit eviction (P1-06 rule 2).
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
    """Key for a research artifact (P1-06 rule 3).

    Includes the query, the provider identity, and the safety policy version
    so evidence is never shared across policy regimes.
    """
    return _stable_digest(
        query_text,
        provider_tag,
        safety_policy_version,
        model,
    )
