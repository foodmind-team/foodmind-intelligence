"""P1-06: intermediate-artifact cache tests.

Covers: hit/miss/expiry, max-entries eviction, oversized skip, single-flight,
error-not-cached, and the parse/research cache-key invalidation rules
(model/prompt/allow-list/policy changes make old keys unreachable).
"""

import asyncio

import pytest

from cooking_plan_agent.infrastructure.cache import (
    InMemoryTTLCache,
    build_parse_cache_key,
    build_research_cache_key,
)

# ---------------------------------------------------------------------------
# Core cache behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hit_and_miss_are_counted() -> None:
    cache: InMemoryTTLCache[str, str] = InMemoryTTLCache()
    assert await cache.get("k") is None
    assert cache.stats().misses == 1

    await cache.set("k", "v", 60)
    assert await cache.get("k") == "v"
    assert cache.stats().hits == 1


@pytest.mark.asyncio
async def test_ttl_expiry_evicts() -> None:
    cache = InMemoryTTLCache()
    await cache.set("k", "v", 0.05)
    assert await cache.get("k") == "v"

    await asyncio.sleep(0.06)
    assert await cache.get("k") is None
    assert cache.stats().evictions >= 1


@pytest.mark.asyncio
async def test_max_entries_evicts_oldest() -> None:
    cache = InMemoryTTLCache(max_entries=2)
    await cache.set("a", 1, 60)
    await cache.set("b", 2, 60)
    await cache.set("c", 3, 60)  # evicts oldest ("a")

    assert await cache.get("a") is None
    assert await cache.get("b") == 2
    assert await cache.get("c") == 3
    assert cache.stats().evictions == 1


@pytest.mark.asyncio
async def test_oversized_item_is_not_cached() -> None:
    cache = InMemoryTTLCache(max_item_size_bytes=10)
    await cache.set("k", "x" * 100, 60)
    assert await cache.get("k") is None

    await cache.set("small", "abc", 60)
    assert await cache.get("small") == "abc"


@pytest.mark.asyncio
async def test_single_flight_computes_once() -> None:
    """Concurrent callers for the same key share one computation (P1-06)."""
    cache = InMemoryTTLCache()
    compute_calls = 0

    async def compute() -> str:
        nonlocal compute_calls
        compute_calls += 1
        await asyncio.sleep(0.05)
        return "value"

    results = await asyncio.gather(*(cache.get_or_compute("k", 60, compute) for _ in range(5)))

    assert compute_calls == 1
    assert results == ["value"] * 5


@pytest.mark.asyncio
async def test_failed_compute_is_not_cached_and_propagates() -> None:
    cache = InMemoryTTLCache()

    async def boom() -> str:
        raise RuntimeError("compute exploded")

    with pytest.raises(RuntimeError):
        await cache.get_or_compute("k", 60, boom)

    # The failed result must never be cached.
    assert await cache.get("k") is None


@pytest.mark.asyncio
async def test_get_or_compute_tracks_compute_ms() -> None:
    cache = InMemoryTTLCache()

    async def compute() -> str:
        await asyncio.sleep(0.01)
        return "v"

    await cache.get_or_compute("k", 60, compute)
    assert cache.stats().compute_ms_total > 0


# ---------------------------------------------------------------------------
# Cache-key invalidation rules
# ---------------------------------------------------------------------------


def test_parse_key_changes_with_text_model_prompt_schema() -> None:
    base = dict(parser_type="LLM", model="model-a", prompt_version="v1", schema_version="1.0")
    k1 = build_parse_cache_key("text", **base)
    assert build_parse_cache_key("text", **base) == k1  # deterministic

    variants = [
        build_parse_cache_key("other-text", **base),
        build_parse_cache_key("text", **{**base, "model": "model-b"}),
        build_parse_cache_key("text", **{**base, "prompt_version": "v2"}),
        build_parse_cache_key("text", **{**base, "schema_version": "2.0"}),
        build_parse_cache_key("text", **{**base, "parser_type": "RULE_BASED"}),
    ]
    assert len({k1, *variants}) == 6, "Every key dimension must invalidate"


def test_research_key_includes_provider_policy() -> None:
    base = dict(provider_tag="LLMKnowledgeResearcher", safety_policy_version="1")
    k1 = build_research_cache_key("query", **base)
    assert build_research_cache_key("query", **base) == k1

    variants = [
        build_research_cache_key("different-query", **base),
        build_research_cache_key("query", **{**base, "provider_tag": "OtherResearcher"}),
        build_research_cache_key("query", **{**base, "safety_policy_version": "2"}),
    ]
    assert len({k1, *variants}) == 4


# ---------------------------------------------------------------------------
# Workflow integration: parse artifact reused, final response never shared
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parse_cache_reused_across_runs_but_response_not_shared() -> None:
    """Same recipe text → extraction runs once; READY responses stay distinct.

    Two users with different inventory must NOT share a final result even
    though the parse artifact is cached (P1-06 rule 4).
    """
    from decimal import Decimal

    from cooking_plan_agent.domain.enums import HeatLevel
    from cooking_plan_agent.domain.models import (
        ExtractedIngredient,
        ExtractedRecipeCandidate,
        ExtractedStep,
        GeneratePlanRequest,
        InventoryLotSnapshot,
        KitchenResourceSnapshot,
        ReadyPlanResponse,
    )
    from cooking_plan_agent.infrastructure.cache import InMemoryTTLCache
    from cooking_plan_agent.workflow.context import WorkflowContext
    from cooking_plan_agent.workflow.graph import build_cooking_plan_graph

    class CountingExtractor:
        """Returns a gap-free candidate so the happy path reaches READY."""

        def __init__(self) -> None:
            self.calls = 0

        async def extract(self, source_text: str) -> ExtractedRecipeCandidate:
            self.calls += 1
            return ExtractedRecipeCandidate(
                recipe_id="test-recipe-1",
                dish_name="Test Dish",
                original_servings=2,
                source_language="en",
                ingredients=(
                    ExtractedIngredient(
                        raw_text="chicken 200g",
                        name="chicken breast",
                        quantity=200,
                        unit="g",
                    ),
                ),
                steps=(
                    ExtractedStep(
                        step_number=1,
                        instruction="Cook for 10 minutes",
                        category="heating",
                        active_duration_minutes=10,
                        heat_level=HeatLevel.HIGH,
                        target_temperature_c=Decimal(200),
                        resources_hint=("stove",),
                    ),
                ),
            )

    extractor = CountingExtractor()
    cache: InMemoryTTLCache[str, object] = InMemoryTTLCache()
    ctx = WorkflowContext(recipe_extractor=extractor, cache=cache)  # type: ignore[arg-type]
    graph = build_cooking_plan_graph()

    def _request(user_id: str, on_hand: int) -> GeneratePlanRequest:
        return GeneratePlanRequest(
            request_id=f"req-{user_id}",
            user_id=user_id,
            recipes=({"recipe_id": "r1", "text": "Cook chicken for 10 minutes.", "target_servings": 2},),
            inventory_lots=(
                InventoryLotSnapshot(
                    lot_id="lot-1",
                    item_id="item-1",
                    canonical_name="chicken breast",
                    on_hand=Decimal(on_hand),
                    reserved=Decimal(0),
                    unit="g",
                ),
            ),
            kitchen_resources=(
                KitchenResourceSnapshot(
                    resource_id="stove-1",
                    resource_type="stove",
                    capacity=Decimal(4),
                    capacity_unit="burners",
                ),
            ),
        )

    r1 = await graph.ainvoke({"request": _request("u1", 300)}, context=ctx, config={"recursion_limit": 30})
    calls_after_first = extractor.calls
    assert isinstance(r1["response"], ReadyPlanResponse)

    # Second run (same text) reuses the cached parse artifact.
    r2 = await graph.ainvoke({"request": _request("u2", 300)}, context=ctx, config={"recursion_limit": 30})
    assert isinstance(r2["response"], ReadyPlanResponse)
    assert extractor.calls == calls_after_first, "Parse artifact should be cached across runs"

    # Distinct users (different plan IDs) — final responses are never shared.
    assert r1["response"].plan_id != r2["response"].plan_id
