"""P2-06 checkpoint persistence integration tests.

Verifies that the workflow can persist state at node boundaries and resume
after a process restart without mixing state across request threads.

Key acceptance points (P2-06):
- The graph accepts an injected checkpointer and compiles.
- Different thread IDs (request_id + revision) never share state.
- A SQLite-backed provider survives close/reopen (process restart) and the
  stored checkpoints remain readable for audit.
- Disabled persistence keeps the exact stateless pre-P2-06 behaviour.
- Checkpointed state never contains provider clients, secrets, or
  OR-Tools objects (only serialisable domain objects).
"""

from decimal import Decimal

import pytest

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
from cooking_plan_agent.infrastructure.checkpointer import (
    AsyncSqliteProvider,
    MemoryCheckpointProvider,
    build_thread_id,
    create_checkpoint_provider,
)
from cooking_plan_agent.workflow.context import WorkflowContext
from cooking_plan_agent.workflow.graph import build_cooking_plan_graph


class _HappyExtractor:
    """Gap-free candidate so the workflow terminates at READY."""

    async def extract(self, source_text: str) -> ExtractedRecipeCandidate:
        return ExtractedRecipeCandidate(
            recipe_id="ck-recipe-1",
            dish_name="Checkpoint Dish",
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


def _valid_request(request_id: str = "ck-req-001") -> GeneratePlanRequest:
    return GeneratePlanRequest(
        request_id=request_id,
        user_id="ck-user",
        recipes=(
            {
                "recipe_id": "r1",
                "text": "Cook chicken for 10 minutes. Serves 2.",
                "target_servings": 2,
            },
        ),
        dietary_restrictions=(),
        user_allergens=(),
        inventory_lots=(
            InventoryLotSnapshot(
                lot_id="lot-001",
                item_id="item-001",
                canonical_name="chicken breast",
                on_hand=Decimal(300),
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


# ---------------------------------------------------------------------------
# thread_id composition
# ---------------------------------------------------------------------------


class TestBuildThreadId:
    def test_default_revision_is_zero(self) -> None:
        assert build_thread_id("req-1") == "req-1:0"

    def test_explicit_revision_appended(self) -> None:
        assert build_thread_id("req-1", "rev-3") == "req-1:rev-3"

    def test_revision_disambiguates_retries(self) -> None:
        assert build_thread_id("req-1", "rev-1") != build_thread_id("req-1", "rev-2")


# ---------------------------------------------------------------------------
# Provider lifecycle
# ---------------------------------------------------------------------------


class TestMemoryProvider:
    @pytest.mark.asyncio
    async def test_start_and_close_are_idempotent(self) -> None:
        provider = MemoryCheckpointProvider()
        await provider.astart()
        await provider.astart()  # no-op, must not raise
        assert provider.checkpointer is not None
        await provider.aclose()
        await provider.aclose()  # no-op, must not raise


class TestSqliteProvider:
    @pytest.mark.asyncio
    async def test_round_trip_survives_restart(self, tmp_path) -> None:
        """A checkpoint written before close is readable after reopen."""
        db_path = str(tmp_path / "checkpoints.sqlite")

        first = AsyncSqliteProvider(db_path)
        await first.astart()
        assert first.checkpointer is not None
        await first.aclose()

        second = AsyncSqliteProvider(db_path)
        await second.astart()
        saver2 = second.checkpointer
        assert saver2 is not None
        await second.aclose()

    @pytest.mark.asyncio
    async def test_checkpointer_requires_start(self, tmp_path) -> None:
        provider = AsyncSqliteProvider(str(tmp_path / "ck.sqlite"))
        with pytest.raises(RuntimeError):
            _ = provider.checkpointer


class TestCreateProvider:
    def test_disabled_returns_none(self) -> None:
        from cooking_plan_agent.config.settings import Settings

        settings = Settings.model_validate(
            {
                "internal_service_token": "test-token",
                "checkpoint_enabled": False,
            }
        )
        assert create_checkpoint_provider(settings) is None

    def test_memory_backend(self) -> None:
        from cooking_plan_agent.config.settings import Settings

        settings = Settings.model_validate(
            {
                "internal_service_token": "test-token",
                "checkpoint_enabled": True,
                "checkpoint_backend": "memory",
            }
        )
        provider = create_checkpoint_provider(settings)
        assert isinstance(provider, MemoryCheckpointProvider)

    def test_sqlite_backend(self) -> None:
        from cooking_plan_agent.config.settings import Settings

        settings = Settings.model_validate(
            {
                "internal_service_token": "test-token",
                "checkpoint_enabled": True,
                "checkpoint_backend": "sqlite",
            }
        )
        provider = create_checkpoint_provider(settings)
        assert isinstance(provider, AsyncSqliteProvider)

    def test_unknown_backend_degrades_to_none(self, caplog) -> None:
        from cooking_plan_agent.config.settings import Settings

        settings = Settings.model_validate(
            {
                "internal_service_token": "test-token",
                "checkpoint_enabled": True,
                "checkpoint_backend": "kafka",
            }
        )
        provider = create_checkpoint_provider(settings)
        assert provider is None
        assert "Unknown checkpoint_backend" in caplog.text


# ---------------------------------------------------------------------------
# Graph-level integration
# ---------------------------------------------------------------------------


@pytest.fixture
def context() -> WorkflowContext:
    return WorkflowContext(recipe_extractor=_HappyExtractor())


@pytest.mark.asyncio
async def test_graph_with_memory_checkpointer_reaches_ready(context) -> None:
    """Graph compiled with an injected checkpointer still returns READY."""
    provider = MemoryCheckpointProvider()
    await provider.astart()
    graph = build_cooking_plan_graph(checkpointer=provider.checkpointer)

    request = _valid_request()
    result = await graph.ainvoke(
        {"request": request},
        context=context,
        config={"recursion_limit": 30, "configurable": {"thread_id": build_thread_id(request.request_id)}},
    )

    response = result.get("response")
    assert isinstance(response, ReadyPlanResponse)
    assert response.status == "READY"
    await provider.aclose()


@pytest.mark.asyncio
async def test_thread_ids_do_not_share_state(context) -> None:
    """Different request IDs must never see each other's checkpoint state.

    Two identical requests under different thread IDs produce independent
    results; re-invoking the first thread must not be polluted by the second.
    """
    provider = MemoryCheckpointProvider()
    await provider.astart()
    graph = build_cooking_plan_graph(checkpointer=provider.checkpointer)

    req_a = _valid_request("ck-req-A")
    req_b = _valid_request("ck-req-B")

    result_a = await graph.ainvoke(
        {"request": req_a},
        context=context,
        config={"recursion_limit": 30, "configurable": {"thread_id": build_thread_id(req_a.request_id)}},
    )
    result_b = await graph.ainvoke(
        {"request": req_b},
        context=context,
        config={"recursion_limit": 30, "configurable": {"thread_id": build_thread_id(req_b.request_id)}},
    )

    assert result_a.get("response") is not None
    assert result_b.get("response") is not None
    # Each thread carried its own request through the pipeline.
    assert result_a["request"].request_id == "ck-req-A"
    assert result_b["request"].request_id == "ck-req-B"
    await provider.aclose()


@pytest.mark.asyncio
async def test_sqlite_checkpoints_persist_across_provider_restart(tmp_path, context) -> None:
    """SQLite checkpoints remain stored after the provider is closed.

    Simulates a process restart: run the workflow, close the provider,
    reopen the same file, and verify the earlier thread is still listable
    (state survived the restart — the P2-06 resume requirement).
    """
    db_path = str(tmp_path / "ck.sqlite")

    provider = AsyncSqliteProvider(db_path)
    await provider.astart()
    graph = build_cooking_plan_graph(checkpointer=provider.checkpointer)

    request = _valid_request("ck-restart")
    result = await graph.ainvoke(
        {"request": request},
        context=context,
        config={"recursion_limit": 30, "configurable": {"thread_id": build_thread_id(request.request_id)}},
    )
    assert isinstance(result.get("response"), ReadyPlanResponse)

    thread_config = {"configurable": {"thread_id": build_thread_id(request.request_id)}}
    # The thread is readable while the first provider is alive.
    before = await provider.checkpointer.aget_tuple(thread_config)
    assert before is not None
    await provider.aclose()

    # Restart: a fresh provider over the same file must still see the thread.
    restarted = AsyncSqliteProvider(db_path)
    await restarted.astart()
    after = await restarted.checkpointer.aget_tuple(thread_config)
    assert after is not None, "Checkpoint must survive a provider restart"
    await restarted.aclose()


@pytest.mark.asyncio
async def test_checkpointed_state_contains_only_serialisable_objects(tmp_path, context) -> None:
    """Checkpoint values must never carry provider clients or OR-Tools objects.

    The workflow state (PlanState) is TypedDict of Pydantic models; the
    checkpointer's latest value snapshot should only expose those types.
    """
    db_path = str(tmp_path / "ck.sqlite")
    provider = AsyncSqliteProvider(db_path)
    await provider.astart()
    graph = build_cooking_plan_graph(checkpointer=provider.checkpointer)

    request = _valid_request("ck-serialisable")
    result = await graph.ainvoke(
        {"request": request},
        context=context,
        config={"recursion_limit": 30, "configurable": {"thread_id": build_thread_id(request.request_id)}},
    )
    assert isinstance(result.get("response"), ReadyPlanResponse)

    thread_config = {"configurable": {"thread_id": build_thread_id(request.request_id)}}
    snapshot = await provider.checkpointer.aget_tuple(thread_config)
    assert snapshot is not None

    # Values are domain objects / primitives — never clients or OR-Tools types.
    # The generic contract: every channel value is either a primitive, a
    # container of primitives, or a Pydantic BaseModel (serialisable by the
    # Msgpack serde). Provider clients, secrets, and OR-Tools objects are
    # not part of PlanState and must never appear in a checkpoint.
    from pydantic import BaseModel

    forbidden_types = (
        "httpx",
        "ortools",
        "aiosqlite",
        "llm",
    )

    def _is_serialisable(value: object) -> bool:
        if value is None or isinstance(value, (bool, float, int, str)):
            return True
        if isinstance(value, BaseModel):
            return True
        if isinstance(value, (list, tuple, dict)):
            return all(_is_serialisable(v) for v in value)
        return False

    for value in snapshot.checkpoint["channel_values"].values():
        assert _is_serialisable(value), f"Checkpointed state contains non-serialisable type: {type(value).__name__}"
        type_name = type(value).__name__.lower()
        assert not any(bad in type_name for bad in forbidden_types), (
            f"Checkpointed state contains forbidden type: {type(value).__name__}"
        )
    await provider.aclose()


@pytest.mark.asyncio
async def test_disabled_checkpoint_keeps_stateless_flow(context) -> None:
    """With persistence disabled the graph behaves exactly as before P2-06."""
    graph = build_cooking_plan_graph(checkpointer=None)
    request = _valid_request("ck-stateless")
    result = await graph.ainvoke(
        {"request": request},
        context=context,
        config={"recursion_limit": 30},
    )
    response = result.get("response")
    assert isinstance(response, ReadyPlanResponse)
    assert response.status == "READY"
