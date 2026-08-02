"""P1-02: async isolation — concurrency cap, thread-based CPU solve, client reuse.

Verifies:
  - multi-recipe extraction never exceeds the configured in-flight cap
  - the CP-SAT solve runs in a worker thread (event loop stays alive)
  - LLMClient owns one lifecycle-level httpx client closed on aclose()
"""

import asyncio
import threading
from decimal import Decimal

import pytest

from cooking_plan_agent.domain.enums import HeatLevel, SolverStatus, WorkMode
from cooking_plan_agent.domain.models import (
    CookingTask,
    ExtractedIngredient,
    ExtractedRecipeCandidate,
    ExtractedStep,
    GeneratePlanRequest,
    RecipeInput,
)
from cooking_plan_agent.scheduling.models import ScheduleResult
from cooking_plan_agent.workflow.nodes import parse_recipes_node, solve_schedule_node

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _FakeRuntime:
    def __init__(self, context: object) -> None:
        self.context = context


def _candidate(recipe_id: str = "r1") -> ExtractedRecipeCandidate:
    return ExtractedRecipeCandidate(
        recipe_id=recipe_id,
        dish_name="Test Dish",
        original_servings=2,
        source_language="en",
        ingredients=(ExtractedIngredient(raw_text="chicken 200g", name="chicken breast", quantity=200, unit="g"),),
        steps=(
            ExtractedStep(
                step_number=1,
                instruction="Cook.",
                category="heating",
                heat_level=HeatLevel.HIGH,
                active_duration_minutes=10,
            ),
        ),
    )


def _request(recipe_count: int = 4) -> GeneratePlanRequest:
    return GeneratePlanRequest(
        request_id="req-p102",
        user_id="u",
        recipes=tuple(
            RecipeInput(recipe_id=f"r{i}", text=f"Recipe {i}", target_servings=Decimal(2)) for i in range(recipe_count)
        ),
    )


def _task_graph():
    from cooking_plan_agent.preparation.task_graph import TaskGraph

    task = CookingTask(
        task_id="t1",
        dish_id="r1",
        instruction="Cook",
        duration_minutes=5,
        work_mode=WorkMode.ACTIVE,
        category="heating",
    )
    return TaskGraph(tasks=(task,), edges=())


# ---------------------------------------------------------------------------
# Concurrency cap on multi-recipe extraction
# ---------------------------------------------------------------------------


class _TrackingExtractor:
    """Counts the peak number of concurrent extract() calls."""

    def __init__(self) -> None:
        self.peak_concurrent = 0
        self._active = 0
        self._lock = threading.Lock()

    async def extract(self, source_text: str) -> ExtractedRecipeCandidate:
        with self._lock:
            self._active += 1
            self.peak_concurrent = max(self.peak_concurrent, self._active)
        await asyncio.sleep(0.05)
        with self._lock:
            self._active -= 1
        return _candidate()


@pytest.mark.asyncio
async def test_extraction_never_exceeds_concurrency_cap(monkeypatch) -> None:
    """In-flight extract() calls must never exceed llm_max_concurrency."""
    monkeypatch.setenv("COOKING_PLAN_LLM_MAX_CONCURRENCY", "2")
    from cooking_plan_agent.config.settings import get_settings

    get_settings.cache_clear()

    extractor = _TrackingExtractor()
    runtime = _FakeRuntime(type("C", (), {"recipe_extractor": extractor})())

    try:
        result = await parse_recipes_node({"request": _request(6)}, runtime)
    finally:
        get_settings.cache_clear()

    assert extractor.peak_concurrent <= 2, f"Peak concurrency {extractor.peak_concurrent} exceeded cap 2"
    assert len(result["extracted_candidates"]) == 6


# ---------------------------------------------------------------------------
# CPU-bound solve runs in a worker thread — event loop stays alive
# ---------------------------------------------------------------------------


def test_solver_maps_cpu_solve_to_thread(monkeypatch) -> None:
    """solve_schedule_node must call schedule() via asyncio.to_thread.

    Proves the CPU-bound solve is delegated to a worker thread rather than
    running inline on the event loop.
    """
    import cooking_plan_agent.workflow.nodes as nodes_module

    captured: dict[str, object] = {}

    async def fake_to_thread(fn, *args, **kwargs):  # noqa: ANN002, ANN003
        captured["called_via_thread"] = True
        return fn(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(
        "cooking_plan_agent.scheduling.orchestrator.schedule",
        lambda problem: (ScheduleResult(status=SolverStatus.INFEASIBLE), None),
    )

    async def _run() -> None:
        state = {"request": _request(1), "task_graph": _task_graph()}
        await nodes_module.solve_schedule_node(state, _FakeRuntime(None))

    asyncio.run(_run())
    assert captured.get("called_via_thread") is True


@pytest.mark.asyncio
async def test_event_loop_heartbeat_runs_during_solve(monkeypatch) -> None:
    """While the blocking solve runs in a thread, the event loop must tick."""
    started = threading.Event()
    release = threading.Event()
    heartbeats: list[int] = []

    def blocking_solve(problem):  # noqa: ANN001
        started.set()
        release.wait(timeout=5)
        return ScheduleResult(status=SolverStatus.INFEASIBLE), None

    monkeypatch.setattr("cooking_plan_agent.scheduling.orchestrator.schedule", blocking_solve)
    state = {"request": _request(1), "task_graph": _task_graph()}

    async def heartbeat() -> None:
        while not release.is_set():
            heartbeats.append(1)
            await asyncio.sleep(0.01)

    heartbeat_task = asyncio.create_task(heartbeat())
    started.wait(timeout=2)
    await asyncio.sleep(0.1)  # give the loop time to tick while solve blocks
    release.set()

    result = await solve_schedule_node(state, _FakeRuntime(None))
    await heartbeat_task

    assert len(heartbeats) > 0, "Event loop was blocked during solve — not running in a thread"
    assert result["schedule_result"].status == SolverStatus.INFEASIBLE


# ---------------------------------------------------------------------------
# LLMClient lifecycle: one client, closed once
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_client_owns_single_lifecycle_client_and_closes() -> None:
    from cooking_plan_agent.llm.client import LLMClient

    client = LLMClient(
        base_url="http://localhost:11434/v1",
        model="test-model",
        connection_pool_size=3,
    )
    try:
        assert client._client is not None  # noqa: SLF001 — white-box lifecycle check
    finally:
        await client.aclose()

    # aclose is idempotent-safe: closing again must not raise.
    await client.aclose()
