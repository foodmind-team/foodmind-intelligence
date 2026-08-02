"""P4-01: schedule explanation node tests.

Covers the five explanation paths (disabled / llm success / llm failure /
empty output / no explainer), the compact summary shape, the D3/D4 privacy
boundary (no recipe text / user identity in the summary), and the graph
integration so a verified READY flow carries the explanation.
"""

from __future__ import annotations

import pytest

from cooking_plan_agent.domain.enums import SolverStatus, WorkMode
from cooking_plan_agent.domain.models import (
    CookingTask,
    GeneratePlanRequest,
)
from cooking_plan_agent.preparation.task_graph import TaskGraph
from cooking_plan_agent.scheduling.models import ScheduledInterval, ScheduleResult
from cooking_plan_agent.workflow.context import WorkflowContext
from cooking_plan_agent.workflow.graph import build_cooking_plan_graph
from cooking_plan_agent.workflow.nodes import (
    _build_schedule_summary,
    _deterministic_explanation,
    explain_schedule_node,
)
from cooking_plan_agent.workflow.state import PlanState


class _FakeRuntime:
    """Minimal runtime stand-in — nodes only need .context at call time."""

    def __init__(self, context: object) -> None:
        self.context = context


def _ctx(explainer: object | None) -> object:
    """Minimal WorkflowContext stand-in exposing .explainer."""
    return type("C", (), {"explainer": explainer})()


class _FakeExplainer:
    """Captures the summary it receives and mimics LLM success/failure."""

    def __init__(
        self, *, text: str = "Parallel cooking lets the last dish finish on time.", fail: bool = False
    ) -> None:
        self._text = text
        self._fail = fail
        self.captured: list[dict[str, object]] = []

    async def explain(self, schedule_summary: dict[str, object]) -> str:
        self.captured.append(schedule_summary)
        if self._fail:
            raise RuntimeError("injected LLM failure")
        return self._text


def _patch_settings(monkeypatch: pytest.MonkeyPatch, *, enabled: bool) -> None:
    """Point get_settings() at a hermetic Settings for the duration of a test."""
    from cooking_plan_agent.config import settings as settings_module

    monkeypatch.setattr(
        settings_module,
        "get_settings",
        lambda: settings_module.Settings(
            internal_service_token="test-token",
            explanation_enabled=enabled,
        ),
    )


def _make_state_with_schedule() -> PlanState:
    """Deterministic verified schedule: two dishes, overlapping ACTIVE tasks."""
    tasks = (
        CookingTask(
            task_id="t1",
            dish_id="dish-a",
            instruction="Boil water",
            duration_minutes=10,
            work_mode=WorkMode.ACTIVE,
            category="heating",
        ),
        CookingTask(
            task_id="t2",
            dish_id="dish-a",
            instruction="Cook pasta",
            duration_minutes=8,
            work_mode=WorkMode.ACTIVE,
            category="heating",
        ),
        CookingTask(
            task_id="t3",
            dish_id="dish-b",
            instruction="Chop vegetables",
            duration_minutes=5,
            work_mode=WorkMode.ACTIVE,
            category="cutting",
        ),
        CookingTask(
            task_id="t4",
            dish_id="dish-b",
            instruction="Stir-fry",
            duration_minutes=7,
            work_mode=WorkMode.ACTIVE,
            category="heating",
        ),
    )
    schedule = ScheduleResult(
        status=SolverStatus.OPTIMAL,
        makespan_minutes=18,
        intervals=(
            ScheduledInterval(task_id="t1", start_minute=0, end_minute=10),
            ScheduledInterval(task_id="t2", start_minute=10, end_minute=18),
            ScheduledInterval(task_id="t3", start_minute=0, end_minute=5),
            ScheduledInterval(task_id="t4", start_minute=5, end_minute=12),
        ),
    )
    return PlanState(
        request=GeneratePlanRequest(
            request_id="req-1",
            user_id="user-1",
            recipes=({"recipe_id": "r1", "text": "Cook.", "target_servings": 2},),
        ),
        schedule_result=schedule,
        task_graph=TaskGraph(tasks=tasks, edges=()),
    )


# ---------------------------------------------------------------------------
# Summary shape (D3: only verified schedule facts; D4: no sensitive content)
# ---------------------------------------------------------------------------


def test_summary_shape_and_dish_mapping() -> None:
    summary = _build_schedule_summary(_make_state_with_schedule())

    assert summary["makespan_minutes"] == 18
    # dish completions: dish-a ends at 18 (t2), dish-b ends at 12 (t4)
    assert sorted(summary["dish_completions"], key=lambda d: str(d["dish"])) == [
        {"dish": "dish-a", "completion_minute": 18},
        {"dish": "dish-b", "completion_minute": 12},
    ]
    # peak concurrency over [0,18] (half-open): t1∥t3 (2), t4 starts as t3 ends,
    # t2 starts as t1 ends -> peak 2
    assert summary["parallel_groups"] == 2


def test_summary_contains_no_sensitive_content() -> None:
    """D4: the summary must never carry recipe text, user id, or inventory."""
    summary = _build_schedule_summary(_make_state_with_schedule())
    serialized = repr(summary)
    assert "user-1" not in serialized
    assert "Cook" not in serialized
    assert "req-1" not in serialized


# ---------------------------------------------------------------------------
# Node behaviour — five paths all keep READY safe
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disabled_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_settings(monkeypatch, enabled=False)
    result = await explain_schedule_node(_make_state_with_schedule(), _FakeRuntime(_ctx(_FakeExplainer())))

    assert result["explanation"] is None
    assert result["explanation_source"] == "disabled"


@pytest.mark.asyncio
async def test_llm_success_returns_text(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_settings(monkeypatch, enabled=True)
    explainer = _FakeExplainer(text="Boil in parallel to finish on time.")
    result = await explain_schedule_node(_make_state_with_schedule(), _FakeRuntime(_ctx(explainer)))

    assert result["explanation"] == "Boil in parallel to finish on time."
    assert result["explanation_source"] == "llm"
    assert len(explainer.captured) == 1


@pytest.mark.asyncio
async def test_llm_failure_falls_back_to_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_settings(monkeypatch, enabled=True)
    result = await explain_schedule_node(_make_state_with_schedule(), _FakeRuntime(_ctx(_FakeExplainer(fail=True))))

    assert result["explanation_source"] == "deterministic"
    assert "18 minutes" in str(result["explanation"])


@pytest.mark.asyncio
async def test_llm_empty_output_falls_back_to_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_settings(monkeypatch, enabled=True)
    result = await explain_schedule_node(_make_state_with_schedule(), _FakeRuntime(_ctx(_FakeExplainer(text="   "))))

    assert result["explanation_source"] == "deterministic"


@pytest.mark.asyncio
async def test_no_explainer_uses_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_settings(monkeypatch, enabled=True)
    result = await explain_schedule_node(_make_state_with_schedule(), _FakeRuntime(_ctx(None)))

    assert result["explanation_source"] == "deterministic"


def test_deterministic_explanation_states_verified_facts_only() -> None:
    text = _deterministic_explanation(
        {
            "makespan_minutes": 18,
            "dish_completions": [{"dish": "dish-b", "completion_minute": 12}],
        }
    )
    assert "18 minutes" in text
    assert "dish-b at 12 min" in text


# ---------------------------------------------------------------------------
# Graph integration — verified READY flow carries the explanation
# ---------------------------------------------------------------------------


class _FakeRecipeExtractor:
    """Deterministic gap-free extractor so the graph reaches READY."""

    async def extract(self, source_text: str):
        from cooking_plan_agent.domain.enums import HeatLevel
        from cooking_plan_agent.domain.models import (
            ExtractedIngredient,
            ExtractedRecipeCandidate,
            ExtractedStep,
        )

        return ExtractedRecipeCandidate(
            recipe_id="test-recipe-1",
            dish_name="Test Dish",
            original_servings=2,
            source_language="en",
            ingredients=(ExtractedIngredient(raw_text="chicken 200g", name="chicken breast", quantity=200, unit="g"),),
            steps=(
                ExtractedStep(
                    step_number=1,
                    instruction="Cook for 10 minutes",
                    category="heating",
                    active_duration_minutes=10,
                    heat_level=HeatLevel.HIGH,
                    target_temperature_c=200,
                    resources_hint=("stove",),
                ),
            ),
        )


def _valid_request() -> GeneratePlanRequest:
    from decimal import Decimal

    from cooking_plan_agent.domain.models import InventoryLotSnapshot, KitchenResourceSnapshot

    return GeneratePlanRequest(
        request_id="req-explain",
        user_id="user-explain",
        recipes=({"recipe_id": "r1", "text": "Cook chicken for 10 minutes. Serves 2.", "target_servings": 2},),
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


@pytest.mark.asyncio
async def test_ready_flow_carries_explanation(monkeypatch: pytest.MonkeyPatch) -> None:
    """P4-01: with the feature enabled, READY responses include explanation."""
    from cooking_plan_agent.domain.models import ReadyPlanResponse

    _patch_settings(monkeypatch, enabled=True)
    explainer = _FakeExplainer(text="Parallel cooking finishes the last dish on time.")
    context = WorkflowContext(recipe_extractor=_FakeRecipeExtractor(), explainer=explainer)
    graph = build_cooking_plan_graph()

    result = await graph.ainvoke(
        {"request": _valid_request()},
        context=context,
        config={"recursion_limit": 30},
    )

    response = result["response"]
    assert isinstance(response, ReadyPlanResponse), f"expected READY, got {type(response).__name__}"
    assert response.explanation == "Parallel cooking finishes the last dish on time."
    assert response.explanation_source == "llm"
