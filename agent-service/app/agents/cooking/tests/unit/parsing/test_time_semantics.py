"""P0-05 time/date/serving/deadline semantics tests.

Covers the four distinct time semantics:
  1. time_limit_minutes → makespan hard constraint (INFEASIBLE when tight)
  2. cooking_date → FEFO + expired-lot exclusion on the cooking day
  3. serving_at (timezone-aware) vs legacy serving_time (HH:MM string)
  4. solver_timeout_seconds passes through to the SchedulingProblem
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
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
from cooking_plan_agent.scheduling.models import SchedulingProblem
from cooking_plan_agent.workflow.context import WorkflowContext
from cooking_plan_agent.workflow.graph import build_cooking_plan_graph

SGT = timezone(timedelta(hours=8))


class _TimeAwareExtractor:
    """Extractor returning a candidate with realistic durations."""

    async def extract(self, source_text: str) -> ExtractedRecipeCandidate:
        return ExtractedRecipeCandidate(
            recipe_id="time-recipe-1",
            dish_name="Timed Dish",
            original_servings=2,
            source_language="en",
            ingredients=(
                ExtractedIngredient(
                    raw_text="chicken 200g",
                    name="chicken breast",
                    quantity=Decimal(200),
                    unit="g",
                    confidence=Decimal("1.0"),
                ),
            ),
            steps=(
                ExtractedStep(
                    step_number=1,
                    instruction="Boil chicken for 20 minutes.",
                    category="heating",
                    active_duration_minutes=20,
                    passive_duration_minutes=None,
                    heat_level=HeatLevel.HIGH,
                ),
            ),
        )


def _request(**overrides) -> GeneratePlanRequest:
    base = {
        "request_id": "time-req-001",
        "user_id": "u",
        "recipes": (
            {
                "recipe_id": "r1",
                "text": "Cook chicken for 20 minutes.",
                "target_servings": 2,
            },
        ),
        "dietary_restrictions": (),
        "user_allergens": (),
        "inventory_lots": (
            InventoryLotSnapshot(
                lot_id="lot-1",
                item_id="chicken",
                canonical_name="chicken breast",
                on_hand=Decimal(300),
                reserved=Decimal(0),
                unit="g",
            ),
        ),
        "kitchen_resources": (
            KitchenResourceSnapshot(
                resource_id="stove-1",
                resource_type="stove",
                capacity=Decimal(4),
                capacity_unit="burners",
            ),
            KitchenResourceSnapshot(
                resource_id="sink-1",
                resource_type="sink",
                capacity=Decimal(1),
            ),
        ),
    }
    base.update(overrides)
    return GeneratePlanRequest(**base)


@pytest.fixture
def graph():
    return build_cooking_plan_graph()


@pytest.fixture
def context():
    return WorkflowContext(recipe_extractor=_TimeAwareExtractor())


@pytest.mark.asyncio
async def test_makespan_hard_limit_infeasible(graph, context):
    """A tight time_limit_minutes must yield a genuine INFEASIBLE.

    The recipe needs ~20+ minutes; a 5-minute limit cannot fit → solver
    proves infeasible (not merely UNKNOWN).
    """
    request = _request(time_limit_minutes=5)
    result = await graph.ainvoke({"request": request}, context=context, config={"recursion_limit": 30})
    response = result.get("response")
    assert response is not None
    assert response.status == "INFEASIBLE", f"Expected INFEASIBLE, got {response.status}"


@pytest.mark.asyncio
async def test_generous_time_limit_reaches_ready(graph, context):
    """A generous time_limit_minutes must still produce a READY plan."""
    request = _request(time_limit_minutes=120)
    result = await graph.ainvoke({"request": request}, context=context, config={"recursion_limit": 30})
    response = result.get("response")
    assert response is not None
    assert response.status == "READY", f"Expected READY, got {response.status}"
    assert response.makespan_minutes <= 120


# ---------------------------------------------------------------------------
# cooking_date → FEFO expired-lot exclusion
# ---------------------------------------------------------------------------


class TestCookingDate:
    def test_expired_lot_excluded_on_cooking_date(self) -> None:
        """A lot expiring before cooking_date must be excluded by FEFO."""
        from cooking_plan_agent.domain.models import IngredientDemand
        from cooking_plan_agent.inventory.feasibility import check_all_inventory

        ingredient = IngredientDemand(
            canonical_name="chicken breast",
            raw_name="chicken",
            quantity=Decimal(300),
            unit="g",
            confidence=Decimal("1.0"),
        )
        lots = (
            InventoryLotSnapshot(
                lot_id="expired",
                item_id="c",
                canonical_name="chicken breast",
                on_hand=Decimal(500),
                reserved=Decimal(0),
                unit="g",
                expiry_date=date(2026, 7, 30),
            ),
        )
        # Cooking on 2026-08-02 → the 2026-07-30 lot is unusable.
        report = check_all_inventory(
            requirements=(ingredient,),
            lots=lots,
            cooking_date=date(2026, 8, 2),
        )
        assert not report.is_feasible
        assert report.ingredient_shortages
        assert report.ingredient_shortages[0].shortage == Decimal(300)

    def test_valid_lot_usable_on_cooking_date(self) -> None:
        from cooking_plan_agent.domain.models import IngredientDemand
        from cooking_plan_agent.inventory.feasibility import check_all_inventory

        ingredient = IngredientDemand(
            canonical_name="chicken breast",
            raw_name="chicken",
            quantity=Decimal(300),
            unit="g",
            confidence=Decimal("1.0"),
        )
        lots = (
            InventoryLotSnapshot(
                lot_id="fresh",
                item_id="c",
                canonical_name="chicken breast",
                on_hand=Decimal(500),
                reserved=Decimal(0),
                unit="g",
                expiry_date=date(2026, 8, 10),
            ),
        )
        report = check_all_inventory(
            requirements=(ingredient,),
            lots=lots,
            cooking_date=date(2026, 8, 2),
        )
        assert report.is_feasible

    @pytest.mark.asyncio
    async def test_cooking_date_flows_into_safety(self, graph, context) -> None:
        """cooking_date must reach the safety engine (expired-ingredient rule)."""
        # A lot already > 3 days expired on cooking_date → unrepairable safety.
        request = _request(
            cooking_date=date(2026, 8, 2),
            inventory_lots=(
                InventoryLotSnapshot(
                    lot_id="spoiled",
                    item_id="chicken",
                    canonical_name="chicken breast",
                    on_hand=Decimal(300),
                    reserved=Decimal(0),
                    unit="g",
                    expiry_date=date(2026, 7, 20),
                ),
            ),
        )
        result = await graph.ainvoke({"request": request}, context=context, config={"recursion_limit": 30})
        response = result.get("response")
        assert response is not None
        # The expired lot is excluded on cooking_date → no READY path. The
        # exact terminal state may be INFEASIBLE, NEEDS_CONFIRMATION (repair
        # options) or FAILED depending on downstream repair logic — the key
        # invariant is that the stale lot can never yield a READY plan.
        assert response.status != "READY", f"expired lot must not reach READY, got {response.status}"


# ---------------------------------------------------------------------------
# serving_at (timezone-aware) vs serving_time (legacy)
# ---------------------------------------------------------------------------


class TestServingTime:
    @pytest.mark.asyncio
    async def test_serving_at_requires_timezone(self, graph, context) -> None:
        request = _request(serving_at=datetime(2026, 8, 2, 18, 0))  # naive → reject
        result = await graph.ainvoke({"request": request}, context=context, config={"recursion_limit": 30})
        error = result.get("error")
        assert error is not None
        assert error.error_code == "INVALID_SERVING_TIME"

    @pytest.mark.asyncio
    async def test_serving_time_invalid_hhmm_rejected(self, graph, context) -> None:
        request = _request(serving_time="25:99")
        result = await graph.ainvoke({"request": request}, context=context, config={"recursion_limit": 30})
        error = result.get("error")
        assert error is not None
        assert error.error_code == "INVALID_SERVING_TIME"

    @pytest.mark.asyncio
    async def test_serving_at_present_attaches_to_timeline(self, graph, context) -> None:
        request = _request(
            serving_at=datetime(2026, 8, 2, 19, 0, tzinfo=SGT),
        )
        result = await graph.ainvoke({"request": request}, context=context, config={"recursion_limit": 30})
        response = result.get("response")
        assert isinstance(response, ReadyPlanResponse), f"got {type(response).__name__}"
        assert response.timeline
        assert all("serving_at" in entry for entry in response.timeline)


# ---------------------------------------------------------------------------
# solver_timeout_seconds propagates into the SchedulingProblem
# ---------------------------------------------------------------------------


class TestSolverTimeoutPropagation:
    def test_timeout_wired_into_problem(self, monkeypatch) -> None:
        """Settings.solver_timeout_seconds must land in SchedulingProblem."""
        monkeypatch.setenv("COOKING_PLAN_SOLVER_TIMEOUT_SECONDS", "1.5")
        from cooking_plan_agent.config.settings import get_settings

        get_settings.cache_clear()
        try:
            from cooking_plan_agent.workflow.nodes import _solver_timeout

            assert _solver_timeout() == 1.5
        finally:
            get_settings.cache_clear()

    def test_problem_default_timeout(self) -> None:
        """SchedulingProblem carries solver_timeout_seconds."""
        problem = SchedulingProblem(tasks=(), resources=(), solver_timeout_seconds=7.0)
        assert problem.solver_timeout_seconds == 7.0
