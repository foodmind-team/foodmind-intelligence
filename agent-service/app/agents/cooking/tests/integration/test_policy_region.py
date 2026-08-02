"""P3-04 integration: regional policy through the full graph.

Verifies (plan verification section):
  - unknown region → FAILED (SAFETY_POLICY_UNAVAILABLE) — never READY
  - a valid region (SG) produces READY whose response records the policy
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from cooking_plan_agent.domain.enums import HeatLevel
from cooking_plan_agent.domain.models import (
    ExtractedIngredient,
    ExtractedRecipeCandidate,
    ExtractedStep,
    FailedPlanResponse,
    GeneratePlanRequest,
    InventoryLotSnapshot,
    KitchenResourceSnapshot,
    ReadyPlanResponse,
)
from cooking_plan_agent.workflow.context import WorkflowContext
from cooking_plan_agent.workflow.graph import build_cooking_plan_graph
from cooking_plan_agent.workflow.state import PlanState


class FakeRecipeExtractor:
    """Returns a gap-free candidate so the happy path reaches READY."""

    async def extract(self, source_text: str) -> ExtractedRecipeCandidate:
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


@pytest.fixture
def graph():
    return build_cooking_plan_graph()


@pytest.fixture
def context():
    return WorkflowContext(recipe_extractor=FakeRecipeExtractor())


def _base_request(region: str | None) -> GeneratePlanRequest:
    return GeneratePlanRequest(
        request_id="test-req-policy",
        user_id="test-user",
        recipes=(
            {
                "recipe_id": "r1",
                "text": "Cook chicken for 10 minutes. Serves 2.",
                "target_servings": 2,
            },
        ),
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
        region=region,
    )


@pytest.mark.asyncio
async def test_unknown_region_never_reaches_ready(graph, context) -> None:
    """Unknown region → FAILED with the stable policy error code."""
    initial_state: PlanState = {"request": _base_request(region="XX")}

    result = await graph.ainvoke(
        initial_state,
        context=context,
        config={"recursion_limit": 30},
    )

    response = result.get("response")
    assert isinstance(response, FailedPlanResponse), f"Expected FAILED, got {type(response).__name__}"
    assert response.status == "FAILED"
    assert response.error_code == "SAFETY_POLICY_UNAVAILABLE"


@pytest.mark.asyncio
async def test_sg_region_reads_ready_and_records_policy(graph, context) -> None:
    """A valid region produces READY carrying the policy provenance."""
    initial_state: PlanState = {"request": _base_request(region="SG")}

    result = await graph.ainvoke(
        initial_state,
        context=context,
        config={"recursion_limit": 30},
    )

    response = result.get("response")
    assert isinstance(response, ReadyPlanResponse), f"Expected READY, got {type(response).__name__}"
    assert response.status == "READY"
    assert response.safety_policy is not None
    assert response.safety_policy.region == "SG"
    assert response.safety_policy.version == "1.0"
    assert response.safety_policy.sources


@pytest.mark.asyncio
async def test_us_default_region_records_policy(graph, context) -> None:
    """No request region → deployment default (US) recorded on the response."""
    initial_state: PlanState = {"request": _base_request(region=None)}

    result = await graph.ainvoke(
        initial_state,
        context=context,
        config={"recursion_limit": 30},
    )

    response = result.get("response")
    assert isinstance(response, ReadyPlanResponse), f"Expected READY, got {type(response).__name__}"
    assert response.safety_policy is not None
    assert response.safety_policy.region == "US"
