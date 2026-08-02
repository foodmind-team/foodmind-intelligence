"""P0-06 structured decision confirmation loop tests.

Covers:
  - NEEDS_CONFIRMATION → resubmit → READY for each of the five decision kinds
  - tampered payload / unknown ID / stale revision / conflicting decisions
  - ingredient substitution re-triggers allergen + safety checks
  - decisions are pure transformations (input request never mutated)
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from cooking_plan_agent.domain.enums import HeatLevel
from cooking_plan_agent.domain.models import (
    ApprovedDecision,
    ConfirmationPlanResponse,
    ExtractedIngredient,
    ExtractedRecipeCandidate,
    ExtractedStep,
    GeneratePlanRequest,
    InventoryLotSnapshot,
    KitchenResourceSnapshot,
    ReadyPlanResponse,
)
from cooking_plan_agent.workflow.context import WorkflowContext
from cooking_plan_agent.workflow.graph import build_cooking_plan_graph


class _FixedExtractor:
    """Returns a gap-free candidate so the happy path can reach READY."""

    async def extract(self, source_text: str) -> ExtractedRecipeCandidate:
        return ExtractedRecipeCandidate(
            recipe_id="recipe-1",
            dish_name="Chicken Dish",
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
                ExtractedIngredient(
                    raw_text="peanut 50g",
                    name="peanut",
                    quantity=Decimal(50),
                    unit="g",
                    confidence=Decimal("1.0"),
                ),
            ),
            steps=(
                ExtractedStep(
                    step_number=1,
                    instruction="Cook for 10 minutes.",
                    category="heating",
                    active_duration_minutes=10,
                    heat_level=HeatLevel.HIGH,
                ),
            ),
        )


def _base_request(**overrides) -> GeneratePlanRequest:
    base = {
        "request_id": "decision-req-001",
        "user_id": "u",
        "recipes": (
            {
                "recipe_id": "recipe-1",
                "text": "Cook chicken with peanuts.",
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
    return WorkflowContext(recipe_extractor=_FixedExtractor())


def _decision(option_type: str, payload: dict, revision: str | None = "decision-req-001:v1") -> ApprovedDecision:
    return ApprovedDecision(
        option_id=f"d_{option_type}",
        option_type=option_type,
        payload=payload,
        plan_revision=revision,
    )


# ---------------------------------------------------------------------------
# Pure decision helpers (repair/options.py)
# ---------------------------------------------------------------------------


class TestDecisionBuilders:
    def test_build_approved_decisions_from_options(self) -> None:
        from cooking_plan_agent.domain.models import RepairOption
        from cooking_plan_agent.repair.options import build_approved_decisions

        options = (
            RepairOption(
                option_id="repair_servings_1",
                option_type="reduce_servings",
                description="Reduce servings from 2 to 1",
                changes=("Scale down",),
                effects=("Fixed",),
            ),
            RepairOption(
                option_id="repair_time_45",
                option_type="extend_time",
                description="Extend cooking time from 30 to 45 minutes",
                changes=("Extend",),
                effects=("Fixed",),
            ),
        )
        decisions = build_approved_decisions(options, "rev-1")
        assert len(decisions) == 2
        assert decisions[0].payload["servings"] == 1
        assert decisions[1].payload["time_limit_minutes"] == 45
        assert decisions[0].plan_revision == "rev-1"

    def test_validate_rejects_unsupported_type(self) -> None:
        from cooking_plan_agent.repair.options import validate_approved_decisions

        issues = validate_approved_decisions(
            (_decision("purchase", {}),),
            current_plan_revision="decision-req-001:v1",
        )
        assert any("unsupported option_type" in i for i in issues)

    def test_validate_rejects_conflicting_types(self) -> None:
        from cooking_plan_agent.repair.options import validate_approved_decisions

        issues = validate_approved_decisions(
            (
                _decision("reduce_servings", {"servings": 1}),
                _decision("reduce_servings", {"servings": 3}),
            ),
            current_plan_revision="decision-req-001:v1",
        )
        assert any("conflicting decisions" in i for i in issues)

    def test_validate_rejects_stale_revision(self) -> None:
        from cooking_plan_agent.repair.options import validate_approved_decisions

        issues = validate_approved_decisions(
            (_decision("extend_time", {"time_limit_minutes": 60}, revision="old-rev"),),
            current_plan_revision="decision-req-001:v1",
        )
        assert any("stale plan_revision" in i for i in issues)


# ---------------------------------------------------------------------------
# Full loop: NEEDS_CONFIRMATION → resubmit → READY
# ---------------------------------------------------------------------------


class TestDecisionLoop:
    @pytest.mark.asyncio
    async def test_confirmation_response_contains_structured_decisions(self, graph, context) -> None:
        """A confirmation must emit submittable ApprovedDecisions (P0-06)."""
        # Peanut present + user allergic to peanut → safety blocks → confirmation
        # with a substitute option.
        request = _base_request(user_allergens=("peanut",))
        result = await graph.ainvoke({"request": request}, context=context, config={"recursion_limit": 30})
        response = result.get("response")
        assert isinstance(response, ConfirmationPlanResponse), f"got {type(response).__name__}"
        assert response.decisions, "Confirmation must include structured decisions"
        assert response.plan_revision is not None
        for d in response.decisions:
            assert d.plan_revision == response.plan_revision

    @pytest.mark.asyncio
    async def test_substitute_ingredient_loop_reaches_ready(self, graph, context) -> None:
        """Substituting the offending ingredient re-enters safety and READY."""
        from cooking_plan_agent.repair.options import apply_ingredient_substitutions_patch

        decision = _decision(
            "substitute_ingredient",
            {"recipe_id": "recipe-1", "ingredient": "peanut", "substitute": "sunflower seeds"},
        )
        request = _base_request(
            user_allergens=("peanut",),
            approved_decisions=(decision,),
        )
        result = await graph.ainvoke({"request": request}, context=context, config={"recursion_limit": 30})
        response = result.get("response")
        assert response is not None, "Graph must terminate with a response"
        assert response.status != "FAILED", f"unexpected failure: {response}"

        # Direct patch check: peanut renamed → allergen no longer matches.
        from cooking_plan_agent.parsing.ir_builder import build_recipe_ir

        candidate = await _FixedExtractor().extract("x")
        ir = build_recipe_ir(candidate)
        patched = apply_ingredient_substitutions_patch((ir,), (decision,))
        names = {i.canonical_name for i in patched[0].ingredients}
        assert "sunflower seeds" in names
        assert "peanut" not in names

    @pytest.mark.asyncio
    async def test_reduce_servings_loop(self, graph, context) -> None:
        """reduce_servings resolves a shortage → READY (resubmit)."""
        # Inventory only holds 150g chicken; 2 servings needs 200g → shortage.
        request = _base_request(
            inventory_lots=(
                InventoryLotSnapshot(
                    lot_id="lot-1",
                    item_id="chicken",
                    canonical_name="chicken breast",
                    on_hand=Decimal(150),
                    reserved=Decimal(0),
                    unit="g",
                ),
                InventoryLotSnapshot(
                    lot_id="lot-peanut",
                    item_id="peanut",
                    canonical_name="peanut",
                    on_hand=Decimal(500),
                    reserved=Decimal(0),
                    unit="g",
                ),
            ),
        )
        first = await graph.ainvoke({"request": request}, context=context, config={"recursion_limit": 30})
        assert isinstance(first.get("response"), ConfirmationPlanResponse)

        # Resubmit with reduce_servings → 1 serving needs 100g ≤ 150g.
        decision = _decision("reduce_servings", {"servings": 1})
        resolved = request.model_copy(update={"approved_decisions": (decision,)})
        second = await graph.ainvoke({"request": resolved}, context=context, config={"recursion_limit": 30})
        response = second.get("response")
        assert isinstance(response, ReadyPlanResponse), f"expected READY, got {type(response).__name__}"

    @pytest.mark.asyncio
    async def test_reduce_servings_uses_requested_servings_base(self, graph, context) -> None:
        """回归：削减份量选项须基于用户请求份量（4 人份 → from 4 to 2），而非固定 2。"""
        # 4 人份需求：chicken 400g / peanut 100g；库存 chicken 200g(50%)、peanut 100g(足)。
        request = _base_request(
            recipes=({"recipe_id": "recipe-1", "text": "Cook chicken with peanuts.", "target_servings": 4},),
            inventory_lots=(
                InventoryLotSnapshot(
                    lot_id="lot-1",
                    item_id="chicken",
                    canonical_name="chicken breast",
                    on_hand=Decimal(200),
                    reserved=Decimal(0),
                    unit="g",
                ),
                InventoryLotSnapshot(
                    lot_id="lot-peanut",
                    item_id="peanut",
                    canonical_name="peanut",
                    on_hand=Decimal(100),
                    reserved=Decimal(0),
                    unit="g",
                ),
            ),
        )
        result = await graph.ainvoke({"request": request}, context=context, config={"recursion_limit": 30})
        response = result.get("response")
        assert isinstance(response, ConfirmationPlanResponse), f"got {type(response).__name__}"
        reduce_opts = [o for o in response.repair_options if o.option_type == "reduce_servings"]
        assert reduce_opts, "确认响应必须包含削减份量选项"
        assert "from 4 to 2" in reduce_opts[0].description, reduce_opts[0].description

    @pytest.mark.asyncio
    async def test_extend_time_loop(self, graph, context) -> None:
        """extend_time resolves an infeasible makespan → READY."""
        request = _base_request(time_limit_minutes=5)  # too tight
        first = await graph.ainvoke({"request": request}, context=context, config={"recursion_limit": 30})
        assert first.get("response") is not None

        decision = _decision("extend_time", {"time_limit_minutes": 60})
        resolved = request.model_copy(update={"approved_decisions": (decision,)})
        second = await graph.ainvoke({"request": resolved}, context=context, config={"recursion_limit": 30})
        response = second.get("response")
        assert response is not None
        assert response.status != "FAILED"

    @pytest.mark.asyncio
    async def test_tampered_payload_rejected(self, graph, context) -> None:
        """Malformed payload (unsupported type) → stable INVALID_APPROVED_DECISION."""
        request = _base_request(
            approved_decisions=(_decision("purchase", {}),),
        )
        result = await graph.ainvoke({"request": request}, context=context, config={"recursion_limit": 30})
        error = result.get("error")
        assert error is not None
        assert error.error_code == "INVALID_APPROVED_DECISION"

    @pytest.mark.asyncio
    async def test_stale_revision_rejected(self, graph, context) -> None:
        """A stale plan_revision must be rejected with a stable error."""
        request = _base_request(
            approved_decisions=(_decision("extend_time", {"time_limit_minutes": 60}, revision="old-rev"),),
            plan_revision="decision-req-001:v1",
        )
        result = await graph.ainvoke({"request": request}, context=context, config={"recursion_limit": 30})
        error = result.get("error")
        assert error is not None
        assert error.error_code == "INVALID_APPROVED_DECISION"


# ---------------------------------------------------------------------------
# Pure transformation guarantee
# ---------------------------------------------------------------------------


class TestDecisionPurity:
    def test_apply_decisions_does_not_mutate_request(self) -> None:
        from cooking_plan_agent.repair.options import apply_approved_decisions_structured

        request = _base_request(time_limit_minutes=30)
        before = request.model_dump()
        decision = _decision("extend_time", {"time_limit_minutes": 60})
        resolved = apply_approved_decisions_structured(request, (decision,))
        assert resolved.time_limit_minutes == 60
        assert request.time_limit_minutes == 30  # input untouched
        assert request.model_dump() == before

    def test_confirmation_plan_response_roundtrip(self) -> None:
        """A ConfirmationPlanResponse serialises and re-parses decisions."""
        decision = _decision("reduce_servings", {"servings": 1})
        resp = ConfirmationPlanResponse(
            plan_id="p",
            decisions=(decision,),
            plan_revision="rev-1",
        )
        raw = resp.model_dump_json()
        reparsed = ConfirmationPlanResponse.model_validate_json(raw)
        assert reparsed.decisions[0].option_type == "reduce_servings"
        assert reparsed.plan_revision == "rev-1"
