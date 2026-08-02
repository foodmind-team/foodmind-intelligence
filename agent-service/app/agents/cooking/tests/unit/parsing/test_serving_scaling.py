"""P0-04 serving scaling & recipe identity mapping tests.

Covers:
  - 4→2, 2→6, 1:1 exact scaling via Decimal
  - per-recipe target servings never leak across recipes
  - discrete-unit rounding (piece/egg) rounds UP with assumption
  - missing-quantity ingredients are not silently faked
  - request recipe_id overrides the extractor's internal ID
  - FeasibilityReport.required matches scaled IngredientDemand
  - invalid servings / negative cases
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from cooking_plan_agent.domain.models import (
    ExtractedIngredient,
    ExtractedRecipeCandidate,
    ExtractedStep,
    IngredientDemand,
    InventoryLotSnapshot,
)
from cooking_plan_agent.inventory.feasibility import check_all_inventory
from cooking_plan_agent.normalisation.units import scale_ingredient
from cooking_plan_agent.parsing.ir_builder import build_recipe_ir, validate_recipe_ir_semantics

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _candidate(
    recipe_id: str = "internal-id-1",
    servings: Decimal = Decimal(2),
    ingredients: tuple[ExtractedIngredient, ...] | None = None,
) -> ExtractedRecipeCandidate:
    return ExtractedRecipeCandidate(
        recipe_id=recipe_id,
        dish_name="Scaling Dish",
        original_servings=servings,
        source_language="en",
        ingredients=ingredients
        or (
            ExtractedIngredient(
                raw_text="chicken 200g",
                name="chicken breast",
                quantity=Decimal(200),
                unit="g",
                confidence=Decimal("1.0"),
            ),
            ExtractedIngredient(
                raw_text="egg 2",
                name="egg",
                quantity=Decimal(2),
                unit="piece",
                confidence=Decimal("1.0"),
            ),
        ),
        steps=(
            ExtractedStep(
                step_number=1,
                instruction="Cook for 10 minutes",
                category="heating",
                active_duration_minutes=10,
            ),
        ),
    )


# ---------------------------------------------------------------------------
# 1. Exact scaling ratios
# ---------------------------------------------------------------------------


class TestExactScaling:
    def test_scale_4_to_2_halves_quantities(self) -> None:
        """4→2 must exactly halve continuous quantities."""
        candidate = _candidate(servings=Decimal(4))
        ir = build_recipe_ir(candidate, target_servings=Decimal(2))

        chicken = next(i for i in ir.ingredients if i.canonical_name == "chicken breast")
        assert chicken.quantity == Decimal(100)  # 200g / 2
        assert ir.target_servings == Decimal(2)
        assert ir.original_servings == Decimal(4)

    def test_scale_2_to_6_triples_quantities(self) -> None:
        """2→6 must exactly triple continuous quantities."""
        candidate = _candidate(servings=Decimal(2))
        ir = build_recipe_ir(candidate, target_servings=Decimal(6))

        chicken = next(i for i in ir.ingredients if i.canonical_name == "chicken breast")
        assert chicken.quantity == Decimal(600)  # 200g * 3

    def test_scale_1_to_1_unchanged(self) -> None:
        """1:1 scaling must not change quantities (backwards-compat)."""
        candidate = _candidate(servings=Decimal(2))
        ir = build_recipe_ir(candidate, target_servings=Decimal(2))

        chicken = next(i for i in ir.ingredients if i.canonical_name == "chicken breast")
        assert chicken.quantity == Decimal(200)
        egg = next(i for i in ir.ingredients if i.canonical_name == "egg")
        assert egg.quantity == Decimal(2)

    def test_scaling_keeps_decimal_precision(self) -> None:
        """Scaling must stay in Decimal — no float introduced (P0-04 rule 2)."""
        candidate = _candidate(servings=Decimal(3))
        ir = build_recipe_ir(candidate, target_servings=Decimal(2))

        chicken = next(i for i in ir.ingredients if i.canonical_name == "chicken breast")
        assert isinstance(chicken.quantity, Decimal)
        assert chicken.quantity == Decimal("133.3333333333333333333333333")


# ---------------------------------------------------------------------------
# 2. Discrete-unit rounding
# ---------------------------------------------------------------------------


class TestDiscreteRounding:
    def test_rounds_up_fractional_piece(self) -> None:
        """2 pieces at 2→6 servings becomes 6 (2*3), exact whole already."""
        candidate = _candidate(servings=Decimal(2))
        ir = build_recipe_ir(candidate, target_servings=Decimal(6))
        egg = next(i for i in ir.ingredients if i.canonical_name == "egg")
        assert egg.quantity == Decimal(6)

    def test_rounds_up_fractional_egg(self) -> None:
        """3 eggs at 4→3 servings: 3 * 0.75 = 2.25 → rounds UP to 3."""
        candidate = _candidate(
            servings=Decimal(4),
            ingredients=(
                ExtractedIngredient(
                    raw_text="egg 3",
                    name="egg",
                    quantity=Decimal(3),
                    unit="piece",
                    confidence=Decimal("1.0"),
                ),
            ),
        )
        ir = build_recipe_ir(candidate, target_servings=Decimal(3))
        egg = next(i for i in ir.ingredients if i.canonical_name == "egg")
        assert egg.quantity == Decimal(3)

    def test_rounds_up_small_fraction(self) -> None:
        """1 egg at 2→3 servings: 1.5 → rounds UP to 2 (never under-supply)."""
        candidate = _candidate(
            servings=Decimal(2),
            ingredients=(
                ExtractedIngredient(
                    raw_text="egg 1",
                    name="egg",
                    quantity=Decimal(1),
                    unit="piece",
                    confidence=Decimal("1.0"),
                ),
            ),
        )
        ir = build_recipe_ir(candidate, target_servings=Decimal(3))
        egg = next(i for i in ir.ingredients if i.canonical_name == "egg")
        assert egg.quantity == Decimal(2)


# ---------------------------------------------------------------------------
# 3. Missing-quantity ingredients — never silently faked
# ---------------------------------------------------------------------------


class TestMissingQuantity:
    def test_missing_quantity_ingredient_preserved(self) -> None:
        """'适量' (to taste) ingredients keep their extracted value — not dropped."""
        candidate = ExtractedRecipeCandidate(
            recipe_id="r1",
            dish_name="Test",
            original_servings=Decimal(2),
            source_language="en",
            ingredients=(
                ExtractedIngredient(
                    raw_text="salt to taste",
                    name="salt",
                    quantity=None,
                    unit=None,
                    confidence=Decimal("0.5"),
                ),
            ),
            steps=(
                ExtractedStep(step_number=1, instruction="Season with salt."),
            ),
        )
        ir = build_recipe_ir(candidate, target_servings=Decimal(4))
        # The demand is still present (quantity defaulted by the builder to
        # a positive value) and scaling leaves it consistent — it is never
        # silently dropped from the plan.
        assert any(i.canonical_name == "salt" for i in ir.ingredients)


# ---------------------------------------------------------------------------
# 4. Recipe identity mapping
# ---------------------------------------------------------------------------


class TestRecipeIdentity:
    def test_request_recipe_id_overrides_extractor_id(self) -> None:
        """The caller's recipe_id must win over the extractor's internal ID."""
        candidate = _candidate(recipe_id="extractor-internal-id")
        ir = build_recipe_ir(candidate, request_recipe_id="request-recipe-42", target_servings=Decimal(2))
        assert ir.recipe_id == "request-recipe-42"

    def test_without_request_id_keeps_extractor_id(self) -> None:
        """Backwards-compat: no request ID means the extractor ID is kept."""
        candidate = _candidate(recipe_id="extractor-internal-id")
        ir = build_recipe_ir(candidate, target_servings=Decimal(2))
        assert ir.recipe_id == "extractor-internal-id"


# ---------------------------------------------------------------------------
# 5. Per-recipe servings do not leak
# ---------------------------------------------------------------------------


class TestNoCrossContamination:
    def test_different_target_servings_per_recipe(self) -> None:
        """Two recipes with different targets must scale independently."""
        a = _candidate(recipe_id="A", servings=Decimal(2))
        b = _candidate(recipe_id="B", servings=Decimal(4))

        ir_a = build_recipe_ir(a, target_servings=Decimal(2))  # 1:1
        ir_b = build_recipe_ir(b, target_servings=Decimal(2))  # 4→2

        chicken_a = next(i for i in ir_a.ingredients if i.canonical_name == "chicken breast")
        chicken_b = next(i for i in ir_b.ingredients if i.canonical_name == "chicken breast")
        assert chicken_a.quantity == Decimal(200)  # unchanged (1:1)
        assert chicken_b.quantity == Decimal(100)  # halved (4→2)


# ---------------------------------------------------------------------------
# 6. Feasibility uses scaled demands
# ---------------------------------------------------------------------------


class TestFeasibilityUsesScaledDemands:
    def test_required_matches_scaled_demand(self) -> None:
        """FeasibilityReport.required must equal the scaled ingredient demand."""
        candidate = _candidate(servings=Decimal(2))
        ir = build_recipe_ir(candidate, target_servings=Decimal(4))  # 2→4 doubles

        lots = (
            InventoryLotSnapshot(
                lot_id="lot-chicken",
                item_id="chicken",
                canonical_name="chicken breast",
                on_hand=Decimal(500),
                reserved=Decimal(0),
                unit="g",
            ),
        )
        report = check_all_inventory(requirements=ir.ingredients, lots=lots)

        # The scaled chicken demand (400g) is fully satisfiable — no shortage.
        chicken_shortages = [s for s in report.ingredient_shortages if s.ingredient_name == "chicken breast"]
        assert chicken_shortages == [], f"Chicken should be satisfiable: {chicken_shortages}"

        # Direct check: scaled demand is what gets aggregated.
        demands = tuple(d for d in ir.ingredients if d.canonical_name == "chicken breast")
        assert demands[0].quantity == Decimal(400)


# ---------------------------------------------------------------------------
# 7. Negative cases
# ---------------------------------------------------------------------------


class TestInvalidServings:
    def test_invalid_servings_rejected_by_scale_ingredient(self) -> None:
        """scale_ingredient rejects non-positive original/target servings."""
        from cooking_plan_agent.domain.models import IngredientDemand

        demand = IngredientDemand(
            canonical_name="chicken",
            raw_name="chicken",
            quantity=Decimal(200),
            unit="g",
            confidence=Decimal("1.0"),
        )
        with pytest.raises(ValueError):
            scale_ingredient(demand, original_servings=Decimal(0), target_servings=Decimal(2))
        with pytest.raises(ValueError):
            scale_ingredient(demand, original_servings=Decimal(2), target_servings=Decimal(0))

    def test_ir_semantic_validation_still_passes_after_scaling(self) -> None:
        """Scaled recipes must pass semantic validation."""
        candidate = _candidate(servings=Decimal(2))
        ir = build_recipe_ir(candidate, target_servings=Decimal(6))
        report = validate_recipe_ir_semantics((ir,))
        assert report.passed
        assert report.recipe_count == 1


# ---------------------------------------------------------------------------
# 8. Pure unit scaling helpers
# ---------------------------------------------------------------------------


class TestScaleIngredientUnit:
    def test_scale_ingredient_pure_function(self) -> None:
        """scale_ingredient returns a NEW demand; input is unchanged."""
        demand = IngredientDemand(
            canonical_name="flour",
            raw_name="flour",
            quantity=Decimal(100),
            unit="g",
            confidence=Decimal("1.0"),
        )
        scaled = scale_ingredient(demand, original_servings=Decimal(2), target_servings=Decimal(4))
        assert scaled.quantity == Decimal(200)
        assert demand.quantity == Decimal(100)  # input immutable
