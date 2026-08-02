"""Unit tests for the safety rule engine.

Handbook 11.3: cover each safety rule in isolation and the engine
composition. Uses table-driven tests for repeatable coverage.

Tests are organised by rule:
  1. CrossContaminationRule
  2. AllergenDetectionRule
  3. ProteinSafetyTemperatureRule
  4. DietaryCompatibilityRule
  5. SafetyEngine composition
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from cooking_plan_agent.domain.enums import HeatLevel
from cooking_plan_agent.domain.models import (
    IngredientDemand,
    InventoryLotSnapshot,
    RecipeIR,
    RecipeStep,
    SafetyContext,
    SafetyReport,
)
from cooking_plan_agent.safety.engine import SafetyEngine
from cooking_plan_agent.safety.rules import (
    AllergenDetectionRule,
    CrossContaminationRule,
    DietaryCompatibilityRule,
    ExpiredIngredientRule,
    HoldingTimeRule,
    ProteinSafetyTemperatureRule,
)

# =============================================================================
# Shared test helpers
# =============================================================================


def _make_ingredient(
    canonical_name: str,
    raw_name: str = "",
    input_state: str = "raw",
    allergen_tags: tuple[str, ...] = (),
) -> IngredientDemand:
    """Factory for an IngredientDemand with minimal required fields."""
    return IngredientDemand(
        canonical_name=canonical_name,
        raw_name=raw_name or canonical_name,
        quantity=Decimal(500),
        unit="g",
        input_state=input_state,
        allergen_tags=allergen_tags,
        confidence=Decimal("1.0"),
    )


def _make_step(
    step_number: int,
    instruction: str,
    category: str = "heating",
    heat_level: HeatLevel = HeatLevel.MEDIUM,
    target_temperature_c: Decimal | None = None,
    passive_duration_minutes: int | None = None,
) -> RecipeStep:
    """Factory for a RecipeStep with minimal required fields."""
    return RecipeStep(
        step_number=step_number,
        instruction=instruction,
        category=category,
        heat_level=heat_level,
        target_temperature_c=target_temperature_c,
        passive_duration_minutes=passive_duration_minutes,
    )


def _make_recipe(
    recipe_id: str,
    dish_name: str,
    ingredients: tuple[IngredientDemand, ...] = (),
    steps: tuple[RecipeStep, ...] = (),
) -> RecipeIR:
    """Factory for a RecipeIR with sensible defaults."""
    return RecipeIR(
        recipe_id=recipe_id,
        dish_name=dish_name,
        original_servings=Decimal(2),
        target_servings=Decimal(2),
        source_language="zh",
        ingredients=ingredients or (_make_ingredient("rice"),),
        steps=steps or (_make_step(1, "Cook rice"),),
    )


def _make_context(
    recipes: tuple[RecipeIR, ...] = (),
    dietary_restrictions: tuple[str, ...] = (),
    user_allergens: tuple[str, ...] = (),
) -> SafetyContext:
    """Factory for a SafetyContext with defaults."""
    if not recipes:
        recipes = (_make_recipe("r1", "Test Dish"),)
    return SafetyContext(
        recipes=recipes,
        dietary_restrictions=dietary_restrictions,
        user_allergens=user_allergens,
    )


# =============================================================================
# 1. CrossContaminationRule
# =============================================================================


class TestCrossContaminationRule:
    """Cross-contamination: raw protein + ready-to-eat steps."""

    def test_no_raw_protein_returns_none(self):
        """Recipe without raw protein → no finding."""
        rule = CrossContaminationRule()
        recipe = _make_recipe(
            "r1",
            "Vegetable Stir-Fry",
            ingredients=(_make_ingredient("broccoli"), _make_ingredient("carrot")),
            steps=(_make_step(1, "Stir-fry vegetables"),),
        )
        ctx = _make_context(recipes=(recipe,))

        result = rule.evaluate(ctx)
        assert result is None

    def test_raw_protein_no_rte_returns_none(self):
        """Recipe with raw protein but no RTE step → no finding."""
        rule = CrossContaminationRule()
        recipe = _make_recipe(
            "r1",
            "Pan-Seared Chicken",
            ingredients=(_make_ingredient("chicken breast", input_state="raw"),),
            steps=(_make_step(1, "Sear chicken in pan"),),
        )
        ctx = _make_context(recipes=(recipe,))

        result = rule.evaluate(ctx)
        assert result is None

    def test_raw_protein_with_rte_step_returns_finding(self):
        """Raw protein + RTE step (e.g., plating) → cross-contamination finding."""
        rule = CrossContaminationRule()
        recipe = _make_recipe(
            "r1",
            "Chicken Salad",
            ingredients=(
                _make_ingredient("chicken breast", input_state="raw"),
                _make_ingredient("lettuce", input_state="raw"),
            ),
            steps=(
                _make_step(1, "Cook chicken", category="heating"),
                _make_step(2, "Plate the salad", category="plating"),
            ),
        )
        ctx = _make_context(recipes=(recipe,))

        result = rule.evaluate(ctx)
        assert result is not None
        assert result.rule_id == "SAFETY_CROSS_CONTAMINATION"
        assert result.severity == "hard_repairable"
        assert "chicken breast" in result.affected_ingredient_names


# =============================================================================
# 2. AllergenDetectionRule
# =============================================================================


class TestAllergenDetectionRule:
    """Allergen detection against user-declared allergens."""

    def test_no_user_allergens_returns_none(self):
        """No user allergens → no finding."""
        rule = AllergenDetectionRule()
        recipe = _make_recipe(
            "r1",
            "Peanut Noodles",
            ingredients=(_make_ingredient("peanut", allergen_tags=("peanut",)),),
        )
        ctx = _make_context(recipes=(recipe,), user_allergens=())

        result = rule.evaluate(ctx)
        assert result is None

    def test_priority_allergen_match_returns_hard_unrepairable(self):
        """Peanut (priority allergen) match → hard_unrepairable."""
        rule = AllergenDetectionRule()
        recipe = _make_recipe(
            "r1",
            "Pad Thai",
            ingredients=(_make_ingredient("peanut", allergen_tags=("peanut",)),),
        )
        ctx = _make_context(recipes=(recipe,), user_allergens=("peanut",))

        result = rule.evaluate(ctx)
        assert result is not None
        assert result.rule_id == "SAFETY_ALLERGEN_DETECTION"
        assert result.severity == "hard_unrepairable"
        assert "peanut" in result.affected_ingredient_names

    def test_non_priority_allergen_match_returns_hard_repairable(self):
        """A non-priority allergen (e.g., corn sensitivity) → hard_repairable."""
        rule = AllergenDetectionRule()
        # "corn" is not in the priority or keyword lists, so use allergen_tags
        recipe = _make_recipe(
            "r1",
            "Corn Soup",
            ingredients=(_make_ingredient("corn", allergen_tags=("corn",)),),
        )
        ctx = _make_context(recipes=(recipe,), user_allergens=("corn",))

        result = rule.evaluate(ctx)
        assert result is not None
        # "corn" is not in _priority_allergens, so it falls to hard_repairable
        assert result.severity == "hard_repairable"

    def test_keyword_matched_allergen_works(self):
        """Ingredient name keyword match (shrimp → shellfish) detects allergen."""
        rule = AllergenDetectionRule()
        recipe = _make_recipe(
            "r1",
            "Garlic Shrimp",
            ingredients=(_make_ingredient("shrimp"),),
        )
        ctx = _make_context(recipes=(recipe,), user_allergens=("shellfish",))

        result = rule.evaluate(ctx)
        assert result is not None
        assert result.rule_id == "SAFETY_ALLERGEN_DETECTION"
        assert "shrimp" in result.affected_ingredient_names

    def test_no_allergen_match_returns_none(self):
        """User allergic to peanut but dish has no peanut → no finding."""
        rule = AllergenDetectionRule()
        recipe = _make_recipe(
            "r1",
            "Plain Rice",
            ingredients=(_make_ingredient("rice"),),
        )
        ctx = _make_context(recipes=(recipe,), user_allergens=("peanut",))

        result = rule.evaluate(ctx)
        assert result is None


# =============================================================================
# 3. ProteinSafetyTemperatureRule
# =============================================================================


class TestProteinSafetyTemperatureRule:
    """USDA safe internal temperature checks for protein cooking."""

    def test_non_heating_step_returns_none(self):
        """No heating steps → no finding."""
        rule = ProteinSafetyTemperatureRule()
        recipe = _make_recipe(
            "r1",
            "Salad",
            ingredients=(_make_ingredient("chicken breast"),),
            steps=(_make_step(1, "Mix ingredients", category="mixing", heat_level=HeatLevel.NONE),),
        )
        ctx = _make_context(recipes=(recipe,))

        result = rule.evaluate(ctx)
        assert result is None

    def test_chicken_with_safe_temperature_returns_none(self):
        """Chicken cooked at 74°C (safe) → no finding."""
        rule = ProteinSafetyTemperatureRule()
        recipe = _make_recipe(
            "r1",
            "Roast Chicken",
            ingredients=(_make_ingredient("chicken"),),
            steps=(
                _make_step(
                    1,
                    "Roast chicken until done",
                    target_temperature_c=Decimal(74),
                ),
            ),
        )
        ctx = _make_context(recipes=(recipe,))

        result = rule.evaluate(ctx)
        assert result is None

    def test_chicken_below_safe_temperature_returns_finding(self):
        """Chicken cooked at 60°C (below 74°C) → finding."""
        rule = ProteinSafetyTemperatureRule()
        recipe = _make_recipe(
            "r1",
            "Undercooked Chicken",
            ingredients=(_make_ingredient("chicken breast"),),
            steps=(
                _make_step(
                    1,
                    "Cook chicken breast",
                    target_temperature_c=Decimal(60),
                ),
            ),
        )
        ctx = _make_context(recipes=(recipe,))

        result = rule.evaluate(ctx)
        assert result is not None
        assert result.rule_id == "SAFETY_PROTEIN_TEMPERATURE"
        assert result.severity == "hard_repairable"
        assert "74" in result.description  # Should mention safe minimum 74°C

    def test_chicken_no_temperature_specified_returns_finding(self):
        """Chicken step with no temperature → finding with recommendation."""
        rule = ProteinSafetyTemperatureRule()
        recipe = _make_recipe(
            "r1",
            "Pan-Fried Chicken",
            ingredients=(_make_ingredient("chicken thigh"),),
            steps=(
                _make_step(
                    1,
                    "Fry chicken until golden",
                    target_temperature_c=None,
                ),
            ),
        )
        ctx = _make_context(recipes=(recipe,))

        result = rule.evaluate(ctx)
        assert result is not None
        assert "no target temperature" in result.description.lower()

    def test_beef_steak_at_safe_temperature_returns_none(self):
        """Beef at 63°C (safe for steaks) → no finding."""
        rule = ProteinSafetyTemperatureRule()
        recipe = _make_recipe(
            "r1",
            "Beef Steak",
            ingredients=(_make_ingredient("beef sirloin"),),
            steps=(
                _make_step(
                    1,
                    "Sear beef steak",
                    target_temperature_c=Decimal(63),
                ),
            ),
        )
        ctx = _make_context(recipes=(recipe,))

        result = rule.evaluate(ctx)
        assert result is None

    def test_vegetarian_dish_with_heating_returns_none(self):
        """Vegetarian dish (no protein keywords) → no finding."""
        rule = ProteinSafetyTemperatureRule()
        recipe = _make_recipe(
            "r1",
            "Stir-Fried Vegetables",
            ingredients=(_make_ingredient("broccoli"), _make_ingredient("carrot")),
            steps=(_make_step(1, "Stir-fry vegetables", heat_level=HeatLevel.HIGH),),
        )
        ctx = _make_context(recipes=(recipe,))

        result = rule.evaluate(ctx)
        assert result is None


# =============================================================================
# 4. DietaryCompatibilityRule
# =============================================================================


class TestDietaryCompatibilityRule:
    """Dietary restriction compatibility checks."""

    def test_no_restrictions_returns_none(self):
        """No dietary restrictions → no finding."""
        rule = DietaryCompatibilityRule()
        recipe = _make_recipe(
            "r1",
            "Pork Belly",
            ingredients=(_make_ingredient("pork belly"),),
        )
        ctx = _make_context(recipes=(recipe,), dietary_restrictions=())

        result = rule.evaluate(ctx)
        assert result is None

    def test_halal_violation_with_pork_returns_unrepairable(self):
        """Halal restriction + pork ingredient → hard_unrepairable."""
        rule = DietaryCompatibilityRule()
        recipe = _make_recipe(
            "r1",
            "Braised Pork",
            ingredients=(_make_ingredient("pork belly"),),
        )
        ctx = _make_context(recipes=(recipe,), dietary_restrictions=("halal",))

        result = rule.evaluate(ctx)
        assert result is not None
        assert result.rule_id == "SAFETY_DIETARY_COMPATIBILITY"
        assert result.severity == "hard_unrepairable"
        assert "pork" in result.description.lower()

    def test_vegetarian_violation_with_chicken_returns_unrepairable(self):
        """Vegetarian restriction + chicken → hard_unrepairable."""
        rule = DietaryCompatibilityRule()
        recipe = _make_recipe(
            "r1",
            "Chicken Soup",
            ingredients=(_make_ingredient("chicken breast"),),
        )
        ctx = _make_context(recipes=(recipe,), dietary_restrictions=("vegetarian",))

        result = rule.evaluate(ctx)
        assert result is not None
        assert result.severity == "hard_unrepairable"

    def test_vegan_violation_with_egg_returns_unrepairable(self):
        """Vegan restriction + egg → hard_unrepairable."""
        rule = DietaryCompatibilityRule()
        recipe = _make_recipe(
            "r1",
            "Omelette",
            ingredients=(_make_ingredient("egg"),),
        )
        ctx = _make_context(recipes=(recipe,), dietary_restrictions=("vegan",))

        result = rule.evaluate(ctx)
        assert result is not None
        assert result.severity == "hard_unrepairable"

    def test_compatible_recipe_returns_none(self):
        """Halal restriction + chicken (allowed) → no finding."""
        rule = DietaryCompatibilityRule()
        recipe = _make_recipe(
            "r1",
            "Halal Chicken Rice",
            ingredients=(
                _make_ingredient("chicken breast"),
                _make_ingredient("rice"),
            ),
        )
        ctx = _make_context(recipes=(recipe,), dietary_restrictions=("halal",))

        result = rule.evaluate(ctx)
        assert result is None


# =============================================================================
# 5. SafetyEngine composition
# =============================================================================


class TestSafetyEngine:
    """SafetyEngine integration: all rules running together."""

    def test_empty_context_returns_safe(self):
        """No allergens, no restrictions, simple recipe → is_safe=True."""
        engine = SafetyEngine()
        recipe = _make_recipe("r1", "Plain Rice")
        ctx = _make_context(recipes=(recipe,))

        report = engine.evaluate(ctx)
        assert isinstance(report, SafetyReport)
        assert report.is_safe
        assert not report.has_unrepairable

    def test_allergen_recipe_returns_unsafe(self):
        """Allergen recipe → is_safe=False, has_unrepairable=True."""
        engine = SafetyEngine()
        recipe = _make_recipe(
            "r1",
            "Peanut Sauce",
            ingredients=(_make_ingredient("peanut", allergen_tags=("peanut",)),),
        )
        ctx = _make_context(recipes=(recipe,), user_allergens=("peanut",))

        report = engine.evaluate(ctx)
        assert not report.is_safe
        assert report.has_unrepairable
        assert len(report.findings) >= 1

    def test_cross_contamination_generates_safety_task_ids(self):
        """Cross-contamination (repairable) → required_safety_task_ids populated."""
        # Use a custom rule set to isolate cross-contamination from
        # other rules (e.g., protein temperature which would also fire).
        engine = SafetyEngine(rules=(CrossContaminationRule(),))
        recipe = _make_recipe(
            "r1",
            "Chicken Salad",
            ingredients=(
                _make_ingredient("chicken breast", input_state="raw"),
                _make_ingredient("lettuce"),
            ),
            steps=(
                _make_step(1, "Cook chicken", category="heating"),
                _make_step(2, "Plate salad", category="plating"),
            ),
        )
        ctx = _make_context(recipes=(recipe,))

        report = engine.evaluate(ctx)
        assert len(report.required_safety_task_ids) >= 1
        for task_id in report.required_safety_task_ids:
            assert task_id.startswith("safety_safety_cross_contamination_")

    def test_multiple_findings_aggregated(self):
        """Multiple violations → all findings in report."""
        engine = SafetyEngine()
        recipe = _make_recipe(
            "r1",
            "Pork Satay with Peanut Sauce",
            ingredients=(
                _make_ingredient("pork belly", input_state="raw"),
                _make_ingredient("peanut", allergen_tags=("peanut",)),
            ),
            steps=(
                _make_step(1, "Grill pork satay"),
                _make_step(2, "Plate with sauce", category="plating"),
            ),
        )
        ctx = _make_context(
            recipes=(recipe,),
            dietary_restrictions=("halal",),
            user_allergens=("peanut",),
        )

        report = engine.evaluate(ctx)
        # Expect at least dietary, allergen, and cross-contamination findings
        assert len(report.findings) >= 3
        assert not report.is_safe
        assert report.has_unrepairable

    def test_report_id_is_generated(self):
        """Every report gets a unique report_id."""
        engine = SafetyEngine()
        recipe = _make_recipe("r1", "Test Dish")
        ctx = _make_context(recipes=(recipe,))

        report = engine.evaluate(ctx)
        assert report.report_id.startswith("safety_")
        assert len(report.report_id) > 10

    def test_custom_rules_supported(self):
        """Engine accepts custom rule sets."""
        rule = CrossContaminationRule()
        engine = SafetyEngine(rules=(rule,))
        recipe = _make_recipe(
            "r1",
            "Chicken Salad",
            ingredients=(
                _make_ingredient("chicken breast", input_state="raw"),
                _make_ingredient("lettuce"),
            ),
            steps=(
                _make_step(1, "Cook chicken", category="heating"),
                _make_step(2, "Plate salad", category="plating"),
            ),
        )
        ctx = _make_context(recipes=(recipe,))

        report = engine.evaluate(ctx)
        assert len(report.findings) == 1
        assert report.findings[0].rule_id == "SAFETY_CROSS_CONTAMINATION"


# =============================================================================
# 6. ExpiredIngredientRule
# =============================================================================


def _make_lot(
    lot_id: str,
    canonical_name: str,
    on_hand: Decimal = Decimal(500),
    reserved: Decimal = Decimal(0),
    unit: str = "g",
    expiry_date: date | None = None,
) -> InventoryLotSnapshot:
    """Factory for an InventoryLotSnapshot."""
    return InventoryLotSnapshot(
        lot_id=lot_id,
        item_id=f"item_{lot_id}",
        canonical_name=canonical_name,
        on_hand=on_hand,
        reserved=reserved,
        unit=unit,
        expiry_date=expiry_date,
    )


class TestExpiredIngredientRule:
    """Expired ingredient detection in inventory lots."""

    def test_no_cooking_date_returns_none(self):
        """Without a cooking_date, expiry cannot be checked → None."""
        rule = ExpiredIngredientRule()
        ctx = _make_context(recipes=(_make_recipe("r1", "Chicken Soup"),))
        assert rule.evaluate(ctx) is None

    def test_no_inventory_lots_returns_none(self):
        """Without inventory lots, no expiry to check → None."""
        rule = ExpiredIngredientRule()
        recipe = _make_recipe(
            "r1",
            "Chicken Soup",
            ingredients=(_make_ingredient("chicken breast", input_state="raw"),),
        )
        ctx = SafetyContext(recipes=(recipe,), cooking_date=date(2026, 8, 15))
        assert rule.evaluate(ctx) is None

    def test_fresh_lot_returns_none(self):
        """Lot with expiry_date after cooking_date → no finding."""
        rule = ExpiredIngredientRule()
        recipe = _make_recipe(
            "r1",
            "Chicken Soup",
            ingredients=(_make_ingredient("chicken breast"),),
        )
        lot = _make_lot("L1", "chicken breast", expiry_date=date(2026, 8, 20))
        ctx = SafetyContext(
            recipes=(recipe,),
            cooking_date=date(2026, 8, 15),
            inventory_lots=(lot,),
        )
        assert rule.evaluate(ctx) is None

    def test_slightly_expired_perishable_returns_repairable(self):
        """Lot 1 day past expiry → hard_repairable (can inspect)."""
        rule = ExpiredIngredientRule()
        recipe = _make_recipe(
            "r1",
            "Chicken Soup",
            ingredients=(_make_ingredient("chicken breast"),),
        )
        lot = _make_lot("L1", "chicken breast", expiry_date=date(2026, 8, 14))
        ctx = SafetyContext(
            recipes=(recipe,),
            cooking_date=date(2026, 8, 15),
            inventory_lots=(lot,),
        )
        result = rule.evaluate(ctx)
        assert result is not None
        assert result.severity == "hard_repairable"
        assert "chicken breast" in result.affected_ingredient_names

    def test_deeply_expired_perishable_returns_unrepairable(self):
        """Lot 5 days past expiry → hard_unrepairable (likely spoiled)."""
        rule = ExpiredIngredientRule()
        recipe = _make_recipe(
            "r1",
            "Beef Stew",
            ingredients=(_make_ingredient("beef chuck"),),
        )
        lot = _make_lot("L1", "beef chuck", expiry_date=date(2026, 8, 10))
        ctx = SafetyContext(
            recipes=(recipe,),
            cooking_date=date(2026, 8, 15),
            inventory_lots=(lot,),
        )
        result = rule.evaluate(ctx)
        assert result is not None
        assert result.severity == "hard_unrepairable"

    def test_expired_non_perishable_returns_none(self):
        """Expired rice (non-perishable) → no finding."""
        rule = ExpiredIngredientRule()
        recipe = _make_recipe(
            "r1",
            "Fried Rice",
            ingredients=(_make_ingredient("rice"),),
        )
        lot = _make_lot("L1", "rice", expiry_date=date(2025, 1, 1))
        ctx = SafetyContext(
            recipes=(recipe,),
            cooking_date=date(2026, 8, 15),
            inventory_lots=(lot,),
        )
        assert rule.evaluate(ctx) is None

    def test_lot_not_used_in_recipes_returns_none(self):
        """Lot for unused ingredient → skipped, no false positive."""
        rule = ExpiredIngredientRule()
        recipe = _make_recipe(
            "r1",
            "Vegetable Soup",
            ingredients=(_make_ingredient("carrot"),),
        )
        lot = _make_lot("L1", "chicken breast", expiry_date=date(2026, 8, 10))
        ctx = SafetyContext(
            recipes=(recipe,),
            cooking_date=date(2026, 8, 15),
            inventory_lots=(lot,),
        )
        assert rule.evaluate(ctx) is None


# =============================================================================
# 7. HoldingTimeRule
# =============================================================================


class TestHoldingTimeRule:
    """Holding-time risk for dishes with long passive phases."""

    def test_no_perishable_protein_returns_none(self):
        """Vegetarian dish → no holding-time risk."""
        rule = HoldingTimeRule()
        recipe = _make_recipe(
            "r1",
            "Vegetable Soup",
            ingredients=(_make_ingredient("carrot"), _make_ingredient("broccoli")),
            steps=(_make_step(1, "Boil vegetables", passive_duration_minutes=180),),
        )
        ctx = _make_context(recipes=(recipe,))
        assert rule.evaluate(ctx) is None

    def test_short_passive_returns_none(self):
        """Perishable protein but passive time ≤ 120 min → no finding."""
        rule = HoldingTimeRule()
        recipe = _make_recipe(
            "r1",
            "Quick Chicken",
            ingredients=(_make_ingredient("chicken breast"),),
            steps=(_make_step(1, "Boil chicken", passive_duration_minutes=60),),
        )
        ctx = _make_context(recipes=(recipe,))
        assert rule.evaluate(ctx) is None

    def test_long_passive_with_perishable_returns_finding(self):
        """Perishable protein + passive > 120 min → holding-time risk."""
        rule = HoldingTimeRule()
        recipe = _make_recipe(
            "r1",
            "Slow-Cooked Beef",
            ingredients=(_make_ingredient("beef chuck"),),
            steps=(_make_step(1, "Simmer beef", passive_duration_minutes=180),),
        )
        ctx = _make_context(recipes=(recipe,))
        result = rule.evaluate(ctx)
        assert result is not None
        assert result.severity == "hard_repairable"
        assert result.rule_id == "SAFETY_HOLDING_TIME"
        assert "beef" in result.description.lower() or "Slow-Cooked" in result.description

    def test_multiple_recipes_one_risky(self):
        """One risky dish among several → finding includes only the risky one."""
        rule = HoldingTimeRule()
        risky = _make_recipe(
            "r1",
            "Brisket",
            ingredients=(_make_ingredient("beef brisket"),),
            steps=(_make_step(1, "Simmer brisket", passive_duration_minutes=240),),
        )
        safe = _make_recipe(
            "r2",
            "Salad",
            ingredients=(_make_ingredient("lettuce"), _make_ingredient("tomato")),
            steps=(_make_step(1, "Toss salad", passive_duration_minutes=0),),
        )
        ctx = _make_context(recipes=(risky, safe))
        result = rule.evaluate(ctx)
        assert result is not None
        assert "Brisket" in result.description

    def test_edge_case_exactly_120_returns_none(self):
        """Passive == 120 min (boundary) → no finding (strict > check)."""
        rule = HoldingTimeRule()
        recipe = _make_recipe(
            "r1",
            "Roast Chicken",
            ingredients=(_make_ingredient("chicken"),),
            steps=(_make_step(1, "Roast chicken", passive_duration_minutes=120),),
        )
        ctx = _make_context(recipes=(recipe,))
        assert rule.evaluate(ctx) is None
