"""Unit tests for repair options — handbook 5.17–5.25."""

from decimal import Decimal

from cooking_plan_agent.domain.models import (
    FeasibilityReport,
    GeneratePlanRequest,
    IngredientFeasibility,
    RepairOption,
)
from cooking_plan_agent.repair.options import (
    apply_approved_decisions,
    calculate_exact_shortages,
    propose_dish_replacements,
    propose_equipment_alternatives,
    propose_ingredient_substitutions,
    propose_portion_adjustments,
    propose_time_extension,
    rank_repair_options,
    validate_repair_option,
)

# ======================================================================
# 5.17  calculate_exact_shortages
# ======================================================================


class TestCalculateExactShortages:
    def test_extracts_ingredient_shortages(self):
        report = FeasibilityReport(
            report_id="r1",
            ingredient_shortages=(
                IngredientFeasibility(
                    ingredient_name="chicken breast",
                    required=Decimal(400),
                    available=Decimal(200),
                    shortage=Decimal(200),
                    unit="g",
                ),
            ),
            is_feasible=False,
        )
        result = calculate_exact_shortages(report)
        assert len(result) == 1
        assert result[0].item == "chicken breast"
        assert result[0].required == Decimal(400)
        assert result[0].available == Decimal(200)

    def test_extracts_resource_shortages(self):
        report = FeasibilityReport(
            report_id="r2",
            missing_resources=("oven", "wok"),
            is_feasible=False,
        )
        result = calculate_exact_shortages(report)
        assert len(result) == 2
        assert {r.item for r in result} == {"oven", "wok"}

    def test_mixed_shortages(self):
        report = FeasibilityReport(
            report_id="r3",
            ingredient_shortages=(
                IngredientFeasibility(
                    ingredient_name="tomato",
                    required=Decimal(4),
                    available=Decimal(2),
                    shortage=Decimal(2),
                    unit="piece",
                ),
            ),
            missing_resources=("stove",),
            is_feasible=False,
        )
        result = calculate_exact_shortages(report)
        assert len(result) == 2

    def test_feasible_report_returns_empty(self):
        report = FeasibilityReport(report_id="r4", is_feasible=True)
        assert calculate_exact_shortages(report) == ()

    def test_skips_fully_satisfied_ingredients(self):
        report = FeasibilityReport(
            report_id="r5",
            ingredient_shortages=(
                IngredientFeasibility(
                    ingredient_name="chicken breast",
                    required=Decimal(400),
                    available=Decimal(400),
                    shortage=Decimal(0),
                    unit="g",
                ),
            ),
            is_feasible=True,
        )
        assert calculate_exact_shortages(report) == ()


# ======================================================================
# 5.18  propose_ingredient_substitutions
# ======================================================================


class TestProposeIngredientSubstitutions:
    def test_known_substitute(self):
        shortages = (
            IngredientFeasibility(
                ingredient_name="chicken breast",
                required=Decimal(400),
                available=Decimal(200),
                shortage=Decimal(200),
                unit="g",
            ),
        )
        options = propose_ingredient_substitutions(shortages)
        assert len(options) >= 2  # chicken thigh + tofu
        types = {o.option_type for o in options}
        assert "substitute_ingredient" in types

    def test_unknown_ingredient_falls_back_to_purchase(self):
        shortages = (
            IngredientFeasibility(
                ingredient_name="truffle oil",
                required=Decimal(50),
                available=Decimal(0),
                shortage=Decimal(50),
                unit="ml",
            ),
        )
        options = propose_ingredient_substitutions(shortages)
        assert len(options) == 1
        assert options[0].option_type == "purchase"

    def test_empty_shortages(self):
        assert propose_ingredient_substitutions(()) == ()

    def test_each_option_has_required_fields(self):
        shortages = (
            IngredientFeasibility(
                ingredient_name="butter",
                required=Decimal(100),
                available=Decimal(0),
                shortage=Decimal(100),
                unit="g",
            ),
        )
        options = propose_ingredient_substitutions(shortages)
        for opt in options:
            assert opt.option_id.startswith("repair_")
            assert opt.description
            assert opt.changes
            assert opt.effects
            assert opt.revalidation_status == "validated"


# ======================================================================
# 5.19  propose_portion_adjustments
# ======================================================================


class TestProposePortionAdjustments:
    def test_reduce_servings_by_half(self):
        """50% shortage → reduce from 4 to 2 servings."""
        shortages = (
            IngredientFeasibility(
                ingredient_name="chicken breast",
                required=Decimal(400),
                available=Decimal(200),
                shortage=Decimal(200),
                unit="g",
            ),
        )
        options = propose_portion_adjustments(shortages, original_servings=4)
        assert len(options) == 1
        assert options[0].option_type == "reduce_servings"
        assert "2" in options[0].description

    def test_no_reduction_needed(self):
        shortages = (
            IngredientFeasibility(
                ingredient_name="chicken breast",
                required=Decimal(400),
                available=Decimal(400),
                shortage=Decimal(0),
                unit="g",
            ),
        )
        options = propose_portion_adjustments(shortages, original_servings=2)
        # shortage=0 ingredients are filtered by caller, but we handle defensively
        # ratio = 400/400 = 1.0 → new_servings = 2, no reduction
        assert options == ()

    def test_single_serving_no_reduction(self):
        shortages = (
            IngredientFeasibility(
                ingredient_name="chicken breast",
                required=Decimal(400),
                available=Decimal(100),
                shortage=Decimal(300),
                unit="g",
            ),
        )
        options = propose_portion_adjustments(shortages, original_servings=1)
        assert options == ()

    def test_empty_shortages(self):
        assert propose_portion_adjustments((), original_servings=4) == ()

    def test_uses_most_limiting_ingredient(self):
        """Two shortages: 50% and 25% → use 25% (most limiting)."""
        shortages = (
            IngredientFeasibility(
                ingredient_name="chicken breast",
                required=Decimal(400),
                available=Decimal(200),  # 50%
                shortage=Decimal(200),
                unit="g",
            ),
            IngredientFeasibility(
                ingredient_name="rice",
                required=Decimal(400),
                available=Decimal(100),  # 25% — most limiting
                shortage=Decimal(300),
                unit="g",
            ),
        )
        options = propose_portion_adjustments(shortages, original_servings=4)
        assert len(options) == 1
        # 25% of 4 servings = 1 serving
        assert "to 1" in options[0].description or "from 4 to 1" in options[0].description


# ======================================================================
# 5.20  propose_equipment_alternatives
# ======================================================================


class TestProposeEquipmentAlternatives:
    def test_known_alternative(self):
        options = propose_equipment_alternatives(("oven",))
        assert len(options) >= 2  # air fryer, toaster oven, etc.
        types = {o.option_type for o in options}
        assert types == {"alternative_equipment"}

    def test_unknown_equipment_no_alternative(self):
        options = propose_equipment_alternatives(("laser_cutter",))
        assert len(options) == 1
        assert "No known alternative" in options[0].description

    def test_multiple_missing(self):
        options = propose_equipment_alternatives(("oven", "wok"))
        assert len(options) > 2  # Multiple alternatives per type

    def test_strips_capability_suffix(self):
        """'stove:induction' → should look up 'stove' alternatives."""
        options = propose_equipment_alternatives(("stove:induction",))
        assert len(options) >= 1
        assert all(o.option_type == "alternative_equipment" for o in options)

    def test_empty_missing(self):
        assert propose_equipment_alternatives(()) == ()


# ======================================================================
# 5.21  propose_dish_replacements
# ======================================================================


class TestProposeDishReplacements:
    def test_unsubstitutable_ingredients_suggest_removal(self):
        shortages = (
            IngredientFeasibility(
                ingredient_name="truffle oil",
                required=Decimal(50),
                available=Decimal(0),
                shortage=Decimal(50),
                unit="ml",
            ),
        )
        options = propose_dish_replacements(shortages, ("Pasta",))
        assert len(options) == 1
        assert options[0].option_type == "replace_dish"
        assert "truffle oil" in options[0].description.lower()

    def test_all_substitutable_no_suggestion(self):
        shortages = (
            IngredientFeasibility(
                ingredient_name="chicken breast",
                required=Decimal(400),
                available=Decimal(200),
                shortage=Decimal(200),
                unit="g",
            ),
        )
        options = propose_dish_replacements(shortages, ("Stir-fry",))
        assert options == ()

    def test_empty_shortages(self):
        assert propose_dish_replacements((), ()) == ()


# ======================================================================
# 5.22  propose_time_extension
# ======================================================================


class TestProposeTimeExtension:
    def test_reasonable_extension(self):
        opt = propose_time_extension(current_time_limit=30, minimum_required_minutes=45)
        assert opt is not None
        assert opt.option_type == "extend_time"
        assert "45" in opt.description
        assert "15 minutes" in opt.description.lower()

    def test_no_limit_no_extension(self):
        assert propose_time_extension(current_time_limit=None, minimum_required_minutes=45) is None

    def test_limit_already_sufficient(self):
        assert propose_time_extension(current_time_limit=60, minimum_required_minutes=45) is None

    def test_extreme_extension_rejected(self):
        """> 3× the current limit → rejected."""
        assert propose_time_extension(current_time_limit=10, minimum_required_minutes=50) is None

    def test_just_within_threshold(self):
        """Exactly 3× is accepted."""
        opt = propose_time_extension(current_time_limit=10, minimum_required_minutes=30)
        assert opt is not None


# ======================================================================
# 5.23  validate_repair_option
# ======================================================================


class TestValidateRepairOption:
    def test_valid_option(self):
        opt = RepairOption(
            option_id="repair_test_1",
            option_type="substitute_ingredient",
            description="Substitute X with Y",
            changes=("Change X to Y",),
            effects=("Resolves shortage",),
        )
        result = validate_repair_option(opt)
        assert result.is_valid is True
        assert result.issues == ()

    def test_missing_description(self):
        opt = RepairOption(
            option_id="repair_test",
            option_type="reduce_servings",
            description="",
            changes=("Change",),
            effects=("Effect",),
        )
        result = validate_repair_option(opt)
        assert result.is_valid is False
        assert any("description" in i for i in result.issues)

    def test_unknown_option_type(self):
        opt = RepairOption(
            option_id="repair_test",
            option_type="magic_wand",
            description="Fix everything",
            changes=("Magic",),
            effects=("Solved",),
        )
        result = validate_repair_option(opt)
        assert result.is_valid is False
        assert any("option_type" in i for i in result.issues)

    def test_empty_changes(self):
        opt = RepairOption(
            option_id="repair_test",
            option_type="extend_time",
            description="Extend time",
            changes=(),
            effects=("Effect",),
        )
        result = validate_repair_option(opt)
        assert result.is_valid is False

    def test_invalid_revalidation_status(self):
        opt = RepairOption(
            option_id="repair_test",
            option_type="purchase",
            description="Buy it",
            changes=("Buy",),
            effects=("Have it",),
            revalidation_status="pending",
        )
        result = validate_repair_option(opt)
        assert result.is_valid is False


# ======================================================================
# 5.24  rank_repair_options
# ======================================================================


class TestRankRepairOptions:
    def test_sorts_by_priority(self):
        options = (
            RepairOption(
                option_id="r_purchase",
                option_type="purchase",
                description="Buy",
                changes=("Buy",),
                effects=("Get",),
            ),
            RepairOption(
                option_id="r_reduce",
                option_type="reduce_servings",
                description="Reduce",
                changes=("Reduce",),
                effects=("Less",),
            ),
            RepairOption(
                option_id="r_sub",
                option_type="substitute_ingredient",
                description="Sub",
                changes=("Sub",),
                effects=("Swap",),
            ),
        )
        ranked = rank_repair_options(options)
        assert ranked[0].option_type == "reduce_servings"  # Priority 1
        assert ranked[1].option_type == "substitute_ingredient"  # Priority 3
        assert ranked[2].option_type == "purchase"  # Priority 6

    def test_filters_non_validated(self):
        options = (
            RepairOption(
                option_id="r1",
                option_type="reduce_servings",
                description="Ok",
                changes=("Ok",),
                effects=("Ok",),
                revalidation_status="rejected",
            ),
            RepairOption(
                option_id="r2",
                option_type="purchase",
                description="Ok",
                changes=("Ok",),
                effects=("Ok",),
            ),
        )
        ranked = rank_repair_options(options)
        assert len(ranked) == 1
        assert ranked[0].option_id == "r2"

    def test_empty_options(self):
        assert rank_repair_options(()) == ()


# ======================================================================
# 5.25  apply_approved_decisions
# ======================================================================


class TestApplyApprovedDecisions:
    def test_apply_time_extension(self):
        request = GeneratePlanRequest(
            request_id="req-1",
            user_id="user-1",
            recipes=({"recipe_id": "r1", "text": "test", "target_servings": 2},),
            time_limit_minutes=30,
        )
        options = (
            RepairOption(
                option_id="repair_time_45",
                option_type="extend_time",
                description="Extend cooking time from 30 to 45 minutes (adds 15 minutes)",
                changes=("Increase time limit to 45 minutes",),
                effects=("All tasks can be scheduled within 45 minutes",),
            ),
        )
        result = apply_approved_decisions(request, ("repair_time_45",), options)
        assert result["applied_count"] == 1
        assert result["modifications"]["time_limit_minutes"] == 45

    def test_apply_portion_adjustment(self):
        request = GeneratePlanRequest(
            request_id="req-1",
            user_id="user-1",
            recipes=({"recipe_id": "r1", "text": "test", "target_servings": 4},),
        )
        options = (
            RepairOption(
                option_id="repair_servings_2",
                option_type="reduce_servings",
                description="Reduce servings from 4 to 2",
                changes=("Scale all ingredient quantities to 2 servings",),
                effects=("All ingredient shortages resolved",),
            ),
        )
        result = apply_approved_decisions(request, ("repair_servings_2",), options)
        assert result["applied_count"] == 1
        assert result["modifications"]["target_servings"] == 2

    def test_no_approved_ids(self):
        request = GeneratePlanRequest(
            request_id="req-1",
            user_id="user-1",
            recipes=({"recipe_id": "r1", "text": "test", "target_servings": 2},),
        )
        options = (
            RepairOption(
                option_id="repair_sub",
                option_type="substitute_ingredient",
                description="Sub",
                changes=("Sub",),
                effects=("Effect",),
            ),
        )
        result = apply_approved_decisions(request, (), options)
        assert result["applied_count"] == 0
        assert result["modifications"] == {}

    def test_unknown_option_id(self):
        request = GeneratePlanRequest(
            request_id="req-1",
            user_id="user-1",
            recipes=({"recipe_id": "r1", "text": "test", "target_servings": 2},),
        )
        options = (
            RepairOption(
                option_id="repair_sub",
                option_type="substitute_ingredient",
                description="Sub",
                changes=("Sub",),
                effects=("Effect",),
            ),
        )
        result = apply_approved_decisions(request, ("nonexistent_id",), options)
        assert result["applied_count"] == 0

    def test_partial_approval(self):
        """Only some options approved."""
        request = GeneratePlanRequest(
            request_id="req-1",
            user_id="user-1",
            recipes=({"recipe_id": "r1", "text": "test", "target_servings": 2},),
            time_limit_minutes=30,
        )
        options = (
            RepairOption(
                option_id="r_time",
                option_type="extend_time",
                description="Extend cooking time from 30 to 45 minutes (adds 15 minutes)",
                changes=("Extend",),
                effects=("Effect",),
            ),
            RepairOption(
                option_id="r_sub",
                option_type="substitute_ingredient",
                description="Sub",
                changes=("Sub",),
                effects=("Effect",),
            ),
        )
        result = apply_approved_decisions(request, ("r_time",), options)
        assert result["applied_count"] == 1
        assert result["modifications"]["time_limit_minutes"] == 45
