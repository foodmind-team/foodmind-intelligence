"""Domain model invariant and error-code tests.

Handbook 11.3: cover Pydantic invariants, error codes, frozen immutability,
and field-level validators for every StrictModel subclass.

Uses table-driven tests (pytest parametrize) rather than one nearly
identical test function per model.
"""

from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from cooking_plan_agent.domain.enums import HeatLevel, PlanStatus, SolverStatus, WorkMode
from cooking_plan_agent.domain.errors import DomainErrorCode, WorkflowException
from cooking_plan_agent.domain.models import (
    Assumption,
    ConfirmationPlanResponse,
    CookingEvidence,
    CookingTask,
    EvidenceQuery,
    EvidenceRef,
    ExtractedIngredient,
    ExtractedRecipeCandidate,
    ExtractedStep,
    FailedPlanResponse,
    FeasibilityReport,
    GeneratePlanRequest,
    InfeasiblePlanResponse,
    IngredientDemand,
    IngredientFeasibility,
    InventoryLotSnapshot,
    KitchenResourceSnapshot,
    LotAllocation,
    ReadyPlanResponse,
    RecipeGap,
    RecipeIR,
    RecipeStep,
    ReconciledEvidence,
    RepairOption,
    ResourceNeed,
    SafetyFinding,
    SafetyReport,
    TaskDependency,
    WorkflowError,
)

# =============================================================================
# 1. StrictModel invariants (applies to all subclasses)
# =============================================================================


class TestStrictModelInvariants:
    """Every StrictModel subclass enforces frozen + extra=forbid."""

    # Represented by one example per model category
    @pytest.mark.parametrize(
        "model_factory,model_name",
        [
            # lambda creates an instance of each StrictModel type
            (
                lambda: IngredientDemand(
                    canonical_name="test",
                    raw_name="test",
                    quantity=Decimal(1),
                    unit="g",
                    confidence=Decimal("0.5"),
                ),
                "IngredientDemand",
            ),
            (lambda: RecipeStep(step_number=1, instruction="Test"), "RecipeStep"),
            (
                lambda: CookingTask(
                    task_id="t1",
                    dish_id="r1",
                    instruction="Test",
                    duration_minutes=1,
                    work_mode=WorkMode.ACTIVE,
                    category="test",
                ),
                "CookingTask",
            ),
            (
                lambda: InventoryLotSnapshot(
                    lot_id="l1",
                    item_id="i1",
                    canonical_name="test",
                    on_hand=Decimal(100),
                    reserved=Decimal(0),
                    unit="g",
                ),
                "InventoryLotSnapshot",
            ),
            (
                lambda: GeneratePlanRequest(
                    request_id="r1",
                    user_id="u1",
                    recipes=({"recipe_id": "r", "text": "boil", "target_servings": "2"},),
                ),
                "GeneratePlanRequest",
            ),
            (
                lambda: ReadyPlanResponse(
                    plan_id="p1",
                    status="READY",
                    solver_status="OPTIMAL",
                    makespan_minutes=10,
                    timeline=(),
                    completion_checklist=(),
                    mise_en_place=(),
                    dish_completions=(),
                ),
                "ReadyPlanResponse",
            ),
            (
                lambda: FailedPlanResponse(
                    status="FAILED",
                    error_code="INTERNAL_ERROR",
                    correlation_id="c1",
                    message="test",
                ),
                "FailedPlanResponse",
            ),
            (
                lambda: SafetyFinding(
                    rule_id="r1",
                    severity="warning",
                    description="test",
                ),
                "SafetyFinding",
            ),
            (
                lambda: WorkflowError(
                    error_code="INTERNAL_ERROR",
                    message="test",
                    correlation_id="c1",
                ),
                "WorkflowError",
            ),
        ],
    )
    def test_extra_fields_rejected(self, model_factory, model_name):
        """Every StrictModel rejects unknown fields (extra=forbid)."""
        instance = model_factory()
        # Try to construct with an extra field — Pydantic should reject
        with pytest.raises(ValidationError):
            type(instance)(**{**instance.model_dump(), "unknown_field": "intruder"})

    @pytest.mark.parametrize(
        "model_factory,model_name",
        [
            (
                lambda: IngredientDemand(
                    canonical_name="test",
                    raw_name="test",
                    quantity=Decimal(1),
                    unit="g",
                    confidence=Decimal("0.5"),
                ),
                "IngredientDemand",
            ),
            (
                lambda: CookingTask(
                    task_id="t1",
                    dish_id="r1",
                    instruction="Test",
                    duration_minutes=1,
                    work_mode=WorkMode.ACTIVE,
                    category="test",
                ),
                "CookingTask",
            ),
        ],
    )
    def test_frozen_immutable(self, model_factory, model_name):
        """Frozen models reject attribute mutation."""
        instance = model_factory()
        with pytest.raises((ValidationError, ValueError, AttributeError)):
            # Attempt mutation on a frozen model
            instance.quantity = Decimal(999)  # type: ignore[misc]


# =============================================================================
# 2. IngredientDemand
# =============================================================================


class TestIngredientDemand:
    def test_positive_quantity_required(self):
        with pytest.raises(ValidationError, match="greater than 0"):
            IngredientDemand(
                canonical_name="test",
                raw_name="test",
                quantity=Decimal(0),
                unit="g",
                confidence=Decimal("0.5"),
            )

    def test_confidence_range(self):
        # Valid range [0, 1]
        IngredientDemand(
            canonical_name="test",
            raw_name="test",
            quantity=Decimal(10),
            unit="g",
            confidence=Decimal(0),
        )
        IngredientDemand(
            canonical_name="test",
            raw_name="test",
            quantity=Decimal(10),
            unit="g",
            confidence=Decimal(1),
        )

    def test_confidence_out_of_range_high(self):
        with pytest.raises(ValidationError):
            IngredientDemand(
                canonical_name="test",
                raw_name="test",
                quantity=Decimal(10),
                unit="g",
                confidence=Decimal("1.1"),
            )

    def test_confidence_out_of_range_low(self):
        with pytest.raises(ValidationError):
            IngredientDemand(
                canonical_name="test",
                raw_name="test",
                quantity=Decimal(10),
                unit="g",
                confidence=Decimal("-0.1"),
            )

    def test_allergen_tags_default_empty(self):
        demand = IngredientDemand(
            canonical_name="salt",
            raw_name="salt",
            quantity=Decimal(5),
            unit="g",
            confidence=Decimal(1),
        )
        assert demand.allergen_tags == ()

    def test_creation_with_allergens(self):
        demand = IngredientDemand(
            canonical_name="shrimp",
            raw_name="shrimp",
            quantity=Decimal(200),
            unit="g",
            confidence=Decimal("0.9"),
            allergen_tags=("shellfish",),
        )
        assert "shellfish" in demand.allergen_tags


# =============================================================================
# 3. RecipeStep
# =============================================================================


class TestRecipeStep:
    def test_step_number_must_be_positive(self):
        with pytest.raises(ValidationError):
            RecipeStep(step_number=0, instruction="Test")

    def test_minimal_valid_step(self):
        step = RecipeStep(step_number=1, instruction="Do something")
        assert step.step_number == 1
        assert step.category == "general"
        assert step.pattern == "simple"
        assert step.heat_level == HeatLevel.NONE

    def test_with_full_fields(self):
        step = RecipeStep(
            step_number=3,
            instruction="Bake at 200C for 25 minutes",
            category="heating",
            pattern="bake",
            active_duration_minutes=5,
            passive_duration_minutes=25,
            heat_level=HeatLevel.MEDIUM,
            target_temperature_c=Decimal(200),
            interval_minutes=5,
            resources_hint=("oven",),
        )
        assert step.target_temperature_c == Decimal(200)
        assert step.resources_hint == ("oven",)


# =============================================================================
# 4. RecipeIR
# =============================================================================


class TestRecipeIR:
    def test_must_have_at_least_one_ingredient(self):
        step = RecipeStep(step_number=1, instruction="Test")
        with pytest.raises(ValidationError, match="at least one ingredient"):
            RecipeIR(
                recipe_id="r1",
                dish_name="Test",
                original_servings=Decimal(2),
                target_servings=Decimal(2),
                source_language="eng",
                ingredients=(),
                steps=(step,),
            )

    def test_must_have_at_least_one_step(self):
        ingredient = IngredientDemand(
            canonical_name="salt",
            raw_name="salt",
            quantity=Decimal(5),
            unit="g",
            confidence=Decimal(1),
        )
        with pytest.raises(ValidationError, match="at least one step"):
            RecipeIR(
                recipe_id="r1",
                dish_name="Test",
                original_servings=Decimal(2),
                target_servings=Decimal(2),
                source_language="eng",
                ingredients=(ingredient,),
                steps=(),
            )

    def test_valid_recipe_ir(self):
        ingredient = IngredientDemand(
            canonical_name="salt",
            raw_name="salt",
            quantity=Decimal(5),
            unit="g",
            confidence=Decimal(1),
        )
        step = RecipeStep(step_number=1, instruction="Test")
        ir = RecipeIR(
            recipe_id="r1",
            dish_name="Test Dish",
            original_servings=Decimal(2),
            target_servings=Decimal(4),
            source_language="eng",
            ingredients=(ingredient,),
            steps=(step,),
        )
        assert ir.dish_name == "Test Dish"
        assert ir.original_servings == Decimal(2)
        assert ir.target_servings == Decimal(4)


# =============================================================================
# 5. ResourceNeed
# =============================================================================


class TestResourceNeed:
    def test_quantity_must_be_positive(self):
        with pytest.raises(ValidationError):
            ResourceNeed(resource_type="stove", quantity=0)

    def test_minimal_resource(self):
        need = ResourceNeed(resource_type="stove", quantity=1)
        assert need.resource_type == "stove"
        assert need.minimum_capacity is None

    def test_with_capacity(self):
        need = ResourceNeed(
            resource_type="pot",
            quantity=1,
            minimum_capacity=Decimal("2.0"),
            capacity_unit="L",
        )
        assert need.minimum_capacity == Decimal("2.0")

    def test_with_capabilities(self):
        need = ResourceNeed(
            resource_type="stove",
            quantity=1,
            required_capabilities=("induction",),
        )
        assert "induction" in need.required_capabilities


# =============================================================================
# 6. TaskDependency
# =============================================================================


class TestTaskDependency:
    def test_minimum_lag_defaults_to_zero(self):
        dep = TaskDependency(predecessor_id="t1")
        assert dep.minimum_lag_minutes == 0
        assert dep.maximum_lag_minutes is None

    def test_minimum_lag_non_negative(self):
        with pytest.raises(ValidationError):
            TaskDependency(predecessor_id="t1", minimum_lag_minutes=-1)

    def test_maximum_lag_non_negative(self):
        with pytest.raises(ValidationError):
            TaskDependency(predecessor_id="t1", maximum_lag_minutes=-1)

    def test_with_both_lags(self):
        dep = TaskDependency(
            predecessor_id="t1",
            minimum_lag_minutes=5,
            maximum_lag_minutes=10,
        )
        assert dep.minimum_lag_minutes == 5
        assert dep.maximum_lag_minutes == 10


# =============================================================================
# 7. CookingTask
# =============================================================================


class TestCookingTask:
    def test_duration_must_be_positive(self):
        with pytest.raises(ValidationError):
            CookingTask(
                task_id="t1",
                dish_id="r1",
                instruction="Test",
                duration_minutes=0,
                work_mode=WorkMode.ACTIVE,
                category="test",
            )

    def test_negative_duration_rejected(self):
        with pytest.raises(ValidationError):
            CookingTask(
                task_id="t1",
                dish_id="r1",
                instruction="Test",
                duration_minutes=-5,
                work_mode=WorkMode.ACTIVE,
                category="test",
            )

    def test_default_values(self):
        task = CookingTask(
            task_id="t1",
            dish_id="r1",
            instruction="Test",
            duration_minutes=5,
            work_mode=WorkMode.ACTIVE,
            category="test",
        )
        assert task.heat_level == HeatLevel.NONE
        assert task.dependencies == ()
        assert task.resources == ()
        assert task.consumes_states == ()
        assert task.produces_states == ()

    def test_with_safety_tags(self):
        task = CookingTask(
            task_id="t1",
            dish_id="r1",
            instruction="Handle raw chicken",
            duration_minutes=3,
            work_mode=WorkMode.ACTIVE,
            category="prep",
            safety_tags=("raw_meat",),
        )
        assert "raw_meat" in task.safety_tags


# =============================================================================
# 8. InventoryLotSnapshot
# =============================================================================


class TestInventoryLotSnapshot:
    def test_reservation_cannot_exceed_on_hand(self):
        with pytest.raises(ValidationError, match="reserved quantity exceeds"):
            InventoryLotSnapshot(
                lot_id="l1",
                item_id="i1",
                canonical_name="rice",
                on_hand=Decimal(100),
                reserved=Decimal(150),
                unit="g",
            )

    def test_valid_lot(self):
        lot = InventoryLotSnapshot(
            lot_id="l1",
            item_id="i1",
            canonical_name="rice",
            on_hand=Decimal(100),
            reserved=Decimal(50),
            unit="g",
            expiry_date=date(2026, 12, 31),
        )
        assert lot.on_hand == Decimal(100)
        assert lot.reserved == Decimal(50)
        assert lot.expiry_date == date(2026, 12, 31)

    def test_zero_reserved_allowed(self):
        lot = InventoryLotSnapshot(
            lot_id="l1",
            item_id="i1",
            canonical_name="rice",
            on_hand=Decimal(0),
            reserved=Decimal(0),
            unit="g",
        )
        assert lot.on_hand == Decimal(0)

    def test_negative_on_hand_rejected(self):
        with pytest.raises(ValidationError):
            InventoryLotSnapshot(
                lot_id="l1",
                item_id="i1",
                canonical_name="rice",
                on_hand=Decimal(-1),
                reserved=Decimal(0),
                unit="g",
            )


# =============================================================================
# 9. KitchenResourceSnapshot
# =============================================================================


class TestKitchenResourceSnapshot:
    def test_default_available(self):
        res = KitchenResourceSnapshot(
            resource_id="stove:1",
            resource_type="stove",
        )
        assert res.available is True

    def test_unavailable_resource(self):
        res = KitchenResourceSnapshot(
            resource_id="oven:broken",
            resource_type="oven",
            available=False,
        )
        assert res.available is False

    def test_with_capabilities(self):
        res = KitchenResourceSnapshot(
            resource_id="stove:induction",
            resource_type="stove",
            capacity=Decimal(4),
            capacity_unit="burners",
            capabilities=("induction", "temperature_control"),
        )
        assert "induction" in res.capabilities


# =============================================================================
# 10. Response contract models
# =============================================================================


class TestPlanResponses:
    def test_ready_plan_response(self):
        r = ReadyPlanResponse(
            plan_id="p1",
            status="READY",
            solver_status="OPTIMAL",
            makespan_minutes=45,
            timeline=(),
            completion_checklist=(),
            mise_en_place=(),
            dish_completions=(),
        )
        assert r.status == "READY"
        assert r.solver_status == "OPTIMAL"

    def test_confirmation_response(self):
        r = ConfirmationPlanResponse(
            plan_id="p1",
            status="NEEDS_CONFIRMATION",
            questions=("Proceed?",),
        )
        assert r.status == "NEEDS_CONFIRMATION"

    def test_infeasible_response(self):
        r = InfeasiblePlanResponse(
            plan_id="p1",
            status="INFEASIBLE",
            reasons=("Not enough chicken",),
        )
        assert r.status == "INFEASIBLE"

    def test_failed_response(self):
        r = FailedPlanResponse(
            status="FAILED",
            error_code="INTERNAL_ERROR",
            correlation_id="c1",
            message="test error",
        )
        assert r.error_code == "INTERNAL_ERROR"


# =============================================================================
# 11. Safety models
# =============================================================================


class TestSafetyModels:
    def test_safety_finding(self):
        f = SafetyFinding(
            rule_id="RAW_MEAT_CROSS_CONTAMINATION",
            severity="hard_unrepairable",
            description="Raw meat and ready-to-eat share cutting board",
        )
        assert f.severity == "hard_unrepairable"

    def test_safety_report(self):
        finding = SafetyFinding(
            rule_id="r1",
            severity="warning",
            description="test",
        )
        report = SafetyReport(
            report_id="sr1",
            findings=(finding,),
            is_safe=True,
            has_unrepairable=False,
        )
        assert report.is_safe is True
        assert len(report.findings) == 1

    def test_safety_report_with_required_tasks(self):
        report = SafetyReport(
            report_id="sr1",
            findings=(),
            is_safe=True,
            has_unrepairable=False,
            required_safety_task_ids=("sanitize_cutting_board",),
        )
        assert "sanitize_cutting_board" in report.required_safety_task_ids


# =============================================================================
# 12. Feasibility and Repair models
# =============================================================================


class TestFeasibilityModels:
    def test_feasibility_report_feasible(self):
        report = FeasibilityReport(
            report_id="fr1",
            ingredient_shortages=(),
            missing_resources=(),
            is_feasible=True,
        )
        assert report.is_feasible is True

    def test_feasibility_report_with_shortages(self):
        shortage = IngredientFeasibility(
            ingredient_name="chicken",
            required=Decimal(500),
            available=Decimal(300),
            shortage=Decimal(200),
            unit="g",
        )
        report = FeasibilityReport(
            report_id="fr1",
            ingredient_shortages=(shortage,),
            missing_resources=(),
            is_feasible=False,
        )
        assert report.is_feasible is False
        assert report.ingredient_shortages[0].shortage == Decimal(200)

    def test_repair_option(self):
        opt = RepairOption(
            option_id="ro1",
            option_type="reduce_servings",
            description="Reduce to 2 servings",
            changes=("target_servings: 2",),
            effects=("Chicken needed: 250g",),
        )
        assert opt.option_type == "reduce_servings"
        assert opt.revalidation_status == "validated"


# =============================================================================
# 13. Evidence models
# =============================================================================


class TestEvidenceModels:
    def test_evidence_ref(self):
        ref = EvidenceRef(
            source_type="web_search",
            title="Serious Eats",
            url="https://example.com",
            retrieved_at="2026-07-01T00:00:00Z",
        )
        assert ref.source_type == "web_search"

    def test_cooking_evidence(self):
        ev = CookingEvidence(
            operation="stir-fry",
            heat_level=HeatLevel.HIGH,
            duration_min_minutes=3,
            duration_max_minutes=5,
            source_url="https://example.com",
            source_title="Test",
            source_excerpt="Cook 3-5 min.",
        )
        assert ev.operation == "stir-fry"
        assert ev.duration_min_minutes == 3

    def test_reconciled_evidence(self):
        ev = CookingEvidence(
            operation="bake",
            heat_level=HeatLevel.MEDIUM,
            duration_min_minutes=20,
            duration_max_minutes=25,
            source_url="https://a.com",
            source_title="A",
            source_excerpt="20-25 min.",
        )
        rec = ReconciledEvidence(
            heat_level=HeatLevel.MEDIUM,
            duration_min_minutes=20,
            duration_max_minutes=25,
            source_count=1,
            needs_confirmation=False,
            evidence_items=(ev,),
        )
        assert rec.source_count == 1
        assert not rec.needs_confirmation

    def test_evidence_query(self):
        q = EvidenceQuery(
            query_text="stir fry heat level",
            gap_type="critical",
            recipe_context="chicken stir-fry",
            target_fields=("heat_level",),
        )
        assert q.gap_type == "critical"


# =============================================================================
# 14. Assumption and Gap models
# =============================================================================


class TestAssumption:
    def test_valid_assumption(self):
        a = Assumption(text="Assuming 200C for baking", confidence=Decimal("0.7"))
        assert a.text == "Assuming 200C for baking"
        assert a.confidence == Decimal("0.7")

    def test_assumption_default_evidence(self):
        a = Assumption(text="Test", confidence=Decimal("0.5"))
        assert a.evidence == ()


class TestRecipeGap:
    def test_critical_gap(self):
        gap = RecipeGap(
            gap_id="g1",
            recipe_id="r1",
            field_path="steps[0].heat_level",
            gap_class="critical",
            description="Missing heat level",
            confidence=Decimal("0.3"),
        )
        assert gap.gap_class == "critical"
        assert gap.current_value is None

    def test_gap_with_current_value(self):
        gap = RecipeGap(
            gap_id="g1",
            recipe_id="r1",
            field_path="steps[0].duration",
            current_value="0",
            gap_class="critical",
            description="Duration is zero",
            confidence=Decimal("0.5"),
        )
        assert gap.current_value == "0"


class TestWorkflowError:
    def test_minimal_error(self):
        e = WorkflowError(
            error_code="INTERNAL_ERROR",
            message="test",
            correlation_id="c1",
        )
        assert e.error_code == "INTERNAL_ERROR"
        assert e.recoverable is False

    def test_with_node_name(self):
        e = WorkflowError(
            error_code="SCHEDULE_INFEASIBLE",
            message="Cannot schedule",
            correlation_id="c1",
            node_name="solve_schedule",
            recoverable=True,
        )
        assert e.node_name == "solve_schedule"
        assert e.recoverable is True


# =============================================================================
# 15. Extraction models
# =============================================================================


class TestExtractionModels:
    def test_extracted_ingredient(self):
        ei = ExtractedIngredient(
            raw_text="chicken 200g",
            name="chicken breast",
            quantity=Decimal(200),
            unit="g",
        )
        assert ei.extraction_source == "EXPLICIT"
        assert ei.confidence == Decimal("1.0")

    def test_extracted_step(self):
        es = ExtractedStep(
            step_number=1,
            instruction="Boil water",
            active_duration_minutes=2,
            heat_level=HeatLevel.HIGH,
        )
        assert es.category == "general"
        assert es.extraction_source == "EXPLICIT"

    def test_extracted_recipe_candidate(self):
        ingredient = ExtractedIngredient(
            raw_text="salt",
            name="salt",
            quantity=Decimal(1),
            unit="g",
        )
        step = ExtractedStep(step_number=1, instruction="Mix")
        candidate = ExtractedRecipeCandidate(
            recipe_id="r1",
            dish_name="Test",
            original_servings=Decimal(2),
            source_language="eng",
            ingredients=(ingredient,),
            steps=(step,),
        )
        assert candidate.dish_name == "Test"


# =============================================================================
# 16. WorkflowException — domain error code mapping
# =============================================================================


class TestWorkflowException:
    def test_raises_and_carries_code(self):
        exc = WorkflowException(
            DomainErrorCode.SAFETY_CONSTRAINT_VIOLATION,
            "Cross-contamination detected",
        )
        assert exc.code == DomainErrorCode.SAFETY_CONSTRAINT_VIOLATION
        assert "SAFETY_CONSTRAINT_VIOLATION" in str(exc)

    def test_inherits_from_exception(self):
        exc = WorkflowException(DomainErrorCode.INTERNAL_ERROR, "test")
        assert isinstance(exc, Exception)

    @pytest.mark.parametrize(
        "code",
        [
            DomainErrorCode.INVALID_RECIPE_TEXT,
            DomainErrorCode.UNSUPPORTED_UNIT_CONVERSION,
            DomainErrorCode.SAFETY_CONSTRAINT_VIOLATION,
            DomainErrorCode.INSUFFICIENT_INVENTORY,
            DomainErrorCode.NO_COMPATIBLE_RESOURCE,
            DomainErrorCode.TASK_GRAPH_CYCLE,
            DomainErrorCode.SCHEDULE_INFEASIBLE,
            DomainErrorCode.SCHEDULE_UNKNOWN,
            DomainErrorCode.SCHEDULE_VERIFICATION_FAILED,
            DomainErrorCode.EXTERNAL_PROVIDER_UNAVAILABLE,
            DomainErrorCode.INTERNAL_ERROR,
        ],
    )
    def test_all_error_codes_have_string_value(self, code):
        """Every DomainErrorCode must be usable as a string."""
        assert isinstance(code.value, str)
        assert len(code.value) > 0


# =============================================================================
# 17. Enum integrity
# =============================================================================


class TestEnums:
    def test_plan_status_values(self):
        assert PlanStatus.READY.value == "READY"
        assert PlanStatus.NEEDS_CONFIRMATION.value == "NEEDS_CONFIRMATION"
        assert PlanStatus.INFEASIBLE.value == "INFEASIBLE"
        assert PlanStatus.FAILED.value == "FAILED"

    def test_solver_status_values(self):
        assert SolverStatus.OPTIMAL.value == "OPTIMAL"
        assert SolverStatus.FEASIBLE.value == "FEASIBLE"
        assert SolverStatus.INFEASIBLE.value == "INFEASIBLE"
        assert SolverStatus.MODEL_INVALID.value == "MODEL_INVALID"
        assert SolverStatus.UNKNOWN.value == "UNKNOWN"

    def test_work_mode_values(self):
        assert WorkMode.ACTIVE.value == "ACTIVE"
        assert WorkMode.PASSIVE.value == "PASSIVE"

    def test_heat_level_values(self):
        assert HeatLevel.NONE.value == "NONE"
        assert HeatLevel.LOW.value == "LOW"
        assert HeatLevel.MEDIUM.value == "MEDIUM"
        assert HeatLevel.HIGH.value == "HIGH"


# =============================================================================
# 18. GeneratePlanRequest — API boundary
# =============================================================================


class TestGeneratePlanRequest:
    def test_schema_version_default(self):
        req = GeneratePlanRequest(
            request_id="r1",
            user_id="u1",
            recipes=({"recipe_id": "r", "text": "boil", "target_servings": "2"},),
        )
        assert req.schema_version == "1.0"

    def test_optional_fields_default(self):
        req = GeneratePlanRequest(
            request_id="r1",
            user_id="u1",
            recipes=({"recipe_id": "r", "text": "boil", "target_servings": "2"},),
        )
        assert req.dietary_restrictions == ()
        assert req.user_allergens == ()
        assert req.time_limit_minutes is None
        assert req.inventory_lots == ()
        assert req.kitchen_resources == ()

    def test_with_inventory_and_resources(self):
        lot = InventoryLotSnapshot(
            lot_id="l1",
            item_id="i1",
            canonical_name="chicken",
            on_hand=Decimal(500),
            reserved=Decimal(0),
            unit="g",
        )
        res = KitchenResourceSnapshot(
            resource_id="stove:1",
            resource_type="stove",
        )
        req = GeneratePlanRequest(
            request_id="r1",
            user_id="u1",
            recipes=({"recipe_id": "r", "text": "boil", "target_servings": "2"},),
            inventory_lots=(lot,),
            kitchen_resources=(res,),
        )
        assert len(req.inventory_lots) == 1
        assert len(req.kitchen_resources) == 1


# =============================================================================
# 19. Annotated type guards
# =============================================================================


class TestAnnotatedTypes:
    def test_positive_decimal_rejects_zero(self):
        with pytest.raises(ValidationError):
            LotAllocation(inventory_lot_id="l1", quantity=Decimal(0), unit="g")

    def test_positive_decimal_rejects_negative(self):
        with pytest.raises(ValidationError):
            LotAllocation(inventory_lot_id="l1", quantity=Decimal(-1), unit="g")

    def test_positive_decimal_accepts_positive(self):
        alloc = LotAllocation(inventory_lot_id="l1", quantity=Decimal(100), unit="g")
        assert alloc.quantity == Decimal(100)

    def test_confidence_rejects_gt_1(self):
        with pytest.raises(ValidationError):
            IngredientDemand(
                canonical_name="x",
                raw_name="x",
                quantity=Decimal(1),
                unit="g",
                confidence=Decimal("1.5"),
            )
