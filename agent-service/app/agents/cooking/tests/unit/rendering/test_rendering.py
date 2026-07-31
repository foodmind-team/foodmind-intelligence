"""Unit tests for rendering builders and responses — handbook 11.1–11.10."""

from datetime import date
from decimal import Decimal

import pytest

from cooking_plan_agent.domain.enums import HeatLevel, SolverStatus, WorkMode
from cooking_plan_agent.domain.models import (
    Assumption,
    CompletionItem,
    ConfirmationPlanResponse,
    CookingTask,
    FailedPlanResponse,
    FeasibilityReport,
    GeneratePlanRequest,
    InfeasiblePlanResponse,
    IngredientFeasibility,
    InventoryConsumptionProposal,
    InventoryLotSnapshot,
    LotAllocation,
    ReadyPlanResponse,
    RepairOption,
    SafetyFinding,
    SafetyReport,
    WorkflowError,
)
from cooking_plan_agent.rendering.builder import (
    build_completion_checklist,
    build_dish_completion_summary,
    build_mise_en_place,
    build_timeline,
    validate_completion_checklist,
)
from cooking_plan_agent.rendering.responses import (
    render_confirmation_response,
    render_failed_response,
    render_infeasible_response,
    render_ready_response,
    validate_terminal_response,
)
from cooking_plan_agent.scheduling.models import (
    ScheduledInterval,
    ScheduleResult,
    VerificationIssue,
)
from cooking_plan_agent.workflow.state import PlanState


# ======================================================================
# Fixtures
# ======================================================================


def _make_task(task_id: str, dish_id: str, instruction: str, duration: int,
               category: str = "general", work_mode: WorkMode = WorkMode.ACTIVE,
               resources: tuple = ()) -> CookingTask:
    return CookingTask(
        task_id=task_id,
        dish_id=dish_id,
        instruction=instruction,
        duration_minutes=duration,
        work_mode=work_mode,
        category=category,
        resources=resources,
    )


def _make_prep_task(task_id: str, instruction: str, duration: int,
                    consumes: tuple[str, ...] = (),
                    produces: tuple[str, ...] = ()) -> CookingTask:
    return CookingTask(
        task_id=task_id,
        dish_id="shared",
        instruction=instruction,
        duration_minutes=duration,
        work_mode=WorkMode.ACTIVE,
        category="preparation",
        consumes_states=consumes,
        produces_states=produces,
    )


def _make_state(**overrides) -> PlanState:
    base = PlanState(
        request=GeneratePlanRequest(
            request_id="req-1",
            user_id="user-1",
            recipes=({"recipe_id": "r1", "text": "test", "target_servings": 2},),
        ),
    )
    base.update(overrides)  # type: ignore[arg-type]
    return base


@pytest.fixture
def sample_tasks():
    return (
        _make_task("t1", "dish-a", "Boil water", 10, "heating"),
        _make_task("t2", "dish-a", "Cook pasta", 8, "heating"),
        _make_task("t3", "dish-b", "Chop vegetables", 5, "cutting"),
        _make_task("t4", "dish-b", "Stir-fry", 7, "heating", WorkMode.ACTIVE),
    )


@pytest.fixture
def sample_intervals():
    return (
        ScheduledInterval(task_id="t1", start_minute=0, end_minute=10),
        ScheduledInterval(task_id="t2", start_minute=10, end_minute=18),
        ScheduledInterval(task_id="t3", start_minute=0, end_minute=5),
        ScheduledInterval(task_id="t4", start_minute=5, end_minute=12),
    )


@pytest.fixture
def sample_schedule(sample_intervals):
    return ScheduleResult(
        status=SolverStatus.OPTIMAL,
        makespan_minutes=18,
        intervals=sample_intervals,
    )


@pytest.fixture
def prep_tasks():
    return (
        _make_prep_task("p1", "[Prep] wash 500.0 of chicken breast", 3,
                         produces=("chicken:washed:shared",)),
        _make_prep_task("p2", "[Prep] dice 500.0 of chicken breast", 5,
                         consumes=("chicken:washed:shared",),
                         produces=("chicken:diced:shared",)),
        _make_prep_task("p3", "[Prep] wash 300.0 of tomato", 2,
                         produces=("tomato:washed:shared",)),
    )


@pytest.fixture
def sample_proposal():
    return InventoryConsumptionProposal(
        inventory_snapshot_version="v1_test",
        items=(
            CompletionItem(
                completion_item_id="comp-1",
                ingredient_name="chicken breast",
                recipe_ids=("r1",),
                allocations=(
                    LotAllocation(
                        inventory_lot_id="lot-001",
                        quantity=Decimal(400),
                        unit="g",
                    ),
                ),
            ),
            CompletionItem(
                completion_item_id="comp-2",
                ingredient_name="tomato",
                recipe_ids=("r1", "r2"),
                allocations=(
                    LotAllocation(
                        inventory_lot_id="lot-002",
                        quantity=Decimal(4),
                        unit="piece",
                    ),
                ),
            ),
        ),
    )


# ======================================================================
# build_mise_en_place
# ======================================================================


class TestBuildMiseEnPlace:
    def test_extracts_prep_tasks(self, prep_tasks):
        result = build_mise_en_place(prep_tasks)
        assert len(result) >= 2
        instructions = {i["instruction"] for i in result}
        assert any("chicken" in inst for inst in instructions)
        assert any("tomato" in inst for inst in instructions)

    def test_empty_tasks(self):
        assert build_mise_en_place(()) == ()

    def test_no_prep_tasks(self, sample_tasks):
        # sample_tasks have no preparation category
        result = build_mise_en_place(sample_tasks)
        assert result == ()

    def test_includes_resource_info(self, prep_tasks):
        result = build_mise_en_place(prep_tasks)
        for item in result:
            assert "resources" in item
            assert isinstance(item["resources"], list)

    def test_sorted_by_duration_desc(self):
        tasks = (
            _make_prep_task("p1", "[Prep] quick 100.0 of item", 1),
            _make_prep_task("p2", "[Prep] slow 100.0 of item2", 10),
            _make_prep_task("p3", "[Prep] medium 100.0 of item3", 5),
        )
        result = build_mise_en_place(tasks)
        durations = [i["duration_minutes"] for i in result]
        assert durations == sorted(durations, reverse=True)

    def test_when_needed_from_state(self, prep_tasks):
        result = build_mise_en_place(prep_tasks)
        wash_chicken = [i for i in result if "wash" in i["operation"]]
        if wash_chicken:
            assert wash_chicken[0]["when_needed"] is not None


# ======================================================================
# build_timeline
# ======================================================================


class TestBuildTimeline:
    def test_sorted_by_start_time(self, sample_schedule, sample_tasks):
        result = build_timeline(sample_schedule, sample_tasks)
        assert len(result) == 4
        starts = [i["start_minute"] for i in result]
        assert starts == sorted(starts)

    def test_includes_all_fields(self, sample_schedule, sample_tasks):
        result = build_timeline(sample_schedule, sample_tasks)
        for item in result:
            assert "task_id" in item
            assert "start_minute" in item
            assert "end_minute" in item
            assert "instruction" in item
            assert "dish_id" in item
            assert "work_mode" in item

    def test_empty_intervals(self, sample_tasks):
        schedule = ScheduleResult(status=SolverStatus.INFEASIBLE)
        assert build_timeline(schedule, sample_tasks) == ()

    def test_task_not_found_skipped(self, sample_schedule):
        """Interval references a task not in the task list → skipped."""
        result = build_timeline(sample_schedule, ())  # No tasks
        assert result == ()


# ======================================================================
# build_dish_completion_summary
# ======================================================================


class TestBuildDishCompletionSummary:
    def test_calculates_per_dish_completion(self, sample_schedule, sample_tasks):
        result = build_dish_completion_summary(sample_schedule, sample_tasks)
        # dish-a: max(t1=10, t2=18) = 18; dish-b: max(t3=5, t4=12) = 12
        assert len(result) >= 2

        dish_a = [d for d in result if d["dish_id"] == "dish-a"]
        assert len(dish_a) == 1
        assert dish_a[0]["completion_minute"] == 18

        dish_b = [d for d in result if d["dish_id"] == "dish-b"]
        assert len(dish_b) == 1
        assert dish_b[0]["completion_minute"] == 12

    def test_sorted_by_completion(self, sample_schedule, sample_tasks):
        result = build_dish_completion_summary(sample_schedule, sample_tasks)
        completions = [d["completion_minute"] for d in result]
        assert completions == sorted(completions)

    def test_empty_intervals(self, sample_tasks):
        schedule = ScheduleResult(status=SolverStatus.INFEASIBLE)
        assert build_dish_completion_summary(schedule, sample_tasks) == ()


# ======================================================================
# build_completion_checklist
# ======================================================================


class TestBuildCompletionChecklist:
    def test_passthrough(self, sample_proposal):
        result = build_completion_checklist(sample_proposal)
        assert len(result) == 2
        assert result[0].ingredient_name == "chicken breast"

    def test_empty_proposal(self):
        proposal = InventoryConsumptionProposal(
            inventory_snapshot_version="v1",
            items=(),
        )
        assert build_completion_checklist(proposal) == ()


# ======================================================================
# validate_completion_checklist
# ======================================================================


class TestValidateCompletionChecklist:
    def test_valid_checklist(self, sample_proposal):
        checklist = build_completion_checklist(sample_proposal)
        issues = validate_completion_checklist(sample_proposal, checklist)
        assert issues == ()

    def test_empty_checklist_with_proposal_items(self, sample_proposal):
        issues = validate_completion_checklist(sample_proposal, ())
        assert len(issues) == 1
        assert issues[0].code == "EMPTY_CHECKLIST"

    def test_missing_item_id(self, sample_proposal):
        bad = (
            CompletionItem(
                completion_item_id="",
                ingredient_name="test",
                recipe_ids=(),
                allocations=(LotAllocation(inventory_lot_id="lot-1", quantity=Decimal(1), unit="g"),),
            ),
        )
        issues = validate_completion_checklist(sample_proposal, bad)
        assert any(i.code == "MISSING_ITEM_ID" for i in issues)

    def test_missing_lot_id_caught(self, sample_proposal):
        bad = (
            CompletionItem(
                completion_item_id="comp-1",
                ingredient_name="test",
                recipe_ids=(),
                allocations=(
                    LotAllocation(inventory_lot_id="", quantity=Decimal(1), unit="g"),
                ),
            ),
        )
        issues = validate_completion_checklist(sample_proposal, bad)
        assert any(i.code == "MISSING_LOT_ID" for i in issues)

    def test_duplicate_lot_allocation(self):
        proposal = InventoryConsumptionProposal(
            inventory_snapshot_version="v1",
            items=(
                CompletionItem(
                    completion_item_id="comp-1",
                    ingredient_name="chicken",
                    recipe_ids=(),
                    allocations=(
                        LotAllocation(inventory_lot_id="lot-1", quantity=Decimal(200), unit="g"),
                        LotAllocation(inventory_lot_id="lot-1", quantity=Decimal(200), unit="g"),
                    ),
                ),
            ),
        )
        checklist = build_completion_checklist(proposal)
        issues = validate_completion_checklist(proposal, checklist)
        assert any(i.code == "DUPLICATE_LOT_ALLOCATION" for i in issues)


# ======================================================================
# render_ready_response
# ======================================================================


class TestRenderReadyResponse:
    def test_produces_ready_response(self, sample_schedule, sample_tasks, prep_tasks):
        all_tasks = sample_tasks + prep_tasks
        state = _make_state(
            schedule_result=sample_schedule,
            recipe_tasks=sample_tasks,
            prep_tasks=prep_tasks,
        )
        # Manually inject task_graph since render_ready uses it
        from cooking_plan_agent.preparation.task_graph import TaskGraph
        state["task_graph"] = TaskGraph(tasks=all_tasks, edges=())

        response = render_ready_response(state)
        assert isinstance(response, ReadyPlanResponse)
        assert response.status == "READY"
        assert response.makespan_minutes == 18
        assert response.solver_status == "OPTIMAL"
        assert len(response.timeline) == 4
        assert len(response.mise_en_place) >= 2

    def test_no_schedule_handled_gracefully(self):
        state = _make_state()
        response = render_ready_response(state)
        assert response.status == "READY"
        assert response.solver_status == "UNKNOWN"
        assert response.makespan_minutes == 0
        assert response.timeline == ()

    def test_includes_feasibility_checklist(self):
        feasibility = FeasibilityReport(
            report_id="r1",
            ingredient_shortages=(
                IngredientFeasibility(
                    ingredient_name="chicken breast",
                    required=Decimal(400),
                    available=Decimal(400),
                    shortage=Decimal(0),
                    unit="g",
                    proposed_allocations=(
                        LotAllocation(inventory_lot_id="lot-001", quantity=Decimal(400), unit="g"),
                    ),
                ),
            ),
            is_feasible=True,
        )
        state = _make_state(feasibility_report=feasibility)
        response = render_ready_response(state)
        assert len(response.completion_checklist) == 1
        assert response.completion_checklist[0].ingredient_name == "chicken breast"


# ======================================================================
# render_confirmation_response
# ======================================================================


class TestRenderConfirmationResponse:
    def test_includes_assumptions_and_options(self):
        options = (
            RepairOption(
                option_id="r1",
                option_type="substitute_ingredient",
                description="Sub",
                changes=("Change",),
                effects=("Effect",),
            ),
        )
        state = _make_state(repair_options=options)
        response = render_confirmation_response(state)
        assert isinstance(response, ConfirmationPlanResponse)
        assert response.status == "NEEDS_CONFIRMATION"
        assert len(response.repair_options) == 1
        assert len(response.questions) > 0

    def test_empty_state(self):
        state = _make_state()
        response = render_confirmation_response(state)
        assert response.assumptions == ()
        assert response.repair_options == ()
        assert len(response.questions) == 1  # Fallback question


# ======================================================================
# render_infeasible_response
# ======================================================================


class TestRenderInfeasibleResponse:
    def test_safety_reasons_included(self):
        report = SafetyReport(
            report_id="s1",
            findings=(
                SafetyFinding(
                    rule_id="R1",
                    severity="hard_unrepairable",
                    description="Allergen detected: peanut",
                ),
            ),
            is_safe=False,
            has_unrepairable=True,
        )
        state = _make_state(safety_report=report)
        response = render_infeasible_response(state)
        assert isinstance(response, InfeasiblePlanResponse)
        assert any("peanut" in r for r in response.reasons)

    def test_feasibility_reasons_included(self):
        report = FeasibilityReport(
            report_id="f1",
            ingredient_shortages=(
                IngredientFeasibility(
                    ingredient_name="chicken breast",
                    required=Decimal(400),
                    available=Decimal(100),
                    shortage=Decimal(300),
                    unit="g",
                ),
            ),
            missing_resources=("oven",),
            is_feasible=False,
        )
        state = _make_state(feasibility_report=report)
        response = render_infeasible_response(state)
        assert any("chicken breast" in r for r in response.reasons)
        assert any("oven" in r for r in response.reasons)

    def test_fallback_reason(self):
        state = _make_state()
        response = render_infeasible_response(state)
        assert len(response.reasons) >= 1


# ======================================================================
# render_failed_response
# ======================================================================


class TestRenderFailedResponse:
    def test_error_from_state(self):
        error = WorkflowError(
            error_code="TEST_ERROR",
            message="Test failure",
            correlation_id="corr-123",
        )
        state = _make_state(error=error)
        response = render_failed_response(state)
        assert isinstance(response, FailedPlanResponse)
        assert response.error_code == "TEST_ERROR"
        assert response.correlation_id == "corr-123"
        assert response.message == "Test failure"

    def test_fallback_when_no_error(self):
        state = _make_state()
        response = render_failed_response(state)
        assert response.error_code == "INTERNAL_ERROR"
        assert response.correlation_id == "req-1"


# ======================================================================
# validate_terminal_response
# ======================================================================


class TestValidateTerminalResponse:
    def test_valid_ready(self):
        r = ReadyPlanResponse(
            plan_id="p1", solver_status="OPTIMAL", makespan_minutes=30,
            timeline=(), completion_checklist=(), mise_en_place=(), dish_completions=(),
        )
        assert validate_terminal_response(r) is r

    def test_ready_negative_makespan(self):
        r = ReadyPlanResponse(
            plan_id="p1", solver_status="OPTIMAL", makespan_minutes=0,
            timeline=(), completion_checklist=(), mise_en_place=(), dish_completions=(),
        )
        with pytest.raises(ValueError, match="makespan"):
            validate_terminal_response(r)

    def test_valid_confirmation(self):
        r = ConfirmationPlanResponse(
            plan_id="p1",
            questions=("Proceed?",),
        )
        assert validate_terminal_response(r) is r

    def test_confirmation_empty(self):
        r = ConfirmationPlanResponse(plan_id="p1")
        with pytest.raises(ValueError):
            validate_terminal_response(r)

    def test_valid_infeasible(self):
        r = InfeasiblePlanResponse(plan_id="p1", reasons=("No inventory",))
        assert validate_terminal_response(r) is r

    def test_infeasible_no_reasons(self):
        r = InfeasiblePlanResponse(plan_id="p1", reasons=())
        with pytest.raises(ValueError, match="reason"):
            validate_terminal_response(r)

    def test_valid_failed(self):
        r = FailedPlanResponse(
            error_code="ERR", correlation_id="c1", message="fail",
        )
        assert validate_terminal_response(r) is r

    def test_failed_empty_error_code(self):
        r = FailedPlanResponse(error_code="", correlation_id="c1", message="fail")
        with pytest.raises(ValueError, match="error_code"):
            validate_terminal_response(r)
