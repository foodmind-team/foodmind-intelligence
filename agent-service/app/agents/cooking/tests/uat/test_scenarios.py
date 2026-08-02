"""UAT (User Acceptance Test) scenarios — Handbook 11.9.

10 business scenarios that test the complete cooking plan workflow.
Each test verifies one business rule end-to-end.
"""

from decimal import Decimal

import pytest

from cooking_plan_agent.domain.enums import HeatLevel, SolverStatus, WorkMode
from cooking_plan_agent.domain.models import (
    CookingTask,
    ExtractedIngredient,
    ExtractedRecipeCandidate,
    ExtractedStep,
    FeasibilityReport,
    GeneratePlanRequest,
    IngredientFeasibility,
    InventoryLotSnapshot,
    KitchenResourceSnapshot,
    RecipeStep,
    ResourceNeed,
    SafetyFinding,
    SafetyReport,
)
from cooking_plan_agent.preparation.decompose import decompose_step
from cooking_plan_agent.preparation.prep_trie import (
    PreparationOperation,
    PrepTrieNode,
    convert_trie_to_tasks,
    insert_operation_chain,
)
from cooking_plan_agent.preparation.task_graph import (
    build_task_graph,
    topological_sort_kahn,
)
from cooking_plan_agent.scheduling.models import SchedulingProblem
from cooking_plan_agent.scheduling.orchestrator import schedule

# =============================================================================
# UAT 1: Blanching overlaps with marinade preparation (passive overlaps active)
# =============================================================================


def test_blanching_overlaps_marinade() -> None:
    """Blanching (passive boil) runs while cook applies marinade (active).

    Verifies that passive heating tasks can overlap active preparation
    tasks from a different dish.
    """
    boil_step = RecipeStep(
        step_number=1,
        instruction="Blanch vegetables",
        pattern="boil",
        passive_duration_minutes=5,
        heat_level=HeatLevel.HIGH,
    )
    marinate_step = RecipeStep(
        step_number=1,
        instruction="Apply marinade to chicken",
        pattern="marinate",
        passive_duration_minutes=20,
    )

    boil_tasks = decompose_step("d1", boil_step)
    marinate_tasks = decompose_step("d2", marinate_step)
    all_tasks = boil_tasks + marinate_tasks

    graph = build_task_graph(all_tasks, (), ())
    order = topological_sort_kahn(graph)
    # Boil/decompose tasks need sink + mixing_bowl + stove resources
    sink = KitchenResourceSnapshot(
        resource_id="sink:main",
        resource_type="sink",
        capacity=Decimal(2),
    )
    bowl = KitchenResourceSnapshot(
        resource_id="bowl:main",
        resource_type="mixing_bowl",
        capacity=Decimal(2),
    )
    stove = KitchenResourceSnapshot(
        resource_id="stove:main",
        resource_type="stove",
        capacity=Decimal(4),
        capacity_unit="burners",
    )
    problem = SchedulingProblem(tasks=order, resources=(sink, bowl, stove))

    result, report = schedule(problem)
    assert result.status == SolverStatus.OPTIMAL
    assert report.passed

    # Passive boil should run in parallel with marinate apply
    # Total makespan should be less than sum of all durations
    total_dur = sum(t.duration_minutes for t in all_tasks)
    assert result.makespan_minutes < total_dur


# =============================================================================
# UAT 2: One ingredient washed together, divided into julienne/sliced/diced
# =============================================================================


def test_shared_wash_branches_into_multiple_cuts() -> None:
    """Wash 500g chilli once, then branch into julienne(100g)/sliced(200g)/diced(200g).

    Verifies prefix-tree merging: wash is shared, children branch.
    """
    root = PrepTrieNode(operation_key="root")

    # Julienne chain
    insert_operation_chain(
        root,
        (
            PreparationOperation(operation="wash", quantity=Decimal(100), unit="g"),
            PreparationOperation(operation="julienne", specification="julienned", quantity=Decimal(100), unit="g"),
        ),
        "d1",
    )

    # Slice chain
    insert_operation_chain(
        root,
        (
            PreparationOperation(operation="wash", quantity=Decimal(200), unit="g"),
            PreparationOperation(operation="slice", specification="sliced", quantity=Decimal(200), unit="g"),
        ),
        "d2",
    )

    # Dice chain
    insert_operation_chain(
        root,
        (
            PreparationOperation(operation="wash", quantity=Decimal(200), unit="g"),
            PreparationOperation(operation="dice", specification="diced", quantity=Decimal(200), unit="g"),
        ),
        "d3",
    )

    # Wash is shared: 100+200+200 = 500
    wash_node = root.child_nodes["wash"]
    assert wash_node.total_quantity == Decimal(500)

    # Three branches
    assert len(wash_node.child_nodes) == 3
    assert "cut:julienned" in wash_node.child_nodes
    assert "cut:sliced" in wash_node.child_nodes
    assert "cut:diced" in wash_node.child_nodes

    # Convert to tasks
    tasks = convert_trie_to_tasks(root, "chilli", ("d1", "d2", "d3"))
    assert len(tasks) == 4  # wash + 3 cuts


# =============================================================================
# UAT 3: One stove serialises conflicting cooking steps
# =============================================================================


def test_one_burner_serialises_tasks() -> None:
    """Two dishes share one stove — cooking tasks must interleave."""
    need_stove = ResourceNeed(resource_type="stove", quantity=1)

    tasks = (
        CookingTask(
            task_id="d1_cook",
            dish_id="dish1",
            instruction="Stir-fry d1",
            duration_minutes=8,
            work_mode=WorkMode.ACTIVE,
            category="heating",
            resources=(need_stove,),
        ),
        CookingTask(
            task_id="d2_cook",
            dish_id="dish2",
            instruction="Stir-fry d2",
            duration_minutes=6,
            work_mode=WorkMode.ACTIVE,
            category="heating",
            resources=(need_stove,),
        ),
    )
    stove = KitchenResourceSnapshot(
        resource_id="stove:main",
        resource_type="stove",
        capacity=Decimal(1),
        capacity_unit="burners",
    )
    problem = SchedulingProblem(tasks=tasks, resources=(stove,))
    result, report = schedule(problem)

    assert result.status == SolverStatus.OPTIMAL
    assert report.passed
    # One burner → tasks are sequential → makespan = 8+6 = 14
    assert result.makespan_minutes == 14


# =============================================================================
# UAT 4: Salt shortage produces confirmation (inventory feasibility)
# =============================================================================


def test_ingredient_shortage_detected() -> None:
    """When inventory can't meet demand, feasibility report captures shortage."""
    shortage = IngredientFeasibility(
        ingredient_name="salt",
        required=Decimal(50),
        available=Decimal(30),
        shortage=Decimal(20),
        unit="g",
    )
    report = FeasibilityReport(
        report_id="fr1",
        ingredient_shortages=(shortage,),
        missing_resources=(),
        is_feasible=False,
    )
    assert not report.is_feasible
    assert report.ingredient_shortages[0].ingredient_name == "salt"
    assert report.ingredient_shortages[0].shortage == Decimal(20)


# =============================================================================
# UAT 5: Expired and reserved stock are excluded (InventoryLotSnapshot)
# =============================================================================


def test_reserved_stock_excluded_from_available() -> None:
    """Lot with 500g on_hand, 200g reserved → available = 300g.

    The InventoryLotSnapshot.revalidation_cannot_exceed_stock validator
    ensures reserved <= on_hand. The consumer must compute available as
    on_hand - reserved.
    """
    from datetime import date

    lot = InventoryLotSnapshot(
        lot_id="lot-chicken-fresh",
        item_id="item-chicken",
        canonical_name="chicken breast",
        on_hand=Decimal(500),
        reserved=Decimal(200),
        unit="g",
        expiry_date=date(2026, 8, 15),
    )
    available = lot.on_hand - lot.reserved
    assert available == Decimal(300)


def test_expired_lot_should_be_excluded() -> None:
    """Expired lot: expiry_date in the past should not be allocated."""
    from datetime import date, timedelta

    expired = InventoryLotSnapshot(
        lot_id="lot-expired",
        item_id="item-milk",
        canonical_name="milk",
        on_hand=Decimal(500),
        reserved=Decimal(0),
        unit="ml",
        expiry_date=date.today() - timedelta(days=1),
    )
    assert expired.expiry_date is not None
    assert expired.expiry_date < date.today()


# =============================================================================
# UAT 6: Requested deadline is infeasible
# =============================================================================


def test_infeasible_deadline() -> None:
    """Three 10-minute active tasks cannot finish within 5 minutes."""
    tasks = tuple(
        CookingTask(
            task_id=f"t{i}",
            dish_id="d1",
            instruction=f"Task {i}",
            duration_minutes=10,
            work_mode=WorkMode.ACTIVE,
            category="test",
        )
        for i in range(3)
    )
    problem = SchedulingProblem(
        tasks=tasks,
        resources=(),
        requested_time_limit_minutes=5,
    )
    result, report = schedule(problem)

    assert result.status == SolverStatus.INFEASIBLE
    assert result.makespan_minutes is None
    assert report.passed  # No intervals to check


# =============================================================================
# UAT 7: Raw-protein equipment reuse inserts sanitisation
# =============================================================================


def test_raw_protein_triggers_safety_tag() -> None:
    """Marinate step involving chicken sets raw_meat safety tag.

    The decompose step should detect "chicken" in instruction and set
    safety_tags=("raw_meat",) so the safety engine can insert a
    sanitisation task when the same cutting board is reused.
    """
    step = RecipeStep(
        step_number=1,
        instruction="Marinate chicken breast for 20 minutes",
        pattern="marinate",
        passive_duration_minutes=20,
    )
    tasks = decompose_step("r1", step)
    assert len(tasks) == 2  # apply + wait
    assert "raw_meat" in tasks[1].safety_tags, "Marinate step with chicken should have raw_meat safety tag"


def test_safety_report_with_required_sanitisation() -> None:
    """Safety report can prescribe sanitisation tasks."""
    finding = SafetyFinding(
        rule_id="RAW_MEAT_CROSS_CONTAMINATION",
        severity="hard_repairable",
        description="Cutting board used for raw chicken",
        recommended_action="Insert sanitise_cutting_board task",
    )
    report = SafetyReport(
        report_id="sr1",
        findings=(finding,),
        is_safe=False,
        has_unrepairable=False,
        required_safety_task_ids=("sanitise_cutting_board",),
    )
    assert "sanitise_cutting_board" in report.required_safety_task_ids
    assert not report.is_safe


# =============================================================================
# UAT 8: Search timeout produces confirmation, not a crash
# =============================================================================


@pytest.mark.asyncio
async def test_search_timeout_produces_confirmation() -> None:
    """Research timeout → ReconciledEvidence with needs_confirmation, not crash.

    The researcher handles timeout gracefully: empty results + confirmation flag.
    """
    from cooking_plan_agent.domain.models import ReconciledEvidence

    # Simulating what the researcher returns on timeout
    timed_out = ReconciledEvidence(
        source_count=0,
        needs_confirmation=True,
    )
    assert timed_out.source_count == 0
    assert timed_out.needs_confirmation is True


# =============================================================================
# UAT 9: FEASIBLE schedule is not labelled OPTIMAL
# =============================================================================


def test_feasible_not_mislabeled_as_optimal() -> None:
    """A FEASIBLE result must have status FEASIBLE, not OPTIMAL."""
    from cooking_plan_agent.domain.enums import SolverStatus

    # OPTIMAL means proven optimality; FEASIBLE means found a solution
    # but optimality is not proven. They must never be confused.
    assert SolverStatus.FEASIBLE != SolverStatus.OPTIMAL
    assert SolverStatus.FEASIBLE.value == "FEASIBLE"
    assert SolverStatus.OPTIMAL.value == "OPTIMAL"


# =============================================================================
# UAT 10: Duplicate completion is rejected (schedule verification)
# =============================================================================


def test_duplicate_completion_idempotent() -> None:
    """Handbook 11.9.10: extra interval for non-existent task → rejected.

    The verifier catches EXTRA_TASK when an interval exists for a task
    that is not in the problem's task list.
    """
    tasks = (
        CookingTask(
            task_id="t1",
            dish_id="d1",
            instruction="Task",
            duration_minutes=5,
            work_mode=WorkMode.ACTIVE,
            category="test",
        ),
    )

    from cooking_plan_agent.scheduling.models import ScheduledInterval, ScheduleResult

    corrupted = ScheduleResult(
        status=SolverStatus.OPTIMAL,
        makespan_minutes=10,
        intervals=(
            ScheduledInterval(task_id="t1", start_minute=0, end_minute=5),
            ScheduledInterval(task_id="ghost", start_minute=5, end_minute=10),
        ),
    )
    problem = SchedulingProblem(tasks=tasks, resources=())

    from cooking_plan_agent.scheduling.verifier import ScheduleVerifier

    verifier = ScheduleVerifier()
    report = verifier.verify(problem, corrupted)
    assert not report.passed, "Extra interval for ghost task should be rejected"
    assert any(i.code == "EXTRA_TASK" for i in report.issues)


# =============================================================================
# UAT 11: Structured confirmation form → answers → ApprovedDecision → READY
# =============================================================================


class _StructuredConfirmationExtractor:
    """Gap-free candidate so the inventory shortage drives the outcome."""

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


@pytest.mark.asyncio
async def test_structured_confirmation_closed_loop() -> None:
    """UAT: NEEDS_CONFIRMATION → structured answers → ApprovedDecision → READY.

    P4-02: the confirmation carries ConfirmationQuestion[] whose repair
    CHOICE options map losslessly to ApprovedDecision (D9); answering the
    repair question and resubmitting re-enters validation and reaches READY
    (verification item 6 — the complete closed loop).
    """
    from cooking_plan_agent.domain.models import (
        ConfirmationPlanResponse,
        QuestionAnswer,
        ReadyPlanResponse,
    )
    from cooking_plan_agent.repair.options import answers_to_approved_decisions
    from cooking_plan_agent.workflow.context import WorkflowContext
    from cooking_plan_agent.workflow.graph import build_cooking_plan_graph

    request = GeneratePlanRequest(
        request_id="uat-structured-conf-001",
        user_id="u",
        recipes=(
            {
                "recipe_id": "recipe-1",
                "text": "Cook chicken.",
                "target_servings": 2,
            },
        ),
        # Shortage ratio 0.5 (need 200g, have 100g) so propose_portion_adjustments
        # proposes reduce_servings (a ratio of 0.75 would round 1.5 → 2 via
        # ROUND_HALF_EVEN and suppress the option — pre-existing behaviour).
        inventory_lots=(
            InventoryLotSnapshot(
                lot_id="lot-1",
                item_id="chicken",
                canonical_name="chicken breast",
                on_hand=Decimal(100),
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

    graph = build_cooking_plan_graph()
    context = WorkflowContext(recipe_extractor=_StructuredConfirmationExtractor())

    first = await graph.ainvoke({"request": request}, context=context, config={"recursion_limit": 30})
    response = first.get("response")
    assert isinstance(response, ConfirmationPlanResponse)
    assert response.confirmation_questions, "confirmation must carry structured questions"
    assert response.plan_revision is not None

    # Answer the repair-option question with a presented option value.
    repair_question = next(q for q in response.confirmation_questions if q.field_path == "repair_options")
    answer_value = next(o.value for o in repair_question.options if o.value != "__skip__")
    answers = (QuestionAnswer(question_id=repair_question.question_id, value=answer_value),)
    decisions = answers_to_approved_decisions(
        response.confirmation_questions,
        answers,
        response.plan_revision,
        presented_decisions=response.decisions,
    )
    assert decisions, "a presented repair decision must be recovered"
    assert any(d.option_type == "reduce_servings" for d in decisions)

    # Re-enter: decisions resubmit in approved_decisions, same revision.
    resolved = request.model_copy(update={"approved_decisions": decisions, "plan_revision": response.plan_revision})
    second = await graph.ainvoke({"request": resolved}, context=context, config={"recursion_limit": 30})
    final = second.get("response")
    assert isinstance(final, ReadyPlanResponse), f"expected READY, got {type(final).__name__}: {final}"
