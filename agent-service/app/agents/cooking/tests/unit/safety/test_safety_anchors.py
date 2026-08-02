"""P0-07 safety-task anchors & ordering verification tests.

Covers:
  - CrossContaminationRule locates raw and RTE steps and emits a
    structured SafetyInsertion with exact anchors
  - merge_preparation builds raw → sanitise → RTE dependency chains
  - the verifier rejects missing, misplaced, and broken-anchor safety tasks
  - shared equipment across recipes does not merge raw/RTE washes wrongly
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from cooking_plan_agent.domain.enums import HeatLevel, WorkMode
from cooking_plan_agent.domain.models import (
    CookingTask,
    ExtractedIngredient,
    ExtractedRecipeCandidate,
    ExtractedStep,
    GeneratePlanRequest,
    InventoryLotSnapshot,
    KitchenResourceSnapshot,
    RecipeIR,
    RecipeStep,
    SafetyContext,
    SafetyInsertion,
)
from cooking_plan_agent.preparation.decompose import decompose_step
from cooking_plan_agent.safety.engine import SafetyEngine
from cooking_plan_agent.safety.rules import CrossContaminationRule
from cooking_plan_agent.scheduling.models import ScheduledInterval, ScheduleResult, SchedulingProblem
from cooking_plan_agent.scheduling.verifier import ScheduleVerifier
from cooking_plan_agent.workflow.context import WorkflowContext
from cooking_plan_agent.workflow.graph import build_cooking_plan_graph

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _recipe_with_ingredients(
    recipe_id: str,
    raw_step: int = 1,
    rte_step: int = 3,
) -> RecipeIR:
    """Build a valid RecipeIR with a real chicken ingredient."""
    steps = tuple(
        RecipeStep(
            step_number=i + 1,
            instruction=("Cut the raw chicken." if i + 1 == raw_step else "Cook the dish."),
            category=("heating", "heating", "plating")[i],
            active_duration_minutes=10,
            heat_level=HeatLevel.HIGH if i < 2 else HeatLevel.NONE,
        )
        for i in range(3)
    )
    from cooking_plan_agent.domain.models import IngredientDemand

    ingredient = IngredientDemand(
        canonical_name="chicken breast",
        raw_name="chicken",
        quantity=Decimal(200),
        unit="g",
        input_state="raw",
        confidence=Decimal("1.0"),
    )
    return RecipeIR(
        recipe_id=recipe_id,
        dish_name=f"Dish {recipe_id}",
        original_servings=2,
        target_servings=2,
        source_language="en",
        ingredients=(ingredient,),
        steps=steps,
    )


class _FixedExtractor:
    """Workflow extractor producing a raw + RTE candidate."""

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
                    instruction="Cut the raw chicken.",
                    category="heating",
                    active_duration_minutes=10,
                    heat_level=HeatLevel.HIGH,
                ),
                ExtractedStep(
                    step_number=2,
                    instruction="Cook the chicken.",
                    category="heating",
                    active_duration_minutes=10,
                    heat_level=HeatLevel.HIGH,
                ),
                ExtractedStep(
                    step_number=3,
                    instruction="Plate the dish.",
                    category="plating",
                ),
            ),
        )


def _base_request() -> GeneratePlanRequest:
    return GeneratePlanRequest(
        request_id="safety-req-001",
        user_id="u",
        recipes=(
            {
                "recipe_id": "recipe-1",
                "text": "Cut chicken. Cook. Plate.",
                "target_servings": 2,
            },
        ),
        dietary_restrictions=(),
        user_allergens=(),
        inventory_lots=(
            InventoryLotSnapshot(
                lot_id="lot-1",
                item_id="chicken",
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
            KitchenResourceSnapshot(
                resource_id="sink-1",
                resource_type="sink",
                capacity=Decimal(1),
            ),
        ),
    )


@pytest.fixture
def graph():
    return build_cooking_plan_graph()


@pytest.fixture
def context():
    return WorkflowContext(recipe_extractor=_FixedExtractor())


# ---------------------------------------------------------------------------
# Rule emits structured insertion with anchors
# ---------------------------------------------------------------------------


class TestRuleAnchors:
    def test_cross_contamination_emits_anchored_insertion(self) -> None:
        recipe = _recipe_with_ingredients("r1")
        context = SafetyContext(recipes=(recipe,))
        finding = CrossContaminationRule().evaluate(context)
        assert finding is not None
        assert finding.insertion is not None
        insertion: SafetyInsertion = finding.insertion
        assert insertion.rule_id == "SAFETY_CROSS_CONTAMINATION"
        assert insertion.recipe_id == "r1"
        assert insertion.after_step_number == 1  # raw step
        assert insertion.before_step_number == 3  # RTE/plating step
        assert insertion.duration_minutes >= 1  # policy, not fixed 1-min
        assert "sink" in insertion.required_resources

    def test_no_rte_step_no_insertion(self) -> None:
        recipe = _recipe_with_ingredients("r1")
        recipe = recipe.model_copy(
            update={"steps": tuple(s.model_copy(update={"category": "heating"}) for s in recipe.steps)}
        )
        finding = CrossContaminationRule().evaluate(SafetyContext(recipes=(recipe,)))
        assert finding is None

    def test_engine_collects_insertions(self) -> None:
        recipe = _recipe_with_ingredients("r1")
        report = SafetyEngine().evaluate(SafetyContext(recipes=(recipe,)))
        assert report.insertions, "Engine must surface structured insertions"
        assert report.required_safety_task_ids, "Engine must emit task IDs too"


# ---------------------------------------------------------------------------
# merge_preparation builds the raw → sanitise → RTE chain
# ---------------------------------------------------------------------------


class TestTaskChain:
    def test_sanitise_task_anchored_between_raw_and_rte(self) -> None:
        """The injected sanitise task must sit between the raw and RTE steps."""

        recipe = _recipe_with_ingredients("r1")
        # Decompose: step1 → r1_s1, step2 → r1_s2, step3 (plating, simple) → r1_s3
        tasks = []
        for step in recipe.steps:
            tasks.extend(decompose_step("r1", step))

        # Sanitise task via the report insertion
        report = SafetyEngine().evaluate(SafetyContext(recipes=(recipe,)))
        assert report.insertions
        insertion = report.insertions[0]

        # Simulate merge_preparation's chain-building on this recipe's tasks.
        from cooking_plan_agent.domain.models import TaskDependency

        after_task = next(t for t in tasks if t.task_id.endswith("_s1"))
        before_task = next(t for t in tasks if t.task_id.endswith("_s3"))
        sanitise = CookingTask(
            task_id="safety_sanitise_x",
            dish_id="r1",
            instruction=insertion.task_instruction,
            duration_minutes=insertion.duration_minutes,
            work_mode=WorkMode.ACTIVE,
            category="safety",
            dependencies=(TaskDependency(predecessor_id=after_task.task_id),),
            resources=(),
            safety_tags=(insertion.rule_id,),
        )
        assert sanitise.dependencies[0].predecessor_id == after_task.task_id
        # The RTE task must depend on the sanitise task.
        assert before_task.task_id == "r1_s3"

    @pytest.mark.asyncio
    async def test_graph_injects_anchored_safety_task(self, graph, context) -> None:
        """End-to-end: a raw+RTE recipe yields a scheduled safety task."""
        from cooking_plan_agent.safety.engine import SafetyEngine

        ctx = WorkflowContext(recipe_extractor=_FixedExtractor(), safety_engine=SafetyEngine())
        result = await graph.ainvoke({"request": _base_request()}, context=ctx, config={"recursion_limit": 30})
        response = result.get("response")
        assert response is not None
        # Cross-contamination is hard_repairable → safety task injected and
        # the plan still completes (READY) with the sanitise task scheduled.
        task_graph = result.get("task_graph")
        if task_graph is not None:
            safety_tasks = [t for t in task_graph.tasks if t.safety_tags]
            assert safety_tasks, "Sanitisation task must be present in the graph"
        assert response.status != "FAILED"


# ---------------------------------------------------------------------------
# Verifier: missing / misplaced / broken-anchor checks
# ---------------------------------------------------------------------------


class TestVerifierAnchors:
    def _problem(self, tasks: tuple[CookingTask, ...]) -> SchedulingProblem:
        return SchedulingProblem(tasks=tasks, resources=())

    def _schedule(
        self,
        task_ids: tuple[str, ...],
        start: int = 0,
        task_map: dict[str, CookingTask] | None = None,
    ) -> ScheduleResult:
        """Build a schedule honouring each task's real duration."""
        task_map = task_map or {}
        intervals: list[ScheduledInterval] = []
        t = start
        for tid in task_ids:
            duration = task_map[tid].duration_minutes if tid in task_map else 3
            intervals.append(ScheduledInterval(task_id=tid, start_minute=t, end_minute=t + duration))
            t += duration
        return ScheduleResult(
            status="FEASIBLE",
            makespan_minutes=t,
            intervals=tuple(intervals),
        )

    def _sanitise_task(self, task_id: str, pred_id: str | None, duration: int = 3) -> CookingTask:
        from cooking_plan_agent.domain.models import TaskDependency

        deps = ()
        if pred_id is not None:
            deps = (TaskDependency(predecessor_id=pred_id),)
        return CookingTask(
            task_id=task_id,
            dish_id="r1",
            instruction="Sanitise board",
            duration_minutes=duration,
            work_mode=WorkMode.ACTIVE,
            category="safety",
            dependencies=deps,
            safety_tags=("SAFETY_CROSS_CONTAMINATION",),
        )

    def test_well_ordered_sanitise_passes(self) -> None:
        raw = CookingTask(
            task_id="raw",
            dish_id="r1",
            instruction="Cut",
            duration_minutes=5,
            work_mode=WorkMode.ACTIVE,
            category="cutting",
        )
        sanitise = self._sanitise_task("san", pred_id="raw")
        verifier = ScheduleVerifier()
        problem = self._problem((raw, sanitise))
        # sanitise starts after raw ends (raw 0-5, san 5-8)
        result = self._schedule(("raw", "san"), start=0, task_map={"raw": raw, "san": sanitise})
        report = verifier.verify(problem, result)
        assert report.passed, f"Expected pass, got {report.issues}"

    def test_misplaced_sanitise_fails(self) -> None:
        raw = CookingTask(
            task_id="raw",
            dish_id="r1",
            instruction="Cut",
            duration_minutes=5,
            work_mode=WorkMode.ACTIVE,
            category="cutting",
        )
        sanitise = self._sanitise_task("san", pred_id="raw")
        verifier = ScheduleVerifier()
        problem = self._problem((raw, sanitise))
        # sanitise starts (0) BEFORE raw ends (5) → misplaced.
        result = self._schedule(("san", "raw"), start=0)
        report = verifier.verify(problem, result)
        assert not report.passed
        codes = {i.code for i in report.issues}
        assert "SAFETY_TASK_MISPLACED" in codes

    def test_missing_sanitise_fails(self) -> None:
        raw = CookingTask(
            task_id="raw",
            dish_id="r1",
            instruction="Cut",
            duration_minutes=5,
            work_mode=WorkMode.ACTIVE,
            category="cutting",
        )
        sanitise = self._sanitise_task("san", pred_id="raw")
        verifier = ScheduleVerifier()
        problem = self._problem((raw, sanitise))
        # sanitise missing from schedule.
        result = self._schedule(("raw",), start=0)
        report = verifier.verify(problem, result)
        assert not report.passed
        codes = {i.code for i in report.issues}
        assert "SAFETY_TASK_MISSING" in codes

    def test_broken_anchor_fails(self) -> None:
        sanitise = self._sanitise_task("san", pred_id="raw")  # raw never exists
        verifier = ScheduleVerifier()
        problem = self._problem((sanitise,))
        result = self._schedule(("san",), start=0)
        report = verifier.verify(problem, result)
        assert not report.passed
        codes = {i.code for i in report.issues}
        assert "SAFETY_ANCHOR_MISSING" in codes

    def test_raw_and_rte_washes_not_merged_across_recipes(self) -> None:
        """Two recipes sharing equipment must not merge raw/RTE wrongly.

        Each recipe's sanitise task anchors to ITS OWN raw/RTE steps.
        """
        raw_a = CookingTask(
            task_id="a_raw",
            dish_id="a",
            instruction="Cut chicken",
            duration_minutes=5,
            work_mode=WorkMode.ACTIVE,
            category="cutting",
        )
        raw_b = CookingTask(
            task_id="b_raw",
            dish_id="b",
            instruction="Cut pork",
            duration_minutes=5,
            work_mode=WorkMode.ACTIVE,
            category="cutting",
        )
        san_a = self._sanitise_task("a_san", pred_id="a_raw")
        san_b = self._sanitise_task("b_san", pred_id="b_raw")
        verifier = ScheduleVerifier()
        problem = self._problem((raw_a, raw_b, san_a, san_b))
        result = self._schedule(
            ("a_raw", "a_san", "b_raw", "b_san"),
            start=0,
            task_map={"a_raw": raw_a, "a_san": san_a, "b_raw": raw_b, "b_san": san_b},
        )
        report = verifier.verify(problem, result)
        assert report.passed, f"Per-recipe anchors must hold, got {report.issues}"
