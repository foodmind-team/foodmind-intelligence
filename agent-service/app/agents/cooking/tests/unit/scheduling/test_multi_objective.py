"""P3-03 multi-objective scheduling tests.

Verifies the lexicographic ordering guarantee (D5): later phases (holding,
context switching) never increase the Phase 1 makespan; improvement fixtures
strictly improve the secondary objective; un-improvable fixtures preserve
the result; optimization_level selects the phase depth; the verifier rejects
inconsistent phase metadata.
"""

from cooking_plan_agent.domain.enums import SolverStatus, WorkMode
from cooking_plan_agent.domain.models import (
    CookingTask,
    ResourceNeed,
    TaskDependency,
)
from cooking_plan_agent.scheduling.models import (
    ScheduleResult,
    SchedulingProblem,
)
from cooking_plan_agent.scheduling.orchestrator import ScheduleOrchestrator
from cooking_plan_agent.scheduling.verifier import ScheduleVerifier


def _task(
    task_id: str,
    dish_id: str = "d1",
    duration: int = 5,
    category: str = "general",
    deps: tuple[TaskDependency, ...] = (),
    work_mode: WorkMode = WorkMode.ACTIVE,
    resources: tuple[ResourceNeed, ...] = (),
) -> CookingTask:
    return CookingTask(
        task_id=task_id,
        dish_id=dish_id,
        instruction=f"Task {task_id}",
        duration_minutes=duration,
        work_mode=work_mode,
        category=category,
        dependencies=deps,
        resources=resources,
    )


def _problem(tasks: tuple[CookingTask, ...], timeout: float = 5.0) -> SchedulingProblem:
    return SchedulingProblem(
        tasks=tasks,
        resources=(),
        solver_timeout_seconds=timeout,
    )


class TestMakespanNeverIncreases:
    """P3-03 acceptance: phases 2–4 must never increase Phase 1 makespan."""

    def test_holding_and_context_switch_preserve_makespan(self) -> None:
        # Two dishes: A(10) and B(2). Single cook → makespan 12.
        tasks = (
            _task("A", dish_id="dish1", duration=10, category="heating"),
            _task("B", dish_id="dish2", duration=2, category="cutting"),
        )
        problem = _problem(tasks)

        makespan_only = ScheduleOrchestrator().solve(problem, optimization_level="makespan")[0]
        full = ScheduleOrchestrator().solve(problem, optimization_level="full")[0]

        assert makespan_only.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
        assert full.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
        assert full.makespan_minutes == makespan_only.makespan_minutes == 12

    def test_context_switch_tie_break_never_increases_makespan(self) -> None:
        # Three tasks in two categories; context-switch phase must not push
        # the makespan beyond the Phase-1 optimum.
        tasks = (
            _task("A", dish_id="d1", duration=3, category="cutting"),
            _task("B", dish_id="d2", duration=4, category="heating"),
            _task("C", dish_id="d3", duration=2, category="cutting"),
        )
        problem = _problem(tasks)

        full = ScheduleOrchestrator().solve(problem, optimization_level="full")[0]
        assert full.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
        # All active sequential → makespan 9 regardless of phase.
        assert full.makespan_minutes == 9
        assert "context_switch" in full.optimization_phases


class TestSecondaryObjectiveImprovement:
    """New objectives strictly improve improvable fixtures."""

    def test_holding_improves_improvable_fixture(self) -> None:
        """Holding objective pushes short dish later (test from P2 baseline)."""
        tasks = (
            _task("A", dish_id="dish1", duration=10),
            _task("B", dish_id="dish2", duration=2),
        )
        problem = _problem(tasks)

        orchestrator = ScheduleOrchestrator()
        result, report = orchestrator.solve(problem)
        assert result.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
        assert result.makespan_minutes == 12
        assert report.passed

        # Dish 2 (task B) finishes at makespan → holding objective is 0
        # (no waiting between last task end and serving).
        b = next(i for i in result.intervals if i.task_id == "B")
        assert b.end_minute == 12

    def test_context_switch_groups_same_category(self) -> None:
        """Context-switch phase groups same-category tasks (span proxy).

        Two cutting tasks (A, B) and one heating task (C) on a single cook.
        Grouping the cutting tasks together minimises the cutting-category
        time span. Verify the phase records its objective and the verifier
        stays green.
        """
        tasks = (
            _task("A", dish_id="d1", duration=2, category="cutting"),
            _task("B", dish_id="d2", duration=2, category="cutting"),
            _task("C", dish_id="d3", duration=2, category="heating"),
        )
        problem = _problem(tasks)

        orchestrator = ScheduleOrchestrator()
        result, report = orchestrator.solve(problem, optimization_level="full")

        assert result.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
        assert report.passed
        assert "context_switch" in result.optimization_phases
        assert result.context_switch_objective is not None
        # Cutting category span = latest cutting end - earliest cutting start.
        cutting = sorted((i.start_minute, i.end_minute) for i in result.intervals if i.task_id in ("A", "B"))
        span = cutting[-1][1] - cutting[0][0]
        assert span == 4  # two consecutive 2-min cutting tasks


class TestUnimprovableFixturePreserved:
    def test_unimprovable_fixture_keeps_result(self) -> None:
        """A single-task problem cannot improve holding/context switch.

        Phase 2/3 run but must leave the schedule unchanged (no regression).
        """
        tasks = (_task("t1", dish_id="d1", duration=5, category="heating"),)
        problem = _problem(tasks)

        result, report = ScheduleOrchestrator().solve(problem)
        assert result.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
        assert report.passed
        assert result.makespan_minutes == 5
        assert result.holding_objective is not None
        assert result.context_switch_objective is not None


class TestOptimizationLevelRollback:
    """P3-03 completion definition: configurable rollback to fewer phases."""

    def test_makespan_only_level(self) -> None:
        tasks = (
            _task("A", dish_id="d1", duration=10),
            _task("B", dish_id="d2", duration=2),
        )
        problem = _problem(tasks)
        result, report = ScheduleOrchestrator().solve(problem, optimization_level="makespan")
        assert result.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
        assert result.optimization_phases == ("makespan",)
        assert result.holding_objective is None
        assert report.passed

    def test_phase12_level(self) -> None:
        tasks = (
            _task("A", dish_id="d1", duration=10),
            _task("B", dish_id="d2", duration=2),
        )
        problem = _problem(tasks)
        result, report = ScheduleOrchestrator().solve(problem, optimization_level="phase12")
        assert result.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
        assert result.optimization_phases == ("makespan", "holding")
        assert result.holding_objective is not None
        assert result.context_switch_objective is None
        assert report.passed


class TestVerifierRejectsInconsistentMetadata:
    """The independent verifier rejects broken phase metadata (P3-03)."""

    def test_holding_phase_without_objective_is_rejected(self) -> None:
        result = ScheduleResult(
            status=SolverStatus.OPTIMAL,
            makespan_minutes=10,
            intervals=(),
            optimization_phases=("makespan", "holding"),
            holding_objective=None,  # inconsistent: phase claimed, value lost
        )
        report = ScheduleVerifier().verify(_problem((_task("t1", duration=10),)), result)
        assert not report.passed
        codes = {i.code for i in report.issues}
        assert "HOLDING_OBJECTIVE_MISSING" in codes

    def test_context_switch_phase_without_objective_is_rejected(self) -> None:
        result = ScheduleResult(
            status=SolverStatus.FEASIBLE,
            makespan_minutes=10,
            intervals=(),
            optimization_phases=("makespan", "holding", "context_switch"),
            holding_objective=0,
            context_switch_objective=None,  # inconsistent
        )
        report = ScheduleVerifier().verify(_problem((_task("t1", duration=10),)), result)
        assert not report.passed
        codes = {i.code for i in report.issues}
        assert "CONTEXT_SWITCH_OBJECTIVE_MISSING" in codes

    def test_phases_without_makespan_is_rejected(self) -> None:
        result = ScheduleResult(
            status=SolverStatus.OPTIMAL,
            makespan_minutes=None,  # inconsistent with claimed phases
            intervals=(),
            optimization_phases=("makespan",),
        )
        report = ScheduleVerifier().verify(_problem((_task("t1", duration=10),)), result)
        assert not report.passed
        codes = {i.code for i in report.issues}
        assert "MAKESPAN_MISSING_WITH_PHASES" in codes


class TestPhaseFourGated:
    """Phase 4 (active labour) must never run without equivalent modes."""

    def test_no_equivalent_modes_skips_phase_four(self) -> None:
        # A single resource type per task is NOT an equivalent-mode scenario,
        # so Phase 4 must stay gated.
        problem = _problem(
            (
                _task("A", dish_id="d1", duration=5),
                _task("B", dish_id="d2", duration=3),
            )
        )
        result, report = ScheduleOrchestrator().solve(problem, optimization_level="full")
        assert result.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
        assert "active_labour" not in result.optimization_phases
        assert report.passed
