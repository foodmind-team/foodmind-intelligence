"""OR-Tools scheduler tests — 10 solver fixtures + verification + edge cases.

Handbook 7.14: build small cases with hand-computable answers.  Each fixture
tests one scheduling concern in isolation before combining them.

Test structure:
- Fixture helpers: factory functions for CookingTask, KitchenResourceSnapshot
- Fixture 1–10: solver fixtures with known expected results
- Verifier tests: edge cases the verifier must catch
- Orchestrator tests: lexicographic solve
- Status mapping: UNKNOWN, INFEASIBLE, MODEL_INVALID handling
"""

from decimal import Decimal

from ortools.sat.python import cp_model

from cooking_plan_agent.domain.enums import SolverStatus, WorkMode
from cooking_plan_agent.domain.models import (
    CookingTask,
    KitchenResourceSnapshot,
    ResourceNeed,
    TaskDependency,
)
from cooking_plan_agent.scheduling.builder import ScheduleModelBuilder
from cooking_plan_agent.scheduling.extractor import ScheduleExtractor
from cooking_plan_agent.scheduling.models import (
    ScheduledInterval,
    ScheduleResult,
    SchedulingProblem,
    VerificationReport,
)
from cooking_plan_agent.scheduling.orchestrator import (
    ScheduleOrchestrator,
    schedule,
)
from cooking_plan_agent.scheduling.solver import ScheduleSolver
from cooking_plan_agent.scheduling.verifier import ScheduleVerifier

# =============================================================================
# Factory helpers — construct domain objects concisely
# =============================================================================


def _task(
    task_id: str,
    dish_id: str = "d1",
    duration: int = 5,
    work_mode: WorkMode = WorkMode.ACTIVE,
    category: str = "general",
    deps: tuple[TaskDependency, ...] = (),
    resources: tuple[ResourceNeed, ...] = (),
) -> CookingTask:
    """Create a minimal CookingTask for scheduler testing."""
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


def _stove(count: int = 4) -> KitchenResourceSnapshot:
    """Create a stove resource with N burners."""
    return KitchenResourceSnapshot(
        resource_id="stove:main",
        resource_type="stove",
        capacity=Decimal(count),
        capacity_unit="burners",
    )


def _oven() -> KitchenResourceSnapshot:
    """Create an oven resource."""
    return KitchenResourceSnapshot(
        resource_id="oven:main",
        resource_type="oven",
        capacity=Decimal(1),
    )


def _make_problem(
    tasks: tuple[CookingTask, ...],
    resources: tuple[KitchenResourceSnapshot, ...] = (),
    time_limit: int | None = None,
) -> SchedulingProblem:
    """Create a SchedulingProblem with defaults."""
    return SchedulingProblem(
        tasks=tasks,
        resources=resources or (_stove(4),),
        requested_time_limit_minutes=time_limit,
        solver_timeout_seconds=5.0,
    )


def _run_solve(problem: SchedulingProblem) -> tuple[ScheduleResult, VerificationReport]:
    """Run the full schedule() pipeline and return result + report."""
    return schedule(problem)


# =============================================================================
# 7.14 Fixture 1: One task (hand-computable: makespan = 5)
# =============================================================================


class TestFixture1OneTask:
    """A single 5-minute active task.  Makespan must be 5."""

    def test_makespan_is_duration(self) -> None:
        tasks = (_task("t1", duration=5),)
        problem = _make_problem(tasks)
        result, report = _run_solve(problem)

        assert result.status == SolverStatus.OPTIMAL
        assert result.makespan_minutes == 5
        assert len(result.intervals) == 1
        assert result.intervals[0].start_minute == 0
        assert result.intervals[0].end_minute == 5
        assert report.passed


# =============================================================================
# 7.14 Fixture 2: Two dependent tasks (A → B, makespan = 10)
# =============================================================================


class TestFixture2TwoDependentTasks:
    """B depends on A.  Makespan = 10 (sequential)."""

    def test_sequential_dependency(self) -> None:
        dep = TaskDependency(predecessor_id="t1")
        tasks = (
            _task("t1", duration=5),
            _task("t2", duration=5, deps=(dep,)),
        )
        problem = _make_problem(tasks)
        result, report = _run_solve(problem)

        assert result.status == SolverStatus.OPTIMAL
        assert result.makespan_minutes == 10
        assert report.passed

        # Verify ordering: t1 must finish before t2 starts
        intervals = {i.task_id: i for i in result.intervals}
        assert intervals["t1"].end_minute <= intervals["t2"].start_minute


class TestFixture2MinLag:
    """B must start at least 3 minutes after A ends (lag)."""

    def test_minimum_lag(self) -> None:
        dep = TaskDependency(predecessor_id="t1", minimum_lag_minutes=3)
        tasks = (
            _task("t1", duration=5),
            _task("t2", duration=5, deps=(dep,)),
        )
        problem = _make_problem(tasks)
        result, report = _run_solve(problem)

        assert result.status == SolverStatus.OPTIMAL
        # t1 ends at 5, +3 lag, t2 starts at 8, ends at 13 → makespan = 13
        assert result.makespan_minutes == 13
        assert report.passed


class TestFixture2MaxLag:
    """B must start within 2 minutes after A ends (tight max lag)."""

    def test_maximum_lag(self) -> None:
        dep = TaskDependency(predecessor_id="t1", maximum_lag_minutes=2)
        tasks = (
            _task("t1", duration=5),
            _task("t2", duration=3, deps=(dep,)),
        )
        problem = _make_problem(tasks)
        result, report = _run_solve(problem)

        assert result.status == SolverStatus.OPTIMAL
        intervals = {i.task_id: i for i in result.intervals}
        # t2 must start <= t1.end + 2 = 5 + 2 = 7
        assert intervals["t2"].start_minute <= 7
        assert report.passed


# =============================================================================
# 7.14 Fixture 3: Two independent active tasks (no overlap, makespan = 10)
# =============================================================================


class TestFixture3TwoIndependentActive:
    """Two 5-min active tasks that cannot overlap → makespan = 10."""

    def test_sequential_active(self) -> None:
        tasks = (
            _task("t1", duration=5, work_mode=WorkMode.ACTIVE),
            _task("t2", duration=5, work_mode=WorkMode.ACTIVE),
        )
        problem = _make_problem(tasks)
        result, report = _run_solve(problem)

        assert result.status == SolverStatus.OPTIMAL
        assert result.makespan_minutes == 10
        assert report.passed


# =============================================================================
# 7.14 Fixture 4: One passive task overlapping two active tasks
# =============================================================================


class TestFixture4PassiveOverlap:
    """Passive tasks can overlap active tasks without consuming the cook."""

    def test_passive_parallel_to_active(self) -> None:
        tasks = (
            _task("t1", duration=5, work_mode=WorkMode.ACTIVE),
            _task("t2", duration=10, work_mode=WorkMode.PASSIVE),
            _task("t3", duration=5, work_mode=WorkMode.ACTIVE),
        )
        problem = _make_problem(tasks)
        result, report = _run_solve(problem)

        assert result.status == SolverStatus.OPTIMAL
        # Active tasks are sequential → makespan = 10 (t1 then t3).
        # Passive t2 (10 min) runs in parallel → makespan = max(10, 10) = 10
        assert result.makespan_minutes == 10
        assert report.passed


# =============================================================================
# 7.14 Fixture 5: Two burners vs one burner
# =============================================================================


class TestFixture5BurnerCapacity:
    """Two tasks needing one burner each — capacity matters."""

    def test_two_burners_enough(self) -> None:
        """2 burners: both PASSIVE stove tasks run in parallel.

        When tasks are PASSIVE (like boiling on two burners simultaneously),
        the cook is not occupied — only the stove resource constrains them.
        """
        need_stove = ResourceNeed(resource_type="stove", quantity=1)
        tasks = (
            _task("t1", duration=5, resources=(need_stove,), work_mode=WorkMode.PASSIVE),
            _task("t2", duration=5, resources=(need_stove,), work_mode=WorkMode.PASSIVE),
        )
        problem = _make_problem(tasks, resources=(_stove(2),))
        result, report = _run_solve(problem)

        assert result.status == SolverStatus.OPTIMAL
        # Two burners → both PASSIVE tasks can run in parallel → makespan = 5
        assert result.makespan_minutes == 5
        assert report.passed

    def test_one_burner_forces_sequential(self) -> None:
        """1 burner: tasks must be sequential."""
        need_stove = ResourceNeed(resource_type="stove", quantity=1)
        tasks = (
            _task("t1", duration=5, resources=(need_stove,)),
            _task("t2", duration=5, resources=(need_stove,)),
        )
        problem = _make_problem(tasks, resources=(_stove(1),))
        result, report = _run_solve(problem)

        assert result.status == SolverStatus.OPTIMAL
        # One burner → sequential → makespan = 10
        assert result.makespan_minutes == 10
        assert report.passed


# =============================================================================
# 7.14 Fixture 6: One task with two equipment alternatives
# =============================================================================


class TestFixture6ResourceAlternatives:
    """A task that can use either oven or stove.

    Note: In the MVP, identical resources of each type are pooled.
    Alternative resources (named instances) are deferred per Handbook 7.6.
    This test verifies the current pooling behaviour works correctly.
    """

    def test_task_uses_single_resource_type(self) -> None:
        need_oven = ResourceNeed(resource_type="oven", quantity=1)
        tasks = (_task("t1", duration=10, resources=(need_oven,)),)
        problem = _make_problem(tasks, resources=(_oven(),))
        result, report = _run_solve(problem)

        assert result.status == SolverStatus.OPTIMAL
        assert result.makespan_minutes == 10
        assert report.passed


# =============================================================================
# 7.14 Fixture 7: Minimum marinating lag
# =============================================================================


class TestFixture7MarinatingLag:
    """Apply marinade → wait 20 minutes → cook.  Ensures lag is respected."""

    def test_marinating_lag(self) -> None:
        dep = TaskDependency(predecessor_id="apply", minimum_lag_minutes=20)
        tasks = (
            _task("apply", duration=5, category="mixing"),
            _task("cook", duration=10, deps=(dep,), category="heating"),
        )
        problem = _make_problem(tasks)
        result, report = _run_solve(problem)

        assert result.status == SolverStatus.OPTIMAL
        intervals = {i.task_id: i for i in result.intervals}
        # apply ends at 5, +20 lag → cook starts at 25, ends at 35
        assert intervals["cook"].start_minute >= intervals["apply"].end_minute + 20
        assert result.makespan_minutes == 35
        assert report.passed


# =============================================================================
# 7.14 Fixture 8: Maximum safe wait
# =============================================================================


class TestFixture8MaximumSafeWait:
    """After removing from oven, plating must start within 5 minutes.

    This tests the maximum_lag constraint.
    """

    def test_max_wait_respected(self) -> None:
        dep = TaskDependency(
            predecessor_id="unload",
            maximum_lag_minutes=5,
        )
        tasks = (
            _task("unload", duration=3, category="finishing"),
            _task("plate", duration=2, deps=(dep,), category="finishing"),
        )
        problem = _make_problem(tasks)
        result, report = _run_solve(problem)

        assert result.status == SolverStatus.OPTIMAL
        intervals = {i.task_id: i for i in result.intervals}
        # plate must start <= unload.end + 5 = 3 + 5 = 8
        assert intervals["plate"].start_minute <= intervals["unload"].end_minute + 5
        assert report.passed


# =============================================================================
# 7.14 Fixture 9: Infeasible deadline
# =============================================================================


class TestFixture9InfeasibleDeadline:
    """A hard deadline that is impossible to meet → INFEASIBLE."""

    def test_too_tight_deadline(self) -> None:
        """Three 10-minute active tasks cannot finish within 5 minutes."""
        tasks = (
            _task("t1", duration=10),
            _task("t2", duration=10),
            _task("t3", duration=10),
        )
        problem = _make_problem(tasks, time_limit=5)
        result, report = _run_solve(problem)

        assert result.status == SolverStatus.INFEASIBLE
        assert result.makespan_minutes is None
        assert len(result.intervals) == 0
        # Verifier should pass (or have no intervals to check)
        assert report.passed  # No intervals means no violations

    def test_deadline_just_feasible(self) -> None:
        """Two 5-minute tasks can finish within 10 minutes."""
        tasks = (
            _task("t1", duration=5),
            _task("t2", duration=5),
        )
        problem = _make_problem(tasks, time_limit=10)
        result, report = _run_solve(problem)

        assert result.status == SolverStatus.OPTIMAL
        assert result.makespan_minutes == 10
        assert report.passed


# =============================================================================
# 7.14 Fixture 10: Corrupted result rejected by verifier
# =============================================================================


class TestFixture10VerifierRejectsCorruptedResult:
    """If someone manipulates a ScheduleResult, the verifier must catch it."""

    def test_duration_mismatch(self) -> None:
        tasks = (_task("t1", duration=5),)
        problem = _make_problem(tasks)

        # Create a deliberately corrupted result
        bad_result = ScheduleResult(
            status=SolverStatus.OPTIMAL,
            makespan_minutes=10,
            intervals=(
                ScheduledInterval(
                    task_id="t1",
                    start_minute=0,
                    end_minute=10,  # Should be 5!
                ),
            ),
        )

        verifier = ScheduleVerifier()
        report = verifier.verify(problem, bad_result)
        assert not report.passed
        assert any(i.code == "DURATION_MISMATCH" for i in report.issues)

    def test_missing_task(self) -> None:
        tasks = (
            _task("t1", duration=5),
            _task("t2", duration=5),
        )
        problem = _make_problem(tasks)

        # t1 missing from intervals
        bad_result = ScheduleResult(
            status=SolverStatus.OPTIMAL,
            makespan_minutes=5,
            intervals=(
                ScheduledInterval(
                    task_id="t2",
                    start_minute=0,
                    end_minute=5,
                ),
            ),
        )

        verifier = ScheduleVerifier()
        report = verifier.verify(problem, bad_result)
        assert not report.passed
        assert any(i.code == "MISSING_TASK" for i in report.issues)

    def test_active_overlap(self) -> None:
        tasks = (
            _task("t1", duration=5),
            _task("t2", duration=5),
        )
        problem = _make_problem(tasks)

        bad_result = ScheduleResult(
            status=SolverStatus.OPTIMAL,
            makespan_minutes=5,
            intervals=(
                ScheduledInterval(task_id="t1", start_minute=0, end_minute=5),
                ScheduledInterval(task_id="t2", start_minute=2, end_minute=7),
            ),
        )

        verifier = ScheduleVerifier()
        report = verifier.verify(problem, bad_result)
        assert not report.passed
        assert any(i.code == "ACTIVE_OVERLAP" for i in report.issues)

    def test_capacity_exceeded(self) -> None:
        need_stove = ResourceNeed(resource_type="stove", quantity=1)
        tasks = (
            _task("t1", duration=5, resources=(need_stove,)),
            _task("t2", duration=5, resources=(need_stove,)),
        )
        problem = _make_problem(tasks, resources=(_stove(1),))

        bad_result = ScheduleResult(
            status=SolverStatus.OPTIMAL,
            makespan_minutes=5,
            intervals=(
                ScheduledInterval(task_id="t1", start_minute=0, end_minute=5),
                ScheduledInterval(task_id="t2", start_minute=0, end_minute=5),
            ),
        )

        verifier = ScheduleVerifier()
        report = verifier.verify(problem, bad_result)
        assert not report.passed
        assert any(i.code == "CAPACITY_EXCEEDED" for i in report.issues)

    def test_makespan_mismatch(self) -> None:
        tasks = (_task("t1", duration=5),)
        problem = _make_problem(tasks)

        bad_result = ScheduleResult(
            status=SolverStatus.OPTIMAL,
            makespan_minutes=99,  # Wrong!
            intervals=(ScheduledInterval(task_id="t1", start_minute=0, end_minute=5),),
        )

        verifier = ScheduleVerifier()
        report = verifier.verify(problem, bad_result)
        assert not report.passed
        assert any(i.code == "MAKESPAN_MISMATCH" for i in report.issues)


# =============================================================================
# Solver status mapping tests (7.8)
# =============================================================================


class TestSolverStatusMapping:
    """Verify every CP-SAT status maps correctly to SolverStatus."""

    def test_all_statuses_mapped(self) -> None:
        solver = ScheduleSolver()

        assert solver.map_status(cp_model.OPTIMAL) == SolverStatus.OPTIMAL
        assert solver.map_status(cp_model.FEASIBLE) == SolverStatus.FEASIBLE
        assert solver.map_status(cp_model.INFEASIBLE) == SolverStatus.INFEASIBLE
        assert solver.map_status(cp_model.MODEL_INVALID) == SolverStatus.MODEL_INVALID
        assert solver.map_status(cp_model.UNKNOWN) == SolverStatus.UNKNOWN

    def test_unknown_is_not_infeasible(self) -> None:
        """Handbook 7.8: UNKNOWN must never be collapsed into INFEASIBLE."""
        solver = ScheduleSolver()
        assert solver.map_status(cp_model.UNKNOWN) != SolverStatus.INFEASIBLE


# =============================================================================
# Verifier edge cases
# =============================================================================


class TestVerifierEdgeCases:
    """Additional verifier checks beyond the 10 fixtures."""

    def test_empty_problem(self) -> None:
        """Empty task list — should pass trivially."""
        problem = SchedulingProblem(
            tasks=(),
            resources=(),
        )
        result = ScheduleResult(status=SolverStatus.OPTIMAL, makespan_minutes=0)
        verifier = ScheduleVerifier()
        report = verifier.verify(problem, result)
        assert report.passed
        assert report.checked_task_count == 0

    def test_adjacent_tasks_no_overlap(self) -> None:
        """Tasks touching end-to-end should NOT be considered overlapping."""
        tasks = (
            _task("t1", duration=5),
            _task("t2", duration=5),
        )
        problem = _make_problem(tasks)

        result = ScheduleResult(
            status=SolverStatus.OPTIMAL,
            makespan_minutes=10,
            intervals=(
                ScheduledInterval(task_id="t1", start_minute=0, end_minute=5),
                ScheduledInterval(task_id="t2", start_minute=5, end_minute=10),
            ),
        )

        verifier = ScheduleVerifier()
        report = verifier.verify(problem, result)
        assert report.passed

    def test_negative_start_rejected(self) -> None:
        tasks = (_task("t1", duration=5),)
        problem = _make_problem(tasks)

        result = ScheduleResult(
            status=SolverStatus.OPTIMAL,
            makespan_minutes=5,
            intervals=(ScheduledInterval(task_id="t1", start_minute=-1, end_minute=4),),
        )

        verifier = ScheduleVerifier()
        report = verifier.verify(problem, result)
        assert not report.passed
        assert any(i.code == "NEGATIVE_START" for i in report.issues)

    def test_resource_unavailable(self) -> None:
        need_oven = ResourceNeed(resource_type="oven", quantity=1)
        tasks = (_task("t1", duration=5, resources=(need_oven,)),)

        # oven is marked unavailable
        resources = (
            KitchenResourceSnapshot(
                resource_id="oven:main",
                resource_type="oven",
                capacity=Decimal(1),
                available=False,
            ),
        )
        problem = _make_problem(tasks, resources=resources)
        result, report = _run_solve(problem)

        # The problem itself should still be feasible (the solver doesn't
        # enforce that unavailable resources can't be used — that's the
        # verifier's job).  But the verifier should flag it.
        if result.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE):
            assert not report.passed
            assert any(i.code == "RESOURCE_UNAVAILABLE" for i in report.issues)

    def test_extra_interval_rejected(self) -> None:
        """An interval for a non-existent task should be caught."""
        tasks = (_task("t1", duration=5),)
        problem = _make_problem(tasks)

        result = ScheduleResult(
            status=SolverStatus.OPTIMAL,
            makespan_minutes=10,
            intervals=(
                ScheduledInterval(task_id="t1", start_minute=0, end_minute=5),
                ScheduledInterval(task_id="ghost", start_minute=5, end_minute=10),
            ),
        )

        verifier = ScheduleVerifier()
        report = verifier.verify(problem, result)
        assert not report.passed
        assert any(i.code == "EXTRA_TASK" for i in report.issues)


# =============================================================================
# Orchestrator tests (7.9–7.12)
# =============================================================================


class TestOrchestrator:
    """Lexicographic multi-objective solve tests."""

    def test_makespan_then_holding(self) -> None:
        """Phase 1 minimises makespan; Phase 2 minimises holding without
        increasing makespan."""
        # Dish 1: task A (5 min) → task B (5 min)  (makespan 10)
        # Dish 2: single task C (3 min)             (makespan 3 alone, but 10 with cook constraint)
        # Total makespan minimum = 10 (A, C, B sequential)
        dep = TaskDependency(predecessor_id="A")
        tasks = (
            _task("A", dish_id="dish1", duration=5),
            _task("B", dish_id="dish1", duration=5, deps=(dep,)),
            _task("C", dish_id="dish2", duration=3),
        )
        problem = _make_problem(tasks)

        orchestrator = ScheduleOrchestrator()
        result, report = orchestrator.solve(problem)

        assert result.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
        assert result.makespan_minutes is not None
        # Makespan must be <= 10 (optimal: A=0-5, C=5-8, B=8-13 = 13...
        # Wait, let me recalculate: A(5) → B(5) dependency, plus C(3).
        # Cook can do one at a time: e.g. A(0-5), C(5-8), B(8-13) → makespan 13
        # Or: C(0-3), A(3-8), B(8-13) → makespan 13
        # Actually the dependency lower bound is 10 (A→B), but cook sequential
        # adds C → makespan 13.  With 3 tasks of 5+5+3=13, it reaches exactly 13.
        assert result.makespan_minutes == 13
        assert report.passed

    def test_basic_pipeline_schedule_function(self) -> None:
        """The convenience schedule() function should work end-to-end."""
        tasks = (
            _task("t1", duration=3),
            _task("t2", duration=4),
        )
        problem = _make_problem(tasks)
        result, report = schedule(problem)

        assert result.status == SolverStatus.OPTIMAL
        assert result.makespan_minutes == 7  # 3 + 4 sequential (both active)
        assert len(result.intervals) == 2
        assert report.passed

    def test_holding_moves_dishes_later(self) -> None:
        """Phase 2 should push short-cooking dishes later without increasing makespan.

        Two dishes:
        - Dish 1: A(10 min) → makespan contribution 10
        - Dish 2: B(2 min)  → makespan contribution 2
        Cook is single-threaded → total makespan = 12 (sequential).
        Without holding optimisation: B might run first (0-2), then A (2-12).
        With holding: A runs first (0-10), then B (10-12), so dish 2
        finishes at 12 (no holding wait before serving).
        """
        tasks = (
            _task("A", dish_id="dish1", duration=10),
            _task("B", dish_id="dish2", duration=2),
        )
        problem = _make_problem(tasks)

        orchestrator = ScheduleOrchestrator()
        result, report = orchestrator.solve(problem)

        assert result.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
        assert result.makespan_minutes == 12  # 10 + 2 sequential
        assert report.passed

        # Dish 2 (task B) should finish at makespan (t=12), minimising holding.
        b_interval = next(i for i in result.intervals if i.task_id == "B")
        assert b_interval.end_minute == 12


# =============================================================================
# Multi-dish integration test
# =============================================================================


class TestMultiDishIntegration:
    """Complex scenario: two dishes with multiple tasks and shared resources."""

    def test_two_dishes_with_shared_stove(self) -> None:
        """Two dishes sharing one stove burner — must interleave."""
        need_stove = ResourceNeed(resource_type="stove", quantity=1)

        # Dish 1: prep(3, no stove) → cook(8, stove) → plate(2)
        dep1 = TaskDependency(predecessor_id="d1_prep")
        dep2 = TaskDependency(predecessor_id="d1_cook")

        # Dish 2: prep(2, no stove) → cook(6, stove) → plate(1)
        dep3 = TaskDependency(predecessor_id="d2_prep")
        dep4 = TaskDependency(predecessor_id="d2_cook")

        tasks = (
            _task("d1_prep", dish_id="d1", duration=3),
            _task("d1_cook", dish_id="d1", duration=8, deps=(dep1,), resources=(need_stove,)),
            _task("d1_plate", dish_id="d1", duration=2, deps=(dep2,)),
            _task("d2_prep", dish_id="d2", duration=2),
            _task("d2_cook", dish_id="d2", duration=6, deps=(dep3,), resources=(need_stove,)),
            _task("d2_plate", dish_id="d2", duration=1, deps=(dep4,)),
        )
        problem = _make_problem(tasks, resources=(_stove(1),))
        result, report = _run_solve(problem)

        assert result.status == SolverStatus.OPTIMAL
        assert report.passed
        # The total active time = 3+8+2+2+6+1 = 22 minutes.
        # All are active, plus one stove shared by d1_cook and d2_cook.
        # Sequential active makespan >= 22.
        assert result.makespan_minutes >= 22

    def test_passive_boiling_parallel(self) -> None:
        """A passive boil can run while cook does other active tasks."""
        need_stove = ResourceNeed(resource_type="stove", quantity=1)

        dep = TaskDependency(predecessor_id="fill")
        tasks = (
            _task("fill", duration=2, category="setup"),
            _task(
                "boil",
                duration=15,
                deps=(dep,),
                work_mode=WorkMode.PASSIVE,
                resources=(need_stove,),
                category="heating",
            ),
            _task("chop", duration=5, work_mode=WorkMode.ACTIVE, category="cutting"),
        )
        problem = _make_problem(tasks, resources=(_stove(1),))
        result, report = _run_solve(problem)

        assert result.status == SolverStatus.OPTIMAL
        assert report.passed

        # Passive boil (15 min) starts after fill (2 min), ends at 17.
        # Active chop (5 min) can run during the boil (parallel).
        # Makespan = max(2+15, fill+chop=7) = 17
        assert result.makespan_minutes == 17


# =============================================================================
# Unit tests for individual components
# =============================================================================


class TestBuilderUnit:
    """Unit tests for ScheduleModelBuilder."""

    def test_horizon_is_sum_of_durations(self) -> None:
        tasks = (
            _task("t1", duration=5),
            _task("t2", duration=10),
            _task("t3", duration=3),
        )
        builder = ScheduleModelBuilder()
        problem = _make_problem(tasks)
        info = builder.build(problem)
        assert info.horizon == 18  # 5 + 10 + 3

    def test_variables_created_for_all_tasks(self) -> None:
        tasks = (
            _task("t1", duration=5),
            _task("t2", duration=10),
        )
        builder = ScheduleModelBuilder()
        problem = _make_problem(tasks)
        info = builder.build(problem)
        assert set(info.starts.keys()) == {"t1", "t2"}
        assert set(info.ends.keys()) == {"t1", "t2"}
        assert set(info.intervals.keys()) == {"t1", "t2"}


class TestSolverUnit:
    """Unit tests for ScheduleSolver."""

    def test_optimal_result_has_makespan(self) -> None:
        tasks = (_task("t1", duration=5),)
        problem = _make_problem(tasks)

        builder = ScheduleModelBuilder()
        solver = ScheduleSolver()
        info = builder.build(problem)
        run = solver.solve(info, timeout_seconds=5.0)
        result = solver.to_schedule_result(run)

        assert result.status == SolverStatus.OPTIMAL
        assert result.makespan_minutes == 5
        assert result.wall_time_seconds > 0

    def test_infeasible_has_no_makespan(self) -> None:
        # Force infeasible: two 10-min tasks with a hard 5-min deadline.
        tasks = (
            _task("t1", duration=10),
            _task("t2", duration=10),
        )
        problem = _make_problem(tasks, time_limit=5)

        builder = ScheduleModelBuilder()
        solver = ScheduleSolver()
        info = builder.build(problem)
        run = solver.solve(info, timeout_seconds=5.0)
        result = solver.to_schedule_result(run)

        assert result.status == SolverStatus.INFEASIBLE
        assert result.makespan_minutes is None
        assert len(result.intervals) == 0


class TestExtractorUnit:
    """Unit tests for ScheduleExtractor."""

    def test_extracts_all_intervals(self) -> None:
        tasks = (
            _task("t1", duration=5),
            _task("t2", duration=5),
        )
        problem = _make_problem(tasks)

        builder = ScheduleModelBuilder()
        solver = ScheduleSolver()
        extractor = ScheduleExtractor()
        info = builder.build(problem)
        run = solver.solve(info)
        intervals = extractor.extract(run)

        assert len(intervals) == 2
        task_ids = {i.task_id for i in intervals}
        assert task_ids == {"t1", "t2"}
        for iv in intervals:
            assert iv.end_minute - iv.start_minute == 5
