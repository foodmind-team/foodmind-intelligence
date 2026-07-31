"""Mutation-style negative schedule tests (Handbook 11.6).

Take a valid schedule and deliberately corrupt it. The independent
verifier must reject every mutation.
"""

from decimal import Decimal

from cooking_plan_agent.domain.enums import SolverStatus, WorkMode
from cooking_plan_agent.domain.models import (
    CookingTask,
    KitchenResourceSnapshot,
    ResourceNeed,
    TaskDependency,
)
from cooking_plan_agent.scheduling.models import (
    ScheduledInterval,
    ScheduleResult,
    SchedulingProblem,
)
from cooking_plan_agent.scheduling.verifier import ScheduleVerifier


def _task(tid: str, dur: int = 5, resources: tuple = (), deps: tuple = ()) -> CookingTask:
    return CookingTask(
        task_id=tid,
        dish_id="d1",
        instruction=f"Task {tid}",
        duration_minutes=dur,
        work_mode=WorkMode.ACTIVE,
        category="test",
        dependencies=deps,
        resources=resources,
    )


def _valid_schedule(tasks: tuple[CookingTask, ...]) -> tuple[ScheduleResult, SchedulingProblem]:
    """Build a valid sequential schedule for the given tasks."""
    intervals = tuple(
        ScheduledInterval(task_id=t.task_id, start_minute=i * 5, end_minute=(i + 1) * 5) for i, t in enumerate(tasks)
    )
    makespan = len(tasks) * 5
    problem = SchedulingProblem(tasks=tasks, resources=())
    result = ScheduleResult(
        status=SolverStatus.OPTIMAL,
        makespan_minutes=makespan,
        intervals=intervals,
    )
    return result, problem


def _verify_corruption(corrupted: ScheduleResult, problem: SchedulingProblem, expected_code: str) -> None:
    """Helper: verify that the corrupted result is rejected with the expected code."""
    verifier = ScheduleVerifier()
    report = verifier.verify(problem, corrupted)
    assert not report.passed, "Expected verifier to reject, but it passed"
    assert any(i.code == expected_code for i in report.issues), (
        f"Expected issue code '{expected_code}', got {[i.code for i in report.issues]}"
    )


# =============================================================================
# Mutation 1: Move a task before its predecessor → MIN_LAG_VIOLATION
# =============================================================================


def test_move_task_before_predecessor() -> None:
    """Swap the order of dependent tasks — verifier must reject."""
    dep = TaskDependency(predecessor_id="t1")
    tasks = (_task("t1"), _task("t2", deps=(dep,)))

    corrupted = ScheduleResult(
        status=SolverStatus.OPTIMAL,
        makespan_minutes=10,
        intervals=(
            ScheduledInterval(task_id="t2", start_minute=0, end_minute=5),
            ScheduledInterval(task_id="t1", start_minute=5, end_minute=10),
        ),
    )
    problem = SchedulingProblem(tasks=tasks, resources=())
    _verify_corruption(corrupted, problem, "MIN_LAG_VIOLATION")


# =============================================================================
# Mutation 2: Overlap two active tasks → ACTIVE_OVERLAP
# =============================================================================


def test_overlap_active_tasks() -> None:
    """Two active tasks overlapping — verifier must reject."""
    tasks = (_task("t1", dur=5), _task("t2", dur=5))

    corrupted = ScheduleResult(
        status=SolverStatus.OPTIMAL,
        makespan_minutes=7,
        intervals=(
            ScheduledInterval(task_id="t1", start_minute=0, end_minute=5),
            ScheduledInterval(task_id="t2", start_minute=2, end_minute=7),
        ),
    )
    problem = SchedulingProblem(tasks=tasks, resources=())
    _verify_corruption(corrupted, problem, "ACTIVE_OVERLAP")


# =============================================================================
# Mutation 3: Assign an undersized pot (resource capacity exceed) → CAPACITY_EXCEEDED
# =============================================================================


def test_exceed_resource_capacity() -> None:
    """Two tasks sharing one stove but scheduled in parallel → CAPACITY_EXCEEDED."""
    need_stove = ResourceNeed(resource_type="stove", quantity=1)
    tasks = (
        _task("t1", dur=5, resources=(need_stove,)),
        _task("t2", dur=5, resources=(need_stove,)),
    )
    stove = KitchenResourceSnapshot(
        resource_id="s1",
        resource_type="stove",
        capacity=Decimal(1),
        capacity_unit="burners",
    )

    corrupted = ScheduleResult(
        status=SolverStatus.OPTIMAL,
        makespan_minutes=5,
        intervals=(
            ScheduledInterval(task_id="t1", start_minute=0, end_minute=5),
            ScheduledInterval(task_id="t2", start_minute=0, end_minute=5),
        ),
    )
    problem = SchedulingProblem(tasks=tasks, resources=(stove,))
    _verify_corruption(corrupted, problem, "CAPACITY_EXCEEDED")


# =============================================================================
# Mutation 4: Change reported makespan → MAKESPAN_MISMATCH
# =============================================================================


def test_wrong_makespan() -> None:
    """Reported makespan doesn't match actual intervals → MAKESPAN_MISMATCH."""
    tasks = (_task("t1", dur=5),)

    corrupted = ScheduleResult(
        status=SolverStatus.OPTIMAL,
        makespan_minutes=999,
        intervals=(ScheduledInterval(task_id="t1", start_minute=0, end_minute=5),),
    )
    problem = SchedulingProblem(tasks=tasks, resources=())
    _verify_corruption(corrupted, problem, "MAKESPAN_MISMATCH")


# =============================================================================
# Mutation 5: Duplicate a task → EXTRA_TASK
# =============================================================================


def test_extra_task_interval() -> None:
    """Extra interval for a non-existent task → EXTRA_TASK."""
    tasks = (_task("t1", dur=5),)

    corrupted = ScheduleResult(
        status=SolverStatus.OPTIMAL,
        makespan_minutes=10,
        intervals=(
            ScheduledInterval(task_id="t1", start_minute=0, end_minute=5),
            ScheduledInterval(task_id="ghost_task", start_minute=5, end_minute=10),
        ),
    )
    problem = SchedulingProblem(tasks=tasks, resources=())
    _verify_corruption(corrupted, problem, "EXTRA_TASK")


# =============================================================================
# Mutation 6: Remove a task → MISSING_TASK
# =============================================================================


def test_missing_task_interval() -> None:
    """Missing interval for a required task → MISSING_TASK."""
    tasks = (_task("t1", dur=5), _task("t2", dur=5))

    corrupted = ScheduleResult(
        status=SolverStatus.OPTIMAL,
        makespan_minutes=5,
        intervals=(ScheduledInterval(task_id="t1", start_minute=0, end_minute=5),),
    )
    problem = SchedulingProblem(tasks=tasks, resources=())
    _verify_corruption(corrupted, problem, "MISSING_TASK")


# =============================================================================
# Mutation 7: Wrong duration → DURATION_MISMATCH
# =============================================================================


def test_duration_mismatch() -> None:
    """Interval end - start doesn't match task duration → DURATION_MISMATCH."""
    tasks = (_task("t1", dur=5),)

    corrupted = ScheduleResult(
        status=SolverStatus.OPTIMAL,
        makespan_minutes=10,
        intervals=(ScheduledInterval(task_id="t1", start_minute=0, end_minute=10),),
    )
    problem = SchedulingProblem(tasks=tasks, resources=())
    _verify_corruption(corrupted, problem, "DURATION_MISMATCH")


# =============================================================================
# Mutation 8: Negative start time → NEGATIVE_START
# =============================================================================


def test_negative_start() -> None:
    """A task with a negative start time → NEGATIVE_START."""
    tasks = (_task("t1", dur=5),)

    corrupted = ScheduleResult(
        status=SolverStatus.OPTIMAL,
        makespan_minutes=5,
        intervals=(ScheduledInterval(task_id="t1", start_minute=-1, end_minute=4),),
    )
    problem = SchedulingProblem(tasks=tasks, resources=())
    _verify_corruption(corrupted, problem, "NEGATIVE_START")


# =============================================================================
# Mutation 9: Task exceeds makespan → EXCEEDS_MAKESPAN
# =============================================================================


def test_exceeds_makespan() -> None:
    """Task ends after reported makespan → EXCEEDS_MAKESPAN."""
    tasks = (_task("t1", dur=5),)

    corrupted = ScheduleResult(
        status=SolverStatus.OPTIMAL,
        makespan_minutes=3,
        intervals=(ScheduledInterval(task_id="t1", start_minute=0, end_minute=5),),
    )
    problem = SchedulingProblem(tasks=tasks, resources=())
    _verify_corruption(corrupted, problem, "EXCEEDS_MAKESPAN")


# =============================================================================
# Mutation 10: Removed sanitisation task (resource unavailable)
# =============================================================================


def test_resource_unavailable() -> None:
    """Task requires resource that's marked unavailable → RESOURCE_UNAVAILABLE."""
    need_oven = ResourceNeed(resource_type="oven", quantity=1)
    tasks = (_task("t1", dur=5, resources=(need_oven,)),)

    oven = KitchenResourceSnapshot(
        resource_id="oven:main",
        resource_type="oven",
        capacity=Decimal(1),
        available=False,
    )

    corrupted = ScheduleResult(
        status=SolverStatus.OPTIMAL,
        makespan_minutes=5,
        intervals=(ScheduledInterval(task_id="t1", start_minute=0, end_minute=5),),
    )
    problem = SchedulingProblem(tasks=tasks, resources=(oven,))
    _verify_corruption(corrupted, problem, "RESOURCE_UNAVAILABLE")
