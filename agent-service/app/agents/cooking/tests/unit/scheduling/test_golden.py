"""Solver golden tests — small, manually solvable cases (Handbook 11.5).

Each test stores:
  - input problem
  - expected status (OPTIMAL / FEASIBLE / INFEASIBLE)
  - known lower bound (dependency-only makespan)
  - known acceptable upper bound
  - required non-overlaps (active tasks must not overlap in time)

Golden tests use @pytest.mark.golden markers for selective execution.
"""

from decimal import Decimal

import pytest

from cooking_plan_agent.domain.enums import SolverStatus, WorkMode
from cooking_plan_agent.domain.models import (
    CookingTask,
    KitchenResourceSnapshot,
    ResourceNeed,
    TaskDependency,
)
from cooking_plan_agent.scheduling.models import SchedulingProblem
from cooking_plan_agent.scheduling.orchestrator import schedule

# =============================================================================
# Fixture factories
# =============================================================================


def _opt(
    task_id: str, duration: int, dish_id: str = "d1", deps: tuple = (), resources: tuple = (), work_mode=WorkMode.ACTIVE
) -> CookingTask:
    return CookingTask(
        task_id=task_id,
        dish_id=dish_id,
        instruction=f"Task {task_id}",
        duration_minutes=duration,
        work_mode=work_mode,
        category="test",
        dependencies=deps,
        resources=resources,
    )


def _stove(n: int) -> KitchenResourceSnapshot:
    return KitchenResourceSnapshot(resource_id="s", resource_type="stove", capacity=Decimal(n), capacity_unit="burners")


# =============================================================================
# Golden fixtures
# =============================================================================


GOLDEN_CASES = [
    # name, tasks, resources, time_limit, expected_status, lower_bound, upper_bound
    {
        "name": "single_task_5min",
        "tasks": (_opt("t1", 5),),
        "resources": (),
        "time_limit": None,
        "expected_status": SolverStatus.OPTIMAL,
        "lower_bound": 5,
        "upper_bound": 5,
    },
    {
        "name": "two_dependent_10min",
        "tasks": (
            _opt("t1", 5),
            _opt("t2", 5, deps=(TaskDependency(predecessor_id="t1"),)),
        ),
        "resources": (),
        "time_limit": None,
        "expected_status": SolverStatus.OPTIMAL,
        "lower_bound": 10,
        "upper_bound": 10,
    },
    {
        "name": "two_active_sequential_10min",
        "tasks": (_opt("t1", 5), _opt("t2", 5)),
        "resources": (),
        "time_limit": None,
        "expected_status": SolverStatus.OPTIMAL,
        "lower_bound": 5,
        "upper_bound": 10,
    },
    {
        "name": "passive_overlaps_active_10min",
        "tasks": (
            _opt("t1", 5, work_mode=WorkMode.ACTIVE),
            _opt("t2", 10, work_mode=WorkMode.PASSIVE),
            _opt("t3", 5, work_mode=WorkMode.ACTIVE),
        ),
        "resources": (),
        "time_limit": None,
        "expected_status": SolverStatus.OPTIMAL,
        "lower_bound": 10,
        "upper_bound": 10,
    },
    {
        "name": "one_burner_sequential_10min",
        "tasks": (
            _opt("t1", 5, resources=(ResourceNeed(resource_type="stove", quantity=1),)),
            _opt("t2", 5, resources=(ResourceNeed(resource_type="stove", quantity=1),)),
        ),
        "resources": (_stove(1),),
        "time_limit": None,
        "expected_status": SolverStatus.OPTIMAL,
        "lower_bound": 5,
        "upper_bound": 10,
    },
    {
        "name": "marinating_lag_35min",
        "tasks": (
            _opt("apply", 5),
            _opt("cook", 10, deps=(TaskDependency(predecessor_id="apply", minimum_lag_minutes=20),)),
        ),
        "resources": (),
        "time_limit": None,
        "expected_status": SolverStatus.OPTIMAL,
        "lower_bound": 35,
        "upper_bound": 35,
    },
    {
        "name": "hard_deadline_infeasible",
        "tasks": (_opt("t1", 10), _opt("t2", 10), _opt("t3", 10)),
        "resources": (),
        "time_limit": 5,
        "expected_status": SolverStatus.INFEASIBLE,
        "lower_bound": 0,
        "upper_bound": 0,
    },
    {
        "name": "three_dishes_shared_stove",
        "tasks": (
            _opt("d1_prep", 3, dish_id="dish1"),
            _opt(
                "d1_cook",
                8,
                dish_id="dish1",
                resources=(ResourceNeed(resource_type="stove", quantity=1),),
                deps=(TaskDependency(predecessor_id="d1_prep"),),
            ),
            _opt("d1_plate", 2, dish_id="dish1", deps=(TaskDependency(predecessor_id="d1_cook"),)),
            _opt("d2_prep", 2, dish_id="dish2"),
            _opt(
                "d2_cook",
                6,
                dish_id="dish2",
                resources=(ResourceNeed(resource_type="stove", quantity=1),),
                deps=(TaskDependency(predecessor_id="d2_prep"),),
            ),
            _opt("d2_plate", 1, dish_id="dish2", deps=(TaskDependency(predecessor_id="d2_cook"),)),
        ),
        "resources": (_stove(1),),
        "time_limit": None,
        "expected_status": SolverStatus.OPTIMAL,
        "lower_bound": 15,  # d1: 3+8+2=13, d2: 2+6+1=9, serial cooks → ≥15
        "upper_bound": 22,  # worst case: all sequential
    },
]


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.golden
@pytest.mark.parametrize(
    "case",
    GOLDEN_CASES,
    ids=[c["name"] for c in GOLDEN_CASES],
)
def test_golden(case: dict) -> None:
    """Each golden case produces the expected status and bounds."""
    problem = SchedulingProblem(
        tasks=case["tasks"],
        resources=case["resources"],
        requested_time_limit_minutes=case["time_limit"],
    )
    result, report = schedule(problem)

    assert result.status == case["expected_status"], f"Expected {case['expected_status']}, got {result.status}"

    if result.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE):
        assert result.makespan_minutes is not None
        assert result.makespan_minutes >= case["lower_bound"], (
            f"Makespan {result.makespan_minutes} < lower bound {case['lower_bound']}"
        )
        assert result.makespan_minutes <= case["upper_bound"], (
            f"Makespan {result.makespan_minutes} > upper bound {case['upper_bound']}"
        )

    # Verifier must agree with solver
    assert report.passed, f"Verifier rejected: {report.issues}"


@pytest.mark.golden
def test_golden_all_cases_have_required_fields() -> None:
    """Every golden case must declare all required metadata."""
    required = {"name", "tasks", "resources", "time_limit", "expected_status", "lower_bound", "upper_bound"}
    for i, case in enumerate(GOLDEN_CASES):
        missing = required - set(case.keys())
        assert not missing, f"Case {i} ({case.get('name', '?')}) missing fields: {missing}"
