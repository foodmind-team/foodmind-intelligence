"""P1-04: solver error semantics — only INFEASIBLE is a business outcome.

SCHEDULE_INFEASIBLE means ONLY that the solver proved no solution exists for
a VALID model. Model bugs, undetermined timeouts, and internal invariant
breaks each carry a distinct code and terminate at FAILED — never INFEASIBLE.

P3-03: solve_schedule_node uses ScheduleOrchestrator.solve; the tests patch
that method (not the legacy schedule() convenience function).
"""

from decimal import Decimal

import pytest

from cooking_plan_agent.domain.enums import SolverStatus, WorkMode
from cooking_plan_agent.domain.models import (
    CookingTask,
    GeneratePlanRequest,
    RecipeInput,
)
from cooking_plan_agent.scheduling.models import ScheduleResult
from cooking_plan_agent.workflow.nodes import solve_schedule_node
from cooking_plan_agent.workflow.routing import route_after_solve

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _FakeRuntime:
    """Minimal runtime stand-in — nodes only need .context at call time."""

    def __init__(self, context: object = None) -> None:
        self.context = context


def _request() -> GeneratePlanRequest:
    return GeneratePlanRequest(
        request_id="req-solver",
        user_id="u",
        recipes=(RecipeInput(recipe_id="r1", text="Cook.", target_servings=Decimal(2)),),
    )


def _task_graph():
    from cooking_plan_agent.preparation.task_graph import TaskGraph

    task = CookingTask(
        task_id="t1",
        dish_id="r1",
        instruction="Cook",
        duration_minutes=5,
        work_mode=WorkMode.ACTIVE,
        category="heating",
    )
    return TaskGraph(tasks=(task,), edges=())


def _state(task_graph=None) -> dict[str, object]:
    return {"request": _request(), "task_graph": task_graph}


def _patch_solver(monkeypatch, status, raises: type[Exception] | None = None) -> None:
    """Patch ScheduleOrchestrator.solve to return the given status.

    Returns (ScheduleResult, None) so nodes only see the status mapping.
    """

    def _solve(problem, optimization_level: str = "full"):  # noqa: ANN001
        if raises is not None:
            raise raises("injected solver failure")
        return ScheduleResult(status=status), None

    monkeypatch.setattr(
        "cooking_plan_agent.scheduling.orchestrator.ScheduleOrchestrator",
        type("_FakeOrchestrator", (), {"solve": staticmethod(_solve)}),
    )


# ---------------------------------------------------------------------------
# Node-level: solver status mapping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_infeasible_status_is_business_outcome(monkeypatch) -> None:
    """INFEASIBLE → schedule_result kept, no error → INFEASIBLE terminal."""
    _patch_solver(monkeypatch, SolverStatus.INFEASIBLE)
    result = await solve_schedule_node(_state(_task_graph()), _FakeRuntime())

    assert "schedule_result" in result
    assert "error" not in result
    assert route_after_solve(result) == "render_infeasible_response"


@pytest.mark.asyncio
async def test_unknown_status_maps_to_schedule_unknown(monkeypatch) -> None:
    """UNKNOWN (timeout, undetermined) → SCHEDULE_UNKNOWN → FAILED."""
    _patch_solver(monkeypatch, SolverStatus.UNKNOWN)
    result = await solve_schedule_node(_state(_task_graph()), _FakeRuntime())

    error = result.get("error")
    assert error is not None
    assert error.error_code == "SCHEDULE_UNKNOWN"
    assert route_after_solve(result) == "render_failed_response"


@pytest.mark.asyncio
async def test_model_invalid_status_maps_to_schedule_model_invalid(monkeypatch) -> None:
    """MODEL_INVALID → SCHEDULE_MODEL_INVALID → FAILED (never INFEASIBLE)."""
    _patch_solver(monkeypatch, SolverStatus.MODEL_INVALID)
    result = await solve_schedule_node(_state(_task_graph()), _FakeRuntime())

    error = result.get("error")
    assert error is not None
    assert error.error_code == "SCHEDULE_MODEL_INVALID"
    assert route_after_solve(result) == "render_failed_response"


@pytest.mark.asyncio
async def test_value_error_maps_to_model_invalid(monkeypatch) -> None:
    """Model-construction failure (ValueError/TypeError) → SCHEDULE_MODEL_INVALID."""
    _patch_solver(monkeypatch, SolverStatus.OPTIMAL, raises=ValueError)
    result = await solve_schedule_node(_state(_task_graph()), _FakeRuntime())

    error = result.get("error")
    assert error is not None
    assert error.error_code == "SCHEDULE_MODEL_INVALID"
    assert route_after_solve(result) == "render_failed_response"


@pytest.mark.asyncio
async def test_runtime_error_maps_to_internal_error(monkeypatch) -> None:
    """Solver-internal RuntimeError → INTERNAL_ERROR → FAILED."""
    _patch_solver(monkeypatch, SolverStatus.OPTIMAL, raises=RuntimeError)
    result = await solve_schedule_node(_state(_task_graph()), _FakeRuntime())

    error = result.get("error")
    assert error is not None
    assert error.error_code == "INTERNAL_ERROR"
    assert route_after_solve(result) == "render_failed_response"


@pytest.mark.asyncio
async def test_missing_task_graph_is_internal_invariant_failure() -> None:
    """Missing DAG must NEVER be marked INFEASIBLE (P1-04 rule 3)."""
    result = await solve_schedule_node(_state(task_graph=None), _FakeRuntime())

    error = result.get("error")
    assert error is not None
    assert error.error_code == "INTERNAL_ERROR", f"Got {error.error_code} instead of INTERNAL_ERROR"
    assert error.error_code != "SCHEDULE_INFEASIBLE"
    assert route_after_solve(result) == "render_failed_response"


# ---------------------------------------------------------------------------
# Routing: enum-based status comparisons
# ---------------------------------------------------------------------------


def test_route_after_solve_optimal_verifies() -> None:
    result = {"schedule_result": ScheduleResult(status=SolverStatus.OPTIMAL)}
    assert route_after_solve(result) == "verify_schedule"


def test_route_after_solve_feasible_verifies() -> None:
    result = {"schedule_result": ScheduleResult(status=SolverStatus.FEASIBLE)}
    assert route_after_solve(result) == "verify_schedule"


def test_route_after_solve_infeasible_routes_business() -> None:
    result = {"schedule_result": ScheduleResult(status=SolverStatus.INFEASIBLE)}
    assert route_after_solve(result) == "render_infeasible_response"


def test_route_after_solve_unknown_fails() -> None:
    result = {"schedule_result": ScheduleResult(status=SolverStatus.UNKNOWN)}
    assert route_after_solve(result) == "render_failed_response"


def test_route_after_solve_model_invalid_fails() -> None:
    result = {"schedule_result": ScheduleResult(status=SolverStatus.MODEL_INVALID)}
    assert route_after_solve(result) == "render_failed_response"


def test_route_after_solve_error_takes_precedence() -> None:
    result = {
        "error": object(),
        "schedule_result": ScheduleResult(status=SolverStatus.INFEASIBLE),
    }
    assert route_after_solve(result) == "render_failed_response"
