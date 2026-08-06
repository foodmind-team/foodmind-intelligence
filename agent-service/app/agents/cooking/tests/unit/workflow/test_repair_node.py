"""P5-3: repair_schedule_node 与路由函数。"""

from types import SimpleNamespace

import pytest

from cooking_plan_agent.domain.models import GeneratePlanRequest, RecipeInput, WorkflowError
from cooking_plan_agent.scheduling.models import (
    RepairAttemptRecord,
    VerificationIssue,
    VerificationReport,
)
from cooking_plan_agent.workflow.repair_nodes import repair_schedule_node
from cooking_plan_agent.workflow.routing import route_after_repair, route_after_verification
from cooking_plan_agent.workflow.state import PlanState


def _request() -> GeneratePlanRequest:
    return GeneratePlanRequest(
        request_id="req-repair",
        user_id="user-repair",
        recipes=(RecipeInput(recipe_id="r1", text="300 g tofu", target_servings=2),),
        time_limit_minutes=60,
    )


def _failed_report() -> VerificationReport:
    return VerificationReport(
        passed=False,
        issues=(VerificationIssue(code="CAPACITY_EXCEEDED", message="capacity"),),
    )


@pytest.mark.asyncio
async def test_repair_node_schedules_retry():
    state: PlanState = {
        "request": _request(),
        "verification_report": _failed_report(),
    }
    runtime = SimpleNamespace(context=SimpleNamespace())
    delta = await repair_schedule_node(state, runtime)  # type: ignore[arg-type]
    assert delta["repair_attempts"] == 1
    assert delta["solver_overrides"]["optimization_level"] == "phase12"
    history = delta["repair_history"]
    assert history[-1].outcome == "retrying"


@pytest.mark.asyncio
async def test_repair_node_gives_up_when_exhausted():
    state: PlanState = {
        "request": _request(),
        "verification_report": _failed_report(),
        "repair_attempts": 2,
        "solver_overrides": {"optimization_level": "makespan"},
    }
    runtime = SimpleNamespace(context=SimpleNamespace())
    delta = await repair_schedule_node(state, runtime)  # type: ignore[arg-type]
    assert delta["repair_attempts"] == 3
    assert delta["repair_history"][-1].outcome == "gave_up"


def test_route_after_verification_routes_to_repair():
    state: PlanState = {
        "request": _request(),
        "verification_report": _failed_report(),
    }
    assert route_after_verification(state) == "repair_schedule"


def test_route_after_verification_passed_still_explains():
    state: PlanState = {
        "request": _request(),
        "verification_report": VerificationReport(passed=True, issues=()),
    }
    assert route_after_verification(state) == "explain_schedule"


def test_route_after_verification_error_wins():
    state: PlanState = {
        "request": _request(),
        "error": WorkflowError(error_code="E", message="x", correlation_id="c"),
    }
    assert route_after_verification(state) == "render_failed_response"


def test_route_after_repair_retries():
    state: PlanState = {
        "request": _request(),
        "repair_history": (
            RepairAttemptRecord(attempt=1, issues=("CAPACITY_EXCEEDED",), action="lower", outcome="retrying"),
        ),
    }
    assert route_after_repair(state) == "solve_schedule"


def test_route_after_repair_gives_up():
    state: PlanState = {
        "request": _request(),
        "repair_history": (
            RepairAttemptRecord(attempt=1, issues=("MISSING_TASK",), action="give_up", outcome="gave_up"),
        ),
    }
    assert route_after_repair(state) == "render_failed_response"


def test_route_after_repair_error_wins():
    state: PlanState = {
        "request": _request(),
        "error": WorkflowError(error_code="E", message="x", correlation_id="c"),
    }
    assert route_after_repair(state) == "render_failed_response"
