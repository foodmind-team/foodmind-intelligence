"""P5-2: run_tool_node 与控制器路由函数。"""

from types import SimpleNamespace

import pytest

from cooking_plan_agent.domain.models import GeneratePlanRequest, RecipeInput, WorkflowError
from cooking_plan_agent.workflow.controller_nodes import run_tool_node
from cooking_plan_agent.workflow.routing import route_after_controller
from cooking_plan_agent.workflow.state import PlanState


def _request() -> GeneratePlanRequest:
    return GeneratePlanRequest(
        request_id="req-controller",
        user_id="user-controller",
        recipes=(RecipeInput(recipe_id="r1", text="300 g tofu", target_servings=2),),
        time_limit_minutes=60,
    )


class _FakeExtractor:
    async def extract(self, source_text: str) -> dict[str, object]:
        return {"title": "Tofu", "source_text": source_text}


def _runtime_with_extractor() -> SimpleNamespace:
    return SimpleNamespace(
        context=SimpleNamespace(
            recipe_extractor=_FakeExtractor(),
            recipe_researcher=None,
            safety_engine=None,
        )
    )


# ---------------------------------------------------------------------------
# run_tool_node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_tool_executes_and_backfills_observation() -> None:
    state: PlanState = {
        "request": _request(),
        "pending_decision": {
            "type": "tool_call",
            "tool": "parse_recipe",
            "arguments": {"source_text": "300 g tofu"},
        },
    }
    delta = await run_tool_node(state, _runtime_with_extractor())  # type: ignore[arg-type]
    assert delta["tool_calls"][-1]["tool"] == "parse_recipe"
    observation = delta["observations"][-1]
    assert observation["ok"] is True
    assert "candidate" in observation
    assert delta["agent_step"] == 1
    assert delta["pending_decision"] == {}


@pytest.mark.asyncio
async def test_run_tool_unknown_tool_returns_error_observation() -> None:
    state: PlanState = {
        "request": _request(),
        "pending_decision": {"type": "tool_call", "tool": "nope", "arguments": {}},
    }
    delta = await run_tool_node(state, _runtime_with_extractor())  # type: ignore[arg-type]
    observation = delta["observations"][-1]
    assert observation["ok"] is False
    assert "unknown tool" in str(observation["error"])


@pytest.mark.asyncio
async def test_run_tool_exception_returns_error_observation() -> None:
    class _BrokenExtractor:
        async def extract(self, source_text: str) -> dict[str, object]:
            raise ValueError("boom")

    runtime = SimpleNamespace(
        context=SimpleNamespace(
            recipe_extractor=_BrokenExtractor(),
            recipe_researcher=None,
            safety_engine=None,
        )
    )
    state: PlanState = {
        "request": _request(),
        "pending_decision": {
            "type": "tool_call",
            "tool": "parse_recipe",
            "arguments": {"source_text": "x"},
        },
    }
    delta = await run_tool_node(state, runtime)  # type: ignore[arg-type]
    observation = delta["observations"][-1]
    assert observation["ok"] is False
    assert "ValueError" in str(observation["error"])


# ---------------------------------------------------------------------------
# route_after_controller
# ---------------------------------------------------------------------------


def test_route_after_controller_tool_call_runs_tool() -> None:
    state: PlanState = {
        "request": _request(),
        "agent_mode": "controller",
        "pending_decision": {"type": "tool_call", "tool": "parse_recipe"},
    }
    assert route_after_controller(state) == "run_tool"


def test_route_after_controller_final_goes_dag() -> None:
    state: PlanState = {
        "request": _request(),
        "agent_mode": "controller",
        "pending_decision": {"type": "final", "response": {"status": "READY"}},
    }
    assert route_after_controller(state) == "validate_input"


def test_route_after_controller_deterministic_mode_goes_dag() -> None:
    state: PlanState = {
        "request": _request(),
        "agent_mode": "deterministic",
    }
    assert route_after_controller(state) == "validate_input"


def test_route_after_controller_error_wins() -> None:
    state: PlanState = {
        "request": _request(),
        "error": WorkflowError(error_code="E", message="x", correlation_id="c"),
    }
    assert route_after_controller(state) == "render_failed_response"
