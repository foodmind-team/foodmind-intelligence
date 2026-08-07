"""P5-2: agent_controller_node 与 ReAct 控制器状态/配置。"""

from types import SimpleNamespace

import pytest

from cooking_plan_agent.config.settings import get_settings
from cooking_plan_agent.workflow.controller_nodes import agent_controller_node
from cooking_plan_agent.workflow.state import PlanState

# ---------------------------------------------------------------------------
# 配置（P5-2）
# ---------------------------------------------------------------------------


def test_agent_settings_defaults(monkeypatch) -> None:
    get_settings.cache_clear()
    monkeypatch.delenv("COOKING_PLAN_AGENT_MAX_STEPS", raising=False)
    monkeypatch.delenv("COOKING_PLAN_AGENT_CONTROLLER_ENABLED", raising=False)
    settings = get_settings()
    assert settings.agent_max_steps == 5
    assert settings.agent_controller_enabled is False


def test_agent_settings_env_override(monkeypatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("COOKING_PLAN_AGENT_MAX_STEPS", "3")
    monkeypatch.setenv("COOKING_PLAN_AGENT_CONTROLLER_ENABLED", "true")
    settings = get_settings()
    assert settings.agent_max_steps == 3
    assert settings.agent_controller_enabled is True


# ---------------------------------------------------------------------------
# 状态扩展（P5-2）
# ---------------------------------------------------------------------------


def test_plan_state_accepts_controller_fields() -> None:
    state: PlanState = {
        "messages": ({"role": "user", "content": "hi"},),
        "agent_step": 1,
        "agent_mode": "controller",
        "tool_calls": ({"tool": "parse_recipe"},),
        "observations": ({"ok": True, "candidate": {"title": "T"}},),
        "pending_decision": {"type": "tool_call", "tool": "parse_recipe"},
    }
    assert state["agent_step"] == 1
    assert state["agent_mode"] == "controller"
    assert state["tool_calls"][0]["tool"] == "parse_recipe"
    assert state["pending_decision"]["type"] == "tool_call"


# ---------------------------------------------------------------------------
# agent_controller_node 行为
# ---------------------------------------------------------------------------


class _FakeController:
    """伪造控制器：按脚本依次返回决策。"""

    def __init__(self, decisions: list[dict[str, object]]) -> None:
        self._decisions = list(decisions)
        self.calls = 0

    async def decide(self, state_summary: dict[str, object]) -> dict[str, object]:
        self.calls += 1
        return self._decisions.pop(0)


def _runtime(controller: object) -> SimpleNamespace:
    return SimpleNamespace(context=SimpleNamespace(agent_controller=controller))


def _enable_controller(monkeypatch: pytest.MonkeyPatch, enabled: bool = True, max_steps: int = 5) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("COOKING_PLAN_AGENT_CONTROLLER_ENABLED", "true" if enabled else "false")
    monkeypatch.setenv("COOKING_PLAN_AGENT_MAX_STEPS", str(max_steps))


@pytest.mark.asyncio
async def test_node_no_controller_falls_back(monkeypatch) -> None:
    _enable_controller(monkeypatch)
    delta = await agent_controller_node({}, _runtime(None))  # type: ignore[arg-type]
    assert delta == {"agent_mode": "deterministic"}


@pytest.mark.asyncio
async def test_node_disabled_falls_back(monkeypatch) -> None:
    _enable_controller(monkeypatch, enabled=False)
    delta = await agent_controller_node({}, _runtime(_FakeController([])))  # type: ignore[arg-type]
    assert delta == {"agent_mode": "deterministic"}


@pytest.mark.asyncio
async def test_node_step_limit_falls_back(monkeypatch) -> None:
    _enable_controller(monkeypatch, max_steps=2)
    state: PlanState = {"agent_step": 2}
    delta = await agent_controller_node(state, _runtime(_FakeController([])))  # type: ignore[arg-type]
    assert delta == {"agent_mode": "deterministic"}


@pytest.mark.asyncio
async def test_node_tool_call_decision_applied(monkeypatch) -> None:
    _enable_controller(monkeypatch)
    controller = _FakeController([{"type": "tool_call", "tool": "parse_recipe", "arguments": {"source_text": "x"}}])
    delta = await agent_controller_node({}, _runtime(controller))  # type: ignore[arg-type]
    assert delta["pending_decision"]["type"] == "tool_call"
    assert delta["pending_decision"]["tool"] == "parse_recipe"
    assert delta["agent_trace"][-1]["action"] == "tool_call"
    assert controller.calls == 1


@pytest.mark.asyncio
async def test_node_final_decision_applied(monkeypatch) -> None:
    _enable_controller(monkeypatch)
    controller = _FakeController([{"type": "final", "response": {"status": "READY"}}])
    delta = await agent_controller_node({}, _runtime(controller))  # type: ignore[arg-type]
    assert delta["pending_decision"]["type"] == "final"
    assert delta["agent_trace"][-1]["action"] == "final"


@pytest.mark.asyncio
async def test_node_fallback_decision(monkeypatch) -> None:
    _enable_controller(monkeypatch)
    controller = _FakeController([{"type": "fallback"}])
    delta = await agent_controller_node({}, _runtime(controller))  # type: ignore[arg-type]
    assert delta == {"agent_mode": "deterministic"}


@pytest.mark.asyncio
async def test_node_controller_error_falls_back(monkeypatch) -> None:
    _enable_controller(monkeypatch)

    class _Broken:
        async def decide(self, state_summary: dict[str, object]) -> dict[str, object]:
            raise RuntimeError("llm down")

    delta = await agent_controller_node({}, _runtime(_Broken()))  # type: ignore[arg-type]
    assert delta == {"agent_mode": "deterministic"}


@pytest.mark.asyncio
async def test_node_malformed_decision_falls_back(monkeypatch) -> None:
    _enable_controller(monkeypatch)
    controller = _FakeController([{"type": "nonsense"}])
    delta = await agent_controller_node({}, _runtime(controller))  # type: ignore[arg-type]
    assert delta == {"agent_mode": "deterministic"}


@pytest.mark.asyncio
async def test_node_tool_call_missing_tool_falls_back(monkeypatch) -> None:
    _enable_controller(monkeypatch)
    controller = _FakeController([{"type": "tool_call", "arguments": {}}])
    delta = await agent_controller_node({}, _runtime(controller))  # type: ignore[arg-type]
    assert delta == {"agent_mode": "deterministic"}
