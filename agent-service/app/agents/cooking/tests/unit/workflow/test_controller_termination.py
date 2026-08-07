"""P5-2: ReAct 控制器终止保障与可观测性（图级）。"""

from decimal import Decimal

import pytest

from cooking_plan_agent.config.settings import get_settings
from cooking_plan_agent.domain.enums import HeatLevel
from cooking_plan_agent.domain.models import (
    ExtractedIngredient,
    ExtractedRecipeCandidate,
    ExtractedStep,
    GeneratePlanRequest,
    InventoryLotSnapshot,
    KitchenResourceSnapshot,
    ReadyPlanResponse,
)
from cooking_plan_agent.workflow.context import WorkflowContext
from cooking_plan_agent.workflow.graph import build_cooking_plan_graph


class _FakeExtractor:
    """Returns a gap-free candidate so the DAG path reaches READY."""

    async def extract(self, source_text: str) -> ExtractedRecipeCandidate:
        return ExtractedRecipeCandidate(
            recipe_id="r1",
            dish_name="Test Dish",
            original_servings=2,
            source_language="en",
            ingredients=(
                ExtractedIngredient(
                    raw_text="chicken 200g",
                    name="chicken breast",
                    quantity=200,
                    unit="g",
                ),
            ),
            steps=(
                ExtractedStep(
                    step_number=1,
                    instruction="Cook for 10 minutes",
                    category="heating",
                    active_duration_minutes=10,
                    heat_level=HeatLevel.HIGH,
                    target_temperature_c=Decimal(200),
                    resources_hint=("stove",),
                ),
            ),
        )


def _valid_request() -> GeneratePlanRequest:
    return GeneratePlanRequest(
        request_id="req-agent-term",
        user_id="user-agent",
        recipes=({"recipe_id": "r1", "text": "Cook chicken for 10 minutes. Serves 2.", "target_servings": 2},),
        dietary_restrictions=(),
        user_allergens=(),
        inventory_lots=(
            InventoryLotSnapshot(
                lot_id="lot-1",
                item_id="item-1",
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
        ),
    )


def _context() -> WorkflowContext:
    return WorkflowContext(recipe_extractor=_FakeExtractor())


class _LoopingController:
    """始终输出 tool_call —— 用于验证步数耗尽强制回退。"""

    def __init__(self, tool: str = "nope") -> None:
        self.tool = tool
        self.calls = 0

    async def decide(self, state_summary: dict[str, object]) -> dict[str, object]:
        self.calls += 1
        return {"type": "tool_call", "tool": self.tool, "arguments": {}}


class _BrokenController:
    async def decide(self, state_summary: dict[str, object]) -> dict[str, object]:
        raise RuntimeError("llm down")


def _enable_controller(monkeypatch: pytest.MonkeyPatch, enabled: bool = True, max_steps: int = 5) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("COOKING_PLAN_AGENT_CONTROLLER_ENABLED", "true" if enabled else "false")
    monkeypatch.setenv("COOKING_PLAN_AGENT_MAX_STEPS", str(max_steps))


# ---------------------------------------------------------------------------
# 终止保障
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_controller_step_limit_forces_dag_fallback(monkeypatch) -> None:
    """连续 tool_call 超过 agent_max_steps 后强制回退确定性 DAG，最终 READY。"""
    _enable_controller(monkeypatch, max_steps=2)
    controller = _LoopingController()
    ctx = WorkflowContext(recipe_extractor=_FakeExtractor(), agent_controller=controller)

    graph = build_cooking_plan_graph()
    result = await graph.ainvoke(
        {"request": _valid_request()},
        context=ctx,
        config={"recursion_limit": 40},
    )

    # 步数耗尽 -> agent_mode=deterministic -> DAG 正常走完 -> READY。
    assert result["agent_mode"] == "deterministic"
    response = result.get("response")
    assert isinstance(response, ReadyPlanResponse), f"expected READY, got {type(response).__name__}"
    # 每轮 tool_call 都有留痕，且 agent_step 不超过上限。
    assert len(result.get("tool_calls", ())) == 2
    assert result["agent_step"] == 2
    assert result["agent_trace"][-1]["action"] in ("tool_call", "final", "fallback")


@pytest.mark.asyncio
async def test_controller_exception_falls_back_to_dag(monkeypatch) -> None:
    """控制器抛异常 -> agent_mode=deterministic 回退，DAG 正常走完。"""
    _enable_controller(monkeypatch)
    ctx = WorkflowContext(recipe_extractor=_FakeExtractor(), agent_controller=_BrokenController())

    graph = build_cooking_plan_graph()
    result = await graph.ainvoke(
        {"request": _valid_request()},
        context=ctx,
        config={"recursion_limit": 40},
    )

    assert result["agent_mode"] == "deterministic"
    response = result.get("response")
    assert isinstance(response, ReadyPlanResponse), f"expected READY, got {type(response).__name__}"


@pytest.mark.asyncio
async def test_controller_unknown_tool_observation_does_not_crash(monkeypatch) -> None:
    """未知工具 -> run_tool 返回 ok=False 观察；控制器读到错误后修正（fallback）。"""
    _enable_controller(monkeypatch, max_steps=3)

    class _RecoveringController:
        def __init__(self) -> None:
            self.calls = 0

        async def decide(self, state_summary: dict[str, object]) -> dict[str, object]:
            self.calls += 1
            if self.calls == 1:
                # 第一次：错误地调用不存在的工具。
                return {"type": "tool_call", "tool": "no_such_tool", "arguments": {}}
            # 第二次：看到 ok=False 观察后回退到确定性 DAG（修正路径）。
            return {"type": "fallback"}

    controller = _RecoveringController()
    ctx = WorkflowContext(recipe_extractor=_FakeExtractor(), agent_controller=controller)

    graph = build_cooking_plan_graph()
    result = await graph.ainvoke(
        {"request": _valid_request()},
        context=ctx,
        config={"recursion_limit": 40},
    )

    observations = result.get("observations", ())
    assert observations[0]["ok"] is False
    assert "unknown tool" in str(observations[0]["error"])
    assert result["agent_mode"] == "deterministic"
    response = result.get("response")
    assert isinstance(response, ReadyPlanResponse), f"expected READY, got {type(response).__name__}"


# ---------------------------------------------------------------------------
# 零回归：禁用开关时图行为与改造前一致
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_controller_disabled_behaves_like_legacy(monkeypatch) -> None:
    """agent_controller_enabled=false 时，无 controller 依赖、走原 DAG 直达 READY。"""
    _enable_controller(monkeypatch, enabled=False)
    ctx = WorkflowContext(recipe_extractor=_FakeExtractor())

    graph = build_cooking_plan_graph()
    result = await graph.ainvoke(
        {"request": _valid_request()},
        context=ctx,
        config={"recursion_limit": 40},
    )

    assert result.get("agent_mode") is None
    response = result.get("response")
    assert isinstance(response, ReadyPlanResponse), f"expected READY, got {type(response).__name__}"
