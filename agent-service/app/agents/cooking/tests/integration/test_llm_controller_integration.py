"""P5-2: LLMReActController 图级集成 —— 真实控制器驱动 ReAct 循环。"""

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
from cooking_plan_agent.llm.client import LLMClient, ToolCall
from cooking_plan_agent.llm.controller import LLMReActController
from cooking_plan_agent.tooling.registry import ToolRegistry
from cooking_plan_agent.workflow.context import WorkflowContext
from cooking_plan_agent.workflow.graph import build_cooking_plan_graph


class _FakeExtractor:
    async def extract(self, source_text: str) -> ExtractedRecipeCandidate:
        return ExtractedRecipeCandidate(
            recipe_id="r1",
            dish_name="Tofu Bowl",
            original_servings=2,
            source_language="en",
            ingredients=(
                ExtractedIngredient(
                    raw_text="tofu 200g",
                    name="tofu",
                    quantity=200,
                    unit="g",
                ),
            ),
            steps=(
                ExtractedStep(
                    step_number=1,
                    instruction="Stir-fry tofu",
                    category="heating",
                    active_duration_minutes=8,
                    heat_level=HeatLevel.MEDIUM,
                    target_temperature_c=Decimal(180),
                    resources_hint=("wok",),
                ),
            ),
        )


def _request() -> GeneratePlanRequest:
    return GeneratePlanRequest(
        request_id="req-llm-agent",
        user_id="u1",
        recipes=({"recipe_id": "r1", "text": "Stir-fry tofu", "target_servings": 2},),
        inventory_lots=(
            InventoryLotSnapshot(
                lot_id="lot-1",
                item_id="i1",
                canonical_name="tofu",
                on_hand=Decimal(300),
                reserved=Decimal(0),
                unit="g",
            ),
        ),
        kitchen_resources=(
            KitchenResourceSnapshot(
                resource_id="stove-1", resource_type="stove", capacity=Decimal(4), capacity_unit="burners"
            ),
            KitchenResourceSnapshot(resource_id="wok-1", resource_type="wok", capacity=Decimal(1)),
            KitchenResourceSnapshot(resource_id="spatula-1", resource_type="spatula", capacity=Decimal(2)),
        ),
    )


def _enable(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("COOKING_PLAN_AGENT_CONTROLLER_ENABLED", "true")
    monkeypatch.setenv("COOKING_PLAN_AGENT_MAX_STEPS", "3")


class _ScriptedLLM:
    """按脚本依次返回 LLM 响应：先 tool_call，再 final。"""

    def __init__(self) -> None:
        self.calls: list[tuple[list[dict[str, str]], list[dict[str, object]]]] = []

    async def chat_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, object]],
    ) -> tuple[str | None, tuple[ToolCall, ...]]:
        self.calls.append((messages, tools))
        if len(self.calls) == 1:
            # 第一轮：调用 parse_recipe 工具（真实执行）。
            return None, (ToolCall(id="c1", name="parse_recipe", arguments={"source_text": "300 g tofu"}),)
        # 第二轮：宣布完成，交回确定性 DAG。
        return '{"status": "READY"}', ()


@pytest.mark.asyncio
async def test_llm_controller_drives_loop_then_falls_back_to_dag(monkeypatch) -> None:
    """LLM 控制器先 tool_call（真实工具执行），再 final → 确定性 DAG 走完 READY。"""
    _enable(monkeypatch)
    scripted = _ScriptedLLM()

    base_ctx = WorkflowContext(recipe_extractor=_FakeExtractor())
    registry = ToolRegistry(base_ctx)
    controller = LLMReActController(  # type: ignore[arg-type]
        LLMClient(base_url="http://llm.test", model="m"),  # type: ignore[arg-type]
        tools=registry.specs(),
    )
    controller._client.chat_with_tools = scripted.chat_with_tools  # type: ignore[method-assign]
    ctx = WorkflowContext(
        recipe_extractor=_FakeExtractor(),
        agent_controller=controller,
    )

    graph = build_cooking_plan_graph()
    result = await graph.ainvoke(
        {"request": _request()},
        context=ctx,
        config={"recursion_limit": 40},
    )

    # LLM 确实被调用了两轮（tool_call → final），工具名传给 LLM。
    assert len(scripted.calls) == 2
    tool_names = {t["function"]["name"] for t in scripted.calls[0][1]}
    assert "parse_recipe" in tool_names
    # 工具真实执行过：观察留痕含 parse_recipe 结果。
    assert any(o.get("ok") is True for o in result.get("observations", ()))
    # final → 回退确定性 DAG → READY。
    response = result.get("response")
    assert isinstance(response, ReadyPlanResponse), f"expected READY, got {type(response).__name__}"
    assert result.get("agent_trace")[-1]["action"] == "final"
