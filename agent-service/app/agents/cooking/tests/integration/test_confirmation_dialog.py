"""P5-4: 确认对话的图级挂起-续接（interrupt + checkpointer + resume）。

验证一次 NEEDS_CONFIRMATION 响应 → 用户 answers → 续接重排 → 终态，
全程同一 thread_id（checkpoint 续接）。
"""

from decimal import Decimal

import pytest
from langgraph.types import Command

from cooking_plan_agent.config.settings import get_settings
from cooking_plan_agent.domain.enums import HeatLevel
from cooking_plan_agent.domain.models import (
    ConfirmationPlanResponse,
    ExtractedIngredient,
    ExtractedRecipeCandidate,
    ExtractedStep,
    GeneratePlanRequest,
    InventoryLotSnapshot,
    KitchenResourceSnapshot,
)
from cooking_plan_agent.infrastructure.checkpointer import MemoryCheckpointProvider
from cooking_plan_agent.workflow.context import WorkflowContext
from cooking_plan_agent.workflow.graph import build_cooking_plan_graph


class _GapExtractor:
    """Candidate 缺烘烤温度 → safety-critical gap → 本地推理无法解决 → 确认。"""

    async def extract(self, source_text: str) -> ExtractedRecipeCandidate:
        return ExtractedRecipeCandidate(
            recipe_id="r1",
            dish_name="Baked Chicken",
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
                    instruction="Bake chicken breast in the oven at medium heat",
                    category="heating",
                    active_duration_minutes=5,
                    passive_duration_minutes=25,
                    heat_level=HeatLevel.MEDIUM,
                    target_temperature_c=None,  # safety-critical temperature gap
                    resources_hint=("oven",),
                ),
            ),
        )


def _request() -> GeneratePlanRequest:
    return GeneratePlanRequest(
        request_id="req-dialog",
        user_id="user-dialog",
        recipes=({"recipe_id": "r1", "text": "Bake chicken breast", "target_servings": 2},),
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
                resource_id="oven-1",
                resource_type="oven",
                capacity=Decimal(1),
            ),
        ),
    )


def _enable_dialog(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("COOKING_PLAN_CONFIRMATION_DIALOG_ENABLED", "true")


def _context() -> WorkflowContext:
    return WorkflowContext(recipe_extractor=_GapExtractor())


@pytest.mark.asyncio
async def test_graph_hangs_on_confirmation_then_resumes(monkeypatch) -> None:
    """首次运行挂起于确认（interrupt），answers 续接后重排（不再 NEEDS_CONFIRMATION）。"""
    _enable_dialog(monkeypatch)
    provider = MemoryCheckpointProvider()
    graph = build_cooking_plan_graph(checkpointer=provider.checkpointer)
    config = {"configurable": {"thread_id": "req-dialog:v1"}, "recursion_limit": 40}

    # 首次：走到 build_confirmation_response → apply_confirmation 挂起。
    first = await graph.ainvoke({"request": _request()}, context=_context(), config=config)
    # LangGraph 挂起语义：返回值携带 __interrupt__ 键，response 为 NEEDS_CONFIRMATION。
    assert "__interrupt__" in first
    assert isinstance(first.get("response"), ConfirmationPlanResponse)
    assert first["response"].status == "NEEDS_CONFIRMATION"
    assert first.get("confirmation_context") is not None

    # 续接：带 answers 恢复同一 thread。answers 为空 → 保持确认（无死循环、无异常）。
    resumed = await graph.ainvoke(Command(resume=()), config=config)
    assert resumed.get("needs_confirmation") is True
    assert resumed.get("confirmation_applied") is False

    # 使用不同线程（模拟用户修正后的新请求）不应串状态（P2-06 隔离）。
    config2 = {"configurable": {"thread_id": "req-dialog:v2"}, "recursion_limit": 40}
    second = await graph.ainvoke({"request": _request()}, context=_context(), config=config2)
    assert second.get("response") is None or second.get("response").status == "NEEDS_CONFIRMATION"


@pytest.mark.asyncio
async def test_graph_dialog_disabled_keeps_terminal_confirmation(monkeypatch) -> None:
    """confirmation_dialog_enabled=false（默认）→ 确认仍是终态（零回归）。"""
    get_settings.cache_clear()
    monkeypatch.setenv("COOKING_PLAN_CONFIRMATION_DIALOG_ENABLED", "false")
    provider = MemoryCheckpointProvider()
    graph = build_cooking_plan_graph(checkpointer=provider.checkpointer)
    config = {"configurable": {"thread_id": "req-dialog:legacy"}, "recursion_limit": 40}

    result = await graph.ainvoke({"request": _request()}, context=_context(), config=config)
    # 原终态：响应就是 ConfirmationPlanResponse，不再需要 apply_confirmation。
    from cooking_plan_agent.domain.models import ConfirmationPlanResponse

    response = result.get("response")
    assert isinstance(response, ConfirmationPlanResponse)
    assert response.status == "NEEDS_CONFIRMATION"


@pytest.mark.asyncio
async def test_graph_dialog_enabled_without_checkpointer_keeps_terminal(monkeypatch) -> None:
    """启用对话但未注入 checkpointer → interrupt 不可用，保持原终态（降级保底）。"""
    _enable_dialog(monkeypatch)
    graph = build_cooking_plan_graph(checkpointer=None)
    result = await graph.ainvoke({"request": _request()}, context=_context())
    from cooking_plan_agent.domain.models import ConfirmationPlanResponse

    response = result.get("response")
    assert isinstance(response, ConfirmationPlanResponse)
    assert response.status == "NEEDS_CONFIRMATION"
