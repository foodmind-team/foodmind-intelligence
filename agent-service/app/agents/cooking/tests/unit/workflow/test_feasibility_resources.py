"""Unit tests for the feasibility resource pre-check (workflow/safety_nodes.py).

Regression: LLM 抽取的 resources_hint 是自由文本软提示（如 剪刀/厨房纸/电饭煲/
碗/锅/锅盖），不能因为它们不在厨房快照中就判定计划不可行 —— 一个有刀就能做的
操作必须可行。只有归一化后属于「关键器材类型」的 hint 才参与可行性 gating。
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from cooking_plan_agent.domain.enums import HeatLevel
from cooking_plan_agent.domain.models import (
    GeneratePlanRequest,
    IngredientDemand,
    InventoryLotSnapshot,
    KitchenResourceSnapshot,
    RecipeIR,
    RecipeStep,
)
from cooking_plan_agent.workflow.safety_nodes import check_feasibility_node


class _RuntimeStub:
    """check_feasibility_node 不使用 runtime —— 仅需满足签名。"""

    @property
    def context(self):
        return None


def _ir(steps: tuple[RecipeStep, ...]) -> RecipeIR:
    return RecipeIR(
        recipe_id="r1",
        dish_name="Test Dish",
        original_servings=Decimal(2),
        target_servings=Decimal(2),
        source_language="zho",
        ingredients=(
            IngredientDemand(
                canonical_name="shrimp",
                raw_name="虾",
                quantity=Decimal(200),
                unit="g",
                confidence=Decimal("1.0"),
            ),
        ),
        steps=steps,
    )


def _step(hints: tuple[str, ...], category: str = "preparation") -> RecipeStep:
    return RecipeStep(
        step_number=1,
        instruction="处理食材",
        category=category,
        heat_level=HeatLevel.NONE,
        resources_hint=hints,
    )


def _request(kitchen: tuple[KitchenResourceSnapshot, ...]) -> GeneratePlanRequest:
    return GeneratePlanRequest(
        request_id="req-feasibility-res",
        user_id="u",
        recipes=({"recipe_id": "r1", "text": "test", "target_servings": 2},),
        inventory_lots=(
            InventoryLotSnapshot(
                lot_id="lot-1",
                item_id="shrimp",
                canonical_name="shrimp",
                on_hand=Decimal(500),
                reserved=Decimal(0),
                unit="g",
            ),
        ),
        kitchen_resources=kitchen,
    )


def _snapshot(resource_type: str) -> KitchenResourceSnapshot:
    return KitchenResourceSnapshot(
        resource_id=f"{resource_type}-1",
        resource_type=resource_type,
        capacity=Decimal(1),
    )


@pytest.mark.asyncio
async def test_soft_llm_hints_do_not_block_knife_only_operation():
    """场景 A 回归：LLM hint 含 剪刀/厨房纸/电饭煲/碗/锅/锅盖，厨房快照只有常规
    器材（刀/灶/锅…）—— 操作有刀即可做，不可再判为不可行。"""
    kitchen = (
        _snapshot("knife"),
        _snapshot("stove"),
        _snapshot("pot"),
        _snapshot("cutting_board"),
        _snapshot("sink"),
    )
    state = {
        "request": _request(kitchen),
        "parsed_recipes": (
            _ir((_step(("剪刀", "厨房纸", "电饭煲", "碗", "锅", "锅盖")),)),
        ),
    }

    result = await check_feasibility_node(state, _RuntimeStub())
    report = result["feasibility_report"]
    assert report.is_feasible is True, f"软性 hint 不应阻断可行性: {report.missing_resources}"
    assert report.missing_resources == ()


@pytest.mark.asyncio
async def test_essential_missing_equipment_still_blocks():
    """回归：真正缺关键器材（烤箱）仍须判为不可行并给出替代方案。"""
    state = {
        "request": _request((_snapshot("knife"),)),
        "parsed_recipes": (_ir((_step(("oven", "剪刀", "厨房纸")),)),),
    }

    result = await check_feasibility_node(state, _RuntimeStub())
    report = result["feasibility_report"]
    assert report.is_feasible is False
    assert report.missing_resources == ("oven",)
    assert any(o.option_type == "alternative_equipment" for o in result["repair_options"])


@pytest.mark.asyncio
async def test_capability_suffix_matches_base_type():
    """'锅:炒锅' 等带能力后缀的 hint 归一到 pot 后命中厨房快照。"""
    kitchen = (_snapshot("pot"),)
    state = {
        "request": _request(kitchen),
        "parsed_recipes": (_ir((_step(("锅",)),)),),
    }

    result = await check_feasibility_node(state, _RuntimeStub())
    assert result["feasibility_report"].is_feasible is True


@pytest.mark.asyncio
async def test_mixed_essential_and_soft_hints_report_canonical_missing():
    """混合场景：只上报缺失的关键器材（canonical 类型），软性 hint 不上报。"""
    state = {
        "request": _request((_snapshot("knife"),)),
        "parsed_recipes": (_ir((_step(("oven", "碗", "锅盖")),)),),
    }

    result = await check_feasibility_node(state, _RuntimeStub())
    report = result["feasibility_report"]
    assert report.missing_resources == ("oven",)
    assert "碗" not in report.missing_resources
    assert "锅盖" not in report.missing_resources
