"""E2E：用菜单 fixtures 直连 workflow graph 验证「关键项补全 + 做饭计划生成」。

重点（对应设计预期 3/4）：
  1. 关键项补全 —— 对 7 道菜谱原文做 提取 → gap 检测 → 本地推理补全，
     展示每道菜缺失的关键项（火力/时长/温度）及补全依据（assumptions）。
  2. 做饭计划生成 —— 库存/设备用 mock 且保证充足，跑完整 workflow，
     输出 READY 计划的 makespan、工序时间线、mise en place、各菜完成时间。

用法（在 cooking 目录）：
    COOKING_PLAN_INTERNAL_SERVICE_TOKEN=test uv run python scripts/e2e_menu_plan.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections import Counter
from decimal import Decimal

# 使 scripts/ 与 src/ 可导入（本脚本可能从任意 cwd 启动）
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)

os.environ.setdefault("COOKING_PLAN_INTERNAL_SERVICE_TOKEN", "test-token")
os.environ.setdefault("COOKING_PLAN_LLM_ENABLED", "false")
# 本菜单 7 道菜；默认上限 6，测试脚本放宽到 8
os.environ.setdefault("COOKING_PLAN_MAX_RECIPE_COUNT", "8")

from menu_fixtures import (  # noqa: E402
    MENU_RECIPES,
    MOCK_INVENTORY,
    MOCK_KITCHEN_RESOURCES,
)

from cooking_plan_agent.domain.models import (  # noqa: E402
    ConfirmationPlanResponse,
    ExtractedRecipeCandidate,
    GeneratePlanRequest,
    InfeasiblePlanResponse,
    InventoryLotSnapshot,
    KitchenResourceSnapshot,
    ReadyPlanResponse,
)
from cooking_plan_agent.parsing.extractor import RecipeExtractor  # noqa: E402
from cooking_plan_agent.parsing.gaps import find_recipe_gaps  # noqa: E402
from cooking_plan_agent.parsing.inference import infer_local  # noqa: E402
from cooking_plan_agent.workflow.context import WorkflowContext  # noqa: E402
from cooking_plan_agent.workflow.graph import build_cooking_plan_graph  # noqa: E402

GAP_CLASS_LABEL = {
    "critical": "关键缺口",
    "safety_critical": "安全关键缺口",
    "resource_critical": "设备关键缺口",
    "optimisation": "优化项",
    "cosmetic": "外观项",
}


# ============================================================================
# 1. 提取 + 关键项补全（逐菜谱展示）
# ============================================================================


async def extract_candidates() -> list[ExtractedRecipeCandidate]:
    extractor = RecipeExtractor()
    candidates: list[ExtractedRecipeCandidate] = []
    for recipe in MENU_RECIPES:
        candidate = await extractor.extract(recipe["text"])
        candidates.append(candidate)
    return candidates


def show_gap_filling(candidates: list[ExtractedRecipeCandidate]) -> None:
    """展示每道菜的 gap 检测与本地推理补全结果。"""
    print("=" * 78)
    print("第 1 部分｜关键项补全（提取 → 缺口检测 → 本地推理填补）")
    print("=" * 78)
    for candidate in candidates:
        gaps = find_recipe_gaps(candidate)
        result = infer_local(candidate, gaps)
        print(f"\n--- 菜谱 {candidate.recipe_id}: {candidate.dish_name} "
              f"(食材 {len(candidate.ingredients)} 项 / 步骤 {len(candidate.steps)} 步) ---")
        if not gaps:
            print("  ✓ 无关键项缺口")
            continue
        print(f"  检测到 {len(gaps)} 个缺口:")
        for g in gaps:
            print(f"    · [{GAP_CLASS_LABEL.get(g.gap_class, g.gap_class)}] "
                  f"{g.field_path}: {g.description}")
        if result.unresolved_gaps:
            print(f"  未能本地补全 {len(result.unresolved_gaps)} 个关键缺口:")
            for g in result.unresolved_gaps:
                print(f"    ✗ [{g.gap_class}] {g.field_path}: {g.description}")
        if result.filled_gaps:
            print(f"  本地推理补全 {len(result.filled_gaps)} 个缺口")
        for a in result.assumptions:
            print(f"    补全依据(置信度 {a.confidence}): {a.text}")


# ============================================================================
# 2. 构造「保证充足」的 mock 库存（基准 + 自动补足）
# ============================================================================


def build_abundant_inventory(
    candidates: list[ExtractedRecipeCandidate],
    base: list[dict],
) -> tuple[InventoryLotSnapshot, ...]:
    """以 MOCK_INVENTORY 为基准，按 graph 真实使用的缩放后需求自动补足，
    保证每个需求资源充足（仅用于测试）。

    需求口径与 check_feasibility 完全一致：走 build_recipe_ir（含
    target_servings 缩放），因此补足后必然不会触发库存短缺。
    """
    from cooking_plan_agent.domain.models import IngredientDemand
    from cooking_plan_agent.parsing.ir_builder import build_recipe_ir

    lots = [InventoryLotSnapshot.model_validate(lot) for lot in base]
    covered: dict[str, Decimal] = Counter()
    for lot in lots:
        covered[lot.canonical_name.lower().strip()] += lot.on_hand - lot.reserved

    # 按 canonical_name 聚合缩放后的需求（与 feasibility 相同口径）
    demands: dict[str, IngredientDemand] = {}
    for candidate, recipe in zip(candidates, MENU_RECIPES, strict=True):
        try:
            ir = build_recipe_ir(
                candidate,
                request_recipe_id=recipe["recipe_id"],
                target_servings=Decimal(str(recipe["target_servings"])),
            )
        except (ValueError, TypeError):
            continue  # 单菜 IR 构建失败不阻塞整体测试
        for demand in ir.ingredients:
            key = demand.canonical_name.lower().strip()
            prev = demands.get(key)
            if prev is not None:
                demands[key] = prev.model_copy(update={"quantity": prev.quantity + demand.quantity})
            else:
                demands[key] = demand

    extra: list[InventoryLotSnapshot] = []
    idx = 1000
    for key, demand in demands.items():
        available = covered.get(key, Decimal(0))
        if available >= demand.quantity:
            continue
        # 补足到需求的 10 倍，兜底 +100（单位不变）
        top_up = demand.quantity * 10 + Decimal(100)
        extra.append(
            InventoryLotSnapshot(
                lot_id=f"auto-{idx}",
                item_id=f"auto-item-{idx}",
                canonical_name=demand.canonical_name,
                on_hand=top_up,
                reserved=Decimal(0),
                unit=demand.unit,
            )
        )
        idx += 1
    if extra:
        print(f"\n  自动补足库存 {len(extra)} 项（仅测试用）: "
              f"{', '.join(lot.canonical_name for lot in extra[:8])}{' …' if len(extra) > 8 else ''}")
    return tuple(lots) + tuple(extra)


# ============================================================================
# 3. 运行完整 workflow
# ============================================================================


def build_request(
    candidates: list[ExtractedRecipeCandidate],
    inventory: tuple[InventoryLotSnapshot, ...],
) -> GeneratePlanRequest:
    return GeneratePlanRequest(
        request_id="e2e-menu-2026-08-02",
        user_id="test-user",
        recipes=tuple(
            {
                "recipe_id": r["recipe_id"],
                "text": r["text"],
                "target_servings": r["target_servings"],
            }
            for r in MENU_RECIPES
        ),
        inventory_lots=inventory,
        kitchen_resources=tuple(KitchenResourceSnapshot.model_validate(r) for r in MOCK_KITCHEN_RESOURCES),
        preparsed_candidates=tuple(candidates),  # 复用脚本提取结果，保证库存与需求一一对应
    )


def show_ready(response: ReadyPlanResponse) -> None:
    print("\n" + "=" * 78)
    print("第 3 部分｜做饭计划生成（READY）")
    print("=" * 78)
    print(f"状态: READY | 求解器: {response.solver_status} | 总耗时: {response.makespan_minutes} 分钟")
    print(f"解释: {response.explanation or '(未启用)'}")

    print(f"\n工序时间线（食材处理顺序 / 烹饪时长 / 工序安排，共 {len(response.timeline)} 项）:")
    for entry in response.timeline:
        mode = {"ACTIVE": "操作", "PASSIVE": "等待"}.get(str(entry.get("work_mode")), entry.get("work_mode"))
        print(
            f"  [{int(entry['start_minute']):>4}–{int(entry['end_minute']):>4}分钟] "
            f"{str(entry.get('dish_id')):<8} {mode:<4} {str(entry.get('instruction'))[:56]}"
        )

    print(f"\n各菜完成时间（共 {len(response.dish_completions)} 道）:")
    for dish in response.dish_completions:
        print(f"  {dish.get('dish_id', '?')}: 第 {dish.get('completion_minute')} 分钟完成")

    print(f"\nMise en place 备料清单（共 {len(response.mise_en_place)} 项）:")
    for item in response.mise_en_place:
        print(f"  · {item.get('instruction')}（{item.get('duration_minutes')} 分钟）")

    print(f"\n库存消耗分配 completion_checklist（共 {len(response.completion_checklist)} 项）:")
    for item in response.completion_checklist:
        print(f"  · {item.ingredient_name}: " + ", ".join(
            f"{a.quantity}{a.unit} @ {a.inventory_lot_id}" for a in item.allocations
        ))


def show_confirmation(response: ConfirmationPlanResponse) -> None:
    print("\n" + "=" * 78)
    print("第 3 部分｜做饭计划生成（NEEDS_CONFIRMATION —— 需要用户确认）")
    print("=" * 78)
    if response.assumptions:
        print(f"补全假设（{len(response.assumptions)} 条）:")
        for a in response.assumptions:
            print(f"  · [{a.confidence}] {a.text}")
    if response.repair_options:
        print(f"修复选项（{len(response.repair_options)} 个）:")
        for opt in response.repair_options:
            print(f"  · [{opt.option_type}] {opt.description}")
    if response.decisions:
        print(f"可回传决策（{len(response.decisions)} 个）:")
        for d in response.decisions:
            print(f"  · {d.option_type} @ {d.option_id} payload={d.payload}")
    print("结构化问题单:")
    for q in response.confirmation_questions:
        print(f"  · {q.question_id}: {q.prompt}")


async def main() -> None:
    print(f"菜单共 {len(MENU_RECIPES)} 道菜谱\n")

    # 1) 提取 + 关键项补全
    candidates = await extract_candidates()
    show_gap_filling(candidates)

    # 2) 充足 mock 库存
    print("\n" + "=" * 78)
    print("第 2 部分｜mock 库存（保证充足，仅用于测试）")
    print("=" * 78)
    inventory = build_abundant_inventory(candidates, MOCK_INVENTORY)

    # 3) 完整 workflow
    graph = build_cooking_plan_graph()
    context = WorkflowContext(recipe_extractor=None)  # preparsed_candidates 已注入，跳过 LLM/规则提取
    request = build_request(candidates, inventory)
    result = await graph.ainvoke({"request": request}, context=context, config={"recursion_limit": 40})
    response = result.get("response")

    if isinstance(response, ReadyPlanResponse):
        show_ready(response)
    elif isinstance(response, ConfirmationPlanResponse):
        show_confirmation(response)
    elif isinstance(response, InfeasiblePlanResponse):
        print("\nINFEASIBLE:", response.reasons)
    else:
        print("\nFAILED:", response)


if __name__ == "__main__":
    asyncio.run(main())
