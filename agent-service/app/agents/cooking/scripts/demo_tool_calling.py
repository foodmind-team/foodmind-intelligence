"""P5-1: ToolRunner 分发器演示 —— 两条真实工具链。

链路 1: check_feasibility（真实 check_all_inventory，库存可行性）
链路 2: solve_schedule → verify_schedule（CP-SAT 求解 + 独立验证器复核）

无网络、无外部依赖；每个工具结果以 JSON 打印。

用法（在 cooking 目录）：
    PYTHONPATH=src uv run python scripts/demo_tool_calling.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

# 使 src/ 可导入（脚本可从任意 cwd 启动）
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from cooking_plan_agent.tooling.registry import ToolRegistry  # noqa: E402
from cooking_plan_agent.tooling.runner import ToolRunner  # noqa: E402
from cooking_plan_agent.workflow.context import WorkflowContext  # noqa: E402

# ============================================================================
# 1. 链路 1 输入：1 个 IngredientDemand + 1 个 InventoryLotSnapshot
#    字段与 tests/unit/tooling/test_tools_execution.py 中合法构造一致。
# ============================================================================

FEASIBILITY_ARGS: dict[str, object] = {
    "demands": [
        {
            "canonical_name": "tofu",
            "raw_name": "tofu",
            "quantity": "200",
            "unit": "g",
            "confidence": "0.9",
        }
    ],
    "lots": [
        {
            "lot_id": "lot-1",
            "item_id": "item-1",
            "canonical_name": "tofu",
            "on_hand": "1000",
            "reserved": "0",
            "unit": "g",
        }
    ],
}

# ============================================================================
# 2. 链路 2 输入：1 个极简 CookingTask + 1 个 KitchenResourceSnapshot
# ============================================================================

TASK: dict[str, object] = {
    "task_id": "t1",
    "dish_id": "d1",
    "instruction": "Boil water",
    "duration_minutes": 5,
    "work_mode": "ACTIVE",
    "category": "heating",
}

RESOURCE: dict[str, object] = {
    "resource_id": "stove:1",
    "resource_type": "stove",
    "capacity": "4",
    "capacity_unit": "burners",
    "capabilities": [],
    "available": True,
}

SOLVE_ARGS: dict[str, object] = {
    "tasks": [TASK],
    "resources": [RESOURCE],
    "solver_timeout_seconds": 1.0,
    "optimization_level": "makespan",
}


def _print_chain(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


async def main() -> None:
    # WorkflowContext.recipe_extractor 为必填字段；演示仅用无状态工具
    # （check_feasibility / solve_schedule / verify_schedule 总是注册），
    # 服务字段全部置 None 即可。
    context = WorkflowContext(recipe_extractor=None)  # type: ignore[arg-type]
    registry = ToolRegistry(context)
    runner = ToolRunner(registry)

    # ---- 链路 1: 库存可行性 ----
    _print_chain("链路 1: check_feasibility（真实 check_all_inventory）")
    feasibility = await runner.run("check_feasibility", FEASIBILITY_ARGS)
    print(json.dumps(feasibility, ensure_ascii=False, indent=2))

    # ---- 链路 2: 求解 → 验证 ----
    _print_chain("链路 2: solve_schedule（CP-SAT） → verify_schedule（独立验证）")
    solve = await runner.run("solve_schedule", SOLVE_ARGS)
    print(json.dumps(solve, ensure_ascii=False, indent=2))

    if not solve["ok"]:
        print("\n[!] solve_schedule 失败，跳过 verify_schedule。")
        return

    verify = await runner.run(
        "verify_schedule",
        {
            "problem": {"tasks": [TASK], "resources": [RESOURCE]},
            "result": solve["schedule_result"],
        },
    )
    print(json.dumps(verify, ensure_ascii=False, indent=2))

    # ---- 汇总 ----
    _print_chain("汇总")
    print(f"  check_feasibility : ok={feasibility['ok']}")
    if feasibility["ok"]:
        print(f"    is_feasible      = {feasibility['feasibility_report']['is_feasible']}")
    print(f"  solve_schedule    : ok={solve['ok']}")
    if solve["ok"]:
        print(f"    status           = {solve['schedule_result']['status']}")
        print(f"    makespan_minutes = {solve['schedule_result']['makespan_minutes']}")
    print(f"  verify_schedule   : ok={verify['ok']}")
    if verify["ok"]:
        print(f"    passed           = {verify['verification_report']['passed']}")


if __name__ == "__main__":
    asyncio.run(main())
