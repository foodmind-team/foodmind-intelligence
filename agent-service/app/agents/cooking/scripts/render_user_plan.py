"""Render a readable cooking plan (做饭计划) for the 7 golden recipes.

Re-uses the same composition root as demo_comprehensive_workflow.py (offline
deterministic: rule-based extractor + CP-SAT scheduling, no LLM needed).

Usage:
    cd agent-service/app/agents/cooking
    PYTHONPATH=src uv run python scripts/render_user_plan.py
"""

from __future__ import annotations

import asyncio
import sys

sys.path.insert(0, "scripts")

from demo_comprehensive_workflow import (  # noqa: E402
    RECIPES,
    build_lots_from_candidates,
    make_request,
    run_one,
)

from cooking_plan_agent.parsing.extractor import RecipeExtractor  # noqa: E402
from cooking_plan_agent.workflow.context import WorkflowContext  # noqa: E402
from cooking_plan_agent.workflow.graph import build_cooking_plan_graph  # noqa: E402


async def main() -> None:
    extractor = RecipeExtractor()
    ctx = WorkflowContext(recipe_extractor=extractor)
    graph = build_cooking_plan_graph()

    prepared = [{"name": r["name"], "zh": r["zh"], "text": f"{r['zh']}\n{r['text']}"} for r in RECIPES]
    cands = await asyncio.gather(*(extractor.extract(r["text"]) for r in prepared))
    lots = build_lots_from_candidates(cands)
    dish_ids = {f"r{i}": m["name"] for i, m in enumerate(prepared)}

    request = make_request("user-7dish", tuple(prepared), lots)
    final, elapsed, _ = await run_one(graph, ctx, request, "7-dish meal plan", verbose=False)
    resp = final.get("response")
    if resp is None:
        print("NO RESPONSE, error:", final.get("error_code"))
        return

    print("=" * 74)
    print(f"做饭计划 · status={resp.status} · solver={resp.solver_status} · 总时长={resp.makespan_minutes}min")
    print("=" * 74)

    for e in sorted(resp.timeline, key=lambda x: int(x["start_minute"])):
        dish = dish_ids.get(str(e["dish_id"]), str(e["dish_id"]))
        wm = str(e["work_mode"])[:7]
        print(f"  t={int(e['start_minute']):>3}-{int(e['end_minute']):<3} [{wm:<7}] {e['instruction'][:40]:<42} ({dish})")

    print("-" * 74)
    print("各道菜完成时间:")
    for c in sorted(resp.dish_completions, key=lambda x: int(x["completion_minute"])):
        print(f"  · {dish_ids.get(str(c['dish_id']), str(c['dish_id'])):<22} 第 {int(c['completion_minute'])} 分钟")

    print("-" * 74)
    print(f"食材备料清单 ({len(resp.completion_checklist)} 项):")
    for i, item in enumerate(resp.completion_checklist, 1):
        alloc = ", ".join(f"{a.quantity}{a.unit}" for a in item.allocations)
        print(f"  {i:>2}. {item.ingredient_name} (用量 {alloc} · 覆盖 {len(item.recipe_ids)} 道菜)")
    print(f"mise_en_place: {len(resp.mise_en_place)} 项 · wall={elapsed:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
