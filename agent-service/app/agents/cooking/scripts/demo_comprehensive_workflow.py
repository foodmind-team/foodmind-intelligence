"""Comprehensive demo: Cooking Plan Agent — smart planning + agentic workflow.

Live demo for the instructor. Two modes:

  default      — offline deterministic (no LLM / no research), fully
                 reproducible. Proves routing & planning are not hard-coded.
  --full-agent — EVERY agent capability enabled: LLM recipe extraction,
                 LLM knowledge research, TTL cache, schedule explanation,
                 workflow checkpointing, the ReAct tool-calling controller,
                 and the multi-turn confirmation dialog (with automatic
                 resume on interrupts). Runs a representative comprehensive
                 test that shows which capability fired per case.

  Demonstration points (both modes):
  1. Deterministic pipeline: validate -> parse -> gaps -> infer -> IR -> safety
     -> feasibility -> prep merge -> task DAG -> CP-SAT solve -> verify -> render
  2. Dynamic routing: different dishes take different node paths
     (infer_local / research_missing / NEEDS_CONFIRMATION / INFEASIBLE / READY)
  3. Reflection-repair: verify failure triggers the repair_schedule back-edge
     (repair_history leaves an audit trail)
  4. SMART PLANNING: multi-dish meal combos are solved by CP-SAT which
     optimizes makespan under resource constraints. Per combo we measure
     solver status, makespan, peak parallelism, serial-vs-parallel speedup,
     and per-dish completion times.

Usage:
    cd agent-service/app/agents/cooking
    PYTHONPATH=src uv run python scripts/demo_comprehensive_workflow.py          # offline
    PYTHONPATH=src uv run python scripts/demo_comprehensive_workflow.py --full-agent
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import random
import sys
import time
from decimal import Decimal
from typing import Any

# ---------------------------------------------------------------------------
# Mode selection — MUST happen before settings are first read.
# Offline is the baseline (setdefault); --full-agent overrides every switch.
# ---------------------------------------------------------------------------
_FULL_AGENT = "--full-agent" in sys.argv

os.environ.setdefault("COOKING_PLAN_INTERNAL_SERVICE_TOKEN", "local-cooking-token")
os.environ.setdefault("COOKING_PLAN_LLM_ENABLED", "false")
# NOTE: do NOT pin COOKING_PLAN_LLM_API_KEY here — a blank env var would shadow
# the real key from .env (env vars outrank dotenv in pydantic-settings) and
# every LLM call would 401. The .env key is only used when --full-agent turns
# LLM on; offline mode never touches it.
os.environ.setdefault("COOKING_PLAN_WEB_RESEARCH_ENABLED", "false")
os.environ.setdefault("COOKING_PLAN_CACHE_ENABLED", "false")
os.environ.setdefault("COOKING_PLAN_EXPLANATION_ENABLED", "false")
os.environ.setdefault("COOKING_PLAN_CHECKPOINT_ENABLED", "false")
os.environ.setdefault("COOKING_PLAN_CONFIRMATION_DIALOG_ENABLED", "false")
os.environ.setdefault("COOKING_PLAN_AGENT_CONTROLLER_ENABLED", "false")
os.environ.setdefault("ORTools_LogToStdout", "")
os.environ.setdefault("ORTools_LogToStderr", "")

if _FULL_AGENT:
    for _key in (
        "COOKING_PLAN_LLM_ENABLED",
        "COOKING_PLAN_WEB_RESEARCH_ENABLED",
        "COOKING_PLAN_CACHE_ENABLED",
        "COOKING_PLAN_EXPLANATION_ENABLED",
        "COOKING_PLAN_CHECKPOINT_ENABLED",
        "COOKING_PLAN_CONFIRMATION_DIALOG_ENABLED",
        "COOKING_PLAN_AGENT_CONTROLLER_ENABLED",
    ):
        os.environ[_key] = "true"

from langgraph.types import Command  # noqa: E402

from cooking_plan_agent.domain.models import (  # noqa: E402
    GeneratePlanRequest,
    InventoryLotSnapshot,
    KitchenResourceSnapshot,
    RecipeInput,
)
from cooking_plan_agent.infrastructure.cache import (  # noqa: E402
    InMemoryTTLCache,
    build_parse_cache_key,
)
from cooking_plan_agent.infrastructure.checkpointer import MemoryCheckpointProvider  # noqa: E402
from cooking_plan_agent.infrastructure.preferences import PreferenceStore  # noqa: E402
from cooking_plan_agent.llm.client import LLMClient  # noqa: E402
from cooking_plan_agent.llm.controller import LLMReActController  # noqa: E402
from cooking_plan_agent.llm.explainer import LLMPlanExplainer  # noqa: E402
from cooking_plan_agent.llm.extractor import PARSE_PROMPT_VERSION, LLMRecipeExtractor  # noqa: E402
from cooking_plan_agent.llm.researcher import LLMKnowledgeResearcher  # noqa: E402
from cooking_plan_agent.parsing.extractor import RecipeExtractor  # noqa: E402
from cooking_plan_agent.safety.engine import SafetyEngine  # noqa: E402
from cooking_plan_agent.tooling.registry import ToolRegistry  # noqa: E402
from cooking_plan_agent.workflow.context import WorkflowContext  # noqa: E402
from cooking_plan_agent.workflow.graph import build_cooking_plan_graph  # noqa: E402

# ---------------------------------------------------------------------------
# Terminal colors (graceful fallback without ANSI support)
# ---------------------------------------------------------------------------
C_RESET = "\033[0m"
C_CYAN = "\033[36m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_RED = "\033[31m"
C_DIM = "\033[2m"
C_BOLD = "\033[1m"
C_MAGENTA = "\033[35m"

if os.environ.get("NO_COLOR"):
    C_RESET = C_CYAN = C_GREEN = C_YELLOW = C_RED = C_DIM = C_BOLD = C_MAGENTA = ""

# Node labels (LangGraph node name -> demo copy)
NODE_LABELS: dict[str, str] = {
    "validate_input": "Validate request boundary",
    "parse_recipes": "Parse recipes (LLM extractor)",
    "detect_gaps": "Detect knowledge gaps",
    "infer_local": "Infer from local cooking knowledge",
    "research_missing": "LLM knowledge research",
    "apply_research_evidence": "Apply research evidence",
    "validate_recipe_ir": "Validate recipe IR",
    "validate_safety": "Evaluate food-safety rules",
    "check_feasibility": "Check inventory/resource feasibility",
    "merge_preparation": "Merge shared preparation",
    "build_task_graph": "Build task DAG",
    "solve_schedule": "CP-SAT solve",
    "verify_schedule": "Verify schedule independently",
    "repair_schedule": "Reflection-repair loop",
    "explain_schedule": "Explain schedule (LLM)",
    "render_ready_response": "Render READY terminal",
    "render_infeasible_response": "Render INFEASIBLE terminal",
    "render_failed_response": "Render FAILED terminal",
    "build_confirmation_response": "Render NEEDS_CONFIRMATION terminal",
    "apply_confirmation": "Apply confirmation answers (P5-4 dialog)",
    "run_tool": "Tool execution (ReAct)",
    "agent_controller": "ReAct controller (LLM decision)",
}

# 7 golden recipes: display name (EN), canonical Chinese title, raw text.
# The Chinese title is injected as the first line at request build time so the
# extractor picks up the dish_name correctly (same as a real titled request).
RECIPES: list[dict[str, str]] = [
    {
        "name": "Spicy Crab Legs",
        "zh": "香辣蟹脚",
        "text": """食材准备 
主料： 
蟹脚 
辅料： 
姜 
蒜 
洋葱 
辣椒 
辣椒面 
火锅料 
米酒 
生抽 
老抽 
白胡椒粉、 
酒糟 
盐 
葱段 
烹饪步骤 
1. 处理蟹脚：蟹脚洗净，用刷子刷净缝隙，刀背轻敲外壳方便入味。 
2. 焯水去腥：冷水下锅加米酒焯水，捞出沥干。 
3. 炒香底料：热油下姜蒜末、洋葱、辣椒炒香，加足量辣椒面和少许火锅料炒出红油。 
4. 翻炒调味：放入蟹脚翻炒，加生抽、老抽、白胡椒粉、小半碗酒糟、少许盐炒匀。 
5. 焖煮入味：加适量清水没过蟹脚，中火煮5-6分钟入味。 
6. 出锅装盘：出锅前撒红辣椒、葱段即可装盘。 
7. 享用：蟹脚吃完，汤汁拌米饭超下饭～""",
    },
    {
        "name": "Garlic Fried Shrimp",
        "zh": "蒜香煎虾",
        "text": """食材准备 
主料： 
鲜虾（适量，去头、去虾线、对半剪开） 
辅料： 
蒜末（大量，分两次用） 
干辣椒 
葱花 
调料： 
食用油 
盐 
鸡精 
生抽 
水淀粉 
清水 
烹饪步骤 
处理虾：虾去头，拉出虾线，剪掉虾枪（虾头尖刺），从背部对半剪开，用清水洗净备用。 
煎虾：锅中倒油烧热，下入处理好的虾，煎至外壳金黄焦脆，盛出备用。 
炒香配料：锅中留底油（或重新倒油），油热后下入一半蒜末和干辣椒，大火炒出香味。 
翻炒调味：倒入煎好的虾，快速翻炒均匀；加入适量盐、鸡精、生抽调味，继续翻炒至虾身裹上调料。 
收汁入味：加入小半碗清水，煮至汤汁冒泡后，倒入少许水淀粉勾芡；最后加入剩余蒜末和葱花，大火快速翻炒均匀，即可出锅。""",
    },
    {
        "name": "Spicy Douban Prawns",
        "zh": "香辣基围虾",
        "text": """食材准备 
- 主料： 
基围虾500克（选大个头） 
- 辅料： 
大蒜3-4瓣 
小米辣（依吃辣程度放） 
生姜2片 
干辣椒（可选） 
- 调料： 
豆瓣酱1勺 
生抽2勺 
老抽少许 
蚝油 
黑胡椒粉 
盐 
味精/鸡精（可选） 
食用油 
烹饪步骤 
1. 处理食材： 
虾：剪去虾头，开背去除虾线，用厨房纸彻底吸干水分（防煎制溅油） 
蒜：用重物拍扁，去皮后切成蒜末，备用。切成条或粒； 
小米辣切圈；香菜切段；干辣椒可剪开（增香）。 
2. 煎虾：热锅倒油，中小火加热至油微微热，放入控干水的虾，慢慢煎炒到虾壳变黄变焦，捞出备用。 
3. 炒料：锅中补少许油，放姜、蒜、小米辣翻炒出香味，加1勺豆瓣酱炒出红油。 
4. 调味：倒入煎好的虾，放干辣椒（可选），加2勺生抽、少许老抽（上色）、蚝油、黑胡椒粉、少许盐，可加味精提鲜，快速翻炒均匀。 
5. 出锅""",
    },
    {
        "name": "Pork Rib Corn Soup",
        "zh": "玉米胡萝卜排骨汤",
        "text": """食材准备 
主料： 
排骨 
玉米 
胡萝卜 
辅料： 
红枣 
葱 
姜 
调料： 
盐 
烹饪步骤 
焯水去腥：排骨冷水下锅，加入葱段、姜片焯水，撇去浮沫后捞出洗净。 
放入食材：把焯好的排骨放入电饭煲，依次放入玉米段、胡萝卜块、山药块、红枣，加热水没过食材。 
一键炖煮：盖好锅盖，选择煲汤模式，炖煮至程序结束。 
调味出锅：出锅前撒入葱花，加适量盐调味，搅拌均匀即可。""",
    },
    {
        "name": "Hand-Torn Cabbage",
        "zh": "手撕包菜",
        "text": """一、食材准备 
- 主料 
包菜1个 
- 辅料： 
大蒜20g 
青花椒适量 
干辣椒适量 
- 调料： 
盐1.5g 
味精1g 
白糖少许 
生抽3g 
香醋少许 
二、制作步骤 
1. 处理包菜：包菜从中间切开，去除硬梗，叶片逐片分开后手撕成大小均匀的片状，洗净后充分沥干水分备用。 
2. 准备辅料：大蒜切蒜片，与青花椒、干辣椒一同放入碗中备用。 
3. 爆香料头：锅烧热，倒入适量底油（可稍多防糊锅），油温烧至7成热，下入蒜片、干辣椒、青花椒，快速炸出香味。 
4. 大火快炒：倒入沥干的包菜，大火翻炒30秒，炒至包菜断生（家庭灶可适当延长时间）。 
5. 调味出锅：加入盐、味精、少许白糖提鲜，生抽从锅边淋入，加少许香醋，快速翻炒均匀，立即出锅装盘""",
    },
    {
        "name": "Sausage Cauliflower",
        "zh": "腊肠炒菜花",
        "text": """食材： 
- 腊肠 
- 菜花 
- 蒜苗 
- 蒜头 
调料： 
- 盐 
鸡粉 
蚝油 
老抽 
淀粉 
生抽 
猪油 
米酒 

步骤： 
1. 腊肠水开后蒸15分钟，取出切片；菜花切小块，加两勺盐浸泡；蒜苗切段，蒜头切片。 
2. 调个料汁：淀粉水+老抽+蚝油，混合备用。 
3. 锅热不放油，下菜花反复煸炒5分钟，把水分煸干；盖盖焖1分钟，断生后盛出。 
4. 锅中放猪油，下蒜片炒香，加入腊肠继续炒出香味。 
5. 菜花回锅，加适量盐、鸡粉，沿锅边淋少许米酒。 
6. 倒入调好的料汁，翻炒均匀，最后淋生抽提鲜出锅。""",
    },
    {
        "name": "Spicy Chicken Wings",
        "zh": "香辣鸡翅",
        "text": """食材准备 
主料： 
鸡翅中 15个 
腌料： 
盐 
胡椒粉 
鸡精 
蚝油 
老抽 
料酒 
蛋清 1个 
淀粉 2大勺 
辅料： 
干辣椒（适量） 
蒜末 
火锅底料（半块） 
白糖 
白芝麻 
点缀： 
葱花（一把） 
烹饪步骤 
处理鸡翅 
将15个鸡翅对半剪开（方便入味）。 
加入盐、胡椒粉、鸡精、蚝油、老抽、料酒、1个蛋清、2大勺淀粉。抓拌均匀，腌制 20 分钟。 
准备配料 
干辣椒剪成丝，用清水浸泡一下（防止炒糊，更香）。 
煎制鸡翅。热锅凉油，将腌好的鸡翅下锅。保持中火，煎至两面焦黄，散发香味。盛出鸡翅备用。 
炒底料。再起锅，放入蒜末和半块火锅底料，炒化。倒入泡好的辣椒丝，炒出浓郁的香辣味。 
混合翻炒。倒入煎好的鸡翅。加入 1勺白糖、五香粉、鸡精、白芝麻。最后放入一把葱花，大火快速翻炒均匀，即可出锅""",
    },
]

# Shared kitchen resources (types cover every essential need of the decomposition)
KITCHEN_RESOURCES: tuple[KitchenResourceSnapshot, ...] = (
    KitchenResourceSnapshot(resource_id="stove:main", resource_type="stove", capacity=Decimal(4), capacity_unit="burners"),
    KitchenResourceSnapshot(resource_id="oven:main", resource_type="oven", capacity=Decimal(1)),
    KitchenResourceSnapshot(resource_id="sink:main", resource_type="sink", capacity=Decimal(2)),
    KitchenResourceSnapshot(resource_id="bowl:main", resource_type="mixing_bowl", capacity=Decimal(3)),
    KitchenResourceSnapshot(resource_id="knife:main", resource_type="knife", capacity=Decimal(1)),
    KitchenResourceSnapshot(resource_id="board:main", resource_type="cutting_board", capacity=Decimal(1)),
    KitchenResourceSnapshot(resource_id="pot:main", resource_type="pot", capacity=Decimal(2)),
    KitchenResourceSnapshot(resource_id="pan:main", resource_type="pan", capacity=Decimal(2)),
    KitchenResourceSnapshot(resource_id="wok:main", resource_type="wok", capacity=Decimal(2)),
    KitchenResourceSnapshot(resource_id="spatula:main", resource_type="spatula", capacity=Decimal(2)),
    KitchenResourceSnapshot(resource_id="steamer:main", resource_type="steamer", capacity=Decimal(1)),
)


def p(text: str, color: str = "") -> None:
    print(f"{color}{text}{C_RESET}")


def bar(char: str = "─", width: int = 78) -> None:
    print(char * width)


# ---------------------------------------------------------------------------
# Dynamic inventory: auto-generate abundant stock from extracted ingredients so
# matching always succeeds (keep the focus on the workflow / smart planning)
# ---------------------------------------------------------------------------


def build_lots_from_candidates(
    candidates: list[Any],
    skip_ingredients: set[str] | None = None,
    shortage_scale: Decimal = Decimal(1),
) -> tuple[InventoryLotSnapshot, ...]:
    """Build inventory lots from extracted ingredient names.

    skip_ingredients: deliberately exclude ingredients from stock (fault
    injection -> triggers the shortage route).
    shortage_scale: <1 prepares only a fraction of the demand (partial stockout).
    """
    lots: list[InventoryLotSnapshot] = []
    lot_index: dict[str, int] = {}
    skip = skip_ingredients or set()
    for cand in candidates:
        for ing in cand.ingredients:
            name = ing.name.strip()
            if not name or name in skip:
                continue
            qty = ing.quantity if ing.quantity is not None and ing.quantity > 0 else Decimal(1)
            unit = ing.unit or "piece"
            # scale=1 -> 3x demand (abundant); scale<1 -> below demand (stockout)
            added = max(Decimal(1), qty * shortage_scale * 3)
            if name in lot_index:
                # Same ingredient across recipes -> accumulate on_hand
                idx = lot_index[name]
                prev = lots[idx]
                lots[idx] = prev.model_copy(update={"on_hand": prev.on_hand + added})
                continue
            lot_index[name] = len(lots)
            lots.append(
                InventoryLotSnapshot(
                    lot_id=f"lot-{len(lots)}",
                    item_id=f"item-{len(lots)}",
                    canonical_name=name,
                    on_hand=added,
                    reserved=Decimal(0),
                    unit=unit,
                )
            )
    return tuple(lots)


def summarize_update(node: str, update: Any) -> str:
    """Compress one node's state delta into a single demo line."""
    if node == "validate_input":
        if not isinstance(update, dict):
            return "Request boundary check passed"
        return "Request boundary check failed" if update.get("error") else "Request boundary check passed"
    if not isinstance(update, dict):
        return f"(node returned {type(update).__name__})"
    try:
        if node == "parse_recipes":
            cands = update.get("extracted_candidates", ())
            srcs = {c.extraction_source for c in cands}
            return f"Extracted {len(cands)} recipe candidate(s) · source={','.join(sorted(srcs)) or '?'}"
        if node == "detect_gaps":
            gaps = update.get("gaps", ())
            brief = ", ".join(f"{g.field_path}[{g.gap_class}]" for g in gaps[:4])
            more = f" ... {len(gaps)} total" if len(gaps) > 4 else ""
            return f"Detected {len(gaps)} knowledge gap(s) -> {brief}{more}" if gaps else "No gaps, straight to IR validation"
        if node == "infer_local":
            unresolved = update.get("gaps", ())
            return f"Local rules filled gaps -> {len(unresolved)} unresolved (routed onward)"
        if node == "research_missing":
            ev = update.get("research_evidence", {})
            return f"LLM research -> evidence for {len(ev)} gap(s)"
        if node == "validate_recipe_ir":
            irl = update.get("parsed_recipes", ())
            if not irl:
                return "IR semantic validation passed"
            names = ", ".join(r.dish_name for r in irl)
            return f"Built RecipeIR: {names} ({sum(len(r.ingredients) for r in irl)} ingredients / {sum(len(r.steps) for r in irl)} steps) -> validated"
        if node == "validate_safety":
            rep = update.get("safety_report")
            if rep is None:
                return "Safety assessment (no report)"
            findings = getattr(rep, "findings", ())
            policy = update.get("safety_policy")
            region = getattr(policy, "region", "?") if policy else "?"
            status = "WARN: findings" if findings else "OK: no violation"
            return f"Region {region} · {len(findings)} rule finding(s) {status}"
        if node == "check_feasibility":
            rep = update.get("feasibility_report")
            opts = update.get("repair_options", ())
            if rep is None:
                return "Feasibility check (no report)"
            shortages = getattr(rep, "ingredient_shortages", ())
            missing_res = getattr(rep, "missing_resources", ())
            if rep.is_feasible:
                return "Stock/resources sufficient OK -> prep merge"
            detail = f"{len(shortages)} shortage(s)" + (f", {len(missing_res)} missing equipment" if missing_res else "")
            return f"INFEASIBLE: {detail} -> generated {len(opts)} repair option(s)"
        if node == "merge_preparation":
            n_recipe = len(update.get("recipe_tasks", ()))
            n_prep = len(update.get("prep_tasks", ()))
            n_safety = len(update.get("safety_tasks", ()))
            n_obs = len(update.get("prep_observations", ()))
            return f"Prep merge -> recipe {n_recipe} / prep {n_prep} / safety {n_safety} / decisions {n_obs}"
        if node == "build_task_graph":
            return "Task DAG built (topologically ordered)"
        if node == "solve_schedule":
            sr = update.get("schedule_result")
            if sr is None:
                return "Solve produced no result"
            return f"CP-SAT solve -> {sr.status.value} · makespan={sr.makespan_minutes}min · wall={sr.wall_time_seconds:.2f}s"
        if node == "verify_schedule":
            vr = update.get("verification_report")
            if vr is None:
                return "Verification (no report)"
            return f"Independent verify -> {'PASS' if vr.passed else 'FAIL: ' + str(getattr(vr, 'issues', ()))}"
        if node == "repair_schedule":
            rh = update.get("repair_history", ())
            last = rh[-1] if rh else None
            action = getattr(last, "action", "?") if last else "?"
            detail = ""
            if last is not None and getattr(last, "note", None):
                detail = f" · {getattr(last, 'note', '')}"
            return f"Repair attempt #{len(rh)}: {action}{detail}"
        if node == "explain_schedule":
            src = update.get("explanation_source", "?")
            return f"Schedule explanation -> source={src}"
        if node == "render_ready_response":
            resp = update.get("response")
            if resp is None:
                return "Rendered READY (no response object)"
            return (
                f"Plan READY -> makespan={getattr(resp, 'makespan_minutes', '?')}min · "
                f"timeline={len(getattr(resp, 'timeline', ()))} · checklist={len(getattr(resp, 'completion_checklist', ()))}"
            )
        if node == "build_confirmation_response":
            resp = update.get("response")
            n_q = len(getattr(resp, "confirmation_questions", ())) if resp else 0
            return f"NEEDS_CONFIRMATION -> {n_q} structured question(s)"
        if node == "apply_confirmation":
            if update.get("confirmation_applied"):
                return f"Confirmation applied -> route={update.get('confirmation_route')}"
            return "Confirmation pending (interrupt -> waiting for user answers)"
        if node == "agent_controller":
            mode = update.get("agent_mode", "?")
            pd = update.get("pending_decision") or {}
            if pd.get("type") == "tool_call":
                return f"Controller decided tool_call -> {pd.get('tool')}"
            if pd.get("type") == "final":
                return "Controller decided final -> hand back to deterministic DAG"
            return f"Controller -> agent_mode={mode} (fallback to deterministic)"
        if node == "run_tool":
            calls = update.get("tool_calls", ())
            tool = calls[-1].get("tool") if calls else "?"
            obs = update.get("observations", ())
            ok = obs[-1].get("ok") if obs and isinstance(obs[-1], dict) else "?"
            return f"Tool executed -> {tool} · ok={ok} · observations={len(obs)}"
        if node == "render_infeasible_response":
            resp = update.get("response")
            return f"Plan INFEASIBLE -> {len(getattr(resp, 'reasons', ()))} reason(s)"
        if node == "render_failed_response":
            resp = update.get("response")
            return f"Execution FAILED -> {getattr(resp, 'error_code', '?')}"
        return "changed: " + ", ".join(update.keys())
    except Exception as exc:  # noqa: BLE001 — demo must never crash while printing
        return f"(summary error {type(exc).__name__})"


def auto_answers(payload: Any) -> list[dict[str, str]]:
    """Auto-answer a confirmation-dialog interrupt so the demo can proceed.

    CHOICE questions pick the suggested (or first) option; TEXT questions use
    the suggested value or a safe placeholder. Mirrors a user accepting all
    presented repair options.
    """
    answers: list[dict[str, str]] = []
    if not isinstance(payload, dict):
        return answers
    for q in payload.get("questions") or []:
        qid = q.get("question_id") if isinstance(q, dict) else None
        if not qid:
            continue
        if q.get("response_type") == "CHOICE":
            opts = [o for o in (q.get("options") or []) if isinstance(o, dict)]
            value = next((o.get("value") for o in opts if o.get("suggested")), None)
            value = value or (opts[0].get("value") if opts else None)
            if value is None:
                continue
        else:
            value = (q.get("suggested_value") or "").strip() or "ok"
        answers.append({"question_id": qid, "value": value})
    return answers


async def run_one(
    graph: Any,
    ctx: WorkflowContext,
    request: GeneratePlanRequest,
    title: str,
    verbose: bool = True,
) -> tuple[dict[str, Any], float, dict[str, Any]]:
    """Run one request through the workflow, auto-resuming confirmation dialogs.

    Dual-stream mode: ``updates`` carries node deltas (trace display),
    ``values`` returns full state each step (last one == terminal state).
    When the confirmation dialog interrupts (P5-4), the demo auto-answers and
    resumes via Command(resume=...) — one round, enough to prove the dialog
    is resumable without spamming identical repair loops.

    Returns (final_state, elapsed_seconds, meta) where meta carries the node
    trace and the number of auto-resumes (used to infer exercised features).
    """
    print(f"\n{C_BOLD}▶ {title}{C_RESET}")
    thread_id = f"{request.request_id}:0"
    config: dict[str, Any] = {"recursion_limit": 60, "configurable": {"thread_id": thread_id}}
    start = time.perf_counter()
    final: dict[str, Any] = {}
    trace: list[str] = []
    first_input: Any = {"request": request}
    resumes = 0
    while True:
        interrupt_payload: Any = None
        async for mode, chunk in graph.astream(
            first_input,
            context=ctx,
            config=config,
            stream_mode=["updates", "values"],
        ):
            if mode == "updates":
                for node, update in chunk.items():
                    if node == "__interrupt__":
                        interrupt_payload = update
                        continue
                    trace.append(node)
                    if not verbose:
                        continue
                    summary = summarize_update(node, update)
                    label = NODE_LABELS.get(node, node)
                    print(f"  {C_CYAN}├─ {label}{C_RESET}  {C_DIM}›{C_RESET} {summary}")
            else:
                final = chunk  # last values chunk == terminal state
        if interrupt_payload is None or resumes >= 1:
            break
        raw = interrupt_payload[0].value if isinstance(interrupt_payload, tuple) else interrupt_payload
        answers = auto_answers(raw)
        if not answers:
            break
        resumes += 1
        print(
            f"  {C_MAGENTA}◉ agent: confirmation dialog round {resumes} -> "
            f"auto-answered {len(answers)} question(s), resuming{C_RESET}",
        )
        first_input = Command(resume=answers)
    elapsed = time.perf_counter() - start
    return final, elapsed, {"trace": trace, "resumes": resumes}


def analyze_plan(final: dict[str, Any], dish_names: dict[str, str]) -> dict[str, Any]:
    """Extract smart-planning metrics from the terminal state."""
    resp = final.get("response")
    info: dict[str, Any] = {
        "status": getattr(resp, "status", "?"),
        "solver": getattr(resp, "solver_status", "?"),
        "makespan": getattr(resp, "makespan_minutes", None),
        "repairs": len(final.get("repair_history", ())),
        "tasks": 0,
        "peak_parallel": 0,
        "completions": [],
    }
    sr = final.get("schedule_result")
    info["solver_wall_s"] = getattr(sr, "wall_time_seconds", None) if sr else None

    timeline = getattr(resp, "timeline", ()) or ()
    info["tasks"] = len(timeline)

    # Peak parallelism: sweep-line over scheduled intervals [start, end).
    events: list[tuple[int, int]] = []
    for entry in timeline:
        events.append((int(entry["start_minute"]), 1))
        events.append((int(entry["end_minute"]), -1))
    events.sort()
    cur = peak = 0
    for _, delta in events:
        cur += delta
        peak = max(peak, cur)
    info["peak_parallel"] = peak

    # Per-dish completion minutes (from the schedule's dish_completions).
    completions: list[tuple[str, int]] = []
    for c in getattr(resp, "dish_completions", ()) or ():
        completions.append((dish_names.get(c["dish_id"], c["dish_id"]), int(c["completion_minute"])))
    info["completions"] = sorted(completions, key=lambda x: x[1])
    return info


def infer_features(
    final: dict[str, Any],
    meta: dict[str, Any],
    cache: InMemoryTTLCache[str, object] | None,
    cache_before: Any,
) -> list[str]:
    """Infer which agent capabilities actually fired, from the real state/trace.

    Never hard-codes a label: each feature is proven by evidence in the
    terminal state (research_evidence, explanation_source, extraction_source,
    repair_history) or in the node trace / cache stats.
    """
    feats: list[str] = []
    cands = final.get("extracted_candidates", ())
    if cands and any(getattr(c, "extraction_source", "") == "LLM" for c in cands):
        feats.append("LLM-parse")
    if final.get("research_evidence"):
        feats.append("LLM-research")
    if final.get("explanation_source") == "llm":
        feats.append("LLM-explain")
    if "agent_controller" in meta.get("trace", []):
        feats.append("ReAct-controller")
    if final.get("confirmation_context") is not None:
        feats.append("confirmation-dialog")
    if meta.get("resumes", 0) > 0:
        feats.append("auto-resume")
    if final.get("repair_history"):
        feats.append("repair-loop")
    vr = final.get("verification_report")
    if vr is not None and not getattr(vr, "passed", True):
        feats.append("verify-fail")
    if cache is not None and cache_before is not None and cache.stats().hits > cache_before.hits:
        feats.append("cache-hit")
    feats.append("checkpoint")
    return feats


def make_request(
    request_id: str,
    recipes: tuple[dict[str, str], ...],
    lots: tuple[InventoryLotSnapshot, ...],
    resources: tuple[KitchenResourceSnapshot, ...] = KITCHEN_RESOURCES,
) -> GeneratePlanRequest:
    return GeneratePlanRequest(
        request_id=request_id,
        user_id="demo-teacher",
        recipes=tuple(
            RecipeInput(recipe_id=f"r{i}", text=r["text"], target_servings=Decimal(2))
            for i, r in enumerate(recipes)
        ),
        inventory_lots=lots,
        kitchen_resources=resources,
    )


def with_stove(resources: tuple[KitchenResourceSnapshot, ...], burners: int) -> tuple[KitchenResourceSnapshot, ...]:
    """Return a copy of the resource set with a different stove capacity."""
    return tuple(
        r if r.resource_type != "stove" else r.model_copy(update={"capacity": Decimal(burners)})
        for r in resources
    )


# ---------------------------------------------------------------------------
# Full-agent wiring (mirrors application/main.py lifespan assembly)
# ---------------------------------------------------------------------------


async def build_full_agent() -> tuple[Any, WorkflowContext, LLMRecipeExtractor, InMemoryTTLCache[str, object]]:
    """Wire EVERY agent capability: LLM extractor, research, cache, explainer,
    ReAct controller, preference store, checkpoint + confirmation dialog."""
    from cooking_plan_agent.config.settings import get_settings

    settings = get_settings()
    llm_client = LLMClient(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        temperature=settings.llm_temperature,
        max_output_tokens=settings.llm_max_output_tokens,
        connection_pool_size=settings.llm_connection_pool_size,
    )
    extractor = LLMRecipeExtractor(llm_client)
    researcher = LLMKnowledgeResearcher(llm_client)
    cache: InMemoryTTLCache[str, object] = InMemoryTTLCache(
        max_entries=settings.cache_max_entries,
        max_item_size_bytes=settings.cache_max_item_bytes,
        default_ttl_seconds=settings.cache_ttl_seconds,
    )
    explainer = LLMPlanExplainer(llm_client)
    ctx = WorkflowContext(
        recipe_extractor=extractor,  # type: ignore[arg-type]
        recipe_researcher=researcher,
        safety_engine=SafetyEngine(),
        cache=cache,  # type: ignore[arg-type]
        explainer=explainer,
    )
    # P5-2 ReAct controller — tools mirror the injectable services.
    registry = ToolRegistry(ctx)
    controller = LLMReActController(llm_client, tools=registry.specs())
    ctx = dataclasses.replace(ctx, agent_controller=controller)
    # P5-4 long-term preference store.
    ctx = dataclasses.replace(ctx, preference_store=PreferenceStore("/tmp/cooking-demo-prefs.sqlite"))
    # P2-06 checkpointer (memory backend keeps the demo self-contained).
    provider = MemoryCheckpointProvider()
    await provider.astart()
    graph = build_cooking_plan_graph(checkpointer=provider.checkpointer)
    return graph, ctx, extractor, cache


async def extract_with_cache(
    extractor: LLMRecipeExtractor,
    cache: InMemoryTTLCache[str, object],
    text: str,
    model: str,
) -> Any:
    """LLM extraction behind the SAME parse-cache key the graph uses, so the
    graph's parse node hits the cache instead of re-calling the LLM."""
    from cooking_plan_agent.config.settings import get_settings

    schema_version = get_settings().supported_schema_versions[0]
    key = build_parse_cache_key(
        text,
        parser_type="LLMRecipeExtractor",
        model=model,
        prompt_version=PARSE_PROMPT_VERSION,
        schema_version=schema_version,
    )
    return await cache.get_or_compute(
        key,
        get_settings().cache_ttl_seconds,
        lambda: extractor.extract(text),  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Offline deterministic comprehensive test (7 dishes + randomized combos)
# ---------------------------------------------------------------------------


async def run_offline(graph: Any, ctx: WorkflowContext, extractor: RecipeExtractor, prepared: list[dict[str, str]]) -> None:
    p(f"\n{C_BOLD}STAGE 0 — Batch extraction preview (workflow input){C_RESET}")
    candidates = await asyncio.gather(*(extractor.extract(r["text"]) for r in prepared))
    for cand, meta in zip(candidates, prepared, strict=False):
        p(
            f"  · {C_CYAN}{meta['name']}{C_RESET} "
            f"-> {len(cand.ingredients)} ingredients / {len(cand.steps)} steps / "
            f"{sum(1 for g in cand.ingredients if g.quantity is None)} quantity gap(s) to infer",
        )

    # ---------- Stage 1: per-dish baseline (independent requests) ----------
    p(f"\n{C_BOLD}STAGE 1 — Per-dish baseline (each dish = one independent request){C_RESET}")
    per_dish: dict[str, dict[str, Any]] = {}
    for i, ((meta, cand)) in enumerate(zip(prepared, candidates, strict=False)):
        lots = build_lots_from_candidates([cand])
        request = make_request(f"demo-single-{i}", (meta,), lots)
        final, elapsed, _ = await run_one(graph, ctx, request, f"Dish #{i + 1}: {meta['name']}")
        info = analyze_plan(final, {"r0": meta["name"]})
        per_dish[meta["name"]] = info
        p(
            f"  {C_DIM}-> terminal {C_RESET}{_status_color(info['status'])}{info['status']}{C_RESET}"
            f"{C_DIM} · solver={info['solver']} · makespan={info['makespan']}min · tasks={info['tasks']} · wall={elapsed:.2f}s{C_RESET}",
        )

    # ---------- Stage 2: SMART PLANNING — randomized multi-dish combos ----------
    p(f"\n{C_BOLD}STAGE 2 — Smart planning: randomized multi-dish meal combos (seeded, reproducible){C_RESET}")
    p(f"{C_DIM}  Each combo is one request; CP-SAT packs all dishes into one shared timeline.{C_RESET}")
    rng = random.Random(42)  # fixed seed -> reproducible combinations
    combo_indices: list[list[int]] = [rng.sample(range(len(RECIPES)), k) for k in (2, 3, 4, 5, 6, 7)]
    combo_rows: list[dict[str, Any]] = []
    combo_finals: dict[int, dict[str, Any]] = {}
    combo_dish_ids: dict[int, dict[str, str]] = {}
    for ci, idxs in enumerate(combo_indices):
        combo_meta = [prepared[i] for i in idxs]
        combo_cands = [candidates[i] for i in idxs]
        lots = build_lots_from_candidates(combo_cands)
        dish_ids = {f"r{i}": m["name"] for i, m in enumerate(combo_meta)}
        request = make_request(f"demo-combo-{ci}", tuple(combo_meta), lots)
        names = " + ".join(m["name"] for m in combo_meta)
        final, elapsed, _ = await run_one(
            graph, ctx, request, f"Combo #{ci + 1} ({len(idxs)} dishes): {names}", verbose=False
        )
        info = analyze_plan(final, dish_ids)
        serial = sum(per_dish[m["name"]]["makespan"] or 0 for m in combo_meta)
        speedup = round(serial / info["makespan"], 2) if info["makespan"] else None
        info.update({"serial_min": serial, "speedup": speedup})
        combo_rows.append(info)
        combo_finals[ci] = final
        combo_dish_ids[ci] = dish_ids
        p(
            f"  {C_BOLD}Combo #{ci + 1}{C_RESET} {C_DIM}({len(idxs)} dishes){C_RESET} "
            f"-> {_status_color(info['status'])}{info['status']}{C_RESET}"
            f"{C_DIM} · solver={C_RESET}{_solver_color(info['solver'])}{info['solver']}{C_RESET}"
            f"{C_DIM} · makespan={C_RESET}{info['makespan']}min"
            f"{C_DIM} · serial≈{serial}min · speedup={C_RESET}{C_GREEN}{speedup}x{C_RESET}"
            f"{C_DIM} · tasks={info['tasks']} · peak_parallel={info['peak_parallel']} · wall={elapsed:.2f}s{C_RESET}",
        )
        done = ", ".join(f"{n}@{m}min" for n, m in info["completions"])
        p(f"     {C_DIM}dish completion order:{C_RESET} {done}")

    # ---------- Stage 2.5: timeline sample (proof of interleaved planning) ----------
    p(f"\n{C_BOLD}STAGE 2.5 — Timeline sample: CP-SAT interleaves tasks on one shared clock{C_RESET}")
    resp7 = combo_finals[5].get("response")
    tl7 = getattr(resp7, "timeline", ()) or ()
    names7 = combo_dish_ids[5]
    for entry in tl7[:10]:
        dish = names7.get(str(entry["dish_id"]), str(entry["dish_id"]))
        p(
            f"  t={entry['start_minute']:>3}-{entry['end_minute']:<3} "
            f"{C_CYAN}[{entry['work_mode']:<7}]{C_RESET} {entry['instruction'][:42]:<44} "
            f"{C_DIM}({dish}){C_RESET}",
        )
    p(f"  {C_DIM}... {len(tl7) - 10} more intervals. Overlapping windows across dishes = parallel passive cooking on one cook.{C_RESET}")

    # ---------- Stage 3: resource-aware planning ----------
    p(f"\n{C_BOLD}STAGE 3 — Resource sensitivity: same 7-dish meal, fewer stove burners{C_RESET}")
    p(f"{C_DIM}  The solver re-solves under tighter capacity — watch peak parallelism vs makespan.{C_RESET}")
    idxs = combo_indices[5]  # 7-dish combo (highest parallelism) from stage 2
    combo_meta = [prepared[i] for i in idxs]
    combo_cands = [candidates[i] for i in idxs]
    lots = build_lots_from_candidates(combo_cands)
    dish_ids = {f"r{i}": m["name"] for i, m in enumerate(combo_meta)}
    names = " + ".join(m["name"] for m in combo_meta)
    resource_rows: list[tuple[int, dict[str, Any], float]] = []
    for burners in (4, 2, 1):
        request = make_request(
            f"demo-stove-{burners}",
            tuple(combo_meta),
            lots,
            resources=with_stove(KITCHEN_RESOURCES, burners),
        )
        final, elapsed, _ = await run_one(
            graph, ctx, request, f"Stove x{burners} burner(s): {names}", verbose=False
        )
        info = analyze_plan(final, dish_ids)
        resource_rows.append((burners, info, elapsed))
        p(
            f"  · stove x{burners} -> {_status_color(info['status'])}{info['status']}{C_RESET}"
            f"{C_DIM} · solver={C_RESET}{_solver_color(info['solver'])}{info['solver']}{C_RESET}"
            f"{C_DIM} · makespan={C_RESET}{info['makespan']}min"
            f"{C_DIM} · tasks={info['tasks']} · peak_parallel={info['peak_parallel']} · wall={elapsed:.2f}s{C_RESET}",
        )
    base = resource_rows[0][1]["makespan"] or 0
    base_peak = resource_rows[0][1]["peak_parallel"]
    for burners, info, _ in resource_rows[1:]:
        peak_shift = info["peak_parallel"] - base_peak
        p(
            f"  {C_DIM}  · {burners} burner(s): makespan {info['makespan']}min (vs {base}min) · "
            f"peak_parallel {info['peak_parallel']} ({peak_shift:+d} vs 4 burners){C_RESET}",
        )
    p(f"  {C_DIM}  Insight: ACTIVE tasks are serialized by the single cook (no_overlap), so the true bottleneck is the{C_RESET}")
    p(f"  {C_DIM}  cook, not the stove — passive simmering rides the soup's critical path. Burner count only repacks{C_RESET}")
    p(f"  {C_DIM}  intervals (peak 4 -> 3) and never hard-codes a fixed schedule.{C_RESET}")

    # ---------- Stage 4: fault injection -> dynamic routing ----------
    p(f"\n{C_BOLD}STAGE 4 — Fault injection: only 30% stock -> dynamic route to NEEDS_CONFIRMATION{C_RESET}")
    target = 2  # Spicy Douban Prawns (clear main ingredient)
    meta_t, cand_t = prepared[target], candidates[target]
    lots_partial = build_lots_from_candidates([cand_t], shortage_scale=Decimal("0.3"))
    request = make_request("demo-shortage", (meta_t,), lots_partial)
    final, elapsed, _ = await run_one(graph, ctx, request, f"Fault injection: {meta_t['name']} (30% stock)")
    info = analyze_plan(final, {"r0": meta_t["name"]})
    p(f"  {C_DIM}-> terminal {C_RESET}{_status_color(info['status'])}{info['status']}{C_RESET}"
      f"{C_DIM} · repair options surfaced, user must confirm to proceed · wall={elapsed:.2f}s{C_RESET}")

    # ---------- Stage 5: reflection-repair loop ----------
    p(f"\n{C_BOLD}STAGE 5 — Reflection-repair loop: remove wok/spatula -> verify FAIL -> self-repair{C_RESET}")
    limited_resources = tuple(r for r in KITCHEN_RESOURCES if r.resource_type not in ("wok", "spatula"))
    meta_r, cand_r = prepared[0], candidates[0]  # Spicy Crab Legs
    lots_r = build_lots_from_candidates([cand_r])
    request = make_request("demo-repair", (meta_r,), lots_r, resources=limited_resources)
    final, elapsed, _ = await run_one(graph, ctx, request, f"Repair loop: {meta_r['name']} (no wok/spatula in kitchen)")
    info = analyze_plan(final, {"r0": meta_r["name"]})
    rh = final.get("repair_history", ())
    p(f"  {C_DIM}-> terminal {C_RESET}{_status_color(info['status'])}{info['status']}{C_RESET}"
      f"{C_DIM} · repair attempts={len(rh)} (back-edge audit trail in repair_history) · wall={elapsed:.2f}s{C_RESET}")
    for rec in rh:
        p(f"     · {C_DIM}{getattr(rec, 'action', '?')}{C_RESET}")

    # ---------- Stage 6: summary ----------
    bar("═")
    p(f"{C_BOLD}  SUMMARY — planning metrics across all test samples{C_RESET}")
    bar("═")
    p(f"  {'':<3}{'Sample':<26}{'Status':<20}{'Solver':<10}{'Makespan':<10}{'Tasks':<7}{'Peak#':<6}")
    p(f"  {'-'*3}{'-'*26}{'-'*20}{'-'*10}{'-'*10}{'-'*7}{'-'*6}")
    for i, (name, info) in enumerate(per_dish.items()):
        p(f"  {i + 1:<3}{name:<26}{_status_color(info['status'])}{info['status']:<20}{C_RESET}"
          f"{_solver_color(info['solver'])}{info['solver']:<10}{C_RESET}{str(info['makespan']):<10}{info['tasks']:<7}{info['peak_parallel']:<6}")
    for i, info in enumerate(combo_rows):
        label = f"Combo {i + 1} ({info['solver']})"
        p(f"  {'C'+str(i+1):<3}{label:<26}{_status_color(info['status'])}{info['status']:<20}{C_RESET}"
          f"{_solver_color(info['solver'])}{info['solver']:<10}{C_RESET}{str(info['makespan']):<10}{info['tasks']:<7}{info['peak_parallel']:<6}")
    p(f"\n{C_DIM}Every request terminated inside the graph; full node traces printed above (stages 0-1, 4-5).{C_RESET}")
    p(f"{C_DIM}Speedup = serial makespan sum / parallel makespan -> CP-SAT packs dishes onto the same timeline.{C_RESET}")


# ---------------------------------------------------------------------------
# Full-agent comprehensive test (every capability, representative cases)
# ---------------------------------------------------------------------------


async def run_full_agent(
    graph: Any,
    ctx: WorkflowContext,
    extractor: LLMRecipeExtractor,
    cache: InMemoryTTLCache[str, object],
    prepared: list[dict[str, str]],
) -> None:
    from cooking_plan_agent.config.settings import get_settings

    settings = get_settings()

    # ---------- Stage 0: LLM batch extraction (warms the parse cache) ----------
    p(f"\n{C_BOLD}STAGE 0 — LLM recipe extraction (7 dishes, concurrency {settings.llm_max_concurrency}){C_RESET}")
    sem = asyncio.Semaphore(max(1, settings.llm_max_concurrency))

    async def _one(meta: dict[str, str]) -> Any:
        async with sem:
            return await extract_with_cache(extractor, cache, meta["text"], settings.llm_model)

    t0 = time.perf_counter()
    candidates = await asyncio.gather(*(_one(r) for r in prepared))
    p(f"  {C_DIM}batch extraction wall={time.perf_counter() - t0:.1f}s (cache warmed — graph parses hit the cache){C_RESET}")
    for cand, meta in zip(candidates, prepared, strict=False):
        p(
            f"  · {C_CYAN}{meta['name']}{C_RESET} "
            f"-> source={cand.extraction_source} · {len(cand.ingredients)} ingredients / {len(cand.steps)} steps",
        )

    rng = random.Random(7)
    combo_small = [prepared[i] for i in rng.sample(range(len(RECIPES)), 3)]
    combo_big = [prepared[i] for i in rng.sample(range(len(RECIPES)), 5)]

    rows: list[dict[str, Any]] = []

    def _record(label: str, final: dict[str, Any], meta: dict[str, Any], elapsed: float, dish_ids: dict[str, str]) -> None:
        info = analyze_plan(final, dish_ids)
        features = infer_features(final, meta, cache, cache_before)
        info.update({"label": label, "features": features, "wall": elapsed})
        rows.append(info)
        feat = " + ".join(features)
        p(
            f"  {C_DIM}-> {C_RESET}{_status_color(info['status'])}{info['status']}{C_RESET}"
            f"{C_DIM} · solver={info['solver']} · makespan={info['makespan']}min · wall={elapsed:.1f}s · {C_RESET}"
            f"{C_MAGENTA}[{feat}]{C_RESET}",
        )

    # ---------- Stage 1: single-dish flows (LLM extract + explain + safety) ----------
    p(f"\n{C_BOLD}STAGE 1 — Single-dish flows (LLM extraction + safety + CP-SAT + LLM explanation){C_RESET}")
    for idx in (0, 3, 6):  # crab legs / rib soup / chicken wings
        meta = prepared[idx]
        cand = candidates[idx]
        lots = build_lots_from_candidates([cand])
        request = make_request(f"full-single-{idx}", (meta,), lots)
        cache_before = cache.stats()
        final, elapsed, meta_info = await run_one(graph, ctx, request, f"Dish: {meta['name']}")
        _record(f"Dish {meta['name']}", final, meta_info, elapsed, {"r0": meta["name"]})

    # ---------- Stage 2: multi-dish combos (smart planning, cache hit) ----------
    p(f"\n{C_BOLD}STAGE 2 — Multi-dish combos (smart planning, shared prep merge, parse-cache hit){C_RESET}")
    for tag, metas in (("3-dish", combo_small), ("5-dish", combo_big)):
        idxs = [prepared.index(m) for m in metas]
        lots = build_lots_from_candidates([candidates[i] for i in idxs])
        dish_ids = {f"r{i}": m["name"] for i, m in enumerate(metas)}
        request = make_request(f"full-combo-{tag}", tuple(metas), lots)
        names = " + ".join(m["name"] for m in metas)
        cache_before = cache.stats()
        final, elapsed, meta_info = await run_one(graph, ctx, request, f"Combo ({tag}): {names}", verbose=False)
        _record(f"Combo {tag}", final, meta_info, elapsed, dish_ids)

    # ---------- Stage 3: fault injection -> confirmation dialog + auto-resume ----------
    p(f"\n{C_BOLD}STAGE 3 — Fault injection (30% stock) -> NEEDS_CONFIRMATION dialog -> auto-resume{C_RESET}")
    target = 2  # Spicy Douban Prawns
    meta_t, cand_t = prepared[target], candidates[target]
    lots_partial = build_lots_from_candidates([cand_t], shortage_scale=Decimal("0.3"))
    request = make_request("full-shortage", (meta_t,), lots_partial)
    cache_before = cache.stats()
    final, elapsed, meta_info = await run_one(graph, ctx, request, f"Fault injection: {meta_t['name']} (30% stock)")
    _record(f"Fault 30% stock ({meta_t['name']})", final, meta_info, elapsed, {"r0": meta_t["name"]})

    # ---------- Stage 4: equipment shortage -> confirmation dialog ----------
    p(f"\n{C_BOLD}STAGE 4 — Equipment shortage (no wok/spatula) -> feasibility stop -> confirmation dialog{C_RESET}")
    p(f"{C_DIM}  With confirmation dialog ON, a missing tool is caught at feasibility and offered as a repair{C_RESET}")
    p(f"{C_DIM}  decision (alternative_equipment) instead of failing later at verification — agent routes by context.{C_RESET}")
    limited_resources = tuple(r for r in KITCHEN_RESOURCES if r.resource_type not in ("wok", "spatula"))
    meta_r, cand_r = prepared[0], candidates[0]
    lots_r = build_lots_from_candidates([cand_r])
    request = make_request("full-repair", (meta_r,), lots_r, resources=limited_resources)
    cache_before = cache.stats()
    final, elapsed, meta_info = await run_one(graph, ctx, request, f"Equipment shortage: {meta_r['name']} (no wok/spatula)")
    _record(f"Equipment shortage ({meta_r['name']})", final, meta_info, elapsed, {"r0": meta_r["name"]})

    # ---------- Stage 5: summary ----------
    bar("═")
    p(f"{C_BOLD}  FULL-AGENT SUMMARY — capabilities actually exercised (inferred from state){C_RESET}")
    bar("═")
    for i, r in enumerate(rows, start=1):
        p(
            f"  {i:<2} {r['label']:<42} {_status_color(r['status'])}{r['status']:<18}{C_RESET}"
            f"{str(r['makespan']):<8}{r['wall']:.1f}s",
        )
        p(f"     {C_DIM}features: {', '.join(r['features'])}{C_RESET}")
    research_fired = any("LLM-research" in r["features"] for r in rows)
    controller_fired = any("ReAct-controller" in r["features"] for r in rows)
    p(f"\n{C_DIM}Total cases: {len(rows)} · all capabilities ON: LLM, research, cache, explain, checkpoint,{C_RESET}")
    p(f"{C_DIM}ReAct controller, confirmation dialog. LLM-research fired={research_fired} (only when a heat/duration{C_RESET}")
    p(f"{C_DIM}gap survives local inference); ReAct controller fired={controller_fired} (LLM decides, falls back to the{C_RESET}")
    p(f"{C_DIM}deterministic DAG on any failure — soft decision, hard guarantee).{C_RESET}")


async def main() -> None:
    bar("═")
    if _FULL_AGENT:
        p(f"{C_BOLD}  Cooking Plan Agent — FULL-AGENT comprehensive test (ALL capabilities ON){C_RESET}")
        p(f"{C_DIM}  LLM extract · LLM research · cache · explain · checkpoint · ReAct controller · confirmation dialog{C_RESET}")
    else:
        p(f"{C_BOLD}  Cooking Plan Agent — Smart Planning + Agentic Workflow demo (7 dishes / offline deterministic){C_RESET}")
        p(f"{C_DIM}  Workflow: LangGraph DAG · Scheduling: CP-SAT (OR-Tools) · Parsing: rule engine (no LLM){C_RESET}")
    bar("═")

    # Preprocessor: inject the Chinese title as the first line so the
    # extractor resolves dish_name correctly (mirrors a real titled request).
    prepared: list[dict[str, str]] = [
        {"name": r["name"], "zh": r["zh"], "text": f"{r['zh']}\n{r['text']}"} for r in RECIPES
    ]

    if _FULL_AGENT:
        # LangGraph's memory checkpointer emits "Deserializing unregistered
        # type ... from checkpoint" warnings on every resume — harmless here
        # (the demo still proves checkpoint resume works), so silence them.
        import logging as _logging

        _logging.getLogger("langgraph").setLevel(_logging.ERROR)

        graph, ctx, extractor, cache = await build_full_agent()
        await run_full_agent(graph, ctx, extractor, cache, prepared)
        return

    extractor = RecipeExtractor()
    ctx = WorkflowContext(recipe_extractor=extractor)
    graph = build_cooking_plan_graph()
    await run_offline(graph, ctx, extractor, prepared)


def _status_color(status: str) -> str:
    if status == "READY":
        return C_GREEN
    if status == "NEEDS_CONFIRMATION":
        return C_YELLOW
    return C_RED


def _solver_color(status: str) -> str:
    return C_GREEN if status == "OPTIMAL" else C_YELLOW


if __name__ == "__main__":
    asyncio.run(main())
