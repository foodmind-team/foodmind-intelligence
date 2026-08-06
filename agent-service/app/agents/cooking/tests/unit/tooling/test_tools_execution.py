"""P5-1: 首批工具执行。

每个工具：输入合法 dict → 返回可序列化 dict（json.dumps 不抛错），
并断言关键字段存在。全程无网络、无真实大求解。
"""

import json
from decimal import Decimal
from types import SimpleNamespace

import pytest

from cooking_plan_agent.domain.enums import SolverStatus, WorkMode
from cooking_plan_agent.domain.models import (
    CookingTask,
    EvidenceResult,
    ExtractedIngredient,
    ExtractedRecipeCandidate,
    ExtractedStep,
    IngredientDemand,
    InventoryLotSnapshot,
    RecipeIR,
    RecipeStep,
    SafetyReport,
)
from cooking_plan_agent.scheduling.models import (
    ScheduledInterval,
    ScheduleResult,
    SchedulingProblem,
)
from cooking_plan_agent.tooling.registry import ToolRegistry
from cooking_plan_agent.tooling.schemas import RegisteredTool

TOOL_NAMES = (
    "parse_recipe",
    "research_gap",
    "evaluate_safety",
    "check_feasibility",
    "solve_schedule",
    "verify_schedule",
)


# ---------------------------------------------------------------------------
# Fakes（满足 context 服务 Protocol 的形状）
# ---------------------------------------------------------------------------


class _FakeExtractor:
    async def extract(self, source_text: str) -> ExtractedRecipeCandidate:
        return ExtractedRecipeCandidate(
            recipe_id="r1",
            dish_name="Tofu",
            original_servings=Decimal(2),
            source_language="en",
            ingredients=(
                ExtractedIngredient(
                    raw_text="tofu 200g",
                    name="tofu",
                    quantity=Decimal(200),
                    unit="g",
                ),
            ),
            steps=(ExtractedStep(step_number=1, instruction="Cook tofu"),),
        )


class _FakeResearcher:
    def __init__(self) -> None:
        self.last_query: object | None = None

    async def research(self, query: object) -> list[EvidenceResult]:
        self.last_query = query
        return [
            EvidenceResult(
                source_title="t",
                source_url="https://x",
                snippet="s",
                confidence=Decimal("0.8"),
                extracted_fact="heat",
                fact_type="temperature",
                fact_value="200",
            )
        ]


class _FakeSafetyEngine:
    def __init__(self) -> None:
        self.last_context: object | None = None

    def evaluate(self, context: object) -> SafetyReport:
        self.last_context = context
        return SafetyReport(report_id="sr1", findings=(), is_safe=True, has_unrepairable=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _demand() -> IngredientDemand:
    return IngredientDemand(
        canonical_name="tofu",
        raw_name="tofu",
        quantity=Decimal(200),
        unit="g",
        confidence=Decimal("0.9"),
    )


def _recipe_ir() -> RecipeIR:
    return RecipeIR(
        recipe_id="r1",
        dish_name="Tofu Stir-fry",
        original_servings=Decimal(2),
        target_servings=Decimal(2),
        source_language="en",
        ingredients=(_demand(),),
        steps=(RecipeStep(step_number=1, instruction="Cook tofu"),),
    )


def _task() -> CookingTask:
    return CookingTask(
        task_id="t1",
        dish_id="d1",
        instruction="Boil water",
        duration_minutes=5,
        work_mode=WorkMode.ACTIVE,
        category="heating",
    )


def _registry_and_fakes() -> tuple[ToolRegistry, _FakeExtractor, _FakeResearcher, _FakeSafetyEngine]:
    extractor = _FakeExtractor()
    researcher = _FakeResearcher()
    engine = _FakeSafetyEngine()
    context = SimpleNamespace(
        recipe_extractor=extractor,
        recipe_researcher=researcher,
        safety_engine=engine,
    )
    registry = ToolRegistry(context)  # type: ignore[arg-type]
    return registry, extractor, researcher, engine


def _get_tool(registry: ToolRegistry, name: str) -> RegisteredTool:
    tool = registry.get(name)
    assert tool is not None, f"tool '{name}' not registered"
    return tool


def _assert_json_serialisable(payload: dict) -> None:
    assert isinstance(json.dumps(payload), str)


# ---------------------------------------------------------------------------
# 六工具执行
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parse_recipe_tool() -> None:
    registry, _extractor, _researcher, _engine = _registry_and_fakes()
    tool = _get_tool(registry, "parse_recipe")
    out = await tool.executor({"source_text": "Tofu recipe text"})
    _assert_json_serialisable(out)
    assert out["candidate"]["dish_name"] == "Tofu"
    assert out["candidate"]["recipe_id"] == "r1"


@pytest.mark.asyncio
async def test_research_gap_tool() -> None:
    registry, _extractor, researcher, _engine = _registry_and_fakes()
    tool = _get_tool(registry, "research_gap")
    out = await tool.executor(
        {
            "query_text": "safe internal temperature chicken",
            "gap_type": "critical",
            "recipe_context": "Chicken Curry",
        }
    )
    _assert_json_serialisable(out)
    assert out["count"] == 1
    assert out["results"][0]["source_url"] == "https://x"
    assert researcher.last_query is not None
    assert researcher.last_query.query_text == "safe internal temperature chicken"
    assert researcher.last_query.recipe_context == "Chicken Curry"


@pytest.mark.asyncio
async def test_evaluate_safety_tool() -> None:
    registry, _extractor, _researcher, engine = _registry_and_fakes()
    tool = _get_tool(registry, "evaluate_safety")
    out = await tool.executor(
        {
            "recipes": [_recipe_ir().model_dump(mode="json")],
            "dietary_restrictions": ["vegetarian"],
            "user_allergens": ["peanut"],
        }
    )
    _assert_json_serialisable(out)
    assert out["safety_report"]["report_id"] == "sr1"
    assert out["safety_report"]["is_safe"] is True
    assert engine.last_context is not None
    assert engine.last_context.recipes[0].dish_name == "Tofu Stir-fry"


@pytest.mark.asyncio
async def test_check_feasibility_tool() -> None:
    registry, _extractor, _researcher, _engine = _registry_and_fakes()
    tool = _get_tool(registry, "check_feasibility")
    out = await tool.executor(
        {
            "demands": [_demand().model_dump(mode="json")],
            "lots": [
                InventoryLotSnapshot(
                    lot_id="lot-1",
                    item_id="item-1",
                    canonical_name="tofu",
                    on_hand=Decimal(1000),
                    reserved=Decimal(0),
                    unit="g",
                ).model_dump(mode="json")
            ],
        }
    )
    _assert_json_serialisable(out)
    assert out["feasibility_report"]["is_feasible"] is True
    assert out["feasibility_report"]["ingredient_results"][0]["ingredient_name"] == "tofu"


@pytest.mark.asyncio
async def test_solve_schedule_tool() -> None:
    registry, _extractor, _researcher, _engine = _registry_and_fakes()
    tool = _get_tool(registry, "solve_schedule")
    out = await tool.executor(
        {
            "tasks": [_task().model_dump(mode="json")],
            "resources": [],
            "solver_timeout_seconds": 1.0,
            "optimization_level": "makespan",
        }
    )
    _assert_json_serialisable(out)
    assert out["schedule_result"]["status"] in ("OPTIMAL", "FEASIBLE")
    assert out["schedule_result"]["intervals"][0]["task_id"] == "t1"


@pytest.mark.asyncio
async def test_verify_schedule_tool() -> None:
    registry, _extractor, _researcher, _engine = _registry_and_fakes()
    tool = _get_tool(registry, "verify_schedule")
    problem = SchedulingProblem(tasks=(_task(),), resources=())
    result = ScheduleResult(
        status=SolverStatus.OPTIMAL,
        makespan_minutes=5,
        intervals=(ScheduledInterval(task_id="t1", start_minute=0, end_minute=5),),
    )
    out = await tool.executor(
        {
            "problem": problem.model_dump(mode="json"),
            "result": result.model_dump(mode="json"),
        }
    )
    _assert_json_serialisable(out)
    assert out["verification_report"]["passed"] is True
    assert out["verification_report"]["checked_task_count"] == 1


def test_six_tools_registered() -> None:
    registry, _extractor, _researcher, _engine = _registry_and_fakes()
    for name in TOOL_NAMES:
        assert isinstance(registry.get(name), RegisteredTool), f"tool '{name}' not registered"
