"""P5-3: LLM 诊断为加法能力，最终动作仍由规则裁决。"""
from types import SimpleNamespace

import pytest

from cooking_plan_agent.config.settings import get_settings
from cooking_plan_agent.domain.models import GeneratePlanRequest, RecipeInput
from cooking_plan_agent.scheduling.models import VerificationIssue, VerificationReport
from cooking_plan_agent.workflow.repair_nodes import repair_schedule_node
from cooking_plan_agent.workflow.state import PlanState


class _FakeDiagnoser:
    """伪造诊断器：返回不可信建议，验证规则会忽略越界动作。"""

    def __init__(self) -> None:
        self.calls = 0

    async def diagnose(self, context: dict[str, object]) -> dict[str, object]:
        self.calls += 1
        # 恶意/错误建议：直接指定越级深度 —— 规则不应采纳。
        return {"optimization_level": "makespan"}


@pytest.mark.asyncio
async def test_llm_diagnosis_cannot_override_rules(monkeypatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("COOKING_PLAN_SCHEDULE_REPAIR_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("COOKING_PLAN_SCHEDULE_REPAIR_LLM_ENABLED", "true")
    monkeypatch.setenv("COOKING_PLAN_SOLVER_OPTIMIZATION_LEVEL", "full")

    request = GeneratePlanRequest(
        request_id="req-llm",
        user_id="user-llm",
        recipes=(RecipeInput(recipe_id="r1", text="x", target_servings=2),),
        time_limit_minutes=60,
    )
    state: PlanState = {
        "request": request,
        "verification_report": VerificationReport(
            passed=False,
            issues=(VerificationIssue(code="CAPACITY_EXCEEDED", message="c"),),
        ),
    }
    diagnoser = _FakeDiagnoser()
    runtime = SimpleNamespace(context=SimpleNamespace(repair_diagnoser=diagnoser))

    delta = await repair_schedule_node(state, runtime)  # type: ignore[arg-type]
    # LLM 被调用过，但最终动作仍是规则裁决的阶梯下一级，而非 makespan。
    assert diagnoser.calls == 1
    assert delta["solver_overrides"]["optimization_level"] == "phase12"
