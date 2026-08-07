"""P5-4: 确认对话中间态 —— apply_confirmation_node 与路由。"""

from types import SimpleNamespace

import pytest

from cooking_plan_agent.config.settings import get_settings
from cooking_plan_agent.domain.models import (
    ApprovedDecision,
    ConfirmationPlanResponse,
    ConfirmationQuestion,
    GeneratePlanRequest,
    QuestionAnswer,
    QuestionOption,
    QuestionResponseType,
)
from cooking_plan_agent.workflow.conversation_nodes import apply_confirmation_node
from cooking_plan_agent.workflow.routing import route_after_confirmation
from cooking_plan_agent.workflow.state import PlanState


def _request() -> GeneratePlanRequest:
    return GeneratePlanRequest(
        request_id="req-confirm",
        user_id="user-confirm",
        recipes=({"recipe_id": "r1", "text": "300 g tofu", "target_servings": 2},),
        time_limit_minutes=60,
    )


def _confirmation_context() -> ConfirmationPlanResponse:
    decision = ApprovedDecision(
        option_id="opt-time",
        option_type="extend_time",
        payload={"time_limit_minutes": 120},
        plan_revision="req-confirm:v1",
    )
    return ConfirmationPlanResponse(
        plan_id="req-confirm",
        status="NEEDS_CONFIRMATION",
        confirmation_questions=(
            ConfirmationQuestion(
                question_id="q-extend-time",
                field_path="request.time_limit_minutes",
                prompt="Extend the time limit?",
                response_type=QuestionResponseType.CHOICE,
                options=(QuestionOption(value="opt-time", label="Extend to 120 minutes"),),
            ),
        ),
        decisions=(decision,),
        plan_revision="req-confirm:v1",
    )


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(context=SimpleNamespace())


# ---------------------------------------------------------------------------
# 配置（P5-4）
# ---------------------------------------------------------------------------


def test_confirmation_dialog_settings_defaults(monkeypatch) -> None:
    get_settings.cache_clear()
    monkeypatch.delenv("COOKING_PLAN_CONFIRMATION_DIALOG_ENABLED", raising=False)
    settings = get_settings()
    assert settings.confirmation_dialog_enabled is False


def test_confirmation_dialog_settings_env_override(monkeypatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("COOKING_PLAN_CONFIRMATION_DIALOG_ENABLED", "true")
    settings = get_settings()
    assert settings.confirmation_dialog_enabled is True


# ---------------------------------------------------------------------------
# apply_confirmation_node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_confirmation_applies_answers_to_request() -> None:
    """用户答复（answers）被映射为 decision 并应用到 request（extend_time）。"""
    state: PlanState = {
        "request": _request(),
        "confirmation_context": _confirmation_context(),
        "confirmation_answers": (QuestionAnswer(question_id="q-extend-time", value="opt-time"),),
    }
    delta = await apply_confirmation_node(state, _runtime())  # type: ignore[arg-type]
    assert delta["confirmation_applied"] is True
    assert delta["needs_confirmation"] is False
    new_request = delta["request"]
    assert new_request.time_limit_minutes == 120


@pytest.mark.asyncio
async def test_apply_confirmation_invalid_answer_reports_errors() -> None:
    """无效答复（未知 question_id）→ 字段级错误，不静默放行。"""
    state: PlanState = {
        "request": _request(),
        "confirmation_context": _confirmation_context(),
        "confirmation_answers": (QuestionAnswer(question_id="q-unknown", value="whatever"),),
    }
    delta = await apply_confirmation_node(state, _runtime())  # type: ignore[arg-type]
    assert delta["confirmation_applied"] is False
    assert delta["needs_confirmation"] is True
    assert delta["confirmation_error"]
    assert "unknown question_id" in delta["confirmation_error"][0]


# ---------------------------------------------------------------------------
# route_after_confirmation
# ---------------------------------------------------------------------------


def test_route_after_confirmation_still_needs_confirmation() -> None:
    state: PlanState = {"request": _request(), "needs_confirmation": True}
    assert route_after_confirmation(state) == "build_confirmation_response"


def test_route_after_confirmation_redirects_parse() -> None:
    """答复改变了 request 内容（substitute/replace_dish）→ 从 parse 重新推进。"""
    state: PlanState = {
        "request": _request(),
        "confirmation_applied": True,
        "confirmation_route": "parse_recipes",
    }
    assert route_after_confirmation(state) == "parse_recipes"


def test_route_after_confirmation_reschedules_only_constraint() -> None:
    """答复仅调整约束（extend_time / reduce_servings）→ 直接重排。"""
    state: PlanState = {
        "request": _request(),
        "confirmation_applied": True,
        "confirmation_route": "solve_schedule",
    }
    assert route_after_confirmation(state) == "solve_schedule"


def test_route_after_confirmation_defaults_to_ready() -> None:
    """答复应用后无修改 → READY。"""
    state: PlanState = {
        "request": _request(),
        "confirmation_applied": True,
        "confirmation_route": "render_ready_response",
    }
    assert route_after_confirmation(state) == "render_ready_response"
