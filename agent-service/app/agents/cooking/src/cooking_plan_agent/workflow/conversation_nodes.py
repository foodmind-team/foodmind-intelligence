"""P5-4: 多轮确认对话节点。

把一次性 NEEDS_CONFIRMATION 终态升级为对话中间态：用户在
``apply_confirmation`` 处挂起（LangGraph interrupt/断点），提交
answers 后经 Command(resume=...) 续接，本节点校验、映射为 approved
decisions 并应用到 request，然后按续接方向（parse_recipes /
solve_schedule / render_ready_response）继续推进。

终止性/降级保底：
  - 仅当 confirmation_dialog_enabled 且注入 checkpointer 时图才连到
    本节点；无 checkpointer 时保持原 NEEDS_CONFIRMATION 终态（零回归）；
  - 校验失败 → confirmation_error + needs_confirmation=true（再次确认）；
  - 本节点不抛异常、不写 WorkflowError。
"""

from __future__ import annotations

from langgraph.runtime import Runtime
from langgraph.types import interrupt

from cooking_plan_agent.domain.models import QuestionAnswer
from cooking_plan_agent.repair.decisions import (
    ConfirmationAnswersError,
    answers_to_approved_decisions,
    apply_approved_decisions_structured,
)
from cooking_plan_agent.workflow.context import WorkflowContext
from cooking_plan_agent.workflow.state import PlanState

# 修改 request 内容的决策类型 —— 需要从 parse_recipes 重新推进。
_REROUTE_PARSE_TYPES: frozenset[str] = frozenset({"substitute_ingredient", "alternative_equipment", "replace_dish"})
# 仅调整排程约束的决策类型 —— 可直接重排（跳过重新解析）。
_REROUTE_SOLVE_TYPES: frozenset[str] = frozenset({"extend_time", "reduce_servings"})


def _resolve_route(decisions: object) -> str:
    """按已应用决策的类型决定续接方向（P5-4）。

    - 修改 request 内容 → "parse_recipes"（重新走解析/IR）
    - 仅调整约束 → "solve_schedule"（直接重排）
    - purchase / 无决策 → "render_ready_response"（无需修改）
    """
    if not isinstance(decisions, tuple):
        return "render_ready_response"
    has_content_change = False
    has_constraint_change = False
    for decision in decisions:
        option_type = getattr(decision, "option_type", "")
        if option_type in _REROUTE_PARSE_TYPES:
            has_content_change = True
        elif option_type in _REROUTE_SOLVE_TYPES:
            has_constraint_change = True
    if has_content_change:
        return "parse_recipes"
    if has_constraint_change:
        return "solve_schedule"
    return "render_ready_response"


def _coerce_answers(raw: object) -> tuple[QuestionAnswer, ...]:
    """把 interrupt resume 值（JSON 可序列化）归一为 QuestionAnswer 元组。"""
    if isinstance(raw, QuestionAnswer):
        return (raw,)
    if isinstance(raw, dict):
        try:
            return (QuestionAnswer.model_validate(raw),)
        except Exception:  # noqa: BLE001 —— 非法形状视为空答复
            return ()
    if isinstance(raw, (list, tuple)):
        answers: list[QuestionAnswer] = []
        for item in raw:
            if isinstance(item, QuestionAnswer):
                answers.append(item)
            elif isinstance(item, dict):
                try:
                    answers.append(QuestionAnswer.model_validate(item))
                except Exception:  # noqa: BLE001 —— 单条非法不阻塞整体
                    continue
        return tuple(answers)
    return ()


async def apply_confirmation_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, object]:
    """消费用户确认答复并决定续接方向（P5-4）。

    Returns state deltas：request / confirmation_applied / needs_confirmation
    / confirmation_route / confirmation_error。

    - 无答复且挂起中 → interrupt() 等待用户；resume 值成为答复
    - 校验失败 → confirmation_error + needs_confirmation=true（再次确认）
    - 校验通过 → 应用 decisions 产出新 request，按类型决定续接方向
    """
    request = state["request"]
    confirmation = state.get("confirmation_context")

    # 挂起-恢复：无注入答复时 interrupt 等待用户（需要 checkpointer）。
    answers = state.get("confirmation_answers", ())
    if not answers:
        payload = {
            "plan_id": request.request_id,
            "questions": (
                [q.model_dump(mode="json") for q in confirmation.confirmation_questions]
                if confirmation is not None
                else []
            ),
            "plan_revision": confirmation.plan_revision if confirmation is not None else None,
        }
        answers = _coerce_answers(interrupt(payload))
    if not answers:
        # 仍无答复 —— 保持确认状态（等待用户）。
        return {"needs_confirmation": True, "confirmation_applied": False}
    if confirmation is None:
        # 无确认表单上下文 —— 无法校验，保守保持确认。
        return {"needs_confirmation": True, "confirmation_applied": False}

    # P4-02: 校验 + 无损映射 answer -> ApprovedDecision（复用既有实现）。
    try:
        decisions = answers_to_approved_decisions(
            questions=confirmation.confirmation_questions,
            answers=answers,
            plan_revision=confirmation.plan_revision,
            presented_decisions=confirmation.decisions,
        )
    except ConfirmationAnswersError as exc:
        # 字段级错误提示（P4-02 ConfirmationAnswersError），回确认。
        return {
            "confirmation_applied": False,
            "needs_confirmation": True,
            "confirmation_error": exc.issues,
        }

    # 应用决策为纯变换（P0-06 rule 6）：安全/可行性/排程重新跑新 request。
    new_request = apply_approved_decisions_structured(request, decisions)

    return {
        "request": new_request,
        "confirmation_applied": True,
        "needs_confirmation": False,
        "confirmation_route": _resolve_route(decisions),
    }
