"""P5-3: schedule 反思修复节点。

失败 → 诊断（确定性规则，可叠加 LLM 摘要）→ 记录 repair_history →
路由决定重试求解或 FAILED。本节点永不抛异常、永不写 WorkflowError。
"""

from __future__ import annotations

from langgraph.runtime import Runtime

from cooking_plan_agent.config.settings import get_settings
from cooking_plan_agent.repair.schedule_repair import plan_schedule_repair
from cooking_plan_agent.scheduling.models import RepairAttemptRecord
from cooking_plan_agent.workflow.context import WorkflowContext
from cooking_plan_agent.workflow.state import PlanState


async def repair_schedule_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, object]:
    """对验证失败做修复决策（walk down optimization ladder）。

    Returns state deltas：repair_attempts / repair_history / solver_overrides
    （保留位：后续可叠加 LLM 诊断，见任务 3.5）。
    """
    report = state.get("verification_report")
    if report is None or report.passed:
        return {}

    settings = get_settings()
    attempts = state.get("repair_attempts", 0)
    overrides = dict(state.get("solver_overrides", {}))
    current_level = str(overrides.get("optimization_level") or settings.solver_optimization_level)

    action = plan_schedule_repair(
        issues=report.issues,
        current_attempt=attempts,
        max_attempts=settings.schedule_repair_max_attempts,
        current_optimization_level=current_level,
    )

    issue_codes = tuple(issue.code for issue in report.issues)
    history = state.get("repair_history", ())
    if action is None:
        record = RepairAttemptRecord(
            attempt=attempts + 1, issues=issue_codes, action="give_up", outcome="gave_up"
        )
        return {
            "repair_attempts": attempts + 1,
            "repair_history": history + (record,),
        }

    overrides["optimization_level"] = str(action["optimization_level"])
    record = RepairAttemptRecord(
        attempt=attempts + 1,
        issues=issue_codes,
        action=str(action["action"]),
        outcome="retrying",
    )
    return {
        "repair_attempts": attempts + 1,
        "repair_history": history + (record,),
        "solver_overrides": overrides,
    }
