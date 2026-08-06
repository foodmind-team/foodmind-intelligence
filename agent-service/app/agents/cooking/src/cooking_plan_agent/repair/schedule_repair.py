"""P5-3: schedule 反思修复的确定性决策。

设计红线：修复动作只由规则产生（walk down optimization ladder），
LLM（schedule_repair_llm_enabled）至多提供诊断摘要，不直接产生动作。
"""

from __future__ import annotations

from cooking_plan_agent.scheduling.models import VerificationIssue

# 优化深度阶梯：与 SOLVER_OPTIMIZATION_LEVEL 语义一致，逐级降低。
OPTIMIZATION_LADDER: tuple[str, ...] = ("full", "phase12", "makespan")

# 模型构建缺陷类 issue：不是优化深度问题，重试无法修复。
UNREPAIRABLE_ISSUE_CODES: frozenset[str] = frozenset({"MISSING_TASK"})


def next_optimization_level(current: str) -> str | None:
    """返回当前深度在阶梯上的下一级；已到最底（或未知）返回 None。

    未知级别视为旧配置，重置到 full。
    """
    try:
        idx = OPTIMIZATION_LADDER.index(current)
    except ValueError:
        return OPTIMIZATION_LADDER[0]
    if idx >= len(OPTIMIZATION_LADDER) - 1:
        return None
    return OPTIMIZATION_LADDER[idx + 1]


def plan_schedule_repair(
    issues: tuple[VerificationIssue, ...],
    current_attempt: int,
    max_attempts: int,
    current_optimization_level: str,
) -> dict[str, object] | None:
    """根据验证失败 issue 决定下一次修复动作。

    Returns:
        修复动作 dict（{"action": str, "optimization_level": str}）；
        不可修复 / 重试耗尽 / 阶梯到底时返回 None（路由到 FAILED）。
    """
    if current_attempt >= max_attempts:
        return None
    if any(issue.code in UNREPAIRABLE_ISSUE_CODES for issue in issues):
        return None
    level = next_optimization_level(current_optimization_level)
    if level is None:
        return None
    return {"action": "lower_optimization_level", "optimization_level": level}
