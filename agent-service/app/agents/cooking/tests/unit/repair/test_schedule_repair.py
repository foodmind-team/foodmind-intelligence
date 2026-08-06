"""P5-3: schedule repair 确定性策略。"""

from cooking_plan_agent.repair.schedule_repair import (
    UNREPAIRABLE_ISSUE_CODES,
    next_optimization_level,
    plan_schedule_repair,
)
from cooking_plan_agent.scheduling.models import VerificationIssue


def _issue(code: str) -> VerificationIssue:
    return VerificationIssue(code=code, message=f"issue {code}")


def test_ladder_walks_down():
    assert next_optimization_level("full") == "phase12"
    assert next_optimization_level("phase12") == "makespan"
    assert next_optimization_level("makespan") is None


def test_unknown_level_resets_to_full():
    assert next_optimization_level("bogus") == "full"


def test_plan_repair_returns_lower_level():
    action = plan_schedule_repair(
        issues=(_issue("CAPACITY_EXCEEDED"),),
        current_attempt=0,
        max_attempts=2,
        current_optimization_level="full",
    )
    assert action == {"action": "lower_optimization_level", "optimization_level": "phase12"}


def test_plan_repair_unrepairable_issue():
    assert (
        plan_schedule_repair(
            issues=(_issue("MISSING_TASK"),),
            current_attempt=0,
            max_attempts=2,
            current_optimization_level="full",
        )
        is None
    )


def test_plan_repair_exhausted():
    assert (
        plan_schedule_repair(
            issues=(_issue("CAPACITY_EXCEEDED"),),
            current_attempt=2,
            max_attempts=2,
            current_optimization_level="phase12",
        )
        is None
    )


def test_plan_repair_ladder_bottom():
    assert (
        plan_schedule_repair(
            issues=(_issue("CAPACITY_EXCEEDED"),),
            current_attempt=0,
            max_attempts=2,
            current_optimization_level="makespan",
        )
        is None
    )


def test_unrepairable_codes_non_empty():
    # 契约：模型构建类 issue 必须被显式列为不可修复。
    assert "MISSING_TASK" in UNREPAIRABLE_ISSUE_CODES
