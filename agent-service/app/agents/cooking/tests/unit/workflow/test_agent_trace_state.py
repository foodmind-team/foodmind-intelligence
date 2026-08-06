"""P5: PlanState 通用 agent 追踪字段（阶段 0 基座）。"""
from typing import Any

from cooking_plan_agent.domain.models import StrictModel


class _AgentTraceRecord(StrictModel):
    """State 中每条行动的留痕。"""

    step: int
    action: str          # 例如 "repair", "tool_call", "question"
    detail: dict[str, Any] = {}


def test_trace_record_default_detail():
    record = _AgentTraceRecord(step=1, action="repair")
    assert record.detail == {}


def test_plan_state_accepts_trace_fields():
    # PlanState 是 total=False 的 TypedDict —— 验证新增字段可通过 **state 构造。
    from cooking_plan_agent.workflow.state import PlanState

    state: PlanState = {
        "agent_trace": (
            _AgentTraceRecord(step=1, action="repair").model_dump(),
        ),
    }
    assert state["agent_trace"][0]["action"] == "repair"
