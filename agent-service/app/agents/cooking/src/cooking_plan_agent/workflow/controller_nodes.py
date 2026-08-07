"""P5-2: ReAct 控制器节点。

控制器循环：agent_controller（LLM 决策）→ run_tool（执行）→ 观察回填
→ agent_controller。两条硬保障：
  1. 终止性：agent_step >= agent_max_steps 强制回退确定性 DAG；
  2. 降级保底：控制器缺失 / 未启用 / 抛异常 / 非法决策 → agent_mode=
     "deterministic"，交回原确定性 16 节点 DAG，绝不静默放行。
本模块节点永不抛异常、永不写 WorkflowError。
"""

from __future__ import annotations

from langgraph.runtime import Runtime

from cooking_plan_agent.config.settings import get_settings
from cooking_plan_agent.workflow.context import AgentController, WorkflowContext
from cooking_plan_agent.workflow.state import PlanState

# 控制器决策的合法 type 值（防止 LLM 越界动作）。
_VALID_DECISION_TYPES: frozenset[str] = frozenset({"tool_call", "final", "fallback"})


def _state_summary(state: PlanState) -> dict[str, object]:
    """构建传给控制器的紧凑、非敏感状态摘要（D4）。

    只包含请求 ID、当前步数/模式与已观察计数，绝不携带菜谱原文、
    库存明细或用户身份等敏感信息。
    """
    request = state.get("request")
    return {
        "request_id": request.request_id if request is not None else None,
        "agent_step": state.get("agent_step", 0),
        "agent_mode": state.get("agent_mode", "deterministic"),
        "tool_call_count": len(state.get("tool_calls", ())),
        "observation_count": len(state.get("observations", ())),
        "has_error": state.get("error") is not None,
    }


def _apply_decision(
    decision: dict[str, object],
    step: int,
) -> dict[str, object]:
    """将控制器决策落地为状态 delta。

    - tool_call / final：写入 pending_decision 供路由与 run_tool_node 消费，
      并追加 agent_trace 留痕；
    - fallback / 非法 type / 非法结构：回退确定性 DAG（agent_mode）。
    """
    decision_type = decision.get("type")
    if decision_type not in _VALID_DECISION_TYPES:
        return {"agent_mode": "deterministic"}
    if decision_type == "fallback":
        return {"agent_mode": "deterministic"}
    if decision_type in ("tool_call", "final"):
        if not isinstance(decision.get("tool" if decision_type == "tool_call" else "response"), (str, dict)):
            return {"agent_mode": "deterministic"}
        return {
            "pending_decision": decision,
            "agent_trace": ({"step": step, "action": decision_type, "detail": decision},),
        }
    return {"agent_mode": "deterministic"}


async def agent_controller_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, object]:
    """LLM 控制器一步决策（P5-2）。

    返回 state deltas：pending_decision / agent_mode / agent_trace。
    任何失败路径都收敛为 agent_mode="deterministic"（回退原 DAG）。
    """
    controller: AgentController | None = getattr(runtime.context, "agent_controller", None)
    settings = get_settings()
    step = state.get("agent_step", 0)

    if controller is None or not settings.agent_controller_enabled:
        return {"agent_mode": "deterministic"}
    if step >= settings.agent_max_steps:
        # 终止性保障：步数耗尽强制回退，不产生副作用残留。
        return {"agent_mode": "deterministic"}

    try:
        decision = await controller.decide(_state_summary(state))
    except Exception:  # noqa: BLE001 —— 控制器失败必须回退，不影响主流程
        return {"agent_mode": "deterministic"}

    return _apply_decision(decision, step)


async def run_tool_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, object]:
    """执行控制器选中的工具，observation 回填状态（P5-2）。

    未知工具 / 执行异常统一由 ToolRunner 收敛为 ok=False 观察，
    控制器可读到错误并修正 —— 不中断循环、不产生副作用残留。
    """
    from cooking_plan_agent.tooling.registry import ToolRegistry
    from cooking_plan_agent.tooling.runner import ToolRunner

    decision = state.get("pending_decision") or {}
    tool_name = str(decision.get("tool", ""))
    raw_arguments = decision.get("arguments")
    arguments = dict(raw_arguments) if isinstance(raw_arguments, dict) else {}

    runner = ToolRunner(ToolRegistry(runtime.context))
    outcome = await runner.run(tool_name, arguments)

    return {
        "tool_calls": state.get("tool_calls", ()) + ({"tool": tool_name},),
        "observations": state.get("observations", ()) + (outcome,),
        "agent_step": state.get("agent_step", 0) + 1,
        # 决策已消费，清空以避免重复执行。
        "pending_decision": {},
    }
