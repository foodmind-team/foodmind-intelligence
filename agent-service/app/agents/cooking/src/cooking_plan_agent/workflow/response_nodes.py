# =============================================================================
# 响应渲染节点（workflow/response_nodes）
# -----------------------------------------------------------------------------
# 渲染三个终态响应：READY（已验证排程 + mise en place + 清单）、
# INFEASIBLE（多来源有序原因）、FAILED（稳定错误码 + correlation ID）。
# 仅含一个内聚阶段；公共兼容面仍为 cooking_plan_agent.workflow.nodes。
# =============================================================================

"""Workflow node implementations for a single pipeline stage.

单个流水线阶段的节点实现。

The public compatibility surface remains ``cooking_plan_agent.workflow.nodes``.
This module contains one cohesive stage only.

公共兼容面仍为 ``cooking_plan_agent.workflow.nodes``。本模块仅包含一个内聚阶段。
"""

import logging

from langgraph.runtime import Runtime

from cooking_plan_agent.workflow.context import WorkflowContext
from cooking_plan_agent.workflow.state import PlanState

# Preserve the established logger name as part of the observable diagnostics
# contract while the implementation moves out of the legacy façade module.
# 保留既有 logger 名称，作为可观测诊断契约的一部分，同时实现移出旧的外观模块。
logger = logging.getLogger("cooking_plan_agent.workflow.nodes")


async def render_ready_response_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, object]:
    """Render READY response with verified schedule, mise en place, and checklist.

    渲染 READY 响应 —— 包含已验证排程、mise en place 与检查清单。

    Delegates to rendering.responses.render_ready_response.

    委托给 rendering.responses.render_ready_response。
    """
    from cooking_plan_agent.rendering.responses import render_ready_response

    response = render_ready_response(state)
    return {"response": response}


async def render_infeasible_response_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, object]:
    """Render INFEASIBLE response with ordered reasons from all sources.

    渲染 INFEASIBLE 响应 —— 包含来自所有来源的有序原因。

    Delegates to rendering.responses.render_infeasible_response.

    委托给 rendering.responses.render_infeasible_response。
    """
    from cooking_plan_agent.rendering.responses import render_infeasible_response

    response = render_infeasible_response(state)
    return {"response": response}


async def render_failed_response_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, object]:
    """Render FAILED response with stable error code and correlation ID.

    渲染 FAILED 响应 —— 包含稳定错误码与 correlation ID。

    Delegates to rendering.responses.render_failed_response.

    委托给 rendering.responses.render_failed_response。

    P2-03: emits a structured diagnostic log line (error code, node,
    correlation ID, controlled diagnostics) so failures can be traced via
    correlation ID without writing the raw internal message to the log.

    P2-03：输出一条结构化诊断日志（错误码、节点、correlation ID、受控诊断信息），
    使失败可通过 correlation ID 追溯，同时不把原始内部消息写入日志。
    """
    from cooking_plan_agent.rendering.responses import render_failed_response

    error = state.get("error")
    if error is not None:
        logger.warning(
            "Workflow FAILED | error_code=%s | node=%s | correlation_id=%s | recoverable=%s | diagnostics=%s",
            error.error_code,
            error.node_name,
            error.correlation_id,
            error.recoverable,
            error.diagnostics,
        )

    response = render_failed_response(state)
    return {"response": response}
