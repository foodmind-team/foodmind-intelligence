"""Workflow node implementations for a single pipeline stage.

The public compatibility surface remains ``cooking_plan_agent.workflow.nodes``.
This module contains one cohesive stage only.
"""

import logging

from langgraph.runtime import Runtime

from cooking_plan_agent.workflow.context import WorkflowContext
from cooking_plan_agent.workflow.state import PlanState

# Preserve the established logger name as part of the observable diagnostics
# contract while the implementation moves out of the legacy façade module.
logger = logging.getLogger("cooking_plan_agent.workflow.nodes")


async def render_ready_response_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, object]:
    """Render READY response with verified schedule, mise en place, and checklist.

    Delegates to rendering.responses.render_ready_response.
    """
    from cooking_plan_agent.rendering.responses import render_ready_response

    response = render_ready_response(state)
    return {"response": response}


async def render_infeasible_response_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, object]:
    """Render INFEASIBLE response with ordered reasons from all sources.

    Delegates to rendering.responses.render_infeasible_response.
    """
    from cooking_plan_agent.rendering.responses import render_infeasible_response

    response = render_infeasible_response(state)
    return {"response": response}


async def render_failed_response_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, object]:
    """Render FAILED response with stable error code and correlation ID.

    Delegates to rendering.responses.render_failed_response.

    P2-03: emits a structured diagnostic log line (error code, node,
    correlation ID, controlled diagnostics) so failures can be traced via
    correlation ID without writing the raw internal message to the log.
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
