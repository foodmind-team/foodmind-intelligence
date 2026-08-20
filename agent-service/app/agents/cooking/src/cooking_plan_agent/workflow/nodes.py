# =============================================================================
# 节点公共兼容外观模块（workflow/nodes）
# -----------------------------------------------------------------------------
# LangGraph 烹饪计划节点的公共兼容外观。实现按流水线阶段拆分，使每个模块
# 只承担一项内聚职责。保持这些再导出稳定，以便图接线与扩展使用。
# =============================================================================

"""Public compatibility façade for LangGraph cooking-plan nodes.

LangGraph 烹饪计划节点的公共兼容外观。

Implementations are split by pipeline stage so each module owns one cohesive
responsibility.  Keep these re-exports stable for graph wiring and extensions.

实现按流水线阶段拆分，使每个模块只承担一项内聚职责。保持这些再导出稳定，
以便图接线与扩展使用。
"""

from cooking_plan_agent.workflow.controller_nodes import (
    agent_controller_node,
    run_tool_node,
)
from cooking_plan_agent.workflow.conversation_nodes import (
    apply_confirmation_node,
)
from cooking_plan_agent.workflow.input_parsing import (
    detect_gaps_node,
    infer_local_node,
    parse_recipes_node,
    validate_input_node,
    validate_recipe_ir_node,
)
from cooking_plan_agent.workflow.preparation_nodes import (
    _wire_prep_consumption,
    merge_preparation_node,
)
from cooking_plan_agent.workflow.repair_nodes import (
    repair_schedule_node,
)
from cooking_plan_agent.workflow.research_nodes import (
    apply_research_evidence_node,
    research_missing_node,
)
from cooking_plan_agent.workflow.response_nodes import (
    render_failed_response_node,
    render_infeasible_response_node,
    render_ready_response_node,
)
from cooking_plan_agent.workflow.safety_nodes import (
    build_confirmation_response_node,
    check_feasibility_node,
    validate_safety_node,
)
from cooking_plan_agent.workflow.scheduling_nodes import (
    _build_schedule_summary,
    _deterministic_explanation,
    _max_parallel_active,
    _solver_timeout,
    build_task_graph_node,
    explain_schedule_node,
    solve_schedule_node,
    verify_schedule_node,
)

__all__ = [
    "_build_schedule_summary",
    "_deterministic_explanation",
    "_max_parallel_active",
    "_solver_timeout",
    "_wire_prep_consumption",
    "agent_controller_node",
    "apply_confirmation_node",
    "apply_research_evidence_node",
    "build_confirmation_response_node",
    "build_task_graph_node",
    "check_feasibility_node",
    "detect_gaps_node",
    "explain_schedule_node",
    "infer_local_node",
    "merge_preparation_node",
    "parse_recipes_node",
    "render_failed_response_node",
    "render_infeasible_response_node",
    "render_ready_response_node",
    "repair_schedule_node",
    "research_missing_node",
    "run_tool_node",
    "solve_schedule_node",
    "validate_input_node",
    "validate_recipe_ir_node",
    "validate_safety_node",
    "verify_schedule_node",
]
