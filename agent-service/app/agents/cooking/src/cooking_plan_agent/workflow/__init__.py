"""LangGraph workflow orchestration for cooking plan generation.

Exports the public API: state type, context, graph builder, and node/
routing modules for testing.

Public API:
  - PlanState: TypedDict carrying serialisable state between nodes
  - WorkflowContext: frozen dataclass for dependency injection (handbook 8.3)
  - build_cooking_plan_graph: factory that assembles and compiles the 16-node graph
"""

from cooking_plan_agent.workflow.context import WorkflowContext
from cooking_plan_agent.workflow.graph import build_cooking_plan_graph
from cooking_plan_agent.workflow.state import PlanState

__all__ = [
    "PlanState",
    "WorkflowContext",
    "build_cooking_plan_graph",
]
