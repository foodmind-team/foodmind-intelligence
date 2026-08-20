# =============================================================================
# LangGraph 工作流编排入口（workflow 包）
# -----------------------------------------------------------------------------
# 导出公共 API：状态类型、上下文、图构建器，以及供测试使用的节点 / 路由模块。
# 公共 API：
#   - PlanState                ：在节点间传递可序列化状态的 TypedDict
#   - WorkflowContext          ：依赖注入用的冻结 dataclass（手册 8.3）
#   - build_cooking_plan_graph ：组装并编译 16 节点图的工厂函数
# =============================================================================

"""LangGraph workflow orchestration for cooking plan generation.

LangGraph 工作流编排 —— 用于生成烹饪计划。

Exports the public API: state type, context, graph builder, and node/
routing modules for testing.

导出公共 API：状态类型、上下文、图构建器，以及供测试使用的节点 / 路由模块。

Public API:
  - PlanState: TypedDict carrying serialisable state between nodes
  - WorkflowContext: frozen dataclass for dependency injection (handbook 8.3)
  - build_cooking_plan_graph: factory that assembles and compiles the 16-node graph

公共 API：
  - PlanState                ：在节点间传递可序列化状态的 TypedDict
  - WorkflowContext          ：依赖注入用的冻结 dataclass（手册 8.3）
  - build_cooking_plan_graph ：组装并编译 16 节点图的工厂函数
"""

from cooking_plan_agent.workflow.context import WorkflowContext
from cooking_plan_agent.workflow.graph import build_cooking_plan_graph
from cooking_plan_agent.workflow.state import PlanState

__all__ = [
    "PlanState",
    "WorkflowContext",
    "build_cooking_plan_graph",
]
