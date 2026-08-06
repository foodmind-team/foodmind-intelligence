"""P5-1: 从 WorkflowContext 构建工具注册表。"""

from __future__ import annotations

from cooking_plan_agent.tooling.schemas import RegisteredTool
from cooking_plan_agent.workflow.context import WorkflowContext


class ToolRegistry:
    """把 WorkflowContext 中的服务能力暴露为可调用工具集。

    每个工具只读调用所需参数、返回可序列化 dict；工具内部不做决策，
    决策由调用方（LLM 控制器 / 用户）完成。
    """

    def __init__(self, context: WorkflowContext) -> None:
        self._context = context
        self._tools: tuple[RegisteredTool, ...] = self._build()

    def _build(self) -> tuple[RegisteredTool, ...]:
        """按 context 中可用的服务注册工具（P5-1）。"""
        from cooking_plan_agent.tooling import (
            feasibility_tool,
            parse_recipe_tool,
            research_gap_tool,
            safety_tool,
            scheduling_tool,
        )

        tools: list[RegisteredTool] = []

        extractor = getattr(self._context, "recipe_extractor", None)
        if extractor is not None:
            tools.append(parse_recipe_tool.build(extractor))

        researcher = getattr(self._context, "recipe_researcher", None)
        if researcher is not None:
            tools.append(research_gap_tool.build(researcher))

        safety_engine = getattr(self._context, "safety_engine", None)
        if safety_engine is not None:
            tools.append(safety_tool.build(safety_engine))

        # 无状态工具总是注册。
        tools.append(feasibility_tool.build())
        tools.append(scheduling_tool.build_solve_schedule())
        tools.append(scheduling_tool.build_verify_schedule())

        return tuple(tools)

    def specs(self) -> tuple[RegisteredTool, ...]:
        """暴露给 LLM 的完整工具描述（含 schema）。"""
        return self._tools

    def get(self, name: str) -> RegisteredTool | None:
        for tool in self._tools:
            if tool.name == name:
                return tool
        return None
