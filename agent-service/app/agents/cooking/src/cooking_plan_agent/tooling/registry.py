"""P5-1: 从 WorkflowContext 构建工具注册表。"""

from __future__ import annotations

from typing import Any

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
        """按 context 中可用的服务注册工具（当前：parse_recipe）。

        后续任务将把每个工具抽到独立模块（tooling/xxx_tool.py）；
        本任务先内联最小实现，保证基座独立可交付。
        """
        tools: list[RegisteredTool] = []

        extractor = getattr(self._context, "recipe_extractor", None)
        if extractor is not None:
            tools.append(self._build_parse_recipe(extractor))

        return tuple(tools)

    @staticmethod
    def _build_parse_recipe(extractor: object) -> RegisteredTool:
        async def execute(arguments: dict[str, Any]) -> dict[str, Any]:
            candidate = await extractor.extract(str(arguments["source_text"]))  # type: ignore[attr-defined]
            dumped = candidate.model_dump() if hasattr(candidate, "model_dump") else candidate
            return {"candidate": dumped}

        return RegisteredTool(
            name="parse_recipe",
            description="Parse unstructured recipe text into a structured recipe candidate.",
            parameters={
                "type": "object",
                "properties": {
                    "source_text": {"type": "string", "description": "Raw recipe text to parse."}
                },
                "required": ["source_text"],
            },
            executor=execute,
        )

    def specs(self) -> tuple[RegisteredTool, ...]:
        """暴露给 LLM 的完整工具描述（含 schema）。"""
        return self._tools

    def get(self, name: str) -> RegisteredTool | None:
        for tool in self._tools:
            if tool.name == name:
                return tool
        return None
