"""P5-1: 工具调用分发器。"""

from __future__ import annotations

from typing import Any

from cooking_plan_agent.tooling.registry import ToolRegistry


class ToolRunner:
    """按 name 分发工具调用，异常收敛为可序列化错误结果。"""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    async def run(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        tool = self._registry.get(name)
        if tool is None:
            return {"ok": False, "error": f"unknown tool: {name}"}
        try:
            result = await tool.executor(arguments)
            return {"ok": True, **result}
        except Exception as exc:  # noqa: BLE001 —— 边界收敛，LLM 可读错误
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
