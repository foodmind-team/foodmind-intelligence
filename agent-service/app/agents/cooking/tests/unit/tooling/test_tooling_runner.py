"""P5-1: ToolRunner 分发。"""

from types import SimpleNamespace

import pytest

from cooking_plan_agent.tooling.registry import ToolRegistry
from cooking_plan_agent.tooling.runner import ToolRunner
from cooking_plan_agent.tooling.schemas import RegisteredTool


@pytest.mark.asyncio
async def test_run_unknown_tool() -> None:
    registry = ToolRegistry(SimpleNamespace())  # type: ignore[arg-type]
    runner = ToolRunner(registry)
    outcome = await runner.run("nope", {})
    assert outcome["ok"] is False
    assert "unknown tool" in outcome["error"]


@pytest.mark.asyncio
async def test_run_executor_error_is_contained() -> None:
    async def boom(arguments: dict) -> dict:
        raise ValueError("kaput")

    tool = RegisteredTool(name="boom_tool", description="d", parameters={}, executor=boom)
    registry = SimpleNamespace(get=lambda name: tool if name == "boom_tool" else None)
    runner = ToolRunner(registry)  # type: ignore[arg-type]
    outcome = await runner.run("boom_tool", {})
    assert outcome["ok"] is False
    assert "ValueError" in outcome["error"]


@pytest.mark.asyncio
async def test_run_success_merges_result() -> None:
    async def echo(arguments: dict) -> dict:
        return {"value": arguments.get("x")}

    tool = RegisteredTool(name="echo_tool", description="d", parameters={}, executor=echo)
    registry = SimpleNamespace(get=lambda name: tool if name == "echo_tool" else None)
    runner = ToolRunner(registry)  # type: ignore[arg-type]
    outcome = await runner.run("echo_tool", {"x": 42})
    assert outcome == {"ok": True, "value": 42}
