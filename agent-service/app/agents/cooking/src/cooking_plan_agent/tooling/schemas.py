"""P5-1: 工具描述与执行协议（ToolSpec / RegisteredTool）。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field


class ToolSpec(BaseModel):
    """一个可由 LLM 调用的工具描述。

    parameters 为 JSON Schema（OpenAI tool 格式的 parameters 子集）。
    """

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    description: str
    parameters: dict[str, Any]


class RegisteredTool(ToolSpec):
    """ToolSpec + 执行器。registry 暴露只读 schema，executor 供本地调用。"""

    executor: Callable[..., Awaitable[dict[str, Any]]]
