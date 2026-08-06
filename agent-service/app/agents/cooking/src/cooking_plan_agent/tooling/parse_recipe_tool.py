"""P5-1: parse_recipe 工具。"""

from __future__ import annotations

from typing import Any

from cooking_plan_agent.tooling.schemas import RegisteredTool


def build(extractor: object) -> RegisteredTool:
    async def execute(arguments: dict[str, Any]) -> dict[str, Any]:
        candidate = await extractor.extract(str(arguments["source_text"]))  # type: ignore[attr-defined]
        dumped = candidate.model_dump(mode="json") if hasattr(candidate, "model_dump") else candidate
        return {"candidate": dumped}

    return RegisteredTool(
        name="parse_recipe",
        description="Parse unstructured recipe text into a structured recipe candidate.",
        parameters={
            "type": "object",
            "properties": {"source_text": {"type": "string", "description": "Raw recipe text to parse."}},
            "required": ["source_text"],
        },
        executor=execute,
    )
