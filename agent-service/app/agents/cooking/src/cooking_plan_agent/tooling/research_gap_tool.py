"""P5-1: research_gap 工具。"""

from __future__ import annotations

from typing import Any

from cooking_plan_agent.tooling.schemas import RegisteredTool


def build(researcher: object) -> RegisteredTool:
    async def execute(arguments: dict[str, Any]) -> dict[str, Any]:
        from cooking_plan_agent.domain.models import EvidenceQuery

        query = EvidenceQuery(
            query_text=str(arguments["query_text"]),
            gap_type=str(arguments.get("gap_type", "critical")),
            recipe_context=str(arguments.get("recipe_context", "")),
        )
        results = await researcher.research(query)  # type: ignore[attr-defined]
        return {
            "results": [r.model_dump(mode="json") if hasattr(r, "model_dump") else r for r in results],
            "count": len(results),
        }

    return RegisteredTool(
        name="research_gap",
        description="Search for evidence to fill a recipe gap (heat/duration/temperature).",
        parameters={
            "type": "object",
            "properties": {
                "query_text": {"type": "string", "description": "Search query for the missing value."},
                "gap_type": {"type": "string", "description": "gap_class, e.g. critical."},
                "recipe_context": {"type": "string", "description": "Dish name for context."},
            },
            "required": ["query_text"],
        },
        executor=execute,
    )
