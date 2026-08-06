"""P5-1: evaluate_safety 工具。"""

from __future__ import annotations

from typing import Any

from cooking_plan_agent.tooling.schemas import RegisteredTool


def build(engine: object) -> RegisteredTool:
    async def execute(arguments: dict[str, Any]) -> dict[str, Any]:
        from cooking_plan_agent.domain.models import RecipeIR, SafetyContext

        recipes = tuple(RecipeIR.model_validate(r) for r in arguments.get("recipes", ()))
        context = SafetyContext(
            recipes=recipes,
            dietary_restrictions=tuple(arguments.get("dietary_restrictions", ())),
            user_allergens=tuple(arguments.get("user_allergens", ())),
            cooking_date=arguments.get("cooking_date"),
        )
        report = engine.evaluate(context)  # type: ignore[attr-defined]
        return {"safety_report": report.model_dump(mode="json")}

    return RegisteredTool(
        name="evaluate_safety",
        description="Evaluate food-safety rules against parsed recipes under a regional policy.",
        parameters={
            "type": "object",
            "properties": {
                "recipes": {"type": "array", "items": {"type": "object"}, "description": "Serialised RecipeIR list."},
                "dietary_restrictions": {"type": "array", "items": {"type": "string"}},
                "user_allergens": {"type": "array", "items": {"type": "string"}},
                "cooking_date": {"type": ["string", "null"], "description": "ISO date."},
            },
            "required": ["recipes"],
        },
        executor=execute,
    )
