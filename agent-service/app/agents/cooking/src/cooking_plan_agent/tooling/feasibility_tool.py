"""P5-1: check_feasibility 工具。"""

from __future__ import annotations

from typing import Any

from cooking_plan_agent.tooling.schemas import RegisteredTool


def build() -> RegisteredTool:
    async def execute(arguments: dict[str, Any]) -> dict[str, Any]:
        from cooking_plan_agent.domain.models import IngredientDemand, InventoryLotSnapshot
        from cooking_plan_agent.inventory.feasibility import check_all_inventory

        demands = tuple(IngredientDemand.model_validate(d) for d in arguments.get("demands", ()))
        lots = tuple(InventoryLotSnapshot.model_validate(lot) for lot in arguments.get("lots", ()))
        report = check_all_inventory(demands, lots, cooking_date=arguments.get("cooking_date"))
        return {"feasibility_report": report.model_dump(mode="json")}

    return RegisteredTool(
        name="check_feasibility",
        description="Check ingredient inventory sufficiency with FEFO allocation.",
        parameters={
            "type": "object",
            "properties": {
                "demands": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Serialised IngredientDemand list.",
                },
                "lots": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Serialised InventoryLotSnapshot list.",
                },
                "cooking_date": {"type": ["string", "null"]},
            },
            "required": ["demands"],
        },
        executor=execute,
    )
