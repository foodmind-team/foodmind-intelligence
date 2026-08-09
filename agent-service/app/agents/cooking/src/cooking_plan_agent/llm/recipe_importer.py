"""LLM-backed multi-dish recipe import extraction."""

from __future__ import annotations

from typing import Any

from cooking_plan_agent.domain.recipe_imports import RecipeImportDraft
from cooking_plan_agent.llm.client import LLMClient, LLMError
from cooking_plan_agent.parsing.recipe_imports import DeterministicRecipeImportExtractor

_SYSTEM_PROMPT = (
    "You extract one or more recipes from English user text. Return one JSON object only with "
    "a recipes array. Each recipe must contain exactly: name (string or null), servings "
    "(whole number or null), ingredients (array of strings), and steps (array of strings). "
    "Preserve input order. Never invent a dish name, serving count, ingredient, or step that "
    "the user did not provide; use null or an empty array when required information is missing."
)


class LLMRecipeImportExtractor:
    """Extract a bounded list of partial import drafts with safe fallback."""

    def __init__(self, client: LLMClient, fallback: DeterministicRecipeImportExtractor | None = None) -> None:
        self._client = client
        self._fallback = fallback or DeterministicRecipeImportExtractor()

    async def extract(self, text: str) -> tuple[RecipeImportDraft, ...]:
        try:
            payload = await self._client.chat_json(
                [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ]
            )
            recipes = payload.get("recipes")
            if not isinstance(recipes, list) or not recipes:
                raise LLMError("Recipe import response did not contain recipes")
            return tuple(self._draft(index, item) for index, item in enumerate(recipes, start=1) if isinstance(item, dict))
        except (LLMError, TypeError, ValueError):
            return await self._fallback.extract(text)

    @staticmethod
    def _draft(index: int, value: dict[str, Any]) -> RecipeImportDraft:
        raw_servings = value.get("servings")
        servings: int | None = None
        if isinstance(raw_servings, int) and not isinstance(raw_servings, bool) and 1 <= raw_servings <= 50:
            servings = raw_servings
        name_value = value.get("name")
        name = str(name_value).strip()[:160] if isinstance(name_value, str) and name_value.strip() else None
        ingredients = tuple(
            str(item).strip()[:500]
            for item in value.get("ingredients", [])
            if isinstance(item, str) and item.strip()
        )[:100]
        steps = tuple(
            str(item).strip()[:1000]
            for item in value.get("steps", [])
            if isinstance(item, str) and item.strip()
        )[:100]
        return RecipeImportDraft(
            draft_id=f"dish-{index}",
            name=name,
            servings=servings,
            ingredients=ingredients,
            steps=steps,
        )
