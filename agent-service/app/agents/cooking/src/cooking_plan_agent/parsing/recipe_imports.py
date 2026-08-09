"""Deterministic multi-dish parsing for recipe-import fallback."""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from cooking_plan_agent.domain.recipe_imports import RecipeImportDraft
from cooking_plan_agent.parsing.extractor import RecipeExtractor

_SEPARATOR = re.compile(r"(?m)^\s*(?:-{3,}|={3,})\s*$")
_RECIPE_HEADING = re.compile(r"^\s*(?:recipe|dish)\s*:\s*(.+?)\s*$", re.IGNORECASE)
_MARKDOWN_DISH_HEADING = re.compile(r"^\s*#\s+(.+?)\s*$")
_SERVINGS = re.compile(
    r"\b(?:serves?|servings?|makes?|yield)\s*(?::|for)?\s*(\d{1,2})\b|\b(\d{1,2})\s+servings?\b",
    re.IGNORECASE,
)


@runtime_checkable
class RecipeImportExtractor(Protocol):
    async def extract(self, text: str) -> tuple[RecipeImportDraft, ...]: ...


def split_recipe_blocks(text: str) -> tuple[str, ...]:
    """Split common multi-recipe text formats without guessing dish content."""

    separated = tuple(part.strip() for part in _SEPARATOR.split(text) if part.strip())
    if len(separated) > 1:
        return separated

    lines = text.splitlines()
    heading_indexes = [
        index
        for index, line in enumerate(lines)
        if _RECIPE_HEADING.match(line) or _MARKDOWN_DISH_HEADING.match(line)
    ]
    if len(heading_indexes) <= 1:
        return (text.strip(),)

    blocks: list[str] = []
    for position, start in enumerate(heading_indexes):
        end = heading_indexes[position + 1] if position + 1 < len(heading_indexes) else len(lines)
        block = "\n".join(lines[start:end]).strip()
        if block:
            blocks.append(block)
    return tuple(blocks)


def _normalise_heading(block: str) -> str:
    lines = block.splitlines()
    if not lines:
        return block
    match = _RECIPE_HEADING.match(lines[0]) or _MARKDOWN_DISH_HEADING.match(lines[0])
    if match:
        lines[0] = match.group(1).strip()
    return "\n".join(lines)


def _explicit_servings(block: str) -> int | None:
    match = _SERVINGS.search(block)
    if not match:
        return None
    value = int(match.group(1) or match.group(2))
    return value if 1 <= value <= 50 else None


class DeterministicRecipeImportExtractor:
    """Parse recognised recipe sections through the existing rule extractor."""

    def __init__(self, recipe_extractor: RecipeExtractor | None = None) -> None:
        self._recipe_extractor = recipe_extractor or RecipeExtractor()

    async def extract(self, text: str) -> tuple[RecipeImportDraft, ...]:
        drafts: list[RecipeImportDraft] = []
        for index, raw_block in enumerate(split_recipe_blocks(text), start=1):
            block = _normalise_heading(raw_block)
            candidate = await self._recipe_extractor.extract(block)
            name = candidate.dish_name.strip() if candidate.dish_name else None
            if name in {"Untitled Recipe", "Untitled"}:
                name = None
            drafts.append(
                RecipeImportDraft(
                    draft_id=f"dish-{index}",
                    name=name,
                    servings=_explicit_servings(raw_block),
                    ingredients=tuple(
                        ingredient.raw_text.strip()
                        for ingredient in candidate.ingredients
                        if ingredient.raw_text.strip()
                    ),
                    steps=tuple(step.instruction.strip() for step in candidate.steps if step.instruction.strip()),
                )
            )
        return tuple(drafts)
