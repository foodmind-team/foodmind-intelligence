"""Deterministic multi-dish parsing for recipe-import fallback."""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from cooking_plan_agent.domain.models import ExtractedRecipeCandidate
from cooking_plan_agent.domain.recipe_imports import RecipeImportDraft
from cooking_plan_agent.normalisation.names import clean_dish_name
from cooking_plan_agent.parsing.extractor import RecipeExtractor
from cooking_plan_agent.parsing.preprocess import collapse_blank_lines, normalise_line_endings

_SEPARATOR = re.compile(r"(?m)^\s*(?:-{3,}|={3,})\s*$")
# "Recipe: Lemon Pasta", "Recipe 1: Lemon Pasta", "Dish 2 — Stew"…
_RECIPE_HEADING = re.compile(
    r"^\s*(?:recipe|dish|菜谱|食谱|料理|レシピ|요리|receta|receita|recette|rezept|ricetta)"
    r"\s*(?:\d+\s*)?[：:]?\s*(?:[-–—]\s*)?(.*)$",
    re.IGNORECASE,
)
_MARKDOWN_DISH_HEADING = re.compile(r"^\s*#\s+(.+?)\s*$")
# "Ingredients Preparation", "I. Ingredients Preparation" — a common pasted
# recipe template heading. Deliberately NOT matching "Ingredients:" sections.
_PREP_HEADING = re.compile(r"^\s*(?:[IVX]{1,3}\.\s*)?ingredients?\s+preparation\b", re.IGNORECASE)
# "Main ingredients:" / "- Main ingredients" template section headings (own line).
_MAIN_INGREDIENTS_HEADING = re.compile(r"^\s*-?\s*(?:main\s+)?ingredients?\s*[：:]*\s*$", re.IGNORECASE)
# "Main ingredients: Fresh shrimp (，…)" — dish name glued after the colon.
_MAIN_INGREDIENTS_INLINE = re.compile(r"^\s*-?\s*(?:main\s+)?ingredients?\s*[：:]\s*(.+)$", re.IGNORECASE)
_SERVINGS = re.compile(
    r"\b(?:serves?|servings?|makes?|yield|para|pour|für)\s*(?::|for)?\s*(\d{1,2})\b"
    r"|\b(\d{1,2})\s*(?:servings?|portions?|porciones?|raciones?|porções?|persone?|personen?)\b"
    r"|(?<!\d)(\d{1,2})\s*(?:人份量?|人分|人|份|인분)(?!\d)",
    re.IGNORECASE,
)
# Two or more consecutive blank lines = a likely dish boundary when the text
# carries no explicit "---" / "Recipe:" separator (common for copy-paste).
_DOUBLE_BLANK = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)+")


@runtime_checkable
class RecipeImportExtractor(Protocol):
    async def extract(self, text: str) -> tuple[RecipeImportDraft, ...]: ...


def clean_recipe_text(text: str) -> str:
    """Normalise a pasted multi-dish text before any splitting or extraction.

    Reuses the pipeline's deterministic cleaning stages so every recipe-import
    path (LLM, fan-out, rule fallback) sees the same line endings and blank
    lines — otherwise copy-paste noise would break the block heuristics.
    """

    return collapse_blank_lines(normalise_line_endings(text))


def split_recipe_blocks(text: str) -> tuple[str, ...]:
    """Split common multi-recipe text formats without guessing dish content.

    Recognised boundaries, in priority order:
      1. Explicit separators ("---" / "===")
      2. "Recipe:" / "Dish:" headings (with or without a number), Markdown
         "#" headings, and "Ingredients Preparation" template headings
      3. Two or more consecutive blank lines (copy-paste style)

    Returns a single block when no reliable boundary is found.
    """

    separated = tuple(part.strip() for part in _SEPARATOR.split(text) if part.strip())
    if len(separated) > 1:
        return separated

    lines = text.splitlines()
    heading_indexes = [
        index
        for index, line in enumerate(lines)
        if _RECIPE_HEADING.match(line) or _MARKDOWN_DISH_HEADING.match(line) or _PREP_HEADING.match(line)
    ]
    if len(heading_indexes) > 1:
        blocks: list[str] = []
        for position, start in enumerate(heading_indexes):
            end = heading_indexes[position + 1] if position + 1 < len(heading_indexes) else len(lines)
            block = "\n".join(lines[start:end]).strip()
            if block:
                blocks.append(block)
        return tuple(blocks)

    # No explicit separator — try the blank-line heuristic as a last resort.
    blank_split = tuple(part.strip() for part in _DOUBLE_BLANK.split(text) if part.strip())
    if len(blank_split) > 1:
        return blank_split

    return (text.strip(),)


def split_on_markers(text: str, markers: tuple[str, ...]) -> tuple[str, ...]:
    """Split text at each marker's occurrence, in order.

    Used by the LLM dish-splitting path: the model returns the first line of
    every dish and this locates those substrings in the original text, so no
    text is ever rewritten or lost. Substring (not line) matching also splits
    pasted text where the next dish's heading is glued onto the previous
    dish's last line (e.g. "...serve. Ingredients Preparation").
    """

    lower = text.lower()
    starts: list[int] = []
    cursor = 0
    for marker in markers:
        needle = marker.strip().lower()
        if not needle:
            continue
        position = lower.find(needle, cursor)
        if position < 0:
            continue
        starts.append(position)
        cursor = position + len(needle)
    if len(starts) < 2:
        return (text,)

    blocks: list[str] = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(text)
        block = text[start:end].strip()
        if block:
            blocks.append(block)
    return tuple(blocks)


# "ingredients preparation" (optionally "I. Ingredients Preparation") is the
# pasted-template dish heading itself. Splitting on it as a SUBSTRING (not a
# line start) also catches headings glued to the previous dish's last line.
_PREP_BOUNDARY = re.compile(r"(?:[IVX]{1,3}\.\s*)?ingredients?\s+preparation\b", re.IGNORECASE)


def expand_prep_boundaries(block: str) -> tuple[str, ...]:
    """Split a block on "Ingredients Preparation" headings wherever they occur.

    Deterministic coarse cut for template pastes: each occurrence of the
    heading starts a new dish. Glued headings ("...serve. Ingredients
    Preparation") are still caught because matching is substring-based.
    Returns the input unchanged when there is at most one occurrence.
    """

    matches = list(_PREP_BOUNDARY.finditer(block))
    if len(matches) <= 1:
        return (block,)
    starts = [match.start() for match in matches]
    # Segment k runs [0|starts[k], starts[k+1]|len): the first segment keeps
    # any content before the first heading, later segments start at their own
    # heading so "…serve. Ingredients Preparation" never leaks into a dish.
    seg_starts = [0, *starts[1:]]
    seg_ends = [*starts[1:], len(block)]
    parts: list[str] = []
    for start, end in zip(seg_starts, seg_ends, strict=True):
        part = block[start:end].strip()
        if part:
            parts.append(part)
    return tuple(parts)


def _normalise_heading(block: str) -> str:
    lines = block.splitlines()
    if not lines:
        return block
    # "Recipe: X" / "# X" → keep X as the dish name.
    match = _RECIPE_HEADING.match(lines[0]) or _MARKDOWN_DISH_HEADING.match(lines[0])
    if match:
        lines[0] = match.group(1).strip()
    # Drop pasted template headings ("Ingredients Preparation",
    # "Main ingredients:") so the extractor sees the actual dish name; when
    # the name is glued to the heading line ("Main ingredients: Crab legs")
    # keep only the name.
    while lines:
        stripped = lines[0].strip()
        if _PREP_HEADING.match(stripped) or _MAIN_INGREDIENTS_HEADING.match(stripped):
            lines.pop(0)
            continue
        inline = _MAIN_INGREDIENTS_INLINE.match(stripped)
        if inline:
            name = inline.group(1).strip()
            if name:
                lines[0] = name
            else:
                lines.pop(0)
            continue
        break
    return "\n".join(lines)


def _explicit_servings(block: str) -> int | None:
    match = _SERVINGS.search(block)
    if not match:
        return None
    value = int(next(group for group in match.groups() if group is not None))
    return value if 1 <= value <= 50 else None


def _candidate_to_draft(
    index: int,
    raw_block: str,
    candidate: ExtractedRecipeCandidate,
) -> RecipeImportDraft:
    """Map a full extracted candidate into a partial import draft.

    Shared by the deterministic extractor and the LLM fan-out path so both
    produce identical ``RecipeImportDraft`` semantics: servings are only
    recorded when the raw text explicitly states them (never a rule default),
    and free-text names are surfaced verbatim.
    """

    name = clean_dish_name(candidate.dish_name) if candidate.dish_name else None
    if name in {"Untitled Recipe", "Untitled"}:
        name = None
    return RecipeImportDraft(
        draft_id=f"dish-{index}",
        name=name,
        servings=_explicit_servings(raw_block),
        ingredients=tuple(
            ingredient.raw_text.strip() for ingredient in candidate.ingredients if ingredient.raw_text.strip()
        )[:100],
        steps=tuple(step.instruction.strip() for step in candidate.steps if step.instruction.strip())[:100],
    )


class DeterministicRecipeImportExtractor:
    """Parse recognised recipe sections through the existing rule extractor."""

    def __init__(self, recipe_extractor: RecipeExtractor | None = None) -> None:
        self._recipe_extractor = recipe_extractor or RecipeExtractor()

    async def extract(self, text: str) -> tuple[RecipeImportDraft, ...]:
        cleaned = clean_recipe_text(text)
        drafts: list[RecipeImportDraft] = []
        for index, raw_block in enumerate(split_recipe_blocks(cleaned), start=1):
            block = _normalise_heading(raw_block)
            candidate = await self._recipe_extractor.extract(block)
            drafts.append(_candidate_to_draft(index, raw_block, candidate))
        return tuple(drafts)
