"""LLM-backed multi-dish recipe import extraction.

The import boundary reuses the agent's full natural-language pipeline:

  1. ``clean_recipe_text``  — deterministic cleaning (line endings, blanks)
  2. ``split_recipe_blocks`` — deterministic multi-dish splitting
  3. multi-block input → per-dish ``LLMRecipeExtractor`` fan-out (each block is
     a short, fully-structured extraction; failures degrade to the rule
     extractor for that block only)
  4. single-block input → one LLM call for the whole ``recipes`` array with a
     raised output budget, so multi-dish JSON is never truncated mid-array

Every path falls back to ``DeterministicRecipeImportExtractor`` instead of
failing, so a provider outage never blocks the interactive import flow.
"""

from __future__ import annotations

import asyncio
import json
import unicodedata
from typing import Any

from cooking_plan_agent.application.recipe_import_service import InvalidRecipeImportAnswers
from cooking_plan_agent.config.settings import get_settings
from cooking_plan_agent.domain.recipe_imports import RecipeImportAnswer, RecipeImportDraft, RecipeImportQuestion
from cooking_plan_agent.llm.client import LLMClient, LLMError
from cooking_plan_agent.llm.extractor import LLMRecipeExtractor
from cooking_plan_agent.parsing.extractor import RecipeExtractor
from cooking_plan_agent.parsing.recipe_imports import (
    DeterministicRecipeImportExtractor,
    _candidate_to_draft,
    _normalise_heading,
    clean_recipe_text,
    expand_prep_boundaries,
    split_on_markers,
    split_recipe_blocks,
)

_SYSTEM_PROMPT = (
    "You extract one or more recipes from user text written in any language. Return one JSON object only with "
    "a recipes array. Each recipe must contain exactly: name (string or null), servings "
    "(whole number or null), ingredients (array of strings), and steps (array of strings). "
    "Translate name, ingredients, and steps into clear English before returning them. All recipe strings in "
    "the JSON response must be English even when the source is multilingual. Preserve quantities, units, "
    "temperatures, cooking times, and proper nouns accurately. "
    "Preserve input order. name must be a SHORT dish title only — strip quantities, units, "
    "parenthetical notes, and preparation instructions (e.g. 'Fresh Shrimp', not 'Fresh shrimp "
    "(remove head, tail, and thread)'; 'chicken wings', not '15 chicken wings'). "
    "CRITICAL: when the text describes MULTIPLE distinct dishes — "
    "separated by '---', 'Recipe:' headings, blank lines, or simply listed one after another — "
    "you MUST return one recipe object per dish. Never merge two dishes into a single recipe "
    "object, and never invent a dish name, serving count, ingredient, or step that the user "
    "did not provide; use null or an empty array when required information is missing."
)

_ANSWER_SYSTEM_PROMPT = (
    "Translate recipe clarification answer values into clear English. Return one JSON object only with an "
    "answers array containing exactly question_id and value for every supplied answer. Preserve question_id, "
    "numbers, quantities, units, temperatures, cooking times, line breaks, and list item boundaries. Do not "
    "add or remove recipe facts. Every textual value must be English. For a servings answer, convert a number "
    "written in words or another numeral system to ASCII digits only."
)

_SPLIT_SYSTEM_PROMPT = (
    "You split a pasted cooking text into its separate dishes. Return one JSON object only with "
    "a dishes array. Each element is the exact first line of one dish, copied verbatim from the "
    "input — for example 'Recipe: Lemon Pasta', 'Ingredients Preparation', or a dish name line "
    "such as 'Fried Spare Ribs'. Cover every dish in the text in order. If there is only one "
    "dish, return one element. Never rewrite, translate, or invent lines that do not appear "
    "in the input. IMPORTANT: a dish's name may be glued onto the previous dish's last line "
    "after an ingredient, e.g. '...Cooking oil Fried Spare Ribs' — when you see such a new "
    "dish name, still report it; a marker does not have to start at a line boundary."
)

_TRANSLATION_SYSTEM_PROMPT = (
    "You normalize already-extracted recipe drafts into English. Return one JSON object only with a recipes "
    "array. Preserve each draft_id, servings value, list order, quantities, units, temperatures, times, and "
    "proper nouns. Translate name, every ingredient, and every step into clear English. Do not add, remove, "
    "merge, or reinterpret recipe facts. Every letter in user-visible fields must use Latin script."
)


def _contains_non_latin_letters(value: str) -> bool:
    return any(character.isalpha() and "LATIN" not in unicodedata.name(character, "") for character in value)


def _needs_english_normalisation(drafts: tuple[RecipeImportDraft, ...]) -> bool:
    return any(
        _contains_non_latin_letters(value)
        for draft in drafts
        for value in (draft.name or "", *draft.ingredients, *draft.steps)
    )


class LLMRecipeImportExtractor:
    """Extract a bounded list of partial import drafts with safe fallback."""

    def __init__(
        self,
        client: LLMClient,
        fallback: DeterministicRecipeImportExtractor | None = None,
        *,
        timeout_seconds: float = 20.0,
        max_output_tokens: int | None = None,
    ) -> None:
        self._client = client
        self._fallback = fallback or DeterministicRecipeImportExtractor()
        self._timeout_seconds = timeout_seconds
        self._max_output_tokens = max_output_tokens
        # Per-dish fan-out reuses the workflow's full-field extractor, so every
        # dish benefits from the same rich prompt and rule degradation as the
        # main cooking-plan pipeline.
        self._dish_extractor = LLMRecipeExtractor(client, translate_to_english=True)

    async def normalise_answers(
        self,
        questions: tuple[RecipeImportQuestion, ...],
        answers: tuple[RecipeImportAnswer, ...],
    ) -> tuple[RecipeImportAnswer, ...]:
        """Translate free-text clarification answers while preserving their IDs."""

        field_by_id = {question.question_id: question.field_path for question in questions}
        if not answers:
            return answers
        request_payload = {
            "answers": [
                {
                    "question_id": answer.question_id,
                    "field_path": field_by_id.get(answer.question_id, "text"),
                    "value": answer.value,
                }
                for answer in answers
            ]
        }
        try:
            payload = await asyncio.wait_for(
                self._client.chat_json(
                    [
                        {"role": "system", "content": _ANSWER_SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps(request_payload, ensure_ascii=False)},
                    ],
                    max_tokens=self._max_output_tokens,
                ),
                timeout=self._timeout_seconds,
            )
            translated_items = payload.get("answers")
            if not isinstance(translated_items, list):
                raise LLMError("Recipe answer translation did not contain an answers list")
            translated = {
                str(item.get("question_id")): str(item.get("value", "")).strip()
                for item in translated_items
                if isinstance(item, dict) and item.get("question_id") and str(item.get("value", "")).strip()
            }
            expected_ids = {answer.question_id for answer in answers}
            if len(translated_items) != len(expected_ids) or set(translated) != expected_ids:
                raise LLMError("Recipe answer translation changed the answer identifiers")
        except (TimeoutError, LLMError, TypeError, ValueError) as exc:
            raise InvalidRecipeImportAnswers(
                "Recipe answer translation is temporarily unavailable. Please try again."
            ) from exc

        return tuple(
            RecipeImportAnswer(question_id=answer.question_id, value=translated.get(answer.question_id, answer.value))
            for answer in answers
        )

    async def extract(self, text: str) -> tuple[RecipeImportDraft, ...]:
        cleaned = clean_recipe_text(text)
        # Deterministic coarse cut: separators, headings, blank lines, then
        # "Ingredients Preparation" template boundaries (substring-based, so
        # glued headings like "...serve. Ingredients Preparation" still cut).
        coarse: list[str] = []
        for block in split_recipe_blocks(cleaned):
            coarse.extend(expand_prep_boundaries(block))
        blocks = tuple(coarse)

        # LLM semantic splitting is the primary boundary detector: it handles
        # heading-less dishes and other layouts the deterministic rules cannot
        # see. Running it per block keeps every call short (a full 6-dish paste
        # times out in one shot) and lets still-merged blocks expand. Blocks
        # are independent, so they split concurrently under the LLM semaphore.
        settings = get_settings()
        semaphore = asyncio.Semaphore(max(1, settings.llm_max_concurrency))

        async def _split_one(block: str) -> tuple[str, ...]:
            async with semaphore:
                try:
                    sub_blocks = await self._split_with_llm(block)
                    return sub_blocks if len(sub_blocks) > 1 else (block,)
                except (TimeoutError, LLMError, TypeError, ValueError):
                    return (block,)

        nested = await asyncio.gather(*(_split_one(block) for block in blocks))
        blocks = tuple(part for group in nested for part in group)

        if len(blocks) > 1:
            # Recognisable multi-dish text → extract each block independently.
            try:
                drafts = await self._extract_multi(blocks)
            except (TimeoutError, LLMError, TypeError, ValueError):
                drafts = await self._fallback.extract(cleaned)
            return await self._ensure_english(drafts)
        # Single block → whole-text recipes-array extraction.
        try:
            drafts = await self._extract_single(cleaned)
        except (TimeoutError, LLMError, TypeError, ValueError):
            drafts = await self._fallback.extract(cleaned)
        return await self._ensure_english(drafts)

    async def _ensure_english(self, drafts: tuple[RecipeImportDraft, ...]) -> tuple[RecipeImportDraft, ...]:
        """Retry translation as a bounded final gate instead of persisting mixed-script drafts."""
        if not _needs_english_normalisation(drafts):
            return drafts
        request_payload = {"recipes": [draft.model_dump() for draft in drafts]}
        try:
            payload = await asyncio.wait_for(
                self._client.chat_json(
                    [
                        {"role": "system", "content": _TRANSLATION_SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps(request_payload, ensure_ascii=False)},
                    ],
                    max_tokens=self._max_output_tokens,
                ),
                timeout=self._timeout_seconds,
            )
            values = payload.get("recipes")
            if not isinstance(values, list) or len(values) != len(drafts):
                raise LLMError("Recipe translation changed the draft count")
            translated: list[RecipeImportDraft] = []
            for index, (previous, value) in enumerate(zip(drafts, values, strict=True), start=1):
                if not isinstance(value, dict) or value.get("draft_id") != previous.draft_id:
                    raise LLMError("Recipe translation changed a draft identifier")
                candidate = self._draft(index, value).model_copy(update={"draft_id": previous.draft_id})
                if candidate.servings != previous.servings:
                    raise LLMError("Recipe translation changed a serving count")
                translated.append(candidate)
            result = tuple(translated)
            if _needs_english_normalisation(result):
                raise LLMError("Recipe translation still contains non-Latin text")
            return result
        except (TimeoutError, LLMError, TypeError, ValueError) as exc:
            raise InvalidRecipeImportAnswers(
                "The recipe could not be fully converted to English. Please try parsing it again."
            ) from exc

    async def _split_with_llm(self, text: str) -> tuple[str, ...]:
        """Ask the LLM for the first line of every dish, then cut on those lines.

        Returns a single block when the model reports one dish or its markers
        cannot be located, so the caller falls through to whole-text parsing.
        """
        payload = await asyncio.wait_for(
            self._client.chat_json(
                [
                    {"role": "system", "content": _SPLIT_SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                max_tokens=self._max_output_tokens,
            ),
            # The split reply is tiny — never let it consume the whole budget.
            # Per-block calls stay well under this; whole-text pastes may not.
            timeout=min(self._timeout_seconds, 6.0),
        )
        markers = payload.get("dishes")
        if not isinstance(markers, list):
            raise LLMError("Dish split response did not contain a dishes list")
        clean_markers = tuple(str(marker).strip() for marker in markers if isinstance(marker, str) and marker.strip())
        if len(clean_markers) < 2:
            return (text,)
        blocks = split_on_markers(text, clean_markers)
        return blocks if len(blocks) > 1 else (text,)

    async def _extract_multi(self, blocks: tuple[str, ...]) -> tuple[RecipeImportDraft, ...]:
        settings = get_settings()
        semaphore = asyncio.Semaphore(max(1, settings.llm_max_concurrency))

        async def _one(index: int, raw_block: str) -> RecipeImportDraft:
            block = _normalise_heading(raw_block)
            async with semaphore:
                try:
                    candidate = await asyncio.wait_for(
                        self._dish_extractor.extract(block),
                        # DeepSeek completes a single dish well under this;
                        # a hung provider must not stall the whole import.
                        timeout=min(self._timeout_seconds, 10.0),
                    )
                except (TimeoutError, LLMError, TypeError, ValueError):
                    # Progressive degradation: a single bad block must not
                    # fail the whole import — use the rule extractor for it.
                    candidate = await RecipeExtractor().extract(block)
            return _candidate_to_draft(index, raw_block, candidate)

        drafts = await asyncio.wait_for(
            asyncio.gather(*(_one(index, block) for index, block in enumerate(blocks, start=1))),
            timeout=settings.llm_overall_timeout_seconds,
        )
        return tuple(drafts)

    async def _extract_single(self, text: str) -> tuple[RecipeImportDraft, ...]:
        payload = await asyncio.wait_for(
            self._client.chat_json(
                [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                max_tokens=self._max_output_tokens,
            ),
            timeout=self._timeout_seconds,
        )
        recipes = payload.get("recipes")
        if not isinstance(recipes, list) or not recipes:
            raise LLMError("Recipe import response did not contain recipes")
        return tuple(self._draft(index, item) for index, item in enumerate(recipes, start=1) if isinstance(item, dict))

    @staticmethod
    def _draft(index: int, value: dict[str, Any]) -> RecipeImportDraft:
        raw_servings = value.get("servings")
        servings: int | None = None
        if isinstance(raw_servings, int) and not isinstance(raw_servings, bool) and 1 <= raw_servings <= 50:
            servings = raw_servings
        name_value = value.get("name")
        name = str(name_value).strip()[:160] if isinstance(name_value, str) and name_value.strip() else None
        ingredients = tuple(
            str(item).strip()[:500] for item in value.get("ingredients", []) if isinstance(item, str) and item.strip()
        )[:100]
        steps = tuple(
            str(item).strip()[:1000] for item in value.get("steps", []) if isinstance(item, str) and item.strip()
        )[:100]
        return RecipeImportDraft(
            draft_id=f"dish-{index}",
            name=name,
            servings=servings,
            ingredients=ingredients,
            steps=steps,
        )
