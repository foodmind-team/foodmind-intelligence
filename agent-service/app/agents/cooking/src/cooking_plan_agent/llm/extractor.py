"""LLM-backed recipe extractor implementing the RecipeExtractor Protocol.

Converts free-form recipe text into a structured ExtractedRecipeCandidate
using a local LLM with JSON-mode output. This replaces the rule-based
extractor when LLM is enabled, while preserving the exact Protocol contract
(workflow/context.py) so the workflow graph does not change.

Fallback: if the LLM call fails or its output fails schema validation, the
rule-based extractor is used so the pipeline degrades gracefully.
"""

from __future__ import annotations

import hashlib
import logging
from decimal import Decimal
from typing import Any

from cooking_plan_agent.domain.enums import HeatLevel
from cooking_plan_agent.domain.models import (
    ExtractedIngredient,
    ExtractedRecipeCandidate,
    ExtractedStep,
)
from cooking_plan_agent.llm.client import LLMClient, LLMError
from cooking_plan_agent.normalisation.names import clean_dish_name

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Extraction prompt — instructs the LLM to emit a JSON object matching
# ExtractedRecipeCandidate (snake_case fields). Bounded and deterministic.
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a recipe structuring assistant. Extract structured recipe data "
    "from user-provided cooking text. Respond with a SINGLE JSON object only — "
    "no prose, no markdown fences. The object must use exactly these fields:\n"
    '{"dish_name": string, "original_servings": number, "source_language": '
    '"zho"|"eng"|"und", "ingredients": [{"raw_text": string, "name": string, '
    '"quantity": number|null, "unit": string|null, "preparation": string|null}], '
    '"steps": [{"instruction": string, "category": "general"|"heating"|'
    '"preparation"|"resting"|"mixing", "active_duration_minutes": number|null, '
    '"passive_duration_minutes": number|null, "heat_level": "NONE"|"LOW"|"MEDIUM"|'
    '"HIGH", "target_temperature_c": number|null, "resources_hint": [string]}]}\n'
    "Rules: quantity must be a positive number when given; omit fields the text "
    "does not specify (use null or empty list, never invent values). "
    "dish_name must be a SHORT dish title only — strip quantities, units, "
    "parenthetical notes, and preparation instructions (e.g. 'Fresh Shrimp', not "
    "'Fresh shrimp (remove head, tail, and thread)')."
)

_ENGLISH_OUTPUT_RULE = (
    " Translate every user-visible text field into clear English before returning it. "
    "dish_name, ingredient raw_text/name/preparation, step instruction/category, and resource hints "
    "must be English even when the source is written in another language. Preserve quantities, units, "
    "temperatures, times, and proper nouns accurately. source_language must describe the original input language."
)

# Stable cache tag (P1-06 rule 2): changing the prompt changes this digest, so
# cached parse artifacts keyed on the old prompt are never reused.
PARSE_PROMPT_VERSION = hashlib.sha256(_SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12]


class LLMRecipeExtractor:
    """Extract structured recipe candidates via a local LLM.

    Implements the async extract() contract expected by the workflow
    (RecipeExtractor Protocol in workflow/context.py).
    """

    def __init__(self, client: LLMClient, *, translate_to_english: bool = False) -> None:
        self._client = client
        self._system_prompt = _SYSTEM_PROMPT + (_ENGLISH_OUTPUT_RULE if translate_to_english else "")

    async def extract(self, source_text: str) -> ExtractedRecipeCandidate:
        """Parse recipe text into a structured candidate using the LLM.

        Args:
            source_text: Raw recipe text (preprocessed).

        Returns:
            An ExtractedRecipeCandidate with extraction_source="LLM" on
            success. On LLM failure, falls back to the rule-based extractor
            so the pipeline degrades gracefully (source="RULE_BASED").
        """
        try:
            data = await self._client.chat_json(
                [
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": source_text},
                ]
            )
            return self._to_candidate(source_text, data)
        except LLMError:
            # Degrade to rule-based parsing — never block the workflow.
            logger.warning("LLM extraction failed — falling back to rule-based")
            return await self._rule_based_extract(source_text)

    # ------------------------------------------------------------------
    # Fallback: built-in rule-based extractor (same path as llm disabled)
    # ------------------------------------------------------------------

    @staticmethod
    async def _rule_based_extract(source_text: str) -> ExtractedRecipeCandidate:
        from cooking_plan_agent.parsing.extractor import RecipeExtractor as RuleExtractor

        return await RuleExtractor().extract(source_text)

    # ------------------------------------------------------------------
    # Mapping LLM JSON → domain model (defensive: tolerate missing keys)
    # ------------------------------------------------------------------

    @staticmethod
    def _to_candidate(source_text: str, data: dict[str, Any]) -> ExtractedRecipeCandidate:
        ingredients = tuple(
            LLMRecipeExtractor._to_ingredient(item) for item in data.get("ingredients") or [] if isinstance(item, dict)
        )
        steps = tuple(
            LLMRecipeExtractor._to_step(i, item)
            for i, item in enumerate(data.get("steps") or [], start=1)
            if isinstance(item, dict)
        )
        dish_name = clean_dish_name(str(data.get("dish_name") or "Untitled Recipe"))[:80]
        try:
            servings = Decimal(str(data.get("original_servings") or 2))
        except (TypeError, ValueError):
            servings = Decimal(2)

        return ExtractedRecipeCandidate(
            recipe_id=f"recipe_{dish_name[:40]}",
            dish_name=dish_name,
            original_servings=servings,
            source_language=str(data.get("source_language") or "und"),
            ingredients=ingredients,
            steps=steps,
            extraction_source="LLM",
        )

    @staticmethod
    def _to_ingredient(item: dict[str, Any]) -> ExtractedIngredient:
        raw_text = str(item.get("raw_text") or "").strip()
        name = str(item.get("name") or "").strip()
        quantity_raw = item.get("quantity")
        quantity = None
        try:
            if quantity_raw is not None and str(quantity_raw).strip():
                q = Decimal(str(quantity_raw))
                if q > 0:
                    quantity = q
        except (ValueError, TypeError):
            quantity = None
        unit = str(item.get("unit") or "").strip() or None
        prep = str(item.get("preparation") or "").strip() or None
        return ExtractedIngredient(
            raw_text=raw_text or name,
            name=name or "unknown",
            quantity=quantity,
            unit=unit,
            preparation=prep,
            extraction_source="LLM",
        )

    @staticmethod
    def _to_step(index: int, item: dict[str, Any]) -> ExtractedStep:
        instruction = str(item.get("instruction") or "").strip()
        heat = str(item.get("heat_level") or "NONE").upper()
        if heat not in {"NONE", "LOW", "MEDIUM", "HIGH"}:
            heat = "NONE"
        return ExtractedStep(
            step_number=index,
            instruction=instruction,
            category=str(item.get("category") or "general"),
            active_duration_minutes=LLMRecipeExtractor._to_int(item.get("active_duration_minutes")),
            passive_duration_minutes=LLMRecipeExtractor._to_int(item.get("passive_duration_minutes")),
            heat_level=HeatLevel(heat),
            target_temperature_c=LLMRecipeExtractor._to_decimal(item.get("target_temperature_c")),
            resources_hint=tuple(str(r) for r in (item.get("resources_hint") or []) if isinstance(r, str)),
            extraction_source="LLM",
        )

    @staticmethod
    def _to_int(value: Any) -> int | None:
        try:
            if value is not None:
                v = int(value)
                return v if v > 0 else None
        except (TypeError, ValueError):
            pass
        return None

    @staticmethod
    def _to_decimal(value: Any) -> Decimal | None:
        try:
            if value is not None:
                d = Decimal(str(value))
                return d if d > 0 else None
        except (ValueError, TypeError):
            pass
        return None
