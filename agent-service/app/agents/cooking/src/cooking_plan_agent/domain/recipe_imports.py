"""Typed contract for natural-language recipe imports."""

from __future__ import annotations

import unicodedata
from enum import StrEnum

from pydantic import Field, field_validator, model_validator

from cooking_plan_agent.domain.models import StrictModel

ENGLISH_ONLY_MESSAGE = "Please use English only. Chinese or mixed-language input is not supported."


def contains_non_latin_letters(value: str) -> bool:
    """Return whether text contains a letter outside the Latin script.

    Digits, punctuation, measurement symbols, combining marks, and emoji are
    allowed. This is the product's deterministic input policy, not a
    statistical language detector.
    """

    for character in value:
        if not unicodedata.category(character).startswith("L"):
            continue
        if "LATIN" not in unicodedata.name(character, ""):
            return True
    return False


def require_english_script(value: str) -> str:
    if contains_non_latin_letters(value):
        raise ValueError(ENGLISH_ONLY_MESSAGE)
    return value


class RecipeImportStatus(StrEnum):
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    READY = "READY"


class RecipeImportAnswer(StrictModel):
    question_id: str = Field(min_length=1, max_length=160)
    value: str = Field(min_length=1, max_length=20_000)

    @field_validator("value")
    @classmethod
    def english_only(cls, value: str) -> str:
        return require_english_script(value)


class ParseRecipeImportRequest(StrictModel):
    request_id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1, max_length=100_000)
    answers: tuple[RecipeImportAnswer, ...] = Field(default=(), max_length=24)

    @field_validator("text")
    @classmethod
    def english_only(cls, value: str) -> str:
        return require_english_script(value)

    @model_validator(mode="after")
    def unique_answers(self) -> ParseRecipeImportRequest:
        identifiers = [answer.question_id for answer in self.answers]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Duplicate recipe-import question IDs are not allowed.")
        return self


class RecipeImportDraft(StrictModel):
    draft_id: str = Field(min_length=1, max_length=80)
    name: str | None = Field(default=None, max_length=160)
    servings: int | None = Field(default=None, ge=1, le=50)
    ingredients: tuple[str, ...] = Field(default=(), max_length=100)
    steps: tuple[str, ...] = Field(default=(), max_length=100)


class RecipeImportQuestion(StrictModel):
    question_id: str = Field(min_length=1, max_length=160)
    draft_id: str = Field(min_length=1, max_length=80)
    field_path: str = Field(min_length=1, max_length=80)
    prompt: str = Field(min_length=1, max_length=500)
    response_type: str = "TEXT"
    required: bool = True
    suggested_value: str | None = Field(default=None, max_length=500)


class ParseRecipeImportResponse(StrictModel):
    status: RecipeImportStatus
    drafts: tuple[RecipeImportDraft, ...]
    questions: tuple[RecipeImportQuestion, ...] = ()
