"""Typed contract for natural-language recipe imports."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from cooking_plan_agent.domain.models import StrictModel


class RecipeImportStatus(StrEnum):
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    READY = "READY"


class RecipeImportAnswer(StrictModel):
    question_id: str = Field(min_length=1, max_length=160)
    value: str = Field(min_length=1, max_length=20_000)


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


class ParseRecipeImportRequest(StrictModel):
    request_id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1, max_length=100_000)
    answers: tuple[RecipeImportAnswer, ...] = Field(default=(), max_length=24)
    drafts: tuple[RecipeImportDraft, ...] = Field(default=(), max_length=6)
    questions: tuple[RecipeImportQuestion, ...] = Field(default=(), max_length=24)

    @model_validator(mode="after")
    def valid_resume_snapshot(self) -> ParseRecipeImportRequest:
        answer_identifiers = [answer.question_id for answer in self.answers]
        if len(answer_identifiers) != len(set(answer_identifiers)):
            raise ValueError("Duplicate recipe-import question IDs are not allowed.")
        draft_identifiers = [draft.draft_id for draft in self.drafts]
        if len(draft_identifiers) != len(set(draft_identifiers)):
            raise ValueError("Duplicate recipe-import draft IDs are not allowed.")
        question_identifiers = [question.question_id for question in self.questions]
        if len(question_identifiers) != len(set(question_identifiers)):
            raise ValueError("Duplicate recipe-import question IDs are not allowed.")
        if self.questions and not self.drafts:
            raise ValueError("Recipe-import questions require their draft snapshot.")
        return self


class ParseRecipeImportResponse(StrictModel):
    status: RecipeImportStatus
    drafts: tuple[RecipeImportDraft, ...]
    questions: tuple[RecipeImportQuestion, ...] = ()
