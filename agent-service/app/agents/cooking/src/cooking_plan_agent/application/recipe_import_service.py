"""Application service for multi-dish recipe import clarification."""

from __future__ import annotations

import re

from cooking_plan_agent.config.settings import get_settings
from cooking_plan_agent.domain.recipe_imports import (
    ParseRecipeImportRequest,
    ParseRecipeImportResponse,
    RecipeImportDraft,
    RecipeImportQuestion,
    RecipeImportStatus,
)
from cooking_plan_agent.parsing.recipe_imports import RecipeImportExtractor

_LIST_PREFIX = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


class InvalidRecipeImportAnswers(ValueError):
    """Raised when answers do not match the current structured questions."""


class ParseRecipeImportService:
    """Extract drafts, apply field-scoped answers, and produce questions."""

    def __init__(self, extractor: RecipeImportExtractor) -> None:
        self._extractor = extractor

    async def execute(self, request: ParseRecipeImportRequest) -> ParseRecipeImportResponse:
        drafts = await self._extractor.extract(request.text)
        settings = get_settings()
        if not drafts:
            drafts = (RecipeImportDraft(draft_id="dish-1"),)
        if len(drafts) > settings.max_recipe_count:
            raise InvalidRecipeImportAnswers(
                f"A maximum of {settings.max_recipe_count} recipes can be imported at once."
            )

        current_questions = self._questions(drafts)
        allowed = {question.question_id: question for question in current_questions}
        answers = {answer.question_id: answer.value for answer in request.answers}
        unknown = sorted(set(answers) - set(allowed))
        if unknown:
            raise InvalidRecipeImportAnswers("One or more answers do not match the current questions.")

        updated = tuple(self._apply_answers(draft, allowed, answers) for draft in drafts)
        questions = self._questions(updated)
        status = RecipeImportStatus.NEEDS_CLARIFICATION if questions else RecipeImportStatus.READY
        return ParseRecipeImportResponse(status=status, drafts=updated, questions=questions)

    @staticmethod
    def _apply_answers(
        draft: RecipeImportDraft,
        allowed: dict[str, RecipeImportQuestion],
        answers: dict[str, str],
    ) -> RecipeImportDraft:
        updates: dict[str, object] = {}
        for question_id, value in answers.items():
            question = allowed[question_id]
            if question.draft_id != draft.draft_id:
                continue
            if question.field_path == "name":
                updates["name"] = value.strip()[:160] or None
            elif question.field_path == "servings":
                try:
                    servings = int(value)
                except ValueError:
                    continue
                if 1 <= servings <= 50:
                    updates["servings"] = servings
            elif question.field_path in {"ingredients", "steps"}:
                limit = 500 if question.field_path == "ingredients" else 1000
                items = tuple(
                    cleaned[:limit]
                    for line in re.split(r"[\n;]+", value)
                    if (cleaned := _LIST_PREFIX.sub("", line).strip())
                )[:100]
                if items:
                    updates[question.field_path] = items
        return draft.model_copy(update=updates)

    @staticmethod
    def _questions(drafts: tuple[RecipeImportDraft, ...]) -> tuple[RecipeImportQuestion, ...]:
        questions: list[RecipeImportQuestion] = []
        for draft in drafts:
            label = draft.name or "this dish"
            if not draft.name:
                questions.append(ParseRecipeImportService._question(draft, "name", "What is the dish name?"))
            if draft.servings is None:
                questions.append(
                    ParseRecipeImportService._question(
                        draft,
                        "servings",
                        f"How many servings does {label} make? Enter a whole number from 1 to 50.",
                    )
                )
            if not draft.ingredients:
                questions.append(
                    ParseRecipeImportService._question(
                        draft,
                        "ingredients",
                        f"List the ingredients for {label}, with one ingredient per line.",
                    )
                )
            if not draft.steps:
                questions.append(
                    ParseRecipeImportService._question(
                        draft,
                        "steps",
                        f"List the cooking steps for {label}, with one step per line.",
                    )
                )
        return tuple(questions)

    @staticmethod
    def _question(draft: RecipeImportDraft, field_path: str, prompt: str) -> RecipeImportQuestion:
        return RecipeImportQuestion(
            question_id=f"{draft.draft_id}:{field_path}",
            draft_id=draft.draft_id,
            field_path=field_path,
            prompt=prompt,
        )
