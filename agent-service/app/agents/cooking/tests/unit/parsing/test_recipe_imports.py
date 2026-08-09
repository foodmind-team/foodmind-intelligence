from __future__ import annotations

import pytest
from pydantic import ValidationError

from cooking_plan_agent.application.recipe_import_service import (
    InvalidRecipeImportAnswers,
    ParseRecipeImportService,
)
from cooking_plan_agent.domain.recipe_imports import ParseRecipeImportRequest
from cooking_plan_agent.parsing.recipe_imports import DeterministicRecipeImportExtractor

MULTI_DISH_TEXT = """Recipe: Lemon Pasta
4 servings
Ingredients:
200 g spaghetti
1 lemon
Steps:
1. Boil the spaghetti for 10 minutes.
2. Toss with lemon.
---
Recipe: Tomato Salad
Ingredients:
2 tomatoes
1 tbsp olive oil
Steps:
1. Slice the tomatoes.
2. Toss with olive oil.
"""


@pytest.mark.asyncio
async def test_multi_dish_import_asks_for_missing_servings_then_becomes_ready() -> None:
    service = ParseRecipeImportService(DeterministicRecipeImportExtractor())

    first = await service.execute(ParseRecipeImportRequest(request_id="req-1", text=MULTI_DISH_TEXT))

    assert first.status == "NEEDS_CLARIFICATION"
    assert [draft.name for draft in first.drafts] == ["Lemon Pasta", "Tomato Salad"]
    assert [question.question_id for question in first.questions] == ["dish-2:servings"]

    completed = await service.execute(
        ParseRecipeImportRequest(
            request_id="req-1",
            text=MULTI_DISH_TEXT,
            answers=({"question_id": "dish-2:servings", "value": "4"},),
        )
    )

    assert completed.status == "READY"
    assert completed.questions == ()
    assert [draft.servings for draft in completed.drafts] == [4, 4]
    assert all(draft.ingredients and draft.steps for draft in completed.drafts)


@pytest.mark.asyncio
async def test_unknown_answer_is_rejected() -> None:
    service = ParseRecipeImportService(DeterministicRecipeImportExtractor())

    with pytest.raises(InvalidRecipeImportAnswers):
        await service.execute(
            ParseRecipeImportRequest(
                request_id="req-2",
                text=MULTI_DISH_TEXT,
                answers=({"question_id": "dish-9:name", "value": "Soup"},),
            )
        )


@pytest.mark.parametrize("text", ["番茄 pasta", "トマト salad", "토마토 salad"])
def test_non_latin_or_mixed_input_is_rejected(text: str) -> None:
    with pytest.raises(ValidationError, match="Please use English only"):
        ParseRecipeImportRequest(request_id="req-3", text=text)
