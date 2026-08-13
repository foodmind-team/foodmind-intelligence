"""Preprocess use-case tests."""

from decimal import Decimal
from typing import Any, cast

import pytest

from cooking_plan_agent.application.service import GenerateCookingPlanService
from cooking_plan_agent.domain.enums import HeatLevel
from cooking_plan_agent.domain.models import (
    ExtractedIngredient,
    ExtractedRecipeCandidate,
    ExtractedStep,
    PreprocessRecipesRequest,
)
from cooking_plan_agent.workflow.context import WorkflowContext


class _CompletionExtractor:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def extract(self, source_text: str) -> ExtractedRecipeCandidate:
        self.calls.append(source_text)
        return ExtractedRecipeCandidate(
            recipe_id="llm-generated-id",
            dish_name="Roast Chicken",
            original_servings=Decimal(2),
            source_language="eng",
            ingredients=(
                ExtractedIngredient(
                    raw_text="chicken 500 g",
                    name="chicken",
                    quantity=Decimal(500),
                    unit="g",
                ),
            ),
            steps=(
                ExtractedStep(
                    step_number=1,
                    instruction="Roast the chicken until cooked through.",
                    category="heating",
                    passive_duration_minutes=40,
                    heat_level=HeatLevel.MEDIUM,
                    target_temperature_c=Decimal(75),
                    resources_hint=("oven",),
                    extraction_source="LLM_INFERRED",
                    confidence=Decimal("0.8"),
                ),
            ),
            inferred_fields=("steps[0].target_temperature_c",),
        )


@pytest.mark.asyncio
async def test_preprocess_uses_configured_llm_completion_extractor() -> None:
    extractor = _CompletionExtractor()
    service = GenerateCookingPlanService(
        graph=cast(Any, object()),
        context=WorkflowContext(recipe_extractor=extractor),
    )
    request = PreprocessRecipesRequest(
        request_id="preprocess-1",
        recipes=(
            {
                "recipe_id": "stored-recipe-id",
                "text": "Roast one chicken until cooked through.",
                "target_servings": 2,
            },
        ),
    )

    response = await service.preprocess(request)

    assert extractor.calls == ["Roast one chicken until cooked through."]
    candidate = response.recipes[0]
    assert candidate.recipe_id == "stored-recipe-id"
    assert candidate.steps[0].target_temperature_c == Decimal(75)
    assert candidate.steps[0].extraction_source == "LLM_INFERRED"
