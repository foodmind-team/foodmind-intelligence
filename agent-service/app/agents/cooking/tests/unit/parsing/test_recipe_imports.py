from __future__ import annotations

import asyncio
from typing import Any

import pytest

from cooking_plan_agent.application.recipe_import_service import (
    InvalidRecipeImportAnswers,
    ParseRecipeImportService,
)
from cooking_plan_agent.domain.recipe_imports import (
    ParseRecipeImportRequest,
    RecipeImportDraft,
    RecipeImportQuestion,
)
from cooking_plan_agent.llm.client import LLMError
from cooking_plan_agent.llm.recipe_importer import LLMRecipeImportExtractor
from cooking_plan_agent.parsing.recipe_imports import (
    DeterministicRecipeImportExtractor,
    expand_prep_boundaries,
    split_on_markers,
    split_recipe_blocks,
)

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
async def test_multi_dish_import_defaults_missing_servings_without_clarification() -> None:
    service = ParseRecipeImportService(DeterministicRecipeImportExtractor())

    first = await service.execute(ParseRecipeImportRequest(request_id="req-1", text=MULTI_DISH_TEXT))

    assert first.status == "READY"
    assert [draft.name for draft in first.drafts] == ["Lemon Pasta", "Tomato Salad"]
    assert first.questions == ()
    assert [draft.servings for draft in first.drafts] == [4, 2]
    assert all(draft.ingredients and draft.steps for draft in first.drafts)


class _UnexpectedResumeDependency:
    async def extract(self, _text: str):
        raise AssertionError("A persisted clarification snapshot must not reparse the source text")

    async def normalise_answers(self, _questions, _answers):
        raise AssertionError("An ASCII servings answer must not call the LLM normaliser")


@pytest.mark.asyncio
async def test_numeric_servings_resume_uses_snapshot_without_extractor_or_llm() -> None:
    dependency = _UnexpectedResumeDependency()
    service = ParseRecipeImportService(dependency, answer_normaliser=dependency)  # type: ignore[arg-type]
    draft = RecipeImportDraft(
        draft_id="dish-1",
        name="Spicy Crab Legs",
        ingredients=("Crab legs", "Ginger"),
        steps=("Cook the crab legs.",),
    )
    question = RecipeImportQuestion(
        question_id="dish-1:servings",
        draft_id="dish-1",
        field_path="servings",
        prompt="How many servings does Spicy Crab Legs make?",
    )

    completed = await service.execute(
        ParseRecipeImportRequest(
            request_id="req-resume",
            text="The original multilingual source is retained for audit only.",
            answers=({"question_id": question.question_id, "value": "1"},),
            drafts=(draft,),
            questions=(question,),
        )
    )

    assert completed.status == "READY"
    assert completed.questions == ()
    assert completed.drafts[0].servings == 1
    assert completed.drafts[0].name == "Spicy Crab Legs"


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


@pytest.mark.parametrize("text", ["番茄 pasta", "トマト salad", "토마토 salad", "pasta con tomate"])
def test_multilingual_input_is_accepted(text: str) -> None:
    request = ParseRecipeImportRequest(request_id="req-3", text=text)

    assert request.text == text


class _BlockingLLMClient:
    async def chat_json(self, messages: list[dict[str, str]], **_: Any) -> dict[str, Any]:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


@pytest.mark.asyncio
async def test_recipe_import_falls_back_when_llm_exceeds_interactive_deadline() -> None:
    extractor = LLMRecipeImportExtractor(
        _BlockingLLMClient(),  # type: ignore[arg-type]
        DeterministicRecipeImportExtractor(),
        timeout_seconds=0.01,
    )

    drafts = await extractor.extract(MULTI_DISH_TEXT)

    assert [draft.name for draft in drafts] == ["Lemon Pasta", "Tomato Salad"]


DOUBLE_BLANK_DISH_TEXT = """Lemon Pasta
4 servings
Ingredients:
200 g spaghetti
1 lemon

Steps:
1. Boil the spaghetti for 10 minutes.
2. Toss with lemon.


Tomato Salad
Ingredients:
2 tomatoes
1 tbsp olive oil

Steps:
1. Slice the tomatoes.
2. Toss with olive oil.
"""

NUMBERED_HEADING_TEXT = """Recipe 1: Lemon Pasta
Ingredients:
200 g spaghetti
Steps:
1. Boil.
---
Recipe 2: Tomato Salad
Ingredients:
2 tomatoes
Steps:
1. Slice.
"""

NOISY_SINGLE_DISH_TEXT = """Recipe VIDEO above. Many people consider this to be the world's best one pot rice meal.
This Jambalaya is started on the stove then finished in the oven.
Ingredients
1 cup long grain rice
2 cups chicken broth
Creole Seasoning Mix:
1 tsp paprika
Instructions
Bake for 20 minutes.
Recipe Notes:
Rice should be tender but not mushy.
Nutrition Information:
Calories: 500
"""


def test_recipe_page_metadata_does_not_create_phantom_dishes() -> None:
    blocks = split_recipe_blocks(NOISY_SINGLE_DISH_TEXT)

    assert len(blocks) == 1


@pytest.mark.asyncio
async def test_rule_fallback_recovers_short_title_from_page_introduction() -> None:
    drafts = await DeterministicRecipeImportExtractor().extract(NOISY_SINGLE_DISH_TEXT)

    assert len(drafts) == 1
    assert drafts[0].name == "Jambalaya"
    assert drafts[0].steps == ("Bake for 20 minutes.",)
    assert all("Rice should" not in step and "Calories" not in step for step in drafts[0].steps)


class _NoisySingleRecipeLLMClient:
    async def chat_json(self, messages: list[dict[str, str]], **_: Any) -> dict[str, Any]:
        system = messages[0]["content"]
        if "dishes array" in system:
            assert "Recipe Notes" in system
            return {"dishes": [messages[1]["content"].splitlines()[0]]}
        assert "recipes array" in system
        assert "component mixtures" in system
        return {
            "recipes": [
                {
                    "name": "Jambalaya",
                    "servings": 6,
                    "ingredients": ["1 cup long grain rice", "2 cups chicken broth", "1 tsp paprika"],
                    "steps": ["Bake for 20 minutes."],
                }
            ]
        }


@pytest.mark.asyncio
async def test_llm_treats_recipe_notes_and_seasoning_mix_as_one_dish() -> None:
    drafts = await LLMRecipeImportExtractor(_NoisySingleRecipeLLMClient()).extract(NOISY_SINGLE_DISH_TEXT)  # type: ignore[arg-type]

    assert len(drafts) == 1
    assert drafts[0].name == "Jambalaya"
    assert drafts[0].ingredients[-1] == "1 tsp paprika"


@pytest.mark.asyncio
async def test_blank_line_separated_dishes_are_split_into_individual_drafts() -> None:
    extractor = DeterministicRecipeImportExtractor()

    drafts = await extractor.extract(DOUBLE_BLANK_DISH_TEXT)

    assert [draft.name for draft in drafts] == ["Lemon Pasta", "Tomato Salad"]
    assert drafts[0].servings == 4
    # Second dish states no servings — never defaulted, surfaced for clarification.
    assert drafts[1].servings is None


@pytest.mark.asyncio
async def test_numbered_recipe_headings_are_normalised_to_plain_names() -> None:
    extractor = DeterministicRecipeImportExtractor()

    drafts = await extractor.extract(NUMBERED_HEADING_TEXT)

    assert [draft.name for draft in drafts] == ["Lemon Pasta", "Tomato Salad"]


@pytest.mark.asyncio
async def test_multilingual_headings_and_servings_are_recognised_without_follow_up() -> None:
    extractor = DeterministicRecipeImportExtractor()

    drafts = await extractor.extract("""菜谱：番茄炒蛋
2人份
食材：
3个鸡蛋
200克番茄
步骤：
1. 炒熟鸡蛋。
""")

    assert drafts[0].name == "番茄炒蛋"
    assert drafts[0].servings == 2
    assert drafts[0].ingredients == ("3个鸡蛋", "200克番茄")


class _StructuredLLMClient:
    """Returns full-field candidates for the per-dish fan-out prompt."""

    def __init__(self) -> None:
        self.last_max_tokens: int | None = None

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.last_max_tokens = kwargs.get("max_tokens")
        text = messages[1]["content"]
        if "Lemon" in text:
            return {
                "dish_name": "Lemon Pasta",
                "original_servings": 4,
                "source_language": "eng",
                "ingredients": [
                    {"raw_text": "200 g spaghetti", "name": "spaghetti", "quantity": 200, "unit": "g"},
                    {"raw_text": "1 lemon", "name": "lemon", "quantity": 1, "unit": "piece"},
                ],
                "steps": [{"instruction": "Boil the spaghetti for 10 minutes."}],
            }
        return {
            "dish_name": "Tomato Salad",
            "original_servings": 4,
            "source_language": "eng",
            "ingredients": [{"raw_text": "2 tomatoes", "name": "tomato", "quantity": 2, "unit": "piece"}],
            "steps": [{"instruction": "Slice the tomatoes."}, {"instruction": "Toss with olive oil."}],
        }


@pytest.mark.asyncio
async def test_multi_block_input_fan_outs_per_dish_with_full_field_extraction() -> None:
    client = _StructuredLLMClient()
    extractor = LLMRecipeImportExtractor(client, max_output_tokens=4096)

    drafts = await extractor.extract(MULTI_DISH_TEXT)

    assert [draft.name for draft in drafts] == ["Lemon Pasta", "Tomato Salad"]
    assert drafts[0].ingredients == ("200 g spaghetti", "1 lemon")
    assert drafts[1].steps == ("Slice the tomatoes.", "Toss with olive oil.")
    # Fan-out goes through the full-field dish extractor (llm/extractor.py),
    # which does not need the raised import token budget.
    assert client.last_max_tokens is None


class _RecipesArrayLLMClient:
    """Split prompt → single dish; whole-text prompt → a recipes array."""

    def __init__(self) -> None:
        self.last_max_tokens: int | None = None

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.last_max_tokens = kwargs.get("max_tokens")
        system = messages[0]["content"]
        if "dishes array" in system:
            # The model judges this text as one dish → whole-text parsing.
            return {"dishes": ["Lemon Pasta with spaghetti and a tomato salad on the side"]}
        assert "recipes array" in system
        return {
            "recipes": [
                {"name": "Lemon Pasta", "servings": 4, "ingredients": ["200 g spaghetti"], "steps": ["Boil."]},
                {"name": "Tomato Salad", "servings": None, "ingredients": ["2 tomatoes"], "steps": ["Slice."]},
            ]
        }


@pytest.mark.asyncio
async def test_single_block_input_uses_recipes_array_with_raised_token_budget() -> None:
    client = _RecipesArrayLLMClient()
    extractor = LLMRecipeImportExtractor(client, max_output_tokens=4096)

    drafts = await extractor.extract("Lemon Pasta with spaghetti and a tomato salad on the side")

    assert [draft.name for draft in drafts] == ["Lemon Pasta", "Tomato Salad"]
    assert client.last_max_tokens == 4096


class _MultilingualLLMClient:
    async def chat_json(
        self,
        messages: list[dict[str, str]],
        **_: Any,
    ) -> dict[str, Any]:
        system = messages[0]["content"]
        if "dishes array" in system:
            return {"dishes": ["番茄意面"]}
        if "clarification answer values" in system:
            return {"answers": [{"question_id": "dish-1:steps", "value": "Boil the pasta and stir in the tomatoes."}]}
        assert "written in any language" in system
        assert "must be English" in system
        return {
            "recipes": [
                {
                    "name": "Tomato Pasta",
                    "servings": 2,
                    "ingredients": ["200 g pasta", "2 tomatoes"],
                    "steps": [],
                }
            ]
        }


@pytest.mark.asyncio
async def test_multilingual_recipe_and_answers_are_normalised_to_english() -> None:
    extractor = LLMRecipeImportExtractor(_MultilingualLLMClient(), max_output_tokens=4096)  # type: ignore[arg-type]
    service = ParseRecipeImportService(extractor, answer_normaliser=extractor)
    text = "番茄意面，2人份，200克意大利面和2个番茄"

    first = await service.execute(ParseRecipeImportRequest(request_id="req-multi", text=text))

    assert first.drafts[0].name == "Tomato Pasta"
    assert first.drafts[0].ingredients == ("200 g pasta", "2 tomatoes")
    assert [question.question_id for question in first.questions] == ["dish-1:steps"]

    completed = await service.execute(
        ParseRecipeImportRequest(
            request_id="req-multi",
            text=text,
            answers=({"question_id": "dish-1:steps", "value": "煮意大利面，然后加入番茄翻炒。"},),
        )
    )

    assert completed.status == "READY"
    assert completed.drafts[0].steps == ("Boil the pasta and stir in the tomatoes.",)


class _TranslationUnavailableLLMClient:
    async def chat_json(self, messages: list[dict[str, str]], **_: Any) -> dict[str, Any]:
        system = messages[0]["content"]
        if "dishes array" in system:
            return {"dishes": ["红烧肉"]}
        if "already-extracted recipe drafts" in system:
            raise LLMError("translation provider is unavailable")
        return {
            "recipes": [
                {
                    "name": "红烧肉",
                    "servings": 2,
                    "ingredients": ["500克五花肉"],
                    "steps": ["将五花肉煎至金黄。"],
                }
            ]
        }


@pytest.mark.asyncio
async def test_multilingual_import_remains_reviewable_when_english_normalisation_fails() -> None:
    extractor = LLMRecipeImportExtractor(_TranslationUnavailableLLMClient())  # type: ignore[arg-type]

    drafts = await extractor.extract("红烧肉\n2人份\n500克五花肉\n将五花肉煎至金黄。")

    assert drafts[0].name == "红烧肉"
    assert drafts[0].ingredients == ("500克五花肉",)
    assert drafts[0].steps == ("将五花肉煎至金黄。",)


class _MixedScriptLLMClient:
    async def chat_json(self, messages: list[dict[str, str]], **_: Any) -> dict[str, Any]:
        system = messages[0]["content"]
        if "dishes array" in system:
            return {"dishes": [messages[1]["content"].splitlines()[0]]}
        if "already-extracted recipe drafts" in system:
            return {
                "recipes": [
                    {
                        "draft_id": "dish-1",
                        "name": "Tomato Scrambled Eggs",
                        "servings": 2,
                        "ingredients": ["3 eggs", "200 g tomato"],
                        "steps": ["Scramble the eggs and stir-fry with the tomato."],
                    },
                    {
                        "draft_id": "dish-2",
                        "name": "Garlic Tofu",
                        "servings": 2,
                        "ingredients": ["400 g firm tofu", "20 g garlic"],
                        "steps": ["Pan-fry the tofu and add the garlic."],
                    },
                ]
            }
        text = messages[1]["content"]
        chinese = "番茄" in text
        return {
            "dish_name": "番茄炒蛋" if chinese else "Garlic Tofu",
            "original_servings": 2,
            "source_language": "zho",
            "ingredients": [
                {
                    "raw_text": "3个鸡蛋" if chinese else "400 g firm tofu",
                    "name": "鸡蛋" if chinese else "firm tofu",
                    "quantity": 3 if chinese else 400,
                    "unit": "piece" if chinese else "g",
                }
            ],
            "steps": [{"instruction": "炒熟鸡蛋。" if chinese else "Pan-fry the tofu."}],
        }


@pytest.mark.asyncio
async def test_mixed_script_multi_dish_output_is_normalised_as_one_stable_english_batch() -> None:
    extractor = LLMRecipeImportExtractor(_MixedScriptLLMClient(), max_output_tokens=4096)  # type: ignore[arg-type]

    drafts = await extractor.extract("""菜谱：番茄炒蛋
2人份
食材：
3个鸡蛋
步骤：
1. 炒熟鸡蛋。
---
菜谱：蒜香豆腐
2人份
食材：
400克硬豆腐
步骤：
1. 煎豆腐。
""")

    assert [draft.name for draft in drafts] == ["Tomato Scrambled Eggs", "Garlic Tofu"]
    assert [draft.servings for draft in drafts] == [2, 2]
    assert drafts[0].ingredients == ("3 eggs", "200 g tomato")


@pytest.mark.asyncio
async def test_pasted_ingredients_preparation_template_is_split() -> None:
    template = """Ingredients Preparation
Crab legs
Cooking steps
1. Boil.
Ingredients preparation
Fresh shrimp
Cooking steps
1. Fry.
"""

    drafts = await DeterministicRecipeImportExtractor().extract(template)

    assert [draft.name for draft in drafts] == ["Crab legs", "Fresh shrimp"]


def test_split_on_markers_located_in_order() -> None:
    text = "Fried Spare Ribs\nIngredients:\nSalt\nCooking steps\n1. Soak.\nBaked Chicken\nIngredients:\nChicken\nCooking steps\n1. Roast."

    blocks = split_on_markers(text, ("Fried Spare Ribs", "Baked Chicken"))

    assert len(blocks) == 2
    assert blocks[0].startswith("Fried Spare Ribs")
    assert blocks[1].startswith("Baked Chicken")


def test_expand_prep_boundaries_splits_glued_headings() -> None:
    block = (
        "Ingredients Preparation\n"
        "Crab legs\nCooking steps\n1. Boil.\n"
        "...meal. Ingredients preparation\n"
        "Fresh shrimp\nCooking steps\n1. Fry.\n"
        "...serve. Ingredients preparation\n"
        "Prawns\nCooking steps\n1. Sear."
    )

    parts = expand_prep_boundaries(block)

    assert len(parts) == 3
    assert parts[0].startswith("Ingredients Preparation")
    assert "Crab legs" in parts[0]
    assert parts[1].startswith("Ingredients preparation")
    assert "Fresh shrimp" in parts[1]
    assert parts[2].startswith("Ingredients preparation")
    assert "Prawns" in parts[2]


class _GuidedLLMClient:
    """Split prompt → dish markers; dish prompt → full candidate by content."""

    def __init__(self, markers: tuple[str, ...]) -> None:
        self.markers = markers

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        system = messages[0]["content"]
        if "dishes array" in system:
            return {"dishes": list(self.markers)}
        text = messages[1]["content"]
        if "Ribs" in text:
            return {
                "dish_name": "Fried Spare Ribs",
                "original_servings": 2,
                "source_language": "eng",
                "ingredients": [{"raw_text": "Salt", "name": "salt"}],
                "steps": [{"instruction": "Soak the ribs."}],
            }
        return {
            "dish_name": "Baked Chicken",
            "original_servings": 2,
            "source_language": "eng",
            "ingredients": [{"raw_text": "Chicken", "name": "chicken"}],
            "steps": [{"instruction": "Roast the chicken."}],
        }


@pytest.mark.asyncio
async def test_unstructured_text_is_split_by_llm_markers_then_fanned_out() -> None:
    text = "Fried Spare Ribs\nIngredients:\nSalt\nCooking steps\n1. Soak.\nBaked Chicken\nIngredients:\nChicken\nCooking steps\n1. Roast."
    client = _GuidedLLMClient(("Fried Spare Ribs", "Baked Chicken"))
    extractor = LLMRecipeImportExtractor(client, max_output_tokens=4096)

    drafts = await extractor.extract(text)

    assert [draft.name for draft in drafts] == ["Fried Spare Ribs", "Baked Chicken"]
    assert drafts[0].ingredients == ("Salt",)
    assert drafts[1].steps == ("Roast the chicken.",)


@pytest.mark.asyncio
async def test_rule_fallback_truncates_steps_over_the_draft_limit() -> None:
    many_steps = "One Pot Stew\nIngredients:\nPotatoes\nCooking steps\n" + "\n".join(
        f"{index}. Step number {index}." for index in range(1, 120)
    )

    drafts = await DeterministicRecipeImportExtractor().extract(many_steps)

    assert len(drafts) == 1
    assert len(drafts[0].steps) == 100
