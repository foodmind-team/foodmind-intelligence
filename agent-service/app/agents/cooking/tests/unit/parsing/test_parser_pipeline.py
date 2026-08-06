"""Unit tests for the LLM parsing pipeline — handbook 4.15.

Covers: extraction, gap detection, local inference, IR building, and
semantic validation — across 7 real Chinese golden recipe fixtures:
蟹脚、蒜蓉鲜虾、豆瓣基围虾、排骨汤、手撕包菜、腊肠菜花、辣炒鸡翅.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from cooking_plan_agent.domain.enums import HeatLevel
from cooking_plan_agent.domain.models import ExtractedIngredient, ExtractedRecipeCandidate, ExtractedStep
from cooking_plan_agent.parsing.extractor import RecipeExtractor
from cooking_plan_agent.parsing.gaps import GapClass, find_recipe_gaps
from cooking_plan_agent.parsing.golden_fixtures import GOLDEN_FIXTURES
from cooking_plan_agent.parsing.inference import infer_local, merge_inference
from cooking_plan_agent.parsing.ir_builder import build_recipe_ir, validate_recipe_ir_semantics

# =============================================================================
# RecipeExtractor tests — extraction quality per recipe
# =============================================================================


class TestRecipeExtractor:
    """Rule-based extraction from real Chinese recipes."""

    @pytest.mark.asyncio
    async def test_extract_crab_legs(self) -> None:
        """蟹脚: 14 ingredients, 7 steps, Chinese, detects simmer+mid heat."""
        extractor = RecipeExtractor()
        candidate = await extractor.extract(GOLDEN_FIXTURES["crab_legs"])

        assert "蟹脚" in candidate.dish_name
        assert candidate.source_language == "zho"
        assert len(candidate.ingredients) >= 10
        assert len(candidate.steps) >= 7

        # Step 5: "中火煮5-6分钟" → duration detected
        step5 = candidate.steps[4] if len(candidate.steps) > 4 else None
        assert step5 is not None
        has_duration = step5.passive_duration_minutes is not None or step5.active_duration_minutes is not None
        assert has_duration, f"Step 5 should have duration detected: {step5}"

    @pytest.mark.asyncio
    async def test_extract_garlic_shrimp(self) -> None:
        """蒜蓉鲜虾: detects stir-fry technique, high heat (大火)."""
        extractor = RecipeExtractor()
        candidate = await extractor.extract(GOLDEN_FIXTURES["garlic_shrimp"])

        assert candidate.source_language == "zho"
        assert len(candidate.ingredients) >= 8
        assert len(candidate.steps) >= 5

        # Step 3: "大火炒出香味" → high heat detected
        heat_steps = [s for s in candidate.steps if s.heat_level != HeatLevel.NONE]
        assert len(heat_steps) > 0, "Should detect heat level in at least one step"

        # Shrimp → shellfish allergen check (in IR builder)
        names = [i.name for i in candidate.ingredients]
        assert any("虾" in n for n in names)

    @pytest.mark.asyncio
    async def test_extract_spicy_prawns(self) -> None:
        """豆瓣基围虾: ingredient with quantity parsing (500克, 3-4瓣)."""
        extractor = RecipeExtractor()
        candidate = await extractor.extract(GOLDEN_FIXTURES["spicy_prawns"])

        assert len(candidate.ingredients) >= 10
        assert len(candidate.steps) >= 5

        # "基围虾500克" → quantity parsed
        prawn_ingredients = [i for i in candidate.ingredients if "基围虾" in i.name]
        assert len(prawn_ingredients) > 0

    @pytest.mark.asyncio
    async def test_extract_rib_soup(self) -> None:
        """排骨汤: minimal recipe, few ingredients+steps, detects boil."""
        extractor = RecipeExtractor()
        candidate = await extractor.extract(GOLDEN_FIXTURES["rib_soup"])

        assert candidate.source_language == "zho"
        assert len(candidate.ingredients) >= 5
        assert len(candidate.steps) >= 4

    @pytest.mark.asyncio
    async def test_extract_hand_torn_cabbage(self) -> None:
        """手撕包菜: high-heat stir-fry, specific quantities (1.5g salt, 20g garlic)."""
        extractor = RecipeExtractor()
        candidate = await extractor.extract(GOLDEN_FIXTURES["hand_torn_cabbage"])

        assert len(candidate.steps) >= 5

        # "大火翻炒30秒" → heat detected
        heat_steps = [s for s in candidate.steps if s.heat_level == HeatLevel.HIGH]
        assert len(heat_steps) > 0

    @pytest.mark.asyncio
    async def test_extract_sausage_cauliflower(self) -> None:
        """腊肠菜花: flat ingredient structure (no 主料/辅料 subsections)."""
        extractor = RecipeExtractor()
        candidate = await extractor.extract(GOLDEN_FIXTURES["sausage_cauliflower"])

        assert len(candidate.ingredients) >= 8
        assert len(candidate.steps) >= 6

    @pytest.mark.asyncio
    async def test_extract_chicken_wings(self) -> None:
        """辣炒鸡翅: 15 ingredients, chicken safety,腌料/辅料 subsections."""
        extractor = RecipeExtractor()
        candidate = await extractor.extract(GOLDEN_FIXTURES["chicken_wings"])

        assert len(candidate.ingredients) >= 10
        assert len(candidate.steps) >= 6

        # Should detect "鸡翅" or "鸡翅中" in ingredients
        names = [i.name for i in candidate.ingredients]
        assert any("鸡翅" in n for n in names), f"Ingredient names: {names}"

    @pytest.mark.asyncio
    async def test_extract_noise_free_names(self) -> None:
        """Spicy prawns: ingredient names are cleaned of optional markers,
        trailing qualifiers, trailing punctuation, and parenthetical notes."""
        extractor = RecipeExtractor()
        candidate = await extractor.extract(GOLDEN_FIXTURES["spicy_prawns"])

        names = [i.name for i in candidate.ingredients]
        # 2. 可选标记剥离: "干辣椒（可选）"
        assert "干辣椒" in names, f"干辣椒（可选） should drop the optional marker: {names}"
        # 3. 尾量词剥离: "老抽少许"
        assert "老抽" in names, f"老抽少许 should drop the qualifier: {names}"
        # 5. 括号注释剥离且不再误判为步骤: "小米辣（依吃辣程度放）"
        assert "小米辣" in names, f"小米辣（依吃辣程度放） should survive: {names}"
        assert not any("（可选）" in n or "少许" in n or "（依" in n for n in names)

    @pytest.mark.asyncio
    async def test_extract_trailing_punctuation_cleaned(self) -> None:
        """'白胡椒粉、' — a trailing Chinese comma must not stick to the name."""
        extractor = RecipeExtractor()
        candidate = await extractor.extract("香辣蟹\n食材：\n主料：\n蟹脚\n辅料：\n白胡椒粉、\n步骤：\n1. 翻炒调味。\n")
        names = [i.name for i in candidate.ingredients]
        assert "白胡椒粉" in names, f"白胡椒粉、 should be cleaned: {names}"
        assert not any("、" in n for n in names)

    @pytest.mark.asyncio
    async def test_extract_quantity_range_and_slash_alternatives(self) -> None:
        """Spicy prawns: '大蒜3-4瓣' parses quantity range with 瓣 unit, and
        '味精/鸡精（可选）' splits into independent alternatives."""
        extractor = RecipeExtractor()
        candidate = await extractor.extract(GOLDEN_FIXTURES["spicy_prawns"])

        # 4. 数量范围取上限 + 瓣→piece
        garlic = [i for i in candidate.ingredients if i.name == "大蒜"]
        assert garlic, "大蒜 should be present"
        assert garlic[0].quantity == Decimal(4), f"3-4瓣 should take the upper bound: {garlic[0]}"
        assert garlic[0].unit == "piece", f"瓣 should map to piece: {garlic[0]}"

        # 2. / 候选拆分: "味精/鸡精" → 味精 + 鸡精
        names = [i.name for i in candidate.ingredients]
        assert "味精" in names and "鸡精" in names, f"味精/鸡精 should split: {names}"
        assert not any("/" in n for n in names), f"No slash should remain: {names}"


def test_frying_step_with_marinated_food_is_not_a_marination_step() -> None:
    """“腌好的鸡翅下锅煎制” describes frying, not another marinade wait."""
    candidate = ExtractedRecipeCandidate(
        recipe_id="r7",
        dish_name="香辣鸡翅",
        original_servings=Decimal(2),
        source_language="zho",
        ingredients=(ExtractedIngredient(raw_text="鸡翅中", name="鸡翅中", quantity=Decimal(15), unit="个"),),
        steps=(
            ExtractedStep(
                step_number=1,
                instruction="煎制鸡翅：将腌好的鸡翅下锅，煎至两面焦黄。",
                category="heating",
            ),
        ),
    )

    recipe = build_recipe_ir(candidate, request_recipe_id="r7")
    assert recipe.steps[0].pattern == "stir_fry"

    @pytest.mark.asyncio
    async def test_extract_duration_detection(self) -> None:
        """'5-6分钟' and '20分钟' durations should be detected."""
        extractor = RecipeExtractor()

        # 蟹脚 step 5: "中火煮5-6分钟"
        candidate = await extractor.extract(GOLDEN_FIXTURES["crab_legs"])
        step5 = candidate.steps[4]
        assert step5.passive_duration_minutes is not None or step5.active_duration_minutes is not None

        # 鸡翅 step 2: "腌制20分钟"
        candidate2 = await extractor.extract(GOLDEN_FIXTURES["chicken_wings"])
        step2 = candidate2.steps[1] if len(candidate2.steps) > 1 else None
        assert step2 is not None

    @pytest.mark.asyncio
    async def test_default_servings_is_two(self) -> None:
        """When no explicit servings, default to 2."""
        extractor = RecipeExtractor()
        candidate = await extractor.extract(GOLDEN_FIXTURES["rib_soup"])
        assert candidate.original_servings == 2


# =============================================================================
# Gap detection tests
# =============================================================================


class TestGapDetection:
    """Gap detection and classification on real recipes."""

    @pytest.mark.asyncio
    async def test_well_specified_recipe_few_critical(self) -> None:
        """蟹脚: well-specified → critical gaps should be fillable by inference."""
        extractor = RecipeExtractor()
        candidate = await extractor.extract(GOLDEN_FIXTURES["crab_legs"])
        gaps = find_recipe_gaps(candidate)

        result = infer_local(candidate, gaps)
        unresolved_critical = [
            g for g in result.unresolved_gaps if g.gap_class in (GapClass.CRITICAL, GapClass.SAFETY_CRITICAL)
        ]
        assert len(unresolved_critical) == 0, (
            f"Unresolved critical gaps: {[g.description for g in unresolved_critical]}"
        )

    @pytest.mark.asyncio
    async def test_underspecified_recipe_has_duration_gaps(self) -> None:
        """排骨汤: simmer without explicit time → duration gaps."""
        extractor = RecipeExtractor()
        candidate = await extractor.extract(GOLDEN_FIXTURES["rib_soup"])
        gaps = find_recipe_gaps(candidate)

        # At least one gap should exist (simmer step without duration)
        assert len(gaps) > 0, "Underspecified recipe should produce gaps"

    @pytest.mark.asyncio
    async def test_gap_has_required_fields(self) -> None:
        """Every gap must carry recipe_id, field_path, gap_class, description."""
        extractor = RecipeExtractor()
        candidate = await extractor.extract(GOLDEN_FIXTURES["hand_torn_cabbage"])
        gaps = find_recipe_gaps(candidate)

        for gap in gaps:
            assert gap.recipe_id, f"Missing recipe_id in gap: {gap}"
            assert gap.field_path, f"Missing field_path in gap: {gap}"
            assert gap.gap_class, f"Missing gap_class in gap: {gap}"
            assert gap.description, f"Missing description in gap: {gap}"

    @pytest.mark.asyncio
    async def test_gap_classes_are_valid(self) -> None:
        """All gaps must use one of the five valid gap classes."""
        extractor = RecipeExtractor()
        candidate = await extractor.extract(GOLDEN_FIXTURES["rib_soup"])
        gaps = find_recipe_gaps(candidate)

        valid = {
            GapClass.CRITICAL,
            GapClass.SAFETY_CRITICAL,
            GapClass.RESOURCE_CRITICAL,
            GapClass.OPTIMISATION,
            GapClass.COSMETIC,
        }
        for gap in gaps:
            assert gap.gap_class in valid, f"Invalid gap class: {gap.gap_class}"


# =============================================================================
# Local inference tests
# =============================================================================


class TestLocalInference:
    """Local cooking knowledge inference with real recipes."""

    @pytest.mark.asyncio
    async def test_infer_heat_from_boil_step(self) -> None:
        """'焯水' (blanch/boil) step → should infer HIGH heat."""
        extractor = RecipeExtractor()
        candidate = await extractor.extract(GOLDEN_FIXTURES["rib_soup"])
        gaps = find_recipe_gaps(candidate)

        result = infer_local(candidate, gaps)
        assert len(result.assumptions) > 0, "Should make at least one assumption"

    @pytest.mark.asyncio
    async def test_merge_preserves_step_count(self) -> None:
        """Merge should not change the number of steps."""
        extractor = RecipeExtractor()
        candidate = await extractor.extract(GOLDEN_FIXTURES["rib_soup"])
        gaps = find_recipe_gaps(candidate)

        result = infer_local(candidate, gaps)
        updated = merge_inference(candidate, result)
        assert len(updated.steps) == len(candidate.steps)

    @pytest.mark.asyncio
    async def test_non_critical_gaps_not_inferred(self) -> None:
        """Only critical/safety_critical gaps trigger local inference."""
        extractor = RecipeExtractor()
        candidate = await extractor.extract(GOLDEN_FIXTURES["crab_legs"])
        all_gaps = find_recipe_gaps(candidate)
        non_critical = tuple(g for g in all_gaps if g.gap_class not in (GapClass.CRITICAL, GapClass.SAFETY_CRITICAL))

        result = infer_local(candidate, non_critical)
        assert len(result.filled_gaps) == 0

    @pytest.mark.asyncio
    async def test_chicken_wings_safety_not_guessed(self) -> None:
        """辣炒鸡翅: protein safety temps must not be guessed locally."""
        extractor = RecipeExtractor()
        candidate = await extractor.extract(GOLDEN_FIXTURES["chicken_wings"])
        gaps = find_recipe_gaps(candidate)

        result = infer_local(candidate, gaps)
        for gap in result.filled_gaps:
            if gap.gap_class == GapClass.SAFETY_CRITICAL:
                assert gap.confidence <= Decimal("0.5"), f"Safety-critical gap should have low confidence: {gap}"


# =============================================================================
# IR Builder tests
# =============================================================================


class TestIRBuilder:
    """RecipeIR construction and validation."""

    @pytest.mark.asyncio
    async def test_build_ir_from_garlic_shrimp(self) -> None:
        """Full extract → build IR pipeline on 蒜蓉鲜虾."""
        extractor = RecipeExtractor()
        candidate = await extractor.extract(GOLDEN_FIXTURES["garlic_shrimp"])
        recipe_ir = build_recipe_ir(candidate)

        assert recipe_ir.dish_name == candidate.dish_name
        assert len(recipe_ir.ingredients) == len(candidate.ingredients)
        assert len(recipe_ir.steps) == len(candidate.steps)
        assert recipe_ir.source_language == "zho"

    @pytest.mark.asyncio
    async def test_validate_passes_on_cabbage(self) -> None:
        """手撕包菜 should pass semantic validation."""
        extractor = RecipeExtractor()
        candidate = await extractor.extract(GOLDEN_FIXTURES["hand_torn_cabbage"])
        recipe_ir = build_recipe_ir(candidate)
        report = validate_recipe_ir_semantics((recipe_ir,))

        assert report.passed, f"Validation failed: {[i.message for i in report.issues]}"
        assert report.recipe_count == 1

    @pytest.mark.asyncio
    async def test_shellfish_allergen_detected(self) -> None:
        """虾 in ingredients → shellfish allergen tag."""
        extractor = RecipeExtractor()
        candidate = await extractor.extract(GOLDEN_FIXTURES["garlic_shrimp"])
        recipe_ir = build_recipe_ir(candidate)

        shellfish = [i for i in recipe_ir.ingredients if "shellfish" in i.allergen_tags]
        assert len(shellfish) > 0, "虾 should be tagged as shellfish allergen"

    @pytest.mark.asyncio
    async def test_build_ir_crab_legs(self) -> None:
        """蟹脚: full IR build with 14+ ingredients, 7 steps."""
        extractor = RecipeExtractor()
        candidate = await extractor.extract(GOLDEN_FIXTURES["crab_legs"])
        recipe_ir = build_recipe_ir(candidate)

        assert len(recipe_ir.ingredients) >= 10
        assert len(recipe_ir.steps) >= 7
        assert recipe_ir.source_language == "zho"

    @pytest.mark.asyncio
    async def test_rejects_empty_ingredients(self) -> None:
        """Pydantic model_validator must reject zero-ingredient IRs."""
        from cooking_plan_agent.domain.models import RecipeIR

        with pytest.raises(ValueError):
            RecipeIR(
                recipe_id="test",
                dish_name="Empty",
                original_servings=Decimal(2),
                target_servings=Decimal(2),
                source_language="zho",
                ingredients=(),
                steps=(),
            )


# =============================================================================
# Pipeline integration tests
# =============================================================================


class TestParserPipeline:
    """End-to-end: text → candidate → gaps → inference → IR → validate."""

    @pytest.mark.asyncio
    async def test_full_pipeline_crab_legs(self) -> None:
        """Full pipeline on 蟹脚."""
        extractor = RecipeExtractor()
        candidate = await extractor.extract(GOLDEN_FIXTURES["crab_legs"])
        assert len(candidate.ingredients) >= 10
        assert len(candidate.steps) >= 7

        gaps = find_recipe_gaps(candidate)
        result = infer_local(candidate, gaps)
        updated = merge_inference(candidate, result)
        recipe_ir = build_recipe_ir(updated)
        report = validate_recipe_ir_semantics((recipe_ir,))

        assert report.passed, f"Failed: {[i.message for i in report.issues]}"

    @pytest.mark.asyncio
    async def test_full_pipeline_garlic_shrimp(self) -> None:
        """Full pipeline on 蒜蓉鲜虾: gaps → inference → IR."""
        extractor = RecipeExtractor()
        candidate = await extractor.extract(GOLDEN_FIXTURES["garlic_shrimp"])
        gaps = find_recipe_gaps(candidate)
        result = infer_local(candidate, gaps)
        updated = merge_inference(candidate, result)

        unresolved = [g for g in result.unresolved_gaps if g.gap_class == GapClass.CRITICAL]
        assert len(unresolved) == 0, f"Unresolved critical: {[g.description for g in unresolved]}"

        recipe_ir = build_recipe_ir(updated)
        report = validate_recipe_ir_semantics((recipe_ir,))
        assert report.passed

    @pytest.mark.asyncio
    async def test_full_pipeline_rib_soup(self) -> None:
        """Full pipeline on 排骨汤 (underspecified → inference fills gaps)."""
        extractor = RecipeExtractor()
        candidate = await extractor.extract(GOLDEN_FIXTURES["rib_soup"])
        gaps = find_recipe_gaps(candidate)
        result = infer_local(candidate, gaps)
        updated = merge_inference(candidate, result)
        recipe_ir = build_recipe_ir(updated)
        report = validate_recipe_ir_semantics((recipe_ir,))
        assert report.passed

    @pytest.mark.asyncio
    async def test_all_seven_fixtures_validate(self) -> None:
        """Every one of the 7 golden fixtures must produce a valid RecipeIR."""
        extractor = RecipeExtractor()

        for name, text in GOLDEN_FIXTURES.items():
            candidate = await extractor.extract(text)
            gaps = find_recipe_gaps(candidate)
            result = infer_local(candidate, gaps)
            updated = merge_inference(candidate, result)
            recipe_ir = build_recipe_ir(updated)
            report = validate_recipe_ir_semantics((recipe_ir,))

            assert report.passed, f"Fixture '{name}' failed validation:\n" + "\n".join(
                f"  [{i.severity}] {i.message}" for i in report.issues
            )
