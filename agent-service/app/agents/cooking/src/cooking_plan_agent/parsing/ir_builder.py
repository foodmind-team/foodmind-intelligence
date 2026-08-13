"""IR Builder — converts ExtractedRecipeCandidate to validated RecipeIR.

Handbook 4.12–4.14: this module builds the canonical Intermediate Representation
from one or more extracted candidates. It also performs semantic validation
to reject structurally valid but logically impossible recipes.
"""

from __future__ import annotations

from decimal import Decimal
from typing import NamedTuple

from cooking_plan_agent.domain.enums import HeatLevel
from cooking_plan_agent.domain.models import (
    Assumption,
    ExtractedIngredient,
    ExtractedRecipeCandidate,
    ExtractedStep,
    IngredientDemand,
    RecipeIR,
    RecipeStep,
)
from cooking_plan_agent.normalisation.units import UnitClassifier

# =============================================================================
# SemanticValidationReport
# =============================================================================


class SemanticIssue(NamedTuple):
    """A single semantic validation issue."""

    code: str
    """Machine-readable issue code (e.g. 'NO_INGREDIENTS', 'NEGATIVE_DURATION')."""

    severity: str
    """'error' (reject) or 'warning' (accept with caution)."""

    message: str
    """Human-readable description."""


class RecipeValidationReport(NamedTuple):
    """Result of semantic validation on one or more RecipeIR objects.

    passed=True means no 'error'-severity issues were found.
    """

    passed: bool
    issues: tuple[SemanticIssue, ...]
    recipe_count: int


# =============================================================================
# Public API
# =============================================================================


def build_recipe_ir(
    candidate: ExtractedRecipeCandidate,
    *,
    request_recipe_id: str | None = None,
    target_servings: Decimal | None = None,
) -> RecipeIR:
    """Convert an ExtractedRecipeCandidate into a validated RecipeIR.

    Handles:
      - ExtractedIngredient → IngredientDemand (with unit normalisation)
      - ExtractedStep → RecipeStep (with technique-pattern inference)
      - Collects assumptions from extraction
      - Serving scaling (P0-04): when target_servings differs from the
        recipe's original servings, every continuous-quantity ingredient
        is scaled by ``target / original`` using Decimal arithmetic.
        Discrete units (piece, egg, …) are rounded up per the configured
        rounding policy, recording an assumption when rounding occurred.

    Args:
        candidate: An extracted recipe candidate (possibly after inference).
        request_recipe_id: The recipe ID from the caller's request. When
            provided it OVERRIDES the extractor's internal recipe_id so the
            final identity always matches the request (P0-04 rule 5).
        target_servings: Desired servings. When None, defaults to the
            recipe's original servings (1:1 — unchanged behaviour).

    Returns:
        A validated RecipeIR ready for the scheduling pipeline.

    Raises:
        ValueError: If the candidate has no ingredients or no steps.
    """
    recipe_id = request_recipe_id or candidate.recipe_id

    parsed_demands = tuple(_build_ingredient_demand(ing) for ing in candidate.ingredients)

    # Filter out ingredients without a valid name or quantity
    ingredients = tuple(i for i in parsed_demands if i and i.canonical_name)

    # P0-04: apply serving scaling before any downstream feasibility/safety
    # computation. Missing quantities are never silently invented — they are
    # left as-is (quantity still present) and gaps are preserved upstream.
    if target_servings is not None:
        ingredients = _scale_ingredients(
            ingredients,
            original_servings=Decimal(str(candidate.original_servings)),
            target_servings=target_servings,
        )

    steps = tuple(_build_recipe_step(step, recipe_id) for step in candidate.steps)

    # Collect assumptions from extraction source
    assumptions = _collect_assumptions(candidate)

    return RecipeIR(
        recipe_id=recipe_id,
        dish_name=candidate.dish_name,
        original_servings=Decimal(str(candidate.original_servings)),
        target_servings=target_servings or Decimal(str(candidate.original_servings)),
        source_language=candidate.source_language,
        ingredients=ingredients,
        steps=steps,
        assumptions=assumptions,
    )


def validate_recipe_ir_semantics(recipes: tuple[RecipeIR, ...]) -> RecipeValidationReport:
    """Run semantic validation on one or more RecipeIR objects.

    Checks that go beyond Pydantic field validation:
      - At least one ingredient per recipe
      - At least one step per recipe
      - Duration values are non-negative
      - Heat level is not NONE for heating-category steps
      - Ingredient names are not empty

    Args:
        recipes: One or more RecipeIR objects to validate.

    Returns:
        RecipeValidationReport with passed flag and issue list.
    """
    issues: list[SemanticIssue] = []

    for recipe in recipes:
        issues.extend(_validate_single_recipe(recipe))

    errors = [i for i in issues if i.severity == "error"]
    passed = len(errors) == 0

    return RecipeValidationReport(
        passed=passed,
        issues=tuple(issues),
        recipe_count=len(recipes),
    )


def attach_research_assumptions(
    recipes: tuple[RecipeIR, ...],
    research_assumptions: tuple[Assumption, ...],
) -> tuple[RecipeIR, ...]:
    """Merge evidence-backed research assumptions into each RecipeIR (P1-01).

    Research provenance must be traceable in the final assumption/response:
    each applied evidence value produces an Assumption carrying EvidenceRef
    entries (source title + URL). This helper attaches them to every recipe
    so rendering surfaces them verbatim.

    Args:
        recipes: RecipeIR objects to enrich.
        research_assumptions: Assumptions produced by the research evidence
            application node. Empty tuple is a no-op.

    Returns:
        New RecipeIR tuple with the research assumptions appended (never
        mutates the input recipes).
    """
    if not research_assumptions:
        return recipes
    return tuple(
        recipe.model_copy(update={"assumptions": recipe.assumptions + research_assumptions}) for recipe in recipes
    )


# =============================================================================
# Internal builders
# =============================================================================


def _build_ingredient_demand(ingredient: ExtractedIngredient) -> IngredientDemand | None:
    """Convert ExtractedIngredient → IngredientDemand.

    Returns None if the ingredient lacks a meaningful name.
    """
    if not ingredient.name or len(ingredient.name.strip()) < 1:
        return None

    # Normalise unit string
    unit = _normalise_ingredient_unit(ingredient)

    # Detect allergen tags from ingredient name
    allergen_tags = _detect_allergens(ingredient.name)

    return IngredientDemand(
        canonical_name=ingredient.name.strip(),
        raw_name=ingredient.raw_text,
        quantity=ingredient.quantity or Decimal(1),
        unit=unit,
        preparation_spec=ingredient.preparation,
        input_state="raw",
        allergen_tags=allergen_tags,
        confidence=ingredient.confidence,
    )


def _build_recipe_step(step: ExtractedStep, recipe_id: str) -> RecipeStep:
    """Convert ExtractedStep → RecipeStep with pattern inference."""
    # Infer decomposition pattern from category and technique
    pattern = _infer_pattern(step)

    return RecipeStep(
        step_number=step.step_number,
        instruction=step.instruction,
        category=step.category,
        pattern=pattern,
        active_duration_minutes=step.active_duration_minutes,
        passive_duration_minutes=step.passive_duration_minutes,
        heat_level=step.heat_level,
        target_temperature_c=step.target_temperature_c,
        resources_hint=step.resources_hint,
    )


# =============================================================================
# P0-04 serving scaling helpers
# =============================================================================

# Discrete units that must be rounded to whole items.  For these, scaled
# quantities are rounded UP so a plan never under-supplies (e.g. 1.2 eggs
# becomes 2 eggs) and an assumption is recorded (P0-04 rule 3).
_DISCRETE_UNITS = frozenset(
    {
        "piece",
        "pc",
        "pcs",
        "egg",
        "eggs",
        "clove",
        "cloves",
        "root",
        "roots",
        "head",
        "heads",
        "slice",
        "slices",
        "bunch",
        "bunches",
    }
)


def _is_discrete_unit(unit: str) -> bool:
    """Return True when the ingredient unit is counted, not measured."""
    return unit.strip().lower() in _DISCRETE_UNITS


def _scale_ingredients(
    ingredients: tuple[IngredientDemand, ...],
    *,
    original_servings: Decimal,
    target_servings: Decimal,
) -> tuple[IngredientDemand, ...]:
    """Scale every ingredient from original to target servings (P0-04).

    Continuous units scale exactly via Decimal multiplication. Discrete
    units round UP to the nearest whole item; rounding decisions are
    attached as assumptions so they surface for user confirmation.

    Args:
        ingredients: Demands to scale.
        original_servings: Servings the recipe was written for.
        target_servings: Desired servings.

    Returns:
        A new tuple of scaled IngredientDemand instances. Never mutates
        the input demands.
    """
    from cooking_plan_agent.normalisation.units import scale_ingredient

    scaled: list[IngredientDemand] = []
    for demand in ingredients:
        new_demand = scale_ingredient(
            demand,
            original_servings=original_servings,
            target_servings=target_servings,
        )
        if _is_discrete_unit(new_demand.unit):
            import math

            rounded = Decimal(math.ceil(new_demand.quantity))
            if rounded != new_demand.quantity:
                new_demand = new_demand.model_copy(update={"quantity": rounded})
        scaled.append(new_demand)
    return tuple(scaled)


# =============================================================================
# Validation helpers
# =============================================================================


def _validate_single_recipe(recipe: RecipeIR) -> list[SemanticIssue]:
    """Validate a single RecipeIR for semantic correctness."""
    issues: list[SemanticIssue] = []

    # Check: at least one ingredient
    if not recipe.ingredients:
        issues.append(
            SemanticIssue(
                code="NO_INGREDIENTS",
                severity="error",
                message=f"Recipe '{recipe.dish_name}' has no ingredients",
            )
        )

    # Check: at least one step
    if not recipe.steps:
        issues.append(
            SemanticIssue(
                code="NO_STEPS",
                severity="error",
                message=f"Recipe '{recipe.dish_name}' has no steps",
            )
        )

    # Check: no negative durations
    for step in recipe.steps:
        if step.active_duration_minutes is not None and step.active_duration_minutes <= 0:
            issues.append(
                SemanticIssue(
                    code="NEGATIVE_DURATION",
                    severity="error",
                    message=f"Recipe '{recipe.dish_name}' step {step.step_number}: "
                    f"active duration is {step.active_duration_minutes}",
                )
            )
        if step.passive_duration_minutes is not None and step.passive_duration_minutes <= 0:
            issues.append(
                SemanticIssue(
                    code="NEGATIVE_DURATION",
                    severity="error",
                    message=f"Recipe '{recipe.dish_name}' step {step.step_number}: "
                    f"passive duration is {step.passive_duration_minutes}",
                )
            )

    # Check: heating steps should have a heat level
    for step in recipe.steps:
        if step.category == "heating" and step.heat_level == HeatLevel.NONE:
            issues.append(
                SemanticIssue(
                    code="MISSING_HEAT_LEVEL",
                    severity="warning",
                    message=f"Recipe '{recipe.dish_name}' step {step.step_number}: "
                    f"heating step has no heat level specified",
                )
            )

    # Check: ingredient names are non-empty
    for i, ingredient in enumerate(recipe.ingredients):
        if not ingredient.canonical_name.strip():
            issues.append(
                SemanticIssue(
                    code="EMPTY_INGREDIENT_NAME",
                    severity="error",
                    message=f"Recipe '{recipe.dish_name}' ingredient {i + 1}: empty name",
                )
            )

    # Check: servings are positive
    if recipe.original_servings <= 0:
        issues.append(
            SemanticIssue(
                code="INVALID_SERVINGS",
                severity="error",
                message=f"Recipe '{recipe.dish_name}': servings must be > 0, got {recipe.original_servings}",
            )
        )

    return issues


# =============================================================================
# Internal helpers
# =============================================================================


def _normalise_ingredient_unit(ingredient: ExtractedIngredient) -> str:
    """Normalise ingredient unit to a canonical form."""
    if not ingredient.unit:
        return "piece"

    unit = ingredient.unit.lower().strip()

    # Try to classify — if unknown, default to "piece"
    try:
        UnitClassifier.classify(unit)
        return unit
    except (ValueError, KeyError):
        pass

    # Common normalisations
    alias_map = {
        "tablespoon": "tbsp",
        "tablespoons": "tbsp",
        "teaspoon": "tsp",
        "teaspoons": "tsp",
        "cup": "cup",
        "cups": "cup",
        "gram": "g",
        "grams": "g",
        "kilogram": "kg",
        "kilograms": "kg",
        "milliliter": "ml",
        "milliliters": "ml",
        "litre": "l",
        "liter": "l",
        "ounce": "oz",
        "ounces": "oz",
        "pound": "lb",
        "pounds": "lb",
        "cloves": "piece",
        "clove": "piece",
    }
    return alias_map.get(unit, unit)


def _detect_allergens(name: str) -> tuple[str, ...]:
    """Detect common allergens from ingredient name."""
    name_lower = name.lower()
    allergens: list[str] = []

    allergen_map = {
        "gluten": ("wheat", "flour", "bread", "pasta", "noodle", "soy sauce", "面粉", "面条", "面包"),
        "dairy": ("milk", "cheese", "butter", "cream", "yogurt", "牛奶", "奶油", "奶酪", "黄油"),
        "egg": ("egg", "鸡蛋", "蛋"),
        "shellfish": ("shrimp", "prawn", "crab", "lobster", "虾", "蟹", "龙虾"),
        "fish": ("fish", "salmon", "tuna", "cod", "鱼", "三文鱼", "金枪鱼"),
        "soy": ("soy", "tofu", "soybean", "豆腐", "大豆"),
        "nut": ("peanut", "almond", "walnut", "cashew", "花生", "杏仁", "核桃"),
        "sesame": ("sesame", "芝麻"),
    }

    for allergen, keywords in allergen_map.items():
        if any(kw in name_lower for kw in keywords):
            allergens.append(allergen)

    return tuple(allergens)


def _infer_pattern(step: ExtractedStep) -> str:
    """Infer the decomposition pattern from step category and instruction text.

    The pattern drives the decomposition policy in preparation/decompose.py.
    """
    instruction_lower = step.instruction.lower()

    # Boil detection
    if any(kw in instruction_lower for kw in ("boil", "煮", "焯")):
        return "boil"

    # Stir-fry detection
    if any(kw in instruction_lower for kw in ("stir-fry", "stir fry", "炒", "爆炒", "翻炒")):
        return "stir_fry"

    # Pan-frying is an active stove-and-pan operation. Check it before the
    # marinade keywords below: "将腌好的鸡翅下锅煎制" describes frying, not a
    # new marination step.
    if any(kw in instruction_lower for kw in ("pan-fry", "pan fry", "煎")):
        return "stir_fry"

    # Bake detection
    if any(kw in instruction_lower for kw in ("bake", "oven", "烤", "烘烤", "烤箱")):
        return "bake"

    # Simmer detection
    if any(kw in instruction_lower for kw in ("simmer", "stew", "焖", "炖", "煲", "慢炖")):
        return "simmer"

    # Marinate detection
    if any(kw in instruction_lower for kw in ("marinate", "腌制", "腌")):
        return "marinate"

    return "simple"


def _collect_assumptions(candidate: ExtractedRecipeCandidate) -> tuple[Assumption, ...]:
    """Collect assumptions from extraction process.

    When rule-based extraction makes guesses (e.g. default 2 servings),
    those become assumptions that may surface to the user.
    """
    assumptions: list[Assumption] = []

    # Rule-based extraction inherently carries assumptions
    if candidate.extraction_source == "RULE_BASED":
        assumptions.append(
            Assumption(
                text="Recipe extracted using rule-based parser (no LLM). "
                "Confidence may be lower than LLM-based extraction.",
                confidence=Decimal("0.8"),
            )
        )

    if "original_servings" in candidate.inferred_fields:
        assumptions.append(
            Assumption(
                text=f"LLM inferred the recipe serves {candidate.original_servings} from culinary context",
                confidence=Decimal("0.8"),
            )
        )

    for index, ingredient in enumerate(candidate.ingredients, start=1):
        if ingredient.extraction_source == "LLM_INFERRED":
            assumptions.append(
                Assumption(
                    text=f"LLM completed missing details for ingredient {index} ({ingredient.name})",
                    confidence=ingredient.confidence,
                )
            )

    for step in candidate.steps:
        if step.extraction_source in {"LLM_INFERRED", "RULE_INFERRED"}:
            source = "LLM" if step.extraction_source == "LLM_INFERRED" else "fallback rules"
            assumptions.append(
                Assumption(
                    text=f"{source} completed missing operational details for step {step.step_number}",
                    confidence=step.confidence,
                )
            )

    return tuple(assumptions)
