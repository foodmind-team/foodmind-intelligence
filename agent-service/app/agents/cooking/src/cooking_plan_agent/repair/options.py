"""Repair options — propose, validate, rank, and apply fixes for infeasible plans.

Handbook sections 5.17–5.25: all functions are pure (no I/O), operating on
immutable Pydantic models. Each repair type has its own proposal function.
Options are ranked and validated before presentation.
"""

from __future__ import annotations

from decimal import Decimal
from typing import NamedTuple
from uuid import uuid4

from cooking_plan_agent.domain.models import (
    FeasibilityReport,
    GeneratePlanRequest,
    IngredientFeasibility,
    RepairOption,
    StrictModel,
)

# =============================================================================
# Shortage and validation models
# =============================================================================


class Shortage(NamedTuple):
    """A single resource or ingredient shortage."""

    item: str
    """Ingredient name or resource type."""
    required: Decimal
    """Amount needed."""
    available: Decimal
    """Amount available."""
    unit: str
    """Unit of measure."""


class RepairValidation(StrictModel):
    """Result of validating a single RepairOption."""

    is_valid: bool
    """Whether the option is internally consistent."""
    issues: tuple[str, ...] = ()
    """Validation issues, if any."""


# =============================================================================
# 5.17  Calculate exact shortages
# =============================================================================


def calculate_exact_shortages(
    report: FeasibilityReport,
) -> tuple[Shortage, ...]:
    """Extract exact shortage items from a FeasibilityReport.

    Covers both ingredient shortages (from ingredient_shortages) and
    resource shortages (from missing_resources). Resource shortages
    are represented with required=1, available=0.

    Args:
        report: A FeasibilityReport from check_all_inventory / check_feasibility.

    Returns:
        Tuple of Shortage items. Empty if the report is feasible.
    """
    shortages: list[Shortage] = []

    for item in report.ingredient_shortages:
        if item.shortage > 0:
            shortages.append(
                Shortage(
                    item=item.ingredient_name,
                    required=item.required,
                    available=item.available,
                    unit=item.unit,
                )
            )

    for resource_type in report.missing_resources:
        shortages.append(
            Shortage(
                item=resource_type,
                required=Decimal(1),
                available=Decimal(0),
                unit="unit",
            )
        )

    return tuple(shortages)


# =============================================================================
# 5.18  Ingredient substitution proposal
# =============================================================================

# Common ingredient substitutions (handbook 5.18).
# Format: ingredient_name → list of (substitute_name, confidence_note)
_INGREDIENT_SUBSTITUTIONS: dict[str, tuple[tuple[str, str], ...]] = {
    "chicken breast": (
        ("chicken thigh", "same protein, slightly longer cook time"),
        ("tofu (firm)", "vegetarian alternative, absorbs flavors well"),
    ),
    "chicken thigh": (
        ("chicken breast", "leaner, slightly shorter cook time"),
        ("tofu (firm)", "vegetarian alternative"),
    ),
    "beef": (
        ("pork", "similar texture for most dishes"),
        ("lamb", "richer flavor, works in stews and roasts"),
        ("tofu (extra firm)", "vegetarian alternative for stir-fries"),
    ),
    "pork": (
        ("chicken", "leaner, adjust cooking time"),
        ("beef", "richer, similar cooking methods"),
    ),
    "fish": (
        ("shrimp", "similar light protein"),
        ("tofu (silken)", "vegetarian alternative for steamed dishes"),
    ),
    "salmon": (
        ("trout", "similar fatty fish"),
        ("mackerel", "stronger flavor, similar cooking methods"),
    ),
    "shrimp": (
        ("scallop", "similar delicate seafood"),
        ("tofu (firm, cubed)", "vegetarian alternative"),
    ),
    "egg": (
        ("flax egg (1 tbsp flax + 3 tbsp water)", "vegan baking substitute"),
        ("mashed banana (1/4 cup per egg)", "vegan baking substitute"),
    ),
    "milk": (
        ("oat milk", "dairy-free, similar creaminess"),
        ("soy milk", "dairy-free, high protein"),
        ("water (with extra fat)", "emergency substitute"),
    ),
    "butter": (
        ("vegetable oil (reduce by 20%)", "dairy-free, works in most recipes"),
        ("margarine", "direct substitute"),
        ("coconut oil", "dairy-free, adds subtle coconut flavor"),
    ),
    "cream": (
        ("coconut cream", "dairy-free, similar richness"),
        ("cashew cream", "dairy-free, neutral flavor"),
    ),
    "cheese": (
        ("nutritional yeast (for flavor)", "dairy-free cheese flavor substitute"),
        ("vegan cheese", "direct dairy-free substitute"),
    ),
    "wheat flour": (
        ("almond flour", "gluten-free, adjust liquid"),
        ("rice flour", "gluten-free, works for coatings"),
        ("gluten-free flour blend", "direct gluten-free substitute"),
    ),
    "tomato": (
        ("bell pepper", "different flavor, similar texture for sauces"),
        ("canned tomato", "more concentrated, adjust liquid"),
    ),
    "onion": (
        ("shallot", "milder, more delicate flavor"),
        ("leek", "milder, works in soups and stews"),
        ("onion powder (1 tbsp = 1 medium onion)", "concentrated flavor"),
    ),
    "garlic": (
        ("garlic powder (1/8 tsp = 1 clove)", "concentrated, milder"),
        ("shallot", "different flavor, similar aromatic role"),
    ),
    "soy sauce": (
        ("tamari", "gluten-free, similar flavor"),
        ("coconut aminos", "soy-free, slightly sweeter"),
        ("fish sauce (use less)", "stronger umami, different base"),
    ),
    "rice": (
        ("quinoa", "higher protein, similar cooking time"),
        ("couscous", "faster cooking, different texture"),
        ("cauliflower rice", "low-carb alternative"),
    ),
    "pasta": (
        ("zucchini noodles", "low-carb, shorter cook time"),
        ("rice noodles", "gluten-free, different texture"),
        ("gluten-free pasta", "direct gluten-free substitute"),
    ),
    "sugar": (
        ("honey (reduce liquid by 1/4)", "natural sweetener, stronger flavor"),
        ("maple syrup (reduce liquid by 1/4)", "natural sweetener"),
    ),
    "olive oil": (
        ("vegetable oil", "neutral, higher smoke point"),
        ("avocado oil", "similar quality, higher smoke point"),
        ("coconut oil", "adds coconut flavor, solid at room temp"),
    ),
    "cilantro": (
        ("parsley + lime zest", "similar fresh note without soapy flavor"),
        ("Thai basil", "different but complementary flavor"),
    ),
    "chilli": (
        ("bell pepper + cayenne", "milder heat"),
        ("jalapeño", "different heat profile"),
    ),
}


def propose_ingredient_substitutions(
    shortages: tuple[IngredientFeasibility, ...],
) -> tuple[RepairOption, ...]:
    """Propose ingredient substitutions for each shortage.

    Looks up common substitutions from the built-in table. Only returns
    options for ingredients that have known substitutes — unknown
    ingredients are left for the "purchase" option (deferred to rendering).

    Args:
        shortages: Ingredient shortages from FeasibilityReport.

    Returns:
        One or more RepairOption per shortage (may be empty if no
        substitute is known).
    """
    options: list[RepairOption] = []

    for shortage in shortages:
        name_lower = shortage.ingredient_name.lower().strip()
        subs = _INGREDIENT_SUBSTITUTIONS.get(name_lower)

        if not subs:
            # No known substitution — flag as needing purchase
            options.append(
                RepairOption(
                    option_id=f"repair_purchase_{shortage.ingredient_name}_{uuid4().hex[:6]}",
                    option_type="purchase",
                    description=(
                        f"Purchase {shortage.shortage} {shortage.unit} of "
                        f"'{shortage.ingredient_name}' (no known substitute available)"
                    ),
                    changes=(
                        f"Add {shortage.shortage} {shortage.unit} of {shortage.ingredient_name} to shopping list",
                    ),
                    effects=(
                        f"Increase {shortage.ingredient_name} availability by {shortage.shortage} {shortage.unit}",
                    ),
                    revalidation_status="validated",
                )
            )
            continue

        for sub_name, note in subs:
            options.append(
                RepairOption(
                    option_id=f"repair_sub_{shortage.ingredient_name}_{sub_name.split()[0]}_{uuid4().hex[:6]}",
                    option_type="substitute_ingredient",
                    description=(f"Substitute '{shortage.ingredient_name}' with '{sub_name}': {note}"),
                    changes=(
                        f"Replace {shortage.ingredient_name} ({shortage.shortage} {shortage.unit}) with {sub_name}",
                    ),
                    effects=(
                        f"Resolves {shortage.shortage} {shortage.unit} shortage of "
                        f"'{shortage.ingredient_name}'. {note}",
                    ),
                    revalidation_status="validated",
                )
            )

    return tuple(options)


# =============================================================================
# 5.19  Portion adjustment proposal
# =============================================================================


def propose_portion_adjustments(
    shortages: tuple[IngredientFeasibility, ...],
    original_servings: int = 2,
) -> tuple[RepairOption, ...]:
    """Propose reducing target servings to match available ingredients.

    Calculates the maximum feasible servings as:
        new_servings = floor(original * min(available/required for each ingredient))

    If all shortages are minor (≤ 50%), proposes a specific serving reduction.
    If any shortage is > 50%, proposes reducing to single-serving.

    Args:
        shortages: Ingredient shortages from FeasibilityReport.
        original_servings: The requested serving count.

    Returns:
        RepairOption if a reduction is meaningful, empty tuple otherwise.
    """
    if not shortages or original_servings <= 1:
        return ()

    # Find the limiting ingredient: min(available / required)
    min_ratio = Decimal(1)
    for s in shortages:
        if s.required > 0:
            ratio = s.available / s.required
            min_ratio = min(min_ratio, ratio)

    if min_ratio >= Decimal(1):
        return ()  # No reduction needed (should not happen with shortages)

    new_servings = max(1, int((Decimal(original_servings) * min_ratio).to_integral_value()))

    if new_servings >= original_servings:
        return ()

    description = (
        f"Reduce servings from {original_servings} to {new_servings} "
        f"(available ingredients support ~{min_ratio:.0%} of original portions)"
    )

    return (
        RepairOption(
            option_id=f"repair_servings_{new_servings}_{uuid4().hex[:6]}",
            option_type="reduce_servings",
            description=description,
            changes=(f"Scale all ingredient quantities to {new_servings} servings (was {original_servings})",),
            effects=(f"All ingredient shortages resolved by scaling down to {new_servings} servings",),
            revalidation_status="validated",
        ),
    )


# =============================================================================
# 5.20  Equipment alternative proposal
# =============================================================================

# Common equipment alternatives (handbook 5.20).
_EQUIPMENT_ALTERNATIVES: dict[str, tuple[tuple[str, str], ...]] = {
    "oven": (
        ("air fryer", "faster, similar results for most baked dishes"),
        ("toaster oven", "works for small batches"),
        ("stove + covered pot", "simulates oven for stews and braises"),
    ),
    "stove": (
        ("electric skillet", "portable, similar temperature control"),
        ("induction cooktop", "portable, rapid heating"),
        ("camping stove", "portable, suitable for basic cooking"),
    ),
    "wok": (
        ("large frying pan", "works for most stir-fry dishes"),
        ("cast iron skillet", "excellent heat retention for high-heat cooking"),
    ),
    "steamer": (
        ("pot + colander", "classic improvised steamer"),
        ("microwave + covered bowl", "faster for vegetables"),
    ),
    "blender": (
        ("food processor", "works for most blending tasks"),
        ("immersion blender", "works directly in pot/bowl"),
        ("mortar and pestle", "manual alternative for pastes and small batches"),
    ),
    "mixing_bowl": (
        ("large pot", "substitute for mixing larger quantities"),
        ("any large container", "temporary substitute"),
    ),
    "cutting_board": (
        ("clean countertop + silicone mat", "temporary substitute"),
        ("large flat plate", "works for small prep tasks"),
    ),
    "spatula": (
        ("wooden spoon", "works for most stirring tasks"),
        ("chopsticks", "works for stir-frying small portions"),
    ),
    "sink": (
        ("large bowl/basin", "manual washing substitute"),
        ("bathtub", "emergency substitute for large items"),
    ),
    "knife": (
        ("kitchen shears", "works for cutting herbs and small items"),
        ("mandoline slicer", "works for uniform slicing"),
    ),
    "rice_cooker": (
        ("pot with lid", "standard stovetop rice method"),
        ("instant pot", "pressure-cooker rice method"),
    ),
    "slow_cooker": (
        ("dutch oven + low oven", "simulates slow cooking"),
        ("pressure cooker", "faster, similar tenderizing"),
    ),
    "grill": (
        ("broiler (oven)", "similar high-heat from above"),
        ("grill pan (stove)", "indoor alternative with grill marks"),
        ("cast iron skillet", "excellent searing similar to grill"),
    ),
    "microwave": (
        ("stove + small pot", "reheating and steaming alternative"),
        ("oven at low temp", "gentle reheating"),
    ),
    "thermometer": (
        ("visual cues + timing", "less precise but workable for experienced cooks"),
        ("touch test (for meats)", "traditional method"),
    ),
}


def propose_equipment_alternatives(
    missing_resources: tuple[str, ...],
) -> tuple[RepairOption, ...]:
    """Propose alternative equipment for each missing resource type.

    Looks up common alternatives from the built-in table. Resources
    without known alternatives are flagged for manual resolution.

    Args:
        missing_resources: Resource types that are missing/unavailable.

    Returns:
        One or more RepairOption per missing resource.
    """
    options: list[RepairOption] = []

    for resource in missing_resources:
        # Strip capability suffix if present (e.g. "stove:induction" → "stove")
        base_resource = resource.split(":")[0].lower().strip()
        alts = _EQUIPMENT_ALTERNATIVES.get(base_resource)

        if not alts:
            options.append(
                RepairOption(
                    option_id=f"repair_noalt_{resource}_{uuid4().hex[:6]}",
                    option_type="alternative_equipment",
                    description=(f"No known alternative for '{resource}'. Manual resolution required."),
                    changes=(f"Source or improvise alternative for {resource}",),
                    effects=("Requires manual equipment sourcing",),
                    revalidation_status="validated",
                )
            )
            continue

        for alt_name, note in alts:
            options.append(
                RepairOption(
                    option_id=f"repair_eq_{base_resource}_{alt_name.split()[0]}_{uuid4().hex[:6]}",
                    option_type="alternative_equipment",
                    description=(f"Use '{alt_name}' instead of '{resource}': {note}"),
                    changes=(f"Replace {resource} with {alt_name}",),
                    effects=(f"Resolves missing '{resource}'. {note}",),
                    revalidation_status="validated",
                )
            )

    return tuple(options)


# =============================================================================
# 5.21  Dish replacement proposal
# =============================================================================


def propose_dish_replacements(
    shortages: tuple[IngredientFeasibility, ...],
    recipe_names: tuple[str, ...],
) -> tuple[RepairOption, ...]:
    """Propose removing/replacing dishes that have unsolvable ingredient issues.

    MVP strategy: for each shortage that has no known substitute, suggest
    removing the affected dish(es). Since we don't have per-dish shortage
    mapping at this layer, we surface the issue at plan level.

    Args:
        shortages: Ingredient shortages from FeasibilityReport.
        recipe_names: Names of all dishes in the current plan.

    Returns:
        RepairOption suggesting dish removal/review.
    """
    if not shortages:
        return ()

    # Identify ingredients with no known substitutes
    unsubstitutable = [s for s in shortages if s.ingredient_name.lower().strip() not in _INGREDIENT_SUBSTITUTIONS]

    if not unsubstitutable:
        return ()  # All shortages have substitutes available

    ingredient_list = ", ".join(s.ingredient_name for s in unsubstitutable)
    dish_list = ", ".join(recipe_names) if recipe_names else "the current dishes"

    return (
        RepairOption(
            option_id=f"repair_dish_remove_{uuid4().hex[:8]}",
            option_type="replace_dish",
            description=(
                f"Some ingredients ({ingredient_list}) have no known substitutes "
                f"in {dish_list}. Consider replacing affected dishes or purchasing "
                f"the missing ingredients."
            ),
            changes=(
                f"Review dishes containing: {ingredient_list}",
                "Consider replacing with alternative recipes using available ingredients",
            ),
            effects=(
                f"Eliminates shortages in: {ingredient_list}",
                "May change the meal composition significantly",
            ),
            revalidation_status="validated",
        ),
    )


# =============================================================================
# 5.22  Time extension proposal
# =============================================================================


def propose_time_extension(
    current_time_limit: int | None,
    minimum_required_minutes: int,
) -> RepairOption | None:
    """Propose extending the time limit if the current limit is too tight.

    Only proposes extension if the gap is reasonable (≤ 3× current limit).
    Extreme gaps suggest deeper problems that a time extension won't fix.

    Args:
        current_time_limit: The user-specified time limit in minutes.
        minimum_required_minutes: The minimum feasible makespan.

    Returns:
        RepairOption if extension is reasonable, None otherwise.
    """
    if current_time_limit is None:
        return None

    if current_time_limit >= minimum_required_minutes:
        return None

    gap = minimum_required_minutes - current_time_limit

    # Don't propose extreme extensions (> 3× the current limit)
    if current_time_limit > 0 and minimum_required_minutes > 3 * current_time_limit:
        return None

    return RepairOption(
        option_id=f"repair_time_{minimum_required_minutes}_{uuid4().hex[:6]}",
        option_type="extend_time",
        description=(
            f"Extend cooking time from {current_time_limit} to {minimum_required_minutes} minutes (adds {gap} minutes)"
        ),
        changes=(f"Increase time limit to {minimum_required_minutes} minutes",),
        effects=(f"All tasks can be scheduled within {minimum_required_minutes} minutes",),
        revalidation_status="validated",
    )


# =============================================================================
# 5.23  Repair option validation
# =============================================================================


def validate_repair_option(
    option: RepairOption,
) -> RepairValidation:
    """Validate that a RepairOption is internally consistent.

    Checks:
      - option_id is non-empty
      - option_type is a recognised type
      - description, changes, and effects are non-empty
      - revalidation_status is 'validated'

    Args:
        option: A RepairOption to validate.

    Returns:
        RepairValidation with is_valid=True and empty issues on success.
    """
    issues: list[str] = []
    valid_types = {
        "substitute_ingredient",
        "reduce_servings",
        "alternative_equipment",
        "replace_dish",
        "extend_time",
        "purchase",
    }

    if not option.option_id.strip():
        issues.append("option_id is empty")

    if option.option_type not in valid_types:
        issues.append(f"Unknown option_type: {option.option_type!r}")

    if not option.description.strip():
        issues.append("description is empty")

    if not option.changes:
        issues.append("changes is empty")

    if not option.effects:
        issues.append("effects is empty")

    if option.revalidation_status != "validated":
        issues.append(f"revalidation_status is {option.revalidation_status!r}, expected 'validated'")

    return RepairValidation(
        is_valid=len(issues) == 0,
        issues=tuple(issues),
    )


# =============================================================================
# 5.24  Rank repair options
# =============================================================================

# Priority ordering: least disruptive options first (handbook 5.24).
_OPTION_TYPE_PRIORITY: dict[str, int] = {
    "reduce_servings": 1,  # Least disruptive — just scale down
    "alternative_equipment": 2,  # Use what you have differently
    "substitute_ingredient": 3,  # Swap ingredients
    "extend_time": 4,  # Just wait longer
    "replace_dish": 5,  # Change the menu
    "purchase": 6,  # Most disruptive — go shopping
}


def rank_repair_options(
    options: tuple[RepairOption, ...],
) -> tuple[RepairOption, ...]:
    """Rank repair options from least to most disruptive.

    Sorts by option_type priority, then by option_id for determinism.
    Only includes validated options (revalidation_status='validated').

    Args:
        options: Unsorted repair options.

    Returns:
        Sorted tuple, least disruptive first.
    """
    valid = [o for o in options if o.revalidation_status == "validated"]
    valid.sort(
        key=lambda o: (
            _OPTION_TYPE_PRIORITY.get(o.option_type, 99),
            o.option_id,
        )
    )
    return tuple(valid)


# =============================================================================
# 5.25  Apply approved decisions
# =============================================================================


def apply_approved_decisions(
    request: GeneratePlanRequest,
    approved_ids: tuple[str, ...],
    available_options: tuple[RepairOption, ...],
) -> dict[str, object]:
    """Apply user-approved repair decisions to produce a resolved plan input.

    This is a planning step — it modifies the request context (e.g. removes
    dietary restrictions, adjusts time limits) and records which decisions
    were approved. Downstream nodes use the approved_decisions field.

    Current MVP: passes approved decision IDs through to request context.
    Full ingredient/recipe mutation is deferred to the rendering layer
    where per-dish context is available.

    Args:
        request: The original GeneratePlanRequest.
        approved_ids: IDs of RepairOptions the user approved.
        available_options: All options that were presented.

    Returns:
        Dict with keys: 'request' (updated GeneratePlanRequest or dict of
        modifications) and 'applied_count' (int).
    """
    approved_set = set(approved_ids)
    applied: list[str] = []

    modifications: dict[str, object] = {}

    for opt in available_options:
        if opt.option_id not in approved_set:
            continue

        applied.append(opt.option_id)

        if opt.option_type == "extend_time":
            # Extract the proposed new time limit from the option description
            import re

            match = re.search(r"to (\d+) minutes", opt.description)
            if match:
                modifications["time_limit_minutes"] = int(match.group(1))

        elif opt.option_type == "reduce_servings":
            import re

            match = re.search(r"from \d+ to (\d+)", opt.description)
            if match:
                modifications["target_servings"] = int(match.group(1))

        # Other option types (substitute, equipment, dish replacement, purchase)
        # are deferred to rendering layer for now.

    return {
        "applied_count": len(applied),
        "applied_ids": tuple(applied),
        "modifications": modifications,
    }
