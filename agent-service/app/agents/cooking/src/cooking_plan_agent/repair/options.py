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
    ApprovedDecision,
    ConfirmationQuestion,
    FeasibilityReport,
    GeneratePlanRequest,
    IngredientFeasibility,
    QuestionAnswer,
    QuestionResponseType,
    RecipeIR,
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


def _fmt_servings(value: Decimal) -> str:
    """份量显示：整数值不带小数点，小数保留（保证与决策正则可解析）。"""
    if value == value.to_integral_value():
        return str(int(value))
    return format(value, "f")


def propose_portion_adjustments(
    shortages: tuple[IngredientFeasibility, ...],
    original_servings: Decimal | int = 2,
) -> tuple[RepairOption, ...]:
    """Propose reducing target servings to match available ingredients.

    Calculates the maximum feasible servings as:
        new_servings = floor(original * min(available/required for each ingredient))

    If all shortages are minor (≤ 50%), proposes a specific serving reduction.
    If any shortage is > 50%, proposes reducing to single-serving.

    Args:
        shortages: Ingredient shortages from FeasibilityReport.
        original_servings: The requested serving count (per-recipe target
            servings). Accepts Decimal so callers pass the user's actual
            serving size instead of a fixed default.

    Returns:
        RepairOption if a reduction is meaningful, empty tuple otherwise.
    """
    if not shortages or original_servings <= 1:
        return ()

    original = Decimal(str(original_servings))

    # Find the limiting ingredient: min(available / required)
    min_ratio = Decimal(1)
    for s in shortages:
        if s.required > 0:
            ratio = s.available / s.required
            min_ratio = min(min_ratio, ratio)

    if min_ratio >= Decimal(1):
        return ()  # No reduction needed (should not happen with shortages)

    new_servings = max(Decimal(1), (original * min_ratio).to_integral_value())

    if new_servings >= original:
        return ()

    description = (
        f"Reduce servings from {_fmt_servings(original)} to {_fmt_servings(new_servings)} "
        f"(available ingredients support ~{min_ratio:.0%} of original portions)"
    )

    return (
        RepairOption(
            option_id=f"repair_servings_{_fmt_servings(new_servings)}_{uuid4().hex[:6]}",
            option_type="reduce_servings",
            description=description,
            changes=(
                f"Scale all ingredient quantities to {_fmt_servings(new_servings)} servings "
                f"(was {_fmt_servings(original)})",
            ),
            effects=(f"All ingredient shortages resolved by scaling down to {_fmt_servings(new_servings)} servings",),
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

            match = re.search(r"from ([\d.]+) to ([\d.]+)", opt.description)
            if match:
                modifications["target_servings"] = Decimal(match.group(2))

        # Other option types (substitute, equipment, dish replacement, purchase)
        # are deferred to rendering layer for now.

    return {
        "applied_count": len(applied),
        "applied_ids": tuple(applied),
        "modifications": modifications,
    }


# =============================================================================
# 5.26  Structured decision loop (P0-06)
# =============================================================================

# The six decision kinds the confirmation loop supports (P0-06 rule 5).
# "purchase" (外出采购) is confirmable end-to-end: selecting it emits a
# structured decision the client echoes back; applying it is a no-op on
# the schedule inputs — the client buys the missing ingredients, updates
# the inventory snapshot in the backend, and resubmits the request.
SUPPORTED_DECISION_TYPES = frozenset(
    {
        "reduce_servings",
        "extend_time",
        "substitute_ingredient",
        "alternative_equipment",
        "replace_dish",
        "purchase",
    }
)


def build_approved_decisions(
    repair_options: tuple[RepairOption, ...],
    plan_revision: str | None,
) -> tuple[ApprovedDecision, ...]:
    """Convert presented RepairOptions into structured, submittable decisions.

    Every supported option becomes an ApprovedDecision whose payload is
    populated from the option's structured description. The client can
    resubmit these verbatim; the server re-validates them (P0-06 rule 2).
    """
    import re as _re

    decisions: list[ApprovedDecision] = []
    for option in repair_options:
        if option.option_type not in SUPPORTED_DECISION_TYPES:
            continue
        payload: dict[str, object] = {}
        if option.option_type == "extend_time":
            match = _re.search(r"to (\d+) minutes", option.description)
            if match:
                payload["time_limit_minutes"] = int(match.group(1))
        elif option.option_type == "reduce_servings":
            match = _re.search(r"from ([\d.]+) to ([\d.]+)", option.description)
            if match:
                # 削减后的新份量始终为整数（to_integral_value），故保留 int 语义
                payload["servings"] = int(Decimal(match.group(2)))
        decisions.append(
            ApprovedDecision(
                option_id=option.option_id,
                option_type=option.option_type,
                payload=payload,
                plan_revision=plan_revision,
            )
        )
    return tuple(decisions)


def validate_approved_decisions(
    decisions: tuple[ApprovedDecision, ...],
    current_plan_revision: str | None,
) -> tuple[str, ...]:
    """Validate a client's resubmitted decisions (P0-06 rule 3).

    Checks:
      - option_type is one of the six supported kinds
      - payload is not conflicting (mutually exclusive decision kinds)
      - option_id is non-empty
      - plan_revision matches the confirmation the client is answering
        (stale confirmation rejected)

    Returns a tuple of issue strings. Empty = all valid.
    """
    issues: list[str] = []
    seen_option_ids: set[str] = set()
    seen_types: set[str] = set()

    for decision in decisions:
        if not decision.option_id.strip():
            issues.append("decision has empty option_id")
        if decision.option_id in seen_option_ids:
            issues.append(f"duplicate option_id: {decision.option_id}")
        seen_option_ids.add(decision.option_id)

        if decision.option_type not in SUPPORTED_DECISION_TYPES:
            issues.append(
                f"unsupported option_type {decision.option_type!r}; supported: {sorted(SUPPORTED_DECISION_TYPES)}"
            )
        else:
            # Conflicting decisions: e.g. reduce_servings + replace_dish both
            # change portions. Reject mutually exclusive combinations.
            if decision.option_type in seen_types:
                issues.append(f"conflicting decisions of type {decision.option_type}")
            seen_types.add(decision.option_type)

        if decision.plan_revision is not None and current_plan_revision is not None:
            if decision.plan_revision != current_plan_revision:
                issues.append(f"stale plan_revision {decision.plan_revision!r}, current is {current_plan_revision!r}")

    return tuple(issues)


def apply_approved_decisions_structured(
    request: GeneratePlanRequest,
    decisions: tuple[ApprovedDecision, ...],
) -> GeneratePlanRequest:
    """Apply approved decisions to produce a resolved request (P0-06 rule 4).

    Pure transformation: never mutates the input request. Returns a new
    GeneratePlanRequest with the applicable constraints updated:
      - reduce_servings   → target_servings of every recipe
      - extend_time       → time_limit_minutes
      - substitute_ingredient → recorded in approved payload for the IR
        builder (ingredient substitution applied downstream as a patch)
      - alternative_equipment → kitchen resource snapshot adjusted
      - replace_dish      → recipe removed from the request
    """
    new_request = request
    new_kitchen: list[object] = list(request.kitchen_resources)

    for decision in decisions:
        payload = decision.payload
        if decision.option_type == "reduce_servings" and payload.get("servings") is not None:
            servings = Decimal(str(payload["servings"]))
            new_recipes = tuple(
                r.model_copy(update={"target_servings": servings}) if r.target_servings != servings else r
                for r in new_request.recipes
            )
            new_request = new_request.model_copy(update={"recipes": new_recipes})

        elif decision.option_type == "extend_time" and payload.get("time_limit_minutes") is not None:
            new_request = new_request.model_copy(update={"time_limit_minutes": int(str(payload["time_limit_minutes"]))})

        elif decision.option_type == "replace_dish" and payload.get("recipe_id"):
            target = str(payload["recipe_id"])
            new_recipes = tuple(r for r in new_request.recipes if r.recipe_id != target)
            if len(new_recipes) == len(new_request.recipes):
                # No-op replace of an unknown dish is tolerated but unused.
                continue
            new_request = new_request.model_copy(update={"recipes": new_recipes})

        elif decision.option_type == "alternative_equipment" and payload.get("resource_type"):
            from cooking_plan_agent.domain.models import KitchenResourceSnapshot

            target_type = str(payload["resource_type"]).lower()
            alternative = str(payload.get("alternative", "")).lower()
            if not alternative:
                continue
            # Replace resources of the target type with an alternative type.
            kept = [
                r
                for r in new_kitchen
                if not isinstance(r, KitchenResourceSnapshot) or r.resource_type.lower() != target_type
            ]
            kept.append(
                KitchenResourceSnapshot(
                    resource_id=f"alt-{alternative}",
                    resource_type=alternative,
                    capacity=Decimal(1),
                )
            )
            new_kitchen = kept

        # substitute_ingredient is handled as a patch by the IR builder
        # (payload: {recipe_id, ingredient, substitute}) — see
        # apply_ingredient_substitutions_patch.

        elif decision.option_type == "purchase":
            # 外出采购：决策本身不改变排程输入（agent 无法代购）。
            # 用户购买后由后端更新库存快照（inventory_lots）并重新提交请求；
            # 若库存未变，重跑仍会返回 NEEDS_CONFIRMATION（可再次选择）。
            pass  # no-op：继续到最终 model_copy（保留 approved_decisions）

    new_request = new_request.model_copy(update={"kitchen_resources": tuple(new_kitchen)})
    return new_request


def apply_ingredient_substitutions_patch(
    recipes: tuple[RecipeIR, ...],
    decisions: tuple[ApprovedDecision, ...],
) -> tuple[RecipeIR, ...]:
    """Patch RecipeIR ingredients per substitute_ingredient decisions.

    Pure transformation: each decision with option_type
    ``substitute_ingredient`` renames the target ingredient's canonical
    name to the substitute so safety (allergen) and feasibility checks
    re-run against the NEW ingredient (P0-06 rule 6).
    """
    substitutes = {
        (d.payload.get("recipe_id"), d.payload.get("ingredient")): d.payload.get("substitute")
        for d in decisions
        if d.option_type == "substitute_ingredient"
        and d.payload.get("recipe_id")
        and d.payload.get("ingredient")
        and d.payload.get("substitute")
    }
    if not substitutes:
        return recipes

    patched: list[RecipeIR] = []
    for recipe in recipes:
        changed = False
        new_ingredients = list(recipe.ingredients)
        for i, ingredient in enumerate(recipe.ingredients):
            key = (recipe.recipe_id, ingredient.canonical_name)
            substitute = substitutes.get(key)
            if substitute is not None:
                new_ingredients[i] = ingredient.model_copy(
                    update={
                        "canonical_name": str(substitute),
                        "raw_name": str(substitute),
                    }
                )
                changed = True
        if changed:
            recipe = recipe.model_copy(update={"ingredients": tuple(new_ingredients)})
        patched.append(recipe)
    return tuple(patched)


# =============================================================================
# P4-02  Structured confirmation answers → ApprovedDecision mapping
# =============================================================================

# Bounded free-text answer length (P4-02 rule 5: bound length/types).
_MAX_TEXT_ANSWER_LENGTH = 500


class ConfirmationAnswersError(ValueError):
    """Raised when a set of confirmation answers is invalid (P4-02).

    Carries the individual issues (unknown question_id, invalid option,
    missing required answer, duplicate answer, over-length text) so the
    caller can produce field-level fix guidance (P2-04 fault matrix).
    """

    def __init__(self, issues: tuple[str, ...]) -> None:
        self.issues = issues
        super().__init__("; ".join(issues))


def answers_to_approved_decisions(
    questions: tuple[ConfirmationQuestion, ...],
    answers: tuple[QuestionAnswer, ...],
    plan_revision: str | None,
    presented_decisions: tuple[ApprovedDecision, ...] = (),
) -> tuple[ApprovedDecision, ...]:
    """Validate client answers and map them losslessly to ApprovedDecision.

    Validation (P4-02 rule 4 / P2-04 fault matrix):
      - every answer's question_id must exist in the presented questions;
      - no duplicate answers for the same question;
      - every required question must be answered;
      - CHOICE answers must hit exactly one of the question's option values;
      - TEXT answers must be non-empty and bounded in length.

    Mapping (D9): only CHOICE answers that select a presented repair
    decision emit an ApprovedDecision — the EXACT object that was
    presented (looked up by option_id), so the payload is preserved
    verbatim with zero rewriting. Gap/assumption answers are validated
    but have no ApprovedDecision carrier yet (contract v2).

    Args:
        questions: The ConfirmationQuestions presented to the client.
        answers: The client's submitted QuestionAnswers.
        plan_revision: The revision of the confirmation being answered.
        presented_decisions: The ApprovedDecisions carried by the
            confirmation response (used to map option values verbatim).

    Returns:
        The decisions to resubmit in the next request's
        ``approved_decisions`` field.

    Raises:
        ConfirmationAnswersError: With field-level fix guidance when any
            answer fails validation.
    """
    issues: list[str] = []
    by_id: dict[str, ConfirmationQuestion] = {q.question_id: q for q in questions}
    answered_ids: set[str] = set()

    for answer in answers:
        question = by_id.get(answer.question_id)
        if question is None:
            issues.append(f"unknown question_id: {answer.question_id}")
            continue
        if answer.question_id in answered_ids:
            issues.append(f"duplicate answer for question_id: {answer.question_id}")
        answered_ids.add(answer.question_id)

        value = answer.value.strip()
        if question.response_type == QuestionResponseType.CHOICE:
            valid_values = {option.value for option in question.options}
            if value not in valid_values:
                issues.append(
                    f"invalid option for question {answer.question_id!r}: {answer.value!r}; "
                    f"allowed: {sorted(valid_values)}"
                )
        else:
            if not value:
                issues.append(f"empty answer for question {answer.question_id!r}")
            elif len(value) > _MAX_TEXT_ANSWER_LENGTH:
                issues.append(
                    f"answer for question {answer.question_id!r} exceeds {_MAX_TEXT_ANSWER_LENGTH} characters"
                )

    # Required questions must all be answered.
    for question in questions:
        if question.required and question.question_id not in answered_ids:
            issues.append(f"missing required answer for question {question.question_id!r}")

    if issues:
        raise ConfirmationAnswersError(tuple(issues))

    # Lossless mapping: an answer selects a presented decision verbatim
    # (by option_id); payload is never rebuilt from prose (D9).
    decisions_by_option_id: dict[str, ApprovedDecision] = {d.option_id: d for d in presented_decisions}
    mapped: list[ApprovedDecision] = []
    for answer in answers:
        decision = decisions_by_option_id.get(answer.value)
        if decision is None:
            continue
        if plan_revision is not None and decision.plan_revision != plan_revision:
            # Keep the decision's payload/type; only rebind the revision the
            # client is answering. This is a metadata update, not a payload
            # rewrite (D9).
            decision = decision.model_copy(update={"plan_revision": plan_revision})
        mapped.append(decision)
    return tuple(mapped)
