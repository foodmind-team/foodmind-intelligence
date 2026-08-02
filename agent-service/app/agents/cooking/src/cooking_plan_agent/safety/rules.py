"""Safety rules — individual, independently evaluable food safety constraints.

Each rule implements the SafetyRule protocol: evaluate(SafetyContext) →
SafetyFinding | None. Rules are designed to be composed by SafetyEngine
with no cross-rule dependencies. Every rule is unit-testable in isolation.

Handbook 5.7: safety rules are the first hard gate after parsing.
Handbook 5.8: rules produce three severity levels:
  - hard_unrepairable → block the plan entirely (INFEASIBLE)
  - hard_repairable   → inject safety_tasks (e.g., sanitise board)
  - warning           → surface to user but do not block
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Protocol

from cooking_plan_agent.domain.enums import HeatLevel
from cooking_plan_agent.domain.models import (
    InventoryLotSnapshot,
    RecipeIR,
    RecipeStep,
    SafetyContext,
    SafetyFinding,
    SafetyInsertion,
)
from cooking_plan_agent.safety.policies.usda import USDA_SAFE_MINIMUM_TEMPERATURES_C
from cooking_plan_agent.safety.policy import SafetyPolicy

# =============================================================================
# P0-07 safety-task policy — durations and resources for inserted safety tasks
# =============================================================================

# Sanitisation task default duration.  Replaces the old hard-coded 1-minute
# placeholder: the inserted task must occupy a realistic window so the
# verifier can check raw→sanitise→RTE ordering meaningfully.
_SANITISE_DURATION_MINUTES = 3
# Resources a sanitisation task requires (Policy 6.1: sink).
_SANITISE_REQUIRED_RESOURCES = ("sink",)

# =============================================================================
# SafetyRule protocol — contract for all rule implementations
# =============================================================================


class SafetyRule(Protocol):
    """A single independently evaluable safety constraint.

    Each rule receives the full SafetyContext and returns either a
    SafetyFinding (violation detected) or None (rule satisfied).
    Rules MUST NOT mutate the context or have side effects.

    rule_id is declared read-only because every concrete rule is a
    frozen dataclass — instance attributes are never settable.
    """

    @property
    def rule_id(self) -> str: ...

    def evaluate(self, context: SafetyContext) -> SafetyFinding | None: ...


# =============================================================================
# Safe minimum internal temperatures (P3-04)
# =============================================================================

# Protein internal-temperature thresholds are no longer hard-coded here — they
# live in versioned, source-backed regional policy packs (safety/policies/).
# The USDA table is imported only as the backward-compatible default so rules
# constructed without an explicit policy keep their historical behaviour.
# Production wiring binds rules to a resolved policy via build_rules(policy).

# Protein keywords for matching ingredient names to protein categories
_PROTEIN_KEYWORDS: dict[str, str] = {
    # Poultry
    "chicken": "chicken",
    "turkey": "turkey",
    "duck": "duck",
    "goose": "goose",
    # Red meat
    "beef": "beef",
    "pork": "pork",
    "lamb": "lamb",
    "veal": "veal",
    # Seafood
    "fish": "fish",
    "salmon": "salmon",
    "tuna": "fish",
    "shrimp": "shrimp",
    "prawn": "shellfish",
    "crab": "shellfish",
    "lobster": "shellfish",
    "mussel": "shellfish",
    "clam": "shellfish",
    "oyster": "shellfish",
    "squid": "shellfish",
    "octopus": "shellfish",
    # Other
    "egg": "egg",
}


# =============================================================================
# Dietary restriction keyword matching
# =============================================================================

# Ingredients prohibited per dietary restriction
_DIETARY_PROHIBITED: dict[str, tuple[str, ...]] = {
    "halal": (
        "pork",
        "bacon",
        "ham",
        "lard",
        "sausage",
        "alcohol",
        "wine",
        "beer",
        "sake",
        "mirin",
        "rum",
        "gelatin",  # unless halal-certified
    ),
    "vegetarian": (
        "chicken",
        "beef",
        "pork",
        "lamb",
        "mutton",
        "veal",
        "fish",
        "salmon",
        "tuna",
        "shrimp",
        "prawn",
        "crab",
        "lobster",
        "mussel",
        "clam",
        "oyster",
        "squid",
        "octopus",
        "bacon",
        "ham",
        "sausage",
        "meat",
    ),
    "vegan": (
        "chicken",
        "beef",
        "pork",
        "lamb",
        "mutton",
        "veal",
        "fish",
        "salmon",
        "tuna",
        "shrimp",
        "prawn",
        "crab",
        "lobster",
        "mussel",
        "clam",
        "oyster",
        "squid",
        "octopus",
        "egg",
        "milk",
        "cheese",
        "butter",
        "cream",
        "yogurt",
        "honey",
        "gelatin",
        "bacon",
        "ham",
        "sausage",
        "meat",
    ),
    "kosher": (
        "pork",
        "bacon",
        "ham",
        "lard",
        "shellfish",
        "shrimp",
        "prawn",
        "crab",
        "lobster",
        "mussel",
        "clam",
        "oyster",
        "squid",
        "octopus",
        # Meat + dairy mixing is complex; flag for now
    ),
}


# =============================================================================
# Rule 1: CrossContaminationRule
# =============================================================================


@dataclass(frozen=True)
class CrossContaminationRule:
    """Detect raw protein handling near ready-to-eat ingredients.

    When a recipe uses raw proteins (meat, poultry, seafood, eggs) AND
    contains steps handling ready-to-eat items, a sanitisation task must
    be injected between them. This rule flags the violation; the
    merge_preparation node injects the sanitisation task.

    Severity: hard_repairable — can always insert a sanitise step.
    """

    rule_id: str = "SAFETY_CROSS_CONTAMINATION"

    # Ingredients considered "raw protein" — matched against canonical_name
    _raw_protein_keywords: tuple[str, ...] = (
        "chicken",
        "beef",
        "pork",
        "lamb",
        "mutton",
        "veal",
        "fish",
        "salmon",
        "tuna",
        "shrimp",
        "prawn",
        "crab",
        "lobster",
        "mussel",
        "clam",
        "oyster",
        "squid",
        "octopus",
        "egg",
        "meat",
        "poultry",
        "seafood",
    )

    # Step categories that imply ready-to-eat handling
    _rte_categories: tuple[str, ...] = (
        "plating",
        "garnishing",
        "dressing",
        "mixing",
    )

    def evaluate(self, context: SafetyContext) -> SafetyFinding | None:
        """Check each recipe for raw-protein / RTE co-existence.

        When both exist in the SAME recipe, locate the anchor steps:
          - after_step_number: last step that handles raw protein
          - before_step_number: first step that is RTE/plating
        The finding carries a structured SafetyInsertion so merge_preparation
        can build the raw → sanitise → RTE dependency chain (P0-07).
        """
        for recipe in context.recipes:
            raw_steps = _raw_protein_steps(recipe, self._raw_protein_keywords)
            rte_steps = _rte_steps(recipe, self._rte_categories)

            if not raw_steps or not rte_steps:
                continue

            # Anchors: last raw step → sanitise → first RTE step.
            after_step = raw_steps[-1].step_number
            before_step = rte_steps[0].step_number
            if after_step >= before_step:
                # Raw handling already precedes RTE within the same recipe
                # with no interleaving — still insert between them.
                pass

            raw_ingredients = [
                ing.raw_name
                for ing in recipe.ingredients
                if _matches_keywords(ing.canonical_name.lower(), self._raw_protein_keywords)
            ]

            insertion = SafetyInsertion(
                insertion_id=f"{self.rule_id.lower()}_{recipe.recipe_id}",
                rule_id=self.rule_id,
                recipe_id=recipe.recipe_id,
                after_step_number=after_step,
                before_step_number=before_step,
                task_instruction=(
                    "Sanitise cutting board and utensils after raw protein handling and before ready-to-eat assembly."
                ),
                duration_minutes=_SANITISE_DURATION_MINUTES,
                required_resources=_SANITISE_REQUIRED_RESOURCES,
            )

            return SafetyFinding(
                rule_id=self.rule_id,
                severity="hard_repairable",
                description=(
                    f"Cross-contamination risk: raw protein and ready-to-eat "
                    f"handling coexist in dish '{recipe.dish_name}' (steps "
                    f"{after_step} → {before_step}). A sanitisation task must "
                    f"be inserted between raw and RTE steps."
                ),
                affected_ingredient_names=tuple(raw_ingredients),
                recommended_action=(
                    "Insert a 'Sanitise cutting board and utensils' task between "
                    "raw protein handling and ready-to-eat assembly."
                ),
                insertion=insertion,
            )

        return None


# =============================================================================
# Rule 2: AllergenDetectionRule
# =============================================================================


@dataclass(frozen=True)
class AllergenDetectionRule:
    """Match recipe ingredients against the user's declared allergens.

    Checks both IngredientDemand.allergen_tags (explicit tags from extraction)
    and ingredient name keyword matching for common allergens.

    Severity: hard_unrepairable for the "big 9" allergens if present,
              hard_repairable for other sensitivities (can substitute).
    """

    rule_id: str = "SAFETY_ALLERGEN_DETECTION"

    # Big 9 priority allergens (FAO/WHO) — hard_unrepairable
    _priority_allergens: tuple[str, ...] = (
        "peanut",
        "tree nut",
        "milk",
        "egg",
        "fish",
        "shellfish",
        "soy",
        "wheat",
        "sesame",
    )

    # Keyword mapping for ingredient name → allergen type
    _allergen_keywords: dict[str, str] = field(
        default_factory=lambda: {
            "peanut": "peanut",
            "almond": "tree nut",
            "walnut": "tree nut",
            "cashew": "tree nut",
            "pecan": "tree nut",
            "pistachio": "tree nut",
            "hazelnut": "tree nut",
            "milk": "milk",
            "cream": "milk",
            "butter": "milk",
            "cheese": "milk",
            "yogurt": "milk",
            "whey": "milk",
            "egg": "egg",
            "fish": "fish",
            "salmon": "fish",
            "tuna": "fish",
            "shrimp": "shellfish",
            "prawn": "shellfish",
            "crab": "shellfish",
            "lobster": "shellfish",
            "mussel": "shellfish",
            "clam": "shellfish",
            "oyster": "shellfish",
            "squid": "shellfish",
            "soy": "soy",
            "soybean": "soy",
            "tofu": "soy",
            "wheat": "wheat",
            "flour": "wheat",
            "bread": "wheat",
            "pasta": "wheat",
            "noodle": "wheat",
            "sesame": "sesame",
            "tahini": "sesame",
            "gluten": "wheat",
        }
    )

    def evaluate(self, context: SafetyContext) -> SafetyFinding | None:
        """Check all ingredients against user allergens."""
        if not context.user_allergens:
            return None

        user_allergens_lower = {a.lower() for a in context.user_allergens}
        matches_priority: list[str] = []
        matches_other: list[str] = []
        affected_ingredients: list[str] = []

        for recipe in context.recipes:
            for ingredient in recipe.ingredients:
                # Check explicit allergen tags from extraction
                for tag in ingredient.allergen_tags:
                    tag_lower = tag.lower()
                    for user_allergen in user_allergens_lower:
                        if user_allergen in tag_lower or tag_lower in user_allergen:
                            affected_ingredients.append(ingredient.raw_name)
                            if tag_lower in self._priority_allergens:
                                matches_priority.append(f"{ingredient.raw_name}({tag})")
                            else:
                                matches_other.append(f"{ingredient.raw_name}({tag})")

                # Check ingredient name against allergen keyword map
                name_lower = ingredient.canonical_name.lower()
                for kw, allergen_type in self._allergen_keywords.items():
                    if kw in name_lower and allergen_type in user_allergens_lower:
                        if ingredient.raw_name not in affected_ingredients:
                            affected_ingredients.append(ingredient.raw_name)
                            if allergen_type in self._priority_allergens:
                                matches_priority.append(f"{ingredient.raw_name}({allergen_type})")
                            else:
                                matches_other.append(f"{ingredient.raw_name}({allergen_type})")

        if not affected_ingredients:
            return None

        if matches_priority:
            return SafetyFinding(
                rule_id=self.rule_id,
                severity="hard_unrepairable",
                description=(
                    f"Priority allergen detected: {', '.join(matches_priority)}. "
                    f"The user has declared allergies to these ingredients. "
                    f"The dish cannot be safely prepared without complete substitution."
                ),
                affected_ingredient_names=tuple(affected_ingredients),
                recommended_action=(
                    "Remove or substitute all flagged ingredients. "
                    "Cross-contamination risk cannot be eliminated for priority allergens."
                ),
            )

        return SafetyFinding(
            rule_id=self.rule_id,
            severity="hard_repairable",
            description=(
                f"Allergen detected (non-priority): {', '.join(matches_other)}. User is sensitive to these ingredients."
            ),
            affected_ingredient_names=tuple(affected_ingredients),
            recommended_action="Substitute flagged ingredients with safe alternatives.",
        )


# =============================================================================
# Rule 3: ProteinSafetyTemperatureRule
# =============================================================================


@dataclass(frozen=True)
class ProteinSafetyTemperatureRule:
    """Verify that protein cooking steps reach the region's safe internal temperatures.

    For each recipe step that involves heating a protein, check that
    target_temperature_c is at or above the safe minimum defined by the
    active regional policy (P3-04). Steps without a specified temperature are
    flagged with the recommended safe temp. Protein categories the policy does
    not document are skipped (never flagged).

    Severity: hard_repairable — temperature can always be specified.
    """

    rule_id: str = "SAFETY_PROTEIN_TEMPERATURE"

    # Per-protein safe minimum internal temperatures (°C). Backward-compatible
    # default is the USDA pack; production binds the resolved regional policy.
    safe_temperatures_c: dict[str, Decimal] = field(default_factory=lambda: dict(USDA_SAFE_MINIMUM_TEMPERATURES_C))

    def evaluate(self, context: SafetyContext) -> SafetyFinding | None:
        """Check all protein heating steps across recipes."""
        unsafe_steps: list[str] = []

        for recipe in context.recipes:
            for step in recipe.steps:
                if not _is_protein_heating_step(step):
                    continue

                # Determine protein type from the recipe ingredients
                protein_type = _dominant_protein_type(recipe)
                safe_temp = self.safe_temperatures_c.get(protein_type)

                if safe_temp is None:
                    continue  # Not a tracked protein — skip

                if step.target_temperature_c is None:
                    unsafe_steps.append(
                        f"'{recipe.dish_name}' step {step.step_number}: "
                        f"no target temperature specified — "
                        f"recommend ≥{safe_temp}°C for {protein_type}"
                    )
                elif step.target_temperature_c < safe_temp:
                    unsafe_steps.append(
                        f"'{recipe.dish_name}' step {step.step_number}: "
                        f"target {step.target_temperature_c}°C is below "
                        f"safe minimum {safe_temp}°C for {protein_type}"
                    )

        if not unsafe_steps:
            return None

        detail = "; ".join(unsafe_steps)
        return SafetyFinding(
            rule_id=self.rule_id,
            severity="hard_repairable",
            description=(f"Protein cooking temperature below regional safe minimum: {detail}"),
            recommended_action=(
                "Set target temperatures to at or above the safe minimum "
                "internal temperatures defined by the active regional safety "
                "policy (see the plan's policy sources)."
            ),
        )


# =============================================================================
# Rule 4: DietaryCompatibilityRule
# =============================================================================


@dataclass(frozen=True)
class DietaryCompatibilityRule:
    """Check recipe ingredients against user dietary restrictions.

    Matches ingredient names against prohibited keyword lists for common
    dietary patterns (halal, vegetarian, vegan, kosher).

    Severity: hard_unrepairable — dietary restrictions are non-negotiable.
    """

    rule_id: str = "SAFETY_DIETARY_COMPATIBILITY"

    def evaluate(self, context: SafetyContext) -> SafetyFinding | None:
        """Check all ingredients against dietary restrictions."""
        if not context.dietary_restrictions:
            return None

        violations: list[str] = []
        affected_ingredients: list[str] = []

        for restriction in context.dietary_restrictions:
            restricted_lower = restriction.lower()
            prohibited = _DIETARY_PROHIBITED.get(restricted_lower)
            if not prohibited:
                continue

            for recipe in context.recipes:
                for ingredient in recipe.ingredients:
                    name_lower = ingredient.canonical_name.lower()
                    for prohibited_kw in prohibited:
                        if prohibited_kw in name_lower:
                            violations.append(
                                f"'{ingredient.raw_name}' in '{recipe.dish_name}' (violates {restriction})"
                            )
                            if ingredient.raw_name not in affected_ingredients:
                                affected_ingredients.append(ingredient.raw_name)

        if not violations:
            return None

        restriction_list = ", ".join(context.dietary_restrictions)
        violation_detail = "; ".join(violations)
        return SafetyFinding(
            rule_id=self.rule_id,
            severity="hard_unrepairable",
            description=(f"Dietary restriction violation ({restriction_list}): {violation_detail}"),
            affected_ingredient_names=tuple(affected_ingredients),
            recommended_action=(
                "Remove or substitute flagged ingredients to comply with "
                "dietary restrictions. Consider alternative recipes if core "
                "ingredients cannot be substituted."
            ),
        )


# =============================================================================
# Rule 5: ExpiredIngredientRule — check inventory lots for spoilage
# =============================================================================


@dataclass(frozen=True)
class ExpiredIngredientRule:
    """Detect ingredients that have passed their expiry date.

    When a cooking plan uses inventory lots that will expire before the
    cooking date, the ingredient is flagged.  Lots ≤ 3 days past expiry
    are hard_repairable (can inspect); lots > 3 days past expiry are
    hard_unrepairable (likely spoiled — must discard).

    Only evaluates when both context.cooking_date and context.inventory_lots
    are provided.  Non-perishable items (rice, flour, etc.) past their
    best-before date are NOT flagged.
    """

    rule_id: str = "SAFETY_EXPIRED_INGREDIENT"

    _perishable_keywords: tuple[str, ...] = (
        "chicken",
        "beef",
        "pork",
        "lamb",
        "mutton",
        "veal",
        "fish",
        "salmon",
        "tuna",
        "shrimp",
        "prawn",
        "crab",
        "lobster",
        "mussel",
        "clam",
        "oyster",
        "squid",
        "octopus",
        "egg",
        "milk",
        "cream",
        "yogurt",
        "cheese",
        "butter",
        "meat",
        "poultry",
        "seafood",
        "dairy",
    )

    def evaluate(self, context: SafetyContext) -> SafetyFinding | None:
        if context.cooking_date is None or not context.inventory_lots:
            return None

        used_lots = self._lots_used_in_recipes(context)
        if not used_lots:
            return None

        expired_repairable: list[str] = []
        expired_unrepairable: list[str] = []
        affected_names: list[str] = []

        for lot in used_lots:
            if lot.expiry_date is None:
                continue
            if context.cooking_date <= lot.expiry_date:
                continue
            if not self._is_perishable(lot.canonical_name):
                continue

            days_past = (context.cooking_date - lot.expiry_date).days
            label = f"'{lot.canonical_name}' (lot {lot.lot_id}, expired {lot.expiry_date}, {days_past}d past)"
            if lot.canonical_name not in affected_names:
                affected_names.append(lot.canonical_name)
            if days_past <= 3:
                expired_repairable.append(label)
            else:
                expired_unrepairable.append(label)

        if not expired_repairable and not expired_unrepairable:
            return None

        all_expired = expired_unrepairable + expired_repairable
        if expired_unrepairable:
            return SafetyFinding(
                rule_id=self.rule_id,
                severity="hard_unrepairable",
                description=(f"Expired perishable ingredients (likely spoiled): {'; '.join(all_expired)}"),
                affected_ingredient_names=tuple(affected_names),
                recommended_action=(
                    "Discard expired perishable items. Purchase fresh "
                    "replacements. Do not consume items > 3 days past "
                    "expiry without professional food-safety assessment."
                ),
            )

        return SafetyFinding(
            rule_id=self.rule_id,
            severity="hard_repairable",
            description=(f"Recently expired ingredients (inspect before use): {'; '.join(expired_repairable)}"),
            affected_ingredient_names=tuple(affected_names),
            recommended_action=(
                "Inspect each flagged item for spoilage signs (odour, "
                "texture, colour). If in doubt, discard and replace."
            ),
        )

    def _lots_used_in_recipes(self, context: SafetyContext) -> list[InventoryLotSnapshot]:
        used_names: set[str] = set()
        for recipe in context.recipes:
            for ing in recipe.ingredients:
                used_names.add(ing.canonical_name.lower().strip())
        return [lot for lot in context.inventory_lots if lot.canonical_name.lower().strip() in used_names]

    def _is_perishable(self, name: str) -> bool:
        name_lower = name.lower()
        return any(kw in name_lower for kw in self._perishable_keywords)


# =============================================================================
# Rule 6: HoldingTimeRule — food held at unsafe temperatures too long
# =============================================================================

# Room-temperature holding limit comes from the active regional policy pack
# (P3-04) — default 120 min is the USDA 2-hour rule for backward compatibility.

_PERISHABLE_FOOD_KEYWORDS: tuple[str, ...] = (
    "chicken",
    "beef",
    "pork",
    "lamb",
    "mutton",
    "veal",
    "meat",
    "fish",
    "salmon",
    "tuna",
    "shrimp",
    "prawn",
    "crab",
    "lobster",
    "mussel",
    "clam",
    "oyster",
    "squid",
    "octopus",
    "egg",
    "milk",
    "cream",
    "yogurt",
    "cheese",
    "butter",
    "seafood",
    "poultry",
    "dairy",
)


@dataclass(frozen=True)
class HoldingTimeRule:
    """Flag dishes where long passive phases risk temperature abuse.

    Recipes with perishable proteins and total passive time above the active
    regional policy's room-temperature holding limit are flagged — food may
    cool into the danger zone where bacteria multiply rapidly. The scheduler
    can resolve this by placing completions near serving time.

    Severity: hard_repairable — can add cooling/reheating or adjust schedule.
    """

    rule_id: str = "SAFETY_HOLDING_TIME"

    # Max minutes perishable food may sit at room temperature before it is
    # flagged. Backward-compatible default is the USDA 2-hour rule (120 min);
    # production binds the resolved regional policy (P3-04).
    max_holding_minutes_room_temp: int = 120

    def evaluate(self, context: SafetyContext) -> SafetyFinding | None:
        risky: list[str] = []

        for recipe in context.recipes:
            if not self._has_perishable_protein(recipe):
                continue

            total_passive = sum((s.passive_duration_minutes or 0) for s in recipe.steps)
            total_active = sum((s.active_duration_minutes or 5) for s in recipe.steps)

            if total_passive > self.max_holding_minutes_room_temp:
                risky.append(
                    f"'{recipe.dish_name}' (~{total_passive + total_active}min total, {total_passive}min passive)"
                )

        if not risky:
            return None

        return SafetyFinding(
            rule_id=self.rule_id,
            severity="hard_repairable",
            description=(
                f"Holding-time risk: dishes with long passive phases "
                f"containing perishable proteins: {'; '.join(risky)}. "
                f"Passive time exceeds the regional room-temperature holding "
                f"limit of {self.max_holding_minutes_room_temp} minutes."
            ),
            recommended_action=(
                "Serve perishable food within the regional room-temperature "
                "holding limit of the plan's cooking completion. "
                "Keep hot food above the policy's hot-holding minimum or "
                "refrigerate below its cold-holding maximum. "
                "Stagger dish completions near serving time."
            ),
        )

    @staticmethod
    def _has_perishable_protein(recipe: RecipeIR) -> bool:
        for ingredient in recipe.ingredients:
            name_lower = ingredient.canonical_name.lower()
            if any(kw in name_lower for kw in _PERISHABLE_FOOD_KEYWORDS):
                return True
        return False


# =============================================================================
# Default rule set — composed by SafetyEngine
# =============================================================================

default_rules: tuple[
    CrossContaminationRule
    | AllergenDetectionRule
    | ProteinSafetyTemperatureRule
    | DietaryCompatibilityRule
    | ExpiredIngredientRule
    | HoldingTimeRule,
    ...,
] = (
    CrossContaminationRule(),
    AllergenDetectionRule(),
    ProteinSafetyTemperatureRule(),
    DietaryCompatibilityRule(),
    ExpiredIngredientRule(),
    HoldingTimeRule(),
)


def build_rules(policy: SafetyPolicy) -> tuple[SafetyRule, ...]:
    """Build the standard rule set bound to a resolved regional policy (P3-04).

    Threshold-driven rules (protein temperatures, room-temperature holding)
    are constructed from ``policy.thresholds``; structural rules are
    region-agnostic. This is the only place production wiring creates rules —
    the module-level ``default_rules`` stays only as a backward-compatible
    USDA-bound fallback for tests and legacy engines.
    """
    return (
        CrossContaminationRule(),
        AllergenDetectionRule(),
        ProteinSafetyTemperatureRule(safe_temperatures_c=dict(policy.thresholds.safe_minimum_temperatures_c)),
        DietaryCompatibilityRule(),
        ExpiredIngredientRule(),
        HoldingTimeRule(max_holding_minutes_room_temp=policy.thresholds.max_room_temp_holding_minutes),
    )


# =============================================================================
# Internal helpers
# =============================================================================


def _recipe_has_raw_protein(
    recipe: RecipeIR,
    raw_protein_keywords: tuple[str, ...],
) -> bool:
    """Check if any ingredient in the recipe is a raw protein."""
    for ingredient in recipe.ingredients:
        if ingredient.input_state == "raw" and _matches_keywords(
            ingredient.canonical_name.lower(), raw_protein_keywords
        ):
            return True
    return False


def _raw_protein_steps(
    recipe: RecipeIR,
    raw_protein_keywords: tuple[str, ...],
) -> tuple[RecipeStep, ...]:
    """Return steps that handle raw protein (P0-07 anchor detection).

    A step is considered raw-protein handling when the recipe contains a
    raw protein ingredient AND the step's instruction references it. Falls
    back to the first steps that mention raw keywords in their text.
    """
    if not _recipe_has_raw_protein(recipe, raw_protein_keywords):
        return ()
    matches = [s for s in recipe.steps if _matches_keywords(s.instruction.lower(), raw_protein_keywords)]
    if matches:
        return tuple(matches)
    # No explicit keyword in step text — assume the protein-handling step is
    # the earliest heating/mixing step, so we still anchor the insertion.
    return tuple(recipe.steps[:1])


def _recipe_has_rte_step(
    recipe: RecipeIR,
    rte_categories: tuple[str, ...],
) -> bool:
    """Check if any step handles ready-to-eat (plating, garnishing, etc.)."""
    for step in recipe.steps:
        if step.category.lower() in rte_categories:
            return True
    return False


def _rte_steps(
    recipe: RecipeIR,
    rte_categories: tuple[str, ...],
) -> tuple[RecipeStep, ...]:
    """Return ready-to-eat steps in order (P0-07 anchor detection)."""
    return tuple(s for s in recipe.steps if s.category.lower() in rte_categories)


def _matches_keywords(name: str, keywords: tuple[str, ...]) -> bool:
    """Check if name contains any of the given keywords."""
    return any(kw in name for kw in keywords)


def _is_protein_heating_step(step: RecipeStep) -> bool:
    """Check if a step applies heat to a protein (from instruction keywords)."""
    if step.heat_level == HeatLevel.NONE:
        return False

    instruction_lower = step.instruction.lower()
    return any(kw in instruction_lower for kw in _PROTEIN_KEYWORDS)


def _dominant_protein_type(recipe: RecipeIR) -> str:
    """Determine the dominant protein type of a recipe from its ingredients."""
    for ingredient in recipe.ingredients:
        name_lower = ingredient.canonical_name.lower()
        for kw, protein_type in _PROTEIN_KEYWORDS.items():
            if kw in name_lower:
                return protein_type
    return "unknown"
