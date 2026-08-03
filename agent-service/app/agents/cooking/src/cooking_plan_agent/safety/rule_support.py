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

from typing import Protocol

from cooking_plan_agent.domain.enums import HeatLevel
from cooking_plan_agent.domain.models import (
    RecipeIR,
    RecipeStep,
    SafetyContext,
    SafetyFinding,
)

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
