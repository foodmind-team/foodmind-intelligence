"""Independently evaluable food-safety rule."""

from __future__ import annotations

from dataclasses import dataclass

from cooking_plan_agent.domain.models import (
    SafetyContext,
    SafetyFinding,
    SafetyInsertion,
)
from cooking_plan_agent.safety.rule_support import (
    _SANITISE_DURATION_MINUTES,
    _SANITISE_REQUIRED_RESOURCES,
    _matches_keywords,
    _raw_protein_steps,
    _rte_steps,
)


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
