"""Independently evaluable food-safety rule."""

from __future__ import annotations

from dataclasses import dataclass

from cooking_plan_agent.domain.models import (
    SafetyContext,
    SafetyFinding,
)
from cooking_plan_agent.safety.rule_support import _DIETARY_PROHIBITED


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
