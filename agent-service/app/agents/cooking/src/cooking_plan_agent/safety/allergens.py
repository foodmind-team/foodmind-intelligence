"""Independently evaluable food-safety rule."""

from __future__ import annotations

from dataclasses import dataclass, field

from cooking_plan_agent.domain.models import (
    SafetyContext,
    SafetyFinding,
)


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
