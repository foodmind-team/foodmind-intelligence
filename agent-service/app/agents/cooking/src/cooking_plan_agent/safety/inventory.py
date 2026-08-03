"""Independently evaluable food-safety rule."""

from __future__ import annotations

from dataclasses import dataclass

from cooking_plan_agent.domain.models import (
    InventoryLotSnapshot,
    SafetyContext,
    SafetyFinding,
)


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
