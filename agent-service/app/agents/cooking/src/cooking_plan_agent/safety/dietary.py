# =============================================================================
# 膳食兼容性规则（safety/dietary）
# -----------------------------------------------------------------------------
# DietaryCompatibilityRule：将菜谱食材与用户的膳食限制（清真、素食、
# 纯素、犹太洁食）比对，检测违反膳食限制的食材。
# =============================================================================

"""Independently evaluable food-safety rule.

可独立评估的食品安全规则。
"""

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

    对照用户膳食限制检查菜谱食材。

    Matches ingredient names against prohibited keyword lists for common
    dietary patterns (halal, vegetarian, vegan, kosher).

    将食材名称与常见膳食模式（清真、素食、纯素、犹太洁食）的禁用关键词表匹配。

    Severity: hard_unrepairable — dietary restrictions are non-negotiable.

    严重级别：hard_unrepairable —— 膳食限制不可协商。
    """

    rule_id: str = "SAFETY_DIETARY_COMPATIBILITY"

    def evaluate(self, context: SafetyContext) -> SafetyFinding | None:
        """Check all ingredients against dietary restrictions. 对照膳食限制检查所有食材。"""
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
# 规则 5：过期食材规则 —— 检查库存批次是否变质
# =============================================================================
