# =============================================================================
# 保温时间规则（safety/holding）
# -----------------------------------------------------------------------------
# HoldingTimeRule：标记长时间被动阶段存在温度滥用风险的菜品。
# 当含易腐蛋白质的菜品被动时长超过区域策略的室温保温上限时予以标记，
# 以防食物冷却进入危险温度区、细菌大量繁殖。
# =============================================================================

"""Independently evaluable food-safety rule. 可独立评估的食品安全规则。"""

from __future__ import annotations

from dataclasses import dataclass

from cooking_plan_agent.domain.models import (
    RecipeIR,
    SafetyContext,
    SafetyFinding,
)

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

    标记长时间被动阶段存在温度滥用风险的菜品。

    Recipes with perishable proteins and total passive time above the active
    regional policy's room-temperature holding limit are flagged — food may
    cool into the danger zone where bacteria multiply rapidly. The scheduler
    can resolve this by placing completions near serving time.

    含易腐蛋白质且总被动时长超过当前区域策略室温保温上限的菜谱会被标记 ——
    食物可能冷却进入危险温度区，细菌在此迅速繁殖。
    调度器可通过将完成时间安排在临近上菜时来解决。

    Severity: hard_repairable — can add cooling/reheating or adjust schedule.

    严重级别：hard_repairable —— 可增加冷却 / 复热或调整排程。
    """

    rule_id: str = "SAFETY_HOLDING_TIME"

    # Max minutes perishable food may sit at room temperature before it is
    # flagged. Backward-compatible default is the USDA 2-hour rule (120 min);
    # production binds the resolved regional policy (P3-04).
    # 易腐食品在室温下可放置、直至被标记的最长分钟数。
    # 向后兼容的默认值为 USDA 两小时规则（120 分钟）；
    # 生产环境绑定已解析的区域策略（P3-04）。
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
