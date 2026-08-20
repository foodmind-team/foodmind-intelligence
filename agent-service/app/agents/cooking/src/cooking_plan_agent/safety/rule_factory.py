# =============================================================================
# 规则工厂（safety/rule_factory）
# -----------------------------------------------------------------------------
# 本文件构造标准、绑定策略的食品安全规则集：default_rules 提供向后兼容的
# 默认规则元组，build_rules 则根据已解析的区域策略构建绑定阈值的规则集。
# =============================================================================

"""Factories for the standard, policy-bound food-safety rule set.

标准、绑定策略的食品安全规则集的工厂。
"""

from cooking_plan_agent.safety.allergens import AllergenDetectionRule
from cooking_plan_agent.safety.cross_contamination import CrossContaminationRule
from cooking_plan_agent.safety.dietary import DietaryCompatibilityRule
from cooking_plan_agent.safety.holding import HoldingTimeRule
from cooking_plan_agent.safety.inventory import ExpiredIngredientRule
from cooking_plan_agent.safety.policy import SafetyPolicy
from cooking_plan_agent.safety.rule_support import SafetyRule
from cooking_plan_agent.safety.temperatures import ProteinSafetyTemperatureRule

default_rules: tuple[SafetyRule, ...] = (
    CrossContaminationRule(),
    AllergenDetectionRule(),
    ProteinSafetyTemperatureRule(),
    DietaryCompatibilityRule(),
    ExpiredIngredientRule(),
    HoldingTimeRule(),
)


def build_rules(policy: SafetyPolicy) -> tuple[SafetyRule, ...]:
    """Build the standard rule set bound to a resolved regional policy. 构建绑定到已解析区域策略的标准规则集。"""
    return (
        CrossContaminationRule(),
        AllergenDetectionRule(),
        ProteinSafetyTemperatureRule(safe_temperatures_c=dict(policy.thresholds.safe_minimum_temperatures_c)),
        DietaryCompatibilityRule(),
        ExpiredIngredientRule(),
        HoldingTimeRule(max_holding_minutes_room_temp=policy.thresholds.max_room_temp_holding_minutes),
    )
