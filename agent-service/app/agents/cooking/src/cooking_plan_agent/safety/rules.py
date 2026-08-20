# =============================================================================
# 可组合食品安全规则的向后兼容公共 API（safety/rules）
# -----------------------------------------------------------------------------
# 本模块汇总导出所有可组合的食品安全规则类、SafetyRule 协议、
# build_rules 工厂函数与 default_rules 默认规则集，供工作流层直接引用。
# =============================================================================

"""Backward-compatible public API for composable food-safety rules. 可组合食品安全规则的向后兼容公共 API。"""

from cooking_plan_agent.safety.allergens import AllergenDetectionRule
from cooking_plan_agent.safety.cross_contamination import CrossContaminationRule
from cooking_plan_agent.safety.dietary import DietaryCompatibilityRule
from cooking_plan_agent.safety.holding import HoldingTimeRule
from cooking_plan_agent.safety.inventory import ExpiredIngredientRule
from cooking_plan_agent.safety.rule_factory import build_rules, default_rules
from cooking_plan_agent.safety.rule_support import SafetyRule
from cooking_plan_agent.safety.temperatures import ProteinSafetyTemperatureRule

__all__ = [
    "AllergenDetectionRule",
    "CrossContaminationRule",
    "DietaryCompatibilityRule",
    "ExpiredIngredientRule",
    "HoldingTimeRule",
    "ProteinSafetyTemperatureRule",
    "SafetyRule",
    "build_rules",
    "default_rules",
]
