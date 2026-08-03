"""Backward-compatible public API for composable food-safety rules."""

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
