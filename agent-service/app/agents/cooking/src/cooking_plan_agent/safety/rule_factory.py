"""Factories for the standard, policy-bound food-safety rule set."""

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
    """Build the standard rule set bound to a resolved regional policy."""
    return (
        CrossContaminationRule(),
        AllergenDetectionRule(),
        ProteinSafetyTemperatureRule(safe_temperatures_c=dict(policy.thresholds.safe_minimum_temperatures_c)),
        DietaryCompatibilityRule(),
        ExpiredIngredientRule(),
        HoldingTimeRule(max_holding_minutes_room_temp=policy.thresholds.max_room_temp_holding_minutes),
    )
