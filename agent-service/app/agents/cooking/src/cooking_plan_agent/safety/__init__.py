"""Safety rule engine — evaluates food safety constraints against parsed recipes.

Handbook 5.7–5.9: safety validation is a mandatory pre-scheduling gate.
The engine applies independent rules (cross-contamination, allergens,
safe cooking temperatures, dietary compatibility) and produces a
SafetyReport consumed by the LangGraph routing layer.
"""

from cooking_plan_agent.safety.engine import SafetyEngine
from cooking_plan_agent.safety.rules import (
    AllergenDetectionRule,
    CrossContaminationRule,
    DietaryCompatibilityRule,
    ProteinSafetyTemperatureRule,
    SafetyRule,
    default_rules,
)

__all__ = [
    "SafetyEngine",
    "SafetyRule",
    "CrossContaminationRule",
    "AllergenDetectionRule",
    "ProteinSafetyTemperatureRule",
    "DietaryCompatibilityRule",
    "default_rules",
]
