"""Backward-compatible public API for repair option services.

Implementation is separated into proposal generation, decision processing, and
small value objects.  Keep these imports stable for API consumers.
"""

from cooking_plan_agent.repair.decisions import (
    ConfirmationAnswersError,
    answers_to_approved_decisions,
    apply_approved_decisions,
    apply_approved_decisions_structured,
    apply_ingredient_substitutions_patch,
    build_approved_decisions,
    validate_approved_decisions,
)
from cooking_plan_agent.repair.models import RepairValidation, Shortage
from cooking_plan_agent.repair.proposals import (
    calculate_exact_shortages,
    propose_dish_replacements,
    propose_equipment_alternatives,
    propose_ingredient_substitutions,
    propose_portion_adjustments,
    propose_time_extension,
    rank_repair_options,
    validate_repair_option,
)

__all__ = [
    "ConfirmationAnswersError",
    "RepairValidation",
    "Shortage",
    "answers_to_approved_decisions",
    "apply_approved_decisions",
    "apply_approved_decisions_structured",
    "apply_ingredient_substitutions_patch",
    "build_approved_decisions",
    "calculate_exact_shortages",
    "propose_dish_replacements",
    "propose_equipment_alternatives",
    "propose_ingredient_substitutions",
    "propose_portion_adjustments",
    "propose_time_extension",
    "rank_repair_options",
    "validate_approved_decisions",
    "validate_repair_option",
]
