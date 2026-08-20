# =============================================================================
# 修复模块（repair 包）— 为不可行计划提出、校验、排序并应用修复方案
# -----------------------------------------------------------------------------
# 导出修复选项的公共 API，并对既有导入方保持兼容。
# =============================================================================

"""Repair options — propose, validate, rank, and apply fixes for infeasible plans.

修复选项 —— 为不可行计划提出、校验、排序并应用修复方案。
"""

from cooking_plan_agent.repair.options import (
    RepairValidation,
    Shortage,
    apply_approved_decisions,
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
    "RepairValidation",
    "Shortage",
    "apply_approved_decisions",
    "calculate_exact_shortages",
    "propose_dish_replacements",
    "propose_equipment_alternatives",
    "propose_ingredient_substitutions",
    "propose_portion_adjustments",
    "propose_time_extension",
    "rank_repair_options",
    "validate_repair_option",
]
