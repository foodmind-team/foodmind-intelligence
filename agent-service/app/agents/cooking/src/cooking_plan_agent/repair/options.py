# =============================================================================
# 修复选项服务公共 API 模块（repair/options）
# -----------------------------------------------------------------------------
# 向后兼容的公共入口：实现已拆分到提议生成、决策处理与小型值对象模块，
# 此文件保持导出稳定，供 API 消费者使用。
# =============================================================================

"""Backward-compatible public API for repair option services.

修复选项服务的向后兼容公共 API。

Implementation is separated into proposal generation, decision processing, and
small value objects.  Keep these imports stable for API consumers.

实现被拆分为提议生成、决策处理和小型值对象。保持这些导入对 API 消费者稳定。
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
