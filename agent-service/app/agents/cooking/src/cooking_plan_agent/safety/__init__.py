# =============================================================================
# 安全验证模块入口（safety/__init__）
# -----------------------------------------------------------------------------
# 本模块是安全验证的公共入口：导出 SafetyEngine、SafetyRule 协议以及
# 各独立安全规则类与默认规则集，供 LangGraph 路由层组合使用。
# =============================================================================

"""Safety rule engine — evaluates food safety constraints against parsed recipes.

安全规则引擎 —— 对已解析的菜谱评估食品安全约束。

Handbook 5.7–5.9: safety validation is a mandatory pre-scheduling gate.
The engine applies independent rules (cross-contamination, allergens,
safe cooking temperatures, dietary compatibility) and produces a
SafetyReport consumed by the LangGraph routing layer.

手册 5.7–5.9：安全验证是调度之前的强制性门禁。
引擎应用独立规则（交叉污染、过敏原、安全烹饪温度、膳食兼容性），
并产出供 LangGraph 路由层消费的 SafetyReport。
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
