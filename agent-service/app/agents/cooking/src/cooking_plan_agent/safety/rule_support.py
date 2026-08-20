# =============================================================================
# 食品安全规则支撑模块（safety/rule_support）
# -----------------------------------------------------------------------------
# 本文件提供所有食品安全规则共用的协议定义与辅助工具，包括：
#   - SafetyRule 协议：所有规则实现必须遵守的统一接口契约
#   - 蛋白质关键词、膳食限制禁用词表、即食（RTE）步骤识别等辅助函数
#   - 插入式安全任务（消毒）的默认时长与所需资源
# 规则本身不在此实现，而是由 safety/rules.py 等模块组合引用。
# =============================================================================

"""Safety rules — individual, independently evaluable food safety constraints.

食品安全规则 —— 单个、可独立评估的食品安全约束。

Each rule implements the SafetyRule protocol: evaluate(SafetyContext) →
SafetyFinding | None. Rules are designed to be composed by SafetyEngine
with no cross-rule dependencies. Every rule is unit-testable in isolation.

每个规则都实现 SafetyRule 协议：evaluate(SafetyContext) → SafetyFinding | None。
规则被设计为可由 SafetyEngine 组合且彼此之间无跨规则依赖，
每条规则均可独立进行单元测试。

Handbook 5.7: safety rules are the first hard gate after parsing.
Handbook 5.8: rules produce three severity levels:
  - hard_unrepairable → block the plan entirely (INFEASIBLE)
  - hard_repairable   → inject safety_tasks (e.g., sanitise board)
  - warning           → surface to user but do not block

手册 5.7：安全规则是解析之后的第一个硬性门禁。
手册 5.8：规则产生三个严重级别：
  - hard_unrepairable → 彻底阻止计划（INFEASIBLE）
  - hard_repairable   → 注入 safety_tasks（例如消毒砧板）
  - warning           → 提示给用户但不阻止
"""

from __future__ import annotations

from typing import Protocol

from cooking_plan_agent.domain.enums import HeatLevel
from cooking_plan_agent.domain.models import (
    RecipeIR,
    RecipeStep,
    SafetyContext,
    SafetyFinding,
)

# =============================================================================
# P0-07 safety-task policy — durations and resources for inserted safety tasks
# P0-07 安全任务策略 —— 插入式安全任务的时长与所需资源
# =============================================================================

# Sanitisation task default duration.  Replaces the old hard-coded 1-minute
# placeholder: the inserted task must occupy a realistic window so the
# verifier can check raw→sanitise→RTE ordering meaningfully.
# 消毒任务的默认时长。取代旧的一分钟硬编码占位值：
# 插入的任务必须占据一个真实的时长窗口，以便验证器能够有意义地检查
# 生食 → 消毒 → 即食 的先后顺序。
_SANITISE_DURATION_MINUTES = 3
# Resources a sanitisation task requires (Policy 6.1: sink).
# 消毒任务所需资源（策略 6.1：水槽）。
_SANITISE_REQUIRED_RESOURCES = ("sink",)

# =============================================================================
# SafetyRule protocol — contract for all rule implementations
# SafetyRule 协议 —— 所有规则实现的契约
# =============================================================================


class SafetyRule(Protocol):
    """A single independently evaluable safety constraint.

    一条可独立评估的安全约束。

    Each rule receives the full SafetyContext and returns either a
    SafetyFinding (violation detected) or None (rule satisfied).
    Rules MUST NOT mutate the context or have side effects.

    每条规则接收完整的 SafetyContext，并返回 SafetyFinding（检测到违规）
    或 None（规则满足）。规则不得修改上下文，也不得产生副作用。

    rule_id is declared read-only because every concrete rule is a
    frozen dataclass — instance attributes are never settable.

    rule_id 被声明为只读，因为每个具体规则都是冻结的数据类 ——
    实例属性永远不可被赋值。
    """

    @property
    def rule_id(self) -> str: ...

    def evaluate(self, context: SafetyContext) -> SafetyFinding | None: ...


# =============================================================================
# Safe minimum internal temperatures (P3-04)
# 安全最低内部温度（P3-04）
# =============================================================================

# Protein internal-temperature thresholds are no longer hard-coded here — they
# live in versioned, source-backed regional policy packs (safety/policies/).
# The USDA table is imported only as the backward-compatible default so rules
# constructed without an explicit policy keep their historical behaviour.
# Production wiring binds rules to a resolved policy via build_rules(policy).
# 蛋白质内部温度阈值不再硬编码于此 —— 它们位于带版本、有来源依据的
# 区域策略包（safety/policies/）中。USDA 表仅作为向后兼容的默认值导入，
# 以便未显式指定策略构造的规则保持其历史行为。
# 生产环境通过 build_rules(policy) 将规则绑定到已解析的策略。

# Protein keywords for matching ingredient names to protein categories
# 用于将食材名称匹配到蛋白质类别的蛋白质关键词
_PROTEIN_KEYWORDS: dict[str, str] = {
    # Poultry
    # 禽类
    "chicken": "chicken",
    "turkey": "turkey",
    "duck": "duck",
    "goose": "goose",
    # Red meat
    # 红肉
    "beef": "beef",
    "pork": "pork",
    "lamb": "lamb",
    "veal": "veal",
    # Seafood
    # 海鲜
    "fish": "fish",
    "salmon": "salmon",
    "tuna": "fish",
    "shrimp": "shrimp",
    "prawn": "shellfish",
    "crab": "shellfish",
    "lobster": "shellfish",
    "mussel": "shellfish",
    "clam": "shellfish",
    "oyster": "shellfish",
    "squid": "shellfish",
    "octopus": "shellfish",
    # Other
    # 其他
    "egg": "egg",
}


# =============================================================================
# Dietary restriction keyword matching
# 膳食限制关键词匹配
# =============================================================================

# Ingredients prohibited per dietary restriction
# 每种膳食限制所禁止的食材
_DIETARY_PROHIBITED: dict[str, tuple[str, ...]] = {
    "halal": (
        "pork",
        "bacon",
        "ham",
        "lard",
        "sausage",
        "alcohol",
        "wine",
        "beer",
        "sake",
        "mirin",
        "rum",
        "gelatin",  # unless halal-certified
        # 除非经过清真认证
    ),
    "vegetarian": (
        "chicken",
        "beef",
        "pork",
        "lamb",
        "mutton",
        "veal",
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
        "bacon",
        "ham",
        "sausage",
        "meat",
    ),
    "vegan": (
        "chicken",
        "beef",
        "pork",
        "lamb",
        "mutton",
        "veal",
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
        "cheese",
        "butter",
        "cream",
        "yogurt",
        "honey",
        "gelatin",
        "bacon",
        "ham",
        "sausage",
        "meat",
    ),
    "kosher": (
        "pork",
        "bacon",
        "ham",
        "lard",
        "shellfish",
        "shrimp",
        "prawn",
        "crab",
        "lobster",
        "mussel",
        "clam",
        "oyster",
        "squid",
        "octopus",
        # Meat + dairy mixing is complex; flag for now
        # 肉类与奶制品混用较为复杂；暂先标记
    ),
}


def _recipe_has_raw_protein(
    recipe: RecipeIR,
    raw_protein_keywords: tuple[str, ...],
) -> bool:
    """Check if any ingredient in the recipe is a raw protein. 检查菜谱中是否有任何食材为生蛋白质。"""
    for ingredient in recipe.ingredients:
        if ingredient.input_state == "raw" and _matches_keywords(
            ingredient.canonical_name.lower(), raw_protein_keywords
        ):
            return True
    return False


def _raw_protein_steps(
    recipe: RecipeIR,
    raw_protein_keywords: tuple[str, ...],
) -> tuple[RecipeStep, ...]:
    """Return steps that handle raw protein (P0-07 anchor detection).

    返回处理生蛋白质的步骤（P0-07 锚点检测）。

    A step is considered raw-protein handling when the recipe contains a
    raw protein ingredient AND the step's instruction references it. Falls
    back to the first steps that mention raw keywords in their text.

    当菜谱包含生蛋白质食材且步骤说明引用了它时，该步骤被视为处理生蛋白质。
    否则回退到文本中提到生食关键词的最靠前步骤。
    """
    if not _recipe_has_raw_protein(recipe, raw_protein_keywords):
        return ()
    matches = [s for s in recipe.steps if _matches_keywords(s.instruction.lower(), raw_protein_keywords)]
    if matches:
        return tuple(matches)
    # No explicit keyword in step text — assume the protein-handling step is
    # the earliest heating/mixing step, so we still anchor the insertion.
    # 步骤文本中没有显式关键词 —— 假定蛋白质处理步骤是最早的加热/搅拌步骤，
    # 这样我们仍能锚定插入位置。
    return tuple(recipe.steps[:1])


def _recipe_has_rte_step(
    recipe: RecipeIR,
    rte_categories: tuple[str, ...],
) -> bool:
    """Check if any step handles ready-to-eat (plating, garnishing, etc.). 检查是否有步骤处理即食食材（装盘、点缀等）。"""
    for step in recipe.steps:
        if step.category.lower() in rte_categories:
            return True
    return False


def _rte_steps(
    recipe: RecipeIR,
    rte_categories: tuple[str, ...],
) -> tuple[RecipeStep, ...]:
    """Return ready-to-eat steps in order (P0-07 anchor detection).

    按顺序返回即食步骤（P0-07 锚点检测）。
    """
    return tuple(s for s in recipe.steps if s.category.lower() in rte_categories)


def _matches_keywords(name: str, keywords: tuple[str, ...]) -> bool:
    """Check if name contains any of the given keywords.

    检查 name 是否包含任意给定关键词。
    """
    return any(kw in name for kw in keywords)


def _is_protein_heating_step(step: RecipeStep) -> bool:
    """Check if a step applies heat to a protein (from instruction keywords). 检查某步骤是否对蛋白质加热（根据说明中的关键词判断）。"""
    if step.heat_level == HeatLevel.NONE:
        return False

    instruction_lower = step.instruction.lower()
    return any(kw in instruction_lower for kw in _PROTEIN_KEYWORDS)


def _dominant_protein_type(recipe: RecipeIR) -> str:
    """Determine the dominant protein type of a recipe from its ingredients. 根据食材确定菜谱的主导蛋白质类型。"""
    for ingredient in recipe.ingredients:
        name_lower = ingredient.canonical_name.lower()
        for kw, protein_type in _PROTEIN_KEYWORDS.items():
            if kw in name_lower:
                return protein_type
    return "unknown"
