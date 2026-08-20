# =============================================================================
# 菜谱缺口检测与分类模块（parsing/gaps）
# -----------------------------------------------------------------------------
# 实现手册 4.9–4.10：检测 ExtractedRecipeCandidate 中“缺失或低置信度”的字段，
# 并按严重程度把每个缺口分类。缺口分类驱动 LangGraph 工作流的路由决策。
# 缺口类别（GapClass）：
#   - critical            ：使调度不可靠的缺失信息（火力、时长）
#   - safety_critical     ：可能导致食物不安全的缺失信息（蛋白质的温度）
#   - resource_critical   ：可行性检查所需的缺失资源信息
#   - optimisation        ：阻碍优化但不阻断可行性的缺失信息
#   - cosmetic            ：次要缺失信息（装饰细节、可选备注）
# =============================================================================

"""Recipe gap detection and classification — handbook 4.9–4.10.

菜谱缺口检测与分类 —— 手册 4.9–4.10。

Detects missing or low-confidence fields in ExtractedRecipeCandidate and
classifies each gap by severity (critical, safety_critical, resource_critical,
optimisation, cosmetic). The gap classification drives routing decisions in
the LangGraph workflow.

检测 ExtractedRecipeCandidate 中缺失或低置信度的字段，并按严重程度
（critical、safety_critical、resource_critical、optimisation、cosmetic）分类。
缺口分类驱动 LangGraph 工作流的路由决策。
"""

from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

from cooking_plan_agent.domain.enums import HeatLevel
from cooking_plan_agent.domain.models import (
    ExtractedRecipeCandidate,
    ExtractedStep,
    RecipeGap,
)

# =============================================================================
# Gap classification
# 缺口分类
# =============================================================================


class GapClass:
    """缺口严重程度类别，用于路由决策。"""

    CRITICAL = "critical"
    """Missing info that makes scheduling unreliable (heat, duration).
    使调度不可靠的缺失信息（火力、时长）。"""

    SAFETY_CRITICAL = "safety_critical"
    """Missing info that could lead to unsafe food (temperature for proteins).
    可能导致食物不安全的缺失信息（蛋白质的温度）。"""

    RESOURCE_CRITICAL = "resource_critical"
    """Missing resource info needed for feasibility check.
    可行性检查所需的缺失资源信息。"""

    OPTIMISATION = "optimisation"
    """Missing info that prevents optimisation but doesn't block feasibility.
    阻碍优化但不阻断可行性的缺失信息。"""

    COSMETIC = "cosmetic"
    """Minor missing info (garnish details, optional notes).
    次要缺失信息（装饰细节、可选备注）。"""


# =============================================================================
# Gap detection
# 缺口检测
# =============================================================================


def find_recipe_gaps(candidate: ExtractedRecipeCandidate) -> tuple[RecipeGap, ...]:
    """检测单个提取菜谱候选中的所有缺口。

    Detect all gaps in a single extracted recipe candidate.

    Checks each step for missing heat, duration, temperature, and resource
    information. Also checks candidate-level fields (servings, dish name).

    检查每个步骤缺失的火力、时长、温度与资源信息，也检查候选级字段（份数、菜名）。

    Args:
        candidate: An extracted recipe candidate from RecipeExtractor.
            candidate：来自 RecipeExtractor 的提取菜谱候选。

    Returns:
        Tuple of RecipeGap objects, one per detected gap.
        每个检测到的缺口一个 RecipeGap 对象组成的元组。
    """
    gaps: list[RecipeGap] = []

    # --- Candidate-level gaps 候选级缺口 ---
    gaps.extend(_check_candidate_gaps(candidate))

    # --- Step-level gaps 步骤级缺口 ---
    for i, step in enumerate(candidate.steps):
        gaps.extend(_check_step_gaps(candidate.recipe_id, i, step))

    return tuple(gaps)


# =============================================================================
# Internal check helpers
# 内部检查辅助函数
# =============================================================================


def _check_candidate_gaps(candidate: ExtractedRecipeCandidate) -> list[RecipeGap]:
    """检查候选级字段的缺口。"""
    gaps: list[RecipeGap] = []

    if not candidate.dish_name or candidate.dish_name == "Untitled Recipe":
        gaps.append(
            RecipeGap(
                gap_id=_gap_id(candidate.recipe_id, "dish_name"),
                recipe_id=candidate.recipe_id,
                field_path="dish_name",
                gap_class=GapClass.COSMETIC,
                description="Dish name could not be extracted from recipe text",
                confidence=Decimal("1.0"),
            )
        )

    if candidate.original_servings <= 1:
        gaps.append(
            RecipeGap(
                gap_id=_gap_id(candidate.recipe_id, "servings"),
                recipe_id=candidate.recipe_id,
                field_path="original_servings",
                gap_class=GapClass.OPTIMISATION,
                description="Serving count not specified; defaulting to 2",
                confidence=Decimal("0.8"),
            )
        )

    return gaps


def _check_step_gaps(recipe_id: str, step_index: int, step: ExtractedStep) -> list[RecipeGap]:
    """检查单个步骤的缺口。"""
    gaps: list[RecipeGap] = []
    field_prefix = f"steps[{step_index}]"

    # Heating steps must have heat level
    # 加热步骤必须有火力档位
    if step.category == "heating":
        gaps.extend(_check_heat_gaps(recipe_id, step_index, step, field_prefix))
        gaps.extend(_check_duration_gaps(recipe_id, step_index, step, field_prefix))
        gaps.extend(_check_temperature_gaps(recipe_id, step_index, step, field_prefix))

    # All steps should have at least one resource hint for feasibility
    # 所有步骤应至少有一个资源提示，用于可行性检查
    if not step.resources_hint:
        gaps.append(
            RecipeGap(
                gap_id=_gap_id(recipe_id, f"{field_prefix}.resources"),
                recipe_id=recipe_id,
                field_path=f"{field_prefix}.resources_hint",
                gap_class=GapClass.RESOURCE_CRITICAL,
                description=f"Step {step.step_number} has no resource hints — feasibility check may be incomplete",
                confidence=Decimal("0.7"),
            )
        )

    return gaps


def _check_heat_gaps(recipe_id: str, step_index: int, step: ExtractedStep, prefix: str) -> list[RecipeGap]:
    """检查加热步骤缺失的火力档位。"""
    if step.heat_level == HeatLevel.NONE:
        return [
            RecipeGap(
                gap_id=_gap_id(recipe_id, f"{prefix}.heat"),
                recipe_id=recipe_id,
                field_path=f"{prefix}.heat_level",
                gap_class=_heat_gap_class(step),
                description=f"Step {step.step_number}: heat level not specified for '{step.instruction[:60]}...'",
                confidence=Decimal("1.0"),
            )
        ]
    return []


def _check_duration_gaps(recipe_id: str, step_index: int, step: ExtractedStep, prefix: str) -> list[RecipeGap]:
    """检查需要时长的步骤缺失的时长。"""
    gaps: list[RecipeGap] = []

    # Heating steps that don't specify any duration
    # 未指定任何时长的加热步骤
    if step.active_duration_minutes is None and step.passive_duration_minutes is None:
        # Boiling/baking without time is a critical gap
        # 煮沸 / 烘焙无时长是关键缺口
        is_time_sensitive = _is_time_sensitive_step(step)
        gap_class = GapClass.CRITICAL if is_time_sensitive else GapClass.OPTIMISATION
        gaps.append(
            RecipeGap(
                gap_id=_gap_id(recipe_id, f"{prefix}.duration"),
                recipe_id=recipe_id,
                field_path=f"{prefix}.passive_duration_minutes",
                gap_class=gap_class,
                description=f"Step {step.step_number}: no duration specified for '{step.instruction[:60]}...'",
                confidence=Decimal("1.0"),
            )
        )

    return gaps


def _check_temperature_gaps(recipe_id: str, step_index: int, step: ExtractedStep, prefix: str) -> list[RecipeGap]:
    """检查需要精确温度的步骤缺失的目标温度。"""
    # Only check if the step looks like it needs precise temperature
    # (baking, roasting, or involves proteins)
    # 仅当步骤看起来需要精确温度（烘焙、烤，或涉及蛋白质）时检查
    if step.target_temperature_c is None and _needs_temperature(step):
        return [
            RecipeGap(
                gap_id=_gap_id(recipe_id, f"{prefix}.temperature"),
                recipe_id=recipe_id,
                field_path=f"{prefix}.target_temperature_c",
                gap_class=GapClass.SAFETY_CRITICAL if _involves_protein(step) else GapClass.CRITICAL,
                description=f"Step {step.step_number}: temperature not specified for '{step.instruction[:60]}...'",
                confidence=Decimal("1.0"),
            )
        ]
    return []


# =============================================================================
# Classification helpers
# 分类辅助函数
# =============================================================================


def _heat_gap_class(step: ExtractedStep) -> str:
    """确定缺失火力档位的缺口类别。"""
    # Baking/roasting without temperature info is safety-critical for proteins
    # 对蛋白质而言，烘焙 / 烤缺少温度信息是安全关键
    instruction_lower = step.instruction.lower()
    if any(kw in instruction_lower for kw in ("chicken", "beef", "pork", "fish", "meat", "seafood")):
        return GapClass.SAFETY_CRITICAL
    return GapClass.CRITICAL


def _is_time_sensitive_step(step: ExtractedStep) -> bool:
    """判断步骤是否对时间敏感（煮、炖、烤、烘焙等）。"""
    time_sensitive_keywords = (
        "boil",
        "simmer",
        "bake",
        "roast",
        "grill",
        "steam",
        "fry",
        "marinate",
        "stew",
        "braise",
        "poach",
        "煮",
        "焖",
        "炖",
        "烤",
        "蒸",
        "炸",
        "煎",
        "腌",
    )
    lower = step.instruction.lower()
    return any(kw in lower for kw in time_sensitive_keywords)


def _needs_temperature(step: ExtractedStep) -> bool:
    """判断步骤是否需要精确温度。"""
    temp_keywords = ("bake", "roast", "oven", "烤", "烘")
    lower = step.instruction.lower()
    return any(kw in lower for kw in temp_keywords)


def _involves_protein(step: ExtractedStep) -> bool:
    """判断步骤是否可能涉及生蛋白质（安全关键温度）。"""
    protein_keywords = (
        "chicken",
        "beef",
        "pork",
        "fish",
        "shrimp",
        "meat",
        "poultry",
        "seafood",
        "lamb",
        "turkey",
        "duck",
        "鸡肉",
        "牛肉",
        "猪肉",
        "鱼肉",
        "虾",
        "肉",
        "羊肉",
        "鸭",
    )
    lower = step.instruction.lower()
    return any(kw in lower for kw in protein_keywords)


def _gap_id(recipe_id: str, field_path: str) -> str:
    """生成稳定的缺口 ID。"""
    suffix = field_path.replace("[", "_").replace("]", "").replace(".", "_")
    short = uuid4().hex[:8]
    return f"gap_{recipe_id}_{suffix}_{short}"
