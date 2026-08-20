# =============================================================================
# IR 构建器模块（parsing/ir_builder）
# -----------------------------------------------------------------------------
# 实现手册 4.12–4.14：把 ExtractedRecipeCandidate 转换为规范中间表示（RecipeIR），
# 并执行语义校验，拒绝“结构合法但逻辑上不可能”的菜谱。
# 核心职责：
#   - build_recipe_ir           ：把提取候选转换为已校验的 RecipeIR（含份数缩放 P0-04）
#   - validate_recipe_ir_semantics：对 RecipeIR 做语义校验（超出 Pydantic 字段校验）
#   - attach_research_assumptions：把联网证据假设合并进每个 RecipeIR（P1-01）
# 关键点：份数缩放（P0-04）在可行性 / 安全计算之前应用；连续量精确缩放，
#         离散量（个/颗/瓣）向上取整，绝不供应不足。
# =============================================================================

"""IR Builder — converts ExtractedRecipeCandidate to validated RecipeIR.

IR 构建器 —— 把 ExtractedRecipeCandidate 转换为已校验的 RecipeIR。

Handbook 4.12–4.14: this module builds the canonical Intermediate Representation
from one or more extracted candidates. It also performs semantic validation
to reject structurally valid but logically impossible recipes.

手册 4.12–4.14：本模块从一个或多个提取候选构建规范中间表示，
并执行语义校验以拒绝“结构合法但逻辑上不可能”的菜谱。
"""

from __future__ import annotations

from decimal import Decimal
from typing import NamedTuple

from cooking_plan_agent.domain.enums import HeatLevel
from cooking_plan_agent.domain.models import (
    Assumption,
    ExtractedIngredient,
    ExtractedRecipeCandidate,
    ExtractedStep,
    IngredientDemand,
    RecipeIR,
    RecipeStep,
)
from cooking_plan_agent.normalisation.units import UnitClassifier

# =============================================================================
# SemanticValidationReport
# 语义校验报告
# =============================================================================


class SemanticIssue(NamedTuple):
    """单条语义校验问题。"""

    code: str
    """Machine-readable issue code (e.g. 'NO_INGREDIENTS', 'NEGATIVE_DURATION').
    机器可读问题码（如 'NO_INGREDIENTS'、'NEGATIVE_DURATION'）。"""

    severity: str
    """'error' (reject) or 'warning' (accept with caution).
    'error'（拒绝）或 'warning'（谨慎接受）。"""

    message: str
    """Human-readable description.
    可读描述。"""


class RecipeValidationReport(NamedTuple):
    """对一个或多个 RecipeIR 对象做语义校验的结果。

    Result of semantic validation on one or more RecipeIR objects.

    passed=True means no 'error'-severity issues were found.

    passed=True 表示未发现 'error' 级问题。
    """

    passed: bool
    issues: tuple[SemanticIssue, ...]
    recipe_count: int


# =============================================================================
# Public API
# 公共 API
# =============================================================================


def build_recipe_ir(
    candidate: ExtractedRecipeCandidate,
    *,
    request_recipe_id: str | None = None,
    target_servings: Decimal | None = None,
) -> RecipeIR:
    """把 ExtractedRecipeCandidate 转换为已校验的 RecipeIR。

    Convert an ExtractedRecipeCandidate into a validated RecipeIR.

    Handles:
      - ExtractedIngredient → IngredientDemand (with unit normalisation)
      - ExtractedStep → RecipeStep (with technique-pattern inference)
      - Collects assumptions from extraction
      - Serving scaling (P0-04): when target_servings differs from the
        recipe's original servings, every continuous-quantity ingredient
        is scaled by ``target / original`` using Decimal arithmetic.
        Discrete units (piece, egg, …) are rounded up per the configured
        rounding policy, recording an assumption when rounding occurred.

    处理：
      - ExtractedIngredient → IngredientDemand（含单位规范化）
      - ExtractedStep → RecipeStep（含技法模式推断）
      - 从提取过程收集假设
      - 份数缩放（P0-04）：当 target_servings 与菜谱原始份数不同时，
        每个连续量食材按 target / original 用 Decimal 算术缩放。
        离散单位（个、鸡蛋、…）按配置的取整策略向上取整，取整时记录假设。

    Args:
        candidate: An extracted recipe candidate (possibly after inference).
            candidate：提取的菜谱候选（可能已推断）。
        request_recipe_id: The recipe ID from the caller's request. When
            provided it OVERRIDES the extractor's internal recipe_id so the
            final identity always matches the request (P0-04 rule 5).
            request_recipe_id：调用方请求中的 recipe ID。提供时它覆盖提取器内部
            recipe_id，使最终标识始终匹配请求（P0-04 规则 5）。
        target_servings: Desired servings. When None, defaults to the
            recipe's original servings (1:1 — unchanged behaviour).
            target_servings：期望份数。None 时默认用菜谱原始份数（1:1 —— 行为不变）。

    Returns:
        A validated RecipeIR ready for the scheduling pipeline.
        供调度流水线使用的已校验 RecipeIR。

    Raises:
        ValueError: If the candidate has no ingredients or no steps.
        ValueError：候选无食材或无步骤时抛出。
    """
    recipe_id = request_recipe_id or candidate.recipe_id

    parsed_demands = tuple(_build_ingredient_demand(ing) for ing in candidate.ingredients)

    # Filter out ingredients without a valid name or quantity
    # 过滤掉无有效名称或数量的食材
    ingredients = tuple(i for i in parsed_demands if i and i.canonical_name)

    # P0-04: apply serving scaling before any downstream feasibility/safety
    # computation. Missing quantities are never silently invented — they are
    # left as-is (quantity still present) and gaps are preserved upstream.
    # P0-04：在任何下游可行性 / 安全计算之前应用份数缩放。缺失数量绝不静默臆造 ——
    # 它们保持原样（数量仍存在），缺口在上游保留。
    if target_servings is not None:
        ingredients = _scale_ingredients(
            ingredients,
            original_servings=Decimal(str(candidate.original_servings)),
            target_servings=target_servings,
        )

    steps = tuple(_build_recipe_step(step, recipe_id) for step in candidate.steps)

    # Collect assumptions from extraction source
    # 从提取来源收集假设
    assumptions = _collect_assumptions(candidate)

    return RecipeIR(
        recipe_id=recipe_id,
        dish_name=candidate.dish_name,
        original_servings=Decimal(str(candidate.original_servings)),
        target_servings=target_servings or Decimal(str(candidate.original_servings)),
        source_language=candidate.source_language,
        ingredients=ingredients,
        steps=steps,
        assumptions=assumptions,
    )


def validate_recipe_ir_semantics(recipes: tuple[RecipeIR, ...]) -> RecipeValidationReport:
    """对一个或多个 RecipeIR 对象运行语义校验。

    Run semantic validation on one or more RecipeIR objects.

    Checks that go beyond Pydantic field validation:
      - At least one ingredient per recipe
      - At least one step per recipe
      - Duration values are non-negative
      - Heat level is not NONE for heating-category steps
      - Ingredient names are not empty

    超出 Pydantic 字段校验的检查：
      - 每个菜谱至少一个食材
      - 每个菜谱至少一个步骤
      - 时长值非负
      - 加热类步骤的火力档位不是 NONE
      - 食材名非空

    Args:
        recipes: One or more RecipeIR objects to validate.
            recipes：要校验的一个或多个 RecipeIR。

    Returns:
        RecipeValidationReport with passed flag and issue list.
        含 passed 标志与问题列表的 RecipeValidationReport。
    """
    issues: list[SemanticIssue] = []

    for recipe in recipes:
        issues.extend(_validate_single_recipe(recipe))

    errors = [i for i in issues if i.severity == "error"]
    passed = len(errors) == 0

    return RecipeValidationReport(
        passed=passed,
        issues=tuple(issues),
        recipe_count=len(recipes),
    )


def attach_research_assumptions(
    recipes: tuple[RecipeIR, ...],
    research_assumptions: tuple[Assumption, ...],
) -> tuple[RecipeIR, ...]:
    """把有证据支撑的研究假设合并进每个 RecipeIR（P1-01）。

    Merge evidence-backed research assumptions into each RecipeIR (P1-01).

    Research provenance must be traceable in the final assumption/response:
    each applied evidence value produces an Assumption carrying EvidenceRef
    entries (source title + URL). This helper attaches them to every recipe
    so rendering surfaces them verbatim.

    研究溯源必须在最终假设 / 响应中可追溯：每个应用的证据值产生一个携带 EvidenceRef
    （来源标题 + URL）的 Assumption。本辅助函数把它们附加到每个菜谱，使渲染能原样呈现。

    Args:
        recipes: RecipeIR objects to enrich.
            recipes：要丰富的 RecipeIR 对象。
        research_assumptions: Assumptions produced by the research evidence
            application node. Empty tuple is a no-op.
            research_assumptions：研究证据应用节点产生的假设。空元组为 no-op。

    Returns:
        New RecipeIR tuple with the research assumptions appended (never
        mutates the input recipes).
        已追加研究假设的新 RecipeIR 元组（绝不修改输入菜谱）。
    """
    if not research_assumptions:
        return recipes
    return tuple(
        recipe.model_copy(update={"assumptions": recipe.assumptions + research_assumptions}) for recipe in recipes
    )


# =============================================================================
# Internal builders
# 内部构建器
# =============================================================================


def _build_ingredient_demand(ingredient: ExtractedIngredient) -> IngredientDemand | None:
    """把 ExtractedIngredient 转换为 IngredientDemand。

    Convert ExtractedIngredient → IngredientDemand.

    Returns None if the ingredient lacks a meaningful name.

    若食材缺少有意义名称则返回 None。
    """
    if not ingredient.name or len(ingredient.name.strip()) < 1:
        return None

    # Normalise unit string  规范化单位字符串
    unit = _normalise_ingredient_unit(ingredient)

    # Detect allergen tags from ingredient name  从食材名检测过敏原标签
    allergen_tags = _detect_allergens(ingredient.name)

    return IngredientDemand(
        canonical_name=ingredient.name.strip(),
        raw_name=ingredient.raw_text,
        quantity=ingredient.quantity or Decimal(1),
        unit=unit,
        preparation_spec=ingredient.preparation,
        input_state="raw",
        allergen_tags=allergen_tags,
        confidence=ingredient.confidence,
    )


def _build_recipe_step(step: ExtractedStep, recipe_id: str) -> RecipeStep:
    """把 ExtractedStep 转换为 RecipeStep（含模式推断）。"""
    # Infer decomposition pattern from category and technique
    # 从类别与技法推断分解模式
    pattern = _infer_pattern(step)

    return RecipeStep(
        step_number=step.step_number,
        instruction=step.instruction,
        category=step.category,
        pattern=pattern,
        active_duration_minutes=step.active_duration_minutes,
        passive_duration_minutes=step.passive_duration_minutes,
        heat_level=step.heat_level,
        target_temperature_c=step.target_temperature_c,
        resources_hint=step.resources_hint,
    )


# =============================================================================
# P0-04 serving scaling helpers
# P0-04 份数缩放辅助函数
# =============================================================================

# Discrete units that must be rounded to whole items.  For these, scaled
# quantities are rounded UP so a plan never under-supplies (e.g. 1.2 eggs
# becomes 2 eggs) and an assumption is recorded (P0-04 rule 3).
# 必须取整到整件的离散单位。对这些，缩放后的数量向上取整，使计划绝不供应不足
# （如 1.2 个鸡蛋变为 2 个鸡蛋），并记录假设（P0-04 规则 3）。
_DISCRETE_UNITS = frozenset(
    {
        "piece",
        "pc",
        "pcs",
        "egg",
        "eggs",
        "clove",
        "cloves",
        "root",
        "roots",
        "head",
        "heads",
        "slice",
        "slices",
        "bunch",
        "bunches",
    }
)


def _is_discrete_unit(unit: str) -> bool:
    """当食材单位是“计数”而非“计量”时返回 True。"""
    return unit.strip().lower() in _DISCRETE_UNITS


def _scale_ingredients(
    ingredients: tuple[IngredientDemand, ...],
    *,
    original_servings: Decimal,
    target_servings: Decimal,
) -> tuple[IngredientDemand, ...]:
    """把每个食材从原始份数缩放到目标份数（P0-04）。

    Scale every ingredient from original to target servings (P0-04).

    Continuous units scale exactly via Decimal multiplication. Discrete
    units round UP to the nearest whole item; rounding decisions are
    attached as assumptions so they surface for user confirmation.

    连续单位用 Decimal 乘法精确缩放。离散单位向上取整到最近整件；
    取整决策作为假设附加，使它们呈现给用户确认。

    Args:
        ingredients: Demands to scale.
            ingredients：要缩放的食材需求。
        original_servings: Servings the recipe was written for.
            original_servings：菜谱原本的份数。
        target_servings: Desired servings.
            target_servings：期望份数。

    Returns:
        A new tuple of scaled IngredientDemand instances. Never mutates
        the input demands.
        缩放后 IngredientDemand 实例的新元组。绝不修改输入。
    """
    from cooking_plan_agent.normalisation.units import scale_ingredient

    scaled: list[IngredientDemand] = []
    for demand in ingredients:
        new_demand = scale_ingredient(
            demand,
            original_servings=original_servings,
            target_servings=target_servings,
        )
        if _is_discrete_unit(new_demand.unit):
            import math

            rounded = Decimal(math.ceil(new_demand.quantity))
            if rounded != new_demand.quantity:
                new_demand = new_demand.model_copy(update={"quantity": rounded})
        scaled.append(new_demand)
    return tuple(scaled)


# =============================================================================
# Validation helpers
# 校验辅助函数
# =============================================================================


def _validate_single_recipe(recipe: RecipeIR) -> list[SemanticIssue]:
    """校验单个 RecipeIR 的语义正确性。"""
    issues: list[SemanticIssue] = []

    # Check: at least one ingredient  检查：至少一个食材
    if not recipe.ingredients:
        issues.append(
            SemanticIssue(
                code="NO_INGREDIENTS",
                severity="error",
                message=f"Recipe '{recipe.dish_name}' has no ingredients",
            )
        )

    # Check: at least one step  检查：至少一个步骤
    if not recipe.steps:
        issues.append(
            SemanticIssue(
                code="NO_STEPS",
                severity="error",
                message=f"Recipe '{recipe.dish_name}' has no steps",
            )
        )

    # Check: no negative durations  检查：无负时长
    for step in recipe.steps:
        if step.active_duration_minutes is not None and step.active_duration_minutes <= 0:
            issues.append(
                SemanticIssue(
                    code="NEGATIVE_DURATION",
                    severity="error",
                    message=f"Recipe '{recipe.dish_name}' step {step.step_number}: "
                    f"active duration is {step.active_duration_minutes}",
                )
            )
        if step.passive_duration_minutes is not None and step.passive_duration_minutes <= 0:
            issues.append(
                SemanticIssue(
                    code="NEGATIVE_DURATION",
                    severity="error",
                    message=f"Recipe '{recipe.dish_name}' step {step.step_number}: "
                    f"passive duration is {step.passive_duration_minutes}",
                )
            )

    # Check: heating steps should have a heat level  检查：加热步骤应有火力档位
    for step in recipe.steps:
        if step.category == "heating" and step.heat_level == HeatLevel.NONE:
            issues.append(
                SemanticIssue(
                    code="MISSING_HEAT_LEVEL",
                    severity="warning",
                    message=f"Recipe '{recipe.dish_name}' step {step.step_number}: "
                    f"heating step has no heat level specified",
                )
            )

    # Check: ingredient names are non-empty  检查：食材名非空
    for i, ingredient in enumerate(recipe.ingredients):
        if not ingredient.canonical_name.strip():
            issues.append(
                SemanticIssue(
                    code="EMPTY_INGREDIENT_NAME",
                    severity="error",
                    message=f"Recipe '{recipe.dish_name}' ingredient {i + 1}: empty name",
                )
            )

    # Check: servings are positive  检查：份数为正
    if recipe.original_servings <= 0:
        issues.append(
            SemanticIssue(
                code="INVALID_SERVINGS",
                severity="error",
                message=f"Recipe '{recipe.dish_name}': servings must be > 0, got {recipe.original_servings}",
            )
        )

    return issues


# =============================================================================
# Internal helpers
# 内部辅助函数
# =============================================================================


def _normalise_ingredient_unit(ingredient: ExtractedIngredient) -> str:
    """把食材单位规范化为规范形式。"""
    if not ingredient.unit:
        return "piece"

    unit = ingredient.unit.lower().strip()

    # Try to classify — if unknown, default to "piece"
    # 尝试分类 —— 若未知，默认 "piece"
    try:
        UnitClassifier.classify(unit)
        return unit
    except (ValueError, KeyError):
        pass

    # Common normalisations  常见规范化
    alias_map = {
        "tablespoon": "tbsp",
        "tablespoons": "tbsp",
        "teaspoon": "tsp",
        "teaspoons": "tsp",
        "cup": "cup",
        "cups": "cup",
        "gram": "g",
        "grams": "g",
        "kilogram": "kg",
        "kilograms": "kg",
        "milliliter": "ml",
        "milliliters": "ml",
        "litre": "l",
        "liter": "l",
        "ounce": "oz",
        "ounces": "oz",
        "pound": "lb",
        "pounds": "lb",
        "cloves": "piece",
        "clove": "piece",
    }
    return alias_map.get(unit, unit)


def _detect_allergens(name: str) -> tuple[str, ...]:
    """从食材名检测常见过敏原。"""
    name_lower = name.lower()
    allergens: list[str] = []

    allergen_map = {
        "gluten": ("wheat", "flour", "bread", "pasta", "noodle", "soy sauce", "面粉", "面条", "面包"),
        "dairy": ("milk", "cheese", "butter", "cream", "yogurt", "牛奶", "奶油", "奶酪", "黄油"),
        "egg": ("egg", "鸡蛋", "蛋"),
        "shellfish": ("shrimp", "prawn", "crab", "lobster", "虾", "蟹", "龙虾"),
        "fish": ("fish", "salmon", "tuna", "cod", "鱼", "三文鱼", "金枪鱼"),
        "soy": ("soy", "tofu", "soybean", "豆腐", "大豆"),
        "nut": ("peanut", "almond", "walnut", "cashew", "花生", "杏仁", "核桃"),
        "sesame": ("sesame", "芝麻"),
    }

    for allergen, keywords in allergen_map.items():
        if any(kw in name_lower for kw in keywords):
            allergens.append(allergen)

    return tuple(allergens)


def _infer_pattern(step: ExtractedStep) -> str:
    """从步骤类别与指令文本推断分解模式。

    Infer the decomposition pattern from step category and instruction text.

    The pattern drives the decomposition policy in preparation/decompose.py.

    该模式驱动 preparation/decompose.py 中的分解策略。
    """
    instruction_lower = step.instruction.lower()

    # Boil detection  煮检测
    if any(kw in instruction_lower for kw in ("boil", "煮", "焯")):
        return "boil"

    # Stir-fry detection  爆炒检测
    if any(kw in instruction_lower for kw in ("stir-fry", "stir fry", "炒", "爆炒", "翻炒")):
        return "stir_fry"

    # Pan-frying is an active stove-and-pan operation. Check it before the
    # marinade keywords below: "将腌好的鸡翅下锅煎制" describes frying, not a
    # new marination step.
    # 煎是主动的灶台 + 锅操作。在下方“腌”关键词之前检查它：
    # "将腌好的鸡翅下锅煎制" 描述的是煎，而非新的腌制步骤。
    if any(kw in instruction_lower for kw in ("pan-fry", "pan fry", "煎")):
        return "stir_fry"

    # Bake detection  烘焙检测
    if any(kw in instruction_lower for kw in ("bake", "oven", "烤", "烘烤", "烤箱")):
        return "bake"

    # Simmer detection  炖煮检测
    if any(kw in instruction_lower for kw in ("simmer", "stew", "焖", "炖", "煲", "慢炖")):
        return "simmer"

    # Marinate detection  腌制检测
    if any(kw in instruction_lower for kw in ("marinate", "腌制", "腌")):
        return "marinate"

    return "simple"


def _collect_assumptions(candidate: ExtractedRecipeCandidate) -> tuple[Assumption, ...]:
    """从提取过程收集假设。

    Collect assumptions from extraction process.

    When rule-based extraction makes guesses (e.g. default 2 servings),
    those become assumptions that may surface to the user.

    当基于规则的提取做出猜测（如默认 2 份）时，这些变为可能呈现给用户的假设。
    """
    assumptions: list[Assumption] = []

    # Rule-based extraction inherently carries assumptions
    # 基于规则的提取本身就带有假设
    if candidate.extraction_source == "RULE_BASED":
        assumptions.append(
            Assumption(
                text="Recipe extracted using rule-based parser (no LLM). "
                "Confidence may be lower than LLM-based extraction.",
                confidence=Decimal("0.8"),
            )
        )

    if "original_servings" in candidate.inferred_fields:
        assumptions.append(
            Assumption(
                text=f"LLM inferred the recipe serves {candidate.original_servings} from culinary context",
                confidence=Decimal("0.8"),
            )
        )

    for index, ingredient in enumerate(candidate.ingredients, start=1):
        if ingredient.extraction_source == "LLM_INFERRED":
            assumptions.append(
                Assumption(
                    text=f"LLM completed missing details for ingredient {index} ({ingredient.name})",
                    confidence=ingredient.confidence,
                )
            )

    for step in candidate.steps:
        if step.extraction_source in {"LLM_INFERRED", "RULE_INFERRED"}:
            source = "LLM" if step.extraction_source == "LLM_INFERRED" else "fallback rules"
            assumptions.append(
                Assumption(
                    text=f"{source} completed missing operational details for step {step.step_number}",
                    confidence=step.confidence,
                )
            )

    return tuple(assumptions)
