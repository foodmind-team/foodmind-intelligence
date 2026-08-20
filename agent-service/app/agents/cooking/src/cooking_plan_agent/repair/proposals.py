# =============================================================================
# 纯修复选项提议与排序服务模块（repair/proposals）
# -----------------------------------------------------------------------------
# 实现修复选项的提议（propose_*）、精确短缺计算与排序（rank_*）等纯函数。
# =============================================================================

"""Pure repair-option proposal and ranking services.

纯修复选项提议与排序服务。
"""

from __future__ import annotations

import hashlib
import re
from decimal import ROUND_FLOOR, Decimal
from uuid import uuid4

from cooking_plan_agent.domain.models import FeasibilityReport, IngredientFeasibility, RepairOption
from cooking_plan_agent.repair.catalogs import _EQUIPMENT_ALTERNATIVES, _INGREDIENT_SUBSTITUTIONS
from cooking_plan_agent.repair.models import RepairValidation, Shortage

_OPTION_ID_MAX_LENGTH = 128
_OPTION_ID_HASH_LENGTH = 12
_OPTION_ID_RANDOM_LENGTH = 6
_OPTION_ID_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


def _bounded_option_id(prefix: str, raw_label: str) -> str:
    """Build an ASCII option ID that always fits the public persistence contract.

    构建一个始终符合公共持久化契约的 ASCII 选项 ID。

    User-provided ingredient and equipment names belong in the option payload,
    not unbounded protocol identifiers. Keep a short readable slug, plus a
    content hash and random suffix so separately generated options remain
    collision-resistant without exceeding the Backend's 128-character ID cap.

    用户提供的食材与设备名称应放在选项负载中，而不是无界的协议标识符。
    保留一个简短可读的 slug，并加上内容哈希和随机后缀，使分别生成的选项
    保持抗碰撞，且不超过 Backend 的 128 字符 ID 上限。
    """

    digest = hashlib.sha256(raw_label.encode("utf-8")).hexdigest()[:_OPTION_ID_HASH_LENGTH]
    suffix = uuid4().hex[:_OPTION_ID_RANDOM_LENGTH]
    slug = _OPTION_ID_SLUG_PATTERN.sub("_", raw_label.casefold()).strip("_") or "item"
    reserved = len(prefix) + len(digest) + len(suffix) + 3
    slug = slug[: max(1, _OPTION_ID_MAX_LENGTH - reserved)].rstrip("_") or "item"
    return f"{prefix}_{slug}_{digest}_{suffix}"


def calculate_exact_shortages(
    report: FeasibilityReport,
) -> tuple[Shortage, ...]:
    """Extract exact shortage items from a FeasibilityReport.

    从 FeasibilityReport 中提取精确的短缺项。

    Covers both ingredient shortages (from ingredient_shortages) and
    resource shortages (from missing_resources). Resource shortages
    are represented with required=1, available=0.

    同时覆盖食材短缺（来自 ingredient_shortages）和资源短缺
    （来自 missing_resources）。资源短缺以 required=1、available=0 表示。

    Args:
        report: A FeasibilityReport from check_all_inventory / check_feasibility.
            来自 check_all_inventory / check_feasibility 的 FeasibilityReport。

    Returns:
        Tuple of Shortage items. Empty if the report is feasible.
            Shortage 项元组；若报告可行则为空。
    """
    shortages: list[Shortage] = []

    for item in report.ingredient_shortages:
        if item.shortage > 0:
            shortages.append(
                Shortage(
                    item=item.ingredient_name,
                    required=item.required,
                    available=item.available,
                    unit=item.unit,
                )
            )

    for resource_type in report.missing_resources:
        shortages.append(
            Shortage(
                item=resource_type,
                required=Decimal(1),
                available=Decimal(0),
                unit="unit",
            )
        )

    return tuple(shortages)


def _json_number(value: Decimal) -> int | float:
    """Return a JSON-native number without losing integral quantities.

    返回 JSON 原生数字，且不丢失整数数量。
    """
    if value == value.to_integral_value():
        return int(value)
    return float(value)


# =============================================================================
# 5.18  Ingredient substitution proposal
#      食材替代提议
# =============================================================================


def propose_ingredient_substitutions(
    shortages: tuple[IngredientFeasibility, ...],
) -> tuple[RepairOption, ...]:
    """Build one purchase item for every ingredient shortage.

    The public name is retained for compatibility with existing imports, but
    ingredient substitution is intentionally not offered in the Cooking Plan
    shortage flow. The only inventory strategies are purchase and (when the
    request is above one serving) portion reduction.

    Args:
        shortages: Ingredient shortages from FeasibilityReport.

    Returns:
        Exactly one purchase RepairOption per shortage.
    """
    options: list[RepairOption] = []

    for shortage in shortages:
        options.append(
            RepairOption(
                option_id=_bounded_option_id("repair_purchase", shortage.ingredient_name),
                option_type="purchase",
                description=(f"Purchase {shortage.shortage} {shortage.unit} of '{shortage.ingredient_name}'"),
                changes=(f"Add {shortage.shortage} {shortage.unit} of {shortage.ingredient_name} to shopping list",),
                effects=("Add the purchased quantity to real inventory before revalidation",),
                payload={
                    "ingredient_name": shortage.ingredient_name,
                    "quantity": _json_number(shortage.shortage),
                    "unit": shortage.unit,
                },
                revalidation_status="validated",
            )
        )

    return tuple(options)


# =============================================================================
# 5.19  Portion adjustment proposal
#      份量调整提议
# =============================================================================


def _fmt_servings(value: Decimal) -> str:
    """份量显示：整数值不带小数点，小数保留（保证与决策正则可解析）。"""
    if value == value.to_integral_value():
        return str(int(value))
    return format(value, "f")


def propose_portion_adjustments(
    shortages: tuple[IngredientFeasibility, ...],
    original_servings: Decimal | int = 2,
) -> tuple[RepairOption, ...]:
    """Propose reducing target servings to match available ingredients.

    提议缩减目标份量以匹配可用食材。

    Calculates the maximum feasible servings as:
        new_servings = floor(original * min(available/required for each ingredient))

    按如下公式计算最大可行份量：
        new_servings = floor(original * min(每种食材的 available/required))

    If all shortages are minor (≤ 50%), proposes a specific serving reduction.
    If any shortage is > 50%, proposes reducing to single-serving.

    若所有短缺都较小（≤ 50%），提议具体的份量缩减。
    若任一短缺 > 50%，则提议缩减为单份。

    Args:
        shortages: Ingredient shortages from FeasibilityReport.
            来自 FeasibilityReport 的食材短缺。
        original_servings: The requested serving count (per-recipe target
            servings). Accepts Decimal so callers pass the user's actual
            serving size instead of a fixed default.
            请求的份数（每个菜谱的目标份量）。接受 Decimal，以便调用方
            传入用户实际份量，而非固定默认值。

    Returns:
        RepairOption if a reduction is meaningful, empty tuple otherwise.
            若缩减有意义则返回 RepairOption，否则返回空元组。
    """
    if not shortages or original_servings <= 1:
        return ()

    original = Decimal(str(original_servings))

    # Find the limiting ingredient: min(available / required)
    # 找到限制性食材：min(可用 / 所需)
    min_ratio = Decimal(1)
    for s in shortages:
        if s.required > 0:
            ratio = s.available / s.required
            min_ratio = min(min_ratio, ratio)

    if min_ratio >= Decimal(1):
        return ()  # No reduction needed (should not happen with shortages)
        # ↑ 无需缩减（在存在短缺时不应发生）

    new_servings = max(Decimal(1), (original * min_ratio).to_integral_value(rounding=ROUND_FLOOR))

    if new_servings >= original:
        return ()

    description = (
        f"Reduce servings from {_fmt_servings(original)} to {_fmt_servings(new_servings)} "
        f"(available ingredients support ~{min_ratio:.0%} of original portions)"
    )

    return (
        RepairOption(
            option_id=f"repair_servings_{_fmt_servings(new_servings)}_{uuid4().hex[:6]}",
            option_type="reduce_servings",
            description=description,
            changes=(
                f"Scale all ingredient quantities to {_fmt_servings(new_servings)} servings "
                f"(was {_fmt_servings(original)})",
            ),
            effects=(f"Recheck inventory for {_fmt_servings(new_servings)} servings before generating the plan",),
            payload={"servings": int(new_servings)},
            revalidation_status="validated",
        ),
    )


# =============================================================================
# 5.20  Equipment alternative proposal
#      设备替代提议
# =============================================================================


def propose_equipment_alternatives(
    missing_resources: tuple[str, ...],
) -> tuple[RepairOption, ...]:
    """Propose alternative equipment for each missing resource type.

    为每个缺失的资源类型提议替代设备。

    Looks up common alternatives from the built-in table. Resources
    without known alternatives are flagged for manual resolution.

    从内置表中查找常见替代。没有已知替代的资源会被标记为需人工解决。

    Args:
        missing_resources: Resource types that are missing/unavailable.
            缺失/不可用的资源类型。

    Returns:
        One or more RepairOption per missing resource.
            每个缺失资源一个或多个 RepairOption。
    """
    options: list[RepairOption] = []

    for resource in missing_resources:
        # Strip capability suffix if present (e.g. "stove:induction" → "stove")
        # 若存在能力后缀则去除（例如 "stove:induction" → "stove"）
        base_resource = resource.split(":")[0].lower().strip()
        alts = _EQUIPMENT_ALTERNATIVES.get(base_resource)

        if not alts:
            options.append(
                RepairOption(
                    option_id=_bounded_option_id("repair_noalt", resource),
                    option_type="alternative_equipment",
                    description=(f"No known alternative for '{resource}'. Manual resolution required."),
                    changes=(f"Source or improvise alternative for {resource}",),
                    effects=("Requires manual equipment sourcing",),
                    revalidation_status="validated",
                )
            )
            continue

        for alt_name, note in alts:
            options.append(
                RepairOption(
                    option_id=_bounded_option_id("repair_eq", f"{base_resource}:{alt_name}"),
                    option_type="alternative_equipment",
                    description=(f"Use '{alt_name}' instead of '{resource}': {note}"),
                    changes=(f"Replace {resource} with {alt_name}",),
                    effects=(f"Resolves missing '{resource}'. {note}",),
                    revalidation_status="validated",
                )
            )

    return tuple(options)


# =============================================================================
# 5.21  Dish replacement proposal
#      菜品替换提议
# =============================================================================


def propose_dish_replacements(
    shortages: tuple[IngredientFeasibility, ...],
    recipe_names: tuple[str, ...],
) -> tuple[RepairOption, ...]:
    """Propose removing/replacing dishes that have unsolvable ingredient issues.

    MVP strategy: for each shortage that has no known substitute, suggest
    removing the affected dish(es). Since we don't have per-dish shortage
    mapping at this layer, we surface the issue at plan level.

    Args:
        shortages: Ingredient shortages from FeasibilityReport.
        recipe_names: Names of all dishes in the current plan.

    Returns:
        RepairOption suggesting dish removal/review.
    """
    if not shortages:
        return ()

    # Identify ingredients with no known substitutes
    unsubstitutable = [s for s in shortages if s.ingredient_name.lower().strip() not in _INGREDIENT_SUBSTITUTIONS]

    if not unsubstitutable:
        return ()  # All shortages have substitutes available
        # ↑ 所有短缺都有可用替代

    ingredient_list = ", ".join(s.ingredient_name for s in unsubstitutable)
    dish_list = ", ".join(recipe_names) if recipe_names else "the current dishes"

    return (
        RepairOption(
            option_id=f"repair_dish_remove_{uuid4().hex[:8]}",
            option_type="replace_dish",
            description=(
                f"Some ingredients ({ingredient_list}) have no known substitutes "
                f"in {dish_list}. Consider replacing affected dishes or purchasing "
                f"the missing ingredients."
            ),
            changes=(
                f"Review dishes containing: {ingredient_list}",
                "Consider replacing with alternative recipes using available ingredients",
            ),
            effects=(
                f"Eliminates shortages in: {ingredient_list}",
                "May change the meal composition significantly",
            ),
            revalidation_status="validated",
        ),
    )


# =============================================================================
# 5.22  Time extension proposal
#      延长时间提议
# =============================================================================


def propose_time_extension(
    current_time_limit: int | None,
    minimum_required_minutes: int,
) -> RepairOption | None:
    """Propose extending the time limit if the current limit is too tight.

    若当前时间上限过紧，则提议延长时间上限。

    Only proposes extension if the gap is reasonable (≤ 3× current limit).
    Extreme gaps suggest deeper problems that a time extension won't fix.

    仅当缺口合理时才提议延长（≤ 当前上限的 3 倍）。
    极端缺口说明存在更深层问题，延长时间无法解决。

    Args:
        current_time_limit: The user-specified time limit in minutes.
            用户指定的时间上限（分钟）。
        minimum_required_minutes: The minimum feasible makespan.
            最小可行完工时间。

    Returns:
        RepairOption if extension is reasonable, None otherwise.
            若延长合理则返回 RepairOption，否则返回 None。
    """
    if current_time_limit is None:
        return None

    if current_time_limit >= minimum_required_minutes:
        return None

    gap = minimum_required_minutes - current_time_limit

    # Don't propose extreme extensions (> 3× the current limit)
    # 不要提议极端延长（> 当前上限的 3 倍）
    if current_time_limit > 0 and minimum_required_minutes > 3 * current_time_limit:
        return None

    return RepairOption(
        option_id=f"repair_time_{minimum_required_minutes}_{uuid4().hex[:6]}",
        option_type="extend_time",
        description=(
            f"Extend cooking time from {current_time_limit} to {minimum_required_minutes} minutes (adds {gap} minutes)"
        ),
        changes=(f"Increase time limit to {minimum_required_minutes} minutes",),
        effects=(f"All tasks can be scheduled within {minimum_required_minutes} minutes",),
        revalidation_status="validated",
    )


# =============================================================================
# 5.23  Repair option validation
#      修复选项校验
# =============================================================================


def validate_repair_option(
    option: RepairOption,
) -> RepairValidation:
    """Validate that a RepairOption is internally consistent.

    校验 RepairOption 是否内部一致。

    Checks:
      - option_id is non-empty
      - option_type is a recognised type
      - description, changes, and effects are non-empty
      - revalidation_status is 'validated'

    检查项：
      - option_id 非空
      - option_type 是已识别的类型
      - description、changes、effects 非空
      - revalidation_status 为 'validated'

    Args:
        option: A RepairOption to validate.
            待校验的 RepairOption。

    Returns:
        RepairValidation with is_valid=True and empty issues on success.
            成功时返回 is_valid=True 且 issues 为空的 RepairValidation。
    """
    issues: list[str] = []
    valid_types = {
        "substitute_ingredient",
        "reduce_servings",
        "alternative_equipment",
        "replace_dish",
        "extend_time",
        "purchase",
    }

    if not option.option_id.strip():
        issues.append("option_id is empty")

    if option.option_type not in valid_types:
        issues.append(f"Unknown option_type: {option.option_type!r}")

    if not option.description.strip():
        issues.append("description is empty")

    if not option.changes:
        issues.append("changes is empty")

    if not option.effects:
        issues.append("effects is empty")

    if option.revalidation_status != "validated":
        issues.append(f"revalidation_status is {option.revalidation_status!r}, expected 'validated'")

    return RepairValidation(
        is_valid=len(issues) == 0,
        issues=tuple(issues),
    )


# =============================================================================
# 5.24  Rank repair options
#      排序修复选项
# =============================================================================

# Priority ordering: least disruptive options first (handbook 5.24).
# 优先级排序：最不打扰的选项优先（手册 5.24）。
_OPTION_TYPE_PRIORITY: dict[str, int] = {
    "reduce_servings": 1,  # Least disruptive — just scale down
    # ↑ 最不打扰 —— 只是缩减份量
    "alternative_equipment": 2,  # Use what you have differently
    # ↑ 以不同方式利用现有设备
    "substitute_ingredient": 3,  # Swap ingredients
    # ↑ 替换食材
    "extend_time": 4,  # Just wait longer
    # ↑ 只是等待更久
    "replace_dish": 5,  # Change the menu
    # ↑ 更换菜单
    "purchase": 6,  # Most disruptive — go shopping
    # ↑ 最打扰 —— 出门采购
}


def rank_repair_options(
    options: tuple[RepairOption, ...],
) -> tuple[RepairOption, ...]:
    """Rank repair options from least to most disruptive.

    将修复选项从最不打扰到最打扰排序。

    Sorts by option_type priority, then by option_id for determinism.
    Only includes validated options (revalidation_status='validated').

    先按 option_type 优先级排序，再按 option_id 排序以保证确定性。
    仅包含已校验的选项（revalidation_status='validated'）。

    Args:
        options: Unsorted repair options.
            未排序的修复选项。

    Returns:
        Sorted tuple, least disruptive first.
            排序后的元组，最不打扰的在前。
    """
    valid = [o for o in options if o.revalidation_status == "validated"]
    valid.sort(
        key=lambda o: (
            _OPTION_TYPE_PRIORITY.get(o.option_type, 99),
            o.option_id,
        )
    )
    return tuple(valid)


# =============================================================================
# 5.25  Apply approved decisions
#      应用已批准的决策
# =============================================================================
