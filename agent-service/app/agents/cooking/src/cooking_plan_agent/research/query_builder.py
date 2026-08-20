# =============================================================================
# 最小查询构建模块（research/query_builder）
# -----------------------------------------------------------------------------
# 根据 RecipeGap 数据构建搜索查询，同时确保任何私有用户字段
# （用户 ID、库存、饮食偏好、预算、位置等）都不会被包含进去。
# =============================================================================

"""Minimal query construction (handbook 10.3).

最小查询构建（手册 10.3）。

Builds search queries from RecipeGap data while ensuring no private user
fields (user ID, inventory, dietary profile, budget, location, etc.) are
ever included.

根据 RecipeGap 数据构建搜索查询，同时确保任何私有用户字段
（用户 ID、库存、饮食偏好、预算、位置等）都不会被包含进去。
"""

from cooking_plan_agent.domain.models import RecipeGap

# ---------------------------------------------------------------------------
# Blocked private fields — these must NEVER appear in a search query
# 被屏蔽的私有字段 —— 这些字段绝不应当出现在搜索查询中
# ---------------------------------------------------------------------------

_BLOCKED_TERMS: frozenset[str] = frozenset(
    {
        "user",
        "user_id",
        "inventory",
        "allergen",
        "allergy",
        "dietary",
        "budget",
        "location",
        "address",
        "group",
        "family",
        "profile",
        "comment",
        "password",
        "token",
    }
)


def _sanitised(gap: RecipeGap) -> str:
    """Return the gap description stripped of any blocked terms.

    返回去除任何被屏蔽字段后的缺口描述。

    If a gap contains a private field, it would fail the allow-list check
    in the caller — this is a defence-in-depth measure.

    若缺口包含私有字段，它本应无法通过调用方的白名单校验 ——
    此处为纵深防御措施。
    """
    desc = gap.description.lower()
    for term in _BLOCKED_TERMS:
        if term in desc:
            raise ValueError(f"Query blocked: gap description contains private term '{term}'")
    return gap.description


def build_minimal_query(
    gap: RecipeGap,
    dish_name: str = "",
) -> str:
    """Construct a generic, minimal search query (handbook 10.3).

    构造一个通用、最小的搜索查询（手册 10.3）。

    The query contains ONLY:
      - dish name
      - cooking technique (from gap field_path or description)
      - ingredient or food class needed to disambiguate
      - requested field (heat level or duration)

    查询仅包含：
      - 菜名
      - 烹饪技法（来自 gap 的 field_path 或描述）
      - 用于消歧的食材或食物类别
      - 请求的字段（火力等级或时长）

    Returns a plain-text string suitable for any search provider.

    返回适用于任何搜索提供方的纯文本字符串。
    """
    # Defence-in-depth: verify no private terms leaked into the gap
    # 纵深防御：校验没有私有字段泄漏进缺口
    _sanitised(gap)

    parts: list[str] = []

    if dish_name:
        parts.append(dish_name)

    # Extract technique hint from field_path (e.g. "steps[0].heat_level")
    # 从 field_path 提取技法提示（例如 "steps[0].heat_level"）
    field = gap.field_path.lower()
    if "heat" in field:
        parts.append("heat level")
    if "duration" in field or "time" in field:
        parts.append("approximate cooking duration")
    if "temperature" in field:
        parts.append("target temperature celsius")

    # Append the gap description if it adds disambiguating context
    # 若缺口描述能提供消歧上下文，则追加
    description = gap.description.strip()
    if description and description not in parts:
        parts.append(description)

    return " ".join(parts)
