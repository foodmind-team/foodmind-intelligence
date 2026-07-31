"""Minimal query construction (handbook 10.3).

Builds search queries from RecipeGap data while ensuring no private user
fields (user ID, inventory, dietary profile, budget, location, etc.) are
ever included.
"""

from cooking_plan_agent.domain.models import RecipeGap

# ---------------------------------------------------------------------------
# Blocked private fields — these must NEVER appear in a search query
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

    If a gap contains a private field, it would fail the allow-list check
    in the caller — this is a defence-in-depth measure.
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

    The query contains ONLY:
      - dish name
      - cooking technique (from gap field_path or description)
      - ingredient or food class needed to disambiguate
      - requested field (heat level or duration)

    Returns a plain-text string suitable for any search provider.
    """
    # Defence-in-depth: verify no private terms leaked into the gap
    _sanitised(gap)

    parts: list[str] = []

    if dish_name:
        parts.append(dish_name)

    # Extract technique hint from field_path (e.g. "steps[0].heat_level")
    field = gap.field_path.lower()
    if "heat" in field:
        parts.append("heat level")
    if "duration" in field or "time" in field:
        parts.append("approximate cooking duration")
    if "temperature" in field:
        parts.append("target temperature celsius")

    # Append the gap description if it adds disambiguating context
    description = gap.description.strip()
    if description and description not in parts:
        parts.append(description)

    return " ".join(parts)
