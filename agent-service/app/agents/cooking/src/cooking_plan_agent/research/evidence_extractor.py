"""Evidence extraction (handbook 10.5, 10.6).

Extracts CookingEvidence from a SearchDocument using a narrow-schema,
rule-based approach. In the MVP, this is rule-based (no LLM dependency).
When an LLM is wired later, the extractor becomes the prompt boundary
where hostile text defence applies.
"""

import re
from decimal import Decimal

from cooking_plan_agent.domain.enums import HeatLevel
from cooking_plan_agent.domain.models import CookingEvidence, SearchDocument
from cooking_plan_agent.research.text_sanitizer import sanitize_document_content

# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

# Heat level keyword mapping (case-insensitive).
# standalone "high"/"low" are excluded — they must co-occur with "heat", "flame",
# or "temperature" to avoid false matches on weather/altitude/etc.
_HEAT_MAP: dict[str, HeatLevel] = {
    "high heat": HeatLevel.HIGH,
    "high flame": HeatLevel.HIGH,
    "medium-high heat": HeatLevel.MEDIUM,
    "medium heat": HeatLevel.MEDIUM,
    "medium-low heat": HeatLevel.LOW,
    "low heat": HeatLevel.LOW,
    "low flame": HeatLevel.LOW,
}

# Duration patterns: "X minutes", "X-Y minutes", "about X min", etc.
_DURATION_RANGE_RE = re.compile(r"(\d+)\s*[-–—to]+\s*(\d+)\s*(minutes?|mins?)\b", re.IGNORECASE)
_DURATION_SINGLE_RE = re.compile(
    r"(?:about\s+|approximately\s+|around\s+)?(\d+)\s*(minutes?|mins?)\b",
    re.IGNORECASE,
)

# Temperature patterns: "200°C", "200 C", "200 Celsius", "400°F"
_TEMP_C_RE = re.compile(r"(\d{2,3})\s*(?:°\s*)?[cC](?:elsius)?\b")
_TEMP_F_RE = re.compile(r"(\d{2,4})\s*(?:°\s*)?[fF](?:ahrenheit)?\b")

# Operation keyword detection from snippet
_OPERATION_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("stir-fry", "stir-fry"),
    ("stir fry", "stir-fry"),
    ("deep-fry", "deep-fry"),
    ("deep fry", "deep-fry"),
    ("roast", "roast"),
    ("bake", "bake"),
    ("grill", "grill"),
    ("broil", "broil"),
    ("boil", "boil"),
    ("simmer", "simmer"),
    ("steam", "steam"),
    ("braise", "braise"),
    ("sauté", "sauté"),
    ("saute", "sauté"),
    ("pan-fry", "pan-fry"),
    ("pan fry", "pan-fry"),
    ("sear", "sear"),
    ("poach", "poach"),
)


def _extract_operation(text: str, dish_name: str = "") -> str:
    """Detect cooking operation from text using keyword matching."""
    text_lower = text.lower()
    for keyword, operation in _OPERATION_KEYWORDS:
        if keyword in text_lower:
            return operation
    # Fallback: use dish name if it contains a technique hint
    if dish_name:
        dish_lower = dish_name.lower()
        for keyword, operation in _OPERATION_KEYWORDS:
            if keyword in dish_lower:
                return operation
    return "cook"  # Generic fallback


def _extract_heat(text: str) -> HeatLevel | None:
    """Extract heat level from text."""
    text_lower = text.lower()
    # Check longer phrases first (e.g. "medium-high" before "medium")
    for phrase, level in sorted(_HEAT_MAP.items(), key=lambda x: -len(x[0])):
        if phrase in text_lower:
            return level
    return None


def _extract_duration(text: str) -> tuple[int | None, int | None]:
    """Extract duration range in minutes from text.

    Returns (min_minutes, max_minutes). Both None if no duration found.
    """
    # Try range first: "10-15 minutes"
    match = _DURATION_RANGE_RE.search(text)
    if match:
        lo, hi = int(match.group(1)), int(match.group(2))
        return lo, hi

    # Try single: "about 10 minutes"
    match = _DURATION_SINGLE_RE.search(text)
    if match:
        minutes = int(match.group(1))
        return minutes, minutes

    return None, None


def _extract_temperature(text: str) -> Decimal | None:
    """Extract temperature in Celsius. Converts Fahrenheit to Celsius."""
    match = _TEMP_C_RE.search(text)
    if match:
        return Decimal(match.group(1))

    match = _TEMP_F_RE.search(text)
    if match:
        f = Decimal(match.group(1))
        # Fahrenheit to Celsius: (F - 32) * 5/9
        return ((f - 32) * 5 / 9).quantize(Decimal("0.1"))

    return None


def _excerpt(text: str, max_chars: int = 200) -> str:
    """Return shortest traceable excerpt — not the full page (handbook 10.6)."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_evidence(
    document: SearchDocument,
    dish_name: str = "",
) -> CookingEvidence | None:
    """Extract CookingEvidence from a single search document.

    Returns None if no cooking-relevant information is found in the document.
    The extractor is rule-based in MVP — narrow schema, no LLM, no hallucination risk.

    Hostile text defence: content is sanitized before extraction.
    If injection patterns are detected, extraction proceeds but is stricter
    (no free-form inference from the text).
    """
    # Sanitize content at the boundary (handbook 10.5)
    text_to_search = document.snippet
    if document.raw_content:
        cleaned, _ = sanitize_document_content(document.raw_content)
        text_to_search = f"{document.snippet}\n{cleaned}"

    operation = _extract_operation(text_to_search, dish_name)
    heat = _extract_heat(text_to_search)
    dur_min, dur_max = _extract_duration(text_to_search)
    temp = _extract_temperature(text_to_search)

    # If no cooking-relevant data found, skip this document
    if heat is None and dur_min is None and temp is None:
        return None

    return CookingEvidence(
        operation=operation,
        heat_level=heat,
        duration_min_minutes=dur_min,
        duration_max_minutes=dur_max,
        explicit_temperature_c=temp,
        source_url=document.url,
        source_title=document.title,
        source_excerpt=_excerpt(document.snippet),
    )
