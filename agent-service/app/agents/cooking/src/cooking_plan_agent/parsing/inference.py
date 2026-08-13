"""Local cooking knowledge inference — handbook 4.11.

Fills detected recipe gaps using deterministic, rule-based cooking knowledge.
No LLM dependency — this is the local inference fallback before web research.

Design principles:
  - Only inference rules, NO provider calls
  - All decisions are deterministic and explainable
  - Confidence scores reflect the reliability of the inference
  - Safety-critical gaps are NEVER inferred with high confidence
"""

from __future__ import annotations

from decimal import Decimal
from typing import NamedTuple

from cooking_plan_agent.domain.enums import HeatLevel
from cooking_plan_agent.domain.models import (
    Assumption,
    ExtractedRecipeCandidate,
    RecipeGap,
)
from cooking_plan_agent.parsing.gaps import GapClass

# =============================================================================
# InferenceResult — output of the local inference step
# =============================================================================


class InferenceResult(NamedTuple):
    """Result of applying local inference to fill recipe gaps.

    Attributes:
        filled_gaps: Gaps that were successfully filled by local rules.
        unresolved_gaps: Critical gaps that local inference could not fill.
        assumptions: Explanatory assumptions made during inference.
    """

    filled_gaps: tuple[RecipeGap, ...]
    unresolved_gaps: tuple[RecipeGap, ...]
    assumptions: tuple[Assumption, ...]


# =============================================================================
# Heat level inference rules
# =============================================================================

# Technique → default heat level mapping
# These are standard culinary conventions, not guesses.
_TECHNIQUE_HEAT_MAP: dict[str, tuple[HeatLevel, str]] = {
    "stir_fry": (HeatLevel.HIGH, "stir-frying typically requires high heat"),
    "deep_fry": (HeatLevel.HIGH, "deep-frying requires high heat for proper oil temperature"),
    "boil": (HeatLevel.HIGH, "boiling water requires high heat to reach 100°C"),
    "simmer": (HeatLevel.LOW, "simmering is done at low heat, just below boiling"),
    "steam": (HeatLevel.HIGH, "steaming requires high heat to produce steam"),
    "bake": (HeatLevel.MEDIUM, "baking typically uses medium heat (180°C/350°F)"),
    "roast": (HeatLevel.HIGH, "roasting typically requires high heat (200-220°C)"),
    "grill": (HeatLevel.HIGH, "grilling requires high heat for proper searing"),
    "sauté": (HeatLevel.MEDIUM, "sautéing is done at medium to medium-high heat"),
    "sear": (HeatLevel.HIGH, "searing requires high heat for Maillard reaction"),
    "braise": (HeatLevel.LOW, "braising is done at low heat for tender results"),
    "poach": (HeatLevel.LOW, "poaching requires low, gentle heat"),
    "heat": (HeatLevel.HIGH, "heating oil/pan typically uses high heat initially"),
}

# =============================================================================
# Duration inference rules
# =============================================================================

# Technique → default duration range in minutes
# These are conservative defaults based on common cooking practice.
_TECHNIQUE_DURATION_MAP: dict[str, tuple[int, int, str]] = {
    "stir_fry": (3, 8, "stir-frying typically takes 3-8 minutes"),
    "deep_fry": (3, 10, "deep-frying typically takes 3-10 minutes depending on food size"),
    "boil": (8, 20, "boiling typically takes 8-20 minutes for most ingredients"),
    "simmer": (20, 60, "simmering typically takes 20-60 minutes"),
    "steam": (8, 15, "steaming typically takes 8-15 minutes"),
    "bake": (20, 45, "baking typically takes 20-45 minutes"),
    "roast": (30, 90, "roasting typically takes 30-90 minutes"),
    "grill": (5, 15, "grilling typically takes 5-15 minutes per side"),
    "sauté": (3, 10, "sautéing typically takes 3-10 minutes"),
    "sear": (2, 5, "searing typically takes 2-5 minutes per side"),
    "braise": (60, 180, "braising is a slow cooking method (1-3 hours)"),
    "poach": (5, 15, "poaching typically takes 5-15 minutes"),
}

# =============================================================================
# Temperature inference rules
# =============================================================================

# Technique → typical target temperature in Celsius
_TECHNIQUE_TEMPERATURE_MAP: dict[str, tuple[Decimal, str]] = {
    "bake": (Decimal(180), "standard baking temperature is 180°C (350°F)"),
    "roast": (Decimal(200), "standard roasting temperature is 200°C (400°F)"),
    "deep_fry": (Decimal(180), "standard deep-frying oil temperature is 180°C"),
    "grill": (Decimal(220), "grilling surface temperature ~220°C"),
}

# =============================================================================
# Resource inference rules
# =============================================================================

# Technique → default required resources
_TECHNIQUE_RESOURCES: dict[str, tuple[str, ...]] = {
    "stir_fry": ("stove", "wok", "spatula"),
    "deep_fry": ("stove", "pot"),
    "boil": ("stove", "pot"),
    "simmer": ("stove", "pot"),
    "steam": ("stove", "steamer"),
    "bake": ("oven",),
    "roast": ("oven",),
    "grill": ("stove", "pan"),
    "sauté": ("stove", "pan", "spatula"),
    "sear": ("stove", "pan"),
    "braise": ("stove", "pot"),
    "poach": ("stove", "pot"),
    "marinate": ("mixing_bowl",),
}

# =============================================================================
# Public API
# =============================================================================


def infer_local(
    candidate: ExtractedRecipeCandidate,
    gaps: tuple[RecipeGap, ...],
) -> InferenceResult:
    """Apply local cooking knowledge to fill detected gaps.

    Processes each gap and attempts to fill it with deterministic rules.
    Critical and safety_critical gaps that can be filled by local knowledge
    are resolved. Safety-critical gaps that CANNOT be reliably inferred
    remain unresolved and must be routed to confirmation or web research.

    Args:
        candidate: The extracted recipe candidate with gaps.
        gaps: Detected gaps from find_recipe_gaps().

    Returns:
        InferenceResult with filled/unresolved gaps and assumptions made.
    """
    filled: list[RecipeGap] = []
    unresolved: list[RecipeGap] = []
    assumptions: list[Assumption] = []

    # Detect the cooking technique for the recipe (from step analysis)
    technique = _detect_primary_technique(candidate)

    for gap in gaps:
        # Only attempt to fill critical and safety_critical gaps
        if gap.gap_class not in (GapClass.CRITICAL, GapClass.SAFETY_CRITICAL):
            unresolved.append(gap)
            continue

        # Try heat inference
        if "heat_level" in gap.field_path:
            result = _infer_heat(gap, candidate, technique)
            if result:
                filled.append(result[0])
                assumptions.append(result[1])
            else:
                unresolved.append(gap)
            continue

        # Try duration inference
        if "duration" in gap.field_path.lower():
            result = _infer_duration(gap, technique)
            if result:
                filled.append(result[0])
                assumptions.append(result[1])
            else:
                unresolved.append(gap)
            continue

        # Try temperature inference
        if "temperature" in gap.field_path.lower():
            if gap.gap_class == GapClass.SAFETY_CRITICAL:
                # NEVER infer safety-critical temperatures locally
                unresolved.append(gap)
                continue
            result = _infer_temperature(gap, technique)
            if result:
                filled.append(result[0])
                assumptions.append(result[1])
            else:
                unresolved.append(gap)
            continue

        # Try resource inference
        if "resource" in gap.field_path.lower():
            result = _infer_resources(gap, technique)
            if result:
                filled.append(result[0])
                assumptions.append(result[1])
            else:
                unresolved.append(gap)
            continue

        # Can't infer this gap type
        unresolved.append(gap)

    return InferenceResult(
        filled_gaps=tuple(filled),
        unresolved_gaps=tuple(unresolved),
        assumptions=tuple(assumptions),
    )


# =============================================================================
# Merge inference results back into the candidate
# =============================================================================


def merge_inference(
    candidate: ExtractedRecipeCandidate,
    result: InferenceResult,
) -> ExtractedRecipeCandidate:
    """Apply filled gaps back into the candidate, producing an updated candidate.

    Only fields that were successfully filled by local inference are applied.
    Unresolved gaps are left as-is — downstream routing handles them.

    Args:
        candidate: The original extracted candidate.
        result: InferenceResult from infer_local().

    Returns:
        Updated ExtractedRecipeCandidate with inferred values applied.
    """
    updated_steps = list(candidate.steps)

    for gap in result.filled_gaps:
        step_idx = _extract_step_index(gap.field_path)
        if step_idx is not None and step_idx < len(updated_steps):
            step = updated_steps[step_idx]

            # Apply the inferred values
            field = gap.field_path.rsplit(".", 1)[-1] if "." in gap.field_path else gap.field_path

            if "heat_level" in field and gap.current_value:
                try:
                    heat = HeatLevel(gap.current_value)
                    updated_steps[step_idx] = step.model_copy(
                        update={
                            "heat_level": heat,
                            "extraction_source": "RULE_INFERRED",
                            "confidence": min(step.confidence, gap.confidence),
                        }
                    )
                except ValueError:
                    pass

            elif "passive_duration" in field and gap.current_value:
                try:
                    minutes = int(gap.current_value)
                    updated_steps[step_idx] = step.model_copy(
                        update={
                            "passive_duration_minutes": minutes,
                            "extraction_source": "RULE_INFERRED",
                            "confidence": min(step.confidence, gap.confidence),
                        }
                    )
                except ValueError:
                    pass

            elif "active_duration" in field and gap.current_value:
                try:
                    minutes = int(gap.current_value)
                    updated_steps[step_idx] = step.model_copy(
                        update={
                            "active_duration_minutes": minutes,
                            "extraction_source": "RULE_INFERRED",
                            "confidence": min(step.confidence, gap.confidence),
                        }
                    )
                except ValueError:
                    pass

            elif "temperature" in field and gap.current_value:
                try:
                    temp = Decimal(gap.current_value)
                    updated_steps[step_idx] = step.model_copy(
                        update={
                            "target_temperature_c": temp,
                            "extraction_source": "RULE_INFERRED",
                            "confidence": min(step.confidence, gap.confidence),
                        }
                    )
                except (ValueError, TypeError):
                    pass

            elif "resource" in field and gap.current_value:
                # Resource values are comma-separated
                resources = tuple(r.strip() for r in gap.current_value.split(",") if r.strip())
                if resources:
                    updated_steps[step_idx] = step.model_copy(
                        update={
                            "resources_hint": resources,
                            "extraction_source": "RULE_INFERRED",
                            "confidence": min(step.confidence, gap.confidence),
                        }
                    )

    return candidate.model_copy(update={"steps": tuple(updated_steps)})


# Technique keyword → technique name mapping for Chinese text
# Used by _detect_primary_technique to bridge extractor output and inference maps.
_CHINESE_TECHNIQUE_KEYWORDS: dict[str, str] = {
    "焯水": "boil",
    "煮": "boil",
    "烧开": "boil",
    "煮沸": "boil",
    "炒": "stir_fry",
    "翻炒": "stir_fry",
    "爆炒": "stir_fry",
    "大火炒": "stir_fry",
    "煸炒": "stir_fry",
    "煎": "sauté",
    "炖": "simmer",
    "焖": "simmer",
    "煲": "simmer",
    "慢炖": "simmer",
    "小火炖": "simmer",
    "蒸": "steam",
    "炸": "deep_fry",
    "油炸": "deep_fry",
    "烤": "bake",
    "烘烤": "bake",
    "烤箱": "bake",
    "腌制": "marinate",
    "腌": "marinate",
    "烧": "braise",
    "红烧": "braise",
}

# Common English spellings whose surface form differs from the canonical
# inference-map key. Keep these aliases next to the language-specific lookup
# so extraction and local gap filling agree on the same technique.
_ENGLISH_TECHNIQUE_ALIASES: dict[str, str] = {
    "pan-fry": "sauté",
    "pan fry": "sauté",
    "pan-fried": "sauté",
    "pan fried": "sauté",
    "saute": "sauté",
    "sauteed": "sauté",
    "sauté": "sauté",
    "sautéed": "sauté",
}


# =============================================================================
# Internal inference helpers
# =============================================================================


def _detect_primary_technique(candidate: ExtractedRecipeCandidate) -> str:
    """Detect the primary cooking technique from the recipe's steps.

    Searches step instruction text for both English and Chinese technique
    keywords. Returns the most frequently mentioned technique, or 'general'
    if none are found.
    """
    heating_steps = [s for s in candidate.steps if s.category == "heating"]
    if not heating_steps:
        return "general"

    from collections import Counter

    technique_counts: Counter[str] = Counter()

    for step in heating_steps:
        instruction = step.instruction.lower()
        # Check English technique names
        for tech in _TECHNIQUE_HEAT_MAP:
            term = tech.replace("_", " ")
            if term in instruction or tech.replace("_", "-") in instruction:
                technique_counts[tech] += 1

        for keyword, tech in _ENGLISH_TECHNIQUE_ALIASES.items():
            if keyword in instruction:
                technique_counts[tech] += 1

        # Check Chinese technique keywords
        for zh_keyword, tech in _CHINESE_TECHNIQUE_KEYWORDS.items():
            if zh_keyword in step.instruction:
                technique_counts[tech] += 1

    if technique_counts:
        return technique_counts.most_common(1)[0][0]

    return "general"


def _infer_heat(
    gap: RecipeGap,
    candidate: ExtractedRecipeCandidate,
    technique: str,
) -> tuple[RecipeGap, Assumption] | None:
    """Infer missing heat level from technique."""
    if technique not in _TECHNIQUE_HEAT_MAP:
        return None

    heat, explanation = _TECHNIQUE_HEAT_MAP[technique]
    confidence = Decimal("0.7")  # Moderate — technique-level inference

    filled_gap = gap.model_copy(
        update={
            "current_value": heat.value,
            "confidence": confidence,
        }
    )
    assumption = Assumption(
        text=f"Assuming {heat.value} heat for {technique}: {explanation}",
        confidence=confidence,
    )
    return filled_gap, assumption


def _infer_duration(
    gap: RecipeGap,
    technique: str,
) -> tuple[RecipeGap, Assumption] | None:
    """Infer missing duration from technique."""
    if technique not in _TECHNIQUE_DURATION_MAP:
        return None

    dur_min, dur_max, explanation = _TECHNIQUE_DURATION_MAP[technique]
    # Use midpoint as the inferred passive duration
    inferred = (dur_min + dur_max) // 2
    confidence = Decimal("0.5")  # Low confidence — duration varies a lot

    filled_gap = gap.model_copy(
        update={
            "current_value": str(inferred),
            "confidence": confidence,
        }
    )
    assumption = Assumption(
        text=f"Assuming ~{inferred} minutes for {technique}: {explanation}",
        confidence=confidence,
    )
    return filled_gap, assumption


def _infer_temperature(
    gap: RecipeGap,
    technique: str,
) -> tuple[RecipeGap, Assumption] | None:
    """Infer missing temperature from technique."""
    if technique not in _TECHNIQUE_TEMPERATURE_MAP:
        return None

    temp, explanation = _TECHNIQUE_TEMPERATURE_MAP[technique]
    confidence = Decimal("0.6")

    filled_gap = gap.model_copy(
        update={
            "current_value": str(temp),
            "confidence": confidence,
        }
    )
    assumption = Assumption(
        text=f"Assuming {temp}°C for {technique}: {explanation}",
        confidence=confidence,
    )
    return filled_gap, assumption


def _infer_resources(
    gap: RecipeGap,
    technique: str,
) -> tuple[RecipeGap, Assumption] | None:
    """Infer missing resource hints from technique."""
    if technique not in _TECHNIQUE_RESOURCES:
        return None

    resources = _TECHNIQUE_RESOURCES[technique]
    confidence = Decimal("0.75")

    filled_gap = gap.model_copy(
        update={
            "current_value": ", ".join(resources),
            "confidence": confidence,
        }
    )
    assumption = Assumption(
        text=f"Assuming required equipment for {technique}: {', '.join(resources)}",
        confidence=confidence,
    )
    return filled_gap, assumption


def _extract_step_index(field_path: str) -> int | None:
    """Extract step index from field path like 'steps[2].heat_level'."""
    import re

    match = re.search(r"steps\[(\d+)\]", field_path)
    if match:
        return int(match.group(1))
    return None
