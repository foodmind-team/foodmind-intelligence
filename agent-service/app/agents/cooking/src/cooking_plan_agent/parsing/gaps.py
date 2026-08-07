"""Recipe gap detection and classification — handbook 4.9–4.10.

Detects missing or low-confidence fields in ExtractedRecipeCandidate and
classifies each gap by severity (critical, safety_critical, resource_critical,
optimisation, cosmetic). The gap classification drives routing decisions in
the LangGraph workflow.
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
# =============================================================================


class GapClass:
    """Gap severity classes for routing decisions."""

    CRITICAL = "critical"
    """Missing info that makes scheduling unreliable (heat, duration)."""

    SAFETY_CRITICAL = "safety_critical"
    """Missing info that could lead to unsafe food (temperature for proteins)."""

    RESOURCE_CRITICAL = "resource_critical"
    """Missing resource info needed for feasibility check."""

    OPTIMISATION = "optimisation"
    """Missing info that prevents optimisation but doesn't block feasibility."""

    COSMETIC = "cosmetic"
    """Minor missing info (garnish details, optional notes)."""


# =============================================================================
# Gap detection
# =============================================================================


def find_recipe_gaps(candidate: ExtractedRecipeCandidate) -> tuple[RecipeGap, ...]:
    """Detect all gaps in a single extracted recipe candidate.

    Checks each step for missing heat, duration, temperature, and resource
    information. Also checks candidate-level fields (servings, dish name).

    Args:
        candidate: An extracted recipe candidate from RecipeExtractor.

    Returns:
        Tuple of RecipeGap objects, one per detected gap.
    """
    gaps: list[RecipeGap] = []

    # --- Candidate-level gaps ---
    gaps.extend(_check_candidate_gaps(candidate))

    # --- Step-level gaps ---
    for i, step in enumerate(candidate.steps):
        gaps.extend(_check_step_gaps(candidate.recipe_id, i, step))

    return tuple(gaps)


# =============================================================================
# Internal check helpers
# =============================================================================


def _check_candidate_gaps(candidate: ExtractedRecipeCandidate) -> list[RecipeGap]:
    """Check candidate-level fields for gaps."""
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
    """Check a single step for gaps."""
    gaps: list[RecipeGap] = []
    field_prefix = f"steps[{step_index}]"

    # Heating steps must have heat level
    if step.category == "heating":
        gaps.extend(_check_heat_gaps(recipe_id, step_index, step, field_prefix))
        gaps.extend(_check_duration_gaps(recipe_id, step_index, step, field_prefix))
        gaps.extend(_check_temperature_gaps(recipe_id, step_index, step, field_prefix))

    # All steps should have at least one resource hint for feasibility
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
    """Check for missing heat level in heating steps."""
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
    """Check for missing duration in steps that need timing."""
    gaps: list[RecipeGap] = []

    # Heating steps that don't specify any duration
    if step.active_duration_minutes is None and step.passive_duration_minutes is None:
        # Boiling/baking without time is a critical gap
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
    """Check for missing target temperature in steps that need precise heat."""
    # Only check if the step looks like it needs precise temperature
    # (baking, roasting, or involves proteins)
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
# =============================================================================


def _heat_gap_class(step: ExtractedStep) -> str:
    """Determine gap class for missing heat level."""
    # Baking/roasting without temperature info is safety-critical for proteins
    instruction_lower = step.instruction.lower()
    if any(kw in instruction_lower for kw in ("chicken", "beef", "pork", "fish", "meat", "seafood")):
        return GapClass.SAFETY_CRITICAL
    return GapClass.CRITICAL


def _is_time_sensitive_step(step: ExtractedStep) -> bool:
    """Check if a step is time-sensitive (boiling, baking, roasting, etc.)."""
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
    """Check if a step requires precise temperature."""
    temp_keywords = ("bake", "roast", "oven", "烤", "烘")
    lower = step.instruction.lower()
    return any(kw in lower for kw in temp_keywords)


def _involves_protein(step: ExtractedStep) -> bool:
    """Check if a step likely involves raw protein (safety-critical temperature)."""
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
    """Generate a stable gap ID."""
    suffix = field_path.replace("[", "_").replace("]", "").replace(".", "_")
    short = uuid4().hex[:8]
    return f"gap_{recipe_id}_{suffix}_{short}"
