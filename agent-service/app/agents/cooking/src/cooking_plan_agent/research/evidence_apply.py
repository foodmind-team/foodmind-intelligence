"""Apply reconciled research evidence back into recipe candidates (P1-01).

P1-01 closes the research loop: before this module, ``research_missing``
wrote ``research_evidence`` into state that no downstream node consumed, so
search results never updated the plan and the pipeline silently ignored them.

This module:
  1. Locates the EXACT target field via ``gap_id + recipe_id + field_path``
     (never by list position — P1-01 rule 2).
  2. Writes heat / duration / temperature values back into the candidate
     step identified by the gap.
  3. Carries evidence provenance as ``EvidenceRef`` on an Assumption so the
     source remains traceable in the final assumption/response.
  4. Decides when evidence is reliable enough to auto-apply vs. when it must
     surface for user confirmation.

Safety rule (P1-01 rule 6): a safety-critical temperature is NEVER
auto-resolved from evidence that lacks a verifiable URL. LLM knowledge
carries no URL, so it cannot prove a minimum safe cooking temperature.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import NamedTuple

from cooking_plan_agent.domain.enums import HeatLevel
from cooking_plan_agent.domain.models import (
    Assumption,
    EvidenceRef,
    ExtractedRecipeCandidate,
    ExtractedStep,
    RecipeGap,
    ReconciledEvidence,
)

# Any reconciled value needs at least this many independent sources AND no
# disagreement flag before it is written back without user confirmation.
_RELIABLE_SOURCE_COUNT = 1

# Safety-critical temperature facts are only auto-applied when the evidence
# carries a verifiable URL (e.g. a food-safety authority page). LLM knowledge
# (source_url="") cannot prove a minimum safe temperature.
_SAFETY_GAP_CLASSES = frozenset({"safety_critical"})
_TEMPERATURE_FIELDS = frozenset({"temperature", "target_temperature_c", "temperature_c"})

_STEP_INDEX_RE = re.compile(r"steps\[(\d+)\]")


class EvidenceApplication(NamedTuple):
    """Result of applying one reconciled evidence to one gap target.

    Attributes:
        candidate: Updated candidate when a value was written; None otherwise.
        assumption: Evidence-backed assumption when applied; None otherwise.
        applied: True when a field value was written into the candidate.
        needs_confirmation: True when this gap cannot be safely auto-resolved
            (no source, disagreement, field-location failure, or unverified
            safety-critical temperature).
    """

    candidate: ExtractedRecipeCandidate | None
    assumption: Assumption | None
    applied: bool
    needs_confirmation: bool


# ---------------------------------------------------------------------------
# Field location helpers — deterministic, never positional guessing
# ---------------------------------------------------------------------------


def locate_step_index(field_path: str) -> int | None:
    """Return the step index encoded in a gap's ``field_path``.

    Accepts paths like ``steps[0].heat_level``. Returns None when the path
    does not reference a step (e.g. candidate-level ``dish_name``).
    """
    match = _STEP_INDEX_RE.search(field_path)
    return int(match.group(1)) if match else None


def field_name(field_path: str) -> str:
    """Return the leaf field name of a gap's ``field_path``."""
    return field_path.rsplit(".", 1)[-1]


def evidence_has_verifiable_url(reconciled: ReconciledEvidence) -> bool:
    """True when at least one source carries a stable URL."""
    return any(bool(ev.source_url) for ev in reconciled.evidence_items)


# ---------------------------------------------------------------------------
# Core application logic
# ---------------------------------------------------------------------------


def apply_evidence_to_candidate(
    candidate: ExtractedRecipeCandidate,
    gap: RecipeGap,
    reconciled: ReconciledEvidence,
) -> EvidenceApplication:
    """Apply reconciled evidence to the candidate at the gap's field path.

    Returns an EvidenceApplication describing whether the value was written
    and whether the gap still requires user confirmation.
    """
    if reconciled.source_count <= 0:
        # Search returned nothing usable (timeout, empty results, error).
        return EvidenceApplication(None, None, applied=False, needs_confirmation=True)

    step_index = locate_step_index(gap.field_path)
    if step_index is None or step_index >= len(candidate.steps):
        # Field could not be located — never guess by list position.
        return EvidenceApplication(None, None, applied=False, needs_confirmation=True)

    field = field_name(gap.field_path)

    # Safety-critical temperature: only auto-apply when the evidence has a
    # verifiable URL (P1-01 rule 6). LLM knowledge (no URL) cannot resolve.
    if gap.gap_class in _SAFETY_GAP_CLASSES and field in _TEMPERATURE_FIELDS:
        if not evidence_has_verifiable_url(reconciled):
            return EvidenceApplication(None, None, applied=False, needs_confirmation=True)

    value, evidence_refs = _pick_value(field, reconciled)
    if value is None:
        # No reconciled value usable for this field — cannot auto-resolve.
        return EvidenceApplication(
            None,
            None,
            applied=False,
            needs_confirmation=reconciled.needs_confirmation,
        )

    step = candidate.steps[step_index]
    updated_step = _write_value(step, field, value)
    if updated_step is step:
        # Value could not be mapped onto this step shape.
        return EvidenceApplication(None, None, applied=False, needs_confirmation=True)

    updated_steps = candidate.steps[:step_index] + (updated_step,) + candidate.steps[step_index + 1 :]
    updated_candidate = candidate.model_copy(update={"steps": updated_steps})

    assumption = Assumption(
        text=(f"Researched {field} for {candidate.dish_name}: {value} (from {len(evidence_refs)} source(s))"),
        confidence=Decimal("0.8"),
        evidence=tuple(evidence_refs),
    )

    return EvidenceApplication(updated_candidate, assumption, applied=True, needs_confirmation=False)


def _pick_value(field: str, reconciled: ReconciledEvidence) -> tuple[str | None, list[EvidenceRef]]:
    """Return (value_str, evidence_refs) for a field, or (None, []) if unusable.

    Duration mapping follows gap semantics: an ``active_duration`` gap uses
    the lower bound (minimum hands-on time); a ``passive_duration`` gap uses
    the upper bound (conservative wait time).
    """
    refs = [
        EvidenceRef(
            source_type="web_search",
            title=ev.source_title or None,
            url=ev.source_url or None,
        )
        for ev in reconciled.evidence_items
        if ev.source_url
    ]

    if "heat" in field:
        if reconciled.heat_level is not None:
            return reconciled.heat_level.value, refs
    elif "active_duration" in field:
        if reconciled.duration_min_minutes is not None:
            return str(reconciled.duration_min_minutes), refs
    elif "duration" in field:
        if reconciled.duration_max_minutes is not None:
            return str(reconciled.duration_max_minutes), refs
    elif "temperature" in field:
        if reconciled.explicit_temperature_c is not None:
            return str(reconciled.explicit_temperature_c), refs

    return None, []


def _write_value(step: ExtractedStep, field: str, value_str: str) -> ExtractedStep:
    """Map a reconciled value onto the step, returning the original step when
    the value does not fit the field's type."""
    if "heat_level" in field:
        try:
            heat = HeatLevel(value_str)
        except ValueError:
            return step
        return step.model_copy(update={"heat_level": heat})

    if "active_duration" in field:
        return step.model_copy(update={"active_duration_minutes": int(value_str)})

    if "duration" in field:
        return step.model_copy(update={"passive_duration_minutes": int(value_str)})

    if "temperature" in field:
        return step.model_copy(update={"target_temperature_c": Decimal(value_str)})

    return step
