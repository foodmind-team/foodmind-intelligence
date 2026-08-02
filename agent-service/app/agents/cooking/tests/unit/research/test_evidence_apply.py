"""P1-01: tests for research evidence application and post-research routing.

Covers the evidence-merge loop:
  - reliable evidence is written back to the exact field identified by
    gap_id + recipe_id + field_path (never list-position guessing)
  - provenance (EvidenceRef) lands on the produced Assumption
  - timeout / empty / conflicting / field-location failure evidence is NOT
    auto-applied and routes to NEEDS_CONFIRMATION
  - safety-critical temperatures without a verifiable URL are never
    auto-resolved (LLM knowledge has no URL)
"""

from decimal import Decimal

import pytest

from cooking_plan_agent.domain.enums import HeatLevel
from cooking_plan_agent.domain.models import (
    CookingEvidence,
    ExtractedIngredient,
    ExtractedRecipeCandidate,
    ExtractedStep,
    RecipeGap,
    ReconciledEvidence,
)
from cooking_plan_agent.research.evidence_apply import (
    apply_evidence_to_candidate,
    evidence_has_verifiable_url,
    field_name,
    locate_step_index,
)
from cooking_plan_agent.workflow.nodes import apply_research_evidence_node
from cooking_plan_agent.workflow.routing import route_after_research

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _candidate() -> ExtractedRecipeCandidate:
    """Candidate with one heating step lacking heat/duration/temperature."""
    return ExtractedRecipeCandidate(
        recipe_id="r1",
        dish_name="Chicken Stir-Fry",
        original_servings=2,
        source_language="en",
        ingredients=(
            ExtractedIngredient(
                raw_text="chicken 200g",
                name="chicken breast",
                quantity=200,
                unit="g",
            ),
        ),
        steps=(
            ExtractedStep(
                step_number=1,
                instruction="Stir-fry chicken in a wok",
                category="heating",
                heat_level=HeatLevel.NONE,
            ),
        ),
    )


def _heat_gap() -> RecipeGap:
    return RecipeGap(
        gap_id="gap-heat",
        recipe_id="r1",
        field_path="steps[0].heat_level",
        current_value="NONE",
        gap_class="critical",
        description="Missing heat level",
        confidence=Decimal("1.0"),
    )


def _reconciled_heat(url: str = "https://www.seriouseats.com/stir-fry") -> ReconciledEvidence:
    """Reliable single-source heat evidence with a verifiable URL."""
    return ReconciledEvidence(
        heat_level=HeatLevel.HIGH,
        source_count=1,
        needs_confirmation=False,
        evidence_items=(
            CookingEvidence(
                operation="stir-fry",
                heat_level=HeatLevel.HIGH,
                source_url=url,
                source_title="Serious Eats Stir-Fry",
                source_excerpt="Use high heat.",
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Field-location helpers
# ---------------------------------------------------------------------------


def test_locate_step_index_parses_field_path() -> None:
    assert locate_step_index("steps[0].heat_level") == 0
    assert locate_step_index("steps[2].passive_duration_minutes") == 2
    assert locate_step_index("dish_name") is None


def test_field_name_extracts_leaf() -> None:
    assert field_name("steps[0].heat_level") == "heat_level"
    assert field_name("original_servings") == "original_servings"


# ---------------------------------------------------------------------------
# Application logic
# ---------------------------------------------------------------------------


def test_reliable_heat_evidence_is_applied_to_exact_step() -> None:
    candidate = _candidate()
    result = apply_evidence_to_candidate(candidate, _heat_gap(), _reconciled_heat())

    assert result.applied is True
    assert result.needs_confirmation is False
    assert result.candidate is not None
    step = result.candidate.steps[0]
    assert step.heat_level == HeatLevel.HIGH
    # Provenance must be traceable on the assumption.
    assert result.assumption is not None
    assert result.assumption.evidence
    assert result.assumption.evidence[0].url == "https://www.seriouseats.com/stir-fry"


def test_duration_evidence_uses_conservative_upper_bound() -> None:
    candidate = _candidate()
    gap = RecipeGap(
        gap_id="gap-dur",
        recipe_id="r1",
        field_path="steps[0].passive_duration_minutes",
        gap_class="critical",
        description="Missing duration",
        confidence=Decimal("1.0"),
    )
    reconciled = ReconciledEvidence(
        duration_min_minutes=3,
        duration_max_minutes=5,
        source_count=1,
        evidence_items=(),
    )

    result = apply_evidence_to_candidate(candidate, gap, reconciled)

    assert result.applied is True
    assert result.candidate is not None
    assert result.candidate.steps[0].passive_duration_minutes == 5


def test_temperature_evidence_is_applied() -> None:
    candidate = _candidate()
    gap = RecipeGap(
        gap_id="gap-temp",
        recipe_id="r1",
        field_path="steps[0].target_temperature_c",
        gap_class="critical",
        description="Missing temperature",
        confidence=Decimal("1.0"),
    )
    reconciled = ReconciledEvidence(
        explicit_temperature_c=Decimal(200),
        source_count=1,
        evidence_items=(),
    )

    result = apply_evidence_to_candidate(candidate, gap, reconciled)

    assert result.applied is True
    assert result.candidate is not None
    assert result.candidate.steps[0].target_temperature_c == Decimal(200)


def test_empty_evidence_requires_confirmation() -> None:
    candidate = _candidate()
    reconciled = ReconciledEvidence(source_count=0, needs_confirmation=True)

    result = apply_evidence_to_candidate(candidate, _heat_gap(), reconciled)

    assert result.applied is False
    assert result.needs_confirmation is True


def test_field_location_failure_requires_confirmation() -> None:
    candidate = _candidate()
    gap = _heat_gap().model_copy(update={"field_path": "steps[9].heat_level"})

    result = apply_evidence_to_candidate(candidate, gap, _reconciled_heat())

    assert result.applied is False
    assert result.needs_confirmation is True


@pytest.mark.asyncio
async def test_node_unknown_recipe_id_never_guesses_by_position() -> None:
    """A gap whose recipe_id is absent must never hit another recipe's fields."""
    candidate = _candidate()
    other_recipe = candidate.model_copy(
        update={
            "recipe_id": "r2",
            "steps": (
                ExtractedStep(
                    step_number=1,
                    instruction="Boil rice",
                    category="heating",
                    heat_level=HeatLevel.NONE,
                ),
            ),
        }
    )
    gap = _heat_gap().model_copy(update={"recipe_id": "r2", "gap_id": "gap-heat-r2"})
    state = {
        "extracted_candidates": (candidate, other_recipe),
        "gaps": (gap,),
        "research_evidence": {"gap-heat-r2": _reconciled_heat()},
    }

    result = await apply_research_evidence_node(state, _FakeRuntime(None))

    # The targeted recipe gets the value; the first recipe must NOT be touched.
    assert result["needs_confirmation"] is False
    assert result["extracted_candidates"][0].steps[0].heat_level == HeatLevel.NONE
    assert result["extracted_candidates"][1].steps[0].heat_level == HeatLevel.HIGH
    assert result["gaps"] == ()


def test_safety_critical_temperature_without_url_is_not_applied() -> None:
    """P1-01 rule 6: LLM knowledge (no URL) cannot resolve safe temperature."""
    candidate = _candidate()
    gap = RecipeGap(
        gap_id="gap-safe-temp",
        recipe_id="r1",
        field_path="steps[0].target_temperature_c",
        gap_class="safety_critical",
        description="Missing safe internal temperature for chicken",
        confidence=Decimal("1.0"),
    )
    reconciled = ReconciledEvidence(
        explicit_temperature_c=Decimal(74),
        source_count=1,
        needs_confirmation=False,
        evidence_items=(),  # LLM knowledge carries no URL
    )

    result = apply_evidence_to_candidate(candidate, gap, reconciled)

    assert result.applied is False
    assert result.needs_confirmation is True


def test_safety_critical_temperature_with_url_is_applied() -> None:
    candidate = _candidate()
    gap = RecipeGap(
        gap_id="gap-safe-temp",
        recipe_id="r1",
        field_path="steps[0].target_temperature_c",
        gap_class="safety_critical",
        description="Missing safe internal temperature for chicken",
        confidence=Decimal("1.0"),
    )
    reconciled = ReconciledEvidence(
        explicit_temperature_c=Decimal(74),
        source_count=1,
        needs_confirmation=False,
        evidence_items=(
            CookingEvidence(
                operation="bake",
                explicit_temperature_c=Decimal(74),
                source_url="https://www.fda.gov/chicken",
                source_title="FDA",
                source_excerpt="Chicken must reach 74C.",
            ),
        ),
    )

    result = apply_evidence_to_candidate(candidate, gap, reconciled)

    assert result.applied is True
    assert result.needs_confirmation is False


def test_evidence_has_verifiable_url() -> None:
    assert evidence_has_verifiable_url(_reconciled_heat()) is True
    assert evidence_has_verifiable_url(ReconciledEvidence(source_count=1, evidence_items=())) is False


# ---------------------------------------------------------------------------
# Node-level behaviour
# ---------------------------------------------------------------------------


class _FakeRuntime:
    """Minimal runtime stand-in — nodes only need .context at call time."""

    def __init__(self, context: object) -> None:
        self.context = context


@pytest.mark.asyncio
async def test_node_applies_evidence_and_keeps_unresolved_gaps() -> None:
    candidate = _candidate()
    heat_gap = _heat_gap()
    # A second gap that research did NOT touch stays unresolved.
    duration_gap = RecipeGap(
        gap_id="gap-dur",
        recipe_id="r1",
        field_path="steps[0].passive_duration_minutes",
        gap_class="critical",
        description="Missing duration",
        confidence=Decimal("1.0"),
    )
    state = {
        "extracted_candidates": (candidate,),
        "gaps": (heat_gap, duration_gap),
        "research_evidence": {"gap-heat": _reconciled_heat()},
    }

    result = await apply_research_evidence_node(state, _FakeRuntime(None))

    updated = result["extracted_candidates"]
    assert updated[0].steps[0].heat_level == HeatLevel.HIGH
    # Only the resolved gap is dropped.
    assert [g.gap_id for g in result["gaps"]] == ["gap-dur"]
    assert result["research_assumptions"], "Applied evidence must produce an assumption"
    assert result["research_assumptions"][0].evidence
    assert result["needs_confirmation"] is True  # critical duration gap remains


@pytest.mark.asyncio
async def test_node_conflicting_evidence_routes_to_confirmation() -> None:
    candidate = _candidate()
    heat_gap = _heat_gap()
    conflicting = _reconciled_heat().model_copy(
        update={"needs_confirmation": True}  # MAD over threshold
    )
    state = {
        "extracted_candidates": (candidate,),
        "gaps": (heat_gap,),
        "research_evidence": {"gap-heat": conflicting},
    }

    result = await apply_research_evidence_node(state, _FakeRuntime(None))

    # The value may be applied but the disagreement must still surface.
    assert result["needs_confirmation"] is True


@pytest.mark.asyncio
async def test_node_no_research_is_noop() -> None:
    state = {"extracted_candidates": (_candidate(),), "gaps": (_heat_gap(),), "research_evidence": {}}

    result = await apply_research_evidence_node(state, _FakeRuntime(None))

    assert result == {}
    assert state["gaps"][0].gap_id == "gap-heat"


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def test_route_after_research_reliable_evidence_proceeds_to_ir() -> None:
    assert route_after_research({"needs_confirmation": False, "gaps": ()}) == "validate_recipe_ir"


def test_route_after_research_needs_confirmation() -> None:
    assert route_after_research({"needs_confirmation": True, "gaps": ()}) == "build_confirmation_response"


def test_route_after_research_remaining_critical_gap() -> None:
    state = {"needs_confirmation": False, "gaps": (_heat_gap(),)}
    assert route_after_research(state) == "build_confirmation_response"
