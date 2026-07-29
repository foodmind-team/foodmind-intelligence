"""WorkflowContext — dependency injection container for LangGraph nodes.

Per handbook 8.3: services are passed through runtime context, keeping
state serialisable even without checkpoint persistence in MVP.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from cooking_plan_agent.domain.models import (
        EvidenceQuery,
        EvidenceResult,
        ExtractedRecipeCandidate,
    )


# ---------------------------------------------------------------------------
# Service protocols for the workflow context
# ---------------------------------------------------------------------------
# Protocols are used instead of ABCs so that ANY object satisfying the
# shape (duck typing) can be injected — no explicit inheritance needed.
# This keeps domain services decoupled from the workflow layer.


@runtime_checkable
class RecipeExtractor(Protocol):
    """Parse unstructured recipe text into ExtractedRecipeCandidate.

    May use LLM, regex, or both — the workflow does not care about the
    implementation, only the async extract() contract.
    """

    async def extract(self, source_text: str) -> "ExtractedRecipeCandidate": ...


@runtime_checkable
class RecipeResearcher(Protocol):
    """Search for evidence to fill recipe gaps.

    Returns a list because a single query may yield multiple evidence
    sources (e.g., temperatures from different databases).
    """

    async def research(self, query: "EvidenceQuery") -> list["EvidenceResult"]: ...


# ---------------------------------------------------------------------------
# Context dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkflowContext:
    """Immutable dependency context for all workflow nodes.

    Created at app startup (FastAPI lifespan). Each node receives this
    via LangGraph runtime.context. All services are swappable for testing.

    Frozen=True ensures nodes cannot mutate shared state through the context
    — all state changes must flow through PlanState returns.
    """

    recipe_extractor: RecipeExtractor
    # recipe_researcher is None in MVP; when wired, it enables the
    # research_missing node to fill low-confidence critical gaps
    recipe_researcher: RecipeResearcher | None = None
    # Future services (all optional until implemented):
    # safety_engine: SafetyRuleEngine | None = None
    # optimiser: ScheduleOptimiser | None = None
