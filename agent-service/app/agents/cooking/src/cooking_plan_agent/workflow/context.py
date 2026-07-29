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


@runtime_checkable
class RecipeExtractor(Protocol):
    """Parse unstructured recipe text into ExtractedRecipeCandidate."""

    async def extract(self, source_text: str) -> "ExtractedRecipeCandidate": ...


@runtime_checkable
class RecipeResearcher(Protocol):
    """Search for evidence to fill recipe gaps."""

    async def research(self, query: "EvidenceQuery") -> list["EvidenceResult"]: ...


# ---------------------------------------------------------------------------
# Context dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkflowContext:
    """Immutable dependency context for all workflow nodes.

    Created at app startup (FastAPI lifespan). Each node receives this
    via LangGraph runtime.context. All services are swappable for testing.
    """

    recipe_extractor: RecipeExtractor
    recipe_researcher: RecipeResearcher | None = None
    # Future services (all optional until implemented):
    # safety_engine: SafetyRuleEngine | None = None
    # optimiser: ScheduleOptimiser | None = None
