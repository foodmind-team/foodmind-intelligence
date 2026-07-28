# Standard-library imports for type annotations and interface contracts.
from datetime import (
    datetime,  # UTC timestamp for the Clock port; avoids third-party date libraries.
)
from typing import (
    Protocol,  # Structural subtyping — callers depend on shapes, not nominal inheritance.
)

# Domain model imports define the I/O types that port methods accept and return.
# These are pure data classes; importing them here keeps the port technology-agnostic.
from cooking_plan_agent.domain.models import (
    EvidenceQuery,  # Structured search question — the input to RecipeResearcher.research().
    EvidenceResult,  # One cited piece of evidence — the output item of RecipeResearcher.research().
    RecipeIR,  # Intermediate Representation of a parsed recipe — output of RecipeExtractor.extract().
)


class RecipeExtractor(Protocol):
    """Parse unstructured recipe text into a validated RecipeIR."""

    # Signature implies an asynchronous, potentially remote call (e.g. to an LLM).
    async def extract(self, source_text: str) -> RecipeIR: ...


class RecipeResearcher(Protocol):
    """Search for evidence to answer a structured query."""

    # Returns a list because one query may yield multiple relevant sources.
    async def research(self, query: EvidenceQuery) -> list[EvidenceResult]: ...


class Clock(Protocol):
    """Return current UTC time — synchronous because no I/O is involved."""

    # Synchronous method: obtaining system time requires no network or disk wait.
    def now_utc(self) -> datetime: ...
