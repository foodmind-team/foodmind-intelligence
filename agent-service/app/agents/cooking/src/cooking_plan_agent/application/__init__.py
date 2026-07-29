"""Application layer — use cases and service ports.

This package defines what the system *does* in technology-agnostic terms.
It contains the entry use case (GenerateCookingPlanService) and service
ports (Protocol classes) that decouple the domain from infrastructure.

Handbook sections: 9.4 (Application Service), 3.6 (Ports & Adapters).

Re-exports the public API so callers only need one import:
  >>> from cooking_plan_agent.application import GenerateCookingPlanService
"""

from cooking_plan_agent.application.ports import (
    Clock,
    RecipeExtractor,
    RecipeResearcher,
)
from cooking_plan_agent.application.service import GenerateCookingPlanService

__all__ = [
    "Clock",
    "GenerateCookingPlanService",
    "RecipeExtractor",
    "RecipeResearcher",
]
