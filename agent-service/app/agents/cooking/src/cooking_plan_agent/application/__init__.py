"""Application layer — use cases.

This package defines what the system *does* in technology-agnostic terms.
It contains the entry use cases (GenerateCookingPlanService and
ParseRecipeImportService).

Handbook section: 9.4 (Application Service).

Re-exports the public API so callers only need one import:
  >>> from cooking_plan_agent.application import GenerateCookingPlanService
"""

from cooking_plan_agent.application.recipe_import_service import ParseRecipeImportService
from cooking_plan_agent.application.service import GenerateCookingPlanService

__all__ = [
    "GenerateCookingPlanService",
    "ParseRecipeImportService",
]
