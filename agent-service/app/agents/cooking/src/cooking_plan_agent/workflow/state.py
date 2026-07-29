"""LangGraph workflow state — TypedDict per handbook 8.2.

Uses TypedDict for workflow state (serialisable) and Pydantic models
for values at boundaries. Provider clients and secrets are NOT stored here.
"""

from typing import TypedDict

from cooking_plan_agent.domain.models import (
    ConfirmationPlanResponse,
    CookingTask,
    ExtractedRecipeCandidate,
    FeasibilityReport,
    GeneratePlanRequest,
    PlanResponse,
    RecipeGap,
    RecipeIR,
    RepairOption,
    SafetyReport,
    WorkflowError,
)
from cooking_plan_agent.preparation.task_graph import TaskGraph
from cooking_plan_agent.scheduling.models import ScheduleResult, VerificationReport


class PlanState(TypedDict, total=False):
    """Workflow state carried between LangGraph nodes.

    Only serialisable domain objects. No provider clients, request-scoped
    secrets, or OR-Tools model objects (handbook 8.2).
    """

    # --- Input ---
    request: GeneratePlanRequest

    # --- Parsing & inference ---
    extracted_candidates: tuple[ExtractedRecipeCandidate, ...]
    gaps: tuple[RecipeGap, ...]
    evidence: tuple[dict, ...]  # EvidenceRef-like dicts (serialisable)

    # --- Validation ---
    parsed_recipes: tuple[RecipeIR, ...]
    safety_report: SafetyReport
    feasibility_report: FeasibilityReport
    repair_options: tuple[RepairOption, ...]

    # --- Preparation & scheduling ---
    recipe_tasks: tuple[CookingTask, ...]
    prep_tasks: tuple[CookingTask, ...]
    safety_tasks: tuple[CookingTask, ...]
    task_graph: TaskGraph
    schedule_result: ScheduleResult
    verification_report: VerificationReport

    # --- Terminal output ---
    response: PlanResponse
    needs_confirmation: bool
    confirmation_context: ConfirmationPlanResponse

    # --- Error ---
    error: WorkflowError
