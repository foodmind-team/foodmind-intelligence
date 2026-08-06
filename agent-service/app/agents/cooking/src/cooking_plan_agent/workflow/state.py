"""LangGraph workflow state — TypedDict per handbook 8.2.

Uses TypedDict for workflow state (serialisable) and Pydantic models
for values at boundaries. Provider clients and secrets are NOT stored here.
"""

from typing import TypedDict

from cooking_plan_agent.domain.models import (
    Assumption,
    ConfirmationPlanResponse,
    CookingTask,
    ExtractedRecipeCandidate,
    FeasibilityReport,
    GeneratePlanRequest,
    PlanResponse,
    RecipeGap,
    RecipeIR,
    ReconciledEvidence,
    RepairOption,
    SafetyPolicyRecord,
    SafetyReport,
    WorkflowError,
)
from cooking_plan_agent.preparation.task_graph import TaskGraph
from cooking_plan_agent.scheduling.models import ScheduleResult, VerificationReport


class PlanState(TypedDict, total=False):
    """Workflow state carried between LangGraph nodes.

    Only serialisable domain objects. No provider clients, request-scoped
    secrets, or OR-Tools model objects (handbook 8.2).

    total=False means every field is Optional — nodes only return the
    CHANGED subset, and downstream nodes use .get() to safely read.
    """

    # --- Input (populated once, never mutated) ---
    request: GeneratePlanRequest

    # --- Parsing & inference ---
    # Candidates are raw extraction output; RecipeIR is the validated form
    extracted_candidates: tuple[ExtractedRecipeCandidate, ...]
    gaps: tuple[RecipeGap, ...]
    # Evidence stored as plain dicts to keep state serialisable (no dataclass coupling)
    evidence: tuple[dict[str, object], ...]
    # Per-gap reconciled research results (keyed by gap_id for traceability)
    research_evidence: dict[str, ReconciledEvidence]
    # Evidence-backed assumptions produced by apply_research_evidence_node
    # (P1-01). Merged into RecipeIR.assumptions so provenance is traceable.
    research_assumptions: tuple[Assumption, ...]

    # --- Validation ---
    parsed_recipes: tuple[RecipeIR, ...]
    safety_report: SafetyReport
    # P3-04: resolved regional policy provenance, recorded on terminal
    # responses and retained in state so historical plans stay auditable.
    safety_policy: SafetyPolicyRecord
    feasibility_report: FeasibilityReport
    # Repair options injected by check_feasibility when infeasible but fixable
    repair_options: tuple[RepairOption, ...]

    # --- Preparation & scheduling ---
    # Tasks are split into three categories so merge_preparation can
    # apply different dedup/priority rules per category
    recipe_tasks: tuple[CookingTask, ...]
    prep_tasks: tuple[CookingTask, ...]
    safety_tasks: tuple[CookingTask, ...]
    # P2-01: human-readable summaries of shared-prep merge/branch/isolate
    # decisions, for observability and regression tests.
    prep_observations: tuple[str, ...]
    task_graph: TaskGraph
    schedule_result: ScheduleResult
    # Independent verification report (separate from solver's internal check)
    verification_report: VerificationReport
    # P4-01: additive schedule explanation set by explain_schedule_node
    # between verify and render READY. Display-only — never affects the
    # schedule. explanation_source ∈ {"llm", "deterministic", "disabled"}.
    explanation: str | None
    explanation_source: str | None

    # --- Terminal output ---
    # Exactly ONE terminal field is populated by the terminal node that fires
    response: PlanResponse
    needs_confirmation: bool
    confirmation_context: ConfirmationPlanResponse

    # --- Error ---
    # Set by any node that encounters a recoverable/terminal error;
    # the graph uses this to route to error terminal nodes
    error: WorkflowError

    # --- P5: agent trace (shared by all agentic phases) ---
    # 每步 agent 决策（repair / tool_call / question）的留痕，保持可序列化。
    # 元素为 plain dict（{"step": int, "action": str, "detail": dict}）。
    agent_trace: tuple[dict[str, object], ...]
