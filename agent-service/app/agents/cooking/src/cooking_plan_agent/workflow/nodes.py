"""LangGraph workflow nodes — thin wrappers around domain services.

Per handbook 8.4: each node calls ONE application/domain service and
returns only CHANGED state fields. No in-place state mutation.
No broad exception catching that masks errors as partial success.
"""

# LangGraph runtime type — context is injected by the framework
from langgraph.runtime import Runtime

from cooking_plan_agent.domain.models import (
    ConfirmationPlanResponse,
    FailedPlanResponse,
    FeasibilityReport,
    InfeasiblePlanResponse,
    ReadyPlanResponse,
    SafetyReport,
    WorkflowError,
)
from cooking_plan_agent.workflow.context import WorkflowContext
from cooking_plan_agent.workflow.state import PlanState

# ============================================================================
# Input & parsing nodes
# ============================================================================


async def validate_input_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict:
    """Validate the incoming GeneratePlanRequest.

    Checks: non-empty recipes, reasonable serving/task counts, schema version.
    STUB: passes through. Full validation when contract is finalized.

    Returns an empty dict on success or {'error': WorkflowError} on failure.
    The graph does NOT have an error edge from validate_input, so an error
    here will propagate as a runtime exception.
    """
    request = state["request"]
    if not request.recipes:
        return {
            "error": WorkflowError(
                error_code="INVALID_RECIPE_TEXT",
                message="Request contains no recipes",
                correlation_id=request.request_id,
                node_name="validate_input",
            )
        }
    return {}


async def parse_recipes_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict:
    """Extract structured candidates from each recipe's raw text.

    STUB: passes through. Calls runtime.context.recipe_extractor
    when RecipeExtractor is fully implemented.
    """
    return {"extracted_candidates": ()}


async def detect_gaps_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict:
    """Identify missing/inferred fields in extracted candidates.

    STUB: returns empty gaps.
    """
    return {"gaps": ()}


async def infer_local_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict:
    """Apply local cooking rules to fill detected gaps.

    STUB: passes through.
    """
    return {}


async def research_missing_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict:
    """Query web research for low-confidence critical gaps.

    STUB: passes through.
    """
    return {}


async def validate_recipe_ir_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict:
    """Build validated RecipeIR objects from candidates.

    STUB: passes through.
    """
    return {}


# ============================================================================
# Safety & feasibility nodes
# ============================================================================


async def validate_safety_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict:
    """Evaluate all safety rules against the recipe set.

    STUB: returns a safe report.
    """
    return {
        "safety_report": SafetyReport(
            report_id="stub-safety",
            findings=(),
            is_safe=True,
            has_unrepairable=False,
            required_safety_task_ids=(),
        )
    }


async def check_feasibility_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict:
    """Check inventory sufficiency and resource compatibility.

    STUB: returns feasible.
    """
    return {
        "feasibility_report": FeasibilityReport(
            report_id="stub-feasibility",
            ingredient_shortages=(),
            missing_resources=(),
            is_feasible=True,
        )
    }


async def build_confirmation_response_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict:
    """Render NEEDS_CONFIRMATION response with assumptions/repair options.

    STUB: returns minimal confirmation response.
    """
    response = ConfirmationPlanResponse(
        plan_id=state["request"].request_id,
        status="NEEDS_CONFIRMATION",
        assumptions=(),
        repair_options=state.get("repair_options", ()),
        questions=("Would you like to proceed with these options?",),
    )
    return {"response": response}


# ============================================================================
# Preparation & scheduling nodes
# ============================================================================


async def merge_preparation_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict:
    """Decompose recipe steps + merge shared preparation into CookingTasks.

    Uses existing preparation/decompose.py and preparation/prep_trie.py.
    STUB: returns empty task sets. Full wiring requires ingredient demand
    to operation chain extraction bridge.

    Returns three task tuples — downstream nodes concatenate them before
    building the DAG.
    """
    return {
        "recipe_tasks": (),
        "prep_tasks": (),
        "safety_tasks": (),
    }


async def build_task_graph_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict:
    """Build the task DAG from recipe, prep, and safety tasks.

    Wired to existing preparation/task_graph.py.

    Lazy-imports build_task_graph inside the function so the module
    can be imported even when preparation dependencies are missing.
    """
    from cooking_plan_agent.preparation.task_graph import build_task_graph

    recipe_tasks = state.get("recipe_tasks", ())
    prep_tasks = state.get("prep_tasks", ())
    safety_tasks = state.get("safety_tasks", ())

    # Defensive: if merge_preparation returned nothing, skip building
    if not recipe_tasks and not prep_tasks:
        return {}

    try:
        graph = build_task_graph(
            recipe_tasks=recipe_tasks,
            prep_tasks=prep_tasks,
            safety_tasks=safety_tasks,
        )
        return {"task_graph": graph}
    except (ValueError, TypeError, RuntimeError) as exc:
        # Cycle detection or invalid dependencies -> workflow error
        return {
            "error": WorkflowError(
                error_code="TASK_GRAPH_CYCLE",
                message=str(exc),
                correlation_id=state["request"].request_id,
                node_name="build_task_graph",
            )
        }


async def solve_schedule_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict:
    """Solve the CP-SAT scheduling problem.

    Wired to existing scheduling/orchestrator.py.
    schedule() returns tuple[ScheduleResult, VerificationReport] — we store
    only the result; verification is done independently in verify_schedule_node.

    Lazy-imports inside the function avoid coupling to OR-Tools at import time
    (OR-Tools is a heavy C++ dependency).
    """
    from cooking_plan_agent.scheduling.models import SchedulingProblem
    from cooking_plan_agent.scheduling.orchestrator import schedule as solve_schedule_fn

    task_graph = state.get("task_graph")
    if task_graph is None:
        return {
            "error": WorkflowError(
                error_code="SCHEDULE_INFEASIBLE",
                message="No task graph available for scheduling",
                correlation_id=state["request"].request_id,
                node_name="solve_schedule",
            )
        }

    problem = SchedulingProblem(
        tasks=task_graph.tasks,
        resources=state["request"].kitchen_resources,
    )

    try:
        # schedule() returns (ScheduleResult, VerificationReport)
        result, _ = solve_schedule_fn(problem)
        return {"schedule_result": result}
    except (ValueError, TypeError, RuntimeError) as exc:
        return {
            "error": WorkflowError(
                error_code="SCHEDULE_INFEASIBLE",
                message=str(exc),
                correlation_id=state["request"].request_id,
                node_name="solve_schedule",
            )
        }


async def verify_schedule_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict:
    """Independent verification of solver output.

    Wired to existing scheduling/verifier.py.
    verify() signature: verify(problem: SchedulingProblem, result: ScheduleResult)

    Verification is done in a SEPARATE node (not inside solve_schedule) so that:
    - verification can be skipped/instrumented independently
    - the verifier catches bugs in the solver itself
    """
    from cooking_plan_agent.scheduling.models import SchedulingProblem
    from cooking_plan_agent.scheduling.verifier import ScheduleVerifier

    schedule_result = state.get("schedule_result")
    if schedule_result is None:
        return {
            "error": WorkflowError(
                error_code="SCHEDULE_VERIFICATION_FAILED",
                message="No schedule result to verify",
                correlation_id=state["request"].request_id,
                node_name="verify_schedule",
            )
        }

    task_graph = state.get("task_graph")
    if task_graph is None:
        return {
            "error": WorkflowError(
                error_code="SCHEDULE_VERIFICATION_FAILED",
                message="No task graph available for verification",
                correlation_id=state["request"].request_id,
                node_name="verify_schedule",
            )
        }

    problem = SchedulingProblem(
        tasks=task_graph.tasks,
        resources=state["request"].kitchen_resources,
    )

    try:
        verifier = ScheduleVerifier()
        report = verifier.verify(problem, schedule_result)
        return {"verification_report": report}
    except (ValueError, TypeError, RuntimeError) as exc:
        return {
            "error": WorkflowError(
                error_code="SCHEDULE_VERIFICATION_FAILED",
                message=str(exc),
                correlation_id=state["request"].request_id,
                node_name="verify_schedule",
            )
        }


# ============================================================================
# Terminal response nodes
# ============================================================================


async def render_ready_response_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict:
    """Render READY response with verified schedule and completion checklist.

    STUB: returns minimal ready response.
    """
    result = state.get("schedule_result")
    response = ReadyPlanResponse(
        plan_id=state["request"].request_id,
        status="READY",
        solver_status=result.solver_status if result else "UNKNOWN",
        makespan_minutes=result.makespan_minutes if result else 0,
        timeline=(),
        completion_checklist=(),
        mise_en_place=(),
        dish_completions=(),
    )
    return {"response": response}


async def render_infeasible_response_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict:
    """Render INFEASIBLE response with hard reasons.

    Merges reasons from safety findings, feasibility shortages, and
    workflow errors into a single response. Order matters: safety
    reasons come first as they are the strongest blockers.
    """
    error = state.get("error")
    safety = state.get("safety_report")
    feasibility = state.get("feasibility_report")

    reasons: list[str] = []
    if safety is not None and not safety.is_safe:
        reasons.extend(f.description for f in safety.findings)
    if feasibility is not None and not feasibility.is_feasible:
        for shortage in feasibility.ingredient_shortages:
            reasons.append(
                f"Shortage: {shortage.ingredient_name} needs {shortage.shortage} {shortage.unit}"
            )
    if error is not None:
        reasons.append(error.message)

    response = InfeasiblePlanResponse(
        plan_id=state["request"].request_id,
        status="INFEASIBLE",
        reasons=tuple(reasons) if reasons else ("The plan cannot be fulfilled with current constraints.",),
        safe_alternatives=(),
    )
    return {"response": response}


async def render_failed_response_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict:
    """Render FAILED response with stable error code and correlation ID.

    Graceful fallback: if no error is in state, returns INTERNAL_ERROR
    with the request ID. This prevents the graph from producing a
    response with missing fields.
    """
    error = state.get("error")
    response = FailedPlanResponse(
        status="FAILED",
        error_code=error.error_code if error else "INTERNAL_ERROR",
        correlation_id=error.correlation_id if error else state["request"].request_id,
        message=error.message if error else "An unexpected error occurred",
    )
    return {"response": response}
