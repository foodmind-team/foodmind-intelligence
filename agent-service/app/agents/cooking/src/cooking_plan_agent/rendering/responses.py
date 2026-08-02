"""Response renderers — convert workflow state into terminal PlanResponse objects.

Handbook sections 11.6–11.10: each renderer is a pure function that takes
PlanState and returns exactly one PlanResponse subtype. validate_terminal_response
ensures the response is well-formed before it exits the graph.
"""

from __future__ import annotations

from uuid import uuid4

from cooking_plan_agent.domain.errors import (
    DomainErrorCode,
    is_known_error_code,
    public_message_for,
)
from cooking_plan_agent.domain.models import (
    Assumption,
    CompletionItem,
    ConfirmationPlanResponse,
    FailedPlanResponse,
    InfeasiblePlanResponse,
    PlanResponse,
    ReadyPlanResponse,
)
from cooking_plan_agent.rendering.builder import (
    build_dish_completion_summary,
    build_mise_en_place,
    build_timeline,
)
from cooking_plan_agent.workflow.state import PlanState

# =============================================================================
# 11.6  READY response
# =============================================================================


def render_ready_response(state: PlanState) -> ReadyPlanResponse:
    """Build a complete READY response from workflow state.

    Aggregates: timeline from schedule + tasks, mise en place from
    prep tasks, dish completions, and completion checklist from
    the reservation proposal.

    Args:
        state: Workflow state at the verify_schedule → READY transition.

    Returns:
        ReadyPlanResponse with full schedule details.
    """
    request = state["request"]
    schedule = state.get("schedule_result")

    # Collect all tasks from the task graph
    task_graph = state.get("task_graph")
    all_tasks = task_graph.tasks if task_graph else ()

    # Build timeline
    timeline: tuple[dict[str, object], ...] = ()
    makespan = 0
    solver_status = "UNKNOWN"
    if schedule:
        solver_status = schedule.status.value
        makespan = schedule.makespan_minutes or 0
        timeline = build_timeline(schedule, all_tasks)

    # P0-05: when the caller supplied an absolute serving instant, attach
    # it to the timeline as display context (plan start derives from it).
    # This is a display aid only — scheduling itself uses integer minutes.
    if timeline and request.serving_at is not None:
        serving_iso = request.serving_at.isoformat()
        timeline = tuple(
            {
                **entry,
                "serving_at": serving_iso,
                "offset_from_serving_minutes": -int(str(entry["end_minute"])),
            }
            if isinstance(entry, dict)
            else entry
            for entry in timeline
        )

    # Build mise en place from prep tasks
    prep_tasks = state.get("prep_tasks", ())
    safety_tasks = state.get("safety_tasks", ())
    recipe_tasks = state.get("recipe_tasks", ())
    combined = recipe_tasks + prep_tasks + safety_tasks
    mise_en_place = build_mise_en_place(combined) if combined else ()

    # Build dish completion summary
    dish_completions: tuple[dict[str, object], ...] = ()
    if schedule:
        dish_completions = build_dish_completion_summary(schedule, all_tasks)

    # Build completion checklist from feasibility report allocations
    # (deferred to when reservation_proposal is wired)
    completion_checklist: tuple[CompletionItem, ...] = ()
    feasibility = state.get("feasibility_report")
    if feasibility and feasibility.ingredient_shortages:
        from cooking_plan_agent.inventory.feasibility import build_reservation_proposal

        proposal = build_reservation_proposal(feasibility)
        completion_checklist = proposal.items

    return ReadyPlanResponse(
        plan_id=request.request_id,
        status="READY",
        solver_status=solver_status,
        makespan_minutes=makespan,
        timeline=timeline,
        completion_checklist=completion_checklist,
        mise_en_place=mise_en_place,
        dish_completions=dish_completions,
        # P3-04: regional safety-policy provenance (region/version/sources).
        safety_policy=state.get("safety_policy"),
    )


# =============================================================================
# 11.7  CONFIRMATION response
# =============================================================================


def render_confirmation_response(state: PlanState) -> ConfirmationPlanResponse:
    """Build a NEEDS_CONFIRMATION response with assumptions and repair options.

    Sources assumptions from parsed recipes (inferred fields) and repair
    options from the feasibility check. Generates user-facing questions.

    P0-06: also emits structured, client-submittable ApprovedDecisions so
    the client can resubmit them verbatim instead of opaque string IDs.

    Args:
        state: Workflow state at any CONFIRMATION transition.

    Returns:
        ConfirmationPlanResponse with assumptions, options, decisions,
        and questions.
    """
    request = state["request"]

    # Collect assumptions from parsed recipes
    parsed = state.get("parsed_recipes", ())
    assumptions: list[Assumption] = []
    for recipe in parsed:
        assumptions.extend(recipe.assumptions)

    # P1-01: research evidence that was applied but still warrants
    # confirmation (e.g. conflict over threshold) is surfaced here with its
    # EvidenceRef provenance — never silently dropped.
    assumptions.extend(state.get("research_assumptions", ()))

    # Collect repair options
    repair_options = state.get("repair_options", ())

    # Generate questions based on what needs confirmation
    questions: list[str] = []
    if assumptions:
        questions.append("Are the inferred cooking parameters acceptable?")
    if repair_options:
        questions.append("Which repair options would you like to apply?")
    if not questions:
        questions.append("Would you like to proceed with these options?")

    # P0-06: plan revision — the confirmation the client will answer.
    plan_revision = f"{request.request_id}:v1"

    from cooking_plan_agent.repair.options import build_approved_decisions

    decisions = build_approved_decisions(repair_options, plan_revision)

    return ConfirmationPlanResponse(
        plan_id=request.request_id,
        status="NEEDS_CONFIRMATION",
        assumptions=tuple(assumptions),
        repair_options=repair_options,
        questions=tuple(questions),
        decisions=decisions,
        plan_revision=plan_revision,
        # P3-04: regional safety-policy provenance (region/version/sources).
        safety_policy=state.get("safety_policy"),
    )


# =============================================================================
# 11.8  INFEASIBLE response
# =============================================================================


def render_infeasible_response(state: PlanState) -> InfeasiblePlanResponse:
    """Build an INFEASIBLE response with reasons from safety/feasibility/errors.

    Merges reasons from all sources, ordered by severity:
      safety → feasibility → scheduling → generic.

    Args:
        state: Workflow state at any INFEASIBLE transition.

    Returns:
        InfeasiblePlanResponse with ordered reasons.
    """
    request = state["request"]
    error = state.get("error")
    safety = state.get("safety_report")
    feasibility = state.get("feasibility_report")
    schedule = state.get("schedule_result")

    reasons: list[str] = []

    # Safety reasons (strongest blockers)
    if safety is not None and not safety.is_safe:
        reasons.extend(f.description for f in safety.findings)

    # Feasibility reasons
    if feasibility is not None and not feasibility.is_feasible:
        for shortage in feasibility.ingredient_shortages:
            reasons.append(
                f"Insufficient '{shortage.ingredient_name}': "
                f"need {shortage.required} {shortage.unit}, "
                f"have {shortage.available} {shortage.unit}"
            )
        for res in feasibility.missing_resources:
            reasons.append(f"Missing equipment: {res}")

    # Scheduling reasons
    if schedule is not None:
        status = schedule.status.value
        if status == "INFEASIBLE":
            reasons.append("The scheduler proved no feasible timeline exists with current constraints.")
        elif status == "MODEL_INVALID":
            reasons.append("The scheduling model is invalid — likely a data inconsistency.")

    # Fallback reason
    if not reasons:
        if error:
            reasons.append(error.message)
        else:
            reasons.append("The plan cannot be fulfilled with current constraints.")

    return InfeasiblePlanResponse(
        plan_id=request.request_id if request else "unknown",
        status="INFEASIBLE",
        reasons=tuple(reasons),
        safe_alternatives=(),
    )


# =============================================================================
# 11.9  FAILED response
# =============================================================================


def render_failed_response(state: PlanState) -> FailedPlanResponse:
    """Build a FAILED response from the public message catalog (P2-03).

    Client-facing text always comes from the catalog — never from the node's
    internal ``message``. An error code without a catalog row fails closed to
    INTERNAL_ERROR instead of echoing raw exception text.

    Graceful fallback: if no error is in state, returns INTERNAL_ERROR.

    Args:
        state: Workflow state at any FAILED transition.

    Returns:
        FailedPlanResponse with stable error code, public message and
        correlation ID for support traceability.
    """
    request = state["request"]
    error = state.get("error")

    if error is None:
        correlation_id = request.request_id if request else str(uuid4().hex[:8])
        return FailedPlanResponse(
            status="FAILED",
            error_code=DomainErrorCode.INTERNAL_ERROR.value,
            correlation_id=correlation_id,
            message=public_message_for(DomainErrorCode.INTERNAL_ERROR.value),
        )

    # Missing catalog row is an invariant break — fail closed to
    # INTERNAL_ERROR (P2-03), never echo the node's raw message.
    error_code = error.error_code if is_known_error_code(error.error_code) else DomainErrorCode.INTERNAL_ERROR.value
    message = error.public_message or public_message_for(error_code)

    return FailedPlanResponse(
        status="FAILED",
        error_code=error_code,
        correlation_id=error.correlation_id,
        message=message,
    )


# =============================================================================
# 11.10  Terminal response validation
# =============================================================================


def validate_terminal_response(response: PlanResponse) -> PlanResponse:
    """Validate that a terminal response is well-formed before graph exit.

    Checks:
      - READY: makespan > 0, solver_status is non-empty
      - CONFIRMATION: has at least one assumption or repair_option or question
      - INFEASIBLE: has at least one reason
      - FAILED: error_code and correlation_id are non-empty

    Returns the response unchanged on success. Raises ValueError on
    validation failure (which the caller should catch and map to FAILED).

    Args:
        response: Any PlanResponse subtype.

    Returns:
        The same response object if valid.

    Raises:
        ValueError: If the response fails validation.
    """
    status = getattr(response, "status", None)

    if status == "READY":
        if isinstance(response, ReadyPlanResponse):
            if not response.solver_status:
                raise ValueError("READY response: solver_status is empty")
            if response.makespan_minutes <= 0:
                raise ValueError(f"READY response: makespan must be > 0, got {response.makespan_minutes}")

    elif status == "NEEDS_CONFIRMATION":
        if isinstance(response, ConfirmationPlanResponse):
            has_content = bool(response.assumptions or response.repair_options or response.questions)
            if not has_content:
                raise ValueError("CONFIRMATION response: must have at least one assumption, repair_option, or question")

    elif status == "INFEASIBLE":
        if isinstance(response, InfeasiblePlanResponse):
            if not response.reasons:
                raise ValueError("INFEASIBLE response: must have at least one reason")

    elif status == "FAILED":
        if isinstance(response, FailedPlanResponse):
            if not response.error_code.strip():
                raise ValueError("FAILED response: error_code is empty")
            if not response.correlation_id.strip():
                raise ValueError("FAILED response: correlation_id is empty")

    else:
        raise ValueError(f"Unknown response status: {status!r}")

    return response
