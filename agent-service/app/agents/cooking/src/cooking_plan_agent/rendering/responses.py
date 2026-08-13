"""Response renderers — convert workflow state into terminal PlanResponse objects.

Handbook sections 11.6–11.10: each renderer is a pure function that takes
PlanState and returns exactly one PlanResponse subtype. validate_terminal_response
ensures the response is well-formed before it exits the graph.
"""

from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

from cooking_plan_agent.domain.errors import (
    DomainErrorCode,
    is_known_error_code,
    public_message_for,
)
from cooking_plan_agent.domain.models import (
    ApprovedDecision,
    Assumption,
    CompletionItem,
    ConfirmationPlanResponse,
    ConfirmationQuestion,
    FailedPlanResponse,
    InfeasiblePlanResponse,
    PlanResponse,
    QuestionOption,
    QuestionResponseType,
    ReadyPlanResponse,
    RepairOption,
)
from cooking_plan_agent.rendering.builder import (
    build_dish_completion_summary,
    build_execution_flow,
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

    # Build completion checklist from feasibility report allocations.
    # Uses the FULL ingredient_results (satisfied + short) so a READY plan
    # always carries the inventory consumption plan — previously it was
    # empty because ingredient_shortages only retains short items.
    completion_checklist: tuple[CompletionItem, ...] = ()
    feasibility = state.get("feasibility_report")
    if feasibility and (feasibility.ingredient_results or feasibility.ingredient_shortages):
        from cooking_plan_agent.inventory.feasibility import build_reservation_proposal

        proposal = build_reservation_proposal(feasibility)
        completion_checklist = proposal.items

    return ReadyPlanResponse(
        plan_id=request.request_id,
        status="READY",
        solver_status=solver_status,
        makespan_minutes=makespan,
        timeline=timeline,
        execution_flow=build_execution_flow(all_tasks),
        completion_checklist=completion_checklist,
        mise_en_place=mise_en_place,
        dish_completions=dish_completions,
        # P3-04: regional safety-policy provenance (region/version/sources).
        safety_policy=state.get("safety_policy"),
        # P4-01: additive schedule explanation (llm | deterministic | disabled).
        explanation=state.get("explanation"),
        explanation_source=state.get("explanation_source"),
    )


# =============================================================================
# 11.7  CONFIRMATION response
# =============================================================================

# P4-02: parsed-recipe assumptions at or above this confidence are treated
# as trustworthy and are NOT promoted to a confirmation question (they stay
# informational in response.assumptions). Research-backed assumptions always
# surface as questions — the graph only routes here when they warranted
# confirmation (P1-01), regardless of their numeric confidence.
_ASSUMPTION_CONFIDENCE_THRESHOLD = Decimal("0.5")


def _stable_question_key(*parts: str) -> str:
    """Derive a stable, bounded question key from domain fields (P4-02 D6).

    SHA-256 of the joined stable keys so the same input always reproduces
    the same question_id, and the ID never depends on array position or on
    random identifiers (e.g. the random suffix inside a gap_id).
    """
    import hashlib

    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _gap_question_field_path(recipe_id: str, field_path: str) -> str:
    """Return a self-contained path identifying both recipe and field."""
    if field_path.startswith("recipe."):
        return field_path
    return f"recipe.{recipe_id}.{field_path}"


def _servings_option_label(payload: dict[str, object]) -> str:
    servings = payload.get("servings")
    if servings is None:
        return "Reduce servings"
    count = int(str(servings))
    noun = "serving" if count == 1 else "servings"
    return f"Reduce to {count} {noun}"


def _append_repair_strategy_question(
    questions: list[ConfirmationQuestion],
    decisions: tuple[ApprovedDecision, ...],
) -> set[str]:
    """Collapse purchase + reduce_servings into one inventory strategy choice.

    Creates a strategy question whenever a purchase or reduce_servings
    decision exists. Individual per-item repair
    questions are suppressed when a strategy question is emitted so the
    user never sees a long list of Apply/Do-not-apply items.
    """
    purchase = next((decision for decision in decisions if decision.option_type == "purchase"), None)
    reduce = next((decision for decision in decisions if decision.option_type == "reduce_servings"), None)
    if purchase is None and reduce is None:
        return set()

    options: list[QuestionOption] = []
    suggested_value: str | None = None
    if reduce is not None:
        options.append(
            QuestionOption(value=reduce.option_id, label=_servings_option_label(reduce.payload), suggested=True)
        )
        suggested_value = reduce.option_id
    if purchase is not None:
        options.append(
            QuestionOption(value=purchase.option_id, label="Buy missing ingredients", suggested=reduce is None)
        )
        if suggested_value is None:
            suggested_value = purchase.option_id
    # Strategy question is emitted whenever a plan-level repair exists —
    # even a single one (e.g. only purchase) — so the user always decides
    # at plan level and never sees per-item Apply/Do-not-apply questions.
    prompt = "Some ingredients are missing. Choose how to continue."
    questions.append(
        ConfirmationQuestion(
            question_id="repair:strategy",
            field_path="repair_strategy",
            prompt=prompt,
            response_type=QuestionResponseType.CHOICE,
            options=tuple(options),
            required=True,
            suggested_value=suggested_value,
        )
    )
    grouped_ids: set[str] = set()
    if purchase is not None:
        grouped_ids.add(purchase.option_id)
    if reduce is not None:
        grouped_ids.add(reduce.option_id)
    return grouped_ids


def _build_confirmation_questions(
    state: PlanState,
    repair_options: tuple[RepairOption, ...],
    decisions: tuple[ApprovedDecision, ...],
) -> tuple[ConfirmationQuestion, ...]:
    """Build the field-level structured confirmation form (P4-02).

    Question sources, in order:
      1. Blocking gaps — every unresolved critical / safety_critical gap
         becomes exactly one required TEXT question asking for the missing
         value. question_id is derived from stable domain keys
         (recipe_id + field_path), never from array position (D6).
      2. Assumptions — each low-confidence parsed-recipe assumption, and
         every research-backed assumption, becomes one required CHOICE
         question offering to accept the suggested value or provide an
         alternative.
      3. Repair options — each supported RepairOption becomes one CHOICE
         question (apply / skip); the apply option's value is the presented
         decision's option_id so the answer maps back to the decision
         verbatim (D9).

    Legacy ``questions`` strings are derived from these structured
    questions (dual-emit for old clients).
    """
    questions: list[ConfirmationQuestion] = []

    # Backend-preprocessed requests normally arrive complete. They may still
    # contain a gap when the LLM and deterministic fallback could not infer a
    # reasonable value, so actual blocking gaps must never be hidden merely
    # because a preprocess call happened.
    request = state.get("request")
    backend_preprocessed = bool(request and request.preparsed_candidates)

    # 1. Blocking gaps → one required TEXT question per gap (one-to-one).
    for gap in state.get("gaps", ()):
        if gap.gap_class not in ("critical", "safety_critical"):
            continue
        questions.append(
            ConfirmationQuestion(
                question_id=f"gap:{_stable_question_key(gap.recipe_id, gap.field_path)}",
                field_path=_gap_question_field_path(gap.recipe_id, gap.field_path),
                prompt=(
                    f"The {gap.field_path} for recipe '{gap.recipe_id}' is missing "
                    f"({gap.description}). Please provide the correct value."
                ),
                response_type=QuestionResponseType.TEXT,
                required=True,
                suggested_value=gap.current_value,
            )
        )

    # 2. Assumptions → one required CHOICE question per surfaced assumption.
    #    Skipped for backend-preprocessed requests (assumptions accepted).
    if not backend_preprocessed:
        for recipe_id, assumption in _confirmation_assumptions(state):
            questions.append(
                ConfirmationQuestion(
                    question_id=f"assumption:{_stable_question_key(recipe_id, assumption.text)}",
                    field_path=f"recipe.{recipe_id}.assumptions",
                    prompt=f"Assumption: {assumption.text}. Accept this suggested value?",
                    response_type=QuestionResponseType.CHOICE,
                    options=(
                        QuestionOption(value="accept", label="Accept suggested value", suggested=True),
                        QuestionOption(value="provide_alternative", label="Provide an alternative value"),
                    ),
                    required=True,
                    suggested_value=assumption.text,
                )
            )

    # 3. Repair options → strategy question when any plan-level repair
    # (purchase / reduce_servings / substitute) exists; otherwise one CHOICE
    # question per supported decision. When a strategy question is emitted,
    # per-item repair questions are never shown — the user decides at plan
    # level only.
    label_by_option_id = {option.option_id: option.description for option in repair_options}
    grouped_decision_ids = _append_repair_strategy_question(questions, decisions)
    if not grouped_decision_ids:
        for decision in decisions:
            label = label_by_option_id.get(decision.option_id, decision.option_type)
            questions.append(
                ConfirmationQuestion(
                    question_id=f"repair:{decision.option_id}",
                    field_path="repair_options",
                    prompt=f"Apply the repair option '{label}'?",
                    response_type=QuestionResponseType.CHOICE,
                    options=(
                        QuestionOption(value=decision.option_id, label="Apply", suggested=True),
                        QuestionOption(value="__skip__", label="Do not apply"),
                    ),
                    required=False,
                    suggested_value=decision.option_id,
                )
            )

    return tuple(questions)


def _confirmation_assumptions(state: PlanState) -> tuple[tuple[str, Assumption], ...]:
    """Assumptions that warrant a confirmation question (P4-02).

    - Parsed-recipe assumptions below the confidence threshold (uncertain
      inferences the user should confirm).
    - Every research-backed assumption: the graph only routes to
      confirmation when research warranted it (disagreement over threshold,
      no sources, unverifiable safety-critical value), so it always
      surfaces regardless of its numeric confidence.
    """
    result: list[tuple[str, Assumption]] = []
    for recipe in state.get("parsed_recipes", ()):
        for assumption in recipe.assumptions:
            if assumption.confidence < _ASSUMPTION_CONFIDENCE_THRESHOLD:
                result.append((recipe.recipe_id, assumption))
    for assumption in state.get("research_assumptions", ()):
        result.append(("_research", assumption))
    return tuple(result)


def render_confirmation_response(state: PlanState) -> ConfirmationPlanResponse:
    """Build a NEEDS_CONFIRMATION response with assumptions and repair options.

    Sources assumptions from parsed recipes (inferred fields) and repair
    options from the feasibility check. Generates user-facing questions.

    P0-06: also emits structured, client-submittable ApprovedDecisions so
    the client can resubmit them verbatim instead of opaque string IDs.

    P4-02: also emits ``confirmation_questions`` — a field-level structured
    form the client renders and answers directly. The legacy ``questions``
    strings are derived from it (dual-emit, deprecated since P4-02).

    Args:
        state: Workflow state at any CONFIRMATION transition.

    Returns:
        ConfirmationPlanResponse with assumptions, options, decisions,
        structured confirmation_questions, and legacy questions.
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

    # P0-06: plan revision — the confirmation the client will answer.
    plan_revision = f"{request.request_id}:v1"

    from cooking_plan_agent.repair.options import build_approved_decisions

    decisions = build_approved_decisions(repair_options, plan_revision)

    # P4-02: field-level structured form + legacy plain-string dual-emit.
    confirmation_questions = _build_confirmation_questions(state, repair_options, decisions)
    if confirmation_questions:
        questions = tuple(f"{q.prompt} ({q.question_id})" for q in confirmation_questions)
    else:
        # No field-level question is meaningful — keep the legacy fallback.
        questions = ("Would you like to proceed with these options?",)

    return ConfirmationPlanResponse(
        plan_id=request.request_id,
        status="NEEDS_CONFIRMATION",
        assumptions=tuple(assumptions),
        repair_options=repair_options,
        questions=questions,
        confirmation_questions=confirmation_questions,
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
            if not response.confirmation_questions:
                raise ValueError("CONFIRMATION response: must contain at least one actionable structured question")

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
