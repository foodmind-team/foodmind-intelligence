"""Pure routing functions for LangGraph conditional edges.

Per handbook 8.6: each function returns explicit node name literals.
No side effects, no service calls — only state inspection.
"""

from typing import Literal

from cooking_plan_agent.workflow.state import PlanState

# ---------------------------------------------------------------------------
# 8.6 Routing after gap detection
# ---------------------------------------------------------------------------


def route_after_gap_detection(
    state: PlanState,
) -> Literal["infer_local", "validate_recipe_ir"]:
    """If gaps exist, try local inference before validation.

    Otherwise, proceed directly to IR validation.
    The gap detection may return empty tuples — only non-empty gaps trigger inference.
    """
    if state.get("gaps"):
        return "infer_local"
    return "validate_recipe_ir"


# ---------------------------------------------------------------------------
# 8.6 Routing after local inference
# ---------------------------------------------------------------------------


def route_after_local_inference(
    state: PlanState,
) -> Literal["research_missing", "build_confirmation_response", "validate_recipe_ir"]:
    """After local inference, remaining critical gaps route to:
    - web research (if enabled and gap is heat/duration related)
    - confirmation (if evidence insufficient or research disabled)
    - IR validation (if gaps resolved).

    Only "critical" and "safety_critical" gaps block progress — minor gaps
    (e.g., garnish variation) are tolerated and passed through.
    """
    gaps = state.get("gaps", ())
    critical_gaps = [g for g in gaps if g.gap_class in ("critical", "safety_critical")]

    if not critical_gaps:
        return "validate_recipe_ir"

    # Check if web research is enabled via Settings (handbook 10.1)
    from cooking_plan_agent.config.settings import get_settings

    settings = get_settings()
    if not settings.web_research_enabled:
        # Research disabled — all critical gaps → confirmation
        return "build_confirmation_response"

    # Only route to research for heat/duration/temperature gaps (handbook 10.1)
    _researchable_fields = {"heat_level", "duration", "temperature", "target_temperature_c"}
    researchable = [g for g in critical_gaps if any(f in g.field_path.lower() for f in _researchable_fields)]

    if researchable:
        return "research_missing"

    # Non-researchable critical gaps → confirmation
    return "build_confirmation_response"


# ---------------------------------------------------------------------------
# 8.6 Routing after safety validation
# ---------------------------------------------------------------------------


def route_after_safety(
    state: PlanState,
) -> Literal["check_feasibility", "render_infeasible_response"]:
    """Hard unrepairable safety findings -> INFEASIBLE.

    Otherwise, proceed to feasibility check.
    Safety findings that ARE repairable are injected as safety_tasks in merge_preparation.
    """
    safety_report = state.get("safety_report")
    if safety_report is not None and safety_report.has_unrepairable:
        return "render_infeasible_response"
    return "check_feasibility"


# ---------------------------------------------------------------------------
# 8.6 Routing after feasibility check
# ---------------------------------------------------------------------------


def route_after_feasibility(
    state: PlanState,
) -> Literal["merge_preparation", "build_confirmation_response", "render_infeasible_response"]:
    """If infeasible: confirmation with repair options (if any) or INFEASIBLE.

    A None report means feasibility was not checked (e.g., safety short-circuited)
    — in that case, default to merge_preparation (downstream nodes handle it).
    """
    report = state.get("feasibility_report")
    if report is None:
        return "merge_preparation"

    if not report.is_feasible:
        repair_options = state.get("repair_options", ())
        if repair_options:
            return "build_confirmation_response"
        return "render_infeasible_response"

    return "merge_preparation"


# ---------------------------------------------------------------------------
# 8.6 Routing after schedule solve
# ---------------------------------------------------------------------------


def route_after_solve(
    state: PlanState,
) -> Literal["verify_schedule", "render_infeasible_response", "render_failed_response"]:
    """Solver result determines next step.

    None result = solver errored without producing output -> FAILED.
    OPTIMAL/FEASIBLE proceed to independent verification.
    INFEASIBLE means the solver proved no solution exists.
    MODEL_INVALID/UNKNOWN -> FAILED (likely a model construction bug).
    """
    result = state.get("schedule_result")
    if result is None:
        return "render_failed_response"

    status = result.status
    if status in ("OPTIMAL", "FEASIBLE"):
        return "verify_schedule"
    if status == "INFEASIBLE":
        return "render_infeasible_response"
    # MODEL_INVALID, UNKNOWN -> FAILED
    return "render_failed_response"


# ---------------------------------------------------------------------------
# 8.6 Routing after verification
# ---------------------------------------------------------------------------


def route_after_verification(
    state: PlanState,
) -> Literal["render_ready_response", "render_failed_response"]:
    """Verification passes -> READY; fails -> FAILED.

    The verifier checks constraint satisfaction independently from the solver,
    catching optimiser bugs before they reach the user.
    """
    report = state.get("verification_report")
    if report is not None and report.passed:
        return "render_ready_response"
    return "render_failed_response"
