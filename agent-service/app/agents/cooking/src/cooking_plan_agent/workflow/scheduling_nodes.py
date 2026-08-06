"""Workflow node implementations for a single pipeline stage.

The public compatibility surface remains ``cooking_plan_agent.workflow.nodes``.
This module contains one cohesive stage only.
"""

import logging

from langgraph.runtime import Runtime

from cooking_plan_agent.domain.errors import DomainErrorCode
from cooking_plan_agent.domain.models import (
    WorkflowError,
)
from cooking_plan_agent.workflow.context import WorkflowContext
from cooking_plan_agent.workflow.state import PlanState

logger = logging.getLogger(__name__)


def _solver_timeout() -> float:
    """Return the configured CP-SAT solver timeout in seconds."""
    from cooking_plan_agent.config.settings import get_settings

    return get_settings().solver_timeout_seconds


async def build_task_graph_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, object]:
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
        # Cycle detection or invalid dependencies -> workflow error. P2-03:
        # only the exception type is retained as diagnostic context.
        return {
            "error": WorkflowError(
                error_code=DomainErrorCode.TASK_GRAPH_CYCLE.value,
                message="Task graph construction failed",
                correlation_id=state["request"].request_id,
                node_name="build_task_graph",
                diagnostics={"exception_type": type(exc).__name__},
            )
        }


async def solve_schedule_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, object]:
    """Solve the CP-SAT scheduling problem.

    Wired to existing scheduling/orchestrator.py.
    schedule() returns tuple[ScheduleResult, VerificationReport] — we store
    only the result; verification is done independently in verify_schedule_node.

    Error semantics (P1-04): ``SCHEDULE_INFEASIBLE`` means ONLY that the
    solver proved no solution exists for a VALID model. Everything else uses
    a distinct code:
      - MODEL_INVALID → SCHEDULE_MODEL_INVALID (model construction bug)
      - UNKNOWN       → SCHEDULE_UNKNOWN (solver hit its limit, undetermined)
      - missing task graph → INTERNAL_ERROR (invariant break, never INFEASIBLE)
      - ValueError/TypeError during solve → SCHEDULE_MODEL_INVALID
      - RuntimeError from the solver → INTERNAL_ERROR
    """
    import asyncio

    from cooking_plan_agent.domain.enums import SolverStatus
    from cooking_plan_agent.scheduling.models import SchedulingProblem
    from cooking_plan_agent.scheduling.orchestrator import ScheduleOrchestrator

    task_graph = state.get("task_graph")
    request = state["request"]
    if task_graph is None:
        # Missing DAG is an internal invariant failure, not a business
        # infeasibility — solve must never run without a task graph.
        return {
            "error": WorkflowError(
                error_code=DomainErrorCode.INTERNAL_ERROR.value,
                message="No task graph available for scheduling — internal invariant violated",
                correlation_id=request.request_id,
                node_name="solve_schedule",
            )
        }

    problem = SchedulingProblem(
        tasks=task_graph.tasks,
        resources=request.kitchen_resources,
        requested_time_limit_minutes=request.time_limit_minutes,
        solver_timeout_seconds=_solver_timeout(),
    )

    try:
        # CP-SAT solving is CPU-bound — run it in a worker thread so the
        # event loop stays responsive (P1-02). The verifier is synchronous
        # and stays inside the solve call; it is not moved to a thread.
        # P3-03: ScheduleOrchestrator runs the lexicographic phases
        # (makespan → holding → context switch); Phase 4 stays gated.
        # The depth is configurable for rollback (solver_optimization_level).
        from cooking_plan_agent.config.settings import get_settings

        overrides = state.get("solver_overrides", {})
        optimization_level = str(
            overrides.get("optimization_level") or get_settings().solver_optimization_level
        )

        orchestrator = ScheduleOrchestrator()
        result, _ = await asyncio.to_thread(
            orchestrator.solve,
            problem,
            optimization_level,
        )
    except (ValueError, TypeError) as exc:
        # Model-construction phase: bad variable shapes, contradictory
        # constraints → the model was never valid. P2-03: keep only the
        # exception type as diagnostic context.
        return {
            "error": WorkflowError(
                error_code=DomainErrorCode.SCHEDULE_MODEL_INVALID.value,
                message="Scheduling model construction failed",
                correlation_id=request.request_id,
                node_name="solve_schedule",
                diagnostics={"exception_type": type(exc).__name__},
            )
        }
    except RuntimeError as exc:
        # Solver-internal failure (runtime) — not a business outcome.
        return {
            "error": WorkflowError(
                error_code=DomainErrorCode.INTERNAL_ERROR.value,
                message="Scheduling solver failed",
                correlation_id=request.request_id,
                node_name="solve_schedule",
                diagnostics={"exception_type": type(exc).__name__},
            )
        }

    # Map solver status to a stable, distinct error code. Only INFEASIBLE is a
    # business outcome (routes to render_infeasible_response); MODEL_INVALID
    # and UNKNOWN are FAILED responses (P1-04).
    status = result.status
    if status == SolverStatus.MODEL_INVALID:
        return {
            "error": WorkflowError(
                error_code=DomainErrorCode.SCHEDULE_MODEL_INVALID.value,
                message="The scheduling model is invalid — likely a data inconsistency",
                correlation_id=request.request_id,
                node_name="solve_schedule",
            )
        }
    if status == SolverStatus.UNKNOWN:
        return {
            "error": WorkflowError(
                error_code=DomainErrorCode.SCHEDULE_UNKNOWN.value,
                message="The solver could not determine feasibility within the time limit",
                correlation_id=request.request_id,
                node_name="solve_schedule",
            )
        }

    return {"schedule_result": result}


async def verify_schedule_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, object]:
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
                error_code=DomainErrorCode.SCHEDULE_VERIFICATION_FAILED.value,
                message="No schedule result to verify",
                correlation_id=state["request"].request_id,
                node_name="verify_schedule",
            )
        }

    task_graph = state.get("task_graph")
    if task_graph is None:
        return {
            "error": WorkflowError(
                error_code=DomainErrorCode.SCHEDULE_VERIFICATION_FAILED.value,
                message="No task graph available for verification",
                correlation_id=state["request"].request_id,
                node_name="verify_schedule",
            )
        }

    problem = SchedulingProblem(
        tasks=task_graph.tasks,
        resources=state["request"].kitchen_resources,
        requested_time_limit_minutes=state["request"].time_limit_minutes,
        solver_timeout_seconds=_solver_timeout(),
    )

    try:
        verifier = ScheduleVerifier()
        report = verifier.verify(problem, schedule_result)
        return {"verification_report": report}
    except (ValueError, TypeError, RuntimeError) as exc:
        # Verification failure is an invariant break — the solver output
        # must never reach the client. P2-03: public text comes from the
        # catalog; keep only the exception type as diagnostic context.
        return {
            "error": WorkflowError(
                error_code=DomainErrorCode.SCHEDULE_VERIFICATION_FAILED.value,
                message="Schedule verification failed",
                correlation_id=state["request"].request_id,
                node_name="verify_schedule",
                diagnostics={"exception_type": type(exc).__name__},
            )
        }


# ============================================================================
# P4-01: schedule explanation (between verify and READY render)
# ============================================================================


def _build_schedule_summary(state: PlanState) -> dict[str, object]:
    """Build the compact, non-sensitive summary the explainer consumes (D3/D4).

    Only facts already present in the verified schedule are included:
    makespan minutes, per-dish completion minutes, and the maximum number of
    concurrently ACTIVE tasks (parallel groups). No recipe text, inventory,
    or user identity is ever included.
    """
    from cooking_plan_agent.rendering.builder import build_dish_completion_summary

    schedule = state.get("schedule_result")
    makespan: int = (schedule.makespan_minutes or 0) if schedule is not None else 0

    dish_completions: list[dict[str, object]] = []
    if schedule is not None:
        task_graph = state.get("task_graph")
        tasks = task_graph.tasks if task_graph is not None else ()
        for entry in build_dish_completion_summary(schedule, tasks):
            # builder emits "dish_id"; the explainer consumes "dish".
            raw_completion = entry.get("completion_minute")
            dish_completions.append(
                {
                    "dish": str(entry.get("dish_id") or "?"),
                    "completion_minute": int(raw_completion) if isinstance(raw_completion, int) else 0,
                }
            )

    return {
        "makespan_minutes": makespan,
        "dish_completions": dish_completions,
        "parallel_groups": _max_parallel_active(state),
    }


def _max_parallel_active(state: PlanState) -> int:
    """Maximum number of concurrently ACTIVE tasks across the timeline (D3).

    A simple sweep over (start, end) events of ACTIVE tasks gives the peak
    concurrency. Falls back to 0 when no schedule/timeline is available.
    """
    from cooking_plan_agent.domain.enums import WorkMode
    from cooking_plan_agent.rendering.builder import build_timeline

    schedule = state.get("schedule_result")
    if schedule is None:
        return 0

    task_graph = state.get("task_graph")
    tasks = task_graph.tasks if task_graph is not None else ()
    events: list[tuple[int, int]] = []  # (minute, +1 start / -1 end)
    for entry in build_timeline(schedule, tasks):
        if entry.get("work_mode") != WorkMode.ACTIVE.value:
            continue
        raw_start = entry.get("start_minute")
        raw_end = entry.get("end_minute")
        start = int(raw_start) if isinstance(raw_start, int) else 0
        end = int(raw_end) if isinstance(raw_end, int) else start
        events.append((start, 1))
        events.append((end, -1))
    # Half-open intervals [start, end): at a shared boundary the ending task
    # is already done before the starting task begins, so -1 sorts before +1.
    events.sort(key=lambda event: (event[0], event[1]))
    current = 0
    peak = 0
    for _minute, delta in events:
        current += delta
        peak = max(peak, current)
    return peak


def _deterministic_explanation(summary: dict[str, object]) -> str:
    """Deterministic fallback: re-states only verified schedule facts (D3).

    Used when the LLM explainer is absent or fails. The content is always
    derived from the summary — no new claims are introduced.
    """
    makespan = summary.get("makespan_minutes")
    parts = [f"Plan completes in approximately {makespan} minutes."]
    raw_completions = summary.get("dish_completions")
    completions = raw_completions if isinstance(raw_completions, list) else []
    if completions:
        parts.append(
            "Dishes finish at: "
            + ", ".join(
                f"{entry.get('dish', '?')} at {entry.get('completion_minute', '?')} min" for entry in completions
            )
        )
    return " ".join(parts)


async def explain_schedule_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, object]:
    """Attach a short, additive explanation to a verified schedule (P4-01).

    Placed between verify_schedule and render_ready_response. The node NEVER
    writes a WorkflowError: an absent explainer, LLM timeout, malformed output
    or any exception degrades to a deterministic summary, so the verified
    READY response is never blocked (P2-02 fault matrix).

    Returns state fields:
      - explanation: prose or None (feature disabled).
      - explanation_source: "llm" | "deterministic" | "disabled".
    """
    from cooking_plan_agent.config.settings import get_settings

    if not get_settings().explanation_enabled:
        return {"explanation": None, "explanation_source": "disabled"}

    summary = _build_schedule_summary(state)
    explainer = runtime.context.explainer
    if explainer is not None:
        try:
            text = await explainer.explain(summary)
            if isinstance(text, str) and text.strip():
                return {"explanation": text, "explanation_source": "llm"}
        except Exception:  # noqa: BLE001 — additive capability must never fail READY
            logger.warning("Schedule explanation failed — using deterministic fallback")

    return {
        "explanation": _deterministic_explanation(summary),
        "explanation_source": "deterministic",
    }
