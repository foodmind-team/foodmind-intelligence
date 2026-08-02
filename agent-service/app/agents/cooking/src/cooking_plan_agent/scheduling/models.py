"""Scheduling domain models — problem input, solution output, verification.

Domain-neutral scheduler input/output.  All durations are
integer minutes because CP-SAT uses integer variables.

See also
--------
domain/models.py : CookingTask, KitchenResourceSnapshot, TaskDependency
domain/enums.py  : SolverStatus
"""

from cooking_plan_agent.domain.enums import SolverStatus
from cooking_plan_agent.domain.models import (
    CookingTask,
    KitchenResourceSnapshot,
    StrictModel,
)

# ============================================================================
# 7.2  SchedulingProblem — validated input for the CP-SAT scheduler
# ============================================================================


class SchedulingProblem(StrictModel):
    """A self-contained scheduling problem ready for CP-SAT model construction.

    All tasks, resources, and time limits are validated before reaching the
    solver.  Durations are integer minutes — never floats.
    """

    tasks: tuple[CookingTask, ...]
    """All tasks to schedule (must include recipe, prep, and safety tasks)."""

    resources: tuple[KitchenResourceSnapshot, ...]
    """Available kitchen resources (stove burners, ovens, sinks, etc.)."""

    requested_time_limit_minutes: int | None = None
    """Optional hard deadline.  If set, the solver enforces makespan <= this value."""

    solver_timeout_seconds: float = 10.0
    """Maximum wall-clock time for a single CP-SAT solve call."""


# ============================================================================
# 7.2  ScheduleResult — solver output consumed by downstream services
# ============================================================================


class ScheduledInterval(StrictModel):
    """A single task's placement on the global timeline.

    start_minute / end_minute are integer offsets from t=0 (start of cooking).
    """

    task_id: str
    """Maps back to CookingTask.task_id."""

    start_minute: int
    """Inclusive start time in minutes."""

    end_minute: int
    """Exclusive end time in minutes (start + duration)."""

    assigned_resource_ids: tuple[str, ...] = ()
    """Which specific resource instances this task occupies (e.g. 'stove:1')."""


class ScheduleResult(StrictModel):
    """The output of one CP-SAT solve attempt.

    Note: map CP-SAT statuses before exposing to callers.
    """

    status: SolverStatus
    """Mapped solver status — never raw CP-SAT enum."""

    makespan_minutes: int | None = None
    """Total elapsed time from start to last finish.  None if infeasible/unknown."""

    intervals: tuple[ScheduledInterval, ...] = ()
    """Every scheduled interval.  Empty if no feasible solution found."""

    wall_time_seconds: float = 0.0
    """Actual solver wall-clock time for this solve call."""

    best_objective_bound: int | None = None
    """Lower bound on the objective (makespan).  Only meaningful for FEASIBLE/OPTIMAL."""

    # --- Multi-objective extension (P3-03) ---
    optimization_phases: tuple[str, ...] = ()
    """Phases actually applied, in priority order (e.g. ("makespan", "holding", "context_switch"))."""

    holding_objective: int | None = None
    """Weighted holding-time value from Phase 2 (None when not run)."""

    context_switch_objective: int | None = None
    """Context-switch cost value from Phase 3 (None when not run)."""

    active_labour_objective: int | None = None
    """Active-labour value from Phase 4 (None when not run / no equivalent modes)."""


# ============================================================================
# 7.13  VerificationReport — independent correctness check
# ============================================================================


class VerificationIssue(StrictModel):
    """A single correctness violation found by the verifier."""

    code: str
    """Machine-readable issue code (e.g. 'MISSING_TASK', 'CAPACITY_EXCEEDED')."""

    message: str
    """Human-readable description of the violation."""

    task_ids: tuple[str, ...] = ()
    """Tasks involved in this violation, if any."""


class VerificationReport(StrictModel):
    """Independent verification report — does not use OR-Tools APIs.

    Note: the verifier must not reuse CP-SAT variables or assume
    the model was constructed correctly.
    """

    passed: bool
    """True only if zero issues were found."""

    issues: tuple[VerificationIssue, ...]
    """Every correctness violation found.  Empty means the schedule passed."""

    checked_task_count: int = 0
    """How many tasks were verified."""

    checked_resource_count: int = 0
    """How many resources were verified."""
