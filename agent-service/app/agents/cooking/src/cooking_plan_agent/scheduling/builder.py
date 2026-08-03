"""CP-SAT model construction — variables, intervals, constraints, objective.

Design decisions:
- Horizon = sum of all task durations (safe but loose upper bound).
- Variables are stored in dicts keyed by task_id — never depend on list position.
- Only ACTIVE tasks are added to the cook's no_overlap constraint (7.5).
- Reusable resources use add_cumulative (7.6).
- Makespan is computed as the max of all task ends (or only final dish tasks).
"""

from ortools.sat.python import cp_model

from cooking_plan_agent.domain.enums import WorkMode
from cooking_plan_agent.domain.models import (
    CookingTask,
)
from cooking_plan_agent.normalisation.names import normalise_resource_type
from cooking_plan_agent.scheduling.models import SchedulingProblem

# ============================================================================
# 7.3  ScheduleModelBuilder — creates CP-SAT model from SchedulingProblem
# ============================================================================


class ScheduleModelBuilder:
    """Creates a CP-SAT CpModel with variables and constraints for scheduling.

    Note: this class is responsible for model construction only.
    Solving and extraction are handled by separate classes.

    Usage::
        builder = ScheduleModelBuilder()
        model_info = builder.build(problem)
        # model_info contains the CpModel, variable dicts, and horizon
    """

    def __init__(self) -> None:
        self._model: cp_model.CpModel | None = None
        self._starts: dict[str, cp_model.IntVar] = {}
        self._ends: dict[str, cp_model.IntVar] = {}
        self._intervals: dict[str, cp_model.IntervalVar] = {}
        self._horizon: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, problem: SchedulingProblem) -> "ModelInfo":
        """Construct the full CP-SAT model from a scheduling problem.

        Stages A through E are applied in order.  The returned ModelInfo
        wraps the CpModel and variable dictionaries for extraction.

        Args:
            problem: Validated scheduling problem with tasks and resources.

        Returns:
            ModelInfo with CpModel, variable dicts, and horizon.
        """
        self._model = cp_model.CpModel()
        self._starts = {}
        self._ends = {}
        self._intervals = {}
        self._horizon = self.compute_horizon(problem.tasks)

        # Stages A–D: variables and constraints (reusable)
        self._starts, self._ends, self._intervals = self.build_constraints(self._model, problem, self._horizon)

        # Stage E: makespan objective
        self._build_objective(problem)

        return ModelInfo(
            model=self._model,
            starts=self._starts,
            ends=self._ends,
            intervals=self._intervals,
            horizon=self._horizon,
            makespan_var=self._makespan_var,
        )

    # ------------------------------------------------------------------
    # 7.3 Stage A — Variables and intervals
    # ------------------------------------------------------------------

    def compute_horizon(self, tasks: tuple[CookingTask, ...]) -> int:
        """Compute a safe upper bound: sum of all durations + all lags.

        The bare sum of durations is insufficient when tasks have
        minimum_lag_minutes dependencies — those lags extend the
        required horizon.  We compute a worst-case chain to ensure
        the horizon is always feasible.

        This is a loose bound — the real makespan will be much shorter
        after resource contention is resolved.  The solver prunes
        infeasible values automatically.
        """
        base = sum(t.duration_minutes for t in tasks)
        # Add the maximum minimum lag across all dependencies.
        max_lag = 0
        for task in tasks:
            for dep in task.dependencies:
                max_lag = max(max_lag, dep.minimum_lag_minutes)
        # Each dependency could add lag to the critical path.
        # A simple but safe bound: base + max_lag * number of dependencies.
        # Even safer: sum all lags across all edges.
        total_lag = sum(dep.minimum_lag_minutes for task in tasks for dep in task.dependencies)
        return base + total_lag

    def build_constraints(
        self,
        model: cp_model.CpModel,
        problem: SchedulingProblem,
        horizon: int,
    ) -> tuple[
        dict[str, cp_model.IntVar],
        dict[str, cp_model.IntVar],
        dict[str, cp_model.IntervalVar],
    ]:
        """Build Stages A–D (variables + constraints) on an existing model.

        Does NOT set an objective — the caller is responsible for Stage E.
        Returns variable dicts for downstream use (e.g. Phase 2 objectives).
        """
        starts: dict[str, cp_model.IntVar] = {}
        ends: dict[str, cp_model.IntVar] = {}
        intervals: dict[str, cp_model.IntervalVar] = {}

        # Stage A: create start, end, and interval variables per task
        for task in problem.tasks:
            tid = task.task_id
            start = model.new_int_var(0, horizon, f"start:{tid}")
            end = model.new_int_var(0, horizon, f"end:{tid}")
            iv = model.new_interval_var(
                start,
                task.duration_minutes,
                end,
                f"interval:{tid}",
            )
            starts[tid] = start
            ends[tid] = end
            intervals[tid] = iv

        # Stage B: enforce precedence and lag constraints
        for task in problem.tasks:
            for dep in task.dependencies:
                pred_id = dep.predecessor_id
                succ_id = task.task_id

                if pred_id not in starts or succ_id not in starts:
                    continue  # Skip dependencies on tasks not in this problem

                # Minimum lag: successor.start >= predecessor.end + min_lag
                model.add(starts[succ_id] >= ends[pred_id] + dep.minimum_lag_minutes)

                # Maximum lag: successor.start <= predecessor.end + max_lag
                if dep.maximum_lag_minutes is not None:
                    model.add(starts[succ_id] <= ends[pred_id] + dep.maximum_lag_minutes)

        # Stage C: single active cook — no_overlap for ACTIVE tasks
        # Passive tasks are NOT included (they don't occupy the cook).
        active_ivs = [intervals[t.task_id] for t in problem.tasks if t.work_mode == WorkMode.ACTIVE]
        if active_ivs:
            model.add_no_overlap(active_ivs)

        # Stage D: enforce resource capacity via cumulative constraints
        resource_capacity: dict[str, int] = {}
        resource_available: dict[str, bool] = {}
        for r in problem.resources:
            resource_type = normalise_resource_type(r.resource_type)
            if resource_type not in resource_capacity:
                resource_capacity[resource_type] = 0
                resource_available[resource_type] = True
            resource_capacity[resource_type] += int(r.capacity) if r.capacity else 1
            if not r.available:
                resource_available[resource_type] = False

        for res_type, capacity in resource_capacity.items():
            if not resource_available[res_type]:
                continue

            res_ivs: list[cp_model.IntervalVar] = []
            res_demands: list[int] = []

            for task in problem.tasks:
                for need in task.resources:
                    if normalise_resource_type(need.resource_type) == res_type:
                        res_ivs.append(intervals[task.task_id])
                        res_demands.append(need.quantity)
                        break  # Each task counts once per resource type

            if res_ivs:
                model.add_cumulative(
                    intervals=res_ivs,
                    demands=res_demands,
                    capacity=capacity,
                )

        return starts, ends, intervals

    # ------------------------------------------------------------------
    # 7.7 Stage E — Makespan objective
    # ------------------------------------------------------------------

    def _build_objective(self, problem: SchedulingProblem) -> None:
        """Stage E: minimise makespan = max(all task end times).

        Handbook 7.7: use final dish completion tasks when available,
        otherwise use all task ends.

        If requested_time_limit_minutes is set, add a hard constraint.
        """
        model = self._model
        assert model is not None

        # Identify final tasks — those that are not predecessors of any other task.
        predecessor_ids: set[str] = set()
        for task in problem.tasks:
            for dep in task.dependencies:
                predecessor_ids.add(dep.predecessor_id)

        all_task_ids = {t.task_id for t in problem.tasks}
        final_task_ids = all_task_ids - predecessor_ids
        if not final_task_ids:
            final_task_ids = all_task_ids  # Fallback: use all tasks

        final_ends = [self._ends[tid] for tid in final_task_ids]

        makespan = model.new_int_var(0, self._horizon, "makespan")
        model.add_max_equality(makespan, final_ends)
        model.minimize(makespan)

        # Hard deadline if requested
        if problem.requested_time_limit_minutes is not None:
            model.add(makespan <= problem.requested_time_limit_minutes)

        self._makespan_var = makespan


# ============================================================================
# ModelInfo — container for CP-SAT model and variable references
# ============================================================================


class ModelInfo:
    """Container for a constructed CP-SAT model and its variable dictionaries.

    Note: separates model construction (builder) from solving (solver)
    and extraction (extractor).  This class bridges the gap.
    """

    def __init__(
        self,
        model: cp_model.CpModel,
        starts: dict[str, cp_model.IntVar],
        ends: dict[str, cp_model.IntVar],
        intervals: dict[str, cp_model.IntervalVar],
        horizon: int,
        makespan_var: cp_model.IntVar,
    ) -> None:
        self.model = model
        self.starts = starts
        self.ends = ends
        self.intervals = intervals
        self.horizon = horizon
        self.makespan_var = makespan_var
