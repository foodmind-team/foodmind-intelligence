"""CP-SAT model construction — variables, intervals, constraints, objective.

Handbook 7.3–7.7: Stages A through E build the model in small, testable layers.

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
from cooking_plan_agent.scheduling.models import SchedulingProblem

# ============================================================================
# 7.3  ScheduleModelBuilder — creates CP-SAT model from SchedulingProblem
# ============================================================================


class ScheduleModelBuilder:
    """Creates a CP-SAT CpModel with variables and constraints for scheduling.

    Handbook 7.1: this class is responsible for model construction only.
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
        self._horizon = self._compute_horizon(problem.tasks)

        # Stage A: create interval variables for every task
        self._build_variables(problem.tasks)

        # Stage B: enforce precedence and lag constraints
        self._build_precedence(problem.tasks)

        # Stage C: single active cook — no_overlap for ACTIVE tasks
        self._build_cook_no_overlap(problem.tasks)

        # Stage D: reusable resource constraints (cumulative)
        self._build_resources(problem)

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

    def _compute_horizon(self, tasks: tuple[CookingTask, ...]) -> int:
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
        total_lag = sum(
            dep.minimum_lag_minutes
            for task in tasks
            for dep in task.dependencies
        )
        return base + total_lag

    def _build_variables(self, tasks: tuple[CookingTask, ...]) -> None:
        """Stage A: create start, end, and interval variables per task.

        Handbook 7.3: store in dicts keyed by task_id.
        """
        model = self._model
        assert model is not None

        for task in tasks:
            tid = task.task_id
            start = model.new_int_var(0, self._horizon, f"start:{tid}")
            end = model.new_int_var(0, self._horizon, f"end:{tid}")
            interval = model.new_interval_var(
                start,
                task.duration_minutes,
                end,
                f"interval:{tid}",
            )
            self._starts[tid] = start
            self._ends[tid] = end
            self._intervals[tid] = interval

    # ------------------------------------------------------------------
    # 7.4 Stage B — Precedence and lag constraints
    # ------------------------------------------------------------------

    def _build_precedence(self, tasks: tuple[CookingTask, ...]) -> None:
        """Stage B: for each TaskDependency, add start >= end + lag.

        Handbook 7.4:
        - Minimum lag: successor.start >= predecessor.end + minimum_lag
        - Maximum lag: successor.start <= predecessor.end + maximum_lag
        """
        model = self._model
        assert model is not None

        for task in tasks:
            for dep in task.dependencies:
                pred_id = dep.predecessor_id
                succ_id = task.task_id

                if pred_id not in self._starts or succ_id not in self._starts:
                    continue  # Skip dependencies on tasks not in this problem

                # Minimum lag: successor starts at or after predecessor ends + lag
                model.add(
                    self._starts[succ_id]
                    >= self._ends[pred_id] + dep.minimum_lag_minutes
                )

                # Maximum lag: successor must start by predecessor end + max_lag
                if dep.maximum_lag_minutes is not None:
                    model.add(
                        self._starts[succ_id]
                        <= self._ends[pred_id] + dep.maximum_lag_minutes
                    )

    # ------------------------------------------------------------------
    # 7.5 Stage C — Single active cook (no_overlap)
    # ------------------------------------------------------------------

    def _build_cook_no_overlap(self, tasks: tuple[CookingTask, ...]) -> None:
        """Stage C: active tasks cannot overlap — the human cook is single-threaded.

        Handbook 7.5: passive intervals are NOT added to this list.
        They still occupy their cookware/appliance resources (Stage D).
        """
        model = self._model
        assert model is not None

        active_intervals = [
            self._intervals[t.task_id]
            for t in tasks
            if t.work_mode == WorkMode.ACTIVE
        ]

        if active_intervals:
            model.add_no_overlap(active_intervals)

    # ------------------------------------------------------------------
    # 7.6 Stage D — Reusable resources
    # ------------------------------------------------------------------

    def _build_resources(self, problem: SchedulingProblem) -> None:
        """Stage D: enforce resource capacity via cumulative constraints.

        For each resource type with interchangeable identical capacity,
        group tasks that require that type and add add_cumulative.

        Handbook 7.6: Implement identical resources first.
        """
        model = self._model
        assert model is not None

        # Group resources by type and compute total available capacity.
        resource_capacity: dict[str, int] = {}
        resource_available: dict[str, bool] = {}
        for r in problem.resources:
            if r.resource_type not in resource_capacity:
                resource_capacity[r.resource_type] = 0
                resource_available[r.resource_type] = True
            resource_capacity[r.resource_type] += int(r.capacity) if r.capacity else 1
            if not r.available:
                resource_available[r.resource_type] = False

        # Build cumulative constraints per resource type.
        for res_type, capacity in resource_capacity.items():
            if not resource_available[res_type]:
                continue

            # Collect tasks that require this resource type.
            res_intervals: list[cp_model.IntervalVar] = []
            res_demands: list[int] = []

            for task in problem.tasks:
                for need in task.resources:
                    if need.resource_type == res_type:
                        res_intervals.append(self._intervals[task.task_id])
                        res_demands.append(need.quantity)
                        break  # Each task counts once per resource type

            if res_intervals:
                model.add_cumulative(
                    intervals=res_intervals,
                    demands=res_demands,
                    capacity=capacity,
                )

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

    Handbook 7.1: separates model construction (builder) from solving (solver)
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
