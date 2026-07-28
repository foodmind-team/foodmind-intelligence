"""Schedule orchestrator — lexicographic multi-objective optimisation.

Handbook 7.9–7.12: sequential solves for makespan, holding, context switching,
and active labour.  Each later optimisation MUST NOT increase the fixed value
from earlier optimisations.

Design:
- Phase 1: minimise makespan
- Phase 2: minimise dish holding time (7.9)
- Phase 3: minimise context switching (7.10 — defer to post-tie-break in MVP)
- Phase 4: minimise active labour (7.11 — defer until alternative modes exist)

This module also provides the top-level ``schedule()`` convenience function
that ties together builder → solver → extractor → verifier.
"""

import time

from ortools.sat.python import cp_model

from cooking_plan_agent.domain.enums import SolverStatus
from cooking_plan_agent.scheduling.builder import ModelInfo, ScheduleModelBuilder
from cooking_plan_agent.scheduling.extractor import ScheduleExtractor
from cooking_plan_agent.scheduling.models import (
    ScheduleResult,
    SchedulingProblem,
)
from cooking_plan_agent.scheduling.solver import ScheduleSolver, SolverRun
from cooking_plan_agent.scheduling.verifier import ScheduleVerifier, VerificationReport

# ============================================================================
# Top-level schedule() — convenience function for the full pipeline
# ============================================================================


def schedule(problem: SchedulingProblem) -> tuple[ScheduleResult, VerificationReport]:
    """Run the full scheduling pipeline: build → solve → extract → verify.

    This is a convenience function for simple cases.  For multi-objective
    optimisation, use ``ScheduleOrchestrator`` directly.

    Args:
        problem: Validated scheduling problem.

    Returns:
        A tuple of (ScheduleResult, VerificationReport).
    """
    builder = ScheduleModelBuilder()
    solver = ScheduleSolver()
    extractor = ScheduleExtractor()
    verifier = ScheduleVerifier()

    model_info = builder.build(problem)
    solver_run = solver.solve(model_info, problem.solver_timeout_seconds)

    # Build base result with status and makespan
    result = solver.to_schedule_result(solver_run)

    # Extract intervals only for feasible solutions
    if result.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE):
        intervals = extractor.extract(solver_run)
        result = result.model_copy(
            update={"intervals": intervals}
        )

    report = verifier.verify(problem, result)
    return result, report


# ============================================================================
# ScheduleOrchestrator — lexicographic multi-objective solve
# ============================================================================


class ScheduleOrchestrator:
    """Orchestrates lexicographic multi-objective solves.

    Handbook 7.12: sequential solves, never combine objectives with arbitrary
    weights.  Each later solve fixes the earlier objective value as a hard
    constraint, guaranteeing priority ordering.

    Usage::

        orchestrator = ScheduleOrchestrator()
        result, report = orchestrator.solve(problem)
    """

    def __init__(self) -> None:
        self._builder = ScheduleModelBuilder()
        self._solver = ScheduleSolver()
        self._extractor = ScheduleExtractor()
        self._verifier = ScheduleVerifier()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        problem: SchedulingProblem,
    ) -> tuple[ScheduleResult, VerificationReport]:
        """Run Phase 1 (makespan) + Phase 2 (holding minimisation).

        Phases 3 (context switching) and 4 (active labour) are deferred
        per Handbook 7.10–7.11.

        Returns:
            (best_feasible_result, verification_report)
        """
        # Phase 1: minimise makespan
        result = self._phase_makespan(problem)
        if result.status not in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE):
            report = ScheduleVerifier().verify(problem, result)
            return result, report

        makespan = result.makespan_minutes
        assert makespan is not None

        # Phase 2: minimise holding (keep makespan fixed)
        result = self._phase_holding(problem, makespan, result)

        report = self._verifier.verify(problem, result)
        return result, report

    # ------------------------------------------------------------------
    # Phase 1: minimise makespan
    # ------------------------------------------------------------------

    def _phase_makespan(
        self,
        problem: SchedulingProblem,
    ) -> ScheduleResult:
        """Build and solve the basic makespan-minimisation model.

        Returns a ScheduleResult with intervals extracted if feasible.
        """
        model_info = self._builder.build(problem)
        solver_run = self._solver.solve(model_info, problem.solver_timeout_seconds)
        result = self._solver.to_schedule_result(solver_run)

        if result.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE):
            intervals = self._extractor.extract(solver_run)
            result = result.model_copy(update={"intervals": intervals})

        return result

    # ------------------------------------------------------------------
    # 7.9  Phase 2: minimise dish holding time
    # ------------------------------------------------------------------

    def _phase_holding(
        self,
        problem: SchedulingProblem,
        makespan: int,
        phase1_result: ScheduleResult,
    ) -> ScheduleResult:
        """Minimise weighted holding time while keeping makespan fixed.

        Handbook 7.9: holding cost H = sum_d w_d * (T* - C_d) where:
        - T* is the fixed minimum makespan
        - C_d is dish d's completion time
        - w_d is a weight (higher = serve sooner)

        This moves heat-sensitive dishes later without increasing makespan.
        """
        model = cp_model.CpModel()

        # Re-use the builder for basic variable creation (stages A-D).
        phase1_model = self._builder.build(problem)
        horizon = phase1_model.horizon

        # Reconstruct the same model, but now fix makespan.
        starts, ends, intervals = self._rebuild_model(
            model, problem, horizon, makespan
        )

        # --- Holding objective ---
        # Group tasks by dish_id and compute a per-dish completion time.
        dish_final_tasks: dict[str, list[str]] = {}
        for task in problem.tasks:
            dish_final_tasks.setdefault(task.dish_id, []).append(task.task_id)

        # The completion time of a dish is the max end time of its tasks.
        # We create a completion variable per dish and minimise sum of
        # (makespan - completion), weighted by dish priority.
        # For MVP, all dishes have equal weight = 1.
        dish_holding_terms: list[cp_model.LinearExpr] = []

        for dish_id, task_ids in dish_final_tasks.items():
            dish_end_vars = [ends[tid] for tid in task_ids if tid in ends]
            if not dish_end_vars:
                continue
            dish_completion = model.new_int_var(0, horizon, f"completion:{dish_id}")
            model.add_max_equality(dish_completion, dish_end_vars)

            # Holding penalty: (makespan - completion).  Minimising sum of
            # these pushes dish completions later toward makespan, which
            # reduces holding time.
            holding = model.new_int_var(0, horizon, f"holding:{dish_id}")
            model.add(holding == makespan - dish_completion)
            dish_holding_terms.append(holding)

        if dish_holding_terms:
            total_holding = sum(dish_holding_terms)
            model.minimize(total_holding)

        # Solve Phase 2
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = problem.solver_timeout_seconds
        solver.parameters.num_search_workers = 4

        start_time = time.monotonic()
        cp_status = solver.solve(model)
        elapsed = time.monotonic() - start_time

        status = self._solver.map_status(cp_status)

        # If Phase 2 succeeded, extract the new intervals.
        # If it failed (timeout etc.), return Phase 1 result.
        if status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE):
            run = SolverRun(
                solver=solver,
                cp_status=cp_status,
                wall_time_seconds=elapsed,
                model_info=ModelInfo(
                    model=model,
                    starts=starts,
                    ends=ends,
                    intervals={tid: intervals[tid] for tid in intervals},
                    horizon=horizon,
                    makespan_var=phase1_model.makespan_var,
                ),
            )
            intervals = self._extractor.extract(run)
            return ScheduleResult(
                status=status,
                makespan_minutes=makespan,
                intervals=intervals,
                wall_time_seconds=elapsed,
                best_objective_bound=int(solver.best_objective_bound)
                if status == SolverStatus.OPTIMAL
                else None,
            )

        # Fall back to Phase 1 result.
        return phase1_result

    # ------------------------------------------------------------------
    # Helper: rebuild model for Phase 2 with fixed makespan
    # ------------------------------------------------------------------

    def _rebuild_model(
        self,
        model: cp_model.CpModel,
        problem: SchedulingProblem,
        horizon: int,
        makespan: int,
    ) -> tuple[
        dict[str, cp_model.IntVar],
        dict[str, cp_model.IntVar],
        dict[str, cp_model.IntervalVar],
    ]:
        """Rebuild Stages A-D with makespan fixed."""
        from cooking_plan_agent.domain.enums import WorkMode

        starts: dict[str, cp_model.IntVar] = {}
        ends: dict[str, cp_model.IntVar] = {}
        intervals: dict[str, cp_model.IntervalVar] = {}

        # Stage A: variables
        for task in problem.tasks:
            tid = task.task_id
            start = model.new_int_var(0, horizon, f"start:{tid}")
            end = model.new_int_var(0, horizon, f"end:{tid}")
            iv = model.new_interval_var(start, task.duration_minutes, end, f"interval:{tid}")
            starts[tid] = start
            ends[tid] = end
            intervals[tid] = iv

        # Stage B: precedence
        for task in problem.tasks:
            for dep in task.dependencies:
                if dep.predecessor_id in starts and task.task_id in starts:
                    model.add(starts[task.task_id] >= ends[dep.predecessor_id] + dep.minimum_lag_minutes)
                    if dep.maximum_lag_minutes is not None:
                        model.add(starts[task.task_id] <= ends[dep.predecessor_id] + dep.maximum_lag_minutes)

        # Stage C: active no-overlap
        active_ivs = [
            intervals[t.task_id]
            for t in problem.tasks
            if t.work_mode == WorkMode.ACTIVE
        ]
        if active_ivs:
            model.add_no_overlap(active_ivs)

        # Stage D: resources
        res_capacity: dict[str, int] = {}
        for r in problem.resources:
            if r.available:
                cap = int(r.capacity) if r.capacity else 1
                res_capacity[r.resource_type] = res_capacity.get(r.resource_type, 0) + cap

        for res_type, capacity in res_capacity.items():
            res_ivs: list[cp_model.IntervalVar] = []
            res_demands: list[int] = []
            for task in problem.tasks:
                for need in task.resources:
                    if need.resource_type == res_type:
                        res_ivs.append(intervals[task.task_id])
                        res_demands.append(need.quantity)
                        break
            if res_ivs:
                model.add_cumulative(intervals=res_ivs, demands=res_demands, capacity=capacity)

        # Fix makespan
        final_task_ids = self._final_task_ids(problem)
        final_ends_list = [ends[tid] for tid in final_task_ids if tid in ends]
        if final_ends_list:
            makespan_var = model.new_int_var(0, horizon, "makespan")
            model.add_max_equality(makespan_var, final_ends_list)
            model.add(makespan_var == makespan)

        return starts, ends, intervals

    # ------------------------------------------------------------------
    # Helper: identify final tasks (no successors)
    # ------------------------------------------------------------------

    def _final_task_ids(self, problem: SchedulingProblem) -> set[str]:
        predecessor_ids: set[str] = set()
        for task in problem.tasks:
            for dep in task.dependencies:
                predecessor_ids.add(dep.predecessor_id)
        all_ids = {t.task_id for t in problem.tasks}
        final = all_ids - predecessor_ids
        return final if final else all_ids
