"""Schedule orchestrator — lexicographic multi-objective optimisation (P3-03).

Handbook 7.9–7.12: sequential solves for makespan, holding, context switching,
and active labour.  Each later optimisation MUST NOT increase the fixed value
from earlier optimisations.

Design:
- Phase 1: minimise makespan
- Phase 2: minimise dish holding time (7.9)
- Phase 3: minimise context switching (7.10) — modelled as the per-category
  time span proxy (grouping same-category tasks reduces hot-zone/station
  changes); a linear, verifiable surrogate for switch cost.
- Phase 4: minimise active labour (7.11) — only when equivalent execution
  modes exist (alternative resources/workers); otherwise gated off (the model
  must never invent options that do not exist).

Lexicographic guarantee (D5): every later phase fixes the earlier phase's
optimal value as a hard constraint. A phase that fails (timeout / unknown /
model error) falls back to the previous phase's feasible result — the solver
never produces a worse solution than the best known so far.

Budget: the single-request solver budget (problem.solver_timeout_seconds) is
split across the phases actually executed, so total wall time stays bounded.
"""

import time
from dataclasses import dataclass

from ortools.sat.python import cp_model

from cooking_plan_agent.domain.enums import SolverStatus
from cooking_plan_agent.scheduling.builder import ModelInfo, ScheduleModelBuilder
from cooking_plan_agent.scheduling.extractor import ScheduleExtractor
from cooking_plan_agent.scheduling.models import (
    ScheduleResult,
    SchedulingProblem,
    VerificationReport,
)
from cooking_plan_agent.scheduling.solver import ScheduleSolver, SolverRun
from cooking_plan_agent.scheduling.verifier import ScheduleVerifier

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
    return ScheduleOrchestrator().solve(problem)


@dataclass(frozen=True)
class _PhaseOutcome:
    """Result of one optimisation phase (P3-03)."""

    result: ScheduleResult
    objective_value: int | None = None


class ScheduleOrchestrator:
    """Orchestrates lexicographic multi-objective solves.

    Handbook 7.12: sequential solves, never combine objectives with arbitrary
    weights.  Each later solve fixes the earlier objective value as a hard
    constraint, guaranteeing priority ordering (D5).

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
        optimization_level: str = "full",
    ) -> tuple[ScheduleResult, VerificationReport]:
        """Run the lexicographic phases in priority order with budget control.

        ``optimization_level`` selects how many objectives are optimised:
          - "makespan": Phase 1 only (legacy behaviour; fastest, no tie-break)
          - "phase12":   Phase 1 + 2 (makespan, then holding)
          - "full":      Phase 1 + 2 + 3 (+ Phase 4 when equivalent modes exist)

        Each phase fixes the previous phase's optimal value.  Phases that
        fail fall back to the previous feasible result; the returned
        ScheduleResult records which phases were applied.

        Returns:
            (best_feasible_result, verification_report)
        """
        deadline = time.monotonic() + problem.solver_timeout_seconds

        # Phase 1: minimise makespan (receives the full budget — it is the
        # only phase whose failure is fatal).
        phase1 = self._phase_makespan(problem, deadline)
        if phase1.result.status not in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE):
            report = self._verifier.verify(problem, phase1.result)
            return phase1.result, report

        phases: list[str] = ["makespan"]
        best = phase1.result
        makespan = phase1.result.makespan_minutes
        assert makespan is not None

        if optimization_level in ("phase12", "full") and self._has_time_remaining(deadline):
            # Phase 2: minimise holding (fix makespan). A phase counts as
            # applied ONLY when it produced its own objective value — a
            # fallback to Phase 1 (timeout/UNKNOWN or infeasible) must not
            # be recorded, or the verifier would report the objective as
            # missing and reject a valid schedule (P3-03 regression).
            phase2 = self._phase_holding(problem, makespan, best, deadline)
            if phase2.result.holding_objective is not None:
                best = phase2.result
                phases.append("holding")

        if optimization_level == "full" and self._has_time_remaining(deadline):
            # Phase 3: minimise context switching (fix makespan + holding).
            holding_fixed = best.holding_objective
            phase3 = self._phase_context_switch(problem, makespan, holding_fixed, best, deadline)
            if phase3.result.context_switch_objective is not None:
                best = phase3.result
                phases.append("context_switch")

            # Phase 4: minimise active labour (only when equivalent execution
            # modes exist; gated otherwise — P3-03 step 4).
            if self._has_equivalent_modes(problem) and self._has_time_remaining(deadline):
                phase4 = self._phase_active_labour(problem, makespan, best)
                if phase4.result is not best and phase4.result.active_labour_objective is not None:
                    best = phase4.result
                    phases.append("active_labour")

        best = best.model_copy(update={"optimization_phases": tuple(phases)})
        report = self._verifier.verify(problem, best)
        return best, report

    # ------------------------------------------------------------------
    # Phase 1: minimise makespan
    # ------------------------------------------------------------------

    def _phase_makespan(self, problem: SchedulingProblem, deadline: float) -> _PhaseOutcome:
        """Build and solve the basic makespan-minimisation model.

        Returns a ScheduleResult with intervals extracted if feasible.
        """
        model_info = self._builder.build(problem)
        remaining = self._remaining_timeout(deadline)
        if remaining <= 0:
            return _PhaseOutcome(result=ScheduleResult(status=SolverStatus.UNKNOWN, wall_time_seconds=0.0))
        solver_run = self._solver.solve(model_info, remaining)
        result = self._solver.to_schedule_result(solver_run)

        if result.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE):
            intervals = self._extractor.extract(solver_run)
            result = result.model_copy(update={"intervals": intervals})

        return _PhaseOutcome(result=result, objective_value=result.makespan_minutes)

    # ------------------------------------------------------------------
    # Phase 2: minimise dish holding time (7.9)
    # ------------------------------------------------------------------

    def _phase_holding(
        self,
        problem: SchedulingProblem,
        makespan: int,
        phase1_result: ScheduleResult,
        deadline: float,
    ) -> _PhaseOutcome:
        """Minimise weighted holding time while keeping makespan fixed.

        Handbook 7.9: holding cost H = sum_d w_d * (T* - C_d) where:
        - T* is the fixed minimum makespan
        - C_d is dish d's completion time
        - w_d is a weight (higher = serve sooner)

        This moves heat-sensitive dishes later without increasing makespan.
        """
        model = cp_model.CpModel()
        horizon = self._builder.compute_horizon(problem.tasks)
        starts, ends, interval_vars = self._builder.build_constraints(model, problem, horizon)

        # Fix makespan to Phase 1's optimal value.
        final_task_ids = self._final_task_ids(problem)
        final_ends_list = [ends[tid] for tid in final_task_ids if tid in ends]
        makespan_var = model.new_int_var(0, horizon, "makespan")
        model.add_max_equality(makespan_var, final_ends_list)
        model.add(makespan_var == makespan)

        # Holding objective: minimise sum of per-dish (makespan - completion),
        # weighted by dish priority (all weights = 1 in MVP — a documented,
        # explainable source; weights become configurable with dish priority).
        dish_holding_terms: list[cp_model.LinearExpr] = []
        for dish_id, task_ids in self._dish_tasks(problem).items():
            dish_end_vars = [ends[tid] for tid in task_ids if tid in ends]
            if not dish_end_vars:
                continue
            dish_completion = model.new_int_var(0, horizon, f"completion:{dish_id}")
            model.add_max_equality(dish_completion, dish_end_vars)
            holding = model.new_int_var(0, horizon, f"holding:{dish_id}")
            model.add(holding == makespan - dish_completion)
            dish_holding_terms.append(holding)

        if dish_holding_terms:
            model.minimize(sum(dish_holding_terms))

        outcome = self._solve_model(model, starts, ends, interval_vars, horizon, makespan_var, deadline)
        if outcome.result.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE):
            result = outcome.result.model_copy(update={"holding_objective": outcome.objective_value})
            return _PhaseOutcome(result=result, objective_value=outcome.objective_value)

        # Fall back to Phase 1 result.
        return _PhaseOutcome(result=phase1_result)

    # ------------------------------------------------------------------
    # Phase 3: minimise context switching (7.10)
    # ------------------------------------------------------------------

    def _phase_context_switch(
        self,
        problem: SchedulingProblem,
        makespan: int,
        holding_fixed: int | None,
        phase2_result: ScheduleResult,
        deadline: float,
    ) -> _PhaseOutcome:
        """Minimise context switching while keeping makespan + holding fixed.

        Context switching (7.10) is modelled as the sum over task categories
        of (latest end - earliest start) for that category.  Grouping
        same-category tasks (cutting together, heating together) shrinks
        these spans and reduces station/tool/hot-zone switching.  This is a
        linear, verifiable proxy for switch cost and — being a pure tie-break
        — never increases makespan or holding (D5).
        """
        model = cp_model.CpModel()
        horizon = self._builder.compute_horizon(problem.tasks)
        starts, ends, interval_vars = self._builder.build_constraints(model, problem, horizon)

        # Fix makespan.
        final_task_ids = self._final_task_ids(problem)
        final_ends_list = [ends[tid] for tid in final_task_ids if tid in ends]
        makespan_var = model.new_int_var(0, horizon, "makespan")
        model.add_max_equality(makespan_var, final_ends_list)
        model.add(makespan_var == makespan)

        # Fix holding to Phase 2's value (lexicographic tie-break, D5).
        holding_var = self._add_holding_objective(problem, model, starts, ends, horizon, makespan_var)
        if holding_var is not None and holding_fixed is not None:
            model.add(holding_var <= holding_fixed)
        elif holding_var is not None:
            model.minimize(holding_var)

        # Context-switch objective: minimise summed category time spans.
        category_span_terms: list[cp_model.LinearExpr] = []
        for category, task_ids in self._category_tasks(problem).items():
            cat_ends = [ends[tid] for tid in task_ids if tid in ends]
            cat_starts = [starts[tid] for tid in task_ids if tid in starts]
            if len(cat_ends) < 2:
                continue
            earliest = model.new_int_var(0, horizon, f"earliest:{category}")
            latest = model.new_int_var(0, horizon, f"latest:{category}")
            model.add_min_equality(earliest, cat_starts)
            model.add_max_equality(latest, cat_ends)
            span = model.new_int_var(0, horizon, f"span:{category}")
            model.add(span == latest - earliest)
            category_span_terms.append(span)

        if category_span_terms:
            model.minimize(sum(category_span_terms))

        outcome = self._solve_model(model, starts, ends, interval_vars, horizon, makespan_var, deadline)
        if outcome.result.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE):
            result = outcome.result.model_copy(
                update={
                    "holding_objective": holding_fixed,
                    "context_switch_objective": outcome.objective_value,
                }
            )
            return _PhaseOutcome(result=result, objective_value=outcome.objective_value)

        # Fall back to Phase 2 result.
        return _PhaseOutcome(result=phase2_result)

    # ------------------------------------------------------------------
    # Phase 4: minimise active labour (7.11 — gated)
    # ------------------------------------------------------------------

    def _has_equivalent_modes(self, problem: SchedulingProblem) -> bool:
        """Return True only when tasks have equivalent execution modes.

        Phase 4 may only optimise active labour when genuinely alternative
        execution modes exist (e.g. two equivalent stations/workers, or a
        task with a documented passive alternative).  The current model has
        no such options, so this returns False and Phase 4 is skipped — we
        must never fabricate alternatives (P3-03 step 4).
        """
        # A task with more than one compatible resource for the same need is
        # a candidate for alternative-mode scheduling.  Currently the model
        # never creates such tasks; kept as an explicit gate for future work.
        for task in problem.tasks:
            if len(task.resources) > 1:
                return True
        return False

    def _phase_active_labour(
        self,
        problem: SchedulingProblem,
        makespan: int,
        phase3_result: ScheduleResult,
    ) -> _PhaseOutcome:
        """Minimise total active labour (sum of ACTIVE durations) under
        equivalent execution modes.  Gated: only reached when
        ``_has_equivalent_modes`` is True; unreachable in the current model.
        """
        return _PhaseOutcome(result=phase3_result)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _solve_model(
        self,
        model: cp_model.CpModel,
        starts: dict[str, cp_model.IntVar],
        ends: dict[str, cp_model.IntVar],
        interval_vars: dict[str, cp_model.IntervalVar],
        horizon: int,
        makespan_var: cp_model.IntVar,
        deadline: float,
    ) -> _PhaseOutcome:
        """Solve an already-built phase model and extract intervals.

        Returns the ScheduleResult (with intervals) and the solver's
        objective value when the solve succeeded.
        """
        remaining = self._remaining_timeout(deadline)
        if remaining <= 0:
            return _PhaseOutcome(result=ScheduleResult(status=SolverStatus.UNKNOWN, wall_time_seconds=0.0))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = remaining
        solver.parameters.num_search_workers = 4

        start_time = time.monotonic()
        cp_status = solver.solve(model)
        elapsed = time.monotonic() - start_time

        status = self._solver.map_status(cp_status)
        if status not in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE):
            return _PhaseOutcome(result=ScheduleResult(status=status, wall_time_seconds=elapsed))

        run = SolverRun(
            solver=solver,
            cp_status=cp_status,
            wall_time_seconds=elapsed,
            model_info=ModelInfo(
                model=model,
                starts=starts,
                ends=ends,
                intervals={tid: interval_vars[tid] for tid in interval_vars},
                horizon=horizon,
                makespan_var=makespan_var,
            ),
        )
        intervals = self._extractor.extract(run)
        result = ScheduleResult(
            status=status,
            makespan_minutes=int(solver.objective_value)
            if self._is_makespan_objective(model)
            else int(max(solver.value(e) for e in ends.values())),
            intervals=intervals,
            wall_time_seconds=elapsed,
            best_objective_bound=int(solver.best_objective_bound) if status == SolverStatus.OPTIMAL else None,
        )
        objective_value: int | None = None
        try:
            objective_value = int(solver.objective_value)
        except (ValueError, TypeError):
            objective_value = None
        return _PhaseOutcome(result=result, objective_value=objective_value)

    @staticmethod
    def _remaining_timeout(deadline: float) -> float:
        """Return the solver time left in the current request budget."""
        return max(0.0, deadline - time.monotonic())

    @classmethod
    def _has_time_remaining(cls, deadline: float) -> bool:
        """Avoid building another phase when the request budget is exhausted."""
        return cls._remaining_timeout(deadline) > 0.001

    @staticmethod
    def _is_makespan_objective(model: cp_model.CpModel) -> bool:
        """Best-effort heuristic: a model minimising a single 'makespan'
        variable reports that value directly; phase objectives report their
        own value.  We conservatively recompute makespan from ends instead,
        so this helper is unused today (kept for clarity).
        """
        return False

    def _dish_tasks(self, problem: SchedulingProblem) -> dict[str, list[str]]:
        """Group task IDs by dish_id."""
        groups: dict[str, list[str]] = {}
        for task in problem.tasks:
            groups.setdefault(task.dish_id, []).append(task.task_id)
        return groups

    def _category_tasks(self, problem: SchedulingProblem) -> dict[str, list[str]]:
        """Group task IDs by task category (for context-switch spans)."""
        groups: dict[str, list[str]] = {}
        for task in problem.tasks:
            groups.setdefault(task.category, []).append(task.task_id)
        return groups

    def _add_holding_objective(
        self,
        problem: SchedulingProblem,
        model: cp_model.CpModel,
        starts: dict[str, cp_model.IntVar],
        ends: dict[str, cp_model.IntVar],
        horizon: int,
        makespan_var: cp_model.IntVar,
    ) -> cp_model.IntVar | None:
        """Build the holding objective expression on a fresh model.

        Returns a linear expression variable for the summed holding terms, or
        None when there are no dishes to hold (empty problem).
        """
        terms: list[cp_model.LinearExpr] = []
        for dish_id, task_ids in self._dish_tasks(problem).items():
            dish_end_vars = [ends[tid] for tid in task_ids if tid in ends]
            if not dish_end_vars:
                continue
            dish_completion = model.new_int_var(0, horizon, f"completion:{dish_id}")
            model.add_max_equality(dish_completion, dish_end_vars)
            holding = model.new_int_var(0, horizon, f"holding:{dish_id}")
            model.add(holding == makespan_var - dish_completion)
            terms.append(holding)
        if not terms:
            return None
        total = model.new_int_var(0, horizon * len(terms), "total_holding")
        model.add(total == sum(terms))
        return total

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
        return final or all_ids
