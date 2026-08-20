# ============================================================================
# ScheduleOrchestrator — 字典序多目标优化编排（P3-03）
# ============================================================================

"""Schedule orchestrator — lexicographic multi-objective optimisation (P3-03).

调度编排器 — 字典序多目标优化（P3-03）。

Handbook 7.9–7.12: sequential solves for makespan, holding, context switching,
and active labour.  Each later optimisation MUST NOT increase the fixed value
from earlier optimisations.
手册 7.9–7.12：依次求解 makespan、保温、上下文切换与主动劳动力。每个后续优化不得增大先前优化已固定的值。

Design:
设计：
- Phase 1: minimise makespan
- 阶段 1：最小化 makespan
- Phase 2: minimise dish holding time (7.9)
- 阶段 2：最小化菜品保温时间（7.9）
- Phase 3: minimise context switching (7.10) — modelled as the per-category
  time span proxy (grouping same-category tasks reduces hot-zone/station
  changes); a linear, verifiable surrogate for switch cost.
- 阶段 3：最小化上下文切换（7.10）——建模为每个类别的时间跨度代理（将同类任务分组可减少热区/工位切换）；这是切换成本的线性、可验证替代指标。
- Phase 4: minimise active labour (7.11) — only when equivalent execution
  modes exist (alternative resources/workers); otherwise gated off (the model
  must never invent options that do not exist).
- 阶段 4：最小化主动劳动力（7.11）——仅在存在等价执行模式（替代资源/人员）时进行；否则关闭（模型绝不能凭空捏造不存在的选项）。

Lexicographic guarantee (D5): every later phase fixes the earlier phase's
optimal value as a hard constraint. A phase that fails (timeout / unknown /
model error) falls back to the previous phase's feasible result — the solver
never produces a worse solution than the best known so far.
字典序保证（D5）：每个后续阶段都将先前阶段的最优值固定为硬约束。失败的阶段（超时/未知/模型错误）回退到前一阶段的可行结果——求解器绝不会产出比已知最优更差的解。

Budget: the single-request solver budget (problem.solver_timeout_seconds) is
split across the phases actually executed, so total wall time stays bounded.
预算：单次请求的求解预算（problem.solver_timeout_seconds）在实际执行的阶段间分摊，使总墙钟时间保持有界。
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
# 顶层 schedule() — 完整流水线的便捷函数
# ============================================================================


def schedule(problem: SchedulingProblem) -> tuple[ScheduleResult, VerificationReport]:
    """Run the full scheduling pipeline: build → solve → extract → verify.

    运行完整调度流水线：构建 → 求解 → 提取 → 校验。

    This is a convenience function for simple cases.  For multi-objective
    optimisation, use ``ScheduleOrchestrator`` directly.
    这是简单场景的便捷函数。对于多目标优化，请直接使用 ``ScheduleOrchestrator``。

    Args:
        problem: Validated scheduling problem.
        problem: 已验证的调度问题。

    Returns:
        A tuple of (ScheduleResult, VerificationReport).
        一个 (ScheduleResult, VerificationReport) 元组。
    """
    return ScheduleOrchestrator().solve(problem)


@dataclass(frozen=True)
class _PhaseOutcome:
    """Result of one optimisation phase (P3-03).

    单个优化阶段的结果（P3-03）。
    """

    result: ScheduleResult
    objective_value: int | None = None


class ScheduleOrchestrator:
    """Orchestrates lexicographic multi-objective solves.

    编排字典序多目标求解。

    Handbook 7.12: sequential solves, never combine objectives with arbitrary
    weights.  Each later solve fixes the earlier objective value as a hard
    constraint, guaranteeing priority ordering (D5).
    手册 7.12：顺序求解，绝不使用任意权重组合目标。每次后续求解都将先前目标值固定为硬约束，从而保证优先级顺序（D5）。

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
    # 公共 API
    # ------------------------------------------------------------------

    def solve(
        self,
        problem: SchedulingProblem,
        optimization_level: str = "full",
    ) -> tuple[ScheduleResult, VerificationReport]:
        """Run the lexicographic phases in priority order with budget control.

        按优先级顺序运行字典序阶段，并控制预算。

        ``optimization_level`` selects how many objectives are optimised:
        ``optimization_level`` 选择要优化多少个目标：
          - "makespan": Phase 1 only (legacy behaviour; fastest, no tie-break)
          - "makespan"：仅阶段 1（旧行为；最快，无并列打破）
          - "phase12":   Phase 1 + 2 (makespan, then holding)
          - "phase12"：阶段 1 + 2（先 makespan，后保温）
          - "full":      Phase 1 + 2 + 3 (+ Phase 4 when equivalent modes exist)
          - "full"：阶段 1 + 2 + 3（存在等价模式时再加阶段 4）

        Each phase fixes the previous phase's optimal value.  Phases that
        fail fall back to the previous feasible result; the returned
        ScheduleResult records which phases were applied.
        每个阶段都固定前一阶段的最优值。失败的阶段回退到前一可行结果；返回的 ScheduleResult 记录实际应用了哪些阶段。

        Returns:
            (best_feasible_result, verification_report)
            (best_feasible_result, verification_report)：(最优可行结果, 校验报告)
        """
        deadline = time.monotonic() + problem.solver_timeout_seconds

        # Phase 1: minimise makespan (receives the full budget — it is the
        # only phase whose failure is fatal).
        # 阶段 1：最小化 makespan（获得全部预算——它是唯一失败即致命的阶段）。
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
            # 阶段 2：最小化保温（固定 makespan）。仅当阶段产出自身目标值时才算"已应用"——
            # 回退到阶段 1（超时/UNKNOWN 或不可行）不得被记录，否则校验器会报告目标缺失
            # 并拒绝一个本应有效的调度（P3-03 回归）。
            phase2 = self._phase_holding(problem, makespan, best, deadline)
            if phase2.result.holding_objective is not None:
                best = phase2.result
                phases.append("holding")

        if optimization_level == "full" and self._has_time_remaining(deadline):
            # Phase 3: minimise context switching (fix makespan + holding).
            # 阶段 3：最小化上下文切换（固定 makespan + 保温）。
            holding_fixed = best.holding_objective
            phase3 = self._phase_context_switch(problem, makespan, holding_fixed, best, deadline)
            if phase3.result.context_switch_objective is not None:
                best = phase3.result
                phases.append("context_switch")

            # Phase 4: minimise active labour (only when equivalent execution
            # modes exist; gated otherwise — P3-03 step 4).
            # 阶段 4：最小化主动劳动力（仅在存在等价执行模式时；否则关闭——P3-03 第 4 步）。
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
    # 阶段 1：最小化 makespan
    # ------------------------------------------------------------------

    def _phase_makespan(self, problem: SchedulingProblem, deadline: float) -> _PhaseOutcome:
        """Build and solve the basic makespan-minimisation model.

        构建并求解基础的 makespan 最小化模型。

        Returns a ScheduleResult with intervals extracted if feasible.
        若可行，返回已提取区间的 ScheduleResult。
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
    # 阶段 2：最小化菜品保温时间（7.9）
    # ------------------------------------------------------------------

    def _phase_holding(
        self,
        problem: SchedulingProblem,
        makespan: int,
        phase1_result: ScheduleResult,
        deadline: float,
    ) -> _PhaseOutcome:
        """Minimise weighted holding time while keeping makespan fixed.

        在保持 makespan 不变的前提下最小化加权保温时间。

        Handbook 7.9: holding cost H = sum_d w_d * (T* - C_d) where:
        手册 7.9：保温成本 H = sum_d w_d * (T* - C_d)，其中：
        - T* is the fixed minimum makespan
        - T* 是固定的最小 makespan
        - C_d is dish d's completion time
        - C_d 是菜品 d 的完成时间
        - w_d is a weight (higher = serve sooner)
        - w_d 是权重（越高表示越早上菜）

        This moves heat-sensitive dishes later without increasing makespan.
        这将在不增大 makespan 的前提下，把对温度敏感的菜品推迟。
        """
        model = cp_model.CpModel()
        horizon = self._builder.compute_horizon(problem.tasks)
        starts, ends, interval_vars = self._builder.build_constraints(model, problem, horizon)

        # Fix makespan to Phase 1's optimal value.
        # 将 makespan 固定为阶段 1 的最优值。
        final_task_ids = self._final_task_ids(problem)
        final_ends_list = [ends[tid] for tid in final_task_ids if tid in ends]
        makespan_var = model.new_int_var(0, horizon, "makespan")
        model.add_max_equality(makespan_var, final_ends_list)
        model.add(makespan_var == makespan)

        # Holding objective: minimise sum of per-dish (makespan - completion),
        # weighted by dish priority (all weights = 1 in MVP — a documented,
        # explainable source; weights become configurable with dish priority).
        # 保温目标：最小化每个菜品的 (makespan - 完成时间) 之和，
        # 按菜品优先级加权（MVP 中所有权重 = 1——一个有据可查、可解释的来源；权重将随菜品优先级变为可配置）。
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
        # 回退到阶段 1 的结果。
        return _PhaseOutcome(result=phase1_result)

    # ------------------------------------------------------------------
    # Phase 3: minimise context switching (7.10)
    # 阶段 3：最小化上下文切换（7.10）
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

        在保持 makespan + 保温值不变的前提下最小化上下文切换。

        Context switching (7.10) is modelled as the sum over task categories
        of (latest end - earliest start) for that category.  Grouping
        same-category tasks (cutting together, heating together) shrinks
        these spans and reduces station/tool/hot-zone switching.  This is a
        linear, verifiable proxy for switch cost and — being a pure tie-break
        — never increases makespan or holding (D5).
        上下文切换（7.10）建模为：对每个任务类别求（最晚结束 - 最早开始）并求和。将同类任务分组（一起切、一起加热）可缩小这些跨度，减少工位/工具/热区切换。这是切换成本的线性、可验证代理指标，并且——作为纯并列打破——绝不会增大 makespan 或保温值（D5）。
        """
        model = cp_model.CpModel()
        horizon = self._builder.compute_horizon(problem.tasks)
        starts, ends, interval_vars = self._builder.build_constraints(model, problem, horizon)

        # Fix makespan.
        # 固定 makespan。
        final_task_ids = self._final_task_ids(problem)
        final_ends_list = [ends[tid] for tid in final_task_ids if tid in ends]
        makespan_var = model.new_int_var(0, horizon, "makespan")
        model.add_max_equality(makespan_var, final_ends_list)
        model.add(makespan_var == makespan)

        # Fix holding to Phase 2's value (lexicographic tie-break, D5).
        # 将保温值固定为阶段 2 的值（字典序并列打破，D5）。
        holding_var = self._add_holding_objective(problem, model, starts, ends, horizon, makespan_var)
        if holding_var is not None and holding_fixed is not None:
            model.add(holding_var <= holding_fixed)
        elif holding_var is not None:
            model.minimize(holding_var)

        # Context-switch objective: minimise summed category time spans.
        # 上下文切换目标：最小化类别时间跨度之和。
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
        # 回退到阶段 2 的结果。
        return _PhaseOutcome(result=phase2_result)

    # ------------------------------------------------------------------
    # Phase 4: minimise active labour (7.11 — gated)
    # 阶段 4：最小化主动劳动力（7.11 — 受门控）
    # ------------------------------------------------------------------

    def _has_equivalent_modes(self, problem: SchedulingProblem) -> bool:
        """Return True only when tasks have equivalent execution modes.

        仅当任务存在等价执行模式时返回 True。

        Phase 4 may only optimise active labour when genuinely alternative
        execution modes exist (e.g. two equivalent stations/workers, or a
        task with a documented passive alternative).  The current model has
        no such options, so this returns False and Phase 4 is skipped — we
        must never fabricate alternatives (P3-03 step 4).
        只有当真正存在替代执行模式（例如两个等价工位/人员，或有据可查的被动替代方式）时，阶段 4 才可优化主动劳动力。当前模型没有此类选项，因此返回 False 并跳过阶段 4——我们绝不能凭空捏造替代方案（P3-03 第 4 步）。
        """
        # A task with more than one compatible resource for the same need is
        # a candidate for alternative-mode scheduling.  Currently the model
        # never creates such tasks; kept as an explicit gate for future work.
        # 同一需求拥有多个兼容资源的任务是替代模式调度的候选。当前模型从不创建此类任务；保留此判断作为未来工作的显式门控。
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

        在等价执行模式下最小化总主动劳动力（ACTIVE 时长之和）。受门控：仅当 ``_has_equivalent_modes`` 为 True 时才会到达；当前模型中不可达。
        """
        return _PhaseOutcome(result=phase3_result)

    # ------------------------------------------------------------------
    # Shared helpers
    # 共享辅助方法
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

        求解已构建的阶段模型并提取区间。

        Returns the ScheduleResult (with intervals) and the solver's
        objective value when the solve succeeded.
        求解成功时返回 ScheduleResult（含区间）与求解器的目标值。
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
        """Return the solver time left in the current request budget.

        返回当前请求预算中剩余的求解时间。
        """
        return max(0.0, deadline - time.monotonic())

    @classmethod
    def _has_time_remaining(cls, deadline: float) -> bool:
        """Avoid building another phase when the request budget is exhausted.

        当请求预算耗尽时，避免再构建另一个阶段。
        """
        return cls._remaining_timeout(deadline) > 0.001

    @staticmethod
    def _is_makespan_objective(model: cp_model.CpModel) -> bool:
        """Best-effort heuristic: a model minimising a single 'makespan'
        variable reports that value directly; phase objectives report their
        own value.  We conservatively recompute makespan from ends instead,
        so this helper is unused today (kept for clarity).

        尽力而为的启发式：最小化单个 'makespan' 变量的模型直接报告该值；阶段目标报告其自身的值。我们保守地改为从 ends 重新计算 makespan，因此该辅助函数目前未被使用（仅为清晰起见保留）。
        """
        return False

    def _dish_tasks(self, problem: SchedulingProblem) -> dict[str, list[str]]:
        """Group task IDs by dish_id.

        按 dish_id 分组任务 ID。
        """
        groups: dict[str, list[str]] = {}
        for task in problem.tasks:
            groups.setdefault(task.dish_id, []).append(task.task_id)
        return groups

    def _category_tasks(self, problem: SchedulingProblem) -> dict[str, list[str]]:
        """Group task IDs by task category (for context-switch spans).

        按任务类别分组任务 ID（用于上下文切换跨度）。
        """
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

        在新模型上构建保温目标表达式。

        Returns a linear expression variable for the summed holding terms, or
        None when there are no dishes to hold (empty problem).
        返回求和保温项的线性表达式变量；若没有需要保温的菜品（空问题），则返回 None。
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
    # 辅助方法：识别最终任务（无后继）
    # ------------------------------------------------------------------

    def _final_task_ids(self, problem: SchedulingProblem) -> set[str]:
        predecessor_ids: set[str] = set()
        for task in problem.tasks:
            for dep in task.dependencies:
                predecessor_ids.add(dep.predecessor_id)
        all_ids = {t.task_id for t in problem.tasks}
        final = all_ids - predecessor_ids
        return final or all_ids
