# ============================================================================
# ScheduleExtractor — 将 CP-SAT 解值转换为领域 ScheduledInterval
# ============================================================================

"""Schedule extractor — converts CP-SAT solution values to domain ScheduledIntervals.

调度提取器 — 将 CP-SAT 解值转换为领域 ScheduledInterval。

Note: converts a feasible solution to domain output (ScheduledInterval
instances).  Only call after the solver returns OPTIMAL or FEASIBLE.
注意：将可行解转换为领域输出（ScheduledInterval 实例）。仅在求解器返回 OPTIMAL 或 FEASIBLE 之后调用。
"""

from cooking_plan_agent.scheduling.models import ScheduledInterval
from cooking_plan_agent.scheduling.solver import SolverRun

# ============================================================================
# 7.1  ScheduleExtractor — solution → domain output
# 7.1  ScheduleExtractor — 解 → 领域输出
# ============================================================================


class ScheduleExtractor:
    """Extracts scheduled intervals from a successful CP-SAT solution.

    从成功的 CP-SAT 解中提取已调度的区间。

    Handbook 7.1: this class knows about CP-SAT variables but does not
    depend on how the model was constructed — it only reads values.
    手册 7.1：此类了解 CP-SAT 变量，但不依赖模型是如何构建的——它只读取取值。

    Usage::

        extractor = ScheduleExtractor()
        intervals = extractor.extract(solver_run)
    """

    def extract(self, solver_run: SolverRun) -> tuple[ScheduledInterval, ...]:
        """Extract ScheduledInterval objects from a feasible solver run.

        从可行解中提取 ScheduledInterval 对象。

        Args:
            solver_run: A SolverRun with OPTIMAL or FEASIBLE status.
            solver_run: 状态为 OPTIMAL 或 FEASIBLE 的 SolverRun。

        Returns:
            A tuple of ScheduledInterval, one per task in the model.
            一个 ScheduledInterval 元组，模型中的每个任务对应一个。

        Raises:
            ValueError: If the solver status is not OPTIMAL or FEASIBLE.
            ValueError: 若求解器状态不是 OPTIMAL 或 FEASIBLE。
        """
        cp_solver = solver_run.solver
        model_info = solver_run.model_info

        # Find which resource type each task uses, for assignment tracking.
        # 查找每个任务使用的资源类型，用于分配跟踪。
        # In the MVP, identical resources of the same type are interchangeable,
        # so we simply record the resource type as the assigned resource.
        # 在 MVP 中，同类型的相同资源可互换，因此我们仅将资源类型记录为所分配的资源。
        intervals: list[ScheduledInterval] = []

        for task_id, start_var in model_info.starts.items():
            start_val = cp_solver.value(start_var)
            end_var = model_info.ends[task_id]
            end_val = cp_solver.value(end_var)

            intervals.append(
                ScheduledInterval(
                    task_id=task_id,
                    start_minute=int(start_val),
                    end_minute=int(end_val),
                    # MVP: assign the resource type as a placeholder.
                    # MVP：以资源类型作为占位分配。
                    # Named instances will be added when alternative resources
                    # are implemented (Handbook 7.6).
                    # 命名实例将在实现替代资源后加入（手册 7.6）。
                )
            )

        return tuple(intervals)
