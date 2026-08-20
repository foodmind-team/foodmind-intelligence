# ============================================================================
# ScheduleSolver — 运行 CP-SAT 模型并映射求解器状态
# ============================================================================

"""CP-SAT solver — runs the model and maps solver statuses.

CP-SAT 求解器 — 运行模型并映射求解器状态。

Note: configure and run CP-SAT, then map every official
CpSolverStatus to a domain SolverStatus.  Never treat UNKNOWN as INFEASIBLE.
注意：配置并运行 CP-SAT，然后将每个官方 CpSolverStatus 映射为领域 SolverStatus。绝不要将 UNKNOWN 当作 INFEASIBLE。
"""

import time

from ortools.sat.python import cp_model

from cooking_plan_agent.domain.enums import SolverStatus
from cooking_plan_agent.scheduling.builder import ModelInfo
from cooking_plan_agent.scheduling.models import ScheduleResult

# ============================================================================
# 7.8  ScheduleSolver — runs CP-SAT and maps statuses
# 7.8  ScheduleSolver — 运行 CP-SAT 并映射状态
# ============================================================================


class ScheduleSolver:
    """Configures and runs the CP-SAT solver on a ModelInfo.

    在 ModelInfo 上配置并运行 CP-SAT 求解器。

    Note: solves only — does not build models or extract results.
    注意：仅求解——不构建模型，也不提取结果。

    Usage::

        solver = ScheduleSolver()
        result = solver.solve(model_info, timeout_seconds=10.0)
    """

    def solve(
        self,
        model_info: ModelInfo,
        timeout_seconds: float = 10.0,
    ) -> "SolverRun":
        """Solve the CP-SAT model and return a SolverRun with status mapping.

        求解 CP-SAT 模型并返回带状态映射的 SolverRun。

        Args:
            model_info: Constructed CP-SAT model with variable references.
            model_info: 已构建的、带变量引用的 CP-SAT 模型。
            timeout_seconds: Maximum wall-clock time for the solve.
            timeout_seconds: 求解的最大墙钟时间。

        Returns:
            SolverRun containing the CpSolver, CpSolverStatus, and wall time.
            SolverRun：包含 CpSolver、CpSolverStatus 与墙钟时间。
        """
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = timeout_seconds
        # Use multiple workers for better performance on multi-core machines.
        # 使用多个工作线程以在多核机器上获得更好性能。
        solver.parameters.num_search_workers = 4
        # Log search progress only in verbose mode (not in tests).
        # 仅在详细模式下记录搜索进度（测试中不记录）。
        solver.parameters.log_search_progress = False

        # Run the solver and measure wall-clock time.
        # 运行求解器并测量墙钟时间。
        start_time = time.monotonic()
        cp_status = solver.solve(model_info.model)
        elapsed = time.monotonic() - start_time

        return SolverRun(
            solver=solver,
            cp_status=cp_status,
            wall_time_seconds=elapsed,
            model_info=model_info,
        )

    def map_status(self, cp_status: cp_model.CpSolverStatus) -> SolverStatus:
        """Map CP-SAT's CpSolverStatus enum to domain SolverStatus.

        将 CP-SAT 的 CpSolverStatus 枚举映射为领域 SolverStatus。

        Note:
        注意：
        - OPTIMAL       → OPTIMAL
        - FEASIBLE      → FEASIBLE
        - INFEASIBLE    → INFEASIBLE
        - MODEL_INVALID → MODEL_INVALID
        - UNKNOWN       → UNKNOWN   (never collapse to INFEASIBLE)
        - UNKNOWN       → UNKNOWN   （绝不要归并为 INFEASIBLE）
        """
        if cp_status == cp_model.OPTIMAL:
            return SolverStatus.OPTIMAL
        elif cp_status == cp_model.FEASIBLE:
            return SolverStatus.FEASIBLE
        elif cp_status == cp_model.INFEASIBLE:
            return SolverStatus.INFEASIBLE
        elif cp_status == cp_model.MODEL_INVALID:
            return SolverStatus.MODEL_INVALID
        elif cp_status == cp_model.UNKNOWN:
            return SolverStatus.UNKNOWN
        else:
            return SolverStatus.UNKNOWN

    def to_schedule_result(
        self,
        solver_run: "SolverRun",
    ) -> ScheduleResult:
        """Convert a SolverRun to a domain ScheduleResult.

        将 SolverRun 转换为领域 ScheduleResult。

        Only extracts values when the status is OPTIMAL or FEASIBLE.
        Never attempts to read solver values for INFEASIBLE/MODEL_INVALID/UNKNOWN.
        仅当状态为 OPTIMAL 或 FEASIBLE 时才提取值。绝不尝试为 INFEASIBLE/MODEL_INVALID/UNKNOWN 读取求解器值。
        """
        status = self.map_status(solver_run.cp_status)

        makespan: int | None = None
        best_bound: int | None = None

        if status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE):
            makespan = int(solver_run.solver.objective_value)
            try:
                best_bound = int(solver_run.solver.best_objective_bound)
            except (ValueError, TypeError):
                best_bound = None

        return ScheduleResult(
            status=status,
            makespan_minutes=makespan,
            wall_time_seconds=solver_run.wall_time_seconds,
            best_objective_bound=best_bound,
        )


# ============================================================================
# SolverRun — holds the result of one CP-SAT solve call
# SolverRun — 保存一次 CP-SAT 求解调用的结果
# ============================================================================


class SolverRun:
    """Container for a single CP-SAT solve execution.

    单次 CP-SAT 求解执行的容器。

    Carries the CpSolver instance (for value extraction), the raw
    CpSolverStatus, wall-clock time, and the ModelInfo for variable access.
    携带 CpSolver 实例（用于取值）、原始 CpSolverStatus、墙钟时间，以及用于变量访问的 ModelInfo。
    """

    def __init__(
        self,
        solver: cp_model.CpSolver,
        cp_status: cp_model.CpSolverStatus,
        wall_time_seconds: float,
        model_info: ModelInfo,
    ) -> None:
        self.solver = solver
        self.cp_status = cp_status
        self.wall_time_seconds = wall_time_seconds
        self.model_info = model_info
