"""CP-SAT solver — runs the model and maps solver statuses.

Note: configure and run CP-SAT, then map every official
CpSolverStatus to a domain SolverStatus.  Never treat UNKNOWN as INFEASIBLE.
"""

import time

from ortools.sat.python import cp_model

from cooking_plan_agent.domain.enums import SolverStatus
from cooking_plan_agent.scheduling.builder import ModelInfo
from cooking_plan_agent.scheduling.models import ScheduleResult

# ============================================================================
# 7.8  ScheduleSolver — runs CP-SAT and maps statuses
# ============================================================================


class ScheduleSolver:
    """Configures and runs the CP-SAT solver on a ModelInfo.

    Note: solves only — does not build models or extract results.

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

        Args:
            model_info: Constructed CP-SAT model with variable references.
            timeout_seconds: Maximum wall-clock time for the solve.

        Returns:
            SolverRun containing the CpSolver, CpSolverStatus, and wall time.
        """
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = timeout_seconds
        # Use multiple workers for better performance on multi-core machines.
        solver.parameters.num_search_workers = 4
        # Log search progress only in verbose mode (not in tests).
        solver.parameters.log_search_progress = False

        # Run the solver and measure wall-clock time.
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

        Note:
        - OPTIMAL       → OPTIMAL
        - FEASIBLE      → FEASIBLE
        - INFEASIBLE    → INFEASIBLE
        - MODEL_INVALID → MODEL_INVALID
        - UNKNOWN       → UNKNOWN   (never collapse to INFEASIBLE)
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

        Only extracts values when the status is OPTIMAL or FEASIBLE.
        Never attempts to read solver values for INFEASIBLE/MODEL_INVALID/UNKNOWN.
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
# ============================================================================


class SolverRun:
    """Container for a single CP-SAT solve execution.

    Carries the CpSolver instance (for value extraction), the raw
    CpSolverStatus, wall-clock time, and the ModelInfo for variable access.
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
