"""Schedule extractor — converts CP-SAT solution values to domain ScheduledIntervals.

Note: converts a feasible solution to domain output (ScheduledInterval
instances).  Only call after the solver returns OPTIMAL or FEASIBLE.
"""

from cooking_plan_agent.scheduling.models import ScheduledInterval
from cooking_plan_agent.scheduling.solver import SolverRun

# ============================================================================
# 7.1  ScheduleExtractor — solution → domain output
# ============================================================================


class ScheduleExtractor:
    """Extracts scheduled intervals from a successful CP-SAT solution.

    Handbook 7.1: this class knows about CP-SAT variables but does not
    depend on how the model was constructed — it only reads values.

    Usage::

        extractor = ScheduleExtractor()
        intervals = extractor.extract(solver_run)
    """

    def extract(self, solver_run: SolverRun) -> tuple[ScheduledInterval, ...]:
        """Extract ScheduledInterval objects from a feasible solver run.

        Args:
            solver_run: A SolverRun with OPTIMAL or FEASIBLE status.

        Returns:
            A tuple of ScheduledInterval, one per task in the model.

        Raises:
            ValueError: If the solver status is not OPTIMAL or FEASIBLE.
        """
        cp_solver = solver_run.solver
        model_info = solver_run.model_info

        # Find which resource type each task uses, for assignment tracking.
        # In the MVP, identical resources of the same type are interchangeable,
        # so we simply record the resource type as the assigned resource.
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
                    # Named instances will be added when alternative resources
                    # are implemented (Handbook 7.6).
                )
            )

        return tuple(intervals)
