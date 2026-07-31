"""Independent schedule verifier — checks correctness without OR-Tools.

Handbook 7.13: the verifier must not reuse CP-SAT variables or assume the
model was constructed correctly.  It uses a sweep-line algorithm for resource
capacity checking.

All checks are deterministic and purely based on the domain input/output.
"""

from cooking_plan_agent.domain.enums import SolverStatus, WorkMode
from cooking_plan_agent.scheduling.models import (
    ScheduledInterval,
    ScheduleResult,
    SchedulingProblem,
    VerificationIssue,
    VerificationReport,
)

# ============================================================================
# 7.13  ScheduleVerifier — independent correctness check
# ============================================================================


class ScheduleVerifier:
    """Verifies a ScheduleResult against the original SchedulingProblem.

    Uses only domain models — no OR-Tools APIs.  Sweep-line algorithm for
    resource capacity.  At equal times, processes ending events before
    starting events (adjacent tasks do not overlap).

    Usage::

        verifier = ScheduleVerifier()
        report = verifier.verify(problem, result)
        assert report.passed
    """

    def verify(
        self,
        problem: SchedulingProblem,
        result: ScheduleResult,
    ) -> VerificationReport:
        """Run all verification checks and return a report.

        Returns passed=True only if zero issues were found.

        When the solver status is INFEASIBLE or MODEL_INVALID, there are
        no intervals to verify — we skip structural checks and report
        pass (the absence of a schedule is not a verification failure).
        """
        issues: list[VerificationIssue] = []

        # Build lookup maps
        task_map = {t.task_id: t for t in problem.tasks}
        interval_map = {i.task_id: i for i in result.intervals}

        # --- Pre-check: no intervals for infeasible → pass trivially ---
        if result.status in (SolverStatus.INFEASIBLE, SolverStatus.MODEL_INVALID):
            return VerificationReport(
                passed=True,
                issues=(),
                checked_task_count=len(task_map),
                checked_resource_count=len(problem.resources),
            )

        # --- Check 1: every task is present exactly once ---
        issues.extend(self._check_task_presence(task_map, interval_map))

        # --- Check 2: end - start == duration for every task ---
        issues.extend(self._check_durations(task_map, interval_map))

        # --- Check 3: all min/max lag constraints hold ---
        issues.extend(self._check_lags(task_map, interval_map))

        # --- Check 4: active tasks do not overlap ---
        issues.extend(self._check_active_no_overlap(task_map, interval_map))

        # --- Check 5: resource assignments are compatible ---
        issues.extend(self._check_resource_compatibility(task_map, problem))

        # --- Check 6: resource capacity never exceeded (sweep-line) ---
        issues.extend(self._check_resource_capacity(task_map, interval_map, problem))

        # --- Check 7: all values are non-negative and within makespan ---
        issues.extend(self._check_value_ranges(interval_map, result))

        # --- Check 8: reported makespan matches latest finish ---
        issues.extend(self._check_makespan_consistency(interval_map, result))

        return VerificationReport(
            passed=len(issues) == 0,
            issues=tuple(issues),
            checked_task_count=len(task_map),
            checked_resource_count=len(problem.resources),
        )

    # ------------------------------------------------------------------
    # Check 1: task presence
    # ------------------------------------------------------------------

    def _check_task_presence(
        self,
        task_map: dict[str, object],
        interval_map: dict[str, ScheduledInterval],
    ) -> list[VerificationIssue]:
        issues: list[VerificationIssue] = []

        task_ids = set(task_map.keys())
        interval_ids = set(interval_map.keys())

        missing = task_ids - interval_ids
        if missing:
            issues.append(
                VerificationIssue(
                    code="MISSING_TASK",
                    message=f"Tasks not found in schedule: {sorted(missing)}",
                    task_ids=tuple(sorted(missing)),
                )
            )

        extra = interval_ids - task_ids
        if extra:
            issues.append(
                VerificationIssue(
                    code="EXTRA_TASK",
                    message=f"Intervals without matching tasks: {sorted(extra)}",
                    task_ids=tuple(sorted(extra)),
                )
            )

        return issues

    # ------------------------------------------------------------------
    # Check 2: durations
    # ------------------------------------------------------------------

    def _check_durations(
        self,
        task_map: dict[str, object],
        interval_map: dict[str, ScheduledInterval],
    ) -> list[VerificationIssue]:
        issues: list[VerificationIssue] = []

        for task_id, interval in interval_map.items():
            task = task_map.get(task_id)
            if task is None:
                continue
            actual_duration = interval.end_minute - interval.start_minute
            expected_duration = task.duration_minutes  # type: ignore[union-attr]
            if actual_duration != expected_duration:
                issues.append(
                    VerificationIssue(
                        code="DURATION_MISMATCH",
                        message=(f"Task '{task_id}': expected duration {expected_duration}, got {actual_duration}"),
                        task_ids=(task_id,),
                    )
                )

        return issues

    # ------------------------------------------------------------------
    # Check 3: precedence and lag constraints
    # ------------------------------------------------------------------

    def _check_lags(
        self,
        task_map: dict[str, object],
        interval_map: dict[str, ScheduledInterval],
    ) -> list[VerificationIssue]:
        issues: list[VerificationIssue] = []

        for task_id, task in task_map.items():
            interval = interval_map.get(task_id)
            if interval is None:
                continue

            for dep in task.dependencies:  # type: ignore[union-attr]
                pred_interval = interval_map.get(dep.predecessor_id)
                if pred_interval is None:
                    continue

                # Minimum lag: successor.start >= predecessor.end + min_lag
                min_start = pred_interval.end_minute + dep.minimum_lag_minutes
                if interval.start_minute < min_start:
                    issues.append(
                        VerificationIssue(
                            code="MIN_LAG_VIOLATION",
                            message=(
                                f"Task '{task_id}' starts at {interval.start_minute}, "
                                f"but must be >= {min_start} "
                                f"(predecessor '{dep.predecessor_id}' ends at "
                                f"{pred_interval.end_minute} + {dep.minimum_lag_minutes} min lag)"
                            ),
                            task_ids=(task_id, dep.predecessor_id),
                        )
                    )

                # Maximum lag: successor.start <= predecessor.end + max_lag
                if dep.maximum_lag_minutes is not None:
                    max_start = pred_interval.end_minute + dep.maximum_lag_minutes
                    if interval.start_minute > max_start:
                        issues.append(
                            VerificationIssue(
                                code="MAX_LAG_VIOLATION",
                                message=(
                                    f"Task '{task_id}' starts at {interval.start_minute}, "
                                    f"but must be <= {max_start} "
                                    f"(predecessor '{dep.predecessor_id}' ends at "
                                    f"{pred_interval.end_minute} + {dep.maximum_lag_minutes} max lag)"
                                ),
                                task_ids=(task_id, dep.predecessor_id),
                            )
                        )

        return issues

    # ------------------------------------------------------------------
    # Check 4: active tasks do not overlap
    # ------------------------------------------------------------------

    def _check_active_no_overlap(
        self,
        task_map: dict[str, object],
        interval_map: dict[str, ScheduledInterval],
    ) -> list[VerificationIssue]:
        issues: list[VerificationIssue] = []

        # Collect intervals for ACTIVE tasks only.
        active_intervals: list[tuple[int, int, str]] = []
        for task_id, interval in interval_map.items():
            task = task_map.get(task_id)
            if task is None:
                continue
            if task.work_mode == WorkMode.ACTIVE:  # type: ignore[union-attr]
                active_intervals.append((interval.start_minute, interval.end_minute, task_id))

        # Sort by start time, then by end time.
        active_intervals.sort()

        for i in range(len(active_intervals)):
            s1, e1, t1 = active_intervals[i]
            for j in range(i + 1, len(active_intervals)):
                s2, e2, t2 = active_intervals[j]
                # If intervals overlap (not just touching).
                if s2 < e1 and s1 < e2:
                    issues.append(
                        VerificationIssue(
                            code="ACTIVE_OVERLAP",
                            message=(f"Active tasks '{t1}' [{s1}, {e1}) and '{t2}' [{s2}, {e2}) overlap"),
                            task_ids=(t1, t2),
                        )
                    )
                # Since sorted by start, if s2 >= e1, no later task can overlap.
                if s2 >= e1:
                    break

        return issues

    # ------------------------------------------------------------------
    # Check 5: resource compatibility
    # ------------------------------------------------------------------

    def _check_resource_compatibility(
        self,
        task_map: dict[str, object],
        problem: SchedulingProblem,
    ) -> list[VerificationIssue]:
        issues: list[VerificationIssue] = []

        available_types = {r.resource_type for r in problem.resources if r.available}

        for task in problem.tasks:
            for need in task.resources:
                if need.resource_type not in available_types:
                    issues.append(
                        VerificationIssue(
                            code="RESOURCE_UNAVAILABLE",
                            message=(f"Task '{task.task_id}' requires '{need.resource_type}' but it is not available"),
                            task_ids=(task.task_id,),
                        )
                    )

        return issues

    # ------------------------------------------------------------------
    # Check 6: resource capacity (sweep-line algorithm)
    # ------------------------------------------------------------------

    def _check_resource_capacity(
        self,
        task_map: dict[str, object],
        interval_map: dict[str, ScheduledInterval],
        problem: SchedulingProblem,
    ) -> list[VerificationIssue]:
        """Sweep-line over start/end events for each resource type.

        Handbook 7.13: at equal times, process END before START so adjacent
        tasks are not treated as overlapping.
        """
        issues: list[VerificationIssue] = []

        # Build capacity map per resource type.
        res_capacity: dict[str, int] = {}
        for r in problem.resources:
            if r.available:
                cap = int(r.capacity) if r.capacity else 1
                res_capacity[r.resource_type] = res_capacity.get(r.resource_type, 0) + cap

        for res_type, capacity in res_capacity.items():
            # Collect events: (time, is_start, task_id, demand).
            events: list[tuple[int, bool, str, int]] = []
            for task_id, interval in interval_map.items():
                task = task_map.get(task_id)
                if task is None:
                    continue
                for need in task.resources:  # type: ignore[union-attr]
                    if need.resource_type == res_type:
                        events.append((interval.start_minute, True, task_id, need.quantity))
                        events.append((interval.end_minute, False, task_id, need.quantity))
                        break  # Each task counts once per resource type

            # Sort: by time ascending; END (False) before START (True) at equal times.
            events.sort(key=lambda e: (e[0], e[1]))  # False < True → END before START

            current_usage = 0
            for time, is_start, task_id, demand in events:
                if is_start:
                    current_usage += demand
                else:
                    current_usage -= demand

                if current_usage > capacity:
                    issues.append(
                        VerificationIssue(
                            code="CAPACITY_EXCEEDED",
                            message=(
                                f"Resource type '{res_type}' exceeded capacity "
                                f"({capacity}) at t={time}: usage={current_usage}"
                            ),
                            task_ids=(task_id,),
                        )
                    )

        return issues

    # ------------------------------------------------------------------
    # Check 7: value ranges
    # ------------------------------------------------------------------

    def _check_value_ranges(
        self,
        interval_map: dict[str, ScheduledInterval],
        result: ScheduleResult,
    ) -> list[VerificationIssue]:
        issues: list[VerificationIssue] = []

        if result.makespan_minutes is None:
            return issues

        for task_id, interval in interval_map.items():
            if interval.start_minute < 0:
                issues.append(
                    VerificationIssue(
                        code="NEGATIVE_START",
                        message=f"Task '{task_id}' has negative start: {interval.start_minute}",
                        task_ids=(task_id,),
                    )
                )
            if interval.end_minute < 0:
                issues.append(
                    VerificationIssue(
                        code="NEGATIVE_END",
                        message=f"Task '{task_id}' has negative end: {interval.end_minute}",
                        task_ids=(task_id,),
                    )
                )
            if interval.end_minute > result.makespan_minutes:
                issues.append(
                    VerificationIssue(
                        code="EXCEEDS_MAKESPAN",
                        message=(
                            f"Task '{task_id}' ends at {interval.end_minute}, but makespan is {result.makespan_minutes}"
                        ),
                        task_ids=(task_id,),
                    )
                )
            if interval.start_minute >= interval.end_minute:
                issues.append(
                    VerificationIssue(
                        code="INVALID_INTERVAL",
                        message=(f"Task '{task_id}': start ({interval.start_minute}) >= end ({interval.end_minute})"),
                        task_ids=(task_id,),
                    )
                )

        return issues

    # ------------------------------------------------------------------
    # Check 8: makespan consistency
    # ------------------------------------------------------------------

    def _check_makespan_consistency(
        self,
        interval_map: dict[str, ScheduledInterval],
        result: ScheduleResult,
    ) -> list[VerificationIssue]:
        issues: list[VerificationIssue] = []

        if not interval_map or result.makespan_minutes is None:
            return issues

        actual_max = max(iv.end_minute for iv in interval_map.values())
        if actual_max != result.makespan_minutes:
            issues.append(
                VerificationIssue(
                    code="MAKESPAN_MISMATCH",
                    message=(
                        f"Reported makespan {result.makespan_minutes} does not match actual max end time {actual_max}"
                    ),
                )
            )

        return issues
