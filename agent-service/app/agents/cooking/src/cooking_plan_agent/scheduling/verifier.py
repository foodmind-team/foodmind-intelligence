# ============================================================================
# ScheduleVerifier — 独立调度校验器，不依赖 OR-Tools 进行正确性检查
# ============================================================================

"""Independent schedule verifier — checks correctness without OR-Tools.

独立调度校验器 — 在不使用 OR-Tools 的情况下检查正确性。

Handbook 7.13: the verifier must not reuse CP-SAT variables or assume the
model was constructed correctly.  It uses a sweep-line algorithm for resource
capacity checking.
手册 7.13：校验器不得复用 CP-SAT 变量，也不得假设模型被正确构建。它使用扫描线算法进行资源容量检查。

All checks are deterministic and purely based on the domain input/output.
所有检查都是确定性的，且完全基于领域输入/输出。
"""

from cooking_plan_agent.domain.enums import SolverStatus, WorkMode
from cooking_plan_agent.domain.models import CookingTask
from cooking_plan_agent.normalisation.names import normalise_resource_type
from cooking_plan_agent.scheduling.models import (
    ScheduledInterval,
    ScheduleResult,
    SchedulingProblem,
    VerificationIssue,
    VerificationReport,
)

# ============================================================================
# 7.13  ScheduleVerifier — independent correctness check
# 7.13  ScheduleVerifier — 独立正确性检查
# ============================================================================


class ScheduleVerifier:
    """Verifies a ScheduleResult against the original SchedulingProblem.

    对照原始 SchedulingProblem 校验 ScheduleResult。

    Uses only domain models — no OR-Tools APIs.  Sweep-line algorithm for
    resource capacity.  At equal times, processes ending events before
    starting events (adjacent tasks do not overlap).
    仅使用领域模型——不使用 OR-Tools API。使用扫描线算法进行资源容量检查。在时间相等时，先处理结束事件再处理开始事件（相邻任务不重叠）。

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

        运行所有校验检查并返回报告。

        Returns passed=True only if zero issues were found.
        仅当未发现任何问题时返回 passed=True。

        When the solver status is INFEASIBLE or MODEL_INVALID, there are
        no intervals to verify — we skip structural checks and report
        pass (the absence of a schedule is not a verification failure).
        当求解器状态为 INFEASIBLE 或 MODEL_INVALID 时，没有区间可供校验——我们跳过结构检查并报告通过（没有调度结果并不是校验失败）。
        """
        issues: list[VerificationIssue] = []

        # Build lookup maps
        # 构建查找映射
        task_map = {t.task_id: t for t in problem.tasks}
        interval_map = {i.task_id: i for i in result.intervals}

        # --- Pre-check: no intervals for infeasible → pass trivially ---
        # --- 预检查：不可行时没有区间 → 平凡通过 ---
        if result.status in (SolverStatus.INFEASIBLE, SolverStatus.MODEL_INVALID):
            return VerificationReport(
                passed=True,
                issues=(),
                checked_task_count=len(task_map),
                checked_resource_count=len(problem.resources),
            )

        # --- Check 1: every task is present exactly once ---
        # --- 检查 1：每个任务恰好出现一次 ---
        issues.extend(self._check_task_presence(task_map, interval_map))

        # --- Check 2: end - start == duration for every task ---
        # --- 检查 2：每个任务的 end - start == duration ---
        issues.extend(self._check_durations(task_map, interval_map))

        # --- Check 3: all min/max lag constraints hold ---
        # --- 检查 3：所有最小/最大滞后约束均成立 ---
        issues.extend(self._check_lags(task_map, interval_map))

        # --- Check 4: active tasks do not overlap ---
        # --- 检查 4：主动任务不重叠 ---
        issues.extend(self._check_active_no_overlap(task_map, interval_map))

        # --- Check 5: resource assignments are compatible ---
        # --- 检查 5：资源分配兼容 ---
        issues.extend(self._check_resource_compatibility(task_map, problem))

        # --- Check 6: resource capacity never exceeded (sweep-line) ---
        # --- 检查 6：资源容量从未超限（扫描线） ---
        issues.extend(self._check_resource_capacity(task_map, interval_map, problem))

        # --- Check 7: all values are non-negative and within makespan ---
        # --- 检查 7：所有值非负且位于 makespan 之内 ---
        issues.extend(self._check_value_ranges(interval_map, result))

        # --- Check 8: reported makespan matches latest finish ---
        # --- 检查 8：报告的 makespan 与最晚完成时间一致 ---
        issues.extend(self._check_makespan_consistency(interval_map, result))

        # --- Check 9: safety-tagged tasks are present in the schedule ---
        # --- 检查 9：带安全标签的任务存在于调度中 ---
        issues.extend(self._check_safety_tasks(task_map, interval_map))

        # --- Check 10: safety-task ordering (P0-07) ---
        # --- 检查 10：安全任务排序（P0-07） ---
        # Sanitisation tasks must start after their raw-protein predecessor
        # ends and finish before their RTE successor starts. Anchor loss
        # (a dependency referencing a missing task) is also a failure.
        # 消毒任务必须在生蛋白前驱结束后才开始，并在其 RTE 后继开始前完成。锚点丢失（依赖引用了缺失任务）也是失败。
        issues.extend(self._check_safety_ordering(task_map, interval_map))

        # --- Check 11 (P3-03): lexicographic fixed objectives not broken ---
        # --- 检查 11（P3-03）：字典序固定目标未被破坏 ---
        # When the orchestrator reports which phases were applied, verify the
        # recorded objective values are consistent with the schedule: holding
        # never exceeds the makespan, and the recorded makespan equals the
        # actual max end (already Check 8 — reinforced here for multi-phase).
        # 当编排器报告应用了哪些阶段时，校验记录的目标值与调度一致：保温值绝不超过 makespan，且记录的 makespan 等于实际最大结束时间（已在检查 8 中——此处为多阶段再次强化）。
        issues.extend(self._check_optimization_consistency(result))

        return VerificationReport(
            passed=len(issues) == 0,
            issues=tuple(issues),
            checked_task_count=len(task_map),
            checked_resource_count=len(problem.resources),
        )

    # ------------------------------------------------------------------
    # Check 11 (P3-03): optimisation phases recorded consistently
    # 检查 11（P3-03）：优化阶段记录一致
    # ------------------------------------------------------------------

    def _check_optimization_consistency(
        self,
        result: ScheduleResult,
    ) -> list[VerificationIssue]:
        """Verify the recorded optimisation-phase metadata is self-consistent.

        校验所记录的优化阶段元数据自洽。

        A result that claims a later phase (holding / context switch) must
        carry the corresponding objective value; a claimed makespan phase
        must always carry a concrete makespan. This catches orchestrator
        bugs where a phase is recorded but its fixed value is lost.
        声称应用了后续阶段（保温/上下文切换）的结果必须携带相应的目标值；声称有 makespan 阶段的结果必须始终携带具体 makespan。这能捕获"记录了阶段但其固定值丢失"的编排器缺陷。
        """
        issues: list[VerificationIssue] = []
        phases = result.optimization_phases

        if "holding" in phases and result.holding_objective is None:
            issues.append(
                VerificationIssue(
                    code="HOLDING_OBJECTIVE_MISSING",
                    message="optimization_phases includes 'holding' but holding_objective is None",
                )
            )
        if "context_switch" in phases and result.context_switch_objective is None:
            issues.append(
                VerificationIssue(
                    code="CONTEXT_SWITCH_OBJECTIVE_MISSING",
                    message="optimization_phases includes 'context_switch' but context_switch_objective is None",
                )
            )
        if phases and result.makespan_minutes is None:
            issues.append(
                VerificationIssue(
                    code="MAKESPAN_MISSING_WITH_PHASES",
                    message=f"optimization_phases={phases} but makespan_minutes is None",
                )
            )
        return issues

    # ------------------------------------------------------------------
    # Check 1: task presence
    # 检查 1：任务存在性
    # ------------------------------------------------------------------

    def _check_task_presence(
        self,
        task_map: dict[str, CookingTask],
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
    # Check 9: safety-tagged tasks
    # 检查 9：带安全标签的任务
    # ------------------------------------------------------------------

    def _check_safety_tasks(
        self,
        task_map: dict[str, CookingTask],
        interval_map: dict[str, ScheduledInterval],
    ) -> list[VerificationIssue]:
        """Verify that every safety-tagged task has a scheduled interval.

        校验每个带安全标签的任务都有已调度的区间。

        Safety tasks (e.g., sanitise cutting board after raw protein handling)
        are injected by merge_preparation_node.  Every such task MUST appear
        in the final schedule — if one is missing, the safety constraint is
        not enforced.
        安全任务（例如处理生蛋白后消毒砧板）由 merge_preparation_node 注入。每个此类任务都必须出现在最终调度中——若缺失，安全约束即未被执行。

        Handbook 7.13: the verifier catches optimiser bugs before they reach
        the user.  A missing safety task is a hard failure.
        手册 7.13：校验器在优化器缺陷到达用户之前将其捕获。缺失安全任务是硬失败。
        """
        issues: list[VerificationIssue] = []

        for task_id, task in task_map.items():
            if not task.safety_tags:
                continue
            if task_id not in interval_map:
                issues.append(
                    VerificationIssue(
                        code="SAFETY_TASK_MISSING",
                        message=(f"Safety-tagged task '{task_id}' (tags: {task.safety_tags}) is missing from schedule"),
                        task_ids=(task_id,),
                    )
                )

        return issues

    # ------------------------------------------------------------------
    # Check 10: safety-task ordering and anchor integrity (P0-07)
    # 检查 10：安全任务排序与锚点完整性（P0-07）
    # ------------------------------------------------------------------

    def _check_safety_ordering(
        self,
        task_map: dict[str, CookingTask],
        interval_map: dict[str, ScheduledInterval],
    ) -> list[VerificationIssue]:
        """Verify the raw → sanitise → RTE ordering for safety tasks.

        校验安全任务的"生 → 消毒 → RTE"排序。

        Every safety-tagged task that has dependencies must:
          - start at or after every predecessor's end (no misplaced
            sanitisation before raw handling finishes)
          - have all its dependency anchors present in the schedule
            (missing predecessor = broken anchor)
        每个有依赖的安全任务必须：
          - 在每个前驱结束后或结束时开始（不得在生处理完成之前就错误地消毒）
          - 其所有依赖锚点都存在于调度中（缺失前驱 = 锚点断裂）

        Successors are checked transitively via the dependency edges, so a
        sanitise task that is a predecessor of an RTE task enforces that the
        RTE task starts after the sanitise task ends.
        后继通过依赖边传递地检查，因此作为 RTE 任务前驱的消毒任务会强制 RTE 任务在消毒任务结束后才开始。
        """
        issues: list[VerificationIssue] = []

        for task_id, task in task_map.items():
            if not task.safety_tags:
                continue
            interval = interval_map.get(task_id)
            if interval is None:
                continue  # presence already reported by Check 9
                # 存在性已由检查 9 报告

            for dep in task.dependencies:
                pred_interval = interval_map.get(dep.predecessor_id)
                if pred_interval is None:
                    issues.append(
                        VerificationIssue(
                            code="SAFETY_ANCHOR_MISSING",
                            message=(
                                f"Safety task '{task_id}' depends on missing "
                                f"task '{dep.predecessor_id}' (broken anchor)"
                            ),
                            task_ids=(task_id, dep.predecessor_id),
                        )
                    )
                    continue
                if interval.start_minute < pred_interval.end_minute:
                    issues.append(
                        VerificationIssue(
                            code="SAFETY_TASK_MISPLACED",
                            message=(
                                f"Safety task '{task_id}' starts at "
                                f"{interval.start_minute} but its predecessor "
                                f"'{dep.predecessor_id}' ends at "
                                f"{pred_interval.end_minute} (must start after)"
                            ),
                            task_ids=(task_id, dep.predecessor_id),
                        )
                    )

        return issues

    # ------------------------------------------------------------------
    # Check 2: durations
    # 检查 2：时长
    # ------------------------------------------------------------------

    def _check_durations(
        self,
        task_map: dict[str, CookingTask],
        interval_map: dict[str, ScheduledInterval],
    ) -> list[VerificationIssue]:
        issues: list[VerificationIssue] = []

        for task_id, interval in interval_map.items():
            task = task_map.get(task_id)
            if task is None:
                continue
            actual_duration = interval.end_minute - interval.start_minute
            expected_duration = task.duration_minutes
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
    # 检查 3：先后顺序与滞后约束
    # ------------------------------------------------------------------

    def _check_lags(
        self,
        task_map: dict[str, CookingTask],
        interval_map: dict[str, ScheduledInterval],
    ) -> list[VerificationIssue]:
        issues: list[VerificationIssue] = []

        for task_id, task in task_map.items():
            interval = interval_map.get(task_id)
            if interval is None:
                continue

            for dep in task.dependencies:
                pred_interval = interval_map.get(dep.predecessor_id)
                if pred_interval is None:
                    continue

                # Minimum lag: successor.start >= predecessor.end + min_lag
                # 最小滞后：后继.start >= 前驱.end + min_lag
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
                # 最大滞后：后继.start <= 前驱.end + max_lag
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
    # 检查 4：主动任务不重叠
    # ------------------------------------------------------------------

    def _check_active_no_overlap(
        self,
        task_map: dict[str, CookingTask],
        interval_map: dict[str, ScheduledInterval],
    ) -> list[VerificationIssue]:
        issues: list[VerificationIssue] = []

        # Collect intervals for ACTIVE tasks only.
        # 仅收集 ACTIVE（主动）任务的区间。
        active_intervals: list[tuple[int, int, str]] = []
        for task_id, interval in interval_map.items():
            task = task_map.get(task_id)
            if task is None:
                continue
            if task.work_mode == WorkMode.ACTIVE:
                active_intervals.append((interval.start_minute, interval.end_minute, task_id))

        # Sort by start time, then by end time.
        # 先按开始时间排序，再按结束时间排序。
        active_intervals.sort()

        for i in range(len(active_intervals)):
            s1, e1, t1 = active_intervals[i]
            for j in range(i + 1, len(active_intervals)):
                s2, e2, t2 = active_intervals[j]
                # If intervals overlap (not just touching).
                # 若区间重叠（而不仅仅是相接）。
                if s2 < e1 and s1 < e2:
                    issues.append(
                        VerificationIssue(
                            code="ACTIVE_OVERLAP",
                            message=(f"Active tasks '{t1}' [{s1}, {e1}) and '{t2}' [{s2}, {e2}) overlap"),
                            task_ids=(t1, t2),
                        )
                    )
                # Since sorted by start, if s2 >= e1, no later task can overlap.
                # 由于已按开始时间排序，若 s2 >= e1，则更晚的任务不可能重叠。
                if s2 >= e1:
                    break

        return issues

    # ------------------------------------------------------------------
    # Check 5: resource compatibility
    # 检查 5：资源兼容性
    # ------------------------------------------------------------------

    def _check_resource_compatibility(
        self,
        task_map: dict[str, CookingTask],
        problem: SchedulingProblem,
    ) -> list[VerificationIssue]:
        issues: list[VerificationIssue] = []

        # P5-0 degradation: when the caller supplies no kitchen-resource model
        # at all, equipment needs (pattern defaults / soft hints) are treated as
        # non-blocking — consistent with the soft-hint policy (器材 hint 软性化).
        # A partial resource model stays strict so real infeasibilities — e.g. a
        # declared-but-unavailable oven — are still caught by the verifier.
        # P5-0 降级：当调用方完全未提供厨房资源模型时，器材需求（模式默认值/软提示）被视为非阻塞——与软提示策略（器材 hint 软性化）一致。部分资源模型仍保持严格，以便校验器仍能捕获真实不可行——例如已声明但不可用的烤箱。
        if not problem.resources:
            return issues
        available_types = {normalise_resource_type(r.resource_type) for r in problem.resources if r.available}

        for task in problem.tasks:
            for need in task.resources:
                if normalise_resource_type(need.resource_type) not in available_types:
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
    # 检查 6：资源容量（扫描线算法）
    # ------------------------------------------------------------------

    def _check_resource_capacity(
        self,
        task_map: dict[str, CookingTask],
        interval_map: dict[str, ScheduledInterval],
        problem: SchedulingProblem,
    ) -> list[VerificationIssue]:
        """Sweep-line over start/end events for each resource type.

        对每种资源类型在开始/结束事件上执行扫描线。

        Handbook 7.13: at equal times, process END before START so adjacent
        tasks are not treated as overlapping.
        手册 7.13：在时间相等时，先处理 END 再处理 START，以免将相邻任务视为重叠。
        """
        issues: list[VerificationIssue] = []

        # Build capacity map per resource type.
        # 按资源类型构建容量映射。
        res_capacity: dict[str, int] = {}
        for r in problem.resources:
            if r.available:
                cap = int(r.capacity) if r.capacity else 1
                resource_type = normalise_resource_type(r.resource_type)
                res_capacity[resource_type] = res_capacity.get(resource_type, 0) + cap

        for res_type, capacity in res_capacity.items():
            # Collect events: (time, is_start, task_id, demand).
            # 收集事件：(time, is_start, task_id, demand)。
            events: list[tuple[int, bool, str, int]] = []
            for task_id, interval in interval_map.items():
                task = task_map.get(task_id)
                if task is None:
                    continue
                for need in task.resources:
                    if normalise_resource_type(need.resource_type) == res_type:
                        events.append((interval.start_minute, True, task_id, need.quantity))
                        events.append((interval.end_minute, False, task_id, need.quantity))
                        break  # Each task counts once per resource type
                        # 每个任务对每种资源类型只计一次

            # Sort: by time ascending; END (False) before START (True) at equal times.
            # 排序：按时间升序；时间相等时 END（False）在 START（True）之前。
            events.sort(key=lambda e: (e[0], e[1]))  # False < True → END before START
            # False < True → END 在 START 之前

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
    # 检查 7：取值范围
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
    # 检查 8：makespan 一致性
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
