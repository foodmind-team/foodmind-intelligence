# ============================================================================
# Scheduling 领域模型 — 问题输入、解输出与校验
# ============================================================================

"""Scheduling domain models — problem input, solution output, verification.

调度领域模型 — 问题输入、解输出与校验。

Domain-neutral scheduler input/output.  All durations are
integer minutes because CP-SAT uses integer variables.
领域无关的调度器输入/输出。所有时长均为整数分钟，因为 CP-SAT 使用整数变量。

See also
参见
--------
domain/models.py : CookingTask, KitchenResourceSnapshot, TaskDependency
domain/enums.py  : SolverStatus
"""

from cooking_plan_agent.domain.enums import SolverStatus
from cooking_plan_agent.domain.models import (
    CookingTask,
    KitchenResourceSnapshot,
    StrictModel,
)

# ============================================================================
# 7.2  SchedulingProblem — validated input for the CP-SAT scheduler
# 7.2  SchedulingProblem — CP-SAT 调度器的已验证输入
# ============================================================================


class SchedulingProblem(StrictModel):
    """A self-contained scheduling problem ready for CP-SAT model construction.

    一个自包含的、可直接用于 CP-SAT 模型构建的调度问题。

    All tasks, resources, and time limits are validated before reaching the
    solver.  Durations are integer minutes — never floats.
    所有任务、资源与时间限制在进入求解器前均已校验。时长为整数分钟——绝不为浮点数。
    """

    tasks: tuple[CookingTask, ...]
    """All tasks to schedule (must include recipe, prep, and safety tasks).

    所有需要调度的任务（必须包含食谱、备菜与安全任务）。
    """

    resources: tuple[KitchenResourceSnapshot, ...]
    """Available kitchen resources (stove burners, ovens, sinks, etc.).

    可用的厨房资源（炉头、烤箱、水槽等）。
    """

    requested_time_limit_minutes: int | None = None
    """Optional hard deadline.  If set, the solver enforces makespan <= this value.

    可选的硬截止时间。若设置，求解器将强制 makespan <= 该值。
    """

    solver_timeout_seconds: float = 10.0
    """Maximum wall-clock time for a single CP-SAT solve call.

    单次 CP-SAT 求解调用的最大墙钟时间。
    """


# ============================================================================
# 7.2  ScheduleResult — solver output consumed by downstream services
# 7.2  ScheduleResult — 下游服务消费的求解器输出
# ============================================================================


class ScheduledInterval(StrictModel):
    """A single task's placement on the global timeline.

    单个任务在全局时间轴上的位置。

    start_minute / end_minute are integer offsets from t=0 (start of cooking).
    start_minute / end_minute 是相对 t=0（烹饪开始）的整数偏移。
    """

    task_id: str
    """Maps back to CookingTask.task_id.

    映射回 CookingTask.task_id。
    """

    start_minute: int
    """Inclusive start time in minutes.

    起始时间（分钟，含起点）。
    """

    end_minute: int
    """Exclusive end time in minutes (start + duration).

    结束时间（分钟，不含终点，等于 start + duration）。
    """

    assigned_resource_ids: tuple[str, ...] = ()
    """Which specific resource instances this task occupies (e.g. 'stove:1').

    该任务占用的具体资源实例（例如 'stove:1'）。
    """


class ScheduleResult(StrictModel):
    """The output of one CP-SAT solve attempt.

    一次 CP-SAT 求解尝试的输出。

    Note: map CP-SAT statuses before exposing to callers.
    注意：在暴露给调用方之前，先映射 CP-SAT 状态。
    """

    status: SolverStatus
    """Mapped solver status — never raw CP-SAT enum.

    映射后的求解器状态——绝不是原始 CP-SAT 枚举。
    """

    makespan_minutes: int | None = None
    """Total elapsed time from start to last finish.  None if infeasible/unknown.

    从开始到最后一个完成的总耗时。不可行/未知时为 None。
    """

    intervals: tuple[ScheduledInterval, ...] = ()
    """Every scheduled interval.  Empty if no feasible solution found.

    所有已调度的区间。未找到可行解时为空。
    """

    wall_time_seconds: float = 0.0
    """Actual solver wall-clock time for this solve call.

    本次求解调用的实际求解器墙钟时间。
    """

    best_objective_bound: int | None = None
    """Lower bound on the objective (makespan).  Only meaningful for FEASIBLE/OPTIMAL.

    目标（makespan）的下界。仅在 FEASIBLE/OPTIMAL 时有意义。
    """

    # --- Multi-objective extension (P3-03) ---
    # --- 多目标扩展（P3-03） ---
    optimization_phases: tuple[str, ...] = ()
    """Phases actually applied, in priority order (e.g. ("makespan", "holding", "context_switch")).

    实际应用的阶段，按优先级顺序（例如 ("makespan", "holding", "context_switch")）。
    """

    holding_objective: int | None = None
    """Weighted holding-time value from Phase 2 (None when not run).

    Phase 2 的加权保温时间值（未运行时为 None）。
    """

    context_switch_objective: int | None = None
    """Context-switch cost value from Phase 3 (None when not run).

    Phase 3 的上下文切换成本值（未运行时为 None）。
    """

    active_labour_objective: int | None = None
    """Active-labour value from Phase 4 (None when not run / no equivalent modes).

    Phase 4 的主动劳动力值（未运行/无等价模式时为 None）。
    """


# ============================================================================
# 7.13  VerificationReport — independent correctness check
# 7.13  VerificationReport — 独立正确性检查
# ============================================================================


class VerificationIssue(StrictModel):
    """A single correctness violation found by the verifier.

    校验器发现的一处正确性违规。
    """

    code: str
    """Machine-readable issue code (e.g. 'MISSING_TASK', 'CAPACITY_EXCEEDED').

    机器可读的问题代码（例如 'MISSING_TASK'、'CAPACITY_EXCEEDED'）。
    """

    message: str
    """Human-readable description of the violation.

    违规的可读描述。
    """

    task_ids: tuple[str, ...] = ()
    """Tasks involved in this violation, if any.

    涉及此违规的任务（若有）。
    """


class VerificationReport(StrictModel):
    """Independent verification report — does not use OR-Tools APIs.

    独立校验报告 — 不使用 OR-Tools API。

    Note: the verifier must not reuse CP-SAT variables or assume
    the model was constructed correctly.
    注意：校验器不得复用 CP-SAT 变量，也不得假设模型被正确构建。
    """

    passed: bool
    """True only if zero issues were found.

    仅当未发现问题时为 True。
    """

    issues: tuple[VerificationIssue, ...]
    """Every correctness violation found.  Empty means the schedule passed.

    发现的所有正确性违规。为空表示调度通过。
    """

    checked_task_count: int = 0
    """How many tasks were verified.

    被校验的任务数量。
    """

    checked_resource_count: int = 0
    """How many resources were verified.

    被校验的资源数量。
    """


# ============================================================================
# P5-3  RepairAttemptRecord — schedule 反思修复尝试留痕
# ============================================================================


class RepairAttemptRecord(StrictModel):
    """P5-3: 单次 schedule 反思修复尝试的留痕。

    outcome 取值：
      - "retrying"：本次尝试未恢复，计划降级后重试求解；
      - "gave_up"：不可修复或已达重试上限，路由到 FAILED；
      - "recovered"：保留位，供未来"重试后恢复"场景记录。
    """

    attempt: int
    issues: tuple[str, ...] = ()
    action: str = ""
    outcome: str = "retrying"
