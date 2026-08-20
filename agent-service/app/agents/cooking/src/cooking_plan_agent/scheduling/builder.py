# ============================================================================
# ScheduleModelBuilder — CP-SAT 模型构建：变量、区间、约束、目标函数
# ============================================================================

"""CP-SAT model construction — variables, intervals, constraints, objective.

CP-SAT 模型构建 — 变量、区间、约束与目标函数。

Design decisions:
设计决策：
- Horizon = sum of all task durations (safe but loose upper bound).
- Horizon（时间上界）= 所有任务时长之和（安全但宽松的上界）。
- Variables are stored in dicts keyed by task_id — never depend on list position.
- 变量以 task_id 为键存储在字典中 — 绝不依赖列表位置。
- Only ACTIVE tasks are added to the cook's no_overlap constraint (7.5).
- 只有 ACTIVE（主动）任务被加入厨师的 no_overlap 约束（7.5）。
- Reusable resources use add_cumulative (7.6).
- 可复用资源使用 add_cumulative（7.6）。
- Makespan is computed as the max of all task ends (or only final dish tasks).
- Makespan（完成时间）取所有任务结束时间的最大值（或仅取最终菜品任务）。
"""

from ortools.sat.python import cp_model

from cooking_plan_agent.domain.enums import WorkMode
from cooking_plan_agent.domain.models import (
    CookingTask,
)
from cooking_plan_agent.normalisation.names import normalise_resource_type
from cooking_plan_agent.scheduling.models import SchedulingProblem

# ============================================================================
# 7.3  ScheduleModelBuilder — creates CP-SAT model from SchedulingProblem
# 7.3  ScheduleModelBuilder — 根据 SchedulingProblem 创建 CP-SAT 模型
# ============================================================================


class ScheduleModelBuilder:
    """Creates a CP-SAT CpModel with variables and constraints for scheduling.

    创建带变量与约束的 CP-SAT CpModel 用于调度。

    Note: this class is responsible for model construction only.
    注意：此类仅负责模型构建。
    Solving and extraction are handled by separate classes.
    求解与提取由其他类负责。

    Usage::
        builder = ScheduleModelBuilder()
        model_info = builder.build(problem)
        # model_info contains the CpModel, variable dicts, and horizon
        # model_info 包含 CpModel、变量字典与时间上界
    """

    def __init__(self) -> None:
        self._model: cp_model.CpModel | None = None
        self._starts: dict[str, cp_model.IntVar] = {}
        self._ends: dict[str, cp_model.IntVar] = {}
        self._intervals: dict[str, cp_model.IntervalVar] = {}
        self._horizon: int = 0

    # ------------------------------------------------------------------
    # Public API
    # 公共 API
    # ------------------------------------------------------------------

    def build(self, problem: SchedulingProblem) -> "ModelInfo":
        """Construct the full CP-SAT model from a scheduling problem.

        根据调度问题构建完整的 CP-SAT 模型。

        Stages A through E are applied in order.  The returned ModelInfo
        wraps the CpModel and variable dictionaries for extraction.
        阶段 A 到 E 依序执行。返回的 ModelInfo 封装 CpModel 与变量字典，供后续提取使用。

        Args:
            problem: Validated scheduling problem with tasks and resources.
            problem: 已验证的、含任务与资源的调度问题。

        Returns:
            ModelInfo with CpModel, variable dicts, and horizon.
            ModelInfo：包含 CpModel、变量字典与时间上界。
        """
        self._model = cp_model.CpModel()
        self._starts = {}
        self._ends = {}
        self._intervals = {}
        self._horizon = self.compute_horizon(problem.tasks)

        # Stages A–D: variables and constraints (reusable)
        # 阶段 A–D：变量与约束（可复用）
        self._starts, self._ends, self._intervals = self.build_constraints(self._model, problem, self._horizon)

        # Stage E: makespan objective
        # 阶段 E：makespan 目标函数
        self._build_objective(problem)

        return ModelInfo(
            model=self._model,
            starts=self._starts,
            ends=self._ends,
            intervals=self._intervals,
            horizon=self._horizon,
            makespan_var=self._makespan_var,
        )

    # ------------------------------------------------------------------
    # 7.3 Stage A — Variables and intervals
    # 7.3 阶段 A — 变量与区间
    # ------------------------------------------------------------------

    def compute_horizon(self, tasks: tuple[CookingTask, ...]) -> int:
        """Compute a safe upper bound: sum of all durations + all lags.

        计算安全的上界：所有时长之和 + 所有滞后。

        The bare sum of durations is insufficient when tasks have
        minimum_lag_minutes dependencies — those lags extend the
        required horizon.  We compute a worst-case chain to ensure
        the horizon is always feasible.
        当任务存在 minimum_lag_minutes 依赖时，仅取时长之和是不够的——这些滞后期会延长所需的时间上界。我们计算最坏情况的链，以确保时间上界始终可行。

        This is a loose bound — the real makespan will be much shorter
        after resource contention is resolved.  The solver prunes
        infeasible values automatically.
        这是一个宽松的上界——资源争用解决后，真实的 makespan 会短得多。求解器会自动剪枝不可行的取值。
        """
        base = sum(t.duration_minutes for t in tasks)
        # Add the maximum minimum lag across all dependencies.
        # 累加所有依赖中的最大最小滞后。
        max_lag = 0
        for task in tasks:
            for dep in task.dependencies:
                max_lag = max(max_lag, dep.minimum_lag_minutes)
        # Each dependency could add lag to the critical path.
        # 每个依赖都可能给关键路径增加滞后。
        # A simple but safe bound: base + max_lag * number of dependencies.
        # 一个简单但安全的界：base + max_lag × 依赖数量。
        # Even safer: sum all lags across all edges.
        # 更安全：对所有边上的滞后求和。
        total_lag = sum(dep.minimum_lag_minutes for task in tasks for dep in task.dependencies)
        return base + total_lag

    def build_constraints(
        self,
        model: cp_model.CpModel,
        problem: SchedulingProblem,
        horizon: int,
    ) -> tuple[
        dict[str, cp_model.IntVar],
        dict[str, cp_model.IntVar],
        dict[str, cp_model.IntervalVar],
    ]:
        """Build Stages A–D (variables + constraints) on an existing model.

        在现有模型上构建阶段 A–D（变量 + 约束）。

        Does NOT set an objective — the caller is responsible for Stage E.
        Returns variable dicts for downstream use (e.g. Phase 2 objectives).
        不设置目标函数——阶段 E 由调用方负责。返回变量字典供下游使用（例如 Phase 2 的目标）。
        """
        starts: dict[str, cp_model.IntVar] = {}
        ends: dict[str, cp_model.IntVar] = {}
        intervals: dict[str, cp_model.IntervalVar] = {}

        # Stage A: create start, end, and interval variables per task
        # 阶段 A：为每个任务创建 start、end 与 interval 变量
        for task in problem.tasks:
            tid = task.task_id
            start = model.new_int_var(0, horizon, f"start:{tid}")
            end = model.new_int_var(0, horizon, f"end:{tid}")
            iv = model.new_interval_var(
                start,
                task.duration_minutes,
                end,
                f"interval:{tid}",
            )
            starts[tid] = start
            ends[tid] = end
            intervals[tid] = iv

        # Stage B: enforce precedence and lag constraints
        # 阶段 B：施加先后顺序与滞后约束
        for task in problem.tasks:
            for dep in task.dependencies:
                pred_id = dep.predecessor_id
                succ_id = task.task_id

                if pred_id not in starts or succ_id not in starts:
                    continue  # Skip dependencies on tasks not in this problem
                    # 跳过对不在本问题中的任务的依赖

                # Minimum lag: successor.start >= predecessor.end + min_lag
                # 最小滞后：后继.start >= 前驱.end + min_lag
                model.add(starts[succ_id] >= ends[pred_id] + dep.minimum_lag_minutes)

                # Maximum lag: successor.start <= predecessor.end + max_lag
                # 最大滞后：后继.start <= 前驱.end + max_lag
                if dep.maximum_lag_minutes is not None:
                    model.add(starts[succ_id] <= ends[pred_id] + dep.maximum_lag_minutes)

        # Stage C: single active cook — no_overlap for ACTIVE tasks
        # 阶段 C：单个主动厨师 — 为 ACTIVE 任务施加 no_overlap
        # Passive tasks are NOT included (they don't occupy the cook).
        # 被动（PASSIVE）任务不包含在内（它们不占用厨师）。
        active_ivs = [intervals[t.task_id] for t in problem.tasks if t.work_mode == WorkMode.ACTIVE]
        if active_ivs:
            model.add_no_overlap(active_ivs)

        # Stage D: enforce resource capacity via cumulative constraints
        # 阶段 D：通过 cumulative 约束施加资源容量限制
        resource_capacity: dict[str, int] = {}
        resource_available: dict[str, bool] = {}
        for r in problem.resources:
            resource_type = normalise_resource_type(r.resource_type)
            if resource_type not in resource_capacity:
                resource_capacity[resource_type] = 0
                resource_available[resource_type] = True
            resource_capacity[resource_type] += int(r.capacity) if r.capacity else 1
            if not r.available:
                resource_available[resource_type] = False

        for res_type, capacity in resource_capacity.items():
            if not resource_available[res_type]:
                continue

            res_ivs: list[cp_model.IntervalVar] = []
            res_demands: list[int] = []

            for task in problem.tasks:
                for need in task.resources:
                    if normalise_resource_type(need.resource_type) == res_type:
                        res_ivs.append(intervals[task.task_id])
                        res_demands.append(need.quantity)
                        break  # Each task counts once per resource type
                        # 每个任务对每种资源类型只计一次

            if res_ivs:
                model.add_cumulative(
                    intervals=res_ivs,
                    demands=res_demands,
                    capacity=capacity,
                )

        return starts, ends, intervals

    # ------------------------------------------------------------------
    # 7.7 Stage E — Makespan objective
    # 7.7 阶段 E — Makespan 目标函数
    # ------------------------------------------------------------------

    def _build_objective(self, problem: SchedulingProblem) -> None:
        """Stage E: minimise makespan = max(all task end times).

        阶段 E：最小化 makespan = max(所有任务结束时间)。

        Handbook 7.7: use final dish completion tasks when available,
        otherwise use all task ends.
        手册 7.7：可用时使用最终菜品完成时间，否则使用所有任务结束时间。

        If requested_time_limit_minutes is set, add a hard constraint.
        若设置了 requested_time_limit_minutes，则添加硬约束。
        """
        model = self._model
        assert model is not None

        # Identify final tasks — those that are not predecessors of any other task.
        # 识别最终任务——即不是任何其他任务前驱的任务。
        predecessor_ids: set[str] = set()
        for task in problem.tasks:
            for dep in task.dependencies:
                predecessor_ids.add(dep.predecessor_id)

        all_task_ids = {t.task_id for t in problem.tasks}
        final_task_ids = all_task_ids - predecessor_ids
        if not final_task_ids:
            final_task_ids = all_task_ids  # Fallback: use all tasks
            # 回退：使用所有任务

        final_ends = [self._ends[tid] for tid in final_task_ids]

        makespan = model.new_int_var(0, self._horizon, "makespan")
        model.add_max_equality(makespan, final_ends)
        model.minimize(makespan)

        # Hard deadline if requested
        # 若请求了硬截止时间
        if problem.requested_time_limit_minutes is not None:
            model.add(makespan <= problem.requested_time_limit_minutes)

        self._makespan_var = makespan


# ============================================================================
# ModelInfo — container for CP-SAT model and variable references
# ModelInfo — CP-SAT 模型与变量引用的容器
# ============================================================================


class ModelInfo:
    """Container for a constructed CP-SAT model and its variable dictionaries.

    已构建 CP-SAT 模型及其变量字典的容器。

    Note: separates model construction (builder) from solving (solver)
    and extraction (extractor).  This class bridges the gap.
    注意：将模型构建（builder）、求解（solver）与提取（extractor）分离。此类起到桥梁作用。
    """

    def __init__(
        self,
        model: cp_model.CpModel,
        starts: dict[str, cp_model.IntVar],
        ends: dict[str, cp_model.IntVar],
        intervals: dict[str, cp_model.IntervalVar],
        horizon: int,
        makespan_var: cp_model.IntVar,
    ) -> None:
        self.model = model
        self.starts = starts
        self.ends = ends
        self.intervals = intervals
        self.horizon = horizon
        self.makespan_var = makespan_var
